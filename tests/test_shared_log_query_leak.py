"""Spoken text must not be recoverable by another local user from the server log.

The speech hooks send the assistant response as the ``text`` query parameter of
GET /synthesize. server.py starts uvicorn without turning the access log off, so
the query string is written to stdout. setup.sh and afterwords.sh point launchd's
StandardOutPath and StandardErrorPath at the same fixed path. These tests drive
the real hooks against the real server entrypoint and append that process's
stdout and stderr the way launchd 842.92.1 opens StandardOutPath
(``open(path, O_WRONLY|O_CREAT|O_APPEND|O_NOCTTY, DEFFILEMODE)`` in
``job_setup_fd``, src/core.c). They fail while another local account can read
the spoken text back out of that file.

Not driven here: ``clone-voice.sh`` (the preview request is reached only after
an interactive clone) and ``scripts/internal/gen-comparison-audio.sh`` (a manual
demo-asset generator). Both issue the same GET and would land in this log.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import subprocess
import textwrap
import time
import urllib.parse
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PORT = "7860"
HEALTH = f"http://127.0.0.1:{PORT}/health"

# launchd-842.92.1 src/core.c job_setup_fd: open mode is DEFFILEMODE.
# xnu bsd/sys/stat.h defines that as
# S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH (0666). The job's umask
# is applied by open(). The product plist does not set a Umask key, so the
# fresh-create test uses the common user umask 022 (file mode 0644).
_LAUNCHD_OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOCTTY
_DEFFILEMODE = 0o666

_SERVER_RUNNER = r"""
import os, sys
sys.path.insert(0, os.environ["REPO"])
import numpy as np
from backends.base import PreparedVoice, RefTextPolicy
import backends

class Fake:
    name = "qwen3-0.6b"
    display_name = "fake"
    sample_rate = 24000
    ref_text_policy = RefTextPolicy.OPTIONAL
    supported_langs = ("en",)
    def load(self):
        return None
    def validate_extras(self, extras):
        return None
    def prepare_voice(self, ref_audio_path, ref_text, extras):
        return PreparedVoice(ref_audio_path=ref_audio_path, ref_text=ref_text, extras=extras)
    def synthesize(self, text, prepared, lang):
        with open(os.environ["SPOKE"], "a", encoding="utf-8") as fh:
            fh.write(text.replace("\n", " ") + "\n")
        return np.zeros(self.sample_rate // 10, dtype=np.float32), self.sample_rate

def register_all(with_17b=False):
    backends.reset_for_tests()
    backends.register(Fake())

backends.register_all = register_all
import server
sys.argv = ["server.py", "--no-warmup", "--port", os.environ["PORT"], "--host", "127.0.0.1"]
server.main()
"""


def _extract(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[i : j + len(end)]


def _run_bash(script: str, env: dict, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script],
        input="",
        text=True,
        capture_output=True,
        env=env,
        check=check,
        cwd=str(REPO),
    )


def _plist_strings(plist: str, key: str) -> list[str]:
    needle = f"<key>{key}</key>"
    found = []
    rest = plist
    while needle in rest:
        rest = rest.split(needle, 1)[1]
        chunk = rest.split("</string>", 1)[0]
        found.append(chunk.split("<string>", 1)[1].strip())
    return found


def configured_log_paths(home: Path) -> list[Path]:
    """Run the product's plist writers and return every stdout/stderr path."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "Library" / "LaunchAgents").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["AFTERWORDS_REPO_DIR"] = str(REPO)
    env["SCRIPT_DIR"] = str(REPO)

    afterwords = (REPO / "afterwords.sh").read_text()
    head = afterwords.split("# ── Main dispatch", 1)[0]
    ran = _run_bash(head + '\nmkdir -p "$(dirname "$PLIST_PATH")"\nwrite_plist\ncat "$PLIST_PATH"\n', env)
    paths = _plist_strings(ran.stdout, "StandardOutPath") + _plist_strings(ran.stdout, "StandardErrorPath")

    setup = (REPO / "setup.sh").read_text()
    block = _extract(setup, 'PLIST_NAME="com.afterwords.tts-server"\n', '} > "$PLIST_PATH"\n')
    ran = _run_bash(block + '\ncat "$PLIST_PATH"\n', env)
    paths += _plist_strings(ran.stdout, "StandardOutPath") + _plist_strings(ran.stdout, "StandardErrorPath")
    if not paths:
        raise AssertionError("plist writers produced no StandardOutPath/StandardErrorPath")
    # Preserve order, drop duplicates.
    unique: list[Path] = []
    for raw in paths:
        path = Path(os.path.expanduser(raw))
        if path not in unique:
            unique.append(path)
    return unique


def _other_can_read(path: Path) -> bool:
    """True when a different local uid could open this file for reading."""
    if not path.is_file():
        return False
    if not (path.stat().st_mode & stat.S_IROTH):
        return False
    cur = path.parent
    while True:
        mode = cur.stat().st_mode
        if not (mode & stat.S_IXOTH):
            return False
        if cur == cur.parent:
            return True
        cur = cur.parent


def _other_can_plant(path: Path) -> bool:
    """True when a different local uid could create this path in a shared dir."""
    parent = path.parent
    if not parent.is_dir():
        return False
    mode = parent.stat().st_mode
    return bool(mode & stat.S_IWOTH) and bool(mode & stat.S_IXOTH)


def _launchd_open(path: Path) -> int:
    # job_setup_fd does not create missing parent directories. open() fails
    # with ENOENT in that case and launchd leaves the job's stdout untouched.
    return os.open(str(path), _LAUNCHD_OPEN_FLAGS, _DEFFILEMODE)


class _Server:
    def __init__(self, log_path: Path, spoke: Path, plant: bool):
        self.log_path = log_path
        self.spoke = spoke
        self.plant = plant
        self.proc: subprocess.Popen | None = None
        self._fds: list[int] = []

    def start(self) -> None:
        # A sticky shared directory lets another account create the path only
        # when it does not already exist. Do not replace a file the product
        # created while writing the plist.
        if self.plant and not self.log_path.exists() and _other_can_plant(self.log_path):
            fd = os.open(str(self.log_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            os.close(fd)
            os.chmod(self.log_path, 0o666)
        previous = os.umask(0o022)
        try:
            try:
                out_fd = _launchd_open(self.log_path)
            except FileNotFoundError:
                out_fd = os.open(os.devnull, os.O_WRONLY)
                self.log_path = Path(os.devnull)
        finally:
            os.umask(previous)
        err_fd = os.dup(out_fd)
        self._fds = [out_fd, err_fd]
        runner = self.spoke.parent / "run_server.py"
        runner.write_text(_SERVER_RUNNER)
        env = os.environ.copy()
        env["REPO"] = str(REPO)
        env["SPOKE"] = str(self.spoke)
        env["PORT"] = PORT
        self.proc = subprocess.Popen(
            [sys_executable(), str(runner)],
            cwd=str(REPO),
            env=env,
            stdout=out_fd,
            stderr=err_fd,
            start_new_session=True,
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"server exited {self.proc.returncode} before ready\n{self._log_tail()}"
                )
            try:
                probe = subprocess.run(
                    ["curl", "-sf", "--max-time", "2", HEALTH],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError:
                probe = None
            if probe and probe.returncode == 0 and '"ready"' in probe.stdout:
                return
            time.sleep(0.2)
        raise AssertionError(f"server did not become ready\n{self._log_tail()}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass

    def exposes(self, marker: str) -> bool:
        if not self.log_path.is_file() or not _other_can_read(self.log_path):
            return False
        raw = self.log_path.read_text(errors="replace")
        decoded = urllib.parse.unquote(raw)
        return marker in raw or marker in decoded

    def _log_tail(self) -> str:
        if not self.log_path.is_file():
            return "(no log file)"
        text = self.log_path.read_text(errors="replace")
        return text[-2000:]


def sys_executable() -> str:
    return os.environ.get("AFTERWORDS_TEST_PYTHON", shutil.which("python3") or "python3")


def _stat_shim(directory: Path) -> None:
    """BSD ``stat -f%u`` / ``stat -f%z`` used by the macOS hooks.

    GNU stat rejects ``-f%u``. The shim reports the real owner and size so the
    hooks' own-directory check behaves as it does on macOS, then execs the
    system stat for every other invocation.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "stat"
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os, sys
            args = sys.argv[1:]
            if len(args) == 2 and args[0] in ("-f%u", "-f%z"):
                st = os.stat(args[1])
                print(st.st_uid if args[0] == "-f%u" else st.st_size)
                raise SystemExit(0)
            os.execv("/usr/bin/stat", ["stat", *args])
            """
        )
    )
    path.chmod(0o755)


def _hook_env(home: Path, bin_dir: Path) -> dict:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env["CODEX_THREAD_ID"] = "leak-test"
    env.pop("HERMES_SESSION_PLATFORM", None)
    return env


def _install_hooks(home: Path, env: dict) -> None:
    setup = (REPO / "setup.sh").read_text()
    block = _extract(setup, "# Strip-markdown helper\n", 'chmod +x "$HOOKS_DIR/tts-worker.sh"\n')
    env = dict(env)
    env["SCRIPT_DIR"] = str(REPO)
    env["HOOKS_DIR"] = str(home / ".claude" / "hooks")
    (home / ".claude" / "hooks").mkdir(parents=True, exist_ok=True)
    _run_bash(block, env)


def _clear_shared_state() -> None:
    for path in (
        "/tmp/claude-tts-queue",
        "/tmp/codex-tts-queue-leak-test",
        "/tmp/afterwords-play.lock",
        "/tmp/afterwords-play.pid",
        "/tmp/claude-tts-worker.pid",
        "/tmp/claude-tts-worker.lock",
        "/tmp/codex-tts-worker-leak-test.pid",
        "/tmp/codex-tts-worker-leak-test.lock",
    ):
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def _wait_spoke(spoke: Path, marker: str, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if spoke.is_file() and marker in spoke.read_text(errors="replace"):
            return
        time.sleep(0.1)
    got = spoke.read_text(errors="replace") if spoke.is_file() else ""
    raise AssertionError(f"{marker!r} never reached synthesis.\nspoke file:\n{got[-1500:]}")


def _wait_idle(timeout: float = 30) -> None:
    """Wait until a background worker has released the play lock and its pid file."""
    deadline = time.time() + timeout
    pidfiles = (
        "/tmp/claude-tts-worker.pid",
        "/tmp/codex-tts-worker-leak-test.pid",
        "/tmp/afterwords-play.pid",
    )
    while time.time() < deadline:
        busy = os.path.isdir("/tmp/afterwords-play.lock")
        for pidfile in pidfiles:
            if not os.path.isfile(pidfile):
                continue
            try:
                pid = int(Path(pidfile).read_text().strip())
            except ValueError:
                continue
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            busy = True
        if not busy:
            return
        time.sleep(0.1)


def _run(cmd: list[str], env: dict, stdin: str | None = None, timeout: float = 60, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=stdin,
        text=True,
        capture_output=True,
        env=env,
        cwd=str(cwd or REPO),
        timeout=timeout,
        check=False,
    )


def drive_clients(home: Path, bin_dir: Path, spoke: Path) -> list[tuple[str, str]]:
    """Run every agent speech path. Returns (name, marker) in drive order."""
    env = _hook_env(home, bin_dir)
    _install_hooks(home, env)
    # Mute skips afplay only. Synthesis still runs.
    Path("/tmp/afterwords-muted").write_text("1")
    work = home / "work"
    work.mkdir(parents=True, exist_ok=True)
    clients: list[tuple[str, str]] = []

    def queue_hook(name: str, marker: str, cmd: list[str], stdin: str, cwd: Path | None = None) -> None:
        _clear_shared_state()
        ran = _run(cmd, env, stdin=stdin, cwd=cwd)
        if ran.returncode != 0:
            raise AssertionError(f"{name} hook exited {ran.returncode}: {ran.stderr}")
        _wait_spoke(spoke, marker)
        _wait_idle()
        clients.append((name, marker))

    queue_hook(
        "claude-stop",
        "SpokeClaudeStopMarker",
        ["bash", str(home / ".claude" / "hooks" / "tts-hook.sh")],
        json.dumps({"last_assistant_message": "SpokeClaudeStopMarker", "agent_type": ""}),
        cwd=work,
    )
    queue_hook(
        "cursor",
        "SpokeCursorMarker",
        ["bash", str(REPO / ".claude" / "hooks" / "cursor-tts-hook.sh")],
        json.dumps({"text": "SpokeCursorMarker", "workspace_roots": [str(work)]}),
        cwd=work,
    )
    queue_hook(
        "gemini",
        "SpokeGeminiMarker",
        ["bash", str(REPO / ".claude" / "hooks" / "gemini-tts-hook.sh")],
        json.dumps({"prompt_response": "SpokeGeminiMarker"}),
        cwd=work,
    )
    transcript = home / "agy-transcript.jsonl"
    transcript.write_text(
        json.dumps(
            {"source": "MODEL", "type": "PLANNER_RESPONSE", "content": "SpokeAntigravityMarker"}
        )
        + "\n"
    )
    queue_hook(
        "antigravity",
        "SpokeAntigravityMarker",
        ["bash", str(home / ".claude" / "hooks" / "agy-tts-hook.sh")],
        json.dumps({"transcriptPath": str(transcript), "workspacePaths": [str(work)]}),
        cwd=work,
    )

    _clear_shared_state()
    codex_q = Path("/tmp/codex-tts-queue-leak-test")
    codex_q.mkdir(mode=0o700)
    (codex_q / "item.json").write_text(
        json.dumps({"project_dir": str(work), "agent": "codex", "text": "SpokeCodexMarker"})
    )
    ran = _run(["bash", str(REPO / ".claude" / "hooks" / "codex-tts-worker.sh")], env, cwd=work)
    if ran.returncode != 0:
        raise AssertionError(f"codex worker exited {ran.returncode}: {ran.stderr}")
    _wait_spoke(spoke, "SpokeCodexMarker")
    _wait_idle()
    clients.append(("codex", "SpokeCodexMarker"))

    handler = _load_handler()
    _clear_shared_state()

    async def _hermes():
        await handler.handle(
            "agent:end",
            {"response": "SpokeHermesHandlerMarker", "platform": "cli", "cwd": str(work)},
        )
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), timeout=30)

    # Path.home() follows HOME. The handler module reads it at call time.
    previous_home = os.environ.get("HOME")
    os.environ["HOME"] = str(home)
    try:
        asyncio.run(_hermes())
    finally:
        if previous_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = previous_home
    _wait_spoke(spoke, "SpokeHermesHandlerMarker")
    _wait_idle()
    clients.append(("hermes-handler", "SpokeHermesHandlerMarker"))

    def post_llm(name: str, marker: str, platform: str) -> None:
        _clear_shared_state()
        payload = json.dumps(
            {"platform": platform, "response": marker, "cwd": str(work), "chat_id": "chat"}
        )
        ran = _run(["bash", str(REPO / "scripts" / "afterwords-post-llm.sh")], env, stdin=payload, cwd=work)
        if ran.returncode != 0:
            raise AssertionError(f"{name} exited {ran.returncode}: {ran.stderr}")
        _wait_spoke(spoke, marker)
        _wait_idle()
        clients.append((name, marker))

    post_llm("hermes-post-llm-cli", "SpokePostLlmCliMarker", "cli")
    post_llm("hermes-post-llm-discord", "SpokePostLlmDiscordMarker", "discord")

    def command(name: str, marker: str, platform: str | None) -> None:
        _clear_shared_state()
        src = home / f"{name}.txt"
        dst = home / f"{name}.wav"
        src.write_text(marker)
        local = dict(env)
        if platform:
            local["HERMES_SESSION_PLATFORM"] = platform
        else:
            local.pop("HERMES_SESSION_PLATFORM", None)
        ran = _run(
            ["bash", str(REPO / "scripts" / "afterwords-tts-command.sh"), str(src), str(dst)],
            local,
            cwd=work,
        )
        if ran.returncode != 0:
            raise AssertionError(f"{name} exited {ran.returncode}: {ran.stderr}")
        _wait_spoke(spoke, marker, timeout=20 if platform else 30)
        _wait_idle()
        clients.append((name, marker))

    command("hermes-command-cli", "SpokeCommandCliMarker", None)
    command("hermes-command-discord", "SpokeCommandDiscordMarker", "discord")

    _clear_shared_state()
    ran = _run(
        ["bash", str(REPO / "skill" / "scripts" / "speak.sh"), "SpokeSkillSpeakMarker"],
        env,
        cwd=work,
    )
    # afplay is absent here; the request is already done before it runs.
    _wait_spoke(spoke, "SpokeSkillSpeakMarker")
    _wait_idle()
    clients.append(("skill-speak", "SpokeSkillSpeakMarker"))

    ran = _run(
        ["bash", str(REPO / "scripts" / "hermes-tts.sh"), "SpokeHermesWrapperMarker"],
        env,
        cwd=work,
    )
    _wait_spoke(spoke, "SpokeHermesWrapperMarker")
    _wait_idle()
    clients.append(("hermes-tts-wrapper", "SpokeHermesWrapperMarker"))

    demo = "The quick brown fox jumps over the lazy dog."
    ran = _run(["bash", str(REPO / "afterwords.sh"), "voices", "--demo"], env, cwd=work, timeout=180)
    if ran.returncode != 0:
        raise AssertionError(f"voices --demo exited {ran.returncode}: {ran.stderr[-2000:]}")
    _wait_spoke(spoke, demo, timeout=180)
    _wait_idle()
    clients.append(("voices-demo", demo))
    return clients


def _load_handler():
    import importlib.util

    path = REPO / "hermes" / "hooks" / "afterwords-tts" / "handler.py"
    spec = importlib.util.spec_from_file_location("afterwords_hermes_handler_leak", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_not_exposed(server: _Server, clients: list[tuple[str, str]]) -> None:
    leaked = [(name, marker) for name, marker in clients if server.exposes(marker)]
    if leaked:
        shown = ", ".join(name for name, _marker in leaked)
        mode = oct(server.log_path.stat().st_mode & 0o777)
        raise AssertionError(
            f"{shown} left spoken text in {server.log_path} "
            f"(mode {mode}), which another local account can read"
        )


@pytest.fixture()
def speech_home(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    _stat_shim(bin_dir)
    spoke = tmp_path / "spoke.txt"
    yield home, bin_dir, spoke
    Path("/tmp/afterwords-muted").unlink(missing_ok=True)
    _clear_shared_state()


def _drive_against(log_path: Path, home: Path, bin_dir: Path, spoke: Path, plant: bool):
    server = _Server(log_path, spoke, plant=plant)
    try:
        server.start()
        clients = drive_clients(home, bin_dir, spoke)
        _assert_not_exposed(server, clients)
    finally:
        server.stop()


def _paths_after_install(home: Path) -> list[Path]:
    """Run the plist writers twice.

    The first run discovers the paths. Clearing them and running the writers
    again lets a fixed installer recreate a private log, and leaves the
    vulnerable installer with no file for launchd to create or an attacker to
    plant.
    """
    paths = configured_log_paths(home)
    for path in paths:
        if path.is_symlink() or path.is_file():
            path.unlink()
    return configured_log_paths(home)


def test_attacker_planted_server_log_does_not_reveal_spoken_text(speech_home):
    home, bin_dir, spoke = speech_home
    for log_path in _paths_after_install(home):
        _drive_against(log_path, home, bin_dir, spoke, plant=True)


def test_launchd_created_server_log_does_not_reveal_spoken_text(speech_home, tmp_path):
    home, bin_dir, spoke = speech_home
    for log_path in _paths_after_install(tmp_path / "plist-home"):
        _drive_against(log_path, home, bin_dir, spoke, plant=False)
