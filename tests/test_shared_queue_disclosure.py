"""Other local users must not be able to read agent responses from the TTS queue.

The Cursor, Gemini, and Antigravity hooks write each assistant response into
the fixed directory /tmp/claude-tts-queue. The Codex hook writes into
/tmp/codex-tts-queue-$CODEX_THREAD_ID (or /tmp/codex-tts-queue-global when that
variable is unset). These tests plant that directory as a second local account,
or replace it at the moment a queue file is created, and require that the
response text never lands there.

They fail while the hooks create the queue file in a directory the hook did
not both own and hold open. A check of the path that is followed by another
lookup of the same path still fails: the preload swaps the directory on the
creating open(). The Claude Code hook embedded in setup.sh is not covered; it
already refuses a symlink and a directory it does not own before writing.
Hermes does not use these queue paths.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import struct
import subprocess
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ATTACKER = "awattacker"
MARKER = "queue-disclosure-marker"
CLAUDE_QUEUE = Path("/tmp/claude-tts-queue")
HELPERS = Path(__file__).resolve().parent / "helpers"
IN_CREATE = 0x100
IN_MOVED_TO = 0x80
O_NONBLOCK = 0o4000


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def _sudo(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["sudo", *args], check=check, text=True, capture_output=True)


def ensure_attacker() -> None:
    if subprocess.run(["id", ATTACKER], capture_output=True).returncode != 0:
        _run(["sudo", "useradd", "--create-home", "--shell", "/bin/bash", ATTACKER])


def extract_worker() -> str:
    text = (ROOT / "setup.sh").read_text(encoding="utf-8")
    start = text.index("<<'WORKEREOF'\n") + len("<<'WORKEREOF'\n")
    end = text.index("\nWORKEREOF\n", start)
    return text[start:end] + "\n"


def install_home(home: Path) -> Path:
    hooks = home / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    shutil.copy(ROOT / "strip_markdown.py", hooks / "strip-markdown.py")
    shutil.copy(ROOT / "agy_session_hook.py", hooks / "agy-session-hook.py")
    for name in ("cursor-tts-hook.sh", "gemini-tts-hook.sh", "agy-tts-hook.sh"):
        dest = hooks / name
        shutil.copy(ROOT / ".claude" / "hooks" / name, dest)
        dest.chmod(0o755)
    worker = hooks / "tts-worker.sh"
    worker.write_text(extract_worker(), encoding="utf-8")
    worker.chmod(0o755)
    return hooks


def _clear_path(path: Path) -> None:
    _sudo("chattr", "-a", str(path), check=False)
    _sudo("rm", "-rf", str(path), check=False)


def reset_runtime() -> None:
    for path in (
        CLAUDE_QUEUE,
        Path("/tmp/claude-tts-queue.held"),
        Path("/tmp/aw-queue-evil"),
        Path("/tmp/aw-queue-target"),
        Path("/tmp/claude-tts-worker.pid"),
        Path("/tmp/claude-tts-worker.lock"),
        Path("/tmp/codex-tts-watch.log"),
        Path("/tmp/codex-tts-watch.pid"),
        Path("/tmp/codex-tts-queue-global"),
        Path("/tmp/codex-tts-queue-codex-thread-marker"),
        Path("/tmp/codex-tts-queue-codex-thread-marker.held"),
        Path("/tmp/codex-tts-worker-global.pid"),
        Path("/tmp/codex-tts-worker-global.lock"),
        Path("/tmp/codex-tts-worker-codex-thread-marker.pid"),
        Path("/tmp/codex-tts-worker-codex-thread-marker.lock"),
        Path("/tmp/afterwords-play.lock"),
        Path("/tmp/afterwords-play.pid"),
    ):
        _clear_path(path)
    subprocess.run(["sudo", "pkill", "-f", "tts-worker.sh"], capture_output=True)
    subprocess.run(["sudo", "pkill", "-f", "codex-tts-watch.sh"], capture_output=True)


def plant_foreign(path: Path) -> Path:
    """Attacker-owned, world-writable, append-only directory at path."""
    _clear_path(path)
    _run(["sudo", "-u", ATTACKER, "mkdir", "-m", "777", str(path)])
    _run(["sudo", "chattr", "+a", str(path)])
    return path


def plant_symlink(path: Path) -> Path:
    target = Path("/tmp/aw-queue-target")
    plant_foreign(target)
    _clear_path(path)
    _run(["sudo", "-u", ATTACKER, "ln", "-s", str(target), str(path)])
    return target


def plant_toctou(queue: Path) -> tuple[Path, Path, int]:
    """Victim-owned queue plus a mode-777 directory the preload can swap in.

    A second account cannot rename a directory it does not own inside sticky
    /tmp, so the preload performs the replacement inside the hook process.
    """
    evil = Path("/tmp/aw-queue-evil")
    held = Path(str(queue) + ".held")
    _clear_path(queue)
    _clear_path(evil)
    _clear_path(held)
    queue.mkdir(mode=0o700)
    evil.mkdir(mode=0o777)
    evil.chmod(0o777)
    return evil, held, evil.stat().st_ino


class CreateWatch:
    """Record IN_CREATE and IN_MOVED_TO on one directory inode."""

    def __init__(self, directory: Path):
        import ctypes

        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self._libc.inotify_init1.argtypes = [ctypes.c_int]
        self._libc.inotify_init1.restype = ctypes.c_int
        self._libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self._libc.inotify_add_watch.restype = ctypes.c_int
        self.fd = self._libc.inotify_init1(O_NONBLOCK)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        wd = self._libc.inotify_add_watch(
            self.fd, os.fsencode(directory), IN_CREATE | IN_MOVED_TO
        )
        if wd < 0:
            err = ctypes.get_errno()
            os.close(self.fd)
            raise OSError(err, f"inotify_add_watch {directory}")
        self.events: list[str] = []
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _consume(self, data: bytes) -> None:
        off = 0
        while off + 16 <= len(data):
            _wd, mask, _cookie, name_len = struct.unpack_from("iIII", data, off)
            raw = data[off + 16 : off + 16 + name_len].split(b"\x00", 1)[0]
            if mask & (IN_CREATE | IN_MOVED_TO):
                self.events.append(raw.decode("utf-8", "replace"))
            off += 16 + name_len

    def _run(self) -> None:
        while not self._stop:
            try:
                data = os.read(self.fd, 4096)
            except BlockingIOError:
                time.sleep(0.01)
                continue
            self._consume(data)

    def flush(self) -> None:
        time.sleep(0.05)

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=2)
        while True:
            try:
                data = os.read(self.fd, 4096)
            except BlockingIOError:
                break
            self._consume(data)
        os.close(self.fd)


def attacker_text(path: Path) -> str:
    script = r"""
set +e
target=$1
if [ -f "$target" ]; then
  printf 'FILE %s\n' "$(stat -c '%a %U' "$target" 2>/dev/null || echo unknown)"
  cat "$target"
  printf '\n'
  exit 0
fi
if [ -d "$target" ] || [ -L "$target" ]; then
  find -L "$target" -type f -print 2>/dev/null | while IFS= read -r f; do
    printf 'FILE %s %s\n' "$(stat -c '%a %U' "$f" 2>/dev/null || echo unknown)" "$f"
    cat "$f"
    printf '\n'
  done
fi
"""
    result = subprocess.run(
        ["sudo", "-u", ATTACKER, "bash", "-c", script, "bash", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout


def files_in_inode(ino: int, candidates: list[Path]) -> list[str]:
    for candidate in candidates:
        try:
            st = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(st.st_mode) and st.st_ino == ino:
            return sorted(str(p.relative_to(candidate)) for p in candidate.rglob("*") if p.is_file())
    return []


def files_under(path: Path) -> list[str]:
    try:
        st = path.lstat()
    except FileNotFoundError:
        return []
    if stat.S_ISLNK(st.st_mode):
        path = Path(os.path.realpath(path))
    if not path.is_dir():
        return []
    return sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())


def bsd_stat_path(directory: Path) -> Path:
    bindir = directory
    bindir.mkdir(parents=True, exist_ok=True)
    shim = bindir / "stat"
    shim.write_text(
        "#!/bin/sh\nexec python3 " + sh_quote(HELPERS / "bsd_stat.py") + ' "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return bindir


def sh_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def payload_for(hook: str, home: Path) -> str:
    if hook == "cursor":
        return json.dumps({"text": MARKER, "workspace_roots": ["/tmp/aw-project"]})
    if hook == "gemini":
        return json.dumps({"prompt_response": MARKER})
    if hook == "agy":
        transcript = home / "transcript.jsonl"
        transcript.write_text(
            json.dumps(
                {"source": "MODEL", "type": "PLANNER_RESPONSE", "content": MARKER}
            )
            + "\n",
            encoding="utf-8",
        )
        return json.dumps(
            {"transcriptPath": str(transcript), "workspacePaths": ["/tmp/aw-project"]}
        )
    if hook == "codex":
        # Compact JSON: codex-tts-watch.sh matches the line with a glob that
        # requires "type":"message" with no space after the colon.
        return json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": MARKER}],
                },
            },
            separators=(",", ":"),
        )
    raise AssertionError(hook)


def hook_path(hook: str, hooks: Path) -> Path:
    if hook == "codex":
        return ROOT / ".claude" / "hooks" / "codex-tts-hook.sh"
    return hooks / f"{hook}-tts-hook.sh"


def run_hook(
    hook: str,
    home: Path,
    hooks: Path,
    stat_bin: Path,
    extra_env: dict[str, str] | None = None,
    thread_id: str | None = "codex-thread-marker",
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = str(stat_bin) + os.pathsep + env.get("PATH", "")
    env.pop("LD_PRELOAD", None)
    if hook == "codex":
        if thread_id is None:
            env.pop("CODEX_THREAD_ID", None)
        else:
            env["CODEX_THREAD_ID"] = thread_id
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(hook_path(hook, hooks))],
        input=payload_for(hook, home),
        text=True,
        capture_output=True,
        env=env,
        cwd=str(ROOT),
        timeout=20,
    )


def assert_not_disclosed(files: list[str], readable: str, events: list[str], detail: str) -> None:
    assert files == [], f"queue file created where another user can receive it\n{files}\n{readable}\n{detail}"
    assert MARKER not in readable, f"other local user read the response\n{readable}\n{detail}"
    assert events == [], f"queue file was created in the planted directory\n{events}\n{detail}"


@pytest.fixture(autouse=True)
def _runtime():
    ensure_attacker()
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture(scope="session")
def preload_lib(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("preload")
    lib = work / "queue_toctou.so"
    _run(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-o",
            str(lib),
            str(HELPERS / "queue_toctou_preload.c"),
            "-ldl",
        ]
    )
    return lib


@pytest.fixture
def home(tmp_path: Path) -> Path:
    install_home(tmp_path / "home")
    return tmp_path / "home"


@pytest.fixture
def stat_bin(tmp_path: Path) -> Path:
    return bsd_stat_path(tmp_path / "bin")


@pytest.mark.parametrize("hook", ["cursor", "gemini", "agy"])
def test_foreign_owned_queue_does_not_receive_response(hook: str, home: Path, stat_bin: Path):
    planted = plant_foreign(CLAUDE_QUEUE)
    watch = CreateWatch(planted)
    try:
        result = run_hook(hook, home, home / ".claude" / "hooks", stat_bin)
        time.sleep(0.4)
        watch.flush()
        readable = attacker_text(planted)
        files = files_under(planted)
        assert_not_disclosed(files, readable, watch.events, result.stderr)
    finally:
        watch.close()


@pytest.mark.parametrize("hook", ["cursor", "gemini", "agy"])
def test_symlinked_queue_does_not_receive_response(hook: str, home: Path, stat_bin: Path):
    target = plant_symlink(CLAUDE_QUEUE)
    watch = CreateWatch(target)
    try:
        result = run_hook(hook, home, home / ".claude" / "hooks", stat_bin)
        time.sleep(0.4)
        watch.flush()
        readable = attacker_text(CLAUDE_QUEUE) + attacker_text(target)
        files = files_under(target)
        assert_not_disclosed(files, readable, watch.events, result.stderr)
    finally:
        watch.close()


@pytest.mark.parametrize("hook", ["cursor", "gemini", "agy"])
def test_queue_create_follows_the_directory_that_was_checked(
    hook: str, home: Path, stat_bin: Path, preload_lib: Path
):
    evil, held, ino = plant_toctou(CLAUDE_QUEUE)
    watch = CreateWatch(evil)
    try:
        result = run_hook(
            hook,
            home,
            home / ".claude" / "hooks",
            stat_bin,
            extra_env={
                "LD_PRELOAD": str(preload_lib),
                "AW_TOCTOU_QUEUE": str(CLAUDE_QUEUE),
                "AW_TOCTOU_EVIL": str(evil),
                "AW_TOCTOU_HELD": str(held),
            },
        )
        time.sleep(0.4)
        watch.flush()
        readable = attacker_text(CLAUDE_QUEUE) + attacker_text(evil) + attacker_text(held)
        files = files_in_inode(ino, [CLAUDE_QUEUE, evil, held])
        assert_not_disclosed(files, readable, watch.events, result.stderr)
    finally:
        watch.close()


@pytest.mark.parametrize("thread_id", ["codex-thread-marker", None])
def test_codex_hook_foreign_queue_does_not_receive_response(
    thread_id: str | None, home: Path, stat_bin: Path
):
    name = thread_id or "global"
    queue = Path(f"/tmp/codex-tts-queue-{name}")
    planted = plant_foreign(queue)
    watch = CreateWatch(planted)
    try:
        result = run_hook(
            "codex", home, home / ".claude" / "hooks", stat_bin, thread_id=thread_id
        )
        time.sleep(0.4)
        watch.flush()
        readable = attacker_text(planted)
        assert_not_disclosed(files_under(planted), readable, watch.events, result.stderr)
    finally:
        watch.close()


def test_codex_hook_symlink_does_not_receive_response(home: Path, stat_bin: Path):
    queue = Path("/tmp/codex-tts-queue-codex-thread-marker")
    target = plant_symlink(queue)
    watch = CreateWatch(target)
    try:
        result = run_hook("codex", home, home / ".claude" / "hooks", stat_bin)
        time.sleep(0.4)
        watch.flush()
        readable = attacker_text(queue) + attacker_text(target)
        assert_not_disclosed(files_under(target), readable, watch.events, result.stderr)
    finally:
        watch.close()


def test_codex_queue_create_follows_the_directory_that_was_checked(
    home: Path, stat_bin: Path, preload_lib: Path
):
    queue = Path("/tmp/codex-tts-queue-codex-thread-marker")
    evil, held, ino = plant_toctou(queue)
    watch = CreateWatch(evil)
    try:
        result = run_hook(
            "codex",
            home,
            home / ".claude" / "hooks",
            stat_bin,
            thread_id="codex-thread-marker",
            extra_env={
                "LD_PRELOAD": str(preload_lib),
                "AW_TOCTOU_QUEUE": str(queue),
                "AW_TOCTOU_EVIL": str(evil),
                "AW_TOCTOU_HELD": str(held),
            },
        )
        time.sleep(0.4)
        watch.flush()
        readable = attacker_text(queue) + attacker_text(evil) + attacker_text(held)
        files = files_in_inode(ino, [queue, evil, held])
        assert_not_disclosed(files, readable, watch.events, result.stderr)
    finally:
        watch.close()


def test_codex_watcher_log_lets_other_user_read_the_response(home: Path, stat_bin: Path):
    """The watcher logs the thread id into a world-readable file before the first answer."""
    thread_id = "codex-thread-marker"
    sessions = home / ".codex" / "sessions" / "demo"
    sessions.mkdir(parents=True)
    session = sessions / "rollout.jsonl"
    session.write_text(
        json.dumps({"id": thread_id, "type": "session_meta"}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    log_path = Path("/tmp/codex-tts-watch.log")
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = str(stat_bin) + os.pathsep + env.get("PATH", "")
    env["CODEX_THREAD_ID"] = thread_id
    env["PROJECT_DIR"] = "/tmp/aw-project"
    env["CODEX_WATCH_LOG"] = str(log_path)
    watcher = subprocess.Popen(
        ["bash", str(ROOT / ".claude" / "hooks" / "codex-tts-watch.sh")],
        env=env,
        cwd=str(ROOT),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    planted: Path | None = None
    try:
        deadline = time.time() + 5
        log_text = ""
        while time.time() < deadline:
            result = subprocess.run(
                ["sudo", "-u", ATTACKER, "cat", str(log_path)],
                capture_output=True,
                text=True,
            )
            log_text = result.stdout
            if thread_id in log_text and result.returncode == 0:
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"attacker could not read the thread id from the watch log\n{log_text}")

        planted = plant_foreign(Path(f"/tmp/codex-tts-queue-{thread_id}"))
        with session.open("a", encoding="utf-8") as handle:
            handle.write(payload_for("codex", home) + "\n")
        deadline = time.time() + 6
        while time.time() < deadline:
            if "final_answer" in log_path.read_text(encoding="utf-8"):
                break
            time.sleep(0.1)
        else:
            pytest.fail("watcher did not observe the final answer")
        time.sleep(0.6)
        readable = attacker_text(planted)
        assert_not_disclosed(files_under(planted), readable, [], log_text)
    finally:
        os.killpg(watcher.pid, signal.SIGTERM)
        try:
            watcher.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(watcher.pid, signal.SIGKILL)
        if planted is not None:
            _clear_path(planted)
