"""Voice names from a project ``.afterwords`` must not choose archive paths.

A repository can ship ``.afterwords``. The Hermes native hook and the Claude
and Codex workers copy the resolved voice into the archive filename. These
tests drive those writers and require every created or modified file to stay
inside the archive directory.

They fail while a voice containing ``..`` or an absolute path can redirect
the write. ``scripts/afterwords-post-llm.sh`` and
``scripts/afterwords-tts-command.sh`` name archives from a slug of the spoken
text (non-alphanumerics become ``-``) and are not covered here.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HANDLER_PATH = REPO / "hermes/hooks/afterwords-tts/handler.py"
SETUP_SH = REPO / "setup.sh"
CODEX_WORKER = REPO / ".claude/hooks/codex-tts-worker.sh"

ORIGINAL = "ORIGINAL-KEEP\n"
REPLY = "Planted reply text from the agent."


def _load_handler():
    spec = importlib.util.spec_from_file_location(
        "afterwords_hermes_handler_traversal", HANDLER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _snapshot(roots: list[Path]) -> dict[Path, bytes]:
    found: dict[Path, bytes] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                found[path.resolve()] = path.read_bytes()
    return found


def _mutations(before: dict[Path, bytes], roots: list[Path], archive: Path) -> list[str]:
    """Files created, edited, or removed outside ``archive``."""
    archive_res = archive.resolve()
    after = _snapshot(roots)
    problems: list[str] = []
    for path, data in after.items():
        if before.get(path) == data:
            continue
        if path.is_relative_to(archive_res):
            continue
        problems.append(f"wrote {path}: {data[:80]!r}")
    for path, data in before.items():
        if path.is_relative_to(archive_res):
            continue
        if not path.exists() or path.read_bytes() != data:
            problems.append(f"mutated {path}")
    return problems


def _clear_play_lock() -> None:
    pid = Path("/tmp/afterwords-play.pid")
    lock = Path("/tmp/afterwords-play.lock")
    pid.unlink(missing_ok=True)
    if lock.is_symlink() or lock.is_file():
        lock.unlink()
    elif lock.is_dir():
        try:
            lock.rmdir()
        except OSError:
            pass


class _Resp:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    async def read(self) -> bytes:
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _Session:
    def __init__(self, owner: "_Aiohttp"):
        self._owner = owner

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def get(self, url, timeout=None):
        self._owner.urls.append(url)
        if url.endswith("/health"):
            return _Resp(200, b"{}")
        self._owner.synth_urls.append(url)
        if self._owner.on_synth is not None:
            self._owner.on_synth()
        return _Resp(self._owner.synth_status, self._owner.synth_body)


class _Timeout:
    def __init__(self, total=None):
        self.total = total


class _Aiohttp:
    def __init__(self, synth_status: int, synth_body: bytes, on_synth=None):
        self.synth_status = synth_status
        self.synth_body = synth_body
        self.on_synth = on_synth
        self.urls: list[str] = []
        self.synth_urls: list[str] = []
        self.ClientTimeout = _Timeout

    def ClientSession(self):
        return _Session(self)


def _write_afterwords(project: Path, text: str) -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / ".afterwords").write_text(text, encoding="utf-8")


def _run_handler(
    monkeypatch,
    tmp_path: Path,
    *,
    afterwords: str,
    voice_cwd: Path,
    cwd_via: str,
    synth_status: int = 400,
    synth_body: bytes = b"",
    watch: list[Path] | None = None,
    archive_name: str = ".hermes/tts-archive",
    wrap_playback: bool = False,
    use_global: bool = False,
):
    """Run ``handle('agent:end')`` through the real archive writer.

    ``cwd_via`` is ``context`` (Hermes ``context['cwd']``) or ``terminal``
    (empty context cwd, ``TERMINAL_CWD``). Health is forced to HTTP 200 so
    the hook proceeds; synthesis returns ``synth_status`` without a server.
    """
    home = tmp_path / "home"
    home.mkdir()
    project = voice_cwd
    project.mkdir(parents=True, exist_ok=True)
    if use_global:
        (home / ".afterwords").write_text(afterwords, encoding="utf-8")
    else:
        _write_afterwords(project, afterwords)
    archive = home.joinpath(*archive_name.split("/"))
    roots = [home]
    if watch:
        roots.extend(watch)
    before = _snapshot(roots)
    intact_at_synth: list[bool] = []

    def on_synth():
        intact_at_synth.append(not _mutations(before, roots, archive))

    mod = _load_handler()
    fake = _Aiohttp(synth_status, synth_body, on_synth)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(mod, "GLOBAL_AFTERWORDS", home / ".afterwords")
    monkeypatch.setattr(mod, "aiohttp", fake)
    calls: list[list[str]] = []
    if wrap_playback:
        real_call = mod.subprocess.call

        def _call(args, *a, **k):
            calls.append([str(part) for part in args])
            # afplay and lame are macOS/optional. Record the path the hook
            # built and return so the rest of the writer still runs.
            if args and args[0] in {"afplay", "lame"}:
                return 0
            return real_call(args, *a, **k)

        monkeypatch.setattr(mod.subprocess, "call", _call)

    if cwd_via == "context":
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        context = {"response": REPLY, "platform": "cli", "cwd": str(project)}
    elif cwd_via == "terminal":
        monkeypatch.setenv("TERMINAL_CWD", str(project))
        context = {"response": REPLY, "platform": "cli", "cwd": ""}
    else:
        raise AssertionError(cwd_via)

    entered: list = []
    original_inner = mod._speak_chunked_inner

    async def _inner(*args, **kwargs):
        entered.append(args[1] if len(args) > 1 else kwargs.get("voice"))
        return await original_inner(*args, **kwargs)

    monkeypatch.setattr(mod, "_speak_chunked_inner", _inner)

    scheduled: list[asyncio.Task] = []
    real_create = asyncio.create_task

    def _create(coro, *args, **kwargs):
        task = real_create(coro, *args, **kwargs)
        scheduled.append(task)
        return task

    monkeypatch.setattr(mod.asyncio, "create_task", _create)
    _clear_play_lock()
    try:
        async def _drive():
            await mod.handle("agent:end", context)
            assert scheduled, "handle() did not schedule playback"
            await asyncio.gather(*scheduled)

        asyncio.run(_drive())
    finally:
        _clear_play_lock()

    assert fake.urls, "health check was not issued"
    assert any(url.endswith("/health") for url in fake.urls)
    assert entered, "archive writer did not run"
    return {
        "mutations": _mutations(before, roots, archive),
        "intact_at_synth": intact_at_synth,
        "synth_urls": fake.synth_urls,
        "calls": calls,
        "archive": archive,
        "voice": entered[0],
    }


@pytest.mark.parametrize(
    "case",
    [
        "relative-context-cwd",
        "relative-terminal-cwd",
        "dot-slash-prefix",
        "absolute",
        "simple-mode",
        "dotless",
        "global-afterwords",
    ],
)
def test_hermes_archive_write_stays_inside_archive(monkeypatch, tmp_path, case):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "requirements.txt").write_text(ORIGINAL, encoding="utf-8")
    project = tmp_path / "repo"
    # home is tmp_path/home, archive is home/.hermes/tts-archive, so three
    # ".." segments land in tmp_path. Paths are walked by the kernel, which
    # requires each existing directory; a missing extra segment makes open()
    # fail and hides the write.
    relative = "../../../outside/requirements.x"
    cwd_via = "context"
    use_global = False
    if case == "absolute":
        text = f"hermes: {outside / 'requirements.x'}\n"
    elif case == "simple-mode":
        # No colon: the whole line is the voice.
        text = relative + "\n"
    elif case == "dotless":
        text = "hermes: ../../../outside/loose\n"
    elif case == "dot-slash-prefix":
        # Does not start with "..". Path join collapses "./", and a lexical
        # is_relative_to() check on the un-resolved path still returns true.
        text = f"hermes: ./{relative}\n"
    elif case == "global-afterwords":
        text = f"hermes: {relative}\n"
        use_global = True
    elif case == "relative-terminal-cwd":
        text = f"hermes: {relative}\n"
        cwd_via = "terminal"
    else:
        text = f"hermes: {relative}\n"

    result = _run_handler(
        monkeypatch,
        tmp_path,
        afterwords=text,
        voice_cwd=project,
        cwd_via=cwd_via,
        watch=[outside],
        use_global=use_global,
    )

    # The sidecar is written before /synthesize. A rejection of the voice
    # must not leave the outside file already replaced.
    assert result["intact_at_synth"] in ([], [True]), (
        "the txt sidecar was written before the server could reject the voice: "
        f"{result['mutations']}"
    )
    assert result["mutations"] == [], result["mutations"]


def test_hermes_mp3_archive_path_stays_inside_archive(monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "requirements.txt").write_text(ORIGINAL, encoding="utf-8")
    result = _run_handler(
        monkeypatch,
        tmp_path,
        afterwords="hermes: ../../../outside/requirements.x\n",
        voice_cwd=tmp_path / "repo",
        cwd_via="context",
        synth_status=200,
        synth_body=b"x" * 1200,
        watch=[outside],
        wrap_playback=True,
    )
    lame_outs = [Path(cmd[-1]) for cmd in result["calls"] if cmd and cmd[0] == "lame"]
    assert lame_outs, f"mp3 archive path was never built; calls={result['calls']}"
    archive = result["archive"].resolve()
    escaped = [str(path) for path in lame_outs if not path.resolve().is_relative_to(archive)]
    assert not escaped, escaped
    assert result["mutations"] == [], result["mutations"]


def _stat_shim(directory: Path) -> Path:
    """BSD ``stat -f%u`` / ``stat -f%z`` for the workers' queue-owner check.

    GNU stat rejects ``-f%u`` and the worker then exits before the archive
    write. On macOS the check passes when the queue directory is owned by
    the current uid. Print ``os.stat``'s uid and size, which match ``id -u``
    for a directory this user just created.
    """
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "stat"
    script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os, sys
            args = sys.argv[1:]
            if len(args) >= 2 and args[0] in ("-f%u", "-f%z"):
                st = os.stat(args[1])
                print(st.st_uid if args[0] == "-f%u" else st.st_size)
                raise SystemExit(0)
            os.execv("/usr/bin/stat", ["/usr/bin/stat", *args])
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return directory


def _claude_worker_text() -> str:
    source = SETUP_SH.read_text(encoding="utf-8")
    marker = "<<'WORKEREOF'\n"
    start = source.index(marker) + len(marker)
    end = source.index("\nWORKEREOF\n", start)
    script = source[start:end] + "\n"
    assert 'ARCHIVE_BASE="$ARCHIVE_DIR/${VOICE:-default}-${STAMP}"' in script
    return script


def _run_worker(script: str, env: dict, queue: Path) -> subprocess.CompletedProcess:
    _clear_play_lock()
    try:
        return subprocess.run(
            ["bash", "-c", script],
            env=env,
            cwd=str(queue),
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
    finally:
        _clear_play_lock()


def _assert_worker_confined(proc: subprocess.CompletedProcess, queue: Path, before, roots, archive):
    assert proc.returncode == 0, proc.stderr
    leftover = sorted(path.name for path in queue.glob("*.json"))
    assert leftover == [], (
        "worker did not consume the queue item, so the archive write did not run\n"
        f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    )
    problems = _mutations(before, roots, archive)
    assert problems == [], problems


def test_codex_worker_voice_cannot_escape_archive(tmp_path):
    home = tmp_path / "home"
    outside = home / "victim"
    outside.mkdir(parents=True)
    (outside / "requirements.txt").write_text(ORIGINAL, encoding="utf-8")
    project = tmp_path / "repo"
    project.mkdir()
    (project / ".afterwords").write_text(
        "default: galadriel\ncodex: ../../victim/requirements.x\n",
        encoding="utf-8",
    )
    thread = f"traversal-{tmp_path.name}"
    queue = Path(f"/tmp/codex-tts-queue-{thread}")
    lock = Path(f"/tmp/codex-tts-worker-{thread}.lock")
    pid = Path(f"/tmp/codex-tts-worker-{thread}.pid")
    for path in (queue, lock):
        if path.is_dir() and not path.is_symlink():
            for child in path.iterdir():
                child.unlink()
            path.rmdir()
        elif path.exists() or path.is_symlink():
            path.unlink()
    pid.unlink(missing_ok=True)
    queue.mkdir()
    item = {
        "project_dir": str(project),
        "agent": "codex",
        "text": REPLY,
    }
    (queue / "item.json").write_text(json.dumps(item), encoding="utf-8")
    shim = _stat_shim(tmp_path / "bin")
    env = {
        "HOME": str(home),
        "PATH": f"{shim}{os.pathsep}{os.environ.get('PATH', '')}",
        "CODEX_THREAD_ID": thread,
        "LANG": "C",
    }
    archive = home / ".codex" / "tts-archive"
    before = _snapshot([home])
    proc = _run_worker(CODEX_WORKER.read_text(encoding="utf-8"), env, queue)
    try:
        _assert_worker_confined(proc, queue, before, [home], archive)
    finally:
        if queue.is_dir():
            for child in queue.iterdir():
                child.unlink()
            queue.rmdir()
        pid.unlink(missing_ok=True)


def test_claude_worker_voice_cannot_escape_archive(tmp_path):
    home = tmp_path / "home"
    outside = home / "victim"
    outside.mkdir(parents=True)
    (outside / "requirements.txt").write_text(ORIGINAL, encoding="utf-8")
    project = tmp_path / "repo"
    project.mkdir()
    # Main Claude turns enqueue an empty agent. ``default:`` is the fallback.
    (project / ".afterwords").write_text(
        "default: ../../victim/requirements.x\n",
        encoding="utf-8",
    )
    queue = Path("/tmp/claude-tts-queue")
    lock = Path("/tmp/claude-tts-worker.lock")
    pid = Path("/tmp/claude-tts-worker.pid")
    parked = tmp_path / "parked-claude"
    parked.mkdir()
    moved: list[tuple[Path, Path]] = []
    for path in (queue, lock, pid):
        if path.exists() or path.is_symlink():
            dest = parked / path.name
            path.rename(dest)
            moved.append((dest, path))
    queue.mkdir()
    item = {"project_dir": str(project), "agent": "", "text": REPLY}
    (queue / "item.json").write_text(json.dumps(item), encoding="utf-8")
    shim = _stat_shim(tmp_path / "bin")
    env = {
        "HOME": str(home),
        "PATH": f"{shim}{os.pathsep}{os.environ.get('PATH', '')}",
        "LANG": "C",
    }
    archive = home / ".claude" / "tts-archive"
    before = _snapshot([home])
    try:
        proc = _run_worker(_claude_worker_text(), env, queue)
        _assert_worker_confined(proc, queue, before, [home], archive)
    finally:
        if queue.exists():
            for child in queue.iterdir():
                child.unlink()
            queue.rmdir()
        pid.unlink(missing_ok=True)
        if lock.exists():
            if lock.is_dir():
                lock.rmdir()
            else:
                lock.unlink()
        for dest, path in moved:
            if path.exists():
                continue
            dest.rename(path)
