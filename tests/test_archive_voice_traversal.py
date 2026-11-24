"""Voice names from a project ``.afterwords`` must not choose archive paths.

The Claude worker installed by ``setup.sh``, the Codex worker, and the Hermes
handler build an archive filename from that voice. These tests drive the real
hooks and workers. They fail while a voice containing ``..`` or a symlink
component can create the transcript (and the chunk mp3 beside it) outside the
archive directory, and they pass once every such write stays inside it.

A voice that merely does not *start* with ``..`` (a ``./`` prefix) and a voice
that is a symlink living inside the archive are both covered, so a prefix
check or a second lookup of the unnormalized path still fails.
"""

from __future__ import annotations

import asyncio
import ctypes
import importlib.util
import json
import os
import select
import shutil
import struct
import subprocess
import textwrap
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SETUP = REPO / "setup.sh"
HANDLER = REPO / "hermes" / "hooks" / "afterwords-tts" / "handler.py"

# Longer than one chunk-text sentence so the worker still emits a single chunk,
# and distinct enough to identify in an escaped file.
SENTINEL = "archive-voice-traversal-sentinel"


def _extract_heredoc(marker: str) -> str:
    """Return the quoted heredoc setup.sh installs, unchanged."""
    source = SETUP.read_text()
    token = f"<<'{marker}'"
    start = source.find(token)
    if start < 0:
        raise AssertionError(f"{token} not found in setup.sh")
    body = source.find("\n", start)
    end = source.find(f"\n{marker}\n", body)
    if body < 0 or end < 0:
        raise AssertionError(f"unterminated {marker} heredoc")
    script = source[body + 1 : end + 1]
    if not script.startswith("#!/usr/bin/env bash"):
        raise AssertionError(f"{marker} heredoc is not the hook script")
    return script


def _wav_bytes() -> bytes:
    """PCM WAV larger than the workers' 1000-byte archive threshold."""
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 4000)
    data = buf.getvalue()
    assert len(data) > 1000
    return data


WAV = _wav_bytes()


class _SynthHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = WAV
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


@pytest.fixture(scope="module")
def synth_server():
    """Local stand-in for the TTS server the workers curl on port 7860.

    The archive path is chosen before this response is read. The body only has
    to be large enough for the worker's size check to reach the mp3 write.
    """
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", 7860), _SynthHandler)
    except OSError as exc:
        pytest.fail(f"127.0.0.1:7860 is not available for the worker: {exc}")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


@pytest.fixture(scope="module")
def toolbin(tmp_path_factory):
    """PATH shims for the two macOS/external tools the workers invoke.

    ``stat -f%u`` / ``stat -f%z`` are the formats in the worker. Apple
    file_cmds ``stat.c`` prints ``st_uid`` for ``%u`` and ``st_size`` for
    ``%z`` as unsigned decimals, then a newline. GNU stat rejects ``-f``.

    ``lame --quiet -V 2 infile outfile`` stores the second positional path and
    ``init_outfile`` opens it with ``lame_fopen(outPath, "w+b")`` (lame 3.100,
    ``frontend/parse.c`` and ``frontend/get_audio.c``). The package is not
    installed here; the shim creates that outfile and does not mkdir parents.
    """
    bindir = tmp_path_factory.mktemp("voice-archive-tools")
    (bindir / "stat").write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os, sys
            if len(sys.argv) != 3 or not sys.argv[1].startswith("-f"):
                sys.stderr.write("stat: expected -f format file\\n")
                raise SystemExit(1)
            fmt = sys.argv[1][2:]
            st = os.stat(sys.argv[2])
            out = []
            i = 0
            while i < len(fmt):
                if fmt[i] == "%" and i + 1 < len(fmt):
                    code = fmt[i + 1]
                    if code == "u":
                        out.append(str(st.st_uid))
                    elif code == "z":
                        out.append(str(st.st_size))
                    elif code == "%":
                        out.append("%")
                    else:
                        sys.stderr.write(f"stat: unsupported %{code}\\n")
                        raise SystemExit(1)
                    i += 2
                else:
                    out.append(fmt[i])
                    i += 1
            sys.stdout.write("".join(out) + "\\n")
            """
        )
    )
    (bindir / "lame").write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import sys
            positional = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
            if len(positional) < 2:
                raise SystemExit(1)
            # Last positional is the outfile (infile precedes it).
            with open(positional[-1], "wb") as handle:
                handle.write(b"MP3")
            """
        )
    )
    os.chmod(bindir / "stat", 0o755)
    os.chmod(bindir / "lame", 0o755)
    return bindir


@pytest.fixture
def runtime(tmp_path, toolbin, synth_server):
    """Per-test HOME, project, and an outside directory whose parent exists."""
    del synth_server  # the server must be up before any worker curls it
    home = tmp_path / "home"
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    home.mkdir()
    project.mkdir()
    outside.mkdir()
    hooks = home / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    (hooks / "tts-hook.sh").write_text(_extract_heredoc("HOOKEOF"))
    (hooks / "tts-worker.sh").write_text(_extract_heredoc("WORKEREOF"))
    os.chmod(hooks / "tts-hook.sh", 0o755)
    os.chmod(hooks / "tts-worker.sh", 0o755)
    shutil.copy(REPO / "strip_markdown.py", hooks / "strip-markdown.py")
    shutil.copy(REPO / "chunk_text.py", hooks / "chunk-text.py")
    shutil.copy(REPO / "agy_session_hook.py", hooks / "agy-session-hook.py")
    shutil.copy(REPO / ".claude" / "hooks" / "agy-tts-hook.sh", hooks / "agy-tts-hook.sh")
    os.chmod(hooks / "agy-tts-hook.sh", 0o755)

    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = str(toolbin) + os.pathsep + env.get("PATH", "")
    env.pop("PROJECT_DIR", None)
    env.pop("AGENT_TYPE", None)
    env.pop("CODEX_THREAD_ID", None)

    _reset_shared_paths()
    Path("/tmp/afterwords-muted").write_text("1")
    try:
        yield {
            "home": home,
            "project": project,
            "outside": outside,
            "hooks": hooks,
            "env": env,
        }
    finally:
        _reset_shared_paths()
        Path("/tmp/afterwords-muted").unlink(missing_ok=True)


def _reset_shared_paths() -> None:
    for path in (
        Path("/tmp/claude-tts-queue"),
        Path("/tmp/claude-tts-worker.lock"),
        Path("/tmp/afterwords-play.lock"),
    ):
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    for path in (
        Path("/tmp/claude-tts-worker.pid"),
        Path("/tmp/afterwords-play.pid"),
    ):
        path.unlink(missing_ok=True)
    for path in Path("/tmp").glob("codex-tts-queue-*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    for path in Path("/tmp").glob("codex-tts-worker-*.pid"):
        path.unlink(missing_ok=True)
    for path in Path("/tmp").glob("codex-tts-worker-*.lock"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _voice_to(archive: Path, outside: Path, name: str = "note") -> str:
    """Return a ``..`` voice that resolves to ``outside/name`` when joined to ``archive``."""
    voice = os.path.relpath(outside / name, archive)
    landed = Path(os.path.normpath(os.path.join(str(archive), voice)))
    if landed != outside / name:
        raise AssertionError(f"voice {voice!r} normalizes to {landed}, not {outside / name}")
    if not voice.startswith(".."):
        raise AssertionError(f"expected a dot-dot voice, got {voice!r}")
    return voice


def _write_afterwords(project: Path, voice: str, *, key: str | None, legacy: bool) -> None:
    if legacy:
        project.joinpath(".afterwords").write_text(voice + "\n")
        return
    if not key:
        raise AssertionError("mapping mode needs a key")
    project.joinpath(".afterwords").write_text(f"{key}: {voice}\n")


def _outside_files(outside: Path) -> list[Path]:
    return sorted(path for path in outside.rglob("*") if path.is_file())


class _CreateWatch:
    """Record inotify create events so a write-then-delete still fails.

    Linux ``inotify_add_watch`` on the outside directory sees ``IN_CREATE``
    for a file opened there, including through a symlink. The worker is not
    allowed to create that file before it decides to refuse.
    """

    # struct inotify_event { int wd; uint32_t mask, cookie, len; char name[]; }
    _IN_CREATE = 0x00000100
    _IN_MOVED_TO = 0x00000080
    _HEADER = struct.Struct("iIII")

    def __init__(self, directory: Path):
        self.directory = directory
        self.names: list[str] = []
        self._fd = None
        self._stop = threading.Event()
        self._thread = None
        libc = ctypes.CDLL(None, use_errno=True)
        self._libc = libc
        fd = libc.inotify_init1(0)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        wd = libc.inotify_add_watch(
            fd, os.fsencode(directory), self._IN_CREATE | self._IN_MOVED_TO
        )
        if wd < 0:
            os.close(fd)
            raise OSError(ctypes.get_errno(), "inotify_add_watch")
        self._fd = fd

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                readable, _, _ = select.select([self._fd], [], [], 0.05)
                if not readable:
                    continue
                blob = os.read(self._fd, 4096)
                offset = 0
                while offset + self._HEADER.size <= len(blob):
                    _wd, mask, _cookie, name_len = self._HEADER.unpack_from(blob, offset)
                    offset += self._HEADER.size
                    raw = blob[offset : offset + name_len]
                    offset += name_len
                    if mask & (self._IN_CREATE | self._IN_MOVED_TO):
                        name = raw.split(b"\x00", 1)[0].decode(errors="replace")
                        if name:
                            self.names.append(name)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def _assert_contained(outside: Path, hook: subprocess.CompletedProcess[str]) -> None:
    created = list(getattr(hook, "outside_creates", []) or [])
    files = _outside_files(outside)
    details = [f"created-name {name}" for name in created]
    for path in files:
        blob = path.read_bytes()[:80]
        details.append(f"{path} {blob!r}")
    assert not files and not created, (
        "voice name created files outside the archive directory:\n"
        + "\n".join(details)
        + f"\nhook rc={hook.returncode} stderr={hook.stderr[-400:]!r}"
    )


def _wait_for_pid(pidfile: Path, timeout: float = 20) -> None:
    deadline = time.time() + timeout
    seen: int | None = None
    while time.time() < deadline:
        if pidfile.exists():
            raw = pidfile.read_text().strip()
            if raw.isdigit():
                seen = int(raw)
        if seen is not None:
            try:
                os.kill(seen, 0)
            except OSError:
                time.sleep(0.05)
                return
        time.sleep(0.02)
    if seen is not None:
        try:
            os.kill(seen, 9)
        except OSError:
            return
    # The worker can exit before the first poll. The caller checks files.


def _run(
    script: Path,
    stdin: str,
    env: dict,
    cwd: Path,
    pidfile: Path,
    outside: Path,
) -> subprocess.CompletedProcess[str]:
    with _CreateWatch(outside) as watch:
        proc = subprocess.run(
            ["bash", str(script)],
            input=stdin,
            text=True,
            env=env,
            cwd=cwd,
            capture_output=True,
            timeout=20,
        )
        _wait_for_pid(pidfile)
    proc.outside_creates = list(watch.names)
    return proc


def _prepare_symlink(archive: Path, outside: Path) -> str:
    """Point ``archive/linked`` at ``outside`` and return the voice that uses it."""
    archive.mkdir(parents=True, exist_ok=True)
    link = archive / "linked"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(outside)
    return "linked/note"


def test_claude_default_dotdot_writes_outside_archive(runtime):
    home, project, outside, env = (
        runtime["home"],
        runtime["project"],
        runtime["outside"],
        runtime["env"],
    )
    voice = _voice_to(home / ".claude" / "tts-archive", outside)
    _write_afterwords(project, voice, key="default", legacy=False)
    hook = _run(
        runtime["hooks"] / "tts-hook.sh",
        json.dumps({"last_assistant_message": SENTINEL}),
        env,
        project,
        Path("/tmp/claude-tts-worker.pid"),
        outside,
    )
    _assert_contained(outside, hook)


def test_claude_agent_key_dotdot_writes_outside_archive(runtime):
    home, project, outside, env = (
        runtime["home"],
        runtime["project"],
        runtime["outside"],
        runtime["env"],
    )
    voice = _voice_to(home / ".claude" / "tts-archive", outside)
    _write_afterwords(project, voice, key="claude", legacy=False)
    hook = _run(
        runtime["hooks"] / "tts-hook.sh",
        json.dumps({"last_assistant_message": SENTINEL, "agent_type": "claude"}),
        env,
        project,
        Path("/tmp/claude-tts-worker.pid"),
        outside,
    )
    _assert_contained(outside, hook)


def test_claude_legacy_line_dotdot_writes_outside_archive(runtime):
    home, project, outside, env = (
        runtime["home"],
        runtime["project"],
        runtime["outside"],
        runtime["env"],
    )
    voice = _voice_to(home / ".claude" / "tts-archive", outside)
    _write_afterwords(project, voice, key=None, legacy=True)
    hook = _run(
        runtime["hooks"] / "tts-hook.sh",
        json.dumps({"last_assistant_message": SENTINEL}),
        env,
        project,
        Path("/tmp/claude-tts-worker.pid"),
        outside,
    )
    _assert_contained(outside, hook)


def test_claude_dot_slash_prefix_still_writes_outside_archive(runtime):
    """A leading ``./`` must not survive a check that only rejects a ``..`` prefix."""
    home, project, outside, env = (
        runtime["home"],
        runtime["project"],
        runtime["outside"],
        runtime["env"],
    )
    voice = "./" + _voice_to(home / ".claude" / "tts-archive", outside)
    assert not voice.startswith("..")
    _write_afterwords(project, voice, key="default", legacy=False)
    hook = _run(
        runtime["hooks"] / "tts-hook.sh",
        json.dumps({"last_assistant_message": SENTINEL}),
        env,
        project,
        Path("/tmp/claude-tts-worker.pid"),
        outside,
    )
    _assert_contained(outside, hook)


def test_claude_symlink_component_writes_outside_archive(runtime):
    """Rejecting ``..`` is not enough: a symlink inside the archive still escapes.

    The joined string is lexically inside the archive. Opening it again after a
    lexical check follows the symlink.
    """
    home, project, outside, env = (
        runtime["home"],
        runtime["project"],
        runtime["outside"],
        runtime["env"],
    )
    voice = _prepare_symlink(home / ".claude" / "tts-archive", outside)
    _write_afterwords(project, voice, key="default", legacy=False)
    hook = _run(
        runtime["hooks"] / "tts-hook.sh",
        json.dumps({"last_assistant_message": SENTINEL}),
        env,
        project,
        Path("/tmp/claude-tts-worker.pid"),
        outside,
    )
    _assert_contained(outside, hook)


def test_cursor_hook_dotdot_writes_outside_archive(runtime):
    home, project, outside, env = (
        runtime["home"],
        runtime["project"],
        runtime["outside"],
        runtime["env"],
    )
    voice = _voice_to(home / ".claude" / "tts-archive", outside)
    _write_afterwords(project, voice, key="cursor", legacy=False)
    hook = _run(
        REPO / ".claude" / "hooks" / "cursor-tts-hook.sh",
        json.dumps({"text": SENTINEL, "workspace_roots": [str(project)]}),
        env,
        project,
        Path("/tmp/claude-tts-worker.pid"),
        outside,
    )
    _assert_contained(outside, hook)


def test_gemini_hook_dotdot_writes_outside_archive(runtime):
    home, project, outside, env = (
        runtime["home"],
        runtime["project"],
        runtime["outside"],
        runtime["env"],
    )
    voice = _voice_to(home / ".claude" / "tts-archive", outside)
    _write_afterwords(project, voice, key="gemini", legacy=False)
    hook = _run(
        REPO / ".claude" / "hooks" / "gemini-tts-hook.sh",
        json.dumps({"prompt_response": SENTINEL}),
        env,
        project,
        Path("/tmp/claude-tts-worker.pid"),
        outside,
    )
    _assert_contained(outside, hook)


def test_agy_hook_dotdot_writes_outside_archive(runtime):
    home, project, outside, env = (
        runtime["home"],
        runtime["project"],
        runtime["outside"],
        runtime["env"],
    )
    voice = _voice_to(home / ".claude" / "tts-archive", outside)
    _write_afterwords(project, voice, key="agy", legacy=False)
    transcript = runtime["home"].parent / "agy.jsonl"
    transcript.write_text(
        json.dumps(
            {"source": "MODEL", "type": "PLANNER_RESPONSE", "content": SENTINEL}
        )
        + "\n"
    )
    hook = _run(
        runtime["hooks"] / "agy-tts-hook.sh",
        json.dumps({"transcriptPath": str(transcript), "workspacePaths": [str(project)]}),
        env,
        project,
        Path("/tmp/claude-tts-worker.pid"),
        outside,
    )
    _assert_contained(outside, hook)


def _run_codex(runtime, voice: str, *, key: str | None, legacy: bool) -> subprocess.CompletedProcess[str]:
    project, env = runtime["project"], runtime["env"].copy()
    _write_afterwords(project, voice, key=key, legacy=legacy)
    session = "travtest"
    env["CODEX_THREAD_ID"] = session
    env["AGENT_TYPE"] = "codex"
    env["PROJECT_DIR"] = str(project)
    event = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": SENTINEL}],
        },
    }
    return _run(
        REPO / ".claude" / "hooks" / "codex-tts-hook.sh",
        json.dumps(event),
        env,
        project,
        Path(f"/tmp/codex-tts-worker-{session}.pid"),
        runtime["outside"],
    )


def test_codex_hook_dotdot_writes_outside_archive(runtime):
    voice = _voice_to(runtime["home"] / ".codex" / "tts-archive", runtime["outside"])
    hook = _run_codex(runtime, voice, key="codex", legacy=False)
    _assert_contained(runtime["outside"], hook)


def test_codex_default_key_dotdot_writes_outside_archive(runtime):
    voice = _voice_to(runtime["home"] / ".codex" / "tts-archive", runtime["outside"])
    hook = _run_codex(runtime, voice, key="default", legacy=False)
    _assert_contained(runtime["outside"], hook)


def test_codex_dot_slash_prefix_still_writes_outside_archive(runtime):
    voice = "./" + _voice_to(runtime["home"] / ".codex" / "tts-archive", runtime["outside"])
    assert not voice.startswith("..")
    hook = _run_codex(runtime, voice, key="codex", legacy=False)
    _assert_contained(runtime["outside"], hook)


def test_codex_symlink_component_writes_outside_archive(runtime):
    voice = _prepare_symlink(runtime["home"] / ".codex" / "tts-archive", runtime["outside"])
    hook = _run_codex(runtime, voice, key="codex", legacy=False)
    _assert_contained(runtime["outside"], hook)


def _load_hermes():
    spec = importlib.util.spec_from_file_location("afterwords_hermes_handler_traversal", HANDLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_hermes(runtime, voice: str, *, key: str | None, legacy: bool, monkeypatch) -> subprocess.CompletedProcess[str]:
    """Resolve the project voice and speak through the real handler.

    ``subprocess.CompletedProcess`` is only the assertion's hook-shaped log.
    """
    home, project, env = runtime["home"], runtime["project"], runtime["env"]
    _write_afterwords(project, voice, key=key, legacy=legacy)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", env["PATH"])
    module = _load_hermes()
    resolved = module.resolve_voice({"cwd": str(project), "platform": "cli"})
    if resolved != voice:
        raise AssertionError(f"handler resolved {resolved!r}, expected {voice!r}")
    outside = runtime["outside"]
    with _CreateWatch(outside) as watch:
        asyncio.run(module._speak_chunked([SENTINEL], resolved))
    proc = subprocess.CompletedProcess(args=["hermes"], returncode=0, stdout="", stderr="")
    proc.outside_creates = list(watch.names)
    return proc


def test_hermes_dotdot_writes_outside_archive(runtime, monkeypatch):
    voice = _voice_to(runtime["home"] / ".hermes" / "tts-archive", runtime["outside"])
    hook = _run_hermes(runtime, voice, key="hermes", legacy=False, monkeypatch=monkeypatch)
    _assert_contained(runtime["outside"], hook)


def test_hermes_default_key_dotdot_writes_outside_archive(runtime, monkeypatch):
    voice = _voice_to(runtime["home"] / ".hermes" / "tts-archive", runtime["outside"])
    hook = _run_hermes(runtime, voice, key="default", legacy=False, monkeypatch=monkeypatch)
    _assert_contained(runtime["outside"], hook)


def test_hermes_legacy_line_dotdot_writes_outside_archive(runtime, monkeypatch):
    voice = _voice_to(runtime["home"] / ".hermes" / "tts-archive", runtime["outside"])
    hook = _run_hermes(runtime, voice, key=None, legacy=True, monkeypatch=monkeypatch)
    _assert_contained(runtime["outside"], hook)


def test_hermes_dot_slash_prefix_still_writes_outside_archive(runtime, monkeypatch):
    voice = "./" + _voice_to(runtime["home"] / ".hermes" / "tts-archive", runtime["outside"])
    assert not voice.startswith("..")
    hook = _run_hermes(runtime, voice, key="hermes", legacy=False, monkeypatch=monkeypatch)
    _assert_contained(runtime["outside"], hook)


def test_hermes_symlink_component_writes_outside_archive(runtime, monkeypatch):
    voice = _prepare_symlink(runtime["home"] / ".hermes" / "tts-archive", runtime["outside"])
    hook = _run_hermes(runtime, voice, key="hermes", legacy=False, monkeypatch=monkeypatch)
    _assert_contained(runtime["outside"], hook)
