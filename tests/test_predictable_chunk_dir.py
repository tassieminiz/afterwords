"""Predictable /tmp chunk directories must not be usable by another local user.

The Codex worker (.claude/hooks/codex-tts-worker.sh) and the Claude worker
embedded in setup.sh both do:

    CHUNK_DIR=/tmp/<prefix>-<ids>
    mkdir -p "$CHUNK_DIR"
    printf ... > "$CHUNK_DIR/N.txt"
    curl -o "$CHUNK_DIR/N.wav"
    ffmpeg -y ... "$CHUNK_DIR/N.trimmed.wav"

mkdir -p adopts a directory that already exists. The session id is the name of
/tmp/codex-tts-queue-<id> (and /tmp/claude-tts-queue is fixed). The worker pid
is written to a world-readable pid file before the worker takes the shared
play lock, and the chunk directory is created only after that lock is held.
Another local user can create the directory in that window and point
1.txt / 1.wav / 1.trimmed.wav at files the victim can write.

These tests drive the real worker scripts. They fail while that window can
clobber files, swap what afplay receives, or leak chunk text. A fix that
only checks the predictable path and then uses it again still fails: the
directory is planted while the worker is blocked in acquire_play_lock, which
is after startup and before mkdir.

Hermes (scripts/afterwords-post-llm.sh) already uses mktemp -d. The Hermes
Python hook names files with a fresh uuid (hermes/hooks/afterwords-tts/handler.py)
that is not published before the write; that path is not this bug and is not
what these tests run.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import struct
import subprocess
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CODEX_WORKER = REPO / ".claude" / "hooks" / "codex-tts-worker.sh"
SETUP_SH = REPO / "setup.sh"

PLAY_LOCK = Path("/tmp/afterwords-play.lock")
PLAY_PID = Path("/tmp/afterwords-play.pid")
MUTE_FILE = Path("/tmp/afterwords-muted")
TTS_PORT = 7860

ORIGINAL_TEXT = b"ORIGINAL-TEXT-CANARY\n"
ORIGINAL_WAV = b"ORIGINAL-WAV-CANARY\n"
ORIGINAL_TRIM = b"ORIGINAL-TRIM-CANARY\n"
ATTACKER_SAMPLE = 20000


def _wav(sample: int, seconds: float = 1.0, rate: int = 8000) -> bytes:
    """PCM WAV long enough for `ffmpeg -ss 0.1 -c copy` and the worker's size check."""
    n = int(rate * seconds)
    data = struct.pack("<h", sample) * n
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def _first_sample(blob: bytes) -> int:
    idx = blob.find(b"data")
    if idx < 0 or idx + 10 > len(blob):
        raise AssertionError(f"played file is not a wav payload: {blob[:24]!r}")
    return struct.unpack_from("<h", blob, idx + 8)[0]


def _ensure_attacker() -> None:
    if subprocess.run(["id", "attacker"], capture_output=True).returncode == 0:
        return
    created = subprocess.run(
        ["sudo", "-n", "useradd", "-M", "-s", "/usr/sbin/nologin", "attacker"],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        raise AssertionError(
            "need a second local user to plant the chunk directory "
            f"(sudo useradd attacker failed: {created.stderr.strip()})"
        )


def _run_as_attacker(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sudo", "-n", "-u", "attacker", *argv],
        capture_output=True,
        text=True,
    )


def _claude_worker_text() -> str:
    """The worker setup.sh installs, not a hand-copied fragment."""
    text = SETUP_SH.read_text()
    marker = "cat > \"$HOOKS_DIR/tts-worker.sh\" <<'WORKEREOF'\n"
    start = text.index(marker) + len(marker)
    end = text.index("\nWORKEREOF\n", start)
    body = text[start:end] + "\n"
    if "/tmp/claude-tts-chunks-$$" not in body:
        raise AssertionError("setup.sh tts-worker heredoc did not contain the chunk dir")
    return body


def _write_shims(bin_dir: Path, capture: Path, sleep_flag: Path) -> None:
    """macOS stat(1) flags the workers use, plus observers for sleep and afplay.

    stat is a format adapter: the scripts pass `-f%u` / `-f%z`, which GNU stat
    rejects. The values are the real uid and size (symlink-following, as BSD
    stat -f does). sleep and afplay still perform the real wait and record the
    bytes the worker asked to play.
    """
    (bin_dir / "stat").write_text(
        """#!/bin/sh
if [ "$1" = "-f%u" ] && [ -n "$2" ]; then
  exec python3 -c 'import os,sys; print(os.stat(sys.argv[1]).st_uid)' "$2"
fi
if [ "$1" = "-f%z" ] && [ -n "$2" ]; then
  exec python3 -c 'import os,sys; print(os.stat(sys.argv[1]).st_size)' "$2"
fi
exec /usr/bin/stat "$@"
"""
    )
    (bin_dir / "sleep").write_text(
        f"""#!/bin/sh
if [ "$1" = "0.3" ]; then
  : > {sh_quote(str(sleep_flag))}
fi
exec /usr/bin/sleep "$@"
"""
    )
    (bin_dir / "afplay").write_text(
        f"""#!/bin/sh
python3 -c 'import shutil,sys; shutil.copyfile(sys.argv[1], sys.argv[2], follow_symlinks=True)' "$1" {sh_quote(str(capture))}
exit 0
"""
    )
    for name in ("stat", "sleep", "afplay"):
        os.chmod(bin_dir / name, 0o755)


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


class _TTSServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, wav: bytes):
        self.wav = wav
        self.requests: list[str] = []
        self.got = threading.Event()
        self.proceed = threading.Event()
        self.proceed.set()
        super().__init__(("127.0.0.1", TTS_PORT), _Handler)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        server: _TTSServer = self.server  # type: ignore[assignment]
        server.requests.append(self.path)
        if self.path.startswith("/health"):
            body = b'{"default_voice":""}'
        elif self.path.startswith("/synthesize"):
            server.got.set()
            if not server.proceed.wait(30):
                body = b""
            else:
                body = server.wav
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _saw_phrase(server: _TTSServer, phrase: str) -> bool:
    for path in server.requests:
        if path.startswith("/synthesize") and phrase in urllib.parse.unquote(path):
            return True
    return False


def _attacker_read(pattern: str) -> str:
    code = (
        "import glob, pathlib, sys\n"
        "out=[]\n"
        "for name in glob.glob(sys.argv[1]):\n"
        "    try:\n"
        "        out.append(pathlib.Path(name).read_text(errors='replace'))\n"
        "    except OSError as exc:\n"
        "        out.append('ERR '+str(exc))\n"
        "print('\\n'.join(out))\n"
    )
    result = _run_as_attacker(["python3", "-c", code, pattern])
    if result.returncode != 0:
        raise AssertionError(
            f"second user could not run the read check: {result.stderr.strip()}"
        )
    return result.stdout


def _plant_symlink_dir(chunk_dir: str, canaries: dict[str, Path]) -> None:
    """Attacker-owned 0777 directory with symlinks at the worker's chunk names."""
    _run_as_attacker(["rm", "-rf", chunk_dir])
    made = _run_as_attacker(["mkdir", "-m", "777", chunk_dir])
    if made.returncode != 0:
        raise AssertionError(f"attacker mkdir failed: {made.stderr.strip()}")
    for name, target in canaries.items():
        linked = _run_as_attacker(["ln", "-s", str(target), f"{chunk_dir}/{name}"])
        if linked.returncode != 0:
            raise AssertionError(f"attacker ln {name} failed: {linked.stderr.strip()}")


def _plant_playback_swap(chunk_dir: str, holder: str) -> None:
    """1.wav is a symlink to audio the victim can read but not overwrite.

    The file lives in an attacker-owned directory under /tmp. /tmp is sticky,
    so the victim cannot unlink it, and mode 0444 stops curl -o from truncating
    it. ffmpeg can still read it and that is what gets played.
    """
    code = r"""
import os, struct, sys
holder, chunk = sys.argv[1], sys.argv[2]
os.makedirs(holder, mode=0o755, exist_ok=True)
os.chmod(holder, 0o755)
os.makedirs(chunk, mode=0o777, exist_ok=True)
os.chmod(chunk, 0o777)
n = 8000
data = struct.pack("<h", 20000) * n
blob = (
    b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
    + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
    + b"data" + struct.pack("<I", len(data)) + data
)
path = os.path.join(holder, "locked.wav")
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o444)
with os.fdopen(fd, "wb") as fh:
    fh.write(blob)
os.chmod(path, 0o444)
os.symlink(path, os.path.join(chunk, "1.wav"))
"""
    result = _run_as_attacker(["python3", "-c", code, holder, chunk_dir])
    if result.returncode != 0:
        raise AssertionError(f"attacker could not plant playback wav: {result.stderr.strip()}")


@pytest.fixture
def attacker():
    _ensure_attacker()
    yield "attacker"


def _start_server(block: bool) -> _TTSServer:
    server = _TTSServer(_wav(0))
    if block:
        server.proceed.clear()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.thread = thread  # type: ignore[attr-defined]
    return server


def _stop_server(server: _TTSServer | None) -> None:
    if server is None:
        return
    server.proceed.set()
    server.shutdown()
    server.server_close()


def _release_play_lock(sleeper: subprocess.Popen | None) -> None:
    if sleeper is not None and sleeper.poll() is None:
        sleeper.send_signal(signal.SIGKILL)
        sleeper.wait(timeout=5)
    shutil.rmtree(PLAY_LOCK, ignore_errors=True)
    PLAY_PID.unlink(missing_ok=True)


def _run_worker(
    kind: str,
    phrase: str,
    tmp: Path,
    *,
    block_synthesize: bool,
    plant,
) -> dict:
    """Queue one item and run the real worker. `plant(pid)` runs while the
    worker is inside acquire_play_lock, before the chunk directory is created.
    """
    _ensure_attacker()
    server = None
    sleeper = None
    proc = None
    muted_aside = None
    log_path = tmp / "worker.log"
    log_f = None
    capture = tmp / "played.wav"
    sleep_flag = tmp / "in-play-lock"
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    _write_shims(bin_dir, capture, sleep_flag)
    home = tmp / "home"
    home.mkdir()
    project = tmp / "project"
    project.mkdir()

    session = uuid.uuid4().hex
    if kind == "codex":
        script = CODEX_WORKER
        queue = Path(f"/tmp/codex-tts-queue-{session}")
        pidfile = Path(f"/tmp/codex-tts-worker-{session}.pid")
        lockdir = Path(f"/tmp/codex-tts-worker-{session}.lock")
        chunk_for = lambda pid: f"/tmp/codex-tts-chunks-{session}-{pid}"  # noqa: E731
        read_glob_for = lambda pid: f"/tmp/codex-tts-chunks-{session}-{pid}/1.txt"  # noqa: E731
    elif kind == "claude":
        script = tmp / "tts-worker.sh"
        script.write_text(_claude_worker_text())
        os.chmod(script, 0o755)
        queue = Path("/tmp/claude-tts-queue")
        pidfile = Path("/tmp/claude-tts-worker.pid")
        lockdir = Path("/tmp/claude-tts-worker.lock")
        chunk_for = lambda pid: f"/tmp/claude-tts-chunks-{pid}"  # noqa: E731
        read_glob_for = lambda pid: f"/tmp/claude-tts-chunks-{pid}/1.txt"  # noqa: E731
    else:
        raise AssertionError(kind)

    extra_cleanup: list[str] = []

    try:
        if MUTE_FILE.exists():
            muted_aside = tmp / "muted-aside"
            os.rename(MUTE_FILE, muted_aside)
        if PLAY_LOCK.exists():
            holder = PLAY_PID.read_text().strip() if PLAY_PID.exists() else ""
            if holder.isdigit() and _alive(int(holder)):
                raise AssertionError(f"/tmp/afterwords-play.lock is held by live pid {holder}")
            shutil.rmtree(PLAY_LOCK, ignore_errors=True)
            PLAY_PID.unlink(missing_ok=True)
        for stale in (pidfile, lockdir):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink(missing_ok=True)

        server = _start_server(block_synthesize)
        queue.mkdir(mode=0o700, exist_ok=True)
        os.chmod(queue, 0o700)
        item = queue / f"{uuid.uuid4().hex}.json"
        item.write_text(json.dumps({
            "project_dir": str(project),
            "agent": "codex",
            "text": phrase,
        }))

        # Hold the shared play lock with a live pid so the worker blocks in
        # acquire_play_lock — after it has published its pid, before mkdir of
        # the chunk directory. That is the window the attack uses.
        sleeper = subprocess.Popen(["/usr/bin/sleep", "120"])
        PLAY_LOCK.mkdir()
        PLAY_PID.write_text(str(sleeper.pid))

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '/usr/bin:/bin')}"
        env["HOME"] = str(home)
        env["CODEX_THREAD_ID"] = session
        env.pop("AFPLAY_CAPTURE", None)
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            ["bash", str(script)],
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + 20
        while not sleep_flag.exists():
            if proc.poll() is not None:
                raise AssertionError(
                    "worker exited before the play lock\n" + log_path.read_text()
                )
            if time.time() > deadline:
                raise AssertionError(
                    "worker never reached acquire_play_lock\n" + log_path.read_text()
                )
            time.sleep(0.05)
        if not pidfile.exists():
            raise AssertionError("worker reached the play lock without writing its pid file")
        pid = int(pidfile.read_text().strip())
        chunk_dir = chunk_for(pid)
        # Still holding the lock, so mkdir of the chunk dir has not run.
        if Path(chunk_dir).exists() or Path(chunk_dir).is_symlink():
            raise AssertionError(f"{chunk_dir} appeared before the lock was released")
        plant(chunk_dir, extra_cleanup)
        _release_play_lock(sleeper)
        sleeper = None

        if block_synthesize:
            if not server.got.wait(20):
                raise AssertionError(
                    "synthesize was not requested\n" + log_path.read_text()
                )
            leaked = _attacker_read(read_glob_for(pid))
            server.proceed.set()
        else:
            leaked = ""

        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            raise AssertionError(
                "worker did not finish\n" + log_path.read_text()
            ) from None
        log_f.close()
        return {
            "phrase_spoken": _saw_phrase(server, phrase),
            "log": log_path.read_text(),
            "capture": capture.read_bytes() if capture.exists() else b"",
            "leaked": leaked,
            "chunk_dir": chunk_dir,
            "pid": pid,
        }
    finally:
        if server is not None:
            server.proceed.set()
        if log_f is not None and not log_f.closed:
            log_f.close()
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        _release_play_lock(sleeper)
        _stop_server(server)
        if muted_aside is not None and muted_aside.exists():
            os.rename(muted_aside, MUTE_FILE)
        shutil.rmtree(queue, ignore_errors=True)
        for stale in (pidfile, lockdir):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink(missing_ok=True)
        for path in extra_cleanup:
            subprocess.run(["sudo", "-n", "rm", "-rf", path], check=False)


@pytest.mark.parametrize("worker", ["codex", "claude"])
def test_planted_chunk_symlinks_are_not_written(worker, attacker, tmp_path):
    """printf, curl -o, and ffmpeg -y must not follow symlinks planted in the
    predictable chunk directory while the worker waits for the play lock."""
    phrase = f"Purple lantern {uuid.uuid4().hex}."
    text_canary = tmp_path / "canary.txt"
    wav_canary = tmp_path / "canary.wav"
    trim_canary = tmp_path / "canary.trim"
    text_canary.write_bytes(ORIGINAL_TEXT)
    wav_canary.write_bytes(ORIGINAL_WAV)
    trim_canary.write_bytes(ORIGINAL_TRIM)

    def plant(chunk_dir: str, cleanup: list[str]) -> None:
        cleanup.append(chunk_dir)
        _plant_symlink_dir(chunk_dir, {
            "1.txt": text_canary,
            "1.wav": wav_canary,
            "1.trimmed.wav": trim_canary,
        })

    result = _run_worker(worker, phrase, tmp_path, block_synthesize=False, plant=plant)
    assert result["phrase_spoken"], "worker never requested synthesis\n" + result["log"]
    followed = []
    if text_canary.read_bytes() != ORIGINAL_TEXT:
        followed.append("printf > 1.txt")
    if wav_canary.read_bytes() != ORIGINAL_WAV:
        followed.append("curl -o 1.wav")
    if trim_canary.read_bytes() != ORIGINAL_TRIM:
        followed.append("ffmpeg -y 1.trimmed.wav")
    assert not followed, (
        f"{worker} worker wrote through symlinks planted in {result['chunk_dir']}: "
        + ", ".join(followed)
    )


@pytest.mark.parametrize("worker", ["codex", "claude"])
def test_attacker_cannot_choose_played_audio(worker, attacker, tmp_path):
    """Audio played from the chunk directory must be the server's synthesis.

    The planted 1.wav is mode 0444 and owned by the other user, so curl -o
    cannot replace it. ffmpeg then copies that audio and afplay is given the
    result. The capture stands in for afplay's open of that path.
    """
    phrase = f"Purple lantern {uuid.uuid4().hex}."
    holder = f"/tmp/afterwords-attacker-wav-{uuid.uuid4().hex}"

    def plant(chunk_dir: str, cleanup: list[str]) -> None:
        cleanup.append(chunk_dir)
        cleanup.append(holder)
        _plant_playback_swap(chunk_dir, holder)

    result = _run_worker(worker, phrase, tmp_path, block_synthesize=False, plant=plant)
    assert result["phrase_spoken"], "worker never requested synthesis\n" + result["log"]
    assert result["capture"], "worker never played audio\n" + result["log"]
    played = _first_sample(result["capture"])
    assert played == 0, (
        f"{worker} worker played attacker audio (first sample {played}) "
        f"from {result['chunk_dir']}"
    )


@pytest.mark.parametrize("worker", ["codex", "claude"])
def test_other_user_cannot_read_chunk_text(worker, attacker, tmp_path):
    """Chunk text must not be readable at the predictable path while synthesis runs."""
    token = uuid.uuid4().hex
    phrase = f"Purple lantern {token}."

    def plant(chunk_dir: str, cleanup: list[str]) -> None:
        # Do not create the directory. The worker creates it, and the other
        # user reads it before the response is released.
        cleanup.append(chunk_dir)

    result = _run_worker(worker, phrase, tmp_path, block_synthesize=True, plant=plant)
    assert result["phrase_spoken"], "worker never requested synthesis\n" + result["log"]
    assert token not in result["leaked"], (
        f"other local user read {worker} chunk text from {result['chunk_dir']}: "
        f"{result['leaked']!r}"
    )
