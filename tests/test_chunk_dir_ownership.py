"""Chunk folders must not be a predictable path another local user can plant.

`setup.sh` installs `~/.claude/hooks/tts-worker.sh` (the heredoc ending at
WORKEREOF). That worker is what Claude Code, Cursor, Gemini CLI, and
Antigravity all start after they queue a response. It sets

    CHUNK_DIR="/tmp/claude-tts-chunks-$$"
    mkdir -p "$CHUNK_DIR"

and then writes each sentence to `$CHUNK_DIR/N.txt` and each synthesized
chunk to `$CHUNK_DIR/N.wav`. `.claude/hooks/codex-tts-worker.sh` does the
same with `/tmp/codex-tts-chunks-$CODEX_THREAD_ID-$$`.

`mkdir -p` succeeds when that path already exists, and the workers never
check that they created it, that they own it, or that it is not a symlink.
`/tmp` is world-writable and sticky, so a second local account can create
the directory (or a symlink of that name) for a worker pid and the worker
cannot unlink it. The queue directory beside this path does check
ownership; these tests cover the chunk directory, which does not.

`scripts/afterwords-post-llm.sh` already creates its chunk folder with
`mktemp -d` and is not part of this bug.

The tests fail while a worker still writes response text or audio through
a path planted by another user. They pass when the worker creates a fresh
private directory and still synthesizes the queued response.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

REPO = Path(__file__).resolve().parent.parent
SETUP_SH = REPO / "setup.sh"
CODEX_WORKER = REPO / ".claude/hooks" / "codex-tts-worker.sh"
CHUNK_TEXT = REPO / "chunk_text.py"

ATTACKER = "aw-chunk-atk"
TTS_PORT = 7860

SENTENCE_A = "Alpha chunk secret purple-heron-441."
SENTENCE_B = "Beta chunk secret amber-lynx-442."
RESPONSE = f"{SENTENCE_A} {SENTENCE_B}"

CANARY_TEXT = "canary-text-untouched"
CANARY_WAV = b"canary-wav-untouched"
AUDIO_BODY = b"AWCHUNKAUDIO" + (b"\x00" * 2048)

# macOS stat(1) format the workers actually call. GNU stat rejects `-f%u`.
# FreeBSD stat(1) (the utility macOS ships) obtains the data with lstat(2)
# unless `-L` is passed, prints `%u` as the owner uid and `%z` as the size
# in bytes. The shim below implements only those two formats so the queue
# ownership check can succeed on Linux for a directory the victim owns.
STAT_SHIM = """#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
fmt = None
files = []
if args and args[0].startswith("-f") and args[0] != "-f":
    fmt = args[0][2:]
    files = args[1:]
elif len(args) >= 2 and args[0] == "-f":
    fmt = args[1]
    files = args[2:]
if fmt not in ("%u", "%z") or not files:
    os.execv("/usr/bin/stat", ["/usr/bin/stat", *args])
rc = 0
for path in files:
    try:
        st = os.lstat(path)
    except OSError:
        rc = 1
        continue
    print(st.st_uid if fmt == "%u" else st.st_size)
sys.exit(rc)
"""

LAUNCHER = """#!/bin/bash
umask 022
echo $$ > "$AW_READY"
for _ in $(seq 1 2000); do
    [ -f "$AW_GO" ] && exec /bin/bash "$AW_WORKER"
    sleep 0.01
done
echo "timed out waiting for the planted chunk path" >&2
exit 99
"""


def _extract_claude_worker() -> str:
    text = SETUP_SH.read_text()
    marker = "cat > \"$HOOKS_DIR/tts-worker.sh\" <<'WORKEREOF'\n"
    start = text.index(marker) + len(marker)
    end = text.index("\nWORKEREOF\n", start)
    body = text[start:end] + "\n"
    if "CHUNK_DIR=" not in body or "mkdir -p" not in body and "mktemp" not in body:
        raise AssertionError("extracted tts-worker.sh has no chunk directory setup")
    return body


def _ensure_attacker() -> None:
    if subprocess.run(["id", ATTACKER], capture_output=True).returncode == 0:
        return
    created = subprocess.run(
        [
            "sudo", "-n", "useradd",
            "--no-create-home", "--shell", "/usr/sbin/nologin",
            ATTACKER,
        ],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.fail(
            "need a second local account to plant the chunk path "
            f"({created.stderr.strip() or created.stdout.strip()})"
        )


def _sudo(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sudo", "-n", "-u", ATTACKER, *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _attacker_reads(directory: Path) -> list[tuple[str, bytes]]:
    """Read every file under `directory` as the other local account."""
    if not directory.exists() and not directory.is_symlink():
        return []
    script = r"""
import os, sys
root = sys.argv[1]
for dirpath, _dirs, names in os.walk(root, followlinks=True):
    for name in names:
        path = os.path.join(dirpath, name)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
            mode = os.stat(path).st_mode & 0o777
        except OSError:
            continue
        sys.stdout.buffer.write(f"{mode:o}\t{path}\t{len(data)}\n".encode())
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.write(b"\0")
"""
    proc = subprocess.run(
        ["sudo", "-n", "-u", ATTACKER, "python3", "-c", script, str(directory)],
        capture_output=True,
    )
    reads: list[tuple[str, bytes]] = []
    buf = proc.stdout
    while buf:
        line, sep, buf = buf.partition(b"\n")
        if not sep:
            break
        try:
            _mode, path, _length = line.decode().split("\t", 2)
        except ValueError:
            break
        data, sep, buf = buf.partition(b"\0")
        if not sep:
            break
        reads.append((path, data))
    return reads


class _TTSServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, plant: Path):
        self.plant = plant
        self.texts: list[str] = []
        self.reads: list[tuple[str, bytes]] = []
        super().__init__(("127.0.0.1", TTS_PORT), _TTSHandler)

    def snapshot(self) -> None:
        self.reads.extend(_attacker_reads(self.plant))


class _TTSHandler(BaseHTTPRequestHandler):
    server: _TTSServer

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            body = b'{"default_voice":""}'
        elif parsed.path == "/synthesize":
            text = parse_qs(parsed.query).get("text", [""])[0]
            self.server.texts.append(text)
            body = AUDIO_BODY
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        if parsed.path == "/synthesize":
            # The .txt chunk is written before curl starts, and curl writes
            # the .wav while this response is still being consumed.
            time.sleep(0.2)
            self.server.snapshot()


def _clean_runtime(extra: list[Path]) -> None:
    for path in [
        Path("/tmp/claude-tts-queue"),
        Path("/tmp/claude-tts-worker.pid"),
        Path("/tmp/claude-tts-worker.lock"),
        Path("/tmp/afterwords-play.lock"),
        Path("/tmp/afterwords-play.pid"),
        *extra,
    ]:
        subprocess.run(["sudo", "-n", "rm", "-rf", "--", str(path)], check=False)
    for pattern in ("/tmp/codex-tts-queue-chunktest*", "/tmp/codex-tts-worker-chunktest*"):
        for match in Path("/tmp").glob(pattern[len("/tmp/"):]):
            subprocess.run(["sudo", "-n", "rm", "-rf", "--", str(match)], check=False)


def _queue_item(queue: Path, project_dir: Path) -> None:
    queue.mkdir(mode=0o700, exist_ok=True)
    os.chmod(queue, 0o700)
    item = queue / "1.json"
    item.write_text(json.dumps({
        "project_dir": str(project_dir),
        "agent": "cursor",
        "text": RESPONSE,
    }))


def _run_worker(tmp_path: Path, *, which: str, attack: str) -> dict:
    """Run one real worker against a chunk path planted by another user.

    `attack="symlink"` plants N.txt and N.wav as symlinks to victim files.
    `attack="read"` plants an empty world-writable directory and records
    whatever the other user can read out of it while the worker runs.
    """
    _ensure_attacker()
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / ".claude" / "hooks").mkdir(parents=True)
    shutil.copy(CHUNK_TEXT, home / ".claude" / "hooks" / "chunk-text.py")

    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    stat_path = shim_dir / "stat"
    stat_path.write_text(STAT_SHIM)
    stat_path.chmod(0o755)

    ready = tmp_path / "ready"
    go = tmp_path / "go"
    launcher = tmp_path / "launcher.sh"
    launcher.write_text(LAUNCHER)
    launcher.chmod(0o755)

    if which == "claude":
        worker = tmp_path / "tts-worker.sh"
        worker.write_text(_extract_claude_worker())
        worker.chmod(0o755)
        queue = Path("/tmp/claude-tts-queue")
        session = ""
        def predictable(pid: int) -> Path:
            return Path(f"/tmp/claude-tts-chunks-{pid}")
    elif which == "codex":
        worker = CODEX_WORKER
        session = "chunktest"
        queue = Path(f"/tmp/codex-tts-queue-{session}")
        def predictable(pid: int) -> Path:
            return Path(f"/tmp/codex-tts-chunks-{session}-{pid}")
    else:
        raise AssertionError(which)

    canaries: dict[int, tuple[Path, Path]] = {}
    planted: Path | None = None
    attacker_dir: Path | None = None
    server: _TTSServer | None = None
    thread: threading.Thread | None = None
    proc: subprocess.Popen[str] | None = None
    poll_stop = threading.Event()
    polled: list[tuple[str, bytes]] = []
    stdout = stderr = ""
    returncode = -1
    canary_snapshot: dict[int, tuple[str, bytes]] = {}

    try:
        _clean_runtime([])
        _queue_item(queue, project)
        env = os.environ.copy()
        env.update({
            "HOME": str(home),
            "PATH": f"{shim_dir}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "AW_READY": str(ready),
            "AW_GO": str(go),
            "AW_WORKER": str(worker),
            "TMPDIR": "/tmp",
        })
        if which == "codex":
            env["CODEX_THREAD_ID"] = session
        else:
            env.pop("CODEX_THREAD_ID", None)
        # Queue JSON is the product input. Do not let a developer .afterwords
        # redirect the worker away from the chunk path under test.
        env.pop("VOICE", None)

        proc = subprocess.Popen(
            ["/bin/bash", str(launcher)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        pid = _wait_ready(ready, proc)
        planted = predictable(pid)
        attacker_dir = _plant(tmp_path, planted, canaries, attack)
        # The worker must be unable to unlink the planted path. Otherwise a
        # later rm && mkdir of the same path would not be the attack.
        removed = subprocess.run(["rm", "-rf", str(planted)], capture_output=True, text=True)
        if not (planted.is_symlink() or planted.is_dir()):
            pytest.fail(
                f"victim removed the planted chunk path ({removed.stderr.strip()}); "
                "the sticky-bit precondition did not hold"
            )

        server = _TTSServer(attacker_dir if attack == "read" else attacker_dir)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def _poll() -> None:
            assert attacker_dir is not None
            while not poll_stop.is_set():
                polled.extend(_attacker_reads(attacker_dir))
                poll_stop.wait(0.01)

        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()
        go.write_text("go")
        stdout, stderr = proc.communicate(timeout=30)
        returncode = proc.returncode
        proc = None
        canary_snapshot = {
            index: (txt.read_text(), wav.read_bytes())
            for index, (txt, wav) in canaries.items()
        }
    finally:
        poll_stop.set()
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, 9)
            except OSError:
                proc.kill()
            proc.wait(timeout=5)
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        extras = [p for p in (planted, attacker_dir) if p is not None]
        extras.extend(path for pair in canaries.values() for path in pair)
        _clean_runtime(extras)

    assert server is not None
    reads = list(server.reads) + polled
    archives = []
    for archive_root in (home / ".claude" / "tts-archive", home / ".codex" / "tts-archive"):
        if archive_root.is_dir():
            archives.extend(p.read_text(errors="replace") for p in archive_root.glob("*.txt"))
    return {
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "texts": list(server.texts),
        "reads": reads,
        "archives": archives,
        "canaries": canary_snapshot,
    }


def _wait_ready(ready: Path, proc: subprocess.Popen[str]) -> int:
    for _ in range(2000):
        if ready.exists():
            return int(ready.read_text().strip())
        if proc.poll() is not None:
            out, err = proc.communicate()
            pytest.fail(f"launcher exited {proc.returncode} before reporting pid\n{out}\n{err}")
        time.sleep(0.01)
    pytest.fail("launcher did not report its pid")


def _plant(
    tmp_path: Path,
    planted: Path,
    canaries: dict[int, tuple[Path, Path]],
    attack: str,
) -> Path:
    """Create `planted` as the other user. Return the directory chunks land in."""
    if planted.exists() or planted.is_symlink():
        pytest.fail(f"{planted} already exists before the attacker plants it")
    if attack == "symlink":
        # Predictable name is a symlink, and 1.txt / 1.wav (and 2) are
        # symlinks to files the victim can write. mkdir -p accepts the
        # directory symlink; the later redirects follow the file symlinks.
        target = Path(f"/tmp/aw-chunk-target-{os.getpid()}-{planted.name}")
        _sudo("mkdir", "-m", "777", str(target))
        _sudo("ln", "-s", str(target), str(planted))
        for index in (1, 2):
            text = tmp_path / f"canary-{index}.txt"
            wav = tmp_path / f"canary-{index}.wav"
            text.write_text(CANARY_TEXT)
            wav.write_bytes(CANARY_WAV)
            canaries[index] = (text, wav)
            _sudo("ln", "-s", str(text), str(target / f"{index}.txt"))
            _sudo("ln", "-s", str(wav), str(target / f"{index}.wav"))
        owned = target
    elif attack == "read":
        _sudo("mkdir", "-m", "777", str(planted))
        owned = planted
    else:
        raise AssertionError(attack)
    if os.lstat(planted).st_uid == os.getuid():
        pytest.fail(f"planted path {planted} is owned by the victim")
    mode = os.stat(owned).st_mode & 0o777
    if mode != 0o777:
        pytest.fail(f"attacker directory mode is {oct(mode)}, expected 0o777")
    return owned


def _assert_still_speaks(result: dict) -> None:
    spoken = "\n".join(result["texts"])
    assert SENTENCE_A in spoken and SENTENCE_B in spoken, (
        "worker did not synthesize the queued response, so this run never "
        f"reached the chunk writes\nstderr:\n{result['stderr']}\n"
        f"stdout:\n{result['stdout']}\ntexts={result['texts']!r}"
    )
    archived = "\n".join(result["archives"])
    assert SENTENCE_A in archived and SENTENCE_B in archived, (
        "worker did not archive the queued response\n"
        f"stderr:\n{result['stderr']}"
    )


def _assert_symlinks_untouched(result: dict) -> None:
    _assert_still_speaks(result)
    for index, (text, wav) in result["canaries"].items():
        if index < 1:
            continue
        assert text == CANARY_TEXT, (
            f"chunk {index}.txt was written through a symlink planted by another "
            f"local user: {text!r}"
        )
        assert wav == CANARY_WAV, (
            f"chunk {index}.wav was written through a symlink planted by another "
            f"local user ({len(wav)} bytes)"
        )


def _assert_other_user_cannot_read(result: dict) -> None:
    _assert_still_speaks(result)
    leaked = []
    for path, data in result["reads"]:
        if SENTENCE_A.encode() in data or SENTENCE_B.encode() in data or b"AWCHUNKAUDIO" in data:
            leaked.append((path, data[:80]))
    assert not leaked, (
        "another local user read chunk text or audio from the pre-created "
        f"chunk directory: {leaked!r}"
    )


@pytest.fixture(autouse=True)
def _runtime_lock():
    """The workers use fixed /tmp queue and play-lock paths."""
    _clean_runtime([])
    yield
    _clean_runtime([])


def test_claude_worker_does_not_write_chunks_through_planted_symlinks(tmp_path: Path) -> None:
    result = _run_worker(tmp_path, which="claude", attack="symlink")
    _assert_symlinks_untouched(result)


def test_claude_worker_does_not_leave_chunks_readable_by_another_user(tmp_path: Path) -> None:
    result = _run_worker(tmp_path, which="claude", attack="read")
    _assert_other_user_cannot_read(result)


def test_codex_worker_does_not_write_chunks_through_planted_symlinks(tmp_path: Path) -> None:
    result = _run_worker(tmp_path, which="codex", attack="symlink")
    _assert_symlinks_untouched(result)


def test_codex_worker_does_not_leave_chunks_readable_by_another_user(tmp_path: Path) -> None:
    result = _run_worker(tmp_path, which="codex", attack="read")
    _assert_other_user_cannot_read(result)
