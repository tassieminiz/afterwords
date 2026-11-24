"""Cloud voice pull must not store or load an unbounded reference clip.

These tests fail while a pull of an oversized audio body is written under
voices/ and a later startup or reload reads that file into one buffer.
They pass once the download is capped and an oversized reference is refused
before its contents are read.

The ceiling is server.MAX_CLONE_UPLOAD_BYTES (25 MiB). That is already the
product's own limit for reference audio accepted by /clone, and it is far
above a short reference clip. A body or file larger than that ceiling is the
same missing check that accepts a multi-gigabyte response.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import backends
import server
from backends.base import PreparedVoice, RefTextPolicy, _read_only
from backends.f5_tts import F5TTSBackend
from backends.qwen3 import Qwen3Backend
from backends.xtts_v2 import XTTSv2Backend

REPO = Path(__file__).resolve().parent.parent
AFTERWORDS = REPO / "afterwords.sh"

# Probe of cap+1 is how a bounded reader detects "one byte too many".
# A full read of the fixture (cap + OVERSHOOT) is larger than this.
PROBE_SLACK = 1
OVERSHOOT = 64 * 1024
API_KEY_PLACEHOLDER = "aw_test_placeholder"
VOICE_ID = "voice-under-test"
REF_TEXT = "this is a short reference transcript"


def _ceiling() -> int:
    """Largest reference body the product may accept.

    Tied to the clone upload cap so a fix cannot raise that constant and
    keep an unbounded pull. The cap is already documented as far beyond the
    audio the cloning path needs.
    """
    cap = server.MAX_CLONE_UPLOAD_BYTES
    assert cap <= 25 * 1024 * 1024
    assert cap >= 64 * 1024
    return cap


def _write_sparse(path: Path, size: int) -> None:
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, size)
        os.lseek(fd, size - 1, os.SEEK_SET)
        os.write(fd, b"Z")
    finally:
        os.close(fd)


def _profile(directory: Path, name: str, backend: str) -> Path:
    payload = {
        "name": name,
        "reference_audio": f"{name}-ref.wav",
        "reference_text": REF_TEXT,
        "backend": backend,
    }
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload) + "\n")
    return path


class _ReadTracker:
    """Count bytes read from one reference path. Optionally extend it on open.

    Extending on open is the check-then-use case: a size check that runs
    before open still sees the short file, then the read observes the bytes
    appended for the same inode.
    """

    def __init__(self, target: Path, grow: bool):
        self.target = os.path.realpath(target)
        self.grow = grow
        self.grew = False
        self.bytes = 0
        self._real_open = open

    def __enter__(self):
        tracker = self

        def tracking_open(file, *args, **kwargs):
            fh = tracker._real_open(file, *args, **kwargs)
            try:
                path = os.path.realpath(file)
            except (TypeError, OSError, ValueError):
                return fh
            if path != tracker.target:
                return fh
            if tracker.grow and not tracker.grew:
                tracker.grew = True
                remaining = _ceiling() + OVERSHOOT
                chunk = b"G" * (1024 * 1024)
                with tracker._real_open(tracker.target, "ab", buffering=0) as extra:
                    while remaining:
                        n = min(remaining, len(chunk))
                        extra.write(chunk[:n])
                        remaining -= n

            original = fh.read

            def read(*a, **k):
                data = original(*a, **k)
                if isinstance(data, (bytes, bytearray)):
                    tracker.bytes += len(data)
                return data

            fh.read = read  # type: ignore[method-assign]
            return fh

        # Backends call the builtin. Patch that, not a local binding.
        import builtins
        tracker._builtins = builtins
        builtins.open = tracking_open
        return tracker

    def __exit__(self, *exc):
        self._builtins.open = self._real_open
        return False


def _assert_bounded_read(name: str, tracker: _ReadTracker) -> None:
    cap = _ceiling()
    assert tracker.bytes <= cap + PROBE_SLACK, (
        f"read {tracker.bytes} bytes from {name}; limit is {cap}"
    )
    profile = server.VOICES.get(name)
    if profile is not None and profile.prepared.data is not None:
        assert len(profile.prepared.data) <= cap + PROBE_SLACK


def _assert_not_loaded(name: str, tracker: _ReadTracker) -> None:
    _assert_bounded_read(name, tracker)
    assert name not in server.VOICES


def _use_backend(backend) -> None:
    backends.reset_for_tests()
    backends.register(backend)


@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    directory = tmp_path / "voices"
    directory.mkdir()
    monkeypatch.setattr(server, "_VOICES_DIR", str(directory))
    previous = dict(server.VOICES)
    server.VOICES.clear()
    yield directory
    server.VOICES.clear()
    server.VOICES.update(previous)


def _load_oversized(directory: Path, backend, name: str, grow: bool) -> _ReadTracker:
    ref = directory / f"{name}-ref.wav"
    if grow:
        ref.write_bytes(b"short-reference")
    else:
        _write_sparse(ref, _ceiling() + OVERSHOOT)
    _profile(directory, name, backend.name)
    _use_backend(backend)
    with _ReadTracker(ref, grow=grow) as tracker:
        try:
            server._load_voice_profiles()
        except Exception:
            # A refusal may surface as an exception. The bytes already read
            # are what the assertions measure.
            pass
    return tracker


@pytest.mark.parametrize(
    "backend_factory",
    [
        lambda: Qwen3Backend("0.6B"),
        lambda: Qwen3Backend("1.7B"),
        F5TTSBackend,
        XTTSv2Backend,
    ],
)
def test_startup_load_refuses_oversized_reference(voices_dir, backend_factory):
    """Server startup reads every profile through prepare_voice."""
    backend = backend_factory()
    tracker = _load_oversized(voices_dir, backend, "oversized", grow=False)
    _assert_not_loaded("oversized", tracker)


class _SpyBackend:
    """prepare_voice reads the whole reference, as the model-backed backends do
    once their weights are loaded (OpenVoice, MockingBird, SV2TTS, NeuTTS, SoproTTS).
    """

    name = "spy-buffer"
    display_name = "spy"
    sample_rate = 24000
    ref_text_policy = RefTextPolicy.OPTIONAL
    supported_langs = ("en",)

    def load(self) -> None:
        return None

    def validate_extras(self, extras) -> None:
        return None

    def prepare_voice(self, ref_audio_path, ref_text, extras):
        with open(ref_audio_path, "rb") as fh:
            data = fh.read()
        return PreparedVoice(
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            extras=_read_only({}),
            data=data,
        )

    def synthesize(self, text, prepared, lang):
        raise AssertionError("synthesize is not part of this path")


@pytest.mark.parametrize(
    "backend_factory",
    [
        lambda: Qwen3Backend("0.6B"),
        F5TTSBackend,
        XTTSv2Backend,
        _SpyBackend,
    ],
)
def test_startup_load_bounds_read_when_reference_grows_after_size_check(voices_dir, backend_factory):
    """A size check before open must not be followed by an unbounded read.

    The file is short at stat time. Opening it appends past the ceiling on
    the same inode, which is what an unbounded fh.read() then consumes.
    Reading only the size observed before that open is within the ceiling.
    """
    backend = backend_factory()
    tracker = _load_oversized(voices_dir, backend, "grows", grow=True)
    assert tracker.grew
    _assert_bounded_read("grows", tracker)


def test_startup_load_refuses_oversized_reference_for_any_backend(voices_dir):
    """The refusal has to happen for every backend, not only Qwen3.

    OpenVoice, MockingBird, SV2TTS, NeuTTS Air, and SoproTTS read the
    reference only after their model has loaded. The spy has that shape:
    prepare_voice reads the whole path. Its model code is not what decides
    the size; the server's call into prepare_voice is.
    """
    tracker = _load_oversized(voices_dir, _SpyBackend(), "spied", grow=False)
    _assert_not_loaded("spied", tracker)


def test_small_reference_is_still_loaded(voices_dir):
    """A clip under the ceiling is still prepared, including its buffered bytes."""
    backend = Qwen3Backend("0.6B")
    ref = voices_dir / "small-ref.wav"
    payload = b"RIFFtiny-reference-clip"
    ref.write_bytes(payload)
    _profile(voices_dir, "small", backend.name)
    _use_backend(backend)
    server._load_voice_profiles()
    assert "small" in server.VOICES
    assert server.VOICES["small"].prepared.data == payload


def test_reload_does_not_commit_oversized_reference(voices_dir, client):
    """POST /reload is the path pull uses when the server is already up."""
    backend = Qwen3Backend("0.6B")
    _use_backend(backend)
    kept_wav = voices_dir / "kept-ref.wav"
    kept_wav.write_bytes(b"kept-clip")
    _profile(voices_dir, "kept", backend.name)
    server._load_voice_profiles()
    assert "kept" in server.VOICES

    ref = voices_dir / "oversized-ref.wav"
    _write_sparse(ref, _ceiling() + OVERSHOOT)
    _profile(voices_dir, "oversized", backend.name)
    server._clone_enabled = True
    try:
        with _ReadTracker(ref, grow=False) as tracker:
            response = client.post("/reload")
        _assert_not_loaded("oversized", tracker)
        assert "kept" in server.VOICES
        # Skipped profiles leave reload successful. Raising aborts the commit.
        assert response.status_code in (200, 500)
        if response.status_code == 200:
            assert "oversized" not in response.json()["reloaded"]
    finally:
        server._clone_enabled = False


class _Cloud(BaseHTTPRequestHandler):
    """Serves the voice metadata and the audio body pull requests."""

    protocol_version = "HTTP/1.1"
    mode = "content-length"
    body_size = 0
    name = "ordinary"

    def log_message(self, fmt, *args):
        return None

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == f"/v1/voices/{VOICE_ID}/audio":
            self._audio()
            return
        if path == f"/v1/voices/{VOICE_ID}":
            body = json.dumps(
                {
                    "name": self.name,
                    "ref_text": REF_TEXT,
                    "backend": "qwen3-0.6b",
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def _audio(self):
        size = type(self).body_size
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        if type(self).mode == "content-length":
            self.send_header("Content-Length", str(size))
            self.end_headers()
            self._write_body(size)
        elif type(self).mode == "chunked":
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            remaining = size
            chunk = b"\x00" * (256 * 1024)
            while remaining:
                n = min(remaining, len(chunk))
                piece = chunk[:n]
                self.wfile.write(f"{n:x}\r\n".encode() + piece + b"\r\n")
                remaining -= n
            self.wfile.write(b"0\r\n\r\n")
        else:
            self.send_header("Connection", "close")
            self.end_headers()
            self._write_body(size)

    def _write_body(self, size: int) -> None:
        remaining = size
        chunk = b"\x00" * (256 * 1024)
        try:
            while remaining:
                n = min(remaining, len(chunk))
                self.wfile.write(chunk[:n])
                remaining -= n
        except (BrokenPipeError, ConnectionResetError):
            return


def _serve_cloud(mode: str, body_size: int, name: str):
    _Cloud.mode = mode
    _Cloud.body_size = body_size
    _Cloud.name = name
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Cloud)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def _watch(directory: Path, stop: threading.Event, seen: list[int]) -> None:
    while not stop.is_set():
        for path in directory.rglob("*"):
            try:
                if path.is_file():
                    seen.append(path.stat().st_size)
            except OSError:
                pass
        time.sleep(0.002)


def _run_pull(repo: Path, cloud_url: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["AFTERWORDS_REPO_DIR"] = str(repo)
    env["AFTERWORDS_CLOUD_URL"] = cloud_url
    env["AFTERWORDS_API_KEY"] = API_KEY_PLACEHOLDER
    return subprocess.run(
        ["bash", str(AFTERWORDS), "pull", VOICE_ID],
        capture_output=True,
        text=True,
        env=env,
        timeout=45,
    )


def _assert_pull_refused(repo: Path, result: subprocess.CompletedProcess, seen: list[int], name: str, original: bytes | None):
    cap = _ceiling()
    wav = repo / "voices" / f"{name}-ref.wav"
    profile = repo / "voices" / f"{name}.json"
    wav_size = wav.stat().st_size if wav.exists() else 0
    largest = max(seen, default=0)
    assert result.returncode != 0 and wav_size <= cap and largest <= cap, (
        f"rc={result.returncode} wav={wav_size} largest_seen={largest} cap={cap}\n"
        f"{result.stdout}{result.stderr}"
    )
    if original is None:
        assert not wav.exists()
        assert not profile.exists()
    else:
        assert wav.read_bytes() == original
        assert profile.read_text() == "original-profile\n"
    # Nothing else in voices/ may hold the oversized body.
    for path in (repo / "voices").rglob("*"):
        if path.is_file():
            assert path.stat().st_size <= cap


@pytest.mark.parametrize("mode", ["content-length", "chunked", "close"])
def test_pull_refuses_oversized_audio(tmp_path, mode):
    """afterwords pull writes the audio response with only a 60s timer."""
    repo = tmp_path
    voices = repo / "voices"
    voices.mkdir()
    sentinel = voices / "sentinel.txt"
    sentinel.write_text("keep")
    httpd = _serve_cloud(mode, _ceiling() + OVERSHOOT, "ordinary")
    seen: list[int] = []
    stop = threading.Event()
    watcher = threading.Thread(target=_watch, args=(voices, stop, seen), daemon=True)
    watcher.start()
    try:
        result = _run_pull(repo, f"http://127.0.0.1:{httpd.server_address[1]}")
    finally:
        stop.set()
        watcher.join(timeout=2)
        httpd.shutdown()
    _assert_pull_refused(repo, result, seen, "ordinary", None)
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("mode", ["content-length", "chunked", "close"])
def test_pull_refuses_oversized_audio_without_clobbering_existing_voice(tmp_path, mode):
    """A refused pull must not truncate a voice that already uses that name.

    curl -o on the destination truncates the file as the body arrives. The
    metadata name is chosen by the server, so it can be an existing slug.
    """
    repo = tmp_path
    voices = repo / "voices"
    voices.mkdir()
    original = b"original-reference-bytes"
    (voices / "ordinary-ref.wav").write_bytes(original)
    (voices / "ordinary.json").write_text("original-profile\n")
    httpd = _serve_cloud(mode, _ceiling() + OVERSHOOT, "ordinary")
    seen: list[int] = []
    stop = threading.Event()
    watcher = threading.Thread(target=_watch, args=(voices, stop, seen), daemon=True)
    watcher.start()
    try:
        result = _run_pull(repo, f"http://127.0.0.1:{httpd.server_address[1]}")
    finally:
        stop.set()
        watcher.join(timeout=2)
        httpd.shutdown()
    _assert_pull_refused(repo, result, seen, "ordinary", original)


def test_pull_accepts_a_short_reference(tmp_path):
    repo = tmp_path
    voices = repo / "voices"
    voices.mkdir()
    body = b"RIFF" + b"\x00" * 2048
    httpd = _serve_cloud("content-length", len(body), "ordinary")
    # The handler streams NULs of body_size, not `body`. Match that.
    try:
        result = _run_pull(repo, f"http://127.0.0.1:{httpd.server_address[1]}")
    finally:
        httpd.shutdown()
    assert result.returncode == 0, result.stdout + result.stderr
    wav = voices / "ordinary-ref.wav"
    profile = voices / "ordinary.json"
    assert wav.is_file()
    assert wav.stat().st_size == len(body)
    saved = json.loads(profile.read_text())
    assert saved["name"] == "ordinary"
    assert saved["reference_audio"] == "ordinary-ref.wav"
    assert saved["reference_text"] == REF_TEXT
    assert saved["backend"] == "qwen3-0.6b"
    assert saved["cloud_voice_id"] == VOICE_ID


def test_pulled_oversized_reference_is_not_loaded_on_startup(tmp_path, voices_dir):
    """The file pull leaves behind is what the next server start reads.

    voices_dir is the server's voices directory. Point the CLI at the same
    tree, pull an oversized body, then run the startup loader.
    """
    # voices_dir is tmp_path/voices from the fixture's tmp_path, which is a
    # different tmp_path than this test's first argument. Use voices_dir's parent.
    repo = voices_dir.parent
    httpd = _serve_cloud("chunked", _ceiling() + OVERSHOOT, "ordinary")
    try:
        result = _run_pull(repo, f"http://127.0.0.1:{httpd.server_address[1]}")
    finally:
        httpd.shutdown()
    ref = voices_dir / "ordinary-ref.wav"
    backend = Qwen3Backend("0.6B")
    _use_backend(backend)
    if ref.exists():
        with _ReadTracker(ref, grow=False) as tracker:
            try:
                server._load_voice_profiles()
            except Exception:
                pass
        _assert_not_loaded("ordinary", tracker)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not ref.exists() or ref.stat().st_size <= _ceiling()
    if ref.exists():
        assert "ordinary" not in server.VOICES
