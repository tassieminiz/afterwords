"""Containment spec for `afterwords pull`.

`cmd_pull` in afterwords.sh builds the local profile and reference-audio
paths from the `name` field of the cloud metadata response. These tests
drive that command against a local stand-in for the configured cloud and
require every filesystem operation in the pull (audio download, the
delete-on-error cleanup, and the profile write) to leave anything outside
`voices/` untouched.

They fail while the name is concatenated into the destination and pass
once a pull whose name is not a slug, or whose resolved destination leaves
`voices/`, is refused before any of those operations.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import select
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
AFTERWORDS = REPO / "afterwords.sh"
VOICE_ID = "voice1"
API_KEY = "aw_test_placeholder"
ORIGINAL_WAV = b"ORIGINAL-WAV-BYTES"
ORIGINAL_JSON = b'{"keep":true}\n'
ATTACK_WAV = b"ATTACK-WAV-BYTES"

# glibc sys/inotify.h — events a directory watch reports for a child write,
# rename, or delete. A refusal must not produce any of them outside voices/.
_IN_MODIFY = 0x00000002
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_WATCH_MASK = (
    _IN_MODIFY
    | _IN_CLOSE_WRITE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_CREATE
    | _IN_DELETE
)


class _CloudServer(ThreadingHTTPServer):
    """Serves the two routes cmd_pull requests, and nothing else."""

    def __init__(self, meta: dict, audio: bytes, audio_status: int):
        super().__init__(("127.0.0.1", 0), _CloudHandler)
        self.meta = meta
        self.audio = audio
        self.audio_status = audio_status
        self.hits = {"meta": 0, "audio": 0}
        self.lock = threading.Lock()


class _CloudHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        srv: _CloudServer = self.server  # type: ignore[assignment]
        path = self.path.split("?", 1)[0]
        if path == f"/v1/voices/{VOICE_ID}":
            with srv.lock:
                srv.hits["meta"] += 1
            body = json.dumps(srv.meta).encode()
            self._send(200, body, "application/json")
        elif path == f"/v1/voices/{VOICE_ID}/audio":
            with srv.lock:
                srv.hits["audio"] += 1
            self._send(srv.audio_status, srv.audio, "audio/wav")
        else:
            self._send(404, b"missing", "text/plain")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _DirWatch:
    """Record child create/modify/delete events under one directory.

    Built from the inotify calls exported by this system's libc
    (inotify_init, inotify_add_watch) and the event struct in
    /usr/include/x86_64-linux-gnu/sys/inotify.h.
    """

    def __init__(self, directory: Path):
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        libc.inotify_init.restype = ctypes.c_int
        libc.inotify_add_watch.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        libc.inotify_add_watch.restype = ctypes.c_int
        fd = libc.inotify_init()
        if fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init")
        wd = libc.inotify_add_watch(fd, os.fsencode(directory), _WATCH_MASK)
        if wd < 0:
            os.close(fd)
            raise OSError(ctypes.get_errno(), "inotify_add_watch")
        self._fd = fd
        self.events: list[tuple[int, str]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select([self._fd], [], [], 0.05)
            if readable:
                self._drain()

    def _drain(self) -> None:
        while True:
            readable, _, _ = select.select([self._fd], [], [], 0)
            if not readable:
                return
            try:
                raw = os.read(self._fd, 65536)
            except OSError:
                return
            if not raw:
                return
            off = 0
            while off + 16 <= len(raw):
                mask = int.from_bytes(raw[off + 4 : off + 8], "little")
                name_len = int.from_bytes(raw[off + 12 : off + 16], "little")
                start = off + 16
                end = start + name_len
                if end > len(raw):
                    return
                name = raw[start:end].split(b"\x00", 1)[0].decode("utf-8", "replace")
                off = end
                if name:
                    with self._lock:
                        self.events.append((mask, name))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._drain()
        os.close(self._fd)

    def names(self) -> list[str]:
        with self._lock:
            return [name for _mask, name in self.events]


def _start_cloud(meta: dict, audio: bytes = ATTACK_WAV, audio_status: int = 200):
    srv = _CloudServer(meta, audio, audio_status)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv


def _stop_cloud(srv: _CloudServer) -> None:
    srv.shutdown()
    srv.server_close()


def _meta(name: str) -> dict:
    return {
        "name": name,
        "ref_text": "attacker transcript",
        "backend": "qwen3-0.6b",
        "family": "attacker-family",
        "lang": "en",
    }


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    voices = repo / "voices"
    outside = tmp_path / "outside"
    voices.mkdir(parents=True)
    outside.mkdir()
    return repo, voices, outside


def _escape_name(voices: Path, outside_file: Path) -> str:
    """Metadata name which, concatenated onto voices/, resolves to outside_file.

    outside_file is the path without the suffix cmd_pull appends.
    """
    return os.path.relpath(outside_file, voices)


def _run_pull(repo: Path, base_url: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["AFTERWORDS_REPO_DIR"] = str(repo)
    env["AFTERWORDS_CLOUD_URL"] = base_url
    env["AFTERWORDS_API_KEY"] = API_KEY
    return subprocess.run(
        ["bash", str(AFTERWORDS), "pull", VOICE_ID],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )


def _snapshot(root: Path) -> dict[str, object]:
    found: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_symlink():
            found[rel] = ("symlink", os.readlink(path))
        elif path.is_file():
            found[rel] = ("file", path.read_bytes())
        elif path.is_dir():
            found[rel] = ("dir",)
    return found


def _explain(result: subprocess.CompletedProcess[str], outside: Path) -> str:
    listing = []
    for path in sorted(outside.rglob("*")):
        if path.is_file() or path.is_symlink():
            listing.append(f"{path.name}={path.read_bytes()!r}")
    return (
        f"rc={result.returncode}\n"
        f"stdout={result.stdout}\n"
        f"stderr={result.stderr}\n"
        f"outside={listing}"
    )


def _assert_metadata_was_fetched(srv: _CloudServer, result) -> None:
    assert srv.hits["meta"] >= 1, (
        "pull never requested voice metadata, so this did not exercise cmd_pull\n"
        + _explain(result, Path("."))
    )


def _arm_watch(outside: Path) -> _DirWatch:
    """Watch `outside` and prove the watch sees a write before the pull."""
    watch = _DirWatch(outside)
    probe = outside / ".probe"
    probe.write_bytes(b"probe")
    probe.unlink()
    deadline = time.monotonic() + 2
    while watch.names().count(".probe") < 2:
        if time.monotonic() > deadline:
            seen = watch.names()
            watch.close()
            pytest.fail(f"inotify watch did not observe the probe write: {seen}")
    time.sleep(0.2)
    watch._drain()
    with watch._lock:
        watch.events.clear()
    return watch


@pytest.fixture
def cloud_pull(tmp_path: Path):
    """Repo, voices/, outside dir, and a cloud stand-in bound to 127.0.0.1."""
    repo, voices, outside = _layout(tmp_path)
    holder: dict[str, _CloudServer] = {}

    def start(meta: dict, audio: bytes = ATTACK_WAV, audio_status: int = 200):
        srv = _start_cloud(meta, audio, audio_status)
        holder["srv"] = srv
        return srv

    yield repo, voices, outside, start
    if "srv" in holder:
        _stop_cloud(holder["srv"])


def test_pull_metadata_name_cannot_overwrite_files_outside_voices(cloud_pull):
    """A name of ../ segments must not replace an existing file outside voices/."""
    repo, voices, outside, start = cloud_pull
    target = outside / "kept"
    wav = Path(str(target) + "-ref.wav")
    profile = Path(str(target) + ".json")
    wav.write_bytes(ORIGINAL_WAV)
    profile.write_bytes(ORIGINAL_JSON)
    name = _escape_name(voices, target)
    assert ".." in name
    srv = start(_meta(name))
    watch = _arm_watch(outside)
    try:
        result = _run_pull(repo, f"http://127.0.0.1:{srv.server_address[1]}")
    finally:
        watch.close()
    _assert_metadata_was_fetched(srv, result)
    detail = _explain(result, outside)
    assert result.returncode != 0, detail
    assert wav.read_bytes() == ORIGINAL_WAV, detail
    assert profile.read_bytes() == ORIGINAL_JSON, detail
    assert watch.names() == [], detail


def test_pull_metadata_name_cannot_create_files_outside_voices(cloud_pull):
    """The same name must not create a new audio file or profile outside voices/."""
    repo, voices, outside, start = cloud_pull
    target = outside / "planted"
    wav = Path(str(target) + "-ref.wav")
    profile = Path(str(target) + ".json")
    name = _escape_name(voices, target)
    before_voices = _snapshot(voices)
    srv = start(_meta(name))
    watch = _arm_watch(outside)
    try:
        result = _run_pull(repo, f"http://127.0.0.1:{srv.server_address[1]}")
    finally:
        watch.close()
    _assert_metadata_was_fetched(srv, result)
    detail = _explain(result, outside)
    assert result.returncode != 0, detail
    assert not wav.exists(), detail
    assert not profile.exists(), detail
    assert _snapshot(voices) == before_voices, detail
    assert watch.names() == [], detail


def test_pull_parent_segments_under_a_prefix_cannot_escape(cloud_pull):
    """A slash prefix in front of ../ is still an escape and must not be written."""
    repo, voices, outside, start = cloud_pull
    target = outside / "nested"
    wav = Path(str(target) + "-ref.wav")
    profile = Path(str(target) + ".json")
    # The intermediate directory has to exist or the kernel rejects the walk
    # before ".." is applied, which would hide the escape.
    (voices / "prefix").mkdir()
    name = "prefix/" + os.path.relpath(target, voices / "prefix")
    assert name.startswith("prefix/") and ".." in name
    before = _snapshot(outside)
    before_voices = _snapshot(voices)
    srv = start(_meta(name))
    watch = _arm_watch(outside)
    try:
        result = _run_pull(repo, f"http://127.0.0.1:{srv.server_address[1]}")
    finally:
        watch.close()
    _assert_metadata_was_fetched(srv, result)
    detail = _explain(result, outside)
    assert result.returncode != 0, detail
    assert not wav.exists(), detail
    assert not profile.exists(), detail
    assert _snapshot(outside) == before, detail
    assert _snapshot(voices) == before_voices, detail
    assert watch.names() == [], detail


def test_pull_audio_error_cannot_delete_file_outside_voices(cloud_pull):
    """A non-200 audio response must not unlink a path the name resolved outside."""
    repo, voices, outside, start = cloud_pull
    target = outside / "kept"
    wav = Path(str(target) + "-ref.wav")
    profile = Path(str(target) + ".json")
    wav.write_bytes(ORIGINAL_WAV)
    profile.write_bytes(ORIGINAL_JSON)
    name = _escape_name(voices, target)
    srv = start(_meta(name), audio=b"nope", audio_status=404)
    watch = _arm_watch(outside)
    try:
        result = _run_pull(repo, f"http://127.0.0.1:{srv.server_address[1]}")
    finally:
        watch.close()
    _assert_metadata_was_fetched(srv, result)
    detail = _explain(result, outside)
    assert wav.is_file() and wav.read_bytes() == ORIGINAL_WAV, detail
    assert profile.is_file() and profile.read_bytes() == ORIGINAL_JSON, detail
    assert result.returncode != 0, detail
    assert watch.names() == [], detail


def test_pull_slug_whose_destination_is_a_symlink_cannot_leave_voices(cloud_pull):
    """A slug is not enough when the opened path resolves outside voices/.

    cmd_pull opens voices/<name>-ref.wav and voices/<name>.json. If those
    entries are symlinks, the write follows them. Checking the name text and
    then opening that path still escapes; the opened path has to be the path
    that was contained.
    """
    repo, voices, outside, start = cloud_pull
    wav = outside / "kept-ref.wav"
    profile = outside / "kept.json"
    wav.write_bytes(ORIGINAL_WAV)
    profile.write_bytes(ORIGINAL_JSON)
    wav_link = voices / "escape-ref.wav"
    profile_link = voices / "escape.json"
    wav_link.symlink_to(wav)
    profile_link.symlink_to(profile)
    before_voices = _snapshot(voices)
    srv = start(_meta("escape"))
    watch = _arm_watch(outside)
    try:
        result = _run_pull(repo, f"http://127.0.0.1:{srv.server_address[1]}")
    finally:
        watch.close()
    _assert_metadata_was_fetched(srv, result)
    detail = _explain(result, outside)
    assert result.returncode != 0, detail
    assert wav.read_bytes() == ORIGINAL_WAV, detail
    assert profile.read_bytes() == ORIGINAL_JSON, detail
    assert wav_link.is_symlink() and os.readlink(wav_link) == str(wav), detail
    assert profile_link.is_symlink() and os.readlink(profile_link) == str(profile), detail
    assert _snapshot(voices) == before_voices, detail
    assert watch.names() == [], detail


def test_pull_safe_slug_writes_only_inside_voices(cloud_pull):
    """A letter/digit/hyphen name still pulls, and only into voices/."""
    repo, voices, outside, start = cloud_pull
    outside_before = _snapshot(outside)
    srv = start(_meta("safe-voice"))
    watch = _arm_watch(outside)
    try:
        result = _run_pull(repo, f"http://127.0.0.1:{srv.server_address[1]}")
    finally:
        watch.close()
    detail = _explain(result, outside)
    _assert_metadata_was_fetched(srv, result)
    assert srv.hits["audio"] == 1, detail
    assert result.returncode == 0, detail
    wav = voices / "safe-voice-ref.wav"
    profile = voices / "safe-voice.json"
    assert wav.read_bytes() == ATTACK_WAV, detail
    saved = json.loads(profile.read_text())
    assert saved["name"] == "safe-voice"
    assert saved["reference_audio"] == "safe-voice-ref.wav"
    assert saved["reference_text"] == "attacker transcript"
    assert saved["backend"] == "qwen3-0.6b"
    assert saved["family"] == "attacker-family"
    assert saved["cloud_voice_id"] == VOICE_ID
    assert "lang" not in saved
    assert _snapshot(outside) == outside_before, detail
    assert watch.names() == [], detail
