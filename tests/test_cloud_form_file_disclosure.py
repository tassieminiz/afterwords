"""Push must not let pulled voice metadata read local files through curl -F.

`afterwords pull` stores cloud `name`, `ref_text`, `backend`, and `family` on
the voice profile. `afterwords push` then passes those strings to curl as
`-F name=value`. Curl treats a value that starts with `<` or `@` as a local
file and uploads that file's bytes.

These tests run that pull-then-push path against a local stand-in of the
cloud API. They fail while a pushed text field is replaced by the contents
of a local file.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import select
import struct
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
AFTERWORDS = REPO / "afterwords.sh"

# inotify masks. A pull that refuses must not create or remove voice files first.
_IN_MODIFY = 0x00000002
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_VOICE_WRITE_MASK = (
    _IN_MODIFY | _IN_CLOSE_WRITE | _IN_MOVED_FROM | _IN_MOVED_TO | _IN_CREATE | _IN_DELETE
)

# Distinct from any path string placed in voice metadata, so a body match
# means the file itself was read.
SENTINEL = b"LOCAL_FILE_SENTINEL_b465c595_not_a_credential\n"
BENIGN_TEXT = "A short benign transcript for the gallery voice."
API_KEY = "aw_test_placeholder_not_a_secret"
GALLERY_NAME = "gallery-voice"
GALLERY_FAMILY = "gallery-voice"

_WAV = (
    b"RIFF$\x00\x00\x00WAVEfmt "
    b"\x10\x00\x00\x00\x01\x00\x01\x00\x80\x5e\x00\x00\x00\xbc\x00\x00"
    b"\x02\x00\x10\x00data\x00\x00\x00\x00"
)


class _CloudHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path.endswith("/audio"):
            self._send(200, "audio/wav", self.server.audio)  # type: ignore[attr-defined]
            return
        if path.startswith("/v1/voices/"):
            body = json.dumps(self.server.meta).encode()  # type: ignore[attr-defined]
            self._send(200, "application/json", body)
            return
        self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.posts.append(  # type: ignore[attr-defined]
            {
                "path": self.path.split("?", 1)[0],
                "content_type": self.headers.get("Content-Type", ""),
                "body": body,
            }
        )
        self._send(201, "application/json", b'{"voice_id":"returned-voice-id"}')


class Cloud:
    def __init__(self, meta: dict) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _CloudHandler)
        self._server.meta = meta  # type: ignore[attr-defined]
        self._server.audio = _WAV  # type: ignore[attr-defined]
        self._server.posts = []  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def posts(self) -> list[dict]:
        return self._server.posts  # type: ignore[attr-defined]

    def __enter__(self) -> "Cloud":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


def _voices_snapshot(voices: Path) -> dict[str, bytes]:
    return {
        path.relative_to(voices).as_posix(): path.read_bytes()
        for path in voices.rglob("*")
        if path.is_file()
    }


def _write_gallery(voices: Path) -> None:
    voices.mkdir(parents=True, exist_ok=True)
    (voices / f"{GALLERY_NAME}-ref.wav").write_bytes(_WAV)
    (voices / f"{GALLERY_NAME}.json").write_text(
        json.dumps(
            {
                "name": GALLERY_NAME,
                "reference_audio": f"{GALLERY_NAME}-ref.wav",
                "reference_text": BENIGN_TEXT,
                "backend": "qwen3-0.6b",
                "family": GALLERY_FAMILY,
            },
            indent=2,
        )
        + "\n"
    )


def _form_parts(content_type: str, body: bytes) -> dict[str, bytes]:
    match = re.search(r"boundary=([^;\s]+)", content_type, re.IGNORECASE)
    assert match, content_type
    boundary = match.group(1).strip('"')
    parts: dict[str, bytes] = {}
    for chunk in body.split(("--" + boundary).encode()):
        if b"Content-Disposition" not in chunk:
            continue
        header, _, data = chunk.partition(b"\r\n\r\n")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        name_match = re.search(br'name="([^"]*)"', header)
        if not name_match:
            continue
        parts[name_match.group(1).decode()] = data
    return parts


class _VoiceDirWatch:
    """Record create, write, and delete events under a directory."""

    def __init__(self, directory: Path) -> None:
        self.events: list[str] = []
        libc = ctypes.CDLL(None, use_errno=True)
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        add = libc.inotify_add_watch
        add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add.restype = ctypes.c_int
        self._fd = init(0x00080000)  # IN_CLOEXEC
        if self._fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        if add(self._fd, os.fsencode(directory), _IN_VOICE_WRITE_MASK) < 0:
            os.close(self._fd)
            raise OSError(ctypes.get_errno(), "inotify_add_watch")
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=1)

    def _loop(self) -> None:
        self._ready.set()
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.05)
            if not ready:
                continue
            try:
                buf = os.read(self._fd, 4096)
            except OSError:
                return
            offset = 0
            while offset + 16 <= len(buf):
                _, mask, _, length = struct.unpack_from("iIII", buf, offset)
                name = buf[offset + 16 : offset + 16 + length].split(b"\x00", 1)[0]
                self.events.append(f"{mask:#x}:{name.decode(errors='replace')}")
                offset += 16 + length

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        os.close(self._fd)


def _run(args: list[str], repo: Path, cloud_url: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(repo / "home")
    env["AFTERWORDS_REPO_DIR"] = str(repo)
    env["AFTERWORDS_API_KEY"] = API_KEY
    env["AFTERWORDS_CLOUD_URL"] = cloud_url
    (repo / "home").mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["bash", str(AFTERWORDS), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _profiles(voices: Path) -> list[dict]:
    found = []
    for path in sorted(voices.glob("*.json")):
        found.append(json.loads(path.read_text()))
    return found


def _assert_push_does_not_read_local_files(
    posts: list[dict],
    voices: Path,
    sentinel_file: Path,
    *,
    expect_gallery: bool,
) -> None:
    assert sentinel_file.read_bytes() == SENTINEL
    assert posts, "push made no request, so the text-field assertion did not run"
    for post in posts:
        assert post["path"] == "/v1/voices"
        assert SENTINEL not in post["body"]
        parts = _form_parts(post["content_type"], post["body"])
        assert "name" in parts
        matched = [profile for profile in _profiles(voices) if profile.get("name", "").encode() == parts["name"]]
        assert matched, f"uploaded name is not the profile text: {parts['name']!r}"
        profile = matched[0]
        expected = {
            "name": profile.get("name", ""),
            "ref_text": profile.get("reference_text") or "",
            "backend": profile.get("backend") or "qwen3-0.6b",
            "family": profile.get("family") or "",
        }
        for field, value in expected.items():
            if field == "ref_text" and not value:
                assert field not in parts
                continue
            if field == "family" and not value:
                assert field not in parts
                continue
            assert field in parts, field
            assert parts[field] == value.encode(), field
    if expect_gallery:
        assert any(BENIGN_TEXT.encode() in post["body"] for post in posts)


def _exercise(
    tmp_path: Path,
    meta: dict,
    push_name: str,
    sentinel_file: Path,
    *,
    expect_gallery: bool,
) -> None:
    repo = tmp_path / "repo"
    voices = repo / "voices"
    _write_gallery(voices)
    before = _voices_snapshot(voices)
    with Cloud(meta) as cloud:
        watch = _VoiceDirWatch(voices)
        try:
            pull = _run(["pull", "shared-voice"], repo, cloud.url, tmp_path)
        finally:
            watch.close()
        if pull.returncode != 0:
            # Refusing the pull is safe only when it has not already created,
            # replaced, or removed a voice file, including one it then deletes.
            assert _voices_snapshot(voices) == before, pull.stdout + pull.stderr
            assert not watch.events, pull.stdout + pull.stderr
            assert not cloud.posts, pull.stdout + pull.stderr
        push = _run(["push", push_name], repo, cloud.url, tmp_path)
        leaked = [post for post in cloud.posts if SENTINEL in post["body"]]
        assert not leaked, push.stdout + push.stderr
        if expect_gallery:
            assert any(BENIGN_TEXT.encode() in post["body"] for post in cloud.posts), (
                push.stdout + push.stderr
            )
        if cloud.posts:
            _assert_push_does_not_read_local_files(
                cloud.posts,
                voices,
                sentinel_file,
                expect_gallery=expect_gallery,
            )


@pytest.mark.parametrize("prefix", ["<", "@"])
def test_family_push_does_not_upload_file_named_by_pulled_ref_text(prefix, tmp_path: Path) -> None:
    """Pushing a gallery voice also uploads same-family profiles from pull."""
    secret = tmp_path / "private" / "marker.txt"
    secret.parent.mkdir()
    secret.write_bytes(SENTINEL)
    meta = {
        "name": "planted-voice",
        "ref_text": f"{prefix}{secret}",
        "backend": "qwen3-0.6b",
        "family": GALLERY_FAMILY,
    }
    _exercise(tmp_path, meta, GALLERY_NAME, secret, expect_gallery=True)


@pytest.mark.parametrize("prefix", ["<", "@"])
def test_family_push_relative_ref_text_does_not_upload_dotenv(prefix, tmp_path: Path) -> None:
    """A pulled ref_text of `<.env` or `@.env` is resolved against the cwd."""
    secret = tmp_path / ".env"
    secret.write_bytes(SENTINEL)
    meta = {
        "name": "planted-voice",
        "ref_text": f"{prefix}.env",
        "backend": "qwen3-0.6b",
        "family": GALLERY_FAMILY,
    }
    _exercise(tmp_path, meta, GALLERY_NAME, secret, expect_gallery=True)


@pytest.mark.parametrize("prefix", ["<", "@"])
def test_family_push_does_not_upload_file_named_by_pulled_backend(prefix, tmp_path: Path) -> None:
    secret = tmp_path / "private" / "marker.txt"
    secret.parent.mkdir()
    secret.write_bytes(SENTINEL)
    meta = {
        "name": "planted-voice",
        "ref_text": BENIGN_TEXT,
        "backend": f"{prefix}{secret}",
        "family": GALLERY_FAMILY,
    }
    _exercise(tmp_path, meta, GALLERY_NAME, secret, expect_gallery=True)


@pytest.mark.parametrize("prefix", ["<", "@"])
def test_family_push_does_not_upload_file_named_by_pulled_name(prefix, tmp_path: Path) -> None:
    """The profile filename and the form name both come from the pulled name."""
    secret = tmp_path / "marker.txt"
    secret.write_bytes(SENTINEL)
    meta = {
        "name": f"{prefix}marker.txt",
        "ref_text": BENIGN_TEXT,
        "backend": "qwen3-0.6b",
        "family": GALLERY_FAMILY,
    }
    _exercise(tmp_path, meta, GALLERY_NAME, secret, expect_gallery=True)


@pytest.mark.parametrize("prefix", ["<", "@"])
def test_push_does_not_upload_file_named_by_pulled_family(prefix, tmp_path: Path) -> None:
    """Family is the payload, so the victim pushes the pulled voice by name."""
    secret = tmp_path / "private" / "marker.txt"
    secret.parent.mkdir()
    secret.write_bytes(SENTINEL)
    meta = {
        "name": "planted-voice",
        "ref_text": BENIGN_TEXT,
        "backend": "qwen3-0.6b",
        "family": f"{prefix}{secret}",
    }
    _exercise(tmp_path, meta, "planted-voice", secret, expect_gallery=False)
