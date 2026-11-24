"""Cloud pull must not plant a profile path that push will execute.

`afterwords push` reads every family member (and every same-name member when
the pushed profile has no family) by splicing the profile path into a
single-quoted open() inside `python3 -c`. A path containing a single quote
is Python source. `afterwords pull` writes `voices/<name>.json` from the
cloud metadata `name` with no slug check, so a pulled name is enough to
plant that path.

These tests drive the real `afterwords.sh` pull and push commands. They
fail while that splice is still there, and they pass once a profile path
is only ever data (argv or an environment variable) and a pulled name
cannot create a file outside a lowercase slug.
"""
from __future__ import annotations

import json
import os
import select
import struct
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parent.parent
AFTERWORDS = REPO / "afterwords.sh"

# Closes the single-quoted open() in afterwords.sh and runs touch on $MARKER.
# The marker directory is injected at runtime from the environment so the
# filename itself contains no slash (a slash would be a different path).
# Closes a single-quoted open('…') in the python -c string.
PAYLOAD_STEM = (
    "'+(__import__('os').system('touch '+__import__('os').environ['MARKER']) or 'x')+'"
)
# Closes a double-quoted open("…") if the same path is only flipped to the other quote.
PAYLOAD_STEM_DQ = (
    '"+(__import__(\'os\').system(\'touch \'+__import__(\'os\').environ[\'MARKER\']) or \'x\')+"'
)
PAYLOAD_STEMS = (PAYLOAD_STEM, PAYLOAD_STEM_DQ)
API_KEY = "aw_test_placeholder"
VOICE_ID = "vwtest1"
WAV_BYTES = b"RIFF" + b"\x00" * 32


def _files(root: Path) -> set[str]:
    return {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
    }


def _write_profile(voices: Path, filename: str, payload: dict) -> None:
    (voices / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seed_picard(repo: Path, *, family: str | None) -> None:
    voices = repo / "voices"
    voices.mkdir(parents=True, exist_ok=True)
    profile = {
        "name": "picard",
        "backend": "qwen3-0.6b",
        "reference_audio": "picard-ref.wav",
        "reference_text": "Make it so.",
    }
    if family is not None:
        profile["family"] = family
    _write_profile(voices, "picard.json", profile)
    (voices / "picard-ref.wav").write_bytes(WAV_BYTES)


def _plant_payload_profile(repo: Path, *, family: str | None, logical_name: str) -> None:
    """Place voices/*.json files whose filenames are the Python payloads."""
    profile = {
        "name": logical_name,
        "backend": "qwen3-0.6b",
        "reference_audio": "missing-ref.wav",
        "reference_text": "planted",
    }
    if family is not None:
        profile["family"] = family
    for stem in PAYLOAD_STEMS:
        _write_profile(repo / "voices", stem + ".json", profile)


class _CloudState:
    def __init__(self, metas: dict[str, dict]):
        self.metas = metas
        self.gets: list[str] = []
        self.posts: list[str] = []
        self.auths: list[str] = []


def _start_cloud(meta: dict | None = None, metas: dict[str, dict] | None = None) -> tuple[ThreadingHTTPServer, _CloudState]:
    mapping = dict(metas or {})
    if meta is not None:
        mapping[VOICE_ID] = meta
    state = _CloudState(mapping)

    def _send(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    class Handler(BaseHTTPRequestHandler):
        def _auth(self) -> None:
            state.auths.append(self.headers.get("Authorization", ""))

        def do_GET(self) -> None:  # noqa: N802
            self._auth()
            path = urlparse(self.path).path
            state.gets.append(path)
            prefix = "/v1/voices/"
            if path.startswith(prefix):
                rest = path[len(prefix):]
                if rest.endswith("/audio"):
                    voice_id = rest[: -len("/audio")]
                    if voice_id in state.metas:
                        _send(self, 200, WAV_BYTES, "audio/wav")
                        return
                elif rest in state.metas:
                    body = json.dumps(state.metas[rest]).encode()
                    _send(self, 200, body, "application/json")
                    return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            self._auth()
            path = urlparse(self.path).path
            state.posts.append(path)
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length:
                self.rfile.read(length)
            body = b'{"voice_id":"vid-placeholder"}'
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, state


def _run(repo: Path, marker: Path, cloud_url: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({
        "AFTERWORDS_REPO_DIR": str(repo),
        "AFTERWORDS_API_KEY": API_KEY,
        "AFTERWORDS_CLOUD_URL": cloud_url,
        "MARKER": str(marker),
        "HOME": str(repo / "home"),
    })
    (repo / "home").mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", str(AFTERWORDS), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


class _DirWatch:
    """Record create/rename events so a file that is written and then removed still fails the test."""

    def __init__(self, directories: list[Path]):
        import ctypes
        self._ctypes = ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.inotify_init1.argtypes = [ctypes.c_int]
        libc.inotify_init1.restype = ctypes.c_int
        libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        libc.inotify_add_watch.restype = ctypes.c_int
        self._libc = libc
        # IN_NONBLOCK so draining the queue cannot wait forever after the command exits.
        self.fd = libc.inotify_init1(0x800)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        # IN_CREATE | IN_MOVED_TO | IN_CLOSE_WRITE | IN_DELETE
        mask = 0x100 | 0x80 | 0x8 | 0x200
        for directory in directories:
            wd = libc.inotify_add_watch(self.fd, os.fsencode(directory), mask)
            if wd < 0:
                raise OSError(ctypes.get_errno(), f"inotify_add_watch {directory}")
        self.events: list[tuple[int, str]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([self.fd], [], [], 0.1)
            if ready:
                self._drain()

    def _drain(self) -> None:
        while True:
            try:
                raw = os.read(self.fd, 65536)
            except BlockingIOError:
                return
            if not raw:
                return
            offset = 0
            while offset + 16 <= len(raw):
                _wd, mask, _cookie, name_len = struct.unpack_from("iIII", raw, offset)
                offset += 16
                name = raw[offset:offset + name_len].split(b"\x00", 1)[0].decode("utf-8", "replace")
                offset += name_len
                self.events.append((mask, name))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._drain()
        os.close(self.fd)


def _assert_payload_not_executed(marker: Path, result: subprocess.CompletedProcess[str]) -> None:
    assert not marker.exists(), (
        "push executed a profile filename as Python.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _assert_sibling_was_read(result: subprocess.CompletedProcess[str]) -> None:
    # Reached only after the five open() calls return the JSON fields.
    # On the vulnerable splice those calls run the payload and print nothing,
    # so this string is absent and the assertion fails. Both planted filenames
    # use the same missing reference, so a quote-flip that still interpolates
    # one of them never gets this far (the marker assertion fails first).
    assert result.stdout.count("missing-ref.wav") >= len(PAYLOAD_STEMS), (
        "push never parsed every sibling profile, so the test did not reach "
        f"the open() calls for each filename.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_push_family_sibling_filename_is_not_executed(tmp_path: Path) -> None:
    """Pushing picard must read a picard-family sibling path as data, not as code."""
    repo = tmp_path / "repo"
    marker = tmp_path / "marker"
    _seed_picard(repo, family="picard")
    _plant_payload_profile(repo, family="picard", logical_name="sibling")
    server, state = _start_cloud(meta={})
    try:
        port = server.server_address[1]
        result = _run(repo, marker, f"http://127.0.0.1:{port}", "push", "picard")
    finally:
        server.shutdown()
    _assert_payload_not_executed(marker, result)
    _assert_sibling_was_read(result)
    # The legitimate profile is still uploaded. Its reference file is real.
    assert "/v1/voices" in state.posts, result.stdout + result.stderr
    assert API_KEY not in (result.stdout + result.stderr)


def test_push_same_name_sibling_without_family_is_not_executed(tmp_path: Path) -> None:
    """With no family, push still opens every profile whose JSON name matches."""
    repo = tmp_path / "repo"
    marker = tmp_path / "marker"
    _seed_picard(repo, family=None)
    _plant_payload_profile(repo, family=None, logical_name="picard")
    server, state = _start_cloud(meta={})
    try:
        port = server.server_address[1]
        result = _run(repo, marker, f"http://127.0.0.1:{port}", "push", "picard")
    finally:
        server.shutdown()
    _assert_payload_not_executed(marker, result)
    _assert_sibling_was_read(result)
    assert "/v1/voices" in state.posts, result.stdout + result.stderr


def test_pull_does_not_write_quoted_name_and_push_does_not_run_it(tmp_path: Path) -> None:
    """A cloud name that closes either Python quote must not be written or executed."""
    repo = tmp_path / "repo"
    marker = tmp_path / "marker"
    _seed_picard(repo, family="picard")
    before = _files(repo)
    metas = {
        f"vw{i}": {
            "name": stem,
            "ref_text": "from the cloud",
            "backend": "qwen3-0.6b",
            "family": "picard",
            "lang": "en",
        }
        for i, stem in enumerate(PAYLOAD_STEMS)
    }
    server, state = _start_cloud(metas=metas)
    watch = _DirWatch([repo, repo / "voices"])
    pulled_logs: list[str] = []
    try:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}"
        for voice_id in metas:
            pulled = _run(repo, marker, url, "pull", voice_id)
            pulled_logs.append(pulled.stdout + pulled.stderr)
            assert f"/v1/voices/{voice_id}" in state.gets, pulled.stdout + pulled.stderr
        pushed = _run(repo, marker, url, "push", "picard")
    finally:
        watch.close()
        server.shutdown()

    assert state.auths and state.auths[0] == f"Bearer {API_KEY}"
    assert not marker.exists(), (
        "pull or the following push executed the cloud voice name.\n"
        f"pulls:\n{''.join(pulled_logs)}\npush stdout:\n{pushed.stdout}"
    )
    created = _files(repo) - before
    assert created == set(), (
        "pull wrote a profile for an unsanitized cloud name "
        f"(or wrote it and left it behind): {sorted(created)}\n"
        f"pulls:\n{''.join(pulled_logs)}"
    )
    # A create followed by a delete still shows up here.
    interesting = [
        name for _mask, name in watch.events
        if name and ("'" in name or '"' in name)
    ]
    assert interesting == [], f"pull created then removed {interesting}"


def test_pull_dotdot_name_does_not_write_outside_voices(tmp_path: Path) -> None:
    """A cloud name of ../escaped must not create files in the repo root."""
    repo = tmp_path / "repo"
    marker = tmp_path / "marker"
    (repo / "voices").mkdir(parents=True)
    before = _files(repo)
    meta = {
        "name": "../escaped",
        "ref_text": "nope",
        "backend": "qwen3-0.6b",
        "family": "picard",
    }
    server, state = _start_cloud(meta)
    watch = _DirWatch([repo, repo / "voices"])
    try:
        port = server.server_address[1]
        result = _run(repo, marker, f"http://127.0.0.1:{port}", "pull", VOICE_ID)
    finally:
        watch.close()
        server.shutdown()

    assert f"/v1/voices/{VOICE_ID}" in state.gets, result.stdout + result.stderr
    assert _files(repo) == before, (
        f"pull wrote outside the slug: {sorted(_files(repo) - before)}\n{result.stdout}"
    )
    escaped = [
        name for _mask, name in watch.events
        if name.startswith("escaped")
    ]
    assert escaped == [], f"pull created then removed {escaped}"
    assert not (repo / "escaped.json").exists()
    assert not (repo / "escaped-ref.wav").exists()
    assert not marker.exists()


def test_pull_and_push_lowercase_slug_still_round_trips(tmp_path: Path) -> None:
    """A slug name is still pulled into voices/ and pushed back."""
    repo = tmp_path / "repo"
    marker = tmp_path / "marker"
    (repo / "voices").mkdir(parents=True)
    meta = {
        "name": "newbie",
        "ref_text": "hello there",
        "backend": "qwen3-0.6b",
        "family": "newbie",
        "lang": "en",
    }
    server, state = _start_cloud(meta)
    try:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}"
        pulled = _run(repo, marker, url, "pull", VOICE_ID)
        assert (repo / "voices" / "newbie.json").is_file(), pulled.stdout + pulled.stderr
        assert (repo / "voices" / "newbie-ref.wav").is_file(), pulled.stdout + pulled.stderr
        profile = json.loads((repo / "voices" / "newbie.json").read_text())
        assert profile["name"] == "newbie"
        assert profile["family"] == "newbie"
        assert profile["reference_audio"] == "newbie-ref.wav"
        assert profile["reference_text"] == "hello there"
        pushed = _run(repo, marker, url, "push", "newbie")
    finally:
        server.shutdown()
    assert not marker.exists()
    assert "/v1/voices" in state.posts, pushed.stdout + pushed.stderr
    assert pushed.returncode == 0, pushed.stdout + pushed.stderr


# The five python -c open() calls in cmd_push are one loop. Each push test
# above reaches every one of them: a single remaining splice still runs the
# payload before the reference-audio check. voice_id is expanded inside the
# curl URL's double quotes and is not this bug; it is not covered here.
