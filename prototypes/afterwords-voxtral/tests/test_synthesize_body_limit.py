"""Oversized POST /synthesize bodies must be refused before they are buffered.

A website can send this request as a simple cross-origin POST (Content-Type
text/plain, application/x-www-form-urlencoded, or multipart/form-data). The
browser delivers that request without a CORS preflight. The server has to
answer 413 once the body exceeds a few tens of kilobytes, and it has to stop
reading at that cap for both a declared Content-Length and a chunked body.
The existing 5000-character text check stays behind the cap.

64 KiB is the most body these tests allow the server to read. A fix that
still pulls the rest of the payload into memory fails them, including a fix
that checks the size and then reads the body anyway.

POST /health and POST /voices do not share the bug: a POST to either is
405 and does not read the body, so those routes are not covered here.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

_PROTO_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "voxtral_prototype_server", _PROTO_DIR / "server.py"
)
assert _SPEC is not None and _SPEC.loader is not None
server = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = server
_SPEC.loader.exec_module(server)

# Top of "a few tens of kilobytes". A 5000-character JSON body still fits.
MAX_SYNTHESIZE_BODY_BYTES = 64 * 1024
CHUNK_BYTES = 4 * 1024
OVERSIZE_BYTES = 256 * 1024


def _chunks(body: bytes, size: int = CHUNK_BYTES) -> list[bytes]:
    return [body[i : i + size] for i in range(0, len(body), size)] or [b""]


def _header_list(
    content_type: str | None,
    content_length: int | None,
) -> list[tuple[bytes, bytes]]:
    headers = [(b"host", b"127.0.0.1:7861")]
    if content_type is not None:
        headers.append((b"content-type", content_type.encode("ascii")))
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    return headers


def _invoke(
    body: bytes,
    *,
    content_type: str | None,
    declare_length: bool,
    path: str = "/synthesize",
) -> tuple[int | None, int, bytes]:
    """Drive the ASGI app, returning (status, body bytes the app pulled, response)."""
    parts = _chunks(body)
    read = 0
    index = 0
    status: int | None = None
    response = bytearray()

    async def receive():
        nonlocal read, index
        if index >= len(parts):
            return {"type": "http.request", "body": b"", "more_body": False}
        chunk = parts[index]
        index += 1
        read += len(chunk)
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(parts),
        }

    async def send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            response.extend(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": _header_list(
            content_type, len(body) if declare_length else None
        ),
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 7861),
    }

    async def run():
        await server.app(scope, receive, send)
        # Let a same-loop follow-up read show up before we sample `read`.
        await asyncio.sleep(0)

    asyncio.run(run())
    return status, read, bytes(response)


def _assert_refused_before_buffered(status: int | None, read: int) -> None:
    assert status == 413, f"expected 413, got {status} after reading {read} bytes"
    assert read <= MAX_SYNTHESIZE_BODY_BYTES + CHUNK_BYTES, (
        f"server read {read} bytes before refusing; "
        f"cap is {MAX_SYNTHESIZE_BODY_BYTES}"
    )


def test_text_plain_content_length_rejected_before_buffering():
    """Simple text/plain POST with Content-Length above the cap."""
    body = b"a" * OVERSIZE_BYTES
    status, read, _ = _invoke(body, content_type="text/plain", declare_length=True)
    _assert_refused_before_buffered(status, read)


def test_text_plain_chunked_rejected_before_buffering():
    """Chunked text/plain POST with no Content-Length, the same simple request."""
    body = b"a" * OVERSIZE_BYTES
    status, read, _ = _invoke(body, content_type="text/plain", declare_length=False)
    _assert_refused_before_buffered(status, read)


def test_form_urlencoded_rejected_before_buffering():
    """application/x-www-form-urlencoded is also sent without a preflight."""
    body = b"text=" + (b"a" * OVERSIZE_BYTES)
    status, read, _ = _invoke(
        body,
        content_type="application/x-www-form-urlencoded",
        declare_length=True,
    )
    _assert_refused_before_buffered(status, read)


def test_multipart_rejected_before_buffering():
    """multipart/form-data is also a simple content type."""
    body = b"a" * OVERSIZE_BYTES
    status, read, _ = _invoke(
        body,
        content_type="multipart/form-data; boundary=----bound",
        declare_length=True,
    )
    _assert_refused_before_buffered(status, read)


def test_missing_content_type_rejected_before_buffering():
    """A POST with no Content-Type is still a simple request and is fully read."""
    body = b"a" * OVERSIZE_BYTES
    status, read, _ = _invoke(body, content_type=None, declare_length=True)
    _assert_refused_before_buffered(status, read)


def test_json_over_cap_rejected_before_text_check():
    """application/json larger than the cap is 413 before the 5000-char check."""
    body = json.dumps(
        {"text": "a" * OVERSIZE_BYTES, "voice": server.DEFAULT_VOICE}
    ).encode()
    status, read, raw = _invoke(
        body, content_type="application/json", declare_length=True
    )
    _assert_refused_before_buffered(status, read)
    assert b"text too long" not in raw


def test_json_over_5000_chars_still_rejected_as_too_long():
    """A body under the cap still hits the 5000-character check."""
    body = json.dumps({"text": "a" * 5001, "voice": server.DEFAULT_VOICE}).encode()
    assert len(body) < MAX_SYNTHESIZE_BODY_BYTES
    status, read, raw = _invoke(
        body, content_type="application/json", declare_length=True
    )
    assert status == 400
    assert b"text too long" in raw
    assert read == len(body)


def test_small_json_is_not_rejected_for_size():
    """A normal synthesize body still reaches the handler."""
    server._ready.clear()
    body = json.dumps({"text": "hello", "voice": server.DEFAULT_VOICE}).encode()
    status, read, raw = _invoke(
        body, content_type="application/json", declare_length=True
    )
    assert status == 503
    assert b"warming up" in raw
    assert read == len(body)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_accepts(port: int, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else b""
            raise RuntimeError(f"server exited early: {output!r}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("server did not accept connections")


def _read_headers(sock: socket.socket, timeout: float) -> bytes:
    sock.settimeout(timeout)
    data = b""
    try:
        while b"\r\n\r\n" not in data:
            part = sock.recv(8192)
            if not part:
                break
            data += part
    except TimeoutError:
        pass
    return data


def _start_server(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "server.py",
            "--no-warmup",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(server.__file__).rsplit("/", 1)[0],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _live_exchange(
    head: bytes, body_pieces: list[tuple[int, bytes]]
) -> tuple[int | None, int]:
    """Speak HTTP/1.1 to a real server.py process.

    `body_pieces` is (payload bytes, bytes on the wire). Returns
    (status, payload bytes written before the status line arrived).
    If the server never answers early, the remaining pieces are still
    written so the vulnerable server can finish and the assertion can
    see that status.
    """
    port = _free_port()
    proc = _start_server(port)
    try:
        _wait_until_accepts(port, proc)
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.sendall(head)
            early = _read_headers(sock, 0.3)
            sent = 0
            if b"HTTP/" in early:
                return int(early.split()[1]), sent
            waited = False
            for payload, wire in body_pieces:
                sock.sendall(wire)
                sent += payload
                if sent > MAX_SYNTHESIZE_BODY_BYTES and not waited:
                    waited = True
                    seen = _read_headers(sock, 1.0)
                    if b"HTTP/" in seen:
                        return int(seen.split()[1]), sent
            seen = _read_headers(sock, 2.0)
            if b"HTTP/" not in seen:
                return None, sent
            return int(seen.split()[1]), sent
    finally:
        _stop_server(proc)


def test_live_server_chunked_text_plain_stops_reading():
    """server.py on 127.0.0.1 must 413 a chunked text/plain body at the cap."""
    head = (
        "POST /synthesize HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: text/plain\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")
    pieces = []
    for chunk in _chunks(b"a" * OVERSIZE_BYTES):
        wire = f"{len(chunk):x}\r\n".encode("ascii") + chunk + b"\r\n"
        pieces.append((len(chunk), wire))
    pieces.append((0, b"0\r\n\r\n"))
    status, sent = _live_exchange(head, pieces)
    assert status == 413
    assert sent <= MAX_SYNTHESIZE_BODY_BYTES + CHUNK_BYTES


def test_live_server_content_length_text_plain_stops_reading():
    """server.py must 413 from Content-Length before the whole body is read."""
    body = b"a" * OVERSIZE_BYTES
    head = (
        "POST /synthesize HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: text/plain\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")
    pieces = [(len(chunk), chunk) for chunk in _chunks(body)]
    status, sent = _live_exchange(head, pieces)
    assert status == 413
    assert sent <= MAX_SYNTHESIZE_BODY_BYTES + CHUNK_BYTES
