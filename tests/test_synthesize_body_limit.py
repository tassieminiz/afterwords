"""POST /synthesize must cap the request body before it is buffered.

These tests fail while the server copies an unlimited POST body into RAM and
only then applies the 5000-character check (or returns 404/422). They pass
once POST /synthesize rejects an oversized body with 413 without reading past
a few tens of kilobytes.

The cap has to accept a 5000-character body, including 4-byte UTF-8, and has
to reject anything larger than 64 KiB. Both Content-Length and chunked
bodies are covered, for text/plain and application/json, with the host check
on and voice cloning left at its default off. The byte counts are asserted
so a fix that looks at Content-Length and then still reads the payload fails.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import pytest

import server

# Top of "a few tens of kilobytes". A 5000-character body of 4-byte UTF-8
# plus a small JSON envelope is about 20 KiB and must still be accepted.
BODY_CAP = 64 * 1024
CHUNK = 4 * 1024
# One chunk past the cap is how a streaming limiter notices it has crossed.
READ_CEILING = BODY_CAP + CHUNK
OVERSIZED_DECLARED_LENGTH = 2_000_000


@pytest.fixture(scope="module")
def live_server():
    """Serve the real app through uvicorn, the same way server.main() does."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    import uvicorn

    config = uvicorn.Config(
        server.app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        lifespan="on",
    )
    instance = uvicorn.Server(config)
    previous_host = server._enforce_host_check
    previous_clone = server._clone_enabled
    server._enforce_host_check = True
    server._clone_enabled = False

    thread = threading.Thread(target=instance.run, name="synthesize-body-limit", daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        instance.should_exit = True
        raise RuntimeError("uvicorn did not accept connections")

    yield port

    instance.should_exit = True
    thread.join(timeout=5)
    server._enforce_host_check = previous_host
    server._clone_enabled = previous_clone


def _status_line(sock: socket.socket, timeout: float) -> int | None:
    sock.settimeout(timeout)
    try:
        data = sock.recv(4096)
    except TimeoutError:
        return None
    if not data:
        return None
    line = data.split(b"\r\n", 1)[0].decode("iso-8859-1")
    parts = line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise AssertionError(f"unrecognized status line: {line!r}")
    return int(parts[1])


def _post_headers(port: int, content_type: str, *, length: int | None, chunked: bool) -> bytes:
    lines = [
        "POST /synthesize HTTP/1.1",
        f"Host: 127.0.0.1:{port}",
        f"Content-Type: {content_type}",
    ]
    if chunked:
        lines.append("Transfer-Encoding: chunked")
    else:
        lines.append(f"Content-Length: {length}")
    lines.append("Connection: close")
    lines.append("")
    lines.append("")
    return ("\r\n".join(lines)).encode("ascii")


def _open(port: int) -> socket.socket:
    return socket.create_connection(("127.0.0.1", port), timeout=2)


@pytest.mark.parametrize("content_type", ["text/plain", "application/json"])
def test_content_length_over_cap_is_rejected_before_the_body_is_read(live_server, content_type):
    """A declared multi-megabyte body is refused before any payload byte is sent.

    The browser attack sends text/plain so the request is CORS-safelisted.
    application/json is the same route's declared body and must be capped too.
    """
    sock = _open(live_server)
    try:
        sock.sendall(
            _post_headers(
                live_server,
                content_type,
                length=OVERSIZED_DECLARED_LENGTH,
                chunked=False,
            )
        )
        status = _status_line(sock, timeout=1.0)
    finally:
        sock.close()
    assert status == 413, (
        f"{content_type} Content-Length {OVERSIZED_DECLARED_LENGTH} produced {status}, "
        "so the server is waiting to buffer the body instead of rejecting it"
    )


@pytest.mark.parametrize("content_type", ["text/plain", "application/json"])
def test_chunked_body_is_rejected_once_the_cap_is_reached(live_server, content_type):
    """A chunked body with no Content-Length is refused before the message ends.

    No terminating chunk is sent. A reader that accumulates until the end
    never answers; a cap answers 413 once it has seen BODY_CAP bytes.
    """
    payload = b"A" * CHUNK
    chunk_count = (BODY_CAP // CHUNK) + 2  # past the cap and one overshoot chunk
    sock = _open(live_server)
    try:
        sock.sendall(_post_headers(live_server, content_type, length=None, chunked=True))
        for _ in range(chunk_count):
            sock.sendall(f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n")
        status = _status_line(sock, timeout=1.0)
    finally:
        sock.close()
    assert status == 413, (
        f"chunked {content_type} body of {chunk_count * CHUNK} bytes produced {status}; "
        "the server kept reading past the cap"
    )


def test_max_legal_text_is_not_rejected_as_too_large(live_server):
    """5000 characters of 4-byte UTF-8 must fit under the body cap.

    Cloning is off, so a body that was accepted is the usual 404, not 413.
    """
    text = "😀" * 5000
    body = json.dumps({"text": text, "voice": "galadriel"}, ensure_ascii=False).encode()
    assert len(text) == 5000
    assert len(body) > 20_000
    assert len(body) < BODY_CAP
    sock = _open(live_server)
    try:
        sock.sendall(
            _post_headers(live_server, "application/json", length=len(body), chunked=False)
            + body
        )
        status = _status_line(sock, timeout=2.0)
    finally:
        sock.close()
    assert status is not None, "server did not answer a 5000-character request"
    assert status != 413


def _asgi_scope(content_type: str, length: int | None) -> dict:
    headers = [
        (b"host", b"127.0.0.1:7860"),
        (b"content-type", content_type.encode("ascii")),
    ]
    if length is not None:
        headers.append((b"content-length", str(length).encode("ascii")))
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/synthesize",
        "raw_path": b"/synthesize",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 7860),
    }


async def _drive(scope: dict, body: bytes) -> tuple[int, int]:
    """Run the real ASGI app and return (status, bytes the app pulled).

    A cap that rejects on Content-Length pulls nothing. A cap that stops a
    chunked read pulls at most one chunk past the limit.
    """
    pulled = 0
    pos = 0

    async def receive():
        nonlocal pulled, pos
        if pos >= len(body):
            return {"type": "http.request", "body": b"", "more_body": False}
        part = body[pos : pos + CHUNK]
        pos += len(part)
        pulled += len(part)
        return {"type": "http.request", "body": part, "more_body": pos < len(body)}

    status: dict[str, int] = {}

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]

    await server.app(scope, receive, send)
    return status["code"], pulled


@pytest.fixture
def host_check_on():
    previous_host = server._enforce_host_check
    previous_clone = server._clone_enabled
    server._enforce_host_check = True
    server._clone_enabled = False
    yield
    server._enforce_host_check = previous_host
    server._clone_enabled = previous_clone


@pytest.mark.parametrize("content_type", ["text/plain", "application/json"])
def test_app_rejects_declared_length_without_reading_payload(host_check_on, content_type):
    status, pulled = asyncio.run(
        _drive(
            _asgi_scope(content_type, OVERSIZED_DECLARED_LENGTH),
            b"x" * (BODY_CAP + CHUNK),
        )
    )
    assert status == 413
    assert pulled == 0


@pytest.mark.parametrize("content_type", ["text/plain", "application/json"])
def test_app_stops_reading_a_chunked_body_at_the_cap(host_check_on, content_type):
    body = b"B" * (BODY_CAP + CHUNK * 4)
    status, pulled = asyncio.run(_drive(_asgi_scope(content_type, None), body))
    assert status == 413
    assert 0 < pulled <= READ_CEILING
    assert pulled < len(body)


def test_app_does_not_buffer_an_overlong_json_text_before_the_character_check(host_check_on):
    """The 5000-character check must not run on a body the cap already forbids.

    Cloning is enabled here so today's code reaches that check and returns 400
    only after the whole JSON value is in memory. The body is larger than the
    cap, so the response has to be 413 from the bytes read so far.
    """
    server._clone_enabled = True
    text = "a" * (BODY_CAP + CHUNK * 4)
    body = json.dumps({"text": text, "voice": "galadriel"}).encode()
    assert len(body) > READ_CEILING
    status, pulled = asyncio.run(_drive(_asgi_scope("application/json", len(body)), body))
    assert status == 413
    assert pulled <= READ_CEILING
    assert pulled < len(body)
