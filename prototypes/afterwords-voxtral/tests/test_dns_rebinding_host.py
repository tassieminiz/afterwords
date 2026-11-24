"""Host-header allowlist for the Voxtral prototype.

These tests start the prototype the way `python server.py` does (loopback
bind, no warmup) and send the Host header a DNS-rebinding page would send.
They are expected to fail until a loopback bind rejects every non-local Host
with 403 before synthesis runs.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import uvicorn

PROTO_DIR = Path(__file__).resolve().parents[1]
SERVER_PATH = PROTO_DIR / "server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("voxtral_prototype_server", SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SERVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = _load_server()


class _Chunk:
    def __init__(self, audio, sample_rate=24000):
        self.audio = audio
        self.sample_rate = sample_rate


class _CountingLock:
    """Counts context-manager entries around the real synthesis lock."""

    def __init__(self, inner):
        self._inner = inner
        self.acquires = 0

    def __enter__(self):
        self.acquires += 1
        return self._inner.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._inner.__exit__(exc_type, exc, tb)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _http(port: int, method: str, path: str, host: str, body: bytes | None = None) -> tuple[int, bytes]:
    headers = [
        f"{method} {path} HTTP/1.1",
        f"Host: {host}",
        "X-Forwarded-Host: localhost",
        "Connection: close",
    ]
    payload = b""
    if body is not None:
        payload = body
        headers.append("Content-Type: application/json")
        headers.append(f"Content-Length: {len(payload)}")
    raw_request = ("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + payload
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.settimeout(5)
        sock.sendall(raw_request)
        chunks: list[bytes] = []
        while True:
            try:
                data = sock.recv(65536)
            except socket.timeout:
                break
            if not data:
                break
            chunks.append(data)
    raw = b"".join(chunks)
    status_line = raw.split(b"\r\n", 1)[0]
    parts = status_line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise AssertionError(f"unparseable response: {raw[:200]!r}")
    return int(parts[1]), raw


def _routes(app):
    found = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if methods and path:
            found.append((frozenset(methods), path))
    return found


@pytest.fixture(scope="module")
def loopback_server():
    """Run server.main() on the default loopback bind and record synthesis."""
    port = _free_port()
    observed_hosts: list[bytes | None] = []
    generate_calls: list[dict] = []
    model_loads: list[str] = []
    lock = _CountingLock(server._synth_lock)
    server._synth_lock = lock

    def generate(*, text, voice):
        generate_calls.append({"text": text, "voice": voice})
        yield _Chunk(np.zeros(240, dtype=np.float32), 24000)

    class _Model:
        def generate(self, *, text, voice):
            return generate(text=text, voice=voice)

    def get_model():
        model_loads.append("load")
        return _Model()

    server._get_model = get_model

    captured: dict = {}

    def run(app, host="127.0.0.1", port=8000, **kwargs):
        async def spy(scope, receive, send):
            if scope.get("type") == "http":
                header_map = dict(scope.get("headers") or [])
                observed_hosts.append(header_map.get(b"host"))
            await app(scope, receive, send)

        config = uvicorn.Config(
            spy,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        uv_server = uvicorn.Server(config)
        captured["server"] = uv_server
        captured["bind_host"] = host
        captured["port"] = port
        uv_server.run()

    original_run = uvicorn.run
    original_argv = sys.argv[:]
    uvicorn.run = run
    sys.argv = ["server.py", "--no-warmup", "--port", str(port)]
    thread = threading.Thread(target=server.main, name="voxtral-prototype", daemon=True)
    thread.start()
    for _ in range(100):
        if captured.get("server") is not None and captured["server"].started:
            break
        thread.join(0.05)
    else:
        raise RuntimeError("prototype server did not start")

    state = {
        "port": captured["port"],
        "bind_host": captured["bind_host"],
        "observed_hosts": observed_hosts,
        "generate_calls": generate_calls,
        "model_loads": model_loads,
        "lock": lock,
        "app": server.app,
    }
    try:
        yield state
    finally:
        captured["server"].should_exit = True
        thread.join(timeout=5)
        uvicorn.run = original_run
        sys.argv = original_argv


def _snap(state):
    return (
        len(state["generate_calls"]),
        len(state["model_loads"]),
        state["lock"].acquires,
        len(state["observed_hosts"]),
    )


def _assert_refused(state, before, status: int, raw: bytes, host: str):
    gen_before, load_before, lock_before, seen_before = before
    assert state["observed_hosts"][seen_before:] == [host.encode("latin-1")]
    new_calls = state["generate_calls"][gen_before:]
    new_loads = state["model_loads"][load_before:]
    new_locks = state["lock"].acquires - lock_before
    assert status == 403, {
        "status": status,
        "host": host,
        "generate_calls": new_calls,
        "model_loads": new_loads,
        "synth_lock_acquires": new_locks,
        "body_prefix": raw[:180],
    }
    assert b"invalid host header" in raw.lower()
    assert b"RIFF" not in raw
    assert new_calls == []
    assert new_loads == []
    assert new_locks == 0


def test_default_main_binds_loopback(loopback_server):
    assert loopback_server["bind_host"] == "127.0.0.1"


def test_rebinding_host_does_not_synthesize(loopback_server):
    """The attack request: attacker Host, loopback TCP, a full preset-voice job."""
    port = loopback_server["port"]
    host = f"rebind.example:{port}"
    body = json.dumps(
        {"text": "rebinding-canary " + ("A" * 4980), "voice": "casual_male"}
    ).encode()
    assert len(json.loads(body)["text"]) <= 5000
    before = _snap(loopback_server)
    status, raw = _http(port, "POST", "/synthesize", host, body)
    _assert_refused(loopback_server, before, status, raw, host)


def test_rebinding_host_rejected_before_handler_validation(loopback_server):
    """A bad voice or oversized body must still be a host rejection, not a 400."""
    port = loopback_server["port"]
    host = f"rebind.example:{port}"
    cases = [
        {"text": "hello", "voice": "not-a-voice"},
        {"text": "B" * 5001, "voice": "casual_male"},
        {"text": "   ", "voice": "casual_male"},
    ]
    for payload in cases:
        before = _snap(loopback_server)
        status, raw = _http(
            port,
            "POST",
            "/synthesize",
            host,
            json.dumps(payload).encode(),
        )
        _assert_refused(loopback_server, before, status, raw, host)


@pytest.mark.parametrize(
    "host_template",
    [
        "rebind.example",
        "rebind.example:{port}",
        "Rebind.Example:{port}",
        "rebind.example.:{port}",
        "localhost.rebind.example:{port}",
        "127.0.0.1.rebind.example:{port}",
        "[2001:db8::1]",
        "[2001:db8::1]:{port}",
        "0.0.0.0:{port}",
        "evil.example%00.localhost",
    ],
)
def test_foreign_host_shapes_do_not_synthesize(loopback_server, host_template):
    port = loopback_server["port"]
    host = host_template.format(port=port)
    before = _snap(loopback_server)
    status, raw = _http(
        port,
        "POST",
        "/synthesize",
        host,
        json.dumps({"text": "rebinding-canary", "voice": "casual_female"}).encode(),
    )
    _assert_refused(loopback_server, before, status, raw, host)


def test_every_route_rejects_foreign_host(loopback_server):
    port = loopback_server["port"]
    host = f"rebind.example:{port}"
    body = json.dumps({"text": "rebinding-canary", "voice": "neutral_male"}).encode()
    seen = []
    for methods, path in _routes(loopback_server["app"]):
        for method in sorted(methods):
            if method == "OPTIONS":
                continue
            seen.append((method, path))
            payload = body if method == "POST" else None
            before = _snap(loopback_server)
            status, raw = _http(port, method, path, host, payload)
            _assert_refused(loopback_server, before, status, raw, host)
    assert ("POST", "/synthesize") in seen
    assert ("GET", "/health") in seen
    assert ("GET", "/voices") in seen
    assert ("GET", "/openapi.json") in seen
    assert ("GET", "/docs") in seen
    assert ("GET", "/redoc") in seen


@pytest.mark.parametrize(
    "host_template",
    [
        "localhost",
        "localhost:{port}",
        "LOCALHOST:{port}",
        "127.0.0.1",
        "127.0.0.1:{port}",
        "[::1]",
        "[::1]:{port}",
    ],
)
def test_loopback_hosts_still_synthesize(loopback_server, host_template):
    port = loopback_server["port"]
    host = host_template.format(port=port)
    text = f"local-ok {host}"
    before_gen = len(loopback_server["generate_calls"])
    status, raw = _http(
        port,
        "POST",
        "/synthesize",
        host,
        json.dumps({"text": text, "voice": "casual_male"}).encode(),
    )
    assert status == 200, raw[:300]
    assert b"RIFF" in raw
    new_calls = loopback_server["generate_calls"][before_gen:]
    assert new_calls == [{"text": text, "voice": "casual_male"}]
    assert loopback_server["observed_hosts"][-1] == host.encode("latin-1")


def test_localhost_health_still_ok_after_rebinding_attempt(loopback_server):
    port = loopback_server["port"]
    evil = f"rebind.example:{port}"
    before = _snap(loopback_server)
    status, raw = _http(port, "GET", "/health", evil)
    _assert_refused(loopback_server, before, status, raw, evil)

    status, raw = _http(port, "GET", "/health", f"127.0.0.1:{port}")
    assert status == 200
    payload = json.loads(raw.split(b"\r\n\r\n", 1)[1])
    assert payload["status"] == "ok"
    assert payload["mode"] == "preset-voices-only"
