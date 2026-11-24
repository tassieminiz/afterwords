"""Cross-site requests to the loopback TTS server must not run.

The server binds to 127.0.0.1 and, for that bind, only checks the Host
header. A browser on the same machine still sends Host: 127.0.0.1 for
http://127.0.0.1:7860, so a page the user opens can issue the request.
These tests send that request with the headers a browser attaches and
require the server to reject it before any side effect:

- multipart parsing (Starlette spools a file part to disk once it passes
  the 1MB spool limit, and it does that before the endpoint body runs)
- noise reduction, Whisper, or a voices/ write
- the synthesis lock
- reload, including prune
- session deletion

Requests with no Origin and no Sec-Fetch-Site (curl, the agent hooks)
still work. So do same-origin browser requests and Sec-Fetch-Site: none.
A cross-site Origin, the Origin value ``null``, or any Sec-Fetch-Site
other than ``same-origin`` or ``none`` is rejected with 403 even when
Host is a loopback name the host allowlist accepts.

GET /health is covered because it is the same unauthenticated listener.
Nothing else on this app accepts a request.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import types

import pytest

import backends
import server
from backends.base import RefTextPolicy

LOOPBACK_HOST = "127.0.0.1:7860"
EVIL_ORIGIN = "https://evil.example"


def _headers(extra: dict | None = None) -> dict:
    headers = {"Host": LOOPBACK_HOST}
    if extra:
        headers.update(extra)
    return headers


def _wav_bytes(seconds: float) -> bytes:
    """16-bit mono PCM WAV at 24 kHz. seconds=22 is larger than the 1MB spool."""
    sr = 24000
    n_samples = int(sr * seconds)
    data_size = n_samples * 2
    header = b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    header += b"data" + struct.pack("<I", data_size)
    return header + (b"\x00" * data_size)


def _write_wav(path, seconds: float) -> None:
    path.write_bytes(_wav_bytes(seconds))


@pytest.fixture(autouse=True)
def _restore_voices():
    """Drop voices a rejected request must not have registered.

    Setup runs before sample_voice, and teardown runs after it, so this
    restores the registry without putting the sample voice back.
    """
    snapshot = dict(server.VOICES)
    yield
    server.VOICES.clear()
    server.VOICES.update(snapshot)


@pytest.fixture
def loopback_clone(monkeypatch):
    """Production loopback flags: host allowlist on, clone endpoints on."""
    monkeypatch.setattr(server, "_enforce_host_check", True)
    monkeypatch.setattr(server, "_clone_enabled", True)


@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    directory = tmp_path / "voices"
    directory.mkdir()
    monkeypatch.setattr(server, "_VOICES_DIR", str(directory))
    return directory


@pytest.fixture
def clone_pipeline(monkeypatch):
    """Stand-ins for the libraries the clone handler imports lazily.

    They record calls. A rejected request must not import them into the
    request path at all; the assertions are on the counters.
    """
    calls = {"noisereduce": 0, "whisper": 0}

    def reduce_noise(y, sr, **kwargs):
        calls["noisereduce"] += 1
        return y

    class _Word:
        def __init__(self, word):
            self.word = word

    class _Segment:
        words = [_Word("hello")]

    class WhisperModel:
        def __init__(self, *args, **kwargs):
            calls["whisper"] += 1

        def transcribe(self, path):
            calls["whisper"] += 1
            return [_Segment()], None

    nr_mod = types.ModuleType("noisereduce")
    nr_mod.reduce_noise = reduce_noise
    whisper_mod = types.ModuleType("faster_whisper")
    whisper_mod.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "noisereduce", nr_mod)
    monkeypatch.setitem(sys.modules, "faster_whisper", whisper_mod)

    backend = backends.get("qwen3-0.6b")
    monkeypatch.setattr(backend, "ref_text_policy", RefTextPolicy.REQUIRED)
    return calls


@pytest.fixture
def form_spools(monkeypatch):
    """Record every multipart spool the parser constructs during the request."""
    import starlette.formparsers as formparsers

    created: list = []
    real = formparsers.SpooledTemporaryFile

    class TrackingSpool(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", TrackingSpool)
    return created


@pytest.fixture
def named_temps(monkeypatch):
    created: list[str] = []
    real = tempfile.NamedTemporaryFile

    def wrapper(*args, **kwargs):
        handle = real(*args, **kwargs)
        created.append(handle.name)
        return handle

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", wrapper)
    return created


@pytest.fixture
def synth_calls(sample_voice):
    backend = backends.get("fake")
    calls: list[str] = []
    real = backend.synthesize

    def wrapped(text, prepared, lang):
        # _synthesize_audio holds _synth_lock across this call. A cross-site
        # request that reaches here has already blocked every other speaker.
        assert server._synth_lock.locked()
        calls.append(text)
        return real(text, prepared, lang)

    backend.synthesize = wrapped
    return calls


def _assert_rejected(response) -> None:
    assert response.status_code == 403, response.text
    assert response.headers.get("content-type", "").startswith("application/json")
    assert "access-control-allow-origin" not in response.headers
    assert response.headers.get("access-control-allow-private-network", "").lower() != "true"


def _assert_clone_rejected(response, voices_dir, spools, temps, calls, voices_before) -> None:
    """Status and side effects together, so a late 403 still fails."""
    rolled = [bool(getattr(spool, "_rolled", False)) for spool in spools]
    files = sorted(path.name for path in voices_dir.iterdir())
    new_voices = sorted(set(server.VOICES) - voices_before)
    problems = []
    if response.status_code != 403:
        problems.append(f"status {response.status_code}: {response.text[:300]}")
    if "access-control-allow-origin" in response.headers:
        problems.append(f"ACAO={response.headers['access-control-allow-origin']}")
    if response.headers.get("access-control-allow-private-network", "").lower() == "true":
        problems.append("Access-Control-Allow-Private-Network granted")
    if spools:
        problems.append(f"multipart spool count={len(spools)} rolled={rolled}")
    if temps:
        problems.append(f"named temp files={temps}")
    if calls["noisereduce"]:
        problems.append(f"noisereduce calls={calls['noisereduce']}")
    if calls["whisper"]:
        problems.append(f"whisper calls={calls['whisper']}")
    if files:
        problems.append(f"voices/ files={files}")
    if new_voices:
        problems.append(f"registered voices={new_voices}")
    assert not problems, "cross-site clone ran:\n" + "\n".join(problems)


# Headers a non-browser client, an address-bar navigation, or a same-origin
# page on this server would send. Each of these must keep working.
_ALLOWED = [
    pytest.param({}, id="no-browser-headers"),
    pytest.param({"Origin": ""}, id="empty-origin"),
    pytest.param(
        {"Sec-Fetch-Site": "none", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"},
        id="sec-fetch-none",
    ),
    pytest.param(
        {
            "Origin": "http://127.0.0.1:7860",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors",
        },
        id="same-origin-loopback",
    ),
    pytest.param(
        {
            "Host": "localhost:7860",
            "Origin": "http://localhost:7860",
            "Sec-Fetch-Site": "same-origin",
        },
        id="same-origin-localhost",
    ),
    pytest.param(
        {
            "Host": "[::1]:7860",
            "Origin": "http://[::1]:7860",
            "Sec-Fetch-Site": "same-origin",
        },
        id="same-origin-ipv6",
    ),
]

# Headers a public page's fetch/img/form would attach. Host stays loopback:
# that is the value the host allowlist accepts.
_CROSS_SITE = [
    pytest.param(
        {
            "Origin": EVIL_ORIGIN,
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Dest": "empty",
        },
        id="origin-and-cross-site",
    ),
    pytest.param(
        {
            "Origin": "http://evil.example",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "no-cors",
        },
        id="http-origin",
    ),
    pytest.param(
        {
            "Origin": "null",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "no-cors",
        },
        id="null-origin-downgrade",
    ),
    pytest.param(
        {
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Dest": "image",
        },
        id="image-no-origin",
    ),
    pytest.param({"Origin": EVIL_ORIGIN}, id="origin-only"),
    pytest.param({"Origin": "null"}, id="null-origin-only"),
    pytest.param({"Sec-Fetch-Site": "cross-site"}, id="cross-site-only"),
    pytest.param({"Sec-Fetch-Site": "same-site"}, id="same-site"),
    pytest.param(
        {"Origin": "http://127.0.0.1:7860", "Sec-Fetch-Site": "cross-site"},
        id="loopback-origin-but-cross-site",
    ),
    pytest.param(
        {"Origin": EVIL_ORIGIN, "Sec-Fetch-Site": "none"},
        id="evil-origin-with-sec-fetch-none",
    ),
    pytest.param(
        {
            "Host": "[::1]:7860",
            "Origin": EVIL_ORIGIN,
            "Sec-Fetch-Site": "cross-site",
        },
        id="ipv6-host-cross-site",
    ),
]


@pytest.mark.parametrize("extra", _CROSS_SITE)
def test_cross_site_get_synthesize_rejected(client, loopback_clone, synth_calls, extra):
    voices_before = set(server.VOICES)
    response = client.get(
        "/synthesize",
        params={"text": "a" * 5000, "voice": "testvoice"},
        headers=_headers(extra),
    )
    _assert_rejected(response)
    assert synth_calls == []
    assert set(server.VOICES) == voices_before


@pytest.mark.parametrize("extra", _ALLOWED)
def test_allowed_get_synthesize_still_works(client, loopback_clone, synth_calls, extra):
    response = client.get(
        "/synthesize",
        params={"text": "Hello", "voice": "testvoice"},
        headers=_headers(extra),
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/wav")
    assert synth_calls == ["Hello"]


def test_cross_site_synthesize_rejected_when_clone_disabled(
    client, monkeypatch, synth_calls
):
    """GET /synthesize is on without --allow-clone. The flood does not need it."""
    monkeypatch.setattr(server, "_enforce_host_check", True)
    monkeypatch.setattr(server, "_clone_enabled", False)
    response = client.get(
        "/synthesize",
        params={"text": "a" * 5000, "voice": "testvoice"},
        headers=_headers(
            {
                "Origin": EVIL_ORIGIN,
                "Sec-Fetch-Site": "cross-site",
                "Sec-Fetch-Mode": "no-cors",
                "Sec-Fetch-Dest": "image",
            }
        ),
    )
    _assert_rejected(response)
    assert synth_calls == []


@pytest.mark.parametrize("extra", _CROSS_SITE)
def test_cross_site_post_synthesize_rejected(client, loopback_clone, synth_calls, extra):
    response = client.post(
        "/synthesize",
        json={"text": "a" * 5000, "voice": "testvoice"},
        headers=_headers(extra),
    )
    _assert_rejected(response)
    assert synth_calls == []


@pytest.mark.parametrize("extra", _ALLOWED)
def test_allowed_post_synthesize_still_works(client, loopback_clone, synth_calls, extra):
    response = client.post(
        "/synthesize",
        json={"text": "Hello", "voice": "testvoice"},
        headers=_headers(extra),
    )
    assert response.status_code == 200, response.text
    assert synth_calls == ["Hello"]


def _post_clone(client, headers, seconds: float, session_id: str = "victim-session"):
    return client.post(
        "/clone",
        files={"audio": ("clip.wav", _wav_bytes(seconds), "audio/wav")},
        data={"session_id": session_id, "emotion": "neutral"},
        headers=headers,
    )


@pytest.mark.parametrize("extra", _CROSS_SITE)
def test_cross_site_clone_rejected_before_any_side_effect(
    client,
    loopback_clone,
    voices_dir,
    clone_pipeline,
    form_spools,
    named_temps,
    extra,
):
    # 22s of 16-bit 24 kHz mono is ~1.05MB, past MultiPartParser.spool_max_size
    # (1MB). If parsing starts, the spool rolls onto a real temp file.
    voices_before = set(server.VOICES)
    response = _post_clone(client, _headers(extra), seconds=22)
    _assert_clone_rejected(
        response, voices_dir, form_spools, named_temps, clone_pipeline, voices_before
    )


@pytest.mark.parametrize("extra", _ALLOWED)
def test_allowed_clone_still_registers_voice(
    client,
    loopback_clone,
    voices_dir,
    clone_pipeline,
    extra,
):
    session_id = "allowed-session"
    try:
        response = _post_clone(
            client, _headers(extra), seconds=2, session_id=session_id
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["voice"].startswith(f"{session_id}-")
        assert body["voice"] in server.VOICES
        written = sorted(path.name for path in voices_dir.iterdir())
        assert f"{body['voice']}-ref.wav" in written
        assert f"{body['voice']}.json" in written
        assert clone_pipeline["noisereduce"] == 1
        # transcript omitted, and the backend requires ref text, so Whisper ran.
        assert clone_pipeline["whisper"] >= 1
    finally:
        server._unregister_session(session_id)


def test_cross_site_clone_does_not_extend_existing_session(
    client,
    loopback_clone,
    voices_dir,
    clone_pipeline,
    form_spools,
    named_temps,
    tmp_path,
):
    """A guessed session id must not gain an attacker-voiced palette entry."""
    wav = tmp_path / "existing-ref.wav"
    _write_wav(wav, 0.1)
    backend = backends.get("fake")
    prepared = backend.prepare_voice(str(wav), "already here", {})
    server._register_voice(
        "victim-session-001",
        "fake",
        str(wav),
        "already here",
        prepared,
        metadata={"session_id": "victim-session", "sequence": 1},
    )
    voices_before = set(server.VOICES)
    try:
        response = _post_clone(
            client,
            _headers(
                {
                    "Origin": EVIL_ORIGIN,
                    "Sec-Fetch-Site": "cross-site",
                    "Sec-Fetch-Mode": "no-cors",
                }
            ),
            seconds=22,
        )
        _assert_clone_rejected(
            response, voices_dir, form_spools, named_temps, clone_pipeline, voices_before
        )
        assert server.VOICES["victim-session-001"].ref_text == "already here"
    finally:
        server.VOICES.pop("victim-session-001", None)


@pytest.mark.parametrize("prune", [False, True], ids=["add-only", "prune"])
@pytest.mark.parametrize("extra", _CROSS_SITE)
def test_cross_site_reload_rejected(
    client, loopback_clone, voices_dir, sample_voice, prune, extra
):
    _write_wav(voices_dir / "diskvoice-ref.wav", 0.1)
    (voices_dir / "diskvoice.json").write_text(
        json.dumps(
            {
                "name": "diskvoice",
                "backend": "fake",
                "reference_audio": "diskvoice-ref.wav",
                "reference_text": "on disk",
            }
        )
    )
    backend = backends.get("fake")
    prepares = {"n": 0}
    real_prepare = backend.prepare_voice

    def counting_prepare(*args, **kwargs):
        prepares["n"] += 1
        return real_prepare(*args, **kwargs)

    backend.prepare_voice = counting_prepare
    ghost_wav = voices_dir / "ghost-ref.wav"
    _write_wav(ghost_wav, 0.1)
    ghost_prepared = backend.prepare_voice(str(ghost_wav), "ghost", {})
    prepares["n"] = 0
    server._register_voice(
        "ghostvoice",
        "fake",
        str(ghost_wav),
        "ghost",
        ghost_prepared,
        metadata={"session_id": None},
    )
    # _register_voice derives session_id from the name when metadata has None.
    # Gallery voices are prunable only when session_id is None; force that.
    server.VOICES["ghostvoice"] = server.VoiceProfile(
        **{
            **server.VOICES["ghostvoice"].__dict__,
            "session_id": None,
        }
    )
    try:
        response = client.post(
            "/reload",
            params={"prune": str(prune).lower()},
            headers=_headers(extra),
        )
        _assert_rejected(response)
        assert prepares["n"] == 0
        assert "ghostvoice" in server.VOICES
        assert "diskvoice" not in server.VOICES
        assert (voices_dir / "diskvoice.json").is_file()
        assert ghost_wav.is_file()
    finally:
        server.VOICES.pop("ghostvoice", None)
        server.VOICES.pop("diskvoice", None)


@pytest.mark.parametrize("extra", _ALLOWED)
def test_allowed_reload_still_works(client, loopback_clone, voices_dir, extra):
    _write_wav(voices_dir / "diskvoice-ref.wav", 0.1)
    (voices_dir / "diskvoice.json").write_text(
        json.dumps(
            {
                "name": "diskvoice",
                "backend": "fake",
                "reference_audio": "diskvoice-ref.wav",
                "reference_text": "on disk",
            }
        )
    )
    try:
        response = client.post("/reload", headers=_headers(extra))
        assert response.status_code == 200, response.text
        assert "diskvoice" in response.json()["reloaded"]
        assert "diskvoice" in server.VOICES
    finally:
        server.VOICES.pop("diskvoice", None)


def test_cross_site_delete_does_not_remove_session(
    client, loopback_clone, voices_dir, tmp_path
):
    wav = voices_dir / "victim-session-001-ref.wav"
    profile = voices_dir / "victim-session-001.json"
    _write_wav(wav, 0.1)
    profile.write_text("{}\n")
    backend = backends.get("fake")
    prepared = backend.prepare_voice(str(wav), "keep me", {})
    server._register_voice(
        "victim-session-001",
        "fake",
        str(wav),
        "keep me",
        prepared,
        metadata={"session_id": "victim-session", "sequence": 1},
    )
    try:
        response = client.delete(
            "/session/victim-session",
            headers=_headers(
                {"Origin": EVIL_ORIGIN, "Sec-Fetch-Site": "cross-site"}
            ),
        )
        _assert_rejected(response)
        assert "victim-session-001" in server.VOICES
        assert wav.is_file()
        assert profile.is_file()
    finally:
        server.VOICES.pop("victim-session-001", None)


@pytest.mark.parametrize("extra", _ALLOWED)
def test_allowed_delete_still_removes_session(
    client, loopback_clone, voices_dir, extra
):
    wav = voices_dir / "allowed-session-001-ref.wav"
    profile = voices_dir / "allowed-session-001.json"
    _write_wav(wav, 0.1)
    profile.write_text("{}\n")
    backend = backends.get("fake")
    prepared = backend.prepare_voice(str(wav), "remove me", {})
    server._register_voice(
        "allowed-session-001",
        "fake",
        str(wav),
        "remove me",
        prepared,
        metadata={"session_id": "allowed-session", "sequence": 1},
    )
    response = client.delete("/session/allowed-session", headers=_headers(extra))
    assert response.status_code == 200, response.text
    assert "allowed-session-001" not in server.VOICES
    assert not wav.exists()
    assert not profile.exists()


@pytest.mark.parametrize("extra", _CROSS_SITE)
def test_cross_site_health_rejected(client, loopback_clone, extra):
    response = client.get("/health", headers=_headers(extra))
    _assert_rejected(response)


@pytest.mark.parametrize("extra", _ALLOWED)
def test_allowed_health_still_works(client, loopback_clone, extra):
    response = client.get("/health", headers=_headers(extra))
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ok"


def test_foreign_host_still_rejected(client, loopback_clone, synth_calls):
    """The host allowlist stays in force for names it was built to block."""
    response = client.get(
        "/synthesize",
        params={"text": "Hello", "voice": "testvoice"},
        headers={"Host": "evil.example", "Origin": EVIL_ORIGIN},
    )
    _assert_rejected(response)
    assert synth_calls == []


@pytest.mark.parametrize(
    "path,method",
    [
        ("/clone", "OPTIONS"),
        ("/synthesize", "OPTIONS"),
        ("/reload", "OPTIONS"),
        ("/session/victim-session", "OPTIONS"),
        ("/health", "OPTIONS"),
    ],
)
def test_preflight_does_not_approve_cross_site(client, loopback_clone, path, method):
    response = client.request(
        method,
        path,
        headers=_headers(
            {
                "Origin": EVIL_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
                "Access-Control-Request-Private-Network": "true",
                "Sec-Fetch-Site": "cross-site",
                "Sec-Fetch-Mode": "cors",
            }
        ),
    )
    _assert_rejected(response)


def test_cross_site_clone_rejected_for_guessable_session_id(
    client,
    loopback_clone,
    voices_dir,
    clone_pipeline,
    form_spools,
    named_temps,
):
    """session_id is caller-chosen and the documented example is my-voice."""
    voices_before = set(server.VOICES)
    response = _post_clone(
        client,
        _headers({"Origin": EVIL_ORIGIN, "Sec-Fetch-Site": "cross-site"}),
        seconds=22,
        session_id="my-voice",
    )
    _assert_clone_rejected(
        response, voices_dir, form_spools, named_temps, clone_pipeline, voices_before
    )
