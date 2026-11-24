"""Prototype Voxtral preset-voice TTS server for Apple Silicon."""

from __future__ import annotations

import argparse
import io
import logging
import re
import threading
import time
from collections.abc import Iterable

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voxtral")

MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
DEFAULT_VOICE = "casual_male"
SUPPORTED_VOICES = (
    "casual_male",
    "casual_female",
    "cheerful_female",
    "neutral_male",
    "neutral_female",
    "fr_male",
    "fr_female",
    "es_male",
    "es_female",
    "de_male",
    "de_female",
    "it_male",
    "it_female",
    "pt_male",
    "pt_female",
    "nl_male",
    "nl_female",
    "ar_male",
    "hi_male",
    "hi_female",
)

app = FastAPI(title="Afterwords Voxtral Prototype")

_LOOPBACK_BIND_HOSTS = ("127.0.0.1", "localhost", "::1")
_WILDCARD_BIND_HOSTS = ("0.0.0.0", "::", "")
_LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "[::1]"})
_HOST_HEADER_RE = re.compile(r"(\[[0-9a-f:.]+\]|[a-z0-9.\-]+)(?::[0-9]{1,5})?", re.ASCII)

# Host-header allowlist: blocks DNS rebinding, where a browser sends requests
# to this loopback port under an attacker-controlled hostname. Enforced on
# every request regardless of bind; main() may only widen it with names the
# operator passes explicitly.
_allowed_hostnames: frozenset[str] = _LOOPBACK_HOSTNAMES


def _host_allowed(host_header: str) -> bool:
    match = _HOST_HEADER_RE.fullmatch(host_header.lower())
    return match is not None and match.group(1) in _allowed_hostnames


@app.middleware("http")
async def _validate_host(request, call_next):
    hosts = request.headers.getlist("host")
    if len(hosts) != 1 or not _host_allowed(hosts[0]):
        # HEAD responses carry no body, so the reason also goes in a header.
        return JSONResponse(
            {"error": "invalid host header"},
            status_code=403,
            headers={"X-Error": "invalid host header"},
        )
    return await call_next(request)


def _normalize_hostname(raw: str) -> str:
    name = raw.strip().lower()
    if ":" in name and not name.startswith("["):
        name = f"[{name}]"
    match = _HOST_HEADER_RE.fullmatch(name)
    if match is None or match.group(1) != name:
        raise SystemExit(f"invalid host name (no port, no scheme): {raw!r}")
    return name


def _resolve_bind(host: str, *, bind_public: bool, extra_hosts: list[str]) -> frozenset[str]:
    """Validate the bind address and return the Host-header allowlist for it."""
    if host not in _LOOPBACK_BIND_HOSTS and not bind_public:
        raise SystemExit(
            f"refusing to bind {host}: non-loopback binds expose the GPU to the "
            "network. Re-run with --bind-public if this is intentional."
        )
    allowed = set(_LOOPBACK_HOSTNAMES)
    if host not in _LOOPBACK_BIND_HOSTS and host not in _WILDCARD_BIND_HOSTS:
        allowed.add(_normalize_hostname(host))
    allowed.update(_normalize_hostname(h) for h in extra_hosts)
    return frozenset(allowed)


_model = None
_model_lock = threading.Lock()
_synth_lock = threading.Lock()
_ready = threading.Event()


class SynthesizeRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE


def _get_model():
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        from mlx_audio.tts.utils import load

        log.info("Loading model %s ...", MODEL_ID)
        t0 = time.time()
        _model = load(MODEL_ID)
        log.info("Model loaded in %.1fs", time.time() - t0)
        return _model


def _coerce_audio_array(audio) -> np.ndarray:
    """Convert a model audio object into a 1D float32 numpy array."""
    if hasattr(audio, "tolist"):
        return np.asarray(audio.tolist(), dtype=np.float32)
    return np.asarray(audio, dtype=np.float32)


def _collect_audio(results: Iterable) -> tuple[np.ndarray, int]:
    """Collect streaming Voxtral generation output into one waveform."""
    chunks: list[np.ndarray] = []
    sample_rate = 24000

    for result in results:
        audio = getattr(result, "audio", None)
        if audio is None:
            continue
        sr = getattr(result, "sample_rate", None)
        if isinstance(sr, int) and sr > 0:
            sample_rate = sr
        chunk = _coerce_audio_array(audio)
        if chunk.size:
            chunks.append(chunk.reshape(-1))

    if not chunks:
        raise RuntimeError("generation produced no audio")
    return np.concatenate(chunks), sample_rate


def _synthesize(text: str, voice: str) -> Response:
    if not text.strip():
        return JSONResponse({"error": "text is empty"}, status_code=400)
    if len(text) > 5000:
        return JSONResponse({"error": "text too long (max 5000 chars)"}, status_code=400)
    if voice not in SUPPORTED_VOICES:
        return JSONResponse(
            {"error": f"unknown voice: {voice}", "available": list(SUPPORTED_VOICES)},
            status_code=400,
        )
    if not _ready.is_set():
        return JSONResponse({"error": "server warming up, try again shortly"}, status_code=503)

    model = _get_model()
    log.info("synthesize: %d chars, voice=%s", len(text), voice)
    t0 = time.time()

    try:
        # MLX/Metal synthesis is treated as single-threaded for safety.
        with _synth_lock:
            audio, sr = _collect_audio(model.generate(text=text, voice=voice))
        elapsed = time.time() - t0
        duration = len(audio) / sr if sr else 0

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)

        log.info(
            "done: %.1fs audio in %.1fs (RTF=%.2fx)",
            duration,
            elapsed,
            elapsed / duration if duration > 0 else 0,
        )
        return Response(
            content=buf.read(),
            media_type="audio/wav",
            headers={
                "X-Synthesis-Time": f"{elapsed:.3f}",
                "X-Duration": f"{duration:.3f}",
                "X-Sample-Rate": str(sr),
            },
        )
    except Exception as exc:
        log.error("synthesis failed: %s", exc, exc_info=True)
        return JSONResponse({"error": "synthesis failed"}, status_code=500)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "backend": "mlx",
        "mode": "preset-voices-only",
        "model_loaded": _model is not None,
        "ready": _ready.is_set(),
        "default_voice": DEFAULT_VOICE,
        "voices": list(SUPPORTED_VOICES),
    }


@app.get("/voices")
def voices():
    return {"voices": list(SUPPORTED_VOICES), "default_voice": DEFAULT_VOICE}


@app.post("/synthesize")
def synthesize(body: SynthesizeRequest):
    return _synthesize(body.text, body.voice)


def main():
    parser = argparse.ArgumentParser(description="Afterwords Voxtral prototype server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--bind-public",
        action="store_true",
        help="Allow binding a non-loopback address (exposes synthesis to the network)",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="NAME",
        help="Extra hostname or IP clients may send in the Host header (repeatable)",
    )
    args = parser.parse_args()

    global _allowed_hostnames
    _allowed_hostnames = _resolve_bind(
        args.host, bind_public=args.bind_public, extra_hosts=args.allowed_host
    )

    if not args.no_warmup:
        try:
            _get_model()
        except Exception as exc:
            log.error("Failed to load model: %s", exc)
            raise SystemExit(1)

    _ready.set()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
