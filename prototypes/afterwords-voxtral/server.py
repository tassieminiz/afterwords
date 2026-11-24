"""Prototype Voxtral preset-voice TTS server for Apple Silicon."""

from __future__ import annotations

import argparse
import io
import logging
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

# A 5000-character text where every character is a JSON-escaped surrogate
# pair (12 bytes each) still fits.
MAX_REQUEST_BODY_BYTES = 64 * 1024


class BodySizeLimitMiddleware:
    """Answer 413 once a request body passes max_bytes, without reading the rest.

    The body is read here (bounded) and replayed to the app, so neither
    Content-Length nor a chunked body can make the app buffer more.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    def _too_large(self) -> JSONResponse:
        return JSONResponse(
            {"error": f"request body too large (max {self.max_bytes} bytes)"},
            status_code=413,
            headers={"Connection": "close"},
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", ()):
            if name.lower() != b"content-length":
                continue
            if not value.strip().isdigit():
                response = JSONResponse(
                    {"error": "invalid content-length"},
                    status_code=400,
                    headers={"Connection": "close"},
                )
                await response(scope, receive, send)
                return
            if int(value) > self.max_bytes:
                await self._too_large()(scope, receive, send)
                return

        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                return
            body += message.get("body", b"")
            if len(body) > self.max_bytes:
                await self._too_large()(scope, receive, send)
                return
            more_body = message.get("more_body", False)

        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


app = FastAPI(title="Afterwords Voxtral Prototype")
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)

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
    args = parser.parse_args()

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
