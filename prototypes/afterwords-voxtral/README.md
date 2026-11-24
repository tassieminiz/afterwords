# Afterwords Voxtral Prototype

Isolated preset-voice prototype for evaluating `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` on Apple Silicon without disturbing the main zero-shot cloning server.

## Goal

This prototype answers one question: is Voxtral good enough as a local preset-voice TTS backend to justify a deeper integration later?

It deliberately does **not** attempt voice cloning, `.afterwords` profile reuse, or Claude/Codex hook integration.

## What it does

- Runs a small FastAPI server on `localhost:7861`
- Loads Voxtral through `mlx-audio`
- Exposes `GET /health`
- Exposes `GET /voices`
- Exposes `POST /synthesize` with preset voices
- Serializes synthesis through a lock, matching the MLX safety model used by the main repo

## Expected tradeoff

- Better multilingual preset voices
- Simpler backend shape
- No proof yet of local MLX-compatible reference-audio cloning

## Setup

```bash
cd prototypes/afterwords-voxtral
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py --port 7861
```

First run will download model weights.

The server only answers requests whose `Host` header is `localhost`, `127.0.0.1`, or `[::1]` (any port) and returns 403 otherwise, which blocks DNS-rebinding pages from driving the GPU. Binding a non-loopback address requires `--bind-public`; clients then need a name passed with `--allowed-host` (repeatable), or the bind IP itself when `--host` is a specific address:

```bash
python server.py --host 0.0.0.0 --bind-public --allowed-host mymac.local
```

## Test

```bash
cd prototypes/afterwords-voxtral
PYTHONPATH=. pytest
```

## Smoke test

```bash
curl localhost:7861/health
curl localhost:7861/voices
curl -X POST localhost:7861/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from Voxtral","voice":"casual_male"}' \
  --output voxtral.wav
```
