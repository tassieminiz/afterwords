"""Spec for the POST /clone decoded-audio limit.

POST /clone rejects uploads larger than 25 MB of compressed bytes, then
``soundfile.read`` decodes the whole file at float64 with no cap on frames,
channels, or sample rate. libsndfile picks the format from the bytes, so a
``.wav`` name does not matter, and the frame count it reports is the
STREAMINFO total (a short FLAC can claim tens of millions of samples).

These tests fail until a clone upload is refused, before that allocation and
before any ``voices/`` file is created, unless all of the following hold:

* decoded duration is at most 120 seconds
* channel count is at most 2
* sample rate is between 8 kHz and 96 kHz inclusive
* the read that actually pulls samples passes an explicit frame bound of at
  most 120 seconds at 96 kHz, including when the file changes after a metadata
  check of the same path

A request that carries a non-loopback ``Origin`` (or ``Origin: null``) is
refused on every state-changing endpoint: POST /clone, POST /synthesize,
POST /reload, and DELETE /session/{id}. A request with no Origin header is
still accepted.

``clone-voice.sh`` is not covered: it trims the clip to 15 seconds with
ffmpeg before it reads samples. The voice-maintenance scripts read a path the
operator passed in, not this upload.
"""

from __future__ import annotations

import io
import os
import tempfile
import threading

import numpy as np
import pytest
import soundfile as sf

import server

MAX_DECODED_SECONDS = 120
MAX_CHANNELS = 2
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 96_000
# Bound that must be applied at the read itself, not only by an earlier check.
MAX_READ_FRAMES = MAX_DECODED_SECONDS * MAX_SAMPLE_RATE
# A header-only allocation of this many float64 samples is far past the cap
# and far past the 25 MB byte check (the file itself is a few KB).
BOMB_FRAMES = 20_000_000
ALLOC_SLOP_BYTES = 16 * 1024 * 1024
REJECT_STATUSES = range(400, 500)


def _vm_size_bytes() -> int:
    """Current virtual size. ``np.empty`` reserves this before pages are touched."""
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024
    raise AssertionError("VmSize missing; the allocation check cannot be skipped")


def _call_watching_vm_size(fn):
    """Run ``fn`` and return ``(result, peak VmSize growth)``.

    VmPeak is a process-wide high-water mark, so a later allocation that fits
    under an earlier peak does not move it. Sampling the current VmSize catches
    that reservation while it is alive.
    """
    samples: list[int] = []
    stop = threading.Event()
    started = threading.Event()

    def watch() -> None:
        started.set()
        while not stop.is_set():
            samples.append(_vm_size_bytes())

    watcher = threading.Thread(target=watch)
    watcher.start()
    started.wait()
    baseline = _vm_size_bytes()
    try:
        return fn(), max([baseline, *samples]) - baseline
    finally:
        stop.set()
        watcher.join()


def _wav_bytes(seconds: float, sample_rate: int, channels: int = 1) -> bytes:
    frames = int(seconds * sample_rate)
    if channels == 1:
        data = np.zeros(frames, dtype=np.int16)
    else:
        data = np.zeros((frames, channels), dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _flac_silence(seconds: float, sample_rate: int, channels: int = 1) -> bytes:
    frames = int(seconds * sample_rate)
    buf = io.BytesIO()
    block = np.zeros((8192, channels) if channels > 1 else 8192, dtype=np.int16)
    with sf.SoundFile(
        buf, "w", sample_rate, channels, format="FLAC", subtype="PCM_16"
    ) as out:
        written = 0
        while written < frames:
            chunk = block
            remaining = frames - written
            if remaining < len(block):
                chunk = block[:remaining]
            out.write(chunk)
            written += len(chunk)
    return buf.getvalue()


def _set_flac_total_samples(streaminfo: bytearray, total: int) -> None:
    """Patch the 36-bit total-samples field in a 34-byte STREAMINFO block."""
    word = int.from_bytes(streaminfo[10:18], "big")
    word = (word & ~((1 << 36) - 1)) | (total & ((1 << 36) - 1))
    streaminfo[10:18] = word.to_bytes(8, "big")


def _metadata_blocks(data: bytes) -> tuple[list[tuple[int, int, bool, int]], int]:
    assert data[:4] == b"fLaC"
    blocks = []
    index = 4
    while index < len(data):
        header = data[index]
        length = int.from_bytes(data[index + 1 : index + 4], "big")
        blocks.append((index, header & 0x7F, bool(header & 0x80), length))
        index += 4 + length
        if header & 0x80:
            return blocks, index
    raise AssertionError("FLAC metadata has no last-block flag")


def _header_bomb() -> bytes:
    """A few KB of FLAC whose STREAMINFO claims ``BOMB_FRAMES`` samples.

    The encoded payload is about a second of silence. libsndfile reports the
    STREAMINFO total as the frame count, and ``soundfile.read`` allocates a
    float64 array of that length before it discovers the payload is short.
    Padding keeps the file above the 1_000-byte minimum in the clone handler.
    """
    raw = bytearray(_flac_silence(1.0, 48_000, 1))
    blocks, audio_at = _metadata_blocks(raw)
    info_at, info_type, _last, info_len = blocks[0]
    assert info_type == 0 and info_len == 34
    payload = bytearray(raw[info_at + 4 : info_at + 4 + info_len])
    _set_flac_total_samples(payload, BOMB_FRAMES)
    raw[info_at + 4 : info_at + 4 + info_len] = payload

    blocks, audio_at = _metadata_blocks(raw)
    last_at, _typ, last, _length = blocks[-1]
    assert last
    raw[last_at] &= 0x7F
    pad_len = 4000
    padding = bytes([0x80 | 1]) + pad_len.to_bytes(3, "big") + b"\x00" * pad_len
    bombed = bytes(raw[:audio_at]) + padding + bytes(raw[audio_at:])
    info = sf.info(io.BytesIO(bombed))
    assert info.frames == BOMB_FRAMES
    assert info.format == "FLAC"
    assert 1000 <= len(bombed) < server.MAX_CLONE_UPLOAD_BYTES
    return bombed


def _session_files(session_id: str) -> list[str]:
    folder = server._VOICES_DIR
    return sorted(
        name
        for name in os.listdir(folder)
        if name.startswith(session_id)
    )


def _wipe(session_id: str) -> None:
    server._unregister_session(session_id)
    folder = server._VOICES_DIR
    for name in _session_files(session_id):
        try:
            os.remove(os.path.join(folder, name))
        except OSError:
            pass


def _post_clone(client, audio: bytes, session_id: str, filename: str = "clip.wav", headers=None):
    return client.post(
        "/clone",
        files={"audio": (filename, audio, "application/octet-stream")},
        data={
            "session_id": session_id,
            "emotion": "neutral",
            "transcript": "bounded clone reference",
        },
        headers=headers,
    )


def _read_is_bounded(frames, start, stop) -> bool:
    if stop is not None:
        if start is None:
            start = 0
        if start < 0 or stop < start:
            return False
        return (stop - start) <= MAX_READ_FRAMES
    if frames is None or frames < 0:
        return False
    return frames <= MAX_READ_FRAMES


@pytest.fixture
def clone_on(monkeypatch):
    monkeypatch.setattr(server, "_clone_enabled", True)


def test_short_pcm_without_origin_is_accepted(client, clone_on):
    """Local clients that send no Origin header can still clone a short clip."""
    session_id = "zzdeclim-ok"
    try:
        response = _post_clone(client, _wav_bytes(2, 24_000), session_id)
        assert response.status_code == 200, response.text
        assert response.json()["duration_s"] <= MAX_DECODED_SECONDS
    finally:
        _wipe(session_id)


@pytest.mark.parametrize(
    ("sample_rate", "channels"),
    [(MIN_SAMPLE_RATE, 1), (MAX_SAMPLE_RATE, 1), (24_000, MAX_CHANNELS)],
)
def test_pcm_at_the_inclusive_bounds_is_accepted(client, clone_on, sample_rate, channels):
    session_id = f"zzdeclim-bound-{sample_rate}-{channels}"
    try:
        audio = _wav_bytes(2, sample_rate, channels)
        response = _post_clone(client, audio, session_id)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["duration_s"] <= MAX_DECODED_SECONDS
    finally:
        _wipe(session_id)


def test_pcm_longer_than_cap_is_refused_before_it_is_stored(client, clone_on):
    """130 s of 8 kHz PCM is under the 25 MB byte cap and over the 120 s cap."""
    session_id = "zzdeclim-longwav"
    audio = _wav_bytes(MAX_DECODED_SECONDS + 10, MIN_SAMPLE_RATE)
    assert len(audio) < server.MAX_CLONE_UPLOAD_BYTES
    try:
        response = _post_clone(client, audio, session_id)
        assert response.status_code in REJECT_STATUSES, response.text
        assert _session_files(session_id) == []
    finally:
        _wipe(session_id)


def test_flac_longer_than_cap_is_refused_before_it_is_stored(client, clone_on):
    """Silence FLAC is a few KB and decodes to well over 120 seconds."""
    session_id = "zzdeclim-longflac"
    audio = _flac_silence(MAX_DECODED_SECONDS + 10, MIN_SAMPLE_RATE)
    info = sf.info(io.BytesIO(audio))
    assert info.format == "FLAC"
    assert info.duration > MAX_DECODED_SECONDS
    assert len(audio) < 100_000
    try:
        # Saved under a .wav suffix, the same way the handler names its temp file.
        response = _post_clone(client, audio, session_id, filename="clip.wav")
        assert response.status_code in REJECT_STATUSES, response.text
        assert _session_files(session_id) == []
    finally:
        _wipe(session_id)


def _decoded_bytes(monkeypatch) -> list[int]:
    """Record the byte size ``SoundFile`` is about to allocate, then allocate it."""
    sizes: list[int] = []
    real = server.sf.SoundFile._create_empty_array

    def wrapped(self, frames, always_2d, dtype):
        sizes.append(int(frames) * int(self.channels) * np.dtype(dtype).itemsize)
        return real(self, frames, always_2d, dtype)

    monkeypatch.setattr(server.sf.SoundFile, "_create_empty_array", wrapped)
    return sizes


def test_header_bomb_is_refused_without_allocating_claimed_frames(client, clone_on, monkeypatch):
    """STREAMINFO can claim a huge frame count in a file of a few KB.

    ``soundfile.read`` allocates float64 samples for that claim while the
    clone handler holds the synthesis lock. Refusal has to happen before that
    allocation; a 500 from the decoder means the bomb was already read.
    """
    import noisereduce  # noqa: F401  (the handler imports it on the same path)

    session_id = "zzdeclim-bomb"
    audio = _header_bomb()
    claimed = BOMB_FRAMES * np.dtype(np.float64).itemsize
    reserved = _decoded_bytes(monkeypatch)

    def post():
        return _post_clone(client, audio, session_id, filename="clip.wav")

    try:
        response, growth = _call_watching_vm_size(post)
        asked = max(reserved, default=0)
        assert (
            asked < ALLOC_SLOP_BYTES
            and growth < ALLOC_SLOP_BYTES
            and response.status_code in REJECT_STATUSES
        ), (
            f"status={response.status_code} reserved={asked} virtual_growth={growth} "
            f"claimed_float64_bytes={claimed} body={response.text[:400]}"
        )
        assert _session_files(session_id) == []
    finally:
        _wipe(session_id)


@pytest.mark.parametrize(
    ("sample_rate", "channels"),
    [(4_000, 1), (192_000, 1), (24_000, MAX_CHANNELS + 1)],
)
def test_extreme_rate_or_channel_count_is_refused(client, clone_on, sample_rate, channels):
    session_id = f"zzdeclim-shape-{sample_rate}-{channels}"
    audio = _wav_bytes(2, sample_rate, channels)
    assert len(audio) < server.MAX_CLONE_UPLOAD_BYTES
    try:
        response = _post_clone(client, audio, session_id)
        assert response.status_code in REJECT_STATUSES, response.text
        assert _session_files(session_id) == []
    finally:
        _wipe(session_id)


def test_read_stays_capped_when_the_file_changes_after_it_is_checked(client, clone_on, monkeypatch):
    """A metadata check and the later read are two opens of the same path.

    After the first read-only open closes, the bytes are replaced with a FLAC
    whose frame count is above ``MAX_READ_FRAMES``. An unbounded second read
    decodes that file. The read call itself has to pass a frame bound.
    """
    session_id = "zzdeclim-recheck"
    replacement = _flac_silence(MAX_DECODED_SECONDS + 60, MAX_SAMPLE_RATE)
    replacement_frames = sf.info(io.BytesIO(replacement)).frames
    assert replacement_frames > MAX_READ_FRAMES

    calls: list[tuple] = []
    state = {"swapped": False, "inside_module_read": False}
    real_init = server.sf.SoundFile.__init__
    real_close = server.sf.SoundFile.close
    real_module_read = server.sf.read
    real_instance_read = server.sf.SoundFile.read
    tmp_root = os.path.realpath(tempfile.gettempdir())
    voices_root = os.path.realpath(server._VOICES_DIR)

    def _is_upload(path) -> bool:
        if not isinstance(path, str):
            return False
        real = os.path.realpath(path)
        return real.startswith(tmp_root + os.sep) and not real.startswith(voices_root + os.sep)

    def init(self, file, mode="r", *args, **kwargs):
        real_init(self, file, mode, *args, **kwargs)
        object.__setattr__(
            self,
            "_decode_limit_path",
            file if mode == "r" and _is_upload(file) else None,
        )

    def close(self):
        path = getattr(self, "_decode_limit_path", None)
        real_close(self)
        if path and not state["swapped"] and os.path.exists(path):
            with open(path, "wb") as handle:
                handle.write(replacement)
            state["swapped"] = True

    def module_read(file, frames=-1, start=0, stop=None, **kwargs):
        calls.append(("soundfile.read", frames, start, stop))
        state["inside_module_read"] = True
        try:
            return real_module_read(file, frames=frames, start=start, stop=stop, **kwargs)
        finally:
            state["inside_module_read"] = False

    def instance_read(self, frames=-1, *args, **kwargs):
        # soundfile.read() turns frames=-1 into the file length before this
        # call. Record direct SoundFile.read users, not that inner call.
        if not state["inside_module_read"]:
            calls.append(("SoundFile.read", frames, 0, None))
        return real_instance_read(self, frames, *args, **kwargs)

    monkeypatch.setattr(server.sf.SoundFile, "__init__", init)
    monkeypatch.setattr(server.sf.SoundFile, "close", close)
    monkeypatch.setattr(server.sf, "read", module_read)
    monkeypatch.setattr(server.sf.SoundFile, "read", instance_read)

    try:
        response = _post_clone(client, _wav_bytes(2, 24_000), session_id)
        unbounded = [call for call in calls if not _read_is_bounded(call[1], call[2], call[3])]
        assert not unbounded, (
            f"sample read was not frame-capped (swap_installed={state['swapped']}): {calls}"
        )
        if response.status_code == 200:
            assert response.json()["duration_s"] <= MAX_DECODED_SECONDS
        else:
            assert response.status_code in REJECT_STATUSES, response.text
            assert _session_files(session_id) == []
    finally:
        _wipe(session_id)


def test_cross_origin_clone_is_refused_before_the_bomb_is_decoded(client, clone_on, monkeypatch):
    """The host allowlist accepts ``127.0.0.1:7860``. A browser POST still sends that host.

    A non-loopback Origin has to be refused before the decoder allocates.
    """
    import noisereduce  # noqa: F401

    monkeypatch.setattr(server, "_enforce_host_check", True)
    session_id = "zzdeclim-origin-bomb"
    audio = _header_bomb()
    reserved = _decoded_bytes(monkeypatch)

    def post():
        return _post_clone(
            client,
            audio,
            session_id,
            headers={"host": "127.0.0.1:7860", "origin": "https://evil.example"},
        )

    try:
        response, growth = _call_watching_vm_size(post)
        asked = max(reserved, default=0)
        assert (
            asked < ALLOC_SLOP_BYTES
            and growth < ALLOC_SLOP_BYTES
            and response.status_code in REJECT_STATUSES
        ), (
            f"status={response.status_code} reserved={asked} virtual_growth={growth} "
            f"body={response.text[:400]}"
        )
        assert _session_files(session_id) == []
    finally:
        _wipe(session_id)


@pytest.mark.parametrize("origin", ["https://evil.example", "null"])
def test_cross_origin_clone_of_a_valid_clip_is_refused(client, clone_on, origin):
    session_id = "zzdeclim-origin-okclip"
    try:
        response = _post_clone(
            client,
            _wav_bytes(2, 24_000),
            session_id,
            headers={"origin": origin},
        )
        assert response.status_code in REJECT_STATUSES, response.text
        assert _session_files(session_id) == []
    finally:
        _wipe(session_id)


def test_cross_origin_synthesize_is_refused(client, clone_on, sample_voice):
    response = client.post(
        "/synthesize",
        json={"text": "Hello", "voice": sample_voice},
        headers={"origin": "https://evil.example"},
    )
    assert response.status_code in REJECT_STATUSES, response.headers.get("content-type")
    assert response.headers.get("content-type", "").split(";")[0] != "audio/wav"


def test_cross_origin_reload_is_refused(client, clone_on):
    saved = dict(server.VOICES)
    try:
        response = client.post("/reload", headers={"origin": "https://evil.example"})
        if response.status_code not in REJECT_STATUSES:
            pytest.fail(f"reload status={response.status_code}")
    finally:
        server.VOICES.clear()
        server.VOICES.update(saved)


def test_cross_origin_delete_does_not_remove_the_session(client, clone_on):
    session_id = "zzdeclim-origin-del"
    try:
        created = _post_clone(client, _wav_bytes(2, 24_000), session_id)
        assert created.status_code == 200, created.text
        voice = created.json()["voice"]
        assert voice in server.VOICES
        response = client.delete(
            f"/session/{session_id}",
            headers={"origin": "https://evil.example"},
        )
        assert response.status_code in REJECT_STATUSES, response.text
        assert voice in server.VOICES
        assert _session_files(session_id) != []
    finally:
        _wipe(session_id)
