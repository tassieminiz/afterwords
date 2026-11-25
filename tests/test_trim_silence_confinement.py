"""Confinement for silence-gap trim.

`scripts/trim-silence-gaps.py --apply` must not copy a reference whose
resolved path leaves ``voices/``. These tests fail while the trim treats audit
issue text as a silence-gap signal and copies the raw ``reference_audio``
path (including through a symlink that was retargeted after the audit's
confinement check).

``scripts/qa-voices.py`` also joins ``reference_audio`` and will read a path
that exists, but it does not copy that file into the tree. It is not covered
here.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import struct
import sys
import wave
from pathlib import Path

import pytest

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
TRIM_SCRIPT = REPO / "scripts" / "trim-silence-gaps.py"
# Obvious placeholder. Not a credential.
MARKER = b"afterwords-test-marker-not-a-secret\n"
PHRASE = "mid-clip silence"


def _load_trim():
    spec = importlib.util.spec_from_file_location("trim_silence_gaps_under_test", TRIM_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRIM = _load_trim()


class _Segment:
    text = "hello from the reference"


class _WhisperModel:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, *args, **kwargs):
        return [_Segment()], None


class _Writer:
    """File object that records bytes written, then delegates."""

    def __init__(self, raw, path, leaks, secret):
        self._raw = raw
        self._path = path
        self._chunks: list[bytes] = []
        self._leaks = leaks
        self._secret = secret

    def write(self, data):
        blob = data.encode() if isinstance(data, str) else bytes(data)
        self._chunks.append(blob)
        return self._raw.write(data)

    def __enter__(self):
        self._raw.__enter__()
        return self

    def __exit__(self, *exc):
        blob = b"".join(self._chunks)
        try:
            dest = Path(self._path).resolve()
        except OSError:
            dest = None
        if MARKER in blob and dest != self._secret:
            self._leaks.append(os.fspath(self._path))
        return self._raw.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._raw, name)


def _write_profile(voices: Path, reference_audio: str) -> None:
    (voices / "planted.json").write_text(
        json.dumps(
            {
                "name": "planted",
                "reference_audio": reference_audio,
                "reference_text": "hello from the reference",
            }
        )
    )


def _write_gapped_wav(path: Path) -> None:
    """0.5s tone, 2.0s silence, 0.5s tone. The silence is a mid-clip gap."""
    sr = 16000

    def tone(n: int) -> list[int]:
        return [
            int(0.4 * 32767 * math.sin(2 * math.pi * 440 * i / sr))
            for i in range(n)
        ]

    samples = tone(sr // 2) + [0] * (sr * 2) + tone(sr // 2)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(
            b"".join(struct.pack("<h", max(-32767, min(32767, sample))) for sample in samples)
        )


def _files_with_marker(root: Path, secret: Path) -> list[Path]:
    found = []
    secret_resolved = secret.resolve()
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            if path.resolve() == secret_resolved:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if MARKER in data:
            found.append(path)
    return found


def _run_apply(voices: Path, secret: Path, monkeypatch, *, retarget=None) -> list[str]:
    """Run the real trim ``main`` with ``--apply`` against ``voices``.

    Returns every destination the trim wrote the marker to, including a
    destination created and then removed before ``main`` returned.
    """
    leaks: list[str] = []
    secret_resolved = secret.resolve()

    real_open = open
    real_copy2 = TRIM.shutil.copy2
    retargeted = {"done": False}

    def _retarget_if_using_link(path) -> None:
        """Point the symlink at the outside file once the trim opens that path.

        Resolving the link still sees the in-voices clip. Opening or copying
        the link path itself follows the replacement, so a check that passed
        on the resolved clip does not authorize a later open of the same name.
        """
        if retarget is None or retargeted["done"]:
            return
        link, new_target = retarget
        try:
            candidate = Path(os.fspath(path))
        except TypeError:
            return
        if candidate == link:
            retargeted["done"] = True
            link.unlink()
            link.symlink_to(new_target)

    def tracking_copy2(src, dst, *args, **kwargs):
        _retarget_if_using_link(src)
        real_copy2(src, dst, *args, **kwargs)
        try:
            blob = Path(dst).read_bytes()
        except OSError:
            blob = b""
        if MARKER in blob:
            leaks.append(os.fspath(dst))
        return dst

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            _retarget_if_using_link(file)
        handle = real_open(file, mode, *args, **kwargs)
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            return _Writer(handle, file, leaks, secret_resolved)
        return handle

    monkeypatch.setattr(TRIM.shutil, "copy2", tracking_copy2)
    monkeypatch.setattr("builtins.open", tracking_open)

    import types

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = _WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(TRIM, "VOICES", voices)
    monkeypatch.setattr(
        sys, "argv", ["trim-silence-gaps.py", "--apply", "--json"]
    )

    try:
        TRIM.main()
    except Exception:
        # Decode of a non-audio file fails after the backup copy. The copy
        # is the failure this test is looking for, so keep going.
        pass

    for path in _files_with_marker(voices, secret):
        leaks.append(os.fspath(path))
    parent = secret.parent
    if parent.resolve() != voices.resolve():
        for path in _files_with_marker(parent, secret):
            leaks.append(os.fspath(path))
    return leaks


def test_apply_does_not_materialize_symlink_target_inside_voices(tmp_path, monkeypatch):
    """A profile whose reference name contains the gap phrase, and whose file
    is a symlink to a file outside voices, must not be copied to a regular
    backup inside voices.
    """
    secret = tmp_path / "private" / "credential"
    secret.parent.mkdir()
    secret.write_bytes(MARKER)
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / f"{PHRASE}.wav").symlink_to(secret)
    _write_profile(voices, f"{PHRASE}.wav")

    leaks = _run_apply(voices, secret, monkeypatch)

    assert leaks == []
    assert secret.read_bytes() == MARKER
    assert not (voices / f"{PHRASE}.wav.bak").exists()


def test_apply_does_not_copy_relative_escape_beside_original(tmp_path, monkeypatch):
    secret = tmp_path / "outside" / f"{PHRASE}.wav"
    secret.parent.mkdir()
    secret.write_bytes(MARKER)
    voices = tmp_path / "voices"
    voices.mkdir()
    _write_profile(voices, f"../outside/{PHRASE}.wav")

    leaks = _run_apply(voices, secret, monkeypatch)

    assert leaks == []
    assert not Path(str(secret) + ".bak").exists()
    assert secret.read_bytes() == MARKER


def test_apply_does_not_copy_absolute_reference_beside_original(tmp_path, monkeypatch):
    secret = tmp_path / "outside" / f"{PHRASE}.wav"
    secret.parent.mkdir()
    secret.write_bytes(MARKER)
    voices = tmp_path / "voices"
    voices.mkdir()
    _write_profile(voices, str(secret))

    leaks = _run_apply(voices, secret, monkeypatch)

    assert leaks == []
    assert not Path(str(secret) + ".bak").exists()
    assert secret.read_bytes() == MARKER


def test_apply_does_not_copy_through_directory_symlink(tmp_path, monkeypatch):
    """A directory symlink whose name contains the gap phrase must not make
    the trim copy a file from outside voices to a sibling of that file.
    """
    secret = tmp_path / "secrets" / "credential"
    secret.parent.mkdir()
    secret.write_bytes(MARKER)
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / PHRASE).symlink_to(secret.parent, target_is_directory=True)
    _write_profile(voices, f"{PHRASE}/credential")

    leaks = _run_apply(voices, secret, monkeypatch)

    assert leaks == []
    assert not (secret.parent / "credential.bak").exists()
    assert secret.read_bytes() == MARKER


def test_apply_does_not_follow_symlink_retargeted_after_confinement_check(
    tmp_path, monkeypatch
):
    """The path that was confined is the path that may be copied.

    The reference starts as a symlink to a real in-voices clip that has a
    mid-clip gap, so the audit's resolve stays inside voices. Before the
    trim copies, that symlink is pointed at a file outside voices. Copying
    the original path again follows the new target.
    """
    secret = tmp_path / "private" / "credential"
    secret.parent.mkdir()
    secret.write_bytes(MARKER)
    voices = tmp_path / "voices"
    voices.mkdir()
    good = voices / "good.wav"
    _write_gapped_wav(good)
    audio, sr = sf.read(good)
    assert TRIM.audit.detect_silence_gaps(audio, sr), "fixture clip has no mid-clip gap"
    link = voices / f"{PHRASE}.wav"
    link.symlink_to(good)
    _write_profile(voices, link.name)

    leaks = _run_apply(voices, secret, monkeypatch, retarget=(link, secret))

    assert leaks == []
    assert secret.read_bytes() == MARKER
    assert not (secret.parent / "credential.bak").exists()
    bak = voices / f"{PHRASE}.wav.bak"
    if bak.exists() and not bak.is_symlink():
        assert MARKER not in bak.read_bytes()
