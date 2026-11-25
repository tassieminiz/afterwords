#!/usr/bin/env python3
"""Trim mid-clip silence from voice references and refresh transcripts.

Walks every voices/*.json the audit tool flags for mid-clip silence, applies
ffmpeg silenceremove to densify the WAV, re-runs Whisper, writes the updated
reference_text, and appends a notes line summarising the change.

Usage:
    python scripts/trim-silence-gaps.py             # dry run
    python scripts/trim-silence-gaps.py --apply
    python scripts/trim-silence-gaps.py --apply --voice the-doctor
"""
from __future__ import annotations

import argparse
import io
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VOICES = REPO / "voices"
SCRIPTS = REPO / "scripts"

import importlib.util
spec = importlib.util.spec_from_file_location("audit_vt", SCRIPTS / "audit-voice-transcripts.py")
audit = importlib.util.module_from_spec(spec)
sys.modules["audit_vt"] = audit  # required for dataclasses on py3.14+
spec.loader.exec_module(audit)


def trim_wav(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-af",
        "silenceremove=stop_periods=-1:stop_duration=0.6:stop_threshold=-30dB",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _create_file(dir_fd: int, name: str, data: bytes, src_st: os.stat_result | None = None) -> None:
    """Create `name` in `dir_fd`; never opens an existing entry or follows a symlink."""
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                 0o644, dir_fd=dir_fd)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
        fh.flush()
        if src_st is not None:
            os.fchmod(fh.fileno(), stat.S_IMODE(src_st.st_mode))
            os.utime(fh.fileno(), ns=(src_st.st_atime_ns, src_st.st_mtime_ns))


def trim_reference(finding) -> bytes:
    """Back up, trim, and replace the exact file the audit decoded; return the new WAV.

    The file is reopened from voices/ without following symlinks and must be the
    same (st_dev, st_ino) the audit recorded, so a reference swapped after the
    audit is refused before anything is written.
    """
    import soundfile as sf

    dir_fd, name = audit.open_parent_confined(VOICES, finding.wav_path)
    try:
        data, st = audit.read_regular_nofollow(dir_fd, name)
        if (st.st_dev, st.st_ino) != finding.wav_id:
            raise RuntimeError(f"{name} changed since it was audited; refusing to trim")
        sf.info(io.BytesIO(data))

        try:
            _create_file(dir_fd, name + ".bak", data, st)
        except FileExistsError:
            pass

        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
            src.write_bytes(data)
            trim_wav(src, dst)
            trimmed = dst.read_bytes()
        sf.info(io.BytesIO(trimmed))

        tmp_name = f".{name}.{secrets.token_hex(4)}.trimmed"
        _create_file(dir_fd, tmp_name, trimmed)
        try:
            cur = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            if (cur.st_dev, cur.st_ino) != finding.wav_id:
                raise RuntimeError(f"{name} changed while trimming; refusing to replace")
            os.replace(tmp_name, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except BaseException:
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
            raise
        return trimmed
    finally:
        os.close(dir_fd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Actually overwrite WAVs and JSONs (otherwise dry-run)")
    ap.add_argument("--voice", help="Only process a single voice by name")
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON to stdout; suppress human output")
    args = ap.parse_args()

    profiles = sorted(VOICES.glob("*.json"))
    if args.voice:
        profiles = [p for p in profiles if p.stem == args.voice]
        if not profiles:
            print(f"Voice '{args.voice}' not found.", file=sys.stderr)
            sys.exit(2)

    from faster_whisper import WhisperModel
    print("loading whisper...", file=sys.stderr)
    model = WhisperModel(args.model, compute_type="int8")

    quiet = args.json
    cache: dict[Path, str] = {}
    results: list[dict] = []                 # one entry per processed voice
    targets: list[tuple[Path, object, dict]] = []
    for jp in profiles:
        finding = audit.audit_one(jp, VOICES, cache, model)
        gap_count = len(finding.silence_gaps)
        rec = {"name": jp.stem, "gap_count": gap_count, "changed": False}
        results.append(rec)
        if gap_count > 0:
            targets.append((jp, finding, rec))

    # Dry run: emit results and stop BEFORE applying (this is the bug the QA caught)
    if not args.apply:
        if quiet:
            print(json.dumps({"voices": results}))
        elif not targets:
            print("no silence-gap voices to trim")
        else:
            print(f"would process {len(targets)} voice(s):")
            for jp, finding, _ in targets:
                print(f"  {jp.stem}  ({finding.wav_path.name})")
            print("\ndry run — pass --apply to actually trim and rewrite transcripts")
        return 1 if any(r["gap_count"] for r in results) else 0

    # Apply path
    if targets and shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH", file=sys.stderr)
        return 2
    import soundfile as sf
    trimmed_by_id: dict[tuple[int, int], bytes] = {}   # profiles often share one WAV
    for jp, finding, rec in targets:
        trimmed = trimmed_by_id.get(finding.wav_id)
        if trimmed is None:
            trimmed = trim_reference(finding)
            trimmed_by_id[finding.wav_id] = trimmed

        # Re-transcribe trimmed audio
        new_text = audit.transcribe(trimmed, model, "en")
        # Also recompute duration for the notes line
        info = sf.info(io.BytesIO(trimmed))

        profile = json.loads(jp.read_text())
        old_text = profile.get("reference_text", "")
        profile["reference_text"] = new_text
        notes = profile.get("notes", "")
        addition = (
            f"Reference re-trimmed via ffmpeg silenceremove "
            f"(now {info.duration:.1f}s dense). Transcript refreshed from Whisper."
        )
        profile["notes"] = (notes + " " + addition).strip() if notes else addition

        jp.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")
        rec["changed"] = True
        if not quiet:
            print(f"  ✓ {jp.stem}: {info.duration:.1f}s")
            if old_text != new_text:
                print(f"      old: {old_text[:80]}")
                print(f"      new: {new_text[:80]}")

    if quiet:
        print(json.dumps({"voices": results}))
    elif targets:
        print(f"\ntrimmed {len(targets)} voice(s). re-run audit to verify.")
    return 0   # apply succeeded — exit 0 (Unix convention). Dry-run returns 1 for "gaps found".


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
