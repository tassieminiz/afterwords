#!/usr/bin/env python3
"""
QA all base voice profiles: transcribe ref WAV, synthesize a test phrase,
transcribe synthesis output. Reports ref WER and synth WER per voice.

Usage:
    python3 scripts/qa-voices.py [--synth] [--ref-only] [--voice NAME] [--out results.tsv]

--synth      Also run synthesis tests (slow, ~10-30s per voice).
--ref-only   Only transcribe ref WAVs, skip synthesis even if --synth is set.
--voice NAME Test a single voice by name.
--out        Output file (default: $XDG_CACHE_HOME/afterwords/voice-qa.tsv,
             falling back to ~/.cache/afterwords/voice-qa.tsv)

The report holds every profile's stored transcript, so it is always written
owner-only (0600) and atomically replaced into place: an existing file or
symlink at the output path is swapped out, never written through.
"""
import argparse, json, os, re, secrets, stat, sys, tempfile, urllib.parse, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).parent.parent
TEST_PHRASE = "Good morning. Testing one, two, three. The quick brown fox jumps over the lazy dog."

REF_WER_THRESHOLD = 0.15  # machine/exit threshold (spec §5); human WARN-REF stays at 0.6
SERVER = "http://localhost:7860"

VARIANT_SUFFIXES = re.compile(r'-qwen3-\d+|-(ibuki|sanchin|picard)$')

TEST_WORDS = TEST_PHRASE.lower().split()


def edit_distance(a: list, b: list) -> int:
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(a)][len(b)]


def wer(ref: str, hyp: str) -> float:
    ref_w = ref.lower().split()
    hyp_w = hyp.lower().split()
    if not ref_w:
        return 0.0
    return edit_distance(ref_w, hyp_w) / len(ref_w)


def phrase_wer(ref: str, hyp: str) -> float:
    """WER against best-matching window in hyp — ignores preamble before the test phrase."""
    ref_w = ref.lower().split()
    hyp_w = hyp.lower().split()
    n = len(ref_w)
    if not n:
        return 0.0
    if not hyp_w:
        return 1.0
    # Slide a window of size n..n+n//2 over hyp and take the best WER
    slack = n // 2
    best = 1.0
    for start in range(len(hyp_w)):
        window = hyp_w[start:start + n + slack]
        if not window:
            break
        d = edit_distance(ref_w, window[:n])
        best = min(best, d / n)
        if best == 0.0:
            break
    return best


def transcribe(path: str, model):
    segs, info = model.transcribe(path, beam_size=3)
    text = " ".join(s.text.strip() for s in segs).strip()
    return text, info.language, info.language_probability


def build_flags(ref_wer: float, ref_lang: str, synth_wer: float | None) -> str:
    flags = []
    if ref_lang not in ("en", None, "") and ref_lang is not None:
        flags.append(f"WARN-REF-LANG({ref_lang})")
    if ref_wer > 0.6:
        flags.append("WARN-REF")
    if synth_wer is None:
        if not flags:
            return "REF-OK"
    else:
        if synth_wer > 0.4:
            flags.append("WARN-SYNTH")
    return "|".join(flags) if flags else "PASS"


def default_report_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "afterwords" / "voice-qa.tsv"


def _open_private_dir(path: Path) -> int:
    """Return an fd for an owner-only directory at ``path``, creating it if needed.

    The fd pins the directory that was checked; every later file operation is
    relative to it, so swapping the path afterwards has no effect.
    """
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if st.st_uid != os.geteuid():
            raise PermissionError(f"report directory {path} is not owned by the current user")
        if stat.S_IMODE(st.st_mode) & 0o077:
            os.fchmod(fd, 0o700)
    except BaseException:
        os.close(fd)
        raise
    return fd


def write_report(out: str | None, header: list, rows: list) -> Path:
    if out:
        target = Path(out)
        dir_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    else:
        target = default_report_path()
        dir_fd = _open_private_dir(target.parent)
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        for _ in range(100):
            tmp_name = f".{target.name}.{secrets.token_hex(8)}.tmp"
            try:
                fd = os.open(tmp_name, flags, 0o600, dir_fd=dir_fd)
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(f"could not create a temporary report in {target.parent}")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("\t".join(header) + "\n")
                for row in rows:
                    f.write("\t".join(str(c) for c in row) + "\n")
            # rename() replaces a symlink at the target name instead of following it.
            os.replace(tmp_name, target.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except BaseException:
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
            raise
    finally:
        os.close(dir_fd)
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--ref-only", action="store_true")
    ap.add_argument("--voice", default="")
    ap.add_argument("--out", default=None,
                    help="Report path (default: ~/.cache/afterwords/voice-qa.tsv, mode 0600)")
    ap.add_argument("--json", action="store_true", help="Emit JSON to stdout; suppress human output")
    args = ap.parse_args()

    quiet = args.json  # suppress all human stdout when --json is set

    run_synth = args.synth and not args.ref_only

    import faster_whisper
    if not quiet:
        print("Loading Whisper base model...", flush=True)
    model = faster_whisper.WhisperModel("base", device="cpu", compute_type="int8")

    jsons = sorted(p for p in (REPO / "voices").glob("*.json")
                   if not VARIANT_SUFFIXES.search(p.stem))

    if args.voice:
        jsons = [p for p in jsons if p.stem == args.voice or
                 json.loads(p.read_text()).get("name") == args.voice]
        if not jsons:
            print(f"Voice '{args.voice}' not found.", file=sys.stderr)
            sys.exit(2)

    if not quiet:
        print(f"Found {len(jsons)} base voice profiles.\n", flush=True)

    rows = []
    header = ["voice", "ref_wer", "ref_lang", "ref_transcript", "ref_stored",
              "synth_wer", "synth_transcript", "flag"]

    for i, jpath in enumerate(jsons, 1):
        with open(jpath) as f:
            profile = json.load(f)

        name = profile.get("name", jpath.stem)
        stored_ref = profile.get("reference_text", "")
        ref_wav = REPO / "voices" / profile.get("reference_audio", f"{name}-ref.wav")

        if not quiet:
            print(f"[{i:03d}/{len(jsons)}] {name}", end="  ", flush=True)

        # --- Ref WAV transcription ---
        if ref_wav.exists():
            ref_transcript, ref_lang, ref_lang_prob = transcribe(str(ref_wav), model)
            r_wer = wer(stored_ref, ref_transcript)
        else:
            ref_transcript = "MISSING"
            ref_lang = ""
            ref_lang_prob = 0.0
            r_wer = 1.0

        if not quiet:
            print(f"ref_wer={r_wer:.2f}", end="", flush=True)
            if ref_lang and ref_lang != "en":
                print(f"(lang={ref_lang}:{ref_lang_prob:.0%})", end="", flush=True)

        # --- Synthesis test ---
        s_wer = None
        synth_transcript = ""
        if run_synth:
            try:
                url = (f"{SERVER}/synthesize"
                       f"?text={urllib.parse.quote(TEST_PHRASE)}"
                       f"&voice={urllib.parse.quote(name)}")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                req = urllib.request.Request(url, headers={"Accept": "audio/wav"})
                with urllib.request.urlopen(req, timeout=90) as resp:
                    with open(tmp_path, "wb") as f:
                        f.write(resp.read())
                synth_transcript, _, _ = transcribe(tmp_path, model)
                s_wer = phrase_wer(TEST_PHRASE, synth_transcript)
                os.unlink(tmp_path)
                if not quiet:
                    print(f"  synth_wer={s_wer:.2f}", end="", flush=True)
            except Exception as e:
                synth_transcript = f"ERROR: {e}"
                s_wer = 1.0
                if not quiet:
                    print(f"  synth=ERROR", end="", flush=True)

        f_str = build_flags(r_wer, ref_lang, s_wer)
        if not quiet:
            print(f"  [{f_str}]", flush=True)

        rows.append([
            name,
            f"{r_wer:.3f}",
            ref_lang,
            ref_transcript,
            stored_ref,
            f"{s_wer:.3f}" if s_wer is not None else "",
            synth_transcript,
            f_str,
        ])

    report_path = write_report(args.out, header, rows)

    n_over = sum(1 for r in rows if float(r[1]) > REF_WER_THRESHOLD)

    if args.json:
        out = {
            "threshold": REF_WER_THRESHOLD,
            "voices": [
                {"name": r[0], "ref_wer": float(r[1]),
                 **({"synth_wer": float(r[5])} if r[5] else {})}
                for r in rows
            ],
        }
        print(json.dumps(out))
    else:
        print(f"\nResults written to {report_path}")

        warn = [r for r in rows if "WARN" in r[-1]]
        print(f"\nSummary: {len(rows)} voices, {len(warn)} warnings")
        if warn:
            print("\nWarnings:")
            for r in warn:
                print(f"  {r[0]:30s}  ref_wer={r[1]}  {r[-1]}")
                print(f"    stored:      {r[4][:80]}")
                print(f"    transcribed: {r[3][:80]}")

    # exit 1 if any voice is over the attention threshold; 0 otherwise
    sys.exit(1 if n_over else 0)


if __name__ == "__main__":
    try:
        sys.exit(main())          # preserves a returned code AND a sys.exit() inside main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
