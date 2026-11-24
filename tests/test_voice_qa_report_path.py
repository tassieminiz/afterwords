"""Spec for the fixed /tmp/voice-qa.tsv quality report.

These tests exercise the real ``scripts/qa-voices.py`` write, including the
``afterwords qa`` and ``afterwords refine`` entry points that call it without
overriding ``--out``. They fail while that path is opened in place with the
process umask: the report is other-readable, and an open of the fixed path
follows a symlink the kernel permits.

``afterwords clone`` is not invoked here. After a successful clone it calls
``cmd_refine``, which is the refine entry point covered below.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
QA_SCRIPT = REPO / "scripts" / "qa-voices.py"
AFTERWORDS = REPO / "afterwords.sh"
REPORT_PATH = Path("/tmp/voice-qa.tsv")
VOICE = "zzqa-priv"
ORIGINAL = "OPERATOR_OWNED_ORIGINAL\n"

# Directories that cannot contain the quality report and are expensive to walk.
_SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}


def _write_whisper_stub(dest: Path) -> None:
    """Stand-in for faster-whisper that returns one segment.

    The product imports the real package inside ``main`` only to transcribe
    audio. The report path, the stored transcript, and the open() that writes
    them do not pass through that package. This stub lets the real writer run
    without downloading a model. Its transcribe() result is a single text
    segment, which is the only shape ``qa-voices.py`` reads (``.text``).
    """
    pkg = dest / "faster_whisper"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        "class _Info:\n"
        "    language = 'en'\n"
        "    language_probability = 0.99\n"
        "class _Seg:\n"
        "    def __init__(self):\n"
        "        self.text = 'stub transcription'\n"
        "class WhisperModel:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        pass\n"
        "    def transcribe(self, path, beam_size=3):\n"
        "        return [_Seg()], _Info()\n"
    )


def _write_open_planter(dest: Path) -> None:
    """Plant a symlink at the fixed report path inside the write open.

    ``qa-voices.py`` calls the builtin ``open``. A fix that checks the path
    and then calls ``open`` or ``os.open`` again still reaches one of these
    wrappers, which swaps in the symlink only after those checks return.
    """
    (dest / "sitecustomize.py").write_text(
        "import os, stat, builtins\n"
        "_real_open = builtins.open\n"
        "_real_os_open = os.open\n"
        "_target = os.environ.get('QA_REPORT_PATH', '/tmp/voice-qa.tsv')\n"
        "_sentinel = os.environ.get('QA_SENTINEL', '')\n"
        "\n"
        "def _plant():\n"
        "    if not _sentinel:\n"
        "        return\n"
        "    try:\n"
        "        st = os.lstat(_target)\n"
        "    except FileNotFoundError:\n"
        "        os.symlink(_sentinel, _target)\n"
        "        return\n"
        "    if stat.S_ISLNK(st.st_mode):\n"
        "        return\n"
        "    os.unlink(_target)\n"
        "    os.symlink(_sentinel, _target)\n"
        "\n"
        "def _open(file, mode='r', *args, **kwargs):\n"
        "    if os.path.abspath(str(file)) == os.path.abspath(_target) and any(c in mode for c in 'wax+'):\n"
        "        _plant()\n"
        "    return _real_open(file, mode, *args, **kwargs)\n"
        "\n"
        "def _os_open(path, flags, mode=0o777, *args, **kwargs):\n"
        "    writing = flags & (os.O_WRONLY | os.O_RDWR)\n"
        "    if writing and os.path.abspath(str(path)) == os.path.abspath(_target):\n"
        "        _plant()\n"
        "    return _real_os_open(path, flags, mode, *args, **kwargs)\n"
        "\n"
        "builtins.open = _open\n"
        "os.open = _os_open\n"
    )


def _private_profile(marker: str) -> tuple[Path, Path]:
    voice_json = REPO / "voices" / f"{VOICE}.json"
    voice_wav = REPO / "voices" / f"{VOICE}-ref.wav"
    voice_json.write_text(json.dumps({
        "name": VOICE,
        "reference_text": marker,
        "reference_audio": voice_wav.name,
    }))
    # Other accounts must not already be able to read the stored transcript.
    voice_json.chmod(0o600)
    voice_wav.write_bytes(b"")
    return voice_json, voice_wav


def _cleanup_profile(voice_json: Path, voice_wav: Path) -> None:
    voice_json.unlink(missing_ok=True)
    voice_wav.unlink(missing_ok=True)
    if REPORT_PATH.is_symlink() or REPORT_PATH.exists():
        try:
            REPORT_PATH.unlink()
        except OSError:
            subprocess.run(["sudo", "-n", "rm", "-f", str(REPORT_PATH)], check=False)


def _ensure_venv() -> bool:
    """Return True when this test created the repo .venv directory."""
    venv = REPO / ".venv"
    if venv.exists():
        return False
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "activate").write_text("# test stub\n")
    return True


def _drop_venv(created: bool) -> None:
    if not created:
        return
    venv = REPO / ".venv"
    if venv.exists():
        subprocess.run(["rm", "-rf", str(venv)], check=False)


def _run(entry: str, stub_dir: Path, *, plant: bool, sentinel: Path | None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stub_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("QA_SENTINEL", None)
    env.pop("QA_REPORT_PATH", None)
    if plant:
        env["QA_SENTINEL"] = str(sentinel)
        env["QA_REPORT_PATH"] = str(REPORT_PATH)
    if entry == "script":
        argv = [
            "python3", str(QA_SCRIPT),
            "--voice", VOICE, "--ref-only", "--json",
        ]
    elif entry == "qa":
        argv = [
            "bash", str(AFTERWORDS),
            "qa", "--voice", VOICE, "--ref-only", "--json",
        ]
    elif entry == "refine":
        argv = [
            "bash", str(AFTERWORDS),
            "refine", VOICE, "--quick", "--yes",
        ]
    else:
        raise AssertionError(entry)
    # The product never sets umask. Pin the historical default so the mode
    # does not depend on whatever this shell inherited.
    return subprocess.run(
        ["bash", "-c", "umask 022; exec \"$@\"", "run", *argv],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )


def _files_containing(marker: str, ignore: set[Path]) -> list[Path]:
    needle = marker.encode()
    ignore_resolved = {p.resolve() for p in ignore}
    found: list[Path] = []
    roots = [Path("/tmp"), REPO, Path.home()]
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                path = Path(dirpath) / name
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if resolved in ignore_resolved or resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    st = path.lstat()
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode):
                    continue
                if stat.S_ISREG(st.st_mode) and st.st_size > 2_000_000:
                    continue
                try:
                    data = path.read_bytes()
                except OSError:
                    continue
                if needle in data:
                    found.append(path)
    return found


def _nobody_reads(path: Path) -> bool:
    result = subprocess.run(
        ["sudo", "-n", "-u", "nobody", "cat", "--", str(path)],
        capture_output=True,
    )
    return result.returncode == 0


def _classify(marker: str, ignore: set[Path]) -> tuple[list[str], list[Path]]:
    leaks: list[str] = []
    private: list[Path] = []
    for path in _files_containing(marker, ignore):
        if path.is_symlink() or _nobody_reads(path):
            leaks.append(f"{path} readable_by_other_account")
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            leaks.append(f"{path} mode {oct(mode)}")
            continue
        private.append(path)
    return leaks, private


@pytest.fixture()
def report_case(tmp_path, request):
    marker = f"PRIVATE_REF_TRANSCRIPT_{uuid.uuid4().hex}"
    _write_whisper_stub(tmp_path)
    if request.param["plant"]:
        _write_open_planter(tmp_path)
    voice_json, voice_wav = _private_profile(marker)
    venv_created = False
    sentinel = None
    try:
        if REPORT_PATH.is_symlink() or REPORT_PATH.exists():
            try:
                REPORT_PATH.unlink()
            except OSError:
                subprocess.run(["sudo", "-n", "rm", "-f", str(REPORT_PATH)], check=True)
        if request.param["symlink"] == "same-user":
            sentinel = tmp_path / "operator-owned"
            sentinel.write_text(ORIGINAL)
            sentinel.chmod(0o600)
            REPORT_PATH.symlink_to(sentinel)
        elif request.param["symlink"] == "other-user":
            sentinel = tmp_path / "operator-owned"
            sentinel.write_text(ORIGINAL)
            sentinel.chmod(0o600)
            subprocess.run(
                ["sudo", "-n", "-u", "nobody", "ln", "-s", str(sentinel), str(REPORT_PATH)],
                check=True,
            )
        elif request.param["plant"]:
            sentinel = tmp_path / "operator-owned"
            sentinel.write_text(ORIGINAL)
            sentinel.chmod(0o600)
        if request.param["entry"] in {"qa", "refine"}:
            venv_created = _ensure_venv()
        yield {
            "marker": marker,
            "profile": voice_json,
            "sentinel": sentinel,
            "entry": request.param["entry"],
            "plant": request.param["plant"],
            "stub": tmp_path,
        }
    finally:
        _cleanup_profile(voice_json, voice_wav)
        _drop_venv(venv_created)


# Each case is one product entry that reaches open(default_out, "w").
# symlink:
#   absent    — attacker leaves the fixed path missing (world-readable create)
#   same-user — symlink the follower is allowed to follow (uids match)
#   other-user — another account owns the symlink in sticky /tmp
# plant: swap the path to a symlink inside the write open, after any check
_CASES = [
    {"entry": "script", "symlink": "absent", "plant": False},
    {"entry": "qa", "symlink": "absent", "plant": False},
    {"entry": "refine", "symlink": "absent", "plant": False},
    {"entry": "script", "symlink": "same-user", "plant": False},
    {"entry": "refine", "symlink": "same-user", "plant": False},
    {"entry": "script", "symlink": "other-user", "plant": False},
    {"entry": "refine", "symlink": "other-user", "plant": False},
    {"entry": "script", "symlink": "absent", "plant": True},
    {"entry": "refine", "symlink": "absent", "plant": True},
]


def _case_id(case: dict) -> str:
    planted = "planted-at-open" if case["plant"] else case["symlink"]
    return f"{case['entry']}-{planted}"


@pytest.mark.parametrize("report_case", _CASES, indirect=True, ids=_case_id)
def test_quality_report_stays_private_and_does_not_replace_a_target(report_case):
    result = _run(
        report_case["entry"],
        report_case["stub"],
        plant=report_case["plant"],
        sentinel=report_case["sentinel"],
    )
    profile = report_case["profile"]
    assert not _nobody_reads(profile), (
        "the fixture transcript is already readable by another account"
    )
    sentinel = report_case["sentinel"]
    if sentinel is not None:
        assert sentinel.read_text() == ORIGINAL, (
            "writing the quality report replaced the file at the end of "
            f"{REPORT_PATH}:\n{result.stderr}"
        )
    ignore = {report_case["profile"]}
    if sentinel is not None:
        ignore.add(sentinel)
    leaks, private = _classify(report_case["marker"], ignore)
    assert not leaks, (
        "another local account can read the stored transcript from the "
        f"quality report:\n{leaks}\nstderr:\n{result.stderr}"
    )
    assert private, (
        "the stored transcript was not written to an owner-only report "
        f"(exit {result.returncode}):\n{result.stderr}\n{result.stdout}"
    )
