"""Security spec for scripts/internal/qa-transcripts.py source reads.

These tests fail while a mapped source file is followed when it is a symlink,
when a parent directory is a symlink, or when the path changes after a stat
and before the open that reads it. The file bytes then show up in the gemini
child's argv and command line. They pass once that read cannot observe a
path outside SITE, including across the check and the open.

review-content.py is not covered here: its prompt template raises KeyError
before it starts gemini, so a symlink read in that script does not reach a
child process.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HEAD = "PLACEHOLDER_OUTSIDE_FILE_HEAD"
TAIL = "PLACEHOLDER_OUTSIDE_FILE_TAIL"
PUBLIC_SOURCE = "public source sentence about afterwords"
VIDEO_ID = "iawRevXb9i4"

# The open audit event is emitted before the kernel open. Swapping here is
# the window between Path.exists/resolve (stat) and Path.read_text (open).
_STATE = {
    "active": False,
    "busy": False,
    "armed": False,
    "secret": "",
    "watched": frozenset(),
    "poison": frozenset(),
}
_REAL_STAT = os.stat
_REAL_LSTAT = os.lstat
_HOOK_INSTALLED = False


def _norm(path: object) -> str:
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    try:
        return os.path.abspath(os.fspath(path))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""


def _note(path: object) -> None:
    if _STATE["active"] and _norm(path) in _STATE["watched"]:
        _STATE["armed"] = True


def _wrapped_stat(path, *args, **kwargs):
    result = _REAL_STAT(path, *args, **kwargs)
    _note(path)
    return result


def _wrapped_lstat(path, *args, **kwargs):
    result = _REAL_LSTAT(path, *args, **kwargs)
    _note(path)
    return result


def _poison() -> None:
    secret = _STATE["secret"]
    for path in _STATE["poison"]:
        if not path or path == secret:
            continue
        if os.path.lexists(path):
            os.unlink(path)
        os.symlink(secret, path)


def _audit(event: str, args) -> None:
    if event != "open" or not _STATE["active"] or _STATE["busy"] or not _STATE["armed"]:
        return
    if not args or _norm(args[0]) not in _STATE["poison"]:
        return
    _STATE["busy"] = True
    try:
        _poison()
    finally:
        _STATE["busy"] = False


def _install_hooks() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    os.stat = _wrapped_stat  # type: ignore[assignment]
    os.lstat = _wrapped_lstat  # type: ignore[assignment]
    sys.addaudithook(_audit)
    _HOOK_INSTALLED = True


def _activate_toctou(secret: Path, watched: set[Path], poison: set[Path]) -> None:
    _install_hooks()
    _STATE["secret"] = str(secret)
    _STATE["watched"] = frozenset(str(p) for p in watched)
    _STATE["poison"] = frozenset(str(p) for p in poison)
    _STATE["armed"] = False
    _STATE["busy"] = False
    _STATE["active"] = True


def _deactivate_toctou() -> None:
    _STATE["active"] = False
    _STATE["armed"] = False
    _STATE["busy"] = False


@pytest.fixture(autouse=True)
def _toctou_off():
    _deactivate_toctou()
    yield
    _deactivate_toctou()


def _load_qa():
    path = REPO / "scripts" / "internal" / "qa-transcripts.py"
    spec = importlib.util.spec_from_file_location("qa_transcripts_symlink_spec", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["qa_transcripts_symlink_spec"] = module
    spec.loader.exec_module(module)
    return module


QA = _load_qa()


def _secret_text() -> str:
    # Long enough that a missing cap copies the tail, short enough for execve.
    return HEAD + ("." * 20000) + TAIL


def _write_secret(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_secret_text(), encoding="utf-8")
    path.chmod(0o600)
    return path


def _install_gemini(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    capture = tmp / "gemini-capture.json"
    stub = bin_dir / "gemini"
    stub.write_text(
        """#!/usr/bin/env python3
import json, os, select, stat, sys

def read_stdin():
    try:
        mode = os.fstat(0).st_mode
    except OSError:
        return b""
    if not (stat.S_ISFIFO(mode) or stat.S_ISREG(mode)):
        return b""
    chunks = []
    while True:
        ready, _, _ = select.select([0], [], [], 0.5)
        if not ready:
            break
        piece = os.read(0, 1024 * 1024)
        if not piece:
            break
        chunks.append(piece)
    return b"".join(chunks)

stdin = read_stdin()
cmdline = b""
cmdline_mode = ""
proc = "/proc/self/cmdline"
if os.path.exists(proc):
    with open(proc, "rb") as fh:
        cmdline = fh.read()
    cmdline_mode = oct(os.stat(proc).st_mode & 0o777)
payload = {
    "argv": sys.argv,
    "cmdline": cmdline.decode("utf-8", "surrogateescape"),
    "stdin": stdin.decode("utf-8", "surrogateescape"),
    "cmdline_mode": cmdline_mode,
}
with open(os.environ["TRANSCRIPT_QA_CAPTURE"], "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
print(json.dumps({"response": json.dumps({"status": "clean", "issues": []})}))
""",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    real_env = QA.minimal_env

    def wrapped(*args, **kwargs):
        env = real_env(*args, **kwargs)
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        env["TRANSCRIPT_QA_CAPTURE"] = str(capture)
        return env

    monkeypatch.setattr(QA, "minimal_env", wrapped)
    return capture


def _blob(capture: Path, qa_out: Path) -> str:
    parts: list[str] = []
    if capture.exists():
        parts.append(capture.read_text(encoding="utf-8", errors="replace"))
    if qa_out.exists():
        for path in qa_out.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _assert_secret_stays_put(capture: Path, qa_out: Path) -> None:
    blob = _blob(capture, qa_out)
    assert HEAD not in blob
    assert TAIL not in blob


def _content_tree(tmp: Path) -> tuple[Path, Path, Path]:
    site = tmp / "content"
    blog = site / "blog"
    projects = site / "projects"
    blog.mkdir(parents=True)
    projects.mkdir()
    return site, blog, projects


def _bind_qa(
    monkeypatch: pytest.MonkeyPatch,
    site: Path,
    blog: Path,
    projects: Path,
    transcripts: Path,
    qa_out: Path,
) -> None:
    transcripts.mkdir(parents=True, exist_ok=True)
    titles = transcripts / "video-titles.tsv"
    titles.write_text(f"{VIDEO_ID}\tAfterwords\n", encoding="utf-8")
    (transcripts / f"{VIDEO_ID}.json").write_text(
        json.dumps([{"start": 0.0, "word": "hello"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(QA, "SITE", site)
    monkeypatch.setattr(QA, "BLOG", blog)
    monkeypatch.setattr(QA, "PROJ", projects)
    monkeypatch.setattr(QA, "TRANSCRIPTS", transcripts)
    monkeypatch.setattr(QA, "TITLES_FILE", titles)
    monkeypatch.setattr(QA, "QA_OUT", qa_out)
    monkeypatch.setattr(
        QA,
        "CONTENT_MAP",
        {"Afterwords": [projects / "afterwords.md", blog / "afterwords-post.md"]},
    )


def _run_qa(monkeypatch: pytest.MonkeyPatch) -> None:
    # No QA note is present, which is the branch main() actually synthesizes.
    monkeypatch.setattr(sys, "argv", ["qa-transcripts.py", VIDEO_ID])
    QA.main()


def test_qa_symlink_source_is_not_passed_to_gemini(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    site, blog, projects = _content_tree(tmp_path)
    secret = _write_secret(tmp_path / "outside" / "draft.txt")
    (projects / "afterwords.md").symlink_to(secret)
    (blog / "afterwords-post.md").write_text(PUBLIC_SOURCE, encoding="utf-8")
    qa_out = tmp_path / "qa-out"
    _bind_qa(monkeypatch, site, blog, projects, tmp_path / "transcripts", qa_out)
    capture = _install_gemini(tmp_path, monkeypatch)

    _run_qa(monkeypatch)

    _assert_secret_stays_put(capture, qa_out)


def test_qa_parent_directory_symlink_is_not_passed_to_gemini(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    site, blog, _projects = _content_tree(tmp_path)
    outside = tmp_path / "outside-projects"
    outside.mkdir()
    secret = _write_secret(outside / "afterwords.md")
    projects_link = site / "linked-projects"
    projects_link.symlink_to(outside, target_is_directory=True)
    (blog / "afterwords-post.md").write_text(PUBLIC_SOURCE, encoding="utf-8")
    qa_out = tmp_path / "qa-out"
    _bind_qa(monkeypatch, site, blog, projects_link, tmp_path / "transcripts", qa_out)
    capture = _install_gemini(tmp_path, monkeypatch)

    _run_qa(monkeypatch)

    _assert_secret_stays_put(capture, qa_out)
    assert secret.is_file()


def test_qa_toctou_retarget_between_check_and_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Safe at stat time, outside file at the following open.

    afterwords.md starts as a symlink to a regular file inside the content
    tree. The first stat/lstat arms a swap: the next open of either path
    finds a symlink to the outside file.
    """
    site, blog, projects = _content_tree(tmp_path)
    inside = projects / "inside-body.md"
    inside.write_text(PUBLIC_SOURCE, encoding="utf-8")
    source = projects / "afterwords.md"
    source.symlink_to(inside)
    (blog / "afterwords-post.md").write_text(PUBLIC_SOURCE, encoding="utf-8")
    secret = _write_secret(tmp_path / "outside" / "draft.txt")
    qa_out = tmp_path / "qa-out"
    _bind_qa(monkeypatch, site, blog, projects, tmp_path / "transcripts", qa_out)
    capture = _install_gemini(tmp_path, monkeypatch)
    _activate_toctou(secret, {source, inside}, {source, inside})
    try:
        _run_qa(monkeypatch)
    finally:
        _deactivate_toctou()

    _assert_secret_stays_put(capture, qa_out)


def test_qa_regular_file_inside_content_is_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    site, blog, projects = _content_tree(tmp_path)
    (projects / "afterwords.md").write_text(PUBLIC_SOURCE, encoding="utf-8")
    (blog / "afterwords-post.md").write_text("second public page", encoding="utf-8")
    qa_out = tmp_path / "qa-out"
    _bind_qa(monkeypatch, site, blog, projects, tmp_path / "transcripts", qa_out)
    capture = _install_gemini(tmp_path, monkeypatch)

    _run_qa(monkeypatch)

    blob = _blob(capture, qa_out)
    assert PUBLIC_SOURCE in blob
    assert "second public page" in blob
