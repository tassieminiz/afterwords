"""Speech-archive writes must stay inside the archive directory.

A project ``.afterwords`` voice is attacker-controlled when someone else can
commit that file. Workers may archive a slug such as ``galadriel`` under the
integration archive directory. They must not create, modify, or delete
anything outside that directory when the voice contains ``..``, a slash, or
an absolute path.

The path that is checked has to be the path that is written. Retargeting a
directory symlink after a containment lookup must not make the transcript or
the mp3 land outside the archive.

Gemini CLI, Antigravity, and Cursor all drain ``~/.claude/hooks/tts-worker.sh``
as installed from the ``WORKEREOF`` heredoc in ``setup.sh``. That writer is
the Claude case below. The Hermes shell hooks name archives from the spoken
text, not the voice, and are not part of this bug.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import site
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LINK_NAME = "aw-escape-link"

STAT_SHIM = r"""#!/usr/bin/env python3
import os, sys
fmt = None
paths = []
args = sys.argv[1:]
i = 0
while i < len(args):
    a = args[i]
    if a == "-f" and i + 1 < len(args):
        fmt = args[i + 1]
        i += 2
        continue
    if a.startswith("-f") and len(a) > 2:
        fmt = a[2:]
        i += 1
        continue
    if a.startswith("-"):
        sys.stderr.write("unsupported stat args: %r\n" % (args,))
        sys.exit(1)
    paths.append(a)
    i += 1
if fmt is None or not paths:
    sys.stderr.write("stat usage: %r\n" % (args,))
    sys.exit(1)
st = os.stat(paths[-1])
out = []
j = 0
while j < len(fmt):
    if fmt[j] == "%" and j + 1 < len(fmt):
        spec = fmt[j + 1]
        if spec == "u":
            out.append(str(st.st_uid))
        elif spec == "z":
            out.append(str(st.st_size))
        else:
            sys.stderr.write("bad stat spec\n")
            sys.exit(1)
        j += 2
    else:
        out.append(fmt[j])
        j += 1
sys.stdout.write("".join(out) + "\n")
"""

CURL_SHIM = r"""#!/usr/bin/env bash
out=""
prev=""
for a in "$@"; do
  if [ "$prev" = "-o" ]; then
    out="$a"
  fi
  prev="$a"
done
if [ -n "$out" ]; then
  dd if=/dev/zero of="$out" bs=1500 count=1 status=none
fi
exit 0
"""

LAME_SHIM = r"""#!/usr/bin/env bash
printf 'mp3\n' > "${@: -1}"
exit 0
"""

FFMPEG_SHIM = r"""#!/usr/bin/env bash
printf 'trimmed\n' > "${@: -1}"
exit 0
"""

AFPLAY_SHIM = r"""#!/usr/bin/env bash
exit 0
"""

# Flips the named symlink to AW_TOCTOU_TARGET after the first successful
# readlink/readlinkat of that symlink. A containment check that resolves the
# link, then opens the original path, writes through the new target.
TOCTOU_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int flipped = 0;

static int ends_with_link(const char *path, const char *marker) {
    size_t plen = strlen(path);
    size_t mlen = strlen(marker);
    if (plen < mlen)
        return 0;
    if (strcmp(path + plen - mlen, marker) != 0)
        return 0;
    return plen == mlen || path[plen - mlen - 1] == '/';
}

static void flip_if_needed(const char *path) {
    const char *marker;
    const char *target;
    if (flipped || path == NULL || path[0] == '\0')
        return;
    marker = getenv("AW_TOCTOU_LINK");
    target = getenv("AW_TOCTOU_TARGET");
    if (marker == NULL || target == NULL || !ends_with_link(path, marker))
        return;
    flipped = 1;
    if (unlink(path) != 0)
        return;
    (void)symlink(target, path);
}

ssize_t readlink(const char *path, char *buf, size_t size) {
    static ssize_t (*real_readlink)(const char *, char *, size_t) = NULL;
    ssize_t n;
    if (!real_readlink) {
        real_readlink = dlsym(RTLD_NEXT, "readlink");
        if (!real_readlink) {
            errno = ENOSYS;
            return -1;
        }
    }
    n = real_readlink(path, buf, size);
    if (n >= 0)
        flip_if_needed(path);
    return n;
}

ssize_t readlinkat(int dirfd, const char *path, char *buf, size_t size) {
    static ssize_t (*real_readlinkat)(int, const char *, char *, size_t) = NULL;
    ssize_t n;
    if (!real_readlinkat) {
        real_readlinkat = dlsym(RTLD_NEXT, "readlinkat");
        if (!real_readlinkat) {
            errno = ENOSYS;
            return -1;
        }
    }
    n = real_readlinkat(dirfd, path, buf, size);
    if (n >= 0 && path && path[0] == '/')
        flip_if_needed(path);
    return n;
}
"""


def _two_chunk_text(marker: str) -> str:
    first = f"{marker} alpha " + ("syllable " * 16)
    second = f"{marker} beta " + ("syllable " * 16)
    return first.strip() + ". " + second.strip() + "."


def _clear_play_lock() -> None:
    pidfile = Path("/tmp/afterwords-play.pid")
    lock = Path("/tmp/afterwords-play.lock")
    holder = None
    if pidfile.is_file():
        raw = pidfile.read_text(encoding="utf-8", errors="replace").strip()
        if raw.isdigit():
            holder = int(raw)
    if holder is not None and holder != os.getpid():
        try:
            os.kill(holder, 0)
        except OSError:
            holder = None
        else:
            pytest.fail(f"/tmp/afterwords-play.lock is held by live pid {holder}")
    pidfile.unlink(missing_ok=True)
    if lock.exists() or lock.is_symlink():
        if lock.is_dir() and not lock.is_symlink():
            shutil.rmtree(lock, ignore_errors=True)
        else:
            lock.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _play_lock_guard():
    _clear_play_lock()
    yield
    _clear_play_lock()


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _tool_bin(directory: Path) -> Path:
    bindir = directory / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    _write_executable(bindir / "stat", STAT_SHIM)
    _write_executable(bindir / "curl", CURL_SHIM)
    _write_executable(bindir / "lame", LAME_SHIM)
    _write_executable(bindir / "ffmpeg", FFMPEG_SHIM)
    _write_executable(bindir / "afplay", AFPLAY_SHIM)
    return bindir


def _base_env(home: Path, bindir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = f"{bindir}:{env.get('PATH', '')}"
    # The temporary HOME would hide the user site that provides aiohttp.
    user_site = site.getusersitepackages()
    pythonpath = [user_site]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env.pop("LD_PRELOAD", None)
    env.pop("AW_TOCTOU_LINK", None)
    env.pop("AW_TOCTOU_TARGET", None)
    return env


def _entry_inside_archive(path: Path, archive: Path) -> bool:
    """True when the directory entry itself lives inside ``archive``.

    The leaf is not resolved, so a symlink stored inside the archive still
    counts as inside even when its target does not.
    """
    try:
        parent = path.parent.resolve()
        archive_real = archive.resolve()
    except OSError:
        return False
    return parent == archive_real or archive_real in parent.parents


def _outside_state(root: Path, archive: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    if not root.exists():
        return rows
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        for name in dirnames:
            entry = current / name
            if _entry_inside_archive(entry, archive):
                continue
            kind = "link" if entry.is_symlink() else "dir"
            target = os.readlink(entry) if entry.is_symlink() else ""
            rows.append((kind, str(entry.relative_to(root)), target))
        for name in filenames:
            entry = current / name
            if _entry_inside_archive(entry, archive):
                continue
            if entry.is_symlink():
                rows.append(("link", str(entry.relative_to(root)), os.readlink(entry)))
            else:
                digest = hashlib.sha256(entry.read_bytes()).hexdigest()
                rows.append(("file", str(entry.relative_to(root)), digest))
    return sorted(rows)


def _watched_state(roots: list[Path], archive: Path) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for root in roots:
        for kind, rel, extra in _outside_state(root, archive):
            rows.append((str(root), kind, rel, extra))
    return sorted(rows)


def _state_diff(before: list[tuple], after: list[tuple]) -> str:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    parts = []
    if added:
        parts.append("created or changed outside the archive: " + ", ".join(repr(item) for item in added))
    if removed:
        parts.append("removed outside the archive: " + ", ".join(repr(item) for item in removed))
    return "; ".join(parts)


def _prepare_home(tmp: Path) -> tuple[Path, Path, Path]:
    home = tmp / "home"
    project = tmp / "project"
    desktop = home / "Desktop"
    desktop.mkdir(parents=True)
    project.mkdir()
    (desktop / "sentinel.txt").write_text("keep-sentinel\n", encoding="utf-8")
    return home, project, desktop


def _codex_line(text: str) -> str:
    return json.dumps(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": text}],
            },
        }
    )


def _clear_codex_runtime(session: str) -> None:
    pidfile = Path(f"/tmp/codex-tts-worker-{session}.pid")
    lock = Path(f"/tmp/codex-tts-worker-{session}.lock")
    pidfile.unlink(missing_ok=True)
    if lock.exists():
        shutil.rmtree(lock, ignore_errors=True)
    _clear_play_lock()


def _run_codex(env: dict[str, str], project: Path, text: str, agent: str = "codex") -> subprocess.CompletedProcess[str]:
    """Queue through the Codex hook, then drain with the real worker.

    The hook is invoked the same way ``process_line`` in
    ``.claude/hooks/codex-tts-watch.sh`` invokes it. A live pidfile keeps the
    hook from backgrounding a second worker; this process then runs the
    worker in the foreground.
    """
    session = f"trav-{uuid.uuid4().hex}"
    env = env.copy()
    env["CODEX_THREAD_ID"] = session
    env["PROJECT_DIR"] = str(project)
    env["AGENT_TYPE"] = agent
    pidfile = Path(f"/tmp/codex-tts-worker-{session}.pid")
    queue = Path(f"/tmp/codex-tts-queue-{session}")
    try:
        pidfile.write_text(str(os.getpid()), encoding="utf-8")
        queued = subprocess.run(
            ["bash", str(REPO / ".claude/hooks/codex-tts-hook.sh")],
            input=_codex_line(text) + "\n",
            text=True,
            capture_output=True,
            env=env,
            timeout=30,
        )
        assert queued.returncode == 0, queued.stderr
        items = list(queue.glob("*.json")) if queue.is_dir() else []
        assert items, f"hook did not queue an item: {queued.stderr}"
        _clear_codex_runtime(session)
        completed = subprocess.run(
            ["bash", str(REPO / ".claude/hooks/codex-tts-worker.sh")],
            text=True,
            capture_output=True,
            env=env,
            timeout=60,
        )
        leftover = list(queue.glob("*.json")) if queue.is_dir() else []
        assert not leftover, f"worker left the queue item: {completed.stderr}"
        return completed
    finally:
        _clear_codex_runtime(session)
        shutil.rmtree(queue, ignore_errors=True)


def _claude_worker_source() -> str:
    text = (REPO / "setup.sh").read_text(encoding="utf-8")
    marker = "<<'WORKEREOF'\n"
    start = text.index(marker) + len(marker)
    end = text.index("\nWORKEREOF\n", start)
    source = text[start:end] + "\n"
    if 'ARCHIVE_BASE="$ARCHIVE_DIR/${VOICE:-default}-${STAMP}"' not in source:
        raise AssertionError("setup.sh worker heredoc did not contain the archive write")
    return source


def _install_claude_worker(home: Path) -> Path:
    hooks = home / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    worker = hooks / "tts-worker.sh"
    worker.write_text(_claude_worker_source(), encoding="utf-8")
    worker.chmod(0o755)
    shutil.copyfile(REPO / "strip_markdown.py", hooks / "strip-markdown.py")
    shutil.copyfile(REPO / "chunk_text.py", hooks / "chunk-text.py")
    return worker


def _run_claude(env: dict[str, str], project: Path, text: str) -> subprocess.CompletedProcess[str]:
    """Queue through the Cursor hook into the installed Claude worker.

    Cursor, Gemini CLI, and Antigravity share this worker. The hook is held
    off from spawning it by a live pidfile; the worker then runs in the
    foreground against the queue the hook just wrote.
    """
    pidfile = Path("/tmp/claude-tts-worker.pid")
    lock = Path("/tmp/claude-tts-worker.lock")
    queue = Path("/tmp/claude-tts-queue")
    saved = []
    if queue.is_dir():
        for item in queue.iterdir():
            dest = queue / f".held-{item.name}-{uuid.uuid4().hex}"
            item.rename(dest)
            saved.append(dest)
    try:
        pidfile.write_text(str(os.getpid()), encoding="utf-8")
        payload = json.dumps({"text": text, "workspace_roots": [str(project)]})
        queued = subprocess.run(
            ["bash", str(REPO / ".claude/hooks/cursor-tts-hook.sh")],
            input=payload,
            text=True,
            capture_output=True,
            env=env,
            timeout=30,
        )
        assert queued.returncode == 0, queued.stderr
        items = [p for p in queue.glob("*.json")] if queue.is_dir() else []
        assert items, f"cursor hook did not queue an item: {queued.stderr}"
        pidfile.unlink(missing_ok=True)
        if lock.exists():
            shutil.rmtree(lock, ignore_errors=True)
        _clear_play_lock()
        completed = subprocess.run(
            ["bash", str(env["HOME"]) + "/.claude/hooks/tts-worker.sh"],
            text=True,
            capture_output=True,
            env=env,
            timeout=60,
        )
        leftover = [p for p in queue.glob("*.json")] if queue.is_dir() else []
        assert not leftover, f"claude worker left the queue item: {completed.stderr}"
        return completed
    finally:
        pidfile.unlink(missing_ok=True)
        if lock.exists():
            shutil.rmtree(lock, ignore_errors=True)
        _clear_play_lock()
        if queue.is_dir():
            for item in queue.glob("*.json"):
                item.unlink(missing_ok=True)
            for item in queue.glob("*.claimed"):
                item.unlink(missing_ok=True)
        for held in saved:
            original = held.name
            # .held-<original-name>-<uuid> ; original may contain hyphens.
            marker = ".held-"
            rest = original[len(marker):]
            uuid_len = 32
            restored = rest[: -(uuid_len + 1)]
            held.rename(queue / restored)


def _assert_slug_archived(archive: Path, marker: str) -> None:
    files = [p for p in archive.rglob("*") if p.is_file()]
    texts = [p for p in files if p.name.startswith("galadriel-") and p.suffix == ".txt"]
    mp3s = [p for p in files if p.name.startswith("galadriel-") and p.suffix == ".mp3"]
    assert texts, f"slug voice was not archived inside {archive}"
    assert any(marker in p.read_text(encoding="utf-8", errors="replace") for p in texts)
    assert mp3s, f"slug voice did not produce an mp3 inside {archive}"
    for path in texts + mp3s:
        assert _entry_inside_archive(path, archive)


def _chunk_count(text: str) -> int:
    completed = subprocess.run(
        [sys.executable, str(REPO / "chunk_text.py")],
        input=text,
        text=True,
        capture_output=True,
        check=True,
    )
    return len([line for line in completed.stdout.splitlines() if line.strip()])


def _build_preload(directory: Path) -> Path:
    source = directory / "toctou.c"
    library = directory / "toctou.so"
    source.write_text(TOCTOU_C, encoding="utf-8")
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", str(library), str(source), "-ldl"],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


def _preload_env(env: dict[str, str], library: Path, outside: Path) -> dict[str, str]:
    updated = env.copy()
    updated["LD_PRELOAD"] = str(library)
    updated["AW_TOCTOU_LINK"] = LINK_NAME
    updated["AW_TOCTOU_TARGET"] = str(outside)
    return updated


class _AudioHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = b"R" * 2000
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return


@pytest.fixture(scope="module")
def tts_server():
    server = ThreadingHTTPServer(("127.0.0.1", 7860), _AudioHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _load_hermes():
    path = REPO / "hermes" / "hooks" / "afterwords-tts" / "handler.py"
    spec = importlib.util.spec_from_file_location("afterwords_hermes_handler_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _drive_hermes(module, project: Path, text: str) -> None:
    async def _run() -> None:
        await module.handle(
            "agent:end",
            {"response": text, "platform": "cli", "cwd": str(project)},
        )
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        if pending:
            await asyncio.gather(*pending)

    asyncio.run(_run())


def test_preload_retargets_symlink_after_first_resolution(tmp_path: Path) -> None:
    """The check-then-use harness itself resolves the old target, then the new one."""
    library = _build_preload(tmp_path)
    inside = tmp_path / "inside"
    outside = tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    link = tmp_path / LINK_NAME
    link.symlink_to(inside)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,sys; print(os.readlink(sys.argv[1])); print(os.readlink(sys.argv[1]))",
            str(link),
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "LD_PRELOAD": str(library),
            "AW_TOCTOU_LINK": LINK_NAME,
            "AW_TOCTOU_TARGET": str(outside),
        },
        check=True,
    )
    first, second = probe.stdout.splitlines()
    assert first == str(inside)
    assert second == str(outside)


def test_codex_voice_cannot_escape_speech_archive(tmp_path: Path) -> None:
    home, project, desktop = _prepare_home(tmp_path)
    archive = home / ".codex" / "tts-archive"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "sentinel.txt").write_text("keep-sentinel\n", encoding="utf-8")
    env = _base_env(home, _tool_bin(tmp_path))
    benign = _two_chunk_text("benign-codex-marker")
    assert _chunk_count(benign) >= 2
    (project / ".afterwords").write_text("codex: galadriel\n", encoding="utf-8")
    _run_codex(env, project, benign)
    archive.mkdir(parents=True, exist_ok=True)
    _assert_slug_archived(archive, "benign-codex-marker")

    absolute_outside = tmp_path / "absolute-outside"
    absolute_outside.mkdir()
    link = archive / LINK_NAME
    escapes: list[str] = []
    cases = [
        ("codex key", "codex: ../../Desktop/leak\n", "codex", None),
        ("default key", "default: ../../Desktop/leak\n", "codex", None),
        ("legacy line", "../../Desktop/leak\n", "codex", None),
        ("sibling", "codex: ../sibling-leak\n", "codex", None),
        ("slash symlink", f"codex: {LINK_NAME}/note\n", "codex", desktop),
        ("absolute", f"codex: {absolute_outside}/leak\n", "codex", None),
        ("missing parent", "codex: ../../brand-new-dir/leak\n", "codex", None),
    ]
    for label, body, agent, link_target in cases:
        if link.is_symlink() or link.exists():
            link.unlink()
        if link_target is not None:
            link.symlink_to(link_target)
        marker = f"codex-escape-{label.replace(' ', '-')}-{uuid.uuid4().hex}"
        text = _two_chunk_text(marker)
        (project / ".afterwords").write_text(body, encoding="utf-8")
        before = _watched_state([home, absolute_outside], archive)
        _run_codex(env, project, text, agent=agent)
        after = _watched_state([home, absolute_outside], archive)
        if before != after:
            escapes.append(f"{label}: {_state_diff(before, after)}")
        if link.is_symlink() or link.exists():
            link.unlink()
    assert not escapes, "Codex archive write left the archive directory:\n" + "\n".join(escapes)


def test_codex_archive_write_uses_the_checked_path(tmp_path: Path) -> None:
    """A lookup that sees an in-archive symlink must not be followed by a write of the original path."""
    home, project, _desktop = _prepare_home(tmp_path)
    archive = home / ".codex" / "tts-archive"
    archive.mkdir(parents=True)
    inside = archive / "inside-bucket"
    inside.mkdir()
    outside = home / "toctou-outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("keep-sentinel\n", encoding="utf-8")
    link = archive / LINK_NAME
    link.symlink_to(inside)
    env = _preload_env(_base_env(home, _tool_bin(tmp_path)), _build_preload(tmp_path), outside)
    benign = _two_chunk_text("benign-codex-toctou")
    (project / ".afterwords").write_text("codex: galadriel\n", encoding="utf-8")
    plain = env.copy()
    plain.pop("LD_PRELOAD", None)
    _run_codex(plain, project, benign)
    _assert_slug_archived(archive, "benign-codex-toctou")
    # The benign run does not use the symlink. Point it inside again in case
    # a lookup during that run retargeted it.
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(inside)
    marker = "codex-toctou-marker"
    (project / ".afterwords").write_text(f"codex: {LINK_NAME}/note\n", encoding="utf-8")
    before = _outside_state(home, archive)
    _run_codex(env, project, _two_chunk_text(marker))
    after = _outside_state(home, archive)
    assert before == after, _state_diff(before, after)


def test_claude_worker_voice_cannot_escape_speech_archive(tmp_path: Path) -> None:
    home, project, desktop = _prepare_home(tmp_path)
    _install_claude_worker(home)
    archive = home / ".claude" / "tts-archive"
    (home / ".claude" / "sentinel.txt").write_text("keep-sentinel\n", encoding="utf-8")
    env = _base_env(home, _tool_bin(tmp_path))
    benign = _two_chunk_text("benign-claude-marker")
    assert _chunk_count(benign) >= 2
    (project / ".afterwords").write_text("cursor: galadriel\n", encoding="utf-8")
    _run_claude(env, project, benign)
    archive.mkdir(parents=True, exist_ok=True)
    _assert_slug_archived(archive, "benign-claude-marker")

    absolute_outside = tmp_path / "absolute-outside"
    absolute_outside.mkdir()
    link = archive / LINK_NAME
    escapes: list[str] = []
    cases = [
        ("cursor key", "cursor: ../../Desktop/leak\n", desktop),
        ("default key", "default: ../../Desktop/leak\n", None),
        ("legacy line", "../../Desktop/leak\n", None),
        ("sibling", "cursor: ../sibling-leak\n", None),
        ("slash symlink", f"cursor: {LINK_NAME}/note\n", desktop),
        ("absolute", f"cursor: {absolute_outside}/leak\n", None),
        ("missing parent", "cursor: ../../brand-new-dir/leak\n", None),
    ]
    for label, body, link_target in cases:
        if link.is_symlink() or link.exists():
            link.unlink()
        if link_target is not None:
            link.symlink_to(link_target)
        marker = f"claude-escape-{label.replace(' ', '-')}-{uuid.uuid4().hex}"
        (project / ".afterwords").write_text(body, encoding="utf-8")
        before = _watched_state([home, absolute_outside], archive)
        _run_claude(env, project, _two_chunk_text(marker))
        after = _watched_state([home, absolute_outside], archive)
        if before != after:
            escapes.append(f"{label}: {_state_diff(before, after)}")
        if link.is_symlink() or link.exists():
            link.unlink()
    assert not escapes, "Claude worker archive write left the archive directory:\n" + "\n".join(escapes)


def test_claude_worker_archive_write_uses_the_checked_path(tmp_path: Path) -> None:
    home, project, _desktop = _prepare_home(tmp_path)
    _install_claude_worker(home)
    archive = home / ".claude" / "tts-archive"
    archive.mkdir(parents=True)
    inside = archive / "inside-bucket"
    inside.mkdir()
    outside = home / "toctou-outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("keep-sentinel\n", encoding="utf-8")
    link = archive / LINK_NAME
    link.symlink_to(inside)
    env = _preload_env(_base_env(home, _tool_bin(tmp_path)), _build_preload(tmp_path), outside)
    (project / ".afterwords").write_text("cursor: galadriel\n", encoding="utf-8")
    plain = env.copy()
    plain.pop("LD_PRELOAD", None)
    _run_claude(plain, project, _two_chunk_text("benign-claude-toctou"))
    _assert_slug_archived(archive, "benign-claude-toctou")
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(inside)
    (project / ".afterwords").write_text(f"cursor: {LINK_NAME}/note\n", encoding="utf-8")
    before = _outside_state(home, archive)
    _run_claude(env, project, _two_chunk_text("claude-toctou-marker"))
    after = _outside_state(home, archive)
    assert before == after, _state_diff(before, after)


def test_hermes_voice_cannot_escape_speech_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tts_server) -> None:
    del tts_server
    home, project, desktop = _prepare_home(tmp_path)
    archive = home / ".hermes" / "tts-archive"
    archive.mkdir(parents=True)
    (home / ".hermes" / "sentinel.txt").write_text("keep-sentinel\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{_tool_bin(tmp_path)}:{os.environ.get('PATH', '')}")
    module = _load_hermes()
    benign = _two_chunk_text("benign-hermes-marker")
    assert len(module.chunk_text(module.strip_markdown(benign))) >= 2
    (project / ".afterwords").write_text("hermes: galadriel\n", encoding="utf-8")
    _drive_hermes(module, project, benign)
    _assert_slug_archived(archive, "benign-hermes-marker")

    absolute_outside = tmp_path / "absolute-outside"
    absolute_outside.mkdir()
    link = archive / LINK_NAME
    escapes: list[str] = []
    cases = [
        ("hermes key", "hermes: ../../Desktop/leak\n", desktop),
        ("default key", "default: ../../Desktop/leak\n", None),
        ("legacy line", "../../Desktop/leak\n", None),
        ("sibling", "hermes: ../sibling-leak\n", None),
        ("slash symlink", f"hermes: {LINK_NAME}/note\n", desktop),
        ("absolute", f"hermes: {absolute_outside}/leak\n", None),
        ("missing parent", "hermes: ../../brand-new-dir/leak\n", None),
    ]
    for label, body, link_target in cases:
        if link.is_symlink() or link.exists():
            link.unlink()
        if link_target is not None:
            link.symlink_to(link_target)
        (project / ".afterwords").write_text(body, encoding="utf-8")
        before = _watched_state([home, absolute_outside], archive)
        _drive_hermes(module, project, _two_chunk_text(f"hermes-escape-{label.replace(' ', '-')}"))
        after = _watched_state([home, absolute_outside], archive)
        if before != after:
            escapes.append(f"{label}: {_state_diff(before, after)}")
        if link.is_symlink() or link.exists():
            link.unlink()
    assert not escapes, "Hermes archive write left the archive directory:\n" + "\n".join(escapes)


def test_hermes_archive_write_uses_the_checked_path(tmp_path: Path, tts_server) -> None:
    del tts_server
    home, project, _desktop = _prepare_home(tmp_path)
    archive = home / ".hermes" / "tts-archive"
    archive.mkdir(parents=True)
    inside = archive / "inside-bucket"
    inside.mkdir()
    outside = home / "toctou-outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("keep-sentinel\n", encoding="utf-8")
    library = _build_preload(tmp_path)
    script = tmp_path / "drive_hermes.py"
    script.write_text(
        f"""
import asyncio, importlib.util, os, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "afterwords_hermes_handler_under_test",
    {str(REPO / "hermes/hooks/afterwords-tts/handler.py")!r},
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
async def run():
    await module.handle(
        "agent:end",
        {{"response": sys.argv[1], "platform": "cli", "cwd": sys.argv[2]}},
    )
    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending)
asyncio.run(run())
""",
        encoding="utf-8",
    )
    env = _preload_env(_base_env(home, _tool_bin(tmp_path)), library, outside)
    plain = env.copy()
    plain.pop("LD_PRELOAD", None)
    (project / ".afterwords").write_text("hermes: galadriel\n", encoding="utf-8")
    benign = subprocess.run(
        [sys.executable, str(script), _two_chunk_text("benign-hermes-toctou"), str(project)],
        env=plain,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert benign.returncode == 0, benign.stderr
    _assert_slug_archived(archive, "benign-hermes-toctou")
    link = archive / LINK_NAME
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(inside)
    (project / ".afterwords").write_text(f"hermes: {LINK_NAME}/note\n", encoding="utf-8")
    before = _outside_state(home, archive)
    escaped = subprocess.run(
        [sys.executable, str(script), _two_chunk_text("hermes-toctou-marker"), str(project)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert escaped.returncode == 0, escaped.stderr
    after = _outside_state(home, archive)
    assert before == after, _state_diff(before, after)
