"""Other local users must not be able to read queued speech.

The Codex watcher records the thread id in its log and the hook writes each
final answer into ``/tmp/codex-tts-queue-<thread id>``. These tests drive that
path. They fail while the answer, the project path, or the thread id is
readable by a different local user, and they pass once a queued answer stays
in a directory that user cannot read.

The same unchecked ``mkdir -p`` then JSON write is covered for the Cursor,
Gemini CLI, and Antigravity hooks, which share ``/tmp/claude-tts-queue``.

Not covered here:

- ``setup.sh`` already refuses a symlink or a directory it does not own before
  the Claude hook writes (the generated ``tts-hook.sh``).
- Hermes writes speech audio through ``mktemp``, not a shared answer queue.
- ``codex-tts-worker.sh`` writes ``/tmp/codex-tts-chunks-*`` only after its
  ownership check accepts the queue. On this platform that check exits first,
  so those chunk files are never created.
"""

from __future__ import annotations

import ctypes
import fcntl
import json
import os
import select
import shlex
import struct
import subprocess
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOKS = REPO / ".claude" / "hooks"
ATTACKER = "awattacker"
CODEX_HOOK = HOOKS / "codex-tts-hook.sh"
WATCH_LOG = Path("/tmp/codex-tts-watch.log")
WATCH_PID = Path("/tmp/codex-tts-watch.pid")

# linux/inotify.h — events delivered when a watched directory gains a file.
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_NONBLOCK = 0o4000
_WRITE_EVENTS = IN_CREATE | IN_MOVED_TO | IN_CLOSE_WRITE
_EVENT_HEADER = struct.Struct("iIII")

libc = ctypes.CDLL(None, use_errno=True)
libc.inotify_init1.argtypes = [ctypes.c_int]
libc.inotify_init1.restype = ctypes.c_int
libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
libc.inotify_add_watch.restype = ctypes.c_int

# Replaces the queue directory with a second directory the moment a path-based
# open or rename targets a file inside it. A later write through an already
# opened directory fd does not call these with the absolute path, so it stays
# on the directory that was opened.
_SWAP_PRELOAD = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef AT_FDCWD
#define AT_FDCWD (-100)
#endif

static int (*real_open)(const char *, int, ...);
static int (*real_open64)(const char *, int, ...);
static int (*real_openat)(int, const char *, int, ...);
static int (*real_openat64)(int, const char *, int, ...);
static int (*real_rename)(const char *, const char *);
static int (*real_renameat)(int, const char *, int, const char *);
static int (*real_renameat2)(int, const char *, int, const char *, unsigned int);
static int swapped = 0;

__attribute__((constructor)) static void resolve_syms(void) {
    real_open = dlsym(RTLD_NEXT, "open");
    real_open64 = dlsym(RTLD_NEXT, "open64");
    real_openat = dlsym(RTLD_NEXT, "openat");
    real_openat64 = dlsym(RTLD_NEXT, "openat64");
    real_rename = dlsym(RTLD_NEXT, "rename");
    real_renameat = dlsym(RTLD_NEXT, "renameat");
    real_renameat2 = dlsym(RTLD_NEXT, "renameat2");
}

static int under_queue(const char *path) {
    const char *queue = getenv("AW_QUEUE");
    size_t n;
    if (!queue || !path || path[0] != '/')
        return 0;
    n = strlen(queue);
    return strncmp(path, queue, n) == 0 && path[n] == '/';
}

static void swap_once(void) {
    const char *queue;
    const char *exposed;
    const char *saved;
    if (swapped)
        return;
    queue = getenv("AW_QUEUE");
    exposed = getenv("AW_EXPOSED");
    saved = getenv("AW_SAVED");
    if (!queue || !exposed || !saved || !real_rename)
        _exit(97);
    if (real_rename(queue, saved) != 0)
        _exit(97);
    if (real_rename(exposed, queue) != 0) {
        real_rename(saved, queue);
        _exit(98);
    }
    swapped = 1;
}

static int write_open(const char *path, int flags) {
    if (!under_queue(path))
        return 0;
    if (flags & (O_CREAT | O_WRONLY | O_RDWR)) {
        swap_once();
        return 1;
    }
    return 0;
}

int open(const char *path, int flags, ...) {
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    write_open(path, flags);
    if (flags & (O_CREAT | O_TMPFILE))
        return real_open(path, flags, mode);
    return real_open(path, flags);
}

int open64(const char *path, int flags, ...) {
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    write_open(path, flags);
    if (flags & (O_CREAT | O_TMPFILE))
        return real_open64(path, flags, mode);
    return real_open64(path, flags);
}

int openat(int dirfd, const char *path, int flags, ...) {
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    if (dirfd == AT_FDCWD)
        write_open(path, flags);
    if (flags & (O_CREAT | O_TMPFILE))
        return real_openat(dirfd, path, flags, mode);
    return real_openat(dirfd, path, flags);
}

int openat64(int dirfd, const char *path, int flags, ...) {
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    if (dirfd == AT_FDCWD)
        write_open(path, flags);
    if (flags & (O_CREAT | O_TMPFILE))
        return real_openat64(dirfd, path, flags, mode);
    return real_openat64(dirfd, path, flags);
}

int rename(const char *old, const char *new) {
    if (under_queue(new))
        swap_once();
    return real_rename(old, new);
}

int renameat(int olddirfd, const char *old, int newdirfd, const char *new) {
    if (newdirfd == AT_FDCWD && under_queue(new))
        swap_once();
    return real_renameat(olddirfd, old, newdirfd, new);
}

int renameat2(int olddirfd, const char *old, int newdirfd, const char *new,
              unsigned int flags) {
    if (new && (newdirfd == AT_FDCWD || new[0] == '/') && under_queue(new))
        swap_once();
    return real_renameat2(olddirfd, old, newdirfd, new, flags);
}
"""


@pytest.fixture(scope="module")
def attacker_uid() -> int:
    probe = subprocess.run(["id", "-u", ATTACKER], capture_output=True, text=True)
    if probe.returncode != 0:
        subprocess.run(
            [
                "sudo",
                "-n",
                "useradd",
                "--no-create-home",
                "--shell",
                "/usr/sbin/nologin",
                ATTACKER,
            ],
            check=True,
        )
    return int(subprocess.check_output(["id", "-u", ATTACKER], text=True).strip())


@pytest.fixture
def agy_helpers():
    """The Antigravity hook loads its helpers from its own directory."""
    created: list[Path] = []
    mapping = {
        "agy-session-hook.py": REPO / "agy_session_hook.py",
        "strip-markdown.py": REPO / "strip_markdown.py",
    }
    for name, target in mapping.items():
        dest = HOOKS / name
        if dest.exists() or dest.is_symlink():
            continue
        dest.symlink_to(target)
        created.append(dest)
    try:
        yield
    finally:
        for dest in created:
            if dest.is_symlink():
                dest.unlink()


@contextmanager
def queue_lock():
    handle = open("/tmp/afterwords-queue-test.lock", "a")
    fcntl.flock(handle, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


class DirectoryWriteWatch:
    """Record creates, renames into, and write-closes inside one directory inode."""

    def __init__(self, path: Path):
        self.path = path
        self.events: list[tuple[int, str]] = []
        self._fd = -1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "DirectoryWriteWatch":
        self._fd = libc.inotify_init1(IN_NONBLOCK)
        if self._fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        watch = libc.inotify_add_watch(
            self._fd, os.fsencode(self.path), _WRITE_EVENTS
        )
        if watch < 0:
            err = ctypes.get_errno()
            os.close(self._fd)
            raise OSError(err, f"inotify_add_watch {self.path}")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._drain()
        if self._fd >= 0:
            os.close(self._fd)

    def _loop(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if ready:
                self._drain()

    def _drain(self) -> None:
        while True:
            try:
                data = os.read(self._fd, 8192)
            except BlockingIOError:
                return
            if not data:
                return
            offset = 0
            while offset + _EVENT_HEADER.size <= len(data):
                _wd, mask, _cookie, length = _EVENT_HEADER.unpack_from(data, offset)
                start = offset + _EVENT_HEADER.size
                name = data[start : start + length].split(b"\x00", 1)[0]
                offset = start + length
                if mask & _WRITE_EVENTS:
                    self.events.append((mask, name.decode("utf-8", "replace")))


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _attacker_mkdir(path: Path) -> os.stat_result:
    if path.is_symlink() or path.exists():
        subprocess.run(["sudo", "-n", "rm", "-rf", str(path)], check=True)
    subprocess.run(
        ["sudo", "-n", "-u", ATTACKER, "mkdir", "-m", "0777", str(path)],
        check=True,
    )
    st = path.stat()
    assert st.st_uid != os.getuid()
    assert st.st_mode & 0o777 == 0o777
    return st


def _same_directory(path: Path, before: os.stat_result) -> None:
    st = path.lstat()
    assert not path.is_symlink()
    assert st.st_dev == before.st_dev and st.st_ino == before.st_ino
    assert st.st_uid == before.st_uid
    assert st.st_mode & 0o777 == before.st_mode & 0o777


def _attacker_hits(token: str) -> list[str]:
    script = (
        "find /tmp -xdev -maxdepth 4 -type f -size -512k -readable -print0 "
        "2>/dev/null | xargs -0 -r grep -l -F -e "
        + shlex.quote(token)
        + " -- 2>/dev/null || true"
    )
    result = _run(["sudo", "-n", "-u", ATTACKER, "bash", "-c", script])
    return [line for line in result.stdout.splitlines() if line]


def _victim_copies(token: str, roots: list[Path], exclude: list[Path]) -> list[Path]:
    excluded = {path.resolve() for path in exclude}
    found: list[Path] = []
    for root in roots:
        if not root.exists() or root.is_symlink():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            current = Path(dirpath).resolve()
            if current in excluded or any(parent in excluded for parent in current.parents):
                dirnames.clear()
                continue
            dirnames[:] = [
                name
                for name in dirnames
                if (current / name).resolve() not in excluded
            ]
            for name in filenames:
                path = Path(dirpath) / name
                if path.resolve() in excluded or path.is_symlink():
                    continue
                try:
                    if path.stat().st_size > 512_000:
                        continue
                    text = path.read_text(errors="ignore")
                except OSError:
                    continue
                if token in text:
                    found.append(path)
    return found


def _assert_private_delivery(
    token: str,
    roots: list[Path],
    exclude: list[Path],
    planted_events: list[tuple[int, str]] | None = None,
) -> None:
    hits = _attacker_hits(token)
    assert hits == [], f"another local user read queued speech from: {hits}"
    if planted_events is not None:
        assert planted_events == [], (
            "the planted directory was written before the hook finished: "
            f"{planted_events}"
        )
    copies = _victim_copies(token, roots, exclude)
    assert copies, "the speaker did not keep a copy of the answer"


def _codex_event(token: str, *, compact: bool = False) -> str:
    payload = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": token}],
        },
    }
    if compact:
        return json.dumps(payload, separators=(",", ":"))
    return json.dumps(payload)


def _hook_env(home: Path, tmpdir: Path, project: str, thread: str, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "TMPDIR": str(tmpdir),
            "PROJECT_DIR": project,
            "AGENT_TYPE": "codex",
            "CODEX_THREAD_ID": thread,
        }
    )
    env.update(extra)
    return env


def _run_codex_hook(event: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return _run(["bash", str(CODEX_HOOK)], input=event, env=env)


def _stop_watcher() -> None:
    _run(["bash", str(REPO / "afterwords.sh"), "codex-hook", "stop"], cwd=REPO)
    if WATCH_PID.exists():
        try:
            pid = int(WATCH_PID.read_text().strip() or "0")
        except ValueError:
            pid = 0
        if pid > 0:
            subprocess.run(["kill", str(pid)], capture_output=True)
        WATCH_PID.unlink(missing_ok=True)


def _compile_swap_preload(directory: Path) -> Path:
    source = directory / "swap_preload.c"
    library = directory / "swap_preload.so"
    source.write_text(_SWAP_PRELOAD)
    build = _run(["gcc", "-shared", "-fPIC", "-o", str(library), str(source), "-ldl"])
    assert build.returncode == 0, build.stderr
    return library


def test_foreign_queue_directory_does_not_receive_the_codex_answer(
    tmp_path: Path, attacker_uid: int
) -> None:
    """A directory created by another user must not receive the answer."""
    del attacker_uid
    token = f"PlaceholderFinalAnswer-{uuid.uuid4().hex}"
    thread = str(uuid.uuid4())
    queue = Path(f"/tmp/codex-tts-queue-{thread}")
    home = _private_dir(tmp_path / "home")
    tmpdir = _private_dir(tmp_path / "tmpdir")
    project = f"/codex-project/{token}"
    before = _attacker_mkdir(queue)
    try:
        with queue_lock(), DirectoryWriteWatch(queue) as watch:
            _run_codex_hook(
                _codex_event(token),
                _hook_env(home, tmpdir, project, thread),
            )
            # The hook starts the worker. Give that process time to refuse
            # the directory; it must not be what removes the answer.
            subprocess.run(["sleep", "0.4"], check=True)
        _same_directory(queue, before)
        _assert_private_delivery(
            token,
            [home, tmpdir, queue],
            [],
            watch.events,
        )
    finally:
        subprocess.run(["sudo", "-n", "rm", "-rf", str(queue)], check=False)


def test_symlink_queue_does_not_receive_the_codex_answer(
    tmp_path: Path, attacker_uid: int
) -> None:
    """A symlink planted at the queue path must not receive the answer."""
    del attacker_uid
    token = f"PlaceholderFinalAnswer-{uuid.uuid4().hex}"
    thread = str(uuid.uuid4())
    queue = Path(f"/tmp/codex-tts-queue-{thread}")
    target = Path(f"/tmp/codex-tts-target-{thread}")
    home = _private_dir(tmp_path / "home")
    tmpdir = _private_dir(tmp_path / "tmpdir")
    project = f"/codex-project/{token}"
    before = _attacker_mkdir(target)
    subprocess.run(["sudo", "-n", "rm", "-rf", str(queue)], check=False)
    subprocess.run(
        ["sudo", "-n", "-u", ATTACKER, "ln", "-s", str(target), str(queue)],
        check=True,
    )
    try:
        with queue_lock(), DirectoryWriteWatch(target) as watch:
            _run_codex_hook(
                _codex_event(token),
                _hook_env(home, tmpdir, project, thread),
            )
            subprocess.run(["sleep", "0.4"], check=True)
        assert queue.is_symlink()
        _same_directory(target, before)
        _assert_private_delivery(
            token,
            [home, tmpdir, target],
            [],
            watch.events,
        )
    finally:
        subprocess.run(["sudo", "-n", "rm", "-rf", str(queue), str(target)], check=False)


def test_path_recheck_does_not_deliver_the_answer_to_another_directory(
    tmp_path: Path, attacker_uid: int
) -> None:
    """Looking up the queue path again between a check and the write still leaks.

    The queue starts as a private directory. The preload, loaded only into this
    hook, renames a world-readable directory onto that path when ``open`` or
    ``rename`` is aimed at a child of the path. A write that uses a directory
    fd opened before that rename stays on the private inode.
    """
    del attacker_uid
    token = f"PlaceholderFinalAnswer-{uuid.uuid4().hex}"
    thread = str(uuid.uuid4())
    queue = Path(f"/tmp/codex-tts-queue-{thread}")
    exposed = Path(f"/tmp/codex-tts-exposed-{thread}")
    saved = Path(f"/tmp/codex-tts-saved-{thread}")
    home = _private_dir(tmp_path / "home")
    tmpdir = _private_dir(tmp_path / "tmpdir")
    project = f"/codex-project/{token}"
    for path in (queue, exposed, saved):
        subprocess.run(["rm", "-rf", str(path)], check=False)
    queue.mkdir(mode=0o700)
    exposed.mkdir(mode=0o777)
    library = _compile_swap_preload(tmp_path)
    pidfile = Path(f"/tmp/codex-tts-worker-{thread}.pid")
    pidfile.write_text(str(os.getpid()))
    env = _hook_env(
        home,
        tmpdir,
        project,
        thread,
        LD_PRELOAD=str(library),
        AW_QUEUE=str(queue),
        AW_EXPOSED=str(exposed),
        AW_SAVED=str(saved),
    )
    try:
        with queue_lock(), DirectoryWriteWatch(exposed) as watch:
            completed = _run_codex_hook(_codex_event(token), env)
        assert completed.returncode == 0, completed.stderr
        _assert_private_delivery(
            token,
            [home, tmpdir, queue, saved, exposed],
            [],
            watch.events,
        )
    finally:
        pidfile.unlink(missing_ok=True)
        subprocess.run(["rm", "-rf", str(queue), str(exposed), str(saved)], check=False)


def test_watcher_log_does_not_reveal_the_thread_id(
    tmp_path: Path, attacker_uid: int
) -> None:
    """The watcher log must not let another user learn the thread id."""
    del attacker_uid
    thread = str(uuid.uuid4())
    home = _private_dir(tmp_path / "home")
    sessions = home / ".codex" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "rollout.jsonl").write_text(
        '{"type":"session_meta","payload":{"id":"%s"}}\n' % thread
    )
    env = os.environ.copy()
    env.update({"HOME": str(home), "CODEX_THREAD_ID": thread, "AFTERWORDS_REPO_DIR": str(REPO)})
    try:
        with queue_lock():
            _stop_watcher()
            WATCH_LOG.unlink(missing_ok=True)
            started = _run(
                ["bash", str(REPO / "afterwords.sh"), "codex-hook", "start"],
                cwd=REPO,
                env=env,
            )
            assert started.returncode == 0, started.stdout + started.stderr
            assert WATCH_PID.exists()
            pid = int(WATCH_PID.read_text().strip())
            subprocess.run(["kill", "-0", str(pid)], check=True)
            hits = _attacker_hits(thread)
            assert hits == [], f"another local user read the thread id from: {hits}"
    finally:
        _stop_watcher()


def test_watcher_answer_is_not_readable_after_the_log_is_published(
    tmp_path: Path, attacker_uid: int
) -> None:
    """Full path: read the watcher log, plant the queue, read the next answer."""
    del attacker_uid
    token = f"PlaceholderFinalAnswer-{uuid.uuid4().hex}"
    thread = str(uuid.uuid4())
    home = _private_dir(tmp_path / "home")
    sessions = home / ".codex" / "sessions"
    sessions.mkdir(parents=True)
    session = sessions / "rollout.jsonl"
    session.write_text('{"type":"session_meta","payload":{"id":"%s"}}\n' % thread)
    env = os.environ.copy()
    env.update({"HOME": str(home), "CODEX_THREAD_ID": thread, "AFTERWORDS_REPO_DIR": str(REPO)})
    queue = Path(f"/tmp/codex-tts-queue-{thread}")
    try:
        with queue_lock():
            _stop_watcher()
            WATCH_LOG.unlink(missing_ok=True)
            started = _run(
                ["bash", str(REPO / "afterwords.sh"), "codex-hook", "start"],
                cwd=REPO,
                env=env,
            )
            assert started.returncode == 0, started.stdout + started.stderr
            learned = _attacker_hits(thread)
            if learned:
                _attacker_mkdir(queue)
            with session.open("a") as handle:
                handle.write(_codex_event(token, compact=True) + "\n")
            for _ in range(16):
                if _attacker_hits(token) or _victim_copies(token, [home, queue], [sessions]):
                    break
                subprocess.run(["sleep", "0.5"], check=True)
            hits = _attacker_hits(token)
            copies = _victim_copies(token, [home, queue], [sessions])
        assert hits == [], f"another local user read the queued answer from: {hits}"
        assert copies, "the watcher did not queue the answer for the speaker"
    finally:
        _stop_watcher()
        subprocess.run(["sudo", "-n", "rm", "-rf", str(queue)], check=False)


def _shared_queue_case(
    tmp_path: Path,
    script: Path,
    stdin_for,
    cwd: Path | None = None,
    prepare=None,
) -> None:
    token = f"PlaceholderFinalAnswer-{uuid.uuid4().hex}"
    # The three adapters hardcode this directory.
    queue = Path("/tmp/claude-tts-queue")
    home = _private_dir(tmp_path / "home")
    tmpdir = _private_dir(tmp_path / "tmpdir")
    hooks = home / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    strip = hooks / "strip-markdown.py"
    if not strip.exists():
        strip.symlink_to(REPO / "strip_markdown.py")
    if prepare is not None:
        prepare(token, home)
    before = _attacker_mkdir(queue)
    pidfile = Path("/tmp/claude-tts-worker.pid")
    previous = pidfile.read_text() if pidfile.exists() else None
    pidfile.write_text(str(os.getpid()))
    env = os.environ.copy()
    env.update({"HOME": str(home), "TMPDIR": str(tmpdir)})
    try:
        with queue_lock(), DirectoryWriteWatch(queue) as watch:
            completed = _run(
                ["bash", str(script)],
                input=stdin_for(token),
                env=env,
                cwd=str(cwd or tmp_path),
            )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        _same_directory(queue, before)
        _assert_private_delivery(
            token,
            [home, tmpdir, queue],
            [home / "transcript.jsonl"],
            watch.events,
        )
    finally:
        if previous is None:
            pidfile.unlink(missing_ok=True)
        else:
            pidfile.write_text(previous)
        subprocess.run(["sudo", "-n", "rm", "-rf", str(queue)], check=False)


def test_cursor_hook_does_not_write_into_a_foreign_queue(
    tmp_path: Path, attacker_uid: int
) -> None:
    del attacker_uid
    _shared_queue_case(
        tmp_path,
        HOOKS / "cursor-tts-hook.sh",
        lambda token: json.dumps(
            {
                "text": token,
                "workspace_roots": ["/codex-project/cursor-placeholder"],
            }
        ),
    )


def test_gemini_hook_does_not_write_into_a_foreign_queue(
    tmp_path: Path, attacker_uid: int
) -> None:
    del attacker_uid
    project = _private_dir(tmp_path / "gemini-project")
    _shared_queue_case(
        tmp_path,
        HOOKS / "gemini-tts-hook.sh",
        lambda token: json.dumps({"prompt_response": token}),
        cwd=project,
    )


def test_agy_hook_does_not_write_into_a_foreign_queue(
    tmp_path: Path, attacker_uid: int, agy_helpers: None
) -> None:
    del attacker_uid, agy_helpers

    def prepare(token: str, home: Path) -> None:
        transcript = home / "transcript.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "content": token,
                }
            )
            + "\n"
        )
        prepare.transcript = transcript  # type: ignore[attr-defined]

    def stdin_for(token: str) -> str:
        return json.dumps(
            {
                "transcriptPath": str(prepare.transcript),  # type: ignore[attr-defined]
                "workspacePaths": ["/codex-project/agy-placeholder"],
            }
        )

    _shared_queue_case(
        tmp_path,
        HOOKS / "agy-tts-hook.sh",
        stdin_for,
        prepare=prepare,
    )
