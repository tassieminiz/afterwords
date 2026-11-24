#!/usr/bin/env python3
"""Private per-user TTS queue shared by the Afterwords agent hooks and workers.

    tts-queue.py prepare STATE_DIR [QUEUE]
    tts-queue.py enqueue STATE_DIR QUEUE PROJECT_DIR AGENT TEXT
    tts-queue.py check-cwd

STATE_DIR (and QUEUE, a single directory name inside it) are created mode 0700
and must be real directories owned by the current user; anything else is
refused. Each check and the write that follows go through the same directory
file descriptor, opened with O_NOFOLLOW, so a path swapped after the check
cannot redirect the item. Queue items are written mode 0600 to a temp name and
renamed into place, so a worker never sees a partial item.

Workers `cd` into the queue and then run check-cwd, which applies the same
rule to the already-bound working directory; after that they use only
relative names, so every claim and read stays on the directory that passed.
"""
from __future__ import annotations

import json
import os
import secrets
import stat
import sys
import time

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_ITEM_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
)


class Refused(Exception):
    pass


def _check_name(name: str) -> None:
    if not name or name in (".", "..") or "/" in name or "\0" in name:
        raise Refused(f"invalid queue name {name!r}")


def _require_private(fd: int, label: str) -> int:
    st = os.fstat(fd)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid():
        os.close(fd)
        raise Refused(f"{label} is not a directory we own")
    if stat.S_IMODE(st.st_mode) != 0o700:
        os.fchmod(fd, 0o700)
    return fd


def _open_private(name: str, dir_fd: int, label: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=dir_fd)
    except FileExistsError:
        pass
    except OSError as exc:
        raise Refused(f"cannot create {label}: {exc.strerror}") from None
    try:
        fd = os.open(name, _DIR_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        raise Refused(f"{label} is not a plain directory ({exc.strerror})") from None
    return _require_private(fd, label)


def open_state(state_dir: str) -> int:
    state_dir = os.path.abspath(state_dir)
    parent, name = os.path.split(state_dir)
    _check_name(name)
    os.makedirs(parent, exist_ok=True)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))
    try:
        return _open_private(name, parent_fd, state_dir)
    finally:
        os.close(parent_fd)


def open_queue(state_dir: str, queue: str) -> int:
    _check_name(queue)
    state_fd = open_state(state_dir)
    try:
        return _open_private(queue, state_fd, os.path.join(state_dir, queue))
    finally:
        os.close(state_fd)


def enqueue(state_dir: str, queue: str, project_dir: str, agent: str, text: str) -> str:
    payload = json.dumps({"project_dir": project_dir, "agent": agent, "text": text}) + "\n"
    queue_fd = open_queue(state_dir, queue)
    try:
        for _ in range(8):
            base = f"{int(time.time())}-{secrets.randbelow(1 << 31)}"
            tmp = f"{base}.json.tmp"
            try:
                item_fd = os.open(tmp, _ITEM_FLAGS, 0o600, dir_fd=queue_fd)
            except FileExistsError:
                continue
            break
        else:
            raise Refused("could not allocate a queue item name")
        try:
            with os.fdopen(item_fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.rename(tmp, f"{base}.json", src_dir_fd=queue_fd, dst_dir_fd=queue_fd)
        except BaseException:
            try:
                os.unlink(tmp, dir_fd=queue_fd)
            except OSError:
                pass
            raise
        return f"{base}.json"
    finally:
        os.close(queue_fd)


def main(argv: list[str]) -> int:
    try:
        if len(argv) in (2, 3) and argv[0] == "prepare":
            fd = open_queue(argv[1], argv[2]) if len(argv) == 3 else open_state(argv[1])
            os.close(fd)
            return 0
        if len(argv) == 6 and argv[0] == "enqueue":
            enqueue(*argv[1:])
            return 0
        if argv == ["check-cwd"]:
            os.close(_require_private(os.open(".", _DIR_FLAGS), "queue working directory"))
            return 0
    except Refused as exc:
        print(f"afterwords: {exc} — refusing to use it", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"afterwords: TTS queue error: {exc}", file=sys.stderr)
        return 1
    print("usage:\n" + "\n".join(__doc__.strip().splitlines()[2:5]), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
