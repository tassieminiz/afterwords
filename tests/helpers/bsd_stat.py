#!/usr/bin/env python3
"""BSD ``stat -f`` for the queue-disclosure tests.

FreeBSD 15.1 stat(1) obtains file information with lstat(2) unless ``-L`` is
passed, and its ``-f`` format uses ``%u`` for ``st_uid`` and ``%z`` for
``st_size``. The hooks call ``stat -f%u``. GNU stat rejects that flag, which
would make an ownership check fail closed before it opened the queue. This
shim implements the flag the hooks use and forwards every other invocation to
``/usr/bin/stat``.
"""

from __future__ import annotations

import os
import sys


def _format(fmt: str, path: str) -> str:
    st = os.lstat(path)
    fields = {"u": str(st.st_uid), "z": str(st.st_size)}
    out: list[str] = []
    i = 0
    while i < len(fmt):
        if fmt[i] == "%" and i + 1 < len(fmt):
            spec = fmt[i + 1]
            if spec not in fields:
                sys.stderr.write(f"bsd_stat: unsupported format %{spec}\n")
                raise SystemExit(1)
            out.append(fields[spec])
            i += 2
            continue
        out.append(fmt[i])
        i += 1
    return "".join(out)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0].startswith("-f"):
        if argv[0] == "-f":
            if len(argv) < 3:
                sys.stderr.write("bsd_stat: missing format\n")
                return 1
            fmt = argv[1]
            paths = argv[2:]
        else:
            fmt = argv[0][2:]
            paths = argv[1:]
        if not paths:
            sys.stderr.write("bsd_stat: missing path\n")
            return 1
        for path in paths:
            try:
                print(_format(fmt, path))
            except OSError as exc:
                sys.stderr.write(f"bsd_stat: {path}: {exc.strerror}\n")
                return 1
        return 0
    os.execv("/usr/bin/stat", ["/usr/bin/stat", *argv])
    return 127


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
