"""Predictable /tmp WAV paths must not follow a planted symlink.

These tests are expected to FAIL while the scripts still open
``/tmp/<name>-<pid>.wav`` with curl -o or ffmpeg -y. They pass once each
write goes to a file created with mktemp (or an equivalent O_EXCL create)
so a pre-existing symlink cannot redirect it.

Two timings are covered for every truncating open:

* The symlink is planted before the process starts (the reported attack).
  A different local user owns it, so sticky-directory ``rm -f`` cannot
  unlink it between a check and the open.
* The symlink is planted at ``execve`` of the program that receives the
  predictable path, after any earlier check and before that program opens it.

Linux ``fs.protected_symlinks=1`` may turn the cross-user follow into
``EACCES`` instead of a truncation. That is still a failure: the script
must finish the write safely. A name created with ``O_EXCL`` is not one
an unprivileged neighbor can squat.

Not covered here: ``mkdir`` of ``/tmp/claude-tts-chunks-$$`` (the worker
text embedded in setup.sh) and ``/tmp/codex-tts-chunks-$SESSION-$$``.
Those are directory creates, not the truncating WAV open this finding
describes.

Run (Linux, with curl, ffmpeg, ffprobe, and sudo)::

    python3 -m pytest tests/test_predictable_tmp_wav_symlink.py -q
"""

from __future__ import annotations

import ctypes
import os
import re
import signal
import subprocess
import sys
import textwrap
import threading
from ctypes import c_int, c_long, c_uint64, c_void_p
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ATTACKER = "awattacker"
ORIGINAL = b"VICTIM-FILE-ORIGINAL-CONTENT\n"

PREDICTABLE_RE = re.compile(
    r"^/tmp/(?:"
    r"afterwords-output|"
    r"hermes-afterwords|"
    r"hermes-afterwords-trimmed|"
    r"clone-segment|"
    r"clone-test|"
    r"voice-setup-segment|"
    r"afterwords-demo"
    r")-[0-9]+\.wav$"
)

PTRACE_TRACEME = 0
PTRACE_PEEKDATA = 2
PTRACE_SYSCALL = 24
PTRACE_SETOPTIONS = 0x4200
PTRACE_GETREGS = 12
PTRACE_O_TRACESYSGOOD = 0x00000001
PTRACE_O_TRACEFORK = 0x00000002
PTRACE_O_TRACEVFORK = 0x00000004
PTRACE_O_TRACECLONE = 0x00000008
PTRACE_O_TRACEEXEC = 0x00000010
SYS_EXECVE = 59
SYS_EXECVEAT = 322

_TRACKED: list[str] = []


class _Regs(ctypes.Structure):
    _fields_ = [
        (name, c_uint64)
        for name in (
            "r15", "r14", "r13", "r12", "rbp", "rbx", "r11", "r10",
            "r9", "r8", "rax", "rcx", "rdx", "rsi", "rdi", "orig_rax",
            "rip", "cs", "eflags", "rsp", "ss", "fs_base", "gs_base",
            "ds", "es", "fs", "gs",
        )
    ]


_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.ptrace.argtypes = [c_int, c_int, c_void_p, c_void_p]
_libc.ptrace.restype = c_long


def _ptrace(req: int, pid: int, addr: int = 0, data=None) -> int:
    if data is None:
        data = c_void_p(0)
    result = _libc.ptrace(req, pid, c_void_p(addr), data)
    if result == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return result


def _peek(pid: int, addr: int) -> int:
    ctypes.set_errno(0)
    word = _libc.ptrace(PTRACE_PEEKDATA, pid, c_void_p(addr), None)
    if word == -1 and ctypes.get_errno():
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return word & (2**64 - 1)


def _read_cstr(pid: int, addr: int, limit: int = 4096) -> str:
    buf = b""
    while len(buf) < limit:
        chunk = _peek(pid, addr).to_bytes(8, "little")
        if b"\x00" in chunk:
            buf += chunk.split(b"\x00", 1)[0]
            break
        buf += chunk
        addr += 8
    return buf.decode("utf-8", "surrogateescape")


def _read_argv(pid: int, addr: int) -> list[str]:
    args: list[str] = []
    for _ in range(80):
        ptr = _peek(pid, addr)
        if ptr == 0:
            break
        args.append(_read_cstr(pid, ptr))
        addr += 8
    return args


def _wav_bytes() -> bytes:
    import io
    import math
    import struct
    import wave

    sr = 8000
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sr)))
        for i in range(sr)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(frames)
    return buf.getvalue()


def _write_script(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _plant(link: str, target: Path, *, replace: bool = True) -> None:
    if not replace and os.path.lexists(link):
        return
    subprocess.run(["sudo", "-n", "rm", "-f", link], check=False)
    subprocess.check_call(
        ["sudo", "-n", "-u", ATTACKER, "ln", "-s", str(target), link]
    )
    _TRACKED.append(link)


def _cleanup_tracked() -> None:
    while _TRACKED:
        link = _TRACKED.pop()
        subprocess.run(["sudo", "-n", "rm", "-f", link], check=False)


@pytest.fixture(scope="module")
def attacker_account() -> str:
    probe = subprocess.run(["sudo", "-n", "id", ATTACKER], capture_output=True, text=True)
    if probe.returncode != 0:
        created = subprocess.run(
            ["sudo", "-n", "useradd", "-M", "-s", "/usr/sbin/nologin", ATTACKER],
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            pytest.fail(
                "need a second local account to plant the symlink "
                f"({created.stderr.strip() or created.stdout.strip()})"
            )
    return ATTACKER


@pytest.fixture(scope="module")
def fixture_wav(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("wav") / "fixture.wav"
    path.write_bytes(_wav_bytes())
    return path


@pytest.fixture(scope="module")
def tts_server(fixture_wav: Path):
    server_py = Path("/tmp/aw-symlink-test-server.py")
    server_py.write_text(
        textwrap.dedent(
            f"""\
            import json
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
            BODY = open({str(fixture_wav)!r}, "rb").read()

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path.startswith("/health"):
                        data = json.dumps({{
                            "ready": True,
                            "voices": ["galadriel"],
                            "default_voice": "galadriel",
                        }}).encode()
                        ctype = "application/json"
                    elif self.path.startswith("/synthesize"):
                        data = BODY
                        ctype = "audio/wav"
                    else:
                        self.send_response(404)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                def log_message(self, fmt, *args):
                    return

            ThreadingHTTPServer(("127.0.0.1", 7860), Handler).serve_forever()
            """
        )
    )
    proc = subprocess.Popen([sys.executable, str(server_py)])
    import socket
    import time

    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", 7860), 0.2):
                break
        except OSError:
            if proc.poll() is not None:
                pytest.fail("local synthesize stub exited before binding")
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("local synthesize stub did not bind 127.0.0.1:7860")
    yield
    proc.kill()
    proc.wait(timeout=5)
    server_py.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def shim_dir(tmp_path_factory, fixture_wav: Path) -> Path:
    directory = tmp_path_factory.mktemp("shims")
    _write_script(directory / "afplay", "#!/bin/bash\nexit 0\n")
    _write_script(
        directory / "stat",
        textwrap.dedent(
            """\
            #!/bin/bash
            # The scripts use macOS `stat -f%z` to decide the WAV is real.
            if [ "$1" = "-f%z" ]; then
              exec /usr/bin/stat -c %s "$2"
            fi
            exec /usr/bin/stat "$@"
            """
        ),
    )
    _write_script(
        directory / "sysctl",
        textwrap.dedent(
            """\
            #!/bin/bash
            # setup.sh exits before the WAV extract unless the machine
            # looks like Apple Silicon. This only reaches that line.
            if [ "$1" = "-n" ] && [ "$2" = "machdep.cpu.brand_string" ]; then
              echo "Apple M2"
              exit 0
            fi
            if [ "$1" = "-n" ] && [ "$2" = "hw.memsize" ]; then
              echo 34359738368
              exit 0
            fi
            exec /usr/sbin/sysctl "$@"
            """
        ),
    )
    _write_script(
        directory / "uname",
        textwrap.dedent(
            """\
            #!/bin/bash
            if [ "$1" = "-m" ]; then
              echo arm64
              exit 0
            fi
            exec /usr/bin/uname "$@"
            """
        ),
    )
    _write_script(
        directory / "yt-dlp",
        textwrap.dedent(
            f"""\
            #!/bin/bash
            out=""
            while [ $# -gt 0 ]; do
              case "$1" in
                -o) out="$2"; shift 2 ;;
                *) shift ;;
              esac
            done
            dest="${{out//%(ext)s/wav}}"
            cp {str(fixture_wav)} "$dest"
            """
        ),
    )
    _write_script(directory / "pip", "#!/bin/bash\nexit 0\n")
    _write_script(directory / "pip3", "#!/bin/bash\nexit 0\n")
    return directory


@pytest.fixture(scope="module")
def shim_venv(shim_dir: Path) -> Path:
    """Let clone-voice.sh and setup.sh reach the WAV open without MLX."""
    venv = REPO / ".venv"
    created = not venv.exists()
    bindir = venv / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (bindir / "activate").write_text(
        textwrap.dedent(
            f"""\
            VIRTUAL_ENV={venv}
            export VIRTUAL_ENV
            PATH="$VIRTUAL_ENV/bin:$PATH"
            export PATH
            """
        )
    )
    for name in ("afplay", "stat", "pip", "pip3"):
        dest = bindir / name
        if not dest.exists():
            dest.symlink_to(shim_dir / name)
    python = bindir / "python3"
    python.write_text(
        textwrap.dedent(
            r"""\
            #!/bin/bash
            if [ "${1:-}" = "-c" ]; then
              case "$2" in
                *faster_whisper*) exit 0 ;;
                pass) exit 0 ;;
              esac
              exec /usr/bin/python3 "$@"
            fi
            if [ "${1:-}" = "-m" ] && [ "${2:-}" = "backends" ]; then
              case "${3:-}" in
                list) echo qwen3-0.6b; exit 0 ;;
                max-sample-rate) echo 24000; exit 0 ;;
                slug) echo slug; exit 0 ;;
              esac
            fi
            script=$(cat)
            if printf '%s' "$script" | grep -q noisereduce; then
              if [ "${AW_DENOISE_FAIL:-}" = 1 ]; then
                exit 1
              fi
              exit 0
            fi
            if printf '%s' "$script" | grep -q faster_whisper; then
              printf '%s\n' "  [0.0s] hello" "---FULLTEXT---" "hello from the clip"
              exit 0
            fi
            printf '%s' "$script" | /usr/bin/python3 "$@"
            """
        )
    )
    python.chmod(0o755)
    yield venv
    if created:
        import shutil
        shutil.rmtree(venv, ignore_errors=True)


@pytest.fixture
def secret_file(tmp_path: Path) -> Path:
    path = tmp_path / "victim-file"
    path.write_bytes(ORIGINAL)
    path.chmod(0o644)
    return path


@pytest.fixture
def second_secret(tmp_path: Path) -> Path:
    path = tmp_path / "victim-file-trimmed"
    path.write_bytes(ORIGINAL)
    path.chmod(0o644)
    return path


def _env(shim_dir: Path, home: Path, extra: dict | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
    env["HOME"] = str(home)
    env.pop("LD_PRELOAD", None)
    if extra:
        env.update(extra)
    return env


def _run_traced(
    argv: list[str],
    env: dict[str, str],
    *,
    cwd: Path | None = None,
    stdin: bytes = b"",
    preplant: list[tuple[str, Path]] | None = None,
    plant_at_exec: bool = False,
    exec_target: Path | None = None,
    timeout: float = 60,
) -> subprocess.CompletedProcess[str]:
    """Trace argv. Squat predictable WAV names either before exec or at execve."""
    read_out, write_out = os.pipe()
    read_err, write_err = os.pipe()
    read_in, write_in = os.pipe()
    leader_status = {"code": 1}

    pid = os.fork()
    if pid == 0:
        try:
            os.dup2(read_in, 0)
            os.dup2(write_out, 1)
            os.dup2(write_err, 2)
            for fd in (read_out, write_out, read_err, write_err, read_in, write_in):
                os.close(fd)
            os.setpgrp()
            if cwd is not None:
                os.chdir(cwd)
            _ptrace(PTRACE_TRACEME, 0)
            os.kill(os.getpid(), signal.SIGSTOP)
            os.execvpe(argv[0], argv, env)
        finally:
            os._exit(127)

    os.close(read_in)
    os.close(write_out)
    os.close(write_err)
    os.write(write_in, stdin)
    os.close(write_in)

    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []

    def _drain(fd: int, sink: list[bytes]) -> None:
        with os.fdopen(fd, "rb") as handle:
            sink.append(handle.read())

    out_thread = threading.Thread(target=_drain, args=(read_out, out_chunks))
    err_thread = threading.Thread(target=_drain, args=(read_err, err_chunks))
    out_thread.start()
    err_thread.start()

    timed_out = {"flag": False}

    def _kill() -> None:
        timed_out["flag"] = True
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass

    timer = threading.Timer(timeout, _kill)
    timer.daemon = True
    try:
        os.waitpid(pid, 0)
        _ptrace(
            PTRACE_SETOPTIONS,
            pid,
            0,
            c_void_p(
                PTRACE_O_TRACESYSGOOD
                | PTRACE_O_TRACEFORK
                | PTRACE_O_TRACEVFORK
                | PTRACE_O_TRACECLONE
                | PTRACE_O_TRACEEXEC
            ),
        )
        for name, target in preplant or []:
            _plant(f"/tmp/{name}-{pid}.wav", target)
        entering: dict[int, bool] = {pid: True}
        live = {pid}
        _ptrace(PTRACE_SYSCALL, pid)
        timer.start()
        while live and not timed_out["flag"]:
            try:
                wpid, status = os.waitpid(-1, 0)
            except ChildProcessError:
                break
            if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                if wpid == pid:
                    if os.WIFEXITED(status):
                        leader_status["code"] = os.WEXITSTATUS(status)
                    else:
                        leader_status["code"] = 128 + os.WTERMSIG(status)
                live.discard(wpid)
                continue
            if not os.WIFSTOPPED(status):
                continue
            sig = os.WSTOPSIG(status)
            event = (status >> 16) & 0xFFFF
            if event:
                live.add(wpid)
                entering.setdefault(wpid, True)
                _ptrace(PTRACE_SYSCALL, wpid)
                continue
            if sig == (signal.SIGTRAP | 0x80):
                is_entry = entering.get(wpid, True)
                entering[wpid] = not is_entry
                if is_entry and plant_at_exec and exec_target is not None:
                    regs = _Regs()
                    _ptrace(PTRACE_GETREGS, wpid, 0, ctypes.byref(regs))
                    if regs.orig_rax in (SYS_EXECVE, SYS_EXECVEAT):
                        argv_addr = regs.rsi if regs.orig_rax == SYS_EXECVE else regs.rdx
                        try:
                            args = _read_argv(wpid, argv_addr)
                        except OSError:
                            args = []
                        for arg in args:
                            if PREDICTABLE_RE.match(arg):
                                # Leave a symlink planted before exec alone.
                                # Replacing it would hide the size gate below.
                                _plant(arg, exec_target, replace=False)
                _ptrace(PTRACE_SYSCALL, wpid)
                continue
            injected = 0 if sig in (signal.SIGSTOP, signal.SIGTSTP, signal.SIGTRAP) else sig
            _ptrace(PTRACE_SYSCALL, wpid, 0, c_void_p(injected))
    finally:
        timer.cancel()
        if timed_out["flag"]:
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        out_thread.join(timeout=3)
        err_thread.join(timeout=3)

    stdout = b"".join(out_chunks).decode("utf-8", "replace")
    stderr = b"".join(err_chunks).decode("utf-8", "replace")
    if timed_out["flag"]:
        pytest.fail(f"timed out running {argv!r}\n{stdout[-2000:]}\n{stderr[-2000:]}")
    return subprocess.CompletedProcess(argv, leader_status["code"], stdout, stderr)


def _assert_safe(
    secrets: list[Path],
    proc: subprocess.CompletedProcess[str],
    *,
    must_contain: str | None = None,
    must_not_contain: tuple[str, ...] = (),
    expect_exit: int | None = 0,
) -> None:
    rendered = (proc.stdout + "\n" + proc.stderr)[-4000:]
    for secret in secrets:
        data = secret.read_bytes()
        assert data == ORIGINAL, (
            f"{secret} was overwritten ({len(data)} bytes, starts {data[:24]!r}). "
            "The predictable /tmp name followed a symlink.\n"
            f"exit={proc.returncode}\n{rendered}"
        )
    if expect_exit is not None:
        assert proc.returncode == expect_exit, (
            f"exit={proc.returncode}, expected {expect_exit}. "
            "The write did not finish safely.\n"
            f"{rendered}"
        )
    if must_contain is not None:
        combined = proc.stdout + proc.stderr
        assert must_contain in combined, (
            f"expected {must_contain!r} so the WAV open actually completed\n"
            f"exit={proc.returncode}\n{rendered}"
        )
    combined = proc.stdout + proc.stderr
    for banned in must_not_contain:
        assert banned not in combined, (
            f"write did not succeed ({banned!r} present)\n"
            f"exit={proc.returncode}\n{rendered}"
        )


def _finish(secrets: list[Path], proc, **kwargs) -> None:
    try:
        _assert_safe(secrets, proc, **kwargs)
    finally:
        _cleanup_tracked()


def test_speak_preplanted_symlink_does_not_truncate(
    attacker_account, tts_server, shim_dir, secret_file, tmp_path
):
    proc = _run_traced(
        ["bash", str(REPO / "skill/scripts/speak.sh"), "hello from the skill", "galadriel"],
        _env(shim_dir, tmp_path),
        preplant=[("afterwords-output", secret_file)],
    )
    _finish([secret_file], proc, expect_exit=0)


def test_speak_symlink_planted_at_exec_does_not_truncate(
    attacker_account, tts_server, shim_dir, secret_file, tmp_path
):
    proc = _run_traced(
        ["bash", str(REPO / "skill/scripts/speak.sh"), "hello from the skill", "galadriel"],
        _env(shim_dir, tmp_path),
        plant_at_exec=True,
        exec_target=secret_file,
    )
    _finish([secret_file], proc, expect_exit=0)


def test_hermes_preplanted_symlinks_do_not_truncate(
    attacker_account, tts_server, shim_dir, secret_file, second_secret, tmp_path
):
    proc = _run_traced(
        ["bash", str(REPO / "scripts/hermes-tts.sh"), "hello from hermes", "galadriel"],
        _env(shim_dir, tmp_path),
        preplant=[
            ("hermes-afterwords", secret_file),
            ("hermes-afterwords-trimmed", second_secret),
        ],
    )
    # macOS `stat -f%z` is lstat (file_cmds stat.c defaults usestat=0), so a
    # short symlink does not pass the >1000 size gate and ffmpeg never opens
    # the trimmed path. The curl -o of the untrimmed path still truncates.
    _finish([secret_file, second_secret], proc, expect_exit=0)


def _long_victim(root: Path) -> Path:
    """A writable file whose path is longer than 1000 bytes.

    macOS ``stat -f%z`` uses lstat, so the size it reports for a symlink is
    the length of the link text. hermes-tts.sh only opens the trimmed WAV
    when that size is greater than 1000. A short symlink never reaches
    ``ffmpeg -y``; a link whose text is this path does.
    """
    current = root
    # 520 single-letter components keep each name under NAME_MAX and the
    # full path above the script's 1000-byte size gate, under PATH_MAX.
    for _ in range(520):
        current = current / "d"
    current.mkdir(parents=True)
    victim = current / "v"
    victim.write_bytes(ORIGINAL)
    victim.chmod(0o644)
    assert len(str(victim)) > 1000
    return victim


def test_hermes_long_symlink_reaches_trimmed_output_without_truncating(
    attacker_account, tts_server, shim_dir, secret_file, tmp_path
):
    long_root = Path("/tmp") / f"awlong-{os.getpid()}"
    long_victim = _long_victim(long_root)
    try:
        proc = _run_traced(
            ["bash", str(REPO / "scripts/hermes-tts.sh"), "hello from hermes", "galadriel"],
            _env(shim_dir, tmp_path),
            preplant=[
                ("hermes-afterwords", long_victim),
                ("hermes-afterwords-trimmed", secret_file),
            ],
        )
        _finish([secret_file, long_victim], proc, expect_exit=0)
    finally:
        import shutil
        shutil.rmtree(long_root, ignore_errors=True)


def test_hermes_symlink_planted_at_exec_does_not_truncate(
    attacker_account, tts_server, shim_dir, secret_file, tmp_path
):
    proc = _run_traced(
        ["bash", str(REPO / "scripts/hermes-tts.sh"), "hello from hermes", "galadriel"],
        _env(shim_dir, tmp_path),
        plant_at_exec=True,
        exec_target=secret_file,
    )
    _finish([secret_file], proc, expect_exit=0)


def test_clone_voice_preplanted_symlinks_do_not_truncate(
    attacker_account, tts_server, shim_dir, shim_venv, fixture_wav, secret_file, tmp_path
):
    voice = "zz-symlink-probe"
    proc = _run_traced(
        ["bash", str(REPO / "clone-voice.sh"), str(fixture_wav), voice, "0", "--yes"],
        _env(shim_dir, tmp_path),
        cwd=REPO,
        preplant=[
            ("clone-segment", secret_file),
            ("clone-test", secret_file),
        ],
        timeout=90,
    )
    try:
        _assert_safe(
            [secret_file],
            proc,
            must_contain="Synthesis works!",
            expect_exit=None,
        )
    finally:
        _cleanup_tracked()
        for leftover in (
            REPO / "voices" / f"{voice}.json",
            REPO / "voices" / f"{voice}-ref.wav",
        ):
            leftover.unlink(missing_ok=True)


def test_clone_voice_symlink_planted_at_exec_does_not_truncate(
    attacker_account, tts_server, shim_dir, shim_venv, fixture_wav, secret_file, tmp_path
):
    voice = "zz-symlink-probe-exec"
    proc = _run_traced(
        ["bash", str(REPO / "clone-voice.sh"), str(fixture_wav), voice, "0", "--yes"],
        _env(shim_dir, tmp_path),
        cwd=REPO,
        plant_at_exec=True,
        exec_target=secret_file,
        timeout=90,
    )
    try:
        _assert_safe(
            [secret_file],
            proc,
            must_contain="Synthesis works!",
            expect_exit=None,
        )
    finally:
        _cleanup_tracked()
        for leftover in (
            REPO / "voices" / f"{voice}.json",
            REPO / "voices" / f"{voice}-ref.wav",
        ):
            leftover.unlink(missing_ok=True)


def test_setup_preplanted_symlink_does_not_truncate(
    attacker_account, shim_dir, shim_venv, secret_file, tmp_path
):
    voices = REPO / "voices"
    hidden = tmp_path / "hidden-refs"
    hidden.mkdir()
    moved: list[tuple[Path, Path]] = []
    for ref in voices.glob("*-ref.wav"):
        dest = hidden / ref.name
        ref.rename(dest)
        moved.append((dest, ref))
    try:
        proc = _run_traced(
            ["bash", str(REPO / "setup.sh"), "--server-only"],
            _env(shim_dir, tmp_path, {"AW_DENOISE_FAIL": "1", "AW_FIXTURE_WAV": ""}),
            cwd=REPO,
            stdin=b"https://example.invalid/watch\n\nzzsetup\n",
            preplant=[("voice-setup-segment", secret_file)],
            timeout=90,
        )
        _assert_safe(
            [secret_file],
            proc,
            must_contain="Denoising",
            expect_exit=None,
        )
    finally:
        _cleanup_tracked()
        for dest, orig in moved:
            if dest.exists() and not orig.exists():
                dest.rename(orig)
        for leftover in voices.glob("zzsetup*"):
            leftover.unlink()


def test_setup_symlink_planted_at_exec_does_not_truncate(
    attacker_account, shim_dir, shim_venv, secret_file, tmp_path
):
    voices = REPO / "voices"
    hidden = tmp_path / "hidden-refs"
    hidden.mkdir()
    moved: list[tuple[Path, Path]] = []
    for ref in voices.glob("*-ref.wav"):
        dest = hidden / ref.name
        ref.rename(dest)
        moved.append((dest, ref))
    try:
        proc = _run_traced(
            ["bash", str(REPO / "setup.sh"), "--server-only"],
            _env(shim_dir, tmp_path, {"AW_DENOISE_FAIL": "1"}),
            cwd=REPO,
            stdin=b"https://example.invalid/watch\n\nzzsetup\n",
            plant_at_exec=True,
            exec_target=secret_file,
            timeout=90,
        )
        _assert_safe(
            [secret_file],
            proc,
            must_contain="Denoising",
            expect_exit=None,
        )
    finally:
        _cleanup_tracked()
        for dest, orig in moved:
            if dest.exists() and not orig.exists():
                dest.rename(orig)
        for leftover in voices.glob("zzsetup*"):
            leftover.unlink()


def test_demo_preplanted_symlink_does_not_truncate(
    attacker_account, tts_server, shim_dir, secret_file, tmp_path
):
    proc = _run_traced(
        ["bash", str(REPO / "afterwords.sh"), "voices", "--demo"],
        _env(shim_dir, tmp_path),
        preplant=[("afterwords-demo", secret_file)],
    )
    _finish(
        [secret_file],
        proc,
        must_contain="Demo complete",
        must_not_contain=("Synthesis failed", "empty audio"),
        expect_exit=0,
    )


def test_demo_symlink_planted_at_exec_does_not_truncate(
    attacker_account, tts_server, shim_dir, secret_file, tmp_path
):
    proc = _run_traced(
        ["bash", str(REPO / "afterwords.sh"), "voices", "--demo"],
        _env(shim_dir, tmp_path),
        plant_at_exec=True,
        exec_target=secret_file,
    )
    _finish(
        [secret_file],
        proc,
        must_contain="Demo complete",
        must_not_contain=("Synthesis failed", "empty audio"),
        expect_exit=0,
    )
