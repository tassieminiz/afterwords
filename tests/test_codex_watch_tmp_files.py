"""Security regressions for the Codex watcher control files.

The tests call the real ``afterwords.sh codex-hook`` commands and
``.claude/hooks/codex-tts-watch.sh``. They fail while those commands trust the
fixed paths ``/tmp/codex-tts-watch.log`` and ``/tmp/codex-tts-watch.pid``:
start empties and appends the log through a symlink, stop signals whatever
pid is in the pid file, and a live unrelated pid makes start report the
watcher as already running.

A check that the path is a regular file, followed by opening that path again,
still fails the swap tests: the path is safe at the check and a symlink at
the open.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
AFTERWORDS = REPO / "afterwords.sh"
WATCHER = REPO / ".claude/hooks" / "codex-tts-watch.sh"
PID_PATH = Path("/tmp/codex-tts-watch.pid")
LOG_PATH = Path("/tmp/codex-tts-watch.log")
SECRET = "SECRET-CONTENT-KEEP\n"
ATTACKER = "awattacker"
IGNORER = (
    "import signal, sys, time, pathlib; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "pathlib.Path(sys.argv[1]).write_text('ready'); "
    "time.sleep(90)"
)


def _run(argv: list[str], env: dict[str, str], timeout: float = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )


def _base_env(home: Path | None = None, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("BASH_ENV", None)
    env.pop("CODEX_THREAD_ID", None)
    env.pop("CODEX_WATCH_LOG", None)
    env.pop("AW_LINK_TARGET", None)
    if home is not None:
        env["HOME"] = str(home)
    env.update(extra)
    return env


def _remove_shared_paths() -> None:
    for path in (PID_PATH, LOG_PATH):
        if not os.path.lexists(path):
            continue
        try:
            path.unlink()
        except OSError:
            subprocess.run(["sudo", "-n", "rm", "-f", str(path)], check=False)


def _watcher_pids() -> list[int]:
    listed = subprocess.run(
        ["ps", "-axww", "-o", "pid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    found: list[int] = []
    for line in listed.stdout.splitlines():
        line = line.strip()
        if not line or "codex-tts-watch.sh" not in line:
            continue
        pid_text, _, command = line.partition(" ")
        if "bash" not in command and "codex-tts-watch.sh" not in command.split():
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != os.getpid():
            found.append(pid)
    return found


def _kill_watchers() -> None:
    for pid in _watcher_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _spawn_ignorer() -> subprocess.Popen[bytes]:
    ready_dir = Path(tempfile.mkdtemp(prefix="aw-ready-"))
    ready = ready_dir / "ready"
    proc = subprocess.Popen(
        [sys.executable, "-c", IGNORER, str(ready)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(50):
        if ready.exists() or proc.poll() is not None:
            break
        time.sleep(0.02)
    if not ready.exists():
        proc.kill()
        raise RuntimeError("victim process did not install its SIGTERM handler")
    return proc


def _assert_alive(proc: subprocess.Popen[bytes]) -> None:
    assert proc.poll() is None, "unrelated process was signalled"


def _write_swap_trap(path: Path, *, kind: str) -> None:
    """Swap the shared path to a symlink just before a command opens it.

    Stat and test commands are left alone, so a check-then-open fix still
    opens the symlink.
    """
    if kind == "log":
        dest = str(LOG_PATH)
        names = "*CODEX_WATCH_LOG*|*codex-tts-watch.log*"
        opener = '*">"*'
    elif kind == "pid":
        dest = str(PID_PATH)
        names = "*CODEX_WATCH_PID*|*codex-tts-watch.pid*"
        opener = '*"cat "*|*">"*|*"<"*'
    else:
        raise ValueError(kind)
    # functrace makes the DEBUG trap run inside shell functions. Without it,
    # bash only traces the top-level script, and the open happens later in
    # cmd_codex_hook.
    path.write_text(
        f"""set -o functrace
_aw_swap() {{
  local cmd="$BASH_COMMAND"
  case "$cmd" in
    *"[["*|*"[ "*|*"test "*|*"readlink "*|*"stat "*) return 0 ;;
  esac
  case "$cmd" in
    {names})
      case "$cmd" in
        {opener})
          trap - DEBUG
          rm -f {dest}
          ln -s "$AW_LINK_TARGET" {dest}
          ;;
      esac
      ;;
  esac
  return 0
}}
trap _aw_swap DEBUG
"""
    )


def _plant_session(home: Path, thread: str) -> None:
    session_dir = home / ".codex" / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "rollout.jsonl").write_text('{"id":"%s","type":"session_meta"}\n' % thread)


def _sudo_attacker_available() -> bool:
    if shutil.which("sudo") is None or shutil.which("useradd") is None:
        return False
    if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
        return False
    if subprocess.run(["id", ATTACKER], capture_output=True).returncode != 0:
        created = subprocess.run(
            ["sudo", "-n", "useradd", "-M", "-s", "/usr/sbin/nologin", ATTACKER],
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            return False
    return True


def _plant_foreign_pid(pid: int) -> None:
    script = f"printf '%s\\n' {int(pid)} > {PID_PATH} && chmod 644 {PID_PATH}"
    planted = subprocess.run(
        ["sudo", "-n", "-u", ATTACKER, "bash", "-c", script],
        capture_output=True,
        text=True,
    )
    assert planted.returncode == 0, planted.stderr
    assert os.stat(PID_PATH).st_uid != os.getuid()


@pytest.fixture
def shared_paths():
    _kill_watchers()
    _remove_shared_paths()
    try:
        yield
    finally:
        _kill_watchers()
        _remove_shared_paths()


def test_stop_does_not_kill_unrelated_process_named_by_pidfile(shared_paths):
    victim = _spawn_ignorer()
    try:
        PID_PATH.write_text(f"{victim.pid}\n")
        result = _run(["bash", str(AFTERWORDS), "codex-hook", "stop"], _base_env())
        _assert_alive(victim)
        assert "Codex watcher stopped" not in result.stdout
    finally:
        if victim.poll() is None:
            victim.kill()


def test_stop_does_not_follow_a_symlink_pidfile(shared_paths, tmp_path):
    victim = _spawn_ignorer()
    try:
        target = tmp_path / "pid-target"
        target.write_text(f"{victim.pid}\n")
        os.symlink(target, PID_PATH)
        result = _run(["bash", str(AFTERWORDS), "codex-hook", "stop"], _base_env())
        _assert_alive(victim)
        assert "Codex watcher stopped" not in result.stdout
    finally:
        if victim.poll() is None:
            victim.kill()


def test_stop_does_not_kill_pid_swapped_in_after_stat(shared_paths, tmp_path):
    """The pid file is a regular file at the ``[ -f`` check and a symlink at the read."""
    victim = _spawn_ignorer()
    try:
        target = tmp_path / "pid-target"
        target.write_text(f"{victim.pid}\n")
        PID_PATH.write_text("not-a-pid\n")
        trap = tmp_path / "swap-pid.sh"
        _write_swap_trap(trap, kind="pid")
        env = _base_env(BASH_ENV=str(trap), AW_LINK_TARGET=str(target))
        result = _run(["bash", str(AFTERWORDS), "codex-hook", "stop"], env)
        _assert_alive(victim)
        assert "Codex watcher stopped" not in result.stdout
    finally:
        if victim.poll() is None:
            victim.kill()


def test_start_does_not_treat_unrelated_live_pid_as_the_watcher(shared_paths):
    victim = _spawn_ignorer()
    try:
        PID_PATH.write_text(f"{victim.pid}\n")
        env = _base_env(CODEX_THREAD_ID="not-a-real-codex-thread")
        result = _run(["bash", str(AFTERWORDS), "codex-hook", "start"], env)
        _assert_alive(victim)
        assert f"already running (PID {victim.pid})" not in result.stdout
    finally:
        if victim.poll() is None:
            victim.kill()


def test_start_does_not_honor_pid_swapped_in_after_stat(shared_paths, tmp_path):
    victim = _spawn_ignorer()
    try:
        target = tmp_path / "pid-target"
        target.write_text(f"{victim.pid}\n")
        PID_PATH.write_text("not-a-pid\n")
        trap = tmp_path / "swap-pid.sh"
        _write_swap_trap(trap, kind="pid")
        env = _base_env(
            BASH_ENV=str(trap),
            AW_LINK_TARGET=str(target),
            CODEX_THREAD_ID="not-a-real-codex-thread",
        )
        result = _run(["bash", str(AFTERWORDS), "codex-hook", "start"], env)
        _assert_alive(victim)
        assert f"already running (PID {victim.pid})" not in result.stdout
    finally:
        if victim.poll() is None:
            victim.kill()


def test_start_log_redirect_does_not_follow_symlink(shared_paths, tmp_path):
    sentinel = tmp_path / "profile"
    sentinel.write_text(SECRET)
    os.symlink(sentinel, LOG_PATH)
    env = _base_env(CODEX_THREAD_ID="not-a-real-codex-thread")
    _run(["bash", str(AFTERWORDS), "codex-hook", "start"], env)
    assert sentinel.read_text() == SECRET


def test_start_log_redirect_does_not_follow_symlink_swapped_in_after_stat(shared_paths, tmp_path):
    """The log path is a regular file at a stat check and a symlink at the redirect."""
    sentinel = tmp_path / "profile"
    sentinel.write_text(SECRET)
    LOG_PATH.write_text("regular-log\n")
    trap = tmp_path / "swap-log.sh"
    _write_swap_trap(trap, kind="log")
    env = _base_env(
        BASH_ENV=str(trap),
        AW_LINK_TARGET=str(sentinel),
        CODEX_THREAD_ID="not-a-real-codex-thread",
    )
    _run(["bash", str(AFTERWORDS), "codex-hook", "start"], env)
    assert sentinel.read_text() == SECRET


def test_start_pid_write_does_not_follow_dangling_symlink(shared_paths, tmp_path):
    thread = "watcher-stays-up"
    home = tmp_path / "home"
    _plant_session(home, thread)
    created = tmp_path / "should-not-be-created"
    os.symlink(created, PID_PATH)
    env = _base_env(home, CODEX_THREAD_ID=thread)
    _run(["bash", str(AFTERWORDS), "codex-hook", "start"], env, timeout=15)
    assert not created.exists()


def test_watcher_log_append_does_not_follow_symlink(shared_paths, tmp_path):
    sentinel = tmp_path / "profile"
    sentinel.write_text(SECRET)
    os.symlink(sentinel, LOG_PATH)
    env = _base_env()
    assert "CODEX_WATCH_LOG" not in env
    _run(["bash", str(WATCHER)], env)
    assert sentinel.read_text() == SECRET


def test_diagnose_log_append_does_not_follow_symlink(shared_paths, tmp_path):
    sentinel = tmp_path / "profile"
    sentinel.write_text(SECRET)
    os.symlink(sentinel, LOG_PATH)
    env = _base_env()
    _run(["bash", str(AFTERWORDS), "codex-hook", "start", "--diagnose"], env)
    assert sentinel.read_text() == SECRET


@pytest.mark.skipif(not _sudo_attacker_available(), reason="need a second local account")
def test_other_local_user_pidfile_cannot_make_stop_kill(shared_paths):
    victim = _spawn_ignorer()
    try:
        _plant_foreign_pid(victim.pid)
        result = _run(["bash", str(AFTERWORDS), "codex-hook", "stop"], _base_env())
        _assert_alive(victim)
        assert "Codex watcher stopped" not in result.stdout
    finally:
        if victim.poll() is None:
            victim.kill()


@pytest.mark.skipif(not _sudo_attacker_available(), reason="need a second local account")
def test_other_local_user_pidfile_cannot_block_start(shared_paths):
    victim = _spawn_ignorer()
    try:
        _plant_foreign_pid(victim.pid)
        env = _base_env(CODEX_THREAD_ID="not-a-real-codex-thread")
        result = _run(["bash", str(AFTERWORDS), "codex-hook", "start"], env)
        _assert_alive(victim)
        assert f"already running (PID {victim.pid})" not in result.stdout
    finally:
        if victim.poll() is None:
            victim.kill()
