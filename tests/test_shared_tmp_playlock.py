"""Playback lock and PID files must not follow a symlink planted in /tmp.

These tests fail while the workers write fixed paths under the shared sticky
/tmp with a plain redirect or Path.write_text. They pass once those writes
no longer replace the symlink target, and once a lock directory the current
user cannot remove makes acquire give up within a few seconds instead of
spinning.

Linux fs.protected_symlinks (default 1) refuses to follow another user's
symlink in a sticky world-writable directory. macOS open(2) follows a
symlink unless O_NOFOLLOW is set. The fixture clears the sysctl for the
duration of the module so the write is observable the way it is on the
platform this project ships for. The original value is restored afterwards.
"""
from __future__ import annotations

import os
import signal
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ATTACKER = "awattacker"
SENTINEL_TEXT = b"SENTINEL-PLAYLOCK-TARGET\n"
GIVE_UP_SECONDS = 3
SYSCTL = "/proc/sys/fs/protected_symlinks"

FIXED_PATHS = [
    "/tmp/afterwords-play.lock",
    "/tmp/afterwords-play.pid",
    "/tmp/claude-tts-worker.pid",
    "/tmp/claude-tts-worker.lock",
    "/tmp/codex-tts-worker-global.pid",
    "/tmp/codex-tts-worker-global.lock",
    "/tmp/codex-tts-watch.pid",
    "/tmp/codex-tts-watch.log",
]


def _sudo(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["sudo", *args], check=check, text=True, capture_output=True)


def _cleanup() -> None:
    _sudo("rm", "-rf", *FIXED_PATHS, check=False)


@pytest.fixture(scope="module", autouse=True)
def _shared_tmp_as_on_macos():
    if not os.path.exists(SYSCTL):
        pytest.skip("fs.protected_symlinks is not available")
    probe = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("sudo is required to model a second local account")
    original = Path(SYSCTL).read_text().strip()
    _sudo("sysctl", "-w", "fs.protected_symlinks=0")
    if subprocess.run(["id", ATTACKER], capture_output=True).returncode != 0:
        _sudo("useradd", "-M", "-s", "/usr/sbin/nologin", ATTACKER)
    yield
    _cleanup()
    _sudo("sysctl", "-w", f"fs.protected_symlinks={original}", check=False)


@pytest.fixture(autouse=True)
def _clean_fixed_paths():
    _cleanup()
    yield
    _cleanup()


def _extract_heredoc(text: str, marker: str) -> str:
    start = text.index(f"<<'{marker}'")
    start = text.index("\n", start) + 1
    end = text.index(f"\n{marker}\n", start)
    return text[start:end]


def _extract_function(text: str, name: str) -> tuple[str, str]:
    key = f"{name}()"
    start = text.index(key)
    prelude_lines = []
    for line in text[:start].splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for var in (
            "SESSION_ID",
            "PIDFILE",
            "LOCKDIR",
            "PLAY_LOCK",
            "PLAY_PID",
            "QUEUEDIR",
        ):
            if stripped.startswith(f"{var}="):
                prelude_lines.append(stripped)
    brace = text.index("{", start)
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return "\n".join(prelude_lines), text[start : index + 1]
    raise AssertionError(f"unterminated function {name}")


def _run_bash(script: str, timeout: float) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["HOME"] = "/tmp"
    proc = subprocess.Popen(
        ["bash", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=env,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        proc.returncode = 124
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def _plant_symlink(link: str, target: Path) -> None:
    _sudo("rm", "-f", link, check=False)
    _sudo("-u", ATTACKER, "ln", "-s", str(target), link)


def _assert_intact(sentinel: Path) -> None:
    assert sentinel.read_bytes() == SENTINEL_TEXT, (
        f"PID write followed the attacker symlink and replaced {sentinel} with "
        f"{sentinel.read_bytes()!r}"
    )


def _sentinel(tmp_path: Path) -> Path:
    path = tmp_path / "sentinel"
    path.write_bytes(SENTINEL_TEXT)
    path.chmod(0o644)
    return path


ACQUIRE_SOURCES = {
    "codex-tts-worker": REPO / ".claude/hooks/codex-tts-worker.sh",
    "post-llm": REPO / "scripts/afterwords-post-llm.sh",
}


def _acquire_script(label: str) -> str:
    if label == "setup-worker":
        text = _extract_heredoc((REPO / "setup.sh").read_text(), "WORKEREOF")
    else:
        text = ACQUIRE_SOURCES[label].read_text()
    prelude, function = _extract_function(text, "acquire_play_lock")
    return textwrap.dedent(
        f"""\
        set -uo pipefail
        {prelude}
        {function}
        acquire_play_lock
        echo RC:$?
        """
    )


@pytest.mark.parametrize("label", ["setup-worker", "codex-tts-worker", "post-llm"])
def test_acquire_play_lock_does_not_follow_attacker_symlink(label, tmp_path):
    sentinel = _sentinel(tmp_path)
    _plant_symlink("/tmp/afterwords-play.pid", sentinel)
    result = _run_bash(_acquire_script(label), timeout=5)
    assert result.returncode != 124, result.stderr[-500:]
    _assert_intact(sentinel)


@pytest.mark.parametrize("label", ["setup-worker", "codex-tts-worker", "post-llm"])
def test_acquire_play_lock_gives_up_when_lock_dir_is_not_removable(label):
    _sudo("-u", ATTACKER, "mkdir", "/tmp/afterwords-play.lock")
    _sudo("-u", ATTACKER, "bash", "-c", "echo 999999 > /tmp/afterwords-play.pid")
    result = _run_bash(_acquire_script(label), timeout=GIVE_UP_SECONDS)
    assert result.returncode != 124, (
        "acquire_play_lock was still running after "
        f"{GIVE_UP_SECONDS}s with an unremovable lock directory"
    )
    assert "RC:0" not in result.stdout, result.stdout


def test_command_provider_pid_write_does_not_follow_symlink(tmp_path):
    text = (REPO / "scripts/afterwords-tts-command.sh").read_text()
    start = text.index('PLAY_LOCK="/tmp/afterwords-play.lock"')
    end = text.index('bash -c \'echo $PPID\' > "$PLAY_PID"', start)
    end = text.index("\n", end)
    snippet = text[start:end]
    sentinel = _sentinel(tmp_path)
    _plant_symlink("/tmp/afterwords-play.pid", sentinel)
    script = "set -uo pipefail\n" + snippet + "\n"
    result = _run_bash(script, timeout=5)
    assert result.returncode != 124, result.stderr[-500:]
    _assert_intact(sentinel)


def test_worker_pidfiles_do_not_follow_symlink(tmp_path):
    cases = {
        "/tmp/claude-tts-worker.pid": _extract_heredoc(
            (REPO / "setup.sh").read_text(), "WORKEREOF"
        ),
        "/tmp/codex-tts-worker-global.pid": (
            REPO / ".claude/hooks/codex-tts-worker.sh"
        ).read_text(),
    }
    for pidfile, text in cases.items():
        _cleanup()
        sentinel = _sentinel(tmp_path)
        _plant_symlink(pidfile, sentinel)
        # Execute the real `echo $$ > "$PIDFILE"` line in that file, after the
        # assignments that choose the path.
        prelude, _fn = _extract_function(text, "acquire_play_lock")
        line = next(
            line.strip()
            for line in text.splitlines()
            if line.strip() == 'echo $$ > "$PIDFILE"'
        )
        script = f"set -uo pipefail\nexport CODEX_THREAD_ID=global\n{prelude}\n{line}\n"
        result = _run_bash(script, timeout=5)
        assert result.returncode == 0, result.stderr
        _assert_intact(sentinel)


def test_codex_watch_pid_write_does_not_follow_symlink(tmp_path):
    text = (REPO / "afterwords.sh").read_text()
    line = next(
        line.strip()
        for line in text.splitlines()
        if line.strip() == 'echo "$pid" > "$CODEX_WATCH_PID"'
    )
    sentinel = _sentinel(tmp_path)
    _plant_symlink("/tmp/codex-tts-watch.pid", sentinel)
    script = textwrap.dedent(
        f"""\
        set -uo pipefail
        CODEX_WATCH_PID="/tmp/codex-tts-watch.pid"
        pid=$$
        {line}
        """
    )
    result = _run_bash(script, timeout=5)
    assert result.returncode == 0, result.stderr
    _assert_intact(sentinel)


def _python_speak(timeout: float) -> subprocess.CompletedProcess:
    code = textwrap.dedent(
        """\
        import asyncio, importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "afterwords_handler", os.environ["HANDLER"]
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        async def main():
            await mod._speak_chunked(["hello from the play-lock test"], None)
            print("RETURNED")
        asyncio.run(main())
        """
    )
    env = os.environ.copy()
    env["HANDLER"] = str(REPO / "hermes/hooks/afterwords-tts/handler.py")
    proc = subprocess.Popen(
        ["python3", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=env,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        proc.returncode = 124
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def test_handler_pid_write_does_not_follow_symlink(tmp_path):
    sentinel = _sentinel(tmp_path)
    _plant_symlink("/tmp/afterwords-play.pid", sentinel)
    result = _python_speak(timeout=5)
    assert result.returncode != 124, result.stderr[-500:]
    _assert_intact(sentinel)


def test_handler_gives_up_when_lock_dir_is_not_removable():
    _sudo("-u", ATTACKER, "mkdir", "/tmp/afterwords-play.lock")
    _sudo("-u", ATTACKER, "bash", "-c", "echo 999999 > /tmp/afterwords-play.pid")
    result = _python_speak(timeout=GIVE_UP_SECONDS)
    assert result.returncode != 124, (
        "handler _speak_chunked was still running after "
        f"{GIVE_UP_SECONDS}s with an unremovable lock directory"
    )
    assert "RETURNED" in result.stdout
