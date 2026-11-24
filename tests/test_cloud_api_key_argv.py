"""Cloud commands must not hand the API key to another local account.

The cloud API key is placed in the argument vector of curl (voices --cloud,
setup-cloud, push, and both pull requests). macOS installs /bin/ps setuid
root, and that ps prints the full argument vector of every process, so
another account on the machine can read the key while curl is running.

setup-cloud also creates ~/.afterwords-cloud with the process umask and only
afterwards chmod 600, so the file is group- and world-readable until chmod
runs.

These tests fail until both are fixed. They use an obvious placeholder value,
never a real key.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
AFTERWORDS = REPO / "afterwords.sh"

# Obvious placeholder. Not a credential.
PLACEHOLDER = "PLACEHOLDER_CLOUD_API_KEY"
CLOUD_URL = "http://127.0.0.1:9"

AUDIT_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

static int sys_openat(const char *path, int flags, int mode) {
    return (int)syscall(SYS_openat, AT_FDCWD, path, flags, mode);
}

static void append_log(const char *env_name, const char *text) {
    const char *path = getenv(env_name);
    if (!path || !text) return;
    int fd = sys_openat(path, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (fd < 0) return;
    flock(fd, LOCK_EX);
    size_t len = strlen(text);
    const char *p = text;
    while (len) {
        ssize_t n = write(fd, p, len);
        if (n < 0) break;
        p += n;
        len -= (size_t)n;
    }
    flock(fd, LOCK_UN);
    close(fd);
}

static void log_exec(const char *how, char *const argv[]) {
    char buf[65536];
    size_t used = 0;
    int n = snprintf(buf, sizeof buf, "EXEC %s\n", how);
    if (n < 0) return;
    used = (size_t)n;
    if (argv) {
        for (int i = 0; argv[i] && used + 8 < sizeof buf; i++) {
            n = snprintf(buf + used, sizeof buf - used, "ARG %s\n", argv[i]);
            if (n < 0) break;
            if ((size_t)n >= sizeof buf - used) {
                used = sizeof buf - 1;
                break;
            }
            used += (size_t)n;
        }
    }
    if (used + 5 < sizeof buf) {
        memcpy(buf + used, "END\n", 4);
        used += 4;
        buf[used] = '\0';
    }
    append_log("AW_EXEC_LOG", buf);
}

static void log_create(const char *path, int mode) {
    if (!path || !strstr(path, ".afterwords-cloud")) return;
    mode_t um = umask(0);
    umask(um);
    unsigned eff = (unsigned)(mode & ~um & 0777);
    char buf[1024];
    snprintf(buf, sizeof buf, "CREATE %o %s\n", eff, path);
    append_log("AW_CREATE_LOG", buf);
}

static int (*real_execve)(const char *, char *const [], char *const []);
static int (*real_execv)(const char *, char *const []);
static int (*real_execvp)(const char *, char *const []);
static int (*real_execvpe)(const char *, char *const [], char *const []);
static int (*real_posix_spawn)(pid_t *, const char *, const posix_spawn_file_actions_t *,
    const posix_spawnattr_t *, char *const [], char *const []);
static int (*real_posix_spawnp)(pid_t *, const char *, const posix_spawn_file_actions_t *,
    const posix_spawnattr_t *, char *const [], char *const []);
static int (*real_open)(const char *, int, ...);
static int (*real_open64)(const char *, int, ...);
static int (*real_openat)(int, const char *, int, ...);
static int (*real_openat64)(int, const char *, int, ...);

static void resolve(void **slot, const char *name) {
    if (*slot) return;
    *slot = dlsym(RTLD_NEXT, name);
}

int execve(const char *path, char *const argv[], char *const envp[]) {
    resolve((void **)&real_execve, "execve");
    log_exec("execve", argv);
    return real_execve(path, argv, envp);
}
int execv(const char *path, char *const argv[]) {
    resolve((void **)&real_execv, "execv");
    log_exec("execv", argv);
    return real_execv(path, argv);
}
int execvp(const char *path, char *const argv[]) {
    resolve((void **)&real_execvp, "execvp");
    log_exec("execvp", argv);
    return real_execvp(path, argv);
}
int execvpe(const char *path, char *const argv[], char *const envp[]) {
    resolve((void **)&real_execvpe, "execvpe");
    log_exec("execvpe", argv);
    return real_execvpe(path, argv, envp);
}
int posix_spawn(pid_t *pid, const char *path, const posix_spawn_file_actions_t *fa,
    const posix_spawnattr_t *attr, char *const argv[], char *const envp[]) {
    resolve((void **)&real_posix_spawn, "posix_spawn");
    log_exec("posix_spawn", argv);
    return real_posix_spawn(pid, path, fa, attr, argv, envp);
}
int posix_spawnp(pid_t *pid, const char *path, const posix_spawn_file_actions_t *fa,
    const posix_spawnattr_t *attr, char *const argv[], char *const envp[]) {
    resolve((void **)&real_posix_spawnp, "posix_spawnp");
    log_exec("posix_spawnp", argv);
    return real_posix_spawnp(pid, path, fa, attr, argv, envp);
}

static int open_common(int (*real)(const char *, int, ...), const char *path, int flags, int mode, int created) {
    if (created) log_create(path, mode);
    if (created) return real(path, flags, mode);
    return real(path, flags);
}
int open(const char *path, int flags, ...) {
    resolve((void **)&real_open, "open");
    int mode = 0;
    int created = (flags & O_CREAT) != 0;
    if (created) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap);
    }
    return open_common(real_open, path, flags, mode, created);
}
int open64(const char *path, int flags, ...) {
    resolve((void **)&real_open64, "open64");
    int mode = 0;
    int created = (flags & O_CREAT) != 0;
    if (created) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap);
    }
    return open_common(real_open64, path, flags, mode, created);
}
int openat(int dirfd, const char *path, int flags, ...) {
    resolve((void **)&real_openat, "openat");
    int mode = 0;
    int created = (flags & O_CREAT) != 0;
    if (created) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap);
        log_create(path, mode);
    }
    if (created) return real_openat(dirfd, path, flags, mode);
    return real_openat(dirfd, path, flags);
}
int openat64(int dirfd, const char *path, int flags, ...) {
    resolve((void **)&real_openat64, "openat64");
    int mode = 0;
    int created = (flags & O_CREAT) != 0;
    if (created) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap);
        log_create(path, mode);
    }
    if (created) return real_openat64(dirfd, path, flags, mode);
    return real_openat64(dirfd, path, flags);
}
"""

CURL_STUB = r"""#!/usr/bin/env python3
import sys

args = sys.argv[1:]
if any("/health" in a for a in args):
    sys.exit(1)

url = next(a for a in args if a.startswith("http"))
out = None
method = "GET"
for i, arg in enumerate(args):
    if arg == "-o" and i + 1 < len(args):
        out = args[i + 1]
    elif arg == "-X" and i + 1 < len(args):
        method = args[i + 1]

if out == "/dev/null":
    sys.stdout.write("200")
    sys.exit(0)
if out:
    with open(out, "wb") as fh:
        fh.write(b"RIFF")
    sys.stdout.write("200")
    sys.exit(0)
if method == "POST":
    sys.stdout.write('{"voice_id":"vid-1"}\n201')
    sys.exit(0)
if url.rstrip("/").endswith("/v1/voices"):
    sys.stdout.write(
        '[{"name":"probe","backend":"qwen3-0.6b","voice_id":"abcdef1234567890"}]\n200'
    )
    sys.exit(0)
sys.stdout.write(
    '{"name":"pulledvoice","backend":"qwen3-0.6b","ref_text":"hello","voice_id":"vid-1"}\n200'
)
"""


@pytest.fixture(scope="module")
def audit_lib(tmp_path_factory):
    work = tmp_path_factory.mktemp("audit-lib")
    source = work / "audit.c"
    library = work / "audit.so"
    source.write_text(AUDIT_C)
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-Werror", "-o", str(library), str(source), "-ldl"],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


def _exec_records(log: str) -> list[list[str]]:
    records: list[list[str]] = []
    current: list[str] = []
    for line in log.splitlines():
        if line.startswith("EXEC "):
            current = []
        elif line.startswith("ARG "):
            current.append(line[4:])
        elif line == "END":
            records.append(current)
    return records


def _is_curl(record: list[str]) -> bool:
    if not record:
        return False
    name = record[0].rsplit("/", 1)[-1]
    return name == "curl"


def _assert_placeholder_absent(records: list[list[str]]) -> None:
    leaked = sum(
        1 for rec in records if any(PLACEHOLDER in arg for arg in rec)
    )
    assert leaked == 0, (
        "cloud API key was passed in a process argument vector "
        f"({leaked} exec)"
    )


def _prepare_tree(tmp_path: Path, *, with_config: bool, with_voices: bool) -> tuple[Path, Path]:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o755)
    repo = tmp_path / "repo"
    voices = repo / "voices"
    voices.mkdir(parents=True)
    if with_voices:
        for name in ("probe", "probe-alt"):
            (voices / f"{name}-ref.wav").write_bytes(b"RIFF")
            (voices / f"{name}.json").write_text(json.dumps({
                "name": name,
                "reference_audio": f"{name}-ref.wav",
                "reference_text": "hello",
                "backend": "qwen3-0.6b",
                "family": "probe",
            }))
    if with_config:
        config = home / ".afterwords-cloud"
        config.write_text(f"api_key={PLACEHOLDER}\ncloud_url={CLOUD_URL}\n")
        config.chmod(0o600)
    return home, repo


def _run(
    audit_lib: Path,
    home: Path,
    repo: Path,
    args: list[str],
    stdin: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], str]:
    bindir = home.parent / "bin"
    bindir.mkdir(exist_ok=True)
    curl = bindir / "curl"
    curl.write_text(CURL_STUB)
    curl.chmod(0o755)
    exec_log = home.parent / "exec.log"
    create_log = home.parent / "create.log"
    exec_log.write_text("")
    create_log.write_text("")
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(home),
        "AFTERWORDS_REPO_DIR": str(repo),
        "LD_PRELOAD": str(audit_lib),
        "AW_EXEC_LOG": str(exec_log),
        "AW_CREATE_LOG": str(create_log),
        "LANG": "C",
    }
    result = subprocess.run(
        ["bash", str(AFTERWORDS), *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        preexec_fn=lambda: os.umask(0o022),
        check=False,
    )
    records = _exec_records(exec_log.read_text())
    created = create_log.read_text()
    return result, records, created


def _assert_completed(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"command exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_voices_cloud_does_not_pass_api_key_in_process_arguments(audit_lib, tmp_path):
    home, repo = _prepare_tree(tmp_path, with_config=True, with_voices=False)
    result, records, _created = _run(audit_lib, home, repo, ["voices", "--cloud"])
    _assert_completed(result)
    listed = [
        rec for rec in records
        if _is_curl(rec)
        and any(arg.rstrip("/").endswith("/v1/voices") for arg in rec)
        and "POST" not in rec
    ]
    assert listed, "voices --cloud did not request the cloud voice list"
    _assert_placeholder_absent(records)


def test_setup_cloud_does_not_pass_api_key_in_process_arguments(audit_lib, tmp_path):
    home, repo = _prepare_tree(tmp_path, with_config=False, with_voices=False)
    result, records, _created = _run(
        audit_lib, home, repo, ["setup-cloud"],
        stdin=f"{PLACEHOLDER}\n{CLOUD_URL}\n",
    )
    _assert_completed(result)
    probed = [
        rec for rec in records
        if _is_curl(rec)
        and any(arg.rstrip("/").endswith("/v1/voices") for arg in rec)
    ]
    assert probed, "setup-cloud did not test the cloud connection"
    _assert_placeholder_absent(records)


def test_push_does_not_pass_api_key_in_process_arguments(audit_lib, tmp_path):
    home, repo = _prepare_tree(tmp_path, with_config=True, with_voices=True)
    result, records, _created = _run(audit_lib, home, repo, ["push", "probe"])
    _assert_completed(result)
    uploads = [
        rec for rec in records
        if _is_curl(rec)
        and "POST" in rec
        and any("/v1/voices" in arg for arg in rec)
    ]
    assert len(uploads) == 2, "push did not upload every profile in the family"
    _assert_placeholder_absent(records)


def test_pull_does_not_pass_api_key_in_process_arguments(audit_lib, tmp_path):
    home, repo = _prepare_tree(tmp_path, with_config=True, with_voices=False)
    result, records, _created = _run(audit_lib, home, repo, ["pull", "voice-1"])
    _assert_completed(result)
    meta = [
        rec for rec in records
        if _is_curl(rec)
        and any(arg.endswith("/v1/voices/voice-1") for arg in rec)
    ]
    audio = [
        rec for rec in records
        if _is_curl(rec)
        and any(arg.endswith("/v1/voices/voice-1/audio") for arg in rec)
    ]
    assert len(meta) == 1, "pull did not fetch voice metadata"
    assert len(audio) == 1, "pull did not download reference audio"
    _assert_placeholder_absent(records)


def test_setup_cloud_creates_config_without_group_or_world_access(audit_lib, tmp_path):
    home, repo = _prepare_tree(tmp_path, with_config=False, with_voices=False)
    result, _records, created = _run(
        audit_lib, home, repo, ["setup-cloud"],
        stdin=f"{PLACEHOLDER}\n{CLOUD_URL}\n",
    )
    _assert_completed(result)
    config = home / ".afterwords-cloud"
    assert config.is_file(), "setup-cloud did not write the cloud config"
    creates = [line for line in created.splitlines() if line.startswith("CREATE ")]
    assert creates, "did not observe creation of the cloud config file"
    for line in creates:
        mode_text = line.split()[1]
        mode = int(mode_text, 8)
        assert mode & 0o077 == 0, (
            "cloud config was created with group or world access "
            f"(mode {mode:03o}) before a later chmod"
        )
    final = stat.S_IMODE(config.stat().st_mode)
    assert final & 0o077 == 0, f"cloud config left with mode {final:03o}"
