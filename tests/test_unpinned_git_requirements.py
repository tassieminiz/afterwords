"""Direct git requirements must name a full commit.

`pip install -r` on these files clones each `git+` URL and runs that
revision's `setup.py`. With no revision, or with a branch name, pip
checks out the default branch tip, so a later commit on that branch runs
on the next install. A 40-hex commit does not follow the branch.

The pin test fails until every direct git requirement is pinned that way.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pip._internal.network.session import PipSession
from pip._internal.req.constructors import install_req_from_parsed_requirement
from pip._internal.req.req_file import parse_requirements
from pip._internal.vcs.git import Git, looks_like_hash

REPO = Path(__file__).resolve().parent.parent

# requirements-openvoice.txt installs both of these. Pinning the file means
# both URLs stay and each names a full commit.
REQUIRED_GIT_REPOS = (
    "github.com/myshell-ai/OpenVoice",
    "github.com/myshell-ai/MeloTTS",
)


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Pin Test"
    env["GIT_AUTHOR_EMAIL"] = "pin-test@example.com"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    return env


def _git(repo: Path, *args: str, env: dict[str, str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _setup_py(marker: Path, payload: str, version: str) -> str:
    # The marker path is baked into the commit. Pip executes this file, and
    # build isolation does not forward an arbitrary environment variable.
    return (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text({payload!r})\n"
        "from setuptools import setup\n"
        f"setup(name='afterwords-git-pin-standin', version={version!r})\n"
    )


def _commit_repo(repo: Path, marker: Path) -> tuple[str, str]:
    """Two commits on the default branch. The tip setup records 'compromised'."""
    env = _git_env()
    _git(repo, "init", "-b", "main", env=env)
    reviewed = repo / "setup.py"
    reviewed.write_text(_setup_py(marker, "reviewed\n", "0.0.1"))
    _git(repo, "add", "setup.py", env=env)
    _git(repo, "commit", "-m", "reviewed", env=env)
    reviewed_sha = _git(repo, "rev-parse", "HEAD", env=env)
    reviewed.write_text(_setup_py(marker, "compromised\n", "0.0.2"))
    _git(repo, "add", "setup.py", env=env)
    _git(repo, "commit", "-m", "compromised", env=env)
    return reviewed_sha, _git(repo, "rev-parse", "HEAD", env=env)


def _pip_install_requirement(requirement: str, target: Path) -> subprocess.CompletedProcess[str]:
    req_file = target.parent / f"{target.name}.txt"
    req_file.write_text(requirement + "\n")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-deps",
            "--target",
            str(target),
            "-r",
            str(req_file),
        ],
        capture_output=True,
        text=True,
    )


def _install_ref(repo: Path, marker: Path, work: Path, rev: str | None) -> str:
    """Install `git+file://<repo>` the way a requirements line would.

    `rev` is appended exactly as pip parsed it. None means the line had no
    `@revision`, which is what `git+https://...OpenVoice.git` looks like.
    """
    if marker.exists():
        marker.unlink()
    url = f"git+file://{repo}"
    if rev:
        url = f"{url}@{rev}"
    result = _pip_install_requirement(url, work)
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output
    assert marker.is_file(), output
    return marker.read_text()


def _vcs_requirements():
    session = PipSession()
    for path in sorted(REPO.glob("requirements*.txt")):
        for parsed in parse_requirements(str(path), session=session):
            ireq = install_req_from_parsed_requirement(parsed)
            link = ireq.link
            if link is None or not link.is_vcs:
                continue
            url, rev, _auth = Git.get_url_rev_and_auth(link.url)
            yield path, url, rev


def test_full_commit_pin_does_not_run_a_later_default_branch_setup(tmp_path: Path) -> None:
    """A missing revision and a branch name run the tip. A full commit does not."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    marker = tmp_path / "marker.txt"
    reviewed_sha, _tip = _commit_repo(repo, marker)

    assert looks_like_hash(reviewed_sha)
    assert not looks_like_hash("main")

    assert _install_ref(repo, marker, tmp_path / "unpinned", None) == "compromised\n"
    assert (
        _install_ref(repo, marker, tmp_path / "pinned", reviewed_sha) == "reviewed\n"
    )
    assert _install_ref(repo, marker, tmp_path / "branch", "main") == "compromised\n"


def test_shipped_git_requirements_are_full_commit_pins(tmp_path: Path) -> None:
    """Every direct git requirement in requirements*.txt names a full commit.

    For each line that does not, pip install that same revision shape against
    a local stand-in whose default branch has moved. The tip setup runs.
    """
    seen: dict[str, str | None] = {}
    floating: list[str] = []
    demo_root = tmp_path / "demo"
    demo_root.mkdir()
    repo = demo_root / "upstream"
    repo.mkdir()
    marker = demo_root / "marker.txt"
    _commit_repo(repo, marker)

    for index, (path, url, rev) in enumerate(_vcs_requirements()):
        for repo_name in REQUIRED_GIT_REPOS:
            if repo_name in url:
                seen[repo_name] = rev
        if rev and looks_like_hash(rev):
            continue
        payload = _install_ref(repo, marker, demo_root / f"target-{index}", rev)
        assert payload == "compromised\n", (
            f"{path.name} rev {rev!r} did not execute the default-branch tip"
        )
        floating.append(f"{path.relative_to(REPO)}: {url} rev={rev!r}")

    missing = [name for name in REQUIRED_GIT_REPOS if name not in seen]
    assert not missing, (
        "requirements-openvoice.txt must still install "
        + " and ".join(missing)
        + " from git, pinned to a full commit"
    )
    assert floating == [], (
        "pin each git requirement to a full 40-hex commit; "
        "a missing revision or a branch name installs the default branch tip "
        "and runs its setup.py:\n" + "\n".join(floating)
    )


def test_branch_name_is_not_a_full_commit_pin() -> None:
    """`@main` is the revision shape that still follows the default branch."""
    assert not looks_like_hash("main")
    assert looks_like_hash("a" * 40)
    assert not looks_like_hash("a" * 39)
    assert not looks_like_hash("")
