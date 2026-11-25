"""Optional backend git+ requirements must install one reviewed commit.

`pip install -r` on these files is the documented way to enable the optional
backends. Pip 24 treats a git+ URL with no `@<rev>` as the remote default
branch (`Git.fetch_new` leaves the clone at HEAD) and then executes that
tree's setup (`generate_metadata` → `python setup.py egg_info`). A branch
name or tag is the same class of moving ref: pip resolves it to whatever
commit the remote currently advertises. A full 40-hex commit id checks out
that object instead (`looks_like_hash`).

The upstream repositories are not contacted. Each case replays the revision
pip parsed from the product file against a local git remote with two
commits: an earlier reviewed commit, then a later default-branch commit whose
setup reads `$HOME/.afterwords-cloud` (the cloud API key file written by
`afterwords setup-cloud`). HOME is redirected at a placeholder file. The
product requirement passes only when its revision is a full commit SHA and
the setup that runs is the reviewed commit's setup.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from pip._internal.network.session import PipSession
from pip._internal.req.req_file import parse_requirements
from pip._internal.vcs.git import Git, looks_like_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
_LINE_NO = re.compile(r"\(line (\d+)\)")


def _git_direct_requirements() -> list[tuple[str, int, str, str | None]]:
    """Return (relative path, line, requirement, rev) for every git+ line.

    Revision parsing is pip's `Git.get_url_rev_and_auth`, the same split
    `obtain` uses before clone. Comments are dropped by `parse_requirements`,
    so commented `git clone` instructions are not install requirements.
    """
    session = PipSession()
    found: list[tuple[str, int, str, str | None]] = []
    for path in sorted(REPO_ROOT.rglob("requirements*.txt")):
        if any(part in {".git", ".venv"} for part in path.parts):
            continue
        for parsed in parse_requirements(str(path), session):
            requirement = parsed.requirement.strip()
            url = _git_url(requirement)
            if url is None:
                continue
            _remote, rev, _auth = Git.get_url_rev_and_auth(url)
            match = _LINE_NO.search(parsed.comes_from)
            lineno = int(match.group(1)) if match else 0
            rel = str(path.relative_to(REPO_ROOT))
            found.append((rel, lineno, requirement, rev))
    return found


def _git_url(requirement: str) -> str | None:
    if requirement.startswith("git+"):
        return requirement.split()[0]
    index = requirement.find("git+")
    if index == -1:
        return None
    return requirement[index:].split()[0]


GIT_REQUIREMENTS = _git_direct_requirements()


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _write_package(repo: Path, role: str) -> None:
    """Write the setup that pip executes for this commit.

    The attacker revision reads the cloud config path the product uses.
    The reviewed revision does not touch it. The marker path is injected
    so the test can see which file ran; it is not part of the revision
    choice.
    """
    if role == "attacker":
        body = textwrap.dedent(
            """\
            import os
            from pathlib import Path
            from setuptools import setup

            cloud = Path.home() / ".afterwords-cloud"
            read = cloud.read_text(encoding="utf-8")
            saw = "yes" if read.startswith("api_key=placeholder\\n") else "no"
            Path(os.environ["AFTERWORDS_SETUP_MARKER"]).write_text(
                f"attacker read-cloud={saw}\\n", encoding="utf-8"
            )
            setup(name="afterwords-git-standin", version="9.9.9", py_modules=["standin_mod"])
            """
        )
    else:
        body = textwrap.dedent(
            """\
            import os
            from pathlib import Path
            from setuptools import setup

            Path(os.environ["AFTERWORDS_SETUP_MARKER"]).write_text(
                "reviewed\\n", encoding="utf-8"
            )
            setup(name="afterwords-git-standin", version="0.0.1", py_modules=["standin_mod"])
            """
        )
    (repo / "setup.py").write_text(body, encoding="utf-8")
    (repo / "standin_mod.py").write_text("value = 1\n", encoding="utf-8")


def _upstream(tmp: Path) -> tuple[Path, str]:
    """Two-commit remote: reviewed parent, attacker tip of the default branch."""
    repo = tmp / "upstream"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "tests")
    _write_package(repo, "reviewed")
    _git(repo, "add", "setup.py", "standin_mod.py")
    _git(repo, "commit", "-m", "reviewed")
    reviewed = _git(repo, "rev-parse", "HEAD")
    _write_package(repo, "attacker")
    _git(repo, "add", "setup.py")
    _git(repo, "commit", "-m", "default branch moves")
    return repo, reviewed


def _point_movable_rev(repo: Path, rev: str) -> None:
    """Point a branch and a tag of this name at the default-branch tip.

    Pip prefers `refs/remotes/origin/<rev>` and then `refs/tags/<rev>`.
    Both names land on the later commit so a branch pin and a tag pin
    execute that commit's setup.
    """
    subprocess.run(
        ["git", "branch", "-f", rev, "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "tag", "-f", rev, "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def _pip_install_requirement(tmp: Path, repo: Path, rev: str | None) -> str:
    """`pip install -r` of one git+ line, preserving the parsed revision."""
    home = tmp / "home"
    home.mkdir(parents=True)
    cloud = home / ".afterwords-cloud"
    cloud.write_text(
        "api_key=placeholder\ncloud_url=https://example.invalid\n",
        encoding="utf-8",
    )
    cloud.chmod(0o600)
    marker = tmp / "setup-marker"
    if rev is not None and not looks_like_hash(rev):
        _point_movable_rev(repo, rev)
    local = repo.resolve().as_uri()
    line = f"git+{local}" if rev is None else f"git+{local}@{rev}"
    req = tmp / "requirements-replay.txt"
    req.write_text(line + "\n", encoding="utf-8")
    target = tmp / "target"
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["AFTERWORDS_SETUP_MARKER"] = str(marker)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-index",
            "--no-deps",
            "--target",
            str(target),
            "-r",
            str(req),
        ],
        cwd=tmp,
        env=env,
        capture_output=True,
        text=True,
    )
    if not marker.exists():
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-15:])
        return f"<setup did not run; pip exit {proc.returncode}; {tail}>"
    return marker.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize(
    ("rel", "lineno", "requirement", "rev"),
    GIT_REQUIREMENTS,
    ids=[f"{rel}:{lineno}" for rel, lineno, _requirement, _rev in GIT_REQUIREMENTS],
)
def test_git_requirement_installs_only_the_pinned_commit(
    rel: str, lineno: int, requirement: str, rev: str | None, tmp_path: Path
) -> None:
    repo, reviewed = _upstream(tmp_path / "src")
    # A full SHA is replayed as the fixture's reviewed commit: that object
    # id cannot be the upstream's real SHA, and pip's checkout path is the
    # same `@<40 hex>` form. Any other revision is replayed unchanged, so a
    # missing pin, a branch, a tag, or a short hash still selects the tip.
    install_rev: str | None = reviewed if rev is not None and looks_like_hash(rev) else rev
    executed = _pip_install_requirement(tmp_path / "install", repo, install_rev)
    assert rev is not None and looks_like_hash(rev) and executed == "reviewed", (
        f"{rel}:{lineno} requirement {requirement!r} revision {rev!r} "
        f"executed {executed!r}; pip must install a full commit SHA "
        f"(the reviewed commit on the fixture is {reviewed})"
    )
