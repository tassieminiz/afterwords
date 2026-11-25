"""Optional-backend Git installs must not track a moving default branch.

`pip install -r requirements-dia2.txt` (and the sibling optional-backend
requirements files) is the documented way to install those backends. Pip
treats a `git+` URL with no `@<commit>` as "clone the remote and check out
whatever the default branch is now", then runs that tree's package setup.
These tests drive that install with the pip that is installed here and fail
until every shipped `git+` requirement names a full 40-hex commit, the same
shape already used by requirements-cosyvoice2.txt and
requirements-firered-tts-2.txt.

They fail on purpose while those URLs are unpinned.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pip._internal.network.session import PipSession
from pip._internal.req.constructors import parse_req_from_line
from pip._internal.req.req_file import parse_requirements
from pip._internal.vcs.git import Git, looks_like_hash

REPO = Path(__file__).resolve().parent.parent

# Install lines the product ships. Deleting one would drop the backend, so the
# spec is that each of these repositories stays and is pinned.
EXPECTED_GIT_REPOS = (
    ("requirements-dia2.txt", "https://github.com/nari-labs/dia2.git"),
    ("requirements-indextts.txt", "https://github.com/index-tts/index-tts.git"),
    ("requirements-openvoice.txt", "https://github.com/myshell-ai/OpenVoice.git"),
    ("requirements-openvoice.txt", "https://github.com/myshell-ai/MeloTTS.git"),
    (
        "docs/research/tts-backends/openvoice-v2.md",
        "https://github.com/myshell-ai/MeloTTS.git",
    ),
)

CLOUD_TEXT = "api_key=PLACEHOLDER_NOT_A_SECRET\ncloud_url=https://example.invalid\n"
VOICE_TEXT = '{"name":"sample","reference_text":"hello from the voice profile"}\n'


def _requirement_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("requirements*.txt")):
        if any(part in {".git", ".venv", "node_modules"} for part in path.parts):
            continue
        files.append(path)
    return files


def _git_links_from_requirements(root: Path) -> list[tuple[Path, str, str, str | None]]:
    """Return (file, line_source, url_without_vcs_prefix, rev) using pip's parser."""
    session = PipSession()
    found = []
    for path in _requirement_files(root):
        for parsed in parse_requirements(str(path), session=session):
            parts = parse_req_from_line(parsed.requirement, parsed.line_source)
            link = parts.link
            if link is None or not link.is_vcs or not link.scheme.startswith("git+"):
                continue
            url, rev, _auth = Git.get_url_rev_and_auth(link.url)
            found.append((path, parsed.line_source or "", url, rev))
    return found


def _documented_pip_git_urls(root: Path) -> list[tuple[Path, int, str, str | None]]:
    """`pip install git+...` instructions outside requirements files."""
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".sh", ".yml", ".yaml"}:
            continue
        if any(part in {".git", ".venv", "node_modules"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "pip" not in line or "install" not in line or "git+" not in line:
                continue
            for token in line.split():
                if not token.startswith("git+"):
                    continue
                raw = token.strip("`'\"),")
                url, rev, _auth = Git.get_url_rev_and_auth(raw)
                found.append((path, lineno, url, rev))
    return found


def _unpinned(entries: list[tuple]) -> list[str]:
    """entries are (location, ..., url, rev) with url and rev last."""
    bad = []
    for entry in entries:
        url, rev = entry[-2], entry[-1]
        location = entry[0]
        extra = entry[1]
        if not (isinstance(rev, str) and looks_like_hash(rev)):
            bad.append(f"{location}:{extra}: {url} rev={rev!r}")
    return bad


def test_requirements_git_urls_are_full_commit_pins():
    """Every requirements-file git dependency must be a 40-hex commit.

    Pip's looks_like_hash (pip/_internal/vcs/git.py) is the same check pip uses
    before it will treat a revision as a commit id. A branch name, including
    the implicit default branch of a URL with no `@rev`, fails it.
    """
    found = _git_links_from_requirements(REPO)
    by_repo = {(path.relative_to(REPO).as_posix(), url): rev for path, _src, url, rev in found}
    missing = []
    for rel, url in EXPECTED_GIT_REPOS:
        if not rel.endswith(".txt"):
            continue
        if (rel, url) not in by_repo:
            missing.append(f"{rel} does not install {url}")
    unpinned = _unpinned(found)
    assert missing == [], "expected git installs were removed:\n" + "\n".join(missing)
    assert unpinned == [], (
        "git requirements are not pinned to a full commit:\n" + "\n".join(unpinned)
    )


def test_documented_pip_git_installs_are_full_commit_pins():
    found = _documented_pip_git_urls(REPO)
    by_repo = {(path.relative_to(REPO).as_posix(), url): rev for path, _lineno, url, rev in found}
    missing = []
    for rel, url in EXPECTED_GIT_REPOS:
        if rel.endswith(".txt"):
            continue
        if (rel, url) not in by_repo:
            missing.append(f"{rel} does not install {url}")
    unpinned = _unpinned(found)
    assert missing == [], "expected documented git installs were removed:\n" + "\n".join(missing)
    assert unpinned == [], (
        "documented pip git installs are not pinned to a full commit:\n" + "\n".join(unpinned)
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _rev_parse(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Pin Test")


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _rev_parse(repo)


def _write_setup_py(repo: Path, marker: Path, cloud: Path, voice: Path, compromised: bool) -> None:
    if compromised:
        body = (
            "from pathlib import Path\n"
            "from setuptools import setup\n"
            f"secret = Path({str(cloud)!r}).read_text()\n"
            f"profile = Path({str(voice)!r}).read_text()\n"
            f"Path({str(marker)!r}).write_text('compromised\\n' + secret + '\\n' + profile)\n"
            "setup(name='supplychain-probe', version='0.0.2')\n"
        )
    else:
        body = (
            "from pathlib import Path\n"
            "from setuptools import setup\n"
            f"Path({str(marker)!r}).write_text('benign')\n"
            "setup(name='supplychain-probe', version='0.0.1')\n"
        )
    (repo / "setup.py").write_text(body)


def _write_pep517_backend(repo: Path, marker: Path, cloud: Path, voice: Path, compromised: bool) -> None:
    """In-tree build backend. Importing it is the step pip's hook runner takes."""
    if compromised:
        effect = (
            f"secret = Path({str(cloud)!r}).read_text()\n"
            f"profile = Path({str(voice)!r}).read_text()\n"
            f"Path({str(marker)!r}).write_text('compromised\\n' + secret + '\\n' + profile)\n"
        )
        version = "0.0.2"
    else:
        effect = f"Path({str(marker)!r}).write_text('benign')\n"
        version = "0.0.1"
    (repo / "pyproject.toml").write_text(
        "[build-system]\n"
        "requires = []\n"
        'build-backend = "attacker_backend"\n'
        'backend-path = ["."]\n'
    )
    (repo / "attacker_backend.py").write_text(
        "import base64\n"
        "import hashlib\n"
        "import os\n"
        "import zipfile\n"
        "from pathlib import Path\n"
        f"{effect}"
        "def get_requires_for_build_wheel(config_settings=None):\n"
        "    return []\n"
        f"VERSION = {version!r}\n"
        "def _meta():\n"
        "    return (\n"
        "        'Metadata-Version: 2.1\\n'\n"
        "        'Name: supplychain-probe\\n'\n"
        "        'Version: ' + VERSION + '\\n'\n"
        "    )\n"
        "def _wheel_text():\n"
        "    return (\n"
        "        'Wheel-Version: 1.0\\n'\n"
        "        'Generator: probe\\n'\n"
        "        'Root-Is-Purelib: true\\n'\n"
        "        'Tag: py3-none-any\\n'\n"
        "    )\n"
        "def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):\n"
        "    name = 'supplychain_probe-' + VERSION + '.dist-info'\n"
        "    dest = os.path.join(metadata_directory, name)\n"
        "    os.makedirs(dest, exist_ok=True)\n"
        "    Path(os.path.join(dest, 'METADATA')).write_text(_meta())\n"
        "    Path(os.path.join(dest, 'WHEEL')).write_text(_wheel_text())\n"
        "    return name\n"
        "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
        "    filename = 'supplychain_probe-' + VERSION + '-py3-none-any.whl'\n"
        "    path = os.path.join(wheel_directory, filename)\n"
        "    dist = 'supplychain_probe-' + VERSION + '.dist-info'\n"
        "    files = {dist + '/METADATA': _meta(), dist + '/WHEEL': _wheel_text()}\n"
        "    lines = []\n"
        "    for fn, data in files.items():\n"
        "        raw = data.encode()\n"
        "        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode()\n"
        "        lines.append(f'{fn},sha256={digest},{len(raw)}')\n"
        "    lines.append(dist + '/RECORD,,')\n"
        "    files[dist + '/RECORD'] = '\\n'.join(lines) + '\\n'\n"
        "    with zipfile.ZipFile(path, 'w') as zf:\n"
        "        for fn, data in files.items():\n"
        "            zf.writestr(fn, data)\n"
        "    return filename\n"
    )


def _pip_install(spec: str, target: Path) -> subprocess.CompletedProcess[str]:
    target.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PIP_NO_CACHE_DIR"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            spec,
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    cloud = tmp_path / "home" / ".afterwords-cloud"
    voice = tmp_path / "voices" / "sample.json"
    cloud.parent.mkdir(parents=True)
    voice.parent.mkdir(parents=True)
    cloud.write_text(CLOUD_TEXT)
    voice.write_text(VOICE_TEXT)
    return tmp_path / "marker", cloud, voice


def _assert_ran(completed: subprocess.CompletedProcess[str], marker: Path, expected_prefix: str) -> str:
    text = marker.read_text() if marker.exists() else ""
    assert completed.returncode == 0, (
        "pip install failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert text.startswith(expected_prefix), text
    return text


def test_unpinned_git_url_executes_default_branch_setup_py(tmp_path: Path):
    """A setup.py project (OpenVoice, MeloTTS) runs the default-branch file."""
    marker, cloud, voice = _fixtures(tmp_path)
    repo = tmp_path / "upstream"
    _init_repo(repo)
    _write_setup_py(repo, marker, cloud, voice, compromised=False)
    benign = _commit_all(repo, "benign")
    _write_setup_py(repo, marker, cloud, voice, compromised=True)
    moved = _commit_all(repo, "compromised")
    assert benign != moved

    completed = _pip_install(f"git+file://{repo}", tmp_path / "site")
    text = _assert_ran(completed, marker, "compromised")
    assert "PLACEHOLDER_NOT_A_SECRET" in text
    assert "hello from the voice profile" in text
    assert moved in completed.stdout


def test_unpinned_git_url_executes_default_branch_pep517_backend(tmp_path: Path):
    """A pyproject build backend (Dia2, IndexTTS) is imported from the default branch."""
    marker, cloud, voice = _fixtures(tmp_path)
    repo = tmp_path / "upstream"
    _init_repo(repo)
    _write_pep517_backend(repo, marker, cloud, voice, compromised=False)
    _commit_all(repo, "benign")
    _write_pep517_backend(repo, marker, cloud, voice, compromised=True)
    moved = _commit_all(repo, "compromised")

    completed = _pip_install(f"git+file://{repo}", tmp_path / "site")
    text = _assert_ran(completed, marker, "compromised")
    assert "PLACEHOLDER_NOT_A_SECRET" in text
    assert "hello from the voice profile" in text
    assert moved in completed.stdout


def test_full_commit_pin_does_not_run_later_default_branch(tmp_path: Path):
    marker, cloud, voice = _fixtures(tmp_path)
    repo = tmp_path / "upstream"
    _init_repo(repo)
    _write_setup_py(repo, marker, cloud, voice, compromised=False)
    benign = _commit_all(repo, "benign")
    _write_setup_py(repo, marker, cloud, voice, compromised=True)
    _commit_all(repo, "compromised")
    assert looks_like_hash(benign)

    completed = _pip_install(f"git+file://{repo}@{benign}", tmp_path / "site")
    text = _assert_ran(completed, marker, "benign")
    assert "PLACEHOLDER_NOT_A_SECRET" not in text
    assert benign in completed.stdout


def test_branch_name_pin_still_executes_the_moved_tip(tmp_path: Path):
    """`@main` is not a pin: pip checks out the branch tip after it moves."""
    marker, cloud, voice = _fixtures(tmp_path)
    repo = tmp_path / "upstream"
    _init_repo(repo)
    _write_setup_py(repo, marker, cloud, voice, compromised=False)
    _commit_all(repo, "benign")
    _write_setup_py(repo, marker, cloud, voice, compromised=True)
    moved = _commit_all(repo, "compromised")

    completed = _pip_install(f"git+file://{repo}@main", tmp_path / "site")
    text = _assert_ran(completed, marker, "compromised")
    assert "PLACEHOLDER_NOT_A_SECRET" in text
    assert moved in completed.stdout


def test_commit_id_that_is_also_a_branch_name_follows_the_branch(tmp_path: Path):
    """A 40-hex revision that names a remote branch is checked out as that branch.

    Pip resolves `refs/remotes/origin/<rev>` before it treats `<rev>` as a
    detached commit. The requirements-file spec asks for a 40-hex id, which is
    what the already-pinned backends use; this test records that a branch
    created under that same name still moves.
    """
    marker, cloud, voice = _fixtures(tmp_path)
    repo = tmp_path / "upstream"
    _init_repo(repo)
    _write_setup_py(repo, marker, cloud, voice, compromised=False)
    pinned = _commit_all(repo, "benign")
    _git(repo, "branch", pinned)
    _write_setup_py(repo, marker, cloud, voice, compromised=True)
    moved = _commit_all(repo, "compromised")
    _git(repo, "branch", "-f", pinned, moved)
    assert looks_like_hash(pinned)

    completed = _pip_install(f"git+file://{repo}@{pinned}", tmp_path / "site")
    text = _assert_ran(completed, marker, "compromised")
    assert moved in completed.stdout
    assert pinned not in text
