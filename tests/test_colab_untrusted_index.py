"""Colab notebooks must not run code published under an untrusted index name.

``colab/fish-speech-s2-pro.ipynb`` and ``colab/llasa-8b.ipynb`` ask the user to
upload a reference recording and then import a project that the same notebook
installed with ``pip install <name>`` and no version, hash, or direct URL.
These tests stand in for that index with a local PEP 503 directory: each
untrusted name is published twice, and the release pip selects copies every
file in the working directory (the uploaded recording) into a sink.

They fail while that install-and-import path is still in the notebook. They
pass once the notebook no longer requests the project from a package index.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

REPO = Path(__file__).resolve().parent.parent

# Index project names these notebooks install and then import. They are not
# the upstream projects the notebooks are demonstrating:
# fish-speech on PyPI is owned by the user account quocvu (PyPI JSON
# "ownership" field) while Fish Audio installs from a source checkout;
# llasa is not published on PyPI at all (the project URL returns 404), so
# the first person to register the name supplies the code the notebook runs.
UNTRUSTED_INDEX_PROJECTS = frozenset({"fish-speech", "llasa"})

NOTEBOOKS = (
    "colab/fish-speech-s2-pro.ipynb",
    "colab/llasa-8b.ipynb",
)

# Notebook shell lines that are not the install under test. They do not
# carry the package bytes into the interpreter, so the tests do not run them.
SKIPPED_SHELL_PREFIXES = ("nvidia-smi", "huggingface-cli")

_PIP_INSTALL = re.compile(
    r"^(?:python\d?(?:\.\d+)?\s+-m\s+)?pip3?\s+install\b"
)
_TAKES_VALUE = frozenset(
    {
        "-i",
        "--index-url",
        "--extra-index-url",
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-e",
        "--editable",
        "-t",
        "--target",
        "-f",
        "--find-links",
        "--log",
        "--proxy",
        "--cert",
        "--client-cert",
        "--prefix",
        "--root",
        "--src",
        "--upgrade-strategy",
        "--config-settings",
        "--python-version",
        "--implementation",
        "--abi",
        "--platform",
        "--group",
        "--report",
    }
)
_REFERENCE_NAME = "reference.wav"
_REFERENCE_BYTES = b"REFERENCE-VOICE-CANARY\n"


def _cell_source(cell: dict) -> str:
    source = cell.get("source") or ""
    if isinstance(source, list):
        return "".join(source)
    return source


def _pip_commands(source: str) -> list[str]:
    commands = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("!"):
            stripped = stripped[1:].strip()
        if _PIP_INSTALL.match(stripped):
            commands.append(stripped)
    return commands


def _requirement_tokens(command: str) -> list[str]:
    parts = shlex.split(command)
    try:
        start = parts.index("install") + 1
    except ValueError:
        return []
    tokens: list[str] = []
    pending_value = False
    for token in parts[start:]:
        if pending_value:
            pending_value = False
            continue
        if token in _TAKES_VALUE:
            pending_value = True
            continue
        if token.startswith("-"):
            continue
        tokens.append(token)
    return tokens


def _index_requirements(command: str) -> list[Requirement]:
    found = []
    for token in _requirement_tokens(command):
        try:
            req = Requirement(token)
        except InvalidRequirement:
            continue
        if req.url is not None:
            continue
        if canonicalize_name(req.name) in UNTRUSTED_INDEX_PROJECTS:
            found.append(req)
    return found


def _module_for(req: Requirement, notebook_source: str) -> str:
    """Import root the notebook uses for this distribution, if it has one."""
    normalized = canonicalize_name(req.name).replace("-", "_")
    pattern = re.compile(
        r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", re.MULTILINE
    )
    for match in pattern.finditer(notebook_source):
        root = match.group(1).split(".", 1)[0]
        if root.replace("-", "_").lower() == normalized:
            return root
    return normalized


def _wheel_name(project: str) -> str:
    return canonicalize_name(project).replace("-", "_")


def _build_wheel(
    dest_dir: Path, project: str, version: str, package: str, init_source: str
) -> Path:
    dist = _wheel_name(project)
    root = dest_dir / f"_src_{dist}_{version}"
    pkg_dir = root / package
    info = root / f"{dist}-{version}.dist-info"
    pkg_dir.mkdir(parents=True)
    info.mkdir()
    (pkg_dir / "__init__.py").write_text(init_source)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {project}\nVersion: {version}\n"
    )
    (info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: afterwords-test\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n"
    )
    files = [pkg_dir / "__init__.py", info / "METADATA", info / "WHEEL"]
    record = []
    for path in files:
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        rel = path.relative_to(root).as_posix()
        record.append(f"{rel},sha256={digest.decode()},{len(data)}")
    record.append(f"{dist}-{version}.dist-info/RECORD,,")
    (info / "RECORD").write_text("\n".join(record) + "\n")
    wheel = dest_dir / f"{dist}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path in files + [info / "RECORD"]:
            archive.write(path, path.relative_to(root).as_posix())
    return wheel


def _payload(sink: Path, version: str) -> str:
    # Runs when the notebook executes ``from <package> import ...``.
    # Copies the working directory, which is where the upload cell wrote
    # the reference recording, to a directory outside that session.
    return (
        "from pathlib import Path\n"
        f"sink = Path({str(sink)!r})\n"
        "sink.mkdir(parents=True, exist_ok=True)\n"
        f"(sink / 'installed-version').write_text({version!r})\n"
        "for child in Path.cwd().iterdir():\n"
        "    if child.is_file():\n"
        "        (sink / child.name).write_bytes(child.read_bytes())\n"
    )


def _publish_index(
    index_root: Path, releases: list[tuple[str, str, str, str]]
) -> None:
    """Write a PEP 503 directory index.

    ``releases`` entries are (project, version, import package, init source).
    Pip requests ``<index>/<canonical-name>/`` and reads ``index.html``
    anchor targets as the files to install.
    """
    by_project: dict[str, list[Path]] = {}
    for project, version, package, init_source in releases:
        project_dir = index_root / canonicalize_name(project)
        project_dir.mkdir(parents=True, exist_ok=True)
        wheel = _build_wheel(project_dir, project, version, package, init_source)
        by_project.setdefault(project, []).append(wheel)
    for project, wheels in by_project.items():
        project_dir = index_root / canonicalize_name(project)
        links = [
            f'<a href="{wheel.name}">{wheel.name}</a>'
            for wheel in sorted(wheels, key=lambda path: path.name)
        ]
        (project_dir / "index.html").write_text(
            "<!DOCTYPE html><html><body>\n"
            + "\n".join(links)
            + "\n</body></html>\n"
        )


def _write_upload_stub(stub_root: Path, reference_file: Path) -> None:
    """Kernel side of google.colab.files.upload.

    The browser widget is not available here. ``_upload_files`` returns the
    reference bytes the test supplies; the write loop is the one in
    google.colab.files.upload, which writes those bytes into the kernel
    working directory and returns the saved filenames.
    """
    package = stub_root / "google" / "colab"
    package.mkdir(parents=True)
    (stub_root / "google" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "files.py").write_text(
        "import os\n"
        "\n"
        "def _get_unique_filename(filename):\n"
        "    if not os.path.lexists(filename):\n"
        "        return filename\n"
        "    counter = 1\n"
        "    while True:\n"
        "        path, ext = os.path.splitext(filename)\n"
        "        new_filename = '{} ({}){}'.format(path, counter, ext)\n"
        "        if not os.path.lexists(new_filename):\n"
        "            return new_filename\n"
        "        counter += 1\n"
        "\n"
        "def _upload_files(multiple):\n"
        "    name = os.environ['AFTERWORDS_REFERENCE_NAME']\n"
        "    with open(os.environ['AFTERWORDS_REFERENCE_FILE'], 'rb') as handle:\n"
        "        return {name: handle.read()}\n"
        "\n"
        "def upload(target_dir=''):\n"
        "    uploaded_files = _upload_files(multiple=True)\n"
        "    local_filenames = {}\n"
        "    uploaded_files_with_updated_names = {}\n"
        "    if target_dir:\n"
        "        os.makedirs(target_dir, exist_ok=True)\n"
        "    for filename, data in uploaded_files.items():\n"
        "        complete_filename = os.path.join(target_dir, filename)\n"
        "        local_filename = local_filenames.get(complete_filename)\n"
        "        if not local_filename:\n"
        "            local_filename = _get_unique_filename(complete_filename)\n"
        "            local_filenames[complete_filename] = local_filename\n"
        "        with open(local_filename, 'ab') as handle:\n"
        "            handle.write(data)\n"
        "        uploaded_files_with_updated_names[local_filename] = data\n"
        "    return uploaded_files_with_updated_names\n"
    )


def _base_env(**extra: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONNOUSERSITE": "1",
    }
    env.update(extra)
    return env


def _run(argv: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _notebooks_under_test() -> list[Path]:
    found = [REPO / relative for relative in NOTEBOOKS]
    known = {path.resolve() for path in found}
    for path in sorted((REPO / "colab").glob("*.ipynb")):
        if path.resolve() in known:
            continue
        source = path.read_text()
        for command in _pip_commands(source):
            if _index_requirements(command):
                found.append(path)
                break
    return found


def _exercise(notebook_path: Path, work: Path) -> list[str]:
    problems: list[str] = []
    notebook = json.loads(notebook_path.read_text())
    cells = [
        _cell_source(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    full_source = "\n".join(cells)
    commands = [command for cell in cells for command in _pip_commands(cell)]
    index_reqs: list[Requirement] = []
    companion_reqs: list[Requirement] = []
    for command in commands:
        for token in _requirement_tokens(command):
            try:
                req = Requirement(token)
            except InvalidRequirement:
                continue
            if req.url is not None:
                continue
            if canonicalize_name(req.name) in UNTRUSTED_INDEX_PROJECTS:
                index_reqs.append(req)
            else:
                companion_reqs.append(req)
        if re.search(r"(^|\s)(-i|--index-url|--extra-index-url)\b", command):
            problems.append(
                f"{notebook_path.name} install command sets its own index URL "
                f"and was not executed: {command}"
            )
            commands = [item for item in commands if item != command]

    session = work / "session"
    target = work / "target"
    sink = work / "sink"
    index_root = work / "simple"
    session.mkdir()
    target.mkdir()
    sink.mkdir()
    reference = work / "reference-source.wav"
    reference.write_bytes(_REFERENCE_BYTES)

    if index_reqs:
        releases: list[tuple[str, str, str, str]] = []
        seen: set[str] = set()
        for req in index_reqs:
            name = canonicalize_name(req.name)
            if name in seen:
                continue
            seen.add(name)
            package = _module_for(req, full_source)
            # An older release and a newer one. An unpinned install has an
            # empty specifier, so the resolver's sort key selects the greater
            # version. Both releases read the working directory, because the
            # account that owns the name controls every version.
            for version in ("0.1.0", "0.1.1"):
                releases.append((name, version, package, _payload(sink, version)))
        for req in companion_reqs:
            name = canonicalize_name(req.name)
            if name in seen:
                continue
            seen.add(name)
            releases.append((name, "0.0.0", _module_for(req, full_source), ""))
        _publish_index(index_root, releases)
        config = work / "pip.conf"
        config.write_text("")
        shell = os.environ.get("SHELL") or "/bin/sh"
        install_env = _base_env(
            PIP_INDEX_URL=index_root.as_uri() + "/",
            PIP_CONFIG_FILE=str(config),
            PIP_TARGET=str(target),
            PIP_NO_DEPS="1",
        )
        for command in commands:
            if re.search(r"(^|\s)(-i|--index-url|--extra-index-url)\b", command):
                continue
            if "://" in command:
                problems.append(
                    f"{notebook_path.name} install command contains a URL and "
                    f"was not executed: {command}"
                )
                continue
            completed = _run([shell, "-c", command], session, install_env)
            if completed.returncode != 0:
                problems.append(
                    f"{notebook_path.name} command {command!r} exited "
                    f"{completed.returncode}\n{completed.stdout}\n{completed.stderr}"
                )

    stub = work / "stub"
    _write_upload_stub(stub, reference)
    upload_env = _base_env(
        PYTHONPATH=str(stub),
        AFTERWORDS_REFERENCE_NAME=_REFERENCE_NAME,
        AFTERWORDS_REFERENCE_FILE=str(reference),
    )
    import_env = _base_env(PYTHONPATH=str(target))
    upload_ran = False
    for source in cells:
        stripped = source.strip()
        if not stripped:
            continue
        body = stripped[1:].strip() if stripped.startswith("!") else stripped
        if _pip_commands(source):
            continue
        if any(body.startswith(prefix) for prefix in SKIPPED_SHELL_PREFIXES):
            continue
        if stripped.startswith("!"):
            problems.append(
                f"{notebook_path.name} has a shell cell that these tests do not "
                f"run: {body}"
            )
            continue
        env = upload_env if "files.upload" in source else import_env
        if "files.upload" in source:
            upload_ran = True
        _run([sys.executable, "-c", source], session, env)

    if index_reqs and not upload_ran:
        problems.append(
            f"{notebook_path.name} installs {sorted(canonicalize_name(req.name) for req in index_reqs)} "
            "from the package index and has no files.upload() cell to place "
            "the reference recording"
        )
    uploaded = session / _REFERENCE_NAME
    if upload_ran and not uploaded.is_file():
        problems.append(
            f"{notebook_path.name} upload cell did not write {_REFERENCE_NAME}"
        )
    copied = sink / _REFERENCE_NAME
    if copied.is_file() and copied.read_bytes() == _REFERENCE_BYTES:
        version = (sink / "installed-version").read_text() if (
            sink / "installed-version"
        ).is_file() else "unknown"
        names = sorted({canonicalize_name(req.name) for req in index_reqs})
        problems.append(
            f"{notebook_path.name} installed {names} from the index "
            f"(version {version} was imported) and that import copied the "
            f"uploaded reference recording out of the session directory"
        )
    if index_reqs:
        rendered = ", ".join(
            f"{canonicalize_name(req.name)} ({req})" for req in index_reqs
        )
        problems.append(
            f"{notebook_path.name} still requests an untrusted project from "
            f"the package index, with no direct URL: {rendered}"
        )
    return problems


@pytest.mark.parametrize("notebook", [REPO / relative for relative in NOTEBOOKS])
def test_notebook_does_not_run_untrusted_index_package(notebook: Path, tmp_path: Path):
    assert notebook.is_file(), f"missing notebook {notebook}"
    problems = _exercise(notebook, tmp_path)
    assert not problems, "\n".join(problems)


def test_no_other_colab_notebook_installs_the_same_projects(tmp_path: Path):
    extras = [
        path
        for path in _notebooks_under_test()
        if path.name not in {Path(item).name for item in NOTEBOOKS}
    ]
    problems: list[str] = []
    for path in extras:
        problems.extend(_exercise(path, tmp_path / path.stem))
    assert not problems, "\n".join(problems)
