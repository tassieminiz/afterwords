"""Spec: Colab notebooks must not install or import the PyPI project ``llasa``.

``colab/llasa-8b.ipynb`` tells a user to run ``pip install llasa`` (no version,
hash, or index) and then ``from llasa import generate``, passing the uploaded
reference recording into that function. Nothing by that name is published on
PyPI, so the first person to register it supplies the code ``pip`` builds and
Python imports.

These tests fail while that install or import is still in a notebook under
``colab/``. They pass once every notebook stops requesting project ``llasa``
from a package index and stops importing that module. A version pin or hash
does not make the name safe: the registrant chooses the artifact.

The install test serves a local index only as a stand-in for PyPI after the
name is registered. ``pip`` itself downloads the sdist, executes ``setup.py``,
and, in the import test, installs the package. The stand-in does not implement
pip.

Out of scope here: ``colab/fish-speech-s2-pro.ipynb`` installs ``fish-speech``,
which is a registered PyPI project, so it is not this unregistered-name bug.
Research-doc ``pip install`` lines for other backends name registered projects
or a requirements file. Imports built by concatenating strings at runtime are
not detected.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import textwrap
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = REPO_ROOT / "colab"
HOSTILE_PROJECT = "llasa"
REFERENCE_BYTES = b"REFERENCE-VOICE-BYTES\n"
ATTACKER_AUDIO = b"ATTACKER-AUDIO"
PYPI_HOSTS = {
    "pypi.org",
    "pypi.python.org",
    "files.pythonhosted.org",
    "test.pypi.org",
}

# Options that consume the following argument. Unknown options fail closed
# so a new flag cannot hide a requirement.
_VALUE_OPTIONS = {
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
    "--prefix",
    "--root",
    "-f",
    "--find-links",
    "--hash",
    "--config-settings",
    "-C",
    "--install-option",
    "--global-option",
    "--src",
    "--upgrade-strategy",
    "--implementation",
    "--abi",
    "--platform",
    "--python-version",
    "--only-binary",
    "--no-binary",
    "--exists-action",
    "--log",
    "--proxy",
    "--cert",
    "--client-cert",
    "--retries",
    "--timeout",
    "--cache-dir",
    "--trusted-host",
    "-b",
    "--build",
    "--progress-bar",
    "--report",
    "--dry-run",
    "--group",
    "--config-settings",
}
_FLAG_OPTIONS = {
    "-q",
    "--quiet",
    "-U",
    "--upgrade",
    "--no-index",
    "--pre",
    "--no-deps",
    "--user",
    "--ignore-installed",
    "--no-cache-dir",
    "--disable-pip-version-check",
    "--no-build-isolation",
    "--use-pep517",
    "--no-use-pep517",
    "-v",
    "--verbose",
    "--force-reinstall",
    "--compile",
    "--no-compile",
    "--prefer-binary",
    "--no-warn-conflicts",
    "--break-system-packages",
    "--require-hashes",
    "--no-clean",
    "--use-feature",
}


def _canonicalize(name: str) -> str:
    # Same rule pip 24 uses (pip/_vendor/packaging/utils.py).
    return re.sub(r"[-_.]+", "-", name).lower()


def _project_name(token: str) -> str | None:
    """Return the canonical project name of an index requirement token.

    Direct URLs, VCS URLs, and filesystem paths are not index lookups.
    """
    token = token.strip()
    if not token or token.startswith("-"):
        return None
    lowered = token.lower()
    if "://" in token or lowered.startswith(
        ("git+", "hg+", "svn+", "bzr+", "file:")
    ):
        return None
    if token.startswith(("/", ".")):
        return None
    if "@" in token and " " in token.split("@", 1)[0]:
        name = token.split("@", 1)[0].strip()
    elif re.match(r"^[A-Za-z0-9_.-]+\s*@", token):
        name = token.split("@", 1)[0].strip()
    else:
        name = re.split(r"\[|@|<|>|=|!|~|;", token, maxsplit=1)[0].strip()
    if not name or not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", name):
        return None
    return _canonicalize(name)


def _shell_commands(source: str) -> list[str]:
    """Shell lines IPython would hand to the system shell, plus ``%pip``.

    IPython 8.32 turns a leading ``!`` into ``get_ipython().system(rest)``,
    and ``system`` runs ``sh -c`` on that rest. ``%pip install ...`` is the
    magic form of the same install.
    """
    commands: list[str] = []
    pending = ""
    for raw in source.splitlines():
        stripped = raw.strip()
        if pending:
            piece = stripped[:-1].rstrip() if stripped.endswith("\\") else stripped
            pending = f"{pending} {piece}".strip()
            if not stripped.endswith("\\"):
                commands.append(pending)
                pending = ""
            continue
        if not stripped or stripped.startswith("#"):
            continue
        continued = stripped.endswith("\\")
        body = stripped[:-1].rstrip() if continued else stripped
        if body.startswith("!"):
            body = body[1:].lstrip()
        elif body.startswith("%pip"):
            body = "pip" + body[len("%pip") :]
        elif not re.match(r"^(python[0-9.]*\s+-m\s+)?pip3?\s+install\b", body):
            continue
        if continued:
            pending = body
        else:
            commands.append(body)
    if pending:
        commands.append(pending)
    return commands


class _ParsedInstall:
    def __init__(self, argv: list[str], requirements: list[str], index_urls: list[str], no_index: bool):
        self.argv = argv
        self.requirements = requirements
        self.index_urls = index_urls
        self.no_index = no_index

    @property
    def uses_pypi(self) -> bool:
        if self.no_index:
            return False
        if not self.index_urls:
            return True
        for url in self.index_urls:
            host = urlparse(url).hostname or ""
            if host.lower() in PYPI_HOSTS:
                return True
        return False


def _parse_pip_install(command: str) -> _ParsedInstall | None:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        pytest.fail(f"could not parse install command {command!r}: {exc}")
    if not argv:
        return None
    # Drop a leading interpreter so `python -m pip install` and `pip install` match.
    if Path(argv[0]).name.startswith("python") and len(argv) >= 3 and argv[1] == "-m":
        argv = argv[2:]
    if not argv or Path(argv[0]).name not in {"pip", "pip3"}:
        return None
    if len(argv) < 2 or argv[1] != "install":
        return None
    requirements: list[str] = []
    index_urls: list[str] = []
    no_index = False
    tokens = argv[2:]
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "--":
            requirements.extend(tokens[i + 1 :])
            break
        if token.startswith("-"):
            name, eq, inline = token.partition("=")
            option = name
            if option not in _VALUE_OPTIONS and option not in _FLAG_OPTIONS:
                pytest.fail(
                    f"unrecognized pip option {option!r} in {command!r}; "
                    "refusing to ignore it"
                )
            value = inline if eq else None
            if option in _VALUE_OPTIONS and value is None:
                if i + 1 >= len(tokens):
                    pytest.fail(f"pip option {option!r} is missing its value in {command!r}")
                i += 1
                value = tokens[i]
            if option in {"-i", "--index-url"} and value:
                index_urls.append(value)
            elif option == "--no-index":
                no_index = True
            i += 1
            continue
        requirements.append(token)
        i += 1
    return _ParsedInstall(argv, requirements, index_urls, no_index)


def _hostile_requirements(command: str) -> list[str]:
    parsed = _parse_pip_install(command)
    if parsed is None or not parsed.uses_pypi:
        return []
    found = []
    for token in parsed.requirements:
        if _project_name(token) == HOSTILE_PROJECT:
            found.append(token)
    return found


def _literal_import_names(source: str) -> set[str]:
    """Module roots imported by static import statements and literal calls."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in {"import_module", "__import__"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.add(arg.value.split(".", 1)[0])
    return found


def _notebook_code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text())
    cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        cells.append(source)
    return cells


def _notebook_install_commands(path: Path) -> list[str]:
    commands = []
    for source in _notebook_code_cells(path):
        commands.extend(_shell_commands(source))
        # Also catch install lines hidden in string literals (os.system, subprocess).
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for line in node.value.splitlines():
                    if re.search(r"(^|[!%]\s*)(python[0-9.]*\s+-m\s+)?pip3?\s+install\b", line.strip()) or line.strip().startswith(
                        "%pip"
                    ):
                        commands.extend(_shell_commands(line))
    return commands


def _colab_notebooks() -> list[Path]:
    return sorted(COLAB_DIR.glob("*.ipynb"))


def _hostile_installs() -> list[tuple[Path, str, str]]:
    hits = []
    for path in _colab_notebooks():
        for command in _notebook_install_commands(path):
            for token in _hostile_requirements(command):
                hits.append((path, command, token))
    return hits


def _llasa_import_cells() -> list[tuple[Path, str]]:
    hits = []
    for path in _colab_notebooks():
        for source in _notebook_code_cells(path):
            if HOSTILE_PROJECT in _literal_import_names(source):
                hits.append((path, source))
    return hits


def _write_sdist(dest: Path) -> None:
    root = dest.parent / "sdist-src"
    if root.exists():
        shutil.rmtree(root)
    pkg = root / "llasa"
    pkg.mkdir(parents=True)
    (root / "setup.py").write_text(
        textwrap.dedent(
            """\
            import os
            from pathlib import Path
            marker = os.environ.get("LLASA_INSTALL_MARKER")
            if marker:
                Path(marker).write_text("setup.py-executed\\n")
            from setuptools import setup
            setup(name="llasa", version="1.0.0", packages=["llasa"])
            """
        )
    )
    (pkg / "__init__.py").write_text(
        textwrap.dedent(
            """\
            import os
            from pathlib import Path

            marker = os.environ.get("LLASA_IMPORT_MARKER")
            if marker:
                Path(marker).write_text("imported\\n")

            def generate(text, ref_audio=None, **kwargs):
                read_marker = os.environ.get("LLASA_READ_MARKER")
                paths = (
                    list(ref_audio)
                    if isinstance(ref_audio, (list, tuple))
                    else ([] if ref_audio is None else [ref_audio])
                )
                chunks = []
                for path in paths:
                    chunks.append(Path(path).read_bytes())
                if read_marker:
                    Path(read_marker).write_bytes(b"\\n".join(chunks))
                return (b"ATTACKER-AUDIO", 16000)
            """
        )
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.add(path, arcname=str(Path("llasa-1.0.0") / path.relative_to(root)))


class _Index:
    def __init__(self, root: Path):
        self.root = root
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> str:
        simple = self.root / "simple" / "llasa"
        packages = self.root / "packages"
        packages.mkdir(parents=True)
        sdist = packages / "llasa-1.0.0.tar.gz"
        _write_sdist(sdist)
        simple.mkdir(parents=True)
        (simple / "index.html").write_text(
            "<!DOCTYPE html><html><body>"
            '<a href="/packages/llasa-1.0.0.tar.gz">llasa-1.0.0.tar.gz</a>'
            "</body></html>\n"
        )
        handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(  # noqa: E731
            *args, directory=str(self.root), **kwargs
        )
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/simple/"

    def __exit__(self, *exc) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _run_pip(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pip", *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _pip_env(marker: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    if marker is not None:
        env["LLASA_INSTALL_MARKER"] = str(marker)
    else:
        env.pop("LLASA_INSTALL_MARKER", None)
    return env


def test_parser_sees_unpinned_llasa_and_ignores_direct_urls():
    command = "!pip install llasa torch torchaudio transformers accelerate"
    assert _shell_commands(command) == [
        "pip install llasa torch torchaudio transformers accelerate"
    ]
    assert _hostile_requirements(_shell_commands(command)[0]) == ["llasa"]
    assert _hostile_requirements("pip install llasa==9.9.9 --hash=sha256:" + "ab" * 32) == [
        "llasa==9.9.9"
    ]
    assert _hostile_requirements(
        "pip install git+https://example.invalid/llasa.git@0123456789abcdef"
    ) == []
    assert _hostile_requirements(
        "pip install --index-url https://example.invalid/simple/ llasa"
    ) == []
    assert HOSTILE_PROJECT in _literal_import_names("from llasa import generate\n")
    assert HOSTILE_PROJECT in _literal_import_names('import importlib\nimportlib.import_module("llasa")\n')
    assert HOSTILE_PROJECT not in _literal_import_names("import torchaudio\n")


def test_notebook_pip_install_does_not_run_index_publisher_code(tmp_path: Path):
    """Running the notebook install line must not execute index-supplied code.

    The command is the notebook's ``pip install`` arguments, in order, with
    the index pointed at the local stand-in. ``pip`` prepares each requirement
    before it finishes the command: for an sdist that preparation executes
    ``setup.py``. A marker written there is the publisher's code running,
    including when a later requirement on the same line cannot be resolved.
    """
    hits = _hostile_installs()
    marker = tmp_path / "install-marker"
    ran = False
    if hits:
        _notebook, command, _token = hits[0]
        parsed = _parse_pip_install(command)
        assert parsed is not None
        with _Index(tmp_path / "index") as index_url:
            result = _run_pip(
                [
                    "install",
                    "--no-cache-dir",
                    "--disable-pip-version-check",
                    "--target",
                    str(tmp_path / "target"),
                    "--index-url",
                    index_url,
                    *parsed.requirements,
                ],
                _pip_env(marker),
            )
        ran = marker.is_file() and "setup.py-executed" in marker.read_text()
        if not ran:
            pytest.fail(
                "notebook still installs project llasa from a package index "
                f"({command!r}) but the stand-in install did not execute "
                f"setup.py (pip exit {result.returncode}).\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    assert not ran, (
        "pip executed the index publisher's setup.py while installing the "
        "notebook's llasa requirement. Remove the unregistered project name "
        f"from {hits[0][0]}. A pin or hash still trusts whoever registers that name."
    )
    assert hits == [], (
        "a Colab notebook still pip-installs project llasa from a package index: "
        + ", ".join(f"{path}: {token}" for path, _command, token in hits)
    )


def test_notebook_import_does_not_hand_reference_audio_to_index_llasa(tmp_path: Path):
    """The notebook's import cell must not run an index-supplied ``llasa``.

    After a successful install of the same requirement the install cell names,
    the upload and generate cells are executed as written. ``files.upload``
    writes the reference and returns ``{path: bytes}``; the notebook passes
    those paths to ``generate``. ``torchaudio.save`` writes that return value
    to ``output.wav``, and ``files.download`` reads that file back.
    """
    imports = _llasa_import_cells()
    imported = False
    read_reference = False
    tampered = False
    if imports:
        index_root = tmp_path / "index"
        target = tmp_path / "target"
        import_marker = tmp_path / "import-marker"
        read_marker = tmp_path / "read-marker"
        reference = tmp_path / "reference-source.bin"
        reference.write_bytes(REFERENCE_BYTES)
        download_copy = tmp_path / "downloaded.bin"
        with _Index(index_root) as index_url:
            result = _run_pip(
                [
                    "install",
                    "--no-cache-dir",
                    "--disable-pip-version-check",
                    "--target",
                    str(target),
                    "--index-url",
                    index_url,
                    "llasa",
                ],
                _pip_env(None),
            )
        if result.returncode != 0:
            pytest.fail(
                "could not install the index-supplied llasa package the notebook imports\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        work = tmp_path / "work"
        work.mkdir()
        stubs = tmp_path / "stubs"
        _write_runtime_stubs(stubs)
        cells = []
        for path in _colab_notebooks():
            if path == imports[0][0] or any(item[0] == path for item in imports):
                for source in _notebook_code_cells(path):
                    if _shell_commands(source) and not _literal_import_names(source):
                        continue
                    if source.strip():
                        cells.append(source)
        runner = tmp_path / "run_cells.py"
        runner.write_text(_CELL_RUNNER)
        env = os.environ.copy()
        env.update(
            {
                "LLASA_IMPORT_MARKER": str(import_marker),
                "LLASA_READ_MARKER": str(read_marker),
                "LLASA_REFERENCE_FILE": str(reference),
                "LLASA_DOWNLOAD_COPY": str(download_copy),
                "PYTHONPATH": os.pathsep.join([str(stubs), str(target)]),
                "PYTHONNOUSERSITE": "1",
            }
        )
        completed = subprocess.run(
            [sys.executable, str(runner)],
            cwd=work,
            env=env,
            input=json.dumps(cells),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            pytest.fail(
                "notebook cells failed while importing llasa\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        imported = import_marker.is_file()
        read_reference = (
            read_marker.is_file() and REFERENCE_BYTES.strip() in read_marker.read_bytes()
        )
        tampered = download_copy.is_file() and ATTACKER_AUDIO in download_copy.read_bytes()
    leaked = []
    if imported:
        leaked.append("import executed the publisher's llasa package")
    if read_reference:
        leaked.append("generate() read the uploaded reference recording")
    if tampered:
        leaked.append(
            "files.download read attacker audio that torchaudio.save wrote from generate()"
        )
    assert not leaked, (
        "running the notebook cells after the index supplied llasa: "
        + "; ".join(leaked)
        + ". Delete the llasa install and `from llasa import generate`."
    )
    assert imports == [], (
        "a Colab notebook still imports llasa: "
        + ", ".join(str(path) for path, _source in imports)
    )


def _write_runtime_stubs(stubs: Path) -> None:
    """Stand-ins for the two libraries the notebook imports beside ``llasa``.

    ``google.colab.files.upload`` writes each upload under the working
    directory and returns ``{local_path: contents}`` (colabtools
    ``google/colab/files.py``). ``files.download`` opens that path and sends
    the bytes. ``torchaudio.save`` writes its ``src`` argument to ``filepath``
    (torchaudio 2.4.1 soundfile backend calls ``soundfile.write(file=filepath,
    data=src, ...)``). These stubs keep that data flow and skip the browser
    widget and the tensor encoder.
    """
    (stubs / "google" / "colab").mkdir(parents=True)
    (stubs / "google" / "__init__.py").write_text("")
    (stubs / "google" / "colab" / "__init__.py").write_text("")
    (stubs / "google" / "colab" / "files.py").write_text(
        textwrap.dedent(
            """\
            import os
            from pathlib import Path

            def upload(target_dir=""):
                data = Path(os.environ["LLASA_REFERENCE_FILE"]).read_bytes()
                if target_dir:
                    os.makedirs(target_dir, exist_ok=True)
                local = os.path.join(target_dir, "reference.wav") if target_dir else "reference.wav"
                with open(local, "wb") as handle:
                    handle.write(data)
                return {local: data}

            def download(filename):
                if not os.path.exists(filename):
                    raise FileNotFoundError(filename)
                payload = Path(filename).read_bytes()
                Path(os.environ["LLASA_DOWNLOAD_COPY"]).write_bytes(payload)
            """
        )
    )
    (stubs / "torchaudio.py").write_text(
        textwrap.dedent(
            """\
            from pathlib import Path

            def save(filepath, src, sample_rate, **kwargs):
                data = src if isinstance(src, (bytes, bytearray)) else str(src).encode()
                Path(filepath).write_bytes(data)
            """
        )
    )


_CELL_RUNNER = textwrap.dedent(
    """\
    import json
    import sys

    work_cells = json.load(sys.stdin)
    namespace = {"__name__": "__main__"}
    for source in work_cells:
        exec(compile(source, "<notebook>", "exec"), namespace)
    """
)


def test_documented_notebook_is_one_of_the_scanned_files():
    """The research summary sends readers to the notebook these tests scan."""
    summary = (REPO_ROOT / "docs" / "research" / "tts-backends" / "00-summary.md").read_text()
    assert "colab/llasa-8b.ipynb" in summary
    assert (COLAB_DIR / "llasa-8b.ipynb") in _colab_notebooks()
