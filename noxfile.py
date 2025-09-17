"""Perform test automation with nox.

For further details, see https://nox.thea.codes/en/stable/#

"""

import hashlib
import os
from pathlib import Path

import nox
from nox.logger import logger

#: Default to reusing any pre-existing nox environments.
nox.options.reuse_existing_virtualenvs = True

#: Python versions we can run sessions under
_PY_VERSIONS_ALL = ["3.11", "3.12", "3.13"]
_PY_VERSION_LATEST = _PY_VERSIONS_ALL[-1]

#: One specific python version for docs builds
_PY_VERSION_DOCSBUILD = _PY_VERSION_LATEST

#: Cirrus-CI environment variable hook.
PY_VER = os.environ.get("PY_VER", _PY_VERSIONS_ALL)

#: Default cartopy cache directory.
CARTOPY_CACHE_DIR = os.environ.get("HOME") / Path(".local/share/cartopy")

# https://github.com/numpy/numpy/pull/19478
# https://github.com/matplotlib/matplotlib/pull/22099
#: Common session environment variables.
ENV = dict(NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX")


def session_lockfile(session: nox.sessions.Session) -> Path:
    """Return the path of the session lockfile."""
    # return Path(f"ci/requirements/locks/py{session.python.replace('.', '')}-linux-64.lock")
    return Path(f"ci/requirements/locks/environment-benchmark-linux-64.lock")


def session_cachefile(session: nox.sessions.Session) -> Path:
    """Return the path of the session lockfile cache."""
    lockfile = session_lockfile(session)
    tmp_dir = Path(session.create_tmp())
    cache = tmp_dir / lockfile.name
    return cache


def venv_populated(session: nox.sessions.Session) -> bool:
    """List of packages in the lockfile installed.

    Returns True if the conda venv has been created.
    """
    return session_cachefile(session).is_file()


def venv_changed(session: nox.sessions.Session) -> bool:
    """Return True if the installed session is different.

    Compares to that specified in the lockfile.
    """
    changed = False
    cache = session_cachefile(session)
    lockfile = session_lockfile(session)
    if cache.is_file():
        with open(lockfile, "rb") as fi:
            expected = hashlib.sha256(fi.read()).hexdigest()
        with open(cache, "r") as fi:
            actual = fi.read()
        changed = actual != expected
    return changed


def cache_venv(session: nox.sessions.Session) -> None:
    """Cache the nox session environment.

    This consists of saving a hexdigest (sha256) of the associated
    conda lock file.

    Parameters
    ----------
    session : object
        A `nox.sessions.Session` object.

    """
    lockfile = session_lockfile(session)
    cache = session_cachefile(session)
    with open(lockfile, "rb") as fi:
        hexdigest = hashlib.sha256(fi.read()).hexdigest()
    with open(cache, "w") as fout:
        fout.write(hexdigest)


def cache_cartopy(session: nox.sessions.Session) -> None:
    """Determine whether to cache the cartopy natural earth shapefiles.

    Parameters
    ----------
    session : object
        A `nox.sessions.Session` object.

    """
    if not CARTOPY_CACHE_DIR.is_dir():
        session.run_always(
            "python",
            "-c",
            "import cartopy; cartopy.io.shapereader.natural_earth()",
        )


def prepare_venv(session: nox.sessions.Session) -> None:
    """Create and cache the nox session conda environment.

    Additionally provide conda environment package details and info.

    Note that, iris is installed into the environment using pip.

    Parameters
    ----------
    session : object
        A `nox.sessions.Session` object.

    Notes
    -----
    See
      - https://github.com/theacodes/nox/issues/346
      - https://github.com/theacodes/nox/issues/260

    """
    lockfile = session_lockfile(session)
    print(f"prepare_venv: {lockfile}")
    venv_dir = session.virtualenv.location_name

    if not venv_populated(session):
        # environment has been created but packages not yet installed
        # populate the environment from the lockfile
        logger.debug(f"Populating conda env at {venv_dir}")
        session.conda_install("--file", str(lockfile))
        cache_venv(session)

    elif venv_changed(session):
        # destroy the environment and rebuild it
        logger.debug(f"Lockfile changed. Re-creating conda env at {venv_dir}")
        _re_orig = session.virtualenv.reuse_existing
        session.virtualenv.reuse_existing = False
        session.virtualenv.create()
        session.conda_install("--file", str(lockfile))
        session.virtualenv.reuse_existing = _re_orig
        cache_venv(session)

    logger.debug(f"Environment {venv_dir} is up to date")

    # cache_cartopy(session)

    # Determine whether verbose diagnostics have been requested
    # from the command line.
    verbose = "-v" in session.posargs or "--verbose" in session.posargs

    if verbose:
        session.run_always("conda", "info")
        session.run_always("conda", "list", f"--prefix={venv_dir}")
        session.run_always(
            "conda",
            "list",
            f"--prefix={venv_dir}",
            "--explicit",
        )


@nox.session(python=PY_VER, venv_backend="conda")
def tests(session: nox.sessions.Session):
    """Perform iris system, integration and unit tests.

    Coverage testing is enabled if the "--coverage" or "-c" flag is used.

    Parameters
    ----------
    session : object
        A `nox.sessions.Session` object.

    """
    prepare_venv(session)
    session.install("--no-deps", "--editable", ".")
    session.env.update(ENV)
    run_args = [
        "pytest",
        "-n",
        "auto",
        "lib/iris/tests",
    ]
    if "-c" in session.posargs or "--coverage" in session.posargs:
        run_args[-1:-1] = ["--cov=lib/iris", "--cov-report=xml"]
    session.run(*run_args)


@nox.session(python=_PY_VERSION_DOCSBUILD, venv_backend="conda")
def doctest(session: nox.sessions.Session):
    """Perform iris doctests and gallery.

    Parameters
    ----------
    session : object
        A `nox.sessions.Session` object.

    """
    prepare_venv(session)
    session.install("--no-deps", "--editable", ".")
    session.env.update(ENV)
    session.cd("docs")
    session.run(
        "make",
        "clean",
        "html",
        external=True,
    )
    session.run(
        "make",
        "doctest",
        external=True,
    )


@nox.session(python=_PY_VERSION_DOCSBUILD, venv_backend="conda")
def gallery(session: nox.sessions.Session):
    """Perform iris gallery doc-tests.

    Parameters
    ----------
    session : object
        A `nox.sessions.Session` object.

    """
    prepare_venv(session)
    session.install("--no-deps", "--editable", ".")
    session.env.update(ENV)
    session.run(
        "pytest",
        "-n",
        "auto",
        "docs/gallery_tests",
    )


@nox.session(python=PY_VER, venv_backend="conda")
def wheel(session: nox.sessions.Session):
    """Perform iris local wheel install and import test.

    Parameters
    ----------
    session : object
        A `nox.sessions.Session` object.

    """
    prepare_venv(session)
    session.cd("dist")
    fname = list(Path(".").glob("scitools_iris-*.whl"))
    if len(fname) == 0:
        raise ValueError("Cannot find wheel to install.")
    if len(fname) > 1:
        emsg = f"Expected to find 1 wheel to install, found {len(fname)} instead."
        raise ValueError(emsg)
    session.install(fname[0].name)
    session.run(
        "python",
        "-c",
        "import iris; print(f'{iris.__version__=}')",
        external=True,
    )


@nox.session
def benchmarks(session: nox.sessions.Session):
    """Run the Iris benchmark runner. Run session with `-- --help` for help.

    Parameters
    ----------
    session : object
        A `nox.sessions.Session` object.

    """
    if len(session.posargs) == 0:
        message = (
            "This session MUST be run with at least one argument. The "
            "arguments are passed down to the benchmark runner script. E.g:\n"
            "nox -s benchmarks -- --help\n"
            "nox -s benchmarks -- something --help\n"
            "nox -s benchmarks -- something\n"
        )
        session.error(message)
    session.install("asv", "nox")
    bm_runner_path = Path(__file__).parent / "benchmarks" / "bm_runner.py"
    session.run("python", bm_runner_path, *session.posargs)
