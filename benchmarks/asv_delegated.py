# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Repository-specific adaptation of :mod:`_asv_delegated_abc`."""

import ast
import enum
from os import environ
from os.path import getmtime
from pathlib import Path
import re

from asv import util as asv_util

from _asv_delegated_abc import _DelegatedABC


class Delegated(_DelegatedABC):
    """Specialism of :class:`_DelegatedABC` for benchmarking this repo."""

    tool_name = "delegated"

    def _prep_env_override(self, env_parent_dir: Path) -> Path:
        """Environment preparation specialised for this repo.

        Scans the checked-out commit of Iris to work out the appropriate
        preparation command, including gathering any extra information that said
        command needs.

        Parameters
        ----------
        env_parent_dir : Path
            The directory that the prepared environment should be placed in.

        Returns
        -------
        Path
            The path to the prepared environment.
        """
        # The project checkout.
        build_dir = Path(self._build_root) / self._repo_subdir

        # Older iterations of setup.py are incompatible with setuptools>=80.
        #  (Most dependencies are protected by lock-files, but build
        #   dependencies in pyproject.toml are independent).
        setup_py = build_dir / "setup.py"
        pyproject = build_dir / "pyproject.toml"
        if setup_py.is_file() and "setuptools.command.develop" in setup_py.read_text():
            with pyproject.open("r+") as file_write:
                lines = file_write.readlines()
                for i, line in enumerate(lines):
                    if line == "requires = [\n":
                        next_line = lines[i + 1]
                        indent = next_line[: len(next_line) - len(next_line.lstrip())]

                        lines.insert(i + 1, f'{indent}"setuptools<80",\n')
                        break
                file_write.seek(0)
                file_write.writelines(lines)

        class Mode(enum.Enum):
            """The scenarios where the correct env setup script is known."""

            NOX = enum.auto()
            """``PY_VER=x.xx nox --session=tests --install-only`` is supported."""

        mode = None

        noxfile = build_dir / "noxfile.py"
        if noxfile.is_file():
            # Our noxfile originally did not support `--install-only` - you
            #  could either run the tests, or run nothing at all. Adding
            #  `run_always` to `prepare_venv` enabled environment setup without
            #  running tests.
            noxfile_tree = ast.parse(source=noxfile.read_text())
            prep_session = next(
                filter(
                    lambda node: getattr(node, "name", "") == "prepare_venv",
                    ast.walk(noxfile_tree),
                )
            )
            prep_session_code = ast.unparse(prep_session)
            if (
                "session.run(" not in prep_session_code
                and "session.run_always(" in prep_session_code
            ):
                mode = Mode.NOX

        match mode:
            # Just NOX for now but the architecture is here for future cases.
            case Mode.NOX:
                # Need to determine a single Python version to run with.
                req_dir = build_dir / "ci" / "requirements"
                lockfile_dir = req_dir / "locks"
                if not lockfile_dir.is_dir():
                    lockfile_dir = req_dir / "ci" / "nox.lock"

                if not lockfile_dir.is_dir():
                    message = f"No lockfile directory found in the expected locations, got {lockfile_dir}."
                    raise FileNotFoundError(message)

                def py_ver_from_lockfiles(lockfile: Path) -> str:
                    pattern = re.compile(r"py(\d+)-")
                    search = pattern.search(lockfile.name
                    assert search is not None
                    version = search.group(1)
                    return f"{version[0]}.{version[1:]}"

                python_versions = [
                    py_ver_from_lockfiles(lockfile)
                    for lockfile in lockfile_dir.glob("*.lock")
                ]
                python_version = max(python_versions)

                # Construct and run the environment preparation command.
                local_envs = dict(environ)
                local_envs["PY_VER"] = python_version
                # Prevent Nox re-using env with wrong Python version.
                env_parent_dir = (
                    env_parent_dir / f"nox{python_version.replace('.', '')}"
                )
                env_command = [
                    "nox",
                    f"--envdir={env_parent_dir}",
                    "--session=tests",
                    "--install-only",
                    "--no-error-on-external-run",
                    "--verbose",
                ]
                _ = asv_util.check_output(
                    env_command,
                    timeout=self._install_timeout,
                    cwd=build_dir,
                    env=local_envs,
                )

                env_parent_contents = list(env_parent_dir.iterdir())
                if len(env_parent_contents) != 1:
                    message = (
                        f"{env_parent_dir} contains {len(env_parent_contents)} "
                        "items, expected 1. Cannot determine the environment "
                        "directory."
                    )
                    raise FileNotFoundError(message)
                (delegated_env_path,) = env_parent_contents

            case _:
                message = "No environment setup is known for this commit of Iris."
                raise NotImplementedError(message)

        return delegated_env_path
