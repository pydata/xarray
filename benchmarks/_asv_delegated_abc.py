# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""ASV plug-in providing an alternative :class:`asv.environments.Environment` subclass.

Preps an environment via custom user scripts, then uses that as the
benchmarking environment.

This module is intended as the generic code that can be shared between
repositories. Providing a functional benchmarking environment relies on correct
subclassing of the :class:`_DelegatedABC` class to specialise it for the repo in
question. The parent and subclass are separated into their own dedicated files,
which isolates ALL repo-specific code to a single file, thus simplifying the
templating process.

"""

from abc import ABC, abstractmethod
from contextlib import contextmanager, suppress
from os import environ
from pathlib import Path
import sys

from asv.console import log
from asv.environment import Environment, EnvironmentUnavailable
from asv.repo import Repo


class _DelegatedABC(Environment, ABC):
    """Manage a benchmark environment using custom user scripts, run at each commit.

    Ignores user input variations - ``matrix`` / ``pythons`` /
    ``exclude``, since environment is being managed outside ASV.

    A vanilla :class:`asv.environment.Environment` is created for containing
    the expected ASV configuration files and checked-out project. The actual
    'functional' environment is created/updated using
    :meth:`_prep_env_override`, then the location is recorded via
    a symlink within the ASV environment. The symlink is used as the
    environment path used for any executable calls (e.g.
    ``python my_script.py``).

    Intended as the generic parent class that can be shared between
    repositories. Providing a functional benchmarking environment relies on
    correct subclassing of this class to specialise it for the repo in question.

    Warnings
    --------
    :class:`_DelegatedABC` is an abstract base class. It MUST ONLY be used via
    subclasses implementing their own :meth:`_prep_env_override`, and also
    :attr:`tool_name`, which must be unique.

    """

    tool_name = "delegated-ABC"
    """Required by ASV as a unique identifier of the environment type."""

    DELEGATED_LINK_NAME = "delegated_env"
    """The name of the symlink to the delegated environment."""

    COMMIT_ENVS_VAR = "ASV_COMMIT_ENVS"
    """Env var that instructs a dedicated environment be created per commit."""

    def __init__(self, conf, python, requirements, tagged_env_vars):
        """Get a 'delegated' environment based on the given ASV config object.

        Parameters
        ----------
        conf : dict
            ASV configuration object.

        python : str
            Ignored - environment management is delegated. The value is always
            ``DELEGATED``.

        requirements : dict (str -> str)
            Ignored - environment management is delegated. The value is always
            an empty dict.

        tagged_env_vars : dict (tag, key) -> value
            Ignored - environment management is delegated. The value is always
            an empty dict.

        Raises
        ------
        EnvironmentUnavailable
            The original environment or delegated environment cannot be created.

        """
        ignored = []
        if python:
            ignored.append(f"{python=}")
        if requirements:
            ignored.append(f"{requirements=}")
        if tagged_env_vars:
            ignored.append(f"{tagged_env_vars=}")
        message = (
            f"Ignoring ASV setting(s): {', '.join(ignored)}. Benchmark "
            "environment management is delegated to third party script(s)."
        )
        log.warning(message)
        self._python = "DELEGATED"
        self._requirements = {}
        self._tagged_env_vars = {}
        super().__init__(
            conf,
            self._python,
            self._requirements,
            self._tagged_env_vars,
        )

        self._path_undelegated = Path(self._path)
        """Preserves the 'true' path of the environment so that self._path can
        be safely modified and restored."""

    @property
    def _path_delegated(self) -> Path:
        """The path of the symlink to the delegated environment."""
        return self._path_undelegated / self.DELEGATED_LINK_NAME

    @property
    def _delegated_found(self) -> bool:
        """Whether self._path_delegated successfully resolves to a directory."""
        resolved = None
        with suppress(FileNotFoundError):
            resolved = self._path_delegated.resolve(strict=True)
        result = resolved is not None and resolved.is_dir()
        return result

    def _symlink_to_delegated(self, delegated_env_path: Path) -> None:
        """Create the symlink to the delegated environment."""
        self._path_delegated.unlink(missing_ok=True)
        self._path_delegated.parent.mkdir(parents=True, exist_ok=True)
        self._path_delegated.symlink_to(delegated_env_path, target_is_directory=True)
        assert self._delegated_found

    def _setup(self):
        """Temporarily try to set the user's active env as the delegated env.

        Environment prep will be run anyway once ASV starts checking out
        commits, but this step tries to provide a usable environment (with
        python, etc.) at the moment that ASV expects it.

        """
        current_env = Path(sys.executable).parents[1]
        message = (
            "Temporarily using user's active environment as benchmarking "
            f"environment: {current_env} . "
        )
        try:
            self._symlink_to_delegated(current_env)
            _ = self.find_executable("python")
        except Exception:
            message = (
                f"Delegated environment {self.name} not yet set up (unable to "
                "determine current environment)."
            )
            self._path_delegated.unlink(missing_ok=True)

        message += "Correct environment will be set up at the first commit checkout."
        log.warning(message)

    @abstractmethod
    def _prep_env_override(self, env_parent_dir: Path) -> Path:
        """Run aspects of :meth:`_prep_env` that vary between repos.

        This is the method that is expected to do the preparing
        (:meth:`_prep_env` only performs pre- and post- steps). MUST be
        overridden in any subclass environments before they will work.

        Parameters
        ----------
        env_parent_dir : Path
            The directory that the prepared environment should be placed in.

        Returns
        -------
        Path
            The path to the prepared environment.
        """
        pass

    def _prep_env(self, commit_hash: str) -> None:
        """Prepare the delegated environment for the given commit hash."""
        message = (
            f"Running delegated environment management for: {self.name} "
            f"at commit: {commit_hash[:8]}"
        )
        log.info(message)

        env_parent = Path(self._env_dir).resolve()
        new_env_per_commit = self.COMMIT_ENVS_VAR in environ
        if new_env_per_commit:
            env_parent = env_parent / commit_hash[:8]

        delegated_env_path = self._prep_env_override(env_parent)
        assert delegated_env_path.is_relative_to(env_parent)

        # Record the environment's path via a symlink within this environment.
        self._symlink_to_delegated(delegated_env_path)

        message = f"Environment {self.name} updated to spec at {commit_hash[:8]}"
        log.info(message)

    def checkout_project(self, repo: Repo, commit_hash: str) -> None:
        """Check out the working tree of the project at given commit hash."""
        super().checkout_project(repo, commit_hash)
        self._prep_env(commit_hash)

    @contextmanager
    def _delegate_path(self):
        """Context manager to use the delegated env path as this env's path."""
        if not self._delegated_found:
            message = f"Delegated environment not found at: {self._path_delegated}"
            log.error(message)
            raise EnvironmentUnavailable(message)

        try:
            self._path = str(self._path_delegated)
            yield
        finally:
            self._path = str(self._path_undelegated)

    def find_executable(self, executable):
        """Find an executable (e.g. python, pip) in the DELEGATED environment.

        Raises
        ------
        OSError
            If the executable is not found in the environment.
        """
        if not self._delegated_found:
            # Required during environment setup. OSError expected if executable
            #  not found.
            raise OSError

        with self._delegate_path():
            return super().find_executable(executable)

    def run_executable(self, executable, args, **kwargs):
        """Run a given executable (e.g. python, pip) in the DELEGATED environment."""
        with self._delegate_path():
            return super().run_executable(executable, args, **kwargs)

    def run(self, args, **kwargs):
        # This is not a specialisation - just implementing the abstract method.
        log.debug(f"Running '{' '.join(args)}' in {self.name}")
        return self.run_executable("python", args, **kwargs)
