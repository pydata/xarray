#!/usr/bin/env python3
# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Argparse conveniences for executing common types of benchmark runs."""

from abc import ABC, abstractmethod
import argparse
from datetime import datetime
from importlib import import_module
from os import environ
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import NamedTemporaryFile
from textwrap import dedent
from typing import Literal, Protocol

# The threshold beyond which shifts are 'notable'. See `asv compare`` docs
#  for more.
COMPARE_FACTOR = 1.2

BENCHMARKS_DIR = Path(__file__).parent
ROOT_DIR = BENCHMARKS_DIR.parent
# Storage location for reports used in GitHub actions.
GH_REPORT_DIR = ROOT_DIR.joinpath(".github", "workflows", "benchmark_reports")

# Common ASV arguments for all run_types except `custom`.
ASV_HARNESS = "run {posargs} --attribute rounds=3 --interleave-rounds --show-stderr"


def echo(echo_string: str):
    # Use subprocess for printing to reduce chance of printing out of sequence
    #  with the subsequent calls.
    subprocess.run(["echo", f"BM_RUNNER DEBUG: {echo_string}"])


def _subprocess_runner(args, asv=False, **kwargs):
    # Avoid permanent modifications if the same arguments are used more than once.
    args = args.copy()
    kwargs = kwargs.copy()
    if asv:
        args.insert(0, "asv")
        kwargs["cwd"] = BENCHMARKS_DIR
    echo(" ".join(args))
    kwargs.setdefault("check", True)
    return subprocess.run(args, **kwargs)


def _subprocess_runner_capture(args, **kwargs) -> str:
    result = _subprocess_runner(args, capture_output=True, **kwargs)
    return result.stdout.decode().rstrip()


def _check_requirements(package: str) -> None:
    try:
        import_module(package)
    except ImportError as exc:
        message = (
            f"No {package} install detected. Benchmarks can only "
            f"be run in an environment including {package}."
        )
        raise Exception(message) from exc


def _prep_data_gen_env() -> None:
    """Create or access a separate, unchanging environment for generating test data."""
    python_version = "3.13"
    data_gen_var = "DATA_GEN_PYTHON"
    if data_gen_var in environ:
        echo("Using existing data generation environment.")
    else:
        echo("Setting up the data generation environment ...")
        # Get Nox to build an environment for the `tests` session, but don't
        #  run the session. Will reuse a cached environment if appropriate.
        _subprocess_runner(
            [
                "nox",
                f"--noxfile={ROOT_DIR / 'noxfile.py'}",
                "--session=tests",
                "--install-only",
                f"--python={python_version}",
            ]
        )
        # Find the environment built above, set it to be the data generation
        #  environment.
        env_directory: Path = next((ROOT_DIR / ".nox").rglob(f"tests*"))
        data_gen_python = (env_directory / "bin" / "python").resolve()
        environ[data_gen_var] = str(data_gen_python)

        def clone_resource(name: str, clone_source: str) -> Path:
            resource_dir = data_gen_python.parents[1] / "resources"
            resource_dir.mkdir(exist_ok=True)
            clone_dir = resource_dir / name
            if not clone_dir.is_dir():
                _subprocess_runner(["git", "clone", clone_source, str(clone_dir)])
            return clone_dir

        echo("Installing Mule into data generation environment ...")
        mule_dir = clone_resource("mule", "https://github.com/metomi/mule.git")
        _subprocess_runner(
            [
                str(data_gen_python),
                "-m",
                "pip",
                "install",
                str(mule_dir / "mule"),
            ]
        )

        test_data_var = "OVERRIDE_TEST_DATA_REPOSITORY"
        if test_data_var not in environ:
            echo("Installing iris-test-data into data generation environment ...")
            test_data_dir = clone_resource(
                "iris-test-data", "https://github.com/SciTools/iris-test-data.git"
            )
            environ[test_data_var] = str(test_data_dir / "test_data")

        echo("Data generation environment ready.")


def _setup_common() -> None:
    _check_requirements("asv")
    _check_requirements("nox")

    _prep_data_gen_env()

    echo("Setting up ASV ...")
    _subprocess_runner(["machine", "--yes"], asv=True)

    echo("Setup complete.")


def _asv_compare(
    *commits: str,
    overnight_mode: bool = False,
    fail_on_regression: bool = False,
) -> None:
    """Run through a list of commits comparing each one to the next."""
    commits = tuple(commit[:8] for commit in commits)

    machine_script = [
        "from asv.machine import Machine",
        "print(Machine.get_unique_machine_name())",
    ]
    machine_name = _subprocess_runner_capture(
        ["python", "-c", ";".join(machine_script)]
    )

    for i in range(len(commits) - 1):
        before = commits[i]
        after = commits[i + 1]
        asv_command = shlex.split(
            f"compare {before} {after} "
            f"--machine {machine_name} --factor={COMPARE_FACTOR} --split"
        )

        comparison = _subprocess_runner_capture(asv_command, asv=True)
        echo(comparison)
        shifts = _subprocess_runner_capture([*asv_command, "--only-changed"], asv=True)

        if shifts or (not overnight_mode):
            # For the overnight run: only post if there are shifts.
            _gh_create_reports(after, comparison, shifts)

        if shifts and fail_on_regression:
            # fail_on_regression supports setups that expect CI failures.
            message = (
                f"Performance shifts detected between commits {before} and {after}.\n"
            )
            raise RuntimeError(message)


def _gh_create_reports(commit_sha: str, results_full: str, results_shifts: str) -> None:
    """If running under GitHub Actions: record the results in report(s).

    Posting the reports is done by :func:`_gh_post_reports`, which must be run
    within a separate action to comply with GHA's security limitations.
    """
    if "GITHUB_ACTIONS" not in environ:
        # Only run when within GHA.
        return

    pr_number = environ.get("PR_NUMBER", None)
    on_pull_request = pr_number is not None
    run_id = environ["GITHUB_RUN_ID"]
    repo = environ["GITHUB_REPOSITORY"]
    gha_run_link = f"[`{run_id}`](https://github.com/{repo}/actions/runs/{run_id})"

    GH_REPORT_DIR.mkdir(exist_ok=True)
    commit_dir = GH_REPORT_DIR / commit_sha
    commit_dir.mkdir()
    command_path = commit_dir / "command.txt"
    body_path = commit_dir / "body.txt"

    performance_report = dedent(
        (
            """
            # :stopwatch: Performance Benchmark Report: {commit_sha}

            <details>
            <summary>Performance shifts</summary>

            ```
            {results_shifts}
            ```

            </details>

            <details>
            <summary>Full benchmark results</summary>

            ```
            {results_full}
            ```

            </details>

            Generated by GHA run {gha_run_link}
            """
        )
    )
    performance_report = performance_report.format(
        commit_sha=commit_sha,
        results_shifts=results_shifts,
        results_full=results_full,
        gha_run_link=gha_run_link,
    )

    if on_pull_request:
        # Command to post the report as a comment on the active PR.
        body_path.write_text(performance_report)
        command = (
            f"gh pr comment {pr_number} "
            f"--body-file {body_path.absolute()} "
            f"--repo {repo}"
        )
        command_path.write_text(command)

    else:
        # Command to post the report as new issue.
        commit_msg = _subprocess_runner_capture(
            f"git log {commit_sha}^! --oneline".split(" ")
        )
        # Intended for benchmarking commits on trunk - should include a PR
        #  number due to our squash policy.
        pr_tag_match = re.search("#[0-9]*", commit_msg)

        assignee = ""
        pr_tag = "pull request number unavailable"
        if pr_tag_match is not None:
            pr_tag = pr_tag_match.group(0)

            for login_type in ("author", "mergedBy"):
                gh_query = f'.["{login_type}"]["login"]'
                commandlist = shlex.split(
                    f"gh pr view {pr_tag[1:]} "
                    f"--json {login_type} -q '{gh_query}' "
                    f"--repo {repo}"
                )
                login = _subprocess_runner_capture(commandlist)

                commandlist = [
                    "curl",
                    "-s",
                    f"https://api.github.com/users/{login}",
                ]
                login_info = _subprocess_runner_capture(commandlist)
                is_user = '"type": "User"' in login_info
                if is_user:
                    assignee = login
                    break

        title = f"Performance Shift(s): `{commit_sha}`"
        body = dedent(
            (
                f"""
                Benchmark comparison has identified performance shifts at:

                * commit {commit_sha} ({pr_tag}).

                <p>
                Please review the report below and
                take corrective/congratulatory action as appropriate
                :slightly_smiling_face:
                </p>
                """
            )
        )
        body += performance_report
        body_path.write_text(body)

        command = (
            "gh issue create "
            f'--title "{title}" '
            f"--body-file {body_path.absolute()} "
            '--label "Bot" '
            '--label "Type: Performance" '
            f"--repo {repo}"
        )
        if assignee:
            command += f" --assignee {assignee}"
        command_path.write_text(command)


def _gh_post_reports() -> None:
    """If running under GitHub Actions: post pre-prepared benchmark reports.

    Reports are prepared by :func:`_gh_create_reports`, which must be run
    within a separate action to comply with GHA's security limitations.
    """
    if "GITHUB_ACTIONS" not in environ:
        # Only run when within GHA.
        return

    commit_dirs = [x for x in GH_REPORT_DIR.iterdir() if x.is_dir()]
    for commit_dir in commit_dirs:
        command_path = commit_dir / "command.txt"
        command = command_path.read_text()

        # Security: only accept certain commands to run.
        assert command.startswith(("gh issue create", "gh pr comment"))

        _subprocess_runner(shlex.split(command))


class _SubParserGenerator(ABC):
    """Convenience for holding all the necessary argparse info in 1 place."""

    name: str = NotImplemented
    description: str = NotImplemented
    epilog: str = NotImplemented

    class _SubParsersType(Protocol):
        """Duck typing since argparse._SubParsersAction is private."""

        def add_parser(self, name, **kwargs) -> argparse.ArgumentParser: ...

    def __init__(self, subparsers: _SubParsersType) -> None:
        self.subparser = subparsers.add_parser(
            self.name,
            description=self.description,
            epilog=self.epilog,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self.add_arguments()
        self.add_asv_arguments()
        self.subparser.set_defaults(func=self.func)

    @abstractmethod
    def add_arguments(self) -> None:
        """All custom self.subparser.add_argument() calls."""
        _ = NotImplemented

    def add_asv_arguments(self) -> None:
        self.subparser.add_argument(
            "asv_args",
            nargs=argparse.REMAINDER,
            help="Any number of arguments to pass down to the ASV benchmark command.",
        )

    @staticmethod
    @abstractmethod
    def func(args: argparse.Namespace):
        """Return when the subparser is parsed.

        `func` is then called, performing the user's selected sub-command.

        """
        _ = args
        return NotImplemented


class Overnight(_SubParserGenerator):
    name = "overnight"
    description = (
        "Benchmarks all commits between the input **first_commit** to ``HEAD``, "
        "comparing each to its parent for performance shifts. If running on "
        "GitHub Actions: performance shift(s) will be reported in a new issue.\n"
        "Designed for checking the previous 24 hours' commits, typically in a "
        "scheduled script.\n"
        "Uses `asv run`."
    )
    epilog = (
        "e.g. python bm_runner.py overnight a1b23d4\n"
        "e.g. python bm_runner.py overnight a1b23d4 --bench=regridding"
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "first_commit",
            type=str,
            help="The first commit in the benchmarking commit sequence.",
        )

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _setup_common()

        commit_range = f"{args.first_commit}^^.."
        # git rev-list --first-parent is the command ASV uses.
        git_command = shlex.split(f"git rev-list --first-parent {commit_range}")
        commit_string = _subprocess_runner_capture(git_command)
        commit_list = commit_string.split("\n")

        asv_command = shlex.split(ASV_HARNESS.format(posargs=commit_range))
        try:
            _subprocess_runner([*asv_command, *args.asv_args], asv=True)
        finally:
            # Designed for long running - want to compare/post any valid
            #  results even if some are broken.
            _asv_compare(*reversed(commit_list), overnight_mode=True)


class Branch(_SubParserGenerator):
    name = "branch"
    description = (
        "Performs the same operations as ``overnight``, but always on two "
        "commits only - ``HEAD``, and ``HEAD``'s merge-base with the input "
        "**base_branch**.\n"
        "If running on GitHub Actions: HEAD will be GitHub's "
        "merge commit and merge-base will be the merge target. Performance "
        "comparisons will be posted in a comment on the relevant pull request.\n"
        "Designed for testing if the active branch's changes cause performance "
        "shifts - anticipating what would be caught by ``overnight`` once "
        "merged.\n\n"
        "**For maximum accuracy, avoid using the machine that is running this "
        "session. Run time could be >1 hour for the full benchmark suite.**\n"
        "Uses `asv run`."
    )
    epilog = (
        "e.g. python bm_runner.py branch upstream/main\n"
        "e.g. python bm_runner.py branch upstream/main --bench=regridding"
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "base_branch",
            type=str,
            help="A branch that has the merge-base with ``HEAD`` - ``HEAD`` will be benchmarked against that merge-base.",
        )

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _setup_common()

        git_command = shlex.split("git rev-parse HEAD")
        head_sha = _subprocess_runner_capture(git_command)[:8]

        git_command = shlex.split(f"git merge-base {head_sha} {args.base_branch}")
        merge_base = _subprocess_runner_capture(git_command)[:8]

        with NamedTemporaryFile("w") as hashfile:
            hashfile.writelines([merge_base, "\n", head_sha])
            hashfile.flush()
            commit_range = f"HASHFILE:{hashfile.name}"
            asv_command = shlex.split(ASV_HARNESS.format(posargs=commit_range))
            _subprocess_runner([*asv_command, *args.asv_args], asv=True)

        _asv_compare(merge_base, head_sha)


class _CSPerf(_SubParserGenerator, ABC):
    """Common code used by both CPerf and SPerf."""

    description = (
        "Run the on-demand {} suite of benchmarks (part of the UK Met "
        "Office NG-VAT project) for the ``HEAD`` of ``upstream/main`` only, "
        "and publish the results to the input **publish_dir**, within a "
        "unique subdirectory for this run.\n"
        "Uses `asv run`."
    )
    epilog = (
        "e.g. python bm_runner.py {0} my_publish_dir\n"
        "e.g. python bm_runner.py {0} my_publish_dir --bench=regridding"
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "publish_dir",
            type=str,
            help="HTML results will be published to a sub-dir in this dir.",
        )

    @staticmethod
    def csperf(args: argparse.Namespace, run_type: Literal["cperf", "sperf"]) -> None:
        _setup_common()

        publish_dir = Path(args.publish_dir)
        if not publish_dir.is_dir():
            message = f"Input 'publish directory' is not a directory: {publish_dir}"
            raise NotADirectoryError(message)
        publish_subdir = (
            publish_dir / f"{run_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        publish_subdir.mkdir()

        # Activate on demand benchmarks (C/SPerf are deactivated for
        #  'standard' runs).
        environ["ON_DEMAND_BENCHMARKS"] = "True"
        commit_range = "upstream/main^!"

        asv_command_str = (
            ASV_HARNESS.format(posargs=commit_range) + f" --bench={run_type}"
        )

        # Only do a single round.
        asv_command = shlex.split(re.sub(r"rounds=\d", "rounds=1", asv_command_str))
        try:
            _subprocess_runner([*asv_command, *args.asv_args], asv=True)
        except subprocess.CalledProcessError as err:
            # C/SPerf benchmarks are much bigger than the CI ones:
            # Don't fail the whole run if memory blows on 1 benchmark.
            # ASV produces return code of 2 if the run includes crashes.
            if err.returncode != 2:
                raise

        asv_command = shlex.split(f"publish {commit_range} --html-dir={publish_subdir}")
        _subprocess_runner(asv_command, asv=True)

        # Print completion message.
        location = BENCHMARKS_DIR / ".asv"
        echo(
            f'New ASV results for "{run_type}".\n'
            f'See "{publish_subdir}",'
            f'\n  or JSON files under "{location / "results"}".'
        )


class CPerf(_CSPerf):
    name = "cperf"
    description = _CSPerf.description.format("CPerf")
    epilog = _CSPerf.epilog.format("cperf")

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _CSPerf.csperf(args, "cperf")


class SPerf(_CSPerf):
    name = "sperf"
    description = _CSPerf.description.format("SPerf")
    epilog = _CSPerf.epilog.format("sperf")

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _CSPerf.csperf(args, "sperf")


class Custom(_SubParserGenerator):
    name = "custom"
    description = (
        "Run ASV with the input **ASV sub-command**, without any preset "
        "arguments - must all be supplied by the user. So just like running "
        "ASV manually, with the convenience of re-using the runner's "
        "scripted setup steps."
    )
    epilog = "e.g. python bm_runner.py custom continuous a1b23d4 HEAD --quick"

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "asv_sub_command",
            type=str,
            help="The ASV command to run.",
        )

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _setup_common()
        _subprocess_runner([args.asv_sub_command, *args.asv_args], asv=True)


class TrialRun(_SubParserGenerator):
    name = "trialrun"
    description = (
        "Fast trial-run a given benchmark, to check it works : "
        "in a provided or latest-lockfile environment, "
        "with no repeats for accuracy of measurement."
    )
    epilog = (
        "e.g. python bm_runner.py trialrun "
        "MyBenchmarks.time_calc ${DATA_GEN_PYTHON}"
        "\n\nNOTE: 'runpath' also replaces $DATA_GEN_PYTHON during the run."
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "benchmark",
            type=str,
            help=(
                "A benchmark name, possibly including wildcards, "
                "as supported by the ASV '--bench' argument."
            ),
        )
        self.subparser.add_argument(
            "runpath",
            type=str,
            help=(
                "A path to an existing python executable, "
                "to completely bypass environment building."
            ),
        )

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        if args.runpath:
            # Shortcut creation of a data-gen environment
            # - which is also the trial-run env.
            python_path = Path(args.runpath).resolve()
            environ["DATA_GEN_PYTHON"] = str(python_path)
        _setup_common()
        # get path of data-gen environment, setup by previous call
        python_path = Path(environ["DATA_GEN_PYTHON"])
        # allow 'on-demand' benchmarks
        environ["ON_DEMAND_BENCHMARKS"] = "1"
        asv_command = [
            "run",
            "--bench",
            args.benchmark,
            # no repeats for timing accuracy
            "--quick",
            "--show-stderr",
            # do not build a unique env : run test in data-gen environment
            "--environment",
            f"existing:{python_path}",
        ] + args.asv_args
        _subprocess_runner(asv_command, asv=True)


class Validate(_SubParserGenerator):
    name = "validate"
    description = (
        "Quickly check that the benchmark architecture works as intended with "
        "the current codebase. Things that are checked: env creation/update, "
        "package build/install/uninstall, artificial data creation."
    )
    epilog = "Sole acceptable syntax: python bm_runner.py validate"

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _setup_common()

        git_command = shlex.split("git rev-parse HEAD")
        head_sha = _subprocess_runner_capture(git_command)[:8]

        # Find the most recent commit where the lock-files are not
        #  identical to HEAD - will force environment updates.
        locks_dir = Path(__file__).parents[1] / "ci" / "requirements" / "locks"
        assert locks_dir.is_dir()
        git_command = shlex.split(
            f"git log -1 --pretty=format:%P -- {locks_dir.resolve()}"
        )
        locks_sha = _subprocess_runner_capture(git_command)[:8]

        with NamedTemporaryFile("w") as hashfile:
            hashfile.writelines([locks_sha, "\n", head_sha])
            hashfile.flush()
            asv_command = shlex.split(
                f"run HASHFILE:{hashfile.name} --bench ValidateSetup "
                "--attribute rounds=1 --show-stderr"
            )
            extra_env = environ | {"ON_DEMAND_BENCHMARKS": "1"}
            _subprocess_runner(asv_command, asv=True, env=extra_env)

    # No arguments permitted for this subclass:

    def add_arguments(self) -> None:
        pass

    def add_asv_arguments(self) -> None:
        pass


class GhPost(_SubParserGenerator):
    name = "_gh_post"
    description = (
        "Used by GitHub Actions to post benchmark reports that were prepared "
        "during previous actions. Separated to comply with GitHub's security "
        "requirements."
    )
    epilog = "Sole acceptable syntax: python bm_runner.py _gh_post"

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _gh_post_reports()

    # No arguments permitted for this subclass:

    def add_arguments(self) -> None:
        pass

    def add_asv_arguments(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the repository performance benchmarks (using Airspeed Velocity)."
        ),
        epilog=(
            "More help is available within each sub-command."
            "\n\nNOTE(1): a separate python environment is created to "
            "construct test files.\n   Set $DATA_GEN_PYTHON to avoid the cost "
            "of this."
            "\nNOTE(2): iris-test-data is downloaded and cached within the "
            "data generation environment.\n   Set "
            "$OVERRIDE_TEST_DATA_REPOSITORY to avoid the cost of this."
            "\nNOTE(3): test data is cached within the "
            "benchmarks code directory, and uses a lot of disk space "
            "of disk space (Gb).\n   Set $BENCHMARK_DATA to specify where this "
            "space can be safely allocated."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(required=True)

    parser_generators: tuple[type[_SubParserGenerator], ...] = (
        Overnight,
        Branch,
        CPerf,
        SPerf,
        Custom,
        TrialRun,
        Validate,
        GhPost,
    )

    for gen in parser_generators:
        _ = gen(subparsers).subparser

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
