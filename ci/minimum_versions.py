import asyncio
import bisect
import datetime
import pathlib
import sys
from dataclasses import dataclass, field

import rich_click as click
import yaml
from dateutil.relativedelta import relativedelta
from rattler import Gateway, Version
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.table import Column, Table
from tlz.functoolz import curry, pipe
from tlz.itertoolz import concat, groupby

click.rich_click.SHOW_ARGUMENTS = True

channels = ["conda-forge"]
platforms = ["noarch", "linux-64"]
ignored_packages = [
    "coveralls",
    "pip",
    "pytest",
    "pytest-cov",
    "pytest-env",
    "pytest-xdist",
    "pytest-timeout",
    "hypothesis",
]


@dataclass
class Policy:
    package_months: dict
    default_months: int
    overrides: dict[str, Version] = field(default_factory=dict)

    def minimum_version(self, package_name, releases):
        if (override := self.overrides.get(package_name)) is not None:
            return override

        policy_months = self.package_months.get(package_name, self.default_months)
        today = datetime.date.today()

        cutoff_date = today - relativedelta(months=policy_months)

        index = bisect.bisect_left(
            releases, cutoff_date, key=lambda x: x.timestamp.date()
        )
        return releases[index - 1 if index > 0 else 0]


@dataclass
class Spec:
    name: str
    version: Version | None

    @classmethod
    def parse(cls, spec_text):
        warnings = []
        if ">" in spec_text or "<" in spec_text:
            warnings.append(
                f"package should be pinned with an exact version: {spec_text!r}"
            )

            spec_text = spec_text.replace(">", "").replace("<", "")

        if "=" in spec_text:
            name, version_text = spec_text.split("=", maxsplit=1)
            version = Version(version_text)
            segments = version.segments()

            if len(segments) != 2 or (len(segments) == 3 and segments[2] != 0):
                warnings.append(
                    f"package should be pinned to a minor version (got {version})"
                )
        else:
            name = spec_text
            version = None

        return cls(name, version), (name, warnings)


@dataclass(order=True)
class Release:
    version: Version
    build_number: int
    timestamp: datetime.datetime = field(compare=False)

    @classmethod
    def from_repodata_record(cls, repo_data):
        return cls(
            version=repo_data.version,
            build_number=repo_data.build_number,
            timestamp=repo_data.timestamp,
        )


def parse_environment(text):
    env = yaml.safe_load(text)

    specs = []
    warnings = []
    for dep in env["dependencies"]:
        spec, warnings_ = Spec.parse(dep)

        warnings.append(warnings_)
        specs.append(spec)

    return specs, warnings


def is_preview(version):
    candidates = ["rc", "beta", "alpha"]

    *_, last_segment = version.segments()
    return any(candidate in last_segment for candidate in candidates)


def group_packages(records):
    groups = groupby(lambda r: r.name.normalized, records)
    return {
        name: sorted(map(Release.from_repodata_record, group))
        for name, group in groups.items()
    }


def filter_releases(predicate, releases):
    return {
        name: [r for r in records if predicate(r)] for name, records in releases.items()
    }


def deduplicate_releases(package_info):
    def deduplicate(releases):
        return min(releases, key=lambda p: p.timestamp)

    return {
        name: list(map(deduplicate, groupby(lambda p: p.version, group).values()))
        for name, group in package_info.items()
    }


def find_policy_versions(policy, releases):
    return {
        name: policy.minimum_version(name, package_releases)
        for name, package_releases in releases.items()
    }


def is_suitable_release(release):
    if release.timestamp is None:
        return False

    segments = release.version.extend_to_length(3).segments()

    return segments[2] == [0]


def lookup_spec_release(spec, releases):
    version = spec.version.extend_to_length(3)

    return releases[spec.name][version]


def compare_versions(environments, policy_versions):
    status = {}
    for env, specs in environments.items():
        env_status = any(
            spec.version > policy_versions[spec.name].version for spec in specs
        )
        status[env] = env_status
    return status


def version_comparison_symbol(required, policy):
    if required < policy:
        return "<"
    elif required > policy:
        return ">"
    else:
        return "="


def format_bump_table(specs, policy_versions, releases, warnings):
    table = Table(
        Column("Package", width=20),
        Column("Required", width=8),
        "Required (date)",
        Column("Policy", width=8),
        "Policy (date)",
        "Status",
    )

    heading_style = Style(color="#ff0000", bold=True)
    warning_style = Style(color="#ffff00", bold=True)
    styles = {
        ">": Style(color="#ff0000", bold=True),
        "=": Style(color="#008700", bold=True),
        "<": Style(color="#d78700", bold=True),
    }

    for spec in specs:
        policy_release = policy_versions[spec.name]
        policy_version = policy_release.version.with_segments(0, 2)
        policy_date = policy_release.timestamp

        required_version = spec.version
        required_date = lookup_spec_release(spec, releases).timestamp

        status = version_comparison_symbol(required_version, policy_version)
        style = styles[status]

        table.add_row(
            spec.name,
            str(required_version),
            f"{required_date:%Y-%m-%d}",
            str(policy_version),
            f"{policy_date:%Y-%m-%d}",
            status,
            style=style,
        )

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(style=heading_style, vertical="middle")
    grid.add_column()
    grid.add_row("Version summary", table)

    if any(warnings.values()):
        warning_table = Table(width=table.width, expand=True)
        warning_table.add_column("Package")
        warning_table.add_column("Warning")

        for package, messages in warnings.items():
            if not messages:
                continue
            warning_table.add_row(package, messages[0], style=warning_style)
            for message in messages[1:]:
                warning_table.add_row("", message, style=warning_style)

        grid.add_row("Warnings", warning_table)

    return grid


@click.command()
@click.argument(
    "environment_paths",
    type=click.Path(exists=True, readable=True, path_type=pathlib.Path),
    nargs=-1,
)
def main(environment_paths):
    console = Console()

    parsed_environments = {
        path.stem: parse_environment(path.read_text()) for path in environment_paths
    }

    warnings = {
        env: dict(warnings_) for env, (_, warnings_) in parsed_environments.items()
    }
    environments = {
        env: [spec for spec in specs if spec.name not in ignored_packages]
        for env, (specs, _) in parsed_environments.items()
    }

    all_packages = list(
        dict.fromkeys(spec.name for spec in concat(environments.values()))
    )

    policy_months = {
        "python": 30,
        "numpy": 18,
    }
    policy_months_default = 12
    overrides = {}

    policy = Policy(
        policy_months, default_months=policy_months_default, overrides=overrides
    )

    gateway = Gateway()
    query = gateway.query(channels, platforms, all_packages, recursive=False)
    records = asyncio.run(query)

    package_releases = pipe(
        records,
        concat,
        group_packages,
        curry(filter_releases, lambda r: r.timestamp is not None),
        deduplicate_releases,
    )
    policy_versions = pipe(
        package_releases,
        curry(filter_releases, is_suitable_release),
        curry(find_policy_versions, policy),
    )
    status = compare_versions(environments, policy_versions)

    release_lookup = {
        n: {r.version: r for r in releases} for n, releases in package_releases.items()
    }
    grids = {
        env: format_bump_table(specs, policy_versions, release_lookup, warnings[env])
        for env, specs in environments.items()
    }
    root_grid = Table.grid()
    root_grid.add_column()

    for env, grid in grids.items():
        root_grid.add_row(Panel(grid, title=env, expand=True))

    console.print(root_grid)

    status_code = 1 if any(status.values()) else 0
    sys.exit(status_code)


if __name__ == "__main__":
    main()
