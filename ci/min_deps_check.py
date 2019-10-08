"""Fetch from conda database all available versions of the xarray dependencies and their
publication date. Compare it against requirements/py36-min-all-deps.yml to verify the
policy on obsolete dependencies is being followed. Print a pretty report :)
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, Iterator, Tuple

import yaml

IGNORE_DEPS = {
    "black",
    "coveralls",
    "flake8",
    "hypothesis",
    "mypy",
    "pip",
    "pytest",
    "pytest-cov",
    "pytest-env",
}

POLICY_MONTHS = {"python": 42, "numpy": 24, "pandas": 12, "scipy": 12}
POLICY_MONTHS_DEFAULT = 6

has_errors = False


def error(msg: str) -> None:
    global has_errors
    has_errors = True
    print("ERROR:", msg)


def parse_requirements(fname) -> Iterator[Tuple[str, int, int]]:
    """Load requirements/py36-min-all-deps.yml

    Yield (package name, major version, minor version)
    """
    global has_errors

    with open(fname) as fh:
        contents = yaml.safe_load(fh)
    for row in contents["dependencies"]:
        if isinstance(row, dict) and list(row) == ["pip"]:
            continue
        pkg, eq, version = row.partition("=")
        if pkg.rstrip("<>") in IGNORE_DEPS:
            continue
        if pkg.endswith("<") or pkg.endswith(">") or eq != "=":
            error("package should be pinned with exact version: " + row)
            continue
        try:
            major, minor = version.split(".")
        except ValueError:
            error("expected major.minor (without patch): " + row)
            continue
        try:
            yield pkg, int(major), int(minor)
        except ValueError:
            error("failed to parse version: " + row)


def query_conda(pkg: str) -> Dict[Tuple[int, int], datetime]:
    """Query the conda repository for a specific package

    Return map of {(major version, minor version): publication date}
    """
    stdout = subprocess.check_output(
        ["conda", "search", pkg, "--info", "-c", "defaults", "-c", "conda-forge"]
    )
    out = {}  # type: Dict[Tuple[int, int], datetime]
    major = None
    minor = None

    for row in stdout.decode("utf-8").splitlines():
        label, _, value = row.partition(":")
        label = label.strip()
        if label == "file name":
            value = value.strip()[len(pkg) :]
            major, minor = value.split("-")[1].split(".")[:2]
            major = int(major)
            minor = int(minor)
        if label == "timestamp":
            assert major is not None
            assert minor is not None
            ts = datetime.strptime(value.split()[0].strip(), "%Y-%m-%d")

            if (major, minor) in out:
                out[major, minor] = min(out[major, minor], ts)
            else:
                out[major, minor] = ts

    # Hardcoded fix to work around incorrect dates in conda
    if pkg == "python":
        out.update(
            {
                (2, 7): datetime(2010, 6, 3),
                (3, 5): datetime(2015, 9, 13),
                (3, 6): datetime(2016, 12, 23),
                (3, 7): datetime(2018, 6, 27),
                (3, 8): datetime(2019, 10, 14),
            }
        )

    return out


def process_pkg(
    pkg: str, req_major: int, req_minor: int
) -> Tuple[str, int, int, str, int, int, str, str]:
    """Compare package version from requirements file to available versions in conda.
    Return row to build pandas dataframe:

    - package name
    - major version in requirements file
    - minor version in requirements file
    - publication date of version in requirements file (YYYY-MM-DD)
    - major version suggested by policy
    - minor version suggested by policy
    - publication date of version suggested by policy (YYYY-MM-DD)
    - status ("<", "=", "> (!)")
    """
    print("Analyzing %s..." % pkg)
    versions = query_conda(pkg)

    try:
        req_published = versions[req_major, req_minor]
    except KeyError:
        error("not found in conda: " + pkg)
        return pkg, req_major, req_minor, "-", 0, 0, "-", "(!)"

    policy_months = POLICY_MONTHS.get(pkg, POLICY_MONTHS_DEFAULT)
    policy_published = datetime.now() - timedelta(days=policy_months * 30)

    policy_major = req_major
    policy_minor = req_minor
    policy_published_actual = req_published
    for (major, minor), published in reversed(sorted(versions.items())):
        if published < policy_published:
            break
        policy_major = major
        policy_minor = minor
        policy_published_actual = published

    if (req_major, req_minor) < (policy_major, policy_minor):
        status = "<"
    elif (req_major, req_minor) > (policy_major, policy_minor):
        status = "> (!)"
        error("Package is too new: " + pkg)
    else:
        status = "="

    return (
        pkg,
        req_major,
        req_minor,
        req_published.strftime("%Y-%m-%d"),
        policy_major,
        policy_minor,
        policy_published_actual.strftime("%Y-%m-%d"),
        status,
    )


def main() -> None:
    fname = sys.argv[1]
    with ThreadPoolExecutor(8) as ex:
        futures = [
            ex.submit(process_pkg, pkg, major, minor)
            for pkg, major, minor in parse_requirements(fname)
        ]
        rows = [f.result() for f in futures]

    print("Package       Required          Policy            Status")
    print("------------- ----------------- ----------------- ------")
    fmt = "{:13} {:>1d}.{:<2d} ({:10}) {:>1d}.{:<2d} ({:10}) {}"
    for row in rows:
        print(fmt.format(*row))

    assert not has_errors


if __name__ == "__main__":
    main()
