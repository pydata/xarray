#!/usr/bin/env python
"""Check that requires_* tests are exercised in CI.

The CI test matrix should run tests guarded by ``requires_*`` markers in at
least one environment whenever possible. This script inspects pytest report-log
files from the test matrix and fails if a test is only ever skipped for a
``requires`` reason, unless that skip reason is explicitly allowlisted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

_SKIP_PREFIX = "Skipped: "


def iter_reportlog_paths(paths: Iterable[Path]) -> list[Path]:
    reportlogs: list[Path] = []
    for path in paths:
        if path.is_dir():
            reportlogs.extend(sorted(path.rglob("*.jsonl")))
        elif path.suffix == ".jsonl":
            reportlogs.append(path)
    return reportlogs


def load_allowlist(path: Path | None) -> set[str]:
    if path is None:
        return set()

    allowlist = set()
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            allowlist.add(stripped)
    return allowlist


def _skip_reason(longrepr: object) -> str | None:
    if isinstance(longrepr, str):
        reason = longrepr
    elif isinstance(longrepr, list) and len(longrepr) >= 3:
        reason = str(longrepr[2])
    else:
        return None

    if reason.startswith(_SKIP_PREFIX):
        return reason.removeprefix(_SKIP_PREFIX)
    return reason


def collect_reportlog_data(reportlogs: Iterable[Path]) -> dict[str, dict[str, set[str]]]:
    """Return per-nodeid execution and skip information."""
    data: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"call_outcomes": set(), "skip_reasons": set()}
    )

    for reportlog in reportlogs:
        for line in reportlog.read_text().splitlines():
            record = json.loads(line)
            if record.get("$report_type") != "TestReport":
                continue

            nodeid = record.get("nodeid")
            if not nodeid:
                continue

            when = record.get("when")
            outcome = record.get("outcome")
            if when == "call":
                data[nodeid]["call_outcomes"].add(str(outcome))
            elif when == "setup" and outcome == "skipped":
                reason = _skip_reason(record.get("longrepr"))
                if reason is not None:
                    data[nodeid]["skip_reasons"].add(reason)

    return data


def uncovered_requires_tests(
    reportlogs: Iterable[Path],
    allowlist: set[str],
) -> dict[str, set[str]]:
    """Return tests that only ever skipped for a requires reason."""
    uncovered: dict[str, set[str]] = {}
    for nodeid, info in collect_reportlog_data(reportlogs).items():
        if info["call_outcomes"]:
            continue

        skip_reasons = info["skip_reasons"]
        if not skip_reasons:
            continue

        if all(reason.startswith("requires ") for reason in skip_reasons):
            missing_reasons = skip_reasons - allowlist
            if missing_reasons:
                uncovered[nodeid] = missing_reasons

    return uncovered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate that requires_* tests are exercised somewhere in CI."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Report-log files or directories containing report-log files.",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Optional allowlist of skip reasons that are intentionally uncovered in CI.",
    )
    args = parser.parse_args(argv)

    reportlogs = iter_reportlog_paths(args.paths)
    if not reportlogs:
        raise SystemExit("No report-log files were found.")

    uncovered = uncovered_requires_tests(
        reportlogs, allowlist=load_allowlist(args.allowlist)
    )

    if uncovered:
        print("The following tests are only ever skipped for a requires reason:")
        for nodeid, reasons in sorted(uncovered.items()):
            print(f"- {nodeid}: {', '.join(sorted(reasons))}")
        return 1

    print(f"Checked {len(reportlogs)} report-log file(s); no uncovered requires tests found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
