from __future__ import annotations

import json
from pathlib import Path

from ci.check_requires_coverage import uncovered_requires_tests


def _write_reportlog(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def test_uncovered_requires_test_is_reported(tmp_path: Path) -> None:
    reportlog = _write_reportlog(
        tmp_path / "reportlog.jsonl",
        [
            {"$report_type": "TestReport", "nodeid": "test_mod.py::test_gpu", "when": "setup", "outcome": "skipped", "longrepr": ["test_mod.py", 12, "Skipped: requires foo"]},
            {"$report_type": "TestReport", "nodeid": "test_mod.py::test_gpu", "when": "teardown", "outcome": "passed", "longrepr": None},
        ],
    )

    uncovered = uncovered_requires_tests([reportlog], allowlist=set())

    assert uncovered == {"test_mod.py::test_gpu": {"requires foo"}}


def test_requires_test_is_covered_if_it_runs_elsewhere(tmp_path: Path) -> None:
    reportlog_a = _write_reportlog(
        tmp_path / "a.jsonl",
        [
            {"$report_type": "TestReport", "nodeid": "test_mod.py::test_optional", "when": "setup", "outcome": "skipped", "longrepr": ["test_mod.py", 12, "Skipped: requires foo"]},
        ],
    )
    reportlog_b = _write_reportlog(
        tmp_path / "b.jsonl",
        [
            {"$report_type": "TestReport", "nodeid": "test_mod.py::test_optional", "when": "setup", "outcome": "passed", "longrepr": None},
            {"$report_type": "TestReport", "nodeid": "test_mod.py::test_optional", "when": "call", "outcome": "passed", "longrepr": None},
            {"$report_type": "TestReport", "nodeid": "test_mod.py::test_optional", "when": "teardown", "outcome": "passed", "longrepr": None},
        ],
    )

    assert uncovered_requires_tests([reportlog_a, reportlog_b], allowlist=set()) == {}


def test_requires_skip_reason_allowlist_is_respected(tmp_path: Path) -> None:
    reportlog = _write_reportlog(
        tmp_path / "reportlog.jsonl",
        [
            {"$report_type": "TestReport", "nodeid": "test_mod.py::test_ros3", "when": "setup", "outcome": "skipped", "longrepr": ["test_mod.py", 12, "Skipped: requires h5netcdf>=1.3.0 and h5py with ros3 support"]},
        ],
    )

    assert (
        uncovered_requires_tests(
            [reportlog],
            allowlist={"requires h5netcdf>=1.3.0 and h5py with ros3 support"},
        )
        == {}
    )


def test_requires_skip_with_non_requires_skip_is_considered_covered(
    tmp_path: Path,
) -> None:
    reportlog = _write_reportlog(
        tmp_path / "reportlog.jsonl",
        [
            {
                "$report_type": "TestReport",
                "nodeid": "test_mod.py::test_optional",
                "when": "setup",
                "outcome": "skipped",
                "longrepr": ["test_mod.py", 12, "Skipped: requires foo"],
            },
            {
                "$report_type": "TestReport",
                "nodeid": "test_mod.py::test_optional",
                "when": "setup",
                "outcome": "skipped",
                "longrepr": ["test_mod.py", 12, "Skipped: some other reason"],
            },
        ],
    )

    assert uncovered_requires_tests([reportlog], allowlist=set()) == {}
