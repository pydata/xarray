# type: ignore
import argparse
import functools
import json
import pathlib
import textwrap
from dataclasses import dataclass

from pytest import CollectReport, TestReport


@dataclass
class SessionStart:
    pytest_version: str
    outcome: str = "status"

    @classmethod
    def _from_json(cls, json):
        json_ = json.copy()
        json_.pop("$report_type")
        return cls(**json_)


@dataclass
class SessionFinish:
    exitstatus: str
    outcome: str = "status"

    @classmethod
    def _from_json(cls, json):
        json_ = json.copy()
        json_.pop("$report_type")
        return cls(**json_)


def parse_record(record):
    report_types = {
        "TestReport": TestReport,
        "CollectReport": CollectReport,
        "SessionStart": SessionStart,
        "SessionFinish": SessionFinish,
    }
    cls = report_types.get(record["$report_type"])
    if cls is None:
        raise ValueError(f"unknown report type: {record['$report_type']}")

    return cls._from_json(record)


@functools.singledispatch
def format_summary(report):
    return f"{report.nodeid}: {report}"


@format_summary.register
def _(report: TestReport):
    message = report.longrepr.chain[0][1].message
    return f"{report.nodeid}: {message}"


@format_summary.register
def _(report: CollectReport):
    message = report.longrepr.split("\n")[-1].removeprefix("E").lstrip()
    return f"{report.nodeid}: {message}"


def format_report(reports, py_version):
    newline = "\n"
    summaries = newline.join(format_summary(r) for r in reports)
    message = textwrap.dedent(
        """\
        <details><summary>Python {py_version} Test Summary</summary>

        ```
        {summaries}
        ```

        </details>
        """
    ).format(summaries=summaries, py_version=py_version)
    return message


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=pathlib.Path)
    args = parser.parse_args()

    py_version = args.filepath.stem.split("-")[1]

    print("Parsing logs ...")

    lines = args.filepath.read_text().splitlines()
    reports = [parse_record(json.loads(line)) for line in lines]

    failed = [report for report in reports if report.outcome == "failed"]

    message = format_report(failed, py_version=py_version)

    output_file = pathlib.Path("pytest-logs.txt")
    print(f"Writing output file to: {output_file.absolute()}")
    output_file.write_text(message)
