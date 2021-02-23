# type: ignore
import argparse
import itertools
import pathlib
import textwrap

parser = argparse.ArgumentParser()
parser.add_argument("filepaths", nargs="+", type=pathlib.Path)
args = parser.parse_args()

filepaths = sorted(p for p in args.filepaths if p.is_file())


def extract_short_test_summary_info(lines):
    up_to_start_of_section = itertools.dropwhile(
        lambda l: "=== short test summary info ===" not in l,
        lines,
    )
    up_to_section_content = itertools.islice(up_to_start_of_section, 1, None)
    section_content = itertools.takewhile(
        lambda l: l.startswith("FAILED") or l.startswith("ERROR"), up_to_section_content
    )
    content = "\n".join(section_content)

    return content


def format_log_message(path):
    py_version = path.name.split("-")[1]
    summary = f"Python {py_version} Test Summary Info"
    with open(path) as f:
        data = extract_short_test_summary_info(line.rstrip() for line in f)
    message = (
        textwrap.dedent(
            """\
            <details><summary>{summary}</summary>

            ```
            {data}
            ```

            </details>
            """
        )
        .rstrip()
        .format(summary=summary, data=data)
    )

    return message


print("Parsing logs ...")
message = "\n\n".join(format_log_message(path) for path in filepaths)

output_file = pathlib.Path("pytest-logs.txt")
print(f"Writing output file to: {output_file.absolute()}")
output_file.write_text(message)
