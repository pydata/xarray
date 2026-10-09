"""Helpers to render diagnostic reports (show_versions, show_backends).

Reports are printed as plain text in terminals, displayed as formatted markdown
in notebooks and can be printed as markdown to paste them into GitHub issues.
"""

from __future__ import annotations

import functools
import importlib.metadata
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import TextIO


@functools.cache
def _packages_distributions() -> Mapping[str, list[str]]:
    return importlib.metadata.packages_distributions()


def package_version(module_name: str) -> str | None:
    """Return the installed version of the package providing ``module_name``.

    Uses the package metadata, so the module is not imported.
    Returns None if the package is not installed.
    """
    try:
        return importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        pass
    # the distribution name differs from the module name, e.g. scitools-iris
    for dist_name in _packages_distributions().get(module_name, []):
        try:
            return importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _escape_cell(value: object) -> str:
    return " ".join(str(value).split()).replace("|", r"\|")


def markdown_table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render an aligned markdown table that is also readable as plain text."""
    cells = [[_escape_cell(v) for v in row] for row in [header, *rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    widths = [max(w, 3) for w in widths]

    def fmt(row: Sequence[str]) -> str:
        return (
            "| "
            + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True))
            + " |"
        )

    lines = [fmt(cells[0]), fmt(["-" * w for w in widths])]
    lines.extend(fmt(row) for row in cells[1:])
    return "\n".join(lines)


def _in_rich_frontend() -> bool:
    """Whether we run in a frontend that can display markdown, e.g. Jupyter."""
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return False
    shell = ipython.get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def show_report(
    title: str,
    text: str,
    markdown: str,
    file: TextIO | None = None,
    as_markdown: bool = False,
) -> None:
    """Print a report as plain text or markdown, or display it in notebooks.

    In Jupyter notebooks the markdown is displayed formatted, unless a ``file``
    is given or ``as_markdown`` is requested. The printed markdown is wrapped
    in a ``<details>`` block, so it can be pasted as is into a GitHub issue.
    """
    if file is None and not as_markdown and _in_rich_frontend():
        from IPython.display import Markdown, display

        display(Markdown(f"**{title}**\n\n{markdown}"))
        return

    if as_markdown:
        out = f"<details><summary>{title}</summary>\n\n{markdown}\n\n</details>"
    else:
        out = f"{title}\n{'-' * len(title)}\n{text}"
    print(out, file=sys.stdout if file is None else file)
