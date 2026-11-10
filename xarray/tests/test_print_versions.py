from __future__ import annotations

import importlib.metadata
import io
import sys
import types
from html import escape
from unittest import mock

import pytest

import xarray
from xarray.util import _report


# importing setuptools after distutils (done by some dependencies) warns
@pytest.mark.filterwarnings("ignore:Setuptools is replacing distutils:UserWarning")
def test_show_versions() -> None:
    f = io.StringIO()
    xarray.show_versions(file=f)
    out = f.getvalue()
    assert out.startswith("INSTALLED VERSIONS\n------------------\n")
    assert f"\nxarray: {importlib.metadata.version('xarray')}\n" in out
    assert "Not installed: " in out


@pytest.mark.filterwarnings("ignore:Setuptools is replacing distutils:UserWarning")
def test_show_versions_markdown() -> None:
    f = io.StringIO()
    xarray.show_versions(file=f, markdown=True)
    out = f.getvalue()
    assert out.startswith("<details><summary>INSTALLED VERSIONS</summary>\n\n")
    assert out.endswith("\n</details>\n")
    assert "| package " in out
    assert "| xarray " in out


def test_package_version() -> None:
    assert _report.package_version("xarray") == importlib.metadata.version("xarray")
    assert _report.package_version("not_a_real_package_xyz") is None


def test_package_version_dist_name_differs() -> None:
    with mock.patch.object(
        _report, "_packages_distributions", return_value={"fake_mod": ["pytest"]}
    ):
        assert _report.package_version("fake_mod") == pytest.__version__


def test_markdown_table() -> None:
    actual = _report.markdown_table(("a", "bb"), [("x", "1 | 2"), ("long", None)])
    expected = (
        "| a    | bb     |\n| ---- | ------ |\n| x    | 1 \\| 2 |\n| long | None   |"
    )
    assert actual == expected


def test_show_report(monkeypatch) -> None:
    displayed: list[dict[str, str]] = []

    def display(bundle, raw=False):
        assert raw
        displayed.append(bundle)

    display_mod = types.ModuleType("IPython.display")
    display_mod.display = display  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython.display", display_mod)

    md = "<details><summary>TITLE</summary>\n\n*md*\n\n</details>"

    def show(**kwargs) -> str:
        f = io.StringIO()
        _report.show_report("TITLE", "text", "*md*", "<p>x</p>", file=f, **kwargs)
        return f.getvalue()

    assert show() == "TITLE\n-----\ntext\n"
    assert show(as_markdown=True) == md + "\n"

    # in notebooks the html is displayed, unless printing is requested
    monkeypatch.setattr(_report, "_in_rich_frontend", lambda: True)
    _report.show_report("TITLE", "text", "*md*", "<p>x</p>")
    (bundle,) = displayed
    assert bundle["text/plain"] == "TITLE\n-----\ntext"
    # copying the output copies the markdown
    assert bundle["text/html"].startswith(f'<div data-md="{escape(md)}" oncopy="')
    assert bundle["text/html"].endswith("><h4>TITLE</h4><p>x</p></div>")

    assert show() == "TITLE\n-----\ntext\n"
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    _report.show_report("TITLE", "text", "*md*", "<p>x</p>", as_markdown=True)
    assert stdout.getvalue() == md + "\n"
    assert len(displayed) == 1


def test_in_rich_frontend_without_ipython(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    assert not _report._in_rich_frontend()


def test_html_table() -> None:
    actual = _report.html_table("T<", [("a", "<b>")])
    assert actual == (
        '<div><strong>T&lt;</strong><table><tr><td style="text-align: left">a</td>'
        '<td style="text-align: left">&lt;b&gt;</td></tr></table></div>'
    )
