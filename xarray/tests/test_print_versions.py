from __future__ import annotations

import importlib.metadata
import io
import sys
import types
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
    assert out.startswith("<details><summary>INSTALLED VERSIONS</summary>\n\n")
    assert out.endswith("\n</details>\n")
    assert "| xarray " in out
    assert xarray.__version__ in out
    assert "Not installed: " in out


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


def test_show_report_rich_frontend(monkeypatch) -> None:
    displayed: list[str] = []
    display_mod = types.ModuleType("IPython.display")
    display_mod.Markdown = lambda text: text  # type: ignore[attr-defined]
    display_mod.display = displayed.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython.display", display_mod)
    monkeypatch.setattr(_report, "_in_rich_frontend", lambda: True)

    _report.show_report("TITLE", "body")
    assert displayed == ["**TITLE**\n\nbody"]

    # an explicit file always gets the plain markdown
    f = io.StringIO()
    _report.show_report("TITLE", "body", file=f)
    assert f.getvalue() == "<details><summary>TITLE</summary>\n\nbody\n\n</details>\n"
    assert len(displayed) == 1
