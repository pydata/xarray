from __future__ import annotations

import io

import pytest

import xarray


# importing setuptools after distutils (done by some dependencies) warns
@pytest.mark.filterwarnings("ignore:Setuptools is replacing distutils:UserWarning")
def test_show_versions() -> None:
    f = io.StringIO()
    xarray.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
