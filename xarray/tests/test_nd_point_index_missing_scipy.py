import sys

import numpy as np
import pytest

import xarray as xr
from xarray.indexes import NDPointIndex


def test_ndpointindex_missing_scipy(monkeypatch):
    # Simulate scipy not being installed
    monkeypatch.setitem(sys.modules, "scipy", None)

    xx = xr.DataArray(
        np.random.rand(2, 2),
        dims=("y", "x"),
        name="xx",
    )
    yy = xr.DataArray(
        np.random.rand(2, 2),
        dims=("y", "x"),
        name="yy",
    )

    ds = xr.Dataset(coords={"xx": xx, "yy": yy})

    with pytest.raises(
        ImportError,
        match="scipy.*NDPointIndex|NDPointIndex.*scipy",
    ):
        ds.set_xindex(("xx", "yy"), NDPointIndex)
