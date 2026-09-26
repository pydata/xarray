import numpy as np

import xarray as xr
from xarray.coding.range import RangeIndexCoder
from xarray.indexes import RangeIndex
from xarray.tests import assert_identical


def test_encode() -> None:
    index = RangeIndex.arange(-10, 10, 1, dim="x", coord_name="l")
    ds = xr.Dataset(
        {"a": ("x", np.arange(20))}, coords=xr.Coordinates.from_xindex(index)
    )

    coder = RangeIndexCoder()
    encoded = coder.encode(ds)

    expected = {"l": {"start": -10, "stop": 10, "step": 1, "dim": "x"}}

    assert set(encoded.variables) == {"a"}  # x is gone
    assert not set(encoded.xindexes)  # no indexes
    assert encoded.attrs["ranges"] == expected


def test_decode() -> None:
    ds = xr.Dataset(
        {"a": ("x", np.arange(10))},
        attrs={"ranges": {"l": {"start": -5, "stop": 0, "step": 0.5, "dim": "x"}}},
    )

    coder = RangeIndexCoder()
    actual = coder.decode(ds)

    expected = xr.Dataset(
        {"a": ("x", np.arange(10))},
        coords=xr.Coordinates.from_xindex(
            RangeIndex.arange(-5, 0, 0.5, dim="x", coord_name="l")
        ),
    )
    assert_identical(actual, expected, check_indexes=True)
