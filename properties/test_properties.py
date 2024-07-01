import pytest

pytest.importorskip("hypothesis")

from hypothesis import given

import xarray as xr
import xarray.testing.strategies as xrst


@given(attrs=xrst.simple_attrs)
def test_assert_identical(attrs):
    v = xr.Variable(dims=(), data=0, attrs=attrs)
    xr.testing.assert_identical(v, v.copy(deep=True))

    ds = xr.Dataset(attrs=attrs)
    xr.testing.assert_identical(ds, ds.copy(deep=True))
