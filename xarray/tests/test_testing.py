from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xarray as xr


def test_allclose_regression():
    x = xr.DataArray(1.01)
    y = xr.DataArray(1.02)
    xr.testing.assert_allclose(x, y, atol=0.01)
