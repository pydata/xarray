import numpy as np
import xarray as xr

def test_custom_array_sum():
    data = xr.DataArray([1, 2, 3, 4])
    assert data.sum().item() == 10
