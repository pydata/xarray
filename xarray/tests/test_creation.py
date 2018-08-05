import xarray as xr
import numpy as np
import warnings

def test_coords_are_dataset():
    # Setup an array with coordinates
    n = np.zeros(3)
    coords={'x': np.arange(3)}
    c = xr.Dataset(coords=coords)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "iteration over an xarray.Dataset")
        a = xr.DataArray(n, dims=['x'], coords=c)
