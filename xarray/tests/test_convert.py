import numpy as np
import xarray as xr
import iris
from iris.coords import DimCoord
from iris.cube import Cube


def test_from_iris_no_var_name():
    latitude = DimCoord(np.linspace(-90, 90, 4),
                        standard_name='latitude',
                        units='degrees')
    cube = Cube(np.zeros((4,), np.float32),
                dim_coords_and_dims=[(latitude, 0)])
    xr.DataArray.from_iris(cube)


def test_to_iris_non_numeric_coord():
    data = np.random.rand(3)
    locs = ['IA', 'IL', 'IN']
    da = xr.DataArray(data, coords=[locs], dims=['space'])
    xr.DataArray.to_iris(da)


def test_to_iris_non_monotonic_coord():
    data = np.random.rand(3)
    locs = [0, 2, 1]
    da = xr.DataArray(data, coords=[locs], dims=['space'])
    xr.DataArray.to_iris(da)
