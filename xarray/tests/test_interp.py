from __future__ import absolute_import, division, print_function

import itertools

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core.pycompat import dask_array_type
from xarray.tests import (
    assert_array_equal, assert_equal, raises_regex, requires_bottleneck,
    requires_dask, requires_np112, requires_scipy)

try:
    import scipy
except ImportError:
    pass


def get_example_data(case):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 0.1, 30)
    data = xr.DataArray(
        np.sin(x[:, np.newaxis])*np.cos(y), dims=['x', 'y'],
        coords={'x': x, 'y': y, 'x2': ('x', x**2)})

    if case == 0:
        return data
    elif case == 1:
        return data.chunk({'y': 3})
    elif case == 2:
        return data.chunk({'x': 25, 'y': 3})


@requires_scipy
@pytest.mark.parametrize('method', ['linear'])
@pytest.mark.parametrize('dim', ['x', 'y'])
@pytest.mark.parametrize('case', [0, 1])
def test_interpolate_1d(method, dim, case):
    if dim == 'y' and case == 1:
        pytest.skip('interpolation along chunked dimension is '
                    'not yet supported')

    if not has_dask and case in [1]:
        pytest.skip('dask is not installed in the environment.')

    da = get_example_data(case)
    xdest = np.linspace(0.1, 0.9, 80)

    actual = da.interpolate_at(**{dim: xdest}, method=method)

    # scipy interpolation for the reference
    def func(obj, new_x):
        return scipy.interpolate.interp1d(
            da[dim], obj.data, axis=obj.get_axis_num(dim), bounds_error=False,
            fill_value=np.nan)(new_x)

    if dim == 'x':
        coords = {'x': xdest, 'y': da['y'], 'x2': ('x', func(da['x2'], xdest))}
    else:  # y
        coords = {'x': da['x'], 'y': xdest, 'x2': da['x2']}

    expected = xr.DataArray(func(da, xdest), dims=['x', 'y'], coords=coords)
    assert_equal(actual, expected)


def test_interpolate_vectorize():
    # scipy interpolation for the reference
    def func(obj, dim, new_x):
        shape = [s for i, s in enumerate(obj.shape)
                 if i != obj.get_axis_num(dim)]
        for s in new_x.shape[::-1]:
            shape.insert(obj.get_axis_num(dim), s)

        return scipy.interpolate.interp1d(
            da[dim], obj.data, axis=obj.get_axis_num(dim),
            bounds_error=False, fill_value=np.nan)(new_x).reshape(shape)


    da = get_example_data(0)

    # xdest is 1d but has different dimension
    xdest = xr.DataArray(np.linspace(0.1, 0.9, 30), dims='z',
                         coords={'z': np.random.randn(30),
                                 'z2': ('z', np.random.randn(30))})

    actual = da.interpolate_at(x=xdest, method='linear')

    expected = xr.DataArray(func(da, 'x', xdest), dims=['z', 'y'],
                            coords={'z': xdest['z'], 'z2': xdest['z2'],
                                    'y': da['y'],
                                    'x': ('z', xdest.values),
                                    'x2': ('z', func(da['x2'], 'x', xdest))})
    assert_equal(actual, expected.transpose('y', 'z'))

    # xdest is 2d
    xdest = xr.DataArray(np.linspace(0.1, 0.9, 30).reshape(6, 5),
                         dims=['z', 'w'],
                         coords={'z': np.random.randn(6),
                                 'w': np.random.randn(5),
                                 'z2': ('z', np.random.randn(6))})

    actual = da.interpolate_at(x=xdest, method='linear')

    expected = xr.DataArray(
        func(da, 'x', xdest),
        dims=['z', 'w', 'y'],
        coords={'z': xdest['z'], 'w': xdest['w'], 'z2': xdest['z2'],
                'y': da['y'], 'x': (('z', 'w'), xdest),
                'x2': (('z', 'w'), func(da['x2'], 'x', xdest))})
    assert_equal(actual, expected.transpose('y', 'z', 'w'))
