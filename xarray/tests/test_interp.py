from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.tests import assert_allclose, assert_equal, requires_scipy
from . import has_dask, has_scipy
from .test_dataset import create_test_data

try:
    import scipy
except ImportError:
    pass


def get_example_data(case):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 0.1, 30)
    data = xr.DataArray(
        np.sin(x[:, np.newaxis]) * np.cos(y), dims=['x', 'y'],
        coords={'x': x, 'y': y, 'x2': ('x', x**2)})

    if case == 0:
        return data
    elif case == 1:
        return data.chunk({'y': 3})
    elif case == 2:
        return data.chunk({'x': 25, 'y': 3})
    elif case == 3:
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 0.1, 30)
        z = np.linspace(0.1, 0.2, 10)
        return xr.DataArray(
            np.sin(x[:, np.newaxis, np.newaxis]) * np.cos(
                y[:, np.newaxis]) * z,
            dims=['x', 'y', 'z'],
            coords={'x': x, 'y': y, 'x2': ('x', x**2), 'z': z})
    elif case == 4:
        return get_example_data(3).chunk({'z': 5})


def test_keywargs():
    if not has_scipy:
        pytest.skip('scipy is not installed.')

    da = get_example_data(0)
    assert_equal(da.interp(x=[0.5, 0.8]), da.interp({'x': [0.5, 0.8]}))


@pytest.mark.parametrize('method', ['linear', 'cubic'])
@pytest.mark.parametrize('dim', ['x', 'y'])
@pytest.mark.parametrize('case', [0, 1])
def test_interpolate_1d(method, dim, case):
    if not has_scipy:
        pytest.skip('scipy is not installed.')

    if not has_dask and case in [1]:
        pytest.skip('dask is not installed in the environment.')

    da = get_example_data(case)
    xdest = np.linspace(0.0, 0.9, 80)

    if dim == 'y' and case == 1:
        with pytest.raises(NotImplementedError):
            actual = da.interp(method=method, **{dim: xdest})
        pytest.skip('interpolation along chunked dimension is '
                    'not yet supported')

    actual = da.interp(method=method, **{dim: xdest})

    # scipy interpolation for the reference
    def func(obj, new_x):
        return scipy.interpolate.interp1d(
            da[dim], obj.data, axis=obj.get_axis_num(dim), bounds_error=False,
            fill_value=np.nan, kind=method)(new_x)

    if dim == 'x':
        coords = {'x': xdest, 'y': da['y'], 'x2': ('x', func(da['x2'], xdest))}
    else:  # y
        coords = {'x': da['x'], 'y': xdest, 'x2': da['x2']}

    expected = xr.DataArray(func(da, xdest), dims=['x', 'y'], coords=coords)
    assert_allclose(actual, expected)


@pytest.mark.parametrize('method', ['cubic', 'zero'])
def test_interpolate_1d_methods(method):
    if not has_scipy:
        pytest.skip('scipy is not installed.')

    da = get_example_data(0)
    dim = 'x'
    xdest = np.linspace(0.0, 0.9, 80)

    actual = da.interp(method=method, **{dim: xdest})

    # scipy interpolation for the reference
    def func(obj, new_x):
        return scipy.interpolate.interp1d(
            da[dim], obj.data, axis=obj.get_axis_num(dim), bounds_error=False,
            fill_value=np.nan, kind=method)(new_x)

    coords = {'x': xdest, 'y': da['y'], 'x2': ('x', func(da['x2'], xdest))}
    expected = xr.DataArray(func(da, xdest), dims=['x', 'y'], coords=coords)
    assert_allclose(actual, expected)


@pytest.mark.parametrize('use_dask', [False, True])
def test_interpolate_vectorize(use_dask):
    if not has_scipy:
        pytest.skip('scipy is not installed.')

    if not has_dask and use_dask:
        pytest.skip('dask is not installed in the environment.')

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
    if use_dask:
        da = da.chunk({'y': 5})

    # xdest is 1d but has different dimension
    xdest = xr.DataArray(np.linspace(0.1, 0.9, 30), dims='z',
                         coords={'z': np.random.randn(30),
                                 'z2': ('z', np.random.randn(30))})

    actual = da.interp(x=xdest, method='linear')

    expected = xr.DataArray(func(da, 'x', xdest), dims=['z', 'y'],
                            coords={'z': xdest['z'], 'z2': xdest['z2'],
                                    'y': da['y'],
                                    'x': ('z', xdest.values),
                                    'x2': ('z', func(da['x2'], 'x', xdest))})
    assert_allclose(actual, expected.transpose('z', 'y'))

    # xdest is 2d
    xdest = xr.DataArray(np.linspace(0.1, 0.9, 30).reshape(6, 5),
                         dims=['z', 'w'],
                         coords={'z': np.random.randn(6),
                                 'w': np.random.randn(5),
                                 'z2': ('z', np.random.randn(6))})

    actual = da.interp(x=xdest, method='linear')

    expected = xr.DataArray(
        func(da, 'x', xdest),
        dims=['z', 'w', 'y'],
        coords={'z': xdest['z'], 'w': xdest['w'], 'z2': xdest['z2'],
                'y': da['y'], 'x': (('z', 'w'), xdest),
                'x2': (('z', 'w'), func(da['x2'], 'x', xdest))})
    assert_allclose(actual, expected.transpose('z', 'w', 'y'))


@pytest.mark.parametrize('case', [3, 4])
def test_interpolate_nd(case):
    if not has_scipy:
        pytest.skip('scipy is not installed.')

    if not has_dask and case == 4:
        pytest.skip('dask is not installed in the environment.')

    da = get_example_data(case)

    # grid -> grid
    xdest = np.linspace(0.1, 1.0, 11)
    ydest = np.linspace(0.0, 0.2, 10)
    actual = da.interp(x=xdest, y=ydest, method='linear')

    # linear interpolation is separateable
    expected = da.interp(x=xdest, method='linear')
    expected = expected.interp(y=ydest, method='linear')
    assert_allclose(actual.transpose('x', 'y', 'z'),
                    expected.transpose('x', 'y', 'z'))

    # grid -> 1d-sample
    xdest = xr.DataArray(np.linspace(0.1, 1.0, 11), dims='y')
    ydest = xr.DataArray(np.linspace(0.0, 0.2, 11), dims='y')
    actual = da.interp(x=xdest, y=ydest, method='linear')

    # linear interpolation is separateable
    expected_data = scipy.interpolate.RegularGridInterpolator(
        (da['x'], da['y']), da.transpose('x', 'y', 'z').values,
        method='linear', bounds_error=False,
        fill_value=np.nan)(np.stack([xdest, ydest], axis=-1))
    expected = xr.DataArray(
        expected_data, dims=['y', 'z'],
        coords={'z': da['z'], 'y': ydest, 'x': ('y', xdest.values),
                'x2': da['x2'].interp(x=xdest)})
    assert_allclose(actual.transpose('y', 'z'), expected)

    # reversed order
    actual = da.interp(y=ydest, x=xdest, method='linear')
    assert_allclose(actual.transpose('y', 'z'), expected)


@pytest.mark.parametrize('method', ['linear'])
@pytest.mark.parametrize('case', [0, 1])
def test_interpolate_scalar(method, case):
    if not has_scipy:
        pytest.skip('scipy is not installed.')

    if not has_dask and case in [1]:
        pytest.skip('dask is not installed in the environment.')

    da = get_example_data(case)
    xdest = 0.4

    actual = da.interp(x=xdest, method=method)

    # scipy interpolation for the reference
    def func(obj, new_x):
        return scipy.interpolate.interp1d(
            da['x'], obj.data, axis=obj.get_axis_num('x'), bounds_error=False,
            fill_value=np.nan)(new_x)

    coords = {'x': xdest, 'y': da['y'], 'x2': func(da['x2'], xdest)}
    expected = xr.DataArray(func(da, xdest), dims=['y'], coords=coords)
    assert_allclose(actual, expected)


@pytest.mark.parametrize('method', ['linear'])
@pytest.mark.parametrize('case', [3, 4])
def test_interpolate_nd_scalar(method, case):
    if not has_scipy:
        pytest.skip('scipy is not installed.')

    if not has_dask and case in [4]:
        pytest.skip('dask is not installed in the environment.')

    da = get_example_data(case)
    xdest = 0.4
    ydest = 0.05

    actual = da.interp(x=xdest, y=ydest, method=method)
    # scipy interpolation for the reference
    expected_data = scipy.interpolate.RegularGridInterpolator(
        (da['x'], da['y']), da.transpose('x', 'y', 'z').values,
        method='linear', bounds_error=False,
        fill_value=np.nan)(np.stack([xdest, ydest], axis=-1))

    coords = {'x': xdest, 'y': ydest, 'x2': da['x2'].interp(x=xdest),
              'z': da['z']}
    expected = xr.DataArray(expected_data[0], dims=['z'], coords=coords)
    assert_allclose(actual, expected)


@pytest.mark.parametrize('use_dask', [True, False])
def test_nans(use_dask):
    if not has_scipy:
        pytest.skip('scipy is not installed.')

    da = xr.DataArray([0, 1, np.nan, 2], dims='x', coords={'x': range(4)})

    if not has_dask and use_dask:
        pytest.skip('dask is not installed in the environment.')
        da = da.chunk()

    actual = da.interp(x=[0.5, 1.5])
    # not all values are nan
    assert actual.count() > 0


@pytest.mark.parametrize('use_dask', [True, False])
def test_errors(use_dask):
    if not has_scipy:
        pytest.skip('scipy is not installed.')

    # akima and spline are unavailable
    da = xr.DataArray([0, 1, np.nan, 2], dims='x', coords={'x': range(4)})
    if not has_dask and use_dask:
        pytest.skip('dask is not installed in the environment.')
        da = da.chunk()

    for method in ['akima', 'spline']:
        with pytest.raises(ValueError):
            da.interp(x=[0.5, 1.5], method=method)

    # not sorted
    if use_dask:
        da = get_example_data(3)
    else:
        da = get_example_data(1)

    result = da.interp(x=[-1, 1, 3], kwargs={'fill_value': 0.0})
    assert not np.isnan(result.values).any()
    result = da.interp(x=[-1, 1, 3])
    assert np.isnan(result.values).any()

    # invalid method
    with pytest.raises(ValueError):
        da.interp(x=[2, 0], method='boo')
    with pytest.raises(ValueError):
        da.interp(x=[2, 0], y=2, method='cubic')
    with pytest.raises(ValueError):
        da.interp(y=[2, 0], method='boo')

    # object-type DataArray cannot be interpolated
    da = xr.DataArray(['a', 'b', 'c'], dims='x', coords={'x': [0, 1, 2]})
    with pytest.raises(TypeError):
        da.interp(x=0)


@requires_scipy
def test_dtype():
    ds = xr.Dataset({'var1': ('x', [0, 1, 2]), 'var2': ('x', ['a', 'b', 'c'])},
                    coords={'x': [0.1, 0.2, 0.3], 'z': ('x', ['a', 'b', 'c'])})
    actual = ds.interp(x=[0.15, 0.25])
    assert 'var1' in actual
    assert 'var2' not in actual
    # object array should be dropped
    assert 'z' not in actual.coords


@requires_scipy
def test_sorted():
    # unsorted non-uniform gridded data
    x = np.random.randn(100)
    y = np.random.randn(30)
    z = np.linspace(0.1, 0.2, 10) * 3.0
    da = xr.DataArray(
        np.cos(x[:, np.newaxis, np.newaxis]) * np.cos(
            y[:, np.newaxis]) * z,
        dims=['x', 'y', 'z'],
        coords={'x': x, 'y': y, 'x2': ('x', x**2), 'z': z})

    x_new = np.linspace(0, 1, 30)
    y_new = np.linspace(0, 1, 20)

    da_sorted = da.sortby('x')
    assert_allclose(da.interp(x=x_new),
                    da_sorted.interp(x=x_new, assume_sorted=True))
    da_sorted = da.sortby(['x', 'y'])
    assert_allclose(da.interp(x=x_new, y=y_new),
                    da_sorted.interp(x=x_new, y=y_new, assume_sorted=True))

    with pytest.raises(ValueError):
        da.interp(x=[0, 1, 2], assume_sorted=True)


@requires_scipy
def test_dimension_wo_coords():
    da = xr.DataArray(np.arange(12).reshape(3, 4), dims=['x', 'y'],
                      coords={'y': [0, 1, 2, 3]})
    da_w_coord = da.copy()
    da_w_coord['x'] = np.arange(3)

    assert_equal(da.interp(x=[0.1, 0.2, 0.3]),
                 da_w_coord.interp(x=[0.1, 0.2, 0.3]))
    assert_equal(da.interp(x=[0.1, 0.2, 0.3], y=[0.5]),
                 da_w_coord.interp(x=[0.1, 0.2, 0.3], y=[0.5]))


@requires_scipy
def test_dataset():
    ds = create_test_data()
    ds.attrs['foo'] = 'var'
    ds['var1'].attrs['buz'] = 'var2'
    new_dim2 = xr.DataArray([0.11, 0.21, 0.31], dims='z')
    interpolated = ds.interp(dim2=new_dim2)

    assert_allclose(interpolated['var1'], ds['var1'].interp(dim2=new_dim2))
    assert interpolated['var3'].equals(ds['var3'])

    # make sure modifying interpolated does not affect the original dataset
    interpolated['var1'][:, 1] = 1.0
    interpolated['var2'][:, 1] = 1.0
    interpolated['var3'][:, 1] = 1.0

    assert not interpolated['var1'].equals(ds['var1'])
    assert not interpolated['var2'].equals(ds['var2'])
    assert not interpolated['var3'].equals(ds['var3'])
    # attrs should be kept
    assert interpolated.attrs['foo'] == 'var'
    assert interpolated['var1'].attrs['buz'] == 'var2'


@pytest.mark.parametrize('case', [0, 3])
def test_interpolate_dimorder(case):
    """ Make sure the resultant dimension order is consistent with .sel() """
    if not has_scipy:
        pytest.skip('scipy is not installed.')

    da = get_example_data(case)

    new_x = xr.DataArray([0, 1, 2], dims='x')
    assert da.interp(x=new_x).dims == da.sel(x=new_x, method='nearest').dims

    new_y = xr.DataArray([0, 1, 2], dims='y')
    actual = da.interp(x=new_x, y=new_y).dims
    expected = da.sel(x=new_x, y=new_y, method='nearest').dims
    assert actual == expected
    # reversed order
    actual = da.interp(y=new_y, x=new_x).dims
    expected = da.sel(y=new_y, x=new_x, method='nearest').dims
    assert actual == expected

    new_x = xr.DataArray([0, 1, 2], dims='a')
    assert da.interp(x=new_x).dims == da.sel(x=new_x, method='nearest').dims
    assert da.interp(y=new_x).dims == da.sel(y=new_x, method='nearest').dims
    new_y = xr.DataArray([0, 1, 2], dims='a')
    actual = da.interp(x=new_x, y=new_y).dims
    expected = da.sel(x=new_x, y=new_y, method='nearest').dims
    assert actual == expected

    new_x = xr.DataArray([[0], [1], [2]], dims=['a', 'b'])
    assert da.interp(x=new_x).dims == da.sel(x=new_x, method='nearest').dims
    assert da.interp(y=new_x).dims == da.sel(y=new_x, method='nearest').dims

    if case == 3:
        new_x = xr.DataArray([[0], [1], [2]], dims=['a', 'b'])
        new_z = xr.DataArray([[0], [1], [2]], dims=['a', 'b'])
        actual = da.interp(x=new_x, z=new_z).dims
        expected = da.sel(x=new_x, z=new_z, method='nearest').dims
        assert actual == expected

        actual = da.interp(z=new_z, x=new_x).dims
        expected = da.sel(z=new_z, x=new_x, method='nearest').dims
        assert actual == expected

        actual = da.interp(x=0.5, z=new_z).dims
        expected = da.sel(x=0.5, z=new_z, method='nearest').dims
        assert actual == expected


@requires_scipy
def test_interp_like():
    ds = create_test_data()
    ds.attrs['foo'] = 'var'
    ds['var1'].attrs['buz'] = 'var2'

    other = xr.DataArray(np.random.randn(3), dims=['dim2'],
                         coords={'dim2': [0, 1, 2]})
    interpolated = ds.interp_like(other)

    assert_allclose(interpolated['var1'],
                    ds['var1'].interp(dim2=other['dim2']))
    assert_allclose(interpolated['var1'],
                    ds['var1'].interp_like(other))
    assert interpolated['var3'].equals(ds['var3'])

    # attrs should be kept
    assert interpolated.attrs['foo'] == 'var'
    assert interpolated['var1'].attrs['buz'] == 'var2'

    other = xr.DataArray(np.random.randn(3), dims=['dim3'],
                         coords={'dim3': ['a', 'b', 'c']})

    actual = ds.interp_like(other)
    expected = ds.reindex_like(other)
    assert_allclose(actual, expected)


@requires_scipy
def test_datetime():
    da = xr.DataArray(np.random.randn(24), dims='time',
                      coords={'time': pd.date_range('2000-01-01', periods=24)})

    x_new = pd.date_range('2000-01-02', periods=3)
    actual = da.interp(time=x_new)
    expected = da.isel(time=[1, 2, 3])
    assert_allclose(actual, expected)

    x_new = np.array([np.datetime64('2000-01-01T12:00'),
                      np.datetime64('2000-01-02T12:00')])
    actual = da.interp(time=x_new)
    assert_allclose(actual.isel(time=0).drop('time'),
                    0.5 * (da.isel(time=0) + da.isel(time=1)))
    assert_allclose(actual.isel(time=1).drop('time'),
                    0.5 * (da.isel(time=1) + da.isel(time=2)))
