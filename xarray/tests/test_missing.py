from __future__ import absolute_import, division, print_function

import itertools

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core.missing import (
    NumpyInterpolator, ScipyInterpolator, SplineInterpolator)
from xarray.core.pycompat import dask_array_type
from xarray.tests import (
    assert_array_equal, assert_equal, raises_regex, requires_bottleneck,
    requires_dask, requires_scipy)


@pytest.fixture
def da():
    return xr.DataArray([0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7],
                        dims='time')


@pytest.fixture
def ds():
    ds = xr.Dataset()
    ds['var1'] = xr.DataArray([0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7],
                              dims='time')
    ds['var2'] = xr.DataArray([10, np.nan, 11, 12, np.nan, 13, 14, 15, np.nan,
                               16, 17], dims='x')
    return ds


def make_interpolate_example_data(shape, frac_nan, seed=12345,
                                  non_uniform=False):
    rs = np.random.RandomState(seed)
    vals = rs.normal(size=shape)
    if frac_nan == 1:
        vals[:] = np.nan
    elif frac_nan == 0:
        pass
    else:
        n_missing = int(vals.size * frac_nan)

        ys = np.arange(shape[0])
        xs = np.arange(shape[1])
        if n_missing:
            np.random.shuffle(ys)
            ys = ys[:n_missing]

            np.random.shuffle(xs)
            xs = xs[:n_missing]

            vals[ys, xs] = np.nan

    if non_uniform:
        # construct a datetime index that has irregular spacing
        deltas = pd.TimedeltaIndex(unit='d', data=rs.normal(size=shape[0],
                                                            scale=10))
        coords = {'time': (pd.Timestamp('2000-01-01') + deltas).sort_values()}
    else:
        coords = {'time': pd.date_range('2000-01-01', freq='D',
                                        periods=shape[0])}
    da = xr.DataArray(vals, dims=('time', 'x'), coords=coords)
    df = da.to_pandas()

    return da, df


@requires_scipy
def test_interpolate_pd_compat():
    shapes = [(8, 8), (1, 20), (20, 1), (100, 100)]
    frac_nans = [0, 0.5, 1]
    methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']

    for (shape, frac_nan, method) in itertools.product(shapes, frac_nans,
                                                       methods):

        da, df = make_interpolate_example_data(shape, frac_nan)

        for dim in ['time', 'x']:
            actual = da.interpolate_na(method=method, dim=dim,
                                       fill_value=np.nan)
            expected = df.interpolate(method=method, axis=da.get_axis_num(dim),
                                      fill_value=(np.nan, np.nan))
            # Note, Pandas does some odd things with the left/right fill_value
            # for the linear methods. This next line inforces the xarray
            # fill_value convention on the pandas output. Therefore, this test
            # only checks that interpolated values are the same (not nans)
            expected.values[pd.isnull(actual.values)] = np.nan

            np.testing.assert_allclose(actual.values, expected.values)


@requires_scipy
@pytest.mark.parametrize('method', ['barycentric', 'krog',
                                    'pchip', 'spline', 'akima'])
def test_scipy_methods_function(method):
    # Note: Pandas does some wacky things with these methods and the full
    # integration tests wont work.
    da, _ = make_interpolate_example_data((25, 25), 0.4, non_uniform=True)
    actual = da.interpolate_na(method=method, dim='time')
    assert (da.count('time') <= actual.count('time')).all()


@requires_scipy
def test_interpolate_pd_compat_non_uniform_index():
    shapes = [(8, 8), (1, 20), (20, 1), (100, 100)]
    frac_nans = [0, 0.5, 1]
    methods = ['time', 'index', 'values']

    for (shape, frac_nan, method) in itertools.product(shapes, frac_nans,
                                                       methods):

        da, df = make_interpolate_example_data(shape, frac_nan,
                                               non_uniform=True)
        for dim in ['time', 'x']:
            if method == 'time' and dim != 'time':
                continue
            actual = da.interpolate_na(method='linear', dim=dim,
                                       use_coordinate=True, fill_value=np.nan)
            expected = df.interpolate(method=method, axis=da.get_axis_num(dim),
                                      fill_value=np.nan)

            # Note, Pandas does some odd things with the left/right fill_value
            # for the linear methods. This next line inforces the xarray
            # fill_value convention on the pandas output. Therefore, this test
            # only checks that interpolated values are the same (not nans)
            expected.values[pd.isnull(actual.values)] = np.nan

            np.testing.assert_allclose(actual.values, expected.values)


@requires_scipy
def test_interpolate_pd_compat_polynomial():
    shapes = [(8, 8), (1, 20), (20, 1), (100, 100)]
    frac_nans = [0, 0.5, 1]
    orders = [1, 2, 3]

    for (shape, frac_nan, order) in itertools.product(shapes, frac_nans,
                                                      orders):

        da, df = make_interpolate_example_data(shape, frac_nan)

        for dim in ['time', 'x']:
            actual = da.interpolate_na(method='polynomial', order=order,
                                       dim=dim, use_coordinate=False)
            expected = df.interpolate(method='polynomial', order=order,
                                      axis=da.get_axis_num(dim))
            np.testing.assert_allclose(actual.values, expected.values)


@requires_scipy
def test_interpolate_unsorted_index_raises():
    vals = np.array([1, 2, 3], dtype=np.float64)
    expected = xr.DataArray(vals, dims='x', coords={'x': [2, 1, 3]})
    with raises_regex(ValueError, 'Index must be monotonicly increasing'):
        expected.interpolate_na(dim='x', method='index')


def test_interpolate_no_dim_raises():
    da = xr.DataArray(np.array([1, 2, np.nan, 5], dtype=np.float64), dims='x')
    with raises_regex(NotImplementedError, 'dim is a required argument'):
        da.interpolate_na(method='linear')


def test_interpolate_invalid_interpolator_raises():
    da = xr.DataArray(np.array([1, 2, np.nan, 5], dtype=np.float64), dims='x')
    with raises_regex(ValueError, 'not a valid'):
        da.interpolate_na(dim='x', method='foo')


def test_interpolate_multiindex_raises():
    data = np.random.randn(2, 3)
    data[1, 1] = np.nan
    da = xr.DataArray(data, coords=[('x', ['a', 'b']), ('y', [0, 1, 2])])
    das = da.stack(z=('x', 'y'))
    with raises_regex(TypeError, 'Index must be castable to float64'):
        das.interpolate_na(dim='z')


def test_interpolate_2d_coord_raises():
    coords = {'x': xr.Variable(('a', 'b'), np.arange(6).reshape(2, 3)),
              'y': xr.Variable(('a', 'b'), np.arange(6).reshape(2, 3)) * 2}

    data = np.random.randn(2, 3)
    data[1, 1] = np.nan
    da = xr.DataArray(data, dims=('a', 'b'), coords=coords)
    with raises_regex(ValueError, 'interpolation must be 1D'):
        da.interpolate_na(dim='a', use_coordinate='x')


@requires_scipy
def test_interpolate_kwargs():
    da = xr.DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims='x')
    expected = xr.DataArray(np.array([4, 5, 6], dtype=np.float64), dims='x')
    actual = da.interpolate_na(dim='x', fill_value='extrapolate')
    assert_equal(actual, expected)

    expected = xr.DataArray(np.array([4, 5, -999], dtype=np.float64), dims='x')
    actual = da.interpolate_na(dim='x', fill_value=-999)
    assert_equal(actual, expected)


def test_interpolate():

    vals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    expected = xr.DataArray(vals, dims='x')
    mvals = vals.copy()
    mvals[2] = np.nan
    missing = xr.DataArray(mvals, dims='x')

    actual = missing.interpolate_na(dim='x')

    assert_equal(actual, expected)


def test_interpolate_nonans():

    vals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    expected = xr.DataArray(vals, dims='x')
    actual = expected.interpolate_na(dim='x')
    assert_equal(actual, expected)


@requires_scipy
def test_interpolate_allnans():
    vals = np.full(6, np.nan, dtype=np.float64)
    expected = xr.DataArray(vals, dims='x')
    actual = expected.interpolate_na(dim='x')

    assert_equal(actual, expected)


@requires_bottleneck
def test_interpolate_limits():
    da = xr.DataArray(np.array([1, 2, np.nan, np.nan, np.nan, 6],
                               dtype=np.float64), dims='x')

    actual = da.interpolate_na(dim='x', limit=None)
    assert actual.isnull().sum() == 0

    actual = da.interpolate_na(dim='x', limit=2)
    expected = xr.DataArray(np.array([1, 2, 3, 4, np.nan, 6],
                                     dtype=np.float64), dims='x')

    assert_equal(actual, expected)


@requires_scipy
def test_interpolate_methods():
    for method in ['linear', 'nearest', 'zero', 'slinear', 'quadratic',
                   'cubic']:
        kwargs = {}
        da = xr.DataArray(np.array([0, 1, 2, np.nan, np.nan, np.nan, 6, 7, 8],
                                   dtype=np.float64), dims='x')
        actual = da.interpolate_na('x', method=method, **kwargs)
        assert actual.isnull().sum() == 0

        actual = da.interpolate_na('x', method=method, limit=2, **kwargs)
        assert actual.isnull().sum() == 1


@requires_scipy
def test_interpolators():
    for method, interpolator in [('linear', NumpyInterpolator),
                                 ('linear', ScipyInterpolator),
                                 ('spline', SplineInterpolator)]:
        xi = np.array([-1, 0, 1, 2, 5], dtype=np.float64)
        yi = np.array([-10, 0, 10, 20, 50], dtype=np.float64)
        x = np.array([3, 4], dtype=np.float64)

        f = interpolator(xi, yi, method=method)
        out = f(x)
        assert pd.isnull(out).sum() == 0


def test_interpolate_use_coordinate():
    xc = xr.Variable('x', [100, 200, 300, 400, 500, 600])
    da = xr.DataArray(np.array([1, 2, np.nan, np.nan, np.nan, 6],
                               dtype=np.float64),
                      dims='x', coords={'xc': xc})

    # use_coordinate == False is same as using the default index
    actual = da.interpolate_na(dim='x', use_coordinate=False)
    expected = da.interpolate_na(dim='x')
    assert_equal(actual, expected)

    # possible to specify non index coordinate
    actual = da.interpolate_na(dim='x', use_coordinate='xc')
    expected = da.interpolate_na(dim='x')
    assert_equal(actual, expected)

    # possible to specify index coordinate by name
    actual = da.interpolate_na(dim='x', use_coordinate='x')
    expected = da.interpolate_na(dim='x')
    assert_equal(actual, expected)


@requires_dask
def test_interpolate_dask():
    da, _ = make_interpolate_example_data((40, 40), 0.5)
    da = da.chunk({'x': 5})
    actual = da.interpolate_na('time')
    expected = da.load().interpolate_na('time')
    assert isinstance(actual.data, dask_array_type)
    assert_equal(actual.compute(), expected)

    # with limit
    da = da.chunk({'x': 5})
    actual = da.interpolate_na('time', limit=3)
    expected = da.load().interpolate_na('time', limit=3)
    assert isinstance(actual.data, dask_array_type)
    assert_equal(actual, expected)


@requires_dask
def test_interpolate_dask_raises_for_invalid_chunk_dim():
    da, _ = make_interpolate_example_data((40, 40), 0.5)
    da = da.chunk({'time': 5})
    with raises_regex(ValueError, "dask='parallelized' consists of multiple"):
        da.interpolate_na('time')


@requires_bottleneck
def test_ffill():
    da = xr.DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims='x')
    expected = xr.DataArray(np.array([4, 5, 5], dtype=np.float64), dims='x')
    actual = da.ffill('x')
    assert_equal(actual, expected)


@requires_bottleneck
@requires_dask
def test_ffill_dask():
    da, _ = make_interpolate_example_data((40, 40), 0.5)
    da = da.chunk({'x': 5})
    actual = da.ffill('time')
    expected = da.load().ffill('time')
    assert isinstance(actual.data, dask_array_type)
    assert_equal(actual, expected)

    # with limit
    da = da.chunk({'x': 5})
    actual = da.ffill('time', limit=3)
    expected = da.load().ffill('time', limit=3)
    assert isinstance(actual.data, dask_array_type)
    assert_equal(actual, expected)


@requires_bottleneck
@requires_dask
def test_bfill_dask():
    da, _ = make_interpolate_example_data((40, 40), 0.5)
    da = da.chunk({'x': 5})
    actual = da.bfill('time')
    expected = da.load().bfill('time')
    assert isinstance(actual.data, dask_array_type)
    assert_equal(actual, expected)

    # with limit
    da = da.chunk({'x': 5})
    actual = da.bfill('time', limit=3)
    expected = da.load().bfill('time', limit=3)
    assert isinstance(actual.data, dask_array_type)
    assert_equal(actual, expected)


@requires_bottleneck
def test_ffill_bfill_nonans():

    vals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    expected = xr.DataArray(vals, dims='x')

    actual = expected.ffill(dim='x')
    assert_equal(actual, expected)

    actual = expected.bfill(dim='x')
    assert_equal(actual, expected)


@requires_bottleneck
def test_ffill_bfill_allnans():

    vals = np.full(6, np.nan, dtype=np.float64)
    expected = xr.DataArray(vals, dims='x')

    actual = expected.ffill(dim='x')
    assert_equal(actual, expected)

    actual = expected.bfill(dim='x')
    assert_equal(actual, expected)


@requires_bottleneck
def test_ffill_functions(da):
    result = da.ffill('time')
    assert result.isnull().sum() == 0


@requires_bottleneck
def test_ffill_limit():
    da = xr.DataArray(
        [0, np.nan, np.nan, np.nan, np.nan, 3, 4, 5, np.nan, 6, 7],
        dims='time')
    result = da.ffill('time')
    expected = xr.DataArray([0, 0, 0, 0, 0, 3, 4, 5, 5, 6, 7], dims='time')
    assert_array_equal(result, expected)

    result = da.ffill('time', limit=1)
    expected = xr.DataArray(
        [0, 0, np.nan, np.nan, np.nan, 3, 4, 5, 5, 6, 7], dims='time')


def test_interpolate_dataset(ds):
    actual = ds.interpolate_na(dim='time')
    # no missing values in var1
    assert actual['var1'].count('time') == actual.dims['time']

    # var2 should be the same as it was
    assert_array_equal(actual['var2'], ds['var2'])


@requires_bottleneck
def test_ffill_dataset(ds):
    ds.ffill(dim='time')


@requires_bottleneck
def test_bfill_dataset(ds):
    ds.ffill(dim='time')
