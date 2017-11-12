from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import pytest

from xarray import DataArray

from xarray.core.missing import (NumpyInterpolator, ScipyInterpolator,
                                 FromDerivativesInterpolator,
                                 SplineInterpolator)

from xarray.tests import (
    assert_identical, assert_equal, raises_regex, requires_scipy,
    requires_bottleneck, assert_array_equal)


@pytest.fixture(params=[1])
def da(request):
    if request.param == 1:
        times = pd.date_range('2000-01-01', freq='1D', periods=21)
        values = np.random.random((3, 21, 4))
        da = DataArray(values, dims=('a', 'time', 'x'))
        da['time'] = times
        return da

    if request.param == 2:
        return DataArray(
            [0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7],
            dims='time')


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
    da = DataArray(vals, dims=('time', 'x'), coords=coords)
    df = da.to_pandas()
    return da, df


@requires_scipy
@pytest.mark.parametrize('shape', [(8, 8), (1, 20), (20, 1), (100, 100)])
@pytest.mark.parametrize('frac_nan', [0, 0.5, 1])
@pytest.mark.parametrize('method', ['linear',  'nearest', 'zero', 'slinear',
                                    'quadratic', 'cubic'])
def test_interpolate_pd_compat(shape, frac_nan, method):
    da, df = make_interpolate_example_data(shape, frac_nan)

    for dim in ['time', 'x']:
        actual = da.interpolate_na(method=method, dim=dim)
        expected = df.interpolate(method=method, axis=da.get_axis_num(dim))
        np.testing.assert_allclose(actual.values, expected.values)


@requires_scipy
@pytest.mark.parametrize('shape', [(8, 8), (1, 20), (20, 1)])
@pytest.mark.parametrize('frac_nan', [0, 0.5, 1])
@pytest.mark.parametrize('method', ['time', 'index', 'values', 'linear',
                                    'nearest', 'zero', 'slinear',
                                    'quadratic', 'cubic'])
def test_interpolate_pd_compat_non_uniform_index(shape, frac_nan, method):
    # translate pandas syntax to xarray equivalent
    xmethod = method
    use_coordinate = False
    if method in ['time', 'index', 'values']:
        use_coordinate = True
        xmethod = 'linear'
    elif method in ['nearest', 'slinear', 'quadratic', 'cubic']:
        use_coordinate = True

    da, df = make_interpolate_example_data(shape, frac_nan, non_uniform=True)
    for dim in ['time', 'x']:
        if method == 'time' and dim != 'time':
            continue
        actual = da.interpolate_na(method=xmethod, dim=dim,
                                   use_coordinate=use_coordinate)
        expected = df.interpolate(method=method, axis=da.get_axis_num(dim))
        np.testing.assert_allclose(actual.values, expected.values)


@requires_scipy
@pytest.mark.parametrize('shape', [(8, 8), (100, 100)])
@pytest.mark.parametrize('frac_nan', [0, 0.5, 1])
@pytest.mark.parametrize('order', [1, 2, 3])
def test_interpolate_pd_compat_polynomial(shape, frac_nan, order):
    da, df = make_interpolate_example_data(shape, frac_nan)

    for dim in ['time', 'x']:
        actual = da.interpolate_na(method='polynomial', order=order, dim=dim,
                                   use_coordinate=False)
        expected = df.interpolate(method='polynomial', order=order,
                                  axis=da.get_axis_num(dim))
        np.testing.assert_allclose(actual.values, expected.values)


@requires_scipy
def test_interpolate_unsorted_index_raises():
    vals = np.array([1, 2, 3], dtype=np.float64)
    expected = DataArray(vals, dims=('x'), coords={'x': [2, 1, 3]})
    with raises_regex(ValueError, 'must be monotonicly increasing'):
        expected.interpolate_na(dim='x', method='index')


@requires_scipy
def test_interpolate_no_dim_raises():
    da = DataArray(np.array([1, 2, np.nan, 5], dtype=np.float64), dims=('x'))
    with raises_regex(NotImplementedError, 'dim is a required argument'):
        da.interpolate_na(method='linear')


@requires_scipy
def test_interpolate_kwargs():
    da = DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims=('x'))
    expected = DataArray(np.array([4, 5, 6], dtype=np.float64), dims=('x'))
    actual = da.interpolate_na(dim='x', fill_value='extrapolate')
    assert_equal(actual, expected)

    expected = DataArray(np.array([4, 5, -999], dtype=np.float64), dims=('x'))
    actual = da.interpolate_na(dim='x', fill_value=-999)
    assert_equal(actual, expected)


@requires_scipy
def test_interpolate():

    vals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    expected = DataArray(vals, dims=('x'))
    mvals = vals.copy()
    mvals[2] = np.nan
    missing = DataArray(mvals, dims=('x'))

    actual = missing.interpolate_na(dim='x')

    assert_equal(actual, expected)


@requires_scipy
def test_interpolate_nonans():

    vals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    expected = DataArray(vals, dims=('x'))
    actual = expected.interpolate_na(dim='x')
    assert_equal(actual, expected)


@requires_scipy
def test_interpolate_allnans():
    vals = np.full(6, np.nan, dtype=np.float64)
    expected = DataArray(vals, dims=('x'))
    actual = expected.interpolate_na(dim='x')

    assert_equal(actual, expected)


def test_interpolate_limits():
    da = DataArray(np.array([1, 2, np.nan, np.nan, np.nan, 6],
                            dtype=np.float64), dims=('x'))

    actual = da.interpolate_na(dim='x', limit=None)
    assert actual.isnull().sum() == 0

    actual = da.interpolate_na(dim='x', limit=2)
    expected = DataArray(np.array([1, 2, 3, 4, np.nan, 6],
                                  dtype=np.float64), dims=('x'))

    assert_equal(actual, expected)


@pytest.mark.parametrize(
    'kind, interpolator',
    [('linear', NumpyInterpolator), ('linear', ScipyInterpolator),
     ('spline', SplineInterpolator)])
def test_interpolators(kind, interpolator):
    xi = np.array([-1, 0, 1, 2, 5], dtype=np.float64)
    yi = np.array([-10, 0, 10, 20, 50], dtype=np.float64)
    x = np.array([3, 4], dtype=np.float64)

    f = interpolator(xi, yi, kind=kind)
    out = f(x)
    assert pd.isnull(out).sum() == 0


@requires_bottleneck
def test_ffill():
    da = DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims=('x'))
    expected = DataArray(np.array([4, 5, 5], dtype=np.float64), dims=('x'))
    actual = da.ffill('x')
    assert_equal(actual, expected)


@requires_bottleneck
def test_ffill_bfill_nonans():

    vals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    expected = DataArray(vals, dims='x')

    actual = expected.ffill(dim='x')
    assert_equal(actual, expected)

    actual = expected.bfill(dim='x')
    assert_equal(actual, expected)


@requires_bottleneck
def test_ffill_bfill_allnans():

    vals = np.full(6, np.nan, dtype=np.float64)
    expected = DataArray(vals, dims='x')

    actual = expected.ffill(dim='x')
    assert_equal(actual, expected)

    actual = expected.bfill(dim='x')
    assert_equal(actual, expected)


@pytest.mark.parametrize('da', (1, 2), indirect=True)
def test_ffill_functions(da):
    result = da.ffill('time')
    assert result.isnull().sum() == 0


def test_ffill_limit():
    da = DataArray(
        [0, np.nan, np.nan, np.nan, np.nan, 3, 4, 5, np.nan, 6, 7],
        dims='time')
    result = da.ffill('time')
    expected = DataArray([0, 0, 0, 0, 0, 3, 4, 5, 5, 6, 7], dims='time')
    assert_array_equal(result, expected)

    result = da.ffill('time', limit=1)
    expected = DataArray(
        [0, 0, np.nan, np.nan, np.nan, 3, 4, 5, 5, 6, 7], dims='time')
