from __future__ import absolute_import, division, print_function

from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import pytest
from textwrap import dedent
from numpy import array, nan
import warnings

from xarray import DataArray, Dataset, concat
from xarray.core import duck_array_ops, dtypes
from xarray.core.duck_array_ops import (
    array_notnull_equiv, concatenate, count, first, gradient, last, mean,
    rolling_window, stack, where)
from xarray.core.pycompat import dask_array_type
from xarray.testing import assert_allclose, assert_equal

from . import (
    TestCase, assert_array_equal, has_dask, has_np113, raises_regex,
    requires_dask)


class TestOps(TestCase):

    def setUp(self):
        self.x = array([[[nan, nan, 2., nan],
                         [nan, 5., 6., nan],
                         [8., 9., 10., nan]],

                        [[nan, 13., 14., 15.],
                         [nan, 17., 18., nan],
                         [nan, 21., nan, nan]]])

    def test_first(self):
        expected_results = [array([[nan, 13, 2, 15],
                                   [nan, 5, 6, nan],
                                   [8, 9, 10, nan]]),
                            array([[8, 5, 2, nan],
                                   [nan, 13, 14, 15]]),
                            array([[2, 5, 8],
                                   [13, 17, 21]])]
        for axis, expected in zip([0, 1, 2, -3, -2, -1],
                                  2 * expected_results):
            actual = first(self.x, axis)
            assert_array_equal(expected, actual)

        expected = self.x[0]
        actual = first(self.x, axis=0, skipna=False)
        assert_array_equal(expected, actual)

        expected = self.x[..., 0]
        actual = first(self.x, axis=-1, skipna=False)
        assert_array_equal(expected, actual)

        with raises_regex(IndexError, 'out of bounds'):
            first(self.x, 3)

    def test_last(self):
        expected_results = [array([[nan, 13, 14, 15],
                                   [nan, 17, 18, nan],
                                   [8, 21, 10, nan]]),
                            array([[8, 9, 10, nan],
                                   [nan, 21, 18, 15]]),
                            array([[2, 6, 10],
                                   [15, 18, 21]])]
        for axis, expected in zip([0, 1, 2, -3, -2, -1],
                                  2 * expected_results):
            actual = last(self.x, axis)
            assert_array_equal(expected, actual)

        expected = self.x[-1]
        actual = last(self.x, axis=0, skipna=False)
        assert_array_equal(expected, actual)

        expected = self.x[..., -1]
        actual = last(self.x, axis=-1, skipna=False)
        assert_array_equal(expected, actual)

        with raises_regex(IndexError, 'out of bounds'):
            last(self.x, 3)

    def test_count(self):
        assert 12 == count(self.x)

        expected = array([[1, 2, 3], [3, 2, 1]])
        assert_array_equal(expected, count(self.x, axis=-1))

    def test_where_type_promotion(self):
        result = where([True, False], [1, 2], ['a', 'b'])
        assert_array_equal(result, np.array([1, 'b'], dtype=object))

        result = where([True, False], np.array([1, 2], np.float32), np.nan)
        assert result.dtype == np.float32
        assert_array_equal(result, np.array([1, np.nan], dtype=np.float32))

    def test_stack_type_promotion(self):
        result = stack([1, 'b'])
        assert_array_equal(result, np.array([1, 'b'], dtype=object))

    def test_concatenate_type_promotion(self):
        result = concatenate([[1], ['b']])
        assert_array_equal(result, np.array([1, 'b'], dtype=object))

    def test_all_nan_arrays(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice')
            warnings.filterwarnings('ignore', 'Mean of empty slice')
            assert np.isnan(mean([np.nan, np.nan]))


def test_cumsum_1d():
    inputs = np.array([0, 1, 2, 3])
    expected = np.array([0, 1, 3, 6])
    actual = duck_array_ops.cumsum(inputs)
    assert_array_equal(expected, actual)

    actual = duck_array_ops.cumsum(inputs, axis=0)
    assert_array_equal(expected, actual)

    actual = duck_array_ops.cumsum(inputs, axis=-1)
    assert_array_equal(expected, actual)

    actual = duck_array_ops.cumsum(inputs, axis=(0,))
    assert_array_equal(expected, actual)

    actual = duck_array_ops.cumsum(inputs, axis=())
    assert_array_equal(inputs, actual)


def test_cumsum_2d():
    inputs = np.array([[1, 2], [3, 4]])

    expected = np.array([[1, 3], [4, 10]])
    actual = duck_array_ops.cumsum(inputs)
    assert_array_equal(expected, actual)

    actual = duck_array_ops.cumsum(inputs, axis=(0, 1))
    assert_array_equal(expected, actual)

    actual = duck_array_ops.cumsum(inputs, axis=())
    assert_array_equal(inputs, actual)


def test_cumprod_2d():
    inputs = np.array([[1, 2], [3, 4]])

    expected = np.array([[1, 2], [3, 2 * 3 * 4]])
    actual = duck_array_ops.cumprod(inputs)
    assert_array_equal(expected, actual)

    actual = duck_array_ops.cumprod(inputs, axis=(0, 1))
    assert_array_equal(expected, actual)

    actual = duck_array_ops.cumprod(inputs, axis=())
    assert_array_equal(inputs, actual)


class TestArrayNotNullEquiv():
    @pytest.mark.parametrize("arr1, arr2", [
        (np.array([1, 2, 3]), np.array([1, 2, 3])),
        (np.array([1, 2, np.nan]), np.array([1, np.nan, 3])),
        (np.array([np.nan, 2, np.nan]), np.array([1, np.nan, np.nan])),
    ])
    def test_equal(self, arr1, arr2):
        assert array_notnull_equiv(arr1, arr2)

    def test_some_not_equal(self):
        a = np.array([1, 2, 4])
        b = np.array([1, np.nan, 3])
        assert not array_notnull_equiv(a, b)

    def test_wrong_shape(self):
        a = np.array([[1, np.nan, np.nan, 4]])
        b = np.array([[1, 2], [np.nan, 4]])
        assert not array_notnull_equiv(a, b)

    @pytest.mark.parametrize("val1, val2, val3, null", [
        (1, 2, 3, None),
        (1., 2., 3., np.nan),
        (1., 2., 3., None),
        ('foo', 'bar', 'baz', None),
    ])
    def test_types(self, val1, val2, val3, null):
        arr1 = np.array([val1, null, val3, null])
        arr2 = np.array([val1, val2, null, null])
        assert array_notnull_equiv(arr1, arr2)


def construct_dataarray(dim_num, dtype, contains_nan, dask):
    # dimnum <= 3
    rng = np.random.RandomState(0)
    shapes = [16, 8, 4][:dim_num]
    dims = ('x', 'y', 'z')[:dim_num]

    if np.issubdtype(dtype, np.floating):
        array = rng.randn(*shapes).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        array = rng.randint(0, 10, size=shapes).astype(dtype)
    elif np.issubdtype(dtype, np.bool_):
        array = rng.randint(0, 1, size=shapes).astype(dtype)
    elif dtype == str:
        array = rng.choice(['a', 'b', 'c', 'd'], size=shapes)
    else:
        raise ValueError

    if contains_nan:
        inds = rng.choice(range(array.size), int(array.size * 0.2))
        dtype, fill_value = dtypes.maybe_promote(array.dtype)
        array = array.astype(dtype)
        array.flat[inds] = fill_value

    da = DataArray(array, dims=dims, coords={'x': np.arange(16)}, name='da')

    if dask and has_dask:
        chunks = {d: 4 for d in dims}
        da = da.chunk(chunks)

    return da


def from_series_or_scalar(se):
    try:
        return DataArray.from_series(se)
    except AttributeError:  # scalar case
        return DataArray(se)


def series_reduce(da, func, dim, **kwargs):
    """ convert DataArray to pd.Series, apply pd.func, then convert back to
    a DataArray. Multiple dims cannot be specified."""
    if dim is None or da.ndim == 1:
        se = da.to_series()
        return from_series_or_scalar(getattr(se, func)(**kwargs))
    else:
        da1 = []
        dims = list(da.dims)
        dims.remove(dim)
        d = dims[0]
        for i in range(len(da[d])):
            da1.append(series_reduce(da.isel(**{d: i}), func, dim, **kwargs))

        if d in da.coords:
            return concat(da1, dim=da[d])
        return concat(da1, dim=d)


def assert_dask_array(da, dask):
    if dask and da.ndim > 0:
        assert isinstance(da.data, dask_array_type)


@pytest.mark.parametrize('dim_num', [1, 2])
@pytest.mark.parametrize('dtype', [float, int, np.float32, np.bool_])
@pytest.mark.parametrize('dask', [False, True])
@pytest.mark.parametrize('func', ['sum', 'min', 'max', 'mean', 'var'])
# TODO test cumsum, cumprod
@pytest.mark.parametrize('skipna', [False, True])
@pytest.mark.parametrize('aggdim', [None, 'x'])
def test_reduce(dim_num, dtype, dask, func, skipna, aggdim):

    if aggdim == 'y' and dim_num < 2:
        pytest.skip('dim not in this test')

    if dtype == np.bool_ and func == 'mean':
        pytest.skip('numpy does not support this')

    if dask and not has_dask:
        pytest.skip('requires dask')

    if dask and skipna is False and dtype in [np.bool_]:
        pytest.skip('dask does not compute object-typed array')

    rtol = 1e-04 if dtype == np.float32 else 1e-05

    da = construct_dataarray(dim_num, dtype, contains_nan=True, dask=dask)
    axis = None if aggdim is None else da.get_axis_num(aggdim)

    # TODO: remove these after resolving
    # https://github.com/dask/dask/issues/3245
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        warnings.filterwarnings('ignore', 'All-NaN slice')
        warnings.filterwarnings('ignore', 'invalid value encountered in')

        if has_np113 and da.dtype.kind == 'O' and skipna:
            # Numpy < 1.13 does not handle object-type array.
            try:
                if skipna:
                    expected = getattr(np, 'nan{}'.format(func))(da.values,
                                                                 axis=axis)
                else:
                    expected = getattr(np, func)(da.values, axis=axis)

                actual = getattr(da, func)(skipna=skipna, dim=aggdim)
                assert_dask_array(actual, dask)
                assert np.allclose(actual.values, np.array(expected),
                                   rtol=1.0e-4, equal_nan=True)
            except (TypeError, AttributeError, ZeroDivisionError):
                # TODO currently, numpy does not support some methods such as
                # nanmean for object dtype
                pass

        actual = getattr(da, func)(skipna=skipna, dim=aggdim)

        # for dask case, make sure the result is the same for numpy backend
        expected = getattr(da.compute(), func)(skipna=skipna, dim=aggdim)
        assert_allclose(actual, expected, rtol=rtol)

        # make sure the compatiblility with pandas' results.
        if func == 'var':
            expected = series_reduce(da, func, skipna=skipna, dim=aggdim,
                                     ddof=0)
            assert_allclose(actual, expected, rtol=rtol)
            # also check ddof!=0 case
            actual = getattr(da, func)(skipna=skipna, dim=aggdim, ddof=5)
            if dask:
                assert isinstance(da.data, dask_array_type)
            expected = series_reduce(da, func, skipna=skipna, dim=aggdim,
                                     ddof=5)
            assert_allclose(actual, expected, rtol=rtol)
        else:
            expected = series_reduce(da, func, skipna=skipna, dim=aggdim)
            assert_allclose(actual, expected, rtol=rtol)

        # make sure the dtype argument
        if func not in ['max', 'min']:
            actual = getattr(da, func)(skipna=skipna, dim=aggdim, dtype=float)
            assert_dask_array(actual, dask)
            assert actual.dtype == float

        # without nan
        da = construct_dataarray(dim_num, dtype, contains_nan=False, dask=dask)
        actual = getattr(da, func)(skipna=skipna)
        if dask:
            assert isinstance(da.data, dask_array_type)
        expected = getattr(np, 'nan{}'.format(func))(da.values)
        if actual.dtype == object:
            assert actual.values == np.array(expected)
        else:
            assert np.allclose(actual.values, np.array(expected), rtol=rtol)


@pytest.mark.parametrize('dim_num', [1, 2])
@pytest.mark.parametrize('dtype', [float, int, np.float32, np.bool_, str])
@pytest.mark.parametrize('contains_nan', [True, False])
@pytest.mark.parametrize('dask', [False, True])
@pytest.mark.parametrize('func', ['min', 'max'])
@pytest.mark.parametrize('skipna', [False, True])
@pytest.mark.parametrize('aggdim', ['x', 'y'])
def test_argmin_max(dim_num, dtype, contains_nan, dask, func, skipna, aggdim):
    # pandas-dev/pandas#16830, we do not check consistency with pandas but
    # just make sure da[da.argmin()] == da.min()

    if aggdim == 'y' and dim_num < 2:
        pytest.skip('dim not in this test')

    if dask and not has_dask:
        pytest.skip('requires dask')

    if contains_nan:
        if not skipna:
            pytest.skip("numpy's argmin (not nanargmin) does not handle "
                        "object-dtype")
        if skipna and np.dtype(dtype).kind in 'iufc':
            pytest.skip("numpy's nanargmin raises ValueError for all nan axis")
    da = construct_dataarray(dim_num, dtype, contains_nan=contains_nan,
                             dask=dask)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice')

        actual = da.isel(**{aggdim: getattr(da, 'arg' + func)
                            (dim=aggdim, skipna=skipna).compute()})
        expected = getattr(da, func)(dim=aggdim, skipna=skipna)
        assert_allclose(actual.drop(actual.coords),
                        expected.drop(expected.coords))


def test_argmin_max_error():
    da = construct_dataarray(2, np.bool_, contains_nan=True, dask=False)
    da[0] = np.nan
    with pytest.raises(ValueError):
        da.argmin(dim='y')


@requires_dask
def test_isnull_with_dask():
    da = construct_dataarray(2, np.float32, contains_nan=True, dask=True)
    assert isinstance(da.isnull().data, dask_array_type)
    assert_equal(da.isnull().load(), da.load().isnull())


@pytest.mark.skipif(not has_dask, reason='This is for dask.')
@pytest.mark.parametrize('axis', [0, -1])
@pytest.mark.parametrize('window', [3, 8, 11])
@pytest.mark.parametrize('center', [True, False])
def test_dask_rolling(axis, window, center):
    import dask.array as da

    x = np.array(np.random.randn(100, 40), dtype=float)
    dx = da.from_array(x, chunks=[(6, 30, 30, 20, 14), 8])

    expected = rolling_window(x, axis=axis, window=window, center=center,
                              fill_value=np.nan)
    actual = rolling_window(dx, axis=axis, window=window, center=center,
                            fill_value=np.nan)
    assert isinstance(actual, da.Array)
    assert_array_equal(actual, expected)
    assert actual.shape == expected.shape

    # we need to take care of window size if chunk size is small
    # window/2 should be smaller than the smallest chunk size.
    with pytest.raises(ValueError):
        rolling_window(dx, axis=axis, window=100, center=center,
                       fill_value=np.nan)


@pytest.mark.skipif(not has_dask, reason='This is for dask.')
@pytest.mark.parametrize('axis', [0, -1, 1])
@pytest.mark.parametrize('edge_order', [1, 2])
def test_dask_gradient(axis, edge_order):
    import dask.array as da

    array = np.array(np.random.randn(100, 5, 40))
    x = np.exp(np.linspace(0, 1, array.shape[axis]))

    darray = da.from_array(array, chunks=[(6, 30, 30, 20, 14), 5, 8])
    expected = gradient(array, x, axis=axis, edge_order=edge_order)
    actual = gradient(darray, x, axis=axis, edge_order=edge_order)

    assert isinstance(actual, da.Array)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize('dim_num', [1, 2])
@pytest.mark.parametrize('dtype', [float, int, np.float32, np.bool_])
@pytest.mark.parametrize('dask', [False, True])
@pytest.mark.parametrize('func', ['sum', 'prod'])
@pytest.mark.parametrize('aggdim', [None, 'x'])
def test_min_count(dim_num, dtype, dask, func, aggdim):
    if dask and not has_dask:
        pytest.skip('requires dask')

    da = construct_dataarray(dim_num, dtype, contains_nan=True, dask=dask)
    min_count = 3

    actual = getattr(da, func)(dim=aggdim, skipna=True, min_count=min_count)

    if LooseVersion(pd.__version__) >= LooseVersion('0.22.0'):
        # min_count is only implenented in pandas > 0.22
        expected = series_reduce(da, func, skipna=True, dim=aggdim,
                                 min_count=min_count)
        assert_allclose(actual, expected)

    assert_dask_array(actual, dask)


@pytest.mark.parametrize('func', ['sum', 'prod'])
def test_min_count_dataset(func):
    da = construct_dataarray(2, dtype=float, contains_nan=True, dask=False)
    ds = Dataset({'var1': da}, coords={'scalar': 0})
    actual = getattr(ds, func)(dim='x', skipna=True, min_count=3)['var1']
    expected = getattr(ds['var1'], func)(dim='x', skipna=True, min_count=3)
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dtype', [float, int, np.float32, np.bool_])
@pytest.mark.parametrize('dask', [False, True])
@pytest.mark.parametrize('func', ['sum', 'prod'])
def test_multiple_dims(dtype, dask, func):
    if dask and not has_dask:
        pytest.skip('requires dask')
    da = construct_dataarray(3, dtype, contains_nan=True, dask=dask)

    actual = getattr(da, func)(('x', 'y'))
    expected = getattr(getattr(da, func)('x'), func)('y')
    assert_allclose(actual, expected)


def test_docs():
    # with min_count
    actual = DataArray.sum.__doc__
    expected = dedent("""\
        Reduce this DataArray's data by applying `sum` along some dimension(s).

        Parameters
        ----------
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `sum`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `sum`. Only one of the 'dim'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `sum` is calculated over axes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default None
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result will
            be NA. New in version 0.10.8: Added with the default being None.
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating `sum` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray object with `sum` applied to its data and the
            indicated dimension(s) removed.
        """)
    assert actual == expected

    # without min_count
    actual = DataArray.std.__doc__
    expected = dedent("""\
        Reduce this DataArray's data by applying `std` along some dimension(s).

        Parameters
        ----------
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `std`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `std`. Only one of the 'dim'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `std` is calculated over axes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating `std` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray object with `std` applied to its data and the
            indicated dimension(s) removed.
        """)
    assert actual == expected
