from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pytest
import numpy as np
from numpy import array, nan
from . import assert_array_equal
from xarray.core.duck_array_ops import (
    first, last, count, mean, array_notnull_equiv,
)
from xarray import DataArray

from . import TestCase, raises_regex, has_dask


class TestOps(TestCase):
    def setUp(self):
        self.x = array([[[nan,  nan,   2.,  nan],
                         [nan,   5.,   6.,  nan],
                         [8.,   9.,  10.,  nan]],

                        [[nan,  13.,  14.,  15.],
                         [nan,  17.,  18.,  nan],
                         [nan,  21.,  nan,  nan]]])

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

    def test_all_nan_arrays(self):
        assert np.isnan(mean([np.nan, np.nan]))


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
    shapes = [15, 30, 10][:dim_num]
    dims = ('x', 'y', 'z')[:dim_num]
    da = DataArray(rng.randn(*shapes), dims=dims,
                   coords={'x': np.arange(15)}, name='da').astype(dtype)

    if contains_nan:
        da = da.reindex(x=np.arange(20))
    if dask and has_dask:
        chunks = {d: 5 for d in dims}
        da = da.chunk(chunks)

    return da


def assert_allclose_with_nan(a, b, **kwargs):
    """ Extension of np.allclose with nan-including array """
    for a1, b1 in zip(a.ravel(), b.ravel()):
        assert (np.isnan(a1) and np.isnan(b1)) or np.allclose(a1, b1, **kwargs)


@pytest.mark.parametrize('dim_num', [1, 2, 3])
@pytest.mark.parametrize('dtype', [float, int, np.float32, np.bool_])
@pytest.mark.parametrize('dask', [False, True])
@pytest.mark.parametrize('func', ['sum', 'min', 'max', 'mean'])
@pytest.mark.parametrize('skipna', [False, True])
@pytest.mark.parametrize('aggdim', [None, 'x', 'y'])
def test_reduce(dim_num, dtype, dask, func, skipna, aggdim):

    if aggdim == 'y' and dim_num < 2:
        return

    if dtype == np.bool_ and func == 'mean':
        return  # numpy does not support this

    da = construct_dataarray(dim_num, dtype, contains_nan=True, dask=dask)
    axis = None if aggdim is None else da.get_axis_num(aggdim)

    if dask and not has_dask:
        return

    try:
        if skipna:
            expected = getattr(np, 'nan{}'.format(func))(da.values, axis=axis)
        else:
            expected = getattr(np, func)(da.values, axis=axis)

        actual = getattr(da, func)(skipna=skipna, dim=aggdim)
        assert_allclose_with_nan(actual.values, np.array(expected),
                                 rtol=1.0e-4)
    except (TypeError, AttributeError):
        # TODO currently, numpy does not support nanmean for object dtype
        pass

    # compatible with pandas for 1d case
    if dim_num == 1 or aggdim is None:
        se = da.to_dataframe()
        actual = getattr(da, func)(skipna=skipna, dim=aggdim)
        expected = getattr(se, func)(skipna=skipna)
        assert_allclose_with_nan(actual.values, np.array(expected))

    # without nan
    da = construct_dataarray(dim_num, dtype, contains_nan=False, dask=dask)
    expected = getattr(np, 'nan{}'.format(func))(da.values)
    actual = getattr(da, func)(skipna=skipna)
    assert np.allclose(actual.values, np.array(expected))
