from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from . import assert_identical
from xarray.core.groupby import _consolidate_slices


def test_consolidate_slices():

    assert _consolidate_slices([slice(3), slice(3, 5)]) == [slice(5)]
    assert _consolidate_slices([slice(2, 3), slice(3, 6)]) == [slice(2, 6)]
    assert (_consolidate_slices([slice(2, 3, 1), slice(3, 6, 1)]) ==
            [slice(2, 6, 1)])

    slices = [slice(2, 3), slice(5, 6)]
    assert _consolidate_slices(slices) == slices

    with pytest.raises(ValueError):
        _consolidate_slices([slice(3), 4])


def test_multi_index_groupby_apply():
    # regression test for GH873
    ds = xr.Dataset({'foo': (('x', 'y'), np.random.randn(3, 4))},
                    {'x': ['a', 'b', 'c'], 'y': [1, 2, 3, 4]})
    doubled = 2 * ds
    group_doubled = (ds.stack(space=['x', 'y'])
                     .groupby('space')
                     .apply(lambda x: 2 * x)
                     .unstack('space'))
    assert doubled.equals(group_doubled)


def test_multi_index_groupby_sum():
    # regression test for GH873
    ds = xr.Dataset({'foo': (('x', 'y', 'z'), np.ones((3, 4, 2)))},
                    {'x': ['a', 'b', 'c'], 'y': [1, 2, 3, 4]})
    expected = ds.sum('z')
    actual = (ds.stack(space=['x', 'y'])
              .groupby('space')
              .sum('z')
              .unstack('space'))
    assert expected.equals(actual)


def test_groupby_da_datetime():
    # test groupby with a DataArray of dtype datetime for GH1132
    # create test data
    times = pd.date_range('2000-01-01', periods=4)
    foo = xr.DataArray([1, 2, 3, 4], coords=dict(time=times), dims='time')
    # create test index
    dd = times.to_pydatetime()
    reference_dates = [dd[0], dd[2]]
    labels = reference_dates[0:1] * 2 + reference_dates[1:2] * 2
    ind = xr.DataArray(labels, coords=dict(time=times), dims='time',
                       name='reference_date')
    g = foo.groupby(ind)
    actual = g.sum(dim='time')
    expected = xr.DataArray([3, 7],
                            coords=dict(reference_date=reference_dates),
                            dims='reference_date')
    assert actual.equals(expected)


def test_groupby_duplicate_coordinate_labels():
    # fix for http://stackoverflow.com/questions/38065129
    array = xr.DataArray([1, 2, 3], [('x', [1, 1, 2])])
    expected = xr.DataArray([3, 3], [('x', [1, 2])])
    actual = array.groupby('x').sum()
    assert expected.equals(actual)


def test_groupby_input_mutation():
    # regression test for GH2153
    array = xr.DataArray([1, 2, 3], [('x', [2, 2, 1])])
    array_copy = array.copy()
    expected = xr.DataArray([3, 3], [('x', [1, 2])])
    actual = array.groupby('x').sum()
    assert_identical(expected, actual)
    assert_identical(array, array_copy)  # should not modify inputs


# TODO: move other groupby tests from test_dataset and test_dataarray over here
