import numpy as np
import xarray as xr
from xarray.core.groupby import _consolidate_slices

import pytest


def test_consolidate_slices():

    assert _consolidate_slices([slice(3), slice(3, 5)]) == [slice(5)]
    assert _consolidate_slices([slice(2, 3), slice(3, 6)]) == [slice(2, 6)]
    assert (_consolidate_slices([slice(2, 3, 1), slice(3, 6, 1)])
          == [slice(2, 6, 1)])

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


# TODO: move other groupby tests from test_dataset and test_dataarray over here
