from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import pickle
import pytest
from copy import deepcopy
from textwrap import dedent
from distutils.version import LooseVersion

import xarray as xr

from xarray import (align, broadcast, Dataset, DataArray,
                    IndexVariable, Variable)
from xarray.coding.times import CFDatetimeCoder
from xarray.core.pycompat import iteritems, OrderedDict
from xarray.core.common import full_like
from xarray.tests import (
    TestCase, ReturnItem, source_ndarray, unittest, requires_dask,
    assert_identical, assert_equal, assert_allclose, assert_array_equal,
    raises_regex, requires_scipy, requires_bottleneck)
from xarray.core.ops import NAN_REDUCE_METHODS


def construct_dataarray(dtype, contains_nan, dask):
    da = DataArray(np.random.randn(15, 30), dims=('x', 'y'),
                   coords={'x': np.arange(15)}).astype(dtype)

    if contains_nan:
        da = da.reindex(x=np.arange(20))
    if dask:
        da = da.chunk({'x': 5, 'y': 10})

    return da


def assert_allclose_with_nan(a, b, **kwargs):
    """ Extension of np.allclose with nan-including array """
    index = ~np.isnan(a)
    print(a)
    print(b)
    assert index == ~np.isnan(b)
    assert np.allclose(a[index], b[index], **kwargs)


@pytest.mark.parametrize('dtype', [float, int, np.float32, np.bool_])
@pytest.mark.parametrize('contains_nan', [False, True])
@pytest.mark.parametrize('dask', [False, ])
@pytest.mark.parametrize('func', NAN_REDUCE_METHODS)
@pytest.mark.parametrize('skipna', [False, True])
@pytest.mark.parametrize('dim', [None, 'x', 'y'])
def test_reduce(dtype, contains_nan, dask, func, skipna, dim):
    if dask:  # TODO some reduce methods are not available for dask
        if func in ['sum']:
            return

    da = construct_dataarray(dtype, contains_nan, dask)

    if skipna:
        try:  # TODO currently, we only support methods that numpy supports
            expected = getattr(np, 'nan{}'.format(func))(da.values)
        except TypeError:
            with pytest.raises(NotImplementedError):
                actual = getattr(da, func)(skipna=skipna)
                return
    else:
        expected = getattr(np, func)(da.values)

    actual = getattr(da, func)(skipna=skipna)
    assert_allclose_with_nan(actual.values, np.array(expected))
