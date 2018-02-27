from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from xarray.core import dtypes


@pytest.mark.parametrize("args, expected", [
    ([np.bool], np.bool),
    ([np.bool, np.string_], np.object_),
    ([np.float32, np.float64], np.float64),
    ([np.float32, np.string_], np.object_),
    ([np.unicode_, np.int64], np.object_),
    ([np.unicode_, np.unicode_], np.unicode_),
    ([np.bytes_, np.unicode_], np.object_),
])
def test_result_type(args, expected):
    actual = dtypes.result_type(*args)
    assert actual == expected


def test_result_type_scalar():
    actual = dtypes.result_type(np.arange(3, dtype=np.float32), np.nan)
    assert actual == np.float32


def test_result_type_dask_array():
    # verify it works without evaluating dask arrays
    da = pytest.importorskip('dask.array')
    dask = pytest.importorskip('dask')

    def error():
        raise RuntimeError

    array = da.from_delayed(dask.delayed(error)(), (), np.float64)
    with pytest.raises(RuntimeError):
        array.compute()

    actual = dtypes.result_type(array)
    assert actual == np.float64

    # note that this differs from the behavior for scalar numpy arrays, which
    # would get promoted to float32
    actual = dtypes.result_type(array, np.array([0.5, 1.0], dtype=np.float32))
    assert actual == np.float64


@pytest.mark.parametrize('obj', [1.0, np.inf, 'ab', 1.0 + 1.0j, True])
def test_inf(obj):
    assert dtypes.INF > obj
    assert dtypes.NINF < obj
