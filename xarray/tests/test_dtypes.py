from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from xarray.core import dtypes


@pytest.mark.parametrize("args, expected", [
    ([np.bool], np.bool),
    ([np.bool, np.string_], np.object),
    ([np.float32, np.float64], np.float64),
    ([np.float32, np.string_], np.object),
    ([np.unicode_, np.int64], np.object),
    ([np.unicode_, np.unicode_], np.unicode_),
])
def test_result_type(args, expected):
    actual = dtypes.result_type(*args)
    assert actual == expected


def test_result_type_scalar():
    actual = dtypes.result_type(np.arange(3, dtype=np.float32), np.nan)
    assert actual == np.float32
