import pickle
from collections import OrderedDict
from contextlib import suppress
from distutils.version import LooseVersion
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

import xarray as xr
import xarray.ufuncs as xu
from xarray import DataArray, Dataset, Variable
from xarray.tests import mock
from xarray.core.npcompat import IS_NEP18_ACTIVE


sparse = pytest.importorskip('sparse')


@pytest.mark.parametrize("func", [
    lambda s: s[5:15, 5:15],
    lambda s: s + 1,
    lambda s: s + s,
    lambda s: 2 * s,
    lambda s: np.sin(s),
    lambda s: np.sum(s, axis=0),
    lambda s: np.sum(s, axis=0, keepdims=True),
    pytest.param(lambda s: s.groupby('dim_0').sum(),
                 marks=pytest.mark.xfail)
])
@pytest.mark.skipif(not IS_NEP18_ACTIVE,
                    reason="NUMPY_EXPERIMENTAL_ARRAY_FUNCTION is not enabled")
def test_sparse(func):

    S = sparse.random((100, 100))
    A = xr.DataArray(S)
    assert isinstance(func(A).data, sparse.SparseArray)
