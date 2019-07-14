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


@pytest.mark.skipif(not IS_NEP18_ACTIVE,
                    reason="NUMPY_EXPERIMENTAL_ARRAY_FUNCTION is not enabled")
def test_sparse():

    S = sparse.random((100, 100))
    A = xr.DataArray(S)

    Asub = A[5:15, 5:15]
    Ssub = S[5:15, 5:15]

    assert isinstance(Asub.data, sparse.SparseArray)
    #assert np.allclose(Asub.data, Ssub)

    Asum = A.sum()
    #assert np.allclose(Asum.data, S.sum())

    Amean = A.mean()
    #assert np.allclose(Amean.data, S.mean())

    Asum0 = A.groupby('dim_0').sum()
    #assert np.allclose(Asum0, S.sum(axis=0))

    Asum1 = A.groupby('dim_1').sum()
    #assert np.allclose(Asum1, S.sum(axis=1))

