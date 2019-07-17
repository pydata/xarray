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
from sparse.utils import assert_eq

from . import (
    assert_allclose, assert_array_equal, assert_equal, assert_frame_equal,
    assert_identical, raises_regex)


def xfail(param):
    return pytest.param(param, marks=pytest.mark.xfail)


sparse = pytest.importorskip('sparse')


@pytest.mark.parametrize("func", [
    lambda s: s[5:15, 5:15],
    lambda s: s + 1,
    lambda s: s + s,
    lambda s: 2 * s,
    lambda s: np.sin(s),
    lambda s: s.sum(skipna=True, axis=0),
    lambda s: s.sum(skipna=True, axis=1),
    lambda s: s.sum(skipna=True, axis=0, keepdims=True),
    xfail(lambda s: s.sum(skipna=False, axis=0)),
    xfail(lambda s: s.sum(skipna=True)),
    xfail(lambda s: s.groupby('dim_0').sum())
])
@pytest.mark.skipif(not IS_NEP18_ACTIVE,
                    reason="NUMPY_EXPERIMENTAL_ARRAY_FUNCTION is not enabled")
def test_sparse(func):

    S = sparse.random((10, 10), random_state=0, density=0.2)
    A = xr.DataArray(S)
    assert isinstance(func(A).data, sparse.SparseArray)

    x = xr.Variable('x', sparse.COO.from_numpy([1, 0, 2, 0, 0, 3, 0, 4]))
    y = xr.Variable('y', sparse.COO.from_numpy([0, 4, 0, 3, 2, 0, 0, 1]))
    data = sparse.random((8, 8), random_state=0, density=0.2)
    A = xr.DataArray(data, dims=['x', 'y'])
    A.coords['xx'] = x
    A.coords['yy'] = y
    assert isinstance(func(A).data, sparse.SparseArray)


@pytest.mark.skipif(not IS_NEP18_ACTIVE,
                    reason="NUMPY_EXPERIMENTAL_ARRAY_FUNCTION is not enabled")
class TestSparseVariable:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.data = sparse.random((4, 6), random_state=0, density=0.5)
        self.var = xr.Variable(('x', 'y'), self.data)

    def test_copy(self):
        assert_identical(self.var, self.var.copy())
        assert_identical(self.var, self.var.copy(deep=True))

    def test_indexing(self):
        assert_eq(self.var[0].data, self.data[0])
        assert_eq(self.var[:1].data, self.data[:1])
        assert_eq(self.var[:1].data, self.data[:1])
        assert_eq(self.var[[0, 1, 2]].data, self.data[[0, 1, 2]])
        with raises_regex(TypeError, 'does not support item assignment'):
            self.var[:1] = 0

    # def test_squeeze(self):
    #     assert_eq(self.var[0].squeeze(), self.data[0].squeeze())

    def test_equals(self):
        v = self.var
        assert isinstance(v.data, sparse.SparseArray)
        assert v.equals(v)
        assert v.identical(v)

    def test_transpose(self):
        assert_eq(self.var.T.data, self.data.T)

    # def test_shift(self):
    #     assert_eq(self.var.shift(x=2).data, self.data.shift(x=2))
    #     assert_eq(self.var.shift(x=-2).data, self.data.shift(x=-2))

    # # def test_roll(self):
    # #     assert_eq(self.var.roll(x=2).data, self.data.roll(x=2))

    def test_unary_op(self):
        assert_eq(-self.var.data, -self.data)
        assert_eq(abs(self.var).data, abs(self.data))
        assert_eq(self.var.round().data, self.data.round())

    def test_binary_op(self):
        assert_eq((2 * self.var).data, 2 * self.data)
        assert_eq((self.var + self.var).data, self.data + self.data)
        # assert_eq((self.var[0] + self.var).data, self.data[0] + self.data)

    # def test_repr(self):
    #     pass

    # def test_pickle(self):
    #     pass

    # def test_reduce(self):
    #     pass

    # def test_missing_values(self):
    #     pass

    def test_concat(self):
        v = self.var
        assert_eq(self.data, Variable.concat([v[:2], v[2:]], 'x').data)
        assert_eq(self.data[:2], Variable.concat([v[0], v[1]], 'x').data)

    def test_univariate_ufunc(self):
        assert_eq(np.sin(self.data), xu.sin(self.var).data)

    def test_bivariate_ufunc(self):
        assert_eq(np.maximum(self.data, 0), xu.maximum(self.var, 0).data)
        assert_eq(np.maximum(self.data, 0), xu.maximum(0, self.var).data)


# class TestSparseDataArrayAndDataset:
#     @pytest.fixture(autouse=True)
#     def setUp(self):
#         self.data = sparse.random((4, 6), random_state=0)

#     def test_concat(self):
#         pass

#     def test_groupby(self):
#         pass

#     def test_groupby_first(self):
#         pass

#     def test_reindex(self):
#         pass

#     def test_to_dataset_roundtrip(self):
#         pass

#     def test_merge(self):
#         pass

#     def test_ufuncs(self):
#         pass

#     def test_where_dispatching(self):
#         pass

#     def test_stack(self):
#         pass

#     def test_dot(self):
#         pass

#     def test_dataarray_repr(self):
#         pass

#     def test_dataset_rep(self):
#         pass

#     def test_dataarray_pickle(self):
#         pass

#     def test_dataset_pickle(self):
#         pass

#     def test_dataarray_getattr(self):
#         pass

#     def test_dataset_getattr(self):
#         pass

#     def test_values(self):
#         pass
