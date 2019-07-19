from sparse.utils import assert_eq
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

from . import (
    assert_allclose, assert_array_equal, assert_equal, assert_frame_equal,
    assert_identical, raises_regex)

sparse = pytest.importorskip('sparse')


def xfail(param, **kwargs):
    return pytest.param(param, marks=pytest.mark.xfail(**kwargs))


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

    def test_equals(self):
        v = self.var
        assert isinstance(v.data, sparse.SparseArray)
        assert v.equals(v)
        assert v.identical(v)

    def test_transpose(self):
        assert_eq(self.var.T.data, self.data.T)

    def test_squeeze(self):
        v1 = xr.Variable(('x', 'y'), self.data.todense())
        v2 = xr.Variable(('x', 'y'), self.data)
        assert np.allclose(
            v1[[0]].squeeze(), v2[[0]].squeeze().data.todense(),
            equal_nan=True)

    def test_roll(self):
        v1 = xr.Variable(('x', 'y'), self.data.todense())
        v2 = xr.Variable(('x', 'y'), self.data)
        assert np.allclose(
            v1.roll(x=2).data, v2.roll(x=2).data.todense())

    def test_unary_op(self):
        assert_eq(-self.var.data, -self.data)
        assert_eq(abs(self.var).data, abs(self.data))
        assert_eq(self.var.round().data, self.data.round())

    def test_binary_op(self):
        assert_eq((2 * self.var).data, 2 * self.data)
        assert_eq((self.var + self.var).data, self.data + self.data)
        # assert_eq((self.var[0] + self.var).data, self.data[0] + self.data)

    def test_repr(self):
        expected = dedent("""\
        <xarray.Variable (x: 4, y: 6)>
        <COO: shape=(4, 6), dtype=float64, nnz=12, fill_value=0.0>""")
        assert expected == repr(self.var)

    def test_pickle(self):
        v1 = self.var
        v2 = pickle.loads(pickle.dumps(v1))
        assert_eq(v1.data, v2.data)

    @pytest.mark.parametrize("func", [
        lambda s: s + 1,
        lambda s: s + s,
        lambda s: 2 * s,
        lambda s: np.sin(s),
        lambda s: s.sum(skipna=True, axis=0),
        lambda s: s.sum(skipna=True, axis=1),
        lambda s: s.sum(skipna=True, axis=0, keepdims=True),
        xfail(lambda s: s.sum(skipna=False, axis=0)),
        xfail(lambda s: s.sum(skipna=True),
              reason="Full reduction returns a dense scalar"),
    ])
    def test_reduce(self, func):
        assert isinstance(func(self.var).data, sparse.SparseArray)

    def test_missing_values(self):
        a = np.array([0, 1, np.nan, 3])
        s = sparse.COO.from_numpy(a)
        var_s = Variable('x', s)
        assert np.all(var_s.fillna(2).data.todense() == np.arange(4))
        assert np.all(var_s.count() == 3)

    def test_concat(self):
        v = self.var
        assert_eq(self.data, Variable.concat([v[:2], v[2:]], 'x').data)
        assert_eq(self.data[:2], Variable.concat([v[0], v[1]], 'x').data)

    def test_univariate_ufunc(self):
        assert_eq(np.sin(self.data), xu.sin(self.var).data)

    def test_bivariate_ufunc(self):
        assert_eq(np.maximum(self.data, 0), xu.maximum(self.var, 0).data)
        assert_eq(np.maximum(self.data, 0), xu.maximum(0, self.var).data)

    @pytest.mark.xfail(
        reason="can't concat: 'filler' is ndarray and 'trimmed_data' "
               "is sparse")
    def test_shift(self):
        v1 = xr.Variable(('x', 'y'), self.data.todense())
        v2 = xr.Variable(('x', 'y'), self.data)
        assert np.allclose(
            v1.shift(x=2).data, v2.shift(x=2).data.todense(),
            equal_nan=True)
        assert np.allclose(
            v1.shift(x=-2).data, v2.shift(x=-2).data.todense(),
            equal_nan=True)

    @pytest.mark.xfail(
        reason="sparse.COO objects currently do not accept more than one "
               "iterable index at a time")
    def test_align(self):
        S = self.data
        A1 = xr.DataArray(self.data, dims=['x', 'y'], coords={
            'x': np.arange(S.shape[0]),
            'y': np.arange(S.shape[1])
        })

        A2 = xr.DataArray(self.data, dims=['x', 'y'], coords={
            'x': np.arange(1, S.shape[0] + 1),
            'y': np.arange(1, S.shape[1] + 1)
        })

        B1, B2 = xr.align(A1, A2, join='inner')
        assert np.all(B1.coords['x'] == np.arange(1, S.shape[0]))
        assert np.all(B1.coords['y'] == np.arange(1, S.shape[0]))
        assert np.all(B1.coords['x'] == B2.coords['x'])
        assert np.all(B1.coords['y'] == B2.coords['y'])


class TestSparseDataArrayAndDataset:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.sp_ar = sparse.random((4, 6), random_state=0, density=0.5)
        self.sp_xr = xr.DataArray(self.sp_ar, coords={'x': range(4)},
                                  dims=('x', 'y'), name='foo')
        self.ds_ar = self.sp_ar.todense()
        self.ds_xr = xr.DataArray(self.ds_ar, coords={'x': range(4)},
                                  dims=('x', 'y'), name='foo')

    def test_to_dataset_roundtrip(self):
        x = self.sp_xr
        assert_equal(x, x.to_dataset('x').to_array('x'))

    def test_values(self):
        a = xr.DataArray(sparse.COO.from_numpy([1, 2]))
        with raises_regex(RuntimeError,
                          'Cannot convert a sparse array to dense'):
            a.values

    def test_align(self):
        a1 = xr.DataArray(
            sparse.COO.from_numpy(np.arange(4)),
            dims=['x'],
            coords={'x': ['a', 'b', 'c', 'd']})
        b1 = xr.DataArray(
            sparse.COO.from_numpy(np.arange(4)),
            dims=['x'],
            coords={'x': ['a', 'b', 'd', 'e']})
        a2, b2 = xr.align(a1, b1)
        assert isinstance(a2.data, sparse.SparseArray)
        assert isinstance(b2.data, sparse.SparseArray)
        assert np.all(a2.coords['x'].data == ['a', 'b', 'd'])
        assert np.all(b2.coords['x'].data == ['a', 'b', 'd'])

    @pytest.mark.xfail(reason="fill value leads to sparse-dense operation")
    def test_align_outer(self):
        a1 = xr.DataArray(
            sparse.COO.from_numpy(np.arange(4)),
            dims=['x'],
            coords={'x': ['a', 'b', 'c', 'd']})
        b1 = xr.DataArray(
            sparse.COO.from_numpy(np.arange(4)),
            dims=['x'],
            coords={'x': ['a', 'b', 'd', 'e']})
        a2, b2 = xr.align(a1, b1, join='outer')
        assert isinstance(a2.data, sparse.SparseArray)
        assert isinstance(b2.data, sparse.SparseArray)
        assert np.all(a2.coords['x'].data == ['a', 'b', 'c', 'd'])
        assert np.all(b2.coords['x'].data == ['a', 'b', 'c', 'd'])

    def test_concat(self):
        ds1 = xr.Dataset(data_vars={'d': self.sp_xr})
        ds2 = xr.Dataset(data_vars={'d': self.sp_xr})
        ds3 = xr.Dataset(data_vars={'d': self.sp_xr})
        out = xr.concat([ds1, ds2, ds3], dim='x')
        assert_eq(
            out['d'].data,
            sparse.concatenate([self.sp_ar, self.sp_ar, self.sp_ar], axis=0)
        )

        out = xr.concat([self.sp_xr, self.sp_xr, self.sp_xr], dim='y')
        assert_eq(
            out.data,
            sparse.concatenate([self.sp_ar, self.sp_ar, self.sp_ar], axis=1)
        )

    def test_stack(self):
        data = np.random.normal(size=(2, 3, 4))

        arr = xr.DataArray(data, dims=('w', 'x', 'y'))
        stacked = arr.stack(z=('x', 'y'))

        z = pd.MultiIndex.from_product(
            [np.arange(3), np.arange(4)],
            names=['x', 'y'])
        expected = xr.DataArray(
            data.reshape((2, -1)),
            {'z': z},
            dims=['w', 'z'])

        assert_equal(expected, stacked)

    def test_ufuncs(self):
        x = self.sp_xr
        assert_equal(np.sin(x), xu.sin(x))

    def test_dataarray_repr(self):
        a = xr.DataArray(
            sparse.COO.from_numpy(np.ones((4))),
            dims=['x'],
            coords={'y': ('x', sparse.COO.from_numpy(np.arange(4)))})
        expected = dedent("""\
        <xarray.DataArray (x: 4)>
        <COO: shape=(4,), dtype=float64, nnz=4, fill_value=0.0>
        Coordinates:
            y        (x) int64 ...
        Dimensions without coordinates: x""")
        assert expected == repr(a)

    def test_dataset_repr(self):
        ds = xr.Dataset(
            data_vars={'a': ('x', sparse.COO.from_numpy(np.ones((4))))},
            coords={'y': ('x', sparse.COO.from_numpy(np.arange(4)))})
        expected = dedent("""\
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
            y        (x) int64 ...
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 ...""")
        assert expected == repr(ds)

    def test_dataarray_pickle(self):
        a1 = xr.DataArray(
            sparse.COO.from_numpy(np.ones((4))),
            dims=['x'],
            coords={'y': ('x', sparse.COO.from_numpy(np.arange(4)))})
        a2 = pickle.loads(pickle.dumps(a1))
        assert_identical(a1, a2)

    def test_dataset_pickle(self):
        ds1 = xr.Dataset(
            data_vars={'a': ('x', sparse.COO.from_numpy(np.ones((4))))},
            coords={'y': ('x', sparse.COO.from_numpy(np.arange(4)))})
        ds2 = pickle.loads(pickle.dumps(ds1))
        assert_identical(ds1, ds2)

    @pytest.mark.xfail
    def test_sparse_coords(self):
        xr.DataArray(
            sparse.COO.from_numpy(np.arange(4)),
            dims=['x'],
            coords={'x': sparse.COO.from_numpy([1, 2, 3, 4])})

    @pytest.mark.xfail(reason="No implementation of np.einsum")
    def test_dot(self):
        a1 = self.xp_xr.dot(self.xp_xr[0])
        a2 = self.sp_ar.dot(self.sp_ar[0])
        assert_equal(a1, a2)

    @pytest.mark.xfail(reason="Groupby reductions produce dense output")
    def test_groupby(self):
        x1 = self.ds_xr
        x2 = self.sp_xr
        m1 = x1.groupby('x').mean(xr.ALL_DIMS)
        m2 = x2.groupby('x').mean(xr.ALL_DIMS)
        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail(reason="Groupby reductions produce dense output")
    def test_groupby_first(self):
        x = self.sp_xr.copy()
        x.coords['ab'] = ('x', ['a', 'a', 'b', 'b'])
        x.groupby('ab').first()
        x.groupby('ab').first(skipna=False)

    @pytest.mark.xfail
    def test_reindex(self):
        x1 = self.ds_xr
        x2 = self.sp_xr
        for kwargs in [{'x': [2, 3, 4]},
                       {'x': [1, 100, 2, 101, 3]},
                       {'x': [2.5, 3, 3.5], 'y': [2, 2.5, 3]}]:
            m1 = x1.reindex(**kwargs)
            m2 = x2.reindex(**kwargs)
            assert np.allclose(m1, m2, equal_nan=True)

    @pytest.mark.xfail
    def test_merge(self):
        x = self.sp_xr
        y = xr.merge([x, x.rename('bar')]).to_array()
        assert isinstance(y, sparse.SparseArray)

    @pytest.mark.xfail
    def test_where(self):
        a = np.arange(10)
        cond = a > 3
        xr.DataArray(a).where(cond)

        s = sparse.COO.from_numpy(a)
        cond = s > 3
        xr.DataArray(s).where(cond)

        x = xr.DataArray(s)
        cond = x > 3
        x.where(cond)
