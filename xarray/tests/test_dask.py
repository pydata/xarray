from __future__ import absolute_import, division, print_function

import pickle
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

import xarray as xr
import xarray.ufuncs as xu
from xarray import DataArray, Dataset, Variable
from xarray.core.pycompat import OrderedDict, suppress
from xarray.tests import mock

from . import (
    TestCase, assert_allclose, assert_array_equal, assert_equal,
    assert_frame_equal, assert_identical, raises_regex)

dask = pytest.importorskip('dask')
da = pytest.importorskip('dask.array')
dd = pytest.importorskip('dask.dataframe')


class DaskTestCase(TestCase):
    def assertLazyAnd(self, expected, actual, test):
        with dask.set_options(get=dask.get):
            test(actual, expected)
        if isinstance(actual, Dataset):
            for k, v in actual.variables.items():
                if k in actual.dims:
                    assert isinstance(v.data, np.ndarray)
                else:
                    assert isinstance(v.data, da.Array)
        elif isinstance(actual, DataArray):
            assert isinstance(actual.data, da.Array)
            for k, v in actual.coords.items():
                if k in actual.dims:
                    assert isinstance(v.data, np.ndarray)
                else:
                    assert isinstance(v.data, da.Array)
        elif isinstance(actual, Variable):
            assert isinstance(actual.data, da.Array)
        else:
            assert False


class TestVariable(DaskTestCase):
    def assertLazyAndIdentical(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_identical)

    def assertLazyAndAllClose(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_allclose)

    def setUp(self):
        self.values = np.random.RandomState(0).randn(4, 6)
        self.data = da.from_array(self.values, chunks=(2, 2))

        self.eager_var = Variable(('x', 'y'), self.values)
        self.lazy_var = Variable(('x', 'y'), self.data)

    def test_basics(self):
        v = self.lazy_var
        assert self.data is v.data
        assert self.data.chunks == v.chunks
        assert_array_equal(self.values, v)

    def test_copy(self):
        self.assertLazyAndIdentical(self.eager_var, self.lazy_var.copy())
        self.assertLazyAndIdentical(self.eager_var,
                                    self.lazy_var.copy(deep=True))

    def test_chunk(self):
        for chunks, expected in [(None, ((2, 2), (2, 2, 2))),
                                 (3, ((3, 1), (3, 3))),
                                 ({'x': 3, 'y': 3}, ((3, 1), (3, 3))),
                                 ({'x': 3}, ((3, 1), (2, 2, 2))),
                                 ({'x': (3, 1)}, ((3, 1), (2, 2, 2)))]:
            rechunked = self.lazy_var.chunk(chunks)
            assert rechunked.chunks == expected
            self.assertLazyAndIdentical(self.eager_var, rechunked)

    def test_indexing(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u[0], v[0])
        self.assertLazyAndIdentical(u[:1], v[:1])
        self.assertLazyAndIdentical(u[[0, 1], [0, 1, 2]], v[[0, 1], [0, 1, 2]])
        with raises_regex(TypeError, 'stored in a dask array'):
            v[:1] = 0

    def test_squeeze(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u[0].squeeze(), v[0].squeeze())

    def test_equals(self):
        v = self.lazy_var
        assert v.equals(v)
        assert isinstance(v.data, da.Array)
        assert v.identical(v)
        assert isinstance(v.data, da.Array)

    def test_transpose(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u.T, v.T)

    def test_shift(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u.shift(x=2), v.shift(x=2))
        self.assertLazyAndIdentical(u.shift(x=-2), v.shift(x=-2))
        assert v.data.chunks == v.shift(x=1).data.chunks

    def test_roll(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u.roll(x=2), v.roll(x=2))
        assert v.data.chunks == v.roll(x=1).data.chunks

    def test_unary_op(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(-u, -v)
        self.assertLazyAndIdentical(abs(u), abs(v))
        self.assertLazyAndIdentical(u.round(), v.round())

    def test_binary_op(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(2 * u, 2 * v)
        self.assertLazyAndIdentical(u + u, v + v)
        self.assertLazyAndIdentical(u[0] + u, v[0] + v)

    def test_repr(self):
        expected = dedent("""\
        <xarray.Variable (x: 4, y: 6)>
        dask.array<shape=(4, 6), dtype=float64, chunksize=(2, 2)>""")
        assert expected == repr(self.lazy_var)

    def test_pickle(self):
        # Test that pickling/unpickling does not convert the dask
        # backend to numpy
        a1 = Variable(['x'], build_dask_array('x'))
        a1.compute()
        assert not a1._in_memory
        assert kernel_call_count == 1
        a2 = pickle.loads(pickle.dumps(a1))
        assert kernel_call_count == 1
        assert_identical(a1, a2)
        assert not a1._in_memory
        assert not a2._in_memory

    def test_reduce(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndAllClose(u.mean(), v.mean())
        self.assertLazyAndAllClose(u.std(), v.std())
        self.assertLazyAndAllClose(u.argmax(dim='x'), v.argmax(dim='x'))
        self.assertLazyAndAllClose((u > 1).any(), (v > 1).any())
        self.assertLazyAndAllClose((u < 1).all('x'), (v < 1).all('x'))
        with raises_regex(NotImplementedError, 'dask'):
            v.median()

    def test_missing_values(self):
        values = np.array([0, 1, np.nan, 3])
        data = da.from_array(values, chunks=(2,))

        eager_var = Variable('x', values)
        lazy_var = Variable('x', data)
        self.assertLazyAndIdentical(eager_var, lazy_var.fillna(lazy_var))
        self.assertLazyAndIdentical(Variable('x', range(4)),
                                    lazy_var.fillna(2))
        self.assertLazyAndIdentical(eager_var.count(), lazy_var.count())

    def test_concat(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u, Variable.concat([v[:2], v[2:]], 'x'))
        self.assertLazyAndIdentical(u[:2], Variable.concat([v[0], v[1]], 'x'))
        self.assertLazyAndIdentical(u[:2], Variable.concat([u[0], v[1]], 'x'))
        self.assertLazyAndIdentical(u[:2], Variable.concat([v[0], u[1]], 'x'))
        self.assertLazyAndIdentical(
            u[:3],
            Variable.concat([v[[0, 2]], v[[1]]], 'x', positions=[[0, 2], [1]]))

    def test_missing_methods(self):
        v = self.lazy_var
        try:
            v.argsort()
        except NotImplementedError as err:
            assert 'dask' in str(err)
        try:
            v[0].item()
        except NotImplementedError as err:
            assert 'dask' in str(err)

    def test_univariate_ufunc(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndAllClose(np.sin(u), xu.sin(v))

    def test_bivariate_ufunc(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndAllClose(np.maximum(u, 0), xu.maximum(v, 0))
        self.assertLazyAndAllClose(np.maximum(u, 0), xu.maximum(0, v))

    def test_compute(self):
        u = self.eager_var
        v = self.lazy_var

        assert dask.is_dask_collection(v)
        (v2,) = dask.compute(v + 1)
        assert not dask.is_dask_collection(v2)

        assert ((u + 1).data == v2.data).all()

    def test_persist(self):
        u = self.eager_var
        v = self.lazy_var + 1

        (v2,) = dask.persist(v)
        assert v is not v2
        assert len(v2.__dask_graph__()) < len(v.__dask_graph__())
        assert v2.__dask_keys__() == v.__dask_keys__()
        assert dask.is_dask_collection(v)
        assert dask.is_dask_collection(v2)

        self.assertLazyAndAllClose(u + 1, v)
        self.assertLazyAndAllClose(u + 1, v2)


class TestDataArrayAndDataset(DaskTestCase):
    def assertLazyAndIdentical(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_identical)

    def assertLazyAndAllClose(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_allclose)

    def assertLazyAndEqual(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_equal)

    def setUp(self):
        self.values = np.random.randn(4, 6)
        self.data = da.from_array(self.values, chunks=(2, 2))
        self.eager_array = DataArray(self.values, coords={'x': range(4)},
                                     dims=('x', 'y'), name='foo')
        self.lazy_array = DataArray(self.data, coords={'x': range(4)},
                                    dims=('x', 'y'), name='foo')

    def test_rechunk(self):
        chunked = self.eager_array.chunk({'x': 2}).chunk({'y': 2})
        assert chunked.chunks == ((2,) * 2, (2,) * 3)
        self.assertLazyAndIdentical(self.lazy_array, chunked)

    def test_new_chunk(self):
        chunked = self.eager_array.chunk()
        assert chunked.data.name.startswith('xarray-<this-array>')

    def test_lazy_dataset(self):
        lazy_ds = Dataset({'foo': (('x', 'y'), self.data)})
        assert isinstance(lazy_ds.foo.variable.data, da.Array)

    def test_lazy_array(self):
        u = self.eager_array
        v = self.lazy_array

        self.assertLazyAndAllClose(u, v)
        self.assertLazyAndAllClose(-u, -v)
        self.assertLazyAndAllClose(u.T, v.T)
        self.assertLazyAndAllClose(u.mean(), v.mean())
        self.assertLazyAndAllClose(1 + u, 1 + v)

        actual = xr.concat([v[:2], v[2:]], 'x')
        self.assertLazyAndAllClose(u, actual)

    def test_compute(self):
        u = self.eager_array
        v = self.lazy_array

        assert dask.is_dask_collection(v)
        (v2,) = dask.compute(v + 1)
        assert not dask.is_dask_collection(v2)

        assert ((u + 1).data == v2.data).all()

    def test_persist(self):
        u = self.eager_array
        v = self.lazy_array + 1

        (v2,) = dask.persist(v)
        assert v is not v2
        assert len(v2.__dask_graph__()) < len(v.__dask_graph__())
        assert v2.__dask_keys__() == v.__dask_keys__()
        assert dask.is_dask_collection(v)
        assert dask.is_dask_collection(v2)

        self.assertLazyAndAllClose(u + 1, v)
        self.assertLazyAndAllClose(u + 1, v2)

    def test_concat_loads_variables(self):
        # Test that concat() computes not-in-memory variables at most once
        # and loads them in the output, while leaving the input unaltered.
        d1 = build_dask_array('d1')
        c1 = build_dask_array('c1')
        d2 = build_dask_array('d2')
        c2 = build_dask_array('c2')
        d3 = build_dask_array('d3')
        c3 = build_dask_array('c3')
        # Note: c is a non-index coord.
        # Index coords are loaded by IndexVariable.__init__.
        ds1 = Dataset(data_vars={'d': ('x', d1)}, coords={'c': ('x', c1)})
        ds2 = Dataset(data_vars={'d': ('x', d2)}, coords={'c': ('x', c2)})
        ds3 = Dataset(data_vars={'d': ('x', d3)}, coords={'c': ('x', c3)})

        assert kernel_call_count == 0
        out = xr.concat([ds1, ds2, ds3], dim='n', data_vars='different',
                        coords='different')
        # each kernel is computed exactly once
        assert kernel_call_count == 6
        # variables are loaded in the output
        assert isinstance(out['d'].data, np.ndarray)
        assert isinstance(out['c'].data, np.ndarray)

        out = xr.concat(
            [ds1, ds2, ds3], dim='n', data_vars='all', coords='all')
        # no extra kernel calls
        assert kernel_call_count == 6
        assert isinstance(out['d'].data, dask.array.Array)
        assert isinstance(out['c'].data, dask.array.Array)

        out = xr.concat(
            [ds1, ds2, ds3], dim='n', data_vars=['d'], coords=['c'])
        # no extra kernel calls
        assert kernel_call_count == 6
        assert isinstance(out['d'].data, dask.array.Array)
        assert isinstance(out['c'].data, dask.array.Array)

        out = xr.concat([ds1, ds2, ds3], dim='n', data_vars=[], coords=[])
        # variables are loaded once as we are validing that they're identical
        assert kernel_call_count == 12
        assert isinstance(out['d'].data, np.ndarray)
        assert isinstance(out['c'].data, np.ndarray)

        out = xr.concat([ds1, ds2, ds3], dim='n', data_vars='different',
                        coords='different', compat='identical')
        # compat=identical doesn't do any more kernel calls than compat=equals
        assert kernel_call_count == 18
        assert isinstance(out['d'].data, np.ndarray)
        assert isinstance(out['c'].data, np.ndarray)

        # When the test for different turns true halfway through,
        # stop computing variables as it would not have any benefit
        ds4 = Dataset(data_vars={'d': ('x', [2.0])},
                      coords={'c': ('x', [2.0])})
        out = xr.concat([ds1, ds2, ds4, ds3], dim='n', data_vars='different',
                        coords='different')
        # the variables of ds1 and ds2 were computed, but those of ds3 didn't
        assert kernel_call_count == 22
        assert isinstance(out['d'].data, dask.array.Array)
        assert isinstance(out['c'].data, dask.array.Array)
        # the data of ds1 and ds2 was loaded into numpy and then
        # concatenated to the data of ds3. Thus, only ds3 is computed now.
        out.compute()
        assert kernel_call_count == 24

        # Finally, test that riginals are unaltered
        assert ds1['d'].data is d1
        assert ds1['c'].data is c1
        assert ds2['d'].data is d2
        assert ds2['c'].data is c2
        assert ds3['d'].data is d3
        assert ds3['c'].data is c3

    def test_groupby(self):
        u = self.eager_array
        v = self.lazy_array

        expected = u.groupby('x').mean()
        actual = v.groupby('x').mean()
        self.assertLazyAndAllClose(expected, actual)

    def test_groupby_first(self):
        u = self.eager_array
        v = self.lazy_array

        for coords in [u.coords, v.coords]:
            coords['ab'] = ('x', ['a', 'a', 'b', 'b'])
        with raises_regex(NotImplementedError, 'dask'):
            v.groupby('ab').first()
        expected = u.groupby('ab').first()
        actual = v.groupby('ab').first(skipna=False)
        self.assertLazyAndAllClose(expected, actual)

    def test_reindex(self):
        u = self.eager_array.assign_coords(y=range(6))
        v = self.lazy_array.assign_coords(y=range(6))

        for kwargs in [{'x': [2, 3, 4]},
                       {'x': [1, 100, 2, 101, 3]},
                       {'x': [2.5, 3, 3.5], 'y': [2, 2.5, 3]}]:
            expected = u.reindex(**kwargs)
            actual = v.reindex(**kwargs)
            self.assertLazyAndAllClose(expected, actual)

    def test_to_dataset_roundtrip(self):
        u = self.eager_array
        v = self.lazy_array

        expected = u.assign_coords(x=u['x'])
        self.assertLazyAndEqual(expected, v.to_dataset('x').to_array('x'))

    def test_merge(self):

        def duplicate_and_merge(array):
            return xr.merge([array, array.rename('bar')]).to_array()

        expected = duplicate_and_merge(self.eager_array)
        actual = duplicate_and_merge(self.lazy_array)
        self.assertLazyAndEqual(expected, actual)

    def test_ufuncs(self):
        u = self.eager_array
        v = self.lazy_array
        self.assertLazyAndAllClose(np.sin(u), xu.sin(v))

    def test_where_dispatching(self):
        a = np.arange(10)
        b = a > 3
        x = da.from_array(a, 5)
        y = da.from_array(b, 5)
        expected = DataArray(a).where(b)
        self.assertLazyAndEqual(expected, DataArray(a).where(y))
        self.assertLazyAndEqual(expected, DataArray(x).where(b))
        self.assertLazyAndEqual(expected, DataArray(x).where(y))

    def test_simultaneous_compute(self):
        ds = Dataset({'foo': ('x', range(5)),
                      'bar': ('x', range(5))}).chunk()

        count = [0]

        def counting_get(*args, **kwargs):
            count[0] += 1
            return dask.get(*args, **kwargs)

        ds.load(get=counting_get)
        assert count[0] == 1

    def test_stack(self):
        data = da.random.normal(size=(2, 3, 4), chunks=(1, 3, 4))
        arr = DataArray(data, dims=('w', 'x', 'y'))
        stacked = arr.stack(z=('x', 'y'))
        z = pd.MultiIndex.from_product([np.arange(3), np.arange(4)],
                                       names=['x', 'y'])
        expected = DataArray(data.reshape(2, -1), {'z': z}, dims=['w', 'z'])
        assert stacked.data.chunks == expected.data.chunks
        self.assertLazyAndEqual(expected, stacked)

    def test_dot(self):
        eager = self.eager_array.dot(self.eager_array[0])
        lazy = self.lazy_array.dot(self.lazy_array[0])
        self.assertLazyAndAllClose(eager, lazy)

    def test_dataarray_repr(self):
        # Test that __repr__ converts the dask backend to numpy
        # in neither the data variable nor the non-index coords
        data = build_dask_array('data')
        nonindex_coord = build_dask_array('coord')
        a = DataArray(data, dims=['x'], coords={'y': ('x', nonindex_coord)})
        expected = dedent("""\
        <xarray.DataArray 'data' (x: 1)>
        dask.array<shape=(1,), dtype=int64, chunksize=(1,)>
        Coordinates:
            y        (x) int64 dask.array<shape=(1,), chunksize=(1,)>
        Dimensions without coordinates: x""")
        assert expected == repr(a)
        assert kernel_call_count == 0

    def test_dataset_repr(self):
        # Test that pickling/unpickling converts the dask backend
        # to numpy in neither the data variables nor the non-index coords
        data = build_dask_array('data')
        nonindex_coord = build_dask_array('coord')
        ds = Dataset(data_vars={'a': ('x', data)},
                     coords={'y': ('x', nonindex_coord)})
        expected = dedent("""\
        <xarray.Dataset>
        Dimensions:  (x: 1)
        Coordinates:
            y        (x) int64 dask.array<shape=(1,), chunksize=(1,)>
        Dimensions without coordinates: x
        Data variables:
            a        (x) int64 dask.array<shape=(1,), chunksize=(1,)>""")
        assert expected == repr(ds)
        assert kernel_call_count == 0

    def test_dataarray_pickle(self):
        # Test that pickling/unpickling converts the dask backend
        # to numpy in neither the data variable nor the non-index coords
        data = build_dask_array('data')
        nonindex_coord = build_dask_array('coord')
        a1 = DataArray(data, dims=['x'], coords={'y': ('x', nonindex_coord)})
        a1.compute()
        assert not a1._in_memory
        assert not a1.coords['y']._in_memory
        assert kernel_call_count == 2
        a2 = pickle.loads(pickle.dumps(a1))
        assert kernel_call_count == 2
        assert_identical(a1, a2)
        assert not a1._in_memory
        assert not a2._in_memory
        assert not a1.coords['y']._in_memory
        assert not a2.coords['y']._in_memory

    def test_dataset_pickle(self):
        # Test that pickling/unpickling converts the dask backend
        # to numpy in neither the data variables nor the non-index coords
        data = build_dask_array('data')
        nonindex_coord = build_dask_array('coord')
        ds1 = Dataset(data_vars={'a': ('x', data)},
                      coords={'y': ('x', nonindex_coord)})
        ds1.compute()
        assert not ds1['a']._in_memory
        assert not ds1['y']._in_memory
        assert kernel_call_count == 2
        ds2 = pickle.loads(pickle.dumps(ds1))
        assert kernel_call_count == 2
        assert_identical(ds1, ds2)
        assert not ds1['a']._in_memory
        assert not ds2['a']._in_memory
        assert not ds1['y']._in_memory
        assert not ds2['y']._in_memory

    def test_dataarray_getattr(self):
        # ipython/jupyter does a long list of getattr() calls to when trying to
        # represent an object.
        # Make sure we're not accidentally computing dask variables.
        data = build_dask_array('data')
        nonindex_coord = build_dask_array('coord')
        a = DataArray(data, dims=['x'],
                      coords={'y': ('x', nonindex_coord)})
        with suppress(AttributeError):
            getattr(a, 'NOTEXIST')
        assert kernel_call_count == 0

    def test_dataset_getattr(self):
        # Test that pickling/unpickling converts the dask backend
        # to numpy in neither the data variables nor the non-index coords
        data = build_dask_array('data')
        nonindex_coord = build_dask_array('coord')
        ds = Dataset(data_vars={'a': ('x', data)},
                     coords={'y': ('x', nonindex_coord)})
        with suppress(AttributeError):
            getattr(ds, 'NOTEXIST')
        assert kernel_call_count == 0

    def test_values(self):
        # Test that invoking the values property does not convert the dask
        # backend to numpy
        a = DataArray([1, 2]).chunk()
        assert not a._in_memory
        assert a.values.tolist() == [1, 2]
        assert not a._in_memory

    def test_from_dask_variable(self):
        # Test array creation from Variable with dask backend.
        # This is used e.g. in broadcast()
        a = DataArray(self.lazy_array.variable,
                      coords={'x': range(4)}, name='foo')
        self.assertLazyAndIdentical(self.lazy_array, a)


class TestToDaskDataFrame(TestCase):

    def test_to_dask_dataframe(self):
        # Test conversion of Datasets to dask DataFrames
        x = da.from_array(np.random.randn(10), chunks=4)
        y = np.arange(10, dtype='uint8')
        t = list('abcdefghij')

        ds = Dataset(OrderedDict([('a', ('t', x)),
                                  ('b', ('t', y)),
                                  ('t', ('t', t))]))

        expected_pd = pd.DataFrame({'a': x,
                                    'b': y},
                                   index=pd.Index(t, name='t'))

        # test if 1-D index is correctly set up
        expected = dd.from_pandas(expected_pd, chunksize=4)
        actual = ds.to_dask_dataframe(set_index=True)
        # test if we have dask dataframes
        assert isinstance(actual, dd.DataFrame)

        # use the .equals from pandas to check dataframes are equivalent
        assert_frame_equal(expected.compute(), actual.compute())

        # test if no index is given
        expected = dd.from_pandas(expected_pd.reset_index(drop=False),
                                  chunksize=4)

        actual = ds.to_dask_dataframe(set_index=False)

        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected.compute(), actual.compute())

    def test_to_dask_dataframe_2D(self):
        # Test if 2-D dataset is supplied
        w = da.from_array(np.random.randn(2, 3), chunks=(1, 2))
        ds = Dataset({'w': (('x', 'y'), w)})
        ds['x'] = ('x', np.array([0, 1], np.int64))
        ds['y'] = ('y', list('abc'))

        # dask dataframes do not (yet) support multiindex,
        # but when it does, this would be the expected index:
        exp_index = pd.MultiIndex.from_arrays(
            [[0, 0, 0, 1, 1, 1], ['a', 'b', 'c', 'a', 'b', 'c']],
            names=['x', 'y'])
        expected = pd.DataFrame({'w': w.reshape(-1)},
                                index=exp_index)
        # so for now, reset the index
        expected = expected.reset_index(drop=False)
        actual = ds.to_dask_dataframe(set_index=False)

        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_to_dask_dataframe_2D_set_index(self):
        # This will fail until dask implements MultiIndex support
        w = da.from_array(np.random.randn(2, 3), chunks=(1, 2))
        ds = Dataset({'w': (('x', 'y'), w)})
        ds['x'] = ('x', np.array([0, 1], np.int64))
        ds['y'] = ('y', list('abc'))

        expected = ds.compute().to_dataframe()
        actual = ds.to_dask_dataframe(set_index=True)
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())

    def test_to_dask_dataframe_coordinates(self):
        # Test if coordinate is also a dask array
        x = da.from_array(np.random.randn(10), chunks=4)
        t = da.from_array(np.arange(10) * 2, chunks=4)

        ds = Dataset(OrderedDict([('a', ('t', x)),
                                  ('t', ('t', t))]))

        expected_pd = pd.DataFrame({'a': x},
                                   index=pd.Index(t, name='t'))
        expected = dd.from_pandas(expected_pd, chunksize=4)
        actual = ds.to_dask_dataframe(set_index=True)
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected.compute(), actual.compute())

    def test_to_dask_dataframe_not_daskarray(self):
        # Test if DataArray is not a dask array
        x = np.random.randn(10)
        y = np.arange(10, dtype='uint8')
        t = list('abcdefghij')

        ds = Dataset(OrderedDict([('a', ('t', x)),
                                  ('b', ('t', y)),
                                  ('t', ('t', t))]))

        expected = pd.DataFrame({'a': x, 'b': y},
                                index=pd.Index(t, name='t'))

        actual = ds.to_dask_dataframe(set_index=True)
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())

    def test_to_dask_dataframe_no_coordinate(self):
        x = da.from_array(np.random.randn(10), chunks=4)
        ds = Dataset({'x': ('dim_0', x)})

        expected = ds.compute().to_dataframe().reset_index()
        actual = ds.to_dask_dataframe()
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())

        expected = ds.compute().to_dataframe()
        actual = ds.to_dask_dataframe(set_index=True)
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())

    def test_to_dask_dataframe_dim_order(self):
        values = np.array([[1, 2], [3, 4]], dtype=np.int64)
        ds = Dataset({'w': (('x', 'y'), values)}).chunk(1)

        expected = ds['w'].to_series().reset_index()
        actual = ds.to_dask_dataframe(dim_order=['x', 'y'])
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())

        expected = ds['w'].T.to_series().reset_index()
        actual = ds.to_dask_dataframe(dim_order=['y', 'x'])
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())

        with raises_regex(ValueError, 'does not match the set of dimensions'):
            ds.to_dask_dataframe(dim_order=['x'])


@pytest.mark.parametrize("method", ['load', 'compute'])
def test_dask_kwargs_variable(method):
    x = Variable('y', da.from_array(np.arange(3), chunks=(2,)))
    # args should be passed on to da.Array.compute()
    with mock.patch.object(da.Array, 'compute',
                           return_value=np.arange(3)) as mock_compute:
        getattr(x, method)(foo='bar')
    mock_compute.assert_called_with(foo='bar')


@pytest.mark.parametrize("method", ['load', 'compute', 'persist'])
def test_dask_kwargs_dataarray(method):
    data = da.from_array(np.arange(3), chunks=(2,))
    x = DataArray(data)
    if method in ['load', 'compute']:
        dask_func = 'dask.array.compute'
    else:
        dask_func = 'dask.persist'
    # args should be passed on to "dask_func"
    with mock.patch(dask_func) as mock_func:
        getattr(x, method)(foo='bar')
    mock_func.assert_called_with(data, foo='bar')


@pytest.mark.parametrize("method", ['load', 'compute', 'persist'])
def test_dask_kwargs_dataset(method):
    data = da.from_array(np.arange(3), chunks=(2,))
    x = Dataset({'x': (('y'), data)})
    if method in ['load', 'compute']:
        dask_func = 'dask.array.compute'
    else:
        dask_func = 'dask.persist'
    # args should be passed on to "dask_func"
    with mock.patch(dask_func) as mock_func:
        getattr(x, method)(foo='bar')
    mock_func.assert_called_with(data, foo='bar')


kernel_call_count = 0


def kernel(name):
    """Dask kernel to test pickling/unpickling and __repr__.
    Must be global to make it pickleable.
    """
    print("kernel(%s)" % name)
    global kernel_call_count
    kernel_call_count += 1
    return np.ones(1, dtype=np.int64)


def build_dask_array(name):
    global kernel_call_count
    kernel_call_count = 0
    return dask.array.Array(
        dask={(name, 0): (kernel, name)}, name=name,
        chunks=((1,),), dtype=np.int64)


# test both the perist method and the dask.persist function
# the dask.persist function requires a new version of dask
@pytest.mark.parametrize('persist', [lambda x: x.persist(),
                                     lambda x: dask.persist(x)[0]])
def test_persist_Dataset(persist):
    ds = Dataset({'foo': ('x', range(5)),
                  'bar': ('x', range(5))}).chunk()
    ds = ds + 1
    n = len(ds.foo.data.dask)

    ds2 = persist(ds)

    assert len(ds2.foo.data.dask) == 1
    assert len(ds.foo.data.dask) == n  # doesn't mutate in place


@pytest.mark.parametrize('persist', [lambda x: x.persist(),
                                     lambda x: dask.persist(x)[0]])
def test_persist_DataArray(persist):
    x = da.arange(10, chunks=(5,))
    y = DataArray(x)
    z = y + 1
    n = len(z.data.dask)

    zz = persist(z)

    assert len(z.data.dask) == n
    assert len(zz.data.dask) == zz.data.npartitions


def test_dataarray_with_dask_coords():
    import toolz
    x = xr.Variable('x', da.arange(8, chunks=(4,)))
    y = xr.Variable('y', da.arange(8, chunks=(4,)) * 2)
    data = da.random.random((8, 8), chunks=(4, 4)) + 1
    array = xr.DataArray(data, dims=['x', 'y'])
    array.coords['xx'] = x
    array.coords['yy'] = y

    assert dict(array.__dask_graph__()) == toolz.merge(data.__dask_graph__(),
                                                       x.__dask_graph__(),
                                                       y.__dask_graph__())

    (array2,) = dask.compute(array)
    assert not dask.is_dask_collection(array2)

    assert all(isinstance(v._variable.data, np.ndarray)
               for v in array2.coords.values())


def test_basic_compute():
    ds = Dataset({'foo': ('x', range(5)),
                  'bar': ('x', range(5))}).chunk({'x': 2})
    for get in [dask.threaded.get,
                dask.multiprocessing.get,
                dask.local.get_sync,
                None]:
        with dask.set_options(get=get):
            ds.compute()
            ds.foo.compute()
            ds.foo.variable.compute()
