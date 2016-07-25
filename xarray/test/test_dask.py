import numpy as np
import pandas as pd

import xarray as xr
from xarray import Variable, DataArray, Dataset
import xarray.ufuncs as xu
from xarray.core.pycompat import suppress
from . import TestCase, requires_dask

with suppress(ImportError):
    import dask
    import dask.array as da


def _copy_at_variable_level(arg):
    """We need to copy the argument at the level of xarray.Variable objects, so
    that viewing its values does not trigger lazy loading.
    """
    if isinstance(arg, Variable):
        return arg.copy(deep=False)
    elif isinstance(arg, DataArray):
        ds = arg.to_dataset(name='__copied__')
        return _copy_at_variable_level(ds)['__copied__']
    elif isinstance(arg, Dataset):
        ds = arg.copy()
        for k in list(ds):
            ds._variables[k] = ds._variables[k].copy(deep=False)
        return ds
    else:
        assert False


class DaskTestCase(TestCase):
    def assertLazyAnd(self, expected, actual, test):
        expected_copy = _copy_at_variable_level(expected)
        actual_copy = _copy_at_variable_level(actual)
        with dask.set_options(get=dask.get):
            test(actual_copy, expected_copy)
        var = getattr(actual, 'variable', actual)
        self.assertIsInstance(var.data, da.Array)


@requires_dask
class TestVariable(DaskTestCase):
    def assertLazyAnd(self, expected, actual, test):
        expected_copy = expected.copy(deep=False)
        actual_copy = actual.copy(deep=False)
        with dask.set_options(get=dask.get):
            test(actual_copy, expected_copy)
        var = getattr(actual, 'variable', actual)
        self.assertIsInstance(var.data, da.Array)

    def assertLazyAndIdentical(self, expected, actual):
        self.assertLazyAnd(expected, actual, self.assertVariableIdentical)

    def assertLazyAndAllClose(self, expected, actual):
        self.assertLazyAnd(expected, actual, self.assertVariableAllClose)

    def setUp(self):
        self.values = np.random.RandomState(0).randn(4, 6)
        self.data = da.from_array(self.values, chunks=(2, 2))

        self.eager_var = Variable(('x', 'y'), self.values)
        self.lazy_var = Variable(('x', 'y'), self.data)

    def test_basics(self):
        v = self.lazy_var
        self.assertIs(self.data, v.data)
        self.assertEqual(self.data.chunks, v.chunks)
        self.assertArrayEqual(self.values, v)

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
            self.assertEqual(rechunked.chunks, expected)
            self.assertLazyAndIdentical(self.eager_var, rechunked)

    def test_indexing(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u[0], v[0])
        self.assertLazyAndIdentical(u[:1], v[:1])
        self.assertLazyAndIdentical(u[[0, 1], [0, 1, 2]], v[[0, 1], [0, 1, 2]])
        with self.assertRaisesRegexp(TypeError, 'stored in a dask array'):
            v[:1] = 0

    def test_squeeze(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u[0].squeeze(), v[0].squeeze())

    def test_equals(self):
        v = self.lazy_var
        self.assertTrue(v.equals(v))
        self.assertIsInstance(v.data, da.Array)
        self.assertTrue(v.identical(v))
        self.assertIsInstance(v.data, da.Array)

    def test_transpose(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u.T, v.T)

    def test_shift(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u.shift(x=2), v.shift(x=2))
        self.assertLazyAndIdentical(u.shift(x=-2), v.shift(x=-2))
        self.assertEqual(v.data.chunks, v.shift(x=1).data.chunks)

    def test_roll(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u.roll(x=2), v.roll(x=2))
        self.assertEqual(v.data.chunks, v.roll(x=1).data.chunks)

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

    def test_reduce(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndAllClose(u.mean(), v.mean())
        self.assertLazyAndAllClose(u.std(), v.std())
        self.assertLazyAndAllClose(u.argmax(dim='x'), v.argmax(dim='x'))
        self.assertLazyAndAllClose((u > 1).any(), (v > 1).any())
        self.assertLazyAndAllClose((u < 1).all('x'), (v < 1).all('x'))
        with self.assertRaisesRegexp(NotImplementedError, 'dask'):
            v.prod()
        with self.assertRaisesRegexp(NotImplementedError, 'dask'):
            v.median()

    def test_missing_values(self):
        values = np.array([0, 1, np.nan, 3])
        data = da.from_array(values, chunks=(2,))

        eager_var = Variable('x', values)
        lazy_var = Variable('x', data)
        self.assertLazyAndIdentical(eager_var, lazy_var.fillna(lazy_var))
        self.assertLazyAndIdentical(Variable('x', range(4)), lazy_var.fillna(2))
        self.assertLazyAndIdentical(eager_var.count(), lazy_var.count())

    def test_concat(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u, Variable.concat([v[:2], v[2:]], 'x'))
        self.assertLazyAndIdentical(u[:2], Variable.concat([v[0], v[1]], 'x'))
        self.assertLazyAndIdentical(
            u[:3], Variable.concat([v[[0, 2]], v[[1]]], 'x', positions=[[0, 2], [1]]))

    def test_missing_methods(self):
        v = self.lazy_var
        try:
            v.argsort()
        except NotImplementedError as err:
            self.assertIn('dask', str(err))
        try:
            v[0].item()
        except NotImplementedError as err:
            self.assertIn('dask', str(err))

    def test_univariate_ufunc(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndAllClose(np.sin(u), xu.sin(v))

    def test_bivariate_ufunc(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndAllClose(np.maximum(u, 0), xu.maximum(v, 0))
        self.assertLazyAndAllClose(np.maximum(u, 0), xu.maximum(0, v))


@requires_dask
class TestDataArrayAndDataset(DaskTestCase):
    def assertLazyAndIdentical(self, expected, actual):
        self.assertLazyAnd(expected, actual, self.assertDataArrayIdentical)

    def assertLazyAndAllClose(self, expected, actual):
        self.assertLazyAnd(expected, actual, self.assertDataArrayAllClose)

    def setUp(self):
        self.values = np.random.randn(4, 6)
        self.data = da.from_array(self.values, chunks=(2, 2))
        self.eager_array = DataArray(self.values, dims=('x', 'y'), name='foo')
        self.lazy_array = DataArray(self.data, dims=('x', 'y'), name='foo')

    def test_rechunk(self):
        chunked = self.eager_array.chunk({'x': 2}).chunk({'y': 2})
        self.assertEqual(chunked.chunks, ((2,) * 2, (2,) * 3))

    def test_new_chunk(self):
        chunked = self.eager_array.chunk()
        self.assertTrue(chunked.data.name.startswith('xarray-<this-array>'))

    def test_lazy_dataset(self):
        lazy_ds = Dataset({'foo': (('x', 'y'), self.data)})
        self.assertIsInstance(lazy_ds.foo.variable.data, da.Array)

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
        with self.assertRaisesRegexp(NotImplementedError, 'dask'):
            v.groupby('ab').first()
        expected = u.groupby('ab').first()
        actual = v.groupby('ab').first(skipna=False)
        self.assertLazyAndAllClose(expected, actual)

    def test_reindex(self):
        u = self.eager_array
        v = self.lazy_array

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
        self.assertLazyAndIdentical(expected, v.to_dataset('x').to_array('x'))

    def test_merge(self):

        def duplicate_and_merge(array):
            return xr.merge([array, array.rename('bar')]).to_array()

        expected = duplicate_and_merge(self.eager_array)
        actual = duplicate_and_merge(self.lazy_array)
        self.assertLazyAndIdentical(expected, actual)

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
        self.assertLazyAndIdentical(expected, DataArray(a).where(y))
        self.assertLazyAndIdentical(expected, DataArray(x).where(b))
        self.assertLazyAndIdentical(expected, DataArray(x).where(y))

    def test_simultaneous_compute(self):
        ds = Dataset({'foo': ('x', range(5)),
                      'bar': ('x', range(5))}).chunk()

        count = [0]

        def counting_get(*args, **kwargs):
            count[0] += 1
            return dask.get(*args, **kwargs)

        with dask.set_options(get=counting_get):
            ds.load()
        self.assertEqual(count[0], 1)

    def test_stack(self):
        data = da.random.normal(size=(2, 3, 4), chunks=(1, 3, 4))
        arr = DataArray(data, dims=('w', 'x', 'y'))
        stacked = arr.stack(z=('x', 'y'))
        z = pd.MultiIndex.from_product([np.arange(3), np.arange(4)],
                                       names=['x', 'y'])
        expected = DataArray(data.reshape(2, -1), {'w': [0, 1], 'z': z},
                             dims=['w', 'z'])
        assert stacked.data.chunks == expected.data.chunks
        self.assertLazyAndIdentical(expected, stacked)

    def test_dot(self):
        eager = self.eager_array.dot(self.eager_array[0])
        lazy = self.lazy_array.dot(self.lazy_array[0])
        self.assertLazyAndAllClose(eager, lazy)
