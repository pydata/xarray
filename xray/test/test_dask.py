import numpy as np

from xray import Variable, DataArray, Dataset, concat
import xray.ufuncs as xu
from . import TestCase, requires_dask, unittest, InaccessibleArray

try:
    import dask
    import dask.array as da
except ImportError:
    pass


def _copy_at_variable_level(arg):
    """We need to copy the argument at the level of xray.Variable objects, so
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
        self.values = np.random.randn(4, 6)
        self.data = da.from_array(self.values, chunks=(2, 2))

        self.eager_var = Variable(('x', 'y'), self.values)
        self.lazy_var = Variable(('x', 'y'), self.data)

    def test_basics(self):
        v = self.lazy_var
        self.assertIs(self.data, v.data)
        self.assertEqual(self.data.chunks, v.chunks)
        self.assertArrayEqual(self.values, v)

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
        with self.assertRaisesRegexp(NotImplementedError, 'dask'):
            v.conj()
        with self.assertRaisesRegexp(NotImplementedError, 'dask'):
            v.argsort()
        with self.assertRaisesRegexp(NotImplementedError, 'dask'):
            v[0].item()

    def test_ufuncs(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndAllClose(np.sin(u), xu.sin(v))


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

    def test_chunk(self):
        chunked = self.eager_array.chunk({'x': 2}).chunk({'y': 2})
        self.assertEqual(chunked.chunks, ((2,) * 2, (2,) * 3))

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

        actual = concat([v[:2], v[2:]], 'x')
        self.assertLazyAndAllClose(u, actual)

    @unittest.skip('broken on dask 0.6.0')
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

        expected = u.assign_coords(x=u['x'].astype(str))
        self.assertLazyAndIdentical(expected, v.to_dataset('x').to_array('x'))

    def test_ufuncs(self):
        u = self.eager_array
        v = self.lazy_array
        self.assertLazyAndAllClose(np.sin(u), xu.sin(v))

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
