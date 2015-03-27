import numpy as np

from xray import Variable, DataArray, Dataset
from . import TestCase, requires_dask, unittest

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
        ds = arg.to_dataset('__copied__')
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
        self.data = da.from_array(self.values, blockshape=(2, 2))

        self.eager_var = Variable(('x', 'y'), self.values)
        self.lazy_var = Variable(('x', 'y'), self.data)

    def test_basics(self):
        v = self.lazy_var
        self.assertIs(self.data, v.data)
        self.assertArrayEqual(self.values, v)

    def test_indexing(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u[0], v[0])
        self.assertLazyAndIdentical(u[:1], v[:1])

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

    def test_missing_values(self):
        values = np.array([0, 1, np.nan, 3])
        data = da.from_array(values, blockshape=(2,))

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
            u[:3], Variable.concat([v[[0, 2]], v[[1]]], 'x', indexers=[[0, 2], [1]]))


@requires_dask
class TestDataArray(DaskTestCase):
    def assertLazyAndIdentical(self, expected, actual):
        self.assertLazyAnd(expected, actual, self.assertDataArrayIdentical)

    def assertLazyAndAllClose(self, expected, actual):
        self.assertLazyAnd(expected, actual, self.assertDataArrayAllClose)

    def setUp(self):
        self.values = np.random.randn(4, 6)
        self.data = da.from_array(self.values, blockshape=(2, 2))

    def test_lazy_dataset(self):
        lazy_ds = Dataset({'foo': (('x', 'y'), self.data)})
        self.assertIsInstance(lazy_ds.foo.variable.data, da.Array)

    def test_lazy_array(self):
        eager_array = DataArray(self.values, dims=('x', 'y'))
        lazy_array = DataArray(self.data, dims=('x', 'y'))

        self.assertLazyAndAllClose(eager_array, lazy_array)
        self.assertLazyAndAllClose(-eager_array, -lazy_array)
        self.assertLazyAndAllClose(eager_array.T, lazy_array.T)
        self.assertLazyAndAllClose(eager_array.mean(), lazy_array.mean())
        self.assertLazyAndAllClose(1 + eager_array, 1 + lazy_array)

    @unittest.skip('currently broken')
    def test_groupby(self):
        eager_array = DataArray(self.values, dims=('x', 'y'))
        lazy_array = DataArray(self.data, dims=('x', 'y'))

        actual = lazy_array.groupby('x').mean()
        expected = eager_array.groupby('x').mean()
        self.assertLazyAndAllClose(expected, actual)
