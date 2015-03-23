import numpy as np

from xray.core.variable import Variable
from . import TestCase, requires_dask

try:
    import dask
    import dask.array as da
except ImportError:
    pass


@requires_dask
class TestVariable(TestCase):
    def assertLazyAnd(self, expected, actual, test):
        expected_copy = expected.copy(deep=False)
        actual_copy = actual.copy(deep=False)
        with dask.set_options(get=dask.get):
            test(actual_copy, expected_copy)
        self.assertIsInstance(actual.data, da.Array)

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
