import warnings

import numpy as np

from scidata import Variable, Dataset
from scidata.variable import stack_variables
from . import TestCase


class TestVariable(TestCase):
    def setUp(self):
        self.d = np.random.random((10, 3)).astype(np.float64)

    def test_data(self):
        v = Variable(['time', 'x'], self.d)
        self.assertIs(v.data, self.d)
        with self.assertRaises(ValueError):
            # wrong size
            v.data = np.random.random(5)
        d2 = np.random.random((10, 3))
        v.data = d2
        self.assertIs(v.data, d2)

        with warnings.catch_warnings(record=True) as w:
            v = Variable(['x'], range(5))
            self.assertIn("converting data to np.ndarray", str(w[-1].message))
            self.assertIsInstance(v.data, np.ndarray)
        with warnings.catch_warnings(record=True) as w:
            # don't warn for numpy numbers
            v = Variable([], np.float32(1))
            self.assertFalse(w)

    def test_properties(self):
        v = Variable(['time', 'x'], self.d, {'foo': 'bar'})
        self.assertEqual(v.dimensions, ('time', 'x'))
        self.assertEqual(v.dtype, float)
        self.assertEqual(v.shape, (10, 3))
        self.assertEqual(v.size, 30)
        self.assertEqual(v.ndim, 2)
        self.assertEqual(len(v), 10)
        self.assertEqual(v.attributes, {'foo': u'bar'})

    def test_repr(self):
        v = Variable(['time', 'x'], self.d)
        self.assertEqual('<scidata.Variable (time: 10, x: 3): float64>',
                         repr(v))

    def test_items(self):
        data = np.random.random((10, 11))
        v = Variable(['x', 'y'], data)
        # test slicing
        self.assertVarEqual(v, v[:])
        self.assertVarEqual(v, v[...])
        self.assertVarEqual(Variable(['y'], data[0]), v[0])
        self.assertVarEqual(Variable(['x'], data[:, 0]), v[:, 0])
        self.assertVarEqual(Variable(['x', 'y'], data[:3, :2]), v[:3, :2])
        # test array indexing
        x = Variable(['x'], np.arange(10))
        y = Variable(['y'], np.arange(11))
        self.assertVarEqual(v, v[x.data])
        self.assertVarEqual(v, v[x])
        self.assertVarEqual(v[:3], v[x < 3])
        self.assertVarEqual(v[:, 3:], v[:, y >= 3])
        self.assertVarEqual(v[:3, 3:], v[x < 3, y >= 3])
        self.assertVarEqual(v[:3, :2], v[x[:3], y[:2]])
        self.assertVarEqual(v[:3, :2], v[range(3), range(2)])
        # test iteration
        for n, item in enumerate(v):
            self.assertVarEqual(Variable(['y'], data[n]), item)
        # test setting
        v.data[:] = 0
        self.assertTrue(np.all(v.data == 0))

    def test_views(self):
        v = Variable(['time', 'x'], self.d)
        self.assertVarEqual(v.views(time=slice(None)), v)
        self.assertVarEqual(v.views(time=0), v[0])
        self.assertVarEqual(v.views(time=slice(0, 3)), v[:3])
        self.assertVarEqual(v.views(x=0), v[:, 0])
        with self.assertRaisesRegexp(ValueError, 'do not exist'):
            v.views(not_a_dim=0)

    def test_transpose(self):
        v = Variable(['time', 'x'], self.d)
        v2 = Variable(['x', 'time'], self.d.T)
        self.assertVarEqual(v, v2.transpose())
        x = np.random.randn(2, 3, 4, 5)
        w = Variable(['a', 'b', 'c', 'd'], x)
        w2 = Variable(['d', 'b', 'c', 'a'], np.einsum('abcd->dbca', x))
        self.assertVarEqual(w2, w.transpose('d', 'b', 'c', 'a'))

    def test_1d_math(self):
        x = np.arange(5)
        y = np.ones(5)
        v = Variable(['x'], x)
        # unary ops
        self.assertVarEqual(v, +v)
        self.assertVarEqual(v, abs(v))
        self.assertArrayEqual((-v).data, -x)
        # bianry ops with numbers
        self.assertVarEqual(v, v + 0)
        self.assertVarEqual(v, 0 + v)
        self.assertVarEqual(v, v * 1)
        self.assertArrayEqual((v > 2).data, x > 2)
        self.assertArrayEqual((0 == v).data, 0 == x)
        self.assertArrayEqual((v - 1).data, x - 1)
        self.assertArrayEqual((1 - v).data, 1 - x)
        # binary ops with numpy arrays
        self.assertArrayEqual((v * x).data, x ** 2)
        self.assertArrayEqual((x * v).data, x ** 2)
        self.assertArrayEqual(v - y, v - 1)
        self.assertArrayEqual(y - v, 1 - v)
        # verify attributes
        v2 = Variable(['x'], x, {'units': 'meters'})
        self.assertVarEqual(v2, +v2)
        self.assertVarEqual(v2, 0 + v2)
        # binary ops with all variables
        self.assertArrayEqual(v + v, 2 * v)
        w = Variable(['x'], y, {'foo': 'bar'})
        self.assertVarEqual(v + w, Variable(['x'], x + y))
        self.assertArrayEqual((v * w).data, x * y)
        # something complicated
        self.assertArrayEqual((v ** 2 * w - 1 + x).data, x ** 2 * y - 1 + x)

    def test_broadcasting_math(self):
        x = np.random.randn(2, 3)
        v = Variable(['a', 'b'], x)
        # 1d to 2d broadcasting
        self.assertVarEqual(
            v * v,
            Variable(['a', 'b'], np.einsum('ab,ab->ab', x, x)))
        self.assertVarEqual(
            v * v[0],
            Variable(['a', 'b'], np.einsum('ab,b->ab', x, x[0])))
        self.assertVarEqual(
            v[0] * v,
            Variable(['b', 'a'], np.einsum('b,ab->ba', x[0], x)))
        self.assertVarEqual(
            v[0] * v[:, 0],
            Variable(['b', 'a'], np.einsum('b,a->ba', x[0], x[:, 0])))
        # higher dim broadcasting
        y = np.random.randn(3, 4, 5)
        w = Variable(['b', 'c', 'd'], y)
        self.assertVarEqual(
            v * w, Variable(['a', 'b', 'c', 'd'],
                            np.einsum('ab,bcd->abcd', x, y)))
        self.assertVarEqual(
            w * v, Variable(['b', 'c', 'd', 'a'],
                            np.einsum('bcd,ab->bcda', y, x)))
        self.assertVarEqual(
            v * w[0], Variable(['a', 'b', 'c', 'd'],
                            np.einsum('ab,cd->abcd', x, y[0])))

    def test_broadcasting_failures(self):
        a = Variable(['x'], np.arange(10))
        b = Variable(['x'], np.arange(5))
        c = Variable(['x', 'x'], np.arange(100).reshape(10, 10))
        with self.assertRaisesRegexp(ValueError, 'mismatched lengths'):
            a + b
        with self.assertRaisesRegexp(ValueError, 'duplicate dimensions'):
            a + c

    def test_inplace_math(self):
        x = np.arange(5)
        v = Variable(['x'], x)
        v2 = v
        v2 += 1
        self.assertIs(v, v2)
        # since we provided an ndarray for data, it is also modified in-place
        self.assertIs(v.data, x)
        self.assertArrayEqual(v.data, np.arange(5) + 1)

    def test_array_interface(self):
        x = np.arange(5)
        v = Variable(['x'], x)
        self.assertArrayEqual(np.asarray(v), x)
        # test ufuncs
        self.assertVarEqual(np.sin(v), Variable(['x'], np.sin(x)))

    def test_collapsed(self):
        v = Variable(['time', 'x'], self.d)
        # intentionally test with an operation for which order matters
        self.assertVarEqual(v.collapsed(np.std, 'time'),
                            Variable(['x'], self.d.std(axis=0),
                                     {'cell_methods': 'time: std'}))
        self.assertVarEqual(v.collapsed(np.std, axis=0),
                            v.collapsed(np.std, dimension='time'))
        self.assertVarEqual(v.collapsed(np.std, ['x', 'time']),
                            Variable([], self.d.std(axis=1).std(axis=0),
                                     {'cell_methods': 'x: std time: std'}))
        self.assertVarEqual(v.collapsed(np.std),
                            Variable([], self.d.std(),
                                     {'cell_methods': 'time: x: std'}))
        self.assertVarEqual(v.mean('time'), v.collapsed(np.mean, 'time'))

    def test_aggregated_by(self):
        agg_var = Variable(['y'], np.array(['a', 'a', 'b']))
        v = Variable(['x', 'y'], self.d)
        expected_unique = Variable(['abc'], np.array(['a', 'b']))
        expected_aggregated = Variable(['x', 'abc'],
                                       np.array([self.d[:, :2].sum(axis=1),
                                                 self.d[:, 2:].sum(axis=1)]).T,
                                       {'cell_methods': 'y: sum'})
        actual_unique, actual_aggregated = v.aggregated_by(np.sum, 'abc', agg_var)
        self.assertVarEqual(expected_unique, actual_unique)
        self.assertVarEqual(expected_aggregated, actual_aggregated)
        # should be equivalent to aggregate by a dataview, too
        alt_agg_var = Dataset({'abc': agg_var})['abc']
        actual_unique, actual_aggregated = v.aggregated_by(np.sum, 'abc',
                                                           alt_agg_var)
        self.assertVarEqual(expected_unique, actual_unique)
        self.assertVarEqual(expected_aggregated, actual_aggregated)

    def test_stack_variables(self):
        x = np.arange(5)
        y = np.ones(5)
        v = Variable(['a'], x)
        w = Variable(['a'], y)
        self.assertVarEqual(Variable(['b', 'a'], np.array([x, y])),
                            stack_variables([v, w], 'b'))
        self.assertVarEqual(Variable(['b', 'a'], np.array([x, y])),
                            stack_variables((v, w), 'b'))
        self.assertVarEqual(Variable(['b', 'a'], np.array([x, y])),
                            stack_variables((v, w), 'b', length=2))
        with self.assertRaisesRegexp(ValueError, 'too many'):
            stack_variables([v, w], 'b', length=1)
        with self.assertRaisesRegexp(ValueError, r'only \d+ stack'):
            stack_variables([v, w, w], 'b', length=4)
        with self.assertRaisesRegexp(ValueError, 'inconsistent dimensions'):
            stack_variables([v, Variable(['c'], y)], 'b')


