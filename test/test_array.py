import warnings

import numpy as np

from scidata import Array, Dataset
from . import TestCase


class TestArray(TestCase):
    def setUp(self):
        self.d = np.random.random((10, 3)).astype(np.float64)

    def test_data(self):
        v = Array(['time', 'x'], self.d)
        self.assertIs(v.data, self.d)
        with self.assertRaises(ValueError):
            # wrong size
            v.data = np.random.random(5)
        d2 = np.random.random((10, 3))
        v.data = d2
        self.assertIs(v.data, d2)

        with warnings.catch_warnings(record=True) as w:
            v = Array(['x'], range(5))
            self.assertIn("converting data to np.ndarray", str(w[-1].message))
            self.assertIsInstance(v.data, np.ndarray)
        with warnings.catch_warnings(record=True) as w:
            # don't warn for numpy numbers
            v = Array([], np.float32(1))
            self.assertFalse(w)

    def test_properties(self):
        v = Array(['time', 'x'], self.d, {'foo': 'bar'})
        self.assertEqual(v.dimensions, ('time', 'x'))
        self.assertEqual(v.dtype, float)
        self.assertEqual(v.shape, (10, 3))
        self.assertEqual(v.size, 30)
        self.assertEqual(v.ndim, 2)
        self.assertEqual(len(v), 10)
        self.assertEqual(v.attributes, {'foo': u'bar'})

    def test_repr(self):
        v = Array(['time', 'x'], self.d)
        self.assertEqual('<scidata.Array (time: 10, x: 3): float64>',
                         repr(v))

    def test_items(self):
        data = np.random.random((10, 11))
        v = Array(['x', 'y'], data)
        # test slicing
        self.assertVarEqual(v, v[:])
        self.assertVarEqual(v, v[...])
        self.assertVarEqual(Array(['y'], data[0]), v[0])
        self.assertVarEqual(Array(['x'], data[:, 0]), v[:, 0])
        self.assertVarEqual(Array(['x', 'y'], data[:3, :2]), v[:3, :2])
        # test array indexing
        x = Array(['x'], np.arange(10))
        y = Array(['y'], np.arange(11))
        self.assertVarEqual(v, v[x.data])
        self.assertVarEqual(v, v[x])
        self.assertVarEqual(v[:3], v[x < 3])
        self.assertVarEqual(v[:, 3:], v[:, y >= 3])
        self.assertVarEqual(v[:3, 3:], v[x < 3, y >= 3])
        self.assertVarEqual(v[:3, :2], v[x[:3], y[:2]])
        self.assertVarEqual(v[:3, :2], v[range(3), range(2)])
        # test iteration
        for n, item in enumerate(v):
            self.assertVarEqual(Array(['y'], data[n]), item)
        # test setting
        v.data[:] = 0
        self.assertTrue(np.all(v.data == 0))

    def test_indexed_by(self):
        v = Array(['time', 'x'], self.d)
        self.assertVarEqual(v.indexed_by(time=slice(None)), v)
        self.assertVarEqual(v.indexed_by(time=0), v[0])
        self.assertVarEqual(v.indexed_by(time=slice(0, 3)), v[:3])
        self.assertVarEqual(v.indexed_by(x=0), v[:, 0])
        with self.assertRaisesRegexp(ValueError, 'do not exist'):
            v.indexed_by(not_a_dim=0)

    def test_transpose(self):
        v = Array(['time', 'x'], self.d)
        v2 = Array(['x', 'time'], self.d.T)
        self.assertVarEqual(v, v2.transpose())
        self.assertVarEqual(v.transpose(), v.T)
        x = np.random.randn(2, 3, 4, 5)
        w = Array(['a', 'b', 'c', 'd'], x)
        w2 = Array(['d', 'b', 'c', 'a'], np.einsum('abcd->dbca', x))
        self.assertEqual(w2.shape, (5, 3, 4, 2))
        self.assertVarEqual(w2, w.transpose('d', 'b', 'c', 'a'))
        self.assertVarEqual(w, w2.transpose('a', 'b', 'c', 'd'))
        w3 = Array(['b', 'c', 'd', 'a'], np.einsum('abcd->bcda', x))
        self.assertVarEqual(w, w3.transpose('a', 'b', 'c', 'd'))

    def test_1d_math(self):
        x = np.arange(5)
        y = np.ones(5)
        v = Array(['x'], x)
        # unary ops
        self.assertVarEqual(v, +v)
        self.assertVarEqual(v, abs(v))
        self.assertNDArrayEqual((-v).data, -x)
        # bianry ops with numbers
        self.assertVarEqual(v, v + 0)
        self.assertVarEqual(v, 0 + v)
        self.assertVarEqual(v, v * 1)
        self.assertNDArrayEqual((v > 2).data, x > 2)
        self.assertNDArrayEqual((0 == v).data, 0 == x)
        self.assertNDArrayEqual((v - 1).data, x - 1)
        self.assertNDArrayEqual((1 - v).data, 1 - x)
        # binary ops with numpy arrays
        self.assertNDArrayEqual((v * x).data, x ** 2)
        self.assertNDArrayEqual((x * v).data, x ** 2)
        self.assertNDArrayEqual(v - y, v - 1)
        self.assertNDArrayEqual(y - v, 1 - v)
        # verify attributes
        v2 = Array(['x'], x, {'units': 'meters'})
        self.assertVarEqual(v2, +v2)
        self.assertVarEqual(v2, 0 + v2)
        # binary ops with all variables
        self.assertNDArrayEqual(v + v, 2 * v)
        w = Array(['x'], y, {'foo': 'bar'})
        self.assertVarEqual(v + w, Array(['x'], x + y))
        self.assertNDArrayEqual((v * w).data, x * y)
        # something complicated
        self.assertNDArrayEqual((v ** 2 * w - 1 + x).data, x ** 2 * y - 1 + x)

    def test_broadcasting_math(self):
        x = np.random.randn(2, 3)
        v = Array(['a', 'b'], x)
        # 1d to 2d broadcasting
        self.assertVarEqual(
            v * v,
            Array(['a', 'b'], np.einsum('ab,ab->ab', x, x)))
        self.assertVarEqual(
            v * v[0],
            Array(['a', 'b'], np.einsum('ab,b->ab', x, x[0])))
        self.assertVarEqual(
            v[0] * v,
            Array(['b', 'a'], np.einsum('b,ab->ba', x[0], x)))
        self.assertVarEqual(
            v[0] * v[:, 0],
            Array(['b', 'a'], np.einsum('b,a->ba', x[0], x[:, 0])))
        # higher dim broadcasting
        y = np.random.randn(3, 4, 5)
        w = Array(['b', 'c', 'd'], y)
        self.assertVarEqual(
            v * w, Array(['a', 'b', 'c', 'd'],
                            np.einsum('ab,bcd->abcd', x, y)))
        self.assertVarEqual(
            w * v, Array(['b', 'c', 'd', 'a'],
                            np.einsum('bcd,ab->bcda', y, x)))
        self.assertVarEqual(
            v * w[0], Array(['a', 'b', 'c', 'd'],
                            np.einsum('ab,cd->abcd', x, y[0])))

    def test_broadcasting_failures(self):
        a = Array(['x'], np.arange(10))
        b = Array(['x'], np.arange(5))
        c = Array(['x', 'x'], np.arange(100).reshape(10, 10))
        with self.assertRaisesRegexp(ValueError, 'mismatched lengths'):
            a + b
        with self.assertRaisesRegexp(ValueError, 'duplicate dimensions'):
            a + c

    def test_inplace_math(self):
        x = np.arange(5)
        v = Array(['x'], x)
        v2 = v
        v2 += 1
        self.assertIs(v, v2)
        # since we provided an ndarray for data, it is also modified in-place
        self.assertIs(v.data, x)
        self.assertNDArrayEqual(v.data, np.arange(5) + 1)

    def test_array_interface(self):
        x = np.arange(5)
        v = Array(['x'], x)
        self.assertNDArrayEqual(np.asarray(v), x)
        # test patched in methods
        self.assertNDArrayEqual(v.take([2, 3]), x.take([2, 3]))
        self.assertVarEqual(v.argsort(), v)
        self.assertVarEqual(v.clip(2, 3), Array('x', x.clip(2, 3)))
        # test ufuncs
        self.assertVarEqual(np.sin(v), Array(['x'], np.sin(x)))

    def test_apply(self):
        x = np.arange(5)
        v = Array(['x'], x)
        def numpy_only_square(x):
            return np.asarray(x) ** 2
        self.assertNDArrayEqual(x ** 2, numpy_only_square(v))
        self.assertVarEqual(v ** 2, v.apply(numpy_only_square))

    def test_collapse(self):
        v = Array(['time', 'x'], self.d)
        # intentionally test with an operation for which order matters
        self.assertVarEqual(v.collapse(np.std, 'time'),
                            Array(['x'], self.d.std(axis=0),
                                  {'cell_methods': 'time: std'}))
        self.assertVarEqual(v.collapse(np.std, axis=0),
                            v.collapse(np.std, dimension='time'))
        self.assertVarEqual(v.collapse(np.std, ['x', 'time']),
                            Array([], self.d.std(axis=1).std(axis=0),
                                  {'cell_methods': 'x: std time: std'}))
        self.assertVarEqual(v.collapse(np.std),
                            Array([], self.d.std(),
                                     {'cell_methods': 'time: x: std'}))
        self.assertVarEqual(v.mean('time'), v.collapse(np.mean, 'time'))

    def test_aggregate(self):
        agg_var = Array(['y'], np.array(['a', 'a', 'b']))
        v = Array(['x', 'y'], self.d)
        expected_unique = Array(['abc'], np.array(['a', 'b']))
        expected_aggregated = Array(['x', 'abc'],
                                       np.array([self.d[:, :2].sum(axis=1),
                                                 self.d[:, 2:].sum(axis=1)]).T,
                                       {'cell_methods': 'y: sum'})
        actual_unique, actual_aggregated = v.aggregate(np.sum, 'abc', agg_var)
        self.assertVarEqual(expected_unique, actual_unique)
        self.assertVarEqual(expected_aggregated, actual_aggregated)
        # should be equivalent to aggregate by a dataset array, too
        alt_agg_var = Dataset({'abc': agg_var})['abc']
        actual_unique, actual_aggregated = v.aggregate(np.sum, 'abc',
                                                       alt_agg_var)
        self.assertVarEqual(expected_unique, actual_unique)
        self.assertVarEqual(expected_aggregated, actual_aggregated)

    def test_from_stack(self):
        x = np.arange(5)
        y = np.ones(5)
        v = Array(['a'], x)
        w = Array(['a'], y)
        self.assertVarEqual(Array(['b', 'a'], np.array([x, y])),
                            Array.from_stack([v, w], 'b'))
        self.assertVarEqual(Array(['b', 'a'], np.array([x, y])),
                            Array.from_stack((v, w), 'b'))
        self.assertVarEqual(Array(['b', 'a'], np.array([x, y])),
                            Array.from_stack((v, w), 'b', length=2))
        with self.assertRaisesRegexp(ValueError, 'greater than expected'):
            Array.from_stack([v, w], 'b', length=1)
        with self.assertRaisesRegexp(ValueError, 'but expected length was'):
            Array.from_stack([v, w, w], 'b', length=4)
        with self.assertRaisesRegexp(ValueError, 'inconsistent dimensions'):
            Array.from_stack([v, Array(['c'], y)], 'b')
        # test concatenating along a dimension
        v = Array(['time', 'x'], np.random.random((10, 8)))
        self.assertVarEqual(v, Array.from_stack([v[:5], v[5:]], 'time'))
        self.assertVarEqual(v, Array.from_stack([v[:5], v[5], v[6:]], 'time'))
        self.assertVarEqual(v, Array.from_stack([v[0], v[1:]], 'time'))
        # test dimension order
        self.assertVarEqual(v, Array.from_stack([v[:, :5], v[:, 5:]], 'x'))
        self.assertVarEqual(v.transpose(),
                            Array.from_stack([v[:, 0], v[:, 1:]], 'x'))
