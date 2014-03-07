from copy import deepcopy
from datetime import datetime

import numpy as np

from xray import XArray
from . import TestCase


class TestXArray(TestCase):
    def setUp(self):
        self.d = np.random.random((10, 3)).astype(np.float64)

    def test_data(self):
        v = XArray(['time', 'x'], self.d, indexing_mode='not-supported')
        self.assertIs(v.data, self.d)
        with self.assertRaises(ValueError):
            # wrong size
            v.data = np.random.random(5)
        d2 = np.random.random((10, 3))
        v.data = d2
        self.assertIs(v.data, d2)
        self.assertEqual(v._indexing_mode, 'numpy')

    def test_0d_data(self):
        d = datetime(2000, 1, 1)
        for value, dtype in [(0, int),
                             (np.float32(0.5), np.float32),
                             ('foo', np.string_),
                             (d, None),
                             (np.datetime64(d), np.datetime64)]:
            x = XArray(['x'], [value])
            # check array properties
            self.assertEqual(x[0].shape, ())
            self.assertEqual(x[0].ndim, 0)
            self.assertEqual(x[0].size, 1)
            # check value is equal for both ndarray and XArray
            self.assertEqual(x.data[0], value)
            self.assertEqual(x[0].data, value)
            # check type or dtype is consistent for both ndarray and XArray
            if dtype is None:
                # check output type instead of array dtype
                self.assertEqual(type(x.data[0]), type(value))
                self.assertEqual(type(x[0].data), type(value))
            else:
                self.assertTrue(np.issubdtype(x.data[0].dtype, dtype))
                self.assertTrue(np.issubdtype(x[0].data.dtype, dtype))

    def test_array_equality(self):
        d = np.random.rand(10, 3)
        v1 = XArray(('dim1', 'dim2'), data=d,
                    attributes={'att1': 3, 'att2': [1, 2, 3]})
        v2 = XArray(('dim1', 'dim2'), data=d,
                    attributes={'att1': 3, 'att2': [1, 2, 3]})
        v3 = XArray(('dim1', 'dim3'), data=d,
                   attributes={'att1': 3, 'att2': [1, 2, 3]})
        v4 = XArray(('dim1', 'dim2'), data=d,
                   attributes={'att1': 3, 'att2': [1, 2, 4]})
        v5 = deepcopy(v1)
        v5.data[:] = np.random.rand(10, 3)
        self.assertXArrayEqual(v1, v2)
        self.assertXArrayNotEqual(v1, v3)
        self.assertXArrayNotEqual(v1, v4)
        self.assertXArrayNotEqual(v1, v5)

    def test_properties(self):
        v = XArray(['time', 'x'], self.d, {'foo': 'bar'})
        self.assertEqual(v.dimensions, ('time', 'x'))
        self.assertEqual(v.dtype, float)
        self.assertEqual(v.shape, (10, 3))
        self.assertEqual(v.size, 30)
        self.assertEqual(v.ndim, 2)
        self.assertEqual(len(v), 10)
        self.assertEqual(v.attributes, {'foo': u'bar'})

    def test_repr(self):
        v = XArray(['time', 'x'], self.d)
        self.assertEqual('<xray.XArray (time: 10, x: 3): float64>',
                         repr(v))

    def test_items(self):
        data = np.random.random((10, 11))
        v = XArray(['x', 'y'], data)
        # test slicing
        self.assertXArrayEqual(v, v[:])
        self.assertXArrayEqual(v, v[...])
        self.assertXArrayEqual(XArray(['y'], data[0]), v[0])
        self.assertXArrayEqual(XArray(['x'], data[:, 0]), v[:, 0])
        self.assertXArrayEqual(XArray(['x', 'y'], data[:3, :2]), v[:3, :2])
        # test array indexing
        x = XArray(['x'], np.arange(10))
        y = XArray(['y'], np.arange(11))
        self.assertXArrayEqual(v, v[x.data])
        self.assertXArrayEqual(v, v[x])
        self.assertXArrayEqual(v[:3], v[x < 3])
        self.assertXArrayEqual(v[:, 3:], v[:, y >= 3])
        self.assertXArrayEqual(v[:3, 3:], v[x < 3, y >= 3])
        self.assertXArrayEqual(v[:3, :2], v[x[:3], y[:2]])
        self.assertXArrayEqual(v[:3, :2], v[range(3), range(2)])
        # test iteration
        for n, item in enumerate(v):
            self.assertXArrayEqual(XArray(['y'], data[n]), item)
        # test setting
        v.data[:] = 0
        self.assertTrue(np.all(v.data == 0))

    def test_indexed_by(self):
        v = XArray(['time', 'x'], self.d)
        self.assertXArrayEqual(v.indexed_by(time=slice(None)), v)
        self.assertXArrayEqual(v.indexed_by(time=0), v[0])
        self.assertXArrayEqual(v.indexed_by(time=slice(0, 3)), v[:3])
        self.assertXArrayEqual(v.indexed_by(x=0), v[:, 0])
        with self.assertRaisesRegexp(ValueError, 'do not exist'):
            v.indexed_by(not_a_dim=0)

    def test_transpose(self):
        v = XArray(['time', 'x'], self.d)
        v2 = XArray(['x', 'time'], self.d.T)
        self.assertXArrayEqual(v, v2.transpose())
        self.assertXArrayEqual(v.transpose(), v.T)
        x = np.random.randn(2, 3, 4, 5)
        w = XArray(['a', 'b', 'c', 'd'], x)
        w2 = XArray(['d', 'b', 'c', 'a'], np.einsum('abcd->dbca', x))
        self.assertEqual(w2.shape, (5, 3, 4, 2))
        self.assertXArrayEqual(w2, w.transpose('d', 'b', 'c', 'a'))
        self.assertXArrayEqual(w, w2.transpose('a', 'b', 'c', 'd'))
        w3 = XArray(['b', 'c', 'd', 'a'], np.einsum('abcd->bcda', x))
        self.assertXArrayEqual(w, w3.transpose('a', 'b', 'c', 'd'))

    def test_squeeze(self):
        v = XArray(['x', 'y'], [[1]])
        self.assertXArrayEqual(XArray([], 1), v.squeeze())
        self.assertXArrayEqual(XArray(['y'], [1]), v.squeeze('x'))
        self.assertXArrayEqual(XArray(['y'], [1]), v.squeeze(['x']))
        self.assertXArrayEqual(XArray(['x'], [1]), v.squeeze('y'))
        self.assertXArrayEqual(XArray([], 1), v.squeeze(['x', 'y']))

        v = XArray(['x', 'y'], [[1, 2]])
        self.assertXArrayEqual(XArray(['y'], [1, 2]), v.squeeze())
        self.assertXArrayEqual(XArray(['y'], [1, 2]), v.squeeze('x'))
        with self.assertRaisesRegexp(ValueError, 'cannot select a dimension'):
            v.squeeze('y')

    def test_1d_math(self):
        x = np.arange(5)
        y = np.ones(5)
        v = XArray(['x'], x)
        # unary ops
        self.assertXArrayEqual(v, +v)
        self.assertXArrayEqual(v, abs(v))
        self.assertArrayEqual((-v).data, -x)
        # bianry ops with numbers
        self.assertXArrayEqual(v, v + 0)
        self.assertXArrayEqual(v, 0 + v)
        self.assertXArrayEqual(v, v * 1)
        self.assertArrayEqual((v > 2).data, x > 2)
        self.assertArrayEqual((0 == v).data, 0 == x)
        self.assertArrayEqual((v - 1).data, x - 1)
        self.assertArrayEqual((1 - v).data, 1 - x)
        # binary ops with numpy arrays
        self.assertArrayEqual((v * x).data, x ** 2)
        self.assertArrayEqual((x * v).data, x ** 2)
        self.assertArrayEqual(v - y, v - 1)
        self.assertArrayEqual(y - v, 1 - v)
        # verify math-safe attributes
        v2 = XArray(['x'], x, {'units': 'meters'})
        self.assertXArrayEqual(v, +v2)
        v3 = XArray(['x'], x, {'something': 'else'})
        self.assertXArrayEqual(v3, +v3)
        # binary ops with all variables
        self.assertArrayEqual(v + v, 2 * v)
        w = XArray(['x'], y, {'foo': 'bar'})
        self.assertXArrayEqual(v + w, XArray(['x'], x + y))
        self.assertArrayEqual((v * w).data, x * y)
        # something complicated
        self.assertArrayEqual((v ** 2 * w - 1 + x).data, x ** 2 * y - 1 + x)

    def test_broadcasting_math(self):
        x = np.random.randn(2, 3)
        v = XArray(['a', 'b'], x)
        # 1d to 2d broadcasting
        self.assertXArrayEqual(
            v * v,
            XArray(['a', 'b'], np.einsum('ab,ab->ab', x, x)))
        self.assertXArrayEqual(
            v * v[0],
            XArray(['a', 'b'], np.einsum('ab,b->ab', x, x[0])))
        self.assertXArrayEqual(
            v[0] * v,
            XArray(['b', 'a'], np.einsum('b,ab->ba', x[0], x)))
        self.assertXArrayEqual(
            v[0] * v[:, 0],
            XArray(['b', 'a'], np.einsum('b,a->ba', x[0], x[:, 0])))
        # higher dim broadcasting
        y = np.random.randn(3, 4, 5)
        w = XArray(['b', 'c', 'd'], y)
        self.assertXArrayEqual(
            v * w, XArray(['a', 'b', 'c', 'd'],
                          np.einsum('ab,bcd->abcd', x, y)))
        self.assertXArrayEqual(
            w * v, XArray(['b', 'c', 'd', 'a'],
                          np.einsum('bcd,ab->bcda', y, x)))
        self.assertXArrayEqual(
            v * w[0], XArray(['a', 'b', 'c', 'd'],
                             np.einsum('ab,cd->abcd', x, y[0])))

    def test_broadcasting_failures(self):
        a = XArray(['x'], np.arange(10))
        b = XArray(['x'], np.arange(5))
        c = XArray(['x', 'x'], np.arange(100).reshape(10, 10))
        with self.assertRaisesRegexp(ValueError, 'mismatched lengths'):
            a + b
        with self.assertRaisesRegexp(ValueError, 'duplicate dimensions'):
            a + c

    def test_inplace_math(self):
        x = np.arange(5)
        v = XArray(['x'], x)
        v2 = v
        v2 += 1
        self.assertIs(v, v2)
        # since we provided an ndarray for data, it is also modified in-place
        self.assertIs(v.data, x)
        self.assertArrayEqual(v.data, np.arange(5) + 1)

    def test_array_interface(self):
        x = np.arange(5)
        v = XArray(['x'], x)
        self.assertArrayEqual(np.asarray(v), x)
        # test patched in methods
        self.assertArrayEqual(v.take([2, 3]), x.take([2, 3]))
        self.assertXArrayEqual(v.argsort(), v)
        self.assertXArrayEqual(v.clip(2, 3), XArray('x', x.clip(2, 3)))
        # test ufuncs
        self.assertXArrayEqual(np.sin(v), XArray(['x'], np.sin(x)))

    def test_reduce(self):
        v = XArray(['time', 'x'], self.d)
        # intentionally test with an operation for which order matters
        self.assertXArrayEqual(v.reduce(np.std, 'time'),
                               XArray(['x'], self.d.std(axis=0),
                                      {'cell_methods': 'time: std'}))
        self.assertXArrayEqual(v.reduce(np.std, axis=0),
                               v.reduce(np.std, dimension='time'))
        self.assertXArrayEqual(v.reduce(np.std, ['x', 'time']),
                               XArray([], self.d.std(axis=1).std(axis=0),
                                      {'cell_methods': 'x: std time: std'}))
        self.assertXArrayEqual(v.reduce(np.std),
                               XArray([], self.d.std(),
                                      {'cell_methods': 'time: x: std'}))
        self.assertXArrayEqual(v.mean('time'), v.reduce(np.mean, 'time'))

    def test_groupby(self):
        agg_var = XArray(['y'], np.array(['a', 'a', 'b']))
        v = XArray(['x', 'y'], self.d)

        expected_unique = XArray(['abc'], np.array(['a', 'b']))
        expected_aggregated = XArray(['x', 'abc'],
                                    np.array([self.d[:, :2].sum(axis=1),
                                              self.d[:, 2:].sum(axis=1)]).T,
                                    {'cell_methods': 'y: sum'})

        x = XArray('x', np.arange(10))
        y = XArray('y', np.arange(3))
        self.assertXArrayEqual(v, v.groupby('y', y).apply(lambda x: x))
        self.assertXArrayEqual(v, v.groupby('x', x).apply(lambda x: x))

        grouped = v.groupby('abc', agg_var)
        self.assertXArrayEqual(expected_unique, grouped.unique_coord)
        self.assertXArrayEqual(v, grouped.apply(lambda x: x))
        self.assertXArrayEqual(expected_aggregated, grouped.reduce(np.sum))

        actual = list(grouped)
        expected = zip(expected_unique, [v[:, :2], v[:, 2:]])
        self.assertEqual(len(expected), len(actual))
        for (ke, ve), (ka, va) in zip(expected, actual):
            self.assertXArrayEqual(ke, ka)
            self.assertXArrayEqual(ve, va)

    def test_concat(self):
        x = np.arange(5)
        y = np.ones(5)
        v = XArray(['a'], x)
        w = XArray(['a'], y)
        self.assertXArrayEqual(XArray(['b', 'a'], np.array([x, y])),
                               XArray.concat([v, w], 'b'))
        self.assertXArrayEqual(XArray(['b', 'a'], np.array([x, y])),
                               XArray.concat((v, w), 'b'))
        self.assertXArrayEqual(XArray(['b', 'a'], np.array([x, y])),
                               XArray.concat((v, w), 'b', length=2))
        with self.assertRaisesRegexp(ValueError, 'actual length'):
            XArray.concat([v, w], 'b', length=1)
        with self.assertRaisesRegexp(ValueError, 'actual length'):
            XArray.concat([v, w, w], 'b', length=4)
        with self.assertRaisesRegexp(ValueError, 'inconsistent dimensions'):
            XArray.concat([v, XArray(['c'], y)], 'b')
        # test concatenating along a dimension
        v = XArray(['time', 'x'], np.random.random((10, 8)))
        self.assertXArrayEqual(v, XArray.concat([v[:5], v[5:]], 'time'))
        self.assertXArrayEqual(v, XArray.concat([v[:5], v[5], v[6:]], 'time'))
        self.assertXArrayEqual(v, XArray.concat([v[0], v[1:]], 'time'))
        # test dimension order
        self.assertXArrayEqual(v, XArray.concat([v[:, :5], v[:, 5:]], 'x'))
        self.assertXArrayEqual(v.transpose(),
                               XArray.concat([v[:, 0], v[:, 1:]], 'x'))
