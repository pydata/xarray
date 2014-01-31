import warnings

import numpy as np

import polyglot
from . import TestCase


class TestVariable(TestCase):
    def setUp(self):
        self.d = np.random.random((10, 3))

    def test_data(self):
        v = polyglot.Variable(['time', 'x'], self.d)
        self.assertIs(v.data, self.d)
        with self.assertRaises(ValueError):
            # wrong size
            v.data = np.random.random(5)
        d2 = np.random.random((10, 3))
        v.data = d2
        self.assertIs(v.data, d2)

        with warnings.catch_warnings(record=True) as w:
            v = polyglot.Variable(['x'], range(5))
            self.assertIn("converting data to np.ndarray", str(w[-1].message))
            self.assertIsInstance(v.data, np.ndarray)
        with warnings.catch_warnings(record=True) as w:
            # don't warn for numpy numbers
            v = polyglot.Variable([], np.float32(1))
            self.assertFalse(w)

    def test_properties(self):
        v = polyglot.Variable(['time', 'x'], self.d, {'foo': 'bar'})
        self.assertEqual(v.dimensions, ('time', 'x'))
        self.assertEqual(v.dtype, float)
        self.assertEqual(v.shape, (10, 3))
        self.assertEqual(v.size, 30)
        self.assertEqual(v.ndim, 2)
        self.assertEqual(len(v), 10)
        self.assertEqual(v.attributes, {'foo': u'bar'})

    def test_items(self):
        v = polyglot.Variable(['time', 'x'], self.d)
        self.assertVarEqual(v, v[:])
        self.assertVarEqual(v, v[...])
        self.assertVarEqual(polyglot.Variable(['x'], self.d[0]), v[0])
        self.assertVarEqual(
            polyglot.Variable(['time'], self.d[:, 0]), v[:, 0])
        self.assertVarEqual(
            polyglot.Variable(['time', 'x'], self.d[:3, :2]), v[:3, :2])
        for n, item in enumerate(v):
            self.assertVarEqual(polyglot.Variable(['x'], self.d[n]), item)
        v.data[:] = 0
        self.assertTrue(np.all(v.data == 0))

    def test_views(self):
        v = polyglot.Variable(['time', 'x'], self.d)
        self.assertVarEqual(v.views({'time': slice(None)}), v)
        self.assertVarEqual(v.views({'time': 0}), v[0])
        self.assertVarEqual(v.views({'time': slice(0, 3)}), v[:3])
        self.assertVarEqual(v.views({'x': 0}), v[:, 0])

    def test_1d_math(self):
        x = np.arange(5)
        y = np.ones(5)
        v = polyglot.Variable(['x'], x)
        # unary ops
        self.assertVarEqual(v, +v)
        self.assertVarEqual(v, abs(v))
        self.assertArrayEqual((-v).data, -x)
        # verify attributes
        v2 = polyglot.Variable(['x'], x, {'units': 'meters'})
        self.assertVarEqual(v, +v2)
        v3 = polyglot.Variable(['x'], x, {'some': 'attribute'})
        self.assertVarEqual(v3, +v3)
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
        # binary ops with all variables
        self.assertArrayEqual(v + v, 2 * v)
        w = polyglot.Variable(['x'], y, {'foo': 'bar'})
        self.assertVarEqual(v + w, polyglot.Variable(['x'], x + y, {'foo': 'bar'}))
        self.assertArrayEqual((v * w).data, x * y)
        # something complicated
        self.assertArrayEqual((v ** 2 * w - 1 + x).data, x ** 2 * y - 1 + x)

    def test_broadcasting_math(self):
        x = np.random.randn(2, 3)
        v = polyglot.Variable(['a', 'b'], x)
        # 1d to 2d broadcasting
        self.assertVarEqual(
            v * v,
            polyglot.Variable(['a', 'b'], np.einsum('ab,ab->ab', x, x)))
        self.assertVarEqual(
            v * v[0],
            polyglot.Variable(['a', 'b'], np.einsum('ab,b->ab', x, x[0])))
        self.assertVarEqual(
            v[0] * v,
            polyglot.Variable(['b', 'a'], np.einsum('b,ab->ba', x[0], x)))
        self.assertVarEqual(
            v[0] * v[:, 0],
            polyglot.Variable(['b', 'a'], np.einsum('b,a->ba', x[0], x[:, 0])))
        # higher dim broadcasting
        y = np.random.randn(3, 4, 5)
        w = polyglot.Variable(['b', 'c', 'd'], y)
        self.assertVarEqual(
            v * w, polyglot.Variable(['a', 'b', 'c', 'd'],
                                     np.einsum('ab,bcd->abcd', x, y)))
        self.assertVarEqual(
            w * v, polyglot.Variable(['b', 'c', 'd', 'a'],
                                     np.einsum('bcd,ab->bcda', y, x)))
        self.assertVarEqual(
            v * w[0], polyglot.Variable(['a', 'b', 'c', 'd'],
                                     np.einsum('ab,cd->abcd', x, y[0])))

    def test_inplace_math(self):
        x = np.arange(5)
        v = polyglot.Variable(['x'], x)
        v2 = v
        v2 += 1
        self.assertIs(v, v2)
        # since we provided an ndarray for data, it is also modified in-place
        self.assertIs(v.data, x)
        self.assertArrayEqual(v.data, np.arange(5) + 1)
