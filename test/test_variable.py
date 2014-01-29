import unittest
import numpy as np

import polyglot


class TestVariable(unittest.TestCase):
    def setUp(self):
        self.d = np.random.random((10, 3))

    def test_data(self):
        v = polyglot.Variable(['time', 'x'], self.d, {'foo': 'bar'})
        self.assertIs(v.data, self.d)
        with self.assertRaises(ValueError):
            # wrong size
            v.data = np.random.random(5)
        d2 = np.random.random((10, 3))
        v.data = d2
        self.assertIs(v.data, d2)

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
        self.assertEqual(v, v[:])
        self.assertEqual(v, v[...])
        self.assertEqual(polyglot.Variable(['x'], self.d[0]), v[0])
        self.assertEqual(polyglot.Variable(['time'], self.d[:, 0]), v[:, 0])
        self.assertEqual(polyglot.Variable(['time', 'x'], self.d[:3, :2]),
                         v[:3, :2])
        self.assertItemsEqual(
            [polyglot.Variable(['x'], self.d[i]) for i in range(10)], v)
        v.data[:] = 0
        self.assertTrue(np.all(v.data == 0))

    def test_views(self):
        v = polyglot.Variable(['time', 'x'], self.d)
        self.assertEqual(v.views({'time': slice(None)}), v)
        self.assertEqual(v.views({'time': 0}), v[0])
        self.assertEqual(v.views({'time': slice(0, 3)}), v[:3])
        self.assertEqual(v.views({'x': 0}), v[:, 0])
