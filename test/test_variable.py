from collections import namedtuple
from copy import deepcopy
from datetime import datetime
from textwrap import dedent

import numpy as np
import pandas as pd

from xray import Variable, Dataset, DataArray
from xray.variable import CoordVariable, as_variable
from . import TestCase


class VariableSubclassTestCases(object):
    def test_properties(self):
        data = 0.5 * np.arange(10)
        v = Variable(['time'], data, {'foo': 'bar'})
        self.assertEqual(v.dimensions, ('time',))
        self.assertArrayEqual(v.values, data)
        self.assertTrue(pd.Index(data).equals(v.as_index))
        self.assertEqual(v.dtype, float)
        self.assertEqual(v.shape, (10,))
        self.assertEqual(v.size, 10)
        self.assertEqual(v.ndim, 1)
        self.assertEqual(len(v), 10)
        self.assertEqual(v.attributes, {'foo': u'bar'})

    def test_0d_data(self):
        d = datetime(2000, 1, 1)
        for value, dtype in [(0, int),
                             (np.float32(0.5), np.float32),
                             ('foo', np.string_),
                             (d, None),
                             (np.datetime64(d), np.datetime64)]:
            x = self.cls(['x'], [value])
            # check array properties
            self.assertEqual(x[0].shape, ())
            self.assertEqual(x[0].ndim, 0)
            self.assertEqual(x[0].size, 1)
            # check value is equal for both ndarray and Variable
            self.assertEqual(x.values[0], value)
            self.assertEqual(x[0].values, value)
            # check type or dtype is consistent for both ndarray and Variable
            if dtype is None:
                # check output type instead of array dtype
                self.assertEqual(type(x.values[0]), type(value))
                self.assertEqual(type(x[0].values), type(value))
            else:
                self.assertTrue(np.issubdtype(x.values[0].dtype, dtype))
                self.assertTrue(np.issubdtype(x[0].values.dtype, dtype))

    def test_pandas_data(self):
        v = self.cls(['x'], pd.Series([0, 1, 2], index=[3, 2, 1]))
        self.assertVariableEqual(v, v[[0, 1, 2]])
        v = self.cls(['x'], pd.Index([0, 1, 2]))
        self.assertEqual(v[0].values, v.values[0])

    def test_1d_math(self):
        x = 1.0 * np.arange(5)
        y = np.ones(5)
        v = self.cls(['x'], x)
        # unary ops
        self.assertVariableEqual(v, +v)
        self.assertVariableEqual(v, abs(v))
        self.assertArrayEqual((-v).values, -x)
        # bianry ops with numbers
        self.assertVariableEqual(v, v + 0)
        self.assertVariableEqual(v, 0 + v)
        self.assertVariableEqual(v, v * 1)
        self.assertArrayEqual((v > 2).values, x > 2)
        self.assertArrayEqual((0 == v).values, 0 == x)
        self.assertArrayEqual((v - 1).values, x - 1)
        self.assertArrayEqual((1 - v).values, 1 - x)
        # binary ops with numpy arrays
        self.assertArrayEqual((v * x).values, x ** 2)
        self.assertArrayEqual((x * v).values, x ** 2)
        self.assertArrayEqual(v - y, v - 1)
        self.assertArrayEqual(y - v, 1 - v)
        # verify math-safe attributes
        v2 = self.cls(['x'], x, {'units': 'meters'})
        self.assertVariableEqual(v, +v2)
        v3 = self.cls(['x'], x, {'something': 'else'})
        self.assertVariableEqual(v3, +v3)
        # binary ops with all variables
        self.assertArrayEqual(v + v, 2 * v)
        w = self.cls(['x'], y, {'foo': 'bar'})
        self.assertVariableEqual(v + w, self.cls(['x'], x + y))
        self.assertArrayEqual((v * w).values, x * y)
        # something complicated
        self.assertArrayEqual((v ** 2 * w - 1 + x).values, x ** 2 * y - 1 + x)
        # make sure dtype is preserved (for CoordVariables)
        self.assertEqual(float, (+v).dtype)
        self.assertEqual(float, (+v).values.dtype)
        self.assertEqual(float, (0 + v).dtype)
        self.assertEqual(float, (0 + v).values.dtype)
        # check types of returned data
        self.assertIsInstance(+v, Variable)
        self.assertNotIsInstance(+v, CoordVariable)
        self.assertIsInstance(0 + v, Variable)
        self.assertNotIsInstance(0 + v, CoordVariable)

    def test_1d_reduce(self):
        x = np.arange(5)
        v = self.cls(['x'], x)
        actual = v.sum()
        expected = Variable((), 10, {'cell_methods': 'x: sum'})
        self.assertVariableEqual(expected, actual)
        self.assertIs(type(actual), Variable)

    def test_array_interface(self):
        x = np.arange(5)
        v = self.cls(['x'], x)
        self.assertArrayEqual(np.asarray(v), x)
        # test patched in methods
        self.assertArrayEqual(v.take([2, 3]), x.take([2, 3]))
        self.assertVariableEqual(v.argsort(), v)
        self.assertVariableEqual(v.clip(2, 3), self.cls('x', x.clip(2, 3)))
        # test ufuncs
        self.assertVariableEqual(np.sin(v), self.cls(['x'], np.sin(x)))
        self.assertIsInstance(np.sin(v), Variable)
        self.assertNotIsInstance(np.sin(v), CoordVariable)

    def test_concat(self):
        x = np.arange(5)
        y = np.ones(5)
        v = self.cls(['a'], x)
        w = self.cls(['a'], y)
        self.assertVariableEqual(Variable(['b', 'a'], np.array([x, y])),
                               Variable.concat([v, w], 'b'))
        self.assertVariableEqual(Variable(['b', 'a'], np.array([x, y])),
                               Variable.concat((v, w), 'b'))
        self.assertVariableEqual(Variable(['b', 'a'], np.array([x, y])),
                               Variable.concat((v, w), 'b', length=2))
        with self.assertRaisesRegexp(ValueError, 'actual length'):
            Variable.concat([v, w], 'b', length=1)
        with self.assertRaisesRegexp(ValueError, 'actual length'):
            Variable.concat([v, w, w], 'b', length=4)
        with self.assertRaisesRegexp(ValueError, 'inconsistent dimensions'):
            Variable.concat([v, Variable(['c'], y)], 'b')
        # test concatenating along a dimension
        v = Variable(['time', 'x'], np.random.random((10, 8)))
        self.assertVariableEqual(v, Variable.concat([v[:5], v[5:]], 'time'))
        self.assertVariableEqual(v, Variable.concat([v[:5], v[5], v[6:]], 'time'))
        self.assertVariableEqual(v, Variable.concat([v[0], v[1:]], 'time'))
        # test dimension order
        self.assertVariableEqual(v, Variable.concat([v[:, :5], v[:, 5:]], 'x'))
        self.assertVariableEqual(v.transpose(),
                               Variable.concat([v[:, 0], v[:, 1:]], 'x'))

    def test_copy(self):
        v = self.cls('x', 0.5 * np.arange(10))
        for deep in [True, False]:
            w = v.copy(deep=deep)
            self.assertIs(type(v), type(w))
            self.assertVariableEqual(v, w)
            self.assertEqual(v.dtype, w.dtype)
            if self.cls is Variable:
                if deep:
                    self.assertIsNot(v.values, w.values)
                else:
                    self.assertIs(v.values, w.values)


class TestVariable(TestCase, VariableSubclassTestCases):
    cls = staticmethod(Variable)

    def setUp(self):
        self.d = np.random.random((10, 3)).astype(np.float64)

    def test_data(self):
        v = Variable(['time', 'x'], self.d, indexing_mode='not-supported')
        self.assertIs(v.values, self.d)
        with self.assertRaises(ValueError):
            # wrong size
            v.values = np.random.random(5)
        d2 = np.random.random((10, 3))
        v.values = d2
        self.assertIs(v.values, d2)
        self.assertEqual(v._indexing_mode, 'numpy')

    def test_array_equality(self):
        d = np.random.rand(10, 3)
        v1 = Variable(('dim1', 'dim2'), data=d,
                     attributes={'att1': 3, 'att2': [1, 2, 3]})
        v2 = Variable(('dim1', 'dim2'), data=d,
                     attributes={'att1': 3, 'att2': [1, 2, 3]})
        v3 = Variable(('dim1', 'dim3'), data=d,
                    attributes={'att1': 3, 'att2': [1, 2, 3]})
        v4 = Variable(('dim1', 'dim2'), data=d,
                    attributes={'att1': 3, 'att2': [1, 2, 4]})
        v5 = deepcopy(v1)
        v5.values[:] = np.random.rand(10, 3)
        self.assertVariableEqual(v1, v2)
        self.assertVariableNotEqual(v1, v3)
        self.assertVariableNotEqual(v1, v4)
        self.assertVariableNotEqual(v1, v5)

    def test_as_variable(self):
        data = np.arange(10)
        expected = Variable('x', data)

        self.assertVariableEqual(expected, as_variable(expected))

        ds = Dataset({'x': expected})
        self.assertVariableEqual(expected, as_variable(ds['x']))
        self.assertNotIsInstance(ds['x'], Variable)
        self.assertIsInstance(as_variable(ds['x']), Variable)
        self.assertIsInstance(as_variable(ds['x'], strict=False), DataArray)

        FakeVariable = namedtuple('FakeVariable', 'values dimensions')
        fake_xarray = FakeVariable(expected.values, expected.dimensions)
        self.assertVariableEqual(expected, as_variable(fake_xarray))

        xarray_tuple = (expected.dimensions, expected.values)
        self.assertVariableEqual(expected, as_variable(xarray_tuple))

        with self.assertRaisesRegexp(TypeError, 'cannot convert numpy'):
            as_variable(data)
        with self.assertRaisesRegexp(TypeError, 'cannot convert arg'):
            as_variable(list(data))


    def test_repr(self):
        v = Variable(['time', 'x'], [[1, 2, 3], [4, 5, 6]], {'foo': 'bar'})
        expected = dedent("""
        <xray.Variable (time: 2, x: 3)>
        array([[1, 2, 3],
               [4, 5, 6]])
        Attributes:
            foo: bar
        """).strip()
        self.assertEqual(expected, repr(v))

    def test_items(self):
        data = np.random.random((10, 11))
        v = Variable(['x', 'y'], data)
        # test slicing
        self.assertVariableEqual(v, v[:])
        self.assertVariableEqual(v, v[...])
        self.assertVariableEqual(Variable(['y'], data[0]), v[0])
        self.assertVariableEqual(Variable(['x'], data[:, 0]), v[:, 0])
        self.assertVariableEqual(Variable(['x', 'y'], data[:3, :2]), v[:3, :2])
        # test array indexing
        x = Variable(['x'], np.arange(10))
        y = Variable(['y'], np.arange(11))
        self.assertVariableEqual(v, v[x.values])
        self.assertVariableEqual(v, v[x])
        self.assertVariableEqual(v[:3], v[x < 3])
        self.assertVariableEqual(v[:, 3:], v[:, y >= 3])
        self.assertVariableEqual(v[:3, 3:], v[x < 3, y >= 3])
        self.assertVariableEqual(v[:3, :2], v[x[:3], y[:2]])
        self.assertVariableEqual(v[:3, :2], v[range(3), range(2)])
        # test iteration
        for n, item in enumerate(v):
            self.assertVariableEqual(Variable(['y'], data[n]), item)
        with self.assertRaisesRegexp(TypeError, 'iteration over a 0-d'):
            iter(Variable([], 0))
        # test setting
        v.values[:] = 0
        self.assertTrue(np.all(v.values == 0))

    def test_indexed(self):
        v = Variable(['time', 'x'], self.d)
        self.assertVariableEqual(v.indexed(time=slice(None)), v)
        self.assertVariableEqual(v.indexed(time=0), v[0])
        self.assertVariableEqual(v.indexed(time=slice(0, 3)), v[:3])
        self.assertVariableEqual(v.indexed(x=0), v[:, 0])
        with self.assertRaisesRegexp(ValueError, 'do not exist'):
            v.indexed(not_a_dim=0)

    def test_transpose(self):
        v = Variable(['time', 'x'], self.d)
        v2 = Variable(['x', 'time'], self.d.T)
        self.assertVariableEqual(v, v2.transpose())
        self.assertVariableEqual(v.transpose(), v.T)
        x = np.random.randn(2, 3, 4, 5)
        w = Variable(['a', 'b', 'c', 'd'], x)
        w2 = Variable(['d', 'b', 'c', 'a'], np.einsum('abcd->dbca', x))
        self.assertEqual(w2.shape, (5, 3, 4, 2))
        self.assertVariableEqual(w2, w.transpose('d', 'b', 'c', 'a'))
        self.assertVariableEqual(w, w2.transpose('a', 'b', 'c', 'd'))
        w3 = Variable(['b', 'c', 'd', 'a'], np.einsum('abcd->bcda', x))
        self.assertVariableEqual(w, w3.transpose('a', 'b', 'c', 'd'))

    def test_squeeze(self):
        v = Variable(['x', 'y'], [[1]])
        self.assertVariableEqual(Variable([], 1), v.squeeze())
        self.assertVariableEqual(Variable(['y'], [1]), v.squeeze('x'))
        self.assertVariableEqual(Variable(['y'], [1]), v.squeeze(['x']))
        self.assertVariableEqual(Variable(['x'], [1]), v.squeeze('y'))
        self.assertVariableEqual(Variable([], 1), v.squeeze(['x', 'y']))

        v = Variable(['x', 'y'], [[1, 2]])
        self.assertVariableEqual(Variable(['y'], [1, 2]), v.squeeze())
        self.assertVariableEqual(Variable(['y'], [1, 2]), v.squeeze('x'))
        with self.assertRaisesRegexp(ValueError, 'cannot select a dimension'):
            v.squeeze('y')

    def test_get_axis_num(self):
        v = Variable(['x', 'y', 'z'], np.random.randn(2, 3, 4))
        self.assertEqual(v.get_axis_num('x'), 0)
        self.assertEqual(v.get_axis_num(['x']), (0,))
        self.assertEqual(v.get_axis_num(['x', 'y']), (0, 1))
        self.assertEqual(v.get_axis_num(['z', 'y', 'x']), (2, 1, 0))
        with self.assertRaisesRegexp(ValueError, 'not found in array dim'):
            v.get_axis_num('foobar')

    def test_broadcasting_math(self):
        x = np.random.randn(2, 3)
        v = Variable(['a', 'b'], x)
        # 1d to 2d broadcasting
        self.assertVariableEqual(
            v * v,
            Variable(['a', 'b'], np.einsum('ab,ab->ab', x, x)))
        self.assertVariableEqual(
            v * v[0],
            Variable(['a', 'b'], np.einsum('ab,b->ab', x, x[0])))
        self.assertVariableEqual(
            v[0] * v,
            Variable(['b', 'a'], np.einsum('b,ab->ba', x[0], x)))
        self.assertVariableEqual(
            v[0] * v[:, 0],
            Variable(['b', 'a'], np.einsum('b,a->ba', x[0], x[:, 0])))
        # higher dim broadcasting
        y = np.random.randn(3, 4, 5)
        w = Variable(['b', 'c', 'd'], y)
        self.assertVariableEqual(
            v * w, Variable(['a', 'b', 'c', 'd'],
                          np.einsum('ab,bcd->abcd', x, y)))
        self.assertVariableEqual(
            w * v, Variable(['b', 'c', 'd', 'a'],
                          np.einsum('bcd,ab->bcda', y, x)))
        self.assertVariableEqual(
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
        self.assertIs(v.values, x)
        self.assertArrayEqual(v.values, np.arange(5) + 1)

    def test_reduce(self):
        v = Variable(['x', 'y'], self.d)
        self.assertVariableEqual(v.reduce(np.std, 'x'),
                               Variable(['y'], self.d.std(axis=0),
                                      {'cell_methods': 'x: std'}))
        self.assertVariableEqual(v.reduce(np.std, axis=0),
                               v.reduce(np.std, dimension='x'))
        self.assertVariableEqual(v.reduce(np.std, ['y', 'x']),
                               Variable([], self.d.std(axis=(0, 1)),
                                      {'cell_methods': 'x: y: std'}))
        self.assertVariableEqual(v.reduce(np.std),
                               Variable([], self.d.std(),
                                      {'cell_methods': 'x: y: std'}))
        self.assertVariableEqual(v.reduce(np.mean, 'x').reduce(np.std, 'y'),
                               Variable([], self.d.mean(axis=0).std(),
                                      {'cell_methods': 'x: mean y: std'}))
        self.assertVariableEqual(v.mean('x'), v.reduce(np.mean, 'x'))


class TestCoordVariable(TestCase, VariableSubclassTestCases):
    cls = staticmethod(CoordVariable)

    def test_init(self):
        with self.assertRaisesRegexp(ValueError, 'must be 1-dimensional'):
            CoordVariable((), 0)

    def test_data(self):
        x = CoordVariable('x', [0, 1, 2], dtype=float)
        # data should be initially saved as an ndarray
        self.assertIs(type(x._data), np.ndarray)
        self.assertEqual(float, x.dtype)
        self.assertArrayEqual(np.arange(3), x)
        self.assertEqual(float, x.values.dtype)
        # after inspecting x.values, the CoordVariable will be saved as an Index
        self.assertIsInstance(x._data, pd.Index)
        with self.assertRaisesRegexp(TypeError, 'cannot be modified'):
            x[:] = 0
