import numpy as np
import pandas as pd
from copy import deepcopy
from textwrap import dedent

from xray import (align, concat, broadcast_arrays, Dataset, DataArray,
                  Coordinate, Variable)
from xray.core.pycompat import iteritems, OrderedDict
from . import TestCase, ReturnItem, source_ndarray, unittest, requires_dask


class TestDataArray(TestCase):
    def setUp(self):
        self.attrs = {'attr1': 'value1', 'attr2': 2929}
        self.x = np.random.random((10, 20))
        self.v = Variable(['x', 'y'], self.x)
        self.va = Variable(['x', 'y'], self.x, self.attrs)
        self.ds = Dataset({'foo': self.v})
        self.dv = self.ds['foo']

    def test_repr(self):
        v = Variable(['time', 'x'], [[1, 2, 3], [4, 5, 6]], {'foo': 'bar'})
        data_array = DataArray(v, {'other': np.int64(0)}, name='my_variable')
        expected = dedent("""\
        <xray.DataArray 'my_variable' (time: 2, x: 3)>
        array([[1, 2, 3],
               [4, 5, 6]])
        Coordinates:
            other    int64 0
          * time     (time) int64 0 1
          * x        (x) int64 0 1 2
        Attributes:
            foo: bar""")
        self.assertEqual(expected, repr(data_array))

    def test_properties(self):
        self.assertVariableEqual(self.dv.variable, self.v)
        self.assertArrayEqual(self.dv.values, self.v.values)
        for attr in ['dims', 'dtype', 'shape', 'size', 'nbytes', 'ndim', 'attrs']:
            self.assertEqual(getattr(self.dv, attr), getattr(self.v, attr))
        self.assertEqual(len(self.dv), len(self.v))
        self.assertVariableEqual(self.dv, self.v)
        self.assertItemsEqual(list(self.dv.coords), list(self.ds.coords))
        for k, v in iteritems(self.dv.coords):
            self.assertArrayEqual(v, self.ds.coords[k])
        with self.assertRaises(AttributeError):
            self.dv.dataset
        self.assertIsInstance(self.ds['x'].to_index(), pd.Index)
        with self.assertRaisesRegexp(ValueError, 'must be 1-dimensional'):
            self.ds['foo'].to_index()
        with self.assertRaises(AttributeError):
            self.dv.variable = self.v

    def test_name(self):
        arr = self.dv
        self.assertEqual(arr.name, 'foo')

        copied = arr.copy()
        arr.name = 'bar'
        self.assertEqual(arr.name, 'bar')
        self.assertDataArrayEqual(copied, arr)

        actual = DataArray(Coordinate('x', [3]))
        actual.name = 'y'
        expected = DataArray(Coordinate('y', [3]))
        self.assertDataArrayIdentical(actual, expected)

    def test_dims(self):
        arr = self.dv
        self.assertEqual(arr.dims, ('x', 'y'))

        with self.assertRaisesRegexp(AttributeError, 'you cannot assign'):
            arr.dims = ('w', 'z')

    def test_encoding(self):
        expected = {'foo': 'bar'}
        self.dv.encoding['foo'] = 'bar'
        self.assertEquals(expected, self.dv.encoding)

        expected = {'baz': 0}
        self.dv.encoding = expected
        self.assertEquals(expected, self.dv.encoding)
        self.assertIsNot(expected, self.dv.encoding)

    def test_constructor(self):
        data = np.random.random((2, 3))

        actual = DataArray(data)
        expected = Dataset({None: (['dim_0', 'dim_1'], data)})[None]
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, [['a', 'b'], [-1, -2, -3]])
        expected = Dataset({None: (['dim_0', 'dim_1'], data),
                            'dim_0': ('dim_0', ['a', 'b']),
                            'dim_1': ('dim_1', [-1, -2, -3])})[None]
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, [pd.Index(['a', 'b'], name='x'),
                                  pd.Index([-1, -2, -3], name='y')])
        expected = Dataset({None: (['x', 'y'], data),
                            'x': ('x', ['a', 'b']),
                            'y': ('y', [-1, -2, -3])})[None]
        self.assertDataArrayIdentical(expected, actual)

        coords = [['a', 'b'], [-1, -2, -3]]
        actual = DataArray(data, coords, ['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        coords = [pd.Index(['a', 'b'], name='A'),
                  pd.Index([-1, -2, -3], name='B')]
        actual = DataArray(data, coords, ['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        coords = {'x': ['a', 'b'], 'y': [-1, -2, -3]}
        actual = DataArray(data, coords, ['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        coords = [('x', ['a', 'b']), ('y', [-1, -2, -3])]
        actual = DataArray(data, coords)
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, OrderedDict(coords))
        self.assertDataArrayIdentical(expected, actual)

        expected = Dataset({None: (['x', 'y'], data),
                            'x': ('x', ['a', 'b'])})[None]
        actual = DataArray(data, {'x': ['a', 'b']}, ['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, dims=['x', 'y'])
        expected = Dataset({None: (['x', 'y'], data)})[None]
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, dims=['x', 'y'], name='foo')
        expected = Dataset({'foo': (['x', 'y'], data)})['foo']
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, name='foo')
        expected = Dataset({'foo': (['dim_0', 'dim_1'], data)})['foo']
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, dims=['x', 'y'], attrs={'bar': 2})
        expected = Dataset({None: (['x', 'y'], data, {'bar': 2})})[None]
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, dims=['x', 'y'], encoding={'bar': 2})
        expected = Dataset({None: (['x', 'y'], data, {}, {'bar': 2})})[None]
        self.assertDataArrayIdentical(expected, actual)

    def test_constructor_invalid(self):
        data = np.random.randn(3, 2)

        with self.assertRaisesRegexp(ValueError, 'coords is not dict-like'):
            DataArray(data, [[0, 1, 2]], ['x', 'y'])

        with self.assertRaisesRegexp(ValueError, 'not a subset of the .* dim'):
            DataArray(data, {'x': [0, 1, 2]}, ['a', 'b'])
        with self.assertRaisesRegexp(ValueError, 'not a subset of the .* dim'):
            DataArray(data, {'x': [0, 1, 2]})

        with self.assertRaisesRegexp(TypeError, 'is not a string'):
            DataArray(data, dims=['x', None])

    def test_constructor_from_self_described(self):
        data = [[-0.1, 21], [0, 2]]
        expected = DataArray(data,
                             coords={'x': ['a', 'b'], 'y': [-1, -2]},
                             dims=['x', 'y'], name='foobar',
                             attrs={'bar': 2}, encoding={'foo': 3})
        actual = DataArray(expected)
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(expected.values, actual.coords)
        self.assertDataArrayEqual(expected, actual)

        frame = pd.DataFrame(data, index=pd.Index(['a', 'b'], name='x'),
                             columns=pd.Index([-1, -2], name='y'))
        actual = DataArray(frame)
        self.assertDataArrayEqual(expected, actual)

        series = pd.Series(data[0], index=pd.Index([-1, -2], name='y'))
        actual = DataArray(series)
        self.assertDataArrayEqual(expected[0].reset_coords('x', drop=True),
                                  actual)

        panel = pd.Panel({0: frame})
        actual = DataArray(panel)
        expected = DataArray([data], expected.coords, ['dim_0', 'x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        expected = DataArray(data,
                             coords={'x': ['a', 'b'], 'y': [-1, -2],
                                     'a': 0, 'z': ('x', [-0.5, 0.5])},
                             dims=['x', 'y'])
        actual = DataArray(expected)
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(expected.values, expected.coords)
        self.assertDataArrayIdentical(expected, actual)

        expected = Dataset({'foo': ('foo', ['a', 'b'])})['foo']
        actual = DataArray(pd.Index(['a', 'b'], name='foo'))
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(Coordinate('foo', ['a', 'b']))
        self.assertDataArrayIdentical(expected, actual)

        s = pd.Series(range(2), pd.MultiIndex.from_product([['a', 'b'], [0]]))
        with self.assertRaisesRegexp(NotImplementedError, 'MultiIndex'):
            DataArray(s)

    def test_constructor_from_0d(self):
        expected = Dataset({None: ([], 0)})[None]
        actual = DataArray(0)
        self.assertDataArrayIdentical(expected, actual)

    def test_equals_and_identical(self):
        orig = DataArray(np.arange(5.0), {'a': 42}, dims='x')

        expected = orig
        actual = orig.copy()
        self.assertTrue(expected.equals(actual))
        self.assertTrue(expected.identical(actual))

        actual = expected.rename('baz')
        self.assertTrue(expected.equals(actual))
        self.assertFalse(expected.identical(actual))

        actual = expected.rename({'x': 'xxx'})
        self.assertFalse(expected.equals(actual))
        self.assertFalse(expected.identical(actual))

        actual = expected.copy()
        actual.attrs['foo'] = 'bar'
        self.assertTrue(expected.equals(actual))
        self.assertFalse(expected.identical(actual))

        actual = expected.copy()
        actual['x'] = ('x', -np.arange(5))
        self.assertFalse(expected.equals(actual))
        self.assertFalse(expected.identical(actual))

        actual = expected.reset_coords(drop=True)
        self.assertFalse(expected.equals(actual))
        self.assertFalse(expected.identical(actual))

        actual = orig.copy()
        actual[0] = np.nan
        expected = actual.copy()
        self.assertTrue(expected.equals(actual))
        self.assertTrue(expected.identical(actual))

        actual[:] = np.nan
        self.assertFalse(expected.equals(actual))
        self.assertFalse(expected.identical(actual))

        actual = expected.copy()
        actual['a'] = 100000
        self.assertFalse(expected.equals(actual))
        self.assertFalse(expected.identical(actual))

    def test_equals_failures(self):
        orig = DataArray(np.arange(5.0), {'a': 42}, dims='x')
        self.assertFalse(orig.equals(np.arange(5)))
        self.assertFalse(orig.identical(123))
        self.assertFalse(orig.broadcast_equals({1: 2}))

    def test_broadcast_equals(self):
        a = DataArray([0, 0], {'y': 0}, dims='x')
        b = DataArray([0, 0], {'y': ('x', [0, 0])}, dims='x')
        self.assertTrue(a.broadcast_equals(b))
        self.assertTrue(b.broadcast_equals(a))
        self.assertFalse(a.equals(b))
        self.assertFalse(a.identical(b))

        c = DataArray([0], coords={'x': 0}, dims='y')
        self.assertFalse(a.broadcast_equals(c))
        self.assertFalse(c.broadcast_equals(a))

    def test_getitem(self):
        # strings pull out dataarrays
        self.assertDataArrayIdentical(self.dv, self.ds['foo'])
        x = self.dv['x']
        y = self.dv['y']
        self.assertDataArrayIdentical(self.ds['x'], x)
        self.assertDataArrayIdentical(self.ds['y'], y)

        I = ReturnItem()
        for i in [I[:], I[...], I[x.values], I[x.variable], I[x], I[x, y],
                  I[x.values > -1], I[x.variable > -1], I[x > -1],
                  I[x > -1, y > -1]]:
            self.assertVariableEqual(self.dv, self.dv[i])
        for i in [I[0], I[:, 0], I[:3, :2],
                  I[x.values[:3]], I[x.variable[:3]], I[x[:3]], I[x[:3], y[:4]],
                  I[x.values > 3], I[x.variable > 3], I[x > 3], I[x > 3, y > 3]]:
            self.assertVariableEqual(self.v[i], self.dv[i])

    def test_getitem_dict(self):
        actual = self.dv[{'x': slice(3), 'y': 0}]
        expected = self.dv.isel(x=slice(3), y=0)
        self.assertDataArrayIdentical(expected, actual)

    def test_getitem_coords(self):
        orig = DataArray([[10], [20]],
                         {'x': [1, 2], 'y': [3], 'z': 4,
                          'x2': ('x', ['a', 'b']),
                          'y2': ('y', ['c']),
                          'xy': (['y', 'x'], [['d', 'e']])},
                         dims=['x', 'y'])

        self.assertDataArrayIdentical(orig, orig[:])
        self.assertDataArrayIdentical(orig, orig[:, :])
        self.assertDataArrayIdentical(orig, orig[...])
        self.assertDataArrayIdentical(orig, orig[:2, :1])
        self.assertDataArrayIdentical(orig, orig[[0, 1], [0]])

        actual = orig[0, 0]
        expected = DataArray(
            10, {'x': 1, 'y': 3, 'z': 4, 'x2': 'a', 'y2': 'c', 'xy': 'd'})
        self.assertDataArrayIdentical(expected, actual)

        actual = orig[0, :]
        expected = DataArray(
            [10], {'x': 1, 'y': [3], 'z': 4, 'x2': 'a', 'y2': ('y', ['c']),
                   'xy': ('y', ['d'])},
            dims='y')
        self.assertDataArrayIdentical(expected, actual)

        actual = orig[:, 0]
        expected = DataArray(
            [10, 20], {'x': [1, 2], 'y': 3, 'z': 4, 'x2': ('x', ['a', 'b']),
                       'y2': 'c', 'xy': ('x', ['d', 'e'])},
            dims='x')
        self.assertDataArrayIdentical(expected, actual)

    @requires_dask
    def test_chunk(self):
        unblocked = DataArray(np.ones((3, 4)))
        self.assertIsNone(unblocked.chunks)

        blocked = unblocked.chunk()
        self.assertEqual(blocked.chunks, ((3,), (4,)))

        blocked = unblocked.chunk(chunks=((2, 1), (2, 2)))
        self.assertEqual(blocked.chunks, ((2, 1), (2, 2)))

        blocked = unblocked.chunk(chunks=(3, 3))
        self.assertEqual(blocked.chunks, ((3,), (3, 1)))

        self.assertIsNone(blocked.load().chunks)

    def test_isel(self):
        self.assertDataArrayIdentical(self.dv[0], self.dv.isel(x=0))
        self.assertDataArrayIdentical(self.dv, self.dv.isel(x=slice(None)))
        self.assertDataArrayIdentical(self.dv[:3], self.dv.isel(x=slice(3)))
        self.assertDataArrayIdentical(self.dv[:3, :5],
                                      self.dv.isel(x=slice(3), y=slice(5)))

    def test_sel(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        da = self.ds['foo']
        self.assertDataArrayIdentical(da, da.sel(x=slice(None)))
        self.assertDataArrayIdentical(da[1], da.sel(x='b'))
        self.assertDataArrayIdentical(da[:3], da.sel(x=slice('c')))
        self.assertDataArrayIdentical(da[:3], da.sel(x=['a', 'b', 'c']))
        self.assertDataArrayIdentical(da[:, :4], da.sel(y=(self.ds['y'] < 4)))
        # verify that indexing with a dataarray works
        b = DataArray('b')
        self.assertDataArrayIdentical(da[1], da.sel(x=b))
        self.assertDataArrayIdentical(da[[1]], da.sel(x=slice(b, b)))

    def test_sel_method(self):
        data = DataArray(np.random.randn(3, 4),
                         [('x', [0, 1, 2]), ('y', list('abcd'))])

        expected = data.sel(y=['a', 'b'])
        actual = data.sel(y=['ab', 'ba'], method='pad')
        self.assertDataArrayIdentical(expected, actual)

        expected = data.sel(x=[1, 2])
        actual = data.sel(x=[0.9, 1.9], method='backfill')
        self.assertDataArrayIdentical(expected, actual)

    def test_loc(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        da = self.ds['foo']
        self.assertDataArrayIdentical(da[:3], da.loc[:'c'])
        self.assertDataArrayIdentical(da[1], da.loc['b'])
        self.assertDataArrayIdentical(da[1], da.loc[{'x': 'b'}])
        self.assertDataArrayIdentical(da[1], da.loc['b', ...])
        self.assertDataArrayIdentical(da[:3], da.loc[['a', 'b', 'c']])
        self.assertDataArrayIdentical(da[:3, :4],
                                      da.loc[['a', 'b', 'c'], np.arange(4)])
        self.assertDataArrayIdentical(da[:, :4], da.loc[:, self.ds['y'] < 4])
        da.loc['a':'j'] = 0
        self.assertTrue(np.all(da.values == 0))
        da.loc[{'x': slice('a', 'j')}] = 2
        self.assertTrue(np.all(da.values == 2))

    def test_loc_single_boolean(self):
        data = DataArray([0, 1], coords=[[True, False]])
        self.assertEqual(data.loc[True], 0)
        self.assertEqual(data.loc[False], 1)

    def test_time_components(self):
        dates = pd.date_range('2000-01-01', periods=10)
        da = DataArray(np.arange(1, 11), [('time', dates)])

        self.assertArrayEqual(da['time.dayofyear'], da.values)
        self.assertArrayEqual(da.coords['time.dayofyear'], da.values)

    def test_coords(self):
        # use int64 to ensure repr() consistency on windows
        coords = [Coordinate('x', np.array([-1, -2], 'int64')),
                  Coordinate('y', np.array([0, 1, 2], 'int64'))]
        da = DataArray(np.random.randn(2, 3), coords, name='foo')

        self.assertEquals(2, len(da.coords))

        self.assertEqual(['x', 'y'], list(da.coords))

        self.assertTrue(coords[0].identical(da.coords['x']))
        self.assertTrue(coords[1].identical(da.coords['y']))

        self.assertIn('x', da.coords)
        self.assertNotIn(0, da.coords)
        self.assertNotIn('foo', da.coords)

        with self.assertRaises(KeyError):
            da.coords[0]
        with self.assertRaises(KeyError):
            da.coords['foo']

        expected = dedent("""\
        Coordinates:
          * x        (x) int64 -1 -2
          * y        (y) int64 0 1 2""")
        actual = repr(da.coords)
        self.assertEquals(expected, actual)

    def test_coord_coords(self):
        orig = DataArray([10, 20],
                         {'x': [1, 2], 'x2': ('x', ['a', 'b']), 'z': 4},
                         dims='x')

        actual = orig.coords['x']
        expected = DataArray([1, 2], {'z': 4, 'x2': ('x', ['a', 'b'])},
                             dims='x', name='x')
        self.assertDataArrayIdentical(expected, actual)

        del actual.coords['x2']
        self.assertDataArrayIdentical(
            expected.reset_coords('x2', drop=True), actual)

        actual.coords['x3'] = ('x', ['a', 'b'])
        expected = DataArray([1, 2], {'z': 4, 'x3': ('x', ['a', 'b'])},
                             dims='x', name='x')
        self.assertDataArrayIdentical(expected, actual)

    def test_reset_coords(self):
        data = DataArray(np.zeros((3, 4)),
                         {'bar': ('x', ['a', 'b', 'c']),
                          'baz': ('y', range(4))},
                         dims=['x', 'y'],
                         name='foo')

        actual = data.reset_coords()
        expected = Dataset({'foo': (['x', 'y'], np.zeros((3, 4))),
                            'bar': ('x', ['a', 'b', 'c']),
                            'baz': ('y', range(4))})
        self.assertDatasetIdentical(actual, expected)

        actual = data.reset_coords(['bar', 'baz'])
        self.assertDatasetIdentical(actual, expected)

        actual = data.reset_coords('bar')
        expected = Dataset({'foo': (['x', 'y'], np.zeros((3, 4))),
                            'bar': ('x', ['a', 'b', 'c'])},
                           {'baz': ('y', range(4))})
        self.assertDatasetIdentical(actual, expected)

        actual = data.reset_coords(['bar'])
        self.assertDatasetIdentical(actual, expected)

        actual = data.reset_coords(drop=True)
        expected = DataArray(np.zeros((3, 4)), dims=['x', 'y'], name='foo')
        self.assertDataArrayIdentical(actual, expected)

        actual = data.copy()
        actual.reset_coords(drop=True, inplace=True)
        self.assertDataArrayIdentical(actual, expected)

        actual = data.reset_coords('bar', drop=True)
        expected = DataArray(np.zeros((3, 4)), {'baz': ('y', range(4))},
                             dims=['x', 'y'], name='foo')
        self.assertDataArrayIdentical(actual, expected)

        with self.assertRaisesRegexp(ValueError, 'cannot reset coord'):
            data.reset_coords(inplace=True)
        with self.assertRaises(KeyError):
            data.reset_coords('foo', drop=True)
        with self.assertRaisesRegexp(ValueError, 'cannot be found'):
            data.reset_coords('not_found')
        with self.assertRaisesRegexp(ValueError, 'cannot remove index'):
            data.reset_coords('y')

    def test_assign_coords(self):
        array = DataArray(10)
        actual = array.assign_coords(c = 42)
        expected = DataArray(10, {'c': 42})
        self.assertDataArrayIdentical(actual, expected)

        array = DataArray([1, 2, 3, 4], {'c': ('x', [0, 0, 1, 1])}, dims='x')
        actual = array.groupby('c').assign_coords(d = lambda a: a.mean())
        expected = array.copy()
        expected.coords['d'] = ('x', [1.5, 1.5, 3.5, 3.5])
        self.assertDataArrayIdentical(actual, expected)

    def test_reindex(self):
        foo = self.dv
        bar = self.dv[:2, :2]
        self.assertDataArrayIdentical(foo.reindex_like(bar), bar)

        expected = foo.copy()
        expected[:] = np.nan
        expected[:2, :2] = bar
        self.assertDataArrayIdentical(bar.reindex_like(foo), expected)

        # regression test for #279
        expected = DataArray(np.random.randn(5), dims=["time"])
        time2 = DataArray(np.arange(5), dims="time2")
        actual = expected.reindex(time=time2)
        self.assertDataArrayIdentical(actual, expected)

    def test_reindex_method(self):
        x = DataArray([10, 20], dims='y')
        y = [-0.5, 0.5, 1.5]
        actual = x.reindex(y=y, method='backfill')
        expected = DataArray([10, 20, np.nan], coords=[('y', y)])
        self.assertDataArrayIdentical(expected, actual)

        alt = Dataset({'y': y})
        actual = x.reindex_like(alt, method='backfill')
        self.assertDatasetIdentical(expected, actual)

    def test_rename(self):
        renamed = self.dv.rename('bar')
        self.assertDatasetIdentical(
            renamed.to_dataset(), self.ds.rename({'foo': 'bar'}))
        self.assertEqual(renamed.name, 'bar')

        renamed = self.dv.rename({'foo': 'bar'})
        self.assertDatasetIdentical(
            renamed.to_dataset(), self.ds.rename({'foo': 'bar'}))
        self.assertEqual(renamed.name, 'bar')

    def test_swap_dims(self):
        array = DataArray(np.random.randn(3), {'y': ('x', list('abc'))}, 'x')
        expected = DataArray(array.values,
                             {'y': list('abc'), 'x': ('y', range(3))},
                             dims='y')
        actual = array.swap_dims({'x': 'y'})
        self.assertDataArrayIdentical(expected, actual)

    def test_dataset_getitem(self):
        dv = self.ds['foo']
        self.assertDataArrayIdentical(dv, self.dv)

    def test_array_interface(self):
        self.assertArrayEqual(np.asarray(self.dv), self.x)
        # test patched in methods
        self.assertArrayEqual(self.dv.astype(float), self.v.astype(float))
        self.assertVariableEqual(self.dv.argsort(), self.v.argsort())
        self.assertVariableEqual(self.dv.clip(2, 3), self.v.clip(2, 3))
        # test ufuncs
        expected = deepcopy(self.ds)
        expected['foo'][:] = np.sin(self.x)
        self.assertDataArrayEqual(expected['foo'], np.sin(self.dv))
        self.assertDataArrayEqual(self.dv, np.maximum(self.v, self.dv))
        bar = Variable(['x', 'y'], np.zeros((10, 20)))
        self.assertDataArrayEqual(self.dv, np.maximum(self.dv, bar))

    def test_is_null(self):
        x = np.random.RandomState(42).randn(5, 6)
        x[x < 0] = np.nan
        original = DataArray(x, [-np.arange(5), np.arange(6)], ['x', 'y'])
        expected = DataArray(pd.isnull(x), [-np.arange(5), np.arange(6)],
                             ['x', 'y'])
        self.assertDataArrayIdentical(expected, original.isnull())
        self.assertDataArrayIdentical(~expected, original.notnull())

    def test_math(self):
        x = self.x
        v = self.v
        a = self.dv
        # variable math was already tested extensively, so let's just make sure
        # that all types are properly converted here
        self.assertDataArrayEqual(a, +a)
        self.assertDataArrayEqual(a, a + 0)
        self.assertDataArrayEqual(a, 0 + a)
        self.assertDataArrayEqual(a, a + 0 * v)
        self.assertDataArrayEqual(a, 0 * v + a)
        self.assertDataArrayEqual(a, a + 0 * x)
        self.assertDataArrayEqual(a, 0 * x + a)
        self.assertDataArrayEqual(a, a + 0 * a)
        self.assertDataArrayEqual(a, 0 * a + a)

    def test_math_automatic_alignment(self):
        a = DataArray(range(5), [('x', range(5))])
        b = DataArray(range(5), [('x', range(1, 6))])
        expected = DataArray(np.ones(4), [('x', [1, 2, 3, 4])])
        self.assertDataArrayIdentical(a - b, expected)

        with self.assertRaisesRegexp(ValueError, 'no overlapping labels'):
            a.isel(x=slice(2)) + a.isel(x=slice(2, None))

    def test_inplace_math_basics(self):
        x = self.x
        v = self.v
        a = self.dv
        b = a
        b += 1
        self.assertIs(b, a)
        self.assertIs(b.variable, v)
        self.assertArrayEqual(b.values, x)
        self.assertIs(source_ndarray(b.values), x)
        self.assertDatasetIdentical(b._dataset, self.ds)

    def test_inplace_math_automatic_alignment(self):
        a = DataArray(range(5), [('x', range(5))])
        b = DataArray(range(1, 6), [('x', range(1, 6))])
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            a += b
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            b += a

    def test_math_name(self):
        # Verify that name is preserved only when it can be done unambiguously.
        # The rule (copied from pandas.Series) is keep the current name only if
        # the other object has the same name or no name attribute and this
        # object isn't a coordinate; otherwise reset to None.
        a = self.dv
        self.assertEqual((+a).name, 'foo')
        self.assertEqual((a + 0).name, 'foo')
        self.assertIs((a + a.rename(None)).name, None)
        self.assertIs((a + a.rename('bar')).name, None)
        self.assertEqual((a + a).name, 'foo')
        self.assertIs((+a['x']).name, None)
        self.assertIs((a['x'] + 0).name, None)
        self.assertIs((a + a['x']).name, None)

    def test_math_with_coords(self):
        coords = {'x': [-1, -2], 'y': ['ab', 'cd', 'ef'],
                  'lat': (['x', 'y'], [[1, 2, 3], [-1, -2, -3]]),
                  'c': -999}
        orig = DataArray(np.random.randn(2, 3), coords, dims=['x', 'y'])

        actual = orig + 1
        expected = DataArray(orig.values + 1, orig.coords)
        self.assertDataArrayIdentical(expected, actual)

        actual = 1 + orig
        self.assertDataArrayIdentical(expected, actual)

        actual = orig + orig[0, 0]
        exp_coords = dict((k, v) for k, v in coords.items() if k != 'lat')
        expected = DataArray(orig.values + orig.values[0, 0],
                             exp_coords, dims=['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        actual = orig[0, 0] + orig
        self.assertDataArrayIdentical(expected, actual)

        actual = orig[0, 0] + orig[-1, -1]
        expected = DataArray(orig.values[0, 0] + orig.values[-1, -1],
                             {'c': -999})
        self.assertDataArrayIdentical(expected, actual)

        actual = orig[:, 0] + orig[0, :]
        exp_values = orig[:, 0].values[:, None] + orig[0, :].values[None, :]
        expected = DataArray(exp_values, exp_coords, dims=['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        actual = orig[0, :] + orig[:, 0]
        self.assertDataArrayIdentical(expected.T, actual)

        actual = orig - orig.T
        expected = DataArray(np.zeros((2, 3)), orig.coords)
        self.assertDataArrayIdentical(expected, actual)

        actual = orig.T - orig
        self.assertDataArrayIdentical(expected.T, actual)

        alt = DataArray([1, 1], {'x': [-1, -2], 'c': 'foo', 'd': 555}, 'x')
        actual = orig + alt
        expected = orig + 1
        expected.coords['d'] = 555
        del expected.coords['c']
        self.assertDataArrayIdentical(expected, actual)

        actual = alt + orig
        self.assertDataArrayIdentical(expected, actual)

    def test_index_math(self):
        orig = DataArray(range(3), dims='x', name='x')
        actual = orig + 1
        expected = DataArray(1 + np.arange(3), coords=[('x', range(3))])
        self.assertDataArrayIdentical(expected, actual)

        # regression tests for #254
        actual = orig[0] < orig
        expected = DataArray([False, True, True], coords=[('x', range(3))])
        self.assertDataArrayIdentical(expected, actual)

        actual = orig > orig[0]
        self.assertDataArrayIdentical(expected, actual)

    def test_dataset_math(self):
        # more comprehensive tests with multiple dataset variables
        obs = Dataset({'tmin': ('x', np.arange(5)),
                       'tmax': ('x', 10 + np.arange(5))},
                      {'x': ('x', 0.5 * np.arange(5)),
                       'loc': ('x', range(-2, 3))})

        actual = 2 * obs['tmax']
        expected = DataArray(2 * (10 + np.arange(5)), obs.coords, name='tmax')
        self.assertDataArrayIdentical(actual, expected)

        actual = obs['tmax'] - obs['tmin']
        expected = DataArray(10 * np.ones(5), obs.coords)
        self.assertDataArrayIdentical(actual, expected)

        sim = Dataset({'tmin': ('x', 1 + np.arange(5)),
                       'tmax': ('x', 11 + np.arange(5)),
                       # does *not* include 'loc' as a coordinate
                       'x': ('x', 0.5 * np.arange(5))})

        actual = sim['tmin'] - obs['tmin']
        expected = DataArray(np.ones(5), obs.coords, name='tmin')
        self.assertDataArrayIdentical(actual, expected)

        actual = -obs['tmin'] + sim['tmin']
        self.assertDataArrayIdentical(actual, expected)

        actual = sim['tmin'].copy()
        actual -= obs['tmin']
        self.assertDataArrayIdentical(actual, expected)

        actual = sim.copy()
        actual['tmin'] = sim['tmin'] - obs['tmin']
        expected = Dataset({'tmin': ('x', np.ones(5)),
                            'tmax': ('x', sim['tmax'].values)},
                            obs.coords)
        self.assertDatasetIdentical(actual, expected)

        actual = sim.copy()
        actual['tmin'] -= obs['tmin']
        self.assertDatasetIdentical(actual, expected)

    def test_transpose(self):
        self.assertVariableEqual(self.dv.variable.transpose(),
                                 self.dv.transpose())

    def test_squeeze(self):
        self.assertVariableEqual(self.dv.variable.squeeze(), self.dv.squeeze())

    def test_drop_coordinates(self):
        expected = DataArray(np.random.randn(2, 3), dims=['x', 'y'])
        arr = expected.copy()
        arr.coords['z'] = 2
        actual = arr.drop('z')
        self.assertDataArrayIdentical(expected, actual)

        with self.assertRaises(ValueError):
            arr.drop('not found')

        with self.assertRaisesRegexp(ValueError, 'cannot drop'):
            arr.drop(None)

        renamed = arr.rename('foo')
        with self.assertRaisesRegexp(ValueError, 'cannot drop'):
            renamed.drop('foo')

    def test_drop_index_labels(self):
        arr = DataArray(np.random.randn(2, 3), dims=['x', 'y'])
        actual = arr.drop([0, 1], dim='y')
        expected = arr[:, 2:]
        self.assertDataArrayIdentical(expected, actual)

    def test_dropna(self):
        x = np.random.randn(4, 4)
        x[::2, 0] = np.nan
        arr = DataArray(x, dims=['a', 'b'])

        actual = arr.dropna('a')
        expected = arr[1::2]
        self.assertDataArrayIdentical(actual, expected)

        actual = arr.dropna('b', how='all')
        self.assertDataArrayIdentical(actual, arr)

        actual = arr.dropna('a', thresh=1)
        self.assertDataArrayIdentical(actual, arr)

        actual = arr.dropna('b', thresh=3)
        expected = arr[:, 1:]
        self.assertDataArrayIdentical(actual, expected)

    def test_reduce(self):
        coords = {'x': [-1, -2], 'y': ['ab', 'cd', 'ef'],
                  'lat': (['x', 'y'], [[1, 2, 3], [-1, -2, -3]]),
                  'c': -999}
        orig = DataArray([[-1, 0, 1], [-3, 0, 3]], coords, dims=['x', 'y'])

        actual = orig.mean()
        expected = DataArray(0, {'c': -999})
        self.assertDataArrayIdentical(expected, actual)

        actual = orig.mean(['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        actual = orig.mean('x')
        expected = DataArray([-2, 0, 2], {'y': coords['y'], 'c': -999}, 'y')
        self.assertDataArrayIdentical(expected, actual)

        actual = orig.mean(['x'])
        self.assertDataArrayIdentical(expected, actual)

        actual = orig.mean('y')
        expected = DataArray([0, 0], {'x': coords['x'], 'c': -999}, 'x')
        self.assertDataArrayIdentical(expected, actual)

        self.assertVariableEqual(self.dv.reduce(np.mean, 'x'),
                                 self.v.reduce(np.mean, 'x'))

        orig = DataArray([[1, 0, np.nan], [3, 0, 3]], coords, dims=['x', 'y'])
        actual = orig.count()
        expected = DataArray(5, {'c': -999})
        self.assertDataArrayIdentical(expected, actual)

    def test_reduce_keep_attrs(self):
        # Test dropped attrs
        vm = self.va.mean()
        self.assertEqual(len(vm.attrs), 0)
        self.assertEqual(vm.attrs, OrderedDict())

        # Test kept attrs
        vm = self.va.mean(keep_attrs=True)
        self.assertEqual(len(vm.attrs), len(self.attrs))
        self.assertEqual(vm.attrs, self.attrs)

    def test_fillna(self):
        a = DataArray([np.nan, 1, np.nan, 3], dims='x')
        actual = a.fillna(-1)
        expected = DataArray([-1, 1, -1, 3], dims='x')
        self.assertDataArrayIdentical(expected, actual)

        b = DataArray(range(4), dims='x')
        actual = a.fillna(b)
        expected = b.copy()
        self.assertDataArrayIdentical(expected, actual)

        actual = a.fillna(range(4))
        self.assertDataArrayIdentical(expected, actual)

        actual = a.fillna(b[:3])
        self.assertDataArrayIdentical(expected, actual)

        actual = a.fillna(b[:0])
        self.assertDataArrayIdentical(a, actual)

        with self.assertRaisesRegexp(TypeError, 'fillna on a DataArray'):
            a.fillna({0: 0})

        with self.assertRaisesRegexp(ValueError, 'broadcast'):
            a.fillna([1, 2])

        fill_value = DataArray([0, 1], dims='y')
        actual = a.fillna(fill_value)
        expected = DataArray([[0, 1], [1, 1], [0, 1], [3, 3]], dims=('x', 'y'))
        self.assertDataArrayIdentical(expected, actual)

        expected = b.copy()
        for target in [a, expected]:
            target.coords['b'] = ('x', [0, 0, 1, 1])
        actual = a.groupby('b').fillna(DataArray([0, 2], dims='b'))
        self.assertDataArrayIdentical(expected, actual)

    def test_groupby_iter(self):
        for ((act_x, act_dv), (exp_x, exp_ds)) in \
                zip(self.dv.groupby('y'), self.ds.groupby('y')):
            self.assertEqual(exp_x, act_x)
            self.assertDataArrayIdentical(exp_ds['foo'], act_dv)
        for ((_, exp_dv), act_dv) in zip(self.dv.groupby('x'), self.dv):
            self.assertDataArrayIdentical(exp_dv, act_dv)

    def make_groupby_example_array(self):
        da = self.dv.copy()
        da.coords['abc'] = ('y', np.array(['a'] * 9 + ['c'] + ['b'] * 10))
        da.coords['y'] = 20 + 100 * da['y']
        return da

    def test_groupby_properties(self):
        grouped = self.make_groupby_example_array().groupby('abc')
        expected_unique = Variable('abc', ['a', 'b', 'c'])
        self.assertVariableEqual(expected_unique, grouped.unique_coord)
        self.assertEqual(3, len(grouped))

    def test_groupby_apply_identity(self):
        expected = self.make_groupby_example_array()
        idx = expected.coords['y']
        identity = lambda x: x
        for g in ['x', 'y', 'abc', idx]:
            for shortcut in [False, True]:
                for squeeze in [False, True]:
                    grouped = expected.groupby(g, squeeze=squeeze)
                    actual = grouped.apply(identity, shortcut=shortcut)
                    self.assertDataArrayIdentical(expected, actual)

    def test_groupby_sum(self):
        array = self.make_groupby_example_array()
        grouped = array.groupby('abc')

        expected_sum_all = Dataset(
            {'foo': Variable(['abc'], np.array([self.x[:, :9].sum(),
                                                self.x[:, 10:].sum(),
                                                self.x[:, 9:10].sum()]).T),
             'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
        self.assertDataArrayAllClose(expected_sum_all, grouped.reduce(np.sum))
        self.assertDataArrayAllClose(expected_sum_all, grouped.sum())

        expected = DataArray([array['y'].values[idx].sum() for idx
                              in [slice(9), slice(10, None), slice(9, 10)]],
                             [['a', 'b', 'c']], ['abc'])
        actual = array['y'].groupby('abc').apply(np.sum)
        self.assertDataArrayAllClose(expected, actual)
        actual = array['y'].groupby('abc').sum()
        self.assertDataArrayAllClose(expected, actual)

        expected_sum_axis1 = Dataset(
            {'foo': (['x', 'abc'], np.array([self.x[:, :9].sum(1),
                                             self.x[:, 10:].sum(1),
                                             self.x[:, 9:10].sum(1)]).T),
             'x': self.ds['x'],
             'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
        self.assertDataArrayAllClose(expected_sum_axis1,
                                     grouped.reduce(np.sum, 'y'))
        self.assertDataArrayAllClose(expected_sum_axis1, grouped.sum('y'))

    def test_groupby_count(self):
        array = DataArray([0, 0, np.nan, np.nan, 0, 0],
                          coords={'cat': ('x', ['a', 'b', 'b', 'c', 'c', 'c'])},
                          dims='x')
        actual = array.groupby('cat').count()
        expected = DataArray([1, 1, 2], coords=[('cat', ['a', 'b', 'c'])])
        self.assertDataArrayIdentical(actual, expected)

    @unittest.skip('needs to be fixed for shortcut=False, keep_attrs=False')
    def test_groupby_reduce_attrs(self):
        array = self.make_groupby_example_array()
        array.attrs['foo'] = 'bar'

        for shortcut in [True, False]:
            for keep_attrs in [True, False]:
                print('shortcut=%s, keep_attrs=%s' % (shortcut, keep_attrs))
                actual = array.groupby('abc').reduce(
                    np.mean, keep_attrs=keep_attrs, shortcut=shortcut)
                expected = array.groupby('abc').mean()
                if keep_attrs:
                    expected.attrs['foo'] = 'bar'
                self.assertDataArrayIdentical(expected, actual)

    def test_groupby_apply_center(self):
        def center(x):
            return x - np.mean(x)

        array = self.make_groupby_example_array()
        grouped = array.groupby('abc')

        expected_ds = array.to_dataset()
        exp_data = np.hstack([center(self.x[:, :9]),
                              center(self.x[:, 9:10]),
                              center(self.x[:, 10:])])
        expected_ds['foo'] = (['x', 'y'], exp_data)
        expected_centered = expected_ds['foo']
        self.assertDataArrayAllClose(expected_centered, grouped.apply(center))

    def test_groupby_apply_ndarray(self):
        # regression test for #326
        array = self.make_groupby_example_array()
        grouped = array.groupby('abc')
        actual = grouped.apply(np.asarray)
        self.assertDataArrayEqual(array, actual)

    def test_groupby_apply_changes_metadata(self):
        def change_metadata(x):
            x.coords['x'] = x.coords['x'] * 2
            x.attrs['fruit'] = 'lemon'
            return x

        array = self.make_groupby_example_array()
        grouped = array.groupby('abc')
        actual = grouped.apply(change_metadata)
        expected = array.copy()
        expected = change_metadata(expected)
        self.assertDataArrayEqual(expected, actual)

    def test_groupby_math(self):
        array = self.make_groupby_example_array()
        for squeeze in [True, False]:
            grouped = array.groupby('x', squeeze=squeeze)

            expected = array + array.coords['x']
            actual = grouped + array.coords['x']
            self.assertDataArrayIdentical(expected, actual)

            actual = array.coords['x'] + grouped
            self.assertDataArrayIdentical(expected, actual)

            ds = array.coords['x'].to_dataset()
            expected = array + ds
            actual = grouped + ds
            self.assertDatasetIdentical(expected, actual)

            actual = ds + grouped
            self.assertDatasetIdentical(expected, actual)

        grouped = array.groupby('abc')
        expected_agg = (grouped.mean() - np.arange(3)).rename(None)
        actual = grouped - DataArray(range(3), [('abc', ['a', 'b', 'c'])])
        actual_agg = actual.groupby('abc').mean()
        self.assertDataArrayAllClose(expected_agg, actual_agg)

        with self.assertRaisesRegexp(TypeError, 'only support binary ops'):
            grouped + 1
        with self.assertRaisesRegexp(TypeError, 'only support binary ops'):
            grouped + grouped
        with self.assertRaisesRegexp(TypeError, 'in-place operations'):
            array += grouped

    def test_groupby_math_not_aligned(self):
        array = DataArray(range(4), {'b': ('x', [0, 0, 1, 1])}, dims='x')
        other = DataArray([10], dims='b')
        actual = array.groupby('b') + other
        expected = DataArray([10, 11, np.nan, np.nan], array.coords)
        self.assertDataArrayIdentical(expected, actual)

        other = DataArray([10], coords={'c': 123}, dims='b')
        actual = array.groupby('b') + other
        expected.coords['c'] = (['x'], [123] * 2 + [np.nan] * 2)
        self.assertDataArrayIdentical(expected, actual)

        other = Dataset({'a': ('b', [10])})
        actual = array.groupby('b') + other
        expected = Dataset({'a': ('x', [10, 11, np.nan, np.nan])},
                           array.coords)
        self.assertDatasetIdentical(expected, actual)

    def test_groupby_restore_dim_order(self):
        array = DataArray(np.random.randn(5, 3),
                          coords={'a': ('x', range(5)), 'b': ('y', range(3))},
                          dims=['x', 'y'])
        for by, expected_dims in [('x', ('x', 'y')),
                                  ('y', ('x', 'y')),
                                  ('a', ('a', 'y')),
                                  ('b', ('x', 'b'))]:
            result = array.groupby(by).apply(lambda x: x.squeeze())
            self.assertEqual(result.dims, expected_dims)

    def test_groupby_first_and_last(self):
        array = DataArray([1, 2, 3, 4, 5], dims='x')
        by = DataArray(['a'] * 2 + ['b'] * 3, dims='x', name='ab')

        expected = DataArray([1, 3], [('ab', ['a', 'b'])])
        actual = array.groupby(by).first()
        self.assertDataArrayIdentical(expected, actual)

        expected = DataArray([2, 5], [('ab', ['a', 'b'])])
        actual = array.groupby(by).last()
        self.assertDataArrayIdentical(expected, actual)

        array = DataArray(np.random.randn(5, 3), dims=['x', 'y'])
        expected = DataArray(array[[0, 2]], {'ab': ['a', 'b']}, ['ab', 'y'])
        actual = array.groupby(by).first()
        self.assertDataArrayIdentical(expected, actual)

        actual = array.groupby('x').first()
        expected = array # should be a no-op
        self.assertDataArrayIdentical(expected, actual)

    def test_resample(self):
        times = pd.date_range('2000-01-01', freq='6H', periods=10)
        array = DataArray(np.arange(10), [('time', times)])

        actual = array.resample('6H', dim='time')
        self.assertDataArrayIdentical(array, actual)

        actual = array.resample('24H', dim='time')
        expected = DataArray(array.to_series().resample('24H'))
        self.assertDataArrayIdentical(expected, actual)

        actual = array.resample('24H', dim='time', how=np.mean)
        self.assertDataArrayIdentical(expected, actual)

        with self.assertRaisesRegexp(ValueError, 'index must be monotonic'):
            array[[2, 0, 1]].resample('1D', dim='time')

    def test_resample_first(self):
        times = pd.date_range('2000-01-01', freq='6H', periods=10)
        array = DataArray(np.arange(10), [('time', times)])

        actual = array.resample('1D', dim='time', how='first')
        expected = DataArray([0, 4, 8], [('time', times[::4])])
        self.assertDataArrayIdentical(expected, actual)

        # verify that labels don't use the first value
        actual = array.resample('24H', dim='time', how='first')
        expected = DataArray(array.to_series().resample('24H', how='first'))
        self.assertDataArrayIdentical(expected, actual)

        # missing values
        array = array.astype(float)
        array[:2] = np.nan
        actual = array.resample('1D', dim='time', how='first')
        expected = DataArray([2, 4, 8], [('time', times[::4])])
        self.assertDataArrayIdentical(expected, actual)

        actual = array.resample('1D', dim='time', how='first', skipna=False)
        expected = DataArray([np.nan, 4, 8], [('time', times[::4])])
        self.assertDataArrayIdentical(expected, actual)

    def test_resample_skipna(self):
        times = pd.date_range('2000-01-01', freq='6H', periods=10)
        array = DataArray(np.ones(10), [('time', times)])
        array[1] = np.nan

        actual = array.resample('1D', dim='time', skipna=False)
        expected = DataArray([np.nan, 1, 1], [('time', times[::4])])
        self.assertDataArrayIdentical(expected, actual)

    def test_resample_upsampling(self):
        times = pd.date_range('2000-01-01', freq='1D', periods=5)
        array = DataArray(np.arange(5), [('time', times)])

        expected_time = pd.date_range('2000-01-01', freq='12H', periods=9)
        expected = array.reindex(time=expected_time)
        for how in ['mean', 'median', 'sum', 'first', 'last', np.mean]:
            actual = array.resample('12H', 'time', how=how)
            self.assertDataArrayIdentical(expected, actual)

    def test_concat(self):
        self.ds['bar'] = Variable(['x', 'y'], np.random.randn(10, 20))
        foo = self.ds['foo']
        bar = self.ds['bar']
        # from dataset array:
        expected = DataArray(np.array([foo.values, bar.values]),
                             dims=['w', 'x', 'y'])
        actual = concat([foo, bar], 'w')
        self.assertDataArrayEqual(expected, actual)
        # from iteration:
        grouped = [g for _, g in foo.groupby('x')]
        stacked = concat(grouped, self.ds['x'])
        self.assertDataArrayIdentical(foo, stacked)
        # with an index as the 'dim' argument
        stacked = concat(grouped, self.ds.indexes['x'])
        self.assertDataArrayIdentical(foo, stacked)

        actual = concat([foo[0], foo[1]], pd.Index([0, 1])).reset_coords(drop=True)
        expected = foo[:2].rename({'x': 'concat_dim'})
        self.assertDataArrayIdentical(expected, actual)

        actual = concat([foo[0], foo[1]], [0, 1]).reset_coords(drop=True)
        expected = foo[:2].rename({'x': 'concat_dim'})
        self.assertDataArrayIdentical(expected, actual)

        with self.assertRaisesRegexp(ValueError, 'not identical'):
            concat([foo, bar], compat='identical')

    def test_align(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        dv1, dv2 = align(self.dv, self.dv[:5], join='inner')
        self.assertDataArrayIdentical(dv1, self.dv[:5])
        self.assertDataArrayIdentical(dv2, self.dv[:5])

    def test_align_dtype(self):
        # regression test for #264
        x1 = np.arange(30)
        x2 = np.arange(5, 35)
        a = DataArray(np.random.random((30,)).astype('f32'), {'x': x1})
        b = DataArray(np.random.random((30,)).astype('f32'), {'x': x2})
        c, d = align(a, b, join='outer')
        self.assertEqual(c.dtype, np.float32)

    def test_broadcast_arrays(self):
        x = DataArray([1, 2], coords=[('a', [-1, -2])], name='x')
        y = DataArray([1, 2], coords=[('b', [3, 4])], name='y')
        x2, y2 = broadcast_arrays(x, y)
        expected_coords = [('a', [-1, -2]), ('b', [3, 4])]
        expected_x2 = DataArray([[1, 1], [2, 2]], expected_coords, name='x')
        expected_y2 = DataArray([[1, 2], [1, 2]], expected_coords, name='y')
        self.assertDataArrayIdentical(expected_x2, x2)
        self.assertDataArrayIdentical(expected_y2, y2)

        x = DataArray(np.random.randn(2, 3), dims=['a', 'b'])
        y = DataArray(np.random.randn(3, 2), dims=['b', 'a'])
        x2, y2 = broadcast_arrays(x, y)
        expected_x2 = x
        expected_y2 = y.T
        self.assertDataArrayIdentical(expected_x2, x2)
        self.assertDataArrayIdentical(expected_y2, y2)

        with self.assertRaisesRegexp(ValueError, 'cannot broadcast'):
            z = DataArray([1, 2], coords=[('a', [-10, 20])])
            broadcast_arrays(x, z)

    def test_to_pandas(self):
        # 0d
        actual = DataArray(42).to_pandas()
        expected = np.array(42)
        self.assertArrayEqual(actual, expected)

        # 1d
        values = np.random.randn(3)
        index = pd.Index(['a', 'b', 'c'], name='x')
        da = DataArray(values, coords=[index])
        actual = da.to_pandas()
        self.assertArrayEqual(actual.values, values)
        self.assertArrayEqual(actual.index, index)
        self.assertArrayEqual(actual.index.name, 'x')

        # 2d
        values = np.random.randn(3, 2)
        da = DataArray(values, coords=[('x', ['a', 'b', 'c']), ('y', [0, 1])],
                       name='foo')
        actual = da.to_pandas()
        self.assertArrayEqual(actual.values, values)
        self.assertArrayEqual(actual.index, ['a', 'b', 'c'])
        self.assertArrayEqual(actual.columns, [0, 1])

        # roundtrips
        for shape in [(3,), (3, 4), (3, 4, 5)]:
            dims = list('abc')[:len(shape)]
            da = DataArray(np.random.randn(*shape), dims=dims)
            roundtripped = DataArray(da.to_pandas())
            self.assertDataArrayIdentical(da, roundtripped)

        with self.assertRaisesRegexp(ValueError, 'cannot convert'):
            DataArray(np.random.randn(1, 2, 3, 4, 5)).to_pandas()

    def test_to_dataframe(self):
        # regression test for #260
        arr = DataArray(np.random.randn(3, 4),
                        [('B', [1, 2, 3]), ('A', list('cdef'))])
        expected = arr.to_series()
        actual = arr.to_dataframe()[None]
        self.assertArrayEqual(expected.values, actual.values)
        self.assertArrayEqual(expected.name, actual.name)
        self.assertArrayEqual(expected.index.values, actual.index.values)

        # regression test for coords with different dimensions
        arr.coords['C'] = ('B', [-1, -2, -3])
        expected = arr.to_series().to_frame()
        expected['C'] = [-1] * 4 + [-2] * 4 + [-3] * 4
        expected.columns = [None, 'C']
        actual = arr.to_dataframe()
        self.assertArrayEqual(expected.values, actual.values)
        self.assertArrayEqual(expected.columns.values, actual.columns.values)
        self.assertArrayEqual(expected.index.values, actual.index.values)

    def test_to_and_from_series(self):
        expected = self.dv.to_dataframe()['foo']
        actual = self.dv.to_series()
        self.assertArrayEqual(expected.values, actual.values)
        self.assertArrayEqual(expected.index.values, actual.index.values)
        self.assertEqual('foo', actual.name)
        # test roundtrip
        self.assertDataArrayIdentical(self.dv, DataArray.from_series(actual))
        # test name is None
        actual.name = None
        expected_da = self.dv.rename(None)
        self.assertDataArrayIdentical(expected_da,
                                      DataArray.from_series(actual))

    def test_to_and_from_cdms2(self):
        try:
            import cdms2
        except ImportError:
            raise unittest.SkipTest('cdms2 not installed')

        original = DataArray(np.arange(6).reshape(2, 3),
                             [('distance', [-2, 2], {'units': 'meters'}),
                              ('time', pd.date_range('2000-01-01', periods=3))],
                             name='foo', attrs={'baz': 123})
        expected_coords = [Coordinate('distance', [-2, 2]),
                           Coordinate('time', [0, 1, 2])]
        actual = original.to_cdms2()
        self.assertArrayEqual(actual, original)
        self.assertEqual(actual.id, original.name)
        self.assertItemsEqual(actual.getAxisIds(), original.dims)
        for axis, coord in zip(actual.getAxisList(), expected_coords):
            self.assertEqual(axis.id, coord.name)
            self.assertArrayEqual(axis, coord.values)
        self.assertEqual(actual.baz, original.attrs['baz'])

        component_times = actual.getAxis(1).asComponentTime()
        self.assertEqual(len(component_times), 3)
        self.assertEqual(str(component_times[0]), '2000-1-1 0:0:0.0')

        roundtripped = DataArray.from_cdms2(actual)
        self.assertDataArrayIdentical(original, roundtripped)

    def test_to_dataset_whole(self):
        unnamed = DataArray([1, 2], dims='x')
        actual = unnamed.to_dataset()
        expected = Dataset({None: ('x', [1, 2])})
        self.assertDatasetIdentical(expected, actual)
        self.assertIsNot(unnamed._dataset, actual)

        actual = unnamed.to_dataset(name='foo')
        expected = Dataset({'foo': ('x', [1, 2])})
        self.assertDatasetIdentical(expected, actual)

        named = DataArray([1, 2], dims='x', name='foo')
        actual = named.to_dataset()
        expected = Dataset({'foo': ('x', [1, 2])})
        self.assertDatasetIdentical(expected, actual)

        expected = Dataset({'bar': ('x', [1, 2])})
        with self.assertWarns('order of the arguments'):
            actual = named.to_dataset('bar')
        self.assertDatasetIdentical(expected, actual)

    def test_to_dataset_split(self):
        array = DataArray([1, 2, 3], coords=[('x', list('abc'))],
                          attrs={'a': 1})
        expected = Dataset(OrderedDict([('a', 1), ('b', 2), ('c', 3)]),
                           attrs={'a': 1})
        actual = array.to_dataset('x')
        self.assertDatasetIdentical(expected, actual)

        with self.assertRaises(TypeError):
            array.to_dataset('x', name='foo')

        roundtriped = actual.to_array(dim='x')
        self.assertDataArrayIdentical(array, roundtriped)

        array = DataArray([1, 2, 3], dims='x')
        expected = Dataset(OrderedDict([('0', 1), ('1', 2), ('2', 3)]))
        actual = array.to_dataset('x')
        self.assertDatasetIdentical(expected, actual)
