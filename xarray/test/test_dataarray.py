import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from textwrap import dedent

import xarray as xr

from xarray import (align, broadcast, Dataset, DataArray,
                    Coordinate, Variable)
from xarray.core.pycompat import iteritems, OrderedDict
from xarray.core.common import _full_like

from xarray.test import (TestCase, ReturnItem, source_ndarray, unittest, requires_dask,
               requires_bottleneck)


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
        <xarray.DataArray 'my_variable' (time: 2, x: 3)>
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

    def test_data_property(self):
        array = DataArray(np.zeros((3, 4)))
        actual = array.copy()
        actual.values = np.ones((3, 4))
        self.assertArrayEqual(np.ones((3, 4)), actual.values)
        actual.data = 2 * np.ones((3, 4))
        self.assertArrayEqual(2 * np.ones((3, 4)), actual.data)
        self.assertArrayEqual(actual.data, actual.values)

    def test_struct_array_dims(self):
        """
        This test checks subraction of two DataArrays for the case
        when dimension is a structured array.
        """
        # GH837, GH861
        # checking array subraction when dims are the same
        p_data = np.array([('John', 180), ('Stacy', 150), ('Dick', 200)],
                          dtype=[('name', '|S256'), ('height', object)])

        p_data_1 = np.array([('John', 180), ('Stacy', 150), ('Dick', 200)],
                            dtype=[('name', '|S256'), ('height', object)])

        p_data_2 = np.array([('John', 180), ('Dick', 200)],
                            dtype=[('name', '|S256'), ('height', object)])

        weights_0 = DataArray([80, 56, 120], dims=['participant'],
                              coords={'participant': p_data})

        weights_1 = DataArray([81, 52, 115], dims=['participant'],
                              coords={'participant': p_data_1})

        actual = weights_1 - weights_0

        expected = DataArray([1, -4, -5], dims=['participant'],
                             coords={'participant': p_data})

        self.assertDataArrayIdentical(actual, expected)

        # checking array subraction when dims are not the same
        p_data_1 = np.array([('John', 180), ('Stacy', 151), ('Dick', 200)],
                            dtype=[('name', '|S256'), ('height', object)])

        weights_1 = DataArray([81, 52, 115], dims=['participant'],
                              coords={'participant': p_data_1})

        actual = weights_1 - weights_0

        expected = DataArray([1, -5], dims=['participant'],
                             coords={'participant': p_data_2})

        self.assertDataArrayIdentical(actual, expected)

        # checking array subraction when dims are not the same and one
        # is np.nan
        p_data_1 = np.array([('John', 180), ('Stacy', np.nan), ('Dick', 200)],
                            dtype=[('name', '|S256'), ('height', object)])

        weights_1 = DataArray([81, 52, 115], dims=['participant'],
                              coords={'participant': p_data_1})

        actual = weights_1 - weights_0

        expected = DataArray([1, -5], dims=['participant'],
                             coords={'participant': p_data_2})

        self.assertDataArrayIdentical(actual, expected)

    def test_name(self):
        arr = self.dv
        self.assertEqual(arr.name, 'foo')

        copied = arr.copy()
        arr.name = 'bar'
        self.assertEqual(arr.name, 'bar')
        self.assertDataArrayEqual(copied, arr)

        actual = DataArray(Coordinate('x', [3]))
        actual.name = 'y'
        expected = DataArray([3], {'x': [3]}, name='y')
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

        with self.assertRaisesRegexp(ValueError, 'conflicting sizes for dim'):
            DataArray([1, 2, 3], coords=[('x', [0, 1])])
        with self.assertRaisesRegexp(ValueError, 'conflicting sizes for dim'):
            DataArray([1, 2], coords={'x': [0, 1], 'y': ('x', [1])}, dims='x')


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

    def test_pickle(self):
        data = DataArray(np.random.random((3, 3)), dims=('id', 'time'))
        roundtripped = pickle.loads(pickle.dumps(data))
        self.assertDataArrayIdentical(data, roundtripped)

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

        if pd.__version__ >= '0.17':
            expected = data.sel(x=[1, 2])
            actual = data.sel(x=[0.9, 1.9], method='backfill', tolerance=1)
            self.assertDataArrayIdentical(expected, actual)
        else:
            with self.assertRaisesRegexp(NotImplementedError, 'tolerance'):
                data.sel(x=[0.9, 1.9], method='backfill', tolerance=1)

    def test_isel_points(self):
        shape = (10, 5, 6)
        np_array = np.random.random(shape)
        da = DataArray(np_array, dims=['time', 'y', 'x'])
        y = [1, 3]
        x = [3, 0]

        expected = da.values[:, y, x]

        actual = da.isel_points(y=y, x=x, dim='test_coord')
        assert 'test_coord' in actual.coords
        assert actual.coords['test_coord'].shape == (len(y), )
        assert all(x in actual for x in ['time', 'x', 'y', 'test_coord'])
        assert actual.dims == ('test_coord', 'time')
        actual = da.isel_points(y=y, x=x)
        assert 'points' in actual.coords
        # Note that because xarray always concatenates along the first
        # dimension, We must transpose the result to match the numpy style of
        # concatenation.
        np.testing.assert_equal(actual.T, expected)

        # a few corner cases
        da.isel_points(time=[1, 2], x=[2, 2], y=[3, 4])
        np.testing.assert_allclose(
            da.isel_points(time=[1], x=[2], y=[4]).values.squeeze(),
            np_array[1, 4, 2].squeeze())
        da.isel_points(time=[1, 2])
        y = [-1, 0]
        x = [-2, 2]
        expected = da.values[:, y, x]
        actual = da.isel_points(x=x, y=y).values
        np.testing.assert_equal(actual.T, expected)

        # test that the order of the indexers doesn't matter
        self.assertDataArrayIdentical(
            da.isel_points(y=y, x=x),
            da.isel_points(x=x, y=y))

        # make sure we're raising errors in the right places
        with self.assertRaisesRegexp(ValueError,
                                     'All indexers must be the same length'):
            da.isel_points(y=[1, 2], x=[1, 2, 3])
        with self.assertRaisesRegexp(ValueError,
                                     'dimension bad_key does not exist'):
            da.isel_points(bad_key=[1, 2])
        with self.assertRaisesRegexp(TypeError, 'Indexers must be integers'):
            da.isel_points(y=[1.5, 2.2])
        with self.assertRaisesRegexp(TypeError, 'Indexers must be integers'):
            da.isel_points(x=[1, 2, 3], y=slice(3))
        with self.assertRaisesRegexp(ValueError,
                                     'Indexers must be 1 dimensional'):
            da.isel_points(y=1, x=2)
        with self.assertRaisesRegexp(ValueError,
                                     'Existing dimension names are not'):
            da.isel_points(y=[1, 2], x=[1, 2], dim='x')

        # using non string dims
        actual = da.isel_points(y=[1, 2], x=[1, 2], dim=['A', 'B'])
        assert 'points' in actual.coords

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

    def test_multiindex(self):
        mindex = pd.MultiIndex.from_product([['a', 'b'], [1, 2], [-1, -2]],
                                            names=('one', 'two', 'three'))
        mdata = DataArray(range(8), [('x', mindex)])

        def test_sel(lab_indexer, pos_indexer, replaced_idx=False,
                     renamed_dim=None):
            da = mdata.sel(x=lab_indexer)
            expected_da = mdata.isel(x=pos_indexer)
            if not replaced_idx:
                self.assertDataArrayIdentical(da, expected_da)
            else:
                if renamed_dim:
                    self.assertEqual(da.dims[0], renamed_dim)
                    da = da.rename({renamed_dim: 'x'})
                self.assertVariableIdentical(da, expected_da)
                self.assertVariableNotEqual(da['x'], expected_da['x'])

        test_sel(('a', 1, -1), 0)
        test_sel(('b', 2, -2), -1)
        test_sel(('a', 1), [0, 1], replaced_idx=True, renamed_dim='three')
        test_sel(('a',), range(4), replaced_idx=True)
        test_sel('a', range(4), replaced_idx=True)
        test_sel([('a', 1, -1), ('b', 2, -2)], [0, 7])
        test_sel(slice('a', 'b'), range(8))
        test_sel(slice(('a', 1), ('b', 1)), range(6))
        test_sel({'one': 'a', 'two': 1, 'three': -1}, 0)
        test_sel({'one': 'a', 'two': 1}, [0, 1], replaced_idx=True,
                 renamed_dim='three')
        test_sel({'one': 'a'}, range(4), replaced_idx=True)

        self.assertDataArrayIdentical(mdata.loc['a'], mdata.sel(x='a'))
        self.assertDataArrayIdentical(mdata.loc[('a', 1), ...],
                                      mdata.sel(x=('a', 1)))
        self.assertDataArrayIdentical(mdata.loc[{'one': 'a'}, ...],
                                      mdata.sel(x={'one': 'a'}))
        with self.assertRaises(KeyError):
            mdata.loc[{'one': 'a'}]
        with self.assertRaises(IndexError):
            mdata.loc[('a', 1)]

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

        with self.assertRaisesRegexp(ValueError, 'cannot delete'):
            del da['x']

        with self.assertRaisesRegexp(ValueError, 'cannot delete'):
            del da.coords['x']

    def test_coord_coords(self):
        orig = DataArray([10, 20],
                         {'x': [1, 2], 'x2': ('x', ['a', 'b']), 'z': 4},
                         dims='x')

        actual = orig.coords['x']
        expected = DataArray([1, 2], {'z': 4, 'x2': ('x', ['a', 'b']),
                                      'x': [1, 2]},
                             dims='x', name='x')
        self.assertDataArrayIdentical(expected, actual)

        del actual.coords['x2']
        self.assertDataArrayIdentical(
            expected.reset_coords('x2', drop=True), actual)

        actual.coords['x3'] = ('x', ['a', 'b'])
        expected = DataArray([1, 2], {'z': 4, 'x3': ('x', ['a', 'b']),
                                      'x': [1, 2]},
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
        with self.assertRaisesRegexp(ValueError, 'cannot be found'):
            data.reset_coords('foo', drop=True)
        with self.assertRaisesRegexp(ValueError, 'cannot be found'):
            data.reset_coords('not_found')
        with self.assertRaisesRegexp(ValueError, 'cannot remove index'):
            data.reset_coords('y')

    def test_assign_coords(self):
        array = DataArray(10)
        actual = array.assign_coords(c=42)
        expected = DataArray(10, {'c': 42})
        self.assertDataArrayIdentical(actual, expected)

        array = DataArray([1, 2, 3, 4], {'c': ('x', [0, 0, 1, 1])}, dims='x')
        actual = array.groupby('c').assign_coords(d=lambda a: a.mean())
        expected = array.copy()
        expected.coords['d'] = ('x', [1.5, 1.5, 3.5, 3.5])
        self.assertDataArrayIdentical(actual, expected)

    def test_coords_alignment(self):
        lhs = DataArray([1, 2, 3], [('x', [0, 1, 2])])
        rhs = DataArray([2, 3, 4], [('x', [1, 2, 3])])
        lhs.coords['rhs'] = rhs

        expected = DataArray([1, 2, 3], coords={'rhs': ('x', [np.nan, 2, 3])},
                             dims='x')
        self.assertDataArrayIdentical(lhs, expected)

    def test_coords_replacement_alignment(self):
        # regression test for GH725
        arr = DataArray([0, 1, 2], dims=['abc'])
        new_coord = DataArray([1, 2, 3], dims=['abc'], coords=[[1, 2, 3]])
        arr['abc'] = new_coord
        expected = DataArray([0, 1, 2], coords=[('abc', [1, 2, 3])])
        self.assertDataArrayIdentical(arr, expected)

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

        # regression test for #736, reindex can not change complex nums dtype
        x = np.array([1, 2, 3], dtype=np.complex)
        x = DataArray(x, coords=[[0.1, 0.2, 0.3]])
        y = DataArray([2, 5, 6, 7, 8], coords=[[-1.1, 0.21, 0.31, 0.41, 0.51]])
        re_dtype = x.reindex_like(y, method='pad').dtype
        self.assertEqual(x.dtype, re_dtype)

    def test_reindex_method(self):
        x = DataArray([10, 20], dims='y')
        y = [-0.1, 0.5, 1.1]
        if pd.__version__ >= '0.17':
            actual = x.reindex(y=y, method='backfill', tolerance=0.2)
            expected = DataArray([10, np.nan, np.nan], coords=[('y', y)])
            self.assertDataArrayIdentical(expected, actual)

        alt = Dataset({'y': y})
        actual = x.reindex_like(alt, method='backfill')
        expected = DataArray([10, 20, np.nan], coords=[('y', y)])
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

    def test_non_overlapping_dataarrays_return_empty_result(self):

        a = DataArray(range(5), [('x', range(5))])
        b = DataArray(range(5), [('x', range(1, 6))])
        result = a.isel(x=slice(2)) + a.isel(x=slice(2, None))
        self.assertEqual(len(result['x']), 0)

    def test_empty_dataarrays_return_empty_result(self):

        a = DataArray(data=[])
        result = a * a
        self.assertEqual(len(result['dim_0']), 0)


    def test_inplace_math_basics(self):
        x = self.x
        a = self.dv
        v = a.variable
        b = a
        b += 1
        self.assertIs(b, a)
        self.assertIs(b.variable, v)
        self.assertArrayEqual(b.values, x)
        self.assertIs(source_ndarray(b.values), x)

    def test_inplace_math_automatic_alignment(self):
        a = DataArray(range(5), [('x', range(5))])
        b = DataArray(range(1, 6), [('x', range(1, 6))])
        with self.assertRaises(xr.MergeError):
            a += b
        with self.assertRaises(xr.MergeError):
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
        self.assertIs((+a['x']).name, 'x')
        self.assertIs((a['x'] + 0).name, 'x')
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
        expected = DataArray(1 + np.arange(3), coords=[('x', range(3))],
                             name='x')
        self.assertDataArrayIdentical(expected, actual)

        # regression tests for #254
        actual = orig[0] < orig
        expected = DataArray([False, True, True], coords=[('x', range(3))],
                             name='x')
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

    def test_stack_unstack(self):
        orig = DataArray([[0, 1], [2, 3]], dims=['x', 'y'], attrs={'foo': 2})
        actual = orig.stack(z=['x', 'y']).unstack('z')
        self.assertDataArrayIdentical(orig, actual)

    def test_unstack_pandas_consistency(self):
        df = pd.DataFrame({'foo': range(3),
                           'x': ['a', 'b', 'b'],
                           'y': [0, 0, 1]})
        s = df.set_index(['x', 'y'])['foo']
        expected = DataArray(s.unstack(), name='foo')
        actual = DataArray(s, dims='z').unstack('z')
        self.assertDataArrayIdentical(expected, actual)

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

        with self.assertRaisesRegexp(ValueError, 'cannot be found'):
            arr.drop(None)

        renamed = arr.rename('foo')
        with self.assertRaisesRegexp(ValueError, 'cannot be found'):
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

        def identity(x):
            return x

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

            ds = array.coords['x'].to_dataset('X')
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
        expected = array  # should be a no-op
        self.assertDataArrayIdentical(expected, actual)

    def make_groupby_multidim_example_array(self):
        return DataArray([[[0,1],[2,3]],[[5,10],[15,20]]],
                        coords={'lon': (['ny', 'nx'], [[30., 40.], [40., 50.]] ),
                                'lat': (['ny', 'nx'], [[10., 10.], [20., 20.]] ),},
                        dims=['time', 'ny', 'nx'])

    def test_groupby_multidim(self):
        array = self.make_groupby_multidim_example_array()
        for dim, expected_sum in [
                ('lon', DataArray([5, 28, 23], coords={'lon': [30., 40., 50.]})),
                ('lat', DataArray([16, 40], coords={'lat': [10., 20.]}))]:
            actual_sum = array.groupby(dim).sum()
            self.assertDataArrayIdentical(expected_sum, actual_sum)

    def test_groupby_multidim_apply(self):
        array = self.make_groupby_multidim_example_array()
        actual = array.groupby('lon').apply(
                lambda x : x - x.mean(), shortcut=False)
        expected = DataArray([[[-2.5, -6.], [-5., -8.5]],
                              [[ 2.5,  3.], [ 8.,  8.5]]],
                    coords=array.coords, dims=array.dims)
        self.assertDataArrayIdentical(expected, actual)

    def test_groupby_bins(self):
        array = DataArray(np.arange(4), dims='dim_0')
        # the first value should not be part of any group ("right" binning)
        array[0] = 99
        # bins follow conventions for pandas.cut
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
        bins = [0,1.5,5]
        bin_coords = ['(0, 1.5]', '(1.5, 5]']
        expected = DataArray([1,5], dims='dim_0_bins',
                        coords={'dim_0_bins': bin_coords})
        # the problem with this is that it overwrites the dimensions of array!
        #actual = array.groupby('dim_0', bins=bins).sum()
        actual = array.groupby_bins('dim_0', bins).apply(
                                    lambda x : x.sum(), shortcut=False)
        self.assertDataArrayIdentical(expected, actual)
        # make sure original array dims are unchanged
        # (would fail with shortcut=True above)
        self.assertEqual(len(array.dim_0), 4)

    def test_groupby_bins_multidim(self):
        array = self.make_groupby_multidim_example_array()
        bins = [0,15,20]
        bin_coords = ['(0, 15]', '(15, 20]']
        expected = DataArray([16, 40], dims='lat_bins',
                                coords={'lat_bins': bin_coords})
        actual = array.groupby_bins('lat', bins).apply(
                                    lambda x : x.sum(), shortcut=False)
        self.assertDataArrayIdentical(expected, actual)
        # modify the array coordinates to be non-monotonic after unstacking
        array['lat'].data = np.array([[10., 20.], [20., 10.]])
        expected = DataArray([28, 28], dims='lat_bins',
                    coords={'lat_bins': bin_coords})
        actual = array.groupby_bins('lat', bins).apply(
                                    lambda x : x.sum(), shortcut=False)
        self.assertDataArrayIdentical(expected, actual)

    def test_groupby_bins_sort(self):
        data = xr.DataArray(
            np.arange(100), dims='x',
            coords={'x': np.linspace(-100, 100, num=100)})
        binned_mean = data.groupby_bins('x', bins=11).mean()
        assert binned_mean.to_index().is_monotonic

    def make_rolling_example_array(self):
        times = pd.date_range('2000-01-01', freq='1D', periods=21)
        values = np.random.random((21, 4))
        da = DataArray(values, dims=('time', 'x'))
        da['time'] = times

        return da

    def test_rolling_iter(self):
        da = self.make_rolling_example_array()

        rolling_obj = da.rolling(time=7)

        self.assertEqual(len(rolling_obj.window_labels), len(da['time']))
        self.assertDataArrayIdentical(rolling_obj.window_labels, da['time'])

        for i, (label, window_da) in enumerate(rolling_obj):
            self.assertEqual(label, da['time'].isel(time=i))

    def test_rolling_properties(self):
        da = self.make_rolling_example_array()
        rolling_obj = da.rolling(time=4)

        self.assertEqual(rolling_obj._axis_num, 0)

        # catching invalid args
        with self.assertRaisesRegexp(ValueError, 'exactly one dim/window should'):
            da.rolling(time=7, x=2)
        with self.assertRaisesRegexp(ValueError, 'window must be > 0'):
            da.rolling(time=-2)
        with self.assertRaisesRegexp(ValueError, 'min_periods must be greater'):
            da.rolling(time=2, min_periods=0)

    @requires_bottleneck
    def test_rolling_wrapped_bottleneck(self):
        import bottleneck as bn

        da = self.make_rolling_example_array()

        # Test all bottleneck functions
        rolling_obj = da.rolling(time=7)
        for name in ('sum', 'mean', 'std', 'min', 'max', 'median'):
            func_name = 'move_{0}'.format(name)
            actual = getattr(rolling_obj, name)()
            expected = getattr(bn, func_name)(da.values, window=7, axis=0)
            self.assertArrayEqual(actual.values, expected)

        # Using min_periods
        rolling_obj = da.rolling(time=7, min_periods=1)
        for name in ('sum', 'mean', 'std', 'min', 'max'):
            func_name = 'move_{0}'.format(name)
            actual = getattr(rolling_obj, name)()
            expected = getattr(bn, func_name)(da.values, window=7, axis=0,
                                              min_count=1)
            self.assertArrayEqual(actual.values, expected)

        # Using center=False
        rolling_obj = da.rolling(time=7, center=False)
        for name in ('sum', 'mean', 'std', 'min', 'max', 'median'):
            actual = getattr(rolling_obj, name)()['time']
            self.assertDataArrayEqual(actual, da['time'])

        # Using center=True
        rolling_obj = da.rolling(time=7, center=True)
        for name in ('sum', 'mean', 'std', 'min', 'max', 'median'):
            actual = getattr(rolling_obj, name)()['time']
            self.assertDataArrayEqual(actual, da['time'])

        # catching invalid args
        with self.assertRaisesRegexp(ValueError, 'Rolling.median does not'):
            da.rolling(time=7, min_periods=1).median()

    def test_rolling_pandas_compat(self):
        s = pd.Series(range(10))
        da = DataArray.from_series(s)

        for center in (False, True):
            for window in [1, 2, 3, 4]:
                for min_periods in [None, 1, 2, 3]:
                    if min_periods is not None and window < min_periods:
                        min_periods = window
                    s_rolling = pd.rolling_mean(s, window, center=center,
                                                min_periods=min_periods)
                    da_rolling = da.rolling(index=window, center=center,
                                            min_periods=min_periods).mean()
                    # pandas does some fancy stuff in the last position,
                    # we're not going to do that yet!
                    np.testing.assert_allclose(s_rolling.values[:-1],
                                               da_rolling.values[:-1])
                    np.testing.assert_allclose(s_rolling.index,
                                               da_rolling['index'])

    def test_rolling_reduce(self):
        da = self.make_rolling_example_array()
        for da in [self.make_rolling_example_array(),
                   DataArray([0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7],
                             dims='time')]:
            for center in (False, True):
                for window in [1, 2, 3, 4]:
                    for min_periods in [None, 1, 2, 3]:
                        if min_periods is not None and window < min_periods:
                            min_periods = window
                        # we can use this rolling object for all methods below
                        rolling_obj = da.rolling(time=window, center=center,
                                                 min_periods=min_periods)
                        for name in ['sum', 'mean', 'min', 'max']:
                            # add nan prefix to numpy methods to get similar
                            # behavior as bottleneck
                            actual = rolling_obj.reduce(
                                getattr(np, 'nan%s' % name))
                            expected = getattr(rolling_obj, name)()
                            self.assertDataArrayAllClose(actual, expected)

    def test_resample(self):
        times = pd.date_range('2000-01-01', freq='6H', periods=10)
        array = DataArray(np.arange(10), [('time', times)])

        actual = array.resample('6H', dim='time')
        self.assertDataArrayIdentical(array, actual)

        actual = array.resample('24H', dim='time')
        expected = DataArray(array.to_series().resample('24H', how='mean'))
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

        # regression test for http://stackoverflow.com/questions/33158558/
        array = Dataset({'time': times})['time']
        actual = array.resample('1D', dim='time', how='last')
        expected_times = pd.to_datetime(['2000-01-01T18', '2000-01-02T18',
                                         '2000-01-03T06'])
        expected = DataArray(expected_times, [('time', times[::4])],
                             name='time')
        self.assertDataArrayIdentical(expected, actual)

    def test_resample_first_keep_attrs(self):
        times = pd.date_range('2000-01-01', freq='6H', periods=10)
        array = DataArray(np.arange(10), [('time', times)])
        array.attrs['meta'] = 'data'

        resampled_array = array.resample('1D', dim='time', how='first', keep_attrs=True)
        actual = resampled_array.attrs
        expected = array.attrs
        self.assertEqual(expected, actual)

        resampled_array = array.resample('1D', dim='time', how='first', keep_attrs=False)
        assert resampled_array.attrs == {}

    def test_resample_mean_keep_attrs(self):
        times = pd.date_range('2000-01-01', freq='6H', periods=10)
        array = DataArray(np.arange(10), [('time', times)])
        array.attrs['meta'] = 'data'

        resampled_array = array.resample('1D', dim='time', how='mean', keep_attrs=True)
        actual = resampled_array.attrs
        expected = array.attrs
        self.assertEqual(expected, actual)

        resampled_array = array.resample('1D', dim='time', how='mean', keep_attrs=False)
        assert resampled_array.attrs == {}

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

    def test_align(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        dv1, dv2 = align(self.dv, self.dv[:5], join='inner')
        self.assertDataArrayIdentical(dv1, self.dv[:5])
        self.assertDataArrayIdentical(dv2, self.dv[:5])

    def test_align_dtype(self):
        # regression test for #264
        x1 = np.arange(30)
        x2 = np.arange(5, 35)
        a = DataArray(np.random.random((30,)).astype(np.float32), {'x': x1})
        b = DataArray(np.random.random((30,)).astype(np.float32), {'x': x2})
        c, d = align(a, b, join='outer')
        self.assertEqual(c.dtype, np.float32)

    def test_align_copy(self):
        x = DataArray([1, 2, 3], coords=[('a', [1, 2, 3])])
        y = DataArray([1, 2], coords=[('a', [3, 1])])
        
        expected_x2 = x
        expected_y2 = DataArray([2, np.nan, 1], coords=[('a', [1, 2, 3])])

        x2, y2 = align(x, y, join='outer', copy=False)
        self.assertDataArrayIdentical(expected_x2, x2)
        self.assertDataArrayIdentical(expected_y2, y2)
        assert source_ndarray(x2.data) is source_ndarray(x.data)
    
        x2, y2 = align(x, y, join='outer', copy=True)
        self.assertDataArrayIdentical(expected_x2, x2)
        self.assertDataArrayIdentical(expected_y2, y2)
        assert source_ndarray(x2.data) is not source_ndarray(x.data)

        # Trivial align - 1 element
        x = DataArray([1, 2, 3], coords=[('a', [1, 2, 3])])
        x2, = align(x, copy=False)
        self.assertDataArrayIdentical(x, x2)
        assert source_ndarray(x2.data) is source_ndarray(x.data)
    
        x2, = align(x, copy=True)
        self.assertDataArrayIdentical(x, x2)
        assert source_ndarray(x2.data) is not source_ndarray(x.data)

    def test_align_exclude(self):
        x = DataArray([[1, 2], [3, 4]], coords=[('a', [-1, -2]), ('b', [3, 4])])
        y = DataArray([[1, 2], [3, 4]], coords=[('a', [-1, 20]), ('b', [5, 6])])
        z = DataArray([1], dims=['a'], coords={'a': [20], 'b': 7})
        
        x2, y2, z2 = align(x, y, z, join='outer', exclude=['b'])
        expected_x2 = DataArray([[3, 4], [1, 2], [np.nan, np.nan]], coords=[('a', [-2, -1, 20]), ('b', [3, 4])])
        expected_y2 = DataArray([[np.nan, np.nan], [1, 2], [3, 4]], coords=[('a', [-2, -1, 20]), ('b', [5, 6])])
        expected_z2 = DataArray([np.nan, np.nan, 1], dims=['a'], coords={'a': [-2, -1, 20], 'b': 7})
        self.assertDataArrayIdentical(expected_x2, x2)
        self.assertDataArrayIdentical(expected_y2, y2)
        self.assertDataArrayIdentical(expected_z2, z2)        

    def test_align_indexes(self):
        x = DataArray([1, 2, 3], coords=[('a', [-1, 10, -2])])
        y = DataArray([1, 2], coords=[('a', [-2, -1])])

        x2, y2 = align(x, y, join='outer', indexes={'a': [10, -1, -2]})
        expected_x2 = DataArray([2, 1, 3], coords=[('a', [10, -1, -2])])
        expected_y2 = DataArray([np.nan, 2, 1], coords=[('a', [10, -1, -2])])
        self.assertDataArrayIdentical(expected_x2, x2)
        self.assertDataArrayIdentical(expected_y2, y2)

        x2, = align(x, join='outer', indexes={'a': [-2, 7, 10, -1]})
        expected_x2 = DataArray([3, np.nan, 2, 1], coords=[('a', [-2, 7, 10, -1])])
        self.assertDataArrayIdentical(expected_x2, x2)

    def test_broadcast_arrays(self):
        x = DataArray([1, 2], coords=[('a', [-1, -2])], name='x')
        y = DataArray([1, 2], coords=[('b', [3, 4])], name='y')
        x2, y2 = broadcast(x, y)
        expected_coords = [('a', [-1, -2]), ('b', [3, 4])]
        expected_x2 = DataArray([[1, 1], [2, 2]], expected_coords, name='x')
        expected_y2 = DataArray([[1, 2], [1, 2]], expected_coords, name='y')
        self.assertDataArrayIdentical(expected_x2, x2)
        self.assertDataArrayIdentical(expected_y2, y2)

        x = DataArray(np.random.randn(2, 3), dims=['a', 'b'])
        y = DataArray(np.random.randn(3, 2), dims=['b', 'a'])
        x2, y2 = broadcast(x, y)
        expected_x2 = x
        expected_y2 = y.T
        self.assertDataArrayIdentical(expected_x2, x2)
        self.assertDataArrayIdentical(expected_y2, y2)

    def test_broadcast_arrays_misaligned(self):
        # broadcast on misaligned coords must auto-align
        x = DataArray([[1, 2], [3, 4]], coords=[('a', [-1, -2]), ('b', [3, 4])])
        y = DataArray([1, 2], coords=[('a', [-1, 20])])
        expected_x2 = DataArray([[3, 4], [1, 2], [np.nan, np.nan]], coords=[('a', [-2, -1, 20]), ('b', [3, 4])])
        expected_y2 = DataArray([[np.nan, np.nan], [1, 1], [2, 2]], coords=[('a', [-2, -1, 20]), ('b', [3, 4])])
        x2, y2 = broadcast(x, y)
        self.assertDataArrayIdentical(expected_x2, x2)
        self.assertDataArrayIdentical(expected_y2, y2)

    def test_broadcast_arrays_nocopy(self):
        # Test that input data is not copied over in case no alteration is needed
        x = DataArray([1, 2], coords=[('a', [-1, -2])], name='x')
        y = DataArray(3, name='y')
        expected_x2 = DataArray([1, 2], coords=[('a', [-1, -2])], name='x')
        expected_y2 = DataArray([3, 3], coords=[('a', [-1, -2])], name='y')

        x2, y2 = broadcast(x, y)
        self.assertDataArrayIdentical(expected_x2, x2)
        self.assertDataArrayIdentical(expected_y2, y2)
        assert source_ndarray(x2.data) is source_ndarray(x.data)
        
        # single-element broadcast (trivial case)
        x2, = broadcast(x)
        self.assertDataArrayIdentical(x, x2)
        assert source_ndarray(x2.data) is source_ndarray(x.data)

    def test_broadcast_arrays_exclude(self):
        x = DataArray([[1, 2], [3, 4]], coords=[('a', [-1, -2]), ('b', [3, 4])])
        y = DataArray([1, 2], coords=[('a', [-1, 20])])
        z = DataArray(5, coords={'b': 5})
        
        x2, y2, z2 = broadcast(x, y, z, exclude=['b'])
        expected_x2 = DataArray([[3, 4], [1, 2], [np.nan, np.nan]], coords=[('a', [-2, -1, 20]), ('b', [3, 4])])
        expected_y2 = DataArray([np.nan, 1, 2], coords=[('a', [-2, -1, 20])])
        expected_z2 = DataArray([5, 5, 5], dims=['a'], coords={'a': [-2, -1, 20], 'b': 5})
        self.assertDataArrayIdentical(expected_x2, x2)
        self.assertDataArrayIdentical(expected_y2, y2)
        self.assertDataArrayIdentical(expected_z2, z2)

    def test_broadcast_coordinates(self):
        # regression test for GH649
        ds = Dataset({'a': (['x', 'y'], np.ones((5, 6)))})
        x_bc, y_bc, a_bc = broadcast(ds.x, ds.y, ds.a)
        self.assertDataArrayIdentical(ds.a, a_bc)

        X, Y = np.meshgrid(np.arange(5), np.arange(6), indexing='ij')
        exp_x = DataArray(X, dims=['x', 'y'], name='x')
        exp_y = DataArray(Y, dims=['x', 'y'], name='y')
        self.assertDataArrayIdentical(exp_x, x_bc)
        self.assertDataArrayIdentical(exp_y, y_bc)

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
                        [('B', [1, 2, 3]), ('A', list('cdef'))], name='foo')
        expected = arr.to_series()
        actual = arr.to_dataframe()['foo']
        self.assertArrayEqual(expected.values, actual.values)
        self.assertArrayEqual(expected.name, actual.name)
        self.assertArrayEqual(expected.index.values, actual.index.values)

        # regression test for coords with different dimensions
        arr.coords['C'] = ('B', [-1, -2, -3])
        expected = arr.to_series().to_frame()
        expected['C'] = [-1] * 4 + [-2] * 4 + [-3] * 4
        expected = expected[['C', 'foo']]
        actual = arr.to_dataframe()
        self.assertArrayEqual(expected.values, actual.values)
        self.assertArrayEqual(expected.columns.values, actual.columns.values)
        self.assertArrayEqual(expected.index.values, actual.index.values)

        arr.name = None  # unnamed
        with self.assertRaisesRegexp(ValueError, 'unnamed'):
            arr.to_dataframe()

    def test_to_pandas_name_matches_coordinate(self):
        # coordinate with same name as array
        arr = DataArray([1, 2, 3], dims='x', name='x')
        series = arr.to_series()
        self.assertArrayEqual([1, 2, 3], series.values)
        self.assertArrayEqual([0, 1, 2], series.index.values)
        self.assertEqual('x', series.name)
        self.assertEqual('x', series.index.name)

        frame = arr.to_dataframe()
        expected = series.to_frame()
        self.assertTrue(expected.equals(frame))

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

    def test_series_categorical_index(self):
        # regression test for GH700
        if not hasattr(pd, 'CategoricalIndex'):
            raise unittest.SkipTest('requires pandas with CategoricalIndex')

        s = pd.Series(range(5), index=pd.CategoricalIndex(list('aabbc')))
        arr = DataArray(s)
        assert "'a'" in repr(arr)  # should not error

    def test_to_and_from_dict(self):
        expected = {'name': 'foo',
                    'dims': ('x', 'y'),
                    'data': self.x.tolist(),
                    'attrs': {},
                    'coords': {'y': {'dims': ('y',),
                                     'data': list(range(20)),
                                     'attrs': {}},
                               'x': {'dims': ('x',),
                                     'data': list(range(10)),
                                     'attrs': {}}}}
        actual = self.dv.to_dict()

        # check that they are identical
        self.assertEqual(expected, actual)

        # check roundtrip
        self.assertDataArrayIdentical(self.dv, DataArray.from_dict(actual))

        # a more bare bones representation still roundtrips
        d = {'name': 'foo',
             'dims': ('x', 'y'),
             'data': self.x,
             'coords': {'y': {'dims': 'y', 'data': list(range(20))},
                        'x': {'dims': 'x', 'data': list(range(10))}}}
        self.assertDataArrayIdentical(self.dv, DataArray.from_dict(d))

        # and the most bare bones representation still roundtrips
        d = {'name': 'foo', 'dims': ('x', 'y'), 'data': self.x}
        self.assertDataArrayIdentical(self.dv, DataArray.from_dict(d))

        # missing a dims in the coords
        d = {'dims': ('x', 'y'),
             'data': self.x,
             'coords': {'y': {'data': list(range(20))},
                        'x': {'dims': 'x', 'data': list(range(10))}}}
        with self.assertRaisesRegexp(ValueError, "cannot convert dict when coords are missing the key 'dims'"):
            DataArray.from_dict(d)

        # this one is missing some necessary information
        d = {'dims': ('t')}
        with self.assertRaisesRegexp(ValueError, "cannot convert dict without the key 'data'"):
            DataArray.from_dict(d)

    def test_to_and_from_dict_with_time_dim(self):
        x = np.random.randn(10, 3)
        t = pd.date_range('20130101', periods=10)
        lat = [77.7, 83.2, 76]
        da = DataArray(x, OrderedDict([('t', ('t', t)),
                                       ('lat', ('lat', lat))]))
        roundtripped = DataArray.from_dict(da.to_dict())
        self.assertDataArrayIdentical(da, roundtripped)

    def test_to_and_from_dict_with_nan_nat(self):
        y = np.random.randn(10, 3)
        y[2] = np.nan
        t = pd.Series(pd.date_range('20130101', periods=10))
        t[2] = np.nan
        lat = [77.7, 83.2, 76]
        da = DataArray(y, OrderedDict([('t', ('t', t)),
                                       ('lat', ('lat', lat))]))
        roundtripped = DataArray.from_dict(da.to_dict())
        self.assertDataArrayIdentical(da, roundtripped)

    def test_to_masked_array(self):
        rs = np.random.RandomState(44)
        x = rs.random_sample(size=(10, 20))
        x_masked = np.ma.masked_where(x < 0.5, x)
        da = DataArray(x_masked)

        # Test round trip
        x_masked_2 = da.to_masked_array()
        da_2 = DataArray(x_masked_2)
        self.assertArrayEqual(x_masked, x_masked_2)
        self.assertDataArrayEqual(da, da_2)

        da_masked_array = da.to_masked_array(copy=True)
        self.assertTrue(isinstance(da_masked_array, np.ma.MaskedArray))
        # Test masks
        self.assertArrayEqual(da_masked_array.mask, x_masked.mask)
        # Test that mask is unpacked correctly
        self.assertArrayEqual(da.values, x_masked.filled(np.nan))
        # Test that the underlying data (including nans) hasn't changed
        self.assertArrayEqual(da_masked_array, x_masked.filled(np.nan))

        # Test that copy=False gives access to values
        masked_array = da.to_masked_array(copy=False)
        masked_array[0, 0] = 10.
        self.assertEqual(masked_array[0, 0], 10.)
        self.assertEqual(da[0, 0].values, 10.)
        self.assertTrue(masked_array.base is da.values)
        self.assertIsInstance(masked_array, np.ma.MaskedArray)

        # Test with some odd arrays
        for v in [4, np.nan, True, '4', 'four']:
            da = DataArray(v)
            ma = da.to_masked_array()
            self.assertIsInstance(ma, np.ma.MaskedArray)

        # Fix GH issue 684 - masked arrays mask should be an array not a scalar
        N = 4
        v = range(N)
        da = DataArray(v)
        ma = da.to_masked_array()
        self.assertEqual(len(ma.mask), N)

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
        with self.assertRaisesRegexp(ValueError, 'unable to convert unnamed'):
            unnamed.to_dataset()

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

        roundtripped = actual.to_array(dim='x')
        self.assertDataArrayIdentical(array, roundtripped)

        array = DataArray([1, 2, 3], dims='x')
        expected = Dataset(OrderedDict([(0, 1), (1, 2), (2, 3)]))
        actual = array.to_dataset('x')
        self.assertDatasetIdentical(expected, actual)

    def test_to_dataset_retains_keys(self):

        # use dates as convenient non-str objects. Not a specific date test
        import datetime
        dates = [datetime.date(2000,1,d) for d in range(1,4)]

        array = DataArray([1, 2, 3], coords=[('x', dates)],
                          attrs={'a': 1})

        # convert to dateset and back again
        result = array.to_dataset('x').to_array(dim='x')

        self.assertDatasetEqual(array, result)

    def test__title_for_slice(self):
        array = DataArray(np.ones((4, 3, 2)), dims=['a', 'b', 'c'])
        self.assertEqual('', array._title_for_slice())
        self.assertEqual('c = 0', array.isel(c=0)._title_for_slice())
        title = array.isel(b=1, c=0)._title_for_slice()
        self.assertTrue('b = 1, c = 0' == title or 'c = 0, b = 1' == title)

        a2 = DataArray(np.ones((4, 1)), dims=['a', 'b'])
        self.assertEqual('b = [0]', a2._title_for_slice())

    def test__title_for_slice_truncate(self):
        array = DataArray(np.ones((4)))
        array.coords['a'] = 'a' * 100
        array.coords['b'] = 'b' * 100

        nchar = 80
        title = array._title_for_slice(truncate=nchar)

        self.assertEqual(nchar, len(title))
        self.assertTrue(title.endswith('...'))

    def test_dataarray_diff_n1(self):
        da = self.ds['foo']
        actual = da.diff('y')
        expected = DataArray(np.diff(da.values, axis=1),
                             [da['x'].values, da['y'].values[1:]],
                             ['x', 'y'])
        self.assertDataArrayEqual(expected, actual)

    def test_coordinate_diff(self):
        # regression test for GH634
        arr = DataArray(range(0, 20, 2), dims=['lon'], coords=[range(10)])
        lon = arr.coords['lon']
        expected = DataArray([1] * 9, dims=['lon'], coords=[range(1, 10)],
                             name='lon')
        actual = lon.diff('lon')

    def test_shift(self):
        arr = DataArray([1, 2, 3], dims='x')
        actual = arr.shift(x=1)
        expected = DataArray([np.nan, 1, 2], dims='x')
        self.assertDataArrayIdentical(expected, actual)

        for offset in [-5, -2, -1, 0, 1, 2, 5]:
            expected = DataArray(arr.to_pandas().shift(offset))
            actual = arr.shift(x=offset)
            self.assertDataArrayIdentical(expected, actual)

    def test_roll(self):
        arr = DataArray([1, 2, 3], dims='x')
        actual = arr.roll(x=1)
        expected = DataArray([3, 1, 2], coords=[('x', [2, 0, 1])])
        self.assertDataArrayIdentical(expected, actual)

    def test_real_and_imag(self):
        array = DataArray(1 + 2j)
        self.assertDataArrayIdentical(array.real, DataArray(1))
        self.assertDataArrayIdentical(array.imag, DataArray(2))

    def test_setattr_raises(self):
        array = DataArray(0, coords={'scalar': 1}, attrs={'foo': 'bar'})
        with self.assertRaisesRegexp(AttributeError, 'cannot set attr'):
            array.scalar = 2
        with self.assertRaisesRegexp(AttributeError, 'cannot set attr'):
            array.foo = 2
        with self.assertRaisesRegexp(AttributeError, 'cannot set attr'):
            array.other = 2

    def test_full_like(self):
        da = DataArray(np.random.random(size=(4, 4)), dims=('x', 'y'),
                       attrs={'attr1': 'value1'})
        actual = _full_like(da)

        self.assertEqual(actual.dtype, da.dtype)
        self.assertEqual(actual.shape, da.shape)
        self.assertEqual(actual.dims, da.dims)
        self.assertEqual(actual.attrs, {})

        for name in da.coords:
            self.assertArrayEqual(da[name], actual[name])
            self.assertEqual(da[name].dtype, actual[name].dtype)

        # keep attrs
        actual = _full_like(da, keep_attrs=True)
        self.assertEqual(actual.attrs, da.attrs)

        # Fill value
        actual = _full_like(da, fill_value=True)
        self.assertEqual(actual.dtype, da.dtype)
        self.assertEqual(actual.shape, da.shape)
        self.assertEqual(actual.dims, da.dims)
        np.testing.assert_equal(actual.values, np.nan)

        actual = _full_like(da, fill_value=10)
        self.assertEqual(actual.dtype, da.dtype)
        np.testing.assert_equal(actual.values, 10)

        # make sure filling with nans promotes integer type
        actual = _full_like(DataArray([1, 2, 3]), fill_value=np.nan)
        self.assertEqual(actual.dtype, np.float)
        np.testing.assert_equal(actual.values, np.nan)

    def test_dot(self):
        x = np.linspace(-3, 3, 6)
        y = np.linspace(-3, 3, 5)
        z = range(4)
        da_vals = np.arange(6 * 5 * 4).reshape((6, 5, 4))
        da = DataArray(da_vals, coords=[x, y, z], dims=['x', 'y', 'z'])

        dm_vals = range(4)
        dm = DataArray(dm_vals, coords=[z], dims=['z'])

        # nd dot 1d
        actual = da.dot(dm)
        expected_vals = np.tensordot(da_vals, dm_vals, [2, 0])
        expected = DataArray(expected_vals, coords=[x, y], dims=['x', 'y'])
        self.assertDataArrayEqual(expected, actual)

        # all shared dims
        actual = da.dot(da)
        expected_vals = np.tensordot(da_vals, da_vals, axes=([0, 1, 2], [0, 1, 2]))
        expected = DataArray(expected_vals)
        self.assertDataArrayEqual(expected, actual)

        # multiple shared dims
        dm_vals = np.arange(20 * 5 * 4).reshape((20, 5, 4))
        j = np.linspace(-3, 3, 20)
        dm = DataArray(dm_vals, coords=[j, y, z], dims=['j', 'y', 'z'])
        actual = da.dot(dm)
        expected_vals = np.tensordot(da_vals, dm_vals, axes=([1, 2], [1, 2]))
        expected = DataArray(expected_vals, coords=[x, j], dims=['x', 'j'])
        self.assertDataArrayEqual(expected, actual)

        with self.assertRaises(NotImplementedError):
            da.dot(dm.to_dataset(name='dm'))
        with self.assertRaises(TypeError):
            da.dot(dm.values)
        with self.assertRaisesRegexp(ValueError, 'no shared dimensions'):
            da.dot(DataArray(1))
