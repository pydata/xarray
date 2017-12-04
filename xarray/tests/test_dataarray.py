from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import pickle
import pytest
from copy import deepcopy
from textwrap import dedent
from distutils.version import LooseVersion

import xarray as xr

from xarray import (align, broadcast, Dataset, DataArray,
                    IndexVariable, Variable)
from xarray.core.pycompat import iteritems, OrderedDict
from xarray.core.common import full_like

from xarray.tests import (
    TestCase, ReturnItem, source_ndarray, unittest, requires_dask,
    assert_identical, assert_equal, assert_allclose, assert_array_equal,
    raises_regex, requires_scipy)


class TestDataArray(TestCase):
    def setUp(self):
        self.attrs = {'attr1': 'value1', 'attr2': 2929}
        self.x = np.random.random((10, 20))
        self.v = Variable(['x', 'y'], self.x)
        self.va = Variable(['x', 'y'], self.x, self.attrs)
        self.ds = Dataset({'foo': self.v})
        self.dv = self.ds['foo']

        self.mindex = pd.MultiIndex.from_product([['a', 'b'], [1, 2]],
                                                 names=('level_1', 'level_2'))
        self.mda = DataArray([0, 1, 2, 3], coords={'x': self.mindex}, dims='x')

    def test_repr(self):
        v = Variable(['time', 'x'], [[1, 2, 3], [4, 5, 6]], {'foo': 'bar'})
        coords = OrderedDict([('x', np.arange(3, dtype=np.int64)),
                              ('other', np.int64(0))])
        data_array = DataArray(v, coords, name='my_variable')
        expected = dedent("""\
        <xarray.DataArray 'my_variable' (time: 2, x: 3)>
        array([[1, 2, 3],
               [4, 5, 6]])
        Coordinates:
          * x        (x) int64 0 1 2
            other    int64 0
        Dimensions without coordinates: time
        Attributes:
            foo:      bar""")
        self.assertEqual(expected, repr(data_array))

    def test_repr_multiindex(self):
        expected = dedent("""\
        <xarray.DataArray (x: 4)>
        array([0, 1, 2, 3])
        Coordinates:
          * x        (x) MultiIndex
          - level_1  (x) object 'a' 'a' 'b' 'b'
          - level_2  (x) int64 1 2 1 2""")
        self.assertEqual(expected, repr(self.mda))

    def test_properties(self):
        self.assertVariableEqual(self.dv.variable, self.v)
        self.assertArrayEqual(self.dv.values, self.v.values)
        for attr in ['dims', 'dtype', 'shape', 'size', 'nbytes',
                     'ndim', 'attrs']:
            self.assertEqual(getattr(self.dv, attr), getattr(self.v, attr))
        self.assertEqual(len(self.dv), len(self.v))
        self.assertVariableEqual(self.dv.variable, self.v)
        self.assertItemsEqual(list(self.dv.coords), list(self.ds.coords))
        for k, v in iteritems(self.dv.coords):
            self.assertArrayEqual(v, self.ds.coords[k])
        with pytest.raises(AttributeError):
            self.dv.dataset
        self.assertIsInstance(self.ds['x'].to_index(), pd.Index)
        with raises_regex(ValueError, 'must be 1-dimensional'):
            self.ds['foo'].to_index()
        with pytest.raises(AttributeError):
            self.dv.variable = self.v

    def test_data_property(self):
        array = DataArray(np.zeros((3, 4)))
        actual = array.copy()
        actual.values = np.ones((3, 4))
        self.assertArrayEqual(np.ones((3, 4)), actual.values)
        actual.data = 2 * np.ones((3, 4))
        self.assertArrayEqual(2 * np.ones((3, 4)), actual.data)
        self.assertArrayEqual(actual.data, actual.values)

    def test_indexes(self):
        array = DataArray(np.zeros((2, 3)),
                          [('x', [0, 1]), ('y', ['a', 'b', 'c'])])
        expected = OrderedDict([('x', pd.Index([0, 1])),
                                ('y', pd.Index(['a', 'b', 'c']))])
        assert array.indexes.keys() == expected.keys()
        for k in expected:
            assert array.indexes[k].equals(expected[k])

    def test_get_index(self):
        array = DataArray(np.zeros((2, 3)), coords={'x': ['a', 'b']},
                          dims=['x', 'y'])
        assert array.get_index('x').equals(pd.Index(['a', 'b']))
        assert array.get_index('y').equals(pd.Index([0, 1, 2]))
        with pytest.raises(KeyError):
            array.get_index('z')

    def test_get_index_size_zero(self):
        array = DataArray(np.zeros((0,)), dims=['x'])
        actual = array.get_index('x')
        expected = pd.Index([], dtype=np.int64)
        assert actual.equals(expected)
        assert actual.dtype == expected.dtype

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

        actual = DataArray(IndexVariable('x', [3]))
        actual.name = 'y'
        expected = DataArray([3], [('x', [3])], name='y')
        self.assertDataArrayIdentical(actual, expected)

    def test_dims(self):
        arr = self.dv
        self.assertEqual(arr.dims, ('x', 'y'))

        with raises_regex(AttributeError, 'you cannot assign'):
            arr.dims = ('w', 'z')

    def test_sizes(self):
        array = DataArray(np.zeros((3, 4)), dims=['x', 'y'])
        self.assertEqual(array.sizes, {'x': 3, 'y': 4})
        self.assertEqual(tuple(array.sizes), array.dims)
        with pytest.raises(TypeError):
            array.sizes['foo'] = 5

    def test_encoding(self):
        expected = {'foo': 'bar'}
        self.dv.encoding['foo'] = 'bar'
        assert expected, self.d == encoding

        expected = {'baz': 0}
        self.dv.encoding = expected
        assert expected, self.d == encoding
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

        with raises_regex(ValueError, 'coords is not dict-like'):
            DataArray(data, [[0, 1, 2]], ['x', 'y'])

        with raises_regex(ValueError, 'not a subset of the .* dim'):
            DataArray(data, {'x': [0, 1, 2]}, ['a', 'b'])
        with raises_regex(ValueError, 'not a subset of the .* dim'):
            DataArray(data, {'x': [0, 1, 2]})

        with raises_regex(TypeError, 'is not a string'):
            DataArray(data, dims=['x', None])

        with raises_regex(ValueError, 'conflicting sizes for dim'):
            DataArray([1, 2, 3], coords=[('x', [0, 1])])
        with raises_regex(ValueError, 'conflicting sizes for dim'):
            DataArray([1, 2], coords={'x': [0, 1], 'y': ('x', [1])}, dims='x')

        with raises_regex(ValueError, 'conflicting MultiIndex'):
            DataArray(np.random.rand(4, 4),
                      [('x', self.mindex), ('y', self.mindex)])
        with raises_regex(ValueError, 'conflicting MultiIndex'):
            DataArray(np.random.rand(4, 4),
                      [('x', self.mindex), ('level_1', range(4))])

        with raises_regex(ValueError, 'matching the dimension size'):
            DataArray(data, coords={'x': 0}, dims=['x', 'y'])

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
        expected['dim_0'] = [0]
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

        actual = DataArray(IndexVariable('foo', ['a', 'b']))
        self.assertDataArrayIdentical(expected, actual)

    def test_constructor_from_0d(self):
        expected = Dataset({None: ([], 0)})[None]
        actual = DataArray(0)
        self.assertDataArrayIdentical(expected, actual)

    @requires_dask
    def test_constructor_dask_coords(self):
        # regression test for GH1684
        import dask.array as da

        coord = da.arange(8, chunks=(4,))
        data = da.random.random((8, 8), chunks=(4, 4)) + 1
        actual = DataArray(data, coords={'x': coord, 'y': coord},
                           dims=['x', 'y'])

        ecoord = np.arange(8)
        expected = DataArray(data, coords={'x': ecoord, 'y': ecoord},
                             dims=['x', 'y'])
        assert_equal(actual, expected)

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
                  I[x.values[:3]], I[x.variable[:3]],
                  I[x[:3]], I[x[:3], y[:4]],
                  I[x.values > 3], I[x.variable > 3],
                  I[x > 3], I[x > 3, y > 3]]:
            assert_array_equal(self.v[i], self.dv[i])

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

    def test_getitem_dataarray(self):
        # It should not conflict
        da = DataArray(np.arange(12).reshape((3, 4)), dims=['x', 'y'])
        ind = DataArray([[0, 1], [0, 1]], dims=['x', 'z'])
        actual = da[ind]
        self.assertArrayEqual(actual, da.values[[[0, 1], [0, 1]], :])

        da = DataArray(np.arange(12).reshape((3, 4)), dims=['x', 'y'],
                       coords={'x': [0, 1, 2], 'y': ['a', 'b', 'c', 'd']})
        ind = xr.DataArray([[0, 1], [0, 1]], dims=['X', 'Y'])
        actual = da[ind]
        expected = da.values[[[0, 1], [0, 1]], :]
        self.assertArrayEqual(actual, expected)
        assert actual.dims == ('X', 'Y', 'y')

        # boolean indexing
        ind = xr.DataArray([True, True, False], dims=['x'])
        self.assertDataArrayEqual(da[ind], da[[0, 1], :])
        self.assertDataArrayEqual(da[ind], da[[0, 1]])
        self.assertDataArrayEqual(da[ind], da[ind.values])

    def test_setitem(self):
        # basic indexing should work as numpy's indexing
        tuples = [(0, 0), (0, slice(None, None)),
                  (slice(None, None), slice(None, None)),
                  (slice(None, None), 0),
                  ([1, 0], slice(None, None)),
                  (slice(None, None), [1, 0])]
        for t in tuples:
            expected = np.arange(6).reshape(3, 2)
            orig = DataArray(np.arange(6).reshape(3, 2),
                             {'x': [1, 2, 3], 'y': ['a', 'b'], 'z': 4,
                              'x2': ('x', ['a', 'b', 'c']),
                              'y2': ('y', ['d', 'e'])},
                             dims=['x', 'y'])
            orig[t] = 1
            expected[t] = 1
            self.assertArrayEqual(orig.values, expected)

    def test_contains(self):
        data_array = DataArray(1, coords={'x': 2})
        with pytest.warns(FutureWarning):
            assert 'x' in data_array

    def test_attr_sources_multiindex(self):
        # make sure attr-style access for multi-index levels
        # returns DataArray objects
        self.assertIsInstance(self.mda.level_1, DataArray)

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

        # Check that kwargs are passed
        import dask.array as da
        blocked = unblocked.chunk(name_prefix='testname_')
        self.assertIsInstance(blocked.data, da.Array)
        assert 'testname_' in blocked.data.name

    def test_isel(self):
        self.assertDataArrayIdentical(self.dv[0], self.dv.isel(x=0))
        self.assertDataArrayIdentical(self.dv, self.dv.isel(x=slice(None)))
        self.assertDataArrayIdentical(self.dv[:3], self.dv.isel(x=slice(3)))
        self.assertDataArrayIdentical(self.dv[:3, :5],
                                      self.dv.isel(x=slice(3), y=slice(5)))

    def test_isel_types(self):
        # regression test for #1405
        da = DataArray([1, 2, 3], dims='x')
        # uint64
        self.assertDataArrayIdentical(da.isel(x=np.array([0], dtype="uint64")),
                                      da.isel(x=np.array([0])))
        # uint32
        self.assertDataArrayIdentical(da.isel(x=np.array([0], dtype="uint32")),
                                      da.isel(x=np.array([0])))
        # int64
        self.assertDataArrayIdentical(da.isel(x=np.array([0], dtype="int64")),
                                      da.isel(x=np.array([0])))

    def test_isel_fancy(self):
        shape = (10, 7, 6)
        np_array = np.random.random(shape)
        da = DataArray(np_array, dims=['time', 'y', 'x'],
                       coords={'time': np.arange(0, 100, 10)})
        y = [1, 3]
        x = [3, 0]

        expected = da.values[:, y, x]

        actual = da.isel(y=(('test_coord', ), y), x=(('test_coord', ), x))
        assert actual.coords['test_coord'].shape == (len(y), )
        assert list(actual.coords) == ['time']
        assert actual.dims == ('time', 'test_coord')

        np.testing.assert_equal(actual, expected)

        # a few corner cases
        da.isel(time=(('points',), [1, 2]), x=(('points',), [2, 2]),
                y=(('points',), [3, 4]))
        np.testing.assert_allclose(
            da.isel_points(time=[1], x=[2], y=[4]).values.squeeze(),
            np_array[1, 4, 2].squeeze())
        da.isel(time=(('points', ), [1, 2]))
        y = [-1, 0]
        x = [-2, 2]
        expected = da.values[:, y, x]
        actual = da.isel(x=(('points', ), x), y=(('points', ), y)).values
        np.testing.assert_equal(actual, expected)

        # test that the order of the indexers doesn't matter
        self.assertDataArrayIdentical(
            da.isel(y=(('points', ), y), x=(('points', ), x)),
            da.isel(x=(('points', ), x), y=(('points', ), y)))

        # make sure we're raising errors in the right places
        with raises_regex(IndexError,
                                     'Dimensions of indexers mismatch'):
            da.isel(y=(('points', ), [1, 2]), x=(('points', ), [1, 2, 3]))

        # tests using index or DataArray as indexers
        stations = Dataset()
        stations['station'] = (('station', ), ['A', 'B', 'C'])
        stations['dim1s'] = (('station', ), [1, 2, 3])
        stations['dim2s'] = (('station', ), [4, 5, 1])

        actual = da.isel(x=stations['dim1s'], y=stations['dim2s'])
        assert 'station' in actual.coords
        assert 'station' in actual.dims
        self.assertDataArrayIdentical(actual['station'], stations['station'])

        with raises_regex(ValueError, 'conflicting values for '):
            da.isel(x=DataArray([0, 1, 2], dims='station',
                                coords={'station': [0, 1, 2]}),
                    y=DataArray([0, 1, 2], dims='station',
                                coords={'station': [0, 1, 3]}))

        # multi-dimensional selection
        stations = Dataset()
        stations['a'] = (('a', ), ['A', 'B', 'C'])
        stations['b'] = (('b', ), [0, 1])
        stations['dim1s'] = (('a', 'b'), [[1, 2], [2, 3], [3, 4]])
        stations['dim2s'] = (('a', ), [4, 5, 1])

        actual = da.isel(x=stations['dim1s'], y=stations['dim2s'])
        assert 'a' in actual.coords
        assert 'a' in actual.dims
        assert 'b' in actual.coords
        assert 'b' in actual.dims
        self.assertDataArrayIdentical(actual['a'], stations['a'])
        self.assertDataArrayIdentical(actual['b'], stations['b'])
        expected = da.variable[:, stations['dim2s'].variable,
                               stations['dim1s'].variable]
        self.assertArrayEqual(actual, expected)

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

    def test_sel_dataarray(self):
        # indexing with DataArray
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        da = self.ds['foo']

        ind = DataArray(['a', 'b', 'c'], dims=['x'])
        actual = da.sel(x=ind)
        self.assertDataArrayIdentical(actual, da.isel(x=[0, 1, 2]))

        # along new dimension
        ind = DataArray(['a', 'b', 'c'], dims=['new_dim'])
        actual = da.sel(x=ind)
        self.assertArrayEqual(actual, da.isel(x=[0, 1, 2]))
        assert 'new_dim' in actual.dims

        # with coordinate
        ind = DataArray(['a', 'b', 'c'], dims=['new_dim'],
                        coords={'new_dim': [0, 1, 2]})
        actual = da.sel(x=ind)
        self.assertArrayEqual(actual, da.isel(x=[0, 1, 2]))
        assert 'new_dim' in actual.dims
        assert 'new_dim' in actual.coords
        self.assertDataArrayEqual(actual['new_dim'].drop('x'),
                                  ind['new_dim'])

    def test_sel_no_index(self):
        array = DataArray(np.arange(10), dims='x')
        self.assertDataArrayIdentical(array[0], array.sel(x=0))
        self.assertDataArrayIdentical(array[:5], array.sel(x=slice(5)))
        self.assertDataArrayIdentical(array[[0, -1]], array.sel(x=[0, -1]))
        self.assertDataArrayIdentical(
            array[array < 5], array.sel(x=(array < 5)))

    def test_sel_method(self):
        data = DataArray(np.random.randn(3, 4),
                         [('x', [0, 1, 2]), ('y', list('abcd'))])

        expected = data.sel(y=['a', 'b'])
        actual = data.sel(y=['ab', 'ba'], method='pad')
        self.assertDataArrayIdentical(expected, actual)

        expected = data.sel(x=[1, 2])
        actual = data.sel(x=[0.9, 1.9], method='backfill', tolerance=1)
        self.assertDataArrayIdentical(expected, actual)

    def test_sel_drop(self):
        data = DataArray([1, 2, 3], [('x', [0, 1, 2])])
        expected = DataArray(1)
        selected = data.sel(x=0, drop=True)
        self.assertDataArrayIdentical(expected, selected)

        expected = DataArray(1, {'x': 0})
        selected = data.sel(x=0, drop=False)
        self.assertDataArrayIdentical(expected, selected)

        data = DataArray([1, 2, 3], dims=['x'])
        expected = DataArray(1)
        selected = data.sel(x=0, drop=True)
        self.assertDataArrayIdentical(expected, selected)

    def test_isel_drop(self):
        data = DataArray([1, 2, 3], [('x', [0, 1, 2])])
        expected = DataArray(1)
        selected = data.isel(x=0, drop=True)
        self.assertDataArrayIdentical(expected, selected)

        expected = DataArray(1, {'x': 0})
        selected = data.isel(x=0, drop=False)
        self.assertDataArrayIdentical(expected, selected)

    def test_isel_points(self):
        shape = (10, 5, 6)
        np_array = np.random.random(shape)
        da = DataArray(np_array, dims=['time', 'y', 'x'],
                       coords={'time': np.arange(0, 100, 10)})
        y = [1, 3]
        x = [3, 0]

        expected = da.values[:, y, x]

        actual = da.isel_points(y=y, x=x, dim='test_coord')
        assert actual.coords['test_coord'].shape == (len(y), )
        assert list(actual.coords) == ['time']
        assert actual.dims == ('test_coord', 'time')

        actual = da.isel_points(y=y, x=x)
        assert 'points' in actual.dims
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
        with raises_regex(ValueError,
                                     'All indexers must be the same length'):
            da.isel_points(y=[1, 2], x=[1, 2, 3])
        with raises_regex(ValueError,
                                     'dimension bad_key does not exist'):
            da.isel_points(bad_key=[1, 2])
        with raises_regex(TypeError, 'Indexers must be integers'):
            da.isel_points(y=[1.5, 2.2])
        with raises_regex(TypeError, 'Indexers must be integers'):
            da.isel_points(x=[1, 2, 3], y=slice(3))
        with raises_regex(ValueError,
                                     'Indexers must be 1 dimensional'):
            da.isel_points(y=1, x=2)
        with raises_regex(ValueError,
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

    def test_loc_assign(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        da = self.ds['foo']
        # assignment
        da.loc['a':'j'] = 0
        self.assertTrue(np.all(da.values == 0))
        da.loc[{'x': slice('a', 'j')}] = 2
        self.assertTrue(np.all(da.values == 2))

        da.loc[{'x': slice('a', 'j')}] = 2
        self.assertTrue(np.all(da.values == 2))

        # Multi dimensional case
        da = DataArray(np.arange(12).reshape(3, 4), dims=['x', 'y'])
        da.loc[0, 0] = 0
        assert da.values[0, 0] == 0
        assert da.values[0, 1] != 0

        da = DataArray(np.arange(12).reshape(3, 4), dims=['x', 'y'])
        da.loc[0] = 0
        self.assertTrue(np.all(da.values[0] == np.zeros(4)))
        assert da.values[1, 0] != 0

    def test_loc_single_boolean(self):
        data = DataArray([0, 1], coords=[[True, False]])
        self.assertEqual(data.loc[True], 0)
        self.assertEqual(data.loc[False], 1)

    def test_selection_multiindex(self):
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
                self.assertVariableIdentical(da.variable, expected_da.variable)
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
        with pytest.raises(IndexError):
            mdata.loc[('a', 1)]

        self.assertDataArrayIdentical(mdata.sel(x={'one': 'a', 'two': 1}),
                                      mdata.sel(one='a', two=1))

    def test_virtual_default_coords(self):
        array = DataArray(np.zeros((5,)), dims='x')
        expected = DataArray(range(5), dims='x', name='x')
        self.assertDataArrayIdentical(expected, array['x'])
        self.assertDataArrayIdentical(expected, array.coords['x'])

    def test_virtual_time_components(self):
        dates = pd.date_range('2000-01-01', periods=10)
        da = DataArray(np.arange(1, 11), [('time', dates)])

        self.assertArrayEqual(da['time.dayofyear'], da.values)
        self.assertArrayEqual(da.coords['time.dayofyear'], da.values)

    def test_coords(self):
        # use int64 to ensure repr() consistency on windows
        coords = [IndexVariable('x', np.array([-1, -2], 'int64')),
                  IndexVariable('y', np.array([0, 1, 2], 'int64'))]
        da = DataArray(np.random.randn(2, 3), coords, name='foo')

        assert 2 == len(da.coords)

        self.assertEqual(['x', 'y'], list(da.coords))

        self.assertTrue(coords[0].identical(da.coords['x']))
        self.assertTrue(coords[1].identical(da.coords['y']))

        self.assertIn('x', da.coords)
        self.assertNotIn(0, da.coords)
        self.assertNotIn('foo', da.coords)

        with pytest.raises(KeyError):
            da.coords[0]
        with pytest.raises(KeyError):
            da.coords['foo']

        expected = dedent("""\
        Coordinates:
          * x        (x) int64 -1 -2
          * y        (y) int64 0 1 2""")
        actual = repr(da.coords)
        assert expected == actual

        del da.coords['x']
        expected = DataArray(da.values, {'y': [0, 1, 2]}, dims=['x', 'y'],
                             name='foo')
        self.assertDataArrayIdentical(da, expected)

        with raises_regex(ValueError, 'conflicting MultiIndex'):
            self.mda['level_1'] = np.arange(4)
            self.mda.coords['level_1'] = np.arange(4)

    def test_coords_to_index(self):
        da = DataArray(np.zeros((2, 3)), [('x', [1, 2]), ('y', list('abc'))])

        with raises_regex(ValueError, 'no valid index'):
            da[0, 0].coords.to_index()

        expected = pd.Index(['a', 'b', 'c'], name='y')
        actual = da[0].coords.to_index()
        assert expected.equals(actual)

        expected = pd.MultiIndex.from_product([[1, 2], ['a', 'b', 'c']],
                                              names=['x', 'y'])
        actual = da.coords.to_index()
        assert expected.equals(actual)

        expected = pd.MultiIndex.from_product([['a', 'b', 'c'], [1, 2]],
                                              names=['y', 'x'])
        actual = da.coords.to_index(['y', 'x'])
        assert expected.equals(actual)

        with raises_regex(ValueError, 'ordered_dims must match'):
            da.coords.to_index(['x'])

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
                          'baz': ('y', range(4)),
                          'y': range(4)},
                         dims=['x', 'y'],
                         name='foo')

        actual = data.reset_coords()
        expected = Dataset({'foo': (['x', 'y'], np.zeros((3, 4))),
                            'bar': ('x', ['a', 'b', 'c']),
                            'baz': ('y', range(4)),
                            'y': range(4)})
        self.assertDatasetIdentical(actual, expected)

        actual = data.reset_coords(['bar', 'baz'])
        self.assertDatasetIdentical(actual, expected)

        actual = data.reset_coords('bar')
        expected = Dataset({'foo': (['x', 'y'], np.zeros((3, 4))),
                            'bar': ('x', ['a', 'b', 'c'])},
                           {'baz': ('y', range(4)), 'y': range(4)})
        self.assertDatasetIdentical(actual, expected)

        actual = data.reset_coords(['bar'])
        self.assertDatasetIdentical(actual, expected)

        actual = data.reset_coords(drop=True)
        expected = DataArray(np.zeros((3, 4)), coords={'y': range(4)},
                             dims=['x', 'y'], name='foo')
        self.assertDataArrayIdentical(actual, expected)

        actual = data.copy()
        actual.reset_coords(drop=True, inplace=True)
        self.assertDataArrayIdentical(actual, expected)

        actual = data.reset_coords('bar', drop=True)
        expected = DataArray(np.zeros((3, 4)),
                             {'baz': ('y', range(4)), 'y': range(4)},
                             dims=['x', 'y'], name='foo')
        self.assertDataArrayIdentical(actual, expected)

        with raises_regex(ValueError, 'cannot reset coord'):
            data.reset_coords(inplace=True)
        with raises_regex(ValueError, 'cannot be found'):
            data.reset_coords('foo', drop=True)
        with raises_regex(ValueError, 'cannot be found'):
            data.reset_coords('not_found')
        with raises_regex(ValueError, 'cannot remove index'):
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

        with raises_regex(ValueError, 'conflicting MultiIndex'):
            self.mda.assign_coords(level_1=range(4))

    def test_coords_alignment(self):
        lhs = DataArray([1, 2, 3], [('x', [0, 1, 2])])
        rhs = DataArray([2, 3, 4], [('x', [1, 2, 3])])
        lhs.coords['rhs'] = rhs

        expected = DataArray([1, 2, 3],
                             coords={'rhs': ('x', [np.nan, 2, 3]),
                                     'x': [0, 1, 2]},
                             dims='x')
        self.assertDataArrayIdentical(lhs, expected)

    def test_coords_replacement_alignment(self):
        # regression test for GH725
        arr = DataArray([0, 1, 2], dims=['abc'])
        new_coord = DataArray([1, 2, 3], dims=['abc'], coords=[[1, 2, 3]])
        arr['abc'] = new_coord
        expected = DataArray([0, 1, 2], coords=[('abc', [1, 2, 3])])
        self.assertDataArrayIdentical(arr, expected)

    def test_coords_non_string(self):
        arr = DataArray(0, coords={1: 2})
        actual = arr.coords[1]
        expected = DataArray(2, coords={1: 2}, name=1)
        self.assertDataArrayIdentical(actual, expected)

    def test_reindex_like(self):
        foo = DataArray(np.random.randn(5, 6),
                        [('x', range(5)), ('y', range(6))])
        bar = foo[:2, :2]
        self.assertDataArrayIdentical(foo.reindex_like(bar), bar)

        expected = foo.copy()
        expected[:] = np.nan
        expected[:2, :2] = bar
        self.assertDataArrayIdentical(bar.reindex_like(foo), expected)

    def test_reindex_like_no_index(self):
        foo = DataArray(np.random.randn(5, 6), dims=['x', 'y'])
        self.assertDatasetIdentical(foo, foo.reindex_like(foo))

        bar = foo[:4]
        with raises_regex(
                ValueError, 'different size for unlabeled'):
            foo.reindex_like(bar)

    def test_reindex_regressions(self):
        # regression test for #279
        expected = DataArray(np.random.randn(5), coords=[("time", range(5))])
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
        x = DataArray([10, 20], dims='y', coords={'y': [0, 1]})
        y = [-0.1, 0.5, 1.1]
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

        renamed = self.dv.x.rename({'x': 'z'}).rename('z')
        self.assertDatasetIdentical(
            renamed, self.ds.rename({'x': 'z'}).z)
        self.assertEqual(renamed.name, 'z')
        self.assertEqual(renamed.dims, ('z',))

    def test_swap_dims(self):
        array = DataArray(np.random.randn(3), {'y': ('x', list('abc'))}, 'x')
        expected = DataArray(array.values, {'y': list('abc')}, dims='y')
        actual = array.swap_dims({'x': 'y'})
        self.assertDataArrayIdentical(expected, actual)

    def test_expand_dims_error(self):
        array = DataArray(np.random.randn(3, 4), dims=['x', 'dim_0'],
                          coords={'x': np.linspace(0.0, 1.0, 3.0)},
                          attrs={'key': 'entry'})

        with raises_regex(ValueError, 'dim should be str or'):
            array.expand_dims(0)
        with raises_regex(ValueError, 'lengths of dim and axis'):
            # dims and axis argument should be the same length
            array.expand_dims(dim=['a', 'b'], axis=[1, 2, 3])
        with raises_regex(ValueError, 'Dimension x already'):
            # Should not pass the already existing dimension.
            array.expand_dims(dim=['x'])
        # raise if duplicate
        with raises_regex(ValueError, 'duplicate values.'):
            array.expand_dims(dim=['y', 'y'])
        with raises_regex(ValueError, 'duplicate values.'):
            array.expand_dims(dim=['y', 'z'], axis=[1, 1])
        with raises_regex(ValueError, 'duplicate values.'):
            array.expand_dims(dim=['y', 'z'], axis=[2, -2])

        # out of bounds error, axis must be in [-4, 3]
        with pytest.raises(IndexError):
            array.expand_dims(dim=['y', 'z'], axis=[2, 4])
        with pytest.raises(IndexError):
            array.expand_dims(dim=['y', 'z'], axis=[2, -5])
        # Does not raise an IndexError
        array.expand_dims(dim=['y', 'z'], axis=[2, -4])
        array.expand_dims(dim=['y', 'z'], axis=[2, 3])

    def test_expand_dims(self):
        array = DataArray(np.random.randn(3, 4), dims=['x', 'dim_0'],
                          coords={'x': np.linspace(0.0, 1.0, 3)},
                          attrs={'key': 'entry'})
        # pass only dim label
        actual = array.expand_dims(dim='y')
        expected = DataArray(np.expand_dims(array.values, 0),
                             dims=['y', 'x', 'dim_0'],
                             coords={'x': np.linspace(0.0, 1.0, 3)},
                             attrs={'key': 'entry'})
        self.assertDataArrayIdentical(expected, actual)
        roundtripped = actual.squeeze('y', drop=True)
        self.assertDatasetIdentical(array, roundtripped)

        # pass multiple dims
        actual = array.expand_dims(dim=['y', 'z'])
        expected = DataArray(np.expand_dims(np.expand_dims(array.values, 0),
                                            0),
                             dims=['y', 'z', 'x', 'dim_0'],
                             coords={'x': np.linspace(0.0, 1.0, 3)},
                             attrs={'key': 'entry'})
        self.assertDataArrayIdentical(expected, actual)
        roundtripped = actual.squeeze(['y', 'z'], drop=True)
        self.assertDatasetIdentical(array, roundtripped)

        # pass multiple dims and axis. Axis is out of order
        actual = array.expand_dims(dim=['z', 'y'], axis=[2, 1])
        expected = DataArray(np.expand_dims(np.expand_dims(array.values, 1),
                                            2),
                             dims=['x', 'y', 'z', 'dim_0'],
                             coords={'x': np.linspace(0.0, 1.0, 3)},
                             attrs={'key': 'entry'})
        self.assertDataArrayIdentical(expected, actual)
        # make sure the attrs are tracked
        self.assertTrue(actual.attrs['key'] == 'entry')
        roundtripped = actual.squeeze(['z', 'y'], drop=True)
        self.assertDatasetIdentical(array, roundtripped)

        # Negative axis and they are out of order
        actual = array.expand_dims(dim=['y', 'z'], axis=[-1, -2])
        expected = DataArray(np.expand_dims(np.expand_dims(array.values, -1),
                                            -1),
                             dims=['x', 'dim_0', 'z', 'y'],
                             coords={'x': np.linspace(0.0, 1.0, 3)},
                             attrs={'key': 'entry'})
        self.assertDataArrayIdentical(expected, actual)
        self.assertTrue(actual.attrs['key'] == 'entry')
        roundtripped = actual.squeeze(['y', 'z'], drop=True)
        self.assertDatasetIdentical(array, roundtripped)

    def test_expand_dims_with_scalar_coordinate(self):
        array = DataArray(np.random.randn(3, 4), dims=['x', 'dim_0'],
                          coords={'x': np.linspace(0.0, 1.0, 3), 'z': 1.0},
                          attrs={'key': 'entry'})
        actual = array.expand_dims(dim='z')
        expected = DataArray(np.expand_dims(array.values, 0),
                             dims=['z', 'x', 'dim_0'],
                             coords={'x': np.linspace(0.0, 1.0, 3),
                                     'z': np.ones(1)},
                             attrs={'key': 'entry'})
        self.assertDataArrayIdentical(expected, actual)
        roundtripped = actual.squeeze(['z'], drop=False)
        self.assertDatasetIdentical(array, roundtripped)

    def test_set_index(self):
        indexes = [self.mindex.get_level_values(n) for n in self.mindex.names]
        coords = {idx.name: ('x', idx) for idx in indexes}
        array = DataArray(self.mda.values, coords=coords, dims='x')
        expected = self.mda.copy()
        level_3 = ('x', [1, 2, 3, 4])
        array['level_3'] = level_3
        expected['level_3'] = level_3

        obj = array.set_index(x=self.mindex.names)
        self.assertDataArrayIdentical(obj, expected)

        obj = obj.set_index(x='level_3', append=True)
        expected = array.set_index(x=['level_1', 'level_2', 'level_3'])
        self.assertDataArrayIdentical(obj, expected)

        array.set_index(x=['level_1', 'level_2', 'level_3'], inplace=True)
        self.assertDataArrayIdentical(array, expected)

        array2d = DataArray(np.random.rand(2, 2),
                            coords={'x': ('x', [0, 1]),
                                    'level': ('y', [1, 2])},
                            dims=('x', 'y'))
        with raises_regex(ValueError, 'dimension mismatch'):
            array2d.set_index(x='level')

    def test_reset_index(self):
        indexes = [self.mindex.get_level_values(n) for n in self.mindex.names]
        coords = {idx.name: ('x', idx) for idx in indexes}
        expected = DataArray(self.mda.values, coords=coords, dims='x')

        obj = self.mda.reset_index('x')
        self.assertDataArrayIdentical(obj, expected)
        obj = self.mda.reset_index(self.mindex.names)
        self.assertDataArrayIdentical(obj, expected)
        obj = self.mda.reset_index(['x', 'level_1'])
        self.assertDataArrayIdentical(obj, expected)

        coords = {'x': ('x', self.mindex.droplevel('level_1')),
                  'level_1': ('x', self.mindex.get_level_values('level_1'))}
        expected = DataArray(self.mda.values, coords=coords, dims='x')
        obj = self.mda.reset_index(['level_1'])
        self.assertDataArrayIdentical(obj, expected)

        expected = DataArray(self.mda.values, dims='x')
        obj = self.mda.reset_index('x', drop=True)
        self.assertDataArrayIdentical(obj, expected)

        array = self.mda.copy()
        array.reset_index(['x'], drop=True, inplace=True)
        self.assertDataArrayIdentical(array, expected)

        # single index
        array = DataArray([1, 2], coords={'x': ['a', 'b']}, dims='x')
        expected = DataArray([1, 2], coords={'x_': ('x', ['a', 'b'])},
                             dims='x')
        self.assertDataArrayIdentical(array.reset_index('x'), expected)

    def test_reorder_levels(self):
        midx = self.mindex.reorder_levels(['level_2', 'level_1'])
        expected = DataArray(self.mda.values, coords={'x': midx}, dims='x')

        obj = self.mda.reorder_levels(x=['level_2', 'level_1'])
        self.assertDataArrayIdentical(obj, expected)

        array = self.mda.copy()
        array.reorder_levels(x=['level_2', 'level_1'], inplace=True)
        self.assertDataArrayIdentical(array, expected)

        array = DataArray([1, 2], dims='x')
        with pytest.raises(KeyError):
            array.reorder_levels(x=['level_1', 'level_2'])

        array['x'] = [0, 1]
        with raises_regex(ValueError, 'has no MultiIndex'):
            array.reorder_levels(x=['level_1', 'level_2'])

    def test_dataset_getitem(self):
        dv = self.ds['foo']
        self.assertDataArrayIdentical(dv, self.dv)

    def test_array_interface(self):
        self.assertArrayEqual(np.asarray(self.dv), self.x)
        # test patched in methods
        self.assertArrayEqual(self.dv.astype(float), self.v.astype(float))
        assert_array_equal(self.dv.argsort(), self.v.argsort())
        assert_array_equal(self.dv.clip(2, 3), self.v.clip(2, 3))
        # test ufuncs
        expected = deepcopy(self.ds)
        expected['foo'][:] = np.sin(self.x)
        self.assertDataArrayEqual(expected['foo'], np.sin(self.dv))
        assert_array_equal(self.dv, np.maximum(self.v, self.dv))
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
        with pytest.raises(xr.MergeError):
            a += b
        with pytest.raises(xr.MergeError):
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
        expected = DataArray(1 + np.arange(3), dims='x', name='x')
        self.assertDataArrayIdentical(expected, actual)

        # regression tests for #254
        actual = orig[0] < orig
        expected = DataArray([False, True, True], dims='x', name='x')
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
        actual = orig.stack(z=['x', 'y']).unstack('z').drop(['x', 'y'])
        self.assertDataArrayIdentical(orig, actual)

    def test_stack_unstack_decreasing_coordinate(self):
        # regression test for GH980
        orig = DataArray(np.random.rand(3, 4), dims=('y', 'x'),
                         coords={'x': np.arange(4),
                                 'y': np.arange(3, 0, -1)})
        stacked = orig.stack(allpoints=['y', 'x'])
        actual = stacked.unstack('allpoints')
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
                                 self.dv.transpose().variable)

    def test_squeeze(self):
        assert_equal(self.dv.variable.squeeze(), self.dv.squeeze().variable)

    def test_squeeze_drop(self):
        array = DataArray([1], [('x', [0])])
        expected = DataArray(1)
        actual = array.squeeze(drop=True)
        self.assertDataArrayIdentical(expected, actual)

        expected = DataArray(1, {'x': 0})
        actual = array.squeeze(drop=False)
        self.assertDataArrayIdentical(expected, actual)

    def test_drop_coordinates(self):
        expected = DataArray(np.random.randn(2, 3), dims=['x', 'y'])
        arr = expected.copy()
        arr.coords['z'] = 2
        actual = arr.drop('z')
        self.assertDataArrayIdentical(expected, actual)

        with pytest.raises(ValueError):
            arr.drop('not found')

        with raises_regex(ValueError, 'cannot be found'):
            arr.drop(None)

        renamed = arr.rename('foo')
        with raises_regex(ValueError, 'cannot be found'):
            renamed.drop('foo')

    def test_drop_index_labels(self):
        arr = DataArray(np.random.randn(2, 3), coords={'y': [0, 1, 2]},
                        dims=['x', 'y'])
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

    def test_where(self):
        arr = DataArray(np.arange(4), dims='x')
        expected = arr.sel(x=slice(2))
        actual = arr.where(arr.x < 2, drop=True)
        self.assertDataArrayIdentical(actual, expected)

    def test_cumops(self):
        coords = {'x': [-1, -2], 'y': ['ab', 'cd', 'ef'],
                  'lat': (['x', 'y'], [[1, 2, 3], [-1, -2, -3]]),
                  'c': -999}
        orig = DataArray([[-1, 0, 1], [-3, 0, 3]], coords,
                         dims=['x', 'y'])

        actual = orig.cumsum('x')
        expected = DataArray([[-1, 0, 1], [-4, 0, 4]], coords,
                             dims=['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        actual = orig.cumsum('y')
        expected = DataArray([[-1, -1, 0], [-3, -3, 0]], coords,
                             dims=['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        actual = orig.cumprod('x')
        expected = DataArray([[-1, 0, 1], [3, 0, 3]], coords,
                             dims=['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        actual = orig.cumprod('y')
        expected = DataArray([[-1, 0, 0], [-3, 0, 0]], coords, dims=['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

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

        self.assertVariableEqual(self.dv.reduce(np.mean, 'x').variable,
                                 self.v.reduce(np.mean, 'x'))

        orig = DataArray([[1, 0, np.nan], [3, 0, 3]], coords, dims=['x', 'y'])
        actual = orig.count()
        expected = DataArray(5, {'c': -999})
        self.assertDataArrayIdentical(expected, actual)

        # uint support
        orig = DataArray(np.arange(6).reshape(3, 2).astype('uint'),
                         dims=['x', 'y'])
        assert orig.dtype.kind == 'u'
        actual = orig.mean(dim='x', skipna=True)
        expected = DataArray(orig.values.astype(int),
                             dims=['x', 'y']).mean('x')
        self.assertDataArrayEqual(actual, expected)

    # skip due to bug in older versions of numpy.nanpercentile
    def test_quantile(self):
        for q in [0.25, [0.50], [0.25, 0.75]]:
            for axis, dim in zip([None, 0, [0], [0, 1]],
                                 [None, 'x', ['x'], ['x', 'y']]):
                actual = self.dv.quantile(q, dim=dim)
                expected = np.nanpercentile(self.dv.values, np.array(q) * 100,
                                            axis=axis)
                np.testing.assert_allclose(actual.values, expected)

    def test_reduce_keep_attrs(self):
        # Test dropped attrs
        vm = self.va.mean()
        self.assertEqual(len(vm.attrs), 0)
        self.assertEqual(vm.attrs, OrderedDict())

        # Test kept attrs
        vm = self.va.mean(keep_attrs=True)
        self.assertEqual(len(vm.attrs), len(self.attrs))
        self.assertEqual(vm.attrs, self.attrs)

    def test_assign_attrs(self):
        expected = DataArray([], attrs=dict(a=1, b=2))
        expected.attrs['a'] = 1
        expected.attrs['b'] = 2
        new = DataArray([])
        actual = DataArray([]).assign_attrs(a=1, b=2)
        self.assertDatasetIdentical(actual, expected)
        self.assertEqual(new.attrs, {})

        expected.attrs['c'] = 3
        new_actual = actual.assign_attrs({'c': 3})
        self.assertDatasetIdentical(new_actual, expected)
        self.assertEqual(actual.attrs, {'a': 1, 'b': 2})

    def test_fillna(self):
        a = DataArray([np.nan, 1, np.nan, 3], coords={'x': range(4)}, dims='x')
        actual = a.fillna(-1)
        expected = DataArray([-1, 1, -1, 3], coords={'x': range(4)}, dims='x')
        self.assertDataArrayIdentical(expected, actual)

        b = DataArray(range(4), coords={'x': range(4)}, dims='x')
        actual = a.fillna(b)
        expected = b.copy()
        self.assertDataArrayIdentical(expected, actual)

        actual = a.fillna(range(4))
        self.assertDataArrayIdentical(expected, actual)

        actual = a.fillna(b[:3])
        self.assertDataArrayIdentical(expected, actual)

        actual = a.fillna(b[:0])
        self.assertDataArrayIdentical(a, actual)

        with raises_regex(TypeError, 'fillna on a DataArray'):
            a.fillna({0: 0})

        with raises_regex(ValueError, 'broadcast'):
            a.fillna([1, 2])

        fill_value = DataArray([0, 1], dims='y')
        actual = a.fillna(fill_value)
        expected = DataArray([[0, 1], [1, 1], [0, 1], [3, 3]],
                             coords={'x': range(4)}, dims=('x', 'y'))
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
        expected_groups = {'a': range(0, 9), 'c': [9], 'b': range(10, 20)}
        self.assertItemsEqual(expected_groups.keys(), grouped.groups.keys())
        for key in expected_groups:
            self.assertArrayEqual(expected_groups[key], grouped.groups[key])
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
             'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
        self.assertDataArrayAllClose(expected_sum_axis1,
                                     grouped.reduce(np.sum, 'y'))
        self.assertDataArrayAllClose(expected_sum_axis1, grouped.sum('y'))

    def test_groupby_count(self):
        array = DataArray(
            [0, 0, np.nan, np.nan, 0, 0],
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

        with raises_regex(TypeError, 'only support binary ops'):
            grouped + 1
        with raises_regex(TypeError, 'only support binary ops'):
            grouped + grouped
        with raises_regex(TypeError, 'in-place operations'):
            array += grouped

    def test_groupby_math_not_aligned(self):
        array = DataArray(range(4), {'b': ('x', [0, 0, 1, 1]),
                                     'x': [0, 1, 2, 3]},
                          dims='x')
        other = DataArray([10], coords={'b': [0]}, dims='b')
        actual = array.groupby('b') + other
        expected = DataArray([10, 11, np.nan, np.nan], array.coords)
        self.assertDataArrayIdentical(expected, actual)

        other = DataArray([10], coords={'c': 123, 'b': [0]}, dims='b')
        actual = array.groupby('b') + other
        expected.coords['c'] = (['x'], [123] * 2 + [np.nan] * 2)
        self.assertDataArrayIdentical(expected, actual)

        other = Dataset({'a': ('b', [10])}, {'b': [0]})
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
        return DataArray([[[0, 1], [2, 3]], [[5, 10], [15, 20]]],
                         coords={'lon': (['ny', 'nx'], [[30, 40], [40, 50]]),
                                 'lat': (['ny', 'nx'], [[10, 10], [20, 20]])},
                         dims=['time', 'ny', 'nx'])

    def test_groupby_multidim(self):
        array = self.make_groupby_multidim_example_array()
        for dim, expected_sum in [
                ('lon', DataArray([5, 28, 23],
                                  coords=[('lon', [30., 40., 50.])])),
                ('lat', DataArray([16, 40], coords=[('lat', [10., 20.])]))]:
            actual_sum = array.groupby(dim).sum()
            self.assertDataArrayIdentical(expected_sum, actual_sum)

    def test_groupby_multidim_apply(self):
        array = self.make_groupby_multidim_example_array()
        actual = array.groupby('lon').apply(lambda x: x - x.mean())
        expected = DataArray([[[-2.5, -6.], [-5., -8.5]],
                              [[2.5, 3.], [8., 8.5]]],
                             coords=array.coords, dims=array.dims)
        self.assertDataArrayIdentical(expected, actual)

    def test_groupby_bins(self):
        array = DataArray(np.arange(4), dims='dim_0')
        # the first value should not be part of any group ("right" binning)
        array[0] = 99
        # bins follow conventions for pandas.cut
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
        bins = [0, 1.5, 5]
        bin_coords = pd.cut(array['dim_0'], bins).categories
        expected = DataArray([1, 5], dims='dim_0_bins',
                             coords={'dim_0_bins': bin_coords})
        # the problem with this is that it overwrites the dimensions of array!
        # actual = array.groupby('dim_0', bins=bins).sum()
        actual = array.groupby_bins('dim_0', bins).apply(lambda x: x.sum())
        self.assertDataArrayIdentical(expected, actual)
        # make sure original array dims are unchanged
        self.assertEqual(len(array.dim_0), 4)

    def test_groupby_bins_empty(self):
        array = DataArray(np.arange(4), [('x', range(4))])
        # one of these bins will be empty
        bins = [0, 4, 5]
        bin_coords = pd.cut(array['x'], bins).categories
        actual = array.groupby_bins('x', bins).sum()
        expected = DataArray([6, np.nan], dims='x_bins',
                             coords={'x_bins': bin_coords})
        self.assertDataArrayIdentical(expected, actual)
        # make sure original array is unchanged
        # (was a problem in earlier versions)
        self.assertEqual(len(array.x), 4)

    def test_groupby_bins_multidim(self):
        array = self.make_groupby_multidim_example_array()
        bins = [0, 15, 20]
        bin_coords = pd.cut(array['lat'].values.flat, bins).categories
        expected = DataArray([16, 40], dims='lat_bins',
                             coords={'lat_bins': bin_coords})
        actual = array.groupby_bins('lat', bins).apply(lambda x: x.sum())
        self.assertDataArrayIdentical(expected, actual)
        # modify the array coordinates to be non-monotonic after unstacking
        array['lat'].data = np.array([[10., 20.], [20., 10.]])
        expected = DataArray([28, 28], dims='lat_bins',
                             coords={'lat_bins': bin_coords})
        actual = array.groupby_bins('lat', bins).apply(lambda x: x.sum())
        self.assertDataArrayIdentical(expected, actual)

    def test_groupby_bins_sort(self):
        data = xr.DataArray(
            np.arange(100), dims='x',
            coords={'x': np.linspace(-100, 100, num=100)})
        binned_mean = data.groupby_bins('x', bins=11).mean()
        assert binned_mean.to_index().is_monotonic

    def test_resample(self):
        times = pd.date_range('2000-01-01', freq='6H', periods=10)
        array = DataArray(np.arange(10), [('time', times)])

        actual = array.resample(time='24H').mean()
        expected = DataArray(array.to_series().resample('24H').mean())
        self.assertDataArrayIdentical(expected, actual)

        actual = array.resample(time='24H').reduce(np.mean)
        self.assertDataArrayIdentical(expected, actual)

        with raises_regex(ValueError, 'index must be monotonic'):
            array[[2, 0, 1]].resample(time='1D')

    def test_resample_first(self):
        times = pd.date_range('2000-01-01', freq='6H', periods=10)
        array = DataArray(np.arange(10), [('time', times)])

        actual = array.resample(time='1D').first()
        expected = DataArray([0, 4, 8], [('time', times[::4])])
        self.assertDataArrayIdentical(expected, actual)

        # verify that labels don't use the first value
        actual = array.resample(time='24H').first()
        expected = DataArray(array.to_series().resample('24H').first())
        self.assertDataArrayIdentical(expected, actual)

        # missing values
        array = array.astype(float)
        array[:2] = np.nan
        actual = array.resample(time='1D').first()
        expected = DataArray([2, 4, 8], [('time', times[::4])])
        self.assertDataArrayIdentical(expected, actual)

        actual = array.resample(time='1D').first(skipna=False)
        expected = DataArray([np.nan, 4, 8], [('time', times[::4])])
        self.assertDataArrayIdentical(expected, actual)

        # regression test for http://stackoverflow.com/questions/33158558/
        array = Dataset({'time': times})['time']
        actual = array.resample(time='1D').last()
        expected_times = pd.to_datetime(['2000-01-01T18', '2000-01-02T18',
                                         '2000-01-03T06'])
        expected = DataArray(expected_times, [('time', times[::4])],
                             name='time')
        self.assertDataArrayIdentical(expected, actual)

    def test_resample_bad_resample_dim(self):
        times = pd.date_range('2000-01-01', freq='6H', periods=10)
        array = DataArray(np.arange(10), [('__resample_dim__', times)])
        with raises_regex(ValueError, 'Proxy resampling dimension'):
            array.resample(**{'__resample_dim__': '1D'}).first()

    @requires_scipy
    def test_resample_drop_nondim_coords(self):
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range('2000-01-01', freq='6H', periods=5)
        data = np.tile(np.arange(5), (6, 3, 1))
        xx, yy = np.meshgrid(xs*5, ys*2.5)
        tt = np.arange(len(times), dtype=int)
        array = DataArray(data,
                          {'time': times, 'x': xs, 'y': ys},
                          ('x', 'y', 'time'))
        xcoord = DataArray(xx.T, {'x': xs, 'y': ys}, ('x', 'y'))
        ycoord = DataArray(yy.T, {'x': xs, 'y': ys}, ('x', 'y'))
        tcoord = DataArray(tt, {'time': times}, ('time', ))
        ds = Dataset({'data': array, 'xc': xcoord,
                      'yc': ycoord, 'tc': tcoord})
        ds = ds.set_coords(['xc', 'yc', 'tc'])

        # Select the data now, with the auxiliary coordinates in place
        array = ds['data']

        # Re-sample
        actual = array.resample(time="12H").mean('time')
        assert 'tc' not in actual.coords

        # Up-sample - filling
        actual = array.resample(time="1H").ffill()
        assert 'tc' not in actual.coords

        # Up-sample - interpolation
        actual = array.resample(time="1H").interpolate('linear')
        assert 'tc' not in actual.coords

    def test_resample_old_vs_new_api(self):
        times = pd.date_range('2000-01-01', freq='6H', periods=10)
        array = DataArray(np.ones(10), [('time', times)])

        # Simple mean
        with pytest.warns(DeprecationWarning):
            old_mean = array.resample('1D', 'time', how='mean')
        new_mean = array.resample(time='1D').mean()
        self.assertDataArrayIdentical(old_mean, new_mean)

        # Mean, while keeping attributes
        attr_array = array.copy()
        attr_array.attrs['meta'] = 'data'

        with pytest.warns(DeprecationWarning):
            old_mean = attr_array.resample('1D', dim='time', how='mean',
                                           keep_attrs=True)
        new_mean = attr_array.resample(time='1D').mean(keep_attrs=True)
        self.assertEqual(old_mean.attrs, new_mean.attrs)
        self.assertDatasetIdentical(old_mean, new_mean)

        # Mean, with NaN to skip
        nan_array = array.copy()
        nan_array[1] = np.nan

        with pytest.warns(DeprecationWarning):
            old_mean = nan_array.resample('1D', 'time', how='mean',
                                          skipna=False)
        new_mean = nan_array.resample(time='1D').mean(skipna=False)
        expected = DataArray([np.nan, 1, 1], [('time', times[::4])])
        self.assertDataArrayIdentical(old_mean, expected)
        self.assertDataArrayIdentical(new_mean, expected)

        # Try other common resampling methods
        resampler = array.resample(time='1D')
        for method in ['mean', 'median', 'sum', 'first', 'last', 'count']:
            # Discard attributes on the call using the new api to match
            # convention from old api
            new_api = getattr(resampler, method)(keep_attrs=False)
            with pytest.warns(DeprecationWarning):
                old_api = array.resample('1D', dim='time', how=method)
            self.assertDatasetIdentical(new_api, old_api)
        for method in [np.mean, np.sum, np.max, np.min]:
            new_api = resampler.reduce(method)
            with pytest.warns(DeprecationWarning):
                old_api = array.resample('1D', dim='time', how=method)
            self.assertDatasetIdentical(new_api, old_api)

    def test_upsample(self):
        times = pd.date_range('2000-01-01', freq='6H', periods=5)
        array = DataArray(np.arange(5), [('time', times)])

        # Forward-fill
        actual = array.resample(time='3H').ffill()
        expected = DataArray(array.to_series().resample('3H').ffill())
        self.assertDataArrayIdentical(expected, actual)

        # Backward-fill
        actual = array.resample(time='3H').bfill()
        expected = DataArray(array.to_series().resample('3H').bfill())
        self.assertDataArrayIdentical(expected, actual)

        # As frequency
        actual = array.resample(time='3H').asfreq()
        expected = DataArray(array.to_series().resample('3H').asfreq())
        self.assertDataArrayIdentical(expected, actual)

        # Pad
        actual = array.resample(time='3H').pad()
        expected = DataArray(array.to_series().resample('3H').pad())
        self.assertDataArrayIdentical(expected, actual)

        # Nearest
        rs = array.resample(time='3H')
        actual = rs.nearest()
        new_times = rs._full_index
        expected = DataArray(
            array.reindex(time=new_times, method='nearest')
        )
        self.assertDataArrayIdentical(expected, actual)

    def test_upsample_nd(self):
        # Same as before, but now we try on multi-dimensional DataArrays.
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range('2000-01-01', freq='6H', periods=5)
        data = np.tile(np.arange(5), (6, 3, 1))
        array = DataArray(data,
                          {'time': times, 'x': xs, 'y': ys},
                          ('x', 'y', 'time'))

        # Forward-fill
        actual = array.resample(time='3H').ffill()
        expected_data = np.repeat(data, 2, axis=-1)
        expected_times = times.to_series().resample('3H').asfreq().index
        expected_data = expected_data[..., :len(expected_times)]
        expected = DataArray(expected_data,
                             {'time': expected_times, 'x': xs, 'y': ys},
                             ('x', 'y', 'time'))
        self.assertDataArrayIdentical(expected, actual)

        # Backward-fill
        actual = array.resample(time='3H').ffill()
        expected_data = np.repeat(np.flipud(data.T).T, 2, axis=-1)
        expected_data = np.flipud(expected_data.T).T
        expected_times = times.to_series().resample('3H').asfreq().index
        expected_data = expected_data[..., :len(expected_times)]
        expected = DataArray(expected_data,
                             {'time': expected_times, 'x': xs, 'y': ys},
                             ('x', 'y', 'time'))
        self.assertDataArrayIdentical(expected, actual)

        # As frequency
        actual = array.resample(time='3H').asfreq()
        expected_data = np.repeat(data, 2, axis=-1).astype(float)[..., :-1]
        expected_data[..., 1::2] = np.nan
        expected_times = times.to_series().resample('3H').asfreq().index
        expected = DataArray(expected_data,
                             {'time': expected_times, 'x': xs, 'y': ys},
                             ('x', 'y', 'time'))
        self.assertDataArrayIdentical(expected, actual)

        # Pad
        actual = array.resample(time='3H').pad()
        expected_data = np.repeat(data, 2, axis=-1)
        expected_data[..., 1::2] = expected_data[..., ::2]
        expected_data = expected_data[..., :-1]
        expected_times = times.to_series().resample('3H').asfreq().index
        expected = DataArray(expected_data,
                             {'time': expected_times, 'x': xs, 'y': ys},
                             ('x', 'y', 'time'))
        self.assertDataArrayIdentical(expected, actual)

    @requires_scipy
    def test_upsample_interpolate(self):
        from scipy.interpolate import interp1d
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range('2000-01-01', freq='6H', periods=5)

        z = np.arange(5)**2
        data = np.tile(z, (6, 3, 1))
        array = DataArray(data,
                          {'time': times, 'x': xs, 'y': ys},
                          ('x', 'y', 'time'))

        expected_times = times.to_series().resample('1H').asfreq().index
        # Split the times into equal sub-intervals to simulate the 6 hour
        # to 1 hour up-sampling
        new_times_idx = np.linspace(0, len(times)-1, len(times)*5)
        for kind in ['linear', 'nearest', 'zero', 'slinear', 'quadratic',
                     'cubic']:
            actual = array.resample(time='1H').interpolate(kind)
            f = interp1d(np.arange(len(times)), data, kind=kind, axis=-1,
                         bounds_error=True, assume_sorted=True)
            expected_data = f(new_times_idx)
            expected = DataArray(expected_data,
                                 {'time': expected_times, 'x': xs, 'y': ys},
                                 ('x', 'y', 'time'))
            # Use AllClose because there are some small differences in how
            # we upsample timeseries versus the integer indexing as I've
            # done here due to floating point arithmetic
            self.assertDataArrayAllClose(expected, actual, rtol=1e-16)

    @requires_scipy
    def test_upsample_interpolate_regression_1605(self):
        dates = pd.date_range('2016-01-01', '2016-03-31', freq='1D')
        expected = xr.DataArray(np.random.random((len(dates), 2, 3)),
                                dims=('time', 'x', 'y'),
                                coords={'time': dates})
        actual = expected.resample(time='1D').interpolate('linear')
        assert_allclose(actual, expected, rtol=1e-16)

    @requires_dask
    def test_upsample_interpolate_dask(self):
        import dask.array as da

        times = pd.date_range('2000-01-01', freq='6H', periods=5)
        xs = np.arange(6)
        ys = np.arange(3)

        z = np.arange(5)**2
        data = da.from_array(np.tile(z, (6, 3, 1)), (1, 3, 1))
        array = DataArray(data,
                          {'time': times, 'x': xs, 'y': ys},
                          ('x', 'y', 'time'))

        with raises_regex(TypeError,
                                     "dask arrays are not yet supported"):
            array.resample(time='1H').interpolate('linear')

    def test_align(self):
        array = DataArray(np.random.random((6, 8)),
                          coords={'x': list('abcdef')}, dims=['x', 'y'])
        array1, array2 = align(array, array[:5], join='inner')
        self.assertDataArrayIdentical(array1, array[:5])
        self.assertDataArrayIdentical(array2, array[:5])

    def test_align_dtype(self):
        # regression test for #264
        x1 = np.arange(30)
        x2 = np.arange(5, 35)
        a = DataArray(np.random.random((30,)).astype(np.float32), [('x', x1)])
        b = DataArray(np.random.random((30,)).astype(np.float32), [('x', x2)])
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
        x = DataArray([[1, 2], [3, 4]],
                      coords=[('a', [-1, -2]), ('b', [3, 4])])
        y = DataArray([[1, 2], [3, 4]],
                      coords=[('a', [-1, 20]), ('b', [5, 6])])
        z = DataArray([1], dims=['a'], coords={'a': [20], 'b': 7})

        x2, y2, z2 = align(x, y, z, join='outer', exclude=['b'])
        expected_x2 = DataArray([[3, 4], [1, 2], [np.nan, np.nan]],
                                coords=[('a', [-2, -1, 20]), ('b', [3, 4])])
        expected_y2 = DataArray([[np.nan, np.nan], [1, 2], [3, 4]],
                                coords=[('a', [-2, -1, 20]), ('b', [5, 6])])
        expected_z2 = DataArray([np.nan, np.nan, 1], dims=['a'],
                                coords={'a': [-2, -1, 20], 'b': 7})
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
        expected_x2 = DataArray([3, np.nan, 2, 1],
                                coords=[('a', [-2, 7, 10, -1])])
        self.assertDataArrayIdentical(expected_x2, x2)

    def test_align_without_indexes_exclude(self):
        arrays = [DataArray([1, 2, 3], dims=['x']),
                  DataArray([1, 2], dims=['x'])]
        result0, result1 = align(*arrays, exclude=['x'])
        self.assertDatasetIdentical(result0, arrays[0])
        self.assertDatasetIdentical(result1, arrays[1])

    def test_align_mixed_indexes(self):
        array_no_coord = DataArray([1, 2], dims=['x'])
        array_with_coord = DataArray([1, 2], coords=[('x', ['a', 'b'])])
        result0, result1 = align(array_no_coord, array_with_coord)
        self.assertDatasetIdentical(result0, array_with_coord)
        self.assertDatasetIdentical(result1, array_with_coord)

        result0, result1 = align(array_no_coord, array_with_coord,
                                 exclude=['x'])
        self.assertDatasetIdentical(result0, array_no_coord)
        self.assertDatasetIdentical(result1, array_with_coord)

    def test_align_without_indexes_errors(self):
        with raises_regex(ValueError, 'cannot be aligned'):
            align(DataArray([1, 2, 3], dims=['x']),
                  DataArray([1, 2], dims=['x']))

        with raises_regex(ValueError, 'cannot be aligned'):
            align(DataArray([1, 2, 3], dims=['x']),
                  DataArray([1, 2], coords=[('x', [0, 1])]))

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
        x = DataArray([[1, 2], [3, 4]],
                      coords=[('a', [-1, -2]), ('b', [3, 4])])
        y = DataArray([1, 2], coords=[('a', [-1, 20])])
        expected_x2 = DataArray([[3, 4], [1, 2], [np.nan, np.nan]],
                                coords=[('a', [-2, -1, 20]), ('b', [3, 4])])
        expected_y2 = DataArray([[np.nan, np.nan], [1, 1], [2, 2]],
                                coords=[('a', [-2, -1, 20]), ('b', [3, 4])])
        x2, y2 = broadcast(x, y)
        self.assertDataArrayIdentical(expected_x2, x2)
        self.assertDataArrayIdentical(expected_y2, y2)

    def test_broadcast_arrays_nocopy(self):
        # Test that input data is not copied over in case
        # no alteration is needed
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
        x = DataArray([[1, 2], [3, 4]],
                      coords=[('a', [-1, -2]), ('b', [3, 4])])
        y = DataArray([1, 2], coords=[('a', [-1, 20])])
        z = DataArray(5, coords={'b': 5})

        x2, y2, z2 = broadcast(x, y, z, exclude=['b'])
        expected_x2 = DataArray([[3, 4], [1, 2], [np.nan, np.nan]],
                                coords=[('a', [-2, -1, 20]), ('b', [3, 4])])
        expected_y2 = DataArray([np.nan, 1, 2], coords=[('a', [-2, -1, 20])])
        expected_z2 = DataArray([5, 5, 5], dims=['a'],
                                coords={'a': [-2, -1, 20], 'b': 5})
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
            roundtripped = DataArray(da.to_pandas()).drop(dims)
            self.assertDataArrayIdentical(da, roundtripped)

        with raises_regex(ValueError, 'cannot convert'):
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
        with raises_regex(ValueError, 'unnamed'):
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
        self.assertDataArrayIdentical(
            self.dv,
            DataArray.from_series(actual).drop(['x', 'y']))
        # test name is None
        actual.name = None
        expected_da = self.dv.rename(None)
        self.assertDataArrayIdentical(
            expected_da,
            DataArray.from_series(actual).drop(['x', 'y']))

    def test_series_categorical_index(self):
        # regression test for GH700
        if not hasattr(pd, 'CategoricalIndex'):
            raise unittest.SkipTest('requires pandas with CategoricalIndex')

        s = pd.Series(range(5), index=pd.CategoricalIndex(list('aabbc')))
        arr = DataArray(s)
        assert "'a'" in repr(arr)  # should not error

    def test_to_and_from_dict(self):
        array = DataArray(np.random.randn(2, 3), {'x': ['a', 'b']}, ['x', 'y'],
                          name='foo')
        expected = {'name': 'foo',
                    'dims': ('x', 'y'),
                    'data': array.values.tolist(),
                    'attrs': {},
                    'coords': {'x': {'dims': ('x',),
                                     'data': ['a', 'b'],
                                     'attrs': {}}}}
        actual = array.to_dict()

        # check that they are identical
        self.assertEqual(expected, actual)

        # check roundtrip
        self.assertDataArrayIdentical(array, DataArray.from_dict(actual))

        # a more bare bones representation still roundtrips
        d = {'name': 'foo',
             'dims': ('x', 'y'),
             'data': array.values.tolist(),
             'coords': {'x': {'dims': 'x', 'data': ['a', 'b']}}}
        self.assertDataArrayIdentical(array, DataArray.from_dict(d))

        # and the most bare bones representation still roundtrips
        d = {'name': 'foo', 'dims': ('x', 'y'), 'data': array.values}
        self.assertDataArrayIdentical(array.drop('x'), DataArray.from_dict(d))

        # missing a dims in the coords
        d = {'dims': ('x', 'y'),
             'data': array.values,
             'coords': {'x': {'data': ['a', 'b']}}}
        with raises_regex(
                ValueError,
                "cannot convert dict when coords are missing the key 'dims'"):
            DataArray.from_dict(d)

        # this one is missing some necessary information
        d = {'dims': ('t')}
        with raises_regex(
                ValueError, "cannot convert dict without the key 'data'"):
            DataArray.from_dict(d)

    def test_to_and_from_dict_with_time_dim(self):
        x = np.random.randn(10, 3)
        t = pd.date_range('20130101', periods=10)
        lat = [77.7, 83.2, 76]
        da = DataArray(x, {'t': t, 'lat': lat}, dims=['t', 'lat'])
        roundtripped = DataArray.from_dict(da.to_dict())
        self.assertDataArrayIdentical(da, roundtripped)

    def test_to_and_from_dict_with_nan_nat(self):
        y = np.random.randn(10, 3)
        y[2] = np.nan
        t = pd.Series(pd.date_range('20130101', periods=10))
        t[2] = np.nan
        lat = [77.7, 83.2, 76]
        da = DataArray(y, {'t': t, 'lat': lat}, dims=['t', 'lat'])
        roundtripped = DataArray.from_dict(da.to_dict())
        self.assertDataArrayIdentical(da, roundtripped)

    def test_to_dict_with_numpy_attrs(self):
        # this doesn't need to roundtrip
        x = np.random.randn(10, 3)
        t = list('abcdefghij')
        lat = [77.7, 83.2, 76]
        attrs = {'created': np.float64(1998),
                 'coords': np.array([37, -110.1, 100]),
                 'maintainer': 'bar'}
        da = DataArray(x, {'t': t, 'lat': lat}, dims=['t', 'lat'],
                       attrs=attrs)
        expected_attrs = {'created': np.asscalar(attrs['created']),
                          'coords': attrs['coords'].tolist(),
                          'maintainer': 'bar'}
        actual = da.to_dict()

        # check that they are identical
        self.assertEqual(expected_attrs, actual['attrs'])

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
        pytest.importorskip('cdms2')

        original = DataArray(
            np.arange(6).reshape(2, 3),
            [('distance', [-2, 2], {'units': 'meters'}),
             ('time', pd.date_range('2000-01-01', periods=3))],
            name='foo', attrs={'baz': 123})
        expected_coords = [IndexVariable('distance', [-2, 2]),
                           IndexVariable('time', [0, 1, 2])]
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
        with raises_regex(ValueError, 'unable to convert unnamed'):
            unnamed.to_dataset()

        actual = unnamed.to_dataset(name='foo')
        expected = Dataset({'foo': ('x', [1, 2])})
        self.assertDatasetIdentical(expected, actual)

        named = DataArray([1, 2], dims='x', name='foo')
        actual = named.to_dataset()
        expected = Dataset({'foo': ('x', [1, 2])})
        self.assertDatasetIdentical(expected, actual)

        expected = Dataset({'bar': ('x', [1, 2])})
        with pytest.warns(FutureWarning):
            actual = named.to_dataset('bar')
        self.assertDatasetIdentical(expected, actual)

    def test_to_dataset_split(self):
        array = DataArray([1, 2, 3], coords=[('x', list('abc'))],
                          attrs={'a': 1})
        expected = Dataset(OrderedDict([('a', 1), ('b', 2), ('c', 3)]),
                           attrs={'a': 1})
        actual = array.to_dataset('x')
        self.assertDatasetIdentical(expected, actual)

        with pytest.raises(TypeError):
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
        dates = [datetime.date(2000, 1, d) for d in range(1, 4)]

        array = DataArray([1, 2, 3], coords=[('x', dates)],
                          attrs={'a': 1})

        # convert to dateset and back again
        result = array.to_dataset('x').to_array(dim='x')

        self.assertDatasetEqual(array, result)

    def test__title_for_slice(self):
        array = DataArray(np.ones((4, 3, 2)), dims=['a', 'b', 'c'],
                          coords={'a': range(4), 'b': range(3), 'c': range(2)})
        self.assertEqual('', array._title_for_slice())
        self.assertEqual('c = 0', array.isel(c=0)._title_for_slice())
        title = array.isel(b=1, c=0)._title_for_slice()
        self.assertTrue('b = 1, c = 0' == title or 'c = 0, b = 1' == title)

        a2 = DataArray(np.ones((4, 1)), dims=['a', 'b'])
        self.assertEqual('', a2._title_for_slice())

    def test__title_for_slice_truncate(self):
        array = DataArray(np.ones((4)))
        array.coords['a'] = 'a' * 100
        array.coords['b'] = 'b' * 100

        nchar = 80
        title = array._title_for_slice(truncate=nchar)

        self.assertEqual(nchar, len(title))
        self.assertTrue(title.endswith('...'))

    def test_dataarray_diff_n1(self):
        da = DataArray(np.random.randn(3, 4), dims=['x', 'y'])
        actual = da.diff('y')
        expected = DataArray(np.diff(da.values, axis=1), dims=['x', 'y'])
        self.assertDataArrayEqual(expected, actual)

    def test_coordinate_diff(self):
        # regression test for GH634
        arr = DataArray(range(0, 20, 2), dims=['lon'], coords=[range(10)])
        lon = arr.coords['lon']
        expected = DataArray([1] * 9, dims=['lon'], coords=[range(1, 10)],
                             name='lon')
        actual = lon.diff('lon')
        self.assertDataArrayEqual(expected, actual)

    def test_shift(self):
        arr = DataArray([1, 2, 3], dims='x')
        actual = arr.shift(x=1)
        expected = DataArray([np.nan, 1, 2], dims='x')
        self.assertDataArrayIdentical(expected, actual)

        arr = DataArray([1, 2, 3], [('x', ['a', 'b', 'c'])])
        for offset in [-5, -2, -1, 0, 1, 2, 5]:
            expected = DataArray(arr.to_pandas().shift(offset))
            actual = arr.shift(x=offset)
            self.assertDataArrayIdentical(expected, actual)

    def test_roll(self):
        arr = DataArray([1, 2, 3], coords={'x': range(3)}, dims='x')
        actual = arr.roll(x=1)
        expected = DataArray([3, 1, 2], coords=[('x', [2, 0, 1])])
        self.assertDataArrayIdentical(expected, actual)

    def test_real_and_imag(self):
        array = DataArray(1 + 2j)
        self.assertDataArrayIdentical(array.real, DataArray(1))
        self.assertDataArrayIdentical(array.imag, DataArray(2))

    def test_setattr_raises(self):
        array = DataArray(0, coords={'scalar': 1}, attrs={'foo': 'bar'})
        with raises_regex(AttributeError, 'cannot set attr'):
            array.scalar = 2
        with raises_regex(AttributeError, 'cannot set attr'):
            array.foo = 2
        with raises_regex(AttributeError, 'cannot set attr'):
            array.other = 2

    def test_full_like(self):
        # For more thorough tests, see test_variable.py
        da = DataArray(np.random.random(size=(2, 2)),
                       dims=('x', 'y'),
                       attrs={'attr1': 'value1'},
                       coords={'x': [4, 3]},
                       name='helloworld')

        actual = full_like(da, 2)
        expect = da.copy(deep=True)
        expect.values = [[2.0, 2.0], [2.0, 2.0]]
        self.assertDataArrayIdentical(expect, actual)

        # override dtype
        actual = full_like(da, fill_value=True, dtype=bool)
        expect.values = [[True, True], [True, True]]
        assert expect.dtype == bool
        self.assertDataArrayIdentical(expect, actual)

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
        expected_vals = np.tensordot(da_vals, da_vals,
                                     axes=([0, 1, 2], [0, 1, 2]))
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

        with pytest.raises(NotImplementedError):
            da.dot(dm.to_dataset(name='dm'))
        with pytest.raises(TypeError):
            da.dot(dm.values)
        with raises_regex(ValueError, 'no shared dimensions'):
            da.dot(DataArray(1))

    def test_binary_op_join_setting(self):
        dim = 'x'
        align_type = "outer"
        coords_l, coords_r = [0, 1, 2], [1, 2, 3]
        missing_3 = xr.DataArray(coords_l, [(dim, coords_l)])
        missing_0 = xr.DataArray(coords_r, [(dim, coords_r)])
        with xr.set_options(arithmetic_join=align_type):
            actual = missing_0 + missing_3
        missing_0_aligned, missing_3_aligned = xr.align(missing_0,
                                                        missing_3,
                                                        join=align_type)
        expected = xr.DataArray([np.nan, 2, 4, np.nan], [(dim, [0, 1, 2, 3])])
        self.assertDataArrayEqual(actual, expected)

    def test_combine_first(self):
        ar0 = DataArray([[0, 0], [0, 0]], [('x', ['a', 'b']), ('y', [-1, 0])])
        ar1 = DataArray([[1, 1], [1, 1]], [('x', ['b', 'c']), ('y', [0, 1])])
        ar2 = DataArray([2], [('x', ['d'])])

        actual = ar0.combine_first(ar1)
        expected = DataArray([[0, 0, np.nan], [0, 0, 1], [np.nan, 1, 1]],
                             [('x', ['a', 'b', 'c']), ('y', [-1, 0, 1])])
        self.assertDataArrayEqual(actual, expected)

        actual = ar1.combine_first(ar0)
        expected = DataArray([[0, 0, np.nan], [0, 1, 1], [np.nan, 1, 1]],
                             [('x', ['a', 'b', 'c']), ('y', [-1, 0, 1])])
        self.assertDataArrayEqual(actual, expected)

        actual = ar0.combine_first(ar2)
        expected = DataArray([[0, 0], [0, 0], [2, 2]],
                             [('x', ['a', 'b', 'd']), ('y', [-1, 0])])
        self.assertDataArrayEqual(actual, expected)

    def test_sortby(self):
        da = DataArray([[1, 2], [3, 4], [5, 6]],
                       [('x', ['c', 'b', 'a']), ('y', [1, 0])])

        sorted1d = DataArray([[5, 6], [3, 4], [1, 2]],
                             [('x', ['a', 'b', 'c']), ('y', [1, 0])])

        sorted2d = DataArray([[6, 5], [4, 3], [2, 1]],
                             [('x', ['a', 'b', 'c']), ('y', [0, 1])])

        expected = sorted1d
        dax = DataArray([100, 99, 98], [('x', ['c', 'b', 'a'])])
        actual = da.sortby(dax)
        self.assertDatasetEqual(actual, expected)

        # test descending order sort
        actual = da.sortby(dax, ascending=False)
        self.assertDatasetEqual(actual, da)

        # test alignment (fills in nan for 'c')
        dax_short = DataArray([98, 97], [('x', ['b', 'a'])])
        actual = da.sortby(dax_short)
        self.assertDatasetEqual(actual, expected)

        # test multi-dim sort by 1D dataarray values
        expected = sorted2d
        dax = DataArray([100, 99, 98], [('x', ['c', 'b', 'a'])])
        day = DataArray([90, 80], [('y', [1, 0])])
        actual = da.sortby([day, dax])
        self.assertDataArrayEqual(actual, expected)

        if LooseVersion(np.__version__) < LooseVersion('1.11.0'):
            pytest.skip('numpy 1.11.0 or later to support object data-type.')

        expected = sorted1d
        actual = da.sortby('x')
        self.assertDataArrayEqual(actual, expected)

        expected = sorted2d
        actual = da.sortby(['x', 'y'])
        self.assertDataArrayEqual(actual, expected)


@pytest.fixture(params=[1])
def da(request):
    if request.param == 1:
        times = pd.date_range('2000-01-01', freq='1D', periods=21)
        values = np.random.random((3, 21, 4))
        da = DataArray(values, dims=('a', 'time', 'x'))
        da['time'] = times
        return da

    if request.param == 2:
        return DataArray(
            [0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7],
            dims='time')


@pytest.fixture
def da_dask(seed=123):
    pytest.importorskip('dask.array')
    rs = np.random.RandomState(seed)
    times = pd.date_range('2000-01-01', freq='1D', periods=21)
    values = rs.normal(size=(1, 21, 1))
    da = DataArray(values, dims=('a', 'time', 'x')).chunk({'time': 7})
    da['time'] = times
    return da


def test_rolling_iter(da):

    rolling_obj = da.rolling(time=7)

    assert len(rolling_obj.window_labels) == len(da['time'])
    assert_identical(rolling_obj.window_labels, da['time'])

    for i, (label, window_da) in enumerate(rolling_obj):
        assert label == da['time'].isel(time=i)


def test_rolling_doc(da):
    rolling_obj = da.rolling(time=7)

    # argument substitution worked
    assert '`mean`' in rolling_obj.mean.__doc__


def test_rolling_properties(da):
    rolling_obj = da.rolling(time=4)

    assert rolling_obj.obj.get_axis_num('time') == 1

    # catching invalid args
    with pytest.raises(ValueError) as exception:
        da.rolling(time=7, x=2)
    assert 'exactly one dim/window should' in str(exception)
    with pytest.raises(ValueError) as exception:
        da.rolling(time=-2)
    assert 'window must be > 0' in str(exception)
    with pytest.raises(ValueError) as exception:
        da.rolling(time=2, min_periods=0)
    assert 'min_periods must be greater than zero' in str(exception)


@pytest.mark.parametrize('name', ('sum', 'mean', 'std', 'min', 'max',
                                  'median'))
@pytest.mark.parametrize('center', (True, False, None))
@pytest.mark.parametrize('min_periods', (1, None))
def test_rolling_wrapped_bottleneck(da, name, center, min_periods):
    bn = pytest.importorskip('bottleneck', minversion="1.1")

    # Test all bottleneck functions
    rolling_obj = da.rolling(time=7, min_periods=min_periods)

    func_name = 'move_{0}'.format(name)
    actual = getattr(rolling_obj, name)()
    expected = getattr(bn, func_name)(da.values, window=7, axis=1,
                                      min_count=min_periods)
    assert_array_equal(actual.values, expected)

    # Test center
    rolling_obj = da.rolling(time=7, center=center)
    actual = getattr(rolling_obj, name)()['time']
    assert_equal(actual, da['time'])


@pytest.mark.parametrize('name', ('sum', 'mean', 'std', 'min', 'max',
                                  'median'))
@pytest.mark.parametrize('center', (True, False, None))
@pytest.mark.parametrize('min_periods', (1, None))
def test_rolling_wrapped_bottleneck_dask(da_dask, name, center, min_periods):
    pytest.importorskip('dask.array')
    # dask version
    rolling_obj = da_dask.rolling(time=7, min_periods=min_periods)
    actual = getattr(rolling_obj, name)().load()
    # numpy version
    rolling_obj = da_dask.load().rolling(time=7, min_periods=min_periods)
    expected = getattr(rolling_obj, name)()

    # using all-close because rolling over ghost cells introduces some
    # precision errors
    assert_allclose(actual, expected)


@pytest.mark.parametrize('center', (True, False))
@pytest.mark.parametrize('min_periods', (None, 1, 2, 3))
@pytest.mark.parametrize('window', (1, 2, 3, 4))
def test_rolling_pandas_compat(da, center, window, min_periods):
    s = pd.Series(range(10))
    da = DataArray.from_series(s)

    if min_periods is not None and window < min_periods:
        min_periods = window

    s_rolling = s.rolling(window, center=center,
                          min_periods=min_periods).mean()
    da_rolling = da.rolling(index=window, center=center,
                            min_periods=min_periods).mean()
    # pandas does some fancy stuff in the last position,
    # we're not going to do that yet!
    np.testing.assert_allclose(s_rolling.values[:-1],
                               da_rolling.values[:-1])
    np.testing.assert_allclose(s_rolling.index,
                               da_rolling['index'])


@pytest.mark.parametrize('da', (1, 2), indirect=True)
@pytest.mark.parametrize('center', (True, False))
@pytest.mark.parametrize('min_periods', (None, 1, 2, 3))
@pytest.mark.parametrize('window', (1, 2, 3, 4))
@pytest.mark.parametrize('name', ('sum', 'mean', 'std', 'max'))
def test_rolling_reduce(da, center, min_periods, window, name):

    if min_periods is not None and window < min_periods:
        min_periods = window

    rolling_obj = da.rolling(time=window, center=center,
                             min_periods=min_periods)

    # add nan prefix to numpy methods to get similar # behavior as bottleneck
    actual = rolling_obj.reduce(getattr(np, 'nan%s' % name))
    expected = getattr(rolling_obj, name)()
    assert_allclose(actual, expected)
    assert actual.dims == expected.dims


def test_rolling_count_correct():

    da = DataArray(
        [0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7], dims='time')

    result = da.rolling(time=11, min_periods=1).count()
    expected = DataArray(
        [1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8], dims='time')
    assert_equal(result, expected)

    result = da.rolling(time=11, min_periods=None).count()
    expected = DataArray(
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, np.nan, np.nan, np.nan, 8], dims='time')
    assert_equal(result, expected)

    result = da.rolling(time=7, min_periods=2).count()
    expected = DataArray(
        [np.nan, np.nan, 2, 3, 3, 4, 5, 5, 5, 5, 5], dims='time')
    assert_equal(result, expected)


def test_raise_no_warning_for_nan_in_binary_ops():
    with pytest.warns(None) as record:
        xr.DataArray([1, 2, np.NaN]) > 0
    assert len(record) == 0
