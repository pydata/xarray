from copy import copy, deepcopy
from textwrap import dedent
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    import dask.array as da
except ImportError:
    pass

import numpy as np
import pandas as pd

from xarray import (align, broadcast, concat, conventions, backends, Dataset,
                    DataArray, Variable, Coordinate, auto_combine,
                    open_dataset, set_options)
from xarray.core import indexing, utils
from xarray.core.pycompat import iteritems, OrderedDict

from . import (TestCase, unittest, InaccessibleArray, UnexpectedDataAccess,
               requires_dask)


def create_test_data(seed=None):
    rs = np.random.RandomState(seed)
    _vars = {'var1': ['dim1', 'dim2'],
             'var2': ['dim1', 'dim2'],
             'var3': ['dim3', 'dim1']}
    _dims = {'dim1': 8, 'dim2': 9, 'dim3': 10}

    obj = Dataset()
    obj['time'] = ('time', pd.date_range('2000-01-01', periods=20))
    obj['dim1'] = ('dim1', np.arange(_dims['dim1'], dtype='int64'))
    obj['dim2'] = ('dim2', 0.5 * np.arange(_dims['dim2']))
    obj['dim3'] = ('dim3', list('abcdefghij'))
    for v, dims in sorted(_vars.items()):
        data = rs.normal(size=tuple(_dims[d] for d in dims))
        obj[v] = (dims, data, {'foo': 'variable'})
    obj.coords['numbers'] = ('dim3', np.array([0, 1, 2, 0, 0, 1, 1, 2, 2, 3],
                                              dtype='int64'))
    return obj


class InaccessibleVariableDataStore(backends.InMemoryDataStore):
    def get_variables(self):
        def lazy_inaccessible(x):
            data = indexing.LazilyIndexedArray(InaccessibleArray(x.values))
            return Variable(x.dims, data, x.attrs)
        return dict((k, lazy_inaccessible(v)) for
                    k, v in iteritems(self._variables))


class TestDataset(TestCase):
    def test_repr(self):
        data = create_test_data(seed=123)
        data.attrs['foo'] = 'bar'
        # need to insert str dtype at runtime to handle both Python 2 & 3
        expected = dedent("""\
        <xarray.Dataset>
        Dimensions:  (dim1: 8, dim2: 9, dim3: 10, time: 20)
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03 ...
          * dim1     (dim1) int64 0 1 2 3 4 5 6 7
          * dim2     (dim2) float64 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0
          * dim3     (dim3) %s 'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j'
            numbers  (dim3) int64 0 1 2 0 0 1 1 2 2 3
        Data variables:
            var1     (dim1, dim2) float64 -1.086 0.9973 0.283 -1.506 -0.5786 1.651 ...
            var2     (dim1, dim2) float64 1.162 -1.097 -2.123 1.04 -0.4034 -0.126 ...
            var3     (dim3, dim1) float64 0.5565 -0.2121 0.4563 1.545 -0.2397 0.1433 ...
        Attributes:
            foo: bar""") % data['dim3'].dtype
        actual = '\n'.join(x.rstrip() for x in repr(data).split('\n'))
        print(actual)
        self.assertEqual(expected, actual)

        with set_options(display_width=100):
            max_len = max(map(len, repr(data).split('\n')))
            assert 90 < max_len < 100

        expected = dedent("""\
        <xarray.Dataset>
        Dimensions:  ()
        Coordinates:
            *empty*
        Data variables:
            *empty*""")
        actual = '\n'.join(x.rstrip() for x in repr(Dataset()).split('\n'))
        print(actual)
        self.assertEqual(expected, actual)

        # verify that ... doesn't appear for scalar coordinates
        data = Dataset({'foo': ('x', np.ones(10))}).mean()
        expected = dedent("""\
        <xarray.Dataset>
        Dimensions:  ()
        Coordinates:
            *empty*
        Data variables:
            foo      float64 1.0""")
        actual = '\n'.join(x.rstrip() for x in repr(data).split('\n'))
        print(actual)
        self.assertEqual(expected, actual)

        # verify long attributes are truncated
        data = Dataset(attrs={'foo': 'bar' * 1000})
        self.assertTrue(len(repr(data)) < 1000)

    def test_repr_period_index(self):
        data = create_test_data(seed=456)
        data.coords['time'] = pd.period_range('2000-01-01', periods=20, freq='B')

        # check that creating the repr doesn't raise an error #GH645
        repr(data)

    def test_constructor(self):
        x1 = ('x', 2 * np.arange(100))
        x2 = ('x', np.arange(1000))
        z = (['x', 'y'], np.arange(1000).reshape(100, 10))

        with self.assertRaisesRegexp(ValueError, 'conflicting sizes'):
            Dataset({'a': x1, 'b': x2})
        with self.assertRaisesRegexp(ValueError,
                "variable 'x' has the same name"):
            Dataset({'a': x1, 'x': z})
        with self.assertRaisesRegexp(TypeError, 'must be given by arrays or'):
            Dataset({'x': (1, 2, 3, 4, 5, 6, 7)})
        with self.assertRaisesRegexp(ValueError, 'already exists as a scalar'):
            Dataset({'x': 0, 'y': ('x', [1, 2, 3])})

        # verify handling of DataArrays
        expected = Dataset({'x': x1, 'z': z})
        actual = Dataset({'z': expected['z']})
        self.assertDatasetIdentical(expected, actual)

    def test_constructor_kwargs(self):
        x1 = ('x', 2 * np.arange(100))

        with self.assertRaises(TypeError):
            Dataset(data_vars={'x1': x1}, invalid_kwarg=42)

        import warnings
        # this can be removed once the variables keyword is fully removed
        with warnings.catch_warnings(record=False):
            ds = Dataset(variables={'x1': x1})
        # but assert dataset is still created
        self.assertDatasetEqual(ds, Dataset(data_vars={'x1': x1}))

    def test_constructor_1d(self):
        expected = Dataset({'x': (['x'], 5.0 + np.arange(5))})
        actual = Dataset({'x': 5.0 + np.arange(5)})
        self.assertDatasetIdentical(expected, actual)

        actual = Dataset({'x': [5, 6, 7, 8, 9]})
        self.assertDatasetIdentical(expected, actual)

    def test_constructor_0d(self):
        expected = Dataset({'x': ([], 1)})
        for arg in [1, np.array(1), expected['x']]:
            actual = Dataset({'x': arg})
            self.assertDatasetIdentical(expected, actual)

        d = pd.Timestamp('2000-01-01T12')
        args = [True, None, 3.4, np.nan, 'hello', u'uni', b'raw',
                np.datetime64('2000-01-01T00'), d, d.to_datetime()]
        for arg in args:
            print(arg)
            expected = Dataset({'x': ([], arg)})
            actual = Dataset({'x': arg})
            self.assertDatasetIdentical(expected, actual)

    def test_constructor_auto_align(self):
        a = DataArray([1, 2], [('x', [0, 1])])
        b = DataArray([3, 4], [('x', [1, 2])])

        # verify align uses outer join
        expected = Dataset({'a': ('x', [1, 2, np.nan]),
                            'b': ('x', [np.nan, 3, 4])})
        actual = Dataset({'a': a, 'b': b})
        self.assertDatasetIdentical(expected, actual)

        # regression test for GH346
        self.assertIsInstance(actual.variables['x'], Coordinate)

        # variable with different dimensions
        c = ('y', [3, 4])
        expected2 = expected.merge({'c': c})
        actual = Dataset({'a': a, 'b': b, 'c': c})
        self.assertDatasetIdentical(expected2, actual)

        # variable that is only aligned against the aligned variables
        d = ('x', [3, 2, 1])
        expected3 = expected.merge({'d': d})
        actual = Dataset({'a': a, 'b': b, 'd': d})
        self.assertDatasetIdentical(expected3, actual)

        e = ('x', [0, 0])
        with self.assertRaisesRegexp(ValueError, 'conflicting sizes'):
            Dataset({'a': a, 'b': b, 'e': e})

    def test_constructor_pandas_sequence(self):

        ds = self.make_example_math_dataset()
        pandas_objs = OrderedDict(
            (var_name, ds[var_name].to_pandas()) for var_name in ['foo','bar']
        )
        ds_based_on_pandas = Dataset(data_vars=pandas_objs, coords=ds.coords, attrs=ds.attrs)
        self.assertDatasetEqual(ds, ds_based_on_pandas)

        # reindex pandas obj, check align works
        rearranged_index = reversed(pandas_objs['foo'].index)
        pandas_objs['foo'] = pandas_objs['foo'].reindex(rearranged_index)
        ds_based_on_pandas = Dataset(variables=pandas_objs, coords=ds.coords, attrs=ds.attrs)
        self.assertDatasetEqual(ds, ds_based_on_pandas)

    def test_constructor_pandas_single(self):

        das = [
            DataArray(np.random.rand(4), dims=['a']),  # series
            DataArray(np.random.rand(4,3), dims=['a', 'b']),  # df
            DataArray(np.random.rand(4,3,2), dims=['a','b','c']), # panel
            ]

        for da in das:
            pandas_obj = da.to_pandas()
            ds_based_on_pandas = Dataset(pandas_obj)
            for dim in ds_based_on_pandas.data_vars:
                self.assertArrayEqual(ds_based_on_pandas[dim], pandas_obj[dim])


    def test_constructor_compat(self):
        data = OrderedDict([('x', DataArray(0, coords={'y': 1})),
                            ('y', ('z', [1, 1, 1]))])
        with self.assertRaisesRegexp(ValueError, 'conflicting value'):
            Dataset(data, compat='equals')
        expected = Dataset({'x': 0}, {'y': ('z', [1, 1, 1])})
        actual = Dataset(data)
        self.assertDatasetIdentical(expected, actual)
        actual = Dataset(data, compat='broadcast_equals')
        self.assertDatasetIdentical(expected, actual)

        data = OrderedDict([('y', ('z', [1, 1, 1])),
                            ('x', DataArray(0, coords={'y': 1}))])
        actual = Dataset(data)
        self.assertDatasetIdentical(expected, actual)

        original = Dataset({'a': (('x', 'y'), np.ones((2, 3)))},
                           {'c': (('x', 'y'), np.zeros((2, 3)))})
        expected = Dataset({'a': ('x', np.ones(2)),
                            'b': ('y', np.ones(3))},
                           {'c': (('x', 'y'), np.zeros((2, 3)))})
        # use an OrderedDict to ensure test results are reproducible; otherwise
        # the order of appearance of x and y matters for the order of
        # dimensions in 'c'
        actual = Dataset(OrderedDict([('a', original['a'][:, 0].drop('y')),
                                      ('b', original['a'][0].drop('x'))]))
        self.assertDatasetIdentical(expected, actual)

        data = {'x': DataArray(0, coords={'y': 3}), 'y': ('z', [1, 1, 1])}
        with self.assertRaisesRegexp(ValueError, 'conflicting value'):
            Dataset(data)

        data = {'x': DataArray(0, coords={'y': 1}), 'y': [1, 1]}
        actual = Dataset(data)
        expected = Dataset({'x': 0}, {'y': [1, 1]})
        self.assertDatasetIdentical(expected, actual)

    def test_constructor_with_coords(self):
        with self.assertRaisesRegexp(ValueError, 'redundant variables and co'):
            Dataset({'a': ('x', [1])}, {'a': ('x', [1])})

        ds = Dataset({}, {'a': ('x', [1])})
        self.assertFalse(ds.data_vars)
        self.assertItemsEqual(ds.coords.keys(), ['x', 'a'])

    def test_properties(self):
        ds = create_test_data()
        self.assertEqual(ds.dims,
                         {'dim1': 8, 'dim2': 9, 'dim3': 10, 'time': 20})

        self.assertItemsEqual(ds, list(ds.variables))
        self.assertItemsEqual(ds.keys(), list(ds.variables))
        self.assertNotIn('aasldfjalskdfj', ds.variables)
        self.assertIn('dim1', repr(ds.variables))
        self.assertEqual(len(ds), 8)

        self.assertItemsEqual(ds.data_vars, ['var1', 'var2', 'var3'])
        self.assertItemsEqual(ds.data_vars.keys(), ['var1', 'var2', 'var3'])
        self.assertIn('var1', ds.data_vars)
        self.assertNotIn('dim1', ds.data_vars)
        self.assertNotIn('numbers', ds.data_vars)
        self.assertEqual(len(ds.data_vars), 3)

        self.assertItemsEqual(ds.indexes, ['dim1', 'dim2', 'dim3', 'time'])
        self.assertEqual(len(ds.indexes), 4)
        self.assertIn('dim1', repr(ds.indexes))

        self.assertItemsEqual(ds.coords,
                              ['time', 'dim1', 'dim2', 'dim3', 'numbers'])
        self.assertIn('dim1', ds.coords)
        self.assertIn('numbers', ds.coords)
        self.assertNotIn('var1', ds.coords)
        self.assertEqual(len(ds.coords), 5)

        self.assertEqual(Dataset({'x': np.int64(1),
                                  'y': np.float32([1, 2])}).nbytes, 16)

    def test_attr_access(self):
        ds = Dataset({'tmin': ('x', [42], {'units': 'Celcius'})},
                     attrs={'title': 'My test data'})
        self.assertDataArrayIdentical(ds.tmin, ds['tmin'])
        self.assertDataArrayIdentical(ds.tmin.x, ds.x)

        self.assertEqual(ds.title, ds.attrs['title'])
        self.assertEqual(ds.tmin.units, ds['tmin'].attrs['units'])

        self.assertLessEqual(set(['tmin', 'title']), set(dir(ds)))
        self.assertIn('units', set(dir(ds.tmin)))

        # should defer to variable of same name
        ds.attrs['tmin'] = -999
        self.assertEqual(ds.attrs['tmin'], -999)
        self.assertDataArrayIdentical(ds.tmin, ds['tmin'])

    def test_variable(self):
        a = Dataset()
        d = np.random.random((10, 3))
        a['foo'] = (('time', 'x',), d)
        self.assertTrue('foo' in a.variables)
        self.assertTrue('foo' in a)
        a['bar'] = (('time', 'x',), d)
        # order of creation is preserved
        self.assertEqual(list(a),  ['foo', 'time', 'x', 'bar'])
        self.assertTrue(all([a['foo'][i].values == d[i]
                             for i in np.ndindex(*d.shape)]))
        # try to add variable with dim (10,3) with data that's (3,10)
        with self.assertRaises(ValueError):
            a['qux'] = (('time', 'x'), d.T)

    def test_modify_inplace(self):
        a = Dataset()
        vec = np.random.random((10,))
        attributes = {'foo': 'bar'}
        a['x'] = ('x', vec, attributes)
        self.assertTrue('x' in a.coords)
        self.assertIsInstance(a.coords['x'].to_index(),
            pd.Index)
        self.assertVariableIdentical(a.coords['x'], a.variables['x'])
        b = Dataset()
        b['x'] = ('x', vec, attributes)
        self.assertVariableIdentical(a['x'], b['x'])
        self.assertEqual(a.dims, b.dims)
        # this should work
        a['x'] = ('x', vec[:5])
        a['z'] = ('x', np.arange(5))
        with self.assertRaises(ValueError):
            # now it shouldn't, since there is a conflicting length
            a['x'] = ('x', vec[:4])
        arr = np.random.random((10, 1,))
        scal = np.array(0)
        with self.assertRaises(ValueError):
            a['y'] = ('y', arr)
        with self.assertRaises(ValueError):
            a['y'] = ('y', scal)
        self.assertTrue('y' not in a.dims)

    def test_coords_properties(self):
        # use an OrderedDict for coordinates to ensure order across python
        # versions
        # use int64 for repr consistency on windows
        data = Dataset(OrderedDict([('x', ('x', np.array([-1, -2], 'int64'))),
                                    ('y', ('y', np.array([0, 1, 2], 'int64'))),
                                    ('foo', (['x', 'y'],
                                             np.random.randn(2, 3)))]),
                       OrderedDict([('a', ('x', np.array([4, 5], 'int64'))),
                                    ('b', np.int64(-10))]))

        self.assertEqual(4, len(data.coords))

        self.assertItemsEqual(['x', 'y', 'a', 'b'], list(data.coords))

        self.assertVariableIdentical(data.coords['x'], data['x'].variable)
        self.assertVariableIdentical(data.coords['y'], data['y'].variable)

        self.assertIn('x', data.coords)
        self.assertIn('a', data.coords)
        self.assertNotIn(0, data.coords)
        self.assertNotIn('foo', data.coords)

        with self.assertRaises(KeyError):
            data.coords['foo']
        with self.assertRaises(KeyError):
            data.coords[0]

        expected = dedent("""\
        Coordinates:
          * x        (x) int64 -1 -2
          * y        (y) int64 0 1 2
            a        (x) int64 4 5
            b        int64 -10""")
        actual = repr(data.coords)
        self.assertEqual(expected, actual)

        self.assertEqual({'x': 2, 'y': 3}, data.coords.dims)

    def test_coords_modify(self):
        data = Dataset({'x': ('x', [-1, -2]),
                        'y': ('y', [0, 1, 2]),
                        'foo': (['x', 'y'], np.random.randn(2, 3))},
                       {'a': ('x', [4, 5]), 'b': -10})

        actual = data.copy(deep=True)
        actual.coords['x'] = ('x', ['a', 'b'])
        self.assertArrayEqual(actual['x'], ['a', 'b'])

        actual = data.copy(deep=True)
        actual.coords['z'] = ('z', ['a', 'b'])
        self.assertArrayEqual(actual['z'], ['a', 'b'])

        with self.assertRaisesRegexp(ValueError, 'conflicting sizes'):
            data.coords['x'] = ('x', [-1])

        actual = data.copy()
        del actual.coords['b']
        expected = data.reset_coords('b', drop=True)
        self.assertDatasetIdentical(expected, actual)

        with self.assertRaises(KeyError):
            del data.coords['not_found']

        with self.assertRaises(KeyError):
            del data.coords['foo']

        actual = data.copy(deep=True)
        actual.coords.update({'c': 11})
        expected = data.merge({'c': 11}).set_coords('c')
        self.assertDatasetIdentical(expected, actual)

    def test_coords_set(self):
        one_coord = Dataset({'x': ('x', [0]),
                             'yy': ('x', [1]),
                             'zzz': ('x', [2])})
        two_coords = Dataset({'zzz': ('x', [2])},
                             {'x': ('x', [0]),
                              'yy': ('x', [1])})
        all_coords = Dataset(coords={'x': ('x', [0]),
                                     'yy': ('x', [1]),
                                     'zzz': ('x', [2])})

        actual = one_coord.set_coords('x')
        self.assertDatasetIdentical(one_coord, actual)
        actual = one_coord.set_coords(['x'])
        self.assertDatasetIdentical(one_coord, actual)

        actual = one_coord.set_coords('yy')
        self.assertDatasetIdentical(two_coords, actual)

        actual = one_coord.set_coords(['yy', 'zzz'])
        self.assertDatasetIdentical(all_coords, actual)

        actual = one_coord.reset_coords()
        self.assertDatasetIdentical(one_coord, actual)
        actual = two_coords.reset_coords()
        self.assertDatasetIdentical(one_coord, actual)
        actual = all_coords.reset_coords()
        self.assertDatasetIdentical(one_coord, actual)

        actual = all_coords.reset_coords(['yy', 'zzz'])
        self.assertDatasetIdentical(one_coord, actual)
        actual = all_coords.reset_coords('zzz')
        self.assertDatasetIdentical(two_coords, actual)

        with self.assertRaisesRegexp(ValueError, 'cannot remove index'):
            one_coord.reset_coords('x')

        actual = all_coords.reset_coords('zzz', drop=True)
        expected = all_coords.drop('zzz')
        self.assertDatasetIdentical(expected, actual)
        expected = two_coords.drop('zzz')
        self.assertDatasetIdentical(expected, actual)

    def test_coords_to_dataset(self):
        orig = Dataset({'foo': ('y', [-1, 0, 1])}, {'x': 10, 'y': [2, 3, 4]})
        expected = Dataset(coords={'x': 10, 'y': [2, 3, 4]})
        actual = orig.coords.to_dataset()
        self.assertDatasetIdentical(expected, actual)

    def test_coords_merge(self):
        orig_coords = Dataset(coords={'a': ('x', [1, 2])}).coords
        other_coords = Dataset(coords={'b': ('x', ['a', 'b'])}).coords
        expected = Dataset(coords={'a': ('x', [1, 2]),
                                   'b': ('x', ['a', 'b'])})
        actual = orig_coords.merge(other_coords)
        self.assertDatasetIdentical(expected, actual)
        actual = other_coords.merge(orig_coords)
        self.assertDatasetIdentical(expected, actual)

        other_coords = Dataset(coords={'x': ('x', ['a'])}).coords
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            orig_coords.merge(other_coords)
        other_coords = Dataset(coords={'x': ('x', ['a', 'b'])}).coords
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            orig_coords.merge(other_coords)
        other_coords = Dataset(coords={'x': ('x', ['a', 'b', 'c'])}).coords
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            orig_coords.merge(other_coords)

        other_coords = Dataset(coords={'a': ('x', [8, 9])}).coords
        expected = Dataset(coords={'x': range(2)})
        actual = orig_coords.merge(other_coords)
        self.assertDatasetIdentical(expected, actual)
        actual = other_coords.merge(orig_coords)
        self.assertDatasetIdentical(expected, actual)

        other_coords = Dataset(coords={'x': np.nan}).coords
        actual = orig_coords.merge(other_coords)
        self.assertDatasetIdentical(orig_coords.to_dataset(), actual)
        actual = other_coords.merge(orig_coords)
        self.assertDatasetIdentical(orig_coords.to_dataset(), actual)

    def test_coords_merge_mismatched_shape(self):
        orig_coords = Dataset(coords={'a': ('x', [1, 1])}).coords
        other_coords = Dataset(coords={'a': 1}).coords
        expected = orig_coords.to_dataset()
        actual = orig_coords.merge(other_coords)
        self.assertDatasetIdentical(expected, actual)

        other_coords = Dataset(coords={'a': ('y', [1])}).coords
        expected = Dataset(coords={'a': (['x', 'y'], [[1], [1]])})
        actual = orig_coords.merge(other_coords)
        self.assertDatasetIdentical(expected, actual)

        actual = other_coords.merge(orig_coords)
        self.assertDatasetIdentical(expected.T, actual)

        orig_coords = Dataset(coords={'a': ('x', [np.nan])}).coords
        other_coords = Dataset(coords={'a': np.nan}).coords
        expected = orig_coords.to_dataset()
        actual = orig_coords.merge(other_coords)
        self.assertDatasetIdentical(expected, actual)

    def test_equals_and_identical(self):
        data = create_test_data(seed=42)
        self.assertTrue(data.equals(data))
        self.assertTrue(data.identical(data))

        data2 = create_test_data(seed=42)
        data2.attrs['foobar'] = 'baz'
        self.assertTrue(data.equals(data2))
        self.assertFalse(data.identical(data2))

        del data2['time']
        self.assertFalse(data.equals(data2))

        data = create_test_data(seed=42).rename({'var1': None})
        self.assertTrue(data.equals(data))
        self.assertTrue(data.identical(data))

        data2 = data.reset_coords()
        self.assertFalse(data2.equals(data))
        self.assertFalse(data2.identical(data))

    def test_equals_failures(self):
        data = create_test_data()
        self.assertFalse(data.equals('foo'))
        self.assertFalse(data.identical(123))
        self.assertFalse(data.broadcast_equals({1: 2}))

    def test_broadcast_equals(self):
        data1 = Dataset(coords={'x': 0})
        data2 = Dataset(coords={'x': [0]})
        self.assertTrue(data1.broadcast_equals(data2))
        self.assertFalse(data1.equals(data2))
        self.assertFalse(data1.identical(data2))

    def test_attrs(self):
        data = create_test_data(seed=42)
        data.attrs = {'foobar': 'baz'}
        self.assertTrue(data.attrs['foobar'], 'baz')
        self.assertIsInstance(data.attrs, OrderedDict)

    @requires_dask
    def test_chunk(self):
        data = create_test_data()
        for v in data.variables.values():
            self.assertIsInstance(v.data, np.ndarray)
        self.assertEqual(data.chunks, {})

        reblocked = data.chunk()
        for v in reblocked.variables.values():
            self.assertIsInstance(v.data, da.Array)
        expected_chunks = dict((d, (s,)) for d, s in data.dims.items())
        self.assertEqual(reblocked.chunks, expected_chunks)

        reblocked = data.chunk({'time': 5, 'dim1': 5, 'dim2': 5, 'dim3': 5})
        expected_chunks = {'time': (5,) * 4, 'dim1': (5, 3),
                           'dim2': (5, 4), 'dim3': (5, 5)}
        self.assertEqual(reblocked.chunks, expected_chunks)

        reblocked = data.chunk(expected_chunks)
        self.assertEqual(reblocked.chunks, expected_chunks)

        # reblock on already blocked data
        reblocked = reblocked.chunk(expected_chunks)
        self.assertEqual(reblocked.chunks, expected_chunks)
        self.assertDatasetIdentical(reblocked, data)

        with self.assertRaisesRegexp(ValueError, 'some chunks'):
            data.chunk({'foo': 10})

    @requires_dask
    def test_dask_is_lazy(self):
        store = InaccessibleVariableDataStore()
        create_test_data().dump_to_store(store)
        ds = open_dataset(store).chunk()

        with self.assertRaises(UnexpectedDataAccess):
            ds.load()
        with self.assertRaises(UnexpectedDataAccess):
            ds['var1'].values

        # these should not raise UnexpectedDataAccess:
        ds.var1.data
        ds.isel(time=10)
        ds.isel(time=slice(10), dim1=[0]).isel(dim1=0, dim2=-1)
        ds.transpose()
        ds.mean()
        ds.fillna(0)
        ds.rename({'dim1': 'foobar'})
        ds.set_coords('var1')
        ds.drop('var1')

    def test_isel(self):
        data = create_test_data()
        slicers = {'dim1': slice(None, None, 2), 'dim2': slice(0, 2)}
        ret = data.isel(**slicers)

        # Verify that only the specified dimension was altered
        self.assertItemsEqual(data.dims, ret.dims)
        for d in data.dims:
            if d in slicers:
                self.assertEqual(ret.dims[d],
                                 np.arange(data.dims[d])[slicers[d]].size)
            else:
                self.assertEqual(data.dims[d], ret.dims[d])
        # Verify that the data is what we expect
        for v in data:
            self.assertEqual(data[v].dims, ret[v].dims)
            self.assertEqual(data[v].attrs, ret[v].attrs)
            slice_list = [slice(None)] * data[v].values.ndim
            for d, s in iteritems(slicers):
                if d in data[v].dims:
                    inds = np.nonzero(np.array(data[v].dims) == d)[0]
                    for ind in inds:
                        slice_list[ind] = s
            expected = data[v].values[slice_list]
            actual = ret[v].values
            np.testing.assert_array_equal(expected, actual)

        with self.assertRaises(ValueError):
            data.isel(not_a_dim=slice(0, 2))

        ret = data.isel(dim1=0)
        self.assertEqual({'time': 20, 'dim2': 9, 'dim3': 10}, ret.dims)
        self.assertItemsEqual(data.data_vars, ret.data_vars)
        self.assertItemsEqual(data.coords, ret.coords)
        self.assertItemsEqual(data.indexes, list(ret.indexes) + ['dim1'])

        ret = data.isel(time=slice(2), dim1=0, dim2=slice(5))
        self.assertEqual({'time': 2, 'dim2': 5, 'dim3': 10}, ret.dims)
        self.assertItemsEqual(data.data_vars, ret.data_vars)
        self.assertItemsEqual(data.coords, ret.coords)
        self.assertItemsEqual(data.indexes, list(ret.indexes) + ['dim1'])

        ret = data.isel(time=0, dim1=0, dim2=slice(5))
        self.assertItemsEqual({'dim2': 5, 'dim3': 10}, ret.dims)
        self.assertItemsEqual(data.data_vars, ret.data_vars)
        self.assertItemsEqual(data.coords, ret.coords)
        self.assertItemsEqual(data.indexes,
                              list(ret.indexes) + ['dim1', 'time'])

    def test_sel(self):
        data = create_test_data()
        int_slicers = {'dim1': slice(None, None, 2),
                       'dim2': slice(2),
                       'dim3': slice(3)}
        loc_slicers = {'dim1': slice(None, None, 2),
                       'dim2': slice(0, 0.5),
                       'dim3': slice('a', 'c')}
        self.assertDatasetEqual(data.isel(**int_slicers),
                                data.sel(**loc_slicers))
        data['time'] = ('time', pd.date_range('2000-01-01', periods=20))
        self.assertDatasetEqual(data.isel(time=0),
                                data.sel(time='2000-01-01'))
        self.assertDatasetEqual(data.isel(time=slice(10)),
                                data.sel(time=slice('2000-01-01',
                                                    '2000-01-10')))
        self.assertDatasetEqual(data, data.sel(time=slice('1999', '2005')))
        times = pd.date_range('2000-01-01', periods=3)
        self.assertDatasetEqual(data.isel(time=slice(3)),
                                data.sel(time=times))
        self.assertDatasetEqual(data.isel(time=slice(3)),
                                data.sel(time=(data['time.dayofyear'] <= 3)))

        td = pd.to_timedelta(np.arange(3), unit='days')
        data = Dataset({'x': ('td', np.arange(3)), 'td': td})
        self.assertDatasetEqual(data, data.sel(td=td))
        self.assertDatasetEqual(data, data.sel(td=slice('3 days')))
        self.assertDatasetEqual(data.isel(td=0), data.sel(td='0 days'))
        self.assertDatasetEqual(data.isel(td=0), data.sel(td='0h'))
        self.assertDatasetEqual(data.isel(td=slice(1, 3)),
                                data.sel(td=slice('1 days', '2 days')))

    def test_isel_points(self):
        data = create_test_data()

        pdim1 = [1, 2, 3]
        pdim2 = [4, 5, 1]
        pdim3 = [1, 2, 3]

        actual = data.isel_points(dim1=pdim1, dim2=pdim2, dim3=pdim3,
                                  dim='test_coord')
        assert 'test_coord' in actual.coords
        assert actual.coords['test_coord'].shape == (len(pdim1), )

        actual = data.isel_points(dim1=pdim1, dim2=pdim2)
        assert 'points' in actual.coords
        np.testing.assert_array_equal(pdim1, actual['dim1'])

        # test that the order of the indexers doesn't matter
        self.assertDatasetIdentical(data.isel_points(dim1=pdim1, dim2=pdim2),
                                    data.isel_points(dim2=pdim2, dim1=pdim1))

        # make sure we're raising errors in the right places
        with self.assertRaisesRegexp(ValueError,
                                     'All indexers must be the same length'):
            data.isel_points(dim1=[1, 2], dim2=[1, 2, 3])
        with self.assertRaisesRegexp(ValueError,
                                     'dimension bad_key does not exist'):
            data.isel_points(bad_key=[1, 2])
        with self.assertRaisesRegexp(TypeError, 'Indexers must be integers'):
            data.isel_points(dim1=[1.5, 2.2])
        with self.assertRaisesRegexp(TypeError, 'Indexers must be integers'):
            data.isel_points(dim1=[1, 2, 3], dim2=slice(3))
        with self.assertRaisesRegexp(ValueError,
                                     'Indexers must be 1 dimensional'):
            data.isel_points(dim1=1, dim2=2)
        with self.assertRaisesRegexp(ValueError,
                                     'Existing dimension names are not valid'):
            data.isel_points(dim1=[1, 2], dim2=[1, 2], dim='dim2')

        # test to be sure we keep around variables that were not indexed
        ds = Dataset({'x': [1, 2, 3, 4], 'y': 0})
        actual = ds.isel_points(x=[0, 1, 2])
        self.assertDataArrayIdentical(ds['y'], actual['y'])

        # tests using index or DataArray as a dim
        stations = Dataset()
        stations['station'] = ('station', ['A', 'B', 'C'])
        stations['dim1s'] = ('station', [1, 2, 3])
        stations['dim2s'] = ('station', [4, 5, 1])

        actual = data.isel_points(dim1=stations['dim1s'],
                                  dim2=stations['dim2s'],
                                  dim=stations['station'])
        assert 'station' in actual.coords
        assert 'station' in actual.dims
        self.assertDataArrayIdentical(actual['station'].drop(['dim1', 'dim2']),
                                      stations['station'])

        # make sure we get the default points coordinate when a list is passed
        actual = data.isel_points(dim1=stations['dim1s'],
                                  dim2=stations['dim2s'],
                                  dim=['A', 'B', 'C'])
        assert 'points' in actual.coords

        # can pass a numpy array
        data.isel_points(dim1=stations['dim1s'],
                         dim2=stations['dim2s'],
                         dim=np.array([4, 5, 6]))

    def test_sel_points(self):
        data = create_test_data()

        pdim1 = [1, 2, 3]
        pdim2 = [4, 5, 1]
        pdim3 = [1, 2, 3]
        expected = data.isel_points(dim1=pdim1, dim2=pdim2, dim3=pdim3,
                                    dim='test_coord')
        actual = data.sel_points(dim1=data.dim1[pdim1], dim2=data.dim2[pdim2],
                                 dim3=data.dim3[pdim3], dim='test_coord')
        self.assertDatasetIdentical(expected, actual)

        data = Dataset({'foo': (('x', 'y'), np.arange(9).reshape(3, 3))})
        expected = Dataset({'foo': ('points', [0, 4, 8])},
                           {'x': ('points', range(3)),
                            'y': ('points', range(3))})
        actual = data.sel_points(x=[0.1, 1.1, 2.5], y=[0, 1.2, 2.0],
                                 method='pad')
        self.assertDatasetIdentical(expected, actual)

        if pd.__version__ >= '0.17':
            with self.assertRaises(KeyError):
                data.sel_points(x=[2.5], y=[2.0], method='pad', tolerance=1e-3)

    def test_sel_method(self):
        data = create_test_data()

        if pd.__version__ >= '0.16':
            expected = data.sel(dim1=1)
            actual = data.sel(dim1=0.95, method='nearest')
            self.assertDatasetIdentical(expected, actual)

        if pd.__version__ >= '0.17':
            actual = data.sel(dim1=0.95, method='nearest', tolerance=1)
            self.assertDatasetIdentical(expected, actual)

            with self.assertRaises(KeyError):
                actual = data.sel(dim1=0.5, method='nearest', tolerance=0)

        expected = data.sel(dim2=[1.5])
        actual = data.sel(dim2=[1.45], method='backfill')
        self.assertDatasetIdentical(expected, actual)

        with self.assertRaisesRegexp(NotImplementedError, 'slice objects'):
            data.sel(dim2=slice(1, 3), method='ffill')

        with self.assertRaisesRegexp(TypeError, '``method``'):
            # this should not pass silently
            data.sel(data)

    def test_loc(self):
        data = create_test_data()
        expected = data.sel(dim3='a')
        actual = data.loc[dict(dim3='a')]
        self.assertDatasetIdentical(expected, actual)
        with self.assertRaisesRegexp(TypeError, 'can only lookup dict'):
            data.loc['a']
        with self.assertRaises(TypeError):
            data.loc[dict(dim3='a')] = 0

    def test_reindex_like(self):
        data = create_test_data()
        data['letters'] = ('dim3', 10 * ['a'])

        expected = data.isel(dim1=slice(10), time=slice(13))
        actual = data.reindex_like(expected)
        self.assertDatasetIdentical(actual, expected)

        expected = data.copy(deep=True)
        expected['dim3'] = ('dim3', list('cdefghijkl'))
        expected['var3'][:-2] = expected['var3'][2:]
        expected['var3'][-2:] = np.nan
        expected['letters'] = expected['letters'].astype(object)
        expected['letters'][-2:] = np.nan
        expected['numbers'] = expected['numbers'].astype(float)
        expected['numbers'][:-2] = expected['numbers'][2:].values
        expected['numbers'][-2:] = np.nan
        actual = data.reindex_like(expected)
        self.assertDatasetIdentical(actual, expected)

    def test_reindex(self):
        data = create_test_data()
        self.assertDatasetIdentical(data, data.reindex())

        expected = data.isel(dim1=slice(10))
        actual = data.reindex(dim1=data['dim1'][:10])
        self.assertDatasetIdentical(actual, expected)

        actual = data.reindex(dim1=data['dim1'][:10].values)
        self.assertDatasetIdentical(actual, expected)

        actual = data.reindex(dim1=data['dim1'][:10].to_index())
        self.assertDatasetIdentical(actual, expected)

        # test dict-like argument
        actual = data.reindex({'dim1': data['dim1'][:10]})
        self.assertDatasetIdentical(actual, expected)
        with self.assertRaisesRegexp(ValueError, 'cannot specify both'):
            data.reindex({'x': 0}, x=0)
        with self.assertRaisesRegexp(ValueError, 'dictionary'):
            data.reindex('foo')

        # invalid dimension
        with self.assertRaisesRegexp(ValueError, 'invalid reindex dim'):
            data.reindex(invalid=0)

        # out of order
        expected = data.sel(dim1=data['dim1'][:10:-1])
        actual = data.reindex(dim1=data['dim1'][:10:-1])
        self.assertDatasetIdentical(actual, expected)

        # regression test for #279
        expected = Dataset({'x': ('time', np.random.randn(5))})
        time2 = DataArray(np.arange(5), dims="time2")
        actual = expected.reindex(time=time2)
        self.assertDatasetIdentical(actual, expected)

        # another regression test
        ds = Dataset({'foo': (['x', 'y'], np.zeros((3, 4)))})
        expected = Dataset({'foo': (['x', 'y'], np.zeros((3, 2))),
                            'x': [0, 1, 3]})
        expected['foo'][-1] = np.nan
        actual = ds.reindex(x=[0, 1, 3], y=[0, 1])
        self.assertDatasetIdentical(expected, actual)

    def test_reindex_method(self):
        ds = Dataset({'x': ('y', [10, 20])})
        y = [-0.5, 0.5, 1.5]
        actual = ds.reindex(y=y, method='backfill')
        expected = Dataset({'x': ('y', [10, 20, np.nan]), 'y': y})
        self.assertDatasetIdentical(expected, actual)

        if pd.__version__ >= '0.17':
            actual = ds.reindex(y=y, method='backfill', tolerance=0.1)
            expected = Dataset({'x': ('y', 3 * [np.nan]), 'y': y})
            self.assertDatasetIdentical(expected, actual)
        else:
            with self.assertRaisesRegexp(NotImplementedError, 'tolerance'):
                ds.reindex(y=y, method='backfill', tolerance=0.1)

        actual = ds.reindex(y=y, method='pad')
        expected = Dataset({'x': ('y', [np.nan, 10, 20]), 'y': y})
        self.assertDatasetIdentical(expected, actual)

        alt = Dataset({'y': y})
        actual = ds.reindex_like(alt, method='pad')
        self.assertDatasetIdentical(expected, actual)

    def test_align(self):
        left = create_test_data()
        right = left.copy(deep=True)
        right['dim3'] = ('dim3', list('cdefghijkl'))
        right['var3'][:-2] = right['var3'][2:]
        right['var3'][-2:] = np.random.randn(*right['var3'][-2:].shape)
        right['numbers'][:-2] = right['numbers'][2:]
        right['numbers'][-2:] = -10

        intersection = list('cdefghij')
        union = list('abcdefghijkl')

        left2, right2 = align(left, right, join='inner')
        self.assertArrayEqual(left2['dim3'], intersection)
        self.assertDatasetIdentical(left2, right2)

        left2, right2 = align(left, right, join='outer')
        self.assertVariableEqual(left2['dim3'], right2['dim3'])
        self.assertArrayEqual(left2['dim3'], union)
        self.assertDatasetIdentical(left2.sel(dim3=intersection),
                                    right2.sel(dim3=intersection))
        self.assertTrue(np.isnan(left2['var3'][-2:]).all())
        self.assertTrue(np.isnan(right2['var3'][:2]).all())

        left2, right2 = align(left, right, join='left')
        self.assertVariableEqual(left2['dim3'], right2['dim3'])
        self.assertVariableEqual(left2['dim3'], left['dim3'])
        self.assertDatasetIdentical(left2.sel(dim3=intersection),
                                    right2.sel(dim3=intersection))
        self.assertTrue(np.isnan(right2['var3'][:2]).all())

        left2, right2 = align(left, right, join='right')
        self.assertVariableEqual(left2['dim3'], right2['dim3'])
        self.assertVariableEqual(left2['dim3'], right['dim3'])
        self.assertDatasetIdentical(left2.sel(dim3=intersection),
                                    right2.sel(dim3=intersection))
        self.assertTrue(np.isnan(left2['var3'][-2:]).all())

        with self.assertRaisesRegexp(ValueError, 'invalid value for join'):
            align(left, right, join='foobar')
        with self.assertRaises(TypeError):
            align(left, right, foo='bar')

    def test_broadcast(self):
        ds = Dataset({'foo': 0, 'bar': ('x', [1]), 'baz': ('y', [2, 3])},
                     {'c': ('x', [4])})
        expected = Dataset({'foo': (('x', 'y'), [[0, 0]]),
                            'bar': (('x', 'y'), [[1, 1]]),
                            'baz': (('x', 'y'), [[2, 3]])},
                            {'c': ('x', [4])})
        actual, = broadcast(ds)
        self.assertDatasetIdentical(expected, actual)

        ds_x = Dataset({'foo': ('x', [1])})
        ds_y = Dataset({'bar': ('y', [2, 3])})
        expected_x = Dataset({'foo': (('x', 'y'), [[1, 1]])})
        expected_y = Dataset({'bar': (('x', 'y'), [[2, 3]])})
        actual_x, actual_y = broadcast(ds_x, ds_y)
        self.assertDatasetIdentical(expected_x, actual_x)
        self.assertDatasetIdentical(expected_y, actual_y)

        array_y = ds_y['bar']
        expected_y = expected_y['bar']
        actual_x, actual_y = broadcast(ds_x, array_y)
        self.assertDatasetIdentical(expected_x, actual_x)
        self.assertDataArrayIdentical(expected_y, actual_y)

    def test_variable_indexing(self):
        data = create_test_data()
        v = data['var1']
        d1 = data['dim1']
        d2 = data['dim2']
        self.assertVariableEqual(v, v[d1.values])
        self.assertVariableEqual(v, v[d1])
        self.assertVariableEqual(v[:3], v[d1 < 3])
        self.assertVariableEqual(v[:, 3:], v[:, d2 >= 1.5])
        self.assertVariableEqual(v[:3, 3:], v[d1 < 3, d2 >= 1.5])
        self.assertVariableEqual(v[:3, :2], v[range(3), range(2)])
        self.assertVariableEqual(v[:3, :2], v.loc[d1[:3], d2[:2]])

    def test_drop_variables(self):
        data = create_test_data()

        self.assertDatasetIdentical(data, data.drop([]))

        expected = Dataset(dict((k, data[k]) for k in data if k != 'time'))
        actual = data.drop('time')
        self.assertDatasetIdentical(expected, actual)
        actual = data.drop(['time'])
        self.assertDatasetIdentical(expected, actual)

        expected = Dataset(dict((k, data[k]) for
                                k in ['dim2', 'dim3', 'time', 'numbers']))
        actual = data.drop('dim1')
        self.assertDatasetIdentical(expected, actual)

        with self.assertRaisesRegexp(ValueError, 'cannot be found'):
            data.drop('not_found_here')

    def test_drop_index_labels(self):
        data = Dataset({'A': (['x', 'y'], np.random.randn(2, 3)),
                        'x': ['a', 'b']})

        actual = data.drop(1, 'y')
        expected = data.isel(y=[0, 2])
        self.assertDatasetIdentical(expected, actual)

        actual = data.drop(['a'], 'x')
        expected = data.isel(x=[1])
        self.assertDatasetIdentical(expected, actual)

        actual = data.drop(['a', 'b'], 'x')
        expected = data.isel(x=slice(0, 0))
        self.assertDatasetIdentical(expected, actual)

        with self.assertRaises(ValueError):
            # not contained in axis
            data.drop(['c'], dim='x')

    def test_copy(self):
        data = create_test_data()

        for copied in [data.copy(deep=False), copy(data)]:
            self.assertDatasetIdentical(data, copied)
            for k in data:
                v0 = data.variables[k]
                v1 = copied.variables[k]
                self.assertIs(v0, v1)
            copied['foo'] = ('z', np.arange(5))
            self.assertNotIn('foo', data)

        for copied in [data.copy(deep=True), deepcopy(data)]:
            self.assertDatasetIdentical(data, copied)
            for k in data:
                v0 = data.variables[k]
                v1 = copied.variables[k]
                self.assertIsNot(v0, v1)

    def test_rename(self):
        data = create_test_data()
        newnames = {'var1': 'renamed_var1', 'dim2': 'renamed_dim2'}
        renamed = data.rename(newnames)

        variables = OrderedDict(data.variables)
        for k, v in iteritems(newnames):
            variables[v] = variables.pop(k)

        for k, v in iteritems(variables):
            dims = list(v.dims)
            for name, newname in iteritems(newnames):
                if name in dims:
                    dims[dims.index(name)] = newname

            self.assertVariableEqual(Variable(dims, v.values, v.attrs),
                                     renamed[k])
            self.assertEqual(v.encoding, renamed[k].encoding)
            self.assertEqual(type(v), type(renamed.variables[k]))

        self.assertTrue('var1' not in renamed)
        self.assertTrue('dim2' not in renamed)

        with self.assertRaisesRegexp(ValueError, "cannot rename 'not_a_var'"):
            data.rename({'not_a_var': 'nada'})

        with self.assertRaisesRegexp(ValueError, "'var1' already exists"):
            data.rename({'var2': 'var1'})

        # verify that we can rename a variable without accessing the data
        var1 = data['var1']
        data['var1'] = (var1.dims, InaccessibleArray(var1.values))
        renamed = data.rename(newnames)
        with self.assertRaises(UnexpectedDataAccess):
            renamed['renamed_var1'].values

    def test_rename_same_name(self):
        data = create_test_data()
        newnames = {'var1': 'var1', 'dim2': 'dim2'}
        renamed = data.rename(newnames)
        self.assertDatasetIdentical(renamed, data)

    def test_rename_inplace(self):
        times = pd.date_range('2000-01-01', periods=3)
        data = Dataset({'z': ('x', [2, 3, 4]), 't': ('t', times)})
        copied = data.copy()
        renamed = data.rename({'x': 'y'})
        data.rename({'x': 'y'}, inplace=True)
        self.assertDatasetIdentical(data, renamed)
        self.assertFalse(data.equals(copied))
        self.assertEquals(data.dims, {'y': 3, 't': 3})
        # check virtual variables
        self.assertArrayEqual(data['t.dayofyear'], [1, 2, 3])

    def test_swap_dims(self):
        original = Dataset({'x': [1, 2, 3], 'y': ('x', list('abc')), 'z': 42})
        expected = Dataset({'z': 42}, {'x': ('y', [1, 2, 3]), 'y': list('abc')})
        actual = original.swap_dims({'x': 'y'})
        self.assertDatasetIdentical(expected, actual)
        self.assertIsInstance(actual.variables['y'], Coordinate)
        self.assertIsInstance(actual.variables['x'], Variable)

        roundtripped = actual.swap_dims({'y': 'x'})
        self.assertDatasetIdentical(original.set_coords('y'), roundtripped)

        actual = original.copy()
        actual.swap_dims({'x': 'y'}, inplace=True)
        self.assertDatasetIdentical(expected, actual)

        with self.assertRaisesRegexp(ValueError, 'cannot swap'):
            original.swap_dims({'y': 'x'})
        with self.assertRaisesRegexp(ValueError, 'replacement dimension'):
            original.swap_dims({'x': 'z'})

    def test_stack(self):
        ds = Dataset({'a': ('x', [0, 1]),
                      'b': (('x', 'y'), [[0, 1], [2, 3]]),
                      'y': ['a', 'b']})

        exp_index = pd.MultiIndex.from_product([[0, 1], ['a', 'b']],
                                               names=['x', 'y'])
        expected = Dataset({'a': ('z', [0, 0, 1, 1]),
                            'b': ('z', [0, 1, 2, 3]),
                            'z': exp_index})
        actual = ds.stack(z=['x', 'y'])
        self.assertDatasetIdentical(expected, actual)

        exp_index = pd.MultiIndex.from_product([['a', 'b'], [0, 1]],
                                               names=['y', 'x'])
        expected = Dataset({'a': ('z', [0, 1, 0, 1]),
                            'b': ('z', [0, 2, 1, 3]),
                            'z': exp_index})
        actual = ds.stack(z=['y', 'x'])
        self.assertDatasetIdentical(expected, actual)

    def test_unstack(self):
        index = pd.MultiIndex.from_product([[0, 1], ['a', 'b']],
                                           names=['x', 'y'])
        ds = Dataset({'b': ('z', [0, 1, 2, 3]), 'z': index})
        expected = Dataset({'b': (('x', 'y'), [[0, 1], [2, 3]]),
                            'y': ['a', 'b']})
        actual = ds.unstack('z')
        self.assertDatasetIdentical(actual, expected)

    def test_unstack_errors(self):
        ds = Dataset({'x': [1, 2, 3]})
        with self.assertRaisesRegexp(ValueError, 'invalid dimension'):
            ds.unstack('foo')
        with self.assertRaisesRegexp(ValueError, 'does not have a MultiIndex'):
            ds.unstack('x')

        ds2 = Dataset({'x': pd.Index([(0, 1)])})
        with self.assertRaisesRegexp(ValueError, 'unnamed levels'):
            ds2.unstack('x')

    def test_stack_unstack(self):
        ds = Dataset({'a': ('x', [0, 1]),
                      'b': (('x', 'y'), [[0, 1], [2, 3]]),
                      'y': ['a', 'b']})
        actual = ds.stack(z=['x', 'y']).unstack('z')
        assert actual.broadcast_equals(ds)

        actual = ds[['b']].stack(z=['x', 'y']).unstack('z')
        assert actual.identical(ds[['b']])

    def test_update(self):
        data = create_test_data(seed=0)
        expected = data.copy()
        var2 = Variable('dim1', np.arange(8))
        actual = data.update({'var2': var2})
        expected['var2'] = var2
        self.assertDatasetIdentical(expected, actual)

        actual = data.copy()
        actual_result = actual.update(data, inplace=True)
        self.assertIs(actual_result, actual)
        self.assertDatasetIdentical(expected, actual)

        actual = data.update(data, inplace=False)
        expected = data
        self.assertIsNot(actual, expected)
        self.assertDatasetIdentical(expected, actual)

        other = Dataset(attrs={'new': 'attr'})
        actual = data.copy()
        actual.update(other)
        self.assertDatasetIdentical(expected, actual)

    def test_update_auto_align(self):
        ds = Dataset({'x': ('t', [3, 4])})

        expected = Dataset({'x': ('t', [3, 4]), 'y': ('t', [np.nan, 5])})
        actual = ds.copy()
        other = {'y': ('t', [5]), 't': [1]}
        with self.assertRaisesRegexp(ValueError, 'conflicting sizes'):
            actual.update(other)
        actual.update(Dataset(other))
        self.assertDatasetIdentical(expected, actual)

        actual = ds.copy()
        other = Dataset({'y': ('t', [5]), 't': [100]})
        actual.update(other)
        expected = Dataset({'x': ('t', [3, 4]), 'y': ('t', [np.nan] * 2)})
        self.assertDatasetIdentical(expected, actual)

    def test_merge(self):
        data = create_test_data()
        ds1 = data[['var1']]
        ds2 = data[['var3']]
        expected = data[['var1', 'var3']]
        actual = ds1.merge(ds2)
        self.assertDatasetIdentical(expected, actual)

        actual = ds2.merge(ds1)
        self.assertDatasetIdentical(expected, actual)

        actual = data.merge(data)
        self.assertDatasetIdentical(data, actual)
        actual = data.reset_coords(drop=True).merge(data)
        self.assertDatasetIdentical(data, actual)
        actual = data.merge(data.reset_coords(drop=True))
        self.assertDatasetIdentical(data, actual)

        with self.assertRaises(ValueError):
            ds1.merge(ds2.rename({'var3': 'var1'}))
        with self.assertRaisesRegexp(ValueError, 'cannot merge'):
            data.reset_coords().merge(data)
        with self.assertRaisesRegexp(ValueError, 'cannot merge'):
            data.merge(data.reset_coords())

    def test_merge_broadcast_equals(self):
        ds1 = Dataset({'x': 0})
        ds2 = Dataset({'x': ('y', [0, 0])})
        actual = ds1.merge(ds2)
        self.assertDatasetIdentical(ds2, actual)

        actual = ds2.merge(ds1)
        self.assertDatasetIdentical(ds2, actual)

        actual = ds1.copy()
        actual.update(ds2)
        self.assertDatasetIdentical(ds2, actual)

        ds1 = Dataset({'x': np.nan})
        ds2 = Dataset({'x': ('y', [np.nan, np.nan])})
        actual = ds1.merge(ds2)
        self.assertDatasetIdentical(ds2, actual)

    def test_merge_compat(self):
        ds1 = Dataset({'x': 0})
        ds2 = Dataset({'x': 1})
        for compat in ['broadcast_equals', 'equals', 'identical']:
            with self.assertRaisesRegexp(ValueError, 'conflicting value'):
                ds1.merge(ds2, compat=compat)

        ds2 = Dataset({'x': [0, 0]})
        for compat in ['equals', 'identical']:
            with self.assertRaisesRegexp(ValueError, 'conflicting value'):
                ds1.merge(ds2, compat=compat)

        ds2 = Dataset({'x': ((), 0, {'foo': 'bar'})})
        with self.assertRaisesRegexp(ValueError, 'conflicting value'):
            ds1.merge(ds2, compat='identical')

        with self.assertRaisesRegexp(ValueError, 'compat=\S+ invalid'):
            ds1.merge(ds2, compat='foobar')

    def test_merge_auto_align(self):
        ds1 = Dataset({'a': ('x', [1, 2])})
        ds2 = Dataset({'b': ('x', [3, 4]), 'x': [1, 2]})
        expected = Dataset({'a': ('x', [1, 2, np.nan]),
                            'b': ('x', [np.nan, 3, 4])})
        self.assertDatasetIdentical(expected, ds1.merge(ds2))
        self.assertDatasetIdentical(expected, ds2.merge(ds1))

        expected = expected.isel(x=slice(2))
        self.assertDatasetIdentical(expected, ds1.merge(ds2, join='left'))
        self.assertDatasetIdentical(expected, ds2.merge(ds1, join='right'))

        expected = expected.isel(x=slice(1, 2))
        self.assertDatasetIdentical(expected, ds1.merge(ds2, join='inner'))
        self.assertDatasetIdentical(expected, ds2.merge(ds1, join='inner'))

    def test_getitem(self):
        data = create_test_data()
        self.assertIsInstance(data['var1'], DataArray)
        self.assertVariableEqual(data['var1'], data.variables['var1'])
        with self.assertRaises(KeyError):
            data['notfound']
        with self.assertRaises(KeyError):
            data[['var1', 'notfound']]

        actual = data[['var1', 'var2']]
        expected = Dataset({'var1': data['var1'], 'var2': data['var2']})
        self.assertDatasetEqual(expected, actual)

        actual = data['numbers']
        expected = DataArray(data['numbers'].variable,
                             {'dim3': data['dim3'],
                              'numbers': data['numbers']},
                             dims='dim3', name='numbers')
        self.assertDataArrayIdentical(expected, actual)

        actual = data[dict(dim1=0)]
        expected = data.isel(dim1=0)
        self.assertDatasetIdentical(expected, actual)

    def test_getitem_hashable(self):
        data = create_test_data()
        data[(3, 4)] = data['var1'] + 1
        expected = data['var1'] + 1
        expected.name = (3, 4)
        self.assertDataArrayIdentical(expected, data[(3, 4)])
        with self.assertRaisesRegexp(KeyError, "('var1', 'var2')"):
            data[('var1', 'var2')]

    def test_virtual_variables(self):
        # access virtual variables
        data = create_test_data()
        expected = DataArray(1 + np.arange(20), coords=[data['time']],
                             dims='time', name='dayofyear')
        self.assertDataArrayIdentical(expected, data['time.dayofyear'])
        self.assertArrayEqual(data['time.month'].values,
                              data.variables['time'].to_index().month)
        self.assertArrayEqual(data['time.season'].values, 'DJF')
        # test virtual variable math
        self.assertArrayEqual(data['time.dayofyear'] + 1, 2 + np.arange(20))
        self.assertArrayEqual(np.sin(data['time.dayofyear']),
                              np.sin(1 + np.arange(20)))
        # ensure they become coordinates
        expected = Dataset({}, {'dayofyear': data['time.dayofyear']})
        actual = data[['time.dayofyear']]
        self.assertDatasetEqual(expected, actual)
        # non-coordinate variables
        ds = Dataset({'t': ('x', pd.date_range('2000-01-01', periods=3))})
        self.assertTrue((ds['t.year'] == 2000).all())

    def test_virtual_variable_same_name(self):
        # regression test for GH367
        times = pd.date_range('2000-01-01', freq='H', periods=5)
        data = Dataset({'time': times})
        actual = data['time.time']
        expected = DataArray(times.time, {'time': times}, name='time')
        self.assertDataArrayIdentical(actual, expected)

    def test_time_season(self):
        ds = Dataset({'t': pd.date_range('2000-01-01', periods=12, freq='M')})
        expected = ['DJF'] * 2 + ['MAM'] * 3 + ['JJA'] * 3 + ['SON'] * 3 + ['DJF']
        self.assertArrayEqual(expected, ds['t.season'])

    def test_slice_virtual_variable(self):
        data = create_test_data()
        self.assertVariableEqual(data['time.dayofyear'][:10],
                                 Variable(['time'], 1 + np.arange(10)))
        self.assertVariableEqual(data['time.dayofyear'][0], Variable([], 1))

    def test_setitem(self):
        # assign a variable
        var = Variable(['dim1'], np.random.randn(8))
        data1 = create_test_data()
        data1['A'] = var
        data2 = data1.copy()
        data2['A'] = var
        self.assertDatasetIdentical(data1, data2)
        # assign a dataset array
        dv = 2 * data2['A']
        data1['B'] = dv.variable
        data2['B'] = dv
        self.assertDatasetIdentical(data1, data2)
        # can't assign an ND array without dimensions
        with self.assertRaisesRegexp(ValueError,
                                     'dimensions .* must have the same len'):
            data2['C'] = var.values.reshape(2, 4)
        # but can assign a 1D array
        data1['C'] = var.values
        data2['C'] = ('C', var.values)
        self.assertDatasetIdentical(data1, data2)
        # can assign a scalar
        data1['scalar'] = 0
        data2['scalar'] = ([], 0)
        self.assertDatasetIdentical(data1, data2)
        # can't use the same dimension name as a scalar var
        with self.assertRaisesRegexp(ValueError, 'cannot merge'):
            data1['newvar'] = ('scalar', [3, 4, 5])
        # can't resize a used dimension
        with self.assertRaisesRegexp(ValueError, 'conflicting sizes'):
            data1['dim1'] = data1['dim1'][:5]
        # override an existing value
        data1['A'] = 3 * data2['A']
        self.assertVariableEqual(data1['A'], 3 * data2['A'])

        with self.assertRaises(NotImplementedError):
            data1[{'x': 0}] = 0

    def test_setitem_pandas(self):

        ds = self.make_example_math_dataset()
        ds_copy = ds.copy()
        ds_copy['bar'] = ds['bar'].to_pandas()

        self.assertDatasetEqual(ds, ds_copy)

    def test_setitem_auto_align(self):
        ds = Dataset()
        ds['x'] = ('y', range(3))
        ds['y'] = 1 + np.arange(3)
        expected = Dataset({'x': ('y', range(3)), 'y': 1 + np.arange(3)})
        self.assertDatasetIdentical(ds, expected)

        ds['y'] = DataArray(range(3), dims='y')
        expected = Dataset({'x': ('y', range(3))})
        self.assertDatasetIdentical(ds, expected)

        ds['x'] = DataArray([1, 2], dims='y')
        expected = Dataset({'x': ('y', [1, 2, np.nan])})
        self.assertDatasetIdentical(ds, expected)

        ds['x'] = 42
        expected = Dataset({'x': 42, 'y': range(3)})
        self.assertDatasetIdentical(ds, expected)

        ds['x'] = DataArray([4, 5, 6, 7], dims='y')
        expected = Dataset({'x': ('y', [4, 5, 6])})
        self.assertDatasetIdentical(ds, expected)

    def test_assign(self):
        ds = Dataset()
        actual = ds.assign(x = [0, 1, 2], y = 2)
        expected = Dataset({'x': [0, 1, 2], 'y': 2})
        self.assertDatasetIdentical(actual, expected)
        self.assertEqual(list(actual), ['x', 'y'])
        self.assertDatasetIdentical(ds, Dataset())

        actual = actual.assign(y = lambda ds: ds.x ** 2)
        expected = Dataset({'y': ('x', [0, 1, 4])})
        self.assertDatasetIdentical(actual, expected)

        actual = actual.assign_coords(z = 2)
        expected = Dataset({'y': ('x', [0, 1, 4])}, {'z': 2})
        self.assertDatasetIdentical(actual, expected)

        ds = Dataset({'a': ('x', range(3))}, {'b': ('x', ['A'] * 2 + ['B'])})
        actual = ds.groupby('b').assign(c = lambda ds: 2 * ds.a)
        expected = ds.merge({'c': ('x', [0, 2, 4])})
        self.assertDatasetIdentical(actual, expected)

        actual = ds.groupby('b').assign(c = lambda ds: ds.a.sum())
        expected = ds.merge({'c': ('x', [1, 1, 2])})
        self.assertDatasetIdentical(actual, expected)

        actual = ds.groupby('b').assign_coords(c = lambda ds: ds.a.sum())
        expected = expected.set_coords('c')
        self.assertDatasetIdentical(actual, expected)

    def test_delitem(self):
        data = create_test_data()
        all_items = set(data)
        self.assertItemsEqual(data, all_items)
        del data['var1']
        self.assertItemsEqual(data, all_items - set(['var1']))
        del data['dim1']
        self.assertItemsEqual(data, set(['time', 'dim2', 'dim3', 'numbers']))
        self.assertNotIn('dim1', data.dims)
        self.assertNotIn('dim1', data.coords)

    def test_squeeze(self):
        data = Dataset({'foo': (['x', 'y', 'z'], [[[1], [2]]])})
        for args in [[], [['x']], [['x', 'z']]]:
            def get_args(v):
                return [set(args[0]) & set(v.dims)] if args else []
            expected = Dataset(dict((k, v.squeeze(*get_args(v)))
                                    for k, v in iteritems(data.variables)))
            expected.set_coords(data.coords, inplace=True)
            self.assertDatasetIdentical(expected, data.squeeze(*args))
        # invalid squeeze
        with self.assertRaisesRegexp(ValueError, 'cannot select a dimension'):
            data.squeeze('y')

    def test_groupby(self):
        data = Dataset({'z': (['x', 'y'], np.random.randn(3, 5))},
                       {'x': ('x', list('abc')),
                        'c': ('x', [0, 1, 0])})
        groupby = data.groupby('x')
        self.assertEqual(len(groupby), 3)
        expected_groups = {'a': 0, 'b': 1, 'c': 2}
        self.assertEqual(groupby.groups, expected_groups)
        expected_items = [('a', data.isel(x=0)),
                          ('b', data.isel(x=1)),
                          ('c', data.isel(x=2))]
        for actual, expected in zip(groupby, expected_items):
            self.assertEqual(actual[0], expected[0])
            self.assertDatasetEqual(actual[1], expected[1])

        identity = lambda x: x
        for k in ['x', 'c', 'y']:
            actual = data.groupby(k, squeeze=False).apply(identity)
            self.assertDatasetEqual(data, actual)

    def test_groupby_returns_new_type(self):
        data = Dataset({'z': (['x', 'y'], np.random.randn(3, 5))})

        actual = data.groupby('x').apply(lambda ds: ds['z'])
        expected = data['z']
        self.assertDataArrayIdentical(expected, actual)

        actual = data['z'].groupby('x').apply(lambda x: x.to_dataset())
        expected = data
        self.assertDatasetIdentical(expected, actual)

    def test_groupby_iter(self):
        data = create_test_data()
        for n, (t, sub) in enumerate(list(data.groupby('dim1'))[:3]):
            self.assertEqual(data['dim1'][n], t)
            self.assertVariableEqual(data['var1'][n], sub['var1'])
            self.assertVariableEqual(data['var2'][n], sub['var2'])
            self.assertVariableEqual(data['var3'][:, n], sub['var3'])

    def test_groupby_errors(self):
        data = create_test_data()
        with self.assertRaisesRegexp(ValueError, 'must be 1 dimensional'):
            data.groupby('var1')
        with self.assertRaisesRegexp(ValueError, 'must have a name'):
            data.groupby(np.arange(10))
        with self.assertRaisesRegexp(ValueError, 'length does not match'):
            data.groupby(data['dim1'][:3])
        with self.assertRaisesRegexp(ValueError, "must have a 'dims'"):
            data.groupby(data.coords['dim1'].to_index())

    def test_groupby_reduce(self):
        data = Dataset({'xy': (['x', 'y'], np.random.randn(3, 4)),
                        'xonly': ('x', np.random.randn(3)),
                        'yonly': ('y', np.random.randn(4)),
                        'letters': ('y', ['a', 'a', 'b', 'b'])})

        expected = data.mean('y')
        expected['yonly'] = expected['yonly'].variable.expand_dims({'x': 3})
        actual = data.groupby('x').mean()
        self.assertDatasetAllClose(expected, actual)

        actual = data.groupby('x').mean('y')
        self.assertDatasetAllClose(expected, actual)

        letters = data['letters']
        expected = Dataset({'xy': data['xy'].groupby(letters).mean(),
                            'xonly': (data['xonly'].mean().variable
                                      .expand_dims({'letters': 2})),
                            'yonly': data['yonly'].groupby(letters).mean()})
        actual = data.groupby('letters').mean()
        self.assertDatasetAllClose(expected, actual)

    def test_groupby_math(self):
        reorder_dims = lambda x: x.transpose('dim1', 'dim2', 'dim3', 'time')

        ds = create_test_data()
        for squeeze in [True, False]:
            grouped = ds.groupby('dim1', squeeze=squeeze)

            expected = reorder_dims(ds + ds.coords['dim1'])
            actual = grouped + ds.coords['dim1']
            self.assertDatasetIdentical(expected, reorder_dims(actual))

            actual = ds.coords['dim1'] + grouped
            self.assertDatasetIdentical(expected, reorder_dims(actual))

            ds2 = 2 * ds
            expected = reorder_dims(ds + ds2)
            actual = grouped + ds2
            self.assertDatasetIdentical(expected, reorder_dims(actual))

            actual = ds2 + grouped
            self.assertDatasetIdentical(expected, reorder_dims(actual))

        grouped = ds.groupby('numbers')
        zeros = DataArray([0, 0, 0, 0], [('numbers', range(4))])
        expected = ((ds + Variable('dim3', np.zeros(10)))
                    .transpose('dim3', 'dim1', 'dim2', 'time'))
        actual = grouped + zeros
        self.assertDatasetEqual(expected, actual)

        actual = zeros + grouped
        self.assertDatasetEqual(expected, actual)

        with self.assertRaisesRegexp(ValueError, 'dimensions .* do not exist'):
            grouped + ds
        with self.assertRaisesRegexp(ValueError, 'dimensions .* do not exist'):
            ds + grouped
        with self.assertRaisesRegexp(TypeError, 'only support binary ops'):
            grouped + 1
        with self.assertRaisesRegexp(TypeError, 'only support binary ops'):
            grouped + grouped
        with self.assertRaisesRegexp(TypeError, 'in-place operations'):
            ds += grouped

        ds = Dataset({'x': ('time', np.arange(100)),
                      'time': pd.date_range('2000-01-01', periods=100)})
        with self.assertRaisesRegexp(ValueError, 'incompat.* grouped binary'):
            ds + ds.groupby('time.month')

    def test_groupby_math_virtual(self):
        ds = Dataset({'x': ('t', [1, 2, 3])},
                     {'t': pd.date_range('20100101', periods=3)})
        grouped = ds.groupby('t.day')
        actual = grouped - grouped.mean()
        expected = Dataset({'x': ('t', [0, 0, 0])},
                           ds[['t', 't.day']])
        self.assertDatasetIdentical(actual, expected)

    def test_groupby_nan(self):
        # nan should be excluded from groupby
        ds = Dataset({'foo': ('x', [1, 2, 3, 4])},
                     {'bar': ('x', [1, 1, 2, np.nan])})
        actual = ds.groupby('bar').mean()
        expected = Dataset({'foo': ('bar', [1.5, 3]), 'bar': [1, 2]})
        self.assertDatasetIdentical(actual, expected)

    def test_resample_and_first(self):
        times = pd.date_range('2000-01-01', freq='6H', periods=10)
        ds = Dataset({'foo': (['time', 'x', 'y'], np.random.randn(10, 5, 3)),
                      'bar': ('time', np.random.randn(10), {'meta': 'data'}),
                      'time': times})

        actual = ds.resample('1D', dim='time', how='first')
        expected = ds.isel(time=[0, 4, 8])
        self.assertDatasetIdentical(expected, actual)

        # upsampling
        expected_time = pd.date_range('2000-01-01', freq='3H', periods=19)
        expected = ds.reindex(time=expected_time)
        for how in ['mean', 'sum', 'first', 'last', np.mean]:
            actual = ds.resample('3H', 'time', how=how)
            self.assertDatasetEqual(expected, actual)

    def test_to_array(self):
        ds = Dataset(OrderedDict([('a', 1), ('b', ('x', [1, 2, 3]))]),
                     coords={'c': 42}, attrs={'Conventions': 'None'})
        data = [[1, 1, 1], [1, 2, 3]]
        coords = {'x': range(3), 'c': 42, 'variable': ['a', 'b']}
        dims = ('variable', 'x')
        expected = DataArray(data, coords, dims, attrs=ds.attrs)
        actual = ds.to_array()
        self.assertDataArrayIdentical(expected, actual)

        actual = ds.to_array('abc', name='foo')
        expected = expected.rename({'variable': 'abc'}).rename('foo')
        self.assertDataArrayIdentical(expected, actual)

    def test_to_and_from_dataframe(self):
        x = np.random.randn(10)
        y = np.random.randn(10)
        t = list('abcdefghij')
        ds = Dataset(OrderedDict([('a', ('t', x)),
                                  ('b', ('t', y)),
                                  ('t', ('t', t))]))
        expected = pd.DataFrame(np.array([x, y]).T, columns=['a', 'b'],
                                index=pd.Index(t, name='t'))
        actual = ds.to_dataframe()
        # use the .equals method to check all DataFrame metadata
        assert expected.equals(actual), (expected, actual)

        # verify coords are included
        actual = ds.set_coords('b').to_dataframe()
        assert expected.equals(actual), (expected, actual)

        # check roundtrip
        self.assertDatasetIdentical(ds, Dataset.from_dataframe(actual))

        # test a case with a MultiIndex
        w = np.random.randn(2, 3)
        ds = Dataset({'w': (('x', 'y'), w)})
        ds['y'] = ('y', list('abc'))
        exp_index = pd.MultiIndex.from_arrays(
            [[0, 0, 0, 1, 1, 1], ['a', 'b', 'c', 'a', 'b', 'c']],
            names=['x', 'y'])
        expected = pd.DataFrame(w.reshape(-1), columns=['w'], index=exp_index)
        actual = ds.to_dataframe()
        self.assertTrue(expected.equals(actual))

        # check roundtrip
        self.assertDatasetIdentical(ds, Dataset.from_dataframe(actual))

        # check pathological cases
        df = pd.DataFrame([1])
        actual = Dataset.from_dataframe(df)
        expected = Dataset({0: ('index', [1])})
        self.assertDatasetIdentical(expected, actual)

        df = pd.DataFrame()
        actual = Dataset.from_dataframe(df)
        expected = Dataset(coords={'index':[]})
        self.assertDatasetIdentical(expected, actual)

        # GH697
        df = pd.DataFrame({'A' : []})
        actual = Dataset.from_dataframe(df)
        expected = Dataset({'A': DataArray([], dims=('index',))})
        self.assertDatasetIdentical(expected, actual)

        # regression test for GH278
        # use int64 to ensure consistent results for the pandas .equals method
        # on windows (which requires the same dtype)
        ds = Dataset({'x': pd.Index(['bar']),
                      'a': ('y', np.array([1], 'int64'))}).isel(x=0)
        # use .loc to ensure consistent results on Python 3
        actual = ds.to_dataframe().loc[:, ['a', 'x']]
        expected = pd.DataFrame([[1, 'bar']], index=pd.Index([0], name='y'),
                                columns=['a', 'x'])
        assert expected.equals(actual), (expected, actual)

        ds = Dataset({'x': np.array([0], 'int64'),
                      'y': np.array([1], 'int64')})
        actual = ds.to_dataframe()
        idx = pd.MultiIndex.from_arrays([[0], [1]], names=['x', 'y'])
        expected = pd.DataFrame([[]], index=idx)
        assert expected.equals(actual), (expected, actual)

    def test_from_dataframe_non_unique_columns(self):
        # regression test for GH449
        df = pd.DataFrame(np.zeros((2, 2)))
        df.columns = ['foo', 'foo']
        with self.assertRaisesRegexp(ValueError, 'non-unique columns'):
            Dataset.from_dataframe(df)

    def test_convert_dataframe_with_many_types_and_multiindex(self):
        # regression test for GH737
        df = pd.DataFrame({'a': list('abc'),
                           'b': list(range(1, 4)),
                           'c': np.arange(3, 6).astype('u1'),
                           'd': np.arange(4.0, 7.0, dtype='float64'),
                           'e': [True, False, True],
                           'f': pd.Categorical(list('abc')),
                           'g': pd.date_range('20130101', periods=3),
                           'h': pd.date_range('20130101',
                                              periods=3,
                                              tz='US/Eastern')})
        df.index = pd.MultiIndex.from_product([['a'], range(3)],
                                              names=['one', 'two'])
        roundtripped = Dataset.from_dataframe(df).to_dataframe()
        # we can't do perfectly, but we should be at least as faithful as
        # np.asarray
        expected = df.apply(np.asarray)
        if pd.__version__ < '0.17':
            # datetime with timezone dtype is not consistent on old pandas
            roundtripped = roundtripped.drop(['h'], axis=1)
            expected = expected.drop(['h'], axis=1)
        assert roundtripped.equals(expected)

    def test_pickle(self):
        data = create_test_data()
        roundtripped = pickle.loads(pickle.dumps(data))
        self.assertDatasetIdentical(data, roundtripped)
        # regression test for #167:
        self.assertEqual(data.dims, roundtripped.dims)

    def test_lazy_load(self):
        store = InaccessibleVariableDataStore()
        create_test_data().dump_to_store(store)

        for decode_cf in [True, False]:
            ds = open_dataset(store, decode_cf=decode_cf)
            with self.assertRaises(UnexpectedDataAccess):
                ds.load()
            with self.assertRaises(UnexpectedDataAccess):
                ds['var1'].values

            # these should not raise UnexpectedDataAccess:
            ds.isel(time=10)
            ds.isel(time=slice(10), dim1=[0]).isel(dim1=0, dim2=-1)

    def test_dropna(self):
        x = np.random.randn(4, 4)
        x[::2, 0] = np.nan
        y = np.random.randn(4)
        y[-1] = np.nan
        ds = Dataset({'foo': (('a', 'b'), x), 'bar': (('b', y))})

        expected = ds.isel(a=slice(1, None, 2))
        actual = ds.dropna('a')
        self.assertDatasetIdentical(actual, expected)

        expected = ds.isel(b=slice(1, 3))
        actual = ds.dropna('b')
        self.assertDatasetIdentical(actual, expected)

        actual = ds.dropna('b', subset=['foo', 'bar'])
        self.assertDatasetIdentical(actual, expected)

        expected = ds.isel(b=slice(1, None))
        actual = ds.dropna('b', subset=['foo'])
        self.assertDatasetIdentical(actual, expected)

        expected = ds.isel(b=slice(3))
        actual = ds.dropna('b', subset=['bar'])
        self.assertDatasetIdentical(actual, expected)

        actual = ds.dropna('a', subset=[])
        self.assertDatasetIdentical(actual, ds)

        actual = ds.dropna('a', subset=['bar'])
        self.assertDatasetIdentical(actual, ds)

        actual = ds.dropna('a', how='all')
        self.assertDatasetIdentical(actual, ds)

        actual = ds.dropna('b', how='all', subset=['bar'])
        expected = ds.isel(b=[0, 1, 2])
        self.assertDatasetIdentical(actual, expected)

        actual = ds.dropna('b', thresh=1, subset=['bar'])
        self.assertDatasetIdentical(actual, expected)

        actual = ds.dropna('b', thresh=2)
        self.assertDatasetIdentical(actual, ds)

        actual = ds.dropna('b', thresh=4)
        expected = ds.isel(b=[1, 2, 3])
        self.assertDatasetIdentical(actual, expected)

        actual = ds.dropna('a', thresh=3)
        expected = ds.isel(a=[1, 3])
        self.assertDatasetIdentical(actual, ds)

        with self.assertRaisesRegexp(ValueError, 'a single dataset dimension'):
            ds.dropna('foo')
        with self.assertRaisesRegexp(ValueError, 'invalid how'):
            ds.dropna('a', how='somehow')
        with self.assertRaisesRegexp(TypeError, 'must specify how or thresh'):
            ds.dropna('a', how=None)

    def test_fillna(self):
        ds = Dataset({'a': ('x', [np.nan, 1, np.nan, 3])})

        # fill with -1
        actual = ds.fillna(-1)
        expected = Dataset({'a': ('x', [-1, 1, -1, 3])})
        self.assertDatasetIdentical(expected, actual)

        actual = ds.fillna({'a': -1})
        self.assertDatasetIdentical(expected, actual)

        other = Dataset({'a': -1})
        actual = ds.fillna(other)
        self.assertDatasetIdentical(expected, actual)

        actual = ds.fillna({'a': other.a})
        self.assertDatasetIdentical(expected, actual)

        # fill with range(4)
        b = DataArray(range(4), dims='x')
        actual = ds.fillna(b)
        expected = b.rename('a').to_dataset()
        self.assertDatasetIdentical(expected, actual)

        actual = ds.fillna(expected)
        self.assertDatasetIdentical(expected, actual)

        actual = ds.fillna(range(4))
        self.assertDatasetIdentical(expected, actual)

        actual = ds.fillna(b[:3])
        self.assertDatasetIdentical(expected, actual)

        # okay to only include some data variables
        ds['b'] = np.nan
        actual = ds.fillna({'a': -1})
        expected = Dataset({'a': ('x', [-1, 1, -1, 3]), 'b': np.nan})
        self.assertDatasetIdentical(expected, actual)

        # but new data variables is not okay
        with self.assertRaisesRegexp(ValueError, 'must be contained'):
            ds.fillna({'x': 0})

        # empty argument should be OK
        result = ds.fillna({})
        self.assertDatasetIdentical(ds, result)

        result = ds.fillna(Dataset(coords={'c': 42}))
        expected = ds.assign_coords(c=42)
        self.assertDatasetIdentical(expected, result)

        # groupby
        expected = Dataset({'a': ('x', range(4))})
        for target in [ds, expected]:
            target.coords['b'] = ('x', [0, 0, 1, 1])
        actual = ds.groupby('b').fillna(DataArray([0, 2], dims='b'))
        self.assertDatasetIdentical(expected, actual)

        actual = ds.groupby('b').fillna(Dataset({'a': ('b', [0, 2])}))
        self.assertDatasetIdentical(expected, actual)

    def test_where(self):
        ds = Dataset({'a': ('x', range(5))})
        expected = Dataset({'a': ('x', [np.nan, np.nan, 2, 3, 4])})
        actual = ds.where(ds > 1)
        self.assertDatasetIdentical(expected, actual)

        actual = ds.where(ds.a > 1)
        self.assertDatasetIdentical(expected, actual)

        actual = ds.where(ds.a.values > 1)
        self.assertDatasetIdentical(expected, actual)

        actual = ds.where(True)
        self.assertDatasetIdentical(ds, actual)

        expected = ds.copy(deep=True)
        expected['a'].values = [np.nan] * 5
        actual = ds.where(False)
        self.assertDatasetIdentical(expected, actual)

        # 2d
        ds = Dataset({'a': (('x', 'y'), [[0, 1], [2, 3]])})
        expected = Dataset({'a': (('x', 'y'), [[np.nan, 1], [2, 3]])})
        actual = ds.where(ds > 0)
        self.assertDatasetIdentical(expected, actual)

        # groupby
        ds = Dataset({'a': ('x', range(5))}, {'c': ('x', [0, 0, 1, 1, 1])})
        cond = Dataset({'a': ('c', [True, False])})
        expected = ds.copy(deep=True)
        expected['a'].values = [0, 1] + [np.nan] * 3
        actual = ds.groupby('c').where(cond)
        self.assertDatasetIdentical(expected, actual)

    def test_reduce(self):
        data = create_test_data()

        self.assertEqual(len(data.mean().coords), 0)

        actual = data.max()
        expected = Dataset(dict((k, v.max())
                                for k, v in iteritems(data.data_vars)))
        self.assertDatasetEqual(expected, actual)

        self.assertDatasetEqual(data.min(dim=['dim1']),
                                data.min(dim='dim1'))

        for reduct, expected in [('dim2', ['dim1', 'dim3', 'time']),
                                 (['dim2', 'time'], ['dim1', 'dim3']),
                                 (('dim2', 'time'), ['dim1', 'dim3']),
                                 ((), ['dim1', 'dim2', 'dim3', 'time'])]:
            actual = data.min(dim=reduct).dims
            print(reduct, actual, expected)
            self.assertItemsEqual(actual, expected)

        self.assertDatasetEqual(data.mean(dim=[]), data)

    def test_reduce_bad_dim(self):
        data = create_test_data()
        with self.assertRaisesRegexp(ValueError, 'Dataset does not contain'):
            ds = data.mean(dim='bad_dim')

    def test_reduce_non_numeric(self):
        data1 = create_test_data(seed=44)
        data2 = create_test_data(seed=44)
        add_vars = {'var4': ['dim1', 'dim2']}
        for v, dims in sorted(add_vars.items()):
            size = tuple(data1.dims[d] for d in dims)
            data = np.random.random_integers(0, 100, size=size).astype(np.str_)
            data1[v] = (dims, data, {'foo': 'variable'})

        self.assertTrue('var4' not in data1.mean())
        self.assertDatasetEqual(data1.mean(), data2.mean())
        self.assertDatasetEqual(data1.mean(dim='dim1'),
                                data2.mean(dim='dim1'))

    def test_reduce_strings(self):
        expected = Dataset({'x': 'a'})
        ds = Dataset({'x': ('y', ['a', 'b'])})
        actual = ds.min()
        self.assertDatasetIdentical(expected, actual)

        expected = Dataset({'x': 'b'})
        actual = ds.max()
        self.assertDatasetIdentical(expected, actual)

        expected = Dataset({'x': 0})
        actual = ds.argmin()
        self.assertDatasetIdentical(expected, actual)

        expected = Dataset({'x': 1})
        actual = ds.argmax()
        self.assertDatasetIdentical(expected, actual)

        expected = Dataset({'x': b'a'})
        ds = Dataset({'x': ('y', np.array(['a', 'b'], 'S1'))})
        actual = ds.min()
        self.assertDatasetIdentical(expected, actual)

        expected = Dataset({'x': u'a'})
        ds = Dataset({'x': ('y', np.array(['a', 'b'], 'U1'))})
        actual = ds.min()
        self.assertDatasetIdentical(expected, actual)

    def test_reduce_dtypes(self):
        # regression test for GH342
        expected = Dataset({'x': 1})
        actual = Dataset({'x': True}).sum()
        self.assertDatasetIdentical(expected, actual)

        # regression test for GH505
        expected = Dataset({'x': 3})
        actual = Dataset({'x': ('y', np.array([1, 2], 'uint16'))}).sum()
        self.assertDatasetIdentical(expected, actual)

        expected = Dataset({'x': 1 + 1j})
        actual = Dataset({'x': ('y', [1, 1j])}).sum()
        self.assertDatasetIdentical(expected, actual)

    def test_reduce_keep_attrs(self):
        data = create_test_data()
        _attrs = {'attr1': 'value1', 'attr2': 2929}

        attrs = OrderedDict(_attrs)
        data.attrs = attrs

        # Test dropped attrs
        ds = data.mean()
        self.assertEqual(ds.attrs, {})
        for v in ds.data_vars.values():
            self.assertEqual(v.attrs, {})

        # Test kept attrs
        ds = data.mean(keep_attrs=True)
        self.assertEqual(ds.attrs, attrs)
        for k, v in ds.data_vars.items():
            self.assertEqual(v.attrs, data[k].attrs)

    def test_reduce_argmin(self):
        # regression test for #205
        ds = Dataset({'a': ('x', [0, 1])})
        expected = Dataset({'a': ([], 0)})
        actual = ds.argmin()
        self.assertDatasetIdentical(expected, actual)

        actual = ds.argmin('x')
        self.assertDatasetIdentical(expected, actual)

    def test_reduce_scalars(self):
        ds = Dataset({'x': ('a', [2, 2]), 'y': 2, 'z': ('b', [2])})
        expected = Dataset({'x': 0, 'y': 0, 'z': 0})
        actual = ds.var()
        self.assertDatasetIdentical(expected, actual)

    def test_reduce_only_one_axis(self):

        def mean_only_one_axis(x, axis):
            if not isinstance(axis, (int, np.integer)):
                raise TypeError('non-integer axis')
            return x.mean(axis)

        ds = Dataset({'a': (['x', 'y'], [[0, 1, 2, 3, 4]])})
        expected = Dataset({'a': ('x', [2])})
        actual = ds.reduce(mean_only_one_axis, 'y')
        self.assertDatasetIdentical(expected, actual)

        with self.assertRaisesRegexp(TypeError, 'non-integer axis'):
            ds.reduce(mean_only_one_axis)

        with self.assertRaisesRegexp(TypeError, 'non-integer axis'):
            ds.reduce(mean_only_one_axis, ['x', 'y'])

    def test_count(self):
        ds = Dataset({'x': ('a', [np.nan, 1]), 'y': 0, 'z': np.nan})
        expected = Dataset({'x': 1, 'y': 1, 'z': 0})
        actual = ds.count()
        self.assertDatasetIdentical(expected, actual)

    def test_apply(self):
        data = create_test_data()
        data.attrs['foo'] = 'bar'

        self.assertDatasetIdentical(data.apply(np.mean), data.mean())

        expected = data.mean(keep_attrs=True)
        actual = data.apply(lambda x: x.mean(keep_attrs=True), keep_attrs=True)
        self.assertDatasetIdentical(expected, actual)

        self.assertDatasetIdentical(data.apply(lambda x: x, keep_attrs=True),
                                    data.drop('time'))

        def scale(x, multiple=1):
            return multiple * x

        actual = data.apply(scale, multiple=2)
        self.assertDataArrayEqual(actual['var1'], 2 * data['var1'])
        self.assertDataArrayIdentical(actual['numbers'], data['numbers'])

        actual = data.apply(np.asarray)
        expected = data.drop('time') # time is not used on a data var
        self.assertDatasetEqual(expected, actual)

    def make_example_math_dataset(self):
        variables = OrderedDict(
            [('bar', ('x', np.arange(100, 400, 100))),
             ('foo', (('x', 'y'), 1.0 * np.arange(12).reshape(3, 4)))])
        coords = {'abc': ('x', ['a', 'b', 'c']),
                  'y': 10 * np.arange(4)}
        ds = Dataset(variables, coords)
        ds['foo'][0, 0] = np.nan
        return ds

    def test_dataset_number_math(self):
        ds = self.make_example_math_dataset()

        self.assertDatasetIdentical(ds, +ds)
        self.assertDatasetIdentical(ds, ds + 0)
        self.assertDatasetIdentical(ds, 0 + ds)
        self.assertDatasetIdentical(ds, ds + np.array(0))
        self.assertDatasetIdentical(ds, np.array(0) + ds)

        actual = ds.copy(deep=True)
        actual += 0
        self.assertDatasetIdentical(ds, actual)

    def test_unary_ops(self):
        ds = self.make_example_math_dataset()

        self.assertDatasetIdentical(ds.apply(abs), abs(ds))
        self.assertDatasetIdentical(ds.apply(lambda x: x + 4), ds + 4)

        for func in [lambda x: x.isnull(),
                     lambda x: x.round(),
                     lambda x: x.astype(int)]:
            self.assertDatasetIdentical(ds.apply(func), func(ds))

        self.assertDatasetIdentical(ds.isnull(), ~ds.notnull())

        # don't actually patch these methods in
        with self.assertRaises(AttributeError):
            ds.item
        with self.assertRaises(AttributeError):
            ds.searchsorted

    def test_dataset_array_math(self):
        ds = self.make_example_math_dataset()

        expected = ds.apply(lambda x: x - ds['foo'])
        self.assertDatasetIdentical(expected, ds - ds['foo'])
        self.assertDatasetIdentical(expected, -ds['foo'] + ds)
        self.assertDatasetIdentical(expected, ds - ds['foo'].variable)
        self.assertDatasetIdentical(expected, -ds['foo'].variable + ds)
        actual = ds.copy(deep=True)
        actual -= ds['foo']
        self.assertDatasetIdentical(expected, actual)

        expected = ds.apply(lambda x: x + ds['bar'])
        self.assertDatasetIdentical(expected, ds + ds['bar'])
        actual = ds.copy(deep=True)
        actual += ds['bar']
        self.assertDatasetIdentical(expected, actual)

        expected = Dataset({'bar': ds['bar'] + np.arange(3)})
        self.assertDatasetIdentical(expected, ds[['bar']] + np.arange(3))
        self.assertDatasetIdentical(expected, np.arange(3) + ds[['bar']])

    def test_dataset_dataset_math(self):
        ds = self.make_example_math_dataset()

        self.assertDatasetIdentical(ds, ds + 0 * ds)
        self.assertDatasetIdentical(ds, ds + {'foo': 0, 'bar': 0})

        expected = ds.apply(lambda x: 2 * x)
        self.assertDatasetIdentical(expected, 2 * ds)
        self.assertDatasetIdentical(expected, ds + ds)
        self.assertDatasetIdentical(expected, ds + ds.data_vars)
        self.assertDatasetIdentical(expected, ds + dict(ds.data_vars))

        actual = ds.copy(deep=True)
        expected_id = id(actual)
        actual += ds
        self.assertDatasetIdentical(expected, actual)
        self.assertEqual(expected_id, id(actual))

        self.assertDatasetIdentical(ds == ds, ds.notnull())

        subsampled = ds.isel(y=slice(2))
        expected = 2 * subsampled
        self.assertDatasetIdentical(expected, subsampled + ds)
        self.assertDatasetIdentical(expected, ds + subsampled)

    def test_dataset_math_auto_align(self):
        ds = self.make_example_math_dataset()
        subset = ds.isel(x=slice(2), y=[1, 3])
        expected = 2 * subset
        actual = ds + subset
        self.assertDatasetIdentical(expected, actual)


        actual = ds.isel(x=slice(1)) + ds.isel(x=slice(1, None))
        expected = ds.drop(ds.x, dim='x')
        self.assertDatasetEqual(actual, expected)

        actual = ds + ds[['bar']]
        expected = (2 * ds[['bar']]).merge(ds.coords)
        self.assertDatasetIdentical(expected, actual)

        self.assertDatasetIdentical(ds + Dataset(), ds.coords.to_dataset())
        self.assertDatasetIdentical(Dataset() + Dataset(), Dataset())

        ds2 = Dataset(coords={'bar': 42})
        self.assertDatasetIdentical(ds + ds2, ds.coords.merge(ds2))

        # maybe unary arithmetic with empty datasets should raise instead?
        self.assertDatasetIdentical(Dataset() + 1, Dataset())

        for other in [ds.isel(x=slice(2)), ds.bar.isel(x=slice(0))]:
            actual = ds.copy(deep=True)
            other = ds.isel(x=slice(2))
            actual += other
            expected = ds + other.reindex_like(ds)
            self.assertDatasetIdentical(expected, actual)

    def test_dataset_math_errors(self):
        ds = self.make_example_math_dataset()

        with self.assertRaises(TypeError):
            ds['foo'] += ds
        with self.assertRaises(TypeError):
            ds['foo'].variable += ds
        with self.assertRaisesRegexp(ValueError, 'must have the same'):
            ds += ds[['bar']]

        # verify we can rollback in-place operations if something goes wrong
        # nb. inplace datetime64 math actually will work with an integer array
        # but not floats thanks to numpy's inconsistent handling
        other = DataArray(np.datetime64('2000-01-01T12'), coords={'c': 2})
        actual = ds.copy(deep=True)
        with self.assertRaises(TypeError):
            actual += other
        self.assertDatasetIdentical(actual, ds)

    def test_dataset_transpose(self):
        ds = Dataset({'a': (('x', 'y'), np.random.randn(3, 4)),
                      'b': (('y', 'x'), np.random.randn(4, 3))})

        actual = ds.transpose()
        expected = ds.apply(lambda x: x.transpose())
        self.assertDatasetIdentical(expected, actual)

        actual = ds.T
        self.assertDatasetIdentical(expected, actual)

        actual = ds.transpose('x', 'y')
        expected = ds.apply(lambda x: x.transpose('x', 'y'))
        self.assertDatasetIdentical(expected, actual)

        ds = create_test_data()
        actual = ds.transpose()
        for k in ds:
            self.assertEqual(actual[k].dims[::-1], ds[k].dims)

        new_order = ('dim2', 'dim3', 'dim1', 'time')
        actual = ds.transpose(*new_order)
        for k in ds:
            expected_dims = tuple(d for d in new_order if d in ds[k].dims)
            self.assertEqual(actual[k].dims, expected_dims)

        with self.assertRaisesRegexp(ValueError, 'arguments to transpose'):
            ds.transpose('dim1', 'dim2', 'dim3')
        with self.assertRaisesRegexp(ValueError, 'arguments to transpose'):
            ds.transpose('dim1', 'dim2', 'dim3', 'time', 'extra_dim')

    def test_dataset_retains_period_index_on_transpose(self):

        ds = create_test_data()
        ds['time'] = pd.period_range('2000-01-01', periods=20)

        transposed = ds.transpose()

        self.assertIsInstance(transposed.time.to_index(), pd.PeriodIndex)

    def test_dataset_diff_n1_simple(self):
        ds = Dataset({'foo': ('x', [5, 5, 6, 6])})
        actual = ds.diff('x')
        expected = Dataset({'foo': ('x', [0, 1, 0])})
        expected.coords['x'].values = [1, 2, 3]
        self.assertDatasetEqual(expected, actual)

    def test_dataset_diff_n1_lower(self):
        ds = Dataset({'foo': ('x', [5, 5, 6, 6])})
        actual = ds.diff('x', label='lower')
        expected = Dataset({'foo': ('x', [0, 1, 0])})
        expected.coords['x'].values = [0, 1, 2]
        self.assertDatasetEqual(expected, actual)

    def test_dataset_diff_n1(self):
        ds = create_test_data(seed=1)
        actual = ds.diff('dim2')
        expected = dict()
        expected['var1'] = DataArray(np.diff(ds['var1'].values, axis=1),
                                     [ds['dim1'].values,
                                      ds['dim2'].values[1:]],
                                     ['dim1', 'dim2'])
        expected['var2'] = DataArray(np.diff(ds['var2'].values, axis=1),
                                     [ds['dim1'].values,
                                      ds['dim2'].values[1:]],
                                     ['dim1', 'dim2'])
        expected['var3'] = ds['var3']
        expected = Dataset(expected, coords={'time': ds['time'].values})
        expected.coords['numbers'] = ('dim3', ds['numbers'].values)
        self.assertDatasetEqual(expected, actual)

    def test_dataset_diff_n2(self):
        ds = create_test_data(seed=1)
        actual = ds.diff('dim2', n=2)
        expected = dict()
        expected['var1'] = DataArray(np.diff(ds['var1'].values, axis=1, n=2),
                                     [ds['dim1'].values,
                                      ds['dim2'].values[2:]],
                                     ['dim1', 'dim2'])
        expected['var2'] = DataArray(np.diff(ds['var2'].values, axis=1, n=2),
                                     [ds['dim1'].values,
                                      ds['dim2'].values[2:]],
                                     ['dim1', 'dim2'])
        expected['var3'] = ds['var3']
        expected = Dataset(expected, coords={'time': ds['time'].values})
        expected.coords['numbers'] = ('dim3', ds['numbers'].values)
        self.assertDatasetEqual(expected, actual)

    def test_dataset_diff_exception_n_neg(self):
        ds = create_test_data(seed=1)
        with self.assertRaisesRegexp(ValueError, 'must be non-negative'):
            ds.diff('dim2', n=-1)

    def test_dataset_diff_exception_label_str(self):
        ds = create_test_data(seed=1)
        with self.assertRaisesRegexp(ValueError, '\'label\' argument has to'):
            ds.diff('dim2', label='raise_me')

    def test_shift(self):
        coords = {'bar': ('x', list('abc')), 'x': [-4, 3, 2]}
        attrs = {'meta': 'data'}
        ds = Dataset({'foo': ('x', [1, 2, 3])}, coords, attrs)
        actual = ds.shift(x=1)
        expected = Dataset({'foo': ('x', [np.nan, 1, 2])}, coords, attrs)
        self.assertDatasetIdentical(expected, actual)

        with self.assertRaisesRegexp(ValueError, 'dimensions'):
            ds.shift(foo=123)

    def test_roll(self):
        coords = {'bar': ('x', list('abc')), 'x': [-4, 3, 2]}
        attrs = {'meta': 'data'}
        ds = Dataset({'foo': ('x', [1, 2, 3])}, coords, attrs)
        actual = ds.roll(x=1)

        ex_coords = {'bar': ('x', list('cab')), 'x': [2, -4, 3]}
        expected = Dataset({'foo': ('x', [3, 1, 2])}, ex_coords, attrs)
        self.assertDatasetIdentical(expected, actual)

        with self.assertRaisesRegexp(ValueError, 'dimensions'):
            ds.roll(foo=123)

    def test_real_and_imag(self):
        attrs = {'foo': 'bar'}
        ds = Dataset({'x': ((), 1 + 2j, attrs)}, attrs=attrs)

        expected_re = Dataset({'x': ((), 1, attrs)}, attrs=attrs)
        self.assertDatasetIdentical(ds.real, expected_re)

        expected_im = Dataset({'x': ((), 2, attrs)}, attrs=attrs)
        self.assertDatasetIdentical(ds.imag, expected_im)

    def test_setattr_raises(self):
        ds = Dataset({}, coords={'scalar': 1}, attrs={'foo': 'bar'})
        with self.assertRaisesRegexp(AttributeError, 'cannot set attr'):
            ds.scalar = 2
        with self.assertRaisesRegexp(AttributeError, 'cannot set attr'):
            ds.foo = 2
        with self.assertRaisesRegexp(AttributeError, 'cannot set attr'):
            ds.other = 2
