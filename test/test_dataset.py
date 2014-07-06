from copy import copy, deepcopy
from textwrap import dedent
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import pandas as pd

from xray import (Dataset, DataArray, Index, Variable,
                  backends, utils, align, indexing)
from xray.pycompat import iteritems, OrderedDict

from . import TestCase


_dims = {'dim1': 100, 'dim2': 50, 'dim3': 10}
_vars = {'var1': ['dim1', 'dim2'],
         'var2': ['dim1', 'dim2'],
         'var3': ['dim3', 'dim1'],
         }
_testvar = sorted(_vars.keys())[0]
_testdim = sorted(_dims.keys())[0]


def create_test_data(seed=None):
    rs = np.random.RandomState(seed)
    obj = Dataset()
    obj['time'] = ('time', pd.date_range('2000-01-01', periods=20))
    obj['dim1'] = ('dim1', np.arange(_dims['dim1']))
    obj['dim2'] = ('dim2', 0.5 * np.arange(_dims['dim2']))
    obj['dim3'] = ('dim3', list('abcdefghij'))
    for v, dims in sorted(_vars.items()):
        data = rs.normal(size=tuple(_dims[d] for d in dims))
        obj[v] = (dims, data, {'foo': 'variable'})
    return obj


class UnexpectedDataAccess(Exception):
    pass


class InaccessibleArray(utils.NDArrayMixin):
    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        raise UnexpectedDataAccess("Tried accessing data")


class InaccessibleVariableDataStore(backends.InMemoryDataStore):
    def __init__(self):
        self.dimensions = OrderedDict()
        self._variables = OrderedDict()
        self.attrs = OrderedDict()

    def set_variable(self, name, variable):
        self._variables[name] = variable
        return self._variables[name]

    def open_store_variable(self, var):
        data = indexing.LazilyIndexedArray(InaccessibleArray(var.values))
        return Variable(var.dimensions, data, var.attrs)

    @property
    def store_variables(self):
        return self._variables


class TestDataset(TestCase):
    def test_repr(self):
        data = create_test_data()
        expected = dedent("""
        <xray.Dataset>
        Dimensions:  (dim1: 100, dim2: 50, dim3: 10, time: 20)
        Indexes:
            dim1          X
            dim2                     X
            dim3                               X
            time                                         X
        Non-indexes:
            var1          0          1
            var2          0          1
            var3          1                    0
        Attributes:
            Empty
        """).strip()
        actual = '\n'.join(x.rstrip() for x in repr(data).split('\n'))
        self.assertEqual(expected, actual)

        expected = dedent("""
        <xray.Dataset>
        Dimensions:  ()
        Indexes:
            None
        Non-indexes:
            None
        Attributes:
            Empty
        """).strip()
        actual = '\n'.join(x.rstrip() for x in repr(Dataset()).split('\n'))
        self.assertEqual(expected, actual)

    def test_init(self):
        var1 = Variable('x', 2 * np.arange(100))
        var2 = Variable('x', np.arange(1000))
        var3 = Variable(['x', 'y'], np.arange(1000).reshape(100, 10))
        with self.assertRaisesRegexp(ValueError, 'but already exists'):
            Dataset({'a': var1, 'b': var2})
        with self.assertRaisesRegexp(ValueError, 'must be defined with 1-d'):
            Dataset({'a': var1, 'x': var3})
        # verify handling of DataArrays
        expected = Dataset({'x': var1, 'z': var3})
        actual = Dataset({'z': expected['z']})
        self.assertDatasetIdentical(expected, actual)

    def test_variable(self):
        a = Dataset()
        d = np.random.random((10, 3))
        a['foo'] = (('time', 'x',), d)
        self.assertTrue('foo' in a.variables)
        self.assertTrue('foo' in a)
        a['bar'] = (('time', 'x',), d)
        # order of creation is preserved
        self.assertEqual(list(a.variables.keys()),  ['foo', 'time', 'x', 'bar'])
        self.assertTrue(all([a.variables['foo'][i].values == d[i]
                             for i in np.ndindex(*d.shape)]))
        # try to add variable with dim (10,3) with data that's (3,10)
        with self.assertRaises(ValueError):
            a['qux'] = (('time', 'x'), d.T)

    def test_indexes_create(self):
        a = Dataset()
        vec = np.random.random((10,))
        attributes = {'foo': 'bar'}
        a['x'] = ('x', vec, attributes)
        self.assertTrue('x' in a.indexes)
        self.assertIsInstance(a.indexes['x'].as_index, pd.Index)
        self.assertVariableIdentical(a.indexes['x'], a.variables['x'])
        b = Dataset()
        b['x'] = ('x', vec, attributes)
        self.assertVariableIdentical(a['x'], b['x'])
        self.assertEqual(a.dimensions, b.dimensions)
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
        self.assertTrue('y' not in a.dimensions)

    def test_indexes_properties(self):
        data = Dataset({'x': ('x', [-1, -2]),
                        'y': ('y', [0, 1, 2]),
                        'foo': (['x', 'y'], np.random.randn(2, 3))})

        self.assertEquals(2, len(data.indexes))

        self.assertEquals(set(['x', 'y']), set(data.indexes))

        self.assertVariableIdentical(data.indexes['x'], data['x'].variable)
        self.assertVariableIdentical(data.indexes['y'], data['y'].variable)

        self.assertIn('x', data.indexes)
        self.assertNotIn(0, data.indexes)
        self.assertNotIn('foo', data.indexes)

        with self.assertRaises(KeyError):
            data.indexes['foo']
        with self.assertRaises(KeyError):
            data.indexes[0]

        expected = dedent("""\
        x: Int64Index([-1, -2], dtype='int64')
        y: Int64Index([0, 1, 2], dtype='int64')""")
        actual = repr(data.indexes)
        self.assertEquals(expected, actual)

    def test_equals_and_identical(self):
        data = create_test_data(seed=42)
        self.assertTrue(data.equals(data))
        self.assertTrue(data.identical(data))

        data2 = create_test_data(seed=42)
        data2.attrs['foobar'] = 'baz'
        self.assertTrue(data.equals(data2))
        self.assertTrue(data == data2)
        self.assertFalse(data.identical(data2))

        del data2['time']
        self.assertFalse(data.equals(data2))
        self.assertTrue(data != data2)

        data = create_test_data(seed=42).rename({'var1': None})
        self.assertTrue(data.equals(data))
        self.assertTrue(data.identical(data))

    def test_attrs(self):
        data = create_test_data(seed=42)
        data.attrs = {'foobar': 'baz'}
        self.assertTrue(data.attrs['foobar'], 'baz')
        self.assertIsInstance(data.attrs, OrderedDict)

    def test_isel(self):
        data = create_test_data()
        slicers = {'dim1': slice(None, None, 2), 'dim2': slice(0, 2)}
        ret = data.isel(**slicers)

        # Verify that only the specified dimension was altered
        self.assertItemsEqual(data.dimensions, ret.dimensions)
        for d in data.dimensions:
            if d in slicers:
                self.assertEqual(ret.dimensions[d],
                                 np.arange(data.dimensions[d])[slicers[d]].size)
            else:
                self.assertEqual(data.dimensions[d], ret.dimensions[d])
        # Verify that the data is what we expect
        for v in data.variables:
            self.assertEqual(data[v].dimensions, ret[v].dimensions)
            self.assertEqual(data[v].attrs, ret[v].attrs)
            slice_list = [slice(None)] * data[v].values.ndim
            for d, s in iteritems(slicers):
                if d in data[v].dimensions:
                    inds = np.nonzero(np.array(data[v].dimensions) == d)[0]
                    for ind in inds:
                        slice_list[ind] = s
            expected = data[v].values[slice_list]
            actual = ret[v].values
            np.testing.assert_array_equal(expected, actual)

        with self.assertRaises(ValueError):
            data.isel(not_a_dim=slice(0, 2))

        ret = data.isel(dim1=0)
        self.assertEqual({'time': 20, 'dim2': 50, 'dim3': 10}, ret.dimensions)
        self.assertItemsEqual(list(data.nonindexes) + ['dim1'],
                              ret.nonindexes)

        ret = data.isel(time=slice(2), dim1=0, dim2=slice(5))
        self.assertEqual({'time': 2, 'dim2': 5, 'dim3': 10}, ret.dimensions)
        self.assertItemsEqual(list(data.nonindexes) + ['dim1'],
                              ret.nonindexes)

        ret = data.isel(time=0, dim1=0, dim2=slice(5))
        self.assertItemsEqual({'dim2': 5, 'dim3': 10}, ret.dimensions)
        self.assertItemsEqual(list(data.nonindexes) + ['dim1', 'time'],
                              ret.nonindexes)

    def test_sel(self):
        data = create_test_data()
        int_slicers = {'dim1': slice(None, None, 2),
                       'dim2': slice(2),
                       'dim3': slice(3)}
        loc_slicers = {'dim1': slice(None, None, 2),
                       'dim2': slice(0, 0.5),
                       'dim3': slice('a', 'c')}
        self.assertEqual(data.isel(**int_slicers),
                         data.sel(**loc_slicers))
        data['time'] = ('time', pd.date_range('2000-01-01', periods=20))
        self.assertEqual(data.isel(time=0),
                         data.sel(time='2000-01-01'))
        self.assertEqual(data.isel(time=slice(10)),
                         data.sel(time=slice('2000-01-01',
                                                   '2000-01-10')))
        self.assertEqual(data, data.sel(time=slice('1999', '2005')))
        self.assertEqual(data.isel(time=slice(3)),
                         data.sel(
                            time=pd.date_range('2000-01-01', periods=3)))

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

        actual = data.reindex(dim1=data['dim1'][:10].as_index)
        self.assertDatasetIdentical(actual, expected)

    def test_align(self):
        left = create_test_data()
        right = left.copy(deep=True)
        right['dim3'] = ('dim3', list('cdefghijkl'))
        right['var3'][:-2] = right['var3'][2:]
        right['var3'][-2:] = np.random.randn(*right['var3'][-2:].shape)

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

    def test_select_vars(self):
        data = create_test_data()
        ret = data.select_vars(_testvar)
        self.assertVariableEqual(data[_testvar], ret[_testvar])
        self.assertTrue(sorted(_vars.keys())[1] not in ret.variables)
        self.assertRaises(ValueError, data.select_vars, (_testvar, 'not_a_var'))

    def test_drop_vars(self):
        data = create_test_data()

        self.assertEqual(data, data.drop_vars())

        expected = Dataset(dict((k, data[k]) for k in data if k != 'time'))
        actual = data.drop_vars('time')
        self.assertEqual(expected, actual)

        expected = Dataset(dict((k, data[k]) for k in ['dim2', 'dim3', 'time']))
        actual = data.drop_vars('dim1')
        self.assertEqual(expected, actual)

        with self.assertRaisesRegexp(ValueError, 'does not exist in this'):
            data.drop_vars('not_found_here')

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
            dims = list(v.dimensions)
            for name, newname in iteritems(newnames):
                if name in dims:
                    dims[dims.index(name)] = newname

            self.assertVariableEqual(Variable(dims, v.values, v.attrs),
                                     renamed.variables[k])
            self.assertEqual(v.encoding, renamed.variables[k].encoding)
            self.assertEqual(type(v), type(renamed.variables[k]))

        self.assertTrue('var1' not in renamed.variables)
        self.assertTrue('dim2' not in renamed.variables)

        with self.assertRaisesRegexp(ValueError, "cannot rename 'not_a_var'"):
            data.rename({'not_a_var': 'nada'})

        # verify that we can rename a variable without accessing the data
        var1 = data['var1']
        data['var1'] = (var1.dimensions, InaccessibleArray(var1.values))
        renamed = data.rename(newnames)
        with self.assertRaises(UnexpectedDataAccess):
            renamed['renamed_var1'].values

    def test_update(self):
        data = create_test_data(seed=0)
        var2 = Variable('dim1', np.arange(100))
        actual = data.update({'var2': var2})
        expected_vars = OrderedDict(create_test_data(seed=0).variables)
        expected_vars['var2'] = var2
        expected = Dataset(expected_vars)
        self.assertDatasetIdentical(expected, actual)
        # test in-place
        data2 = data.update(data, inplace=True)
        self.assertIs(data2, data)
        data2 = data.update(data, inplace=False)
        self.assertIsNot(data2, data)

    def test_merge(self):
        data = create_test_data()
        ds1 = data.select_vars('var1')
        ds2 = data.select_vars('var3')
        expected = data.select_vars('var1', 'var3')
        actual = ds1.merge(ds2)
        self.assertEqual(expected, actual)
        with self.assertRaises(ValueError):
            ds1.merge(ds2.isel(dim1=slice(2)))
        with self.assertRaises(ValueError):
            ds1.merge(ds2.rename({'var3': 'var1'}))

    def test_getitem(self):
        data = create_test_data()
        self.assertIsInstance(data['var1'], DataArray)
        self.assertVariableEqual(data['var1'], data.variables['var1'])
        self.assertDatasetIdentical(data['var1'].dataset, data)

    def test_virtual_variables(self):
        # access virtual variables
        data = create_test_data()
        self.assertVariableEqual(data['time.dayofyear'],
                                 Variable('time', 1 + np.arange(20)))
        self.assertArrayEqual(data['time.month'].values,
                              data.variables['time'].as_index.month)
        self.assertArrayEqual(data['time.season'].values, 1)
        # test virtual variable math
        self.assertArrayEqual(data['time.dayofyear'] + 1, 2 + np.arange(20))
        self.assertArrayEqual(np.sin(data['time.dayofyear']),
                              np.sin(1 + np.arange(20)))
        # test slicing the virtual variable -- it should still be virtual
        actual = data['time.dayofyear'][:10].dataset
        expected = data.isel(time=slice(10))
        self.assertDatasetIdentical(expected, actual)

    def test_slice_virtual_variable(self):
        data = create_test_data()
        self.assertVariableEqual(data['time.dayofyear'][:10],
                                 Variable(['time'], 1 + np.arange(10)))
        self.assertVariableEqual(data['time.dayofyear'][0], Variable([], 1))

    def test_setitem(self):
        # assign a variable
        var = Variable(['dim1'], np.random.randn(100))
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
        # assign an array
        with self.assertRaisesRegexp(TypeError, 'variables must be of type'):
            data2['C'] = var.values
        # override an existing value
        data1['A'] = 3 * data2['A']
        self.assertVariableEqual(data1['A'], 3 * data2['A'])
        # can't resize a used dimension
        with self.assertRaisesRegexp(ValueError, 'but already exists with'):
            data1['dim1'] = data1['dim1'][:5]

    def test_delitem(self):
        data = create_test_data()
        all_items = set(['time', 'dim1', 'dim2', 'dim3', 'var1', 'var2', 'var3'])
        self.assertItemsEqual(data, all_items)
        del data['var1']
        self.assertItemsEqual(data, all_items - set(['var1']))
        del data['dim1']
        self.assertItemsEqual(data, set(['time', 'dim2', 'dim3']))

    def test_squeeze(self):
        data = Dataset({'foo': (['x', 'y', 'z'], [[[1], [2]]])})
        for args in [[], [['x']], [['x', 'z']]]:
            def get_args(v):
                return [set(args[0]) & set(v.dimensions)] if args else []
            expected = Dataset(dict((k, v.squeeze(*get_args(v)))
                                    for k, v in iteritems(data.variables)))
            self.assertDatasetIdentical(expected, data.squeeze(*args))
        # invalid squeeze
        with self.assertRaisesRegexp(ValueError, 'cannot select a dimension'):
            data.squeeze('y')

    def test_groupby(self):
        data = Dataset({'x': ('x', list('abc')),
                        'c': ('x', [0, 1, 0]),
                        'z': (['x', 'y'], np.random.randn(3, 5))})
        groupby = data.groupby('x')
        self.assertEqual(len(groupby), 3)
        expected_groups = {'a': 0, 'b': 1, 'c': 2}
        self.assertEqual(groupby.groups, expected_groups)
        expected_items = [('a', data.isel(x=0)),
                          ('b', data.isel(x=1)),
                          ('c', data.isel(x=2))]
        self.assertEqual(list(groupby), expected_items)

        identity = lambda x: x
        for k in ['x', 'c', 'y']:
            actual = data.groupby(k, squeeze=False).apply(identity)
            self.assertEqual(data, actual)

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
        with self.assertRaisesRegexp(ValueError, "must have a 'dimensions'"):
            data.groupby(data.indexes['dim1'].as_index)

    def test_groupby_reduce(self):
        data = Dataset({'xy': (['x', 'y'], np.random.randn(3, 4)),
                        'xonly': ('x', np.random.randn(3)),
                        'yonly': ('y', np.random.randn(4)),
                        'letters': ('y', ['a', 'a', 'b', 'b'])})

        expected = data.mean('y')
        actual = data.groupby('x').mean()
        self.assertDatasetAllClose(expected, actual)

        actual = data.groupby('x').mean('y')
        self.assertDatasetAllClose(expected, actual)

        expected = Dataset({'xy': data['xy'].groupby('letters').mean(),
                            'xonly': data['xonly'].mean(),
                            'yonly': data['yonly'].groupby('letters').mean()})
        actual = data.groupby('letters').mean()
        self.assertDatasetAllClose(expected, actual)

    def test_concat(self):
        data = create_test_data()

        split_data = [data.isel(dim1=slice(10)),
                      data.isel(dim1=slice(10, None))]
        self.assertDatasetIdentical(data, Dataset.concat(split_data, 'dim1'))

        def rectify_dim_order(dataset):
            # return a new dataset with all variable dimensions tranposed into
            # the order in which they are found in `data`
            return Dataset(dict((k, v.transpose(*data[k].dimensions))
                                for k, v in iteritems(dataset.variables)),
                           dataset.attrs)

        for dim in ['dim1', 'dim2', 'dim3']:
            datasets = [g for _, g in data.groupby(dim, squeeze=False)]
            self.assertDatasetIdentical(data, Dataset.concat(datasets, dim))
            self.assertDatasetIdentical(
                data, Dataset.concat(datasets, data[dim]))
            self.assertDatasetIdentical(
                data, Dataset.concat(datasets, data[dim], mode='minimal'))

            datasets = [g for _, g in data.groupby(dim, squeeze=True)]
            concat_over = [k for k, v in iteritems(data.variables)
                           if dim in v.dimensions and k != dim]
            actual = Dataset.concat(datasets, data[dim],
                                    concat_over=concat_over)
            self.assertDatasetIdentical(data, rectify_dim_order(actual))

            actual = Dataset.concat(datasets, data[dim], mode='different')
            self.assertDatasetIdentical(data, rectify_dim_order(actual))

        # Now add a new variable that doesn't depend on any of the current
        # dims and make sure the mode argument behaves as expected
        data['var4'] = ('dim4', np.arange(data.dimensions['dim3']))
        for dim in ['dim1', 'dim2', 'dim3']:
            datasets = [g for _, g in data.groupby(dim, squeeze=False)]
            actual = Dataset.concat(datasets, data[dim], mode='all')
            expected = np.array([data['var4'].values
                                 for _ in range(data.dimensions[dim])])
            self.assertArrayEqual(actual['var4'].values, expected)

            actual = Dataset.concat(datasets, data[dim], mode='different')
            self.assertDataArrayEqual(data['var4'], actual['var4'])
            actual = Dataset.concat(datasets, data[dim], mode='minimal')
            self.assertDataArrayEqual(data['var4'], actual['var4'])

        # verify that the dimension argument takes precedence over
        # concatenating dataset variables of the same name
        dimension = (2 * data['dim1']).rename('dim1')
        datasets = [g for _, g in data.groupby('dim1', squeeze=False)]
        expected = data.copy()
        expected['dim1'] = dimension
        self.assertDatasetIdentical(
            expected, Dataset.concat(datasets, dimension))

        # TODO: factor this into several distinct tests
        data = create_test_data()
        split_data = [data.isel(dim1=slice(10)),
                      data.isel(dim1=slice(10, None))]

        with self.assertRaisesRegexp(ValueError, 'must supply at least one'):
            Dataset.concat([], 'dim1')

        with self.assertRaisesRegexp(ValueError, 'not all elements in'):
            Dataset.concat(split_data, 'dim1', concat_over=['not_found'])

        with self.assertRaisesRegexp(ValueError, 'global attributes not'):
            data0, data1 = deepcopy(split_data)
            data1.attrs['foo'] = 'bar'
            Dataset.concat([data0, data1], 'dim1', compat='identical')
        self.assertDatasetIdentical(
            data, Dataset.concat([data0, data1], 'dim1', compat='equals'))

        with self.assertRaisesRegexp(ValueError, 'encountered unexpected'):
            data0, data1 = deepcopy(split_data)
            data1['foo'] = ('bar', np.random.randn(10))
            Dataset.concat([data0, data1], 'dim1')

        with self.assertRaisesRegexp(ValueError, 'not equal across datasets'):
            data0, data1 = deepcopy(split_data)
            data1['dim2'] = 2 * data1['dim2']
            Dataset.concat([data0, data1], 'dim1')

    def test_to_and_from_dataframe(self):
        x = np.random.randn(10)
        y = np.random.randn(10)
        t = list('abcdefghij')
        ds = Dataset(OrderedDict([('a', ('t', x)),
                                  ('b', ('t', y)),
                                  ('t', ('t', t)),
                                 ]))
        expected = pd.DataFrame(np.array([x, y]).T, columns=['a', 'b'],
                                index=pd.Index(t, name='t'))
        actual = ds.to_dataframe()
        # use the .equals method to check all DataFrame metadata
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

    def test_pickle(self):
        data = create_test_data()
        roundtripped = pickle.loads(pickle.dumps(data))
        self.assertDatasetIdentical(data, roundtripped)
        # regression test for #167:
        self.assertEqual(data.dimensions, roundtripped.dimensions)

    def test_lazy_load(self):
        store = InaccessibleVariableDataStore()
        create_test_data().dump_to_store(store)

        for decode_cf in [False, True]:
            ds = Dataset.load_store(store, decode_cf=decode_cf)
            with self.assertRaises(UnexpectedDataAccess):
                ds.load_data()
            with self.assertRaises(UnexpectedDataAccess):
                ds['var1'].values

            # these should not raise UnexpectedDataAccess:
            ds.isel(time=10)
            ds.isel(time=slice(10), dim1=[0]).isel(dim1=0, dim2=-1)

    def test_reduce(self):
        data = create_test_data()

        self.assertEqual(len(data.mean().indexes), 0)

        expected = data.max()
        for var in data.nonindexes:
            expected = data[var].max()
            actual = expected[var]
            self.assertDataArrayEqual(expected, actual)

        self.assertDatasetEqual(data.min(dimension=['dim1']),
                                data.min(dimension='dim1'))

        for reduct, expected in [('dim2', ['dim1', 'dim3', 'time']),
                                 (['dim2', 'time'], ['dim1', 'dim3']),
                                 (('dim2', 'time'), ['dim1', 'dim3']),
                                 ((), ['dim1', 'dim2', 'dim3', 'time'])]:
            actual = data.min(dimension=reduct).dimensions
            print(reduct, actual, expected)
            self.assertItemsEqual(actual, expected)

        self.assertDatasetEqual(data.mean(dimension=[]), data)

    def test_reduce_bad_dimension(self):
        data = create_test_data()
        with self.assertRaisesRegexp(ValueError, 'Dataset does not contain'):
            ds = data.mean(dimension='bad_dim')

    def test_reduce_non_numeric(self):
        data1 = create_test_data(seed=44)
        data2 = create_test_data(seed=44)
        add_vars = {'var4': ['dim1', 'dim2']}
        for v, dims in sorted(add_vars.items()):
            data = np.random.random_integers(0, 100, size=tuple(_dims[d] for d in dims)).astype(np.str_)
            data1[v] = (dims, data, {'foo': 'variable'})

        self.assertTrue('var4' not in data1.mean())
        self.assertDatasetEqual(data1.mean(), data2.mean())
        self.assertDatasetEqual(data1.mean(dimension='dim1'),
                                data2.mean(dimension='dim1'))

    def test_reduce_keep_attrs(self):
        data = create_test_data()
        _attrs = {'attr1': 'value1', 'attr2': 2929}

        attrs = OrderedDict(_attrs)
        data.attrs = attrs

        # Test dropped attrs
        ds = data.mean()
        self.assertEqual(len(ds.attrs), 0)
        self.assertEqual(ds.attrs, OrderedDict())

        # Test kept attrs
        ds = data.mean(keep_attrs=True)
        self.assertEqual(len(ds.attrs), len(_attrs))
        self.assertTrue(ds.attrs, attrs)
