import mock
import unittest

from collections import OrderedDict
from copy import deepcopy
from textwrap import dedent

import numpy as np
import pandas as pd

from xray import (Dataset, DataArray, XArray, backends, open_dataset, utils,
                  align, conventions)

from . import TestCase, requires_netCDF4


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


def _data_fail(*args, **kwdargs):
    raise UnexpectedDataAccess("Tried Accessing Data")


class InaccessibleArray(XArray):
    def __init__(self, dims, data, *args, **kwargs):
        XArray.__init__(self, dims, data, *args, **kwargs)
        # make sure the only operations on _data
        # are to check ndim, dtype, size and shape
        self._data = mock.Mock()
        self._data.ndim = data.ndim
        self._data.dtype = data.dtype
        self._data.size = data.size
        self._data.shape = data.shape
        # fail if the actual data is accessed from _data
        self._data.__array__ = _data_fail
        self._data.__getitem__ = _data_fail

    @property
    def data(self):
        _data_fail()


class InaccessibleVariableDataStore(backends.InMemoryDataStore):
    def __init__(self):
        self.dimensions = OrderedDict()
        self._variables = OrderedDict()
        self.attributes = OrderedDict()

    def set_variable(self, name, variable):
        self._variables[name] = variable
        return self._variables[name]

    @property
    def variables(self):
        return utils.FrozenOrderedDict(
            (k, InaccessibleArray(v.dimensions, v.data, v.attributes))
            for k, v in self._variables.iteritems())


class TestDataset(TestCase):
    def test_repr(self):
        data = create_test_data()
        expected = dedent("""
        <xray.Dataset>
        Dimensions:     (dim1: 100, dim2: 50, dim3: 10, time: 20)
        Coordinates:
            dim1             X
            dim2                        X
            dim3                                  X
            time                                            X
        Noncoordinates:
            var1             0          1
            var2             0          1
            var3             1                    0
        Attributes:
            Empty
        """).strip()
        actual = '\n'.join(x.rstrip() for x in repr(data).split('\n'))
        self.assertEqual(expected, actual)

        expected = dedent("""
        <xray.Dataset>
        Dimensions:     ()
        Coordinates:
            None
        Noncoordinates:
            None
        Attributes:
            Empty
        """).strip()
        actual = '\n'.join(x.rstrip() for x in repr(Dataset()).split('\n'))
        self.assertEqual(expected, actual)

    def test_init(self):
        var1 = XArray('x', 2 * np.arange(100))
        var2 = XArray('x', np.arange(1000))
        var3 = XArray(['x', 'y'], np.arange(1000).reshape(100, 10))
        with self.assertRaisesRegexp(ValueError, 'but already exists'):
            Dataset({'a': var1, 'b': var2})
        with self.assertRaisesRegexp(ValueError, 'must be defined with 1-d'):
            Dataset({'a': var1, 'x': var3})
        # verify handling of DataArrays
        expected = Dataset({'x': var1, 'z': var3})
        actual = Dataset({'z': expected['z']})
        self.assertDatasetEqual(expected, actual)

    def test_groupby(self):
        data = create_test_data()
        for n, (t, sub) in enumerate(list(data.groupby('dim1'))[:3]):
            self.assertEqual(data['dim1'][n], t)
            self.assertXArrayEqual(data['var1'][n], sub['var1'])
            self.assertXArrayEqual(data['var2'][n], sub['var2'])
            self.assertXArrayEqual(data['var3'][:, n], sub['var3'])

    def test_variable(self):
        a = Dataset()
        d = np.random.random((10, 3))
        a['foo'] = (('time', 'x',), d)
        self.assertTrue('foo' in a.variables)
        self.assertTrue('foo' in a)
        a['bar'] = (('time', 'x',), d)
        # order of creation is preserved
        self.assertTrue(a.variables.keys() == ['foo', 'time', 'x', 'bar'])
        self.assertTrue(all([a.variables['foo'][i].data == d[i]
                             for i in np.ndindex(*d.shape)]))
        # try to add variable with dim (10,3) with data that's (3,10)
        with self.assertRaises(ValueError):
            a['qux'] = (('time', 'x'), d.T)

    def test_coordinate(self):
        a = Dataset()
        vec = np.random.random((10,))
        attributes = {'foo': 'bar'}
        a['x'] = ('x', vec, attributes)
        self.assertTrue('x' in a.coordinates)
        self.assertIsInstance(a.coordinates['x'].index, pd.Index)
        self.assertXArrayEqual(a.coordinates['x'], a.variables['x'])
        b = Dataset()
        b['x'] = ('x', vec, attributes)
        self.assertXArrayEqual(a['x'], b['x'])
        self.assertEquals(a.dimensions, b.dimensions)
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

    def test_indexed_by(self):
        data = create_test_data()
        slicers = {'dim1': slice(None, None, 2), 'dim2': slice(0, 2)}
        ret = data.indexed_by(**slicers)

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
            self.assertEqual(data[v].attributes, ret[v].attributes)
            slice_list = [slice(None)] * data[v].data.ndim
            for d, s in slicers.iteritems():
                if d in data[v].dimensions:
                    inds = np.nonzero(np.array(data[v].dimensions) == d)[0]
                    for ind in inds:
                        slice_list[ind] = s
            expected = data[v].data[slice_list]
            actual = ret[v].data
            np.testing.assert_array_equal(expected, actual)

        with self.assertRaises(ValueError):
            data.indexed_by(not_a_dim=slice(0, 2))

        ret = data.indexed_by(dim1=0)
        self.assertEqual({'time': 20, 'dim2': 50, 'dim3': 10}, ret.dimensions)

        ret = data.indexed_by(time=slice(2), dim1=0, dim2=slice(5))
        self.assertEqual({'time': 2, 'dim2': 5, 'dim3': 10}, ret.dimensions)

        ret = data.indexed_by(time=0, dim1=0, dim2=slice(5))
        self.assertItemsEqual({'dim2': 5, 'dim3': 10}, ret.dimensions)

    def test_labeled_by(self):
        data = create_test_data()
        int_slicers = {'dim1': slice(None, None, 2),
                       'dim2': slice(2),
                       'dim3': slice(3)}
        loc_slicers = {'dim1': slice(None, None, 2),
                       'dim2': slice(0, 0.5),
                       'dim3': slice('a', 'c')}
        self.assertEqual(data.indexed_by(**int_slicers),
                         data.labeled_by(**loc_slicers))
        data['time'] = ('time', pd.date_range('2000-01-01', periods=20))
        self.assertEqual(data.indexed_by(time=0),
                         data.labeled_by(time='2000-01-01'))
        self.assertEqual(data.indexed_by(time=slice(10)),
                         data.labeled_by(time=slice('2000-01-01',
                                                   '2000-01-10')))
        self.assertEqual(data, data.labeled_by(time=slice('1999', '2005')))
        self.assertEqual(data.indexed_by(time=slice(3)),
                         data.labeled_by(
                            time=pd.date_range('2000-01-01', periods=3)))

    def test_reindex_like(self):
        data = create_test_data()
        expected = data.indexed_by(dim1=slice(10), time=slice(13))
        actual = data.reindex_like(expected)
        self.assertDatasetEqual(actual, expected)

        expected = data.copy(deep=True)
        expected['dim3'] = ('dim3', list('cdefghijkl'))
        expected['var3'][:-2] = expected['var3'][2:]
        expected['var3'][-2:] = np.nan
        actual = data.reindex_like(expected)
        self.assertDatasetEqual(actual, expected)

    def test_reindex(self):
        data = create_test_data()
        expected = data.indexed_by(dim1=slice(10))
        actual = data.reindex(dim1=data['dim1'][:10])
        self.assertDatasetEqual(actual, expected)

        actual = data.reindex(dim1=data['dim1'][:10].data)
        self.assertDatasetEqual(actual, expected)

        actual = data.reindex(dim1=data['dim1'][:10].index)
        self.assertDatasetEqual(actual, expected)

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
        self.assertDatasetEqual(left2, right2)

        left2, right2 = align(left, right, join='outer')
        self.assertXArrayEqual(left2['dim3'], right2['dim3'])
        self.assertArrayEqual(left2['dim3'], union)
        self.assertDatasetEqual(left2.labeled_by(dim3=intersection),
                                right2.labeled_by(dim3=intersection))
        self.assertTrue(np.isnan(left2['var3'][-2:]).all())
        self.assertTrue(np.isnan(right2['var3'][:2]).all())

        left2, right2 = align(left, right, join='left')
        self.assertXArrayEqual(left2['dim3'], right2['dim3'])
        self.assertXArrayEqual(left2['dim3'], left['dim3'])
        self.assertDatasetEqual(left2.labeled_by(dim3=intersection),
                                right2.labeled_by(dim3=intersection))
        self.assertTrue(np.isnan(right2['var3'][:2]).all())

        left2, right2 = align(left, right, join='right')
        self.assertXArrayEqual(left2['dim3'], right2['dim3'])
        self.assertXArrayEqual(left2['dim3'], right['dim3'])
        self.assertDatasetEqual(left2.labeled_by(dim3=intersection),
                                right2.labeled_by(dim3=intersection))
        self.assertTrue(np.isnan(left2['var3'][-2:]).all())

    def test_variable_indexing(self):
        data = create_test_data()
        v = data['var1']
        d1 = data['dim1']
        d2 = data['dim2']
        self.assertXArrayEqual(v, v[d1.data])
        self.assertXArrayEqual(v, v[d1])
        self.assertXArrayEqual(v[:3], v[d1 < 3])
        self.assertXArrayEqual(v[:, 3:], v[:, d2 >= 1.5])
        self.assertXArrayEqual(v[:3, 3:], v[d1 < 3, d2 >= 1.5])
        self.assertXArrayEqual(v[:3, :2], v[range(3), range(2)])
        self.assertXArrayEqual(v[:3, :2], v.loc[d1[:3], d2[:2]])

    def test_select(self):
        data = create_test_data()
        ret = data.select(_testvar)
        self.assertXArrayEqual(data[_testvar], ret[_testvar])
        self.assertTrue(_vars.keys()[1] not in ret.variables)
        self.assertRaises(ValueError, data.select, (_testvar, 'not_a_var'))

    @unittest.skip('need to write this test')
    def test_unselect(self):
        pass

    def test_copy(self):
        data = create_test_data()

        copied = data.copy(deep=False)
        self.assertDatasetEqual(data, copied)
        for k in data:
            v0 = data.variables[k]
            v1 = copied.variables[k]
            self.assertIs(v0, v1)
        copied['foo'] = ('z', np.arange(5))
        self.assertNotIn('foo', data)

        copied = data.copy(deep=True)
        self.assertDatasetEqual(data, copied)
        for k in data:
            v0 = data.variables[k]
            v1 = copied.variables[k]
            self.assertIsNot(v0, v1)

    def test_rename(self):
        data = create_test_data()
        newnames = {'var1': 'renamed_var1', 'dim2': 'renamed_dim2'}
        renamed = data.rename(newnames)

        variables = OrderedDict(data.variables)
        for k, v in newnames.iteritems():
            variables[v] = variables.pop(k)

        for k, v in variables.iteritems():
            dims = list(v.dimensions)
            for name, newname in newnames.iteritems():
                if name in dims:
                    dims[dims.index(name)] = newname

            self.assertXArrayEqual(XArray(dims, v.data, v.attributes),
                                   renamed.variables[k])
            self.assertEqual(v.encoding, renamed.variables[k].encoding)
            self.assertEqual(type(v), type(renamed.variables[k]))

        self.assertTrue('var1' not in renamed.variables)
        self.assertTrue('dim2' not in renamed.variables)

        # verify that we can rename a variable without accessing the data
        var1 = data['var1']
        data['var1'] = InaccessibleArray(var1.dimensions, var1.data)
        renamed = data.rename(newnames)
        with self.assertRaises(UnexpectedDataAccess):
            renamed['renamed_var1'].data

    def test_update(self):
        data = create_test_data(seed=0)
        var2 = XArray('dim1', np.arange(100))
        actual = data.update({'var2': var2})
        expected_vars = OrderedDict(create_test_data(seed=0).variables)
        expected_vars['var2'] = var2
        expected = Dataset(expected_vars)
        self.assertDatasetEqual(expected, actual)
        # test in-place
        data2 = data.update(data, inplace=True)
        self.assertIs(data2, data)
        data2 = data.update(data, inplace=False)
        self.assertIsNot(data2, data)

    def test_merge(self):
        data = create_test_data()
        ds1 = data.select('var1')
        ds2 = data.select('var3')
        expected = data.select('var1', 'var3')
        actual = ds1.merge(ds2)
        self.assertEqual(expected, actual)
        with self.assertRaises(ValueError):
            ds1.merge(ds2.indexed_by(dim1=slice(2)))
        with self.assertRaises(ValueError):
            ds1.merge(ds2.rename({'var3': 'var1'}))

    def test_getitem(self):
        data = create_test_data()
        self.assertIsInstance(data['var1'], DataArray)
        self.assertXArrayEqual(data['var1'], data.variables['var1'])
        self.assertIs(data['var1'].dataset, data)

    def test_virtual_variables(self):
        # access virtual variables
        data = create_test_data()
        self.assertXArrayEqual(data['time.dayofyear'],
                               XArray('time', 1 + np.arange(20)))
        self.assertArrayEqual(data['time.month'].data,
                              data.variables['time'].index.month)
        # test virtual variable math
        self.assertArrayEqual(data['time.dayofyear'] + 1, 2 + np.arange(20))
        self.assertArrayEqual(np.sin(data['time.dayofyear']),
                              np.sin(1 + np.arange(20)))
        # test slicing the virtual variable -- it should still be virtual
        actual = data['time.dayofyear'][:10].dataset
        expected = data.indexed_by(time=slice(10))
        self.assertDatasetEqual(expected, actual)

    def test_setitem(self):
        # assign a variable
        var = XArray(['dim1'], np.random.randn(100))
        data1 = create_test_data()
        data1['A'] = var
        data2 = data1.copy()
        data2['A'] = var
        self.assertDatasetEqual(data1, data2)
        # assign a dataset array
        dv = 2 * data2['A']
        data1['B'] = dv.variable
        data2['B'] = dv
        self.assertDatasetEqual(data1, data2)
        # assign an array
        with self.assertRaisesRegexp(TypeError, 'variables must be of type'):
            data2['C'] = var.data
        # override an existing value
        data1['A'] = 3 * data2['A']
        self.assertXArrayEqual(data1['A'], 3 * data2['A'])
        # can't resize a used dimension
        with self.assertRaisesRegexp(ValueError, 'but already exists with'):
            data1['dim1'] = data1['dim1'][:5]

    def test_delitem(self):
        data = create_test_data()
        all_items = {'time', 'dim1', 'dim2', 'dim3', 'var1', 'var2', 'var3'}
        self.assertItemsEqual(data, all_items)
        del data['var1']
        self.assertItemsEqual(data, all_items - {'var1'})
        del data['dim1']
        self.assertItemsEqual(data, {'time', 'dim2', 'dim3'})

    def test_squeeze(self):
        data = Dataset({'foo': (['x', 'y', 'z'], [[[1], [2]]])})
        # squeeze everything
        expected = Dataset({'y': data['y'], 'foo': data['foo'].squeeze()})
        self.assertDatasetEqual(expected, data.squeeze())
        # squeeze only x
        expected = Dataset({'y': data['y'], 'foo': (['y', 'z'], data['foo'].data[0])})
        self.assertDatasetEqual(expected, data.squeeze('x'))
        self.assertDatasetEqual(expected, data.squeeze(['x']))
        # squeeze only z
        expected = Dataset({'y': data['y'], 'foo': (['x', 'y'], data['foo'].data[:, :, 0])})
        self.assertDatasetEqual(expected, data.squeeze('z'))
        self.assertDatasetEqual(expected, data.squeeze(['z']))
        # invalid squeeze
        with self.assertRaisesRegexp(ValueError, 'cannot select a dimension'):
            data.squeeze('y')

    def test_concat(self):
        data = create_test_data()

        split_data = [data.indexed_by(dim1=slice(10)),
                      data.indexed_by(dim1=slice(10, None))]
        self.assertDatasetEqual(data, Dataset.concat(split_data, 'dim1'))

        def rectify_dim_order(dataset):
            # return a new dataset with all variable dimensions tranposed into
            # the order in which they are found in `data`
            return Dataset({k: v.transpose(*data[k].dimensions)
                           for k, v in dataset.variables.iteritems()},
                           dataset.attributes)

        for dim in ['dim1', 'dim2', 'dim3']:
            datasets = [ds for _, ds in data.groupby(dim, squeeze=False)]
            self.assertDatasetEqual(data, Dataset.concat(datasets, dim))
            self.assertDatasetEqual(data, Dataset.concat(datasets, data[dim]))
            self.assertDatasetEqual(data, Dataset.concat(datasets, data[dim].variable))

            datasets = [ds for _, ds in data.groupby(dim, squeeze=True)]
            concat_over = [k for k, v in data.variables.iteritems()
                           if dim in v.dimensions and k != dim]
            actual = Dataset.concat(datasets, data[dim], concat_over=concat_over)
            self.assertDatasetEqual(data, rectify_dim_order(actual))

        # verify that the dimension argument takes precedence over
        # concatenating dataset variables of the same name
        dimension = (2 * data['dim1']).rename('dim1')
        datasets = [ds for _, ds in data.groupby('dim1', squeeze=False)]
        expected = data.copy()
        expected['dim1'] = dimension
        self.assertDatasetEqual(expected, Dataset.concat(datasets, dimension))

        with self.assertRaisesRegexp(ValueError, 'cannot be empty'):
            Dataset.concat([], 'dim1')
        with self.assertRaisesRegexp(ValueError, 'not all elements in'):
            Dataset.concat(split_data, 'dim1', concat_over=['not_found'])
        with self.assertRaisesRegexp(ValueError, 'global attributes not'):
            data0, data1 = deepcopy(split_data)
            data1.attributes['foo'] = 'bar'
            Dataset.concat([data0, data1], 'dim1')
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
        ds = Dataset({'a': ('t', x), 'b': ('t', y), 't': ('t', t)})
        expected = pd.DataFrame(np.array([x, y]).T, columns=['a', 'b'],
                                index=pd.Index(t, name='t'))
        actual = ds.to_dataframe()
        # use the .equals method to check all DataFrame metadata
        self.assertTrue(expected.equals(actual))

        # check roundtrip
        self.assertDatasetEqual(ds, Dataset.from_dataframe(actual))

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
        self.assertDatasetEqual(ds, Dataset.from_dataframe(actual))

    def test_lazy_load(self):
        store = InaccessibleVariableDataStore()
        store.set_variable('dim', XArray(('dim'), np.arange(10)))
        store.set_variable('var', XArray(('dim'), np.random.uniform(size=10)))
        ds = Dataset()
        ds = ds.load_store(store, decode_cf=False)
        self.assertRaises(UnexpectedDataAccess, lambda: ds['var'].data)
        ds = ds.load_store(store, decode_cf=True)
        self.assertRaises(UnexpectedDataAccess, lambda: ds['var'].data)
