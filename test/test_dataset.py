from collections import OrderedDict
from copy import deepcopy
from cStringIO import StringIO
import os.path
import unittest
import tempfile

import numpy as np
import pandas as pd

from xray import Dataset, DatasetArray, XArray, backends, open_dataset
from . import TestCase


_dims = {'dim1':100, 'dim2':50, 'dim3':10}
_vars = {'var1':['dim1', 'dim2'],
         'var2':['dim1', 'dim2'],
         'var3':['dim3', 'dim1'],
         }
_testvar = sorted(_vars.keys())[0]
_testdim = sorted(_dims.keys())[0]


def create_test_data(store=None):
    obj = Dataset() if store is None else Dataset.load_store(store)
    obj['time'] = ('time', pd.date_range('2000-01-01', periods=1000))
    for k, d in sorted(_dims.items()):
        obj[k] = (k, np.arange(d))
    for v, dims in sorted(_vars.items()):
        data = np.random.normal(size=tuple(_dims[d] for d in dims))
        obj[v] = (dims, data, {'foo': 'variable'})
    return obj


class DataTest(TestCase):
    def get_store(self):
        return backends.InMemoryDataStore()

    def test_repr(self):
        data = create_test_data(self.get_store())
        self.assertEqual('<xray.Dataset (time: 1000, dim1: 100, '
                         'dim2: 50, dim3: 10): var1 var2 var3>', repr(data))

    def test_init(self):
        var1 = XArray('x', np.arange(100))
        var2 = XArray('x', np.arange(1000))
        var3 = XArray(['x', 'y'], np.arange(1000).reshape(100, 10))
        with self.assertRaisesRegexp(ValueError, 'but already is saved'):
            Dataset({'a': var1, 'b': var2})
        with self.assertRaisesRegexp(ValueError, 'must be defined with 1-d'):
            Dataset({'a': var1, 'x': var3})

    def test_groupby(self):
        data = create_test_data(self.get_store())
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
        self.assertTrue(a.variables.keys() == ['time', 'x', 'foo', 'bar'])
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
        self.assertIsInstance(a.coordinates['x'].data, pd.Index)
        self.assertXArrayEqual(a.coordinates['x'], a.variables['x'])
        b = Dataset()
        b['x'] = ('x', vec, attributes)
        self.assertXArrayEqual(a['x'], b['x'])
        self.assertEquals(a.dimensions, b.dimensions)
        with self.assertRaises(ValueError):
            a['x'] = ('x', vec[:5])
        arr = np.random.random((10, 1,))
        scal = np.array(0)
        with self.assertRaises(ValueError):
            a['y'] = ('y', arr)
        with self.assertRaises(ValueError):
            a['y'] = ('y', scal)
        self.assertTrue('y' not in a.dimensions)

    @unittest.skip('attribute checks are not yet backend specific')
    def test_attributes(self):
        a = Dataset()
        a.attributes['foo'] = 'abc'
        a.attributes['bar'] = 1
        # numeric scalars are stored as length-1 vectors
        self.assertTrue(isinstance(a.attributes['bar'], np.ndarray) and
                a.attributes['bar'].ndim == 1)
        # __contains__ method
        self.assertTrue('foo' in a.attributes)
        self.assertTrue('bar' in a.attributes)
        self.assertTrue('baz' not in a.attributes)
        # user-defined attributes are not object attributes
        self.assertRaises(AttributeError, object.__getattribute__, a, 'foo')
        # different ways of setting attributes ought to be equivalent
        b = Dataset()
        b.attributes.update(foo='abc')
        self.assertEquals(a.attributes['foo'], b.attributes['foo'])
        b = Dataset()
        b.attributes.update([('foo', 'abc')])
        self.assertEquals(a.attributes['foo'], b.attributes['foo'])
        b = Dataset()
        b.attributes.update({'foo': 'abc'})
        self.assertEquals(a.attributes['foo'], b.attributes['foo'])
        # attributes can be overwritten
        b.attributes['foo'] = 'xyz'
        self.assertEquals(b.attributes['foo'], 'xyz')
        # attributes can be deleted
        del b.attributes['foo']
        self.assertTrue('foo' not in b.attributes)
        # attributes can be cleared
        b.attributes.clear()
        self.assertTrue(len(b.attributes) == 0)
        # attributes can be compared
        a = Dataset()
        b = Dataset()
        a.attributes['foo'] = 'bar'
        b.attributes['foo'] = np.nan
        self.assertFalse(a == b)
        a.attributes['foo'] = np.nan
        self.assertTrue(a == b)
        # attribute names/values must be netCDF-compatible
        self.assertRaises(ValueError, b.attributes.__setitem__, '/', 0)
        self.assertRaises(ValueError, b.attributes.__setitem__, 'foo', np.zeros((2, 2)))
        self.assertRaises(ValueError, b.attributes.__setitem__, 'foo', dict())

    def test_indexed_by(self):
        data = create_test_data(self.get_store())
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
        self.assertEqual({'time': 1000, 'dim2': 50, 'dim3': 10}, ret.dimensions)

        ret = data.indexed_by(time=slice(2), dim1=0, dim2=slice(5))
        self.assertEqual({'time': 2, 'dim2': 5, 'dim3': 10}, ret.dimensions)

        ret = data.indexed_by(time=0, dim1=0, dim2=slice(5))
        self.assertItemsEqual({'dim2': 5, 'dim3': 10}, ret.dimensions)

    def test_labeled_by(self):
        data = create_test_data(self.get_store())
        int_slicers = {'dim1': slice(None, None, 2), 'dim2': slice(0, 2)}
        loc_slicers = {'dim1': slice(None, None, 2), 'dim2': slice(0, 1)}
        self.assertEqual(data.indexed_by(**int_slicers),
                         data.labeled_by(**loc_slicers))
        data['time'] = ('time', np.arange(1000, dtype=np.int32),
                        {'units': 'days since 2000-01-01'})
        self.assertEqual(data.indexed_by(time=0),
                         data.labeled_by(time='2000-01-01'))
        self.assertEqual(data.indexed_by(time=slice(10)),
                         data.labeled_by(time=slice('2000-01-01',
                                                   '2000-01-10')))
        self.assertEqual(data, data.labeled_by(time=slice('1999', '2005')))
        self.assertEqual(data.indexed_by(time=slice(3)),
                         data.labeled_by(
                            time=pd.date_range('2000-01-01', periods=3)))

    def test_variable_indexing(self):
        data = create_test_data(self.get_store())
        v = data['var1']
        d1 = data['dim1']
        d2 = data['dim2']
        self.assertXArrayEqual(v, v[d1.data])
        self.assertXArrayEqual(v, v[d1])
        self.assertXArrayEqual(v[:3], v[d1 < 3])
        self.assertXArrayEqual(v[:, 3:], v[:, d2 >= 3])
        self.assertXArrayEqual(v[:3, 3:], v[d1 < 3, d2 >= 3])
        self.assertXArrayEqual(v[:3, :2], v[d1[:3], d2[:2]])
        self.assertXArrayEqual(v[:3, :2], v[range(3), range(2)])

    def test_select(self):
        data = create_test_data(self.get_store())
        ret = data.select(_testvar)
        self.assertXArrayEqual(data[_testvar], ret[_testvar])
        self.assertTrue(_vars.keys()[1] not in ret.variables)
        self.assertRaises(ValueError, data.select, (_testvar, 'not_a_var'))

    @unittest.skip('need to write this test')
    def test_unselect(self):
        pass

    def test_copy(self):
        data = create_test_data(self.get_store())
        var = data.variables[_testvar]
        var.attributes['foo'] = 'hello world'
        var_copy = var.__deepcopy__()
        self.assertEqual(var.data[2, 3], var_copy.data[2, 3])
        var_copy.data[2, 3] = np.pi
        self.assertNotEqual(var.data[2, 3], np.pi)
        self.assertEqual(var_copy.attributes['foo'], var.attributes['foo'])
        var_copy.attributes['foo'] = 'xyz'
        self.assertNotEqual(var_copy.attributes['foo'], var.attributes['foo'])
        self.assertEqual(var_copy.attributes['foo'], 'xyz')
        self.assertNotEqual(id(var), id(var_copy))
        self.assertNotEqual(id(var.data), id(var_copy.data))
        self.assertNotEqual(id(var.attributes), id(var_copy.attributes))

    def test_rename(self):
        data = create_test_data(self.get_store())
        newnames = {'var1': 'renamed_var1', 'dim2': 'renamed_dim2'}
        renamed = data.renamed(newnames)

        variables = OrderedDict(data.variables)
        for k, v in newnames.iteritems():
            variables[v] = variables.pop(k)

        for k, v in variables.iteritems():
            self.assertTrue(k in renamed.variables)
            self.assertEqual(v.attributes, renamed.variables[k].attributes)
            dims = list(v.dimensions)
            for name, newname in newnames.iteritems():
                if name in dims:
                    dims[dims.index(name)] = newname
            self.assertEqual(dims, list(renamed.variables[k].dimensions))
            self.assertTrue(np.all(v.data == renamed.variables[k].data))
            self.assertEqual(v.attributes, renamed.variables[k].attributes)

        self.assertTrue('var1' not in renamed.variables)
        self.assertTrue('var1' not in renamed.dimensions)
        self.assertTrue('dim2' not in renamed.variables)
        self.assertTrue('dim2' not in renamed.dimensions)

    def test_merge(self):
        data = create_test_data(self.get_store())
        ds1 = data.select('var1')
        ds2 = data.select('var3')
        expected = data.select('var1', 'var3')
        actual = ds1.merge(ds2)
        self.assertEqual(expected, actual)
        with self.assertRaises(ValueError):
            ds1.merge(ds2.indexed_by(dim1=slice(2)))
        with self.assertRaises(ValueError):
            ds1.merge(ds2.renamed({'var3': 'var1'}))

    def test_getitem(self):
        data = create_test_data(self.get_store())
        data['time'] = ('time', np.arange(1000, dtype=np.int32),
                        {'units': 'days since 2000-01-01'})
        self.assertIsInstance(data['var1'], DatasetArray)
        self.assertXArrayEqual(data['var1'], data.variables['var1'])
        self.assertItemsEqual(data['var1'].dataset.variables,
                              {'var1', 'dim1', 'dim2'})
        # access virtual variables
        self.assertXArrayEqual(data['time.dayofyear'][:300],
                            XArray('time', 1 + np.arange(300)))
        self.assertArrayEqual(data['time.month'].data,
                              data.variables['time'].data.month)

    def test_setitem(self):
        # assign a variable
        var = XArray(['dim1'], np.random.randn(100))
        data1 = create_test_data(self.get_store())
        data1['A'] = var
        data2 = data1.copy()
        data2['A'] = var
        self.assertEqual(data1, data2)
        # assign a dataset array
        dv = 2 * data2['A']
        data1['B'] = dv.array
        data2['B'] = dv
        self.assertEqual(data1, data2)
        # assign an array
        with self.assertRaisesRegexp(TypeError, 'variables must be of type'):
            data2['C'] = var.data

    def test_write_store(self):
        expected = create_test_data()
        store = self.get_store()
        expected.dump_to_store(store)
        actual = Dataset.load_store(store)
        self.assertEquals(expected, actual)

    def test_to_dataframe(self):
        x = np.random.randn(10)
        y = np.random.randn(10)
        t = list('abcdefghij')
        ds = Dataset({'a': ('t', x), 'b': ('t', y), 't': ('t', t)})
        expected = pd.DataFrame(np.array([x, y]).T, columns=['a', 'b'],
                                index=pd.Index(t, name='t'))
        actual = ds.to_dataframe()
        # use the .equals method to check all DataFrame metadata
        self.assertTrue(expected.equals(actual))

        # test a case with a MultiIndex
        w = np.random.randn(2, 3)
        ds = Dataset({'w': (('x', 'y'), w)})
        ds['y'] = ('y', list('abc'))
        print ds.dimensions
        exp_index = pd.MultiIndex.from_arrays(
            [[0, 0, 0, 1, 1, 1], ['a', 'b', 'c', 'a', 'b', 'c']],
            names=['x', 'y'])
        expected = pd.DataFrame(w.reshape(-1), columns=['w'], index=exp_index)
        actual = ds.to_dataframe()
        self.assertTrue(expected.equals(actual))


class NetCDF4DataTest(DataTest):
    def get_store(self):
        f, self.tmp_file = tempfile.mkstemp(suffix='.nc')
        os.close(f)
        return backends.NetCDF4DataStore(self.tmp_file, mode='w')

    def test_dump_and_open_dataset(self):
        data = create_test_data(self.get_store())
        f, tmp_file = tempfile.mkstemp(suffix='.nc')
        os.close(f)
        data.dump(tmp_file)

        expected = data.copy()
        actual = open_dataset(tmp_file)
        self.assertEquals(expected, actual)
        os.remove(tmp_file)

    def tearDown(self):
        if hasattr(self, 'tmp_file') and os.path.exists(self.tmp_file):
            os.remove(self.tmp_file)


class ScipyDataTest(DataTest):
    def get_store(self):
        fobj = StringIO()
        return backends.ScipyDataStore(fobj, 'w')

    def test_dump_and_open_dataset(self):
        data = create_test_data(self.get_store())
        serialized = data.dumps()

        expected = data.copy()
        actual = open_dataset(StringIO(serialized))
        self.assertEquals(expected, actual)

    def test_repr(self):
        # scipy.io.netcdf does not keep track of dimension order :(
        pass
