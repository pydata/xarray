from collections import OrderedDict
from copy import deepcopy
from cStringIO import StringIO
import os.path
import unittest
import tempfile

import numpy as np
import pandas as pd

from scidata import Dataset, Variable, backends
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
    obj.add_dimension('time', 1000)
    for d, l in sorted(_dims.items()):
        obj.add_dimension(d, l)
        var = obj.create_variable(name=d, dims=(d,),
                                  data=np.arange(l, dtype=np.int32),
                                  attributes={'units':'integers'})
    for v, dims in sorted(_vars.items()):
        var = obj.create_variable(name=v, dims=tuple(dims),
                data=np.random.normal(size=tuple([_dims[d] for d in dims])))
        var.attributes['foo'] = 'variable'
    return obj

class DataTest(TestCase):
    def get_store(self):
        return backends.InMemoryDataStore()

    def test_repr(self):
        data = create_test_data(self.get_store())
        self.assertEqual('<scidata.Dataset (time: 1000, @dim1: 100, '
                         '@dim2: 50, @dim3: 10): var1 var2 var3>', repr(data))

    @unittest.skip('method needs rewrite and/or removal')
    def test_iterator(self):
        data = create_test_data(self.get_store())
        # iterate over the first dim
        iterdim = _testdim
        for t, sub in data.iterator(dim=iterdim):
            ind = int(np.where(data.variables[iterdim].data == t.data)[0])
            # make sure all the slices match
            for v in _vars.keys():
                if iterdim in data[v].dimensions:
                    dim_axis = list(data[v].dimensions).index(iterdim)
                    expected = data[v].data.take(
                            [ind], axis=dim_axis).reshape(sub[v].data.shape)
                    np.testing.assert_array_equal(sub[v].data, expected)
                self.assertEquals(sub.dimensions[iterdim], 1)
        # test that the yielded objects are copies of the original
        for (t, sub) in data.iterator(dim=iterdim):
            sub[_testvar][:] = -71
        self.assertTrue((data[_testvar].data != -71).all())

    def test_iterarray(self):
        data = create_test_data(self.get_store())
        # iterate over the first dim
        iterdim = _testdim
        for t, d in data.iterarray(dim=iterdim, var=_testvar):
            ind = int(np.where(data.variables[iterdim].data == t)[0])
            # make sure all the slices match
            dim_axis = list(data[_testvar].dimensions).index(iterdim)
            expected = data.variables[_testvar].data.take([ind], axis=dim_axis)
            np.testing.assert_array_equal(d, expected)
        # test that the yielded objects are views of the original
        # This test doesn't make sense for the netCDF4 backend
        # for (t, d) in data.iterarray(dim=iterdim, var=_testvar):
        #     d[:] = -71
        # self.assertTrue((data[_testvar].data == -71).all())

    def test_dimension(self):
        a = Dataset()
        a.add_dimension('time', 10)
        a.add_dimension('x', 5)
        # prevent duplicate creation
        self.assertRaises(ValueError, a.add_dimension, 'time', 0)
        # length must be integer
        self.assertRaises(ValueError, a.add_dimension, 'foo', 'a')
        self.assertRaises(TypeError, a.add_dimension, 'foo', [1,])
        self.assertRaises(ValueError, a.add_dimension, 'foo', -1)
        self.assertTrue('foo' not in a.dimensions)

    def test_variable(self):
        a = Dataset()
        a.add_dimension('time', 10)
        a.add_dimension('x', 3)
        d = np.random.random((10, 3))
        a.create_variable(name='foo', dims=('time', 'x',), data=d)
        self.assertTrue('foo' in a.variables)
        self.assertTrue('foo' in a)
        a.create_variable(name='bar', dims=('time', 'x',), data=d)
        # order of creation is preserved
        self.assertTrue(a.variables.keys() == ['foo', 'bar'])
        self.assertTrue(all([a.variables['foo'][i].data == d[i]
                             for i in np.ndindex(*d.shape)]))
        # prevent duplicate creation
        self.assertRaises(ValueError, a.create_variable,
                name='foo', dims=('time', 'x',), data=d)
        # dimension must be defined
        self.assertRaises(ValueError, a.create_variable,
                name='qux', dims=('time', 'missing_dim',), data=d)
        # try to add variable with dim (10,3) with data that's (3,10)
        self.assertRaises(ValueError, a.create_variable,
                name='qux', dims=('time', 'x'), data=d.T)
        # Variable equality
        d = np.random.rand(10, 3)
        v1 = Variable(('dim1','dim2'), data=d,
                           attributes={'att1': 3, 'att2': [1,2,3]})
        v2 = Variable(('dim1','dim2'), data=d,
                           attributes={'att1': 3, 'att2': [1,2,3]})
        v5 = Variable(('dim1','dim2'), data=d,
                           attributes={'att1': 3, 'att2': [1,2,3]})
        v3 = Variable(('dim1','dim3'), data=d,
                           attributes={'att1': 3, 'att2': [1,2,3]})
        v4 = Variable(('dim1','dim2'), data=d,
                           attributes={'att1': 3, 'att2': [1,2,4]})
        v5 = deepcopy(v1)
        v5.data[:] = np.random.rand(10,3)
        self.assertVarEqual(v1, v2)
        self.assertVarNotEqual(v1, v3)
        self.assertVarNotEqual(v1, v4)
        self.assertVarNotEqual(v1, v5)

    def test_coordinate(self):
        a = Dataset()
        vec = np.random.random((10,))
        attributes = {'foo': 'bar'}
        a.create_coordinate('x', data=vec, attributes=attributes)
        self.assertTrue('x' in a.coordinates)
        self.assertVarEqual(a.coordinates['x'], a.variables['x'])
        b = Dataset()
        b.add_dimension('x', vec.size)
        b.create_variable('x', dims=('x',), data=vec, attributes=attributes)
        self.assertVarEqual(a['x'], b['x'])
        self.assertEquals(a.dimensions, b.dimensions)
        arr = np.random.random((10, 1,))
        scal = np.array(0)
        self.assertRaises(ValueError, a.create_coordinate,
                name='y', data=arr)
        self.assertRaises(ValueError, a.create_coordinate,
                name='y', data=scal)
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

    def test_views(self):
        data = create_test_data(self.get_store())
        slicers = {'dim1': slice(None, None, 2), 'dim2': slice(0, 2)}
        ret = data.views(**slicers)

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
            # Test that our view accesses the same underlying array
            # This test doesn't make sense for the netCDF4 backend
            # actual.fill(np.pi)
            # np.testing.assert_array_equal(expected, actual)

        with self.assertRaises(ValueError):
            data.views(not_a_dim=slice(0, 2))

        ret = data.views(dim1=0)
        self.assertEqual({'time': 1000, 'dim2': 50, 'dim3': 10}, ret.dimensions)

        ret = data.views(time=slice(2), dim1=0, dim2=slice(5))
        self.assertEqual({'time': 2, 'dim2': 5, 'dim3': 10}, ret.dimensions)

        ret = data.views(time=0, dim1=0, dim2=slice(5))
        self.assertItemsEqual({'dim2': 5, 'dim3': 10}, ret.dimensions)

    def test_loc_views(self):
        data = create_test_data(self.get_store())
        int_slicers = {'dim1': slice(None, None, 2), 'dim2': slice(0, 2)}
        loc_slicers = {'dim1': slice(None, None, 2), 'dim2': slice(0, 1)}
        self.assertEqual(data.views(**int_slicers),
                         data.loc_views(**loc_slicers))
        data.create_variable('time', ['time'], np.arange(1000, dtype=np.int32),
                             {'units': 'days since 2000-01-01'})
        self.assertEqual(data.views(time=0),
                         data.loc_views(time='2000-01-01'))
        self.assertEqual(data.views(time=slice(10)),
                         data.loc_views(time=slice('2000-01-01',
                                                   '2000-01-10')))
        self.assertEqual(data, data.loc_views(time=slice('1999', '2005')))
        self.assertEqual(data.views(time=slice(3)),
                         data.loc_views(
                            time=pd.date_range('2000-01-01', periods=3)))

    def test_variable_indexing(self):
        data = create_test_data(self.get_store())
        v = data['var1']
        d1 = data['dim1']
        d2 = data['dim2']
        self.assertVarEqual(v, v[d1.data])
        self.assertVarEqual(v, v[d1])
        self.assertVarEqual(v[:3], v[d1 < 3])
        self.assertVarEqual(v[:, 3:], v[:, d2 >= 3])
        self.assertVarEqual(v[:3, 3:], v[d1 < 3, d2 >= 3])
        self.assertVarEqual(v[:3, :2], v[d1[:3], d2[:2]])
        self.assertVarEqual(v[:3, :2], v[range(3), range(2)])

    @unittest.skip('obsolete method should be removed')
    def test_take(self):
        data = create_test_data(self.get_store())
        slicedim = _testdim
        # using a list
        ret = data.take(indices=range(2, 5), dim=slicedim)
        self.assertEquals(len(ret[slicedim].data), 3)
        # using a numpy vector
        ret = data.take(indices=np.array([2, 3, 4,]), dim=slicedim)
        self.assertEquals(len(ret[slicedim].data), 3)
        # With a random index
        indices = np.random.randint(data.dimensions[slicedim], size=10)
        ret = data.take(indices=indices, dim=slicedim)
        # Verify that only the specified dimension was altered
        for d in data.dimensions:
            if d == slicedim:
                self.assertEqual(ret.dimensions[d], indices.size)
            else:
                self.assertEqual(data.dimensions[d], ret.dimensions[d])
        # Verify that the data is what we expect
        for v in data.variables:
            self.assertEqual(data[v].dimensions, ret[v].dimensions)
            self.assertEqual(data[v].attributes, ret[v].attributes)
            if slicedim in data[v].dimensions:
                expected = data[v].data.take(
                    indices, axis=data[v].dimensions.index(slicedim))
            else:
                expected = data[v].data[:]
            actual = ret[v].data
            np.testing.assert_array_equal(expected, actual)
            # Test that our take is a copy
            ret[v].data.fill(np.pi)
            self.assertTrue(not (data[v].data == np.pi).any())
        self.assertRaises(KeyError, data.take,
                          indices=indices, dim='not_a_dim')
        self.assertRaises(IndexError, data.take,
                          indices=[data.dimensions[slicedim] + 10],
                          dim=slicedim)

    @unittest.skip('method needs rewrite and/or removal')
    def test_squeeze(self):
        data = create_test_data(self.get_store())
        singleton = data.take([1], 'dim2')
        squeezed = singleton.squeeze('dim2')
        assert not 'dim2' in squeezed.dimensions
        for x in [v for v, d in _vars.iteritems() if 'dim2' in d]:
            np.testing.assert_array_equal(singleton[x].data.flatten(),
                                          squeezed[x].data)

    def test_select(self):
        data = create_test_data(self.get_store())
        ret = data.select(_testvar)
        self.assertVarEqual(data[_testvar], ret[_testvar])
        self.assertTrue(_vars.keys()[1] not in ret.variables)
        self.assertRaises(ValueError, data.select, (_testvar, 'not_a_var'))

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
            ds1.merge(ds2.views(dim1=0))
        with self.assertRaises(ValueError):
            ds1.merge(ds2.renamed({'var3': 'var1'}))

    def test_virtual_variables(self):
        # need to fill this out
        pass

    def test_write_store(self):
        expected = create_test_data()
        store = self.get_store()
        expected.dump_to_store(store)
        actual = Dataset.load_store(store)
        self.assertEquals(expected, actual)


class NetCDF4DataTest(DataTest):
    def get_store(self):
        f, self.tmp_file = tempfile.mkstemp(suffix='.nc')
        os.close(f)
        return backends.NetCDF4DataStore(self.tmp_file, mode='w')

    def tearDown(self):
        if hasattr(self, 'tmp_file') and os.path.exists(self.tmp_file):
            os.remove(self.tmp_file)


class ScipyDataTest(DataTest):
    def get_store(self):
        fobj = StringIO()
        return backends.ScipyDataStore(fobj, 'w')

    def test_repr(self):
        # scipy.io.netcdf does not keep track of dimension order :(
        pass


# class StoreTest(TestCase):
#     def test_store_consistency(self):
#         mem_ds = create_test_data()

#         fobj = StringIO()
#         store = backends.ScipyDataStore(fobj, 'w')
#         store = self.get_store()
#         mem_ds.dump_to_store()

#         stored_ds = Dataset.load_store(store)
#         self.assertEquals(mem_ds, stored_ds)


if __name__ == "__main__":
    unittest.main()
