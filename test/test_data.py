import unittest
import os.path
import numpy as np
import scipy.interpolate

from copy import deepcopy
from cStringIO import StringIO

from polyglot import Dataset, Variable, backends

_dims = {'dim1':100, 'dim2':50, 'dim3':10}
_vars = {'var1':['dim1', 'dim2'],
         'var2':['dim1', 'dim2'],
         'var3':['dim3', 'dim1'],
         }
_testvar = sorted(_vars.keys())[0]
_testdim = sorted(_dims.keys())[0]

def create_test_data(store=None):
    obj = Dataset(store=store)
    obj.create_dimension('time', 10)
    for d, l in _dims.items():
        obj.create_dimension(d, l)
        var = obj.create_variable(name=d, dims=(d,),
                                  data=np.arange(l, dtype=np.int32),
                                  attributes={'units':'integers'})
    for v, dims in _vars.items():
        var = obj.create_variable(name=v, dims=tuple(dims),
                data=np.random.normal(size=tuple([_dims[d] for d in dims])))
        var.attributes['foo'] = 'variable'
    return obj

class DataTest(unittest.TestCase):

    def get_store(self):
        return None

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
            expected = data[_testvar].data.take([ind], axis=dim_axis)
            np.testing.assert_array_equal(d, expected)
        # test that the yielded objects are views of the original
        for (t, d) in data.iterarray(dim=iterdim, var=_testvar):
            d[:] = -71
        self.assertTrue((data[_testvar].data == -71).all())

    def test_dimension(self):
        a = Dataset()
        # data objects (currently) do not support record dimensions
        self.assertRaises(ValueError, a.create_dimension, 'time', None)
        a.create_dimension('time', 10)
        a.create_dimension('x', 5)
        # prevent duplicate creation
        self.assertRaises(ValueError, a.create_dimension, 'time', 0)
        # length must be integer
        self.assertRaises(TypeError, a.create_dimension, 'foo', 'a')
        self.assertRaises(TypeError, a.create_dimension, 'foo', [1,])
        self.assertTrue('foo' not in a.dimensions)

    def test_variable(self):
        a = Dataset()
        a.create_dimension('time', 10)
        a.create_dimension('x', 3)
        d = np.random.random((10, 3))
        a.create_variable(name='foo', dims=('time', 'x',), data=d)
        self.assertTrue('foo' in a.variables)
        self.assertTrue('foo' in a)
        a.create_variable(name='bar', dims=('time', 'x',), data=d)
        # order of creation is preserved
        self.assertTrue(a.variables.keys() == ['foo', 'bar'])
        self.assertTrue(all([a['foo'][i] == d[i]
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
        self.assertEquals(v1, v2)
        self.assertFalse(v1 == v3)
        self.assertFalse(v1 == v4)
        self.assertFalse(v1 == v5)
        # Variable hash
        self.assertEquals(hash(v1), hash(v2))

    def test_coordinate(self):
        a = Dataset()
        vec = np.random.random((10,))
        attributes = {'foo': 'bar'}
        a.create_coordinate('x', data=vec, attributes=attributes)
        self.assertTrue('x' in a.coordinates)
        self.assertTrue(a.coordinates['x'] == a.variables['x'])
        b = Dataset()
        b.create_dimension('x', vec.size)
        b.create_variable('x', dims=('x',), data=vec, attributes=attributes)
        self.assertTrue((a['x'].data == b['x'].data).all())
        self.assertEquals(a.dimensions, b.dimensions)
        arr = np.random.random((10, 1,))
        scal = np.array(0)
        self.assertRaises(ValueError, a.create_coordinate,
                name='y', data=arr)
        self.assertRaises(ValueError, a.create_coordinate,
                name='y', data=scal)
        self.assertTrue('y' not in a.dimensions)

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

    def test_view(self):
        data = create_test_data(self.get_store())
        slicedim = _testdim
        s = slice(None, None, 2)
        ret = data.view(s=s, dim=slicedim)
        # Verify that only the specified dimension was altered
        for d in data.dimensions:
            if d == slicedim:
                self.assertEqual(ret.dimensions[d],
                                 np.arange(data.dimensions[d])[s].size)
            else:
                self.assertEqual(data.dimensions[d], ret.dimensions[d])
        # Verify that the data is what we expect
        for v in data.variables:
            self.assertEqual(data[v].dimensions, ret[v].dimensions)
            self.assertEqual(data[v].attributes, ret[v].attributes)
            if slicedim in data[v].dimensions:
                slice_list = [slice(None)] * data[v].data.ndim
                slice_list[data[v].dimensions.index(slicedim)] = s
                expected = data[v].data[slice_list][:]
            else:
                expected = data[v].data[:]
            actual = ret[v].data[:]
            np.testing.assert_array_equal(expected, actual)
            # Test that our view accesses the same underlying array
            actual.fill(np.pi)
            np.testing.assert_array_equal(expected, actual)
        self.assertRaises(KeyError, data.view,
                          s=s, dim='not_a_dim')
        self.assertRaises(IndexError, data.view,
                          s=slice(100, 200), dim=slicedim)

    def test_views(self):
        data = create_test_data(self.get_store())

        data.create_variable('var4', ('dim1', 'dim1'),
                             data = np.empty((data.dimensions['dim1'],
                                              data.dimensions['dim1']),
                                             np.float))
        data['var4'].data[:] = np.random.normal(size=data['var4'].shape)
        slicers = {'dim1': slice(None, None, 2), 'dim2':slice(0, 2)}
        ret = data.views(slicers)
        data.views(slicers)

        # Verify that only the specified dimension was altered
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
            actual.fill(np.pi)
            np.testing.assert_array_equal(expected, actual)
        self.assertRaises(KeyError, data.views,
                          {'not_a_dim': slice(0, 2)})

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
        np.testing.assert_array_equal(data[_testvar].data,
                                      ret[_testvar].data)
        self.assertTrue(_vars.keys()[1] not in ret.variables)
        self.assertRaises(KeyError, data.select, (_testvar, 'not_a_var'))

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
        newnames = {'var1':'renamed_var1', 'dim2':'renamed_dim2'}
        renamed = data.renamed(newnames)

        vars = dict((k, v) for k, v in data.variables.iteritems())
        for k, v in newnames.iteritems():
            vars[v] = vars.pop(k)

        for k, v in vars.iteritems():
            self.assertTrue(k in renamed.variables)
            self.assertEqual(v.attributes, renamed.variables[k].attributes)
            dims = list(v.dimensions)
            for name, newname in newnames.iteritems():
                if name in dims:
                    dims[dims.index(name)] = newname
            self.assertEqual(dims, list(renamed.variables[k].dimensions))
            np.testing.assert_array_equal(v.data[:], renamed.variables[k].data)

        self.assertTrue('var1' not in renamed.variables)
        self.assertTrue('var1' not in renamed.dimensions)
        self.assertTrue('dim2' not in renamed.variables)
        self.assertTrue('dim2' not in renamed.dimensions)

class NetCDF4DataTest(DataTest):

    def get_store(self):
        tmp_file = './delete_me.nc'
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        return backends.NetCDF4DataStore(tmp_file, mode='w')

    # Views on NetCDF4 objects result in copies of the arrays
    # since the netCDF4 package requires data to live on disk
    def test_view(self):
        pass

    def test_views(self):
        pass

    def test_iterarray(self):
        pass

    # TODO: select isn't working for netCDF4 yet.
    def test_select(self):
        pass

class ScipyDataTest(DataTest):

    def get_store(self):
        fobj = StringIO()
        return backends.ScipyDataStore(fobj, 'w')

class StoreTest(unittest.TestCase):

    def test_translate_consistency(self):

        fobj = StringIO()
        store = backends.ScipyDataStore(fobj, 'w')
        expected = create_test_data(store)

        mem_nc = deepcopy(expected)
        self.assertTrue(isinstance(mem_nc.store, backends.InMemoryDataStore))

        fobj = StringIO()
        actual = Dataset(store=backends.ScipyDataStore(fobj, 'w'))
        mem_nc.translate(actual)

        self.assertTrue(actual == expected)

if __name__ == "__main__":
    unittest.main()
