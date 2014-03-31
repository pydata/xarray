import mock
import os.path
import unittest
import tempfile

from cStringIO import StringIO
from collections import OrderedDict
from copy import deepcopy
from textwrap import dedent

import numpy as np
import pandas as pd
import netCDF4 as nc4

from xray import Dataset, DataArray, XArray, backends, open_dataset, utils

from . import TestCase


_test_data_path = os.path.join(os.path.dirname(__file__), 'data')

_dims = {'dim1': 100, 'dim2': 50, 'dim3': 10}
_vars = {'var1': ['dim1', 'dim2'],
         'var2': ['dim1', 'dim2'],
         'var3': ['dim3', 'dim1'],
         }
_testvar = sorted(_vars.keys())[0]
_testdim = sorted(_dims.keys())[0]


def create_test_data():
    obj = Dataset()
    obj['time'] = ('time', pd.date_range('2000-01-01', periods=20))
    obj['dim1'] = ('dim1', np.arange(_dims['dim1']))
    obj['dim2'] = ('dim2', 0.5 * np.arange(_dims['dim2']))
    obj['dim3'] = ('dim3', list('abcdefghij'))
    for v, dims in sorted(_vars.items()):
        data = np.random.normal(size=tuple(_dims[d] for d in dims))
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
        coords = [k for k in self._variables.keys()
                  if k in self.dimensions]

        def mask_noncoords(k, v):
            if k in coords:
                return k, v
            else:
                return k, InaccessibleArray(v.dimensions, v.data, v.attributes)
        return utils.FrozenOrderedDict(mask_noncoords(k, v)
                            for k, v in self._variables.iteritems())



class TestDataset(TestCase):
    def test_repr(self):
        data = create_test_data()
        expected = dedent("""
        <xray.Dataset>
        Coordinates:     (time: 20, dim1: 100, dim2: 50, dim3: 10)
        Non-coordinates:
            var1              -         X          X         -
            var2              -         X          X         -
            var3              -         X          -         X
        Attributes:
            Empty
        """).strip()
        actual = '\n'.join(x.rstrip() for x in repr(data).split('\n'))
        self.assertEqual(expected, actual)

        expected = dedent("""
        <xray.Dataset>
        Coordinates:     ()
        Non-coordinates:
            None
        Attributes:
            Empty
        """).strip()
        actual = '\n'.join(x.rstrip() for x in repr(Dataset()).split('\n'))
        self.assertEqual(expected, actual)

    def test_init(self):
        var1 = XArray('x', np.arange(100))
        var2 = XArray('x', np.arange(1000))
        var3 = XArray(['x', 'y'], np.arange(1000).reshape(100, 10))
        with self.assertRaisesRegexp(ValueError, 'but already is saved'):
            Dataset({'a': var1, 'b': var2})
        with self.assertRaisesRegexp(ValueError, 'must be defined with 1-d'):
            Dataset({'a': var1, 'x': var3})

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
        self.assertIsInstance(a.coordinates['x'].index, pd.Index)
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
        # test accessing a decoded virtual variable
        data.set_variables({'time2': ('time', np.arange(20),
                                     {'units': 'days since 2000-01-01'})},
                           decode_cf=True)
        self.assertXArrayEqual(data['time2.dayofyear'],
                               XArray('time', 1 + np.arange(20)))
        # test virtual variable math
        self.assertArrayEqual(data['time.dayofyear'] + 1, 2 + np.arange(20))
        self.assertArrayEqual(data['time2.dayofyear'] + 1, 2 + np.arange(20))
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
        self.assertEqual(data1, data2)
        # assign a dataset array
        dv = 2 * data2['A']
        data1['B'] = dv.variable
        data2['B'] = dv
        self.assertEqual(data1, data2)
        # assign an array
        with self.assertRaisesRegexp(TypeError, 'variables must be of type'):
            data2['C'] = var.data
        # override an existing value
        data1['A'] = 3 * data2['A']
        self.assertXArrayEqual(data1['A'], 3 * data2['A'])

    def test_delitem(self):
        data = create_test_data()
        all_items = {'time', 'dim1', 'dim2', 'dim3', 'var1', 'var2', 'var3'}
        self.assertItemsEqual(data, all_items)
        del data['var1']
        self.assertItemsEqual(data, all_items - {'var1'})
        del data['dim1']
        self.assertItemsEqual(data, {'time', 'dim2', 'dim3'})

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
        exp_index = pd.MultiIndex.from_arrays(
            [[0, 0, 0, 1, 1, 1], ['a', 'b', 'c', 'a', 'b', 'c']],
            names=['x', 'y'])
        expected = pd.DataFrame(w.reshape(-1), columns=['w'], index=exp_index)
        actual = ds.to_dataframe()
        self.assertTrue(expected.equals(actual))

    def test_lazy_load(self):
        store = InaccessibleVariableDataStore()
        store.set_dimension('dim', 10)
        store.set_variable('dim', XArray(('dim'),
                                          np.arange(10)))
        store.set_variable('var', XArray(('dim'),
                                          np.random.uniform(size=10)))
        ds = Dataset()
        ds = ds.load_store(store, decode_cf=False)
        self.assertRaises(UnexpectedDataAccess, lambda: ds['var'].data)
        ds = ds.load_store(store, decode_cf=True)
        self.assertRaises(UnexpectedDataAccess, lambda: ds['var'].data)


def create_masked_and_scaled_data():
    x = np.array([np.nan, np.nan, 10, 10.1, 10.2])
    encoding = {'_FillValue': -1, 'add_offset': 10,
                'scale_factor': np.float32(0.1), 'dtype': np.int16}
    return Dataset({'x': ('t', x, {}, encoding)})


def create_encoded_masked_and_scaled_data():
    attributes = {'_FillValue': -1, 'add_offset': 10,
                  'scale_factor': np.float32(0.1)}
    return Dataset({'x': XArray('t', [-1, -1, 0, 1, 2], attributes)})


class DatasetIOTestCases(object):
    def get_store(self):
        raise NotImplementedError

    def roundtrip(self, data, **kwargs):
        raise NotImplementedError

    def test_zero_dimensional_variable(self):
        expected = create_test_data()
        expected['xray_awesomeness'] = ([], np.array(1.e9),
                                        {'units': 'units of awesome'})
        store = self.get_store()
        expected.dump_to_store(store)
        actual = Dataset.load_store(store)
        self.assertDatasetEqual(expected, actual)

    def test_write_store(self):
        expected = create_test_data()
        store = self.get_store()
        expected.dump_to_store(store)
        actual = Dataset.load_store(store)
        self.assertDatasetEqual(expected, actual)

    def test_roundtrip_test_data(self):
        expected = create_test_data()
        actual = self.roundtrip(expected)
        self.assertDatasetEqual(expected, actual)

    def test_roundtrip_string_data(self):
        expected = Dataset({'x': ('t', ['abc', 'def'])})
        actual = self.roundtrip(expected)
        self.assertDatasetEqual(expected, actual)

    def test_roundtrip_mask_and_scale(self):
        decoded = create_masked_and_scaled_data()
        encoded = create_encoded_masked_and_scaled_data()
        self.assertDatasetEqual(decoded, self.roundtrip(decoded))
        self.assertDatasetEqual(encoded,
                                self.roundtrip(decoded, decode_cf=False))
        self.assertDatasetEqual(decoded, self.roundtrip(encoded))
        self.assertDatasetEqual(encoded,
                                self.roundtrip(encoded, decode_cf=False))

    def test_roundtrip_example_1_netcdf(self):
        expected = open_dataset(os.path.join(_test_data_path, 'example_1.nc'))
        actual = self.roundtrip(expected)
        self.assertDatasetEqual(expected, actual)

    def test_orthogonal_indexing(self):
        in_memory = create_test_data()
        on_disk = self.roundtrip(in_memory)
        indexers = {'dim1': range(3), 'dim2': range(4), 'dim3': range(5)}
        expected = in_memory.indexed_by(**indexers)
        actual = on_disk.indexed_by(**indexers)
        self.assertDatasetEqual(expected, actual)
        # do it twice, to make sure we're switched from orthogonal -> numpy
        # when we cached the values
        actual = on_disk.indexed_by(**indexers)
        self.assertDatasetEqual(expected, actual)


class NetCDF4DataTest(DatasetIOTestCases, TestCase):
    def get_store(self):
        f, self.tmp_file = tempfile.mkstemp(suffix='.nc')
        os.close(f)
        return backends.NetCDF4DataStore(self.tmp_file, mode='w')

    def tearDown(self):
        if hasattr(self, 'tmp_file') and os.path.exists(self.tmp_file):
            os.remove(self.tmp_file)

    def roundtrip(self, data, **kwargs):
        f, tmp_file = tempfile.mkstemp(suffix='.nc')
        os.close(f)
        data.dump(tmp_file)
        roundtrip_data = open_dataset(tmp_file, **kwargs)
        os.remove(tmp_file)
        return roundtrip_data

    def test_open_encodings(self):
        # Create a netCDF file with explicit time units
        # and make sure it makes it into the encodings
        # and survives a round trip
        f, tmp_file = tempfile.mkstemp(suffix='.nc')
        os.close(f)

        ds = nc4.Dataset(tmp_file, 'w')
        ds.createDimension('time', size=10)
        ds.createVariable('time', np.int32, dimensions=('time',))
        units = 'days since 1999-01-01'
        ds.variables['time'].setncattr('units', units)
        ds.variables['time'][:] = np.arange(10) + 4
        ds.close()

        expected = Dataset()

        time = pd.date_range('1999-01-05', periods=10)
        encoding = {'units': units, 'dtype': np.dtype('int32')}
        expected['time'] = ('time', time, {}, encoding)

        actual = open_dataset(tmp_file)

        self.assertXArrayEqual(actual['time'], expected['time'])
        actual_encoding = {k: v for k, v in actual['time'].encoding.iteritems()
                           if k in expected['time'].encoding}
        self.assertDictEqual(actual_encoding, expected['time'].encoding)

        os.remove(tmp_file)

    def test_dump_and_open_encodings(self):
        # Create a netCDF file with explicit time units
        # and make sure it makes it into the encodings
        # and survives a round trip
        f, tmp_file = tempfile.mkstemp(suffix='.nc')
        os.close(f)

        ds = nc4.Dataset(tmp_file, 'w')
        ds.createDimension('time', size=10)
        ds.createVariable('time', np.int32, dimensions=('time',))
        units = 'days since 1999-01-01'
        ds.variables['time'].setncattr('units', units)
        ds.variables['time'][:] = np.arange(10) + 4
        ds.close()

        xray_dataset = open_dataset(tmp_file)
        os.remove(tmp_file)
        xray_dataset.dump(tmp_file)

        ds = nc4.Dataset(tmp_file, 'r')

        self.assertEqual(ds.variables['time'].getncattr('units'), units)
        self.assertArrayEqual(ds.variables['time'], np.arange(10) + 4)

        ds.close()
        os.remove(tmp_file)

    def test_compression_encoding(self):
        data = create_test_data()
        data['var2'].encoding.update({'zlib': True,
                                      'chunksizes': (10, 10),
                                      'least_significant_digit': 2})
        actual = self.roundtrip(data)
        for k, v in data['var2'].encoding.iteritems():
            self.assertEqual(v, actual['var2'].encoding[k])

    def test_mask_and_scale(self):
        f, tmp_file = tempfile.mkstemp(suffix='.nc')
        os.close(f)

        nc = nc4.Dataset(tmp_file, mode='w')
        nc.createDimension('t', 5)
        nc.createVariable('x', 'int16', ('t',), fill_value=-1)
        v = nc.variables['x']
        v.set_auto_maskandscale(False)
        v.add_offset = 10
        v.scale_factor = 0.1
        v[:] = np.array([-1, -1, 0, 1, 2])
        nc.close()

        # first make sure netCDF4 reads the masked and scaled data correctly
        nc = nc4.Dataset(tmp_file, mode='r')
        expected = np.ma.array([-1, -1, 10, 10.1, 10.2],
                               mask=[True, True, False, False, False])
        actual = nc.variables['x'][:]
        self.assertArrayEqual(expected, actual)

        # now check xray
        ds = open_dataset(tmp_file)
        expected = create_masked_and_scaled_data()
        self.assertDatasetEqual(expected, ds)
        os.remove(tmp_file)

    def test_0dimensional_variable(self):
        # This fix verifies our work-around to this netCDF4-python bug:
        # https://github.com/Unidata/netcdf4-python/pull/220
        f, tmp_file = tempfile.mkstemp(suffix='.nc')
        os.close(f)

        nc = nc4.Dataset(tmp_file, mode='w')
        v = nc.createVariable('x', 'int16')
        v[...] = 123
        nc.close()

        ds = open_dataset(tmp_file)
        expected = Dataset({'x': ((), 123)})
        self.assertDatasetEqual(expected, ds)

    def test_lazy_decode(self):
        data = self.roundtrip(create_test_data(), decode_cf=True)
        self.assertIsInstance(data['var1'].variable._data, nc4.Variable)


class ScipyDataTest(DatasetIOTestCases, TestCase):
    def get_store(self):
        fobj = StringIO()
        return backends.ScipyDataStore(fobj, 'w')

    def roundtrip(self, data, **kwargs):
        serialized = data.dumps()
        return open_dataset(StringIO(serialized), **kwargs)
