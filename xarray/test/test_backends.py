from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from io import BytesIO
from threading import Lock
import contextlib
import itertools
import os.path
import pickle
import shutil
import tempfile
import unittest
import sys

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray import (Dataset, DataArray, open_dataset, open_dataarray,
                    open_mfdataset, backends, save_mfdataset)
from xarray.backends.common import robust_getitem
from xarray.backends.netCDF4_ import _extract_nc4_encoding
from xarray.core import indexing
from xarray.core.pycompat import iteritems, PY2, PY3

from . import (TestCase, requires_scipy, requires_netCDF4, requires_pydap,
               requires_scipy_or_netCDF4, requires_dask, requires_h5netcdf,
               requires_pynio, has_netCDF4, has_scipy, assert_xarray_allclose)
from .test_dataset import create_test_data

try:
    import netCDF4 as nc4
except ImportError:
    pass

try:
    import dask.array as da
except ImportError:
    pass


ON_WINDOWS = sys.platform == 'win32'


def open_example_dataset(name, *args, **kwargs):
    return open_dataset(os.path.join(os.path.dirname(__file__), 'data', name),
                        *args, **kwargs)


def create_masked_and_scaled_data():
    x = np.array([np.nan, np.nan, 10, 10.1, 10.2])
    encoding = {'_FillValue': -1, 'add_offset': 10,
                'scale_factor': np.float32(0.1), 'dtype': 'i2'}
    return Dataset({'x': ('t', x, {}, encoding)})


def create_encoded_masked_and_scaled_data():
    attributes = {'_FillValue': -1, 'add_offset': 10,
                  'scale_factor': np.float32(0.1)}
    return Dataset({'x': ('t', [-1, -1, 0, 1, 2], attributes)})


def create_boolean_data():
    attributes = {'units': '-'}
    return Dataset({'x': ('t', [True, False, False, True], attributes)})


class TestCommon(TestCase):
    def test_robust_getitem(self):

        class UnreliableArrayFailure(Exception):
            pass

        class UnreliableArray(object):
            def __init__(self, array, failures=1):
                self.array = array
                self.failures = failures

            def __getitem__(self, key):
                if self.failures > 0:
                    self.failures -= 1
                    raise UnreliableArrayFailure
                return self.array[key]

        array = UnreliableArray([0])
        with self.assertRaises(UnreliableArrayFailure):
            array[0]
        self.assertEqual(array[0], 0)

        actual = robust_getitem(array, 0, catch=UnreliableArrayFailure,
                                initial_delay=0)
        self.assertEqual(actual, 0)


class Only32BitTypes(object):
    pass


class DatasetIOTestCases(object):
    def create_store(self):
        raise NotImplementedError

    def roundtrip(self, data, **kwargs):
        raise NotImplementedError

    def test_zero_dimensional_variable(self):
        expected = create_test_data()
        expected['float_var'] = ([], 1.0e9, {'units': 'units of awesome'})
        expected['string_var'] = ([], np.array('foobar', dtype='S'))
        with self.roundtrip(expected) as actual:
            self.assertDatasetAllClose(expected, actual)

    def test_write_store(self):
        expected = create_test_data()
        with self.create_store() as store:
            expected.dump_to_store(store)
            # we need to cf decode the store because it has time and
            # non-dimension coordinates
            actual = xr.decode_cf(store)
            self.assertDatasetAllClose(expected, actual)

    def test_roundtrip_test_data(self):
        expected = create_test_data()
        with self.roundtrip(expected) as actual:
            self.assertDatasetAllClose(expected, actual)

    def test_load(self):
        expected = create_test_data()

        @contextlib.contextmanager
        def assert_loads(vars=None):
            if vars is None:
                vars = expected
            with self.roundtrip(expected) as actual:
                for k, v in actual.variables.items():
                    # IndexVariables are eagerly loaded into memory
                    self.assertEqual(v._in_memory, k in actual.dims)
                yield actual
                for k, v in actual.variables.items():
                    if k in vars:
                        self.assertTrue(v._in_memory)
                self.assertDatasetAllClose(expected, actual)

        with self.assertRaises(AssertionError):
            # make sure the contextmanager works!
            with assert_loads() as ds:
                pass

        with assert_loads() as ds:
            ds.load()

        with assert_loads(['var1', 'dim1', 'dim2']) as ds:
            ds['var1'].load()

        # verify we can read data even after closing the file
        with self.roundtrip(expected) as ds:
            actual = ds.load()
        self.assertDatasetAllClose(expected, actual)

    def test_dataset_compute(self):
        expected = create_test_data()

        with self.roundtrip(expected) as actual:
            # Test Dataset.compute()
            for k, v in actual.variables.items():
                # IndexVariables are eagerly cached
                self.assertEqual(v._in_memory, k in actual.dims)

            computed = actual.compute()

            for k, v in actual.variables.items():
                self.assertEqual(v._in_memory, k in actual.dims)
            for v in computed.variables.values():
                self.assertTrue(v._in_memory)

            self.assertDatasetAllClose(expected, actual)
            self.assertDatasetAllClose(expected, computed)

    def test_pickle(self):
        expected = Dataset({'foo': ('x', [42])})
        with self.roundtrip(
                expected, allow_cleanup_failure=ON_WINDOWS) as roundtripped:
            raw_pickle = pickle.dumps(roundtripped)
            # windows doesn't like opening the same file twice
            roundtripped.close()
            unpickled_ds = pickle.loads(raw_pickle)
            self.assertDatasetIdentical(expected, unpickled_ds)

    def test_pickle_dataarray(self):
        expected = Dataset({'foo': ('x', [42])})
        with self.roundtrip(
                expected, allow_cleanup_failure=ON_WINDOWS) as roundtripped:
            unpickled_array = pickle.loads(pickle.dumps(roundtripped['foo']))
            self.assertDatasetIdentical(expected['foo'], unpickled_array)

    def test_dataset_caching(self):
        expected = Dataset({'foo': ('x', [5, 6, 7])})
        with self.roundtrip(expected) as actual:
            assert isinstance(actual.foo.variable._data,
                              indexing.MemoryCachedArray)
            assert not actual.foo.variable._in_memory
            actual.foo.values  # cache
            assert actual.foo.variable._in_memory

        with self.roundtrip(expected, open_kwargs={'cache': False}) as actual:
            assert isinstance(actual.foo.variable._data,
                              indexing.CopyOnWriteArray)
            assert not actual.foo.variable._in_memory
            actual.foo.values  # no caching
            assert not actual.foo.variable._in_memory

    def test_roundtrip_None_variable(self):
        expected = Dataset({None: (('x', 'y'), [[0, 1], [2, 3]])})
        with self.roundtrip(expected) as actual:
            self.assertDatasetAllClose(expected, actual)

    def test_roundtrip_object_dtype(self):
        floats = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=object)
        floats_nans = np.array([np.nan, np.nan, 1.0, 2.0, 3.0], dtype=object)
        letters = np.array(['ab', 'cdef', 'g'], dtype=object)
        letters_nans = np.array(['ab', 'cdef', np.nan], dtype=object)
        all_nans = np.array([np.nan, np.nan], dtype=object)
        original = Dataset({'floats': ('a', floats),
                            'floats_nans': ('a', floats_nans),
                            'letters': ('b', letters),
                            'letters_nans': ('b', letters_nans),
                            'all_nans': ('c', all_nans),
                            'nan': ([], np.nan)})
        expected = original.copy(deep=True)
        if isinstance(self, Only32BitTypes):
            # for netCDF3 tests, expect the results to come back as characters
            expected['letters_nans'] = expected['letters_nans'].astype('S')
            expected['letters'] = expected['letters'].astype('S')
        with self.roundtrip(original) as actual:
            try:
                self.assertDatasetIdentical(expected, actual)
            except AssertionError:
                # Most stores use '' for nans in strings, but some don't
                # first try the ideal case (where the store returns exactly)
                # the original Dataset), then try a more realistic case.
                # ScipyDataTest, NetCDF3ViaNetCDF4DataTest and NetCDF4DataTest
                # all end up using this case.
                expected['letters_nans'][-1] = ''
                self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_string_data(self):
        expected = Dataset({'x': ('t', ['ab', 'cdef'])})
        with self.roundtrip(expected) as actual:
            if isinstance(self, Only32BitTypes):
                expected['x'] = expected['x'].astype('S')
            self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_datetime_data(self):
        times = pd.to_datetime(['2000-01-01', '2000-01-02', 'NaT'])
        expected = Dataset({'t': ('t', times), 't0': times[0]})
        kwds = {'encoding': {'t0': {'units': 'days since 1950-01-01'}}}
        with self.roundtrip(expected, save_kwargs=kwds) as actual:
            self.assertDatasetIdentical(expected, actual)
            self.assertEquals(actual.t0.encoding['units'],
                              'days since 1950-01-01')

    def test_roundtrip_timedelta_data(self):
        time_deltas = pd.to_timedelta(['1h', '2h', 'NaT'])
        expected = Dataset({'td': ('td', time_deltas), 'td0': time_deltas[0]})
        with self.roundtrip(expected) as actual:
            self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_float64_data(self):
        expected = Dataset({'x': ('y', np.array([1.0, 2.0, np.pi], dtype='float64'))})
        with self.roundtrip(expected) as actual:
            self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_example_1_netcdf(self):
        expected = open_example_dataset('example_1.nc')
        with self.roundtrip(expected) as actual:
            # we allow the attributes to differ since that
            # will depend on the encoding used.  For example,
            # without CF encoding 'actual' will end up with
            # a dtype attribute.
            self.assertDatasetEqual(expected, actual)

    def test_roundtrip_coordinates(self):
        original = Dataset({'foo': ('x', [0, 1])},
                           {'x': [2, 3], 'y': ('a', [42]), 'z': ('x', [4, 5])})

        with self.roundtrip(original) as actual:
            self.assertDatasetIdentical(original, actual)

        expected = original.drop('foo')
        with self.roundtrip(expected) as actual:
            self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_boolean_dtype(self):
        original = create_boolean_data()
        self.assertEqual(original['x'].dtype, 'bool')
        with self.roundtrip(original) as actual:
            self.assertDatasetIdentical(original, actual)
            self.assertEqual(actual['x'].dtype, 'bool')

    def test_orthogonal_indexing(self):
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            indexers = {'dim1': np.arange(3), 'dim2': np.arange(4),
                        'dim3': np.arange(5)}
            expected = in_memory.isel(**indexers)
            actual = on_disk.isel(**indexers)
            self.assertDatasetAllClose(expected, actual)
            # do it twice, to make sure we're switched from orthogonal -> numpy
            # when we cached the values
            actual = on_disk.isel(**indexers)
            self.assertDatasetAllClose(expected, actual)


class CFEncodedDataTest(DatasetIOTestCases):

    def test_roundtrip_strings_with_fill_value(self):
        values = np.array(['ab', 'cdef', np.nan], dtype=object)
        encoding = {'_FillValue': np.string_('X'), 'dtype': np.dtype('S1')}
        original = Dataset({'x': ('t', values, {}, encoding)})
        expected = original.copy(deep=True)
        expected['x'][:2] = values[:2].astype('S')
        with self.roundtrip(original) as actual:
            self.assertDatasetIdentical(expected, actual)

        original = Dataset({'x': ('t', values, {}, {'_FillValue': '\x00'})})
        if not isinstance(self, Only32BitTypes):
            # these stores can save unicode strings
            expected = original.copy(deep=True)
        if isinstance(self, BaseNetCDF4Test):
            # netCDF4 can't keep track of an empty _FillValue for VLEN
            # variables
            expected['x'][-1] = ''
        elif (isinstance(self, (NetCDF3ViaNetCDF4DataTest,
                                NetCDF4ClassicViaNetCDF4DataTest)) or
              (has_netCDF4 and type(self) is GenericNetCDFDataTest)):
            # netCDF4 can't keep track of an empty _FillValue for nc3, either:
            # https://github.com/Unidata/netcdf4-python/issues/273
            expected['x'][-1] = np.string_('')
        with self.roundtrip(original) as actual:
            self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_mask_and_scale(self):
        decoded = create_masked_and_scaled_data()
        encoded = create_encoded_masked_and_scaled_data()
        with self.roundtrip(decoded) as actual:
            self.assertDatasetAllClose(decoded, actual)
        with self.roundtrip(decoded, open_kwargs=dict(decode_cf=False)) as actual:
            # TODO: this assumes that all roundtrips will first
            # encode.  Is that something we want to test for?
            self.assertDatasetAllClose(encoded, actual)
        with self.roundtrip(encoded, open_kwargs=dict(decode_cf=False)) as actual:
            self.assertDatasetAllClose(encoded, actual)
        # make sure roundtrip encoding didn't change the
        # original dataset.
        self.assertDatasetIdentical(encoded,
                                    create_encoded_masked_and_scaled_data())
        with self.roundtrip(encoded) as actual:
            self.assertDatasetAllClose(decoded, actual)
        with self.roundtrip(encoded, open_kwargs=dict(decode_cf=False)) as actual:
            self.assertDatasetAllClose(encoded, actual)

    def test_coordinates_encoding(self):
        def equals_latlon(obj):
            return obj == 'lat lon' or obj == 'lon lat'

        original = Dataset({'temp': ('x', [0, 1]), 'precip': ('x', [0, -1])},
                           {'lat': ('x', [2, 3]), 'lon': ('x', [4, 5])})
        with self.roundtrip(original) as actual:
            self.assertDatasetIdentical(actual, original)
        with create_tmp_file() as tmp_file:
            original.to_netcdf(tmp_file)
            with open_dataset(tmp_file, decode_coords=False) as ds:
                self.assertTrue(equals_latlon(ds['temp'].attrs['coordinates']))
                self.assertTrue(equals_latlon(ds['precip'].attrs['coordinates']))
                self.assertNotIn('coordinates', ds.attrs)
                self.assertNotIn('coordinates', ds['lat'].attrs)
                self.assertNotIn('coordinates', ds['lon'].attrs)

        modified = original.drop(['temp', 'precip'])
        with self.roundtrip(modified) as actual:
            self.assertDatasetIdentical(actual, modified)
        with create_tmp_file() as tmp_file:
            modified.to_netcdf(tmp_file)
            with open_dataset(tmp_file, decode_coords=False) as ds:
                self.assertTrue(equals_latlon(ds.attrs['coordinates']))
                self.assertNotIn('coordinates', ds['lat'].attrs)
                self.assertNotIn('coordinates', ds['lon'].attrs)

    def test_roundtrip_endian(self):
        ds = Dataset({'x': np.arange(3, 10, dtype='>i2'),
                      'y': np.arange(3, 20, dtype='<i4'),
                      'z': np.arange(3, 30, dtype='=i8'),
                      'w': ('x', np.arange(3, 10, dtype=np.float))})

        with self.roundtrip(ds) as actual:
            # technically these datasets are slightly different,
            # one hold mixed endian data (ds) the other should be
            # all big endian (actual).  assertDatasetIdentical
            # should still pass though.
            self.assertDatasetIdentical(ds, actual)

        if type(self) is NetCDF4DataTest:
            ds['z'].encoding['endian'] = 'big'
            with self.assertRaises(NotImplementedError):
                with self.roundtrip(ds) as actual:
                    pass

    def test_invalid_dataarray_names_raise(self):
        te = (TypeError, 'string or None')
        ve = (ValueError, 'string must be length 1 or')
        data = np.random.random((2, 2))
        da = xr.DataArray(data)
        for name, e in zip([0, (4, 5), True, ''], [te, te, te, ve]):
            ds = Dataset({name: da})
            with self.assertRaisesRegexp(*e):
                with self.roundtrip(ds) as actual:
                    pass

    def test_encoding_kwarg(self):
        ds = Dataset({'x': ('y', np.arange(10.0))})
        kwargs = dict(encoding={'x': {'dtype': 'f4'}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            self.assertEqual(actual.x.encoding['dtype'], 'f4')
        self.assertEqual(ds.x.encoding, {})

        kwargs = dict(encoding={'x': {'foo': 'bar'}})
        with self.assertRaisesRegexp(ValueError, 'unexpected encoding'):
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                pass

        kwargs = dict(encoding={'x': 'foo'})
        with self.assertRaisesRegexp(ValueError, 'must be castable'):
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                pass

        kwargs = dict(encoding={'invalid': {}})
        with self.assertRaises(KeyError):
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                pass

        ds = Dataset({'t': pd.date_range('2000-01-01', periods=3)})
        units = 'days since 1900-01-01'
        kwargs = dict(encoding={'t': {'units': units}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            self.assertEqual(actual.t.encoding['units'], units)
            self.assertDatasetIdentical(actual, ds)

    def test_default_fill_value(self):
        # Test default encoding for float:
        ds = Dataset({'x': ('y', np.arange(10.0))})
        kwargs = dict(encoding={'x': {'dtype': 'f4'}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            self.assertEqual(actual.x.encoding['_FillValue'],
                             np.nan)
        self.assertEqual(ds.x.encoding, {})

        # Test default encoding for int:
        ds = Dataset({'x': ('y', np.arange(10.0))})
        kwargs = dict(encoding={'x': {'dtype': 'int16'}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            self.assertTrue('_FillValue' not in actual.x.encoding)
        self.assertEqual(ds.x.encoding, {})

        # Test default encoding for implicit int:
        ds = Dataset({'x': ('y', np.arange(10, dtype='int16'))})
        with self.roundtrip(ds) as actual:
            self.assertTrue('_FillValue' not in actual.x.encoding)
        self.assertEqual(ds.x.encoding, {})

    def test_encoding_same_dtype(self):
        ds = Dataset({'x': ('y', np.arange(10.0, dtype='f4'))})
        kwargs = dict(encoding={'x': {'dtype': 'f4'}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            self.assertEqual(actual.x.encoding['dtype'], 'f4')
        self.assertEqual(ds.x.encoding, {})


_counter = itertools.count()


@contextlib.contextmanager
def create_tmp_file(suffix='.nc', allow_cleanup_failure=False):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, 'temp-%s%s' % (next(_counter), suffix))
    try:
        yield path
    finally:
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            if not allow_cleanup_failure:
                raise

@requires_netCDF4
class BaseNetCDF4Test(CFEncodedDataTest):
    def test_open_group(self):
        # Create a netCDF file with a dataset stored within a group
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, 'w') as rootgrp:
                foogrp = rootgrp.createGroup('foo')
                ds = foogrp
                ds.createDimension('time', size=10)
                x = np.arange(10)
                ds.createVariable('x', np.int32, dimensions=('time',))
                ds.variables['x'][:] = x

            expected = Dataset()
            expected['x'] = ('time', x)

            # check equivalent ways to specify group
            for group in 'foo', '/foo', 'foo/', '/foo/':
                with open_dataset(tmp_file, group=group) as actual:
                    self.assertVariableEqual(actual['x'], expected['x'])

            # check that missing group raises appropriate exception
            with self.assertRaises(IOError):
                open_dataset(tmp_file, group='bar')
            with self.assertRaisesRegexp(ValueError, 'must be a string'):
                open_dataset(tmp_file, group=(1, 2, 3))

    def test_open_subgroup(self):
        # Create a netCDF file with a dataset stored within a group within a group
        with create_tmp_file() as tmp_file:
            rootgrp = nc4.Dataset(tmp_file, 'w')
            foogrp = rootgrp.createGroup('foo')
            bargrp = foogrp.createGroup('bar')
            ds = bargrp
            ds.createDimension('time', size=10)
            x = np.arange(10)
            ds.createVariable('x', np.int32, dimensions=('time',))
            ds.variables['x'][:] = x
            rootgrp.close()

            expected = Dataset()
            expected['x'] = ('time', x)

            # check equivalent ways to specify group
            for group in 'foo/bar', '/foo/bar', 'foo/bar/', '/foo/bar/':
                with open_dataset(tmp_file, group=group) as actual:
                    self.assertVariableEqual(actual['x'], expected['x'])

    def test_write_groups(self):
        data1 = create_test_data()
        data2 = data1 * 2
        with create_tmp_file() as tmp_file:
            data1.to_netcdf(tmp_file, group='data/1')
            data2.to_netcdf(tmp_file, group='data/2', mode='a')
            with open_dataset(tmp_file, group='data/1') as actual1:
                self.assertDatasetIdentical(data1, actual1)
            with open_dataset(tmp_file, group='data/2') as actual2:
                self.assertDatasetIdentical(data2, actual2)

    def test_roundtrip_character_array(self):
        with create_tmp_file() as tmp_file:
            values = np.array([['a', 'b', 'c'], ['d', 'e', 'f']], dtype='S')

            with nc4.Dataset(tmp_file, mode='w') as nc:
                nc.createDimension('x', 2)
                nc.createDimension('string3', 3)
                v = nc.createVariable('x', np.dtype('S1'), ('x', 'string3'))
                v[:] = values

            values = np.array(['abc', 'def'], dtype='S')
            expected = Dataset({'x': ('x', values)})
            with open_dataset(tmp_file) as actual:
                self.assertDatasetIdentical(expected, actual)
                # regression test for #157
                with self.roundtrip(actual) as roundtripped:
                    self.assertDatasetIdentical(expected, roundtripped)

    def test_default_to_char_arrays(self):
        data = Dataset({'x': np.array(['foo', 'zzzz'], dtype='S')})
        with self.roundtrip(data) as actual:
            self.assertDatasetIdentical(data, actual)
            self.assertEqual(actual['x'].dtype, np.dtype('S4'))

    def test_open_encodings(self):
        # Create a netCDF file with explicit time units
        # and make sure it makes it into the encodings
        # and survives a round trip
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, 'w') as ds:
                ds.createDimension('time', size=10)
                ds.createVariable('time', np.int32, dimensions=('time',))
                units = 'days since 1999-01-01'
                ds.variables['time'].setncattr('units', units)
                ds.variables['time'][:] = np.arange(10) + 4

            expected = Dataset()

            time = pd.date_range('1999-01-05', periods=10)
            encoding = {'units': units, 'dtype': np.dtype('int32')}
            expected['time'] = ('time', time, {}, encoding)

            with open_dataset(tmp_file) as actual:
                self.assertVariableEqual(actual['time'], expected['time'])
                actual_encoding = dict((k, v) for k, v in
                                       iteritems(actual['time'].encoding)
                                       if k in expected['time'].encoding)
                self.assertDictEqual(actual_encoding, expected['time'].encoding)

    def test_dump_encodings(self):
        # regression test for #709
        ds = Dataset({'x': ('y', np.arange(10.0))})
        kwargs = dict(encoding={'x': {'zlib': True}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            self.assertTrue(actual.x.encoding['zlib'])

    def test_dump_and_open_encodings(self):
        # Create a netCDF file with explicit time units
        # and make sure it makes it into the encodings
        # and survives a round trip
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, 'w') as ds:
                ds.createDimension('time', size=10)
                ds.createVariable('time', np.int32, dimensions=('time',))
                units = 'days since 1999-01-01'
                ds.variables['time'].setncattr('units', units)
                ds.variables['time'][:] = np.arange(10) + 4

            with open_dataset(tmp_file) as xarray_dataset:
                with create_tmp_file() as tmp_file2:
                    xarray_dataset.to_netcdf(tmp_file2)
                    with nc4.Dataset(tmp_file2, 'r') as ds:
                        self.assertEqual(ds.variables['time'].getncattr('units'), units)
                        self.assertArrayEqual(ds.variables['time'], np.arange(10) + 4)

    def test_compression_encoding(self):
        data = create_test_data()
        data['var2'].encoding.update({'zlib': True,
                                      'chunksizes': (5, 5),
                                      'fletcher32': True,
                                      'original_shape': data.var2.shape})
        with self.roundtrip(data) as actual:
            for k, v in iteritems(data['var2'].encoding):
                self.assertEqual(v, actual['var2'].encoding[k])

        # regression test for #156
        expected = data.isel(dim1=0)
        with self.roundtrip(expected) as actual:
            self.assertDatasetEqual(expected, actual)

    def test_mask_and_scale(self):
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode='w') as nc:
                nc.createDimension('t', 5)
                nc.createVariable('x', 'int16', ('t',), fill_value=-1)
                v = nc.variables['x']
                v.set_auto_maskandscale(False)
                v.add_offset = 10
                v.scale_factor = 0.1
                v[:] = np.array([-1, -1, 0, 1, 2])

            # first make sure netCDF4 reads the masked and scaled data correctly
            with nc4.Dataset(tmp_file, mode='r') as nc:
                expected = np.ma.array([-1, -1, 10, 10.1, 10.2],
                                       mask=[True, True, False, False, False])
                actual = nc.variables['x'][:]
                self.assertArrayEqual(expected, actual)

            # now check xarray
            with open_dataset(tmp_file) as ds:
                expected = create_masked_and_scaled_data()
                self.assertDatasetIdentical(expected, ds)

    def test_0dimensional_variable(self):
        # This fix verifies our work-around to this netCDF4-python bug:
        # https://github.com/Unidata/netcdf4-python/pull/220
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode='w') as nc:
                v = nc.createVariable('x', 'int16')
                v[...] = 123

            with open_dataset(tmp_file) as ds:
                expected = Dataset({'x': ((), 123)})
                self.assertDatasetIdentical(expected, ds)

    def test_variable_len_strings(self):
        with create_tmp_file() as tmp_file:
            values = np.array(['foo', 'bar', 'baz'], dtype=object)

            with nc4.Dataset(tmp_file, mode='w') as nc:
                nc.createDimension('x', 3)
                v = nc.createVariable('x', str, ('x',))
                v[:] = values

            expected = Dataset({'x': ('x', values)})
            for kwargs in [{}, {'decode_cf': True}]:
                with open_dataset(tmp_file, **kwargs) as actual:
                    self.assertDatasetIdentical(expected, actual)


@requires_netCDF4
class NetCDF4DataTest(BaseNetCDF4Test, TestCase):

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore(tmp_file, mode='w') as store:
                yield store

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        with create_tmp_file(
                allow_cleanup_failure=allow_cleanup_failure) as tmp_file:
            data.to_netcdf(tmp_file, **save_kwargs)
            with open_dataset(tmp_file, **open_kwargs) as ds:
                yield ds

    def test_variable_order(self):
        # doesn't work with scipy or h5py :(
        ds = Dataset()
        ds['a'] = 1
        ds['z'] = 2
        ds['b'] = 3
        ds.coords['c'] = 4

        with self.roundtrip(ds) as actual:
            self.assertEqual(list(ds), list(actual))

    def test_unsorted_index_raises(self):
        # should be fixed in netcdf4 v1.2.1
        random_data = np.random.random(size=(4, 6))
        dim0 = [0, 1, 2, 3]
        dim1 = [0, 2, 1, 3, 5, 4]  # We will sort this in a later step
        da = xr.DataArray(data=random_data, dims=('dim0', 'dim1'),
                          coords={'dim0': dim0, 'dim1': dim1}, name='randovar')
        ds = da.to_dataset()

        with self.roundtrip(ds) as ondisk:
            inds = np.argsort(dim1)
            ds2 = ondisk.isel(dim1=inds)
            try:
                print(ds2.randovar.values)  # should raise IndexError in netCDF4
            except IndexError as err:
                self.assertIn('first by calling .load', str(err))


@requires_netCDF4
@requires_dask
class NetCDF4ViaDaskDataTest(NetCDF4DataTest):
    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        with NetCDF4DataTest.roundtrip(
                self, data, save_kwargs, open_kwargs,
                allow_cleanup_failure) as ds:
            yield ds.chunk()

    def test_unsorted_index_raises(self):
        # Skip when using dask because dask rewrites indexers to getitem,
        # dask first pulls items by block.
        pass

    def test_dataset_caching(self):
        # caching behavior differs for dask
        pass


@requires_scipy
class ScipyInMemoryDataTest(CFEncodedDataTest, Only32BitTypes, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        fobj = BytesIO()
        yield backends.ScipyDataStore(fobj, 'w')

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        serialized = data.to_netcdf(**save_kwargs)
        with open_dataset(serialized, engine='scipy', **open_kwargs) as ds:
            yield ds

    @pytest.mark.skipif(PY2, reason='cannot pickle BytesIO on Python 2')
    def test_bytesio_pickle(self):
        data = Dataset({'foo': ('x', [1, 2, 3])})
        fobj = BytesIO(data.to_netcdf())
        with open_dataset(fobj) as ds:
            unpickled = pickle.loads(pickle.dumps(ds))
            self.assertDatasetIdentical(unpickled, data)


@requires_scipy
class ScipyOnDiskDataTest(CFEncodedDataTest, Only32BitTypes, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.ScipyDataStore(tmp_file, mode='w') as store:
                yield store

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        with create_tmp_file(
                allow_cleanup_failure=allow_cleanup_failure) as tmp_file:
            data.to_netcdf(tmp_file, engine='scipy', **save_kwargs)
            with open_dataset(tmp_file, engine='scipy', **open_kwargs) as ds:
                yield ds

    def test_array_attrs(self):
        ds = Dataset(attrs={'foo': [[1, 2], [3, 4]]})
        with self.assertRaisesRegexp(ValueError, 'must be 1-dimensional'):
            with self.roundtrip(ds) as roundtripped:
                pass

    def test_roundtrip_example_1_netcdf_gz(self):
        if sys.version_info[:2] < (2, 7):
            with self.assertRaisesRegexp(ValueError,
                                         'gzipped netCDF not supported'):
                open_example_dataset('example_1.nc.gz')
        else:
            with open_example_dataset('example_1.nc.gz') as expected:
                with open_example_dataset('example_1.nc') as actual:
                    self.assertDatasetIdentical(expected, actual)

    def test_netcdf3_endianness(self):
        # regression test for GH416
        expected = open_example_dataset('bears.nc', engine='scipy')
        for var in expected.values():
            self.assertTrue(var.dtype.isnative)


@requires_netCDF4
class NetCDF3ViaNetCDF4DataTest(CFEncodedDataTest, Only32BitTypes, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore(tmp_file, mode='w',
                                           format='NETCDF3_CLASSIC') as store:
                yield store

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        with create_tmp_file(
                allow_cleanup_failure=allow_cleanup_failure) as tmp_file:
            data.to_netcdf(tmp_file, format='NETCDF3_CLASSIC',
                           engine='netcdf4', **save_kwargs)
            with open_dataset(tmp_file, engine='netcdf4', **open_kwargs) as ds:
                yield ds


@requires_netCDF4
class NetCDF4ClassicViaNetCDF4DataTest(CFEncodedDataTest, Only32BitTypes,
                                       TestCase):
    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore(tmp_file, mode='w',
                                           format='NETCDF4_CLASSIC') as store:
                yield store

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        with create_tmp_file(
                allow_cleanup_failure=allow_cleanup_failure) as tmp_file:
            data.to_netcdf(tmp_file, format='NETCDF4_CLASSIC',
                           engine='netcdf4', **save_kwargs)
            with open_dataset(tmp_file, engine='netcdf4', **open_kwargs) as ds:
                yield ds


@requires_scipy_or_netCDF4
class GenericNetCDFDataTest(CFEncodedDataTest, Only32BitTypes, TestCase):
    # verify that we can read and write netCDF3 files as long as we have scipy
    # or netCDF4-python installed

    def test_write_store(self):
        # there's no specific store to test here
        pass

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        with create_tmp_file(
                allow_cleanup_failure=allow_cleanup_failure) as tmp_file:
            data.to_netcdf(tmp_file, format='netcdf3_64bit', **save_kwargs)
            with open_dataset(tmp_file, **open_kwargs) as ds:
                yield ds

    def test_engine(self):
        data = create_test_data()
        with self.assertRaisesRegexp(ValueError, 'unrecognized engine'):
            data.to_netcdf('foo.nc', engine='foobar')
        with self.assertRaisesRegexp(ValueError, 'invalid engine'):
            data.to_netcdf(engine='netcdf4')

        with create_tmp_file() as tmp_file:
            data.to_netcdf(tmp_file)
            with self.assertRaisesRegexp(ValueError, 'unrecognized engine'):
                open_dataset(tmp_file, engine='foobar')

        netcdf_bytes = data.to_netcdf()
        with self.assertRaisesRegexp(ValueError, 'can only read'):
            open_dataset(BytesIO(netcdf_bytes), engine='foobar')

    def test_cross_engine_read_write_netcdf3(self):
        data = create_test_data()
        valid_engines = set()
        if has_netCDF4:
            valid_engines.add('netcdf4')
        if has_scipy:
            valid_engines.add('scipy')

        for write_engine in valid_engines:
            for format in ['NETCDF3_CLASSIC', 'NETCDF3_64BIT']:
                with create_tmp_file() as tmp_file:
                    data.to_netcdf(tmp_file, format=format,
                                   engine=write_engine)
                    for read_engine in valid_engines:
                        with open_dataset(tmp_file,
                                          engine=read_engine) as actual:
                            # hack to allow test to work:
                            # coord comes back as DataArray rather than coord, and so
                            # need to loop through here rather than in the test
                            # function (or we get recursion)
                            [assert_xarray_allclose(data[k].variable, actual[k].variable)
                             for k in data]


@requires_h5netcdf
@requires_netCDF4
class H5NetCDFDataTest(BaseNetCDF4Test, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            yield backends.H5NetCDFStore(tmp_file, 'w')

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        with create_tmp_file(
                allow_cleanup_failure=allow_cleanup_failure) as tmp_file:
            data.to_netcdf(tmp_file, engine='h5netcdf', **save_kwargs)
            with open_dataset(tmp_file, engine='h5netcdf', **open_kwargs) as ds:
                yield ds

    def test_orthogonal_indexing(self):
        # doesn't work for h5py (without using dask as an intermediate layer)
        pass

    def test_complex(self):
        expected = Dataset({'x': ('y', np.ones(5) + 1j * np.ones(5))})
        with self.roundtrip(expected) as actual:
            self.assertDatasetEqual(expected, actual)

    def test_cross_engine_read_write_netcdf4(self):
        # Drop dim3, because its labels include strings. These appear to be
        # not properly read with python-netCDF4, which converts them into
        # unicode instead of leaving them as bytes.
        if PY3:
            raise unittest.SkipTest('see https://github.com/pydata/xarray/issues/535')

        data = create_test_data().drop('dim3')
        data.attrs['foo'] = 'bar'
        valid_engines = ['netcdf4', 'h5netcdf']
        for write_engine in valid_engines:
            with create_tmp_file() as tmp_file:
                data.to_netcdf(tmp_file, engine=write_engine)
                for read_engine in valid_engines:
                    with open_dataset(tmp_file, engine=read_engine) as actual:
                        self.assertDatasetIdentical(data, actual)

    def test_read_byte_attrs_as_unicode(self):
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, 'w') as nc:
                nc.foo = b'bar'
            actual = open_dataset(tmp_file)
            expected = Dataset(attrs={'foo': 'bar'})
            self.assertDatasetIdentical(expected, actual)


@requires_dask
@requires_scipy
@requires_netCDF4
class DaskTest(TestCase, DatasetIOTestCases):
    @contextlib.contextmanager
    def create_store(self):
        yield Dataset()

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        yield data.chunk()

    def test_roundtrip_datetime_data(self):
        # Override method in DatasetIOTestCases - remove not applicable save_kwds
        times = pd.to_datetime(['2000-01-01', '2000-01-02', 'NaT'])
        expected = Dataset({'t': ('t', times), 't0': times[0]})
        with self.roundtrip(expected) as actual:
            self.assertDatasetIdentical(expected, actual)

    def test_write_store(self):
        # Override method in DatasetIOTestCases - not applicable to dask
        pass

    def test_dataset_caching(self):
        expected = Dataset({'foo': ('x', [5, 6, 7])})
        with self.roundtrip(expected) as actual:
            assert not actual.foo.variable._in_memory
            actual.foo.values  # no caching
            assert not actual.foo.variable._in_memory

    def test_open_mfdataset(self):
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)
                with open_mfdataset([tmp1, tmp2]) as actual:
                    self.assertIsInstance(actual.foo.variable.data, da.Array)
                    self.assertEqual(actual.foo.variable.data.chunks,
                                     ((5, 5),))
                    self.assertDatasetAllClose(original, actual)
                with open_mfdataset([tmp1, tmp2], chunks={'x': 3}) as actual:
                    self.assertEqual(actual.foo.variable.data.chunks,
                                     ((3, 2, 3, 2),))

        with self.assertRaisesRegexp(IOError, 'no files to open'):
            open_mfdataset('foo-bar-baz-*.nc')

    def test_preprocess_mfdataset(self):
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)

            def preprocess(ds):
                return ds.assign_coords(z=0)

            expected = preprocess(original)
            with open_mfdataset(tmp, preprocess=preprocess) as actual:
                self.assertDatasetIdentical(expected, actual)

    def test_save_mfdataset_roundtrip(self):
        original = Dataset({'foo': ('x', np.random.randn(10))})
        datasets = [original.isel(x=slice(5)),
                    original.isel(x=slice(5, 10))]
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                save_mfdataset(datasets, [tmp1, tmp2])
                with open_mfdataset([tmp1, tmp2]) as actual:
                    self.assertDatasetIdentical(actual, original)

    def test_save_mfdataset_invalid(self):
        ds = Dataset()
        with self.assertRaisesRegexp(ValueError, 'cannot use mode'):
            save_mfdataset([ds, ds], ['same', 'same'])
        with self.assertRaisesRegexp(ValueError, 'same length'):
            save_mfdataset([ds, ds], ['only one path'])

    def test_open_and_do_math(self):
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)
            with open_mfdataset(tmp) as ds:
                actual = 1.0 * ds
                self.assertDatasetAllClose(original, actual)

    def test_open_mfdataset_concat_dim_none(self):
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                data = Dataset({'x': 0})
                data.to_netcdf(tmp1)
                Dataset({'x': np.nan}).to_netcdf(tmp2)
                with open_mfdataset([tmp1, tmp2], concat_dim=None) as actual:
                    self.assertDatasetIdentical(data, actual)

    def test_open_dataset(self):
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)
            with open_dataset(tmp, chunks={'x': 5}) as actual:
                self.assertIsInstance(actual.foo.variable.data, da.Array)
                self.assertEqual(actual.foo.variable.data.chunks, ((5, 5),))
                self.assertDatasetIdentical(original, actual)
            with open_dataset(tmp, chunks=5) as actual:
                self.assertDatasetIdentical(original, actual)
            with open_dataset(tmp) as actual:
                self.assertIsInstance(actual.foo.variable.data, np.ndarray)
                self.assertDatasetIdentical(original, actual)

    def test_dask_roundtrip(self):
        with create_tmp_file() as tmp:
            data = create_test_data()
            data.to_netcdf(tmp)
            chunks = {'dim1': 4, 'dim2': 4, 'dim3': 4, 'time': 10}
            with open_dataset(tmp, chunks=chunks) as dask_ds:
                self.assertDatasetIdentical(data, dask_ds)
                with create_tmp_file() as tmp2:
                    dask_ds.to_netcdf(tmp2)
                    with open_dataset(tmp2) as on_disk:
                        self.assertDatasetIdentical(data, on_disk)

    def test_deterministic_names(self):
        with create_tmp_file() as tmp:
            data = create_test_data()
            data.to_netcdf(tmp)
            with open_mfdataset(tmp) as ds:
                original_names = dict((k, v.data.name)
                                      for k, v in ds.data_vars.items())
            with open_mfdataset(tmp) as ds:
                repeat_names = dict((k, v.data.name)
                                    for k, v in ds.data_vars.items())
            for var_name, dask_name in original_names.items():
                self.assertIn(var_name, dask_name)
                self.assertIn(tmp, dask_name)
            self.assertEqual(original_names, repeat_names)

    def test_dataarray_compute(self):
        # Test DataArray.compute() on dask backend.
        # The test for Dataset.compute() is already in DatasetIOTestCases;
        # however dask is the only tested backend which supports DataArrays
        actual = DataArray([1,2]).chunk()
        computed = actual.compute()
        self.assertFalse(actual._in_memory)
        self.assertTrue(computed._in_memory)
        self.assertDataArrayAllClose(actual, computed)


@requires_scipy_or_netCDF4
@requires_pydap
class PydapTest(TestCase):
    @contextlib.contextmanager
    def create_datasets(self, **kwargs):
        url = 'http://test.opendap.org/opendap/hyrax/data/nc/bears.nc'
        actual = open_dataset(url, engine='pydap', **kwargs)
        with open_example_dataset('bears.nc') as expected:
            # don't check attributes since pydap doesn't serialize them
            # correctly also skip the "bears" variable since the test DAP
            # server incorrectly concatenates it.
            actual = actual.drop('bears')
            expected = expected.drop('bears')
            yield actual, expected

    def test_cmp_local_file(self):
        with self.create_datasets() as (actual, expected):
            self.assertDatasetEqual(actual, expected)

            # global attributes should be global attributes on the dataset
            self.assertNotIn('NC_GLOBAL', actual.attrs)
            self.assertIn('history', actual.attrs)

        with self.create_datasets() as (actual, expected):
            self.assertDatasetEqual(actual.isel(l=2), expected.isel(l=2))

        with self.create_datasets() as (actual, expected):
            self.assertDatasetEqual(actual.isel(i=0, j=-1),
                                    expected.isel(i=0, j=-1))

        with self.create_datasets() as (actual, expected):
            self.assertDatasetEqual(actual.isel(j=slice(1, 2)),
                                    expected.isel(j=slice(1, 2)))

    @requires_dask
    def test_dask(self):
        with self.create_datasets(chunks={'j': 2}) as (actual, expected):
            self.assertDatasetEqual(actual, expected)


@requires_scipy
@requires_pynio
class TestPyNio(CFEncodedDataTest, Only32BitTypes, TestCase):
    def test_write_store(self):
        # pynio is read-only for now
        pass

    def test_orthogonal_indexing(self):
        # pynio also does not support list-like indexing
        pass

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        with create_tmp_file(
                allow_cleanup_failure=allow_cleanup_failure) as tmp_file:
            data.to_netcdf(tmp_file, engine='scipy', **save_kwargs)
            with open_dataset(tmp_file, engine='pynio', **open_kwargs) as ds:
                yield ds

    def test_weakrefs(self):
        example = Dataset({'foo': ('x', np.arange(5.0))})
        expected = example.rename({'foo': 'bar', 'x': 'y'})

        with create_tmp_file() as tmp_file:
            example.to_netcdf(tmp_file, engine='scipy')
            on_disk = open_dataset(tmp_file, engine='pynio')
            actual = on_disk.rename({'foo': 'bar', 'x': 'y'})
            del on_disk  # trigger garbage collection
            self.assertDatasetIdentical(actual, expected)


class TestEncodingInvalid(TestCase):

    def test_extract_nc4_encoding(self):
        var = xr.Variable(('x',), [1, 2, 3], {}, {'foo': 'bar'})
        with self.assertRaisesRegexp(ValueError, 'unexpected encoding'):
            _extract_nc4_encoding(var, raise_on_invalid=True)

        var = xr.Variable(('x',), [1, 2, 3], {}, {'chunking': (2, 1)})
        encoding = _extract_nc4_encoding(var)
        self.assertEqual({}, encoding)

    def test_extract_h5nc_encoding(self):
        # not supported with h5netcdf (yet)
        var = xr.Variable(('x',), [1, 2, 3], {},
                          {'least_sigificant_digit': 2})
        with self.assertRaisesRegexp(ValueError, 'unexpected encoding'):
            _extract_nc4_encoding(var, raise_on_invalid=True)

class MiscObject:
    pass

@requires_netCDF4
class TestValidateAttrs(TestCase):
    def test_validating_attrs(self):
        def new_dataset():
            return Dataset({'data': ('y', np.arange(10.0))},
                           {'y': np.arange(10)})

        def new_dataset_and_dataset_attrs():
            ds = new_dataset()
            return ds, ds.attrs

        def new_dataset_and_data_attrs():
            ds = new_dataset()
            return ds, ds.data.attrs

        def new_dataset_and_coord_attrs():
            ds = new_dataset()
            return ds, ds.coords['y'].attrs

        for new_dataset_and_attrs in [new_dataset_and_dataset_attrs,
                                      new_dataset_and_data_attrs,
                                      new_dataset_and_coord_attrs]:
            ds, attrs = new_dataset_and_attrs()

            attrs[123] = 'test'
            with self.assertRaisesRegexp(TypeError, 'Invalid name for attr'):
                ds.to_netcdf('test.nc')

            ds, attrs = new_dataset_and_attrs()
            attrs[MiscObject()] = 'test'
            with self.assertRaisesRegexp(TypeError, 'Invalid name for attr'):
                ds.to_netcdf('test.nc')

            ds, attrs = new_dataset_and_attrs()
            attrs[''] = 'test'
            with self.assertRaisesRegexp(ValueError, 'Invalid name for attr'):
                ds.to_netcdf('test.nc')

            # This one should work
            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = 'test'
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = {'a': 5}
            with self.assertRaisesRegexp(TypeError, 'Invalid value for attr'):
                ds.to_netcdf('test.nc')

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = MiscObject()
            with self.assertRaisesRegexp(TypeError, 'Invalid value for attr'):
                ds.to_netcdf('test.nc')

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = 5
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = 3.14
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = [1, 2, 3, 4]
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = (1.9, 2.5)
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = np.arange(5)
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = np.arange(12).reshape(3, 4)
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = 'This is a string'
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = ''
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = np.arange(12).reshape(3, 4)
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

@requires_netCDF4
class TestDataArrayToNetCDF(TestCase):

    def test_dataarray_to_netcdf_no_name(self):
        original_da = DataArray(np.arange(12).reshape((3, 4)))

        with create_tmp_file() as tmp:
            original_da.to_netcdf(tmp)

            with open_dataarray(tmp) as loaded_da:
                self.assertDataArrayIdentical(original_da, loaded_da)


    def test_dataarray_to_netcdf_with_name(self):
        original_da = DataArray(np.arange(12).reshape((3, 4)),
                                name='test')

        with create_tmp_file() as tmp:
            original_da.to_netcdf(tmp)

            with open_dataarray(tmp) as loaded_da:
                self.assertDataArrayIdentical(original_da, loaded_da)


    def test_dataarray_to_netcdf_coord_name_clash(self):
        original_da = DataArray(np.arange(12).reshape((3, 4)),
                                dims=['x', 'y'],
                                name='x')

        with create_tmp_file() as tmp:
            original_da.to_netcdf(tmp)

            with open_dataarray(tmp) as loaded_da:
                self.assertDataArrayIdentical(original_da, loaded_da)
