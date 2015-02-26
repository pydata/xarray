from io import BytesIO
import contextlib
import os.path
import pickle
import tempfile
import unittest
import sys

import numpy as np
import pandas as pd

from xray import Dataset, open_dataset, backends, decode_cf
from xray.core.pycompat import iteritems, PY3

from . import (TestCase, requires_scipy, requires_netCDF4, requires_pydap,
               has_netCDF4)
from .test_dataset import create_test_data

try:
    import netCDF4 as nc4
except ImportError:
    pass


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
            actual = decode_cf(store)
            self.assertDatasetAllClose(expected, actual)

    def test_roundtrip_test_data(self):
        expected = create_test_data()
        with self.roundtrip(expected) as actual:
            self.assertDatasetAllClose(expected, actual)

    def test_load_data(self):
        expected = create_test_data()

        @contextlib.contextmanager
        def assert_loads(vars=None):
            with self.roundtrip(expected) as actual:
                for v in actual.values():
                    self.assertFalse(v._in_memory)
                yield actual
                for k, v in actual.items():
                    if vars is None or k in vars:
                        self.assertTrue(v._in_memory)
                self.assertDatasetAllClose(expected, actual)

        with self.assertRaises(AssertionError):
            # make sure the contextmanager works!
            with assert_loads() as ds:
                pass

        with assert_loads() as ds:
            ds.load_data()

        with assert_loads(['var1', 'dim1', 'dim2']) as ds:
            ds['var1'].load_data()

        # verify we can read data even after closing the file
        with self.roundtrip(expected) as ds:
            actual = ds.load_data()
        self.assertDatasetAllClose(expected, actual)

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
        expected = Dataset({'t': ('t', times)})
        with self.roundtrip(expected) as actual:
            self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_timedelta_data(self):
        time_deltas = pd.to_timedelta(['1h', '2h', 'NaT'])
        expected = Dataset({'td': ('td', time_deltas)})
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

    def test_roundtrip_example_1_netcdf_gz(self):
        if sys.version_info[:2] < (2, 7):
            with self.assertRaisesRegexp(ValueError,
                                         'gzipped netCDF not supported'):
                open_example_dataset('example_1.nc.gz')
        else:
            with open_example_dataset('example_1.nc.gz') as expected:
                with open_example_dataset('example_1.nc') as actual:
                    self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_coordinates(self):
        original = Dataset({'foo': ('x', [0, 1])},
                           {'x': [2, 3], 'y': ('a', [42]), 'z': ('x', [4, 5])})

        with self.roundtrip(original) as actual:
            self.assertDatasetIdentical(original, actual)

        expected = original.drop('foo')
        with self.roundtrip(expected) as actual:
            self.assertDatasetIdentical(expected, actual)

        expected = original.copy()
        expected.attrs['coordinates'] = 'something random'
        with self.assertRaisesRegexp(ValueError, 'cannot serialize'):
            with self.roundtrip(expected):
                pass

        expected = original.copy(deep=True)
        expected['foo'].attrs['coordinates'] = 'something random'
        with self.assertRaisesRegexp(ValueError, 'cannot serialize'):
            with self.roundtrip(expected):
                pass

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

    def test_pickle(self):
        on_disk = open_example_dataset('bears.nc')
        unpickled = pickle.loads(pickle.dumps(on_disk))
        self.assertDatasetIdentical(on_disk, unpickled)


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
        if type(self) is NetCDF4DataTest:
            # the netCDF4 library can't keep track of an empty _FillValue for
            # VLEN variables:
            expected['x'][-1] = ''
        elif type(self) is NetCDF3ViaNetCDF4DataTest:
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
        with self.roundtrip(decoded, decode_cf=False) as actual:
            # TODO: this assumes that all roundtrips will first
            # encode.  Is that something we want to test for?
            self.assertDatasetAllClose(encoded, actual)
        with self.roundtrip(encoded, decode_cf=False) as actual:
            self.assertDatasetAllClose(encoded, actual)
        # make sure roundtrip encoding didn't change the
        # original dataset.
        self.assertDatasetIdentical(encoded,
                                    create_encoded_masked_and_scaled_data())
        with self.roundtrip(encoded) as actual:
            self.assertDatasetAllClose(decoded, actual)
        with self.roundtrip(encoded, decode_cf=False) as actual:
            self.assertDatasetAllClose(encoded, actual)


@contextlib.contextmanager
def create_tmp_file(suffix='.nc'):
    f, path = tempfile.mkstemp(suffix=suffix)
    os.close(f)
    try:
        yield path
    finally:
        os.remove(path)


@requires_netCDF4
class NetCDF4DataTest(CFEncodedDataTest, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            yield backends.NetCDF4DataStore(tmp_file, mode='w')

    @contextlib.contextmanager
    def roundtrip(self, data, **kwargs):
        with create_tmp_file() as tmp_file:
            data.to_netcdf(tmp_file)
            with open_dataset(tmp_file, **kwargs) as ds:
                yield ds

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
                actual_encoding = dict((k, v) for k, v in iteritems(actual['time'].encoding)
                                       if k in expected['time'].encoding)
                self.assertDictEqual(actual_encoding, expected['time'].encoding)

    def test_open_group(self):
        # Create a netCDF file with a dataset stored within a group
        with create_tmp_file() as tmp_file:
            rootgrp = nc4.Dataset(tmp_file, 'w')
            foogrp = rootgrp.createGroup('foo')
            ds = foogrp
            ds.createDimension('time', size=10)
            x = np.arange(10)
            ds.createVariable('x', np.int32, dimensions=('time',))
            ds.variables['x'][:] = x
            rootgrp.close()

            expected = Dataset()
            expected['x'] = ('time', x)

            # check equivalent ways to specify group
            for group in 'foo', '/foo', 'foo/', '/foo/':
                with open_dataset(tmp_file, group=group) as actual:
                    self.assertVariableEqual(actual['x'], expected['x'])

            # check that missing group raises appropriate exception
            with self.assertRaises(IOError):
                open_dataset(tmp_file, group='bar')

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
            actual1 = open_dataset(tmp_file, group='data/1')
            actual2 = open_dataset(tmp_file, group='data/2')
            self.assertDatasetIdentical(data1, actual1)
            self.assertDatasetIdentical(data2, actual2)

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

            with open_dataset(tmp_file) as xray_dataset:
                with create_tmp_file() as tmp_file2:
                    xray_dataset.dump(tmp_file2)
                    with nc4.Dataset(tmp_file2, 'r') as ds:
                        self.assertEqual(ds.variables['time'].getncattr('units'), units)
                        self.assertArrayEqual(ds.variables['time'], np.arange(10) + 4)

    def test_compression_encoding(self):
        data = create_test_data()
        data['var2'].encoding.update({'zlib': True,
                                      'chunksizes': (5, 5),
                                      'least_significant_digit': 2})
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

            # now check xray
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

        ds['z'].encoding['endian'] = 'big'
        with self.assertRaises(NotImplementedError):
            with self.roundtrip(ds) as actual:
                pass

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

            actual = open_dataset(tmp_file)

            self.assertVariableEqual(actual['time'], expected['time'])
            actual_encoding = dict((k, v) for k, v
                                   in iteritems(actual['time'].encoding)
                                   if k in expected['time'].encoding)
            self.assertDictEqual(actual_encoding, expected['time'].encoding)

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

            xray_dataset = open_dataset(tmp_file)

            with create_tmp_file() as tmp_file2:
                xray_dataset.to_netcdf(tmp_file2)

                with nc4.Dataset(tmp_file2, 'r') as ds:
                    self.assertEqual(ds.variables['time'].getncattr('units'), units)
                    self.assertArrayEqual(ds.variables['time'], np.arange(10) + 4)

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


@requires_scipy
class ScipyInMemoryDataTest(CFEncodedDataTest, Only32BitTypes, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        fobj = BytesIO()
        yield backends.ScipyDataStore(fobj, 'w')

    @contextlib.contextmanager
    def roundtrip(self, data, **kwargs):
        serialized = data.to_netcdf()
        with open_dataset(BytesIO(serialized), **kwargs) as ds:
            yield ds


@requires_scipy
class ScipyOnDiskDataTest(CFEncodedDataTest, Only32BitTypes, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            yield backends.ScipyDataStore(tmp_file, mode='w')

    @contextlib.contextmanager
    def roundtrip(self, data, **kwargs):
        if has_netCDF4:
            # this test is redundant with NetCDF4DataTest, but will fail
            # because it assumes only 32-bit types are available
            self.skipTest('only valid if netCDF4 is not installed')

        with create_tmp_file() as tmp_file:
            serialized = data.to_netcdf(tmp_file)
            with open_dataset(tmp_file, **kwargs) as ds:
                yield ds


@requires_netCDF4
class NetCDF3ViaNetCDF4DataTest(CFEncodedDataTest, Only32BitTypes, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            yield backends.NetCDF4DataStore(tmp_file, mode='w',
                                            format='NETCDF3_CLASSIC')

    @contextlib.contextmanager
    def roundtrip(self, data, **kwargs):
        with create_tmp_file() as tmp_file:
            data.to_netcdf(tmp_file, format='NETCDF3_CLASSIC')
            with open_dataset(tmp_file, **kwargs) as ds:
                yield ds


@requires_netCDF4
@requires_pydap
class PydapTest(TestCase):
    def test_cmp_local_file(self):
        url = 'http://test.opendap.org/opendap/hyrax/data/nc/bears.nc'
        actual = Dataset.load_store(backends.PydapDataStore(url))
        with open_example_dataset('bears.nc') as expected:
            # don't check attributes since pydap doesn't serialize them correctly
            # also skip the "bears" variable since the test DAP server incorrectly
            # concatenates it.
            self.assertDatasetEqual(actual.drop('bears'),
                                    expected.drop('bears'))
