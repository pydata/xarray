try:
    import cPickle as pickle
except ImportError:
    import pickle
import contextlib
import os.path
import tempfile
import unittest
try:  # Python 2
    from cStringIO import StringIO as BytesIO
except ImportError:  # Python 3
    from io import BytesIO

import numpy as np
import pandas as pd

from xray import Dataset, open_dataset, backends
from xray.pycompat import iteritems, itervalues, PY3

from . import TestCase, requires_scipy, requires_netCDF4, requires_pydap
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


class DatasetIOTestCases(object):
    def create_store(self):
        raise NotImplementedError

    def roundtrip(self, data, **kwargs):
        raise NotImplementedError

    def test_zero_dimensional_variable(self):
        if PY3 and type(self) is ScipyDataTest:
            # see the fix: https://github.com/scipy/scipy/pull/3617
            raise unittest.SkipTest('scipy.io.netcdf is broken on Python 3')

        expected = create_test_data()
        expected['xray_awesomeness'] = ([], np.array(1.e9),
                                        {'units': 'units of awesome'})
        with self.create_store() as store:
            expected.dump_to_store(store)
            actual = Dataset.load_store(store)
        self.assertDatasetAllClose(expected, actual)

    def test_write_store(self):
        expected = create_test_data()
        with self.create_store() as store:
            expected.dump_to_store(store)
            actual = Dataset.load_store(store)
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
                for v in actual.variables.values():
                    self.assertFalse(v._in_memory())
                yield actual
                for k, v in actual.variables.items():
                    if vars is None or k in vars:
                        self.assertTrue(v._in_memory())
                self.assertDatasetAllClose(expected, actual)

        with self.assertRaises(AssertionError):
            # make sure the contextmanager works!
            with assert_loads() as ds:
                pass

        with assert_loads() as ds:
            ds.load_data()

        with assert_loads(['var1', 'dim1', 'dim2']) as ds:
            ds['var1'].load_data()

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
        if PY3 and type(self) is ScipyDataTest:
            # see the note under test_zero_dimensional_variable
            del original['nan']
        expected = original.copy(deep=True)
        expected['letters_nans'][-1] = ''
        if type(self) is not NetCDF4DataTest:
            # for netCDF3 tests, expect the results to come back as characters
            expected['letters_nans'] = expected['letters_nans'].astype('S')
            expected['letters'] = expected['letters'].astype('S')
        with self.roundtrip(original) as actual:
            self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_string_data(self):
        expected = Dataset({'x': ('t', ['ab', 'cdef'])})
        with self.roundtrip(expected) as actual:
            if type(self) is not NetCDF4DataTest:
                expected['x'] = expected['x'].astype('S')
            self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_strings_with_fill_value(self):
        values = np.array(['ab', 'cdef', np.nan], dtype=object)
        encoding = {'_FillValue': np.string_('X'), 'dtype': np.dtype('S1')}
        original = Dataset({'x': ('t', values, {}, encoding)})
        expected = original.copy(deep=True)
        expected['x'][:2] = values[:2].astype('S')
        with self.roundtrip(original) as actual:
            self.assertDatasetIdentical(expected, actual)

        original = Dataset({'x': ('t', values, {}, {'_FillValue': '\x00'})})
        if type(self) is NetCDF4DataTest:
            # NetCDF4 should still write a VLEN (unicode) string
            expected = original.copy(deep=True)
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
            self.assertDatasetAllClose(encoded, actual)
        with self.roundtrip(encoded) as actual:
            self.assertDatasetAllClose(decoded, actual)
        with self.roundtrip(encoded, decode_cf=False) as actual:
            self.assertDatasetAllClose(encoded, actual)

    def test_roundtrip_example_1_netcdf(self):
        with open_example_dataset('example_1.nc') as expected:
            with self.roundtrip(expected) as actual:
                self.assertDatasetIdentical(expected, actual)

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
        with open_example_dataset('bears.nc') as on_disk:
            unpickled = pickle.loads(pickle.dumps(on_disk))
            self.assertDatasetIdentical(on_disk, unpickled)


@contextlib.contextmanager
def create_tmp_file(suffix='.nc'):
    f, path = tempfile.mkstemp(suffix=suffix)
    os.close(f)
    try:
        yield path
    finally:
        os.remove(path)


@requires_netCDF4
class NetCDF4DataTest(DatasetIOTestCases, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            yield backends.NetCDF4DataStore(tmp_file, mode='w')

    @contextlib.contextmanager
    def roundtrip(self, data, **kwargs):
        with create_tmp_file() as tmp_file:
            data.dump(tmp_file)
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
                                      'chunksizes': (10, 10),
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


@requires_netCDF4
@requires_scipy
class ScipyDataTest(DatasetIOTestCases, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        fobj = BytesIO()
        yield backends.ScipyDataStore(fobj, 'w')

    @contextlib.contextmanager
    def roundtrip(self, data, **kwargs):
        serialized = data.dumps()
        with open_dataset(BytesIO(serialized), **kwargs) as ds:
            yield ds


@requires_netCDF4
class NetCDF3ViaNetCDF4DataTest(DatasetIOTestCases, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            yield backends.NetCDF4DataStore(tmp_file, mode='w',
                                            format='NETCDF3_CLASSIC')

    @contextlib.contextmanager
    def roundtrip(self, data, **kwargs):
        with create_tmp_file() as tmp_file:
            data.dump(tmp_file, format='NETCDF3_CLASSIC')
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
            self.assertDatasetEqual(actual.unselect('bears'),
                                    expected.unselect('bears'))
