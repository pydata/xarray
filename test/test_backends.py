import cPickle as pickle
import contextlib
import os.path
import tempfile
from cStringIO import StringIO

import numpy as np
import pandas as pd

from xray import Dataset, open_dataset, backends

from . import TestCase, requires_scipy, requires_netCDF4, requires_pydap
from test_dataset import create_test_data

try:
    import netCDF4 as nc4
except ImportError:
    pass


def open_example_dataset(name):
    return open_dataset(os.path.join(os.path.dirname(__file__), 'data', name))


def create_masked_and_scaled_data():
    x = np.array([np.nan, np.nan, 10, 10.1, 10.2])
    encoding = {'_FillValue': -1, 'add_offset': 10,
                'scale_factor': np.float32(0.1), 'dtype': np.dtype(np.int16)}
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
        actual = self.roundtrip(expected)
        self.assertDatasetAllClose(expected, actual)

    def test_roundtrip_string_data(self):
        expected = Dataset({'x': ('t', ['abc', 'def'])})
        actual = self.roundtrip(expected)
        self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_mask_and_scale(self):
        decoded = create_masked_and_scaled_data()
        encoded = create_encoded_masked_and_scaled_data()
        self.assertDatasetAllClose(decoded, self.roundtrip(decoded))
        self.assertDatasetAllClose(encoded,
                                   self.roundtrip(decoded, decode_cf=False))
        self.assertDatasetAllClose(decoded, self.roundtrip(encoded))
        self.assertDatasetAllClose(encoded,
                                   self.roundtrip(encoded, decode_cf=False))

    def test_roundtrip_example_1_netcdf(self):
        expected = open_example_dataset('example_1.nc')
        actual = self.roundtrip(expected)
        self.assertDatasetIdentical(expected, actual)

    def test_orthogonal_indexing(self):
        in_memory = create_test_data()
        on_disk = self.roundtrip(in_memory)
        indexers = {'dim1': range(3), 'dim2': range(4), 'dim3': range(5)}
        expected = in_memory.indexed(**indexers)
        actual = on_disk.indexed(**indexers)
        self.assertDatasetAllClose(expected, actual)
        # do it twice, to make sure we're switched from orthogonal -> numpy
        # when we cached the values
        actual = on_disk.indexed(**indexers)
        self.assertDatasetAllClose(expected, actual)

    def test_pickle(self):
        on_disk = open_example_dataset('bears.nc')
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

    def roundtrip(self, data, **kwargs):
        with create_tmp_file() as tmp_file:
            data.dump(tmp_file)
            roundtrip_data = open_dataset(tmp_file, **kwargs)
        return roundtrip_data

    def test_open_encodings(self):
        # Create a netCDF file with explicit time units
        # and make sure it makes it into the encodings
        # and survives a round trip
        with create_tmp_file() as tmp_file:
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

            self.assertVariableEqual(actual['time'], expected['time'])
            actual_encoding = {k: v for k, v in actual['time'].encoding.iteritems()
                               if k in expected['time'].encoding}
            self.assertDictEqual(actual_encoding, expected['time'].encoding)

    def test_dump_and_open_encodings(self):
        # Create a netCDF file with explicit time units
        # and make sure it makes it into the encodings
        # and survives a round trip
        with create_tmp_file() as tmp_file:
            ds = nc4.Dataset(tmp_file, 'w')
            ds.createDimension('time', size=10)
            ds.createVariable('time', np.int32, dimensions=('time',))
            units = 'days since 1999-01-01'
            ds.variables['time'].setncattr('units', units)
            ds.variables['time'][:] = np.arange(10) + 4
            ds.close()

            xray_dataset = open_dataset(tmp_file)

        with create_tmp_file() as tmp_file:
            xray_dataset.dump(tmp_file)

            ds = nc4.Dataset(tmp_file, 'r')

            self.assertEqual(ds.variables['time'].getncattr('units'), units)
            self.assertArrayEqual(ds.variables['time'], np.arange(10) + 4)

            ds.close()

    def test_compression_encoding(self):
        data = create_test_data()
        data['var2'].encoding.update({'zlib': True,
                                      'chunksizes': (10, 10),
                                      'least_significant_digit': 2})
        actual = self.roundtrip(data)
        for k, v in data['var2'].encoding.iteritems():
            self.assertEqual(v, actual['var2'].encoding[k])

    def test_mask_and_scale(self):
        with create_tmp_file() as tmp_file:
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
            self.assertDatasetIdentical(expected, ds)

    def test_0dimensional_variable(self):
        # This fix verifies our work-around to this netCDF4-python bug:
        # https://github.com/Unidata/netcdf4-python/pull/220
        with create_tmp_file() as tmp_file:
            nc = nc4.Dataset(tmp_file, mode='w')
            v = nc.createVariable('x', 'int16')
            v[...] = 123
            nc.close()

            ds = open_dataset(tmp_file)
            expected = Dataset({'x': ((), 123)})
            self.assertDatasetIdentical(expected, ds)

    def test_variable_len_strings(self):
        with create_tmp_file() as tmp_file:
            values = np.array(['foo', 'bar', 'baz'], dtype=object)

            nc = nc4.Dataset(tmp_file, mode='w')
            nc.createDimension('x', 3)
            v = nc.createVariable('x', str, ('x',))
            v[:] = values
            nc.close()

            expected = Dataset({'x': ('x', values)})
            for kwargs in [{}, {'decode_cf': True}]:
                actual = open_dataset(tmp_file, **kwargs)
                self.assertDatasetIdentical(expected, actual)


@requires_netCDF4
@requires_scipy
class ScipyDataTest(DatasetIOTestCases, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        fobj = StringIO()
        yield backends.ScipyDataStore(fobj, 'w')

    def roundtrip(self, data, **kwargs):
        serialized = data.dumps()
        return open_dataset(StringIO(serialized), **kwargs)


def clear_attributes(ds):
    ds.attrs.clear()
    for v in ds.itervalues():
        v.attrs.clear()


@requires_netCDF4
@requires_pydap
class PydapTest(TestCase):
    def test_cmp_local_file(self):
        url = 'http://test.opendap.org/opendap/hyrax/data/nc/bears.nc'
        actual = Dataset.load_store(backends.PydapDataStore(url))
        expected = open_example_dataset('bears.nc')
        # don't check attributes since pydap doesn't serialize them correctly
        self.assertDatasetEqual(actual, expected)
