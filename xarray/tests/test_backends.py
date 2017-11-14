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
from xarray.backends.netCDF4_ import _extract_nc4_variable_encoding
from xarray.core import indexing
from xarray.core.pycompat import (iteritems, PY2, ExitStack, basestring,
                                  dask_array_type)

from . import (TestCase, requires_scipy, requires_netCDF4, requires_pydap,
               requires_scipy_or_netCDF4, requires_dask, requires_h5netcdf,
               requires_pynio, requires_pathlib, has_netCDF4, has_scipy,
               assert_allclose, flaky, network, requires_rasterio,
               assert_identical, raises_regex)
from .test_dataset import create_test_data

from xarray.tests import mock, assert_identical

try:
    import netCDF4 as nc4
except ImportError:
    pass

try:
    import dask.array as da
except ImportError:
    pass

try:
    from pathlib import Path
except ImportError:
    try:
        from pathlib2 import Path
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


def create_unsigned_masked_scaled_data():
    encoding = {'_FillValue': 255, '_Unsigned': 'true', 'dtype': 'i1',
                'add_offset': 10, 'scale_factor': np.float32(0.1)}
    x = np.array([10.0, 10.1, 22.7, 22.8, np.nan])
    return Dataset({'x': ('t', x, {}, encoding)})


def create_encoded_unsigned_masked_scaled_data():
    # These are values as written to the file: the _FillValue will
    # be represented in the signed form.
    attributes = {'_FillValue': -1, '_Unsigned': 'true',
                  'add_offset': 10, 'scale_factor': np.float32(0.1)}
    # Create signed data corresponding to [0, 1, 127, 128, 255] unsigned
    sb = np.asarray([0, 1, 127, -128, -1], dtype='i1')
    return Dataset({'x': ('t', sb, attributes)})


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
        with pytest.raises(UnreliableArrayFailure):
            array[0]
        self.assertEqual(array[0], 0)

        actual = robust_getitem(array, 0, catch=UnreliableArrayFailure,
                                initial_delay=0)
        self.assertEqual(actual, 0)


class NetCDF3Only(object):
    pass


class DatasetIOTestCases(object):
    autoclose = False
    engine = None
    file_format = None

    def create_store(self):
        raise NotImplementedError

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        with create_tmp_file(
                allow_cleanup_failure=allow_cleanup_failure) as path:
            self.save(data, path, **save_kwargs)
            with self.open(path, **open_kwargs) as ds:
                yield ds

    @contextlib.contextmanager
    def roundtrip_append(self, data, save_kwargs={}, open_kwargs={},
                         allow_cleanup_failure=False):
        with create_tmp_file(
                allow_cleanup_failure=allow_cleanup_failure) as path:
            for i, key in enumerate(data.variables):
                mode = 'a' if i > 0 else 'w'
                self.save(data[[key]], path, mode=mode, **save_kwargs)
            with self.open(path, **open_kwargs) as ds:
                yield ds

    # The save/open methods may be overwritten below
    def save(self, dataset, path, **kwargs):
        dataset.to_netcdf(path, engine=self.engine, format=self.file_format,
                          **kwargs)

    @contextlib.contextmanager
    def open(self, path, **kwargs):
        with open_dataset(path, engine=self.engine, autoclose=self.autoclose,
                          **kwargs) as ds:
            yield ds

    def test_zero_dimensional_variable(self):
        expected = create_test_data()
        expected['float_var'] = ([], 1.0e9, {'units': 'units of awesome'})
        expected['bytes_var'] = ([], b'foobar')
        expected['string_var'] = ([], u'foobar')
        with self.roundtrip(expected) as actual:
            self.assertDatasetIdentical(expected, actual)

    def test_write_store(self):
        expected = create_test_data()
        with self.create_store() as store:
            expected.dump_to_store(store)
            # we need to cf decode the store because it has time and
            # non-dimension coordinates
            with xr.decode_cf(store) as actual:
                self.assertDatasetAllClose(expected, actual)

    def check_dtypes_roundtripped(self, expected, actual):
        for k in expected.variables:
            expected_dtype = expected.variables[k].dtype
            if (isinstance(self, NetCDF3Only) and expected_dtype == 'int64'):
                # downcast
                expected_dtype = np.dtype('int32')
            actual_dtype = actual.variables[k].dtype
            # TODO: check expected behavior for string dtypes more carefully
            string_kinds = {'O', 'S', 'U'}
            assert (expected_dtype == actual_dtype or
                    (expected_dtype.kind in string_kinds and
                     actual_dtype.kind in string_kinds))

    def test_roundtrip_test_data(self):
        expected = create_test_data()
        with self.roundtrip(expected) as actual:
            self.check_dtypes_roundtripped(expected, actual)
            self.assertDatasetIdentical(expected, actual)

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
                self.assertDatasetIdentical(expected, actual)

        with pytest.raises(AssertionError):
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
        self.assertDatasetIdentical(expected, actual)

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

            self.assertDatasetIdentical(expected, actual)
            self.assertDatasetIdentical(expected, computed)

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
            self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_object_dtype(self):
        floats = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=object)
        floats_nans = np.array([np.nan, np.nan, 1.0, 2.0, 3.0], dtype=object)
        bytes_ = np.array([b'ab', b'cdef', b'g'], dtype=object)
        bytes_nans = np.array([b'ab', b'cdef', np.nan], dtype=object)
        strings = np.array([u'ab', u'cdef', u'g'], dtype=object)
        strings_nans = np.array([u'ab', u'cdef', np.nan], dtype=object)
        all_nans = np.array([np.nan, np.nan], dtype=object)
        original = Dataset({'floats': ('a', floats),
                            'floats_nans': ('a', floats_nans),
                            'bytes': ('b', bytes_),
                            'bytes_nans': ('b', bytes_nans),
                            'strings': ('b', strings),
                            'strings_nans': ('b', strings_nans),
                            'all_nans': ('c', all_nans),
                            'nan': ([], np.nan)})
        expected = original.copy(deep=True)
        with self.roundtrip(original) as actual:
            try:
                self.assertDatasetIdentical(expected, actual)
            except AssertionError:
                # Most stores use '' for nans in strings, but some don't.
                # First try the ideal case (where the store returns exactly)
                # the original Dataset), then try a more realistic case.
                # This currently includes all netCDF files when encoding is not
                # explicitly set.
                # https://github.com/pydata/xarray/issues/1647
                expected['bytes_nans'][-1] = b''
                expected['strings_nans'][-1] = u''
                self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_string_data(self):
        expected = Dataset({'x': ('t', ['ab', 'cdef'])})
        with self.roundtrip(expected) as actual:
            self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_string_encoded_characters(self):
        expected = Dataset({'x': ('t', [u'ab', u'cdef'])})
        expected['x'].encoding['dtype'] = 'S1'
        with self.roundtrip(expected) as actual:
            self.assertDatasetIdentical(expected, actual)
            self.assertEqual(actual['x'].encoding['_Encoding'], 'utf-8')

        expected['x'].encoding['_Encoding'] = 'ascii'
        with self.roundtrip(expected) as actual:
            self.assertDatasetIdentical(expected, actual)
            self.assertEqual(actual['x'].encoding['_Encoding'], 'ascii')

    def test_roundtrip_datetime_data(self):
        times = pd.to_datetime(['2000-01-01', '2000-01-02', 'NaT'])
        expected = Dataset({'t': ('t', times), 't0': times[0]})
        kwds = {'encoding': {'t0': {'units': 'days since 1950-01-01'}}}
        with self.roundtrip(expected, save_kwargs=kwds) as actual:
            self.assertDatasetIdentical(expected, actual)
            assert actual.t0.encoding['units'] == 'days since 1950-01-01'

    def test_roundtrip_timedelta_data(self):
        time_deltas = pd.to_timedelta(['1h', '2h', 'NaT'])
        expected = Dataset({'td': ('td', time_deltas), 'td0': time_deltas[0]})
        with self.roundtrip(expected) as actual:
            self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_float64_data(self):
        expected = Dataset({'x': ('y', np.array([1.0, 2.0, np.pi],
                                                dtype='float64'))})
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

    def test_roundtrip_global_coordinates(self):
        original = Dataset({'x': [2, 3], 'y': ('a', [42]), 'z': ('x', [4, 5])})
        with self.roundtrip(original) as actual:
            self.assertDatasetIdentical(original, actual)

    def test_roundtrip_coordinates_with_space(self):
        original = Dataset(coords={'x': 0, 'y z': 1})
        expected = Dataset({'y z': 1}, {'x': 0})
        with pytest.warns(xr.SerializationWarning):
            with self.roundtrip(original) as actual:
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
            indexers = {'dim1': [1, 2, 0], 'dim2': [3, 2, 0, 3],
                        'dim3': np.arange(5)}
            expected = in_memory.isel(**indexers)
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)
            # do it twice, to make sure we're switched from orthogonal -> numpy
            # when we cached the values
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)

    def _test_vectorized_indexing(self, vindex_support=True):
        # Make sure vectorized_indexing works or at least raises
        # NotImplementedError
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            indexers = {'dim1': DataArray([0, 2, 0], dims='a'),
                        'dim2': DataArray([0, 2, 3], dims='a')}
            expected = in_memory.isel(**indexers)
            if vindex_support:
                actual = on_disk.isel(**indexers)
                assert_identical(expected, actual)
                # do it twice, to make sure we're switched from
                # orthogonal -> numpy when we cached the values
                actual = on_disk.isel(**indexers)
                assert_identical(expected, actual)
            else:
                with raises_regex(NotImplementedError, 'Vectorized indexing '):
                    actual = on_disk.isel(**indexers)

    def test_vectorized_indexing(self):
        # This test should be overwritten if vindex is supported
        self._test_vectorized_indexing(vindex_support=False)

    def test_isel_dataarray(self):
        # Make sure isel works lazily. GH:issue:1688
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            expected = in_memory.isel(dim2=in_memory['dim2'] < 3)
            actual = on_disk.isel(dim2=on_disk['dim2'] < 3)
            self.assertDatasetIdentical(expected, actual)

    def validate_array_type(self, ds):
        # Make sure that only NumpyIndexingAdapter stores a bare np.ndarray.
        def find_and_validate_array(obj):
            # recursively called function. obj: array or array wrapper.
            if hasattr(obj, 'array'):
                if isinstance(obj.array, indexing.ExplicitlyIndexed):
                    find_and_validate_array(obj.array)
                else:
                    if isinstance(obj.array, np.ndarray):
                        assert isinstance(obj, indexing.NumpyIndexingAdapter)
                    elif isinstance(obj.array, dask_array_type):
                        assert isinstance(obj, indexing.DaskIndexingAdapter)
                    elif isinstance(obj.array, pd.Index):
                        assert isinstance(obj, indexing.PandasIndexAdapter)
                    else:
                        raise TypeError('{} is wrapped by {}'.format(
                                            type(obj.array), type(obj)))

        for k, v in ds.variables.items():
            find_and_validate_array(v._data)

    def test_array_type_after_indexing(self):
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            self.validate_array_type(on_disk)
            indexers = {'dim1': [1, 2, 0], 'dim2': [3, 2, 0, 3],
                        'dim3': np.arange(5)}
            expected = in_memory.isel(**indexers)
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)
            self.validate_array_type(actual)
            # do it twice, to make sure we're switched from orthogonal -> numpy
            # when we cached the values
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)
            self.validate_array_type(actual)

    def test_dropna(self):
        # regression test for GH:issue:1694
        a = np.random.randn(4, 3)
        a[1, 1] = np.NaN
        in_memory = xr.Dataset({'a': (('y', 'x'), a)},
                               coords={'y': np.arange(4), 'x': np.arange(3)})

        assert_identical(in_memory.dropna(dim='x'),
                         in_memory.isel(x=slice(None, None, 2)))

        with self.roundtrip(in_memory) as on_disk:
            self.validate_array_type(on_disk)
            expected = in_memory.dropna(dim='x')
            actual = on_disk.dropna(dim='x')
            assert_identical(expected, actual)


class CFEncodedDataTest(DatasetIOTestCases):

    def test_roundtrip_bytes_with_fill_value(self):
        values = np.array([b'ab', b'cdef', np.nan], dtype=object)
        encoding = {'_FillValue': b'X', 'dtype': 'S1'}
        original = Dataset({'x': ('t', values, {}, encoding)})
        expected = original.copy(deep=True)
        with self.roundtrip(original) as actual:
            self.assertDatasetIdentical(expected, actual)

        original = Dataset({'x': ('t', values, {}, {'_FillValue': b''})})
        with self.roundtrip(original) as actual:
            self.assertDatasetIdentical(expected, actual)

    def test_roundtrip_string_with_fill_value_nchar(self):
        values = np.array([u'ab', u'cdef', np.nan], dtype=object)
        expected = Dataset({'x': ('t', values)})

        encoding = {'dtype': 'S1', '_FillValue': b'X'}
        original = Dataset({'x': ('t', values, {}, encoding)})
        # Not supported yet.
        with pytest.raises(NotImplementedError):
            with self.roundtrip(original) as actual:
                self.assertDatasetIdentical(expected, actual)

    def test_unsigned_roundtrip_mask_and_scale(self):
        decoded = create_unsigned_masked_scaled_data()
        encoded = create_encoded_unsigned_masked_scaled_data()
        with self.roundtrip(decoded) as actual:
            for k in decoded.variables:
                self.assertEqual(decoded.variables[k].dtype,
                                 actual.variables[k].dtype)
            self.assertDatasetAllClose(decoded, actual, decode_bytes=False)
        with self.roundtrip(decoded,
                            open_kwargs=dict(decode_cf=False)) as actual:
            for k in encoded.variables:
                self.assertEqual(encoded.variables[k].dtype,
                                 actual.variables[k].dtype)
            self.assertDatasetAllClose(encoded, actual, decode_bytes=False)
        with self.roundtrip(encoded,
                            open_kwargs=dict(decode_cf=False)) as actual:
            for k in encoded.variables:
                self.assertEqual(encoded.variables[k].dtype,
                                 actual.variables[k].dtype)
            self.assertDatasetAllClose(encoded, actual, decode_bytes=False)
        # make sure roundtrip encoding didn't change the
        # original dataset.
        self.assertDatasetAllClose(
            encoded, create_encoded_unsigned_masked_scaled_data())
        with self.roundtrip(encoded) as actual:
            for k in decoded.variables:
                self.assertEqual(decoded.variables[k].dtype,
                                 actual.variables[k].dtype)
            self.assertDatasetAllClose(decoded, actual, decode_bytes=False)
        with self.roundtrip(encoded,
                            open_kwargs=dict(decode_cf=False)) as actual:
            for k in encoded.variables:
                self.assertEqual(encoded.variables[k].dtype,
                                 actual.variables[k].dtype)
            self.assertDatasetAllClose(encoded, actual, decode_bytes=False)

    def test_roundtrip_mask_and_scale(self):
        decoded = create_masked_and_scaled_data()
        encoded = create_encoded_masked_and_scaled_data()
        with self.roundtrip(decoded) as actual:
            self.assertDatasetAllClose(decoded, actual, decode_bytes=False)
        with self.roundtrip(decoded,
                            open_kwargs=dict(decode_cf=False)) as actual:
            # TODO: this assumes that all roundtrips will first
            # encode.  Is that something we want to test for?
            self.assertDatasetAllClose(encoded, actual, decode_bytes=False)
        with self.roundtrip(encoded,
                            open_kwargs=dict(decode_cf=False)) as actual:
            self.assertDatasetAllClose(encoded, actual, decode_bytes=False)
        # make sure roundtrip encoding didn't change the
        # original dataset.
        self.assertDatasetAllClose(encoded,
                                   create_encoded_masked_and_scaled_data(),
                                   decode_bytes=False)
        with self.roundtrip(encoded) as actual:
            self.assertDatasetAllClose(decoded, actual, decode_bytes=False)
        with self.roundtrip(encoded,
                            open_kwargs=dict(decode_cf=False)) as actual:
            self.assertDatasetAllClose(encoded, actual, decode_bytes=False)

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
            with pytest.raises(NotImplementedError):
                with self.roundtrip(ds) as actual:
                    pass

    def test_invalid_dataarray_names_raise(self):
        te = (TypeError, 'string or None')
        ve = (ValueError, 'string must be length 1 or')
        data = np.random.random((2, 2))
        da = xr.DataArray(data)
        for name, e in zip([0, (4, 5), True, ''], [te, te, te, ve]):
            ds = Dataset({name: da})
            with raises_regex(*e):
                with self.roundtrip(ds) as actual:
                    pass

    def test_encoding_kwarg(self):
        ds = Dataset({'x': ('y', np.arange(10.0))})
        kwargs = dict(encoding={'x': {'dtype': 'f4'}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            self.assertEqual(actual.x.encoding['dtype'], 'f4')
        self.assertEqual(ds.x.encoding, {})

        kwargs = dict(encoding={'x': {'foo': 'bar'}})
        with raises_regex(ValueError, 'unexpected encoding'):
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                pass

        kwargs = dict(encoding={'x': 'foo'})
        with raises_regex(ValueError, 'must be castable'):
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                pass

        kwargs = dict(encoding={'invalid': {}})
        with pytest.raises(KeyError):
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

    def test_explicitly_omit_fill_value(self):
        ds = Dataset({'x': ('y', [np.pi, -np.pi])})
        ds.x.encoding['_FillValue'] = None
        with self.roundtrip(ds) as actual:
            assert '_FillValue' not in actual.x.encoding

    def test_encoding_same_dtype(self):
        ds = Dataset({'x': ('y', np.arange(10.0, dtype='f4'))})
        kwargs = dict(encoding={'x': {'dtype': 'f4'}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            self.assertEqual(actual.x.encoding['dtype'], 'f4')
        self.assertEqual(ds.x.encoding, {})

    def test_append_write(self):
        # regression for GH1215
        data = create_test_data()
        with self.roundtrip_append(data) as actual:
            self.assertDatasetIdentical(data, actual)

    def test_append_overwrite_values(self):
        # regression for GH1215
        data = create_test_data()
        with create_tmp_file(allow_cleanup_failure=False) as tmp_file:
            self.save(data, tmp_file, mode='w')
            data['var2'][:] = -999
            data['var9'] = data['var2'] * 3
            self.save(data[['var2', 'var9']], tmp_file, mode='a')
            with self.open(tmp_file) as actual:
                self.assertDatasetIdentical(data, actual)

    def test_vectorized_indexing(self):
        self._test_vectorized_indexing(vindex_support=False)

    def test_multiindex_not_implemented(self):
        ds = (Dataset(coords={'y': ('x', [1, 2]), 'z': ('x', ['a', 'b'])})
              .set_index(x=['y', 'z']))
        with raises_regex(NotImplementedError, 'MultiIndex'):
            with self.roundtrip(ds):
                pass


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


@contextlib.contextmanager
def create_tmp_files(nfiles, suffix='.nc', allow_cleanup_failure=False):
    with ExitStack() as stack:
        files = [stack.enter_context(create_tmp_file(suffix,
                                                     allow_cleanup_failure))
                 for apath in np.arange(nfiles)]
        yield files


@requires_netCDF4
class BaseNetCDF4Test(CFEncodedDataTest):

    engine = 'netcdf4'

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
            with pytest.raises(IOError):
                open_dataset(tmp_file, group='bar')
            with raises_regex(ValueError, 'must be a string'):
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

    def test_roundtrip_string_with_fill_value_vlen(self):
        values = np.array([u'ab', u'cdef', np.nan], dtype=object)
        expected = Dataset({'x': ('t', values)})

        # netCDF4-based backends don't support an explicit fillvalue
        # for variable length strings yet.
        # https://github.com/Unidata/netcdf4-python/issues/730
        # https://github.com/shoyer/h5netcdf/issues/37
        original = Dataset({'x': ('t', values, {}, {'_FillValue': u'XXX'})})
        with pytest.raises(NotImplementedError):
            with self.roundtrip(original) as actual:
                self.assertDatasetIdentical(expected, actual)

        original = Dataset({'x': ('t', values, {}, {'_FillValue': u''})})
        with pytest.raises(NotImplementedError):
            with self.roundtrip(original) as actual:
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
                                      'shuffle': True,
                                      'original_shape': data.var2.shape})
        with self.roundtrip(data) as actual:
            for k, v in iteritems(data['var2'].encoding):
                self.assertEqual(v, actual['var2'].encoding[k])

        # regression test for #156
        expected = data.isel(dim1=0)
        with self.roundtrip(expected) as actual:
            self.assertDatasetEqual(expected, actual)

    def test_encoding_chunksizes_unlimited(self):
        # regression test for GH1225
        ds = Dataset({'x': [1, 2, 3], 'y': ('x', [2, 3, 4])})
        ds.variables['x'].encoding = {
            'zlib': False,
            'shuffle': False,
            'complevel': 0,
            'fletcher32': False,
            'contiguous': False,
            'chunksizes': (2 ** 20,),
            'original_shape': (3,),
        }
        with self.roundtrip(ds) as actual:
            self.assertDatasetEqual(ds, actual)

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

            # first make sure netCDF4 reads the masked and scaled data
            # correctly
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

    def test_already_open_dataset(self):
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode='w') as nc:
                v = nc.createVariable('x', 'int')
                v[...] = 42

            nc = nc4.Dataset(tmp_file, mode='r')
            with backends.NetCDF4DataStore(nc, autoclose=False) as store:
                with open_dataset(store) as ds:
                    expected = Dataset({'x': ((), 42)})
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
    autoclose = False

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore.open(tmp_file, mode='w') as store:
                yield store

    def test_variable_order(self):
        # doesn't work with scipy or h5py :(
        ds = Dataset()
        ds['a'] = 1
        ds['z'] = 2
        ds['b'] = 3
        ds.coords['c'] = 4

        with self.roundtrip(ds) as actual:
            self.assertEqual(list(ds.variables), list(actual.variables))

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


class NetCDF4DataStoreAutocloseTrue(NetCDF4DataTest):
    autoclose = True


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

    def test_vectorized_indexing(self):
        self._test_vectorized_indexing(vindex_support=True)


class NetCDF4ViaDaskDataTestAutocloseTrue(NetCDF4ViaDaskDataTest):
    autoclose = True


@requires_scipy
class ScipyInMemoryDataTest(CFEncodedDataTest, NetCDF3Only, TestCase):
    engine = 'scipy'

    @contextlib.contextmanager
    def create_store(self):
        fobj = BytesIO()
        yield backends.ScipyDataStore(fobj, 'w')

    def test_to_netcdf_explicit_engine(self):
        # regression test for GH1321
        Dataset({'foo': 42}).to_netcdf(engine='scipy')

    @pytest.mark.skipif(PY2, reason='cannot pickle BytesIO on Python 2')
    def test_bytesio_pickle(self):
        data = Dataset({'foo': ('x', [1, 2, 3])})
        fobj = BytesIO(data.to_netcdf())
        with open_dataset(fobj, autoclose=self.autoclose) as ds:
            unpickled = pickle.loads(pickle.dumps(ds))
            self.assertDatasetIdentical(unpickled, data)


class ScipyInMemoryDataTestAutocloseTrue(ScipyInMemoryDataTest):
    autoclose = True


@requires_scipy
class ScipyFileObjectTest(CFEncodedDataTest, NetCDF3Only, TestCase):
    engine = 'scipy'

    @contextlib.contextmanager
    def create_store(self):
        fobj = BytesIO()
        yield backends.ScipyDataStore(fobj, 'w')

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        with create_tmp_file() as tmp_file:
            with open(tmp_file, 'wb') as f:
                self.save(data, f, **save_kwargs)
            with open(tmp_file, 'rb') as f:
                with self.open(f, **open_kwargs) as ds:
                    yield ds

    @pytest.mark.skip(reason='cannot pickle file objects')
    def test_pickle(self):
        pass

    @pytest.mark.skip(reason='cannot pickle file objects')
    def test_pickle_dataarray(self):
        pass


@requires_scipy
class ScipyFilePathTest(CFEncodedDataTest, NetCDF3Only, TestCase):
    engine = 'scipy'

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.ScipyDataStore(tmp_file, mode='w') as store:
                yield store

    def test_array_attrs(self):
        ds = Dataset(attrs={'foo': [[1, 2], [3, 4]]})
        with raises_regex(ValueError, 'must be 1-dimensional'):
            with self.roundtrip(ds) as roundtripped:
                pass

    def test_roundtrip_example_1_netcdf_gz(self):
        with open_example_dataset('example_1.nc.gz') as expected:
            with open_example_dataset('example_1.nc') as actual:
                self.assertDatasetIdentical(expected, actual)

    def test_netcdf3_endianness(self):
        # regression test for GH416
        expected = open_example_dataset('bears.nc', engine='scipy')
        for var in expected.variables.values():
            self.assertTrue(var.dtype.isnative)

    @requires_netCDF4
    def test_nc4_scipy(self):
        with create_tmp_file(allow_cleanup_failure=True) as tmp_file:
            with nc4.Dataset(tmp_file, 'w', format='NETCDF4') as rootgrp:
                rootgrp.createGroup('foo')

            with raises_regex(TypeError, 'pip install netcdf4'):
                open_dataset(tmp_file, engine='scipy')


class ScipyFilePathTestAutocloseTrue(ScipyFilePathTest):
    autoclose = True


@requires_netCDF4
class NetCDF3ViaNetCDF4DataTest(CFEncodedDataTest, NetCDF3Only, TestCase):
    engine = 'netcdf4'
    file_format = 'NETCDF3_CLASSIC'

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore.open(
                    tmp_file, mode='w', format='NETCDF3_CLASSIC') as store:
                yield store


class NetCDF3ViaNetCDF4DataTestAutocloseTrue(NetCDF3ViaNetCDF4DataTest):
    autoclose = True


@requires_netCDF4
class NetCDF4ClassicViaNetCDF4DataTest(CFEncodedDataTest, NetCDF3Only,
                                       TestCase):
    engine = 'netcdf4'
    file_format = 'NETCDF4_CLASSIC'

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore.open(
                    tmp_file, mode='w', format='NETCDF4_CLASSIC') as store:
                yield store


class NetCDF4ClassicViaNetCDF4DataTestAutocloseTrue(
        NetCDF4ClassicViaNetCDF4DataTest):
    autoclose = True


@requires_scipy_or_netCDF4
class GenericNetCDFDataTest(CFEncodedDataTest, NetCDF3Only, TestCase):
    # verify that we can read and write netCDF3 files as long as we have scipy
    # or netCDF4-python installed
    file_format = 'netcdf3_64bit'

    def test_write_store(self):
        # there's no specific store to test here
        pass

    def test_engine(self):
        data = create_test_data()
        with raises_regex(ValueError, 'unrecognized engine'):
            data.to_netcdf('foo.nc', engine='foobar')
        with raises_regex(ValueError, 'invalid engine'):
            data.to_netcdf(engine='netcdf4')

        with create_tmp_file() as tmp_file:
            data.to_netcdf(tmp_file)
            with raises_regex(ValueError, 'unrecognized engine'):
                open_dataset(tmp_file, engine='foobar')

        netcdf_bytes = data.to_netcdf()
        with raises_regex(ValueError, 'can only read'):
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
                            # coord comes back as DataArray rather than coord,
                            # and so need to loop through here rather than in
                            # the test function (or we get recursion)
                            [assert_allclose(data[k].variable,
                                             actual[k].variable)
                             for k in data.variables]

    def test_encoding_unlimited_dims(self):
        ds = Dataset({'x': ('y', np.arange(10.0))})
        with self.roundtrip(ds,
                            save_kwargs=dict(unlimited_dims=['y'])) as actual:
            self.assertEqual(actual.encoding['unlimited_dims'], set('y'))
            self.assertDatasetEqual(ds, actual)

        ds.encoding = {'unlimited_dims': ['y']}
        with self.roundtrip(ds) as actual:
            self.assertEqual(actual.encoding['unlimited_dims'], set('y'))
            self.assertDatasetEqual(ds, actual)


class GenericNetCDFDataTestAutocloseTrue(GenericNetCDFDataTest):
    autoclose = True


@requires_h5netcdf
@requires_netCDF4
class H5NetCDFDataTest(BaseNetCDF4Test, TestCase):
    engine = 'h5netcdf'

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            yield backends.H5NetCDFStore(tmp_file, 'w')

    def test_orthogonal_indexing(self):
        # doesn't work for h5py (without using dask as an intermediate layer)
        pass

    def test_array_type_after_indexing(self):
        # pynio also does not support list-like indexing
        pass

    def test_complex(self):
        expected = Dataset({'x': ('y', np.ones(5) + 1j * np.ones(5))})
        with self.roundtrip(expected) as actual:
            self.assertDatasetEqual(expected, actual)

    @pytest.mark.xfail(reason='https://github.com/pydata/xarray/issues/535')
    def test_cross_engine_read_write_netcdf4(self):
        # Drop dim3, because its labels include strings. These appear to be
        # not properly read with python-netCDF4, which converts them into
        # unicode instead of leaving them as bytes.
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
            with open_dataset(tmp_file) as actual:
                expected = Dataset(attrs={'foo': 'bar'})
                self.assertDatasetIdentical(expected, actual)

    def test_encoding_unlimited_dims(self):
        ds = Dataset({'x': ('y', np.arange(10.0))})
        with self.roundtrip(ds,
                            save_kwargs=dict(unlimited_dims=['y'])) as actual:
            self.assertEqual(actual.encoding['unlimited_dims'], set('y'))
            self.assertDatasetEqual(ds, actual)
        ds.encoding = {'unlimited_dims': ['y']}
        with self.roundtrip(ds) as actual:
            self.assertEqual(actual.encoding['unlimited_dims'], set('y'))
            self.assertDatasetEqual(ds, actual)


# tests pending h5netcdf fix
@unittest.skip
class H5NetCDFDataTestAutocloseTrue(H5NetCDFDataTest):
    autoclose = True


class OpenMFDatasetManyFilesTest(TestCase):
    def validate_open_mfdataset_autoclose(self, engine, nfiles=10):
        randdata = np.random.randn(nfiles)
        original = Dataset({'foo': ('x', randdata)})
        # test standard open_mfdataset approach with too many files
        with create_tmp_files(nfiles) as tmpfiles:
            for readengine in engine:
                writeengine = (readengine if readengine != 'pynio'
                               else 'netcdf4')
                # split into multiple sets of temp files
                for ii in original.x.values:
                    subds = original.isel(x=slice(ii, ii+1))
                    subds.to_netcdf(tmpfiles[ii], engine=writeengine)

                # check that calculation on opened datasets works properly
                ds = open_mfdataset(tmpfiles, engine=readengine,
                                    autoclose=True)
                self.assertAllClose(ds.x.sum().values, (nfiles*(nfiles-1))/2)
                self.assertAllClose(ds.foo.sum().values, np.sum(randdata))
                self.assertAllClose(ds.sum().foo.values, np.sum(randdata))
                ds.close()

    def validate_open_mfdataset_large_num_files(self, engine):
        self.validate_open_mfdataset_autoclose(engine, nfiles=2000)

    @requires_dask
    @requires_netCDF4
    def test_1_autoclose_netcdf4(self):
        self.validate_open_mfdataset_autoclose(engine=['netcdf4'])

    @requires_dask
    @requires_scipy
    def test_2_autoclose_scipy(self):
        self.validate_open_mfdataset_autoclose(engine=['scipy'])

    @requires_dask
    @requires_pynio
    def test_3_autoclose_pynio(self):
        self.validate_open_mfdataset_autoclose(engine=['pynio'])

    # use of autoclose=True with h5netcdf broken because of
    # probable h5netcdf error
    @requires_dask
    @requires_h5netcdf
    @pytest.mark.xfail
    def test_4_autoclose_h5netcdf(self):
        self.validate_open_mfdataset_autoclose(engine=['h5netcdf'])

    # These tests below are marked as flaky (and skipped by default) because
    # they fail sometimes on Travis-CI, for no clear reason.

    @requires_dask
    @requires_netCDF4
    @flaky
    @pytest.mark.slow
    def test_1_open_large_num_files_netcdf4(self):
        self.validate_open_mfdataset_large_num_files(engine=['netcdf4'])

    @requires_dask
    @requires_scipy
    @flaky
    @pytest.mark.slow
    def test_2_open_large_num_files_scipy(self):
        self.validate_open_mfdataset_large_num_files(engine=['scipy'])

    @requires_dask
    @requires_pynio
    @flaky
    @pytest.mark.slow
    def test_3_open_large_num_files_pynio(self):
        self.validate_open_mfdataset_large_num_files(engine=['pynio'])

    # use of autoclose=True with h5netcdf broken because of
    # probable h5netcdf error
    @requires_dask
    @requires_h5netcdf
    @flaky
    @pytest.mark.xfail
    @pytest.mark.slow
    def test_4_open_large_num_files_h5netcdf(self):
        self.validate_open_mfdataset_large_num_files(engine=['h5netcdf'])


@requires_scipy_or_netCDF4
class OpenMFDatasetWithDataVarsAndCoordsKwTest(TestCase):
    coord_name = 'lon'
    var_name = 'v1'

    @contextlib.contextmanager
    def setup_files_and_datasets(self):
        ds1, ds2 = self.gen_datasets_with_common_coord_and_time()
        with create_tmp_file() as tmpfile1:
            with create_tmp_file() as tmpfile2:

                # save data to the temporary files
                ds1.to_netcdf(tmpfile1)
                ds2.to_netcdf(tmpfile2)

                yield [tmpfile1, tmpfile2], [ds1, ds2]

    def gen_datasets_with_common_coord_and_time(self):
        # create coordinate data
        nx = 10
        nt = 10
        x = np.arange(nx)
        t1 = np.arange(nt)
        t2 = np.arange(nt, 2 * nt, 1)

        v1 = np.random.randn(nt, nx)
        v2 = np.random.randn(nt, nx)

        ds1 = Dataset(data_vars={self.var_name: (['t', 'x'], v1),
                                 self.coord_name: ('x', 2 * x)},
                      coords={
                          't': (['t', ], t1),
                          'x': (['x', ], x)
                      })

        ds2 = Dataset(data_vars={self.var_name: (['t', 'x'], v2),
                                 self.coord_name: ('x', 2 * x)},
                      coords={
                          't': (['t', ], t2),
                          'x': (['x', ], x)
                      })

        return ds1, ds2

    def test_open_mfdataset_does_same_as_concat(self):
        options = ['all', 'minimal', 'different', ]

        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
            for opt in options:
                with open_mfdataset(files, data_vars=opt) as ds:
                    kwargs = dict(data_vars=opt, dim='t')
                    ds_expect = xr.concat([ds1, ds2], **kwargs)
                    self.assertDatasetIdentical(ds, ds_expect)

                with open_mfdataset(files, coords=opt) as ds:
                    kwargs = dict(coords=opt, dim='t')
                    ds_expect = xr.concat([ds1, ds2], **kwargs)
                    self.assertDatasetIdentical(ds, ds_expect)

    def test_common_coord_when_datavars_all(self):
        opt = 'all'

        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
            # open the files with the data_var option
            with open_mfdataset(files, data_vars=opt) as ds:

                coord_shape = ds[self.coord_name].shape
                coord_shape1 = ds1[self.coord_name].shape
                coord_shape2 = ds2[self.coord_name].shape

                var_shape = ds[self.var_name].shape

                self.assertEqual(var_shape, coord_shape)
                self.assertNotEqual(coord_shape1, coord_shape)
                self.assertNotEqual(coord_shape2, coord_shape)

    def test_common_coord_when_datavars_minimal(self):
        opt = 'minimal'

        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
            # open the files using data_vars option
            with open_mfdataset(files, data_vars=opt) as ds:

                coord_shape = ds[self.coord_name].shape
                coord_shape1 = ds1[self.coord_name].shape
                coord_shape2 = ds2[self.coord_name].shape

                var_shape = ds[self.var_name].shape

                self.assertNotEqual(var_shape, coord_shape)
                self.assertEqual(coord_shape1, coord_shape)
                self.assertEqual(coord_shape2, coord_shape)

    def test_invalid_data_vars_value_should_fail(self):

        with self.setup_files_and_datasets() as (files, _):
            with pytest.raises(ValueError):
                with open_mfdataset(files, data_vars='minimum'):
                    pass

            # test invalid coord parameter
            with pytest.raises(ValueError):
                with open_mfdataset(files, coords='minimum'):
                    pass


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

    # Override methods in DatasetIOTestCases - not applicable to dask
    def test_roundtrip_string_encoded_characters(self):
        pass

    def test_roundtrip_coordinates_with_space(self):
        pass

    def test_roundtrip_datetime_data(self):
        # Override method in DatasetIOTestCases - remove not applicable
        # save_kwds
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
                with open_mfdataset([tmp1, tmp2],
                                    autoclose=self.autoclose) as actual:
                    self.assertIsInstance(actual.foo.variable.data, da.Array)
                    self.assertEqual(actual.foo.variable.data.chunks,
                                     ((5, 5),))
                    self.assertDatasetIdentical(original, actual)
                with open_mfdataset([tmp1, tmp2], chunks={'x': 3},
                                    autoclose=self.autoclose) as actual:
                    self.assertEqual(actual.foo.variable.data.chunks,
                                     ((3, 2, 3, 2),))

        with raises_regex(IOError, 'no files to open'):
            open_mfdataset('foo-bar-baz-*.nc', autoclose=self.autoclose)

    @requires_pathlib
    def test_open_mfdataset_pathlib(self):
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                tmp1 = Path(tmp1)
                tmp2 = Path(tmp2)
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)
                with open_mfdataset([tmp1, tmp2],
                                    autoclose=self.autoclose) as actual:
                    self.assertDatasetIdentical(original, actual)

    def test_attrs_mfdataset(self):
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                ds1 = original.isel(x=slice(5))
                ds2 = original.isel(x=slice(5, 10))
                ds1.attrs['test1'] = 'foo'
                ds2.attrs['test2'] = 'bar'
                ds1.to_netcdf(tmp1)
                ds2.to_netcdf(tmp2)
                with open_mfdataset([tmp1, tmp2]) as actual:
                    # presumes that attributes inherited from
                    # first dataset loaded
                    self.assertEqual(actual.test1, ds1.test1)
                    # attributes from ds2 are not retained, e.g.,
                    with raises_regex(AttributeError,
                                                 'no attribute'):
                        actual.test2

    def test_preprocess_mfdataset(self):
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)

            def preprocess(ds):
                return ds.assign_coords(z=0)

            expected = preprocess(original)
            with open_mfdataset(tmp, preprocess=preprocess,
                                autoclose=self.autoclose) as actual:
                self.assertDatasetIdentical(expected, actual)

    def test_save_mfdataset_roundtrip(self):
        original = Dataset({'foo': ('x', np.random.randn(10))})
        datasets = [original.isel(x=slice(5)),
                    original.isel(x=slice(5, 10))]
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                save_mfdataset(datasets, [tmp1, tmp2])
                with open_mfdataset([tmp1, tmp2],
                                    autoclose=self.autoclose) as actual:
                    self.assertDatasetIdentical(actual, original)

    def test_save_mfdataset_invalid(self):
        ds = Dataset()
        with raises_regex(ValueError, 'cannot use mode'):
            save_mfdataset([ds, ds], ['same', 'same'])
        with raises_regex(ValueError, 'same length'):
            save_mfdataset([ds, ds], ['only one path'])

    def test_save_mfdataset_invalid_dataarray(self):
        # regression test for GH1555
        da = DataArray([1, 2])
        with raises_regex(TypeError, 'supports writing Dataset'):
            save_mfdataset([da], ['dataarray'])

    @requires_pathlib
    def test_save_mfdataset_pathlib_roundtrip(self):
        original = Dataset({'foo': ('x', np.random.randn(10))})
        datasets = [original.isel(x=slice(5)),
                    original.isel(x=slice(5, 10))]
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                tmp1 = Path(tmp1)
                tmp2 = Path(tmp2)
                save_mfdataset(datasets, [tmp1, tmp2])
                with open_mfdataset([tmp1, tmp2],
                                    autoclose=self.autoclose) as actual:
                    self.assertDatasetIdentical(actual, original)

    def test_open_and_do_math(self):
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)
            with open_mfdataset(tmp, autoclose=self.autoclose) as ds:
                actual = 1.0 * ds
                self.assertDatasetAllClose(original, actual,
                                           decode_bytes=False)

    def test_open_mfdataset_concat_dim_none(self):
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                data = Dataset({'x': 0})
                data.to_netcdf(tmp1)
                Dataset({'x': np.nan}).to_netcdf(tmp2)
                with open_mfdataset([tmp1, tmp2], concat_dim=None,
                                    autoclose=self.autoclose) as actual:
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
            with open_mfdataset(tmp, autoclose=self.autoclose) as ds:
                original_names = dict((k, v.data.name)
                                      for k, v in ds.data_vars.items())
            with open_mfdataset(tmp, autoclose=self.autoclose) as ds:
                repeat_names = dict((k, v.data.name)
                                    for k, v in ds.data_vars.items())
            for var_name, dask_name in original_names.items():
                self.assertIn(var_name, dask_name)
                self.assertEqual(dask_name[:13], 'open_dataset-')
            self.assertEqual(original_names, repeat_names)

    def test_dataarray_compute(self):
        # Test DataArray.compute() on dask backend.
        # The test for Dataset.compute() is already in DatasetIOTestCases;
        # however dask is the only tested backend which supports DataArrays
        actual = DataArray([1,2]).chunk()
        computed = actual.compute()
        self.assertFalse(actual._in_memory)
        self.assertTrue(computed._in_memory)
        self.assertDataArrayAllClose(actual, computed, decode_bytes=False)

    def test_vectorized_indexing(self):
        self._test_vectorized_indexing(vindex_support=True)


class DaskTestAutocloseTrue(DaskTest):
    autoclose = True


@network
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

    def test_session(self):
        from pydap.cas.urs import setup_session

        session = setup_session('XarrayTestUser', 'Xarray2017')
        with mock.patch('pydap.client.open_url') as mock_func:
            xr.backends.PydapDataStore.open('http://test.url', session=session)
        mock_func.assert_called_with('http://test.url', session=session)

    @requires_dask
    def test_dask(self):
        with self.create_datasets(chunks={'j': 2}) as (actual, expected):
            self.assertDatasetEqual(actual, expected)


@requires_scipy
@requires_pynio
class TestPyNio(CFEncodedDataTest, NetCDF3Only, TestCase):
    def test_write_store(self):
        # pynio is read-only for now
        pass

    def test_orthogonal_indexing(self):
        # pynio also does not support list-like indexing
        with raises_regex(NotImplementedError, 'Outer indexing'):
            super(TestPyNio, self).test_orthogonal_indexing()

    def test_isel_dataarray(self):
        with raises_regex(NotImplementedError, 'Outer indexing'):
            super(TestPyNio, self).test_isel_dataarray()

    def test_array_type_after_indexing(self):
        # pynio also does not support list-like indexing
        pass

    @contextlib.contextmanager
    def open(self, path, **kwargs):
        with open_dataset(path, engine='pynio', autoclose=self.autoclose,
                          **kwargs) as ds:
            yield ds

    def save(self, dataset, path, **kwargs):
        dataset.to_netcdf(path, engine='scipy', **kwargs)

    def test_weakrefs(self):
        example = Dataset({'foo': ('x', np.arange(5.0))})
        expected = example.rename({'foo': 'bar', 'x': 'y'})

        with create_tmp_file() as tmp_file:
            example.to_netcdf(tmp_file, engine='scipy')
            on_disk = open_dataset(tmp_file, engine='pynio')
            actual = on_disk.rename({'foo': 'bar', 'x': 'y'})
            del on_disk  # trigger garbage collection
            self.assertDatasetIdentical(actual, expected)


class TestPyNioAutocloseTrue(TestPyNio):
    autoclose = True


@requires_rasterio
class TestRasterio(TestCase):

    @requires_scipy_or_netCDF4
    def test_serialization(self):
        import rasterio
        from rasterio.transform import from_origin

        # Create a geotiff file in utm proj
        with create_tmp_file(suffix='.tif') as tmp_file:
            # data
            nx, ny, nz = 4, 3, 3
            data = np.arange(nx*ny*nz,
                             dtype=rasterio.float32).reshape(nz, ny, nx)
            transform = from_origin(5000, 80000, 1000, 2000.)
            with rasterio.open(
                    tmp_file, 'w',
                    driver='GTiff', height=ny, width=nx, count=nz,
                    crs={'units': 'm', 'no_defs': True, 'ellps': 'WGS84',
                         'proj': 'utm', 'zone': 18},
                    transform=transform,
                    dtype=rasterio.float32) as s:
                s.write(data)

            # Write it to a netcdf and read again (roundtrip)
            with xr.open_rasterio(tmp_file) as rioda:
                with create_tmp_file(suffix='.nc') as tmp_nc_file:
                    rioda.to_netcdf(tmp_nc_file)
                    with xr.open_dataarray(tmp_nc_file) as ncds:
                        assert_identical(rioda, ncds)

    def test_utm(self):
        import rasterio
        from rasterio.transform import from_origin

        # Create a geotiff file in utm proj
        with create_tmp_file(suffix='.tif') as tmp_file:
            # data
            nx, ny, nz = 4, 3, 3
            data = np.arange(nx*ny*nz,
                             dtype=rasterio.float32).reshape(nz, ny, nx)
            transform = from_origin(5000, 80000, 1000, 2000.)
            with rasterio.open(
                    tmp_file, 'w',
                    driver='GTiff', height=ny, width=nx, count=nz,
                    crs={'units': 'm', 'no_defs': True, 'ellps': 'WGS84',
                         'proj': 'utm', 'zone': 18},
                    transform=transform,
                    dtype=rasterio.float32) as s:
                s.write(data)
                dx, dy = s.res[0], -s.res[1]

            # Tests
            expected = DataArray(data, dims=('band', 'y', 'x'),
                                 coords={
                                     'band': [1, 2, 3],
                                     'y': -np.arange(ny) * 2000 + 80000 + dy/2,
                                     'x': np.arange(nx) * 1000 + 5000 + dx/2,
                                 })
            with xr.open_rasterio(tmp_file) as rioda:
                assert_allclose(rioda, expected)
                assert 'crs' in rioda.attrs
                assert isinstance(rioda.attrs['crs'], basestring)
                assert 'res' in rioda.attrs
                assert isinstance(rioda.attrs['res'], tuple)
                assert 'is_tiled' in rioda.attrs
                assert isinstance(rioda.attrs['is_tiled'], np.uint8)
                assert 'transform' in rioda.attrs
                assert isinstance(rioda.attrs['transform'], tuple)

    def test_platecarree(self):

        import rasterio
        from rasterio.transform import from_origin

        # Create a geotiff file in latlong proj
        with create_tmp_file(suffix='.tif') as tmp_file:
            # data
            nx, ny = 8, 10
            data = np.arange(80, dtype=rasterio.float32).reshape(ny, nx)
            transform = from_origin(1, 2, 0.5, 2.)
            with rasterio.open(
                    tmp_file, 'w',
                    driver='GTiff', height=ny, width=nx, count=1,
                    crs='+proj=latlong',
                    transform=transform,
                    dtype=rasterio.float32) as s:
                s.write(data, indexes=1)
                dx, dy = s.res[0], -s.res[1]

            # Tests
            expected = DataArray(data[np.newaxis, ...],
                                 dims=('band', 'y', 'x'),
                                 coords={'band': [1],
                                         'y': -np.arange(ny)*2 + 2 + dy/2,
                                         'x': np.arange(nx)*0.5 + 1 + dx/2,
                                         })
            with xr.open_rasterio(tmp_file) as rioda:
                assert_allclose(rioda, expected)
                assert 'crs' in rioda.attrs
                assert isinstance(rioda.attrs['crs'], basestring)
                assert 'res' in rioda.attrs
                assert isinstance(rioda.attrs['res'], tuple)
                assert 'is_tiled' in rioda.attrs
                assert isinstance(rioda.attrs['is_tiled'], np.uint8)
                assert 'transform' in rioda.attrs
                assert isinstance(rioda.attrs['transform'], tuple)

    def test_indexing(self):

        import rasterio
        from rasterio.transform import from_origin

        # Create a geotiff file in latlong proj
        with create_tmp_file(suffix='.tif') as tmp_file:
            # data
            nx, ny, nz = 8, 10, 3
            data = np.arange(nx*ny*nz,
                             dtype=rasterio.float32).reshape(nz, ny, nx)
            transform = from_origin(1, 2, 0.5, 2.)
            with rasterio.open(
                    tmp_file, 'w',
                    driver='GTiff', height=ny, width=nx, count=nz,
                    crs='+proj=latlong',
                    transform=transform,
                    dtype=rasterio.float32) as s:
                s.write(data)
                dx, dy = s.res[0], -s.res[1]

            # ref
            expected = DataArray(data, dims=('band', 'y', 'x'),
                                 coords={'x': (np.arange(nx)*0.5 + 1) + dx/2,
                                         'y': (-np.arange(ny)*2 + 2) + dy/2,
                                         'band': [1, 2, 3]})

            with xr.open_rasterio(tmp_file, cache=False) as actual:

                # tests
                # assert_allclose checks all data + coordinates
                assert_allclose(actual, expected)

                # Slicing
                ex = expected.isel(x=slice(2, 5), y=slice(5, 7))
                ac = actual.isel(x=slice(2, 5), y=slice(5, 7))
                assert_allclose(ac, ex)

                ex = expected.isel(band=slice(1, 2), x=slice(2, 5),
                                   y=slice(5, 7))
                ac = actual.isel(band=slice(1, 2), x=slice(2, 5),
                                 y=slice(5, 7))
                assert_allclose(ac, ex)

                # Selecting lists of bands is fine
                ex = expected.isel(band=[1, 2])
                ac = actual.isel(band=[1, 2])
                assert_allclose(ac, ex)
                ex = expected.isel(band=[0, 2])
                ac = actual.isel(band=[0, 2])
                assert_allclose(ac, ex)

                # but on x and y only windowed operations are allowed, more
                # exotic slicing should raise an error
                err_msg = 'not valid on rasterio'
                with raises_regex(IndexError, err_msg):
                    actual.isel(x=[2, 4], y=[1, 3]).values
                with raises_regex(IndexError, err_msg):
                    actual.isel(x=[4, 2]).values
                with raises_regex(IndexError, err_msg):
                    actual.isel(x=slice(5, 2, -1)).values
                # Integer indexing
                ex = expected.isel(band=1)
                ac = actual.isel(band=1)
                assert_allclose(ac, ex)

                ex = expected.isel(x=1, y=2)
                ac = actual.isel(x=1, y=2)
                assert_allclose(ac, ex)

                ex = expected.isel(band=0, x=1, y=2)
                ac = actual.isel(band=0, x=1, y=2)
                assert_allclose(ac, ex)

                # Mixed
                ex = actual.isel(x=slice(2), y=slice(2))
                ac = actual.isel(x=[0, 1], y=[0, 1])
                assert_allclose(ac, ex)

                ex = expected.isel(band=0, x=1, y=slice(5, 7))
                ac = actual.isel(band=0, x=1, y=slice(5, 7))
                assert_allclose(ac, ex)

                ex = expected.isel(band=0, x=slice(2, 5), y=2)
                ac = actual.isel(band=0, x=slice(2, 5), y=2)
                assert_allclose(ac, ex)

                # One-element lists
                ex = expected.isel(band=[0], x=slice(2, 5), y=[2])
                ac = actual.isel(band=[0], x=slice(2, 5), y=[2])
                assert_allclose(ac, ex)

    def test_caching(self):

        import rasterio
        from rasterio.transform import from_origin

        # Create a geotiff file in latlong proj
        with create_tmp_file(suffix='.tif') as tmp_file:
            # data
            nx, ny, nz = 8, 10, 3
            data = np.arange(nx*ny*nz,
                             dtype=rasterio.float32).reshape(nz, ny, nx)
            transform = from_origin(1, 2, 0.5, 2.)
            with rasterio.open(
                    tmp_file, 'w',
                    driver='GTiff', height=ny, width=nx, count=nz,
                    crs='+proj=latlong',
                    transform=transform,
                    dtype=rasterio.float32) as s:
                s.write(data)
                dx, dy = s.res[0], -s.res[1]

            # ref
            expected = DataArray(data, dims=('band', 'y', 'x'),
                                 coords={'x': (np.arange(nx)*0.5 + 1) + dx/2,
                                         'y': (-np.arange(ny)*2 + 2) + dy/2,
                                         'band': [1, 2, 3]})

            # Cache is the default
            with xr.open_rasterio(tmp_file) as actual:

                # Without cache an error is raised
                err_msg = 'not valid on rasterio'
                with raises_regex(IndexError, err_msg):
                    actual.isel(x=[2, 4]).values

                # This should cache everything
                assert_allclose(actual, expected)

                # once cached, non-windowed indexing should become possible
                ac = actual.isel(x=[2, 4])
                ex = expected.isel(x=[2, 4])
                assert_allclose(ac, ex)

    @requires_dask
    def test_chunks(self):

        import rasterio
        from rasterio.transform import from_origin

        # Create a geotiff file in latlong proj
        with create_tmp_file(suffix='.tif') as tmp_file:
            # data
            nx, ny, nz = 8, 10, 3
            data = np.arange(nx*ny*nz,
                             dtype=rasterio.float32).reshape(nz, ny, nx)
            transform = from_origin(1, 2, 0.5, 2.)
            with rasterio.open(
                    tmp_file, 'w',
                    driver='GTiff', height=ny, width=nx, count=nz,
                    crs='+proj=latlong',
                    transform=transform,
                    dtype=rasterio.float32) as s:
                s.write(data)
                dx, dy = s.res[0], -s.res[1]

            # Chunk at open time
            with xr.open_rasterio(tmp_file, chunks=(1, 2, 2)) as actual:

                import dask.array as da
                self.assertIsInstance(actual.data, da.Array)
                assert 'open_rasterio' in actual.data.name

                # ref
                expected = DataArray(data, dims=('band', 'y', 'x'),
                                     coords={'x': np.arange(nx)*0.5 + 1 + dx/2,
                                             'y': -np.arange(ny)*2 + 2 + dy/2,
                                             'band': [1, 2, 3]})

                # do some arithmetic
                ac = actual.mean()
                ex = expected.mean()
                assert_allclose(ac, ex)

                ac = actual.sel(band=1).mean(dim='x')
                ex = expected.sel(band=1).mean(dim='x')
                assert_allclose(ac, ex)


class TestEncodingInvalid(TestCase):

    def test_extract_nc4_variable_encoding(self):
        var = xr.Variable(('x',), [1, 2, 3], {}, {'foo': 'bar'})
        with raises_regex(ValueError, 'unexpected encoding'):
            _extract_nc4_variable_encoding(var, raise_on_invalid=True)

        var = xr.Variable(('x',), [1, 2, 3], {}, {'chunking': (2, 1)})
        encoding = _extract_nc4_variable_encoding(var)
        self.assertEqual({}, encoding)

        # regression test
        var = xr.Variable(('x',), [1, 2, 3], {}, {'shuffle': True})
        encoding = _extract_nc4_variable_encoding(var, raise_on_invalid=True)
        self.assertEqual({'shuffle': True}, encoding)

    def test_extract_h5nc_encoding(self):
        # not supported with h5netcdf (yet)
        var = xr.Variable(('x',), [1, 2, 3], {},
                          {'least_sigificant_digit': 2})
        with raises_regex(ValueError, 'unexpected encoding'):
            _extract_nc4_variable_encoding(var, raise_on_invalid=True)


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
            with raises_regex(TypeError, 'Invalid name for attr'):
                ds.to_netcdf('test.nc')

            ds, attrs = new_dataset_and_attrs()
            attrs[MiscObject()] = 'test'
            with raises_regex(TypeError, 'Invalid name for attr'):
                ds.to_netcdf('test.nc')

            ds, attrs = new_dataset_and_attrs()
            attrs[''] = 'test'
            with raises_regex(ValueError, 'Invalid name for attr'):
                ds.to_netcdf('test.nc')

            # This one should work
            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = 'test'
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = {'a': 5}
            with raises_regex(TypeError, 'Invalid value for attr'):
                ds.to_netcdf('test.nc')

            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = MiscObject()
            with raises_regex(TypeError, 'Invalid value for attr'):
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


@requires_scipy_or_netCDF4
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

    def test_open_dataarray_options(self):
        data = DataArray(
            np.arange(5), coords={'y': ('x', range(5))}, dims=['x'])

        with create_tmp_file() as tmp:
            data.to_netcdf(tmp)

            expected = data.drop('y')
            with open_dataarray(tmp, drop_variables=['y']) as loaded:
                self.assertDataArrayIdentical(expected, loaded)

    def test_dataarray_to_netcdf_return_bytes(self):
        # regression test for GH1410
        data = xr.DataArray([1, 2, 3])
        output = data.to_netcdf()
        assert isinstance(output, bytes)

    @requires_pathlib
    def test_dataarray_to_netcdf_no_name_pathlib(self):
        original_da = DataArray(np.arange(12).reshape((3, 4)))

        with create_tmp_file() as tmp:
            tmp = Path(tmp)
            original_da.to_netcdf(tmp)

            with open_dataarray(tmp) as loaded_da:
                self.assertDataArrayIdentical(original_da, loaded_da)
