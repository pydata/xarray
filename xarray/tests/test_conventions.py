# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest

from xarray import conventions, Variable, Dataset, open_dataset
from xarray.core import utils, indexing
from xarray.testing import assert_identical
from . import (
    TestCase, requires_netCDF4, requires_netcdftime, unittest, raises_regex,
    IndexerMaker, assert_array_equal)
from .test_backends import CFEncodedDataTest
from xarray.core.pycompat import iteritems
from xarray.backends.memory import InMemoryDataStore
from xarray.backends.common import WritableCFDataStore
from xarray.conventions import decode_cf


B = IndexerMaker(indexing.BasicIndexer)
V = IndexerMaker(indexing.VectorizedIndexer)


class TestStackedBytesArray(TestCase):
    def test_wrapper_class(self):
        array = np.array([[b'a', b'b', b'c'], [b'd', b'e', b'f']], dtype='S')
        actual = conventions.StackedBytesArray(array)
        expected = np.array([b'abc', b'def'], dtype='S')
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.size == expected.size
        assert actual.ndim == expected.ndim
        assert len(actual) == len(expected)
        assert_array_equal(expected, actual)
        assert_array_equal(expected[:1], actual[B[:1]])
        with pytest.raises(IndexError):
            actual[B[:, :2]]

    def test_scalar(self):
        array = np.array([b'a', b'b', b'c'], dtype='S')
        actual = conventions.StackedBytesArray(array)

        expected = np.array(b'abc')
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.size == expected.size
        assert actual.ndim == expected.ndim
        with pytest.raises(TypeError):
            len(actual)
        np.testing.assert_array_equal(expected, actual)
        with pytest.raises(IndexError):
            actual[B[:2]]
        assert str(actual) == str(expected)

    def test_char_to_bytes(self):
        array = np.array([['a', 'b', 'c'], ['d', 'e', 'f']])
        expected = np.array(['abc', 'def'])
        actual = conventions.char_to_bytes(array)
        assert_array_equal(actual, expected)

        expected = np.array(['ad', 'be', 'cf'])
        actual = conventions.char_to_bytes(array.T)  # non-contiguous
        assert_array_equal(actual, expected)

    def test_char_to_bytes_ndim_zero(self):
        expected = np.array('a')
        actual = conventions.char_to_bytes(expected)
        assert_array_equal(actual, expected)

    def test_char_to_bytes_size_zero(self):
        array = np.zeros((3, 0), dtype='S1')
        expected = np.array([b'', b'', b''])
        actual = conventions.char_to_bytes(array)
        assert_array_equal(actual, expected)

    def test_bytes_to_char(self):
        array = np.array([['ab', 'cd'], ['ef', 'gh']])
        expected = np.array([[['a', 'b'], ['c', 'd']],
                             [['e', 'f'], ['g', 'h']]])
        actual = conventions.bytes_to_char(array)
        assert_array_equal(actual, expected)

        expected = np.array([[['a', 'b'], ['e', 'f']],
                             [['c', 'd'], ['g', 'h']]])
        actual = conventions.bytes_to_char(array.T)
        assert_array_equal(actual, expected)

    def test_vectorized_indexing(self):
        array = np.array([[b'a', b'b', b'c'], [b'd', b'e', b'f']], dtype='S')
        stacked = conventions.StackedBytesArray(array)
        expected = np.array([[b'abc', b'def'], [b'def', b'abc']])
        indexer = V[np.array([[0, 1], [1, 0]])]
        actual = stacked[indexer]
        assert_array_equal(actual, expected)


class TestBytesToStringArray(TestCase):

    def test_encoding(self):
        encoding = 'utf-8'
        raw_array = np.array([b'abc', u'ß∂µ∆'.encode(encoding)])
        actual = conventions.BytesToStringArray(raw_array, encoding=encoding)
        expected = np.array([u'abc', u'ß∂µ∆'], dtype=object)

        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.size == expected.size
        assert actual.ndim == expected.ndim
        assert_array_equal(expected, actual)
        assert_array_equal(expected[0], actual[B[0]])

    def test_scalar(self):
        expected = np.array(u'abc', dtype=object)
        actual = conventions.BytesToStringArray(
            np.array(b'abc'), encoding='utf-8')
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.size == expected.size
        assert actual.ndim == expected.ndim
        with pytest.raises(TypeError):
            len(actual)
        np.testing.assert_array_equal(expected, actual)
        with pytest.raises(IndexError):
            actual[B[:2]]
        assert str(actual) == str(expected)

    def test_decode_bytes_array(self):
        encoding = 'utf-8'
        raw_array = np.array([b'abc', u'ß∂µ∆'.encode(encoding)])
        expected = np.array([u'abc', u'ß∂µ∆'], dtype=object)
        actual = conventions.decode_bytes_array(raw_array, encoding)
        np.testing.assert_array_equal(actual, expected)


class TestBoolTypeArray(TestCase):
    def test_booltype_array(self):
        x = np.array([1, 0, 1, 1, 0], dtype='i1')
        bx = conventions.BoolTypeArray(x)
        assert bx.dtype == np.bool
        assert_array_equal(bx, np.array([True, False, True, True, False],
                                        dtype=np.bool))


class TestNativeEndiannessArray(TestCase):
    def test(self):
        x = np.arange(5, dtype='>i8')
        expected = np.arange(5, dtype='int64')
        a = conventions.NativeEndiannessArray(x)
        assert a.dtype == expected.dtype
        assert a.dtype == expected[:].dtype
        assert_array_equal(a, expected)


def test_decode_cf_with_conflicting_fill_missing_value():
    var = Variable(['t'], np.arange(10),
                   {'units': 'foobar',
                    'missing_value': 0,
                    '_FillValue': 1})
    with raises_regex(ValueError, "_FillValue and missing_value"):
        conventions.decode_cf_variable('t', var)

    expected = Variable(['t'], np.arange(10), {'units': 'foobar'})

    var = Variable(['t'], np.arange(10),
                   {'units': 'foobar',
                    'missing_value': np.nan,
                    '_FillValue': np.nan})
    actual = conventions.decode_cf_variable('t', var)
    assert_identical(actual, expected)

    var = Variable(['t'], np.arange(10),
                   {'units': 'foobar',
                    'missing_value': np.float32(np.nan),
                    '_FillValue': np.float32(np.nan)})
    actual = conventions.decode_cf_variable('t', var)
    assert_identical(actual, expected)


@requires_netcdftime
class TestEncodeCFVariable(TestCase):
    def test_incompatible_attributes(self):
        invalid_vars = [
            Variable(['t'], pd.date_range('2000-01-01', periods=3),
                     {'units': 'foobar'}),
            Variable(['t'], pd.to_timedelta(['1 day']), {'units': 'foobar'}),
            Variable(['t'], [0, 1, 2], {'add_offset': 0}, {'add_offset': 2}),
            Variable(['t'], [0, 1, 2], {'_FillValue': 0}, {'_FillValue': 2}),
        ]
        for var in invalid_vars:
            with pytest.raises(ValueError):
                conventions.encode_cf_variable(var)

    def test_missing_fillvalue(self):
        v = Variable(['x'], np.array([np.nan, 1, 2, 3]))
        v.encoding = {'dtype': 'int16'}
        with pytest.warns(Warning, match='floating point data as an integer'):
            conventions.encode_cf_variable(v)

    def test_multidimensional_coordinates(self):
        # regression test for GH1763
        # Set up test case with coordinates that have overlapping (but not
        # identical) dimensions.
        zeros1 = np.zeros((1, 5, 3))
        zeros2 = np.zeros((1, 6, 3))
        zeros3 = np.zeros((1, 5, 4))
        orig = Dataset({
            'lon1': (['x1', 'y1'], zeros1.squeeze(0), {}),
            'lon2': (['x2', 'y1'], zeros2.squeeze(0), {}),
            'lon3': (['x1', 'y2'], zeros3.squeeze(0), {}),
            'lat1': (['x1', 'y1'], zeros1.squeeze(0), {}),
            'lat2': (['x2', 'y1'], zeros2.squeeze(0), {}),
            'lat3': (['x1', 'y2'], zeros3.squeeze(0), {}),
            'foo1': (['time', 'x1', 'y1'], zeros1,
                     {'coordinates': 'lon1 lat1'}),
            'foo2': (['time', 'x2', 'y1'], zeros2,
                     {'coordinates': 'lon2 lat2'}),
            'foo3': (['time', 'x1', 'y2'], zeros3,
                     {'coordinates': 'lon3 lat3'}),
            'time': ('time', [0.], {'units': 'hours since 2017-01-01'}),
        })
        orig = conventions.decode_cf(orig)
        # Encode the coordinates, as they would be in a netCDF output file.
        enc, attrs = conventions.encode_dataset_coordinates(orig)
        # Make sure we have the right coordinates for each variable.
        foo1_coords = enc['foo1'].attrs.get('coordinates', '')
        foo2_coords = enc['foo2'].attrs.get('coordinates', '')
        foo3_coords = enc['foo3'].attrs.get('coordinates', '')
        assert set(foo1_coords.split()) == set(['lat1', 'lon1'])
        assert set(foo2_coords.split()) == set(['lat2', 'lon2'])
        assert set(foo3_coords.split()) == set(['lat3', 'lon3'])
        # Should not have any global coordinates.
        assert 'coordinates' not in attrs


@requires_netcdftime
class TestDecodeCF(TestCase):
    def test_dataset(self):
        original = Dataset({
            't': ('t', [0, 1, 2], {'units': 'days since 2000-01-01'}),
            'foo': ('t', [0, 0, 0], {'coordinates': 'y', 'units': 'bar'}),
            'y': ('t', [5, 10, -999], {'_FillValue': -999})
        })
        expected = Dataset({'foo': ('t', [0, 0, 0], {'units': 'bar'})},
                           {'t': pd.date_range('2000-01-01', periods=3),
                            'y': ('t', [5.0, 10.0, np.nan])})
        actual = conventions.decode_cf(original)
        assert_identical(expected, actual)

    def test_invalid_coordinates(self):
        # regression test for GH308
        original = Dataset({'foo': ('t', [1, 2], {'coordinates': 'invalid'})})
        actual = conventions.decode_cf(original)
        assert_identical(original, actual)

    def test_decode_coordinates(self):
        # regression test for GH610
        original = Dataset({'foo': ('t', [1, 2], {'coordinates': 'x'}),
                            'x': ('t', [4, 5])})
        actual = conventions.decode_cf(original)
        assert actual.foo.encoding['coordinates'] == 'x'

    def test_0d_int32_encoding(self):
        original = Variable((), np.int32(0), encoding={'dtype': 'int64'})
        expected = Variable((), np.int64(0))
        actual = conventions.maybe_encode_nonstring_dtype(original)
        assert_identical(expected, actual)

    def test_decode_cf_with_multiple_missing_values(self):
        original = Variable(['t'], [0, 1, 2],
                            {'missing_value': np.array([0, 1])})
        expected = Variable(['t'], [np.nan, np.nan, 2], {})
        with warnings.catch_warnings(record=True) as w:
            actual = conventions.decode_cf_variable('t', original)
            assert_identical(expected, actual)
            assert 'has multiple fill' in str(w[0].message)

    def test_decode_cf_with_drop_variables(self):
        original = Dataset({
            't': ('t', [0, 1, 2], {'units': 'days since 2000-01-01'}),
            'x': ("x", [9, 8, 7], {'units': 'km'}),
            'foo': (('t', 'x'), [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                    {'units': 'bar'}),
            'y': ('t', [5, 10, -999], {'_FillValue': -999})
        })
        expected = Dataset({
            't': pd.date_range('2000-01-01', periods=3),
            'foo': (('t', 'x'), [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                    {'units': 'bar'}),
            'y': ('t', [5, 10, np.nan])
        })
        actual = conventions.decode_cf(original, drop_variables=("x",))
        actual2 = conventions.decode_cf(original, drop_variables="x")
        assert_identical(expected, actual)
        assert_identical(expected, actual2)

    def test_invalid_time_units_raises_eagerly(self):
        ds = Dataset({'time': ('time', [0, 1], {'units': 'foobar since 123'})})
        with raises_regex(ValueError, 'unable to decode time'):
            decode_cf(ds)

    @requires_netcdftime
    def test_dataset_repr_with_netcdf4_datetimes(self):
        # regression test for #347
        attrs = {'units': 'days since 0001-01-01', 'calendar': 'noleap'}
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'unable to decode time')
            ds = decode_cf(Dataset({'time': ('time', [0, 1], attrs)}))
            assert '(time) object' in repr(ds)

        attrs = {'units': 'days since 1900-01-01'}
        ds = decode_cf(Dataset({'time': ('time', [0, 1], attrs)}))
        assert '(time) datetime64[ns]' in repr(ds)

    @requires_netcdftime
    def test_decode_cf_datetime_transition_to_invalid(self):
        # manually create dataset with not-decoded date
        from datetime import datetime
        ds = Dataset(coords={'time': [0, 266 * 365]})
        units = 'days since 2000-01-01 00:00:00'
        ds.time.attrs = dict(units=units)
        ds_decoded = conventions.decode_cf(ds)

        expected = [datetime(2000, 1, 1, 0, 0),
                    datetime(2265, 10, 28, 0, 0)]

        assert_array_equal(ds_decoded.time.values, expected)


class CFEncodedInMemoryStore(WritableCFDataStore, InMemoryDataStore):
    pass


class NullWrapper(utils.NDArrayMixin):
    """
    Just for testing, this lets us create a numpy array directly
    but make it look like its not in memory yet.
    """

    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        return self.array[indexing.orthogonal_indexer(key, self.shape)]


def null_wrap(ds):
    """
    Given a data store this wraps each variable in a NullWrapper so that
    it appears to be out of memory.
    """
    variables = dict((k, Variable(v.dims, NullWrapper(v.values), v.attrs))
                     for k, v in iteritems(ds))
    return InMemoryDataStore(variables=variables, attributes=ds.attrs)


@requires_netCDF4
class TestCFEncodedDataStore(CFEncodedDataTest, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        yield CFEncodedInMemoryStore()

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        store = CFEncodedInMemoryStore()
        data.dump_to_store(store, **save_kwargs)
        yield open_dataset(store, **open_kwargs)

    def test_roundtrip_coordinates(self):
        raise unittest.SkipTest('cannot roundtrip coordinates yet for '
                                'CFEncodedInMemoryStore')

    def test_invalid_dataarray_names_raise(self):
        pass

    def test_encoding_kwarg(self):
        pass
