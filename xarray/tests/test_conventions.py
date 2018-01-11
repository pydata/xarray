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
from . import TestCase, requires_netCDF4, unittest, raises_regex, IndexerMaker
from .test_backends import CFEncodedDataTest
from xarray.core.pycompat import iteritems
from xarray.backends.memory import InMemoryDataStore
from xarray.backends.common import WritableCFDataStore
from xarray.conventions import decode_cf


B = IndexerMaker(indexing.BasicIndexer)
O = IndexerMaker(indexing.OuterIndexer)
V = IndexerMaker(indexing.VectorizedIndexer)


class TestStackedBytesArray(TestCase):
    def test_wrapper_class(self):
        array = np.array([[b'a', b'b', b'c'], [b'd', b'e', b'f']], dtype='S')
        actual = conventions.StackedBytesArray(array)
        expected = np.array([b'abc', b'def'], dtype='S')
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.size, expected.size)
        self.assertEqual(actual.ndim, expected.ndim)
        self.assertEqual(len(actual), len(expected))
        self.assertArrayEqual(expected, actual)
        self.assertArrayEqual(expected[:1], actual[B[:1]])
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
        self.assertArrayEqual(actual, expected)

        expected = np.array(['ad', 'be', 'cf'])
        actual = conventions.char_to_bytes(array.T)  # non-contiguous
        self.assertArrayEqual(actual, expected)

    def test_char_to_bytes_ndim_zero(self):
        expected = np.array('a')
        actual = conventions.char_to_bytes(expected)
        self.assertArrayEqual(actual, expected)

    def test_char_to_bytes_size_zero(self):
        array = np.zeros((3, 0), dtype='S1')
        expected = np.array([b'', b'', b''])
        actual = conventions.char_to_bytes(array)
        self.assertArrayEqual(actual, expected)

    def test_bytes_to_char(self):
        array = np.array([['ab', 'cd'], ['ef', 'gh']])
        expected = np.array([[['a', 'b'], ['c', 'd']],
                             [['e', 'f'], ['g', 'h']]])
        actual = conventions.bytes_to_char(array)
        self.assertArrayEqual(actual, expected)

        expected = np.array([[['a', 'b'], ['e', 'f']],
                             [['c', 'd'], ['g', 'h']]])
        actual = conventions.bytes_to_char(array.T)
        self.assertArrayEqual(actual, expected)

    def test_vectorized_indexing(self):
        array = np.array([[b'a', b'b', b'c'], [b'd', b'e', b'f']], dtype='S')
        stacked = conventions.StackedBytesArray(array)
        expected = np.array([[b'abc', b'def'], [b'def', b'abc']])
        indexer = V[np.array([[0, 1], [1, 0]])]
        actual = stacked[indexer]
        self.assertArrayEqual(actual, expected)


class TestBytesToStringArray(TestCase):

    def test_encoding(self):
        encoding = 'utf-8'
        raw_array = np.array([b'abc', u'ß∂µ∆'.encode(encoding)])
        actual = conventions.BytesToStringArray(raw_array, encoding=encoding)
        expected = np.array([u'abc', u'ß∂µ∆'], dtype=object)

        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.size, expected.size)
        self.assertEqual(actual.ndim, expected.ndim)
        self.assertArrayEqual(expected, actual)
        self.assertArrayEqual(expected[0], actual[B[0]])

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
        self.assertEqual(bx.dtype, np.bool)
        self.assertArrayEqual(bx, np.array([True, False, True, True, False],
                                           dtype=np.bool))


class TestNativeEndiannessArray(TestCase):
    def test(self):
        x = np.arange(5, dtype='>i8')
        expected = np.arange(5, dtype='int64')
        a = conventions.NativeEndiannessArray(x)
        assert a.dtype == expected.dtype
        assert a.dtype == expected[:].dtype
        self.assertArrayEqual(a, expected)


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


@requires_netCDF4
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
        with self.assertWarns('floating point data as an integer'):
            conventions.encode_cf_variable(v)


@requires_netCDF4
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
        self.assertDatasetIdentical(expected, actual)

    def test_invalid_coordinates(self):
        # regression test for GH308
        original = Dataset({'foo': ('t', [1, 2], {'coordinates': 'invalid'})})
        actual = conventions.decode_cf(original)
        self.assertDatasetIdentical(original, actual)

    def test_decode_coordinates(self):
        # regression test for GH610
        original = Dataset({'foo': ('t', [1, 2], {'coordinates': 'x'}),
                            'x': ('t', [4, 5])})
        actual = conventions.decode_cf(original)
        self.assertEqual(actual.foo.encoding['coordinates'], 'x')

    def test_0d_int32_encoding(self):
        original = Variable((), np.int32(0), encoding={'dtype': 'int64'})
        expected = Variable((), np.int64(0))
        actual = conventions.maybe_encode_nonstring_dtype(original)
        self.assertDatasetIdentical(expected, actual)

    def test_decode_cf_with_multiple_missing_values(self):
        original = Variable(['t'], [0, 1, 2],
                            {'missing_value': np.array([0, 1])})
        expected = Variable(['t'], [np.nan, np.nan, 2], {})
        with warnings.catch_warnings(record=True) as w:
            actual = conventions.decode_cf_variable('t', original)
            self.assertDatasetIdentical(expected, actual)
            self.assertIn('has multiple fill', str(w[0].message))

    def test_decode_cf_with_drop_variables(self):
        original = Dataset({
            't': ('t', [0, 1, 2], {'units': 'days since 2000-01-01'}),
            'x': ("x", [9, 8, 7], {'units': 'km'}),
            'foo': (('t', 'x'), [[0, 0, 0], [1, 1, 1], [2, 2, 2]], {'units': 'bar'}),
            'y': ('t', [5, 10, -999], {'_FillValue': -999})
        })
        expected = Dataset({
            't': pd.date_range('2000-01-01', periods=3),
            'foo': (('t', 'x'), [[0, 0, 0], [1, 1, 1], [2, 2, 2]], {'units': 'bar'}),
            'y': ('t', [5, 10, np.nan])
        })
        actual = conventions.decode_cf(original, drop_variables=("x",))
        actual2 = conventions.decode_cf(original, drop_variables="x")
        self.assertDatasetIdentical(expected, actual)
        self.assertDatasetIdentical(expected, actual2)

    def test_invalid_time_units_raises_eagerly(self):
        ds = Dataset({'time': ('time', [0, 1], {'units': 'foobar since 123'})})
        with raises_regex(ValueError, 'unable to decode time'):
            decode_cf(ds)

    @requires_netCDF4
    def test_dataset_repr_with_netcdf4_datetimes(self):
        # regression test for #347
        attrs = {'units': 'days since 0001-01-01', 'calendar': 'noleap'}
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'unable to decode time')
            ds = decode_cf(Dataset({'time': ('time', [0, 1], attrs)}))
            self.assertIn('(time) object', repr(ds))

        attrs = {'units': 'days since 1900-01-01'}
        ds = decode_cf(Dataset({'time': ('time', [0, 1], attrs)}))
        self.assertIn('(time) datetime64[ns]', repr(ds))

    @requires_netCDF4
    def test_decode_cf_datetime_transition_to_invalid(self):
        # manually create dataset with not-decoded date
        from datetime import datetime
        ds = Dataset(coords={'time': [0, 266 * 365]})
        units = 'days since 2000-01-01 00:00:00'
        ds.time.attrs = dict(units=units)
        ds_decoded = conventions.decode_cf(ds)

        expected = [datetime(2000, 1, 1, 0, 0),
                    datetime(2265, 10, 28, 0, 0)]

        self.assertArrayEqual(ds_decoded.time.values, expected)


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
