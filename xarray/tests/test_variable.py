from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
import pytest

from distutils.version import LooseVersion
import numpy as np
import pytz
import pandas as pd

from xarray import Variable, IndexVariable, Coordinate, Dataset
from xarray.core import indexing
from xarray.core.variable import as_variable, as_compatible_data
from xarray.core.indexing import (PandasIndexAdapter, LazilyIndexedArray,
                                  BasicIndexer, OuterIndexer,
                                  VectorizedIndexer, NumpyIndexingAdapter,
                                  CopyOnWriteArray, MemoryCachedArray,
                                  DaskIndexingAdapter)
from xarray.core.pycompat import PY3, OrderedDict
from xarray.core.common import full_like, zeros_like, ones_like
from xarray.core.utils import NDArrayMixin

from . import TestCase, source_ndarray, requires_dask, raises_regex


class VariableSubclassTestCases(object):
    def test_properties(self):
        data = 0.5 * np.arange(10)
        v = self.cls(['time'], data, {'foo': 'bar'})
        self.assertEqual(v.dims, ('time',))
        self.assertArrayEqual(v.values, data)
        self.assertEqual(v.dtype, float)
        self.assertEqual(v.shape, (10,))
        self.assertEqual(v.size, 10)
        self.assertEqual(v.sizes, {'time': 10})
        self.assertEqual(v.nbytes, 80)
        self.assertEqual(v.ndim, 1)
        self.assertEqual(len(v), 10)
        self.assertEqual(v.attrs, {'foo': u'bar'})

    def test_attrs(self):
        v = self.cls(['time'], 0.5 * np.arange(10))
        self.assertEqual(v.attrs, {})
        attrs = {'foo': 'bar'}
        v.attrs = attrs
        self.assertEqual(v.attrs, attrs)
        self.assertIsInstance(v.attrs, OrderedDict)
        v.attrs['foo'] = 'baz'
        self.assertEqual(v.attrs['foo'], 'baz')

    def test_getitem_dict(self):
        v = self.cls(['x'], np.random.randn(5))
        actual = v[{'x': 0}]
        expected = v[0]
        self.assertVariableIdentical(expected, actual)

    def test_getitem_1d(self):
        data = np.array([0, 1, 2])
        v = self.cls(['x'], data)

        v_new = v[dict(x=[0, 1])]
        assert v_new.dims == ('x', )
        self.assertArrayEqual(v_new, data[[0, 1]])

        v_new = v[dict(x=slice(None))]
        assert v_new.dims == ('x', )
        self.assertArrayEqual(v_new, data)

        v_new = v[dict(x=Variable('a', [0, 1]))]
        assert v_new.dims == ('a', )
        self.assertArrayEqual(v_new, data[[0, 1]])

        v_new = v[dict(x=1)]
        assert v_new.dims == ()
        self.assertArrayEqual(v_new, data[1])

        # tuple argument
        v_new = v[slice(None)]
        assert v_new.dims == ('x', )
        self.assertArrayEqual(v_new, data)

    def test_getitem_1d_fancy(self):
        v = self.cls(['x'], [0, 1, 2])
        # 1d-variable should be indexable by multi-dimensional Variable
        ind = Variable(('a', 'b'), [[0, 1], [0, 1]])
        v_new = v[ind]
        assert v_new.dims == ('a', 'b')
        expected = np.array(v._data)[([0, 1], [0, 1]), ]
        self.assertArrayEqual(v_new, expected)

        # boolean indexing
        ind = Variable(('x', ), [True, False, True])
        v_new = v[ind]
        self.assertVariableIdentical(v[[0, 2]], v_new)
        v_new = v[[True, False, True]]
        self.assertVariableIdentical(v[[0, 2]], v_new)

        with raises_regex(IndexError, "Boolean indexer should"):
            ind = Variable(('a', ), [True, False, True])
            v[ind]

    def _assertIndexedLikeNDArray(self, variable, expected_value0,
                                  expected_dtype=None):
        """Given a 1-dimensional variable, verify that the variable is indexed
        like a numpy.ndarray.
        """
        self.assertEqual(variable[0].shape, ())
        self.assertEqual(variable[0].ndim, 0)
        self.assertEqual(variable[0].size, 1)
        # test identity
        self.assertTrue(variable.equals(variable.copy()))
        self.assertTrue(variable.identical(variable.copy()))
        # check value is equal for both ndarray and Variable
        self.assertEqual(variable.values[0], expected_value0)
        self.assertEqual(variable[0].values, expected_value0)
        # check type or dtype is consistent for both ndarray and Variable
        if expected_dtype is None:
            # check output type instead of array dtype
            self.assertEqual(type(variable.values[0]), type(expected_value0))
            self.assertEqual(type(variable[0].values), type(expected_value0))
        elif expected_dtype is not False:
            self.assertEqual(variable.values[0].dtype, expected_dtype)
            self.assertEqual(variable[0].values.dtype, expected_dtype)

    def test_index_0d_int(self):
        for value, dtype in [(0, np.int_),
                             (np.int32(0), np.int32)]:
            x = self.cls(['x'], [value])
            self._assertIndexedLikeNDArray(x, value, dtype)

    def test_index_0d_float(self):
        for value, dtype in [(0.5, np.float_),
                             (np.float32(0.5), np.float32)]:
            x = self.cls(['x'], [value])
            self._assertIndexedLikeNDArray(x, value, dtype)

    def test_index_0d_string(self):
        for value, dtype in [('foo', np.dtype('U3' if PY3 else 'S3')),
                             (u'foo', np.dtype('U3'))]:
            x = self.cls(['x'], [value])
            self._assertIndexedLikeNDArray(x, value, dtype)

    def test_index_0d_datetime(self):
        d = datetime(2000, 1, 1)
        x = self.cls(['x'], [d])
        self._assertIndexedLikeNDArray(x, np.datetime64(d))

        x = self.cls(['x'], [np.datetime64(d)])
        self._assertIndexedLikeNDArray(x, np.datetime64(d), 'datetime64[ns]')

        x = self.cls(['x'], pd.DatetimeIndex([d]))
        self._assertIndexedLikeNDArray(x, np.datetime64(d), 'datetime64[ns]')

    def test_index_0d_timedelta64(self):
        td = timedelta(hours=1)

        x = self.cls(['x'], [np.timedelta64(td)])
        self._assertIndexedLikeNDArray(x, np.timedelta64(td), 'timedelta64[ns]')

        x = self.cls(['x'], pd.to_timedelta([td]))
        self._assertIndexedLikeNDArray(x, np.timedelta64(td), 'timedelta64[ns]')

    def test_index_0d_not_a_time(self):
        d = np.datetime64('NaT', 'ns')
        x = self.cls(['x'], [d])
        self._assertIndexedLikeNDArray(x, d)

    def test_index_0d_object(self):

        class HashableItemWrapper(object):
            def __init__(self, item):
                self.item = item

            def __eq__(self, other):
                return self.item == other.item

            def __hash__(self):
                return hash(self.item)

            def __repr__(self):
                return '%s(item=%r)' % (type(self).__name__, self.item)

        item = HashableItemWrapper((1, 2, 3))
        x = self.cls('x', [item])
        self._assertIndexedLikeNDArray(x, item, expected_dtype=False)

    def test_0d_object_array_with_list(self):
        listarray = np.empty((1,), dtype=object)
        listarray[0] = [1, 2, 3]
        x = self.cls('x', listarray)
        self.assertArrayEqual(x.data, listarray)
        self.assertArrayEqual(x[0].data, listarray.squeeze())
        self.assertArrayEqual(x.squeeze().data, listarray.squeeze())

    def test_index_and_concat_datetime(self):
        # regression test for #125
        date_range = pd.date_range('2011-09-01', periods=10)
        for dates in [date_range, date_range.values,
                      date_range.to_pydatetime()]:
            expected = self.cls('t', dates)
            for times in [[expected[i] for i in range(10)],
                          [expected[i:(i + 1)] for i in range(10)],
                          [expected[[i]] for i in range(10)]]:
                actual = Variable.concat(times, 't')
                self.assertEqual(expected.dtype, actual.dtype)
                self.assertArrayEqual(expected, actual)

    def test_0d_time_data(self):
        # regression test for #105
        x = self.cls('time', pd.date_range('2000-01-01', periods=5))
        expected = np.datetime64('2000-01-01', 'ns')
        self.assertEqual(x[0].values, expected)

    def test_datetime64_conversion(self):
        times = pd.date_range('2000-01-01', periods=3)
        for values, preserve_source in [
                (times, True),
                (times.values, True),
                (times.values.astype('datetime64[s]'), False),
                (times.to_pydatetime(), False),
               ]:
            v = self.cls(['t'], values)
            self.assertEqual(v.dtype, np.dtype('datetime64[ns]'))
            self.assertArrayEqual(v.values, times.values)
            self.assertEqual(v.values.dtype, np.dtype('datetime64[ns]'))
            same_source = source_ndarray(v.values) is source_ndarray(values)
            assert preserve_source == same_source

    def test_timedelta64_conversion(self):
        times = pd.timedelta_range(start=0, periods=3)
        for values, preserve_source in [
                (times, True),
                (times.values, True),
                (times.values.astype('timedelta64[s]'), False),
                (times.to_pytimedelta(), False),
               ]:
            v = self.cls(['t'], values)
            self.assertEqual(v.dtype, np.dtype('timedelta64[ns]'))
            self.assertArrayEqual(v.values, times.values)
            self.assertEqual(v.values.dtype, np.dtype('timedelta64[ns]'))
            same_source = source_ndarray(v.values) is source_ndarray(values)
            assert preserve_source == same_source

    def test_object_conversion(self):
        data = np.arange(5).astype(str).astype(object)
        actual = self.cls('x', data)
        self.assertEqual(actual.dtype, data.dtype)

    def test_pandas_data(self):
        v = self.cls(['x'], pd.Series([0, 1, 2], index=[3, 2, 1]))
        self.assertVariableIdentical(v, v[[0, 1, 2]])
        v = self.cls(['x'], pd.Index([0, 1, 2]))
        self.assertEqual(v[0].values, v.values[0])

    def test_pandas_period_index(self):
        v = self.cls(['x'], pd.period_range(start='2000', periods=20,
                                            freq='B'))
        v = v.load()  # for dask-based Variable
        self.assertEqual(v[0], pd.Period('2000', freq='B'))
        assert "Period('2000-01-03', 'B')" in repr(v)

    def test_1d_math(self):
        x = 1.0 * np.arange(5)
        y = np.ones(5)

        # should we need `.to_base_variable()`?
        # probably a break that `+v` changes type?
        v = self.cls(['x'], x)
        base_v = v.to_base_variable()
        # unary ops
        self.assertVariableIdentical(base_v, +v)
        self.assertVariableIdentical(base_v, abs(v))
        self.assertArrayEqual((-v).values, -x)
        # binary ops with numbers
        self.assertVariableIdentical(base_v, v + 0)
        self.assertVariableIdentical(base_v, 0 + v)
        self.assertVariableIdentical(base_v, v * 1)
        self.assertArrayEqual((v > 2).values, x > 2)
        self.assertArrayEqual((0 == v).values, 0 == x)
        self.assertArrayEqual((v - 1).values, x - 1)
        self.assertArrayEqual((1 - v).values, 1 - x)
        # binary ops with numpy arrays
        self.assertArrayEqual((v * x).values, x ** 2)
        self.assertArrayEqual((x * v).values, x ** 2)
        self.assertArrayEqual(v - y, v - 1)
        self.assertArrayEqual(y - v, 1 - v)
        # verify attributes are dropped
        v2 = self.cls(['x'], x, {'units': 'meters'})
        self.assertVariableIdentical(base_v, +v2)
        # binary ops with all variables
        self.assertArrayEqual(v + v, 2 * v)
        w = self.cls(['x'], y, {'foo': 'bar'})
        self.assertVariableIdentical(v + w, self.cls(['x'], x + y).to_base_variable())
        self.assertArrayEqual((v * w).values, x * y)
        # something complicated
        self.assertArrayEqual((v ** 2 * w - 1 + x).values, x ** 2 * y - 1 + x)
        # make sure dtype is preserved (for Index objects)
        self.assertEqual(float, (+v).dtype)
        self.assertEqual(float, (+v).values.dtype)
        self.assertEqual(float, (0 + v).dtype)
        self.assertEqual(float, (0 + v).values.dtype)
        # check types of returned data
        self.assertIsInstance(+v, Variable)
        self.assertNotIsInstance(+v, IndexVariable)
        self.assertIsInstance(0 + v, Variable)
        self.assertNotIsInstance(0 + v, IndexVariable)

    def test_1d_reduce(self):
        x = np.arange(5)
        v = self.cls(['x'], x)
        actual = v.sum()
        expected = Variable((), 10)
        self.assertVariableIdentical(expected, actual)
        self.assertIs(type(actual), Variable)

    def test_array_interface(self):
        x = np.arange(5)
        v = self.cls(['x'], x)
        self.assertArrayEqual(np.asarray(v), x)
        # test patched in methods
        self.assertArrayEqual(v.astype(float), x.astype(float))
        # think this is a break, that argsort changes the type
        self.assertVariableIdentical(v.argsort(), v.to_base_variable())
        self.assertVariableIdentical(v.clip(2, 3),
                                     self.cls('x', x.clip(2, 3)).to_base_variable())
        # test ufuncs
        self.assertVariableIdentical(np.sin(v), self.cls(['x'], np.sin(x)).to_base_variable())
        self.assertIsInstance(np.sin(v), Variable)
        self.assertNotIsInstance(np.sin(v), IndexVariable)

    def example_1d_objects(self):
        for data in [range(3),
                     0.5 * np.arange(3),
                     0.5 * np.arange(3, dtype=np.float32),
                     pd.date_range('2000-01-01', periods=3),
                     np.array(['a', 'b', 'c'], dtype=object)]:
            yield (self.cls('x', data), data)

    def test___array__(self):
        for v, data in self.example_1d_objects():
            self.assertArrayEqual(v.values, np.asarray(data))
            self.assertArrayEqual(np.asarray(v), np.asarray(data))
            self.assertEqual(v[0].values, np.asarray(data)[0])
            self.assertEqual(np.asarray(v[0]), np.asarray(data)[0])

    def test_equals_all_dtypes(self):
        for v, _ in self.example_1d_objects():
            v2 = v.copy()
            self.assertTrue(v.equals(v2))
            self.assertTrue(v.identical(v2))
            self.assertTrue(v.no_conflicts(v2))
            self.assertTrue(v[0].equals(v2[0]))
            self.assertTrue(v[0].identical(v2[0]))
            self.assertTrue(v[0].no_conflicts(v2[0]))
            self.assertTrue(v[:2].equals(v2[:2]))
            self.assertTrue(v[:2].identical(v2[:2]))
            self.assertTrue(v[:2].no_conflicts(v2[:2]))

    def test_eq_all_dtypes(self):
        # ensure that we don't choke on comparisons for which numpy returns
        # scalars
        expected = Variable('x', 3 * [False])
        for v, _ in self.example_1d_objects():
            actual = 'z' == v
            self.assertVariableIdentical(expected, actual)
            actual = ~('z' != v)
            self.assertVariableIdentical(expected, actual)

    def test_encoding_preserved(self):
        expected = self.cls('x', range(3), {'foo': 1}, {'bar': 2})
        for actual in [expected.T,
                       expected[...],
                       expected.squeeze(),
                       expected.isel(x=slice(None)),
                       expected.set_dims({'x': 3}),
                       expected.copy(deep=True),
                       expected.copy(deep=False)]:

            self.assertVariableIdentical(expected.to_base_variable(),
                                         actual.to_base_variable())
            self.assertEqual(expected.encoding, actual.encoding)

    def test_concat(self):
        x = np.arange(5)
        y = np.arange(5, 10)
        v = self.cls(['a'], x)
        w = self.cls(['a'], y)
        self.assertVariableIdentical(Variable(['b', 'a'], np.array([x, y])),
                                     Variable.concat([v, w], 'b'))
        self.assertVariableIdentical(Variable(['b', 'a'], np.array([x, y])),
                                     Variable.concat((v, w), 'b'))
        self.assertVariableIdentical(Variable(['b', 'a'], np.array([x, y])),
                                     Variable.concat((v, w), 'b'))
        with raises_regex(ValueError, 'inconsistent dimensions'):
            Variable.concat([v, Variable(['c'], y)], 'b')
        # test indexers
        actual = Variable.concat(
            [v, w],
            positions=[np.arange(0, 10, 2), np.arange(1, 10, 2)],
            dim='a')
        expected = Variable('a', np.array([x, y]).ravel(order='F'))
        self.assertVariableIdentical(expected, actual)
        # test concatenating along a dimension
        v = Variable(['time', 'x'], np.random.random((10, 8)))
        self.assertVariableIdentical(v, Variable.concat([v[:5], v[5:]], 'time'))
        self.assertVariableIdentical(v, Variable.concat([v[:5], v[5:6], v[6:]], 'time'))
        self.assertVariableIdentical(v, Variable.concat([v[:1], v[1:]], 'time'))
        # test dimension order
        self.assertVariableIdentical(v, Variable.concat([v[:, :5], v[:, 5:]], 'x'))
        with raises_regex(ValueError, 'all input arrays must have'):
            Variable.concat([v[:, 0], v[:, 1:]], 'x')

    def test_concat_attrs(self):
        # different or conflicting attributes should be removed
        v = self.cls('a', np.arange(5), {'foo': 'bar'})
        w = self.cls('a', np.ones(5))
        expected = self.cls('a', np.concatenate([np.arange(5), np.ones(5)])).to_base_variable()
        self.assertVariableIdentical(expected, Variable.concat([v, w], 'a'))
        w.attrs['foo'] = 2
        self.assertVariableIdentical(expected, Variable.concat([v, w], 'a'))
        w.attrs['foo'] = 'bar'
        expected.attrs['foo'] = 'bar'
        self.assertVariableIdentical(expected, Variable.concat([v, w], 'a'))

    def test_concat_fixed_len_str(self):
        # regression test for #217
        for kind in ['S', 'U']:
            x = self.cls('animal', np.array(['horse'], dtype=kind))
            y = self.cls('animal', np.array(['aardvark'], dtype=kind))
            actual = Variable.concat([x, y], 'animal')
            expected = Variable(
                'animal', np.array(['horse', 'aardvark'], dtype=kind))
            self.assertVariableEqual(expected, actual)

    def test_concat_number_strings(self):
        # regression test for #305
        a = self.cls('x', ['0', '1', '2'])
        b = self.cls('x', ['3', '4'])
        actual = Variable.concat([a, b], dim='x')
        expected = Variable('x', np.arange(5).astype(str).astype(object))
        self.assertVariableIdentical(expected, actual)
        self.assertEqual(expected.dtype, object)
        self.assertEqual(type(expected.values[0]), str)

    def test_copy(self):
        v = self.cls('x', 0.5 * np.arange(10), {'foo': 'bar'})
        for deep in [True, False]:
            w = v.copy(deep=deep)
            self.assertIs(type(v), type(w))
            self.assertVariableIdentical(v, w)
            self.assertEqual(v.dtype, w.dtype)
            if self.cls is Variable:
                if deep:
                    self.assertIsNot(source_ndarray(v.values),
                                     source_ndarray(w.values))
                else:
                    self.assertIs(source_ndarray(v.values),
                                  source_ndarray(w.values))
        self.assertVariableIdentical(v, copy(v))

    def test_copy_index(self):
        midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2], [-1, -2]],
                                          names=('one', 'two', 'three'))
        v = self.cls('x', midx)
        for deep in [True, False]:
            w = v.copy(deep=deep)
            self.assertIsInstance(w._data, PandasIndexAdapter)
            self.assertIsInstance(w.to_index(), pd.MultiIndex)
            self.assertArrayEqual(v._data.array, w._data.array)

    def test_real_and_imag(self):
        v = self.cls('x', np.arange(3) - 1j * np.arange(3), {'foo': 'bar'})
        expected_re = self.cls('x', np.arange(3), {'foo': 'bar'})
        self.assertVariableIdentical(v.real, expected_re)

        expected_im = self.cls('x', -np.arange(3), {'foo': 'bar'})
        self.assertVariableIdentical(v.imag, expected_im)

        expected_abs = self.cls('x', np.sqrt(2 * np.arange(3) ** 2)).to_base_variable()
        self.assertVariableAllClose(abs(v), expected_abs)

    def test_aggregate_complex(self):
        # should skip NaNs
        v = self.cls('x', [1, 2j, np.nan])
        expected = Variable((), 0.5 + 1j)
        self.assertVariableAllClose(v.mean(), expected)

    def test_pandas_cateogrical_dtype(self):
        data = pd.Categorical(np.arange(10, dtype='int64'))
        v = self.cls('x', data)
        print(v)  # should not error
        assert v.dtype == 'int64'

    def test_pandas_datetime64_with_tz(self):
        data = pd.date_range(start='2000-01-01',
                             tz=pytz.timezone('America/New_York'),
                             periods=10, freq='1h')
        v = self.cls('x', data)
        print(v)  # should not error
        if 'America/New_York' in str(data.dtype):
            # pandas is new enough that it has datetime64 with timezone dtype
            assert v.dtype == 'object'

    def test_multiindex(self):
        idx = pd.MultiIndex.from_product([list('abc'), [0, 1]])
        v = self.cls('x', idx)
        self.assertVariableIdentical(Variable((), ('a', 0)), v[0])
        self.assertVariableIdentical(v, v[:])

    def test_load(self):
        array = self.cls('x', np.arange(5))
        orig_data = array._data
        copied = array.copy(deep=True)
        if array.chunks is None:
            array.load()
            assert type(array._data) is type(orig_data)
            assert type(copied._data) is type(orig_data)
            self.assertVariableIdentical(array, copied)

    def test_getitem_advanced(self):
        v = self.cls(['x', 'y'], [[0, 1, 2], [3, 4, 5]])
        v_data = v.compute().data

        # orthogonal indexing
        v_new = v[([0, 1], [1, 0])]
        assert v_new.dims == ('x', 'y')
        self.assertArrayEqual(v_new, v_data[[0, 1]][:, [1, 0]])

        v_new = v[[0, 1]]
        assert v_new.dims == ('x', 'y')
        self.assertArrayEqual(v_new, v_data[[0, 1]])

        # with mixed arguments
        ind = Variable(['a'], [0, 1])
        v_new = v[dict(x=[0, 1], y=ind)]
        assert v_new.dims == ('x', 'a')
        self.assertArrayEqual(v_new, v_data[[0, 1]][:, [0, 1]])

        # boolean indexing
        v_new = v[dict(x=[True, False], y=[False, True, False])]
        assert v_new.dims == ('x', 'y')
        self.assertArrayEqual(v_new, v_data[0][1])

        # with scalar variable
        ind = Variable((), 2)
        v_new = v[dict(y=ind)]
        expected = v[dict(y=2)]
        self.assertArrayEqual(v_new, expected)

        # with boolean variable with wrong shape
        ind = np.array([True, False])
        with raises_regex(IndexError, 'Boolean array size 2 is '):
            v[Variable(('a', 'b'), [[0, 1]]), ind]

        # boolean indexing with different dimension
        ind = Variable(['a'], [True, False, False])
        with raises_regex(IndexError, 'Boolean indexer should be'):
            v[dict(y=ind)]

    def test_getitem_uint_1d(self):
        # regression test for #1405
        v = self.cls(['x'], [0, 1, 2])
        v_data = v.compute().data

        v_new = v[np.array([0])]
        self.assertArrayEqual(v_new, v_data[0])
        v_new = v[np.array([0], dtype="uint64")]
        self.assertArrayEqual(v_new, v_data[0])

    def test_getitem_uint(self):
        # regression test for #1405
        v = self.cls(['x', 'y'], [[0, 1, 2], [3, 4, 5]])
        v_data = v.compute().data

        v_new = v[np.array([0])]
        self.assertArrayEqual(v_new, v_data[[0], :])
        v_new = v[np.array([0], dtype="uint64")]
        self.assertArrayEqual(v_new, v_data[[0], :])

        v_new = v[np.uint64(0)]
        self.assertArrayEqual(v_new, v_data[0, :])

    def test_getitem_0d_array(self):
        # make sure 0d-np.array can be used as an indexer
        v = self.cls(['x'], [0, 1, 2])
        v_data = v.compute().data

        v_new = v[np.array([0])[0]]
        self.assertArrayEqual(v_new, v_data[0])

    def test_getitem_fancy(self):
        v = self.cls(['x', 'y'], [[0, 1, 2], [3, 4, 5]])
        v_data = v.compute().data

        ind = Variable(['a', 'b'], [[0, 1, 1], [1, 1, 0]])
        v_new = v[ind]
        assert v_new.dims == ('a', 'b', 'y')
        self.assertArrayEqual(v_new, v_data[[[0, 1, 1], [1, 1, 0]], :])

        # It would be ok if indexed with the multi-dimensional array including
        # the same name
        ind = Variable(['x', 'b'], [[0, 1, 1], [1, 1, 0]])
        v_new = v[ind]
        assert v_new.dims == ('x', 'b', 'y')
        self.assertArrayEqual(v_new, v_data[[[0, 1, 1], [1, 1, 0]], :])

        ind = Variable(['a', 'b'], [[0, 1, 2], [2, 1, 0]])
        v_new = v[dict(y=ind)]
        assert v_new.dims == ('x', 'a', 'b')
        self.assertArrayEqual(v_new, v_data[:, ([0, 1, 2], [2, 1, 0])])

        ind = Variable(['a', 'b'], [[0, 0], [1, 1]])
        v_new = v[dict(x=[1, 0], y=ind)]
        assert v_new.dims == ('x', 'a', 'b')
        self.assertArrayEqual(v_new, v_data[[1, 0]][:, ind])

        # along diagonal
        ind = Variable(['a'], [0, 1])
        v_new = v[ind, ind]
        assert v_new.dims == ('a',)
        self.assertArrayEqual(v_new, v_data[[0, 1], [0, 1]])

        # with integer
        ind = Variable(['a', 'b'], [[0, 0], [1, 1]])
        v_new = v[dict(x=0, y=ind)]
        assert v_new.dims == ('a', 'b')
        self.assertArrayEqual(v_new[0], v_data[0][[0, 0]])
        self.assertArrayEqual(v_new[1], v_data[0][[1, 1]])

        # with slice
        ind = Variable(['a', 'b'], [[0, 0], [1, 1]])
        v_new = v[dict(x=slice(None), y=ind)]
        assert v_new.dims == ('x', 'a', 'b')
        self.assertArrayEqual(v_new, v_data[:, [[0, 0], [1, 1]]])

        ind = Variable(['a', 'b'], [[0, 0], [1, 1]])
        v_new = v[dict(x=ind, y=slice(None))]
        assert v_new.dims == ('a', 'b', 'y')
        self.assertArrayEqual(v_new, v_data[[[0, 0], [1, 1]], :])

        ind = Variable(['a', 'b'], [[0, 0], [1, 1]])
        v_new = v[dict(x=ind, y=slice(None, 1))]
        assert v_new.dims == ('a', 'b', 'y')
        self.assertArrayEqual(v_new, v_data[[[0, 0], [1, 1]], slice(None, 1)])

        # slice matches explicit dimension
        ind = Variable(['y'], [0, 1])
        v_new = v[ind, :2]
        assert v_new.dims == ('y',)
        self.assertArrayEqual(v_new, v_data[[0, 1], [0, 1]])

        # with multiple slices
        v = self.cls(['x', 'y', 'z'], [[[1, 2, 3], [4, 5, 6]]])
        ind = Variable(['a', 'b'], [[0]])
        v_new = v[ind, :, :]
        expected = Variable(['a', 'b', 'y', 'z'], v.data[np.newaxis, ...])
        self.assertVariableIdentical(v_new, expected)

        v = Variable(['w', 'x', 'y', 'z'], [[[[1, 2, 3], [4, 5, 6]]]])
        ind = Variable(['y'], [0])
        v_new = v[ind, :, 1:2, 2]
        expected = Variable(['y', 'x'], [[6]])
        self.assertVariableIdentical(v_new, expected)

        # slice and vector mixed indexing resulting in the same dimension
        v = Variable(['x', 'y', 'z'], np.arange(60).reshape(3, 4, 5))
        ind = Variable(['x'], [0, 1, 2])
        v_new = v[:, ind]
        expected = Variable(('x', 'z'), np.zeros((3, 5)))
        expected[0] = v.data[0, 0]
        expected[1] = v.data[1, 1]
        expected[2] = v.data[2, 2]
        self.assertVariableIdentical(v_new, expected)

        v_new = v[:, ind.data]
        assert v_new.shape == (3, 3, 5)

    def test_getitem_error(self):
        v = self.cls(['x', 'y'], [[0, 1, 2], [3, 4, 5]])

        with raises_regex(IndexError, "labeled multi-"):
            v[[[0, 1], [1, 2]]]

        ind_x = Variable(['a'], [0, 1, 1])
        ind_y = Variable(['a'], [0, 1])
        with raises_regex(IndexError, "Dimensions of indexers "):
            v[ind_x, ind_y]

        ind = Variable(['a', 'b'], [[True, False], [False, True]])
        with raises_regex(IndexError, '2-dimensional boolean'):
            v[dict(x=ind)]

        v = Variable(['x', 'y', 'z'], np.arange(60).reshape(3, 4, 5))
        ind = Variable(['x'], [0, 1])
        with raises_regex(IndexError, 'Dimensions of indexers mis'):
            v[:, ind]


class TestVariable(TestCase, VariableSubclassTestCases):
    cls = staticmethod(Variable)

    def setUp(self):
        self.d = np.random.random((10, 3)).astype(np.float64)

    def test_data_and_values(self):
        v = Variable(['time', 'x'], self.d)
        self.assertArrayEqual(v.data, self.d)
        self.assertArrayEqual(v.values, self.d)
        self.assertIs(source_ndarray(v.values), self.d)
        with pytest.raises(ValueError):
            # wrong size
            v.values = np.random.random(5)
        d2 = np.random.random((10, 3))
        v.values = d2
        self.assertIs(source_ndarray(v.values), d2)
        d3 = np.random.random((10, 3))
        v.data = d3
        self.assertIs(source_ndarray(v.data), d3)

    def test_numpy_same_methods(self):
        v = Variable([], np.float32(0.0))
        self.assertEqual(v.item(), 0)
        self.assertIs(type(v.item()), float)

        v = IndexVariable('x', np.arange(5))
        self.assertEqual(2, v.searchsorted(2))

    def test_datetime64_conversion_scalar(self):
        expected = np.datetime64('2000-01-01', 'ns')
        for values in [
                 np.datetime64('2000-01-01'),
                 pd.Timestamp('2000-01-01T00'),
                 datetime(2000, 1, 1),
                ]:
            v = Variable([], values)
            self.assertEqual(v.dtype, np.dtype('datetime64[ns]'))
            self.assertEqual(v.values, expected)
            self.assertEqual(v.values.dtype, np.dtype('datetime64[ns]'))

    def test_timedelta64_conversion_scalar(self):
        expected = np.timedelta64(24 * 60 * 60 * 10 ** 9, 'ns')
        for values in [
                 np.timedelta64(1, 'D'),
                 pd.Timedelta('1 day'),
                 timedelta(days=1),
                ]:
            v = Variable([], values)
            self.assertEqual(v.dtype, np.dtype('timedelta64[ns]'))
            self.assertEqual(v.values, expected)
            self.assertEqual(v.values.dtype, np.dtype('timedelta64[ns]'))

    def test_0d_str(self):
        v = Variable([], u'foo')
        self.assertEqual(v.dtype, np.dtype('U3'))
        self.assertEqual(v.values, 'foo')

        v = Variable([], np.string_('foo'))
        self.assertEqual(v.dtype, np.dtype('S3'))
        self.assertEqual(v.values, bytes('foo', 'ascii') if PY3 else 'foo')

    def test_0d_datetime(self):
        v = Variable([], pd.Timestamp('2000-01-01'))
        self.assertEqual(v.dtype, np.dtype('datetime64[ns]'))
        self.assertEqual(v.values, np.datetime64('2000-01-01', 'ns'))

    def test_0d_timedelta(self):
        for td in [pd.to_timedelta('1s'), np.timedelta64(1, 's')]:
            v = Variable([], td)
            self.assertEqual(v.dtype, np.dtype('timedelta64[ns]'))
            self.assertEqual(v.values, np.timedelta64(10 ** 9, 'ns'))

    def test_equals_and_identical(self):
        d = np.random.rand(10, 3)
        d[0, 0] = np.nan
        v1 = Variable(('dim1', 'dim2'), data=d,
                      attrs={'att1': 3, 'att2': [1, 2, 3]})
        v2 = Variable(('dim1', 'dim2'), data=d,
                      attrs={'att1': 3, 'att2': [1, 2, 3]})
        self.assertTrue(v1.equals(v2))
        self.assertTrue(v1.identical(v2))

        v3 = Variable(('dim1', 'dim3'), data=d)
        self.assertFalse(v1.equals(v3))

        v4 = Variable(('dim1', 'dim2'), data=d)
        self.assertTrue(v1.equals(v4))
        self.assertFalse(v1.identical(v4))

        v5 = deepcopy(v1)
        v5.values[:] = np.random.rand(10, 3)
        self.assertFalse(v1.equals(v5))

        self.assertFalse(v1.equals(None))
        self.assertFalse(v1.equals(d))

        self.assertFalse(v1.identical(None))
        self.assertFalse(v1.identical(d))

    def test_broadcast_equals(self):
        v1 = Variable((), np.nan)
        v2 = Variable(('x'), [np.nan, np.nan])
        self.assertTrue(v1.broadcast_equals(v2))
        self.assertFalse(v1.equals(v2))
        self.assertFalse(v1.identical(v2))

        v3 = Variable(('x'), [np.nan])
        self.assertTrue(v1.broadcast_equals(v3))
        self.assertFalse(v1.equals(v3))
        self.assertFalse(v1.identical(v3))

        self.assertFalse(v1.broadcast_equals(None))

        v4 = Variable(('x'), [np.nan] * 3)
        self.assertFalse(v2.broadcast_equals(v4))

    def test_no_conflicts(self):
        v1 = Variable(('x'), [1, 2, np.nan, np.nan])
        v2 = Variable(('x'), [np.nan, 2, 3, np.nan])
        self.assertTrue(v1.no_conflicts(v2))
        self.assertFalse(v1.equals(v2))
        self.assertFalse(v1.broadcast_equals(v2))
        self.assertFalse(v1.identical(v2))

        self.assertFalse(v1.no_conflicts(None))

        v3 = Variable(('y'), [np.nan, 2, 3, np.nan])
        self.assertFalse(v3.no_conflicts(v1))

        d = np.array([1, 2, np.nan, np.nan])
        self.assertFalse(v1.no_conflicts(d))
        self.assertFalse(v2.no_conflicts(d))

        v4 = Variable(('w', 'x'), [d])
        self.assertTrue(v1.no_conflicts(v4))

    def test_as_variable(self):
        data = np.arange(10)
        expected = Variable('x', data)
        expected_extra = Variable('x', data, attrs={'myattr': 'val'},
                                  encoding={'scale_factor': 1})

        self.assertVariableIdentical(expected, as_variable(expected))

        ds = Dataset({'x': expected})
        var = as_variable(ds['x']).to_base_variable()
        self.assertVariableIdentical(expected, var)
        self.assertNotIsInstance(ds['x'], Variable)
        self.assertIsInstance(as_variable(ds['x']), Variable)

        FakeVariable = namedtuple('FakeVariable', 'values dims')
        fake_xarray = FakeVariable(expected.values, expected.dims)
        self.assertVariableIdentical(expected, as_variable(fake_xarray))

        FakeVariable = namedtuple('FakeVariable', 'data dims')
        fake_xarray = FakeVariable(expected.data, expected.dims)
        self.assertVariableIdentical(expected, as_variable(fake_xarray))

        FakeVariable = namedtuple('FakeVariable',
                                  'data values dims attrs encoding')
        fake_xarray = FakeVariable(expected_extra.data, expected_extra.values,
                                   expected_extra.dims, expected_extra.attrs,
                                   expected_extra.encoding)
        self.assertVariableIdentical(expected_extra, as_variable(fake_xarray))

        xarray_tuple = (expected_extra.dims, expected_extra.values,
                        expected_extra.attrs, expected_extra.encoding)
        self.assertVariableIdentical(expected_extra, as_variable(xarray_tuple))

        with raises_regex(TypeError, 'tuples to convert'):
            as_variable(tuple(data))
        with raises_regex(
                TypeError, 'without an explicit list of dimensions'):
            as_variable(data)

        actual = as_variable(data, name='x')
        self.assertVariableIdentical(expected.to_index_variable(), actual)

        actual = as_variable(0)
        expected = Variable([], 0)
        self.assertVariableIdentical(expected, actual)

        data = np.arange(9).reshape((3, 3))
        expected = Variable(('x', 'y'), data)
        with raises_regex(
                ValueError, 'without explicit dimension names'):
            as_variable(data, name='x')
        with raises_regex(
                ValueError, 'has more than 1-dimension'):
            as_variable(expected, name='x')

    def test_repr(self):
        v = Variable(['time', 'x'], [[1, 2, 3], [4, 5, 6]], {'foo': 'bar'})
        expected = dedent("""
        <xarray.Variable (time: 2, x: 3)>
        array([[1, 2, 3],
               [4, 5, 6]])
        Attributes:
            foo:      bar
        """).strip()
        self.assertEqual(expected, repr(v))

    def test_repr_lazy_data(self):
        v = Variable('x', LazilyIndexedArray(np.arange(2e5)))
        self.assertIn('200000 values with dtype', repr(v))
        self.assertIsInstance(v._data, LazilyIndexedArray)

    def test_detect_indexer_type(self):
        """ Tests indexer type was correctly detected. """
        data = np.random.random((10, 11))
        v = Variable(['x', 'y'], data)

        _, ind, _ = v._broadcast_indexes((0, 1))
        assert type(ind) == indexing.BasicIndexer

        _, ind, _ = v._broadcast_indexes((0, slice(0, 8, 2)))
        assert type(ind) == indexing.BasicIndexer

        _, ind, _ = v._broadcast_indexes((0, [0, 1]))
        assert type(ind) == indexing.OuterIndexer

        _, ind, _ = v._broadcast_indexes(([0, 1], 1))
        assert type(ind) == indexing.OuterIndexer

        _, ind, _ = v._broadcast_indexes(([0, 1], [1, 2]))
        assert type(ind) == indexing.OuterIndexer

        _, ind, _ = v._broadcast_indexes(([0, 1], slice(0, 8, 2)))
        assert type(ind) == indexing.OuterIndexer

        vind = Variable(('a', ), [0, 1])
        _, ind, _ = v._broadcast_indexes((vind, slice(0, 8, 2)))
        assert type(ind) == indexing.OuterIndexer

        vind = Variable(('y', ), [0, 1])
        _, ind, _ = v._broadcast_indexes((vind, 3))
        assert type(ind) == indexing.OuterIndexer

        vind = Variable(('a', ), [0, 1])
        _, ind, _ = v._broadcast_indexes((vind, vind))
        assert type(ind) == indexing.VectorizedIndexer

        vind = Variable(('a', 'b'), [[0, 2], [1, 3]])
        _, ind, _ = v._broadcast_indexes((vind, 3))
        assert type(ind) == indexing.VectorizedIndexer

    def test_indexer_type(self):
        # GH:issue:1688. Wrong indexer type induces NotImplementedError
        data = np.random.random((10, 11))
        v = Variable(['x', 'y'], data)

        def assert_indexer_type(key, object_type):
            dims, index_tuple, new_order = v._broadcast_indexes(key)
            assert isinstance(index_tuple, object_type)

        # should return BasicIndexer
        assert_indexer_type((0, 1), BasicIndexer)
        assert_indexer_type((0, slice(None, None)), BasicIndexer)
        assert_indexer_type((Variable([], 3), slice(None, None)), BasicIndexer)
        assert_indexer_type((Variable([], 3), (Variable([], 6))), BasicIndexer)

        # should return OuterIndexer
        assert_indexer_type(([0, 1], 1), OuterIndexer)
        assert_indexer_type(([0, 1], [1, 2]), OuterIndexer)
        assert_indexer_type((Variable(('x'), [0, 1]), 1), OuterIndexer)
        assert_indexer_type((Variable(('x'), [0, 1]), slice(None, None)),
                            OuterIndexer)
        assert_indexer_type((Variable(('x'), [0, 1]), Variable(('y'), [0, 1])),
                            OuterIndexer)

        # should return VectorizedIndexer
        assert_indexer_type((Variable(('y'), [0, 1]), [0, 1]),
                            VectorizedIndexer)
        assert_indexer_type((Variable(('z'), [0, 1]), Variable(('z'), [0, 1])),
                            VectorizedIndexer)
        assert_indexer_type((Variable(('a', 'b'), [[0, 1], [1, 2]]),
                             Variable(('a', 'b'), [[0, 1], [1, 2]])),
                            VectorizedIndexer)

    def test_items(self):
        data = np.random.random((10, 11))
        v = Variable(['x', 'y'], data)
        # test slicing
        self.assertVariableIdentical(v, v[:])
        self.assertVariableIdentical(v, v[...])
        self.assertVariableIdentical(Variable(['y'], data[0]), v[0])
        self.assertVariableIdentical(Variable(['x'], data[:, 0]), v[:, 0])
        self.assertVariableIdentical(Variable(['x', 'y'], data[:3, :2]),
                                     v[:3, :2])
        # test array indexing
        x = Variable(['x'], np.arange(10))
        y = Variable(['y'], np.arange(11))
        self.assertVariableIdentical(v, v[x.values])
        self.assertVariableIdentical(v, v[x])
        self.assertVariableIdentical(v[:3], v[x < 3])
        self.assertVariableIdentical(v[:, 3:], v[:, y >= 3])
        self.assertVariableIdentical(v[:3, 3:], v[x < 3, y >= 3])
        self.assertVariableIdentical(v[:3, :2], v[x[:3], y[:2]])
        self.assertVariableIdentical(v[:3, :2], v[range(3), range(2)])
        # test iteration
        for n, item in enumerate(v):
            self.assertVariableIdentical(Variable(['y'], data[n]), item)
        with raises_regex(TypeError, 'iteration over a 0-d'):
            iter(Variable([], 0))
        # test setting
        v.values[:] = 0
        self.assertTrue(np.all(v.values == 0))
        # test orthogonal setting
        v[range(10), range(11)] = 1
        self.assertArrayEqual(v.values, np.ones((10, 11)))

    def test_getitem_basic(self):
        v = self.cls(['x', 'y'], [[0, 1, 2], [3, 4, 5]])

        v_new = v[dict(x=0)]
        assert v_new.dims == ('y', )
        self.assertArrayEqual(v_new, v._data[0])

        v_new = v[dict(x=0, y=slice(None))]
        assert v_new.dims == ('y', )
        self.assertArrayEqual(v_new, v._data[0])

        v_new = v[dict(x=0, y=1)]
        assert v_new.dims == ()
        self.assertArrayEqual(v_new, v._data[0, 1])

        v_new = v[dict(y=1)]
        assert v_new.dims == ('x', )
        self.assertArrayEqual(v_new, v._data[:, 1])

        # tuple argument
        v_new = v[(slice(None), 1)]
        assert v_new.dims == ('x', )
        self.assertArrayEqual(v_new, v._data[:, 1])

    def test_isel(self):
        v = Variable(['time', 'x'], self.d)
        self.assertVariableIdentical(v.isel(time=slice(None)), v)
        self.assertVariableIdentical(v.isel(time=0), v[0])
        self.assertVariableIdentical(v.isel(time=slice(0, 3)), v[:3])
        self.assertVariableIdentical(v.isel(x=0), v[:, 0])
        with raises_regex(ValueError, 'do not exist'):
            v.isel(not_a_dim=0)

    def test_index_0d_numpy_string(self):
        # regression test to verify our work around for indexing 0d strings
        v = Variable([], np.string_('asdf'))
        self.assertVariableIdentical(v[()], v)

        v = Variable([], np.unicode_(u'asdf'))
        self.assertVariableIdentical(v[()], v)

    def test_indexing_0d_unicode(self):
        # regression test for GH568
        actual = Variable(('x'), [u'tmax'])[0][()]
        expected = Variable((), u'tmax')
        self.assertVariableIdentical(actual, expected)

    def test_shift(self):
        v = Variable('x', [1, 2, 3, 4, 5])

        self.assertVariableIdentical(v, v.shift(x=0))
        self.assertIsNot(v, v.shift(x=0))

        expected = Variable('x', [np.nan, 1, 2, 3, 4])
        self.assertVariableIdentical(expected, v.shift(x=1))

        expected = Variable('x', [np.nan, np.nan, 1, 2, 3])
        self.assertVariableIdentical(expected, v.shift(x=2))

        expected = Variable('x', [2, 3, 4, 5, np.nan])
        self.assertVariableIdentical(expected, v.shift(x=-1))

        expected = Variable('x', [np.nan] * 5)
        self.assertVariableIdentical(expected, v.shift(x=5))
        self.assertVariableIdentical(expected, v.shift(x=6))

        with raises_regex(ValueError, 'dimension'):
            v.shift(z=0)

        v = Variable('x', [1, 2, 3, 4, 5], {'foo': 'bar'})
        self.assertVariableIdentical(v, v.shift(x=0))

        expected = Variable('x', [np.nan, 1, 2, 3, 4], {'foo': 'bar'})
        self.assertVariableIdentical(expected, v.shift(x=1))

    def test_shift2d(self):
        v = Variable(('x', 'y'), [[1, 2], [3, 4]])
        expected = Variable(('x', 'y'), [[np.nan, np.nan], [np.nan, 1]])
        self.assertVariableIdentical(expected, v.shift(x=1, y=1))

    def test_roll(self):
        v = Variable('x', [1, 2, 3, 4, 5])

        self.assertVariableIdentical(v, v.roll(x=0))
        self.assertIsNot(v, v.roll(x=0))

        expected = Variable('x', [5, 1, 2, 3, 4])
        self.assertVariableIdentical(expected, v.roll(x=1))
        self.assertVariableIdentical(expected, v.roll(x=-4))
        self.assertVariableIdentical(expected, v.roll(x=6))

        expected = Variable('x', [4, 5, 1, 2, 3])
        self.assertVariableIdentical(expected, v.roll(x=2))
        self.assertVariableIdentical(expected, v.roll(x=-3))

        with raises_regex(ValueError, 'dimension'):
            v.roll(z=0)

    def test_roll_consistency(self):
        v = Variable(('x', 'y'), np.random.randn(5, 6))

        for axis, dim in [(0, 'x'), (1, 'y')]:
            for shift in [-3, 0, 1, 7, 11]:
                expected = np.roll(v.values, shift, axis=axis)
                actual = v.roll(**{dim: shift}).values
                self.assertArrayEqual(expected, actual)

    def test_transpose(self):
        v = Variable(['time', 'x'], self.d)
        v2 = Variable(['x', 'time'], self.d.T)
        self.assertVariableIdentical(v, v2.transpose())
        self.assertVariableIdentical(v.transpose(), v.T)
        x = np.random.randn(2, 3, 4, 5)
        w = Variable(['a', 'b', 'c', 'd'], x)
        w2 = Variable(['d', 'b', 'c', 'a'], np.einsum('abcd->dbca', x))
        self.assertEqual(w2.shape, (5, 3, 4, 2))
        self.assertVariableIdentical(w2, w.transpose('d', 'b', 'c', 'a'))
        self.assertVariableIdentical(w, w2.transpose('a', 'b', 'c', 'd'))
        w3 = Variable(['b', 'c', 'd', 'a'], np.einsum('abcd->bcda', x))
        self.assertVariableIdentical(w, w3.transpose('a', 'b', 'c', 'd'))

    def test_transpose_0d(self):
        for value in [
                3.5,
                ('a', 1),
                np.datetime64('2000-01-01'),
                np.timedelta64(1, 'h'),
                None,
                object(),
                ]:
            variable = Variable([], value)
            actual = variable.transpose()
            assert actual.identical(variable)

    def test_squeeze(self):
        v = Variable(['x', 'y'], [[1]])
        self.assertVariableIdentical(Variable([], 1), v.squeeze())
        self.assertVariableIdentical(Variable(['y'], [1]), v.squeeze('x'))
        self.assertVariableIdentical(Variable(['y'], [1]), v.squeeze(['x']))
        self.assertVariableIdentical(Variable(['x'], [1]), v.squeeze('y'))
        self.assertVariableIdentical(Variable([], 1), v.squeeze(['x', 'y']))

        v = Variable(['x', 'y'], [[1, 2]])
        self.assertVariableIdentical(Variable(['y'], [1, 2]), v.squeeze())
        self.assertVariableIdentical(Variable(['y'], [1, 2]), v.squeeze('x'))
        with raises_regex(ValueError, 'cannot select a dimension'):
            v.squeeze('y')

    def test_get_axis_num(self):
        v = Variable(['x', 'y', 'z'], np.random.randn(2, 3, 4))
        self.assertEqual(v.get_axis_num('x'), 0)
        self.assertEqual(v.get_axis_num(['x']), (0,))
        self.assertEqual(v.get_axis_num(['x', 'y']), (0, 1))
        self.assertEqual(v.get_axis_num(['z', 'y', 'x']), (2, 1, 0))
        with raises_regex(ValueError, 'not found in array dim'):
            v.get_axis_num('foobar')

    def test_set_dims(self):
        v = Variable(['x'], [0, 1])
        actual = v.set_dims(['x', 'y'])
        expected = Variable(['x', 'y'], [[0], [1]])
        self.assertVariableIdentical(actual, expected)

        actual = v.set_dims(['y', 'x'])
        self.assertVariableIdentical(actual, expected.T)

        actual = v.set_dims(OrderedDict([('x', 2), ('y', 2)]))
        expected = Variable(['x', 'y'], [[0, 0], [1, 1]])
        self.assertVariableIdentical(actual, expected)

        v = Variable(['foo'], [0, 1])
        actual = v.set_dims('foo')
        expected = v
        self.assertVariableIdentical(actual, expected)

        with raises_regex(ValueError, 'must be a superset'):
            v.set_dims(['z'])

    def test_set_dims_object_dtype(self):
        v = Variable([], ('a', 1))
        actual = v.set_dims(('x',), (3,))
        exp_values = np.empty((3,), dtype=object)
        for i in range(3):
            exp_values[i] = ('a', 1)
        expected = Variable(['x'], exp_values)
        assert actual.identical(expected)

    def test_stack(self):
        v = Variable(['x', 'y'], [[0, 1], [2, 3]], {'foo': 'bar'})
        actual = v.stack(z=('x', 'y'))
        expected = Variable('z', [0, 1, 2, 3], v.attrs)
        self.assertVariableIdentical(actual, expected)

        actual = v.stack(z=('x',))
        expected = Variable(('y', 'z'), v.data.T, v.attrs)
        self.assertVariableIdentical(actual, expected)

        actual = v.stack(z=(),)
        self.assertVariableIdentical(actual, v)

        actual = v.stack(X=('x',), Y=('y',)).transpose('X', 'Y')
        expected = Variable(('X', 'Y'), v.data, v.attrs)
        self.assertVariableIdentical(actual, expected)

    def test_stack_errors(self):
        v = Variable(['x', 'y'], [[0, 1], [2, 3]], {'foo': 'bar'})

        with raises_regex(ValueError, 'invalid existing dim'):
            v.stack(z=('x1',))
        with raises_regex(ValueError, 'cannot create a new dim'):
            v.stack(x=('x',))

    def test_unstack(self):
        v = Variable('z', [0, 1, 2, 3], {'foo': 'bar'})
        actual = v.unstack(z=OrderedDict([('x', 2), ('y', 2)]))
        expected = Variable(('x', 'y'), [[0, 1], [2, 3]], v.attrs)
        self.assertVariableIdentical(actual, expected)

        actual = v.unstack(z=OrderedDict([('x', 4), ('y', 1)]))
        expected = Variable(('x', 'y'), [[0], [1], [2], [3]], v.attrs)
        self.assertVariableIdentical(actual, expected)

        actual = v.unstack(z=OrderedDict([('x', 4)]))
        expected = Variable('x', [0, 1, 2, 3], v.attrs)
        self.assertVariableIdentical(actual, expected)

    def test_unstack_errors(self):
        v = Variable('z', [0, 1, 2, 3])
        with raises_regex(ValueError, 'invalid existing dim'):
            v.unstack(foo={'x': 4})
        with raises_regex(ValueError, 'cannot create a new dim'):
            v.stack(z=('z',))
        with raises_regex(ValueError, 'the product of the new dim'):
            v.unstack(z={'x': 5})

    def test_unstack_2d(self):
        v = Variable(['x', 'y'], [[0, 1], [2, 3]])
        actual = v.unstack(y={'z': 2})
        expected = Variable(['x', 'z'], v.data)
        self.assertVariableIdentical(actual, expected)

        actual = v.unstack(x={'z': 2})
        expected = Variable(['y', 'z'], v.data.T)
        self.assertVariableIdentical(actual, expected)

    def test_stack_unstack_consistency(self):
        v = Variable(['x', 'y'], [[0, 1], [2, 3]])
        actual = (v.stack(z=('x', 'y'))
                  .unstack(z=OrderedDict([('x', 2), ('y', 2)])))
        self.assertVariableIdentical(actual, v)

    def test_broadcasting_math(self):
        x = np.random.randn(2, 3)
        v = Variable(['a', 'b'], x)
        # 1d to 2d broadcasting
        self.assertVariableIdentical(
            v * v,
            Variable(['a', 'b'], np.einsum('ab,ab->ab', x, x)))
        self.assertVariableIdentical(
            v * v[0],
            Variable(['a', 'b'], np.einsum('ab,b->ab', x, x[0])))
        self.assertVariableIdentical(
            v[0] * v,
            Variable(['b', 'a'], np.einsum('b,ab->ba', x[0], x)))
        self.assertVariableIdentical(
            v[0] * v[:, 0],
            Variable(['b', 'a'], np.einsum('b,a->ba', x[0], x[:, 0])))
        # higher dim broadcasting
        y = np.random.randn(3, 4, 5)
        w = Variable(['b', 'c', 'd'], y)
        self.assertVariableIdentical(
            v * w, Variable(['a', 'b', 'c', 'd'],
                            np.einsum('ab,bcd->abcd', x, y)))
        self.assertVariableIdentical(
            w * v, Variable(['b', 'c', 'd', 'a'],
                            np.einsum('bcd,ab->bcda', y, x)))
        self.assertVariableIdentical(
            v * w[0], Variable(['a', 'b', 'c', 'd'],
                               np.einsum('ab,cd->abcd', x, y[0])))

    def test_broadcasting_failures(self):
        a = Variable(['x'], np.arange(10))
        b = Variable(['x'], np.arange(5))
        c = Variable(['x', 'x'], np.arange(100).reshape(10, 10))
        with raises_regex(ValueError, 'mismatched lengths'):
            a + b
        with raises_regex(ValueError, 'duplicate dimensions'):
            a + c

    def test_inplace_math(self):
        x = np.arange(5)
        v = Variable(['x'], x)
        v2 = v
        v2 += 1
        self.assertIs(v, v2)
        # since we provided an ndarray for data, it is also modified in-place
        self.assertIs(source_ndarray(v.values), x)
        self.assertArrayEqual(v.values, np.arange(5) + 1)

        with raises_regex(ValueError, 'dimensions cannot change'):
            v += Variable('y', np.arange(5))

    def test_reduce(self):
        v = Variable(['x', 'y'], self.d, {'ignored': 'attributes'})
        self.assertVariableIdentical(v.reduce(np.std, 'x'),
                                     Variable(['y'], self.d.std(axis=0)))
        self.assertVariableIdentical(v.reduce(np.std, axis=0),
                                     v.reduce(np.std, dim='x'))
        self.assertVariableIdentical(v.reduce(np.std, ['y', 'x']),
                                     Variable([], self.d.std(axis=(0, 1))))
        self.assertVariableIdentical(v.reduce(np.std),
                                     Variable([], self.d.std()))
        self.assertVariableIdentical(
            v.reduce(np.mean, 'x').reduce(np.std, 'y'),
            Variable([], self.d.mean(axis=0).std()))
        self.assertVariableAllClose(v.mean('x'), v.reduce(np.mean, 'x'))

        with raises_regex(ValueError, 'cannot supply both'):
            v.mean(dim='x', axis=0)

    @pytest.mark.skipif(LooseVersion(np.__version__) < LooseVersion('1.10.0'),
                        reason='requires numpy version 1.10.0 or later')
    def test_quantile(self):
        v = Variable(['x', 'y'], self.d)
        for q in [0.25, [0.50], [0.25, 0.75]]:
            for axis, dim in zip([None, 0, [0], [0, 1]],
                                 [None, 'x', ['x'], ['x', 'y']]):
                actual = v.quantile(q, dim=dim)

                expected = np.nanpercentile(self.d, np.array(q) * 100,
                                            axis=axis)
                np.testing.assert_allclose(actual.values, expected)

    @requires_dask
    def test_quantile_dask_raises(self):
        # regression for GH1524
        v = Variable(['x', 'y'], self.d).chunk(2)

        with raises_regex(TypeError, 'arrays stored as dask'):
            v.quantile(0.5, dim='x')

    def test_big_endian_reduce(self):
        # regression test for GH489
        data = np.ones(5, dtype='>f4')
        v = Variable(['x'], data)
        expected = Variable([], 5)
        self.assertVariableIdentical(expected, v.sum())

    def test_reduce_funcs(self):
        v = Variable('x', np.array([1, np.nan, 2, 3]))
        self.assertVariableIdentical(v.mean(), Variable([], 2))
        self.assertVariableIdentical(v.mean(skipna=True), Variable([], 2))
        self.assertVariableIdentical(v.mean(skipna=False), Variable([], np.nan))
        self.assertVariableIdentical(np.mean(v), Variable([], 2))

        self.assertVariableIdentical(v.prod(), Variable([], 6))
        self.assertVariableIdentical(v.cumsum(axis=0),
                                     Variable('x', np.array([1, 1, 3, 6])))
        self.assertVariableIdentical(v.cumprod(axis=0),
                                     Variable('x', np.array([1, 1, 2, 6])))
        self.assertVariableIdentical(v.var(), Variable([], 2.0 / 3))

        if LooseVersion(np.__version__) < '1.9':
            with pytest.raises(NotImplementedError):
                v.median()
        else:
            self.assertVariableIdentical(v.median(), Variable([], 2))

        v = Variable('x', [True, False, False])
        self.assertVariableIdentical(v.any(), Variable([], True))
        self.assertVariableIdentical(v.all(dim='x'), Variable([], False))

        v = Variable('t', pd.date_range('2000-01-01', periods=3))
        with pytest.raises(NotImplementedError):
            v.max(skipna=True)
        self.assertVariableIdentical(
            v.max(), Variable([], pd.Timestamp('2000-01-03')))

    def test_reduce_keep_attrs(self):
        _attrs = {'units': 'test', 'long_name': 'testing'}

        v = Variable(['x', 'y'], self.d, _attrs)

        # Test dropped attrs
        vm = v.mean()
        self.assertEqual(len(vm.attrs), 0)
        self.assertEqual(vm.attrs, OrderedDict())

        # Test kept attrs
        vm = v.mean(keep_attrs=True)
        self.assertEqual(len(vm.attrs), len(_attrs))
        self.assertEqual(vm.attrs, _attrs)

    def test_count(self):
        expected = Variable([], 3)
        actual = Variable(['x'], [1, 2, 3, np.nan]).count()
        self.assertVariableIdentical(expected, actual)

        v = Variable(['x'], np.array(['1', '2', '3', np.nan], dtype=object))
        actual = v.count()
        self.assertVariableIdentical(expected, actual)

        actual = Variable(['x'], [True, False, True]).count()
        self.assertVariableIdentical(expected, actual)
        self.assertEqual(actual.dtype, int)

        expected = Variable(['x'], [2, 3])
        actual = Variable(['x', 'y'], [[1, 0, np.nan], [1, 1, 1]]).count('y')
        self.assertVariableIdentical(expected, actual)

    def test_setitem(self):
        v = Variable(['x', 'y'], [[0, 3, 2], [3, 4, 5]])
        v[0, 1] = 1
        self.assertTrue(v[0, 1] == 1)

        v = Variable(['x', 'y'], [[0, 3, 2], [3, 4, 5]])
        v[dict(x=[0, 1])] = 1
        self.assertArrayEqual(v[[0, 1]], np.ones_like(v[[0, 1]]))

        # boolean indexing
        v = Variable(['x', 'y'], [[0, 3, 2], [3, 4, 5]])
        v[dict(x=[True, False])] = 1

        self.assertArrayEqual(v[0], np.ones_like(v[0]))
        v = Variable(['x', 'y'], [[0, 3, 2], [3, 4, 5]])
        v[dict(x=[True, False], y=[False, True, False])] = 1
        self.assertTrue(v[0, 1] == 1)

        # dimension broadcast
        v = Variable(['x', 'y'], np.ones((3, 2)))
        ind = Variable(['a', 'b'], [[0, 1]])
        v[ind, :] = 0
        expected = Variable(['x', 'y'], [[0, 0], [0, 0], [1, 1]])
        self.assertVariableIdentical(expected, v)

        with raises_regex(ValueError, "shape mismatch"):
            v[ind, ind] = np.zeros((1, 2, 1))

        v = Variable(['x', 'y'], [[0, 3, 2], [3, 4, 5]])
        ind = Variable(['a'], [0, 1])
        v[dict(x=ind)] = Variable(['a', 'y'], np.ones((2, 3), dtype=int) * 10)
        self.assertArrayEqual(v[0], np.ones_like(v[0]) * 10)
        self.assertArrayEqual(v[1], np.ones_like(v[1]) * 10)
        assert v.dims == ('x', 'y')  # dimension should not change

        # increment
        v = Variable(['x', 'y'], np.arange(6).reshape(3, 2))
        ind = Variable(['a'], [0, 1])
        v[dict(x=ind)] += 1
        expected = Variable(['x', 'y'], [[1, 2], [3, 4], [4, 5]])
        self.assertVariableIdentical(v, expected)

        ind = Variable(['a'], [0, 0])
        v[dict(x=ind)] += 1
        expected = Variable(['x', 'y'], [[2, 3], [3, 4], [4, 5]])
        self.assertVariableIdentical(v, expected)


@requires_dask
class TestVariableWithDask(TestCase, VariableSubclassTestCases):
    cls = staticmethod(lambda *args: Variable(*args).chunk())

    @pytest.mark.xfail
    def test_0d_object_array_with_list(self):
        super(TestVariableWithDask, self).test_0d_object_array_with_list()

    @pytest.mark.xfail
    def test_array_interface(self):
        # dask array does not have `argsort`
        super(TestVariableWithDask, self).test_array_interface()

    @pytest.mark.xfail
    def test_copy_index(self):
        super(TestVariableWithDask, self).test_copy_index()

    @pytest.mark.xfail
    def test_eq_all_dtypes(self):
        super(TestVariableWithDask, self).test_eq_all_dtypes()

    def test_getitem_fancy(self):
        import dask
        if LooseVersion(dask.__version__) <= LooseVersion('0.15.1'):
            pytest.xfail("vindex from latest dask is required")
        super(TestVariableWithDask, self).test_getitem_fancy()

    def test_getitem_1d_fancy(self):
        import dask
        if LooseVersion(dask.__version__) <= LooseVersion('0.15.1'):
            pytest.xfail("vindex from latest dask is required")
        super(TestVariableWithDask, self).test_getitem_1d_fancy()


class TestIndexVariable(TestCase, VariableSubclassTestCases):
    cls = staticmethod(IndexVariable)

    def test_init(self):
        with raises_regex(ValueError, 'must be 1-dimensional'):
            IndexVariable((), 0)

    def test_to_index(self):
        data = 0.5 * np.arange(10)
        v = IndexVariable(['time'], data, {'foo': 'bar'})
        self.assertTrue(pd.Index(data, name='time').identical(v.to_index()))

    def test_multiindex_default_level_names(self):
        midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
        v = IndexVariable(['x'], midx, {'foo': 'bar'})
        self.assertEqual(v.to_index().names, ('x_level_0', 'x_level_1'))

    def test_data(self):
        x = IndexVariable('x', np.arange(3.0))
        self.assertIsInstance(x._data, PandasIndexAdapter)
        self.assertIsInstance(x.data, np.ndarray)
        self.assertEqual(float, x.dtype)
        self.assertArrayEqual(np.arange(3), x)
        self.assertEqual(float, x.values.dtype)
        with raises_regex(TypeError, 'cannot be modified'):
            x[:] = 0

    def test_name(self):
        coord = IndexVariable('x', [10.0])
        self.assertEqual(coord.name, 'x')

        with pytest.raises(AttributeError):
            coord.name = 'y'

    def test_level_names(self):
        midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]],
                                          names=['level_1', 'level_2'])
        x = IndexVariable('x', midx)
        self.assertEqual(x.level_names, midx.names)

        self.assertIsNone(IndexVariable('y', [10.0]).level_names)

    def test_get_level_variable(self):
        midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]],
                                          names=['level_1', 'level_2'])
        x = IndexVariable('x', midx)
        level_1 = IndexVariable('x', midx.get_level_values('level_1'))
        self.assertVariableIdentical(x.get_level_variable('level_1'), level_1)

        with raises_regex(ValueError, 'has no MultiIndex'):
            IndexVariable('y', [10.0]).get_level_variable('level')

    def test_concat_periods(self):
        periods = pd.period_range('2000-01-01', periods=10)
        coords = [IndexVariable('t', periods[:5]), IndexVariable('t', periods[5:])]
        expected = IndexVariable('t', periods)
        actual = IndexVariable.concat(coords, dim='t')
        assert actual.identical(expected)
        assert isinstance(actual.to_index(), pd.PeriodIndex)

        positions = [list(range(5)), list(range(5, 10))]
        actual = IndexVariable.concat(coords, dim='t', positions=positions)
        assert actual.identical(expected)
        assert isinstance(actual.to_index(), pd.PeriodIndex)

    def test_concat_multiindex(self):
        idx = pd.MultiIndex.from_product([[0, 1, 2], ['a', 'b']])
        coords = [IndexVariable('x', idx[:2]), IndexVariable('x', idx[2:])]
        expected = IndexVariable('x', idx)
        actual = IndexVariable.concat(coords, dim='x')
        assert actual.identical(expected)
        assert isinstance(actual.to_index(), pd.MultiIndex)

    def test_coordinate_alias(self):
        with self.assertWarns('deprecated'):
            x = Coordinate('x', [1, 2, 3])
        self.assertIsInstance(x, IndexVariable)

    # These tests make use of multi-dimensional variables, which are not valid
    # IndexVariable objects:
    @pytest.mark.xfail
    def test_getitem_error(self):
        super(TestIndexVariable, self).test_getitem_error()

    @pytest.mark.xfail
    def test_getitem_advanced(self):
        super(TestIndexVariable, self).test_getitem_advanced()

    @pytest.mark.xfail
    def test_getitem_fancy(self):
        super(TestIndexVariable, self).test_getitem_fancy()

    @pytest.mark.xfail
    def test_getitem_uint(self):
        super(TestIndexVariable, self).test_getitem_fancy()


class TestAsCompatibleData(TestCase):
    def test_unchanged_types(self):
        types = (np.asarray, PandasIndexAdapter, indexing.LazilyIndexedArray)
        for t in types:
            for data in [np.arange(3),
                         pd.date_range('2000-01-01', periods=3),
                         pd.date_range('2000-01-01', periods=3).values]:
                x = t(data)
                self.assertIs(source_ndarray(x),
                              source_ndarray(as_compatible_data(x)))

    def test_converted_types(self):
        for input_array in [[[0, 1, 2]], pd.DataFrame([[0, 1, 2]])]:
            actual = as_compatible_data(input_array)
            self.assertArrayEqual(np.asarray(input_array), actual)
            self.assertEqual(np.ndarray, type(actual))
            self.assertEqual(np.asarray(input_array).dtype, actual.dtype)

    def test_masked_array(self):
        original = np.ma.MaskedArray(np.arange(5))
        expected = np.arange(5)
        actual = as_compatible_data(original)
        self.assertArrayEqual(expected, actual)
        self.assertEqual(np.dtype(int), actual.dtype)

        original = np.ma.MaskedArray(np.arange(5), mask=4 * [False] + [True])
        expected = np.arange(5.0)
        expected[-1] = np.nan
        actual = as_compatible_data(original)
        self.assertArrayEqual(expected, actual)
        self.assertEqual(np.dtype(float), actual.dtype)

    def test_datetime(self):
        expected = np.datetime64('2000-01-01')
        actual = as_compatible_data(expected)
        self.assertEqual(expected, actual)
        self.assertEqual(np.ndarray, type(actual))
        self.assertEqual(np.dtype('datetime64[ns]'), actual.dtype)

        expected = np.array([np.datetime64('2000-01-01')])
        actual = as_compatible_data(expected)
        self.assertEqual(np.asarray(expected), actual)
        self.assertEqual(np.ndarray, type(actual))
        self.assertEqual(np.dtype('datetime64[ns]'), actual.dtype)

        expected = np.array([np.datetime64('2000-01-01', 'ns')])
        actual = as_compatible_data(expected)
        self.assertEqual(np.asarray(expected), actual)
        self.assertEqual(np.ndarray, type(actual))
        self.assertEqual(np.dtype('datetime64[ns]'), actual.dtype)
        self.assertIs(expected, source_ndarray(np.asarray(actual)))

        expected = np.datetime64('2000-01-01', 'ns')
        actual = as_compatible_data(datetime(2000, 1, 1))
        self.assertEqual(np.asarray(expected), actual)
        self.assertEqual(np.ndarray, type(actual))
        self.assertEqual(np.dtype('datetime64[ns]'), actual.dtype)

    def test_full_like(self):
        # For more thorough tests, see test_variable.py
        orig = Variable(dims=('x', 'y'), data=[[1.5 ,2.0], [3.1, 4.3]],
                        attrs={'foo': 'bar'})

        expect = orig.copy(deep=True)
        expect.values = [[2.0, 2.0], [2.0, 2.0]]
        self.assertVariableIdentical(expect, full_like(orig, 2))

        # override dtype
        expect.values = [[True, True], [True, True]]
        assert expect.dtype == bool
        self.assertVariableIdentical(expect, full_like(orig, True, dtype=bool))

    @requires_dask
    def test_full_like_dask(self):
        orig = Variable(dims=('x', 'y'), data=[[1.5, 2.0], [3.1, 4.3]],
                        attrs={'foo': 'bar'}).chunk(((1, 1), (2,)))

        def check(actual, expect_dtype, expect_values):
            self.assertEqual(actual.dtype, expect_dtype)
            self.assertEqual(actual.shape, orig.shape)
            self.assertEqual(actual.dims, orig.dims)
            self.assertEqual(actual.attrs, orig.attrs)
            self.assertEqual(actual.chunks, orig.chunks)
            self.assertArrayEqual(actual.values, expect_values)

        check(full_like(orig, 2),
              orig.dtype, np.full_like(orig.values, 2))
        # override dtype
        check(full_like(orig, True, dtype=bool),
              bool, np.full_like(orig.values, True, dtype=bool))

        # Check that there's no array stored inside dask
        # (e.g. we didn't create a numpy array and then we chunked it!)
        dsk = full_like(orig, 1).data.dask
        for v in dsk.values():
            if isinstance(v, tuple):
                for vi in v:
                    assert not isinstance(vi, np.ndarray)
            else:
                assert not isinstance(v, np.ndarray)

    def test_zeros_like(self):
        orig = Variable(dims=('x', 'y'), data=[[1.5 ,2.0], [3.1, 4.3]],
                        attrs={'foo': 'bar'})
        self.assertVariableIdentical(zeros_like(orig),
                                     full_like(orig, 0))
        self.assertVariableIdentical(zeros_like(orig, dtype=int),
                                     full_like(orig, 0, dtype=int))

    def test_ones_like(self):
        orig = Variable(dims=('x', 'y'), data=[[1.5 ,2.0], [3.1, 4.3]],
                        attrs={'foo': 'bar'})
        self.assertVariableIdentical(ones_like(orig),
                                     full_like(orig, 1))
        self.assertVariableIdentical(ones_like(orig, dtype=int),
                                     full_like(orig, 1, dtype=int))

    def test_unsupported_type(self):
        # Non indexable type
        class CustomArray(NDArrayMixin):
            def __init__(self, array):
                self.array = array

        class CustomIndexable(CustomArray, indexing.ExplicitlyIndexed):
            pass

        array = CustomArray(np.arange(3))
        orig = Variable(dims=('x'), data=array, attrs={'foo': 'bar'})
        assert isinstance(orig._data, np.ndarray)  # should not be CustomArray

        array = CustomIndexable(np.arange(3))
        orig = Variable(dims=('x'), data=array, attrs={'foo': 'bar'})
        assert isinstance(orig._data, CustomIndexable)


def test_raise_no_warning_for_nan_in_binary_ops():
    with pytest.warns(None) as record:
        Variable('x', [1, 2, np.NaN]) > 0
    assert len(record) == 0


class TestBackendIndexing(TestCase):
    """    Make sure all the array wrappers can be indexed. """
    def setUp(self):
        self.d = np.random.random((10, 3)).astype(np.float64)

    def check_orthogonal_indexing(self, v):
        assert np.allclose(v.isel(x=[8, 3], y=[2, 1]),
                           self.d[[8, 3]][:, [2, 1]])

    def check_vectorized_indexing(self, v):
        ind_x = Variable('z', [0, 2])
        ind_y = Variable('z', [2, 1])
        assert np.allclose(v.isel(x=ind_x, y=ind_y), self.d[ind_x, ind_y])

    def test_NumpyIndexingAdapter(self):
        v = Variable(dims=('x', 'y'), data=NumpyIndexingAdapter(self.d))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)
        # could not doubly wrapping
        with raises_regex(TypeError, 'NumpyIndexingAdapter only wraps '):
            v = Variable(dims=('x', 'y'), data=NumpyIndexingAdapter(
                                            NumpyIndexingAdapter(self.d)))

    def test_LazilyIndexedArray(self):
        v = Variable(dims=('x', 'y'), data=LazilyIndexedArray(self.d))
        self.check_orthogonal_indexing(v)
        with raises_regex(NotImplementedError, 'Vectorized indexing for '):
            self.check_vectorized_indexing(v)
        # doubly wrapping
        v = Variable(dims=('x', 'y'),
                     data=LazilyIndexedArray(LazilyIndexedArray(self.d)))
        self.check_orthogonal_indexing(v)
        # hierarchical wrapping
        v = Variable(dims=('x', 'y'),
                     data=LazilyIndexedArray(NumpyIndexingAdapter(self.d)))
        self.check_orthogonal_indexing(v)

    def test_CopyOnWriteArray(self):
        v = Variable(dims=('x', 'y'), data=CopyOnWriteArray(self.d))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)
        # doubly wrapping
        v = Variable(dims=('x', 'y'),
                     data=CopyOnWriteArray(LazilyIndexedArray(self.d)))
        self.check_orthogonal_indexing(v)
        with raises_regex(NotImplementedError, 'Vectorized indexing for '):
            self.check_vectorized_indexing(v)

    def test_MemoryCachedArray(self):
        v = Variable(dims=('x', 'y'), data=MemoryCachedArray(self.d))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)
        # doubly wrapping
        v = Variable(dims=('x', 'y'),
                     data=CopyOnWriteArray(MemoryCachedArray(self.d)))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)

    @requires_dask
    def test_DaskIndexingAdapter(self):
        import dask.array as da
        da = da.asarray(self.d)
        v = Variable(dims=('x', 'y'), data=DaskIndexingAdapter(da))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)
        # doubly wrapping
        v = Variable(dims=('x', 'y'),
                     data=CopyOnWriteArray(DaskIndexingAdapter(da)))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)
