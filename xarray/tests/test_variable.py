import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
import pytz

from xarray import Coordinate, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
    BasicIndexer,
    CopyOnWriteArray,
    DaskIndexingAdapter,
    LazilyOuterIndexedArray,
    MemoryCachedArray,
    NumpyIndexingAdapter,
    OuterIndexer,
    PandasIndexAdapter,
    VectorizedIndexer,
)
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.tests import requires_bottleneck

from . import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_identical,
    raises_regex,
    requires_dask,
    source_ndarray,
)


class VariableSubclassobjects:
    def test_properties(self):
        data = 0.5 * np.arange(10)
        v = self.cls(["time"], data, {"foo": "bar"})
        assert v.dims == ("time",)
        assert_array_equal(v.values, data)
        assert v.dtype == float
        assert v.shape == (10,)
        assert v.size == 10
        assert v.sizes == {"time": 10}
        assert v.nbytes == 80
        assert v.ndim == 1
        assert len(v) == 10
        assert v.attrs == {"foo": "bar"}

    def test_attrs(self):
        v = self.cls(["time"], 0.5 * np.arange(10))
        assert v.attrs == {}
        attrs = {"foo": "bar"}
        v.attrs = attrs
        assert v.attrs == attrs
        assert isinstance(v.attrs, dict)
        v.attrs["foo"] = "baz"
        assert v.attrs["foo"] == "baz"

    def test_getitem_dict(self):
        v = self.cls(["x"], np.random.randn(5))
        actual = v[{"x": 0}]
        expected = v[0]
        assert_identical(expected, actual)

    def test_getitem_1d(self):
        data = np.array([0, 1, 2])
        v = self.cls(["x"], data)

        v_new = v[dict(x=[0, 1])]
        assert v_new.dims == ("x",)
        assert_array_equal(v_new, data[[0, 1]])

        v_new = v[dict(x=slice(None))]
        assert v_new.dims == ("x",)
        assert_array_equal(v_new, data)

        v_new = v[dict(x=Variable("a", [0, 1]))]
        assert v_new.dims == ("a",)
        assert_array_equal(v_new, data[[0, 1]])

        v_new = v[dict(x=1)]
        assert v_new.dims == ()
        assert_array_equal(v_new, data[1])

        # tuple argument
        v_new = v[slice(None)]
        assert v_new.dims == ("x",)
        assert_array_equal(v_new, data)

    def test_getitem_1d_fancy(self):
        v = self.cls(["x"], [0, 1, 2])
        # 1d-variable should be indexable by multi-dimensional Variable
        ind = Variable(("a", "b"), [[0, 1], [0, 1]])
        v_new = v[ind]
        assert v_new.dims == ("a", "b")
        expected = np.array(v._data)[([0, 1], [0, 1]), ...]
        assert_array_equal(v_new, expected)

        # boolean indexing
        ind = Variable(("x",), [True, False, True])
        v_new = v[ind]
        assert_identical(v[[0, 2]], v_new)
        v_new = v[[True, False, True]]
        assert_identical(v[[0, 2]], v_new)

        with raises_regex(IndexError, "Boolean indexer should"):
            ind = Variable(("a",), [True, False, True])
            v[ind]

    def test_getitem_with_mask(self):
        v = self.cls(["x"], [0, 1, 2])
        assert_identical(v._getitem_with_mask(-1), Variable((), np.nan))
        assert_identical(
            v._getitem_with_mask([0, -1, 1]), self.cls(["x"], [0, np.nan, 1])
        )
        assert_identical(v._getitem_with_mask(slice(2)), self.cls(["x"], [0, 1]))
        assert_identical(
            v._getitem_with_mask([0, -1, 1], fill_value=-99),
            self.cls(["x"], [0, -99, 1]),
        )

    def test_getitem_with_mask_size_zero(self):
        v = self.cls(["x"], [])
        assert_identical(v._getitem_with_mask(-1), Variable((), np.nan))
        assert_identical(
            v._getitem_with_mask([-1, -1, -1]),
            self.cls(["x"], [np.nan, np.nan, np.nan]),
        )

    def test_getitem_with_mask_nd_indexer(self):
        v = self.cls(["x"], [0, 1, 2])
        indexer = Variable(("x", "y"), [[0, -1], [-1, 2]])
        assert_identical(v._getitem_with_mask(indexer, fill_value=-1), indexer)

    def _assertIndexedLikeNDArray(self, variable, expected_value0, expected_dtype=None):
        """Given a 1-dimensional variable, verify that the variable is indexed
        like a numpy.ndarray.
        """
        assert variable[0].shape == ()
        assert variable[0].ndim == 0
        assert variable[0].size == 1
        # test identity
        assert variable.equals(variable.copy())
        assert variable.identical(variable.copy())
        # check value is equal for both ndarray and Variable
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "In the future, 'NAT == x'")
            np.testing.assert_equal(variable.values[0], expected_value0)
            np.testing.assert_equal(variable[0].values, expected_value0)
        # check type or dtype is consistent for both ndarray and Variable
        if expected_dtype is None:
            # check output type instead of array dtype
            assert type(variable.values[0]) == type(expected_value0)
            assert type(variable[0].values) == type(expected_value0)
        elif expected_dtype is not False:
            assert variable.values[0].dtype == expected_dtype
            assert variable[0].values.dtype == expected_dtype

    def test_index_0d_int(self):
        for value, dtype in [(0, np.int_), (np.int32(0), np.int32)]:
            x = self.cls(["x"], [value])
            self._assertIndexedLikeNDArray(x, value, dtype)

    def test_index_0d_float(self):
        for value, dtype in [(0.5, np.float_), (np.float32(0.5), np.float32)]:
            x = self.cls(["x"], [value])
            self._assertIndexedLikeNDArray(x, value, dtype)

    def test_index_0d_string(self):
        value = "foo"
        dtype = np.dtype("U3")
        x = self.cls(["x"], [value])
        self._assertIndexedLikeNDArray(x, value, dtype)

    def test_index_0d_datetime(self):
        d = datetime(2000, 1, 1)
        x = self.cls(["x"], [d])
        self._assertIndexedLikeNDArray(x, np.datetime64(d))

        x = self.cls(["x"], [np.datetime64(d)])
        self._assertIndexedLikeNDArray(x, np.datetime64(d), "datetime64[ns]")

        x = self.cls(["x"], pd.DatetimeIndex([d]))
        self._assertIndexedLikeNDArray(x, np.datetime64(d), "datetime64[ns]")

    def test_index_0d_timedelta64(self):
        td = timedelta(hours=1)

        x = self.cls(["x"], [np.timedelta64(td)])
        self._assertIndexedLikeNDArray(x, np.timedelta64(td), "timedelta64[ns]")

        x = self.cls(["x"], pd.to_timedelta([td]))
        self._assertIndexedLikeNDArray(x, np.timedelta64(td), "timedelta64[ns]")

    def test_index_0d_not_a_time(self):
        d = np.datetime64("NaT", "ns")
        x = self.cls(["x"], [d])
        self._assertIndexedLikeNDArray(x, d)

    def test_index_0d_object(self):
        class HashableItemWrapper:
            def __init__(self, item):
                self.item = item

            def __eq__(self, other):
                return self.item == other.item

            def __hash__(self):
                return hash(self.item)

            def __repr__(self):
                return "{}(item={!r})".format(type(self).__name__, self.item)

        item = HashableItemWrapper((1, 2, 3))
        x = self.cls("x", [item])
        self._assertIndexedLikeNDArray(x, item, expected_dtype=False)

    def test_0d_object_array_with_list(self):
        listarray = np.empty((1,), dtype=object)
        listarray[0] = [1, 2, 3]
        x = self.cls("x", listarray)
        assert_array_equal(x.data, listarray)
        assert_array_equal(x[0].data, listarray.squeeze())
        assert_array_equal(x.squeeze().data, listarray.squeeze())

    def test_index_and_concat_datetime(self):
        # regression test for #125
        date_range = pd.date_range("2011-09-01", periods=10)
        for dates in [date_range, date_range.values, date_range.to_pydatetime()]:
            expected = self.cls("t", dates)
            for times in [
                [expected[i] for i in range(10)],
                [expected[i : (i + 1)] for i in range(10)],
                [expected[[i]] for i in range(10)],
            ]:
                actual = Variable.concat(times, "t")
                assert expected.dtype == actual.dtype
                assert_array_equal(expected, actual)

    def test_0d_time_data(self):
        # regression test for #105
        x = self.cls("time", pd.date_range("2000-01-01", periods=5))
        expected = np.datetime64("2000-01-01", "ns")
        assert x[0].values == expected

    def test_datetime64_conversion(self):
        times = pd.date_range("2000-01-01", periods=3)
        for values, preserve_source in [
            (times, True),
            (times.values, True),
            (times.values.astype("datetime64[s]"), False),
            (times.to_pydatetime(), False),
        ]:
            v = self.cls(["t"], values)
            assert v.dtype == np.dtype("datetime64[ns]")
            assert_array_equal(v.values, times.values)
            assert v.values.dtype == np.dtype("datetime64[ns]")
            same_source = source_ndarray(v.values) is source_ndarray(values)
            assert preserve_source == same_source

    def test_timedelta64_conversion(self):
        times = pd.timedelta_range(start=0, periods=3)
        for values, preserve_source in [
            (times, True),
            (times.values, True),
            (times.values.astype("timedelta64[s]"), False),
            (times.to_pytimedelta(), False),
        ]:
            v = self.cls(["t"], values)
            assert v.dtype == np.dtype("timedelta64[ns]")
            assert_array_equal(v.values, times.values)
            assert v.values.dtype == np.dtype("timedelta64[ns]")
            same_source = source_ndarray(v.values) is source_ndarray(values)
            assert preserve_source == same_source

    def test_object_conversion(self):
        data = np.arange(5).astype(str).astype(object)
        actual = self.cls("x", data)
        assert actual.dtype == data.dtype

    def test_pandas_data(self):
        v = self.cls(["x"], pd.Series([0, 1, 2], index=[3, 2, 1]))
        assert_identical(v, v[[0, 1, 2]])
        v = self.cls(["x"], pd.Index([0, 1, 2]))
        assert v[0].values == v.values[0]

    def test_pandas_period_index(self):
        v = self.cls(["x"], pd.period_range(start="2000", periods=20, freq="B"))
        v = v.load()  # for dask-based Variable
        assert v[0] == pd.Period("2000", freq="B")
        assert "Period('2000-01-03', 'B')" in repr(v)

    def test_1d_math(self):
        x = 1.0 * np.arange(5)
        y = np.ones(5)

        # should we need `.to_base_variable()`?
        # probably a break that `+v` changes type?
        v = self.cls(["x"], x)
        base_v = v.to_base_variable()
        # unary ops
        assert_identical(base_v, +v)
        assert_identical(base_v, abs(v))
        assert_array_equal((-v).values, -x)
        # binary ops with numbers
        assert_identical(base_v, v + 0)
        assert_identical(base_v, 0 + v)
        assert_identical(base_v, v * 1)
        # binary ops with numpy arrays
        assert_array_equal((v * x).values, x ** 2)
        assert_array_equal((x * v).values, x ** 2)
        assert_array_equal(v - y, v - 1)
        assert_array_equal(y - v, 1 - v)
        # verify attributes are dropped
        v2 = self.cls(["x"], x, {"units": "meters"})
        assert_identical(base_v, +v2)
        # binary ops with all variables
        assert_array_equal(v + v, 2 * v)
        w = self.cls(["x"], y, {"foo": "bar"})
        assert_identical(v + w, self.cls(["x"], x + y).to_base_variable())
        assert_array_equal((v * w).values, x * y)

        # something complicated
        assert_array_equal((v ** 2 * w - 1 + x).values, x ** 2 * y - 1 + x)
        # make sure dtype is preserved (for Index objects)
        assert float == (+v).dtype
        assert float == (+v).values.dtype
        assert float == (0 + v).dtype
        assert float == (0 + v).values.dtype
        # check types of returned data
        assert isinstance(+v, Variable)
        assert not isinstance(+v, IndexVariable)
        assert isinstance(0 + v, Variable)
        assert not isinstance(0 + v, IndexVariable)

    def test_1d_reduce(self):
        x = np.arange(5)
        v = self.cls(["x"], x)
        actual = v.sum()
        expected = Variable((), 10)
        assert_identical(expected, actual)
        assert type(actual) is Variable

    def test_array_interface(self):
        x = np.arange(5)
        v = self.cls(["x"], x)
        assert_array_equal(np.asarray(v), x)
        # test patched in methods
        assert_array_equal(v.astype(float), x.astype(float))
        # think this is a break, that argsort changes the type
        assert_identical(v.argsort(), v.to_base_variable())
        assert_identical(v.clip(2, 3), self.cls("x", x.clip(2, 3)).to_base_variable())
        # test ufuncs
        assert_identical(np.sin(v), self.cls(["x"], np.sin(x)).to_base_variable())
        assert isinstance(np.sin(v), Variable)
        assert not isinstance(np.sin(v), IndexVariable)

    def example_1d_objects(self):
        for data in [
            range(3),
            0.5 * np.arange(3),
            0.5 * np.arange(3, dtype=np.float32),
            pd.date_range("2000-01-01", periods=3),
            np.array(["a", "b", "c"], dtype=object),
        ]:
            yield (self.cls("x", data), data)

    def test___array__(self):
        for v, data in self.example_1d_objects():
            assert_array_equal(v.values, np.asarray(data))
            assert_array_equal(np.asarray(v), np.asarray(data))
            assert v[0].values == np.asarray(data)[0]
            assert np.asarray(v[0]) == np.asarray(data)[0]

    def test_equals_all_dtypes(self):
        for v, _ in self.example_1d_objects():
            v2 = v.copy()
            assert v.equals(v2)
            assert v.identical(v2)
            assert v.no_conflicts(v2)
            assert v[0].equals(v2[0])
            assert v[0].identical(v2[0])
            assert v[0].no_conflicts(v2[0])
            assert v[:2].equals(v2[:2])
            assert v[:2].identical(v2[:2])
            assert v[:2].no_conflicts(v2[:2])

    def test_eq_all_dtypes(self):
        # ensure that we don't choke on comparisons for which numpy returns
        # scalars
        expected = Variable("x", 3 * [False])
        for v, _ in self.example_1d_objects():
            actual = "z" == v
            assert_identical(expected, actual)
            actual = ~("z" != v)
            assert_identical(expected, actual)

    def test_encoding_preserved(self):
        expected = self.cls("x", range(3), {"foo": 1}, {"bar": 2})
        for actual in [
            expected.T,
            expected[...],
            expected.squeeze(),
            expected.isel(x=slice(None)),
            expected.set_dims({"x": 3}),
            expected.copy(deep=True),
            expected.copy(deep=False),
        ]:

            assert_identical(expected.to_base_variable(), actual.to_base_variable())
            assert expected.encoding == actual.encoding

    def test_concat(self):
        x = np.arange(5)
        y = np.arange(5, 10)
        v = self.cls(["a"], x)
        w = self.cls(["a"], y)
        assert_identical(
            Variable(["b", "a"], np.array([x, y])), Variable.concat([v, w], "b")
        )
        assert_identical(
            Variable(["b", "a"], np.array([x, y])), Variable.concat((v, w), "b")
        )
        assert_identical(
            Variable(["b", "a"], np.array([x, y])), Variable.concat((v, w), "b")
        )
        with raises_regex(ValueError, "inconsistent dimensions"):
            Variable.concat([v, Variable(["c"], y)], "b")
        # test indexers
        actual = Variable.concat(
            [v, w], positions=[np.arange(0, 10, 2), np.arange(1, 10, 2)], dim="a"
        )
        expected = Variable("a", np.array([x, y]).ravel(order="F"))
        assert_identical(expected, actual)
        # test concatenating along a dimension
        v = Variable(["time", "x"], np.random.random((10, 8)))
        assert_identical(v, Variable.concat([v[:5], v[5:]], "time"))
        assert_identical(v, Variable.concat([v[:5], v[5:6], v[6:]], "time"))
        assert_identical(v, Variable.concat([v[:1], v[1:]], "time"))
        # test dimension order
        assert_identical(v, Variable.concat([v[:, :5], v[:, 5:]], "x"))
        with raises_regex(ValueError, "all input arrays must have"):
            Variable.concat([v[:, 0], v[:, 1:]], "x")

    def test_concat_attrs(self):
        # different or conflicting attributes should be removed
        v = self.cls("a", np.arange(5), {"foo": "bar"})
        w = self.cls("a", np.ones(5))
        expected = self.cls(
            "a", np.concatenate([np.arange(5), np.ones(5)])
        ).to_base_variable()
        assert_identical(expected, Variable.concat([v, w], "a"))
        w.attrs["foo"] = 2
        assert_identical(expected, Variable.concat([v, w], "a"))
        w.attrs["foo"] = "bar"
        expected.attrs["foo"] = "bar"
        assert_identical(expected, Variable.concat([v, w], "a"))

    def test_concat_fixed_len_str(self):
        # regression test for #217
        for kind in ["S", "U"]:
            x = self.cls("animal", np.array(["horse"], dtype=kind))
            y = self.cls("animal", np.array(["aardvark"], dtype=kind))
            actual = Variable.concat([x, y], "animal")
            expected = Variable("animal", np.array(["horse", "aardvark"], dtype=kind))
            assert_equal(expected, actual)

    def test_concat_number_strings(self):
        # regression test for #305
        a = self.cls("x", ["0", "1", "2"])
        b = self.cls("x", ["3", "4"])
        actual = Variable.concat([a, b], dim="x")
        expected = Variable("x", np.arange(5).astype(str))
        assert_identical(expected, actual)
        assert actual.dtype.kind == expected.dtype.kind

    def test_concat_mixed_dtypes(self):
        a = self.cls("x", [0, 1])
        b = self.cls("x", ["two"])
        actual = Variable.concat([a, b], dim="x")
        expected = Variable("x", np.array([0, 1, "two"], dtype=object))
        assert_identical(expected, actual)
        assert actual.dtype == object

    @pytest.mark.parametrize("deep", [True, False])
    @pytest.mark.parametrize("astype", [float, int, str])
    def test_copy(self, deep, astype):
        v = self.cls("x", (0.5 * np.arange(10)).astype(astype), {"foo": "bar"})
        w = v.copy(deep=deep)
        assert type(v) is type(w)
        assert_identical(v, w)
        assert v.dtype == w.dtype
        if self.cls is Variable:
            if deep:
                assert source_ndarray(v.values) is not source_ndarray(w.values)
            else:
                assert source_ndarray(v.values) is source_ndarray(w.values)
        assert_identical(v, copy(v))

    def test_copy_index(self):
        midx = pd.MultiIndex.from_product(
            [["a", "b"], [1, 2], [-1, -2]], names=("one", "two", "three")
        )
        v = self.cls("x", midx)
        for deep in [True, False]:
            w = v.copy(deep=deep)
            assert isinstance(w._data, PandasIndexAdapter)
            assert isinstance(w.to_index(), pd.MultiIndex)
            assert_array_equal(v._data.array, w._data.array)

    def test_copy_with_data(self):
        orig = Variable(("x", "y"), [[1.5, 2.0], [3.1, 4.3]], {"foo": "bar"})
        new_data = np.array([[2.5, 5.0], [7.1, 43]])
        actual = orig.copy(data=new_data)
        expected = orig.copy()
        expected.data = new_data
        assert_identical(expected, actual)

    def test_copy_with_data_errors(self):
        orig = Variable(("x", "y"), [[1.5, 2.0], [3.1, 4.3]], {"foo": "bar"})
        new_data = [2.5, 5.0]
        with raises_regex(ValueError, "must match shape of object"):
            orig.copy(data=new_data)

    def test_copy_index_with_data(self):
        orig = IndexVariable("x", np.arange(5))
        new_data = np.arange(5, 10)
        actual = orig.copy(data=new_data)
        expected = orig.copy()
        expected.data = new_data
        assert_identical(expected, actual)

    def test_copy_index_with_data_errors(self):
        orig = IndexVariable("x", np.arange(5))
        new_data = np.arange(5, 20)
        with raises_regex(ValueError, "must match shape of object"):
            orig.copy(data=new_data)

    def test_real_and_imag(self):
        v = self.cls("x", np.arange(3) - 1j * np.arange(3), {"foo": "bar"})
        expected_re = self.cls("x", np.arange(3), {"foo": "bar"})
        assert_identical(v.real, expected_re)

        expected_im = self.cls("x", -np.arange(3), {"foo": "bar"})
        assert_identical(v.imag, expected_im)

        expected_abs = self.cls("x", np.sqrt(2 * np.arange(3) ** 2)).to_base_variable()
        assert_allclose(abs(v), expected_abs)

    def test_aggregate_complex(self):
        # should skip NaNs
        v = self.cls("x", [1, 2j, np.nan])
        expected = Variable((), 0.5 + 1j)
        assert_allclose(v.mean(), expected)

    def test_pandas_cateogrical_dtype(self):
        data = pd.Categorical(np.arange(10, dtype="int64"))
        v = self.cls("x", data)
        print(v)  # should not error
        assert v.dtype == "int64"

    def test_pandas_datetime64_with_tz(self):
        data = pd.date_range(
            start="2000-01-01",
            tz=pytz.timezone("America/New_York"),
            periods=10,
            freq="1h",
        )
        v = self.cls("x", data)
        print(v)  # should not error
        if "America/New_York" in str(data.dtype):
            # pandas is new enough that it has datetime64 with timezone dtype
            assert v.dtype == "object"

    def test_multiindex(self):
        idx = pd.MultiIndex.from_product([list("abc"), [0, 1]])
        v = self.cls("x", idx)
        assert_identical(Variable((), ("a", 0)), v[0])
        assert_identical(v, v[:])

    def test_load(self):
        array = self.cls("x", np.arange(5))
        orig_data = array._data
        copied = array.copy(deep=True)
        if array.chunks is None:
            array.load()
            assert type(array._data) is type(orig_data)
            assert type(copied._data) is type(orig_data)
            assert_identical(array, copied)

    def test_getitem_advanced(self):
        v = self.cls(["x", "y"], [[0, 1, 2], [3, 4, 5]])
        v_data = v.compute().data

        # orthogonal indexing
        v_new = v[([0, 1], [1, 0])]
        assert v_new.dims == ("x", "y")
        assert_array_equal(v_new, v_data[[0, 1]][:, [1, 0]])

        v_new = v[[0, 1]]
        assert v_new.dims == ("x", "y")
        assert_array_equal(v_new, v_data[[0, 1]])

        # with mixed arguments
        ind = Variable(["a"], [0, 1])
        v_new = v[dict(x=[0, 1], y=ind)]
        assert v_new.dims == ("x", "a")
        assert_array_equal(v_new, v_data[[0, 1]][:, [0, 1]])

        # boolean indexing
        v_new = v[dict(x=[True, False], y=[False, True, False])]
        assert v_new.dims == ("x", "y")
        assert_array_equal(v_new, v_data[0][1])

        # with scalar variable
        ind = Variable((), 2)
        v_new = v[dict(y=ind)]
        expected = v[dict(y=2)]
        assert_array_equal(v_new, expected)

        # with boolean variable with wrong shape
        ind = np.array([True, False])
        with raises_regex(IndexError, "Boolean array size 2 is "):
            v[Variable(("a", "b"), [[0, 1]]), ind]

        # boolean indexing with different dimension
        ind = Variable(["a"], [True, False, False])
        with raises_regex(IndexError, "Boolean indexer should be"):
            v[dict(y=ind)]

    def test_getitem_uint_1d(self):
        # regression test for #1405
        v = self.cls(["x"], [0, 1, 2])
        v_data = v.compute().data

        v_new = v[np.array([0])]
        assert_array_equal(v_new, v_data[0])
        v_new = v[np.array([0], dtype="uint64")]
        assert_array_equal(v_new, v_data[0])

    def test_getitem_uint(self):
        # regression test for #1405
        v = self.cls(["x", "y"], [[0, 1, 2], [3, 4, 5]])
        v_data = v.compute().data

        v_new = v[np.array([0])]
        assert_array_equal(v_new, v_data[[0], :])
        v_new = v[np.array([0], dtype="uint64")]
        assert_array_equal(v_new, v_data[[0], :])

        v_new = v[np.uint64(0)]
        assert_array_equal(v_new, v_data[0, :])

    def test_getitem_0d_array(self):
        # make sure 0d-np.array can be used as an indexer
        v = self.cls(["x"], [0, 1, 2])
        v_data = v.compute().data

        v_new = v[np.array([0])[0]]
        assert_array_equal(v_new, v_data[0])

        v_new = v[np.array(0)]
        assert_array_equal(v_new, v_data[0])

        v_new = v[Variable((), np.array(0))]
        assert_array_equal(v_new, v_data[0])

    def test_getitem_fancy(self):
        v = self.cls(["x", "y"], [[0, 1, 2], [3, 4, 5]])
        v_data = v.compute().data

        ind = Variable(["a", "b"], [[0, 1, 1], [1, 1, 0]])
        v_new = v[ind]
        assert v_new.dims == ("a", "b", "y")
        assert_array_equal(v_new, v_data[[[0, 1, 1], [1, 1, 0]], :])

        # It would be ok if indexed with the multi-dimensional array including
        # the same name
        ind = Variable(["x", "b"], [[0, 1, 1], [1, 1, 0]])
        v_new = v[ind]
        assert v_new.dims == ("x", "b", "y")
        assert_array_equal(v_new, v_data[[[0, 1, 1], [1, 1, 0]], :])

        ind = Variable(["a", "b"], [[0, 1, 2], [2, 1, 0]])
        v_new = v[dict(y=ind)]
        assert v_new.dims == ("x", "a", "b")
        assert_array_equal(v_new, v_data[:, ([0, 1, 2], [2, 1, 0])])

        ind = Variable(["a", "b"], [[0, 0], [1, 1]])
        v_new = v[dict(x=[1, 0], y=ind)]
        assert v_new.dims == ("x", "a", "b")
        assert_array_equal(v_new, v_data[[1, 0]][:, ind])

        # along diagonal
        ind = Variable(["a"], [0, 1])
        v_new = v[ind, ind]
        assert v_new.dims == ("a",)
        assert_array_equal(v_new, v_data[[0, 1], [0, 1]])

        # with integer
        ind = Variable(["a", "b"], [[0, 0], [1, 1]])
        v_new = v[dict(x=0, y=ind)]
        assert v_new.dims == ("a", "b")
        assert_array_equal(v_new[0], v_data[0][[0, 0]])
        assert_array_equal(v_new[1], v_data[0][[1, 1]])

        # with slice
        ind = Variable(["a", "b"], [[0, 0], [1, 1]])
        v_new = v[dict(x=slice(None), y=ind)]
        assert v_new.dims == ("x", "a", "b")
        assert_array_equal(v_new, v_data[:, [[0, 0], [1, 1]]])

        ind = Variable(["a", "b"], [[0, 0], [1, 1]])
        v_new = v[dict(x=ind, y=slice(None))]
        assert v_new.dims == ("a", "b", "y")
        assert_array_equal(v_new, v_data[[[0, 0], [1, 1]], :])

        ind = Variable(["a", "b"], [[0, 0], [1, 1]])
        v_new = v[dict(x=ind, y=slice(None, 1))]
        assert v_new.dims == ("a", "b", "y")
        assert_array_equal(v_new, v_data[[[0, 0], [1, 1]], slice(None, 1)])

        # slice matches explicit dimension
        ind = Variable(["y"], [0, 1])
        v_new = v[ind, :2]
        assert v_new.dims == ("y",)
        assert_array_equal(v_new, v_data[[0, 1], [0, 1]])

        # with multiple slices
        v = self.cls(["x", "y", "z"], [[[1, 2, 3], [4, 5, 6]]])
        ind = Variable(["a", "b"], [[0]])
        v_new = v[ind, :, :]
        expected = Variable(["a", "b", "y", "z"], v.data[np.newaxis, ...])
        assert_identical(v_new, expected)

        v = Variable(["w", "x", "y", "z"], [[[[1, 2, 3], [4, 5, 6]]]])
        ind = Variable(["y"], [0])
        v_new = v[ind, :, 1:2, 2]
        expected = Variable(["y", "x"], [[6]])
        assert_identical(v_new, expected)

        # slice and vector mixed indexing resulting in the same dimension
        v = Variable(["x", "y", "z"], np.arange(60).reshape(3, 4, 5))
        ind = Variable(["x"], [0, 1, 2])
        v_new = v[:, ind]
        expected = Variable(("x", "z"), np.zeros((3, 5)))
        expected[0] = v.data[0, 0]
        expected[1] = v.data[1, 1]
        expected[2] = v.data[2, 2]
        assert_identical(v_new, expected)

        v_new = v[:, ind.data]
        assert v_new.shape == (3, 3, 5)

    def test_getitem_error(self):
        v = self.cls(["x", "y"], [[0, 1, 2], [3, 4, 5]])

        with raises_regex(IndexError, "labeled multi-"):
            v[[[0, 1], [1, 2]]]

        ind_x = Variable(["a"], [0, 1, 1])
        ind_y = Variable(["a"], [0, 1])
        with raises_regex(IndexError, "Dimensions of indexers "):
            v[ind_x, ind_y]

        ind = Variable(["a", "b"], [[True, False], [False, True]])
        with raises_regex(IndexError, "2-dimensional boolean"):
            v[dict(x=ind)]

        v = Variable(["x", "y", "z"], np.arange(60).reshape(3, 4, 5))
        ind = Variable(["x"], [0, 1])
        with raises_regex(IndexError, "Dimensions of indexers mis"):
            v[:, ind]

    def test_pad(self):
        data = np.arange(4 * 3 * 2).reshape(4, 3, 2)
        v = self.cls(["x", "y", "z"], data)

        xr_args = [{"x": (2, 1)}, {"y": (0, 3)}, {"x": (3, 1), "z": (2, 0)}]
        np_args = [
            ((2, 1), (0, 0), (0, 0)),
            ((0, 0), (0, 3), (0, 0)),
            ((3, 1), (0, 0), (2, 0)),
        ]
        for xr_arg, np_arg in zip(xr_args, np_args):
            actual = v.pad_with_fill_value(**xr_arg)
            expected = np.pad(
                np.array(v.data.astype(float)),
                np_arg,
                mode="constant",
                constant_values=np.nan,
            )
            assert_array_equal(actual, expected)
            assert isinstance(actual._data, type(v._data))

        # for the boolean array, we pad False
        data = np.full_like(data, False, dtype=bool).reshape(4, 3, 2)
        v = self.cls(["x", "y", "z"], data)
        for xr_arg, np_arg in zip(xr_args, np_args):
            actual = v.pad_with_fill_value(fill_value=False, **xr_arg)
            expected = np.pad(
                np.array(v.data), np_arg, mode="constant", constant_values=False
            )
            assert_array_equal(actual, expected)

    def test_rolling_window(self):
        # Just a working test. See test_nputils for the algorithm validation
        v = self.cls(["x", "y", "z"], np.arange(40 * 30 * 2).reshape(40, 30, 2))
        for (d, w) in [("x", 3), ("y", 5)]:
            v_rolling = v.rolling_window(d, w, d + "_window")
            assert v_rolling.dims == ("x", "y", "z", d + "_window")
            assert v_rolling.shape == v.shape + (w,)

            v_rolling = v.rolling_window(d, w, d + "_window", center=True)
            assert v_rolling.dims == ("x", "y", "z", d + "_window")
            assert v_rolling.shape == v.shape + (w,)

            # dask and numpy result should be the same
            v_loaded = v.load().rolling_window(d, w, d + "_window", center=True)
            assert_array_equal(v_rolling, v_loaded)

            # numpy backend should not be over-written
            if isinstance(v._data, np.ndarray):
                with pytest.raises(ValueError):
                    v_loaded[0] = 1.0


class TestVariable(VariableSubclassobjects):
    cls = staticmethod(Variable)

    @pytest.fixture(autouse=True)
    def setup(self):
        self.d = np.random.random((10, 3)).astype(np.float64)

    def test_data_and_values(self):
        v = Variable(["time", "x"], self.d)
        assert_array_equal(v.data, self.d)
        assert_array_equal(v.values, self.d)
        assert source_ndarray(v.values) is self.d
        with pytest.raises(ValueError):
            # wrong size
            v.values = np.random.random(5)
        d2 = np.random.random((10, 3))
        v.values = d2
        assert source_ndarray(v.values) is d2
        d3 = np.random.random((10, 3))
        v.data = d3
        assert source_ndarray(v.data) is d3

    def test_numpy_same_methods(self):
        v = Variable([], np.float32(0.0))
        assert v.item() == 0
        assert type(v.item()) is float

        v = IndexVariable("x", np.arange(5))
        assert 2 == v.searchsorted(2)

    def test_datetime64_conversion_scalar(self):
        expected = np.datetime64("2000-01-01", "ns")
        for values in [
            np.datetime64("2000-01-01"),
            pd.Timestamp("2000-01-01T00"),
            datetime(2000, 1, 1),
        ]:
            v = Variable([], values)
            assert v.dtype == np.dtype("datetime64[ns]")
            assert v.values == expected
            assert v.values.dtype == np.dtype("datetime64[ns]")

    def test_timedelta64_conversion_scalar(self):
        expected = np.timedelta64(24 * 60 * 60 * 10 ** 9, "ns")
        for values in [
            np.timedelta64(1, "D"),
            pd.Timedelta("1 day"),
            timedelta(days=1),
        ]:
            v = Variable([], values)
            assert v.dtype == np.dtype("timedelta64[ns]")
            assert v.values == expected
            assert v.values.dtype == np.dtype("timedelta64[ns]")

    def test_0d_str(self):
        v = Variable([], "foo")
        assert v.dtype == np.dtype("U3")
        assert v.values == "foo"

        v = Variable([], np.string_("foo"))
        assert v.dtype == np.dtype("S3")
        assert v.values == bytes("foo", "ascii")

    def test_0d_datetime(self):
        v = Variable([], pd.Timestamp("2000-01-01"))
        assert v.dtype == np.dtype("datetime64[ns]")
        assert v.values == np.datetime64("2000-01-01", "ns")

    def test_0d_timedelta(self):
        for td in [pd.to_timedelta("1s"), np.timedelta64(1, "s")]:
            v = Variable([], td)
            assert v.dtype == np.dtype("timedelta64[ns]")
            assert v.values == np.timedelta64(10 ** 9, "ns")

    def test_equals_and_identical(self):
        d = np.random.rand(10, 3)
        d[0, 0] = np.nan
        v1 = Variable(("dim1", "dim2"), data=d, attrs={"att1": 3, "att2": [1, 2, 3]})
        v2 = Variable(("dim1", "dim2"), data=d, attrs={"att1": 3, "att2": [1, 2, 3]})
        assert v1.equals(v2)
        assert v1.identical(v2)

        v3 = Variable(("dim1", "dim3"), data=d)
        assert not v1.equals(v3)

        v4 = Variable(("dim1", "dim2"), data=d)
        assert v1.equals(v4)
        assert not v1.identical(v4)

        v5 = deepcopy(v1)
        v5.values[:] = np.random.rand(10, 3)
        assert not v1.equals(v5)

        assert not v1.equals(None)
        assert not v1.equals(d)

        assert not v1.identical(None)
        assert not v1.identical(d)

    def test_broadcast_equals(self):
        v1 = Variable((), np.nan)
        v2 = Variable(("x"), [np.nan, np.nan])
        assert v1.broadcast_equals(v2)
        assert not v1.equals(v2)
        assert not v1.identical(v2)

        v3 = Variable(("x"), [np.nan])
        assert v1.broadcast_equals(v3)
        assert not v1.equals(v3)
        assert not v1.identical(v3)

        assert not v1.broadcast_equals(None)

        v4 = Variable(("x"), [np.nan] * 3)
        assert not v2.broadcast_equals(v4)

    def test_no_conflicts(self):
        v1 = Variable(("x"), [1, 2, np.nan, np.nan])
        v2 = Variable(("x"), [np.nan, 2, 3, np.nan])
        assert v1.no_conflicts(v2)
        assert not v1.equals(v2)
        assert not v1.broadcast_equals(v2)
        assert not v1.identical(v2)

        assert not v1.no_conflicts(None)

        v3 = Variable(("y"), [np.nan, 2, 3, np.nan])
        assert not v3.no_conflicts(v1)

        d = np.array([1, 2, np.nan, np.nan])
        assert not v1.no_conflicts(d)
        assert not v2.no_conflicts(d)

        v4 = Variable(("w", "x"), [d])
        assert v1.no_conflicts(v4)

    def test_as_variable(self):
        data = np.arange(10)
        expected = Variable("x", data)
        expected_extra = Variable(
            "x", data, attrs={"myattr": "val"}, encoding={"scale_factor": 1}
        )

        assert_identical(expected, as_variable(expected))

        ds = Dataset({"x": expected})
        var = as_variable(ds["x"]).to_base_variable()
        assert_identical(expected, var)
        assert not isinstance(ds["x"], Variable)
        assert isinstance(as_variable(ds["x"]), Variable)

        xarray_tuple = (
            expected_extra.dims,
            expected_extra.values,
            expected_extra.attrs,
            expected_extra.encoding,
        )
        assert_identical(expected_extra, as_variable(xarray_tuple))

        with raises_regex(TypeError, "tuple of form"):
            as_variable(tuple(data))
        with raises_regex(ValueError, "tuple of form"):  # GH1016
            as_variable(("five", "six", "seven"))
        with raises_regex(TypeError, "without an explicit list of dimensions"):
            as_variable(data)

        actual = as_variable(data, name="x")
        assert_identical(expected.to_index_variable(), actual)

        actual = as_variable(0)
        expected = Variable([], 0)
        assert_identical(expected, actual)

        data = np.arange(9).reshape((3, 3))
        expected = Variable(("x", "y"), data)
        with raises_regex(ValueError, "without explicit dimension names"):
            as_variable(data, name="x")
        with raises_regex(ValueError, "has more than 1-dimension"):
            as_variable(expected, name="x")

        # test datetime, timedelta conversion
        dt = np.array([datetime(1999, 1, 1) + timedelta(days=x) for x in range(10)])
        assert as_variable(dt, "time").dtype.kind == "M"
        td = np.array([timedelta(days=x) for x in range(10)])
        assert as_variable(td, "time").dtype.kind == "m"

    def test_repr(self):
        v = Variable(["time", "x"], [[1, 2, 3], [4, 5, 6]], {"foo": "bar"})
        expected = dedent(
            """
        <xarray.Variable (time: 2, x: 3)>
        array([[1, 2, 3],
               [4, 5, 6]])
        Attributes:
            foo:      bar
        """
        ).strip()
        assert expected == repr(v)

    def test_repr_lazy_data(self):
        v = Variable("x", LazilyOuterIndexedArray(np.arange(2e5)))
        assert "200000 values with dtype" in repr(v)
        assert isinstance(v._data, LazilyOuterIndexedArray)

    def test_detect_indexer_type(self):
        """ Tests indexer type was correctly detected. """
        data = np.random.random((10, 11))
        v = Variable(["x", "y"], data)

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

        vind = Variable(("a",), [0, 1])
        _, ind, _ = v._broadcast_indexes((vind, slice(0, 8, 2)))
        assert type(ind) == indexing.OuterIndexer

        vind = Variable(("y",), [0, 1])
        _, ind, _ = v._broadcast_indexes((vind, 3))
        assert type(ind) == indexing.OuterIndexer

        vind = Variable(("a",), [0, 1])
        _, ind, _ = v._broadcast_indexes((vind, vind))
        assert type(ind) == indexing.VectorizedIndexer

        vind = Variable(("a", "b"), [[0, 2], [1, 3]])
        _, ind, _ = v._broadcast_indexes((vind, 3))
        assert type(ind) == indexing.VectorizedIndexer

    def test_indexer_type(self):
        # GH:issue:1688. Wrong indexer type induces NotImplementedError
        data = np.random.random((10, 11))
        v = Variable(["x", "y"], data)

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
        assert_indexer_type((Variable(("x"), [0, 1]), 1), OuterIndexer)
        assert_indexer_type((Variable(("x"), [0, 1]), slice(None, None)), OuterIndexer)
        assert_indexer_type(
            (Variable(("x"), [0, 1]), Variable(("y"), [0, 1])), OuterIndexer
        )

        # should return VectorizedIndexer
        assert_indexer_type((Variable(("y"), [0, 1]), [0, 1]), VectorizedIndexer)
        assert_indexer_type(
            (Variable(("z"), [0, 1]), Variable(("z"), [0, 1])), VectorizedIndexer
        )
        assert_indexer_type(
            (
                Variable(("a", "b"), [[0, 1], [1, 2]]),
                Variable(("a", "b"), [[0, 1], [1, 2]]),
            ),
            VectorizedIndexer,
        )

    def test_items(self):
        data = np.random.random((10, 11))
        v = Variable(["x", "y"], data)
        # test slicing
        assert_identical(v, v[:])
        assert_identical(v, v[...])
        assert_identical(Variable(["y"], data[0]), v[0])
        assert_identical(Variable(["x"], data[:, 0]), v[:, 0])
        assert_identical(Variable(["x", "y"], data[:3, :2]), v[:3, :2])
        # test array indexing
        x = Variable(["x"], np.arange(10))
        y = Variable(["y"], np.arange(11))
        assert_identical(v, v[x.values])
        assert_identical(v, v[x])
        assert_identical(v[:3], v[x < 3])
        assert_identical(v[:, 3:], v[:, y >= 3])
        assert_identical(v[:3, 3:], v[x < 3, y >= 3])
        assert_identical(v[:3, :2], v[x[:3], y[:2]])
        assert_identical(v[:3, :2], v[range(3), range(2)])
        # test iteration
        for n, item in enumerate(v):
            assert_identical(Variable(["y"], data[n]), item)
        with raises_regex(TypeError, "iteration over a 0-d"):
            iter(Variable([], 0))
        # test setting
        v.values[:] = 0
        assert np.all(v.values == 0)
        # test orthogonal setting
        v[range(10), range(11)] = 1
        assert_array_equal(v.values, np.ones((10, 11)))

    def test_getitem_basic(self):
        v = self.cls(["x", "y"], [[0, 1, 2], [3, 4, 5]])

        v_new = v[dict(x=0)]
        assert v_new.dims == ("y",)
        assert_array_equal(v_new, v._data[0])

        v_new = v[dict(x=0, y=slice(None))]
        assert v_new.dims == ("y",)
        assert_array_equal(v_new, v._data[0])

        v_new = v[dict(x=0, y=1)]
        assert v_new.dims == ()
        assert_array_equal(v_new, v._data[0, 1])

        v_new = v[dict(y=1)]
        assert v_new.dims == ("x",)
        assert_array_equal(v_new, v._data[:, 1])

        # tuple argument
        v_new = v[(slice(None), 1)]
        assert v_new.dims == ("x",)
        assert_array_equal(v_new, v._data[:, 1])

        # test that we obtain a modifiable view when taking a 0d slice
        v_new = v[0, 0]
        v_new[...] += 99
        assert_array_equal(v_new, v._data[0, 0])

    def test_getitem_with_mask_2d_input(self):
        v = Variable(("x", "y"), [[0, 1, 2], [3, 4, 5]])
        assert_identical(
            v._getitem_with_mask(([-1, 0], [1, -1])),
            Variable(("x", "y"), [[np.nan, np.nan], [1, np.nan]]),
        )
        assert_identical(v._getitem_with_mask((slice(2), [0, 1, 2])), v)

    def test_isel(self):
        v = Variable(["time", "x"], self.d)
        assert_identical(v.isel(time=slice(None)), v)
        assert_identical(v.isel(time=0), v[0])
        assert_identical(v.isel(time=slice(0, 3)), v[:3])
        assert_identical(v.isel(x=0), v[:, 0])
        with raises_regex(ValueError, "do not exist"):
            v.isel(not_a_dim=0)

    def test_index_0d_numpy_string(self):
        # regression test to verify our work around for indexing 0d strings
        v = Variable([], np.string_("asdf"))
        assert_identical(v[()], v)

        v = Variable([], np.unicode_("asdf"))
        assert_identical(v[()], v)

    def test_indexing_0d_unicode(self):
        # regression test for GH568
        actual = Variable(("x"), ["tmax"])[0][()]
        expected = Variable((), "tmax")
        assert_identical(actual, expected)

    @pytest.mark.parametrize("fill_value", [dtypes.NA, 2, 2.0])
    def test_shift(self, fill_value):
        v = Variable("x", [1, 2, 3, 4, 5])

        assert_identical(v, v.shift(x=0))
        assert v is not v.shift(x=0)

        expected = Variable("x", [np.nan, np.nan, 1, 2, 3])
        assert_identical(expected, v.shift(x=2))

        if fill_value == dtypes.NA:
            # if we supply the default, we expect the missing value for a
            # float array
            fill_value_exp = np.nan
        else:
            fill_value_exp = fill_value

        expected = Variable("x", [fill_value_exp, 1, 2, 3, 4])
        assert_identical(expected, v.shift(x=1, fill_value=fill_value))

        expected = Variable("x", [2, 3, 4, 5, fill_value_exp])
        assert_identical(expected, v.shift(x=-1, fill_value=fill_value))

        expected = Variable("x", [fill_value_exp] * 5)
        assert_identical(expected, v.shift(x=5, fill_value=fill_value))
        assert_identical(expected, v.shift(x=6, fill_value=fill_value))

        with raises_regex(ValueError, "dimension"):
            v.shift(z=0)

        v = Variable("x", [1, 2, 3, 4, 5], {"foo": "bar"})
        assert_identical(v, v.shift(x=0))

        expected = Variable("x", [fill_value_exp, 1, 2, 3, 4], {"foo": "bar"})
        assert_identical(expected, v.shift(x=1, fill_value=fill_value))

    def test_shift2d(self):
        v = Variable(("x", "y"), [[1, 2], [3, 4]])
        expected = Variable(("x", "y"), [[np.nan, np.nan], [np.nan, 1]])
        assert_identical(expected, v.shift(x=1, y=1))

    def test_roll(self):
        v = Variable("x", [1, 2, 3, 4, 5])

        assert_identical(v, v.roll(x=0))
        assert v is not v.roll(x=0)

        expected = Variable("x", [5, 1, 2, 3, 4])
        assert_identical(expected, v.roll(x=1))
        assert_identical(expected, v.roll(x=-4))
        assert_identical(expected, v.roll(x=6))

        expected = Variable("x", [4, 5, 1, 2, 3])
        assert_identical(expected, v.roll(x=2))
        assert_identical(expected, v.roll(x=-3))

        with raises_regex(ValueError, "dimension"):
            v.roll(z=0)

    def test_roll_consistency(self):
        v = Variable(("x", "y"), np.random.randn(5, 6))

        for axis, dim in [(0, "x"), (1, "y")]:
            for shift in [-3, 0, 1, 7, 11]:
                expected = np.roll(v.values, shift, axis=axis)
                actual = v.roll(**{dim: shift}).values
                assert_array_equal(expected, actual)

    def test_transpose(self):
        v = Variable(["time", "x"], self.d)
        v2 = Variable(["x", "time"], self.d.T)
        assert_identical(v, v2.transpose())
        assert_identical(v.transpose(), v.T)
        x = np.random.randn(2, 3, 4, 5)
        w = Variable(["a", "b", "c", "d"], x)
        w2 = Variable(["d", "b", "c", "a"], np.einsum("abcd->dbca", x))
        assert w2.shape == (5, 3, 4, 2)
        assert_identical(w2, w.transpose("d", "b", "c", "a"))
        assert_identical(w, w2.transpose("a", "b", "c", "d"))
        w3 = Variable(["b", "c", "d", "a"], np.einsum("abcd->bcda", x))
        assert_identical(w, w3.transpose("a", "b", "c", "d"))

    def test_transpose_0d(self):
        for value in [
            3.5,
            ("a", 1),
            np.datetime64("2000-01-01"),
            np.timedelta64(1, "h"),
            None,
            object(),
        ]:
            variable = Variable([], value)
            actual = variable.transpose()
            assert actual.identical(variable)

    def test_squeeze(self):
        v = Variable(["x", "y"], [[1]])
        assert_identical(Variable([], 1), v.squeeze())
        assert_identical(Variable(["y"], [1]), v.squeeze("x"))
        assert_identical(Variable(["y"], [1]), v.squeeze(["x"]))
        assert_identical(Variable(["x"], [1]), v.squeeze("y"))
        assert_identical(Variable([], 1), v.squeeze(["x", "y"]))

        v = Variable(["x", "y"], [[1, 2]])
        assert_identical(Variable(["y"], [1, 2]), v.squeeze())
        assert_identical(Variable(["y"], [1, 2]), v.squeeze("x"))
        with raises_regex(ValueError, "cannot select a dimension"):
            v.squeeze("y")

    def test_get_axis_num(self):
        v = Variable(["x", "y", "z"], np.random.randn(2, 3, 4))
        assert v.get_axis_num("x") == 0
        assert v.get_axis_num(["x"]) == (0,)
        assert v.get_axis_num(["x", "y"]) == (0, 1)
        assert v.get_axis_num(["z", "y", "x"]) == (2, 1, 0)
        with raises_regex(ValueError, "not found in array dim"):
            v.get_axis_num("foobar")

    def test_set_dims(self):
        v = Variable(["x"], [0, 1])
        actual = v.set_dims(["x", "y"])
        expected = Variable(["x", "y"], [[0], [1]])
        assert_identical(actual, expected)

        actual = v.set_dims(["y", "x"])
        assert_identical(actual, expected.T)

        actual = v.set_dims({"x": 2, "y": 2})
        expected = Variable(["x", "y"], [[0, 0], [1, 1]])
        assert_identical(actual, expected)

        v = Variable(["foo"], [0, 1])
        actual = v.set_dims("foo")
        expected = v
        assert_identical(actual, expected)

        with raises_regex(ValueError, "must be a superset"):
            v.set_dims(["z"])

    def test_set_dims_object_dtype(self):
        v = Variable([], ("a", 1))
        actual = v.set_dims(("x",), (3,))
        exp_values = np.empty((3,), dtype=object)
        for i in range(3):
            exp_values[i] = ("a", 1)
        expected = Variable(["x"], exp_values)
        assert actual.identical(expected)

    def test_stack(self):
        v = Variable(["x", "y"], [[0, 1], [2, 3]], {"foo": "bar"})
        actual = v.stack(z=("x", "y"))
        expected = Variable("z", [0, 1, 2, 3], v.attrs)
        assert_identical(actual, expected)

        actual = v.stack(z=("x",))
        expected = Variable(("y", "z"), v.data.T, v.attrs)
        assert_identical(actual, expected)

        actual = v.stack(z=())
        assert_identical(actual, v)

        actual = v.stack(X=("x",), Y=("y",)).transpose("X", "Y")
        expected = Variable(("X", "Y"), v.data, v.attrs)
        assert_identical(actual, expected)

    def test_stack_errors(self):
        v = Variable(["x", "y"], [[0, 1], [2, 3]], {"foo": "bar"})

        with raises_regex(ValueError, "invalid existing dim"):
            v.stack(z=("x1",))
        with raises_regex(ValueError, "cannot create a new dim"):
            v.stack(x=("x",))

    def test_unstack(self):
        v = Variable("z", [0, 1, 2, 3], {"foo": "bar"})
        actual = v.unstack(z={"x": 2, "y": 2})
        expected = Variable(("x", "y"), [[0, 1], [2, 3]], v.attrs)
        assert_identical(actual, expected)

        actual = v.unstack(z={"x": 4, "y": 1})
        expected = Variable(("x", "y"), [[0], [1], [2], [3]], v.attrs)
        assert_identical(actual, expected)

        actual = v.unstack(z={"x": 4})
        expected = Variable("x", [0, 1, 2, 3], v.attrs)
        assert_identical(actual, expected)

    def test_unstack_errors(self):
        v = Variable("z", [0, 1, 2, 3])
        with raises_regex(ValueError, "invalid existing dim"):
            v.unstack(foo={"x": 4})
        with raises_regex(ValueError, "cannot create a new dim"):
            v.stack(z=("z",))
        with raises_regex(ValueError, "the product of the new dim"):
            v.unstack(z={"x": 5})

    def test_unstack_2d(self):
        v = Variable(["x", "y"], [[0, 1], [2, 3]])
        actual = v.unstack(y={"z": 2})
        expected = Variable(["x", "z"], v.data)
        assert_identical(actual, expected)

        actual = v.unstack(x={"z": 2})
        expected = Variable(["y", "z"], v.data.T)
        assert_identical(actual, expected)

    def test_stack_unstack_consistency(self):
        v = Variable(["x", "y"], [[0, 1], [2, 3]])
        actual = v.stack(z=("x", "y")).unstack(z={"x": 2, "y": 2})
        assert_identical(actual, v)

    def test_broadcasting_math(self):
        x = np.random.randn(2, 3)
        v = Variable(["a", "b"], x)
        # 1d to 2d broadcasting
        assert_identical(v * v, Variable(["a", "b"], np.einsum("ab,ab->ab", x, x)))
        assert_identical(v * v[0], Variable(["a", "b"], np.einsum("ab,b->ab", x, x[0])))
        assert_identical(v[0] * v, Variable(["b", "a"], np.einsum("b,ab->ba", x[0], x)))
        assert_identical(
            v[0] * v[:, 0], Variable(["b", "a"], np.einsum("b,a->ba", x[0], x[:, 0]))
        )
        # higher dim broadcasting
        y = np.random.randn(3, 4, 5)
        w = Variable(["b", "c", "d"], y)
        assert_identical(
            v * w, Variable(["a", "b", "c", "d"], np.einsum("ab,bcd->abcd", x, y))
        )
        assert_identical(
            w * v, Variable(["b", "c", "d", "a"], np.einsum("bcd,ab->bcda", y, x))
        )
        assert_identical(
            v * w[0], Variable(["a", "b", "c", "d"], np.einsum("ab,cd->abcd", x, y[0]))
        )

    def test_broadcasting_failures(self):
        a = Variable(["x"], np.arange(10))
        b = Variable(["x"], np.arange(5))
        c = Variable(["x", "x"], np.arange(100).reshape(10, 10))
        with raises_regex(ValueError, "mismatched lengths"):
            a + b
        with raises_regex(ValueError, "duplicate dimensions"):
            a + c

    def test_inplace_math(self):
        x = np.arange(5)
        v = Variable(["x"], x)
        v2 = v
        v2 += 1
        assert v is v2
        # since we provided an ndarray for data, it is also modified in-place
        assert source_ndarray(v.values) is x
        assert_array_equal(v.values, np.arange(5) + 1)

        with raises_regex(ValueError, "dimensions cannot change"):
            v += Variable("y", np.arange(5))

    def test_reduce(self):
        v = Variable(["x", "y"], self.d, {"ignored": "attributes"})
        assert_identical(v.reduce(np.std, "x"), Variable(["y"], self.d.std(axis=0)))
        assert_identical(v.reduce(np.std, axis=0), v.reduce(np.std, dim="x"))
        assert_identical(
            v.reduce(np.std, ["y", "x"]), Variable([], self.d.std(axis=(0, 1)))
        )
        assert_identical(v.reduce(np.std), Variable([], self.d.std()))
        assert_identical(
            v.reduce(np.mean, "x").reduce(np.std, "y"),
            Variable([], self.d.mean(axis=0).std()),
        )
        assert_allclose(v.mean("x"), v.reduce(np.mean, "x"))

        with raises_regex(ValueError, "cannot supply both"):
            v.mean(dim="x", axis=0)

    def test_quantile(self):
        v = Variable(["x", "y"], self.d)
        for q in [0.25, [0.50], [0.25, 0.75]]:
            for axis, dim in zip(
                [None, 0, [0], [0, 1]], [None, "x", ["x"], ["x", "y"]]
            ):
                actual = v.quantile(q, dim=dim)

                expected = np.nanpercentile(self.d, np.array(q) * 100, axis=axis)
                np.testing.assert_allclose(actual.values, expected)

    @requires_dask
    def test_quantile_dask_raises(self):
        # regression for GH1524
        v = Variable(["x", "y"], self.d).chunk(2)

        with raises_regex(TypeError, "arrays stored as dask"):
            v.quantile(0.5, dim="x")

    @requires_dask
    @requires_bottleneck
    def test_rank_dask_raises(self):
        v = Variable(["x"], [3.0, 1.0, np.nan, 2.0, 4.0]).chunk(2)
        with raises_regex(TypeError, "arrays stored as dask"):
            v.rank("x")

    @requires_bottleneck
    def test_rank(self):
        import bottleneck as bn

        # floats
        v = Variable(["x", "y"], [[3, 4, np.nan, 1]])
        expect_0 = bn.nanrankdata(v.data, axis=0)
        expect_1 = bn.nanrankdata(v.data, axis=1)
        np.testing.assert_allclose(v.rank("x").values, expect_0)
        np.testing.assert_allclose(v.rank("y").values, expect_1)
        # int
        v = Variable(["x"], [3, 2, 1])
        expect = bn.rankdata(v.data, axis=0)
        np.testing.assert_allclose(v.rank("x").values, expect)
        # str
        v = Variable(["x"], ["c", "b", "a"])
        expect = bn.rankdata(v.data, axis=0)
        np.testing.assert_allclose(v.rank("x").values, expect)
        # pct
        v = Variable(["x"], [3.0, 1.0, np.nan, 2.0, 4.0])
        v_expect = Variable(["x"], [0.75, 0.25, np.nan, 0.5, 1.0])
        assert_equal(v.rank("x", pct=True), v_expect)
        # invalid dim
        with raises_regex(ValueError, "not found"):
            v.rank("y")

    def test_big_endian_reduce(self):
        # regression test for GH489
        data = np.ones(5, dtype=">f4")
        v = Variable(["x"], data)
        expected = Variable([], 5)
        assert_identical(expected, v.sum())

    def test_reduce_funcs(self):
        v = Variable("x", np.array([1, np.nan, 2, 3]))
        assert_identical(v.mean(), Variable([], 2))
        assert_identical(v.mean(skipna=True), Variable([], 2))
        assert_identical(v.mean(skipna=False), Variable([], np.nan))
        assert_identical(np.mean(v), Variable([], 2))

        assert_identical(v.prod(), Variable([], 6))
        assert_identical(v.cumsum(axis=0), Variable("x", np.array([1, 1, 3, 6])))
        assert_identical(v.cumprod(axis=0), Variable("x", np.array([1, 1, 2, 6])))
        assert_identical(v.var(), Variable([], 2.0 / 3))
        assert_identical(v.median(), Variable([], 2))

        v = Variable("x", [True, False, False])
        assert_identical(v.any(), Variable([], True))
        assert_identical(v.all(dim="x"), Variable([], False))

        v = Variable("t", pd.date_range("2000-01-01", periods=3))
        assert v.argmax(skipna=True) == 2

        assert_identical(v.max(), Variable([], pd.Timestamp("2000-01-03")))

    def test_reduce_keepdims(self):
        v = Variable(["x", "y"], self.d)

        assert_identical(
            v.mean(keepdims=True), Variable(v.dims, np.mean(self.d, keepdims=True))
        )
        assert_identical(
            v.mean(dim="x", keepdims=True),
            Variable(v.dims, np.mean(self.d, axis=0, keepdims=True)),
        )
        assert_identical(
            v.mean(dim="y", keepdims=True),
            Variable(v.dims, np.mean(self.d, axis=1, keepdims=True)),
        )
        assert_identical(
            v.mean(dim=["y", "x"], keepdims=True),
            Variable(v.dims, np.mean(self.d, axis=(1, 0), keepdims=True)),
        )

        v = Variable([], 1.0)
        assert_identical(
            v.mean(keepdims=True), Variable([], np.mean(v.data, keepdims=True))
        )

    @requires_dask
    def test_reduce_keepdims_dask(self):
        import dask.array

        v = Variable(["x", "y"], self.d).chunk()

        actual = v.mean(keepdims=True)
        assert isinstance(actual.data, dask.array.Array)

        expected = Variable(v.dims, np.mean(self.d, keepdims=True))
        assert_identical(actual, expected)

        actual = v.mean(dim="y", keepdims=True)
        assert isinstance(actual.data, dask.array.Array)

        expected = Variable(v.dims, np.mean(self.d, axis=1, keepdims=True))
        assert_identical(actual, expected)

    def test_reduce_keep_attrs(self):
        _attrs = {"units": "test", "long_name": "testing"}

        v = Variable(["x", "y"], self.d, _attrs)

        # Test dropped attrs
        vm = v.mean()
        assert len(vm.attrs) == 0
        assert vm.attrs == {}

        # Test kept attrs
        vm = v.mean(keep_attrs=True)
        assert len(vm.attrs) == len(_attrs)
        assert vm.attrs == _attrs

    def test_binary_ops_keep_attrs(self):
        _attrs = {"units": "test", "long_name": "testing"}
        a = Variable(["x", "y"], np.random.randn(3, 3), _attrs)
        b = Variable(["x", "y"], np.random.randn(3, 3), _attrs)
        # Test dropped attrs
        d = a - b  # just one operation
        assert d.attrs == {}
        # Test kept attrs
        with set_options(keep_attrs=True):
            d = a - b
        assert d.attrs == _attrs

    def test_count(self):
        expected = Variable([], 3)
        actual = Variable(["x"], [1, 2, 3, np.nan]).count()
        assert_identical(expected, actual)

        v = Variable(["x"], np.array(["1", "2", "3", np.nan], dtype=object))
        actual = v.count()
        assert_identical(expected, actual)

        actual = Variable(["x"], [True, False, True]).count()
        assert_identical(expected, actual)
        assert actual.dtype == int

        expected = Variable(["x"], [2, 3])
        actual = Variable(["x", "y"], [[1, 0, np.nan], [1, 1, 1]]).count("y")
        assert_identical(expected, actual)

    def test_setitem(self):
        v = Variable(["x", "y"], [[0, 3, 2], [3, 4, 5]])
        v[0, 1] = 1
        assert v[0, 1] == 1

        v = Variable(["x", "y"], [[0, 3, 2], [3, 4, 5]])
        v[dict(x=[0, 1])] = 1
        assert_array_equal(v[[0, 1]], np.ones_like(v[[0, 1]]))

        # boolean indexing
        v = Variable(["x", "y"], [[0, 3, 2], [3, 4, 5]])
        v[dict(x=[True, False])] = 1

        assert_array_equal(v[0], np.ones_like(v[0]))
        v = Variable(["x", "y"], [[0, 3, 2], [3, 4, 5]])
        v[dict(x=[True, False], y=[False, True, False])] = 1
        assert v[0, 1] == 1

    def test_setitem_fancy(self):
        # assignment which should work as np.ndarray does
        def assert_assigned_2d(array, key_x, key_y, values):
            expected = array.copy()
            expected[key_x, key_y] = values
            v = Variable(["x", "y"], array)
            v[dict(x=key_x, y=key_y)] = values
            assert_array_equal(expected, v)

        # 1d vectorized indexing
        assert_assigned_2d(
            np.random.randn(4, 3),
            key_x=Variable(["a"], [0, 1]),
            key_y=Variable(["a"], [0, 1]),
            values=0,
        )
        assert_assigned_2d(
            np.random.randn(4, 3),
            key_x=Variable(["a"], [0, 1]),
            key_y=Variable(["a"], [0, 1]),
            values=Variable((), 0),
        )
        assert_assigned_2d(
            np.random.randn(4, 3),
            key_x=Variable(["a"], [0, 1]),
            key_y=Variable(["a"], [0, 1]),
            values=Variable(("a"), [3, 2]),
        )
        assert_assigned_2d(
            np.random.randn(4, 3),
            key_x=slice(None),
            key_y=Variable(["a"], [0, 1]),
            values=Variable(("a"), [3, 2]),
        )

        # 2d-vectorized indexing
        assert_assigned_2d(
            np.random.randn(4, 3),
            key_x=Variable(["a", "b"], [[0, 1]]),
            key_y=Variable(["a", "b"], [[1, 0]]),
            values=0,
        )
        assert_assigned_2d(
            np.random.randn(4, 3),
            key_x=Variable(["a", "b"], [[0, 1]]),
            key_y=Variable(["a", "b"], [[1, 0]]),
            values=[0],
        )
        assert_assigned_2d(
            np.random.randn(5, 4),
            key_x=Variable(["a", "b"], [[0, 1], [2, 3]]),
            key_y=Variable(["a", "b"], [[1, 0], [3, 3]]),
            values=[2, 3],
        )

        # vindex with slice
        v = Variable(["x", "y", "z"], np.ones((4, 3, 2)))
        ind = Variable(["a"], [0, 1])
        v[dict(x=ind, z=ind)] = 0
        expected = Variable(["x", "y", "z"], np.ones((4, 3, 2)))
        expected[0, :, 0] = 0
        expected[1, :, 1] = 0
        assert_identical(expected, v)

        # dimension broadcast
        v = Variable(["x", "y"], np.ones((3, 2)))
        ind = Variable(["a", "b"], [[0, 1]])
        v[ind, :] = 0
        expected = Variable(["x", "y"], [[0, 0], [0, 0], [1, 1]])
        assert_identical(expected, v)

        with raises_regex(ValueError, "shape mismatch"):
            v[ind, ind] = np.zeros((1, 2, 1))

        v = Variable(["x", "y"], [[0, 3, 2], [3, 4, 5]])
        ind = Variable(["a"], [0, 1])
        v[dict(x=ind)] = Variable(["a", "y"], np.ones((2, 3), dtype=int) * 10)
        assert_array_equal(v[0], np.ones_like(v[0]) * 10)
        assert_array_equal(v[1], np.ones_like(v[1]) * 10)
        assert v.dims == ("x", "y")  # dimension should not change

        # increment
        v = Variable(["x", "y"], np.arange(6).reshape(3, 2))
        ind = Variable(["a"], [0, 1])
        v[dict(x=ind)] += 1
        expected = Variable(["x", "y"], [[1, 2], [3, 4], [4, 5]])
        assert_identical(v, expected)

        ind = Variable(["a"], [0, 0])
        v[dict(x=ind)] += 1
        expected = Variable(["x", "y"], [[2, 3], [3, 4], [4, 5]])
        assert_identical(v, expected)

    def test_coarsen(self):
        v = self.cls(["x"], [0, 1, 2, 3, 4])
        actual = v.coarsen({"x": 2}, boundary="pad", func="mean")
        expected = self.cls(["x"], [0.5, 2.5, 4])
        assert_identical(actual, expected)

        actual = v.coarsen({"x": 2}, func="mean", boundary="pad", side="right")
        expected = self.cls(["x"], [0, 1.5, 3.5])
        assert_identical(actual, expected)

        actual = v.coarsen({"x": 2}, func=np.mean, side="right", boundary="trim")
        expected = self.cls(["x"], [1.5, 3.5])
        assert_identical(actual, expected)

        # working test
        v = self.cls(["x", "y", "z"], np.arange(40 * 30 * 2).reshape(40, 30, 2))
        for windows, func, side, boundary in [
            ({"x": 2}, np.mean, "left", "trim"),
            ({"x": 2}, np.median, {"x": "left"}, "pad"),
            ({"x": 2, "y": 3}, np.max, "left", {"x": "pad", "y": "trim"}),
        ]:
            v.coarsen(windows, func, boundary, side)

    def test_coarsen_2d(self):
        # 2d-mean should be the same with the successive 1d-mean
        v = self.cls(["x", "y"], np.arange(6 * 12).reshape(6, 12))
        actual = v.coarsen({"x": 3, "y": 4}, func="mean")
        expected = v.coarsen({"x": 3}, func="mean").coarsen({"y": 4}, func="mean")
        assert_equal(actual, expected)

        v = self.cls(["x", "y"], np.arange(7 * 12).reshape(7, 12))
        actual = v.coarsen({"x": 3, "y": 4}, func="mean", boundary="trim")
        expected = v.coarsen({"x": 3}, func="mean", boundary="trim").coarsen(
            {"y": 4}, func="mean", boundary="trim"
        )
        assert_equal(actual, expected)

        # if there is nan, the two should be different
        v = self.cls(["x", "y"], 1.0 * np.arange(6 * 12).reshape(6, 12))
        v[2, 4] = np.nan
        v[3, 5] = np.nan
        actual = v.coarsen({"x": 3, "y": 4}, func="mean", boundary="trim")
        expected = (
            v.coarsen({"x": 3}, func="sum", boundary="trim").coarsen(
                {"y": 4}, func="sum", boundary="trim"
            )
            / 12
        )
        assert not actual.equals(expected)
        # adjusting the nan count
        expected[0, 1] *= 12 / 11
        expected[1, 1] *= 12 / 11
        assert_allclose(actual, expected)


@requires_dask
class TestVariableWithDask(VariableSubclassobjects):
    cls = staticmethod(lambda *args: Variable(*args).chunk())

    @pytest.mark.xfail
    def test_0d_object_array_with_list(self):
        super().test_0d_object_array_with_list()

    @pytest.mark.xfail
    def test_array_interface(self):
        # dask array does not have `argsort`
        super().test_array_interface()

    @pytest.mark.xfail
    def test_copy_index(self):
        super().test_copy_index()

    @pytest.mark.xfail
    def test_eq_all_dtypes(self):
        super().test_eq_all_dtypes()

    def test_getitem_fancy(self):
        super().test_getitem_fancy()

    def test_getitem_1d_fancy(self):
        super().test_getitem_1d_fancy()

    def test_getitem_with_mask_nd_indexer(self):
        import dask.array as da

        v = Variable(["x"], da.arange(3, chunks=3))
        indexer = Variable(("x", "y"), [[0, -1], [-1, 2]])
        assert_identical(
            v._getitem_with_mask(indexer, fill_value=-1),
            self.cls(("x", "y"), [[0, -1], [-1, 2]]),
        )


class TestIndexVariable(VariableSubclassobjects):
    cls = staticmethod(IndexVariable)

    def test_init(self):
        with raises_regex(ValueError, "must be 1-dimensional"):
            IndexVariable((), 0)

    def test_to_index(self):
        data = 0.5 * np.arange(10)
        v = IndexVariable(["time"], data, {"foo": "bar"})
        assert pd.Index(data, name="time").identical(v.to_index())

    def test_multiindex_default_level_names(self):
        midx = pd.MultiIndex.from_product([["a", "b"], [1, 2]])
        v = IndexVariable(["x"], midx, {"foo": "bar"})
        assert v.to_index().names == ("x_level_0", "x_level_1")

    def test_data(self):
        x = IndexVariable("x", np.arange(3.0))
        assert isinstance(x._data, PandasIndexAdapter)
        assert isinstance(x.data, np.ndarray)
        assert float == x.dtype
        assert_array_equal(np.arange(3), x)
        assert float == x.values.dtype
        with raises_regex(TypeError, "cannot be modified"):
            x[:] = 0

    def test_name(self):
        coord = IndexVariable("x", [10.0])
        assert coord.name == "x"

        with pytest.raises(AttributeError):
            coord.name = "y"

    def test_level_names(self):
        midx = pd.MultiIndex.from_product(
            [["a", "b"], [1, 2]], names=["level_1", "level_2"]
        )
        x = IndexVariable("x", midx)
        assert x.level_names == midx.names

        assert IndexVariable("y", [10.0]).level_names is None

    def test_get_level_variable(self):
        midx = pd.MultiIndex.from_product(
            [["a", "b"], [1, 2]], names=["level_1", "level_2"]
        )
        x = IndexVariable("x", midx)
        level_1 = IndexVariable("x", midx.get_level_values("level_1"))
        assert_identical(x.get_level_variable("level_1"), level_1)

        with raises_regex(ValueError, "has no MultiIndex"):
            IndexVariable("y", [10.0]).get_level_variable("level")

    def test_concat_periods(self):
        periods = pd.period_range("2000-01-01", periods=10)
        coords = [IndexVariable("t", periods[:5]), IndexVariable("t", periods[5:])]
        expected = IndexVariable("t", periods)
        actual = IndexVariable.concat(coords, dim="t")
        assert actual.identical(expected)
        assert isinstance(actual.to_index(), pd.PeriodIndex)

        positions = [list(range(5)), list(range(5, 10))]
        actual = IndexVariable.concat(coords, dim="t", positions=positions)
        assert actual.identical(expected)
        assert isinstance(actual.to_index(), pd.PeriodIndex)

    def test_concat_multiindex(self):
        idx = pd.MultiIndex.from_product([[0, 1, 2], ["a", "b"]])
        coords = [IndexVariable("x", idx[:2]), IndexVariable("x", idx[2:])]
        expected = IndexVariable("x", idx)
        actual = IndexVariable.concat(coords, dim="x")
        assert actual.identical(expected)
        assert isinstance(actual.to_index(), pd.MultiIndex)

    def test_coordinate_alias(self):
        with pytest.warns(Warning, match="deprecated"):
            x = Coordinate("x", [1, 2, 3])
        assert isinstance(x, IndexVariable)

    def test_datetime64(self):
        # GH:1932  Make sure indexing keeps precision
        t = np.array([1518418799999986560, 1518418799999996560], dtype="datetime64[ns]")
        v = IndexVariable("t", t)
        assert v[0].data == t[0]

    # These tests make use of multi-dimensional variables, which are not valid
    # IndexVariable objects:
    @pytest.mark.xfail
    def test_getitem_error(self):
        super().test_getitem_error()

    @pytest.mark.xfail
    def test_getitem_advanced(self):
        super().test_getitem_advanced()

    @pytest.mark.xfail
    def test_getitem_fancy(self):
        super().test_getitem_fancy()

    @pytest.mark.xfail
    def test_getitem_uint(self):
        super().test_getitem_fancy()

    @pytest.mark.xfail
    def test_pad(self):
        super().test_rolling_window()

    @pytest.mark.xfail
    def test_rolling_window(self):
        super().test_rolling_window()

    @pytest.mark.xfail
    def test_coarsen_2d(self):
        super().test_coarsen_2d()


class TestAsCompatibleData:
    def test_unchanged_types(self):
        types = (np.asarray, PandasIndexAdapter, LazilyOuterIndexedArray)
        for t in types:
            for data in [
                np.arange(3),
                pd.date_range("2000-01-01", periods=3),
                pd.date_range("2000-01-01", periods=3).values,
            ]:
                x = t(data)
                assert source_ndarray(x) is source_ndarray(as_compatible_data(x))

    def test_converted_types(self):
        for input_array in [[[0, 1, 2]], pd.DataFrame([[0, 1, 2]])]:
            actual = as_compatible_data(input_array)
            assert_array_equal(np.asarray(input_array), actual)
            assert np.ndarray == type(actual)
            assert np.asarray(input_array).dtype == actual.dtype

    def test_masked_array(self):
        original = np.ma.MaskedArray(np.arange(5))
        expected = np.arange(5)
        actual = as_compatible_data(original)
        assert_array_equal(expected, actual)
        assert np.dtype(int) == actual.dtype

        original = np.ma.MaskedArray(np.arange(5), mask=4 * [False] + [True])
        expected = np.arange(5.0)
        expected[-1] = np.nan
        actual = as_compatible_data(original)
        assert_array_equal(expected, actual)
        assert np.dtype(float) == actual.dtype

    def test_datetime(self):
        expected = np.datetime64("2000-01-01")
        actual = as_compatible_data(expected)
        assert expected == actual
        assert np.ndarray == type(actual)
        assert np.dtype("datetime64[ns]") == actual.dtype

        expected = np.array([np.datetime64("2000-01-01")])
        actual = as_compatible_data(expected)
        assert np.asarray(expected) == actual
        assert np.ndarray == type(actual)
        assert np.dtype("datetime64[ns]") == actual.dtype

        expected = np.array([np.datetime64("2000-01-01", "ns")])
        actual = as_compatible_data(expected)
        assert np.asarray(expected) == actual
        assert np.ndarray == type(actual)
        assert np.dtype("datetime64[ns]") == actual.dtype
        assert expected is source_ndarray(np.asarray(actual))

        expected = np.datetime64("2000-01-01", "ns")
        actual = as_compatible_data(datetime(2000, 1, 1))
        assert np.asarray(expected) == actual
        assert np.ndarray == type(actual)
        assert np.dtype("datetime64[ns]") == actual.dtype

    def test_full_like(self):
        # For more thorough tests, see test_variable.py
        orig = Variable(
            dims=("x", "y"), data=[[1.5, 2.0], [3.1, 4.3]], attrs={"foo": "bar"}
        )

        expect = orig.copy(deep=True)
        expect.values = [[2.0, 2.0], [2.0, 2.0]]
        assert_identical(expect, full_like(orig, 2))

        # override dtype
        expect.values = [[True, True], [True, True]]
        assert expect.dtype == bool
        assert_identical(expect, full_like(orig, True, dtype=bool))

    @requires_dask
    def test_full_like_dask(self):
        orig = Variable(
            dims=("x", "y"), data=[[1.5, 2.0], [3.1, 4.3]], attrs={"foo": "bar"}
        ).chunk(((1, 1), (2,)))

        def check(actual, expect_dtype, expect_values):
            assert actual.dtype == expect_dtype
            assert actual.shape == orig.shape
            assert actual.dims == orig.dims
            assert actual.attrs == orig.attrs
            assert actual.chunks == orig.chunks
            assert_array_equal(actual.values, expect_values)

        check(full_like(orig, 2), orig.dtype, np.full_like(orig.values, 2))
        # override dtype
        check(
            full_like(orig, True, dtype=bool),
            bool,
            np.full_like(orig.values, True, dtype=bool),
        )

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
        orig = Variable(
            dims=("x", "y"), data=[[1.5, 2.0], [3.1, 4.3]], attrs={"foo": "bar"}
        )
        assert_identical(zeros_like(orig), full_like(orig, 0))
        assert_identical(zeros_like(orig, dtype=int), full_like(orig, 0, dtype=int))

    def test_ones_like(self):
        orig = Variable(
            dims=("x", "y"), data=[[1.5, 2.0], [3.1, 4.3]], attrs={"foo": "bar"}
        )
        assert_identical(ones_like(orig), full_like(orig, 1))
        assert_identical(ones_like(orig, dtype=int), full_like(orig, 1, dtype=int))

    def test_unsupported_type(self):
        # Non indexable type
        class CustomArray(NDArrayMixin):
            def __init__(self, array):
                self.array = array

        class CustomIndexable(CustomArray, indexing.ExplicitlyIndexed):
            pass

        array = CustomArray(np.arange(3))
        orig = Variable(dims=("x"), data=array, attrs={"foo": "bar"})
        assert isinstance(orig._data, np.ndarray)  # should not be CustomArray

        array = CustomIndexable(np.arange(3))
        orig = Variable(dims=("x"), data=array, attrs={"foo": "bar"})
        assert isinstance(orig._data, CustomIndexable)


def test_raise_no_warning_for_nan_in_binary_ops():
    with pytest.warns(None) as record:
        Variable("x", [1, 2, np.NaN]) > 0
    assert len(record) == 0


class TestBackendIndexing:
    """    Make sure all the array wrappers can be indexed. """

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.d = np.random.random((10, 3)).astype(np.float64)

    def check_orthogonal_indexing(self, v):
        assert np.allclose(v.isel(x=[8, 3], y=[2, 1]), self.d[[8, 3]][:, [2, 1]])

    def check_vectorized_indexing(self, v):
        ind_x = Variable("z", [0, 2])
        ind_y = Variable("z", [2, 1])
        assert np.allclose(v.isel(x=ind_x, y=ind_y), self.d[ind_x, ind_y])

    def test_NumpyIndexingAdapter(self):
        v = Variable(dims=("x", "y"), data=NumpyIndexingAdapter(self.d))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)
        # could not doubly wrapping
        with raises_regex(TypeError, "NumpyIndexingAdapter only wraps "):
            v = Variable(
                dims=("x", "y"), data=NumpyIndexingAdapter(NumpyIndexingAdapter(self.d))
            )

    def test_LazilyOuterIndexedArray(self):
        v = Variable(dims=("x", "y"), data=LazilyOuterIndexedArray(self.d))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)
        # doubly wrapping
        v = Variable(
            dims=("x", "y"),
            data=LazilyOuterIndexedArray(LazilyOuterIndexedArray(self.d)),
        )
        self.check_orthogonal_indexing(v)
        # hierarchical wrapping
        v = Variable(
            dims=("x", "y"), data=LazilyOuterIndexedArray(NumpyIndexingAdapter(self.d))
        )
        self.check_orthogonal_indexing(v)

    def test_CopyOnWriteArray(self):
        v = Variable(dims=("x", "y"), data=CopyOnWriteArray(self.d))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)
        # doubly wrapping
        v = Variable(
            dims=("x", "y"), data=CopyOnWriteArray(LazilyOuterIndexedArray(self.d))
        )
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)

    def test_MemoryCachedArray(self):
        v = Variable(dims=("x", "y"), data=MemoryCachedArray(self.d))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)
        # doubly wrapping
        v = Variable(dims=("x", "y"), data=CopyOnWriteArray(MemoryCachedArray(self.d)))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)

    @requires_dask
    def test_DaskIndexingAdapter(self):
        import dask.array as da

        da = da.asarray(self.d)
        v = Variable(dims=("x", "y"), data=DaskIndexingAdapter(da))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)
        # doubly wrapping
        v = Variable(dims=("x", "y"), data=CopyOnWriteArray(DaskIndexingAdapter(da)))
        self.check_orthogonal_indexing(v)
        self.check_vectorized_indexing(v)
