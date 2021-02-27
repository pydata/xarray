import pickle
import sys
import warnings
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex

import xarray as xr
from xarray import (
    DataArray,
    Dataset,
    IndexVariable,
    MergeError,
    Variable,
    align,
    backends,
    broadcast,
    open_dataset,
    set_options,
)
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.npcompat import IS_NEP18_ACTIVE
from xarray.core.pycompat import integer_types
from xarray.core.utils import is_scalar

from . import (
    InaccessibleArray,
    UnexpectedDataAccess,
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_identical,
    has_cftime,
    has_dask,
    raises_regex,
    requires_bottleneck,
    requires_cftime,
    requires_dask,
    requires_numbagg,
    requires_scipy,
    requires_sparse,
    source_ndarray,
)

try:
    import dask.array as da
except ImportError:
    pass

pytestmark = [
    pytest.mark.filterwarnings("error:Mean of empty slice"),
    pytest.mark.filterwarnings("error:All-NaN (slice|axis) encountered"),
]


def create_test_data(seed=None):
    rs = np.random.RandomState(seed)
    _vars = {
        "var1": ["dim1", "dim2"],
        "var2": ["dim1", "dim2"],
        "var3": ["dim3", "dim1"],
    }
    _dims = {"dim1": 8, "dim2": 9, "dim3": 10}

    obj = Dataset()
    obj["time"] = ("time", pd.date_range("2000-01-01", periods=20))
    obj["dim2"] = ("dim2", 0.5 * np.arange(_dims["dim2"]))
    obj["dim3"] = ("dim3", list("abcdefghij"))
    for v, dims in sorted(_vars.items()):
        data = rs.normal(size=tuple(_dims[d] for d in dims))
        obj[v] = (dims, data, {"foo": "variable"})
    obj.coords["numbers"] = (
        "dim3",
        np.array([0, 1, 2, 0, 0, 1, 1, 2, 2, 3], dtype="int64"),
    )
    obj.encoding = {"foo": "bar"}
    assert all(obj.data.flags.writeable for obj in obj.variables.values())
    return obj


def create_append_test_data(seed=None):
    rs = np.random.RandomState(seed)

    lat = [2, 1, 0]
    lon = [0, 1, 2]
    nt1 = 3
    nt2 = 2
    time1 = pd.date_range("2000-01-01", periods=nt1)
    time2 = pd.date_range("2000-02-01", periods=nt2)
    string_var = np.array(["ae", "bc", "df"], dtype=object)
    string_var_to_append = np.array(["asdf", "asdfg"], dtype=object)
    unicode_var = ["áó", "áó", "áó"]
    datetime_var = np.array(
        ["2019-01-01", "2019-01-02", "2019-01-03"], dtype="datetime64[s]"
    )
    datetime_var_to_append = np.array(
        ["2019-01-04", "2019-01-05"], dtype="datetime64[s]"
    )
    bool_var = np.array([True, False, True], dtype=bool)
    bool_var_to_append = np.array([False, True], dtype=bool)

    ds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                rs.rand(3, 3, nt1),
                coords=[lat, lon, time1],
                dims=["lat", "lon", "time"],
            ),
            "string_var": xr.DataArray(string_var, coords=[time1], dims=["time"]),
            "unicode_var": xr.DataArray(
                unicode_var, coords=[time1], dims=["time"]
            ).astype(np.unicode_),
            "datetime_var": xr.DataArray(datetime_var, coords=[time1], dims=["time"]),
            "bool_var": xr.DataArray(bool_var, coords=[time1], dims=["time"]),
        }
    )

    ds_to_append = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                rs.rand(3, 3, nt2),
                coords=[lat, lon, time2],
                dims=["lat", "lon", "time"],
            ),
            "string_var": xr.DataArray(
                string_var_to_append, coords=[time2], dims=["time"]
            ),
            "unicode_var": xr.DataArray(
                unicode_var[:nt2], coords=[time2], dims=["time"]
            ).astype(np.unicode_),
            "datetime_var": xr.DataArray(
                datetime_var_to_append, coords=[time2], dims=["time"]
            ),
            "bool_var": xr.DataArray(bool_var_to_append, coords=[time2], dims=["time"]),
        }
    )

    ds_with_new_var = xr.Dataset(
        data_vars={
            "new_var": xr.DataArray(
                rs.rand(3, 3, nt1 + nt2),
                coords=[lat, lon, time1.append(time2)],
                dims=["lat", "lon", "time"],
            )
        }
    )

    assert all(objp.data.flags.writeable for objp in ds.variables.values())
    assert all(objp.data.flags.writeable for objp in ds_to_append.variables.values())
    return ds, ds_to_append, ds_with_new_var


def create_test_multiindex():
    mindex = pd.MultiIndex.from_product(
        [["a", "b"], [1, 2]], names=("level_1", "level_2")
    )
    return Dataset({}, {"x": mindex})


def create_test_stacked_array():
    x = DataArray(pd.Index(np.r_[:10], name="x"))
    y = DataArray(pd.Index(np.r_[:20], name="y"))
    a = x * y
    b = x * y * y
    return a, b


class InaccessibleVariableDataStore(backends.InMemoryDataStore):
    def __init__(self):
        super().__init__()
        self._indexvars = set()

    def store(self, variables, *args, **kwargs):
        super().store(variables, *args, **kwargs)
        for k, v in variables.items():
            if isinstance(v, IndexVariable):
                self._indexvars.add(k)

    def get_variables(self):
        def lazy_inaccessible(k, v):
            if k in self._indexvars:
                return v
            data = indexing.LazilyOuterIndexedArray(InaccessibleArray(v.values))
            return Variable(v.dims, data, v.attrs)

        return {k: lazy_inaccessible(k, v) for k, v in self._variables.items()}


class TestDataset:
    def test_repr(self):
        data = create_test_data(seed=123)
        data.attrs["foo"] = "bar"
        # need to insert str dtype at runtime to handle different endianness
        expected = dedent(
            """\
            <xarray.Dataset>
            Dimensions:  (dim1: 8, dim2: 9, dim3: 10, time: 20)
            Coordinates:
              * time     (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2000-01-20
              * dim2     (dim2) float64 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0
              * dim3     (dim3) %s 'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j'
                numbers  (dim3) int64 0 1 2 0 0 1 1 2 2 3
            Dimensions without coordinates: dim1
            Data variables:
                var1     (dim1, dim2) float64 -1.086 0.9973 0.283 ... 0.1995 0.4684 -0.8312
                var2     (dim1, dim2) float64 1.162 -1.097 -2.123 ... 0.1302 1.267 0.3328
                var3     (dim3, dim1) float64 0.5565 -0.2121 0.4563 ... -0.2452 -0.3616
            Attributes:
                foo:      bar"""
            % data["dim3"].dtype
        )
        actual = "\n".join(x.rstrip() for x in repr(data).split("\n"))
        print(actual)
        assert expected == actual

        with set_options(display_width=100):
            max_len = max(map(len, repr(data).split("\n")))
            assert 90 < max_len < 100

        expected = dedent(
            """\
            <xarray.Dataset>
            Dimensions:  ()
            Data variables:
                *empty*"""
        )
        actual = "\n".join(x.rstrip() for x in repr(Dataset()).split("\n"))
        print(actual)
        assert expected == actual

        # verify that ... doesn't appear for scalar coordinates
        data = Dataset({"foo": ("x", np.ones(10))}).mean()
        expected = dedent(
            """\
            <xarray.Dataset>
            Dimensions:  ()
            Data variables:
                foo      float64 1.0"""
        )
        actual = "\n".join(x.rstrip() for x in repr(data).split("\n"))
        print(actual)
        assert expected == actual

        # verify long attributes are truncated
        data = Dataset(attrs={"foo": "bar" * 1000})
        assert len(repr(data)) < 1000

    def test_repr_multiindex(self):
        data = create_test_multiindex()
        expected = dedent(
            """\
            <xarray.Dataset>
            Dimensions:  (x: 4)
            Coordinates:
              * x        (x) MultiIndex
              - level_1  (x) object 'a' 'a' 'b' 'b'
              - level_2  (x) int64 1 2 1 2
            Data variables:
                *empty*"""
        )
        actual = "\n".join(x.rstrip() for x in repr(data).split("\n"))
        print(actual)
        assert expected == actual

        # verify that long level names are not truncated
        mindex = pd.MultiIndex.from_product(
            [["a", "b"], [1, 2]], names=("a_quite_long_level_name", "level_2")
        )
        data = Dataset({}, {"x": mindex})
        expected = dedent(
            """\
            <xarray.Dataset>
            Dimensions:                  (x: 4)
            Coordinates:
              * x                        (x) MultiIndex
              - a_quite_long_level_name  (x) object 'a' 'a' 'b' 'b'
              - level_2                  (x) int64 1 2 1 2
            Data variables:
                *empty*"""
        )
        actual = "\n".join(x.rstrip() for x in repr(data).split("\n"))
        print(actual)
        assert expected == actual

    def test_repr_period_index(self):
        data = create_test_data(seed=456)
        data.coords["time"] = pd.period_range("2000-01-01", periods=20, freq="B")

        # check that creating the repr doesn't raise an error #GH645
        repr(data)

    def test_unicode_data(self):
        # regression test for GH834
        data = Dataset({"foø": ["ba®"]}, attrs={"å": "∑"})
        repr(data)  # should not raise

        byteorder = "<" if sys.byteorder == "little" else ">"
        expected = dedent(
            """\
            <xarray.Dataset>
            Dimensions:  (foø: 1)
            Coordinates:
              * foø      (foø) %cU3 %r
            Data variables:
                *empty*
            Attributes:
                å:        ∑"""
            % (byteorder, "ba®")
        )
        actual = str(data)
        assert expected == actual

    @pytest.mark.skipif(not IS_NEP18_ACTIVE, reason="requires __array_function__")
    def test_repr_nep18(self):
        class Array:
            def __init__(self):
                self.shape = (2,)
                self.dtype = np.dtype(np.float64)

            def __array_function__(self, *args, **kwargs):
                pass

            def __repr__(self):
                return "Custom\nArray"

        dataset = Dataset({"foo": ("x", Array())})
        expected = dedent(
            """\
            <xarray.Dataset>
            Dimensions:  (x: 2)
            Dimensions without coordinates: x
            Data variables:
                foo      (x) float64 Custom Array"""
        )
        assert expected == repr(dataset)

    def test_info(self):
        ds = create_test_data(seed=123)
        ds = ds.drop_vars("dim3")  # string type prints differently in PY2 vs PY3
        ds.attrs["unicode_attr"] = "ba®"
        ds.attrs["string_attr"] = "bar"

        buf = StringIO()
        ds.info(buf=buf)

        expected = dedent(
            """\
        xarray.Dataset {
        dimensions:
        \tdim1 = 8 ;
        \tdim2 = 9 ;
        \tdim3 = 10 ;
        \ttime = 20 ;

        variables:
        \tdatetime64[ns] time(time) ;
        \tfloat64 dim2(dim2) ;
        \tfloat64 var1(dim1, dim2) ;
        \t\tvar1:foo = variable ;
        \tfloat64 var2(dim1, dim2) ;
        \t\tvar2:foo = variable ;
        \tfloat64 var3(dim3, dim1) ;
        \t\tvar3:foo = variable ;
        \tint64 numbers(dim3) ;

        // global attributes:
        \t:unicode_attr = ba® ;
        \t:string_attr = bar ;
        }"""
        )
        actual = buf.getvalue()
        assert expected == actual
        buf.close()

    def test_constructor(self):
        x1 = ("x", 2 * np.arange(100))
        x2 = ("x", np.arange(1000))
        z = (["x", "y"], np.arange(1000).reshape(100, 10))

        with raises_regex(ValueError, "conflicting sizes"):
            Dataset({"a": x1, "b": x2})
        with raises_regex(ValueError, "disallows such variables"):
            Dataset({"a": x1, "x": z})
        with raises_regex(TypeError, "tuple of form"):
            Dataset({"x": (1, 2, 3, 4, 5, 6, 7)})
        with raises_regex(ValueError, "already exists as a scalar"):
            Dataset({"x": 0, "y": ("x", [1, 2, 3])})

        # verify handling of DataArrays
        expected = Dataset({"x": x1, "z": z})
        actual = Dataset({"z": expected["z"]})
        assert_identical(expected, actual)

    def test_constructor_invalid_dims(self):
        # regression for GH1120
        with pytest.raises(MergeError):
            Dataset(
                data_vars=dict(v=("y", [1, 2, 3, 4])),
                coords=dict(y=DataArray([0.1, 0.2, 0.3, 0.4], dims="x")),
            )

    def test_constructor_1d(self):
        expected = Dataset({"x": (["x"], 5.0 + np.arange(5))})
        actual = Dataset({"x": 5.0 + np.arange(5)})
        assert_identical(expected, actual)

        actual = Dataset({"x": [5, 6, 7, 8, 9]})
        assert_identical(expected, actual)

    def test_constructor_0d(self):
        expected = Dataset({"x": ([], 1)})
        for arg in [1, np.array(1), expected["x"]]:
            actual = Dataset({"x": arg})
            assert_identical(expected, actual)

        class Arbitrary:
            pass

        d = pd.Timestamp("2000-01-01T12")
        args = [
            True,
            None,
            3.4,
            np.nan,
            "hello",
            b"raw",
            np.datetime64("2000-01-01"),
            d,
            d.to_pydatetime(),
            Arbitrary(),
        ]
        for arg in args:
            print(arg)
            expected = Dataset({"x": ([], arg)})
            actual = Dataset({"x": arg})
            assert_identical(expected, actual)

    def test_constructor_deprecated(self):
        with raises_regex(ValueError, "DataArray dimensions"):
            DataArray([1, 2, 3], coords={"x": [0, 1, 2]})

    def test_constructor_auto_align(self):
        a = DataArray([1, 2], [("x", [0, 1])])
        b = DataArray([3, 4], [("x", [1, 2])])

        # verify align uses outer join
        expected = Dataset(
            {"a": ("x", [1, 2, np.nan]), "b": ("x", [np.nan, 3, 4])}, {"x": [0, 1, 2]}
        )
        actual = Dataset({"a": a, "b": b})
        assert_identical(expected, actual)

        # regression test for GH346
        assert isinstance(actual.variables["x"], IndexVariable)

        # variable with different dimensions
        c = ("y", [3, 4])
        expected2 = expected.merge({"c": c})
        actual = Dataset({"a": a, "b": b, "c": c})
        assert_identical(expected2, actual)

        # variable that is only aligned against the aligned variables
        d = ("x", [3, 2, 1])
        expected3 = expected.merge({"d": d})
        actual = Dataset({"a": a, "b": b, "d": d})
        assert_identical(expected3, actual)

        e = ("x", [0, 0])
        with raises_regex(ValueError, "conflicting sizes"):
            Dataset({"a": a, "b": b, "e": e})

    def test_constructor_pandas_sequence(self):

        ds = self.make_example_math_dataset()
        pandas_objs = {
            var_name: ds[var_name].to_pandas() for var_name in ["foo", "bar"]
        }
        ds_based_on_pandas = Dataset(pandas_objs, ds.coords, attrs=ds.attrs)
        del ds_based_on_pandas["x"]
        assert_equal(ds, ds_based_on_pandas)

        # reindex pandas obj, check align works
        rearranged_index = reversed(pandas_objs["foo"].index)
        pandas_objs["foo"] = pandas_objs["foo"].reindex(rearranged_index)
        ds_based_on_pandas = Dataset(pandas_objs, ds.coords, attrs=ds.attrs)
        del ds_based_on_pandas["x"]
        assert_equal(ds, ds_based_on_pandas)

    def test_constructor_pandas_single(self):

        das = [
            DataArray(np.random.rand(4), dims=["a"]),  # series
            DataArray(np.random.rand(4, 3), dims=["a", "b"]),  # df
        ]

        for a in das:
            pandas_obj = a.to_pandas()
            ds_based_on_pandas = Dataset(pandas_obj)
            for dim in ds_based_on_pandas.data_vars:
                assert_array_equal(ds_based_on_pandas[dim], pandas_obj[dim])

    def test_constructor_compat(self):
        data = {"x": DataArray(0, coords={"y": 1}), "y": ("z", [1, 1, 1])}
        expected = Dataset({"x": 0}, {"y": ("z", [1, 1, 1])})
        actual = Dataset(data)
        assert_identical(expected, actual)

        data = {"y": ("z", [1, 1, 1]), "x": DataArray(0, coords={"y": 1})}
        actual = Dataset(data)
        assert_identical(expected, actual)

        original = Dataset(
            {"a": (("x", "y"), np.ones((2, 3)))},
            {"c": (("x", "y"), np.zeros((2, 3))), "x": [0, 1]},
        )
        expected = Dataset(
            {"a": ("x", np.ones(2)), "b": ("y", np.ones(3))},
            {"c": (("x", "y"), np.zeros((2, 3))), "x": [0, 1]},
        )

        actual = Dataset(
            {"a": original["a"][:, 0], "b": original["a"][0].drop_vars("x")}
        )
        assert_identical(expected, actual)

        data = {"x": DataArray(0, coords={"y": 3}), "y": ("z", [1, 1, 1])}
        with pytest.raises(MergeError):
            Dataset(data)

        data = {"x": DataArray(0, coords={"y": 1}), "y": [1, 1]}
        actual = Dataset(data)
        expected = Dataset({"x": 0}, {"y": [1, 1]})
        assert_identical(expected, actual)

    def test_constructor_with_coords(self):
        with raises_regex(ValueError, "found in both data_vars and"):
            Dataset({"a": ("x", [1])}, {"a": ("x", [1])})

        ds = Dataset({}, {"a": ("x", [1])})
        assert not ds.data_vars
        assert list(ds.coords.keys()) == ["a"]

        mindex = pd.MultiIndex.from_product(
            [["a", "b"], [1, 2]], names=("level_1", "level_2")
        )
        with raises_regex(ValueError, "conflicting MultiIndex"):
            Dataset({}, {"x": mindex, "y": mindex})
            Dataset({}, {"x": mindex, "level_1": range(4)})

    def test_properties(self):
        ds = create_test_data()
        assert ds.dims == {"dim1": 8, "dim2": 9, "dim3": 10, "time": 20}
        assert list(ds.dims) == sorted(ds.dims)
        assert ds.sizes == ds.dims

        # These exact types aren't public API, but this makes sure we don't
        # change them inadvertently:
        assert isinstance(ds.dims, utils.Frozen)
        assert isinstance(ds.dims.mapping, utils.SortedKeysDict)
        assert type(ds.dims.mapping.mapping) is dict

        assert list(ds) == list(ds.data_vars)
        assert list(ds.keys()) == list(ds.data_vars)
        assert "aasldfjalskdfj" not in ds.variables
        assert "dim1" in repr(ds.variables)
        assert len(ds) == 3
        assert bool(ds)

        assert list(ds.data_vars) == ["var1", "var2", "var3"]
        assert list(ds.data_vars.keys()) == ["var1", "var2", "var3"]
        assert "var1" in ds.data_vars
        assert "dim1" not in ds.data_vars
        assert "numbers" not in ds.data_vars
        assert len(ds.data_vars) == 3

        assert set(ds.indexes) == {"dim2", "dim3", "time"}
        assert len(ds.indexes) == 3
        assert "dim2" in repr(ds.indexes)

        assert list(ds.coords) == ["time", "dim2", "dim3", "numbers"]
        assert "dim2" in ds.coords
        assert "numbers" in ds.coords
        assert "var1" not in ds.coords
        assert "dim1" not in ds.coords
        assert len(ds.coords) == 4

        assert Dataset({"x": np.int64(1), "y": np.float32([1, 2])}).nbytes == 16

    def test_asarray(self):
        ds = Dataset({"x": 0})
        with raises_regex(TypeError, "cannot directly convert"):
            np.asarray(ds)

    def test_get_index(self):
        ds = Dataset({"foo": (("x", "y"), np.zeros((2, 3)))}, coords={"x": ["a", "b"]})
        assert ds.get_index("x").equals(pd.Index(["a", "b"]))
        assert ds.get_index("y").equals(pd.Index([0, 1, 2]))
        with pytest.raises(KeyError):
            ds.get_index("z")

    def test_attr_access(self):
        ds = Dataset(
            {"tmin": ("x", [42], {"units": "Celcius"})}, attrs={"title": "My test data"}
        )
        assert_identical(ds.tmin, ds["tmin"])
        assert_identical(ds.tmin.x, ds.x)

        assert ds.title == ds.attrs["title"]
        assert ds.tmin.units == ds["tmin"].attrs["units"]

        assert {"tmin", "title"} <= set(dir(ds))
        assert "units" in set(dir(ds.tmin))

        # should defer to variable of same name
        ds.attrs["tmin"] = -999
        assert ds.attrs["tmin"] == -999
        assert_identical(ds.tmin, ds["tmin"])

    def test_variable(self):
        a = Dataset()
        d = np.random.random((10, 3))
        a["foo"] = (("time", "x"), d)
        assert "foo" in a.variables
        assert "foo" in a
        a["bar"] = (("time", "x"), d)
        # order of creation is preserved
        assert list(a.variables) == ["foo", "bar"]
        assert_array_equal(a["foo"].values, d)
        # try to add variable with dim (10,3) with data that's (3,10)
        with pytest.raises(ValueError):
            a["qux"] = (("time", "x"), d.T)

    def test_modify_inplace(self):
        a = Dataset()
        vec = np.random.random((10,))
        attributes = {"foo": "bar"}
        a["x"] = ("x", vec, attributes)
        assert "x" in a.coords
        assert isinstance(a.coords["x"].to_index(), pd.Index)
        assert_identical(a.coords["x"].variable, a.variables["x"])
        b = Dataset()
        b["x"] = ("x", vec, attributes)
        assert_identical(a["x"], b["x"])
        assert a.dims == b.dims
        # this should work
        a["x"] = ("x", vec[:5])
        a["z"] = ("x", np.arange(5))
        with pytest.raises(ValueError):
            # now it shouldn't, since there is a conflicting length
            a["x"] = ("x", vec[:4])
        arr = np.random.random((10, 1))
        scal = np.array(0)
        with pytest.raises(ValueError):
            a["y"] = ("y", arr)
        with pytest.raises(ValueError):
            a["y"] = ("y", scal)
        assert "y" not in a.dims

    def test_coords_properties(self):
        # use int64 for repr consistency on windows
        data = Dataset(
            {
                "x": ("x", np.array([-1, -2], "int64")),
                "y": ("y", np.array([0, 1, 2], "int64")),
                "foo": (["x", "y"], np.random.randn(2, 3)),
            },
            {"a": ("x", np.array([4, 5], "int64")), "b": np.int64(-10)},
        )

        assert 4 == len(data.coords)

        assert ["x", "y", "a", "b"] == list(data.coords)

        assert_identical(data.coords["x"].variable, data["x"].variable)
        assert_identical(data.coords["y"].variable, data["y"].variable)

        assert "x" in data.coords
        assert "a" in data.coords
        assert 0 not in data.coords
        assert "foo" not in data.coords

        with pytest.raises(KeyError):
            data.coords["foo"]
        with pytest.raises(KeyError):
            data.coords[0]

        expected = dedent(
            """\
        Coordinates:
          * x        (x) int64 -1 -2
          * y        (y) int64 0 1 2
            a        (x) int64 4 5
            b        int64 -10"""
        )
        actual = repr(data.coords)
        assert expected == actual

        assert {"x": 2, "y": 3} == data.coords.dims

    def test_coords_modify(self):
        data = Dataset(
            {
                "x": ("x", [-1, -2]),
                "y": ("y", [0, 1, 2]),
                "foo": (["x", "y"], np.random.randn(2, 3)),
            },
            {"a": ("x", [4, 5]), "b": -10},
        )

        actual = data.copy(deep=True)
        actual.coords["x"] = ("x", ["a", "b"])
        assert_array_equal(actual["x"], ["a", "b"])

        actual = data.copy(deep=True)
        actual.coords["z"] = ("z", ["a", "b"])
        assert_array_equal(actual["z"], ["a", "b"])

        actual = data.copy(deep=True)
        with raises_regex(ValueError, "conflicting sizes"):
            actual.coords["x"] = ("x", [-1])
        assert_identical(actual, data)  # should not be modified

        actual = data.copy()
        del actual.coords["b"]
        expected = data.reset_coords("b", drop=True)
        assert_identical(expected, actual)

        with pytest.raises(KeyError):
            del data.coords["not_found"]

        with pytest.raises(KeyError):
            del data.coords["foo"]

        actual = data.copy(deep=True)
        actual.coords.update({"c": 11})
        expected = data.merge({"c": 11}).set_coords("c")
        assert_identical(expected, actual)

        # regression test for GH3746
        del actual.coords["x"]
        assert "x" not in actual.indexes

    def test_update_index(self):
        actual = Dataset(coords={"x": [1, 2, 3]})
        actual["x"] = ["a", "b", "c"]
        assert actual.indexes["x"].equals(pd.Index(["a", "b", "c"]))

    def test_coords_setitem_with_new_dimension(self):
        actual = Dataset()
        actual.coords["foo"] = ("x", [1, 2, 3])
        expected = Dataset(coords={"foo": ("x", [1, 2, 3])})
        assert_identical(expected, actual)

    def test_coords_setitem_multiindex(self):
        data = create_test_multiindex()
        with raises_regex(ValueError, "conflicting MultiIndex"):
            data.coords["level_1"] = range(4)

    def test_coords_set(self):
        one_coord = Dataset({"x": ("x", [0]), "yy": ("x", [1]), "zzz": ("x", [2])})
        two_coords = Dataset({"zzz": ("x", [2])}, {"x": ("x", [0]), "yy": ("x", [1])})
        all_coords = Dataset(
            coords={"x": ("x", [0]), "yy": ("x", [1]), "zzz": ("x", [2])}
        )

        actual = one_coord.set_coords("x")
        assert_identical(one_coord, actual)
        actual = one_coord.set_coords(["x"])
        assert_identical(one_coord, actual)

        actual = one_coord.set_coords("yy")
        assert_identical(two_coords, actual)

        actual = one_coord.set_coords(["yy", "zzz"])
        assert_identical(all_coords, actual)

        actual = one_coord.reset_coords()
        assert_identical(one_coord, actual)
        actual = two_coords.reset_coords()
        assert_identical(one_coord, actual)
        actual = all_coords.reset_coords()
        assert_identical(one_coord, actual)

        actual = all_coords.reset_coords(["yy", "zzz"])
        assert_identical(one_coord, actual)
        actual = all_coords.reset_coords("zzz")
        assert_identical(two_coords, actual)

        with raises_regex(ValueError, "cannot remove index"):
            one_coord.reset_coords("x")

        actual = all_coords.reset_coords("zzz", drop=True)
        expected = all_coords.drop_vars("zzz")
        assert_identical(expected, actual)
        expected = two_coords.drop_vars("zzz")
        assert_identical(expected, actual)

    def test_coords_to_dataset(self):
        orig = Dataset({"foo": ("y", [-1, 0, 1])}, {"x": 10, "y": [2, 3, 4]})
        expected = Dataset(coords={"x": 10, "y": [2, 3, 4]})
        actual = orig.coords.to_dataset()
        assert_identical(expected, actual)

    def test_coords_merge(self):
        orig_coords = Dataset(coords={"a": ("x", [1, 2]), "x": [0, 1]}).coords
        other_coords = Dataset(coords={"b": ("x", ["a", "b"]), "x": [0, 1]}).coords
        expected = Dataset(
            coords={"a": ("x", [1, 2]), "b": ("x", ["a", "b"]), "x": [0, 1]}
        )
        actual = orig_coords.merge(other_coords)
        assert_identical(expected, actual)
        actual = other_coords.merge(orig_coords)
        assert_identical(expected, actual)

        other_coords = Dataset(coords={"x": ("x", ["a"])}).coords
        with pytest.raises(MergeError):
            orig_coords.merge(other_coords)
        other_coords = Dataset(coords={"x": ("x", ["a", "b"])}).coords
        with pytest.raises(MergeError):
            orig_coords.merge(other_coords)
        other_coords = Dataset(coords={"x": ("x", ["a", "b", "c"])}).coords
        with pytest.raises(MergeError):
            orig_coords.merge(other_coords)

        other_coords = Dataset(coords={"a": ("x", [8, 9])}).coords
        expected = Dataset(coords={"x": range(2)})
        actual = orig_coords.merge(other_coords)
        assert_identical(expected, actual)
        actual = other_coords.merge(orig_coords)
        assert_identical(expected, actual)

        other_coords = Dataset(coords={"x": np.nan}).coords
        actual = orig_coords.merge(other_coords)
        assert_identical(orig_coords.to_dataset(), actual)
        actual = other_coords.merge(orig_coords)
        assert_identical(orig_coords.to_dataset(), actual)

    def test_coords_merge_mismatched_shape(self):
        orig_coords = Dataset(coords={"a": ("x", [1, 1])}).coords
        other_coords = Dataset(coords={"a": 1}).coords
        expected = orig_coords.to_dataset()
        actual = orig_coords.merge(other_coords)
        assert_identical(expected, actual)

        other_coords = Dataset(coords={"a": ("y", [1])}).coords
        expected = Dataset(coords={"a": (["x", "y"], [[1], [1]])})
        actual = orig_coords.merge(other_coords)
        assert_identical(expected, actual)

        actual = other_coords.merge(orig_coords)
        assert_identical(expected.transpose(), actual)

        orig_coords = Dataset(coords={"a": ("x", [np.nan])}).coords
        other_coords = Dataset(coords={"a": np.nan}).coords
        expected = orig_coords.to_dataset()
        actual = orig_coords.merge(other_coords)
        assert_identical(expected, actual)

    def test_data_vars_properties(self):
        ds = Dataset()
        ds["foo"] = (("x",), [1.0])
        ds["bar"] = 2.0

        assert set(ds.data_vars) == {"foo", "bar"}
        assert "foo" in ds.data_vars
        assert "x" not in ds.data_vars
        assert_identical(ds["foo"], ds.data_vars["foo"])

        expected = dedent(
            """\
        Data variables:
            foo      (x) float64 1.0
            bar      float64 2.0"""
        )
        actual = repr(ds.data_vars)
        assert expected == actual

    def test_equals_and_identical(self):
        data = create_test_data(seed=42)
        assert data.equals(data)
        assert data.identical(data)

        data2 = create_test_data(seed=42)
        data2.attrs["foobar"] = "baz"
        assert data.equals(data2)
        assert not data.identical(data2)

        del data2["time"]
        assert not data.equals(data2)

        data = create_test_data(seed=42).rename({"var1": None})
        assert data.equals(data)
        assert data.identical(data)

        data2 = data.reset_coords()
        assert not data2.equals(data)
        assert not data2.identical(data)

    def test_equals_failures(self):
        data = create_test_data()
        assert not data.equals("foo")
        assert not data.identical(123)
        assert not data.broadcast_equals({1: 2})

    def test_broadcast_equals(self):
        data1 = Dataset(coords={"x": 0})
        data2 = Dataset(coords={"x": [0]})
        assert data1.broadcast_equals(data2)
        assert not data1.equals(data2)
        assert not data1.identical(data2)

    def test_attrs(self):
        data = create_test_data(seed=42)
        data.attrs = {"foobar": "baz"}
        assert data.attrs["foobar"], "baz"
        assert isinstance(data.attrs, dict)

    @requires_dask
    def test_chunk(self):
        data = create_test_data()
        for v in data.variables.values():
            assert isinstance(v.data, np.ndarray)
        assert data.chunks == {}

        reblocked = data.chunk()
        for k, v in reblocked.variables.items():
            if k in reblocked.dims:
                assert isinstance(v.data, np.ndarray)
            else:
                assert isinstance(v.data, da.Array)

        expected_chunks = {"dim1": (8,), "dim2": (9,), "dim3": (10,)}
        assert reblocked.chunks == expected_chunks

        def get_dask_names(ds):
            return {k: v.data.name for k, v in ds.items()}

        orig_dask_names = get_dask_names(reblocked)

        reblocked = data.chunk({"time": 5, "dim1": 5, "dim2": 5, "dim3": 5})
        # time is not a dim in any of the data_vars, so it
        # doesn't get chunked
        expected_chunks = {"dim1": (5, 3), "dim2": (5, 4), "dim3": (5, 5)}
        assert reblocked.chunks == expected_chunks

        # make sure dask names change when rechunking by different amounts
        # regression test for GH3350
        new_dask_names = get_dask_names(reblocked)
        for k, v in new_dask_names.items():
            assert v != orig_dask_names[k]

        reblocked = data.chunk(expected_chunks)
        assert reblocked.chunks == expected_chunks

        # reblock on already blocked data
        orig_dask_names = get_dask_names(reblocked)
        reblocked = reblocked.chunk(expected_chunks)
        new_dask_names = get_dask_names(reblocked)
        assert reblocked.chunks == expected_chunks
        assert_identical(reblocked, data)
        # recuhnking with same chunk sizes should not change names
        for k, v in new_dask_names.items():
            assert v == orig_dask_names[k]

        with raises_regex(ValueError, "some chunks"):
            data.chunk({"foo": 10})

    @requires_dask
    def test_dask_is_lazy(self):
        store = InaccessibleVariableDataStore()
        create_test_data().dump_to_store(store)
        ds = open_dataset(store).chunk()

        with pytest.raises(UnexpectedDataAccess):
            ds.load()
        with pytest.raises(UnexpectedDataAccess):
            ds["var1"].values

        # these should not raise UnexpectedDataAccess:
        ds.var1.data
        ds.isel(time=10)
        ds.isel(time=slice(10), dim1=[0]).isel(dim1=0, dim2=-1)
        ds.transpose()
        ds.mean()
        ds.fillna(0)
        ds.rename({"dim1": "foobar"})
        ds.set_coords("var1")
        ds.drop_vars("var1")

    def test_isel(self):
        data = create_test_data()
        slicers = {"dim1": slice(None, None, 2), "dim2": slice(0, 2)}
        ret = data.isel(**slicers)

        # Verify that only the specified dimension was altered
        assert list(data.dims) == list(ret.dims)
        for d in data.dims:
            if d in slicers:
                assert ret.dims[d] == np.arange(data.dims[d])[slicers[d]].size
            else:
                assert data.dims[d] == ret.dims[d]
        # Verify that the data is what we expect
        for v in data.variables:
            assert data[v].dims == ret[v].dims
            assert data[v].attrs == ret[v].attrs
            slice_list = [slice(None)] * data[v].values.ndim
            for d, s in slicers.items():
                if d in data[v].dims:
                    inds = np.nonzero(np.array(data[v].dims) == d)[0]
                    for ind in inds:
                        slice_list[ind] = s
            expected = data[v].values[tuple(slice_list)]
            actual = ret[v].values
            np.testing.assert_array_equal(expected, actual)

        with pytest.raises(ValueError):
            data.isel(not_a_dim=slice(0, 2))
        with raises_regex(
            ValueError,
            r"Dimensions {'not_a_dim'} do not exist. Expected "
            r"one or more of "
            r"[\w\W]*'time'[\w\W]*'dim\d'[\w\W]*'dim\d'[\w\W]*'dim\d'[\w\W]*",
        ):
            data.isel(not_a_dim=slice(0, 2))
        with pytest.warns(
            UserWarning,
            match=r"Dimensions {'not_a_dim'} do not exist. "
            r"Expected one or more of "
            r"[\w\W]*'time'[\w\W]*'dim\d'[\w\W]*'dim\d'[\w\W]*'dim\d'[\w\W]*",
        ):
            data.isel(not_a_dim=slice(0, 2), missing_dims="warn")
        assert_identical(data, data.isel(not_a_dim=slice(0, 2), missing_dims="ignore"))

        ret = data.isel(dim1=0)
        assert {"time": 20, "dim2": 9, "dim3": 10} == ret.dims
        assert set(data.data_vars) == set(ret.data_vars)
        assert set(data.coords) == set(ret.coords)
        assert set(data.indexes) == set(ret.indexes)

        ret = data.isel(time=slice(2), dim1=0, dim2=slice(5))
        assert {"time": 2, "dim2": 5, "dim3": 10} == ret.dims
        assert set(data.data_vars) == set(ret.data_vars)
        assert set(data.coords) == set(ret.coords)
        assert set(data.indexes) == set(ret.indexes)

        ret = data.isel(time=0, dim1=0, dim2=slice(5))
        assert {"dim2": 5, "dim3": 10} == ret.dims
        assert set(data.data_vars) == set(ret.data_vars)
        assert set(data.coords) == set(ret.coords)
        assert set(data.indexes) == set(list(ret.indexes) + ["time"])

    def test_isel_fancy(self):
        # isel with fancy indexing.
        data = create_test_data()

        pdim1 = [1, 2, 3]
        pdim2 = [4, 5, 1]
        pdim3 = [1, 2, 3]
        actual = data.isel(
            dim1=(("test_coord",), pdim1),
            dim2=(("test_coord",), pdim2),
            dim3=(("test_coord",), pdim3),
        )
        assert "test_coord" in actual.dims
        assert actual.coords["test_coord"].shape == (len(pdim1),)

        # Should work with DataArray
        actual = data.isel(
            dim1=DataArray(pdim1, dims="test_coord"),
            dim2=(("test_coord",), pdim2),
            dim3=(("test_coord",), pdim3),
        )
        assert "test_coord" in actual.dims
        assert actual.coords["test_coord"].shape == (len(pdim1),)
        expected = data.isel(
            dim1=(("test_coord",), pdim1),
            dim2=(("test_coord",), pdim2),
            dim3=(("test_coord",), pdim3),
        )
        assert_identical(actual, expected)

        # DataArray with coordinate
        idx1 = DataArray(pdim1, dims=["a"], coords={"a": np.random.randn(3)})
        idx2 = DataArray(pdim2, dims=["b"], coords={"b": np.random.randn(3)})
        idx3 = DataArray(pdim3, dims=["c"], coords={"c": np.random.randn(3)})
        # Should work with DataArray
        actual = data.isel(dim1=idx1, dim2=idx2, dim3=idx3)
        assert "a" in actual.dims
        assert "b" in actual.dims
        assert "c" in actual.dims
        assert "time" in actual.coords
        assert "dim2" in actual.coords
        assert "dim3" in actual.coords
        expected = data.isel(
            dim1=(("a",), pdim1), dim2=(("b",), pdim2), dim3=(("c",), pdim3)
        )
        expected = expected.assign_coords(a=idx1["a"], b=idx2["b"], c=idx3["c"])
        assert_identical(actual, expected)

        idx1 = DataArray(pdim1, dims=["a"], coords={"a": np.random.randn(3)})
        idx2 = DataArray(pdim2, dims=["a"])
        idx3 = DataArray(pdim3, dims=["a"])
        # Should work with DataArray
        actual = data.isel(dim1=idx1, dim2=idx2, dim3=idx3)
        assert "a" in actual.dims
        assert "time" in actual.coords
        assert "dim2" in actual.coords
        assert "dim3" in actual.coords
        expected = data.isel(
            dim1=(("a",), pdim1), dim2=(("a",), pdim2), dim3=(("a",), pdim3)
        )
        expected = expected.assign_coords(a=idx1["a"])
        assert_identical(actual, expected)

        actual = data.isel(dim1=(("points",), pdim1), dim2=(("points",), pdim2))
        assert "points" in actual.dims
        assert "dim3" in actual.dims
        assert "dim3" not in actual.data_vars
        np.testing.assert_array_equal(data["dim2"][pdim2], actual["dim2"])

        # test that the order of the indexers doesn't matter
        assert_identical(
            data.isel(dim1=(("points",), pdim1), dim2=(("points",), pdim2)),
            data.isel(dim2=(("points",), pdim2), dim1=(("points",), pdim1)),
        )
        # make sure we're raising errors in the right places
        with raises_regex(IndexError, "Dimensions of indexers mismatch"):
            data.isel(dim1=(("points",), [1, 2]), dim2=(("points",), [1, 2, 3]))
        with raises_regex(TypeError, "cannot use a Dataset"):
            data.isel(dim1=Dataset({"points": [1, 2]}))

        # test to be sure we keep around variables that were not indexed
        ds = Dataset({"x": [1, 2, 3, 4], "y": 0})
        actual = ds.isel(x=(("points",), [0, 1, 2]))
        assert_identical(ds["y"], actual["y"])

        # tests using index or DataArray as indexers
        stations = Dataset()
        stations["station"] = (("station",), ["A", "B", "C"])
        stations["dim1s"] = (("station",), [1, 2, 3])
        stations["dim2s"] = (("station",), [4, 5, 1])

        actual = data.isel(dim1=stations["dim1s"], dim2=stations["dim2s"])
        assert "station" in actual.coords
        assert "station" in actual.dims
        assert_identical(actual["station"].drop_vars(["dim2"]), stations["station"])

        with raises_regex(ValueError, "conflicting values for "):
            data.isel(
                dim1=DataArray(
                    [0, 1, 2], dims="station", coords={"station": [0, 1, 2]}
                ),
                dim2=DataArray(
                    [0, 1, 2], dims="station", coords={"station": [0, 1, 3]}
                ),
            )

        # multi-dimensional selection
        stations = Dataset()
        stations["a"] = (("a",), ["A", "B", "C"])
        stations["b"] = (("b",), [0, 1])
        stations["dim1s"] = (("a", "b"), [[1, 2], [2, 3], [3, 4]])
        stations["dim2s"] = (("a",), [4, 5, 1])
        actual = data.isel(dim1=stations["dim1s"], dim2=stations["dim2s"])
        assert "a" in actual.coords
        assert "a" in actual.dims
        assert "b" in actual.coords
        assert "b" in actual.dims
        assert "dim2" in actual.coords
        assert "a" in actual["dim2"].dims

        assert_identical(actual["a"].drop_vars(["dim2"]), stations["a"])
        assert_identical(actual["b"], stations["b"])
        expected_var1 = data["var1"].variable[
            stations["dim1s"].variable, stations["dim2s"].variable
        ]
        expected_var2 = data["var2"].variable[
            stations["dim1s"].variable, stations["dim2s"].variable
        ]
        expected_var3 = data["var3"].variable[slice(None), stations["dim1s"].variable]
        assert_equal(actual["a"].drop_vars("dim2"), stations["a"])
        assert_array_equal(actual["var1"], expected_var1)
        assert_array_equal(actual["var2"], expected_var2)
        assert_array_equal(actual["var3"], expected_var3)

    def test_isel_dataarray(self):
        """ Test for indexing by DataArray """
        data = create_test_data()
        # indexing with DataArray with same-name coordinates.
        indexing_da = DataArray(
            np.arange(1, 4), dims=["dim1"], coords={"dim1": np.random.randn(3)}
        )
        actual = data.isel(dim1=indexing_da)
        assert_identical(indexing_da["dim1"], actual["dim1"])
        assert_identical(data["dim2"], actual["dim2"])

        # Conflict in the dimension coordinate
        indexing_da = DataArray(
            np.arange(1, 4), dims=["dim2"], coords={"dim2": np.random.randn(3)}
        )
        with raises_regex(IndexError, "dimension coordinate 'dim2'"):
            actual = data.isel(dim2=indexing_da)
        # Also the case for DataArray
        with raises_regex(IndexError, "dimension coordinate 'dim2'"):
            actual = data["var2"].isel(dim2=indexing_da)
        with raises_regex(IndexError, "dimension coordinate 'dim2'"):
            data["dim2"].isel(dim2=indexing_da)

        # same name coordinate which does not conflict
        indexing_da = DataArray(
            np.arange(1, 4), dims=["dim2"], coords={"dim2": data["dim2"].values[1:4]}
        )
        actual = data.isel(dim2=indexing_da)
        assert_identical(actual["dim2"], indexing_da["dim2"])

        # Silently drop conflicted (non-dimensional) coordinate of indexer
        indexing_da = DataArray(
            np.arange(1, 4),
            dims=["dim2"],
            coords={
                "dim2": data["dim2"].values[1:4],
                "numbers": ("dim2", np.arange(2, 5)),
            },
        )
        actual = data.isel(dim2=indexing_da)
        assert_identical(actual["numbers"], data["numbers"])

        # boolean data array with coordinate with the same name
        indexing_da = DataArray(
            np.arange(1, 10), dims=["dim2"], coords={"dim2": data["dim2"].values}
        )
        indexing_da = indexing_da < 3
        actual = data.isel(dim2=indexing_da)
        assert_identical(actual["dim2"], data["dim2"][:2])

        # boolean data array with non-dimensioncoordinate
        indexing_da = DataArray(
            np.arange(1, 10),
            dims=["dim2"],
            coords={
                "dim2": data["dim2"].values,
                "non_dim": (("dim2",), np.random.randn(9)),
                "non_dim2": 0,
            },
        )
        indexing_da = indexing_da < 3
        actual = data.isel(dim2=indexing_da)
        assert_identical(
            actual["dim2"].drop_vars("non_dim").drop_vars("non_dim2"), data["dim2"][:2]
        )
        assert_identical(actual["non_dim"], indexing_da["non_dim"][:2])
        assert_identical(actual["non_dim2"], indexing_da["non_dim2"])

        # non-dimension coordinate will be also attached
        indexing_da = DataArray(
            np.arange(1, 4),
            dims=["dim2"],
            coords={"non_dim": (("dim2",), np.random.randn(3))},
        )
        actual = data.isel(dim2=indexing_da)
        assert "non_dim" in actual
        assert "non_dim" in actual.coords

        # Index by a scalar DataArray
        indexing_da = DataArray(3, dims=[], coords={"station": 2})
        actual = data.isel(dim2=indexing_da)
        assert "station" in actual
        actual = data.isel(dim2=indexing_da["station"])
        assert "station" in actual

        # indexer generated from coordinates
        indexing_ds = Dataset({}, coords={"dim2": [0, 1, 2]})
        with raises_regex(IndexError, "dimension coordinate 'dim2'"):
            actual = data.isel(dim2=indexing_ds["dim2"])

    def test_sel(self):
        data = create_test_data()
        int_slicers = {"dim1": slice(None, None, 2), "dim2": slice(2), "dim3": slice(3)}
        loc_slicers = {
            "dim1": slice(None, None, 2),
            "dim2": slice(0, 0.5),
            "dim3": slice("a", "c"),
        }
        assert_equal(data.isel(**int_slicers), data.sel(**loc_slicers))
        data["time"] = ("time", pd.date_range("2000-01-01", periods=20))
        assert_equal(data.isel(time=0), data.sel(time="2000-01-01"))
        assert_equal(
            data.isel(time=slice(10)), data.sel(time=slice("2000-01-01", "2000-01-10"))
        )
        assert_equal(data, data.sel(time=slice("1999", "2005")))
        times = pd.date_range("2000-01-01", periods=3)
        assert_equal(data.isel(time=slice(3)), data.sel(time=times))
        assert_equal(
            data.isel(time=slice(3)), data.sel(time=(data["time.dayofyear"] <= 3))
        )

        td = pd.to_timedelta(np.arange(3), unit="days")
        data = Dataset({"x": ("td", np.arange(3)), "td": td})
        assert_equal(data, data.sel(td=td))
        assert_equal(data, data.sel(td=slice("3 days")))
        assert_equal(data.isel(td=0), data.sel(td=pd.Timedelta("0 days")))
        assert_equal(data.isel(td=0), data.sel(td=pd.Timedelta("0h")))
        assert_equal(data.isel(td=slice(1, 3)), data.sel(td=slice("1 days", "2 days")))

    def test_sel_dataarray(self):
        data = create_test_data()

        ind = DataArray([0.0, 0.5, 1.0], dims=["dim2"])
        actual = data.sel(dim2=ind)
        assert_equal(actual, data.isel(dim2=[0, 1, 2]))

        # with different dimension
        ind = DataArray([0.0, 0.5, 1.0], dims=["new_dim"])
        actual = data.sel(dim2=ind)
        expected = data.isel(dim2=Variable("new_dim", [0, 1, 2]))
        assert "new_dim" in actual.dims
        assert_equal(actual, expected)

        # Multi-dimensional
        ind = DataArray([[0.0], [0.5], [1.0]], dims=["new_dim", "new_dim2"])
        actual = data.sel(dim2=ind)
        expected = data.isel(dim2=Variable(("new_dim", "new_dim2"), [[0], [1], [2]]))
        assert "new_dim" in actual.dims
        assert "new_dim2" in actual.dims
        assert_equal(actual, expected)

        # with coordinate
        ind = DataArray(
            [0.0, 0.5, 1.0], dims=["new_dim"], coords={"new_dim": ["a", "b", "c"]}
        )
        actual = data.sel(dim2=ind)
        expected = data.isel(dim2=[0, 1, 2]).rename({"dim2": "new_dim"})
        assert "new_dim" in actual.dims
        assert "new_dim" in actual.coords
        assert_equal(
            actual.drop_vars("new_dim").drop_vars("dim2"), expected.drop_vars("new_dim")
        )
        assert_equal(actual["new_dim"].drop_vars("dim2"), ind["new_dim"])

        # with conflicted coordinate (silently ignored)
        ind = DataArray(
            [0.0, 0.5, 1.0], dims=["dim2"], coords={"dim2": ["a", "b", "c"]}
        )
        actual = data.sel(dim2=ind)
        expected = data.isel(dim2=[0, 1, 2])
        assert_equal(actual, expected)

        # with conflicted coordinate (silently ignored)
        ind = DataArray(
            [0.0, 0.5, 1.0],
            dims=["new_dim"],
            coords={"new_dim": ["a", "b", "c"], "dim2": 3},
        )
        actual = data.sel(dim2=ind)
        assert_equal(
            actual["new_dim"].drop_vars("dim2"), ind["new_dim"].drop_vars("dim2")
        )
        expected = data.isel(dim2=[0, 1, 2])
        expected["dim2"] = (("new_dim"), expected["dim2"].values)
        assert_equal(actual["dim2"].drop_vars("new_dim"), expected["dim2"])
        assert actual["var1"].dims == ("dim1", "new_dim")

        # with non-dimensional coordinate
        ind = DataArray(
            [0.0, 0.5, 1.0],
            dims=["dim2"],
            coords={
                "dim2": ["a", "b", "c"],
                "numbers": ("dim2", [0, 1, 2]),
                "new_dim": ("dim2", [1.1, 1.2, 1.3]),
            },
        )
        actual = data.sel(dim2=ind)
        expected = data.isel(dim2=[0, 1, 2])
        assert_equal(actual.drop_vars("new_dim"), expected)
        assert np.allclose(actual["new_dim"].values, ind["new_dim"].values)

    def test_sel_dataarray_mindex(self):
        midx = pd.MultiIndex.from_product([list("abc"), [0, 1]], names=("one", "two"))
        mds = xr.Dataset(
            {"var": (("x", "y"), np.random.rand(6, 3))},
            coords={"x": midx, "y": range(3)},
        )

        actual_isel = mds.isel(x=xr.DataArray(np.arange(3), dims="x"))
        actual_sel = mds.sel(x=DataArray(mds.indexes["x"][:3], dims="x"))
        assert actual_isel["x"].dims == ("x",)
        assert actual_sel["x"].dims == ("x",)
        assert_identical(actual_isel, actual_sel)

        actual_isel = mds.isel(x=xr.DataArray(np.arange(3), dims="z"))
        actual_sel = mds.sel(x=Variable("z", mds.indexes["x"][:3]))
        assert actual_isel["x"].dims == ("z",)
        assert actual_sel["x"].dims == ("z",)
        assert_identical(actual_isel, actual_sel)

        # with coordinate
        actual_isel = mds.isel(
            x=xr.DataArray(np.arange(3), dims="z", coords={"z": [0, 1, 2]})
        )
        actual_sel = mds.sel(
            x=xr.DataArray(mds.indexes["x"][:3], dims="z", coords={"z": [0, 1, 2]})
        )
        assert actual_isel["x"].dims == ("z",)
        assert actual_sel["x"].dims == ("z",)
        assert_identical(actual_isel, actual_sel)

        # Vectorized indexing with level-variables raises an error
        with raises_regex(ValueError, "Vectorized selection is "):
            mds.sel(one=["a", "b"])

        with raises_regex(
            ValueError,
            "Vectorized selection is not available along MultiIndex variable: x",
        ):
            mds.sel(
                x=xr.DataArray(
                    [np.array(midx[:2]), np.array(midx[-2:])], dims=["a", "b"]
                )
            )

    def test_sel_categorical(self):
        ind = pd.Series(["foo", "bar"], dtype="category")
        df = pd.DataFrame({"ind": ind, "values": [1, 2]})
        ds = df.set_index("ind").to_xarray()
        actual = ds.sel(ind="bar")
        expected = ds.isel(ind=1)
        assert_identical(expected, actual)

    def test_sel_categorical_error(self):
        ind = pd.Series(["foo", "bar"], dtype="category")
        df = pd.DataFrame({"ind": ind, "values": [1, 2]})
        ds = df.set_index("ind").to_xarray()
        with pytest.raises(ValueError):
            ds.sel(ind="bar", method="nearest")
        with pytest.raises(ValueError):
            ds.sel(ind="bar", tolerance="nearest")

    def test_categorical_index(self):
        cat = pd.CategoricalIndex(
            ["foo", "bar", "foo"],
            categories=["foo", "bar", "baz", "qux", "quux", "corge"],
        )
        ds = xr.Dataset(
            {"var": ("cat", np.arange(3))},
            coords={"cat": ("cat", cat), "c": ("cat", [0, 1, 1])},
        )
        # test slice
        actual = ds.sel(cat="foo")
        expected = ds.isel(cat=[0, 2])
        assert_identical(expected, actual)
        # make sure the conversion to the array works
        actual = ds.sel(cat="foo")["cat"].values
        assert (actual == np.array(["foo", "foo"])).all()

        ds = ds.set_index(index=["cat", "c"])
        actual = ds.unstack("index")
        assert actual["var"].shape == (2, 2)

    def test_categorical_reindex(self):
        cat = pd.CategoricalIndex(
            ["foo", "bar", "baz"],
            categories=["foo", "bar", "baz", "qux", "quux", "corge"],
        )
        ds = xr.Dataset(
            {"var": ("cat", np.arange(3))},
            coords={"cat": ("cat", cat), "c": ("cat", [0, 1, 2])},
        )
        actual = ds.reindex(cat=["foo"])["cat"].values
        assert (actual == np.array(["foo"])).all()

    def test_categorical_multiindex(self):
        i1 = pd.Series([0, 0])
        cat = pd.CategoricalDtype(categories=["foo", "baz", "bar"])
        i2 = pd.Series(["baz", "bar"], dtype=cat)

        df = pd.DataFrame({"i1": i1, "i2": i2, "values": [1, 2]}).set_index(
            ["i1", "i2"]
        )
        actual = df.to_xarray()
        assert actual["values"].shape == (1, 2)

    def test_sel_drop(self):
        data = Dataset({"foo": ("x", [1, 2, 3])}, {"x": [0, 1, 2]})
        expected = Dataset({"foo": 1})
        selected = data.sel(x=0, drop=True)
        assert_identical(expected, selected)

        expected = Dataset({"foo": 1}, {"x": 0})
        selected = data.sel(x=0, drop=False)
        assert_identical(expected, selected)

        data = Dataset({"foo": ("x", [1, 2, 3])})
        expected = Dataset({"foo": 1})
        selected = data.sel(x=0, drop=True)
        assert_identical(expected, selected)

    def test_isel_drop(self):
        data = Dataset({"foo": ("x", [1, 2, 3])}, {"x": [0, 1, 2]})
        expected = Dataset({"foo": 1})
        selected = data.isel(x=0, drop=True)
        assert_identical(expected, selected)

        expected = Dataset({"foo": 1}, {"x": 0})
        selected = data.isel(x=0, drop=False)
        assert_identical(expected, selected)

    def test_head(self):
        data = create_test_data()

        expected = data.isel(time=slice(5), dim2=slice(6))
        actual = data.head(time=5, dim2=6)
        assert_equal(expected, actual)

        expected = data.isel(time=slice(0))
        actual = data.head(time=0)
        assert_equal(expected, actual)

        expected = data.isel({dim: slice(6) for dim in data.dims})
        actual = data.head(6)
        assert_equal(expected, actual)

        expected = data.isel({dim: slice(5) for dim in data.dims})
        actual = data.head()
        assert_equal(expected, actual)

        with raises_regex(TypeError, "either dict-like or a single int"):
            data.head([3])
        with raises_regex(TypeError, "expected integer type"):
            data.head(dim2=3.1)
        with raises_regex(ValueError, "expected positive int"):
            data.head(time=-3)

    def test_tail(self):
        data = create_test_data()

        expected = data.isel(time=slice(-5, None), dim2=slice(-6, None))
        actual = data.tail(time=5, dim2=6)
        assert_equal(expected, actual)

        expected = data.isel(dim1=slice(0))
        actual = data.tail(dim1=0)
        assert_equal(expected, actual)

        expected = data.isel({dim: slice(-6, None) for dim in data.dims})
        actual = data.tail(6)
        assert_equal(expected, actual)

        expected = data.isel({dim: slice(-5, None) for dim in data.dims})
        actual = data.tail()
        assert_equal(expected, actual)

        with raises_regex(TypeError, "either dict-like or a single int"):
            data.tail([3])
        with raises_regex(TypeError, "expected integer type"):
            data.tail(dim2=3.1)
        with raises_regex(ValueError, "expected positive int"):
            data.tail(time=-3)

    def test_thin(self):
        data = create_test_data()

        expected = data.isel(time=slice(None, None, 5), dim2=slice(None, None, 6))
        actual = data.thin(time=5, dim2=6)
        assert_equal(expected, actual)

        expected = data.isel({dim: slice(None, None, 6) for dim in data.dims})
        actual = data.thin(6)
        assert_equal(expected, actual)

        with raises_regex(TypeError, "either dict-like or a single int"):
            data.thin([3])
        with raises_regex(TypeError, "expected integer type"):
            data.thin(dim2=3.1)
        with raises_regex(ValueError, "cannot be zero"):
            data.thin(time=0)
        with raises_regex(ValueError, "expected positive int"):
            data.thin(time=-3)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_sel_fancy(self):
        data = create_test_data()

        # add in a range() index
        data["dim1"] = data.dim1

        pdim1 = [1, 2, 3]
        pdim2 = [4, 5, 1]
        pdim3 = [1, 2, 3]
        expected = data.isel(
            dim1=Variable(("test_coord",), pdim1),
            dim2=Variable(("test_coord",), pdim2),
            dim3=Variable(("test_coord"), pdim3),
        )
        actual = data.sel(
            dim1=Variable(("test_coord",), data.dim1[pdim1]),
            dim2=Variable(("test_coord",), data.dim2[pdim2]),
            dim3=Variable(("test_coord",), data.dim3[pdim3]),
        )
        assert_identical(expected, actual)

        # DataArray Indexer
        idx_t = DataArray(
            data["time"][[3, 2, 1]].values, dims=["a"], coords={"a": ["a", "b", "c"]}
        )
        idx_2 = DataArray(
            data["dim2"][[3, 2, 1]].values, dims=["a"], coords={"a": ["a", "b", "c"]}
        )
        idx_3 = DataArray(
            data["dim3"][[3, 2, 1]].values, dims=["a"], coords={"a": ["a", "b", "c"]}
        )
        actual = data.sel(time=idx_t, dim2=idx_2, dim3=idx_3)
        expected = data.isel(
            time=Variable(("a",), [3, 2, 1]),
            dim2=Variable(("a",), [3, 2, 1]),
            dim3=Variable(("a",), [3, 2, 1]),
        )
        expected = expected.assign_coords(a=idx_t["a"])
        assert_identical(expected, actual)

        idx_t = DataArray(
            data["time"][[3, 2, 1]].values, dims=["a"], coords={"a": ["a", "b", "c"]}
        )
        idx_2 = DataArray(
            data["dim2"][[2, 1, 3]].values, dims=["b"], coords={"b": [0, 1, 2]}
        )
        idx_3 = DataArray(
            data["dim3"][[1, 2, 1]].values, dims=["c"], coords={"c": [0.0, 1.1, 2.2]}
        )
        actual = data.sel(time=idx_t, dim2=idx_2, dim3=idx_3)
        expected = data.isel(
            time=Variable(("a",), [3, 2, 1]),
            dim2=Variable(("b",), [2, 1, 3]),
            dim3=Variable(("c",), [1, 2, 1]),
        )
        expected = expected.assign_coords(a=idx_t["a"], b=idx_2["b"], c=idx_3["c"])
        assert_identical(expected, actual)

        # test from sel_points
        data = Dataset({"foo": (("x", "y"), np.arange(9).reshape(3, 3))})
        data.coords.update({"x": [0, 1, 2], "y": [0, 1, 2]})

        expected = Dataset(
            {"foo": ("points", [0, 4, 8])},
            coords={
                "x": Variable(("points",), [0, 1, 2]),
                "y": Variable(("points",), [0, 1, 2]),
            },
        )
        actual = data.sel(
            x=Variable(("points",), [0, 1, 2]), y=Variable(("points",), [0, 1, 2])
        )
        assert_identical(expected, actual)

        expected.coords.update({"x": ("points", [0, 1, 2]), "y": ("points", [0, 1, 2])})
        actual = data.sel(
            x=Variable(("points",), [0.1, 1.1, 2.5]),
            y=Variable(("points",), [0, 1.2, 2.0]),
            method="pad",
        )
        assert_identical(expected, actual)

        idx_x = DataArray([0, 1, 2], dims=["a"], coords={"a": ["a", "b", "c"]})
        idx_y = DataArray([0, 2, 1], dims=["b"], coords={"b": [0, 3, 6]})
        expected_ary = data["foo"][[0, 1, 2], [0, 2, 1]]
        actual = data.sel(x=idx_x, y=idx_y)
        assert_array_equal(expected_ary, actual["foo"])
        assert_identical(actual["a"].drop_vars("x"), idx_x["a"])
        assert_identical(actual["b"].drop_vars("y"), idx_y["b"])

        with pytest.raises(KeyError):
            data.sel(x=[2.5], y=[2.0], method="pad", tolerance=1e-3)

    def test_sel_method(self):
        data = create_test_data()

        expected = data.sel(dim2=1)
        actual = data.sel(dim2=0.95, method="nearest")
        assert_identical(expected, actual)

        actual = data.sel(dim2=0.95, method="nearest", tolerance=1)
        assert_identical(expected, actual)

        with pytest.raises(KeyError):
            actual = data.sel(dim2=np.pi, method="nearest", tolerance=0)

        expected = data.sel(dim2=[1.5])
        actual = data.sel(dim2=[1.45], method="backfill")
        assert_identical(expected, actual)

        with raises_regex(NotImplementedError, "slice objects"):
            data.sel(dim2=slice(1, 3), method="ffill")

        with raises_regex(TypeError, "``method``"):
            # this should not pass silently
            data.sel(method=data)

        # cannot pass method if there is no associated coordinate
        with raises_regex(ValueError, "cannot supply"):
            data.sel(dim1=0, method="nearest")

    def test_loc(self):
        data = create_test_data()
        expected = data.sel(dim3="a")
        actual = data.loc[dict(dim3="a")]
        assert_identical(expected, actual)
        with raises_regex(TypeError, "can only lookup dict"):
            data.loc["a"]
        with pytest.raises(TypeError):
            data.loc[dict(dim3="a")] = 0

    def test_selection_multiindex(self):
        mindex = pd.MultiIndex.from_product(
            [["a", "b"], [1, 2], [-1, -2]], names=("one", "two", "three")
        )
        mdata = Dataset(data_vars={"var": ("x", range(8))}, coords={"x": mindex})

        def test_sel(lab_indexer, pos_indexer, replaced_idx=False, renamed_dim=None):
            ds = mdata.sel(x=lab_indexer)
            expected_ds = mdata.isel(x=pos_indexer)
            if not replaced_idx:
                assert_identical(ds, expected_ds)
            else:
                if renamed_dim:
                    assert ds["var"].dims[0] == renamed_dim
                    ds = ds.rename({renamed_dim: "x"})
                assert_identical(ds["var"].variable, expected_ds["var"].variable)
                assert not ds["x"].equals(expected_ds["x"])

        test_sel(("a", 1, -1), 0)
        test_sel(("b", 2, -2), -1)
        test_sel(("a", 1), [0, 1], replaced_idx=True, renamed_dim="three")
        test_sel(("a",), range(4), replaced_idx=True)
        test_sel("a", range(4), replaced_idx=True)
        test_sel([("a", 1, -1), ("b", 2, -2)], [0, 7])
        test_sel(slice("a", "b"), range(8))
        test_sel(slice(("a", 1), ("b", 1)), range(6))
        test_sel({"one": "a", "two": 1, "three": -1}, 0)
        test_sel({"one": "a", "two": 1}, [0, 1], replaced_idx=True, renamed_dim="three")
        test_sel({"one": "a"}, range(4), replaced_idx=True)

        assert_identical(mdata.loc[{"x": {"one": "a"}}], mdata.sel(x={"one": "a"}))
        assert_identical(mdata.loc[{"x": "a"}], mdata.sel(x="a"))
        assert_identical(mdata.loc[{"x": ("a", 1)}], mdata.sel(x=("a", 1)))
        assert_identical(mdata.loc[{"x": ("a", 1, -1)}], mdata.sel(x=("a", 1, -1)))

        assert_identical(mdata.sel(x={"one": "a", "two": 1}), mdata.sel(one="a", two=1))

    def test_broadcast_like(self):
        original1 = DataArray(
            np.random.randn(5), [("x", range(5))], name="a"
        ).to_dataset()

        original2 = DataArray(np.random.randn(6), [("y", range(6))], name="b")

        expected1, expected2 = broadcast(original1, original2)

        assert_identical(
            original1.broadcast_like(original2), expected1.transpose("y", "x")
        )

        assert_identical(original2.broadcast_like(original1), expected2)

    def test_reindex_like(self):
        data = create_test_data()
        data["letters"] = ("dim3", 10 * ["a"])

        expected = data.isel(dim1=slice(10), time=slice(13))
        actual = data.reindex_like(expected)
        assert_identical(actual, expected)

        expected = data.copy(deep=True)
        expected["dim3"] = ("dim3", list("cdefghijkl"))
        expected["var3"][:-2] = expected["var3"][2:].values
        expected["var3"][-2:] = np.nan
        expected["letters"] = expected["letters"].astype(object)
        expected["letters"][-2:] = np.nan
        expected["numbers"] = expected["numbers"].astype(float)
        expected["numbers"][:-2] = expected["numbers"][2:].values
        expected["numbers"][-2:] = np.nan
        actual = data.reindex_like(expected)
        assert_identical(actual, expected)

    def test_reindex(self):
        data = create_test_data()
        assert_identical(data, data.reindex())

        expected = data.assign_coords(dim1=data["dim1"])
        actual = data.reindex(dim1=data["dim1"])
        assert_identical(actual, expected)

        actual = data.reindex(dim1=data["dim1"].values)
        assert_identical(actual, expected)

        actual = data.reindex(dim1=data["dim1"].to_index())
        assert_identical(actual, expected)

        with raises_regex(ValueError, "cannot reindex or align along dimension"):
            data.reindex(dim1=data["dim1"][:5])

        expected = data.isel(dim2=slice(5))
        actual = data.reindex(dim2=data["dim2"][:5])
        assert_identical(actual, expected)

        # test dict-like argument
        actual = data.reindex({"dim2": data["dim2"]})
        expected = data
        assert_identical(actual, expected)
        with raises_regex(ValueError, "cannot specify both"):
            data.reindex({"x": 0}, x=0)
        with raises_regex(ValueError, "dictionary"):
            data.reindex("foo")

        # invalid dimension
        with raises_regex(ValueError, "invalid reindex dim"):
            data.reindex(invalid=0)

        # out of order
        expected = data.sel(dim2=data["dim2"][:5:-1])
        actual = data.reindex(dim2=data["dim2"][:5:-1])
        assert_identical(actual, expected)

        # multiple fill values
        expected = data.reindex(dim2=[0.1, 2.1, 3.1, 4.1]).assign(
            var1=lambda ds: ds.var1.copy(data=[[-10, -10, -10, -10]] * len(ds.dim1)),
            var2=lambda ds: ds.var2.copy(data=[[-20, -20, -20, -20]] * len(ds.dim1)),
        )
        actual = data.reindex(
            dim2=[0.1, 2.1, 3.1, 4.1], fill_value={"var1": -10, "var2": -20}
        )
        assert_identical(actual, expected)
        # use the default value
        expected = data.reindex(dim2=[0.1, 2.1, 3.1, 4.1]).assign(
            var1=lambda ds: ds.var1.copy(data=[[-10, -10, -10, -10]] * len(ds.dim1)),
            var2=lambda ds: ds.var2.copy(
                data=[[np.nan, np.nan, np.nan, np.nan]] * len(ds.dim1)
            ),
        )
        actual = data.reindex(dim2=[0.1, 2.1, 3.1, 4.1], fill_value={"var1": -10})
        assert_identical(actual, expected)

        # regression test for #279
        expected = Dataset({"x": ("time", np.random.randn(5))}, {"time": range(5)})
        time2 = DataArray(np.arange(5), dims="time2")
        with pytest.raises(ValueError):
            actual = expected.reindex(time=time2)

        # another regression test
        ds = Dataset(
            {"foo": (["x", "y"], np.zeros((3, 4)))}, {"x": range(3), "y": range(4)}
        )
        expected = Dataset(
            {"foo": (["x", "y"], np.zeros((3, 2)))}, {"x": [0, 1, 3], "y": [0, 1]}
        )
        expected["foo"][-1] = np.nan
        actual = ds.reindex(x=[0, 1, 3], y=[0, 1])
        assert_identical(expected, actual)

    def test_reindex_warning(self):
        data = create_test_data()

        with pytest.raises(ValueError):
            # DataArray with different dimension raises Future warning
            ind = xr.DataArray([0.0, 1.0], dims=["new_dim"], name="ind")
            data.reindex(dim2=ind)

        # Should not warn
        ind = xr.DataArray([0.0, 1.0], dims=["dim2"], name="ind")
        with pytest.warns(None) as ws:
            data.reindex(dim2=ind)
            assert len(ws) == 0

    def test_reindex_variables_copied(self):
        data = create_test_data()
        reindexed_data = data.reindex(copy=False)
        for k in data.variables:
            assert reindexed_data.variables[k] is not data.variables[k]

    def test_reindex_method(self):
        ds = Dataset({"x": ("y", [10, 20]), "y": [0, 1]})
        y = [-0.5, 0.5, 1.5]
        actual = ds.reindex(y=y, method="backfill")
        expected = Dataset({"x": ("y", [10, 20, np.nan]), "y": y})
        assert_identical(expected, actual)

        actual = ds.reindex(y=y, method="backfill", tolerance=0.1)
        expected = Dataset({"x": ("y", 3 * [np.nan]), "y": y})
        assert_identical(expected, actual)

        actual = ds.reindex(y=y, method="pad")
        expected = Dataset({"x": ("y", [np.nan, 10, 20]), "y": y})
        assert_identical(expected, actual)

        alt = Dataset({"y": y})
        actual = ds.reindex_like(alt, method="pad")
        assert_identical(expected, actual)

    @pytest.mark.parametrize("fill_value", [dtypes.NA, 2, 2.0, {"x": 2, "z": 1}])
    def test_reindex_fill_value(self, fill_value):
        ds = Dataset({"x": ("y", [10, 20]), "z": ("y", [-20, -10]), "y": [0, 1]})
        y = [0, 1, 2]
        actual = ds.reindex(y=y, fill_value=fill_value)
        if fill_value == dtypes.NA:
            # if we supply the default, we expect the missing value for a
            # float array
            fill_value_x = fill_value_z = np.nan
        elif isinstance(fill_value, dict):
            fill_value_x = fill_value["x"]
            fill_value_z = fill_value["z"]
        else:
            fill_value_x = fill_value_z = fill_value
        expected = Dataset(
            {
                "x": ("y", [10, 20, fill_value_x]),
                "z": ("y", [-20, -10, fill_value_z]),
                "y": y,
            }
        )
        assert_identical(expected, actual)

    @pytest.mark.parametrize("fill_value", [dtypes.NA, 2, 2.0, {"x": 2, "z": 1}])
    def test_reindex_like_fill_value(self, fill_value):
        ds = Dataset({"x": ("y", [10, 20]), "z": ("y", [-20, -10]), "y": [0, 1]})
        y = [0, 1, 2]
        alt = Dataset({"y": y})
        actual = ds.reindex_like(alt, fill_value=fill_value)
        if fill_value == dtypes.NA:
            # if we supply the default, we expect the missing value for a
            # float array
            fill_value_x = fill_value_z = np.nan
        elif isinstance(fill_value, dict):
            fill_value_x = fill_value["x"]
            fill_value_z = fill_value["z"]
        else:
            fill_value_x = fill_value_z = fill_value
        expected = Dataset(
            {
                "x": ("y", [10, 20, fill_value_x]),
                "z": ("y", [-20, -10, fill_value_z]),
                "y": y,
            }
        )
        assert_identical(expected, actual)

    @pytest.mark.parametrize("dtype", [str, bytes])
    def test_reindex_str_dtype(self, dtype):
        data = Dataset({"data": ("x", [1, 2]), "x": np.array(["a", "b"], dtype=dtype)})

        actual = data.reindex(x=data.x)
        expected = data

        assert_identical(expected, actual)
        assert actual.x.dtype == expected.x.dtype

    @pytest.mark.parametrize("fill_value", [dtypes.NA, 2, 2.0, {"foo": 2, "bar": 1}])
    def test_align_fill_value(self, fill_value):
        x = Dataset({"foo": DataArray([1, 2], dims=["x"], coords={"x": [1, 2]})})
        y = Dataset({"bar": DataArray([1, 2], dims=["x"], coords={"x": [1, 3]})})
        x2, y2 = align(x, y, join="outer", fill_value=fill_value)
        if fill_value == dtypes.NA:
            # if we supply the default, we expect the missing value for a
            # float array
            fill_value_foo = fill_value_bar = np.nan
        elif isinstance(fill_value, dict):
            fill_value_foo = fill_value["foo"]
            fill_value_bar = fill_value["bar"]
        else:
            fill_value_foo = fill_value_bar = fill_value

        expected_x2 = Dataset(
            {
                "foo": DataArray(
                    [1, 2, fill_value_foo], dims=["x"], coords={"x": [1, 2, 3]}
                )
            }
        )
        expected_y2 = Dataset(
            {
                "bar": DataArray(
                    [1, fill_value_bar, 2], dims=["x"], coords={"x": [1, 2, 3]}
                )
            }
        )
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)

    def test_align(self):
        left = create_test_data()
        right = left.copy(deep=True)
        right["dim3"] = ("dim3", list("cdefghijkl"))
        right["var3"][:-2] = right["var3"][2:].values
        right["var3"][-2:] = np.random.randn(*right["var3"][-2:].shape)
        right["numbers"][:-2] = right["numbers"][2:].values
        right["numbers"][-2:] = -10

        intersection = list("cdefghij")
        union = list("abcdefghijkl")

        left2, right2 = align(left, right, join="inner")
        assert_array_equal(left2["dim3"], intersection)
        assert_identical(left2, right2)

        left2, right2 = align(left, right, join="outer")

        assert_array_equal(left2["dim3"], union)
        assert_equal(left2["dim3"].variable, right2["dim3"].variable)

        assert_identical(left2.sel(dim3=intersection), right2.sel(dim3=intersection))
        assert np.isnan(left2["var3"][-2:]).all()
        assert np.isnan(right2["var3"][:2]).all()

        left2, right2 = align(left, right, join="left")
        assert_equal(left2["dim3"].variable, right2["dim3"].variable)
        assert_equal(left2["dim3"].variable, left["dim3"].variable)

        assert_identical(left2.sel(dim3=intersection), right2.sel(dim3=intersection))
        assert np.isnan(right2["var3"][:2]).all()

        left2, right2 = align(left, right, join="right")
        assert_equal(left2["dim3"].variable, right2["dim3"].variable)
        assert_equal(left2["dim3"].variable, right["dim3"].variable)

        assert_identical(left2.sel(dim3=intersection), right2.sel(dim3=intersection))

        assert np.isnan(left2["var3"][-2:]).all()

        with raises_regex(ValueError, "invalid value for join"):
            align(left, right, join="foobar")
        with pytest.raises(TypeError):
            align(left, right, foo="bar")

    def test_align_exact(self):
        left = xr.Dataset(coords={"x": [0, 1]})
        right = xr.Dataset(coords={"x": [1, 2]})

        left1, left2 = xr.align(left, left, join="exact")
        assert_identical(left1, left)
        assert_identical(left2, left)

        with raises_regex(ValueError, "indexes .* not equal"):
            xr.align(left, right, join="exact")

    def test_align_override(self):
        left = xr.Dataset(coords={"x": [0, 1, 2]})
        right = xr.Dataset(coords={"x": [0.1, 1.1, 2.1], "y": [1, 2, 3]})
        expected_right = xr.Dataset(coords={"x": [0, 1, 2], "y": [1, 2, 3]})

        new_left, new_right = xr.align(left, right, join="override")
        assert_identical(left, new_left)
        assert_identical(new_right, expected_right)

        new_left, new_right = xr.align(left, right, exclude="x", join="override")
        assert_identical(left, new_left)
        assert_identical(right, new_right)

        new_left, new_right = xr.align(
            left.isel(x=0, drop=True), right, exclude="x", join="override"
        )
        assert_identical(left.isel(x=0, drop=True), new_left)
        assert_identical(right, new_right)

        with raises_regex(ValueError, "Indexes along dimension 'x' don't have"):
            xr.align(left.isel(x=0).expand_dims("x"), right, join="override")

    def test_align_exclude(self):
        x = Dataset(
            {
                "foo": DataArray(
                    [[1, 2], [3, 4]], dims=["x", "y"], coords={"x": [1, 2], "y": [3, 4]}
                )
            }
        )
        y = Dataset(
            {
                "bar": DataArray(
                    [[1, 2], [3, 4]], dims=["x", "y"], coords={"x": [1, 3], "y": [5, 6]}
                )
            }
        )
        x2, y2 = align(x, y, exclude=["y"], join="outer")

        expected_x2 = Dataset(
            {
                "foo": DataArray(
                    [[1, 2], [3, 4], [np.nan, np.nan]],
                    dims=["x", "y"],
                    coords={"x": [1, 2, 3], "y": [3, 4]},
                )
            }
        )
        expected_y2 = Dataset(
            {
                "bar": DataArray(
                    [[1, 2], [np.nan, np.nan], [3, 4]],
                    dims=["x", "y"],
                    coords={"x": [1, 2, 3], "y": [5, 6]},
                )
            }
        )
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)

    def test_align_nocopy(self):
        x = Dataset({"foo": DataArray([1, 2, 3], coords=[("x", [1, 2, 3])])})
        y = Dataset({"foo": DataArray([1, 2], coords=[("x", [1, 2])])})
        expected_x2 = x
        expected_y2 = Dataset(
            {"foo": DataArray([1, 2, np.nan], coords=[("x", [1, 2, 3])])}
        )

        x2, y2 = align(x, y, copy=False, join="outer")
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)
        assert source_ndarray(x["foo"].data) is source_ndarray(x2["foo"].data)

        x2, y2 = align(x, y, copy=True, join="outer")
        assert source_ndarray(x["foo"].data) is not source_ndarray(x2["foo"].data)
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)

    def test_align_indexes(self):
        x = Dataset({"foo": DataArray([1, 2, 3], dims="x", coords=[("x", [1, 2, 3])])})
        (x2,) = align(x, indexes={"x": [2, 3, 1]})
        expected_x2 = Dataset(
            {"foo": DataArray([2, 3, 1], dims="x", coords={"x": [2, 3, 1]})}
        )

        assert_identical(expected_x2, x2)

    def test_align_non_unique(self):
        x = Dataset({"foo": ("x", [3, 4, 5]), "x": [0, 0, 1]})
        x1, x2 = align(x, x)
        assert_identical(x1, x)
        assert_identical(x2, x)

        y = Dataset({"bar": ("x", [6, 7]), "x": [0, 1]})
        with raises_regex(ValueError, "cannot reindex or align"):
            align(x, y)

    def test_align_str_dtype(self):

        a = Dataset({"foo": ("x", [0, 1]), "x": ["a", "b"]})
        b = Dataset({"foo": ("x", [1, 2]), "x": ["b", "c"]})

        expected_a = Dataset({"foo": ("x", [0, 1, np.NaN]), "x": ["a", "b", "c"]})
        expected_b = Dataset({"foo": ("x", [np.NaN, 1, 2]), "x": ["a", "b", "c"]})

        actual_a, actual_b = xr.align(a, b, join="outer")

        assert_identical(expected_a, actual_a)
        assert expected_a.x.dtype == actual_a.x.dtype

        assert_identical(expected_b, actual_b)
        assert expected_b.x.dtype == actual_b.x.dtype

    def test_broadcast(self):
        ds = Dataset(
            {"foo": 0, "bar": ("x", [1]), "baz": ("y", [2, 3])}, {"c": ("x", [4])}
        )
        expected = Dataset(
            {
                "foo": (("x", "y"), [[0, 0]]),
                "bar": (("x", "y"), [[1, 1]]),
                "baz": (("x", "y"), [[2, 3]]),
            },
            {"c": ("x", [4])},
        )
        (actual,) = broadcast(ds)
        assert_identical(expected, actual)

        ds_x = Dataset({"foo": ("x", [1])})
        ds_y = Dataset({"bar": ("y", [2, 3])})
        expected_x = Dataset({"foo": (("x", "y"), [[1, 1]])})
        expected_y = Dataset({"bar": (("x", "y"), [[2, 3]])})
        actual_x, actual_y = broadcast(ds_x, ds_y)
        assert_identical(expected_x, actual_x)
        assert_identical(expected_y, actual_y)

        array_y = ds_y["bar"]
        expected_y = expected_y["bar"]
        actual_x, actual_y = broadcast(ds_x, array_y)
        assert_identical(expected_x, actual_x)
        assert_identical(expected_y, actual_y)

    def test_broadcast_nocopy(self):
        # Test that data is not copied if not needed
        x = Dataset({"foo": (("x", "y"), [[1, 1]])})
        y = Dataset({"bar": ("y", [2, 3])})

        (actual_x,) = broadcast(x)
        assert_identical(x, actual_x)
        assert source_ndarray(actual_x["foo"].data) is source_ndarray(x["foo"].data)

        actual_x, actual_y = broadcast(x, y)
        assert_identical(x, actual_x)
        assert source_ndarray(actual_x["foo"].data) is source_ndarray(x["foo"].data)

    def test_broadcast_exclude(self):
        x = Dataset(
            {
                "foo": DataArray(
                    [[1, 2], [3, 4]], dims=["x", "y"], coords={"x": [1, 2], "y": [3, 4]}
                ),
                "bar": DataArray(5),
            }
        )
        y = Dataset(
            {
                "foo": DataArray(
                    [[1, 2]], dims=["z", "y"], coords={"z": [1], "y": [5, 6]}
                )
            }
        )
        x2, y2 = broadcast(x, y, exclude=["y"])

        expected_x2 = Dataset(
            {
                "foo": DataArray(
                    [[[1, 2]], [[3, 4]]],
                    dims=["x", "z", "y"],
                    coords={"z": [1], "x": [1, 2], "y": [3, 4]},
                ),
                "bar": DataArray(
                    [[5], [5]], dims=["x", "z"], coords={"x": [1, 2], "z": [1]}
                ),
            }
        )
        expected_y2 = Dataset(
            {
                "foo": DataArray(
                    [[[1, 2]], [[1, 2]]],
                    dims=["x", "z", "y"],
                    coords={"z": [1], "x": [1, 2], "y": [5, 6]},
                )
            }
        )
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)

    def test_broadcast_misaligned(self):
        x = Dataset({"foo": DataArray([1, 2, 3], coords=[("x", [-1, -2, -3])])})
        y = Dataset(
            {
                "bar": DataArray(
                    [[1, 2], [3, 4]],
                    dims=["y", "x"],
                    coords={"y": [1, 2], "x": [10, -3]},
                )
            }
        )
        x2, y2 = broadcast(x, y)
        expected_x2 = Dataset(
            {
                "foo": DataArray(
                    [[3, 3], [2, 2], [1, 1], [np.nan, np.nan]],
                    dims=["x", "y"],
                    coords={"y": [1, 2], "x": [-3, -2, -1, 10]},
                )
            }
        )
        expected_y2 = Dataset(
            {
                "bar": DataArray(
                    [[2, 4], [np.nan, np.nan], [np.nan, np.nan], [1, 3]],
                    dims=["x", "y"],
                    coords={"y": [1, 2], "x": [-3, -2, -1, 10]},
                )
            }
        )
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)

    def test_variable_indexing(self):
        data = create_test_data()
        v = data["var1"]
        d1 = data["dim1"]
        d2 = data["dim2"]
        assert_equal(v, v[d1.values])
        assert_equal(v, v[d1])
        assert_equal(v[:3], v[d1 < 3])
        assert_equal(v[:, 3:], v[:, d2 >= 1.5])
        assert_equal(v[:3, 3:], v[d1 < 3, d2 >= 1.5])
        assert_equal(v[:3, :2], v[range(3), range(2)])
        assert_equal(v[:3, :2], v.loc[d1[:3], d2[:2]])

    def test_drop_variables(self):
        data = create_test_data()

        assert_identical(data, data.drop_vars([]))

        expected = Dataset({k: data[k] for k in data.variables if k != "time"})
        actual = data.drop_vars("time")
        assert_identical(expected, actual)
        actual = data.drop_vars(["time"])
        assert_identical(expected, actual)

        with raises_regex(ValueError, "cannot be found"):
            data.drop_vars("not_found_here")

        actual = data.drop_vars("not_found_here", errors="ignore")
        assert_identical(data, actual)

        actual = data.drop_vars(["not_found_here"], errors="ignore")
        assert_identical(data, actual)

        actual = data.drop_vars(["time", "not_found_here"], errors="ignore")
        assert_identical(expected, actual)

        # deprecated approach with `drop` works (straight copy paste from above)

        with pytest.warns(PendingDeprecationWarning):
            actual = data.drop("not_found_here", errors="ignore")
        assert_identical(data, actual)

        with pytest.warns(PendingDeprecationWarning):
            actual = data.drop(["not_found_here"], errors="ignore")
        assert_identical(data, actual)

        with pytest.warns(PendingDeprecationWarning):
            actual = data.drop(["time", "not_found_here"], errors="ignore")
        assert_identical(expected, actual)

        with pytest.warns(PendingDeprecationWarning):
            actual = data.drop({"time", "not_found_here"}, errors="ignore")
        assert_identical(expected, actual)

    def test_drop_index_labels(self):
        data = Dataset({"A": (["x", "y"], np.random.randn(2, 3)), "x": ["a", "b"]})

        with pytest.warns(DeprecationWarning):
            actual = data.drop(["a"], dim="x")
        expected = data.isel(x=[1])
        assert_identical(expected, actual)

        with pytest.warns(DeprecationWarning):
            actual = data.drop(["a", "b"], dim="x")
        expected = data.isel(x=slice(0, 0))
        assert_identical(expected, actual)

        with pytest.raises(KeyError):
            # not contained in axis
            with pytest.warns(DeprecationWarning):
                data.drop(["c"], dim="x")

        with pytest.warns(DeprecationWarning):
            actual = data.drop(["c"], dim="x", errors="ignore")
        assert_identical(data, actual)

        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                data.drop(["c"], dim="x", errors="wrong_value")

        with pytest.warns(DeprecationWarning):
            actual = data.drop(["a", "b", "c"], "x", errors="ignore")
        expected = data.isel(x=slice(0, 0))
        assert_identical(expected, actual)

        # DataArrays as labels are a nasty corner case as they are not
        # Iterable[Hashable] - DataArray.__iter__ yields scalar DataArrays.
        actual = data.drop_sel(x=DataArray(["a", "b", "c"]), errors="ignore")
        expected = data.isel(x=slice(0, 0))
        assert_identical(expected, actual)
        with pytest.warns(DeprecationWarning):
            data.drop(DataArray(["a", "b", "c"]), dim="x", errors="ignore")
        assert_identical(expected, actual)

        actual = data.drop_sel(y=[1])
        expected = data.isel(y=[0, 2])
        assert_identical(expected, actual)

        with raises_regex(KeyError, "not found in axis"):
            data.drop_sel(x=0)

    def test_drop_labels_by_keyword(self):
        data = Dataset(
            {"A": (["x", "y"], np.random.randn(2, 6)), "x": ["a", "b"], "y": range(6)}
        )
        # Basic functionality.
        assert len(data.coords["x"]) == 2

        with pytest.warns(DeprecationWarning):
            ds1 = data.drop(["a"], dim="x")
        ds2 = data.drop_sel(x="a")
        ds3 = data.drop_sel(x=["a"])
        ds4 = data.drop_sel(x=["a", "b"])
        ds5 = data.drop_sel(x=["a", "b"], y=range(0, 6, 2))

        arr = DataArray(range(3), dims=["c"])
        with pytest.warns(FutureWarning):
            data.drop(arr.coords)
        with pytest.warns(FutureWarning):
            data.drop(arr.indexes)

        assert_array_equal(ds1.coords["x"], ["b"])
        assert_array_equal(ds2.coords["x"], ["b"])
        assert_array_equal(ds3.coords["x"], ["b"])
        assert ds4.coords["x"].size == 0
        assert ds5.coords["x"].size == 0
        assert_array_equal(ds5.coords["y"], [1, 3, 5])

        # Error handling if user tries both approaches.
        with pytest.raises(ValueError):
            data.drop(labels=["a"], x="a")
        with pytest.raises(ValueError):
            data.drop(labels=["a"], dim="x", x="a")
        warnings.filterwarnings("ignore", r"\W*drop")
        with pytest.raises(ValueError):
            data.drop(dim="x", x="a")

    def test_drop_labels_by_position(self):
        data = Dataset(
            {"A": (["x", "y"], np.random.randn(2, 6)), "x": ["a", "b"], "y": range(6)}
        )
        # Basic functionality.
        assert len(data.coords["x"]) == 2

        actual = data.drop_isel(x=0)
        expected = data.drop_sel(x="a")
        assert_identical(expected, actual)

        actual = data.drop_isel(x=[0])
        expected = data.drop_sel(x=["a"])
        assert_identical(expected, actual)

        actual = data.drop_isel(x=[0, 1])
        expected = data.drop_sel(x=["a", "b"])
        assert_identical(expected, actual)
        assert actual.coords["x"].size == 0

        actual = data.drop_isel(x=[0, 1], y=range(0, 6, 2))
        expected = data.drop_sel(x=["a", "b"], y=range(0, 6, 2))
        assert_identical(expected, actual)
        assert actual.coords["x"].size == 0

        with pytest.raises(KeyError):
            data.drop_isel(z=1)

    def test_drop_dims(self):
        data = xr.Dataset(
            {
                "A": (["x", "y"], np.random.randn(2, 3)),
                "B": ("x", np.random.randn(2)),
                "x": ["a", "b"],
                "z": np.pi,
            }
        )

        actual = data.drop_dims("x")
        expected = data.drop_vars(["A", "B", "x"])
        assert_identical(expected, actual)

        actual = data.drop_dims("y")
        expected = data.drop_vars("A")
        assert_identical(expected, actual)

        actual = data.drop_dims(["x", "y"])
        expected = data.drop_vars(["A", "B", "x"])
        assert_identical(expected, actual)

        with pytest.raises((ValueError, KeyError)):
            data.drop_dims("z")  # not a dimension

        with pytest.raises((ValueError, KeyError)):
            data.drop_dims(None)

        actual = data.drop_dims("z", errors="ignore")
        assert_identical(data, actual)

        actual = data.drop_dims(None, errors="ignore")
        assert_identical(data, actual)

        with pytest.raises(ValueError):
            actual = data.drop_dims("z", errors="wrong_value")

        actual = data.drop_dims(["x", "y", "z"], errors="ignore")
        expected = data.drop_vars(["A", "B", "x"])
        assert_identical(expected, actual)

    def test_copy(self):
        data = create_test_data()
        data.attrs["Test"] = [1, 2, 3]

        for copied in [data.copy(deep=False), copy(data)]:
            assert_identical(data, copied)
            assert data.encoding == copied.encoding
            # Note: IndexVariable objects with string dtype are always
            # copied because of xarray.core.util.safe_cast_to_index.
            # Limiting the test to data variables.
            for k in data.data_vars:
                v0 = data.variables[k]
                v1 = copied.variables[k]
                assert source_ndarray(v0.data) is source_ndarray(v1.data)
            copied["foo"] = ("z", np.arange(5))
            assert "foo" not in data

            copied.attrs["foo"] = "bar"
            assert "foo" not in data.attrs
            assert data.attrs["Test"] is copied.attrs["Test"]

        for copied in [data.copy(deep=True), deepcopy(data)]:
            assert_identical(data, copied)
            for k, v0 in data.variables.items():
                v1 = copied.variables[k]
                assert v0 is not v1

            assert data.attrs["Test"] is not copied.attrs["Test"]

    def test_copy_with_data(self):
        orig = create_test_data()
        new_data = {k: np.random.randn(*v.shape) for k, v in orig.data_vars.items()}
        actual = orig.copy(data=new_data)

        expected = orig.copy()
        for k, v in new_data.items():
            expected[k].data = v
        assert_identical(expected, actual)

    @pytest.mark.xfail(raises=AssertionError)
    @pytest.mark.parametrize(
        "deep, expected_orig",
        [
            [
                True,
                xr.DataArray(
                    xr.IndexVariable("a", np.array([1, 2])),
                    coords={"a": [1, 2]},
                    dims=["a"],
                ),
            ],
            [
                False,
                xr.DataArray(
                    xr.IndexVariable("a", np.array([999, 2])),
                    coords={"a": [999, 2]},
                    dims=["a"],
                ),
            ],
        ],
    )
    def test_copy_coords(self, deep, expected_orig):
        """The test fails for the shallow copy, and apparently only on Windows
        for some reason. In windows coords seem to be immutable unless it's one
        dataset deep copied from another."""
        ds = xr.DataArray(
            np.ones([2, 2, 2]),
            coords={"a": [1, 2], "b": ["x", "y"], "c": [0, 1]},
            dims=["a", "b", "c"],
            name="value",
        ).to_dataset()
        ds_cp = ds.copy(deep=deep)
        ds_cp.coords["a"].data[0] = 999

        expected_cp = xr.DataArray(
            xr.IndexVariable("a", np.array([999, 2])),
            coords={"a": [999, 2]},
            dims=["a"],
        )
        assert_identical(ds_cp.coords["a"], expected_cp)

        assert_identical(ds.coords["a"], expected_orig)

    def test_copy_with_data_errors(self):
        orig = create_test_data()
        new_var1 = np.arange(orig["var1"].size).reshape(orig["var1"].shape)
        with raises_regex(ValueError, "Data must be dict-like"):
            orig.copy(data=new_var1)
        with raises_regex(ValueError, "only contain variables in original"):
            orig.copy(data={"not_in_original": new_var1})
        with raises_regex(ValueError, "contain all variables in original"):
            orig.copy(data={"var1": new_var1})

    def test_rename(self):
        data = create_test_data()
        newnames = {"var1": "renamed_var1", "dim2": "renamed_dim2"}
        renamed = data.rename(newnames)

        variables = dict(data.variables)
        for k, v in newnames.items():
            variables[v] = variables.pop(k)

        for k, v in variables.items():
            dims = list(v.dims)
            for name, newname in newnames.items():
                if name in dims:
                    dims[dims.index(name)] = newname

            assert_equal(
                Variable(dims, v.values, v.attrs),
                renamed[k].variable.to_base_variable(),
            )
            assert v.encoding == renamed[k].encoding
            assert type(v) is type(renamed.variables[k])  # noqa: E721

        assert "var1" not in renamed
        assert "dim2" not in renamed

        with raises_regex(ValueError, "cannot rename 'not_a_var'"):
            data.rename({"not_a_var": "nada"})

        with raises_regex(ValueError, "'var1' conflicts"):
            data.rename({"var2": "var1"})

        # verify that we can rename a variable without accessing the data
        var1 = data["var1"]
        data["var1"] = (var1.dims, InaccessibleArray(var1.values))
        renamed = data.rename(newnames)
        with pytest.raises(UnexpectedDataAccess):
            renamed["renamed_var1"].values

        renamed_kwargs = data.rename(**newnames)
        assert_identical(renamed, renamed_kwargs)

    def test_rename_old_name(self):
        # regtest for GH1477
        data = create_test_data()

        with raises_regex(ValueError, "'samecol' conflicts"):
            data.rename({"var1": "samecol", "var2": "samecol"})

        # This shouldn't cause any problems.
        data.rename({"var1": "var2", "var2": "var1"})

    def test_rename_same_name(self):
        data = create_test_data()
        newnames = {"var1": "var1", "dim2": "dim2"}
        renamed = data.rename(newnames)
        assert_identical(renamed, data)

    def test_rename_dims(self):
        original = Dataset({"x": ("x", [0, 1, 2]), "y": ("x", [10, 11, 12]), "z": 42})
        expected = Dataset(
            {"x": ("x_new", [0, 1, 2]), "y": ("x_new", [10, 11, 12]), "z": 42}
        )
        expected = expected.set_coords("x")
        dims_dict = {"x": "x_new"}
        actual = original.rename_dims(dims_dict)
        assert_identical(expected, actual)
        actual_2 = original.rename_dims(**dims_dict)
        assert_identical(expected, actual_2)

        # Test to raise ValueError
        dims_dict_bad = {"x_bad": "x_new"}
        with pytest.raises(ValueError):
            original.rename_dims(dims_dict_bad)

        with pytest.raises(ValueError):
            original.rename_dims({"x": "z"})

    def test_rename_vars(self):
        original = Dataset({"x": ("x", [0, 1, 2]), "y": ("x", [10, 11, 12]), "z": 42})
        expected = Dataset(
            {"x_new": ("x", [0, 1, 2]), "y": ("x", [10, 11, 12]), "z": 42}
        )
        expected = expected.set_coords("x_new")
        name_dict = {"x": "x_new"}
        actual = original.rename_vars(name_dict)
        assert_identical(expected, actual)
        actual_2 = original.rename_vars(**name_dict)
        assert_identical(expected, actual_2)

        # Test to raise ValueError
        names_dict_bad = {"x_bad": "x_new"}
        with pytest.raises(ValueError):
            original.rename_vars(names_dict_bad)

    def test_rename_multiindex(self):
        mindex = pd.MultiIndex.from_tuples(
            [([1, 2]), ([3, 4])], names=["level0", "level1"]
        )
        data = Dataset({}, {"x": mindex})
        with raises_regex(ValueError, "conflicting MultiIndex"):
            data.rename({"x": "level0"})

    @requires_cftime
    def test_rename_does_not_change_CFTimeIndex_type(self):
        # make sure CFTimeIndex is not converted to DatetimeIndex #3522

        time = xr.cftime_range(start="2000", periods=6, freq="2MS", calendar="noleap")
        orig = Dataset(coords={"time": time})

        renamed = orig.rename(time="time_new")
        assert "time_new" in renamed.indexes
        assert isinstance(renamed.indexes["time_new"], CFTimeIndex)
        assert renamed.indexes["time_new"].name == "time_new"

        # check original has not changed
        assert "time" in orig.indexes
        assert isinstance(orig.indexes["time"], CFTimeIndex)
        assert orig.indexes["time"].name == "time"

        # note: rename_dims(time="time_new") drops "ds.indexes"
        renamed = orig.rename_dims()
        assert isinstance(renamed.indexes["time"], CFTimeIndex)

        renamed = orig.rename_vars()
        assert isinstance(renamed.indexes["time"], CFTimeIndex)

    def test_rename_does_not_change_DatetimeIndex_type(self):
        # make sure DatetimeIndex is conderved on rename

        time = pd.date_range(start="2000", periods=6, freq="2MS")
        orig = Dataset(coords={"time": time})

        renamed = orig.rename(time="time_new")
        assert "time_new" in renamed.indexes
        assert isinstance(renamed.indexes["time_new"], DatetimeIndex)
        assert renamed.indexes["time_new"].name == "time_new"

        # check original has not changed
        assert "time" in orig.indexes
        assert isinstance(orig.indexes["time"], DatetimeIndex)
        assert orig.indexes["time"].name == "time"

        # note: rename_dims(time="time_new") drops "ds.indexes"
        renamed = orig.rename_dims()
        assert isinstance(renamed.indexes["time"], DatetimeIndex)

        renamed = orig.rename_vars()
        assert isinstance(renamed.indexes["time"], DatetimeIndex)

    def test_swap_dims(self):
        original = Dataset({"x": [1, 2, 3], "y": ("x", list("abc")), "z": 42})
        expected = Dataset({"z": 42}, {"x": ("y", [1, 2, 3]), "y": list("abc")})
        actual = original.swap_dims({"x": "y"})
        assert_identical(expected, actual)
        assert isinstance(actual.variables["y"], IndexVariable)
        assert isinstance(actual.variables["x"], Variable)
        pd.testing.assert_index_equal(actual.indexes["y"], expected.indexes["y"])

        roundtripped = actual.swap_dims({"y": "x"})
        assert_identical(original.set_coords("y"), roundtripped)

        with raises_regex(ValueError, "cannot swap"):
            original.swap_dims({"y": "x"})
        with raises_regex(ValueError, "replacement dimension"):
            original.swap_dims({"x": "z"})

        expected = Dataset(
            {"y": ("u", list("abc")), "z": 42}, coords={"x": ("u", [1, 2, 3])}
        )
        actual = original.swap_dims({"x": "u"})
        assert_identical(expected, actual)

        # as kwargs
        expected = Dataset(
            {"y": ("u", list("abc")), "z": 42}, coords={"x": ("u", [1, 2, 3])}
        )
        actual = original.swap_dims(x="u")
        assert_identical(expected, actual)

        # handle multiindex case
        idx = pd.MultiIndex.from_arrays([list("aab"), list("yzz")], names=["y1", "y2"])
        original = Dataset({"x": [1, 2, 3], "y": ("x", idx), "z": 42})
        expected = Dataset({"z": 42}, {"x": ("y", [1, 2, 3]), "y": idx})
        actual = original.swap_dims({"x": "y"})
        assert_identical(expected, actual)
        assert isinstance(actual.variables["y"], IndexVariable)
        assert isinstance(actual.variables["x"], Variable)
        pd.testing.assert_index_equal(actual.indexes["y"], expected.indexes["y"])

    def test_expand_dims_error(self):
        original = Dataset(
            {
                "x": ("a", np.random.randn(3)),
                "y": (["b", "a"], np.random.randn(4, 3)),
                "z": ("a", np.random.randn(3)),
            },
            coords={
                "a": np.linspace(0, 1, 3),
                "b": np.linspace(0, 1, 4),
                "c": np.linspace(0, 1, 5),
            },
            attrs={"key": "entry"},
        )

        with raises_regex(ValueError, "already exists"):
            original.expand_dims(dim=["x"])

        # Make sure it raises true error also for non-dimensional coordinates
        # which has dimension.
        original = original.set_coords("z")
        with raises_regex(ValueError, "already exists"):
            original.expand_dims(dim=["z"])

        original = Dataset(
            {
                "x": ("a", np.random.randn(3)),
                "y": (["b", "a"], np.random.randn(4, 3)),
                "z": ("a", np.random.randn(3)),
            },
            coords={
                "a": np.linspace(0, 1, 3),
                "b": np.linspace(0, 1, 4),
                "c": np.linspace(0, 1, 5),
            },
            attrs={"key": "entry"},
        )
        with raises_regex(TypeError, "value of new dimension"):
            original.expand_dims({"d": 3.2})
        with raises_regex(ValueError, "both keyword and positional"):
            original.expand_dims({"d": 4}, e=4)

    def test_expand_dims_int(self):
        original = Dataset(
            {"x": ("a", np.random.randn(3)), "y": (["b", "a"], np.random.randn(4, 3))},
            coords={
                "a": np.linspace(0, 1, 3),
                "b": np.linspace(0, 1, 4),
                "c": np.linspace(0, 1, 5),
            },
            attrs={"key": "entry"},
        )

        actual = original.expand_dims(["z"], [1])
        expected = Dataset(
            {
                "x": original["x"].expand_dims("z", 1),
                "y": original["y"].expand_dims("z", 1),
            },
            coords={
                "a": np.linspace(0, 1, 3),
                "b": np.linspace(0, 1, 4),
                "c": np.linspace(0, 1, 5),
            },
            attrs={"key": "entry"},
        )
        assert_identical(expected, actual)
        # make sure squeeze restores the original data set.
        roundtripped = actual.squeeze("z")
        assert_identical(original, roundtripped)

        # another test with a negative axis
        actual = original.expand_dims(["z"], [-1])
        expected = Dataset(
            {
                "x": original["x"].expand_dims("z", -1),
                "y": original["y"].expand_dims("z", -1),
            },
            coords={
                "a": np.linspace(0, 1, 3),
                "b": np.linspace(0, 1, 4),
                "c": np.linspace(0, 1, 5),
            },
            attrs={"key": "entry"},
        )
        assert_identical(expected, actual)
        # make sure squeeze restores the original data set.
        roundtripped = actual.squeeze("z")
        assert_identical(original, roundtripped)

    def test_expand_dims_coords(self):
        original = Dataset({"x": ("a", np.array([1, 2, 3]))})
        expected = Dataset(
            {"x": (("b", "a"), np.array([[1, 2, 3], [1, 2, 3]]))}, coords={"b": [1, 2]}
        )
        actual = original.expand_dims(dict(b=[1, 2]))
        assert_identical(expected, actual)
        assert "b" not in original._coord_names

    def test_expand_dims_existing_scalar_coord(self):
        original = Dataset({"x": 1}, {"a": 2})
        expected = Dataset({"x": (("a",), [1])}, {"a": [2]})
        actual = original.expand_dims("a")
        assert_identical(expected, actual)

    def test_isel_expand_dims_roundtrip(self):
        original = Dataset({"x": (("a",), [1])}, {"a": [2]})
        actual = original.isel(a=0).expand_dims("a")
        assert_identical(actual, original)

    def test_expand_dims_mixed_int_and_coords(self):
        # Test expanding one dimension to have size > 1 that doesn't have
        # coordinates, and also expanding another dimension to have size > 1
        # that DOES have coordinates.
        original = Dataset(
            {"x": ("a", np.random.randn(3)), "y": (["b", "a"], np.random.randn(4, 3))},
            coords={
                "a": np.linspace(0, 1, 3),
                "b": np.linspace(0, 1, 4),
                "c": np.linspace(0, 1, 5),
            },
        )

        actual = original.expand_dims({"d": 4, "e": ["l", "m", "n"]})

        expected = Dataset(
            {
                "x": xr.DataArray(
                    original["x"].values * np.ones([4, 3, 3]),
                    coords=dict(d=range(4), e=["l", "m", "n"], a=np.linspace(0, 1, 3)),
                    dims=["d", "e", "a"],
                ).drop_vars("d"),
                "y": xr.DataArray(
                    original["y"].values * np.ones([4, 3, 4, 3]),
                    coords=dict(
                        d=range(4),
                        e=["l", "m", "n"],
                        b=np.linspace(0, 1, 4),
                        a=np.linspace(0, 1, 3),
                    ),
                    dims=["d", "e", "b", "a"],
                ).drop_vars("d"),
            },
            coords={"c": np.linspace(0, 1, 5)},
        )
        assert_identical(actual, expected)

    def test_expand_dims_kwargs_python36plus(self):
        original = Dataset(
            {"x": ("a", np.random.randn(3)), "y": (["b", "a"], np.random.randn(4, 3))},
            coords={
                "a": np.linspace(0, 1, 3),
                "b": np.linspace(0, 1, 4),
                "c": np.linspace(0, 1, 5),
            },
            attrs={"key": "entry"},
        )
        other_way = original.expand_dims(e=["l", "m", "n"])
        other_way_expected = Dataset(
            {
                "x": xr.DataArray(
                    original["x"].values * np.ones([3, 3]),
                    coords=dict(e=["l", "m", "n"], a=np.linspace(0, 1, 3)),
                    dims=["e", "a"],
                ),
                "y": xr.DataArray(
                    original["y"].values * np.ones([3, 4, 3]),
                    coords=dict(
                        e=["l", "m", "n"],
                        b=np.linspace(0, 1, 4),
                        a=np.linspace(0, 1, 3),
                    ),
                    dims=["e", "b", "a"],
                ),
            },
            coords={"c": np.linspace(0, 1, 5)},
            attrs={"key": "entry"},
        )
        assert_identical(other_way_expected, other_way)

    def test_set_index(self):
        expected = create_test_multiindex()
        mindex = expected["x"].to_index()
        indexes = [mindex.get_level_values(n) for n in mindex.names]
        coords = {idx.name: ("x", idx) for idx in indexes}
        ds = Dataset({}, coords=coords)

        obj = ds.set_index(x=mindex.names)
        assert_identical(obj, expected)

        # ensure set_index with no existing index and a single data var given
        # doesn't return multi-index
        ds = Dataset(data_vars={"x_var": ("x", [0, 1, 2])})
        expected = Dataset(coords={"x": [0, 1, 2]})
        assert_identical(ds.set_index(x="x_var"), expected)

        # Issue 3176: Ensure clear error message on key error.
        with pytest.raises(ValueError) as excinfo:
            ds.set_index(foo="bar")
        assert str(excinfo.value) == "bar is not the name of an existing variable."

    def test_reset_index(self):
        ds = create_test_multiindex()
        mindex = ds["x"].to_index()
        indexes = [mindex.get_level_values(n) for n in mindex.names]
        coords = {idx.name: ("x", idx) for idx in indexes}
        expected = Dataset({}, coords=coords)

        obj = ds.reset_index("x")
        assert_identical(obj, expected)

    def test_reset_index_keep_attrs(self):
        coord_1 = DataArray([1, 2], dims=["coord_1"], attrs={"attrs": True})
        ds = Dataset({}, {"coord_1": coord_1})
        expected = Dataset({}, {"coord_1_": coord_1})
        obj = ds.reset_index("coord_1")
        assert_identical(expected, obj)

    def test_reorder_levels(self):
        ds = create_test_multiindex()
        mindex = ds["x"].to_index()
        midx = mindex.reorder_levels(["level_2", "level_1"])
        expected = Dataset({}, coords={"x": midx})

        reindexed = ds.reorder_levels(x=["level_2", "level_1"])
        assert_identical(reindexed, expected)

        ds = Dataset({}, coords={"x": [1, 2]})
        with raises_regex(ValueError, "has no MultiIndex"):
            ds.reorder_levels(x=["level_1", "level_2"])

    def test_stack(self):
        ds = Dataset(
            {"a": ("x", [0, 1]), "b": (("x", "y"), [[0, 1], [2, 3]]), "y": ["a", "b"]}
        )

        exp_index = pd.MultiIndex.from_product([[0, 1], ["a", "b"]], names=["x", "y"])
        expected = Dataset(
            {"a": ("z", [0, 0, 1, 1]), "b": ("z", [0, 1, 2, 3]), "z": exp_index}
        )
        actual = ds.stack(z=["x", "y"])
        assert_identical(expected, actual)

        actual = ds.stack(z=[...])
        assert_identical(expected, actual)

        # non list dims with ellipsis
        actual = ds.stack(z=(...,))
        assert_identical(expected, actual)

        # ellipsis with given dim
        actual = ds.stack(z=[..., "y"])
        assert_identical(expected, actual)

        exp_index = pd.MultiIndex.from_product([["a", "b"], [0, 1]], names=["y", "x"])
        expected = Dataset(
            {"a": ("z", [0, 1, 0, 1]), "b": ("z", [0, 2, 1, 3]), "z": exp_index}
        )
        actual = ds.stack(z=["y", "x"])
        assert_identical(expected, actual)

    def test_unstack(self):
        index = pd.MultiIndex.from_product([[0, 1], ["a", "b"]], names=["x", "y"])
        ds = Dataset({"b": ("z", [0, 1, 2, 3]), "z": index})
        expected = Dataset(
            {"b": (("x", "y"), [[0, 1], [2, 3]]), "x": [0, 1], "y": ["a", "b"]}
        )
        for dim in ["z", ["z"], None]:
            actual = ds.unstack(dim)
            assert_identical(actual, expected)

    def test_unstack_errors(self):
        ds = Dataset({"x": [1, 2, 3]})
        with raises_regex(ValueError, "does not contain the dimensions"):
            ds.unstack("foo")
        with raises_regex(ValueError, "do not have a MultiIndex"):
            ds.unstack("x")

    def test_unstack_fill_value(self):
        ds = xr.Dataset(
            {"var": (("x",), np.arange(6)), "other_var": (("x",), np.arange(3, 9))},
            coords={"x": [0, 1, 2] * 2, "y": (("x",), ["a"] * 3 + ["b"] * 3)},
        )
        # make ds incomplete
        ds = ds.isel(x=[0, 2, 3, 4]).set_index(index=["x", "y"])
        # test fill_value
        actual = ds.unstack("index", fill_value=-1)
        expected = ds.unstack("index").fillna(-1).astype(int)
        assert actual["var"].dtype == int
        assert_equal(actual, expected)

        actual = ds["var"].unstack("index", fill_value=-1)
        expected = ds["var"].unstack("index").fillna(-1).astype(int)
        assert_equal(actual, expected)

        actual = ds.unstack("index", fill_value={"var": -1, "other_var": 1})
        expected = ds.unstack("index").fillna({"var": -1, "other_var": 1}).astype(int)
        assert_equal(actual, expected)

    @requires_sparse
    def test_unstack_sparse(self):
        ds = xr.Dataset(
            {"var": (("x",), np.arange(6))},
            coords={"x": [0, 1, 2] * 2, "y": (("x",), ["a"] * 3 + ["b"] * 3)},
        )
        # make ds incomplete
        ds = ds.isel(x=[0, 2, 3, 4]).set_index(index=["x", "y"])
        # test fill_value
        actual = ds.unstack("index", sparse=True)
        expected = ds.unstack("index")
        assert actual["var"].variable._to_dense().equals(expected["var"].variable)
        assert actual["var"].data.density < 1.0

        actual = ds["var"].unstack("index", sparse=True)
        expected = ds["var"].unstack("index")
        assert actual.variable._to_dense().equals(expected.variable)
        assert actual.data.density < 1.0

    def test_stack_unstack_fast(self):
        ds = Dataset(
            {
                "a": ("x", [0, 1]),
                "b": (("x", "y"), [[0, 1], [2, 3]]),
                "x": [0, 1],
                "y": ["a", "b"],
            }
        )
        actual = ds.stack(z=["x", "y"]).unstack("z")
        assert actual.broadcast_equals(ds)

        actual = ds[["b"]].stack(z=["x", "y"]).unstack("z")
        assert actual.identical(ds[["b"]])

    def test_stack_unstack_slow(self):
        ds = Dataset(
            {
                "a": ("x", [0, 1]),
                "b": (("x", "y"), [[0, 1], [2, 3]]),
                "x": [0, 1],
                "y": ["a", "b"],
            }
        )
        stacked = ds.stack(z=["x", "y"])
        actual = stacked.isel(z=slice(None, None, -1)).unstack("z")
        assert actual.broadcast_equals(ds)

        stacked = ds[["b"]].stack(z=["x", "y"])
        actual = stacked.isel(z=slice(None, None, -1)).unstack("z")
        assert actual.identical(ds[["b"]])

    def test_to_stacked_array_invalid_sample_dims(self):
        data = xr.Dataset(
            data_vars={"a": (("x", "y"), [[0, 1, 2], [3, 4, 5]]), "b": ("x", [6, 7])},
            coords={"y": ["u", "v", "w"]},
        )
        with pytest.raises(ValueError):
            data.to_stacked_array("features", sample_dims=["y"])

    def test_to_stacked_array_name(self):
        name = "adf9d"

        # make a two dimensional dataset
        a, b = create_test_stacked_array()
        D = xr.Dataset({"a": a, "b": b})
        sample_dims = ["x"]

        y = D.to_stacked_array("features", sample_dims, name=name)
        assert y.name == name

    def test_to_stacked_array_dtype_dims(self):
        # make a two dimensional dataset
        a, b = create_test_stacked_array()
        D = xr.Dataset({"a": a, "b": b})
        sample_dims = ["x"]
        y = D.to_stacked_array("features", sample_dims)
        assert y.indexes["features"].levels[1].dtype == D.y.dtype
        assert y.dims == ("x", "features")

    def test_to_stacked_array_to_unstacked_dataset(self):

        # single dimension: regression test for GH4049
        arr = xr.DataArray(np.arange(3), coords=[("x", [0, 1, 2])])
        data = xr.Dataset({"a": arr, "b": arr})
        stacked = data.to_stacked_array("y", sample_dims=["x"])
        unstacked = stacked.to_unstacked_dataset("y")
        assert_identical(unstacked, data)

        # make a two dimensional dataset
        a, b = create_test_stacked_array()
        D = xr.Dataset({"a": a, "b": b})
        sample_dims = ["x"]
        y = D.to_stacked_array("features", sample_dims).transpose("x", "features")

        x = y.to_unstacked_dataset("features")
        assert_identical(D, x)

        # test on just one sample
        x0 = y[0].to_unstacked_dataset("features")
        d0 = D.isel(x=0)
        assert_identical(d0, x0)

    def test_to_stacked_array_to_unstacked_dataset_different_dimension(self):
        # test when variables have different dimensionality
        a, b = create_test_stacked_array()
        sample_dims = ["x"]
        D = xr.Dataset({"a": a, "b": b.isel(y=0)})

        y = D.to_stacked_array("features", sample_dims)
        x = y.to_unstacked_dataset("features")
        assert_identical(D, x)

    def test_update(self):
        data = create_test_data(seed=0)
        expected = data.copy()
        var2 = Variable("dim1", np.arange(8))
        actual = data.update({"var2": var2})
        expected["var2"] = var2
        assert_identical(expected, actual)

        actual = data.copy()
        actual_result = actual.update(data)
        assert actual_result is actual
        assert_identical(expected, actual)

        other = Dataset(attrs={"new": "attr"})
        actual = data.copy()
        actual.update(other)
        assert_identical(expected, actual)

    def test_update_overwrite_coords(self):
        data = Dataset({"a": ("x", [1, 2])}, {"b": 3})
        data.update(Dataset(coords={"b": 4}))
        expected = Dataset({"a": ("x", [1, 2])}, {"b": 4})
        assert_identical(data, expected)

        data = Dataset({"a": ("x", [1, 2])}, {"b": 3})
        data.update(Dataset({"c": 5}, coords={"b": 4}))
        expected = Dataset({"a": ("x", [1, 2]), "c": 5}, {"b": 4})
        assert_identical(data, expected)

        data = Dataset({"a": ("x", [1, 2])}, {"b": 3})
        data.update({"c": DataArray(5, coords={"b": 4})})
        expected = Dataset({"a": ("x", [1, 2]), "c": 5}, {"b": 3})
        assert_identical(data, expected)

    def test_update_auto_align(self):
        ds = Dataset({"x": ("t", [3, 4])}, {"t": [0, 1]})

        expected = Dataset({"x": ("t", [3, 4]), "y": ("t", [np.nan, 5])}, {"t": [0, 1]})
        actual = ds.copy()
        other = {"y": ("t", [5]), "t": [1]}
        with raises_regex(ValueError, "conflicting sizes"):
            actual.update(other)
        actual.update(Dataset(other))
        assert_identical(expected, actual)

        actual = ds.copy()
        other = Dataset({"y": ("t", [5]), "t": [100]})
        actual.update(other)
        expected = Dataset(
            {"x": ("t", [3, 4]), "y": ("t", [np.nan] * 2)}, {"t": [0, 1]}
        )
        assert_identical(expected, actual)

    def test_getitem(self):
        data = create_test_data()
        assert isinstance(data["var1"], DataArray)
        assert_equal(data["var1"].variable, data.variables["var1"])
        with pytest.raises(KeyError):
            data["notfound"]
        with pytest.raises(KeyError):
            data[["var1", "notfound"]]

        actual = data[["var1", "var2"]]
        expected = Dataset({"var1": data["var1"], "var2": data["var2"]})
        assert_equal(expected, actual)

        actual = data["numbers"]
        expected = DataArray(
            data["numbers"].variable,
            {"dim3": data["dim3"], "numbers": data["numbers"]},
            dims="dim3",
            name="numbers",
        )
        assert_identical(expected, actual)

        actual = data[dict(dim1=0)]
        expected = data.isel(dim1=0)
        assert_identical(expected, actual)

    def test_getitem_hashable(self):
        data = create_test_data()
        data[(3, 4)] = data["var1"] + 1
        expected = data["var1"] + 1
        expected.name = (3, 4)
        assert_identical(expected, data[(3, 4)])
        with raises_regex(KeyError, "('var1', 'var2')"):
            data[("var1", "var2")]

    def test_virtual_variables_default_coords(self):
        dataset = Dataset({"foo": ("x", range(10))})
        expected = DataArray(range(10), dims="x", name="x")
        actual = dataset["x"]
        assert_identical(expected, actual)
        assert isinstance(actual.variable, IndexVariable)

        actual = dataset[["x", "foo"]]
        expected = dataset.assign_coords(x=range(10))
        assert_identical(expected, actual)

    def test_virtual_variables_time(self):
        # access virtual variables
        data = create_test_data()
        expected = DataArray(
            1 + np.arange(20), coords=[data["time"]], dims="time", name="dayofyear"
        )

        assert_array_equal(
            data["time.month"].values, data.variables["time"].to_index().month
        )
        assert_array_equal(data["time.season"].values, "DJF")
        # test virtual variable math
        assert_array_equal(data["time.dayofyear"] + 1, 2 + np.arange(20))
        assert_array_equal(np.sin(data["time.dayofyear"]), np.sin(1 + np.arange(20)))
        # ensure they become coordinates
        expected = Dataset({}, {"dayofyear": data["time.dayofyear"]})
        actual = data[["time.dayofyear"]]
        assert_equal(expected, actual)
        # non-coordinate variables
        ds = Dataset({"t": ("x", pd.date_range("2000-01-01", periods=3))})
        assert (ds["t.year"] == 2000).all()

    def test_virtual_variable_same_name(self):
        # regression test for GH367
        times = pd.date_range("2000-01-01", freq="H", periods=5)
        data = Dataset({"time": times})
        actual = data["time.time"]
        expected = DataArray(times.time, [("time", times)], name="time")
        assert_identical(actual, expected)

    def test_virtual_variable_multiindex(self):
        # access multi-index levels as virtual variables
        data = create_test_multiindex()
        expected = DataArray(
            ["a", "a", "b", "b"],
            name="level_1",
            coords=[data["x"].to_index()],
            dims="x",
        )
        assert_identical(expected, data["level_1"])

        # combine multi-index level and datetime
        dr_index = pd.date_range("1/1/2011", periods=4, freq="H")
        mindex = pd.MultiIndex.from_arrays(
            [["a", "a", "b", "b"], dr_index], names=("level_str", "level_date")
        )
        data = Dataset({}, {"x": mindex})
        expected = DataArray(
            mindex.get_level_values("level_date").hour,
            name="hour",
            coords=[mindex],
            dims="x",
        )
        assert_identical(expected, data["level_date.hour"])

        # attribute style access
        assert_identical(data.level_str, data["level_str"])

    def test_time_season(self):
        ds = Dataset({"t": pd.date_range("2000-01-01", periods=12, freq="M")})
        seas = ["DJF"] * 2 + ["MAM"] * 3 + ["JJA"] * 3 + ["SON"] * 3 + ["DJF"]
        assert_array_equal(seas, ds["t.season"])

    def test_slice_virtual_variable(self):
        data = create_test_data()
        assert_equal(
            data["time.dayofyear"][:10].variable, Variable(["time"], 1 + np.arange(10))
        )
        assert_equal(data["time.dayofyear"][0].variable, Variable([], 1))

    def test_setitem(self):
        # assign a variable
        var = Variable(["dim1"], np.random.randn(8))
        data1 = create_test_data()
        data1["A"] = var
        data2 = data1.copy()
        data2["A"] = var
        assert_identical(data1, data2)
        # assign a dataset array
        dv = 2 * data2["A"]
        data1["B"] = dv.variable
        data2["B"] = dv
        assert_identical(data1, data2)
        # can't assign an ND array without dimensions
        with raises_regex(ValueError, "without explicit dimension names"):
            data2["C"] = var.values.reshape(2, 4)
        # but can assign a 1D array
        data1["C"] = var.values
        data2["C"] = ("C", var.values)
        assert_identical(data1, data2)
        # can assign a scalar
        data1["scalar"] = 0
        data2["scalar"] = ([], 0)
        assert_identical(data1, data2)
        # can't use the same dimension name as a scalar var
        with raises_regex(ValueError, "already exists as a scalar"):
            data1["newvar"] = ("scalar", [3, 4, 5])
        # can't resize a used dimension
        with raises_regex(ValueError, "arguments without labels"):
            data1["dim1"] = data1["dim1"][:5]
        # override an existing value
        data1["A"] = 3 * data2["A"]
        assert_equal(data1["A"], 3 * data2["A"])

        with pytest.raises(NotImplementedError):
            data1[{"x": 0}] = 0

    def test_setitem_pandas(self):

        ds = self.make_example_math_dataset()
        ds["x"] = np.arange(3)
        ds_copy = ds.copy()
        ds_copy["bar"] = ds["bar"].to_pandas()

        assert_equal(ds, ds_copy)

    def test_setitem_auto_align(self):
        ds = Dataset()
        ds["x"] = ("y", range(3))
        ds["y"] = 1 + np.arange(3)
        expected = Dataset({"x": ("y", range(3)), "y": 1 + np.arange(3)})
        assert_identical(ds, expected)

        ds["y"] = DataArray(range(3), dims="y")
        expected = Dataset({"x": ("y", range(3))}, {"y": range(3)})
        assert_identical(ds, expected)

        ds["x"] = DataArray([1, 2], coords=[("y", [0, 1])])
        expected = Dataset({"x": ("y", [1, 2, np.nan])}, {"y": range(3)})
        assert_identical(ds, expected)

        ds["x"] = 42
        expected = Dataset({"x": 42, "y": range(3)})
        assert_identical(ds, expected)

        ds["x"] = DataArray([4, 5, 6, 7], coords=[("y", [0, 1, 2, 3])])
        expected = Dataset({"x": ("y", [4, 5, 6])}, {"y": range(3)})
        assert_identical(ds, expected)

    def test_setitem_dimension_override(self):
        # regression test for GH-3377
        ds = xr.Dataset({"x": [0, 1, 2]})
        ds["x"] = ds["x"][:2]
        expected = Dataset({"x": [0, 1]})
        assert_identical(ds, expected)

        ds = xr.Dataset({"x": [0, 1, 2]})
        ds["x"] = np.array([0, 1])
        assert_identical(ds, expected)

        ds = xr.Dataset({"x": [0, 1, 2]})
        ds.coords["x"] = [0, 1]
        assert_identical(ds, expected)

    def test_setitem_with_coords(self):
        # Regression test for GH:2068
        ds = create_test_data()

        other = DataArray(
            np.arange(10), dims="dim3", coords={"numbers": ("dim3", np.arange(10))}
        )
        expected = ds.copy()
        expected["var3"] = other.drop_vars("numbers")
        actual = ds.copy()
        actual["var3"] = other
        assert_identical(expected, actual)
        assert "numbers" in other.coords  # should not change other

        # with alignment
        other = ds["var3"].isel(dim3=slice(1, -1))
        other["numbers"] = ("dim3", np.arange(8))
        actual = ds.copy()
        actual["var3"] = other
        assert "numbers" in other.coords  # should not change other
        expected = ds.copy()
        expected["var3"] = ds["var3"].isel(dim3=slice(1, -1))
        assert_identical(expected, actual)

        # with non-duplicate coords
        other = ds["var3"].isel(dim3=slice(1, -1))
        other["numbers"] = ("dim3", np.arange(8))
        other["position"] = ("dim3", np.arange(8))
        actual = ds.copy()
        actual["var3"] = other
        assert "position" in actual
        assert "position" in other.coords

        # assigning a coordinate-only dataarray
        actual = ds.copy()
        other = actual["numbers"]
        other[0] = 10
        actual["numbers"] = other
        assert actual["numbers"][0] == 10

        # GH: 2099
        ds = Dataset(
            {"var": ("x", [1, 2, 3])},
            coords={"x": [0, 1, 2], "z1": ("x", [1, 2, 3]), "z2": ("x", [1, 2, 3])},
        )
        ds["var"] = ds["var"] * 2
        assert np.allclose(ds["var"], [2, 4, 6])

    def test_setitem_align_new_indexes(self):
        ds = Dataset({"foo": ("x", [1, 2, 3])}, {"x": [0, 1, 2]})
        ds["bar"] = DataArray([2, 3, 4], [("x", [1, 2, 3])])
        expected = Dataset(
            {"foo": ("x", [1, 2, 3]), "bar": ("x", [np.nan, 2, 3])}, {"x": [0, 1, 2]}
        )
        assert_identical(ds, expected)

    @pytest.mark.parametrize("dtype", [str, bytes])
    def test_setitem_str_dtype(self, dtype):

        ds = xr.Dataset(coords={"x": np.array(["x", "y"], dtype=dtype)})
        ds["foo"] = xr.DataArray(np.array([0, 0]), dims=["x"])

        assert np.issubdtype(ds.x.dtype, dtype)

    def test_assign(self):
        ds = Dataset()
        actual = ds.assign(x=[0, 1, 2], y=2)
        expected = Dataset({"x": [0, 1, 2], "y": 2})
        assert_identical(actual, expected)
        assert list(actual.variables) == ["x", "y"]
        assert_identical(ds, Dataset())

        actual = actual.assign(y=lambda ds: ds.x ** 2)
        expected = Dataset({"y": ("x", [0, 1, 4]), "x": [0, 1, 2]})
        assert_identical(actual, expected)

        actual = actual.assign_coords(z=2)
        expected = Dataset({"y": ("x", [0, 1, 4])}, {"z": 2, "x": [0, 1, 2]})
        assert_identical(actual, expected)

        ds = Dataset({"a": ("x", range(3))}, {"b": ("x", ["A"] * 2 + ["B"])})
        actual = ds.groupby("b").assign(c=lambda ds: 2 * ds.a)
        expected = ds.merge({"c": ("x", [0, 2, 4])})
        assert_identical(actual, expected)

        actual = ds.groupby("b").assign(c=lambda ds: ds.a.sum())
        expected = ds.merge({"c": ("x", [1, 1, 2])})
        assert_identical(actual, expected)

        actual = ds.groupby("b").assign_coords(c=lambda ds: ds.a.sum())
        expected = expected.set_coords("c")
        assert_identical(actual, expected)

    def test_assign_coords(self):
        ds = Dataset()

        actual = ds.assign(x=[0, 1, 2], y=2)
        actual = actual.assign_coords(x=list("abc"))
        expected = Dataset({"x": list("abc"), "y": 2})
        assert_identical(actual, expected)

        actual = ds.assign(x=[0, 1, 2], y=[2, 3])
        actual = actual.assign_coords({"y": [2.0, 3.0]})
        expected = ds.assign(x=[0, 1, 2], y=[2.0, 3.0])
        assert_identical(actual, expected)

    def test_assign_attrs(self):
        expected = Dataset(attrs=dict(a=1, b=2))
        new = Dataset()
        actual = new.assign_attrs(a=1, b=2)
        assert_identical(actual, expected)
        assert new.attrs == {}

        expected.attrs["c"] = 3
        new_actual = actual.assign_attrs({"c": 3})
        assert_identical(new_actual, expected)
        assert actual.attrs == dict(a=1, b=2)

    def test_assign_multiindex_level(self):
        data = create_test_multiindex()
        with raises_regex(ValueError, "conflicting MultiIndex"):
            data.assign(level_1=range(4))
            data.assign_coords(level_1=range(4))
        # raise an Error when any level name is used as dimension GH:2299
        with pytest.raises(ValueError):
            data["y"] = ("level_1", [0, 1])

    def test_merge_multiindex_level(self):
        data = create_test_multiindex()
        other = Dataset({"z": ("level_1", [0, 1])})  # conflict dimension
        with pytest.raises(ValueError):
            data.merge(other)
        other = Dataset({"level_1": ("x", [0, 1])})  # conflict variable name
        with pytest.raises(ValueError):
            data.merge(other)

    def test_setitem_original_non_unique_index(self):
        # regression test for GH943
        original = Dataset({"data": ("x", np.arange(5))}, coords={"x": [0, 1, 2, 0, 1]})
        expected = Dataset({"data": ("x", np.arange(5))}, {"x": range(5)})

        actual = original.copy()
        actual["x"] = list(range(5))
        assert_identical(actual, expected)

        actual = original.copy()
        actual["x"] = ("x", list(range(5)))
        assert_identical(actual, expected)

        actual = original.copy()
        actual.coords["x"] = list(range(5))
        assert_identical(actual, expected)

    def test_setitem_both_non_unique_index(self):
        # regression test for GH956
        names = ["joaquin", "manolo", "joaquin"]
        values = np.random.randint(0, 256, (3, 4, 4))
        array = DataArray(
            values, dims=["name", "row", "column"], coords=[names, range(4), range(4)]
        )
        expected = Dataset({"first": array, "second": array})
        actual = array.rename("first").to_dataset()
        actual["second"] = array
        assert_identical(expected, actual)

    def test_setitem_multiindex_level(self):
        data = create_test_multiindex()
        with raises_regex(ValueError, "conflicting MultiIndex"):
            data["level_1"] = range(4)

    def test_delitem(self):
        data = create_test_data()
        all_items = set(data.variables)
        assert set(data.variables) == all_items
        del data["var1"]
        assert set(data.variables) == all_items - {"var1"}
        del data["numbers"]
        assert set(data.variables) == all_items - {"var1", "numbers"}
        assert "numbers" not in data.coords

        expected = Dataset()
        actual = Dataset({"y": ("x", [1, 2])})
        del actual["y"]
        assert_identical(expected, actual)

    def test_squeeze(self):
        data = Dataset({"foo": (["x", "y", "z"], [[[1], [2]]])})
        for args in [[], [["x"]], [["x", "z"]]]:

            def get_args(v):
                return [set(args[0]) & set(v.dims)] if args else []

            expected = Dataset(
                {k: v.squeeze(*get_args(v)) for k, v in data.variables.items()}
            )
            expected = expected.set_coords(data.coords)
            assert_identical(expected, data.squeeze(*args))
        # invalid squeeze
        with raises_regex(ValueError, "cannot select a dimension"):
            data.squeeze("y")

    def test_squeeze_drop(self):
        data = Dataset({"foo": ("x", [1])}, {"x": [0]})
        expected = Dataset({"foo": 1})
        selected = data.squeeze(drop=True)
        assert_identical(expected, selected)

        expected = Dataset({"foo": 1}, {"x": 0})
        selected = data.squeeze(drop=False)
        assert_identical(expected, selected)

        data = Dataset({"foo": (("x", "y"), [[1]])}, {"x": [0], "y": [0]})
        expected = Dataset({"foo": 1})
        selected = data.squeeze(drop=True)
        assert_identical(expected, selected)

        expected = Dataset({"foo": ("x", [1])}, {"x": [0]})
        selected = data.squeeze(dim="y", drop=True)
        assert_identical(expected, selected)

        data = Dataset({"foo": (("x",), [])}, {"x": []})
        selected = data.squeeze(drop=True)
        assert_identical(data, selected)

    def test_groupby(self):
        data = Dataset(
            {"z": (["x", "y"], np.random.randn(3, 5))},
            {"x": ("x", list("abc")), "c": ("x", [0, 1, 0]), "y": range(5)},
        )
        groupby = data.groupby("x")
        assert len(groupby) == 3
        expected_groups = {"a": 0, "b": 1, "c": 2}
        assert groupby.groups == expected_groups
        expected_items = [
            ("a", data.isel(x=0)),
            ("b", data.isel(x=1)),
            ("c", data.isel(x=2)),
        ]
        for actual, expected in zip(groupby, expected_items):
            assert actual[0] == expected[0]
            assert_equal(actual[1], expected[1])

        def identity(x):
            return x

        for k in ["x", "c", "y"]:
            actual = data.groupby(k, squeeze=False).map(identity)
            assert_equal(data, actual)

    def test_groupby_returns_new_type(self):
        data = Dataset({"z": (["x", "y"], np.random.randn(3, 5))})

        actual = data.groupby("x").map(lambda ds: ds["z"])
        expected = data["z"]
        assert_identical(expected, actual)

        actual = data["z"].groupby("x").map(lambda x: x.to_dataset())
        expected = data
        assert_identical(expected, actual)

    def test_groupby_iter(self):
        data = create_test_data()
        for n, (t, sub) in enumerate(list(data.groupby("dim1"))[:3]):
            assert data["dim1"][n] == t
            assert_equal(data["var1"][n], sub["var1"])
            assert_equal(data["var2"][n], sub["var2"])
            assert_equal(data["var3"][:, n], sub["var3"])

    def test_groupby_errors(self):
        data = create_test_data()
        with raises_regex(TypeError, "`group` must be"):
            data.groupby(np.arange(10))
        with raises_regex(ValueError, "length does not match"):
            data.groupby(data["dim1"][:3])
        with raises_regex(TypeError, "`group` must be"):
            data.groupby(data.coords["dim1"].to_index())

    def test_groupby_reduce(self):
        data = Dataset(
            {
                "xy": (["x", "y"], np.random.randn(3, 4)),
                "xonly": ("x", np.random.randn(3)),
                "yonly": ("y", np.random.randn(4)),
                "letters": ("y", ["a", "a", "b", "b"]),
            }
        )

        expected = data.mean("y")
        expected["yonly"] = expected["yonly"].variable.set_dims({"x": 3})
        actual = data.groupby("x").mean(...)
        assert_allclose(expected, actual)

        actual = data.groupby("x").mean("y")
        assert_allclose(expected, actual)

        letters = data["letters"]
        expected = Dataset(
            {
                "xy": data["xy"].groupby(letters).mean(...),
                "xonly": (data["xonly"].mean().variable.set_dims({"letters": 2})),
                "yonly": data["yonly"].groupby(letters).mean(),
            }
        )
        actual = data.groupby("letters").mean(...)
        assert_allclose(expected, actual)

    def test_groupby_math(self):
        def reorder_dims(x):
            return x.transpose("dim1", "dim2", "dim3", "time")

        ds = create_test_data()
        ds["dim1"] = ds["dim1"]
        for squeeze in [True, False]:
            grouped = ds.groupby("dim1", squeeze=squeeze)

            expected = reorder_dims(ds + ds.coords["dim1"])
            actual = grouped + ds.coords["dim1"]
            assert_identical(expected, reorder_dims(actual))

            actual = ds.coords["dim1"] + grouped
            assert_identical(expected, reorder_dims(actual))

            ds2 = 2 * ds
            expected = reorder_dims(ds + ds2)
            actual = grouped + ds2
            assert_identical(expected, reorder_dims(actual))

            actual = ds2 + grouped
            assert_identical(expected, reorder_dims(actual))

        grouped = ds.groupby("numbers")
        zeros = DataArray([0, 0, 0, 0], [("numbers", range(4))])
        expected = (ds + Variable("dim3", np.zeros(10))).transpose(
            "dim3", "dim1", "dim2", "time"
        )
        actual = grouped + zeros
        assert_equal(expected, actual)

        actual = zeros + grouped
        assert_equal(expected, actual)

        with raises_regex(ValueError, "incompat.* grouped binary"):
            grouped + ds
        with raises_regex(ValueError, "incompat.* grouped binary"):
            ds + grouped
        with raises_regex(TypeError, "only support binary ops"):
            grouped + 1
        with raises_regex(TypeError, "only support binary ops"):
            grouped + grouped
        with raises_regex(TypeError, "in-place operations"):
            ds += grouped

        ds = Dataset(
            {
                "x": ("time", np.arange(100)),
                "time": pd.date_range("2000-01-01", periods=100),
            }
        )
        with raises_regex(ValueError, "incompat.* grouped binary"):
            ds + ds.groupby("time.month")

    def test_groupby_math_virtual(self):
        ds = Dataset(
            {"x": ("t", [1, 2, 3])}, {"t": pd.date_range("20100101", periods=3)}
        )
        grouped = ds.groupby("t.day")
        actual = grouped - grouped.mean(...)
        expected = Dataset({"x": ("t", [0, 0, 0])}, ds[["t", "t.day"]])
        assert_identical(actual, expected)

    def test_groupby_nan(self):
        # nan should be excluded from groupby
        ds = Dataset({"foo": ("x", [1, 2, 3, 4])}, {"bar": ("x", [1, 1, 2, np.nan])})
        actual = ds.groupby("bar").mean(...)
        expected = Dataset({"foo": ("bar", [1.5, 3]), "bar": [1, 2]})
        assert_identical(actual, expected)

    def test_groupby_order(self):
        # groupby should preserve variables order
        ds = Dataset()
        for vn in ["a", "b", "c"]:
            ds[vn] = DataArray(np.arange(10), dims=["t"])
        data_vars_ref = list(ds.data_vars.keys())
        ds = ds.groupby("t").mean(...)
        data_vars = list(ds.data_vars.keys())
        assert data_vars == data_vars_ref
        # coords are now at the end of the list, so the test below fails
        # all_vars = list(ds.variables.keys())
        # all_vars_ref = list(ds.variables.keys())
        # self.assertEqual(all_vars, all_vars_ref)

    def test_resample_and_first(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )

        actual = ds.resample(time="1D").first(keep_attrs=True)
        expected = ds.isel(time=[0, 4, 8])
        assert_identical(expected, actual)

        # upsampling
        expected_time = pd.date_range("2000-01-01", freq="3H", periods=19)
        expected = ds.reindex(time=expected_time)
        actual = ds.resample(time="3H")
        for how in ["mean", "sum", "first", "last"]:
            method = getattr(actual, how)
            result = method()
            assert_equal(expected, result)
        for method in [np.mean]:
            result = actual.reduce(method)
            assert_equal(expected, result)

    def test_resample_min_count(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )
        # inject nan
        ds["foo"] = xr.where(ds["foo"] > 2.0, np.nan, ds["foo"])

        actual = ds.resample(time="1D").sum(min_count=1)
        expected = xr.concat(
            [
                ds.isel(time=slice(i * 4, (i + 1) * 4)).sum("time", min_count=1)
                for i in range(3)
            ],
            dim=actual["time"],
        )
        assert_equal(expected, actual)

    def test_resample_by_mean_with_keep_attrs(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )
        ds.attrs["dsmeta"] = "dsdata"

        resampled_ds = ds.resample(time="1D").mean(keep_attrs=True)
        actual = resampled_ds["bar"].attrs
        expected = ds["bar"].attrs
        assert expected == actual

        actual = resampled_ds.attrs
        expected = ds.attrs
        assert expected == actual

    def test_resample_loffset(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )
        ds.attrs["dsmeta"] = "dsdata"

        actual = ds.resample(time="24H", loffset="-12H").mean("time").time
        expected = xr.DataArray(
            ds.bar.to_series().resample("24H", loffset="-12H").mean()
        ).time
        assert_identical(expected, actual)

    def test_resample_by_mean_discarding_attrs(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )
        ds.attrs["dsmeta"] = "dsdata"

        resampled_ds = ds.resample(time="1D").mean(keep_attrs=False)

        assert resampled_ds["bar"].attrs == {}
        assert resampled_ds.attrs == {}

    def test_resample_by_last_discarding_attrs(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )
        ds.attrs["dsmeta"] = "dsdata"

        resampled_ds = ds.resample(time="1D").last(keep_attrs=False)

        assert resampled_ds["bar"].attrs == {}
        assert resampled_ds.attrs == {}

    @requires_scipy
    def test_resample_drop_nondim_coords(self):
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range("2000-01-01", freq="6H", periods=5)
        data = np.tile(np.arange(5), (6, 3, 1))
        xx, yy = np.meshgrid(xs * 5, ys * 2.5)
        tt = np.arange(len(times), dtype=int)
        array = DataArray(data, {"time": times, "x": xs, "y": ys}, ("x", "y", "time"))
        xcoord = DataArray(xx.T, {"x": xs, "y": ys}, ("x", "y"))
        ycoord = DataArray(yy.T, {"x": xs, "y": ys}, ("x", "y"))
        tcoord = DataArray(tt, {"time": times}, ("time",))
        ds = Dataset({"data": array, "xc": xcoord, "yc": ycoord, "tc": tcoord})
        ds = ds.set_coords(["xc", "yc", "tc"])

        # Re-sample
        actual = ds.resample(time="12H").mean("time")
        assert "tc" not in actual.coords

        # Up-sample - filling
        actual = ds.resample(time="1H").ffill()
        assert "tc" not in actual.coords

        # Up-sample - interpolation
        actual = ds.resample(time="1H").interpolate("linear")
        assert "tc" not in actual.coords

    def test_resample_old_api(self):

        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )

        with raises_regex(TypeError, r"resample\(\) no longer supports"):
            ds.resample("1D", "time")

        with raises_regex(TypeError, r"resample\(\) no longer supports"):
            ds.resample("1D", dim="time", how="mean")

        with raises_regex(TypeError, r"resample\(\) no longer supports"):
            ds.resample("1D", dim="time")

    def test_resample_ds_da_are_the_same(self):
        time = pd.date_range("2000-01-01", freq="6H", periods=365 * 4)
        ds = xr.Dataset(
            {
                "foo": (("time", "x"), np.random.randn(365 * 4, 5)),
                "time": time,
                "x": np.arange(5),
            }
        )
        assert_identical(
            ds.resample(time="M").mean()["foo"], ds.foo.resample(time="M").mean()
        )

    def test_ds_resample_apply_func_args(self):
        def func(arg1, arg2, arg3=0.0):
            return arg1.mean("time") + arg2 + arg3

        times = pd.date_range("2000", freq="D", periods=3)
        ds = xr.Dataset({"foo": ("time", [1.0, 1.0, 1.0]), "time": times})
        expected = xr.Dataset({"foo": ("time", [3.0, 3.0, 3.0]), "time": times})
        actual = ds.resample(time="D").map(func, args=(1.0,), arg3=1.0)
        assert_identical(expected, actual)

    def test_to_array(self):
        ds = Dataset(
            {"a": 1, "b": ("x", [1, 2, 3])},
            coords={"c": 42},
            attrs={"Conventions": "None"},
        )
        data = [[1, 1, 1], [1, 2, 3]]
        coords = {"c": 42, "variable": ["a", "b"]}
        dims = ("variable", "x")
        expected = DataArray(data, coords, dims, attrs=ds.attrs)
        actual = ds.to_array()
        assert_identical(expected, actual)

        actual = ds.to_array("abc", name="foo")
        expected = expected.rename({"variable": "abc"}).rename("foo")
        assert_identical(expected, actual)

    def test_to_and_from_dataframe(self):
        x = np.random.randn(10)
        y = np.random.randn(10)
        t = list("abcdefghij")
        ds = Dataset({"a": ("t", x), "b": ("t", y), "t": ("t", t)})
        expected = pd.DataFrame(
            np.array([x, y]).T, columns=["a", "b"], index=pd.Index(t, name="t")
        )
        actual = ds.to_dataframe()
        # use the .equals method to check all DataFrame metadata
        assert expected.equals(actual), (expected, actual)

        # verify coords are included
        actual = ds.set_coords("b").to_dataframe()
        assert expected.equals(actual), (expected, actual)

        # check roundtrip
        assert_identical(ds, Dataset.from_dataframe(actual))

        # test a case with a MultiIndex
        w = np.random.randn(2, 3)
        ds = Dataset({"w": (("x", "y"), w)})
        ds["y"] = ("y", list("abc"))
        exp_index = pd.MultiIndex.from_arrays(
            [[0, 0, 0, 1, 1, 1], ["a", "b", "c", "a", "b", "c"]], names=["x", "y"]
        )
        expected = pd.DataFrame(w.reshape(-1), columns=["w"], index=exp_index)
        actual = ds.to_dataframe()
        assert expected.equals(actual)

        # check roundtrip
        assert_identical(ds.assign_coords(x=[0, 1]), Dataset.from_dataframe(actual))

        # Check multiindex reordering
        new_order = ["x", "y"]
        actual = ds.to_dataframe(dim_order=new_order)
        assert expected.equals(actual)

        new_order = ["y", "x"]
        exp_index = pd.MultiIndex.from_arrays(
            [["a", "a", "b", "b", "c", "c"], [0, 1, 0, 1, 0, 1]], names=["y", "x"]
        )
        expected = pd.DataFrame(
            w.transpose().reshape(-1), columns=["w"], index=exp_index
        )
        actual = ds.to_dataframe(dim_order=new_order)
        assert expected.equals(actual)

        invalid_order = ["x"]
        with pytest.raises(
            ValueError, match="does not match the set of dimensions of this"
        ):
            ds.to_dataframe(dim_order=invalid_order)

        invalid_order = ["x", "z"]
        with pytest.raises(
            ValueError, match="does not match the set of dimensions of this"
        ):
            ds.to_dataframe(dim_order=invalid_order)

        # check pathological cases
        df = pd.DataFrame([1])
        actual = Dataset.from_dataframe(df)
        expected = Dataset({0: ("index", [1])}, {"index": [0]})
        assert_identical(expected, actual)

        df = pd.DataFrame()
        actual = Dataset.from_dataframe(df)
        expected = Dataset(coords={"index": []})
        assert_identical(expected, actual)

        # GH697
        df = pd.DataFrame({"A": []})
        actual = Dataset.from_dataframe(df)
        expected = Dataset({"A": DataArray([], dims=("index",))}, {"index": []})
        assert_identical(expected, actual)

        # regression test for GH278
        # use int64 to ensure consistent results for the pandas .equals method
        # on windows (which requires the same dtype)
        ds = Dataset({"x": pd.Index(["bar"]), "a": ("y", np.array([1], "int64"))}).isel(
            x=0
        )
        # use .loc to ensure consistent results on Python 3
        actual = ds.to_dataframe().loc[:, ["a", "x"]]
        expected = pd.DataFrame(
            [[1, "bar"]], index=pd.Index([0], name="y"), columns=["a", "x"]
        )
        assert expected.equals(actual), (expected, actual)

        ds = Dataset({"x": np.array([0], "int64"), "y": np.array([1], "int64")})
        actual = ds.to_dataframe()
        idx = pd.MultiIndex.from_arrays([[0], [1]], names=["x", "y"])
        expected = pd.DataFrame([[]], index=idx)
        assert expected.equals(actual), (expected, actual)

    def test_from_dataframe_categorical(self):
        cat = pd.CategoricalDtype(
            categories=["foo", "bar", "baz", "qux", "quux", "corge"]
        )
        i1 = pd.Series(["foo", "bar", "foo"], dtype=cat)
        i2 = pd.Series(["bar", "bar", "baz"], dtype=cat)

        df = pd.DataFrame({"i1": i1, "i2": i2, "values": [1, 2, 3]})
        ds = df.set_index("i1").to_xarray()
        assert len(ds["i1"]) == 3

        ds = df.set_index(["i1", "i2"]).to_xarray()
        assert len(ds["i1"]) == 2
        assert len(ds["i2"]) == 2

    @requires_sparse
    def test_from_dataframe_sparse(self):
        import sparse

        df_base = pd.DataFrame(
            {"x": range(10), "y": list("abcdefghij"), "z": np.arange(0, 100, 10)}
        )

        ds_sparse = Dataset.from_dataframe(df_base.set_index("x"), sparse=True)
        ds_dense = Dataset.from_dataframe(df_base.set_index("x"), sparse=False)
        assert isinstance(ds_sparse["y"].data, sparse.COO)
        assert isinstance(ds_sparse["z"].data, sparse.COO)
        ds_sparse["y"].data = ds_sparse["y"].data.todense()
        ds_sparse["z"].data = ds_sparse["z"].data.todense()
        assert_identical(ds_dense, ds_sparse)

        ds_sparse = Dataset.from_dataframe(df_base.set_index(["x", "y"]), sparse=True)
        ds_dense = Dataset.from_dataframe(df_base.set_index(["x", "y"]), sparse=False)
        assert isinstance(ds_sparse["z"].data, sparse.COO)
        ds_sparse["z"].data = ds_sparse["z"].data.todense()
        assert_identical(ds_dense, ds_sparse)

    def test_to_and_from_empty_dataframe(self):
        # GH697
        expected = pd.DataFrame({"foo": []})
        ds = Dataset.from_dataframe(expected)
        assert len(ds["foo"]) == 0
        actual = ds.to_dataframe()
        assert len(actual) == 0
        assert expected.equals(actual)

    def test_from_dataframe_multiindex(self):
        index = pd.MultiIndex.from_product([["a", "b"], [1, 2, 3]], names=["x", "y"])
        df = pd.DataFrame({"z": np.arange(6)}, index=index)

        expected = Dataset(
            {"z": (("x", "y"), [[0, 1, 2], [3, 4, 5]])},
            coords={"x": ["a", "b"], "y": [1, 2, 3]},
        )
        actual = Dataset.from_dataframe(df)
        assert_identical(actual, expected)

        df2 = df.iloc[[3, 2, 1, 0, 4, 5], :]
        actual = Dataset.from_dataframe(df2)
        assert_identical(actual, expected)

        df3 = df.iloc[:4, :]
        expected3 = Dataset(
            {"z": (("x", "y"), [[0, 1, 2], [3, np.nan, np.nan]])},
            coords={"x": ["a", "b"], "y": [1, 2, 3]},
        )
        actual = Dataset.from_dataframe(df3)
        assert_identical(actual, expected3)

        df_nonunique = df.iloc[[0, 0], :]
        with raises_regex(ValueError, "non-unique MultiIndex"):
            Dataset.from_dataframe(df_nonunique)

    def test_from_dataframe_unsorted_levels(self):
        # regression test for GH-4186
        index = pd.MultiIndex(
            levels=[["b", "a"], ["foo"]], codes=[[0, 1], [0, 0]], names=["lev1", "lev2"]
        )
        df = pd.DataFrame({"c1": [0, 2], "c2": [1, 3]}, index=index)
        expected = Dataset(
            {
                "c1": (("lev1", "lev2"), [[0], [2]]),
                "c2": (("lev1", "lev2"), [[1], [3]]),
            },
            coords={"lev1": ["b", "a"], "lev2": ["foo"]},
        )
        actual = Dataset.from_dataframe(df)
        assert_identical(actual, expected)

    def test_from_dataframe_non_unique_columns(self):
        # regression test for GH449
        df = pd.DataFrame(np.zeros((2, 2)))
        df.columns = ["foo", "foo"]
        with raises_regex(ValueError, "non-unique columns"):
            Dataset.from_dataframe(df)

    def test_convert_dataframe_with_many_types_and_multiindex(self):
        # regression test for GH737
        df = pd.DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
            }
        )
        df.index = pd.MultiIndex.from_product([["a"], range(3)], names=["one", "two"])
        roundtripped = Dataset.from_dataframe(df).to_dataframe()
        # we can't do perfectly, but we should be at least as faithful as
        # np.asarray
        expected = df.apply(np.asarray)
        assert roundtripped.equals(expected)

    def test_to_and_from_dict(self):
        # <xarray.Dataset>
        # Dimensions:  (t: 10)
        # Coordinates:
        #   * t        (t) <U1 'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j'
        # Data variables:
        #     a        (t) float64 0.6916 -1.056 -1.163 0.9792 -0.7865 ...
        #     b        (t) float64 1.32 0.1954 1.91 1.39 0.519 -0.2772 ...
        x = np.random.randn(10)
        y = np.random.randn(10)
        t = list("abcdefghij")
        ds = Dataset({"a": ("t", x), "b": ("t", y), "t": ("t", t)})
        expected = {
            "coords": {"t": {"dims": ("t",), "data": t, "attrs": {}}},
            "attrs": {},
            "dims": {"t": 10},
            "data_vars": {
                "a": {"dims": ("t",), "data": x.tolist(), "attrs": {}},
                "b": {"dims": ("t",), "data": y.tolist(), "attrs": {}},
            },
        }

        actual = ds.to_dict()

        # check that they are identical
        assert expected == actual

        # check roundtrip
        assert_identical(ds, Dataset.from_dict(actual))

        # check the data=False option
        expected_no_data = expected.copy()
        del expected_no_data["coords"]["t"]["data"]
        del expected_no_data["data_vars"]["a"]["data"]
        del expected_no_data["data_vars"]["b"]["data"]
        endiantype = "<U1" if sys.byteorder == "little" else ">U1"
        expected_no_data["coords"]["t"].update({"dtype": endiantype, "shape": (10,)})
        expected_no_data["data_vars"]["a"].update({"dtype": "float64", "shape": (10,)})
        expected_no_data["data_vars"]["b"].update({"dtype": "float64", "shape": (10,)})
        actual_no_data = ds.to_dict(data=False)
        assert expected_no_data == actual_no_data

        # verify coords are included roundtrip
        expected_ds = ds.set_coords("b")
        actual = Dataset.from_dict(expected_ds.to_dict())

        assert_identical(expected_ds, actual)

        # test some incomplete dicts:
        # this one has no attrs field, the dims are strings, and x, y are
        # np.arrays

        d = {
            "coords": {"t": {"dims": "t", "data": t}},
            "dims": "t",
            "data_vars": {"a": {"dims": "t", "data": x}, "b": {"dims": "t", "data": y}},
        }
        assert_identical(ds, Dataset.from_dict(d))

        # this is kind of a flattened version with no coords, or data_vars
        d = {
            "a": {"dims": "t", "data": x},
            "t": {"data": t, "dims": "t"},
            "b": {"dims": "t", "data": y},
        }
        assert_identical(ds, Dataset.from_dict(d))

        # this one is missing some necessary information
        d = {
            "a": {"data": x},
            "t": {"data": t, "dims": "t"},
            "b": {"dims": "t", "data": y},
        }
        with raises_regex(ValueError, "cannot convert dict without the key 'dims'"):
            Dataset.from_dict(d)

    def test_to_and_from_dict_with_time_dim(self):
        x = np.random.randn(10, 3)
        y = np.random.randn(10, 3)
        t = pd.date_range("20130101", periods=10)
        lat = [77.7, 83.2, 76]
        ds = Dataset(
            {
                "a": (["t", "lat"], x),
                "b": (["t", "lat"], y),
                "t": ("t", t),
                "lat": ("lat", lat),
            }
        )
        roundtripped = Dataset.from_dict(ds.to_dict())
        assert_identical(ds, roundtripped)

    def test_to_and_from_dict_with_nan_nat(self):
        x = np.random.randn(10, 3)
        y = np.random.randn(10, 3)
        y[2] = np.nan
        t = pd.Series(pd.date_range("20130101", periods=10))
        t[2] = np.nan

        lat = [77.7, 83.2, 76]
        ds = Dataset(
            {
                "a": (["t", "lat"], x),
                "b": (["t", "lat"], y),
                "t": ("t", t),
                "lat": ("lat", lat),
            }
        )
        roundtripped = Dataset.from_dict(ds.to_dict())
        assert_identical(ds, roundtripped)

    def test_to_dict_with_numpy_attrs(self):
        # this doesn't need to roundtrip
        x = np.random.randn(10)
        y = np.random.randn(10)
        t = list("abcdefghij")
        attrs = {
            "created": np.float64(1998),
            "coords": np.array([37, -110.1, 100]),
            "maintainer": "bar",
        }
        ds = Dataset({"a": ("t", x, attrs), "b": ("t", y, attrs), "t": ("t", t)})
        expected_attrs = {
            "created": attrs["created"].item(),
            "coords": attrs["coords"].tolist(),
            "maintainer": "bar",
        }
        actual = ds.to_dict()

        # check that they are identical
        assert expected_attrs == actual["data_vars"]["a"]["attrs"]

    def test_pickle(self):
        data = create_test_data()
        roundtripped = pickle.loads(pickle.dumps(data))
        assert_identical(data, roundtripped)
        # regression test for #167:
        assert data.dims == roundtripped.dims

    def test_lazy_load(self):
        store = InaccessibleVariableDataStore()
        create_test_data().dump_to_store(store)

        for decode_cf in [True, False]:
            ds = open_dataset(store, decode_cf=decode_cf)
            with pytest.raises(UnexpectedDataAccess):
                ds.load()
            with pytest.raises(UnexpectedDataAccess):
                ds["var1"].values

            # these should not raise UnexpectedDataAccess:
            ds.isel(time=10)
            ds.isel(time=slice(10), dim1=[0]).isel(dim1=0, dim2=-1)

    def test_dropna(self):
        x = np.random.randn(4, 4)
        x[::2, 0] = np.nan
        y = np.random.randn(4)
        y[-1] = np.nan
        ds = Dataset({"foo": (("a", "b"), x), "bar": (("b", y))})

        expected = ds.isel(a=slice(1, None, 2))
        actual = ds.dropna("a")
        assert_identical(actual, expected)

        expected = ds.isel(b=slice(1, 3))
        actual = ds.dropna("b")
        assert_identical(actual, expected)

        actual = ds.dropna("b", subset=["foo", "bar"])
        assert_identical(actual, expected)

        expected = ds.isel(b=slice(1, None))
        actual = ds.dropna("b", subset=["foo"])
        assert_identical(actual, expected)

        expected = ds.isel(b=slice(3))
        actual = ds.dropna("b", subset=["bar"])
        assert_identical(actual, expected)

        actual = ds.dropna("a", subset=[])
        assert_identical(actual, ds)

        actual = ds.dropna("a", subset=["bar"])
        assert_identical(actual, ds)

        actual = ds.dropna("a", how="all")
        assert_identical(actual, ds)

        actual = ds.dropna("b", how="all", subset=["bar"])
        expected = ds.isel(b=[0, 1, 2])
        assert_identical(actual, expected)

        actual = ds.dropna("b", thresh=1, subset=["bar"])
        assert_identical(actual, expected)

        actual = ds.dropna("b", thresh=2)
        assert_identical(actual, ds)

        actual = ds.dropna("b", thresh=4)
        expected = ds.isel(b=[1, 2, 3])
        assert_identical(actual, expected)

        actual = ds.dropna("a", thresh=3)
        expected = ds.isel(a=[1, 3])
        assert_identical(actual, ds)

        with raises_regex(ValueError, "a single dataset dimension"):
            ds.dropna("foo")
        with raises_regex(ValueError, "invalid how"):
            ds.dropna("a", how="somehow")
        with raises_regex(TypeError, "must specify how or thresh"):
            ds.dropna("a", how=None)

    def test_fillna(self):
        ds = Dataset({"a": ("x", [np.nan, 1, np.nan, 3])}, {"x": [0, 1, 2, 3]})

        # fill with -1
        actual = ds.fillna(-1)
        expected = Dataset({"a": ("x", [-1, 1, -1, 3])}, {"x": [0, 1, 2, 3]})
        assert_identical(expected, actual)

        actual = ds.fillna({"a": -1})
        assert_identical(expected, actual)

        other = Dataset({"a": -1})
        actual = ds.fillna(other)
        assert_identical(expected, actual)

        actual = ds.fillna({"a": other.a})
        assert_identical(expected, actual)

        # fill with range(4)
        b = DataArray(range(4), coords=[("x", range(4))])
        actual = ds.fillna(b)
        expected = b.rename("a").to_dataset()
        assert_identical(expected, actual)

        actual = ds.fillna(expected)
        assert_identical(expected, actual)

        actual = ds.fillna(range(4))
        assert_identical(expected, actual)

        actual = ds.fillna(b[:3])
        assert_identical(expected, actual)

        # okay to only include some data variables
        ds["b"] = np.nan
        actual = ds.fillna({"a": -1})
        expected = Dataset(
            {"a": ("x", [-1, 1, -1, 3]), "b": np.nan}, {"x": [0, 1, 2, 3]}
        )
        assert_identical(expected, actual)

        # but new data variables is not okay
        with raises_regex(ValueError, "must be contained"):
            ds.fillna({"x": 0})

        # empty argument should be OK
        result = ds.fillna({})
        assert_identical(ds, result)

        result = ds.fillna(Dataset(coords={"c": 42}))
        expected = ds.assign_coords(c=42)
        assert_identical(expected, result)

        # groupby
        expected = Dataset({"a": ("x", range(4))}, {"x": [0, 1, 2, 3]})
        for target in [ds, expected]:
            target.coords["b"] = ("x", [0, 0, 1, 1])
        actual = ds.groupby("b").fillna(DataArray([0, 2], dims="b"))
        assert_identical(expected, actual)

        actual = ds.groupby("b").fillna(Dataset({"a": ("b", [0, 2])}))
        assert_identical(expected, actual)

        # attrs with groupby
        ds.attrs["attr"] = "ds"
        ds.a.attrs["attr"] = "da"
        actual = ds.groupby("b").fillna(Dataset({"a": ("b", [0, 2])}))
        assert actual.attrs == ds.attrs
        assert actual.a.name == "a"
        assert actual.a.attrs == ds.a.attrs

        da = DataArray(range(5), name="a", attrs={"attr": "da"})
        actual = da.fillna(1)
        assert actual.name == "a"
        assert actual.attrs == da.attrs

        ds = Dataset({"a": da}, attrs={"attr": "ds"})
        actual = ds.fillna({"a": 1})
        assert actual.attrs == ds.attrs
        assert actual.a.name == "a"
        assert actual.a.attrs == ds.a.attrs

    @pytest.mark.parametrize(
        "func", [lambda x: x.clip(0, 1), lambda x: np.float64(1.0) * x, np.abs, abs]
    )
    def test_propagate_attrs(self, func):

        da = DataArray(range(5), name="a", attrs={"attr": "da"})
        ds = Dataset({"a": da}, attrs={"attr": "ds"})

        # test defaults
        assert func(ds).attrs == ds.attrs
        with set_options(keep_attrs=False):
            assert func(ds).attrs != ds.attrs
            assert func(ds).a.attrs != ds.a.attrs

        with set_options(keep_attrs=False):
            assert func(ds).attrs != ds.attrs
            assert func(ds).a.attrs != ds.a.attrs

        with set_options(keep_attrs=True):
            assert func(ds).attrs == ds.attrs
            assert func(ds).a.attrs == ds.a.attrs

    def test_where(self):
        ds = Dataset({"a": ("x", range(5))})
        expected = Dataset({"a": ("x", [np.nan, np.nan, 2, 3, 4])})
        actual = ds.where(ds > 1)
        assert_identical(expected, actual)

        actual = ds.where(ds.a > 1)
        assert_identical(expected, actual)

        actual = ds.where(ds.a.values > 1)
        assert_identical(expected, actual)

        actual = ds.where(True)
        assert_identical(ds, actual)

        expected = ds.copy(deep=True)
        expected["a"].values = [np.nan] * 5
        actual = ds.where(False)
        assert_identical(expected, actual)

        # 2d
        ds = Dataset({"a": (("x", "y"), [[0, 1], [2, 3]])})
        expected = Dataset({"a": (("x", "y"), [[np.nan, 1], [2, 3]])})
        actual = ds.where(ds > 0)
        assert_identical(expected, actual)

        # groupby
        ds = Dataset({"a": ("x", range(5))}, {"c": ("x", [0, 0, 1, 1, 1])})
        cond = Dataset({"a": ("c", [True, False])})
        expected = ds.copy(deep=True)
        expected["a"].values = [0, 1] + [np.nan] * 3
        actual = ds.groupby("c").where(cond)
        assert_identical(expected, actual)

        # attrs with groupby
        ds.attrs["attr"] = "ds"
        ds.a.attrs["attr"] = "da"
        actual = ds.groupby("c").where(cond)
        assert actual.attrs == ds.attrs
        assert actual.a.name == "a"
        assert actual.a.attrs == ds.a.attrs

        # attrs
        da = DataArray(range(5), name="a", attrs={"attr": "da"})
        actual = da.where(da.values > 1)
        assert actual.name == "a"
        assert actual.attrs == da.attrs

        ds = Dataset({"a": da}, attrs={"attr": "ds"})
        actual = ds.where(ds > 0)
        assert actual.attrs == ds.attrs
        assert actual.a.name == "a"
        assert actual.a.attrs == ds.a.attrs

        # lambda
        ds = Dataset({"a": ("x", range(5))})
        expected = Dataset({"a": ("x", [np.nan, np.nan, 2, 3, 4])})
        actual = ds.where(lambda x: x > 1)
        assert_identical(expected, actual)

    def test_where_other(self):
        ds = Dataset({"a": ("x", range(5))}, {"x": range(5)})
        expected = Dataset({"a": ("x", [-1, -1, 2, 3, 4])}, {"x": range(5)})
        actual = ds.where(ds > 1, -1)
        assert_equal(expected, actual)
        assert actual.a.dtype == int

        actual = ds.where(lambda x: x > 1, -1)
        assert_equal(expected, actual)

        with raises_regex(ValueError, "cannot set"):
            ds.where(ds > 1, other=0, drop=True)

        with raises_regex(ValueError, "indexes .* are not equal"):
            ds.where(ds > 1, ds.isel(x=slice(3)))

        with raises_regex(ValueError, "exact match required"):
            ds.where(ds > 1, ds.assign(b=2))

    def test_where_drop(self):
        # if drop=True

        # 1d
        # data array case
        array = DataArray(range(5), coords=[range(5)], dims=["x"])
        expected = DataArray(range(5)[2:], coords=[range(5)[2:]], dims=["x"])
        actual = array.where(array > 1, drop=True)
        assert_identical(expected, actual)

        # dataset case
        ds = Dataset({"a": array})
        expected = Dataset({"a": expected})

        actual = ds.where(ds > 1, drop=True)
        assert_identical(expected, actual)

        actual = ds.where(ds.a > 1, drop=True)
        assert_identical(expected, actual)

        with raises_regex(TypeError, "must be a"):
            ds.where(np.arange(5) > 1, drop=True)

        # 1d with odd coordinates
        array = DataArray(
            np.array([2, 7, 1, 8, 3]), coords=[np.array([3, 1, 4, 5, 9])], dims=["x"]
        )
        expected = DataArray(
            np.array([7, 8, 3]), coords=[np.array([1, 5, 9])], dims=["x"]
        )
        actual = array.where(array > 2, drop=True)
        assert_identical(expected, actual)

        # 1d multiple variables
        ds = Dataset({"a": (("x"), [0, 1, 2, 3]), "b": (("x"), [4, 5, 6, 7])})
        expected = Dataset(
            {"a": (("x"), [np.nan, 1, 2, 3]), "b": (("x"), [4, 5, 6, np.nan])}
        )
        actual = ds.where((ds > 0) & (ds < 7), drop=True)
        assert_identical(expected, actual)

        # 2d
        ds = Dataset({"a": (("x", "y"), [[0, 1], [2, 3]])})
        expected = Dataset({"a": (("x", "y"), [[np.nan, 1], [2, 3]])})
        actual = ds.where(ds > 0, drop=True)
        assert_identical(expected, actual)

        # 2d with odd coordinates
        ds = Dataset(
            {"a": (("x", "y"), [[0, 1], [2, 3]])},
            coords={
                "x": [4, 3],
                "y": [1, 2],
                "z": (["x", "y"], [[np.e, np.pi], [np.pi * np.e, np.pi * 3]]),
            },
        )
        expected = Dataset(
            {"a": (("x", "y"), [[3]])},
            coords={"x": [3], "y": [2], "z": (["x", "y"], [[np.pi * 3]])},
        )
        actual = ds.where(ds > 2, drop=True)
        assert_identical(expected, actual)

        # 2d multiple variables
        ds = Dataset(
            {"a": (("x", "y"), [[0, 1], [2, 3]]), "b": (("x", "y"), [[4, 5], [6, 7]])}
        )
        expected = Dataset(
            {
                "a": (("x", "y"), [[np.nan, 1], [2, 3]]),
                "b": (("x", "y"), [[4, 5], [6, 7]]),
            }
        )
        actual = ds.where(ds > 0, drop=True)
        assert_identical(expected, actual)

    def test_where_drop_empty(self):
        # regression test for GH1341
        array = DataArray(np.random.rand(100, 10), dims=["nCells", "nVertLevels"])
        mask = DataArray(np.zeros((100,), dtype="bool"), dims="nCells")
        actual = array.where(mask, drop=True)
        expected = DataArray(np.zeros((0, 10)), dims=["nCells", "nVertLevels"])
        assert_identical(expected, actual)

    def test_where_drop_no_indexes(self):
        ds = Dataset({"foo": ("x", [0.0, 1.0])})
        expected = Dataset({"foo": ("x", [1.0])})
        actual = ds.where(ds == 1, drop=True)
        assert_identical(expected, actual)

    def test_reduce(self):
        data = create_test_data()

        assert len(data.mean().coords) == 0

        actual = data.max()
        expected = Dataset({k: v.max() for k, v in data.data_vars.items()})
        assert_equal(expected, actual)

        assert_equal(data.min(dim=["dim1"]), data.min(dim="dim1"))

        for reduct, expected in [
            ("dim2", ["dim1", "dim3", "time"]),
            (["dim2", "time"], ["dim1", "dim3"]),
            (("dim2", "time"), ["dim1", "dim3"]),
            ((), ["dim1", "dim2", "dim3", "time"]),
        ]:
            actual = list(data.min(dim=reduct).dims)
            assert actual == expected

        assert_equal(data.mean(dim=[]), data)

    def test_reduce_coords(self):
        # regression test for GH1470
        data = xr.Dataset({"a": ("x", [1, 2, 3])}, coords={"b": 4})
        expected = xr.Dataset({"a": 2}, coords={"b": 4})
        actual = data.mean("x")
        assert_identical(actual, expected)

        # should be consistent
        actual = data["a"].mean("x").to_dataset()
        assert_identical(actual, expected)

    def test_mean_uint_dtype(self):
        data = xr.Dataset(
            {
                "a": (("x", "y"), np.arange(6).reshape(3, 2).astype("uint")),
                "b": (("x",), np.array([0.1, 0.2, np.nan])),
            }
        )
        actual = data.mean("x", skipna=True)
        expected = xr.Dataset(
            {"a": data["a"].mean("x"), "b": data["b"].mean("x", skipna=True)}
        )
        assert_identical(actual, expected)

    def test_reduce_bad_dim(self):
        data = create_test_data()
        with raises_regex(ValueError, "Dataset does not contain"):
            data.mean(dim="bad_dim")

    def test_reduce_cumsum(self):
        data = xr.Dataset(
            {"a": 1, "b": ("x", [1, 2]), "c": (("x", "y"), [[np.nan, 3], [0, 4]])}
        )
        assert_identical(data.fillna(0), data.cumsum("y"))

        expected = xr.Dataset(
            {"a": 1, "b": ("x", [1, 3]), "c": (("x", "y"), [[0, 3], [0, 7]])}
        )
        assert_identical(expected, data.cumsum())

    def test_reduce_cumsum_test_dims(self):
        data = create_test_data()
        for cumfunc in ["cumsum", "cumprod"]:
            with raises_regex(ValueError, "Dataset does not contain"):
                getattr(data, cumfunc)(dim="bad_dim")

            # ensure dimensions are correct
            for reduct, expected in [
                ("dim1", ["dim1", "dim2", "dim3", "time"]),
                ("dim2", ["dim1", "dim2", "dim3", "time"]),
                ("dim3", ["dim1", "dim2", "dim3", "time"]),
                ("time", ["dim1", "dim2", "dim3"]),
            ]:
                actual = getattr(data, cumfunc)(dim=reduct).dims
                assert list(actual) == expected

    def test_reduce_non_numeric(self):
        data1 = create_test_data(seed=44)
        data2 = create_test_data(seed=44)
        add_vars = {"var4": ["dim1", "dim2"]}
        for v, dims in sorted(add_vars.items()):
            size = tuple(data1.dims[d] for d in dims)
            data = np.random.randint(0, 100, size=size).astype(np.str_)
            data1[v] = (dims, data, {"foo": "variable"})

        assert "var4" not in data1.mean()
        assert_equal(data1.mean(), data2.mean())
        assert_equal(data1.mean(dim="dim1"), data2.mean(dim="dim1"))

    @pytest.mark.filterwarnings(
        "ignore:Once the behaviour of DataArray:DeprecationWarning"
    )
    def test_reduce_strings(self):
        expected = Dataset({"x": "a"})
        ds = Dataset({"x": ("y", ["a", "b"])})
        ds.coords["y"] = [-10, 10]
        actual = ds.min()
        assert_identical(expected, actual)

        expected = Dataset({"x": "b"})
        actual = ds.max()
        assert_identical(expected, actual)

        expected = Dataset({"x": 0})
        actual = ds.argmin()
        assert_identical(expected, actual)

        expected = Dataset({"x": 1})
        actual = ds.argmax()
        assert_identical(expected, actual)

        expected = Dataset({"x": -10})
        actual = ds.idxmin()
        assert_identical(expected, actual)

        expected = Dataset({"x": 10})
        actual = ds.idxmax()
        assert_identical(expected, actual)

        expected = Dataset({"x": b"a"})
        ds = Dataset({"x": ("y", np.array(["a", "b"], "S1"))})
        actual = ds.min()
        assert_identical(expected, actual)

        expected = Dataset({"x": "a"})
        ds = Dataset({"x": ("y", np.array(["a", "b"], "U1"))})
        actual = ds.min()
        assert_identical(expected, actual)

    def test_reduce_dtypes(self):
        # regression test for GH342
        expected = Dataset({"x": 1})
        actual = Dataset({"x": True}).sum()
        assert_identical(expected, actual)

        # regression test for GH505
        expected = Dataset({"x": 3})
        actual = Dataset({"x": ("y", np.array([1, 2], "uint16"))}).sum()
        assert_identical(expected, actual)

        expected = Dataset({"x": 1 + 1j})
        actual = Dataset({"x": ("y", [1, 1j])}).sum()
        assert_identical(expected, actual)

    def test_reduce_keep_attrs(self):
        data = create_test_data()
        _attrs = {"attr1": "value1", "attr2": 2929}

        attrs = dict(_attrs)
        data.attrs = attrs

        # Test dropped attrs
        ds = data.mean()
        assert ds.attrs == {}
        for v in ds.data_vars.values():
            assert v.attrs == {}

        # Test kept attrs
        ds = data.mean(keep_attrs=True)
        assert ds.attrs == attrs
        for k, v in ds.data_vars.items():
            assert v.attrs == data[k].attrs

    @pytest.mark.filterwarnings(
        "ignore:Once the behaviour of DataArray:DeprecationWarning"
    )
    def test_reduce_argmin(self):
        # regression test for #205
        ds = Dataset({"a": ("x", [0, 1])})
        expected = Dataset({"a": ([], 0)})
        actual = ds.argmin()
        assert_identical(expected, actual)

        actual = ds.argmin("x")
        assert_identical(expected, actual)

    def test_reduce_scalars(self):
        ds = Dataset({"x": ("a", [2, 2]), "y": 2, "z": ("b", [2])})
        expected = Dataset({"x": 0, "y": 0, "z": 0})
        actual = ds.var()
        assert_identical(expected, actual)

        expected = Dataset({"x": 0, "y": 0, "z": ("b", [0])})
        actual = ds.var("a")
        assert_identical(expected, actual)

    def test_reduce_only_one_axis(self):
        def mean_only_one_axis(x, axis):
            if not isinstance(axis, integer_types):
                raise TypeError("non-integer axis")
            return x.mean(axis)

        ds = Dataset({"a": (["x", "y"], [[0, 1, 2, 3, 4]])})
        expected = Dataset({"a": ("x", [2])})
        actual = ds.reduce(mean_only_one_axis, "y")
        assert_identical(expected, actual)

        with raises_regex(TypeError, "missing 1 required positional argument: 'axis'"):
            ds.reduce(mean_only_one_axis)

        with raises_regex(TypeError, "non-integer axis"):
            ds.reduce(mean_only_one_axis, axis=["x", "y"])

    def test_reduce_no_axis(self):
        def total_sum(x):
            return np.sum(x.flatten())

        ds = Dataset({"a": (["x", "y"], [[0, 1, 2, 3, 4]])})
        expected = Dataset({"a": ((), 10)})
        actual = ds.reduce(total_sum)
        assert_identical(expected, actual)

        with raises_regex(TypeError, "unexpected keyword argument 'axis'"):
            ds.reduce(total_sum, axis=0)

        with raises_regex(TypeError, "unexpected keyword argument 'axis'"):
            ds.reduce(total_sum, dim="x")

    def test_reduce_keepdims(self):
        ds = Dataset(
            {"a": (["x", "y"], [[0, 1, 2, 3, 4]])},
            coords={
                "y": [0, 1, 2, 3, 4],
                "x": [0],
                "lat": (["x", "y"], [[0, 1, 2, 3, 4]]),
                "c": -999.0,
            },
        )

        # Shape should match behaviour of numpy reductions with keepdims=True
        # Coordinates involved in the reduction should be removed
        actual = ds.mean(keepdims=True)
        expected = Dataset(
            {"a": (["x", "y"], np.mean(ds.a, keepdims=True).data)}, coords={"c": ds.c}
        )
        assert_identical(expected, actual)

        actual = ds.mean("x", keepdims=True)
        expected = Dataset(
            {"a": (["x", "y"], np.mean(ds.a, axis=0, keepdims=True).data)},
            coords={"y": ds.y, "c": ds.c},
        )
        assert_identical(expected, actual)

    @pytest.mark.parametrize("skipna", [True, False])
    @pytest.mark.parametrize("q", [0.25, [0.50], [0.25, 0.75]])
    def test_quantile(self, q, skipna):
        ds = create_test_data(seed=123)

        for dim in [None, "dim1", ["dim1"]]:
            ds_quantile = ds.quantile(q, dim=dim, skipna=skipna)
            if is_scalar(q):
                assert "quantile" not in ds_quantile.dims
            else:
                assert "quantile" in ds_quantile.dims

            for var, dar in ds.data_vars.items():
                assert var in ds_quantile
                assert_identical(
                    ds_quantile[var], dar.quantile(q, dim=dim, skipna=skipna)
                )
        dim = ["dim1", "dim2"]
        ds_quantile = ds.quantile(q, dim=dim, skipna=skipna)
        assert "dim3" in ds_quantile.dims
        assert all(d not in ds_quantile.dims for d in dim)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_quantile_skipna(self, skipna):
        q = 0.1
        dim = "time"
        ds = Dataset({"a": ([dim], np.arange(0, 11))})
        ds = ds.where(ds >= 1)

        result = ds.quantile(q=q, dim=dim, skipna=skipna)

        value = 1.9 if skipna else np.nan
        expected = Dataset({"a": value}, coords={"quantile": q})
        assert_identical(result, expected)

    @requires_bottleneck
    def test_rank(self):
        ds = create_test_data(seed=1234)
        # only ds.var3 depends on dim3
        z = ds.rank("dim3")
        assert ["var3"] == list(z.data_vars)
        # same as dataarray version
        x = z.var3
        y = ds.var3.rank("dim3")
        assert_equal(x, y)
        # coordinates stick
        assert list(z.coords) == list(ds.coords)
        assert list(x.coords) == list(y.coords)
        # invalid dim
        with raises_regex(ValueError, "does not contain"):
            x.rank("invalid_dim")

    def test_count(self):
        ds = Dataset({"x": ("a", [np.nan, 1]), "y": 0, "z": np.nan})
        expected = Dataset({"x": 1, "y": 1, "z": 0})
        actual = ds.count()
        assert_identical(expected, actual)

    def test_map(self):
        data = create_test_data()
        data.attrs["foo"] = "bar"

        assert_identical(data.map(np.mean), data.mean())

        expected = data.mean(keep_attrs=True)
        actual = data.map(lambda x: x.mean(keep_attrs=True), keep_attrs=True)
        assert_identical(expected, actual)

        assert_identical(data.map(lambda x: x, keep_attrs=True), data.drop_vars("time"))

        def scale(x, multiple=1):
            return multiple * x

        actual = data.map(scale, multiple=2)
        assert_equal(actual["var1"], 2 * data["var1"])
        assert_identical(actual["numbers"], data["numbers"])

        actual = data.map(np.asarray)
        expected = data.drop_vars("time")  # time is not used on a data var
        assert_equal(expected, actual)

    def test_apply_pending_deprecated_map(self):
        data = create_test_data()
        data.attrs["foo"] = "bar"

        with pytest.warns(PendingDeprecationWarning):
            assert_identical(data.apply(np.mean), data.mean())

    def make_example_math_dataset(self):
        variables = {
            "bar": ("x", np.arange(100, 400, 100)),
            "foo": (("x", "y"), 1.0 * np.arange(12).reshape(3, 4)),
        }
        coords = {"abc": ("x", ["a", "b", "c"]), "y": 10 * np.arange(4)}
        ds = Dataset(variables, coords)
        ds["foo"][0, 0] = np.nan
        return ds

    def test_dataset_number_math(self):
        ds = self.make_example_math_dataset()

        assert_identical(ds, +ds)
        assert_identical(ds, ds + 0)
        assert_identical(ds, 0 + ds)
        assert_identical(ds, ds + np.array(0))
        assert_identical(ds, np.array(0) + ds)

        actual = ds.copy(deep=True)
        actual += 0
        assert_identical(ds, actual)

    def test_unary_ops(self):
        ds = self.make_example_math_dataset()

        assert_identical(ds.map(abs), abs(ds))
        assert_identical(ds.map(lambda x: x + 4), ds + 4)

        for func in [
            lambda x: x.isnull(),
            lambda x: x.round(),
            lambda x: x.astype(int),
        ]:
            assert_identical(ds.map(func), func(ds))

        assert_identical(ds.isnull(), ~ds.notnull())

        # don't actually patch these methods in
        with pytest.raises(AttributeError):
            ds.item
        with pytest.raises(AttributeError):
            ds.searchsorted

    def test_dataset_array_math(self):
        ds = self.make_example_math_dataset()

        expected = ds.map(lambda x: x - ds["foo"])
        assert_identical(expected, ds - ds["foo"])
        assert_identical(expected, -ds["foo"] + ds)
        assert_identical(expected, ds - ds["foo"].variable)
        assert_identical(expected, -ds["foo"].variable + ds)
        actual = ds.copy(deep=True)
        actual -= ds["foo"]
        assert_identical(expected, actual)

        expected = ds.map(lambda x: x + ds["bar"])
        assert_identical(expected, ds + ds["bar"])
        actual = ds.copy(deep=True)
        actual += ds["bar"]
        assert_identical(expected, actual)

        expected = Dataset({"bar": ds["bar"] + np.arange(3)})
        assert_identical(expected, ds[["bar"]] + np.arange(3))
        assert_identical(expected, np.arange(3) + ds[["bar"]])

    def test_dataset_dataset_math(self):
        ds = self.make_example_math_dataset()

        assert_identical(ds, ds + 0 * ds)
        assert_identical(ds, ds + {"foo": 0, "bar": 0})

        expected = ds.map(lambda x: 2 * x)
        assert_identical(expected, 2 * ds)
        assert_identical(expected, ds + ds)
        assert_identical(expected, ds + ds.data_vars)
        assert_identical(expected, ds + dict(ds.data_vars))

        actual = ds.copy(deep=True)
        expected_id = id(actual)
        actual += ds
        assert_identical(expected, actual)
        assert expected_id == id(actual)

        assert_identical(ds == ds, ds.notnull())

        subsampled = ds.isel(y=slice(2))
        expected = 2 * subsampled
        assert_identical(expected, subsampled + ds)
        assert_identical(expected, ds + subsampled)

    def test_dataset_math_auto_align(self):
        ds = self.make_example_math_dataset()
        subset = ds.isel(y=[1, 3])
        expected = 2 * subset
        actual = ds + subset
        assert_identical(expected, actual)

        actual = ds.isel(y=slice(1)) + ds.isel(y=slice(1, None))
        expected = 2 * ds.drop_sel(y=ds.y)
        assert_equal(actual, expected)

        actual = ds + ds[["bar"]]
        expected = (2 * ds[["bar"]]).merge(ds.coords)
        assert_identical(expected, actual)

        assert_identical(ds + Dataset(), ds.coords.to_dataset())
        assert_identical(Dataset() + Dataset(), Dataset())

        ds2 = Dataset(coords={"bar": 42})
        assert_identical(ds + ds2, ds.coords.merge(ds2))

        # maybe unary arithmetic with empty datasets should raise instead?
        assert_identical(Dataset() + 1, Dataset())

        actual = ds.copy(deep=True)
        other = ds.isel(y=slice(2))
        actual += other
        expected = ds + other.reindex_like(ds)
        assert_identical(expected, actual)

    def test_dataset_math_errors(self):
        ds = self.make_example_math_dataset()

        with pytest.raises(TypeError):
            ds["foo"] += ds
        with pytest.raises(TypeError):
            ds["foo"].variable += ds
        with raises_regex(ValueError, "must have the same"):
            ds += ds[["bar"]]

        # verify we can rollback in-place operations if something goes wrong
        # nb. inplace datetime64 math actually will work with an integer array
        # but not floats thanks to numpy's inconsistent handling
        other = DataArray(np.datetime64("2000-01-01"), coords={"c": 2})
        actual = ds.copy(deep=True)
        with pytest.raises(TypeError):
            actual += other
        assert_identical(actual, ds)

    def test_dataset_transpose(self):
        ds = Dataset(
            {
                "a": (("x", "y"), np.random.randn(3, 4)),
                "b": (("y", "x"), np.random.randn(4, 3)),
            },
            coords={
                "x": range(3),
                "y": range(4),
                "xy": (("x", "y"), np.random.randn(3, 4)),
            },
        )

        actual = ds.transpose()
        expected = Dataset(
            {"a": (("y", "x"), ds.a.values.T), "b": (("x", "y"), ds.b.values.T)},
            coords={
                "x": ds.x.values,
                "y": ds.y.values,
                "xy": (("y", "x"), ds.xy.values.T),
            },
        )
        assert_identical(expected, actual)

        actual = ds.transpose(...)
        expected = ds
        assert_identical(expected, actual)

        actual = ds.transpose("x", "y")
        expected = ds.map(lambda x: x.transpose("x", "y", transpose_coords=True))
        assert_identical(expected, actual)

        ds = create_test_data()
        actual = ds.transpose()
        for k in ds.variables:
            assert actual[k].dims[::-1] == ds[k].dims

        new_order = ("dim2", "dim3", "dim1", "time")
        actual = ds.transpose(*new_order)
        for k in ds.variables:
            expected_dims = tuple(d for d in new_order if d in ds[k].dims)
            assert actual[k].dims == expected_dims

        # same as above but with ellipsis
        new_order = ("dim2", "dim3", "dim1", "time")
        actual = ds.transpose("dim2", "dim3", ...)
        for k in ds.variables:
            expected_dims = tuple(d for d in new_order if d in ds[k].dims)
            assert actual[k].dims == expected_dims

        with raises_regex(ValueError, "permuted"):
            ds.transpose("dim1", "dim2", "dim3")
        with raises_regex(ValueError, "permuted"):
            ds.transpose("dim1", "dim2", "dim3", "time", "extra_dim")

        assert "T" not in dir(ds)

    def test_dataset_ellipsis_transpose_different_ordered_vars(self):
        # https://github.com/pydata/xarray/issues/1081#issuecomment-544350457
        ds = Dataset(
            dict(
                a=(("w", "x", "y", "z"), np.ones((2, 3, 4, 5))),
                b=(("x", "w", "y", "z"), np.zeros((3, 2, 4, 5))),
            )
        )
        result = ds.transpose(..., "z", "y")
        assert list(result["a"].dims) == list("wxzy")
        assert list(result["b"].dims) == list("xwzy")

    def test_dataset_retains_period_index_on_transpose(self):

        ds = create_test_data()
        ds["time"] = pd.period_range("2000-01-01", periods=20)

        transposed = ds.transpose()

        assert isinstance(transposed.time.to_index(), pd.PeriodIndex)

    def test_dataset_diff_n1_simple(self):
        ds = Dataset({"foo": ("x", [5, 5, 6, 6])})
        actual = ds.diff("x")
        expected = Dataset({"foo": ("x", [0, 1, 0])})
        assert_equal(expected, actual)

    def test_dataset_diff_n1_label(self):
        ds = Dataset({"foo": ("x", [5, 5, 6, 6])}, {"x": [0, 1, 2, 3]})
        actual = ds.diff("x", label="lower")
        expected = Dataset({"foo": ("x", [0, 1, 0])}, {"x": [0, 1, 2]})
        assert_equal(expected, actual)

        actual = ds.diff("x", label="upper")
        expected = Dataset({"foo": ("x", [0, 1, 0])}, {"x": [1, 2, 3]})
        assert_equal(expected, actual)

    def test_dataset_diff_n1(self):
        ds = create_test_data(seed=1)
        actual = ds.diff("dim2")
        expected = {}
        expected["var1"] = DataArray(
            np.diff(ds["var1"].values, axis=1),
            {"dim2": ds["dim2"].values[1:]},
            ["dim1", "dim2"],
        )
        expected["var2"] = DataArray(
            np.diff(ds["var2"].values, axis=1),
            {"dim2": ds["dim2"].values[1:]},
            ["dim1", "dim2"],
        )
        expected["var3"] = ds["var3"]
        expected = Dataset(expected, coords={"time": ds["time"].values})
        expected.coords["numbers"] = ("dim3", ds["numbers"].values)
        assert_equal(expected, actual)

    def test_dataset_diff_n2(self):
        ds = create_test_data(seed=1)
        actual = ds.diff("dim2", n=2)
        expected = {}
        expected["var1"] = DataArray(
            np.diff(ds["var1"].values, axis=1, n=2),
            {"dim2": ds["dim2"].values[2:]},
            ["dim1", "dim2"],
        )
        expected["var2"] = DataArray(
            np.diff(ds["var2"].values, axis=1, n=2),
            {"dim2": ds["dim2"].values[2:]},
            ["dim1", "dim2"],
        )
        expected["var3"] = ds["var3"]
        expected = Dataset(expected, coords={"time": ds["time"].values})
        expected.coords["numbers"] = ("dim3", ds["numbers"].values)
        assert_equal(expected, actual)

    def test_dataset_diff_exception_n_neg(self):
        ds = create_test_data(seed=1)
        with raises_regex(ValueError, "must be non-negative"):
            ds.diff("dim2", n=-1)

    def test_dataset_diff_exception_label_str(self):
        ds = create_test_data(seed=1)
        with raises_regex(ValueError, "'label' argument has to"):
            ds.diff("dim2", label="raise_me")

    @pytest.mark.parametrize("fill_value", [dtypes.NA, 2, 2.0, {"foo": -10}])
    def test_shift(self, fill_value):
        coords = {"bar": ("x", list("abc")), "x": [-4, 3, 2]}
        attrs = {"meta": "data"}
        ds = Dataset({"foo": ("x", [1, 2, 3])}, coords, attrs)
        actual = ds.shift(x=1, fill_value=fill_value)
        if fill_value == dtypes.NA:
            # if we supply the default, we expect the missing value for a
            # float array
            fill_value = np.nan
        elif isinstance(fill_value, dict):
            fill_value = fill_value.get("foo", np.nan)
        expected = Dataset({"foo": ("x", [fill_value, 1, 2])}, coords, attrs)
        assert_identical(expected, actual)

        with raises_regex(ValueError, "dimensions"):
            ds.shift(foo=123)

    def test_roll_coords(self):
        coords = {"bar": ("x", list("abc")), "x": [-4, 3, 2]}
        attrs = {"meta": "data"}
        ds = Dataset({"foo": ("x", [1, 2, 3])}, coords, attrs)
        actual = ds.roll(x=1, roll_coords=True)

        ex_coords = {"bar": ("x", list("cab")), "x": [2, -4, 3]}
        expected = Dataset({"foo": ("x", [3, 1, 2])}, ex_coords, attrs)
        assert_identical(expected, actual)

        with raises_regex(ValueError, "dimensions"):
            ds.roll(foo=123, roll_coords=True)

    def test_roll_no_coords(self):
        coords = {"bar": ("x", list("abc")), "x": [-4, 3, 2]}
        attrs = {"meta": "data"}
        ds = Dataset({"foo": ("x", [1, 2, 3])}, coords, attrs)
        actual = ds.roll(x=1, roll_coords=False)

        expected = Dataset({"foo": ("x", [3, 1, 2])}, coords, attrs)
        assert_identical(expected, actual)

        with raises_regex(ValueError, "dimensions"):
            ds.roll(abc=321, roll_coords=False)

    def test_roll_coords_none(self):
        coords = {"bar": ("x", list("abc")), "x": [-4, 3, 2]}
        attrs = {"meta": "data"}
        ds = Dataset({"foo": ("x", [1, 2, 3])}, coords, attrs)

        with pytest.warns(FutureWarning):
            actual = ds.roll(x=1, roll_coords=None)

        ex_coords = {"bar": ("x", list("cab")), "x": [2, -4, 3]}
        expected = Dataset({"foo": ("x", [3, 1, 2])}, ex_coords, attrs)
        assert_identical(expected, actual)

    def test_roll_multidim(self):
        # regression test for 2445
        arr = xr.DataArray(
            [[1, 2, 3], [4, 5, 6]],
            coords={"x": range(3), "y": range(2)},
            dims=("y", "x"),
        )
        actual = arr.roll(x=1, roll_coords=True)
        expected = xr.DataArray(
            [[3, 1, 2], [6, 4, 5]], coords=[("y", [0, 1]), ("x", [2, 0, 1])]
        )
        assert_identical(expected, actual)

    def test_real_and_imag(self):
        attrs = {"foo": "bar"}
        ds = Dataset({"x": ((), 1 + 2j, attrs)}, attrs=attrs)

        expected_re = Dataset({"x": ((), 1, attrs)}, attrs=attrs)
        assert_identical(ds.real, expected_re)

        expected_im = Dataset({"x": ((), 2, attrs)}, attrs=attrs)
        assert_identical(ds.imag, expected_im)

    def test_setattr_raises(self):
        ds = Dataset({}, coords={"scalar": 1}, attrs={"foo": "bar"})
        with raises_regex(AttributeError, "cannot set attr"):
            ds.scalar = 2
        with raises_regex(AttributeError, "cannot set attr"):
            ds.foo = 2
        with raises_regex(AttributeError, "cannot set attr"):
            ds.other = 2

    def test_filter_by_attrs(self):
        precip = dict(standard_name="convective_precipitation_flux")
        temp0 = dict(standard_name="air_potential_temperature", height="0 m")
        temp10 = dict(standard_name="air_potential_temperature", height="10 m")
        ds = Dataset(
            {
                "temperature_0": (["t"], [0], temp0),
                "temperature_10": (["t"], [0], temp10),
                "precipitation": (["t"], [0], precip),
            },
            coords={"time": (["t"], [0], dict(axis="T", long_name="time_in_seconds"))},
        )

        # Test return empty Dataset.
        ds.filter_by_attrs(standard_name="invalid_standard_name")
        new_ds = ds.filter_by_attrs(standard_name="invalid_standard_name")
        assert not bool(new_ds.data_vars)

        # Test return one DataArray.
        new_ds = ds.filter_by_attrs(standard_name="convective_precipitation_flux")
        assert new_ds["precipitation"].standard_name == "convective_precipitation_flux"

        assert_equal(new_ds["precipitation"], ds["precipitation"])

        # Test filter coordinates
        new_ds = ds.filter_by_attrs(long_name="time_in_seconds")
        assert new_ds["time"].long_name == "time_in_seconds"
        assert not bool(new_ds.data_vars)

        # Test return more than one DataArray.
        new_ds = ds.filter_by_attrs(standard_name="air_potential_temperature")
        assert len(new_ds.data_vars) == 2
        for var in new_ds.data_vars:
            assert new_ds[var].standard_name == "air_potential_temperature"

        # Test callable.
        new_ds = ds.filter_by_attrs(height=lambda v: v is not None)
        assert len(new_ds.data_vars) == 2
        for var in new_ds.data_vars:
            assert new_ds[var].standard_name == "air_potential_temperature"

        new_ds = ds.filter_by_attrs(height="10 m")
        assert len(new_ds.data_vars) == 1
        for var in new_ds.data_vars:
            assert new_ds[var].height == "10 m"

        # Test return empty Dataset due to conflicting filters
        new_ds = ds.filter_by_attrs(
            standard_name="convective_precipitation_flux", height="0 m"
        )
        assert not bool(new_ds.data_vars)

        # Test return one DataArray with two filter conditions
        new_ds = ds.filter_by_attrs(
            standard_name="air_potential_temperature", height="0 m"
        )
        for var in new_ds.data_vars:
            assert new_ds[var].standard_name == "air_potential_temperature"
            assert new_ds[var].height == "0 m"
            assert new_ds[var].height != "10 m"

        # Test return empty Dataset due to conflicting callables
        new_ds = ds.filter_by_attrs(
            standard_name=lambda v: False, height=lambda v: True
        )
        assert not bool(new_ds.data_vars)

    def test_binary_op_propagate_indexes(self):
        ds = Dataset(
            {"d1": DataArray([1, 2, 3], dims=["x"], coords={"x": [10, 20, 30]})}
        )
        expected = ds.indexes["x"]
        actual = (ds * 2).indexes["x"]
        assert expected is actual

    def test_binary_op_join_setting(self):
        # arithmetic_join applies to data array coordinates
        missing_2 = xr.Dataset({"x": [0, 1]})
        missing_0 = xr.Dataset({"x": [1, 2]})
        with xr.set_options(arithmetic_join="outer"):
            actual = missing_2 + missing_0
        expected = xr.Dataset({"x": [0, 1, 2]})
        assert_equal(actual, expected)

        # arithmetic join also applies to data_vars
        ds1 = xr.Dataset({"foo": 1, "bar": 2})
        ds2 = xr.Dataset({"bar": 2, "baz": 3})
        expected = xr.Dataset({"bar": 4})  # default is inner joining
        actual = ds1 + ds2
        assert_equal(actual, expected)

        with xr.set_options(arithmetic_join="outer"):
            expected = xr.Dataset({"foo": np.nan, "bar": 4, "baz": np.nan})
            actual = ds1 + ds2
            assert_equal(actual, expected)

        with xr.set_options(arithmetic_join="left"):
            expected = xr.Dataset({"foo": np.nan, "bar": 4})
            actual = ds1 + ds2
            assert_equal(actual, expected)

        with xr.set_options(arithmetic_join="right"):
            expected = xr.Dataset({"bar": 4, "baz": np.nan})
            actual = ds1 + ds2
            assert_equal(actual, expected)

    def test_full_like(self):
        # For more thorough tests, see test_variable.py
        # Note: testing data_vars with mismatched dtypes
        ds = Dataset(
            {
                "d1": DataArray([1, 2, 3], dims=["x"], coords={"x": [10, 20, 30]}),
                "d2": DataArray([1.1, 2.2, 3.3], dims=["y"]),
            },
            attrs={"foo": "bar"},
        )
        actual = full_like(ds, 2)

        expected = ds.copy(deep=True)
        expected["d1"].values = [2, 2, 2]
        expected["d2"].values = [2.0, 2.0, 2.0]
        assert expected["d1"].dtype == int
        assert expected["d2"].dtype == float
        assert_identical(expected, actual)

        # override dtype
        actual = full_like(ds, fill_value=True, dtype=bool)
        expected = ds.copy(deep=True)
        expected["d1"].values = [True, True, True]
        expected["d2"].values = [True, True, True]
        assert expected["d1"].dtype == bool
        assert expected["d2"].dtype == bool
        assert_identical(expected, actual)

        # with multiple fill values
        actual = full_like(ds, {"d1": 1, "d2": 2.3})
        expected = ds.assign(d1=("x", [1, 1, 1]), d2=("y", [2.3, 2.3, 2.3]))
        assert expected["d1"].dtype == int
        assert expected["d2"].dtype == float
        assert_identical(expected, actual)

        # override multiple dtypes
        actual = full_like(ds, fill_value={"d1": 1, "d2": 2.3}, dtype={"d1": bool})
        expected = ds.assign(d1=("x", [True, True, True]), d2=("y", [2.3, 2.3, 2.3]))
        assert expected["d1"].dtype == bool
        assert expected["d2"].dtype == float
        assert_identical(expected, actual)

    def test_combine_first(self):
        dsx0 = DataArray([0, 0], [("x", ["a", "b"])]).to_dataset(name="dsx0")
        dsx1 = DataArray([1, 1], [("x", ["b", "c"])]).to_dataset(name="dsx1")

        actual = dsx0.combine_first(dsx1)
        expected = Dataset(
            {"dsx0": ("x", [0, 0, np.nan]), "dsx1": ("x", [np.nan, 1, 1])},
            coords={"x": ["a", "b", "c"]},
        )
        assert_equal(actual, expected)
        assert_equal(actual, xr.merge([dsx0, dsx1]))

        # works just like xr.merge([self, other])
        dsy2 = DataArray([2, 2, 2], [("x", ["b", "c", "d"])]).to_dataset(name="dsy2")
        actual = dsx0.combine_first(dsy2)
        expected = xr.merge([dsy2, dsx0])
        assert_equal(actual, expected)

    def test_sortby(self):
        ds = Dataset(
            {
                "A": DataArray(
                    [[1, 2], [3, 4], [5, 6]], [("x", ["c", "b", "a"]), ("y", [1, 0])]
                ),
                "B": DataArray([[5, 6], [7, 8], [9, 10]], dims=["x", "y"]),
            }
        )

        sorted1d = Dataset(
            {
                "A": DataArray(
                    [[5, 6], [3, 4], [1, 2]], [("x", ["a", "b", "c"]), ("y", [1, 0])]
                ),
                "B": DataArray([[9, 10], [7, 8], [5, 6]], dims=["x", "y"]),
            }
        )

        sorted2d = Dataset(
            {
                "A": DataArray(
                    [[6, 5], [4, 3], [2, 1]], [("x", ["a", "b", "c"]), ("y", [0, 1])]
                ),
                "B": DataArray([[10, 9], [8, 7], [6, 5]], dims=["x", "y"]),
            }
        )

        expected = sorted1d
        dax = DataArray([100, 99, 98], [("x", ["c", "b", "a"])])
        actual = ds.sortby(dax)
        assert_equal(actual, expected)

        # test descending order sort
        actual = ds.sortby(dax, ascending=False)
        assert_equal(actual, ds)

        # test alignment (fills in nan for 'c')
        dax_short = DataArray([98, 97], [("x", ["b", "a"])])
        actual = ds.sortby(dax_short)
        assert_equal(actual, expected)

        # test 1-D lexsort
        # dax0 is sorted first to give indices of [1, 2, 0]
        # and then dax1 would be used to move index 2 ahead of 1
        dax0 = DataArray([100, 95, 95], [("x", ["c", "b", "a"])])
        dax1 = DataArray([0, 1, 0], [("x", ["c", "b", "a"])])
        actual = ds.sortby([dax0, dax1])  # lexsort underneath gives [2, 1, 0]
        assert_equal(actual, expected)

        expected = sorted2d
        # test multi-dim sort by 1D dataarray values
        day = DataArray([90, 80], [("y", [1, 0])])
        actual = ds.sortby([day, dax])
        assert_equal(actual, expected)

        # test exception-raising
        with pytest.raises(KeyError) as excinfo:
            actual = ds.sortby("z")

        with pytest.raises(ValueError) as excinfo:
            actual = ds.sortby(ds["A"])
        assert "DataArray is not 1-D" in str(excinfo.value)

        expected = sorted1d
        actual = ds.sortby("x")
        assert_equal(actual, expected)

        # test pandas.MultiIndex
        indices = (("b", 1), ("b", 0), ("a", 1), ("a", 0))
        midx = pd.MultiIndex.from_tuples(indices, names=["one", "two"])
        ds_midx = Dataset(
            {
                "A": DataArray(
                    [[1, 2], [3, 4], [5, 6], [7, 8]], [("x", midx), ("y", [1, 0])]
                ),
                "B": DataArray([[5, 6], [7, 8], [9, 10], [11, 12]], dims=["x", "y"]),
            }
        )
        actual = ds_midx.sortby("x")
        midx_reversed = pd.MultiIndex.from_tuples(
            tuple(reversed(indices)), names=["one", "two"]
        )
        expected = Dataset(
            {
                "A": DataArray(
                    [[7, 8], [5, 6], [3, 4], [1, 2]],
                    [("x", midx_reversed), ("y", [1, 0])],
                ),
                "B": DataArray([[11, 12], [9, 10], [7, 8], [5, 6]], dims=["x", "y"]),
            }
        )
        assert_equal(actual, expected)

        # multi-dim sort by coordinate objects
        expected = sorted2d
        actual = ds.sortby(["x", "y"])
        assert_equal(actual, expected)

        # test descending order sort
        actual = ds.sortby(["x", "y"], ascending=False)
        assert_equal(actual, ds)

    def test_attribute_access(self):
        ds = create_test_data(seed=1)
        for key in ["var1", "var2", "var3", "time", "dim1", "dim2", "dim3", "numbers"]:
            assert_equal(ds[key], getattr(ds, key))
            assert key in dir(ds)

        for key in ["dim3", "dim1", "numbers"]:
            assert_equal(ds["var3"][key], getattr(ds.var3, key))
            assert key in dir(ds["var3"])
        # attrs
        assert ds["var3"].attrs["foo"] == ds.var3.foo
        assert "foo" in dir(ds["var3"])

    def test_ipython_key_completion(self):
        ds = create_test_data(seed=1)
        actual = ds._ipython_key_completions_()
        expected = ["var1", "var2", "var3", "time", "dim1", "dim2", "dim3", "numbers"]
        for item in actual:
            ds[item]  # should not raise
        assert sorted(actual) == sorted(expected)

        # for dataarray
        actual = ds["var3"]._ipython_key_completions_()
        expected = ["dim3", "dim1", "numbers"]
        for item in actual:
            ds["var3"][item]  # should not raise
        assert sorted(actual) == sorted(expected)

        # MultiIndex
        ds_midx = ds.stack(dim12=["dim1", "dim2"])
        actual = ds_midx._ipython_key_completions_()
        expected = [
            "var1",
            "var2",
            "var3",
            "time",
            "dim1",
            "dim2",
            "dim3",
            "numbers",
            "dim12",
        ]
        for item in actual:
            ds_midx[item]  # should not raise
        assert sorted(actual) == sorted(expected)

        # coords
        actual = ds.coords._ipython_key_completions_()
        expected = ["time", "dim1", "dim2", "dim3", "numbers"]
        for item in actual:
            ds.coords[item]  # should not raise
        assert sorted(actual) == sorted(expected)

        actual = ds["var3"].coords._ipython_key_completions_()
        expected = ["dim1", "dim3", "numbers"]
        for item in actual:
            ds["var3"].coords[item]  # should not raise
        assert sorted(actual) == sorted(expected)

        # data_vars
        actual = ds.data_vars._ipython_key_completions_()
        expected = ["var1", "var2", "var3", "dim1"]
        for item in actual:
            ds.data_vars[item]  # should not raise
        assert sorted(actual) == sorted(expected)

    def test_polyfit_output(self):
        ds = create_test_data(seed=1)

        out = ds.polyfit("dim2", 2, full=False)
        assert "var1_polyfit_coefficients" in out

        out = ds.polyfit("dim1", 2, full=True)
        assert "var1_polyfit_coefficients" in out
        assert "dim1_matrix_rank" in out

        out = ds.polyfit("time", 2)
        assert len(out.data_vars) == 0

    def test_polyfit_warnings(self):
        ds = create_test_data(seed=1)

        with warnings.catch_warnings(record=True) as ws:
            ds.var1.polyfit("dim2", 10, full=False)
            assert len(ws) == 1
            assert ws[0].category == np.RankWarning
            ds.var1.polyfit("dim2", 10, full=True)
            assert len(ws) == 1

    def test_pad(self):
        ds = create_test_data(seed=1)
        padded = ds.pad(dim2=(1, 1), constant_values=42)

        assert padded["dim2"].shape == (11,)
        assert padded["var1"].shape == (8, 11)
        assert padded["var2"].shape == (8, 11)
        assert padded["var3"].shape == (10, 8)
        assert dict(padded.dims) == {"dim1": 8, "dim2": 11, "dim3": 10, "time": 20}

        np.testing.assert_equal(padded["var1"].isel(dim2=[0, -1]).data, 42)
        np.testing.assert_equal(padded["dim2"][[0, -1]].data, np.nan)

    def test_astype_attrs(self):
        data = create_test_data(seed=123)
        data.attrs["foo"] = "bar"

        assert data.attrs == data.astype(float).attrs
        assert data.var1.attrs == data.astype(float).var1.attrs
        assert not data.astype(float, keep_attrs=False).attrs
        assert not data.astype(float, keep_attrs=False).var1.attrs


# Py.test tests


@pytest.fixture(params=[None])
def data_set(request):
    return create_test_data(request.param)


@pytest.mark.parametrize("test_elements", ([1, 2], np.array([1, 2]), DataArray([1, 2])))
def test_isin(test_elements):
    expected = Dataset(
        data_vars={
            "var1": (("dim1",), [0, 1]),
            "var2": (("dim1",), [1, 1]),
            "var3": (("dim1",), [0, 1]),
        }
    ).astype("bool")

    result = Dataset(
        data_vars={
            "var1": (("dim1",), [0, 1]),
            "var2": (("dim1",), [1, 2]),
            "var3": (("dim1",), [0, 1]),
        }
    ).isin(test_elements)

    assert_equal(result, expected)


@pytest.mark.skipif(not has_dask, reason="requires dask")
@pytest.mark.parametrize("test_elements", ([1, 2], np.array([1, 2]), DataArray([1, 2])))
def test_isin_dask(test_elements):
    expected = Dataset(
        data_vars={
            "var1": (("dim1",), [0, 1]),
            "var2": (("dim1",), [1, 1]),
            "var3": (("dim1",), [0, 1]),
        }
    ).astype("bool")

    result = (
        Dataset(
            data_vars={
                "var1": (("dim1",), [0, 1]),
                "var2": (("dim1",), [1, 2]),
                "var3": (("dim1",), [0, 1]),
            }
        )
        .chunk(1)
        .isin(test_elements)
        .compute()
    )

    assert_equal(result, expected)


def test_isin_dataset():
    ds = Dataset({"x": [1, 2]})
    with pytest.raises(TypeError):
        ds.isin(ds)


@pytest.mark.parametrize(
    "unaligned_coords",
    (
        {"x": [2, 1, 0]},
        {"x": (["x"], np.asarray([2, 1, 0]))},
        {"x": (["x"], np.asarray([1, 2, 0]))},
        {"x": pd.Index([2, 1, 0])},
        {"x": Variable(dims="x", data=[0, 2, 1])},
        {"x": IndexVariable(dims="x", data=[0, 1, 2])},
        {"y": 42},
        {"y": ("x", [2, 1, 0])},
        {"y": ("x", np.asarray([2, 1, 0]))},
        {"y": (["x"], np.asarray([2, 1, 0]))},
    ),
)
@pytest.mark.parametrize("coords", ({"x": ("x", [0, 1, 2])}, {"x": [0, 1, 2]}))
def test_dataset_constructor_aligns_to_explicit_coords(unaligned_coords, coords):

    a = xr.DataArray([1, 2, 3], dims=["x"], coords=unaligned_coords)

    expected = xr.Dataset(coords=coords)
    expected["a"] = a

    result = xr.Dataset({"a": a}, coords=coords)

    assert_equal(expected, result)


def test_error_message_on_set_supplied():
    with pytest.raises(TypeError, match="has invalid type <class 'set'>"):
        xr.Dataset(dict(date=[1, 2, 3], sec={4}))


@pytest.mark.parametrize("unaligned_coords", ({"y": ("b", np.asarray([2, 1, 0]))},))
def test_constructor_raises_with_invalid_coords(unaligned_coords):

    with pytest.raises(ValueError, match="not a subset of the DataArray dimensions"):
        xr.DataArray([1, 2, 3], dims=["x"], coords=unaligned_coords)


def test_dir_expected_attrs(data_set):

    some_expected_attrs = {"pipe", "mean", "isnull", "var1", "dim2", "numbers"}
    result = dir(data_set)
    assert set(result) >= some_expected_attrs


def test_dir_non_string(data_set):
    # add a numbered key to ensure this doesn't break dir
    data_set[5] = "foo"
    result = dir(data_set)
    assert 5 not in result

    # GH2172
    sample_data = np.random.uniform(size=[2, 2000, 10000])
    x = xr.Dataset({"sample_data": (sample_data.shape, sample_data)})
    x2 = x["sample_data"]
    dir(x2)


def test_dir_unicode(data_set):
    data_set["unicode"] = "uni"
    result = dir(data_set)
    assert "unicode" in result


@pytest.fixture(params=[1])
def ds(request):
    if request.param == 1:
        return Dataset(
            {
                "z1": (["y", "x"], np.random.randn(2, 8)),
                "z2": (["time", "y"], np.random.randn(10, 2)),
            },
            {
                "x": ("x", np.linspace(0, 1.0, 8)),
                "time": ("time", np.linspace(0, 1.0, 10)),
                "c": ("y", ["a", "b"]),
                "y": range(2),
            },
        )

    if request.param == 2:
        return Dataset(
            {
                "z1": (["time", "y"], np.random.randn(10, 2)),
                "z2": (["time"], np.random.randn(10)),
                "z3": (["x", "time"], np.random.randn(8, 10)),
            },
            {
                "x": ("x", np.linspace(0, 1.0, 8)),
                "time": ("time", np.linspace(0, 1.0, 10)),
                "c": ("y", ["a", "b"]),
                "y": range(2),
            },
        )


def test_coarsen_absent_dims_error(ds):
    with raises_regex(ValueError, "not found in Dataset."):
        ds.coarsen(foo=2)


@pytest.mark.parametrize("dask", [True, False])
@pytest.mark.parametrize(("boundary", "side"), [("trim", "left"), ("pad", "right")])
def test_coarsen(ds, dask, boundary, side):
    if dask and has_dask:
        ds = ds.chunk({"x": 4})

    actual = ds.coarsen(time=2, x=3, boundary=boundary, side=side).max()
    assert_equal(
        actual["z1"], ds["z1"].coarsen(x=3, boundary=boundary, side=side).max()
    )
    # coordinate should be mean by default
    assert_equal(
        actual["time"], ds["time"].coarsen(time=2, boundary=boundary, side=side).mean()
    )


@pytest.mark.parametrize("dask", [True, False])
def test_coarsen_coords(ds, dask):
    if dask and has_dask:
        ds = ds.chunk({"x": 4})

    # check if coord_func works
    actual = ds.coarsen(time=2, x=3, boundary="trim", coord_func={"time": "max"}).max()
    assert_equal(actual["z1"], ds["z1"].coarsen(x=3, boundary="trim").max())
    assert_equal(actual["time"], ds["time"].coarsen(time=2, boundary="trim").max())

    # raise if exact
    with pytest.raises(ValueError):
        ds.coarsen(x=3).mean()
    # should be no error
    ds.isel(x=slice(0, 3 * (len(ds["x"]) // 3))).coarsen(x=3).mean()

    # working test with pd.time
    da = xr.DataArray(
        np.linspace(0, 365, num=364),
        dims="time",
        coords={"time": pd.date_range("15/12/1999", periods=364)},
    )
    actual = da.coarsen(time=2).mean()


@requires_cftime
def test_coarsen_coords_cftime():
    times = xr.cftime_range("2000", periods=6)
    da = xr.DataArray(range(6), [("time", times)])
    actual = da.coarsen(time=3).mean()
    expected_times = xr.cftime_range("2000-01-02", freq="3D", periods=2)
    np.testing.assert_array_equal(actual.time, expected_times)


def test_coarsen_keep_attrs():
    _attrs = {"units": "test", "long_name": "testing"}

    var1 = np.linspace(10, 15, 100)
    var2 = np.linspace(5, 10, 100)
    coords = np.linspace(1, 10, 100)

    ds = Dataset(
        data_vars={"var1": ("coord", var1), "var2": ("coord", var2)},
        coords={"coord": coords},
        attrs=_attrs,
    )

    ds2 = ds.copy(deep=True)

    # Test dropped attrs
    dat = ds.coarsen(coord=5).mean()
    assert dat.attrs == {}

    # Test kept attrs using dataset keyword
    dat = ds.coarsen(coord=5, keep_attrs=True).mean()
    assert dat.attrs == _attrs

    # Test kept attrs using global option
    with set_options(keep_attrs=True):
        dat = ds.coarsen(coord=5).mean()
    assert dat.attrs == _attrs

    # Test kept attrs in original object
    xr.testing.assert_identical(ds, ds2)


@pytest.mark.slow
@pytest.mark.parametrize("ds", (1, 2), indirect=True)
@pytest.mark.parametrize("window", (1, 2, 3, 4))
@pytest.mark.parametrize("name", ("sum", "mean", "std", "var", "min", "max", "median"))
def test_coarsen_reduce(ds, window, name):
    # Use boundary="trim" to accomodate all window sizes used in tests
    coarsen_obj = ds.coarsen(time=window, boundary="trim")

    # add nan prefix to numpy methods to get similar behavior as bottleneck
    actual = coarsen_obj.reduce(getattr(np, f"nan{name}"))
    expected = getattr(coarsen_obj, name)()
    assert_allclose(actual, expected)

    # make sure the order of data_var are not changed.
    assert list(ds.data_vars.keys()) == list(actual.data_vars.keys())

    # Make sure the dimension order is restored
    for key, src_var in ds.data_vars.items():
        assert src_var.dims == actual[key].dims


@pytest.mark.parametrize(
    "funcname, argument",
    [
        ("reduce", (np.mean,)),
        ("mean", ()),
        ("construct", ("window_dim",)),
        ("count", ()),
    ],
)
def test_rolling_keep_attrs(funcname, argument):
    global_attrs = {"units": "test", "long_name": "testing"}
    da_attrs = {"da_attr": "test"}
    da_not_rolled_attrs = {"da_not_rolled_attr": "test"}

    data = np.linspace(10, 15, 100)
    coords = np.linspace(1, 10, 100)

    ds = Dataset(
        data_vars={"da": ("coord", data), "da_not_rolled": ("no_coord", data)},
        coords={"coord": coords},
        attrs=global_attrs,
    )
    ds.da.attrs = da_attrs
    ds.da_not_rolled.attrs = da_not_rolled_attrs

    # attrs are now kept per default
    func = getattr(ds.rolling(dim={"coord": 5}), funcname)
    result = func(*argument)
    assert result.attrs == global_attrs
    assert result.da.attrs == da_attrs
    assert result.da_not_rolled.attrs == da_not_rolled_attrs
    assert result.da.name == "da"
    assert result.da_not_rolled.name == "da_not_rolled"

    # discard attrs
    func = getattr(ds.rolling(dim={"coord": 5}), funcname)
    result = func(*argument, keep_attrs=False)
    assert result.attrs == {}
    assert result.da.attrs == {}
    assert result.da_not_rolled.attrs == {}
    assert result.da.name == "da"
    assert result.da_not_rolled.name == "da_not_rolled"

    # test discard attrs using global option
    func = getattr(ds.rolling(dim={"coord": 5}), funcname)
    with set_options(keep_attrs=False):
        result = func(*argument)

    assert result.attrs == {}
    assert result.da.attrs == {}
    assert result.da_not_rolled.attrs == {}
    assert result.da.name == "da"
    assert result.da_not_rolled.name == "da_not_rolled"

    # keyword takes precedence over global option
    func = getattr(ds.rolling(dim={"coord": 5}), funcname)
    with set_options(keep_attrs=False):
        result = func(*argument, keep_attrs=True)

    assert result.attrs == global_attrs
    assert result.da.attrs == da_attrs
    assert result.da_not_rolled.attrs == da_not_rolled_attrs
    assert result.da.name == "da"
    assert result.da_not_rolled.name == "da_not_rolled"

    func = getattr(ds.rolling(dim={"coord": 5}), funcname)
    with set_options(keep_attrs=True):
        result = func(*argument, keep_attrs=False)

    assert result.attrs == {}
    assert result.da.attrs == {}
    assert result.da_not_rolled.attrs == {}
    assert result.da.name == "da"
    assert result.da_not_rolled.name == "da_not_rolled"


def test_rolling_keep_attrs_deprecated():
    global_attrs = {"units": "test", "long_name": "testing"}
    attrs_da = {"da_attr": "test"}

    data = np.linspace(10, 15, 100)
    coords = np.linspace(1, 10, 100)

    ds = Dataset(
        data_vars={"da": ("coord", data)},
        coords={"coord": coords},
        attrs=global_attrs,
    )
    ds.da.attrs = attrs_da

    # deprecated option
    with pytest.warns(
        FutureWarning, match="Passing ``keep_attrs`` to ``rolling`` is deprecated"
    ):
        result = ds.rolling(dim={"coord": 5}, keep_attrs=False).construct("window_dim")

    assert result.attrs == {}
    assert result.da.attrs == {}

    # the keep_attrs in the reduction function takes precedence
    with pytest.warns(
        FutureWarning, match="Passing ``keep_attrs`` to ``rolling`` is deprecated"
    ):
        result = ds.rolling(dim={"coord": 5}, keep_attrs=True).construct(
            "window_dim", keep_attrs=False
        )

    assert result.attrs == {}
    assert result.da.attrs == {}


def test_rolling_properties(ds):
    # catching invalid args
    with pytest.raises(ValueError, match="window must be > 0"):
        ds.rolling(time=-2)
    with pytest.raises(ValueError, match="min_periods must be greater than zero"):
        ds.rolling(time=2, min_periods=0)
    with pytest.raises(KeyError, match="time2"):
        ds.rolling(time2=2)


@pytest.mark.parametrize("name", ("sum", "mean", "std", "var", "min", "max", "median"))
@pytest.mark.parametrize("center", (True, False, None))
@pytest.mark.parametrize("min_periods", (1, None))
@pytest.mark.parametrize("key", ("z1", "z2"))
def test_rolling_wrapped_bottleneck(ds, name, center, min_periods, key):
    bn = pytest.importorskip("bottleneck", minversion="1.1")

    # Test all bottleneck functions
    rolling_obj = ds.rolling(time=7, min_periods=min_periods)

    func_name = f"move_{name}"
    actual = getattr(rolling_obj, name)()
    if key == "z1":  # z1 does not depend on 'Time' axis. Stored as it is.
        expected = ds[key]
    elif key == "z2":
        expected = getattr(bn, func_name)(
            ds[key].values, window=7, axis=0, min_count=min_periods
        )
    else:
        raise ValueError
    assert_array_equal(actual[key].values, expected)

    # Test center
    rolling_obj = ds.rolling(time=7, center=center)
    actual = getattr(rolling_obj, name)()["time"]
    assert_equal(actual, ds["time"])


@requires_numbagg
def test_rolling_exp(ds):

    result = ds.rolling_exp(time=10, window_type="span").mean()
    assert isinstance(result, Dataset)


@requires_numbagg
def test_rolling_exp_keep_attrs(ds):

    attrs_global = {"attrs": "global"}
    attrs_z1 = {"attr": "z1"}

    ds.attrs = attrs_global
    ds.z1.attrs = attrs_z1

    # attrs are kept per default
    result = ds.rolling_exp(time=10).mean()
    assert result.attrs == attrs_global
    assert result.z1.attrs == attrs_z1

    # discard attrs
    result = ds.rolling_exp(time=10).mean(keep_attrs=False)
    assert result.attrs == {}
    assert result.z1.attrs == {}

    # test discard attrs using global option
    with set_options(keep_attrs=False):
        result = ds.rolling_exp(time=10).mean()
    assert result.attrs == {}
    assert result.z1.attrs == {}

    # keyword takes precedence over global option
    with set_options(keep_attrs=False):
        result = ds.rolling_exp(time=10).mean(keep_attrs=True)
    assert result.attrs == attrs_global
    assert result.z1.attrs == attrs_z1

    with set_options(keep_attrs=True):
        result = ds.rolling_exp(time=10).mean(keep_attrs=False)
    assert result.attrs == {}
    assert result.z1.attrs == {}


@pytest.mark.parametrize("center", (True, False))
@pytest.mark.parametrize("min_periods", (None, 1, 2, 3))
@pytest.mark.parametrize("window", (1, 2, 3, 4))
def test_rolling_pandas_compat(center, window, min_periods):
    df = pd.DataFrame(
        {
            "x": np.random.randn(20),
            "y": np.random.randn(20),
            "time": np.linspace(0, 1, 20),
        }
    )
    ds = Dataset.from_dataframe(df)

    if min_periods is not None and window < min_periods:
        min_periods = window

    df_rolling = df.rolling(window, center=center, min_periods=min_periods).mean()
    ds_rolling = ds.rolling(index=window, center=center, min_periods=min_periods).mean()

    np.testing.assert_allclose(df_rolling["x"].values, ds_rolling["x"].values)
    np.testing.assert_allclose(df_rolling.index, ds_rolling["index"])


@pytest.mark.parametrize("center", (True, False))
@pytest.mark.parametrize("window", (1, 2, 3, 4))
def test_rolling_construct(center, window):
    df = pd.DataFrame(
        {
            "x": np.random.randn(20),
            "y": np.random.randn(20),
            "time": np.linspace(0, 1, 20),
        }
    )

    ds = Dataset.from_dataframe(df)
    df_rolling = df.rolling(window, center=center, min_periods=1).mean()
    ds_rolling = ds.rolling(index=window, center=center)

    ds_rolling_mean = ds_rolling.construct("window").mean("window")
    np.testing.assert_allclose(df_rolling["x"].values, ds_rolling_mean["x"].values)
    np.testing.assert_allclose(df_rolling.index, ds_rolling_mean["index"])

    # with stride
    ds_rolling_mean = ds_rolling.construct("window", stride=2).mean("window")
    np.testing.assert_allclose(df_rolling["x"][::2].values, ds_rolling_mean["x"].values)
    np.testing.assert_allclose(df_rolling.index[::2], ds_rolling_mean["index"])
    # with fill_value
    ds_rolling_mean = ds_rolling.construct("window", stride=2, fill_value=0.0).mean(
        "window"
    )
    assert (ds_rolling_mean.isnull().sum() == 0).to_array(dim="vars").all()
    assert (ds_rolling_mean["x"] == 0.0).sum() >= 0


@pytest.mark.slow
@pytest.mark.parametrize("ds", (1, 2), indirect=True)
@pytest.mark.parametrize("center", (True, False))
@pytest.mark.parametrize("min_periods", (None, 1, 2, 3))
@pytest.mark.parametrize("window", (1, 2, 3, 4))
@pytest.mark.parametrize("name", ("sum", "mean", "std", "var", "min", "max", "median"))
def test_rolling_reduce(ds, center, min_periods, window, name):

    if min_periods is not None and window < min_periods:
        min_periods = window

    if name == "std" and window == 1:
        pytest.skip("std with window == 1 is unstable in bottleneck")

    rolling_obj = ds.rolling(time=window, center=center, min_periods=min_periods)

    # add nan prefix to numpy methods to get similar behavior as bottleneck
    actual = rolling_obj.reduce(getattr(np, "nan%s" % name))
    expected = getattr(rolling_obj, name)()
    assert_allclose(actual, expected)
    assert ds.dims == actual.dims
    # make sure the order of data_var are not changed.
    assert list(ds.data_vars.keys()) == list(actual.data_vars.keys())

    # Make sure the dimension order is restored
    for key, src_var in ds.data_vars.items():
        assert src_var.dims == actual[key].dims


@pytest.mark.parametrize("ds", (2,), indirect=True)
@pytest.mark.parametrize("center", (True, False))
@pytest.mark.parametrize("min_periods", (None, 1))
@pytest.mark.parametrize("name", ("sum", "max"))
@pytest.mark.parametrize("dask", (True, False))
def test_ndrolling_reduce(ds, center, min_periods, name, dask):
    if dask and has_dask:
        ds = ds.chunk({"x": 4})

    rolling_obj = ds.rolling(time=4, x=3, center=center, min_periods=min_periods)

    actual = getattr(rolling_obj, name)()
    expected = getattr(
        getattr(
            ds.rolling(time=4, center=center, min_periods=min_periods), name
        )().rolling(x=3, center=center, min_periods=min_periods),
        name,
    )()
    assert_allclose(actual, expected)
    assert actual.dims == expected.dims

    # Do it in the opposite order
    expected = getattr(
        getattr(
            ds.rolling(x=3, center=center, min_periods=min_periods), name
        )().rolling(time=4, center=center, min_periods=min_periods),
        name,
    )()

    assert_allclose(actual, expected)
    assert actual.dims == expected.dims


@pytest.mark.parametrize("center", (True, False, (True, False)))
@pytest.mark.parametrize("fill_value", (np.nan, 0.0))
@pytest.mark.parametrize("dask", (True, False))
def test_ndrolling_construct(center, fill_value, dask):
    da = DataArray(
        np.arange(5 * 6 * 7).reshape(5, 6, 7).astype(float),
        dims=["x", "y", "z"],
        coords={"x": ["a", "b", "c", "d", "e"], "y": np.arange(6)},
    )
    ds = xr.Dataset({"da": da})
    if dask and has_dask:
        ds = ds.chunk({"x": 4})

    actual = ds.rolling(x=3, z=2, center=center).construct(
        x="x1", z="z1", fill_value=fill_value
    )
    if not isinstance(center, tuple):
        center = (center, center)
    expected = (
        ds.rolling(x=3, center=center[0])
        .construct(x="x1", fill_value=fill_value)
        .rolling(z=2, center=center[1])
        .construct(z="z1", fill_value=fill_value)
    )
    assert_allclose(actual, expected)


def test_raise_no_warning_for_nan_in_binary_ops():
    with pytest.warns(None) as record:
        Dataset(data_vars={"x": ("y", [1, 2, np.NaN])}) > 0
    assert len(record) == 0


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("ds", (2,), indirect=True)
def test_raise_no_warning_assert_close(ds):
    assert_allclose(ds, ds)


@pytest.mark.xfail(reason="See https://github.com/pydata/xarray/pull/4369 or docstring")
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("ds", (2,), indirect=True)
@pytest.mark.parametrize("name", ("mean", "max"))
def test_raise_no_warning_dask_rolling_assert_close(ds, name):
    """
    This is a puzzle — I can't easily find the source of the warning. It
    requires `assert_allclose` to be run, for the `ds` param to be 2, and is
    different for `mean` and `max`. `sum` raises no warning.
    """

    ds = ds.chunk({"x": 4})

    rolling_obj = ds.rolling(time=4, x=3)

    actual = getattr(rolling_obj, name)()
    expected = getattr(getattr(ds.rolling(time=4), name)().rolling(x=3), name)()
    assert_allclose(actual, expected)


@pytest.mark.parametrize("dask", [True, False])
@pytest.mark.parametrize("edge_order", [1, 2])
def test_differentiate(dask, edge_order):
    rs = np.random.RandomState(42)
    coord = [0.2, 0.35, 0.4, 0.6, 0.7, 0.75, 0.76, 0.8]

    da = xr.DataArray(
        rs.randn(8, 6),
        dims=["x", "y"],
        coords={"x": coord, "z": 3, "x2d": (("x", "y"), rs.randn(8, 6))},
    )
    if dask and has_dask:
        da = da.chunk({"x": 4})

    ds = xr.Dataset({"var": da})

    # along x
    actual = da.differentiate("x", edge_order)
    expected_x = xr.DataArray(
        np.gradient(da, da["x"], axis=0, edge_order=edge_order),
        dims=da.dims,
        coords=da.coords,
    )
    assert_equal(expected_x, actual)
    assert_equal(
        ds["var"].differentiate("x", edge_order=edge_order),
        ds.differentiate("x", edge_order=edge_order)["var"],
    )
    # coordinate should not change
    assert_equal(da["x"], actual["x"])

    # along y
    actual = da.differentiate("y", edge_order)
    expected_y = xr.DataArray(
        np.gradient(da, da["y"], axis=1, edge_order=edge_order),
        dims=da.dims,
        coords=da.coords,
    )
    assert_equal(expected_y, actual)
    assert_equal(actual, ds.differentiate("y", edge_order=edge_order)["var"])
    assert_equal(
        ds["var"].differentiate("y", edge_order=edge_order),
        ds.differentiate("y", edge_order=edge_order)["var"],
    )

    with pytest.raises(ValueError):
        da.differentiate("x2d")


@pytest.mark.parametrize("dask", [True, False])
def test_differentiate_datetime(dask):
    rs = np.random.RandomState(42)
    coord = np.array(
        [
            "2004-07-13",
            "2006-01-13",
            "2010-08-13",
            "2010-09-13",
            "2010-10-11",
            "2010-12-13",
            "2011-02-13",
            "2012-08-13",
        ],
        dtype="datetime64",
    )

    da = xr.DataArray(
        rs.randn(8, 6),
        dims=["x", "y"],
        coords={"x": coord, "z": 3, "x2d": (("x", "y"), rs.randn(8, 6))},
    )
    if dask and has_dask:
        da = da.chunk({"x": 4})

    # along x
    actual = da.differentiate("x", edge_order=1, datetime_unit="D")
    expected_x = xr.DataArray(
        np.gradient(
            da, da["x"].variable._to_numeric(datetime_unit="D"), axis=0, edge_order=1
        ),
        dims=da.dims,
        coords=da.coords,
    )
    assert_equal(expected_x, actual)

    actual2 = da.differentiate("x", edge_order=1, datetime_unit="h")
    assert np.allclose(actual, actual2 * 24)

    # for datetime variable
    actual = da["x"].differentiate("x", edge_order=1, datetime_unit="D")
    assert np.allclose(actual, 1.0)

    # with different date unit
    da = xr.DataArray(coord.astype("datetime64[ms]"), dims=["x"], coords={"x": coord})
    actual = da.differentiate("x", edge_order=1)
    assert np.allclose(actual, 1.0)


@pytest.mark.skipif(not has_cftime, reason="Test requires cftime.")
@pytest.mark.parametrize("dask", [True, False])
def test_differentiate_cftime(dask):
    rs = np.random.RandomState(42)
    coord = xr.cftime_range("2000", periods=8, freq="2M")

    da = xr.DataArray(
        rs.randn(8, 6),
        coords={"time": coord, "z": 3, "t2d": (("time", "y"), rs.randn(8, 6))},
        dims=["time", "y"],
    )

    if dask and has_dask:
        da = da.chunk({"time": 4})

    actual = da.differentiate("time", edge_order=1, datetime_unit="D")
    expected_data = np.gradient(
        da, da["time"].variable._to_numeric(datetime_unit="D"), axis=0, edge_order=1
    )
    expected = xr.DataArray(expected_data, coords=da.coords, dims=da.dims)
    assert_equal(expected, actual)

    actual2 = da.differentiate("time", edge_order=1, datetime_unit="h")
    assert_allclose(actual, actual2 * 24)

    # Test the differentiation of datetimes themselves
    actual = da["time"].differentiate("time", edge_order=1, datetime_unit="D")
    assert_allclose(actual, xr.ones_like(da["time"]).astype(float))


@pytest.mark.parametrize("dask", [True, False])
def test_integrate(dask):
    rs = np.random.RandomState(42)
    coord = [0.2, 0.35, 0.4, 0.6, 0.7, 0.75, 0.76, 0.8]

    da = xr.DataArray(
        rs.randn(8, 6),
        dims=["x", "y"],
        coords={
            "x": coord,
            "x2": (("x",), rs.randn(8)),
            "z": 3,
            "x2d": (("x", "y"), rs.randn(8, 6)),
        },
    )
    if dask and has_dask:
        da = da.chunk({"x": 4})

    ds = xr.Dataset({"var": da})

    # along x
    actual = da.integrate("x")
    # coordinate that contains x should be dropped.
    expected_x = xr.DataArray(
        np.trapz(da.compute(), da["x"], axis=0),
        dims=["y"],
        coords={k: v for k, v in da.coords.items() if "x" not in v.dims},
    )
    assert_allclose(expected_x, actual.compute())
    assert_equal(ds["var"].integrate("x"), ds.integrate("x")["var"])

    # make sure result is also a dask array (if the source is dask array)
    assert isinstance(actual.data, type(da.data))

    # along y
    actual = da.integrate("y")
    expected_y = xr.DataArray(
        np.trapz(da, da["y"], axis=1),
        dims=["x"],
        coords={k: v for k, v in da.coords.items() if "y" not in v.dims},
    )
    assert_allclose(expected_y, actual.compute())
    assert_equal(actual, ds.integrate("y")["var"])
    assert_equal(ds["var"].integrate("y"), ds.integrate("y")["var"])

    # along x and y
    actual = da.integrate(("y", "x"))
    assert actual.ndim == 0

    with pytest.raises(ValueError):
        da.integrate("x2d")

    with pytest.warns(FutureWarning):
        da.integrate(dim="x")


@pytest.mark.parametrize("dask", [True, False])
@pytest.mark.parametrize("which_datetime", ["np", "cftime"])
def test_trapz_datetime(dask, which_datetime):
    rs = np.random.RandomState(42)
    if which_datetime == "np":
        coord = np.array(
            [
                "2004-07-13",
                "2006-01-13",
                "2010-08-13",
                "2010-09-13",
                "2010-10-11",
                "2010-12-13",
                "2011-02-13",
                "2012-08-13",
            ],
            dtype="datetime64",
        )
    else:
        if not has_cftime:
            pytest.skip("Test requires cftime.")
        coord = xr.cftime_range("2000", periods=8, freq="2D")

    da = xr.DataArray(
        rs.randn(8, 6),
        coords={"time": coord, "z": 3, "t2d": (("time", "y"), rs.randn(8, 6))},
        dims=["time", "y"],
    )

    if dask and has_dask:
        da = da.chunk({"time": 4})

    actual = da.integrate("time", datetime_unit="D")
    expected_data = np.trapz(
        da.data,
        duck_array_ops.datetime_to_numeric(da["time"].data, datetime_unit="D"),
        axis=0,
    )
    expected = xr.DataArray(
        expected_data,
        dims=["y"],
        coords={k: v for k, v in da.coords.items() if "time" not in v.dims},
    )
    assert_allclose(expected, actual.compute())

    # make sure result is also a dask array (if the source is dask array)
    assert isinstance(actual.data, type(da.data))

    actual2 = da.integrate("time", datetime_unit="h")
    assert_allclose(actual, actual2 / 24.0)


def test_no_dict():
    d = Dataset()
    with pytest.raises(AttributeError):
        d.__dict__


def test_subclass_slots():
    """Test that Dataset subclasses must explicitly define ``__slots__``.

    .. note::
       As of 0.13.0, this is actually mitigated into a FutureWarning for any class
       defined outside of the xarray package.
    """
    with pytest.raises(AttributeError) as e:

        class MyDS(Dataset):
            pass

    assert str(e.value) == "MyDS must explicitly define __slots__"


def test_weakref():
    """Classes with __slots__ are incompatible with the weakref module unless they
    explicitly state __weakref__ among their slots
    """
    from weakref import ref

    ds = Dataset()
    r = ref(ds)
    assert r() is ds


def test_deepcopy_obj_array():
    x0 = Dataset(dict(foo=DataArray(np.array([object()]))))
    x1 = deepcopy(x0)
    assert x0["foo"].values[0] is not x1["foo"].values[0]
