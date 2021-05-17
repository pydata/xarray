import pickle
import sys
import warnings
from copy import deepcopy
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
from pandas.core.computation.ops import UndefinedVariableError
from pandas.tseries.frequencies import to_offset

import xarray as xr
from xarray import (
    DataArray,
    Dataset,
    IndexVariable,
    Variable,
    align,
    broadcast,
    set_options,
)
from xarray.coding.times import CFDatetimeCoder
from xarray.convert import from_cdms2
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.indexes import Index, PandasIndex, propagate_indexes
from xarray.core.utils import is_scalar
from xarray.tests import (
    LooseVersion,
    ReturnItem,
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_identical,
    has_dask,
    raise_if_dask_computes,
    requires_bottleneck,
    requires_dask,
    requires_iris,
    requires_numbagg,
    requires_numexpr,
    requires_scipy,
    requires_sparse,
    source_ndarray,
)

pytestmark = [
    pytest.mark.filterwarnings("error:Mean of empty slice"),
    pytest.mark.filterwarnings("error:All-NaN (slice|axis) encountered"),
]


class TestDataArray:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.attrs = {"attr1": "value1", "attr2": 2929}
        self.x = np.random.random((10, 20))
        self.v = Variable(["x", "y"], self.x)
        self.va = Variable(["x", "y"], self.x, self.attrs)
        self.ds = Dataset({"foo": self.v})
        self.dv = self.ds["foo"]

        self.mindex = pd.MultiIndex.from_product(
            [["a", "b"], [1, 2]], names=("level_1", "level_2")
        )
        self.mda = DataArray([0, 1, 2, 3], coords={"x": self.mindex}, dims="x")

    def test_repr(self):
        v = Variable(["time", "x"], [[1, 2, 3], [4, 5, 6]], {"foo": "bar"})
        coords = {"x": np.arange(3, dtype=np.int64), "other": np.int64(0)}
        data_array = DataArray(v, coords, name="my_variable")
        expected = dedent(
            """\
            <xarray.DataArray 'my_variable' (time: 2, x: 3)>
            array([[1, 2, 3],
                   [4, 5, 6]])
            Coordinates:
              * x        (x) int64 0 1 2
                other    int64 0
            Dimensions without coordinates: time
            Attributes:
                foo:      bar"""
        )
        assert expected == repr(data_array)

    def test_repr_multiindex(self):
        expected = dedent(
            """\
            <xarray.DataArray (x: 4)>
            array([0, 1, 2, 3])
            Coordinates:
              * x        (x) MultiIndex
              - level_1  (x) object 'a' 'a' 'b' 'b'
              - level_2  (x) int64 1 2 1 2"""
        )
        assert expected == repr(self.mda)

    @pytest.mark.skipif(
        LooseVersion(np.__version__) < "1.16",
        reason="old versions of numpy have different printing behavior",
    )
    def test_repr_multiindex_long(self):
        mindex_long = pd.MultiIndex.from_product(
            [["a", "b", "c", "d"], [1, 2, 3, 4, 5, 6, 7, 8]],
            names=("level_1", "level_2"),
        )
        mda_long = DataArray(list(range(32)), coords={"x": mindex_long}, dims="x")
        expected = dedent(
            """\
            <xarray.DataArray (x: 32)>
            array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
            Coordinates:
              * x        (x) MultiIndex
              - level_1  (x) object 'a' 'a' 'a' 'a' 'a' 'a' 'a' ... 'd' 'd' 'd' 'd' 'd' 'd'
              - level_2  (x) int64 1 2 3 4 5 6 7 8 1 2 3 4 5 6 ... 4 5 6 7 8 1 2 3 4 5 6 7 8"""
        )
        assert expected == repr(mda_long)

    def test_properties(self):
        assert_equal(self.dv.variable, self.v)
        assert_array_equal(self.dv.values, self.v.values)
        for attr in ["dims", "dtype", "shape", "size", "nbytes", "ndim", "attrs"]:
            assert getattr(self.dv, attr) == getattr(self.v, attr)
        assert len(self.dv) == len(self.v)
        assert_equal(self.dv.variable, self.v)
        assert set(self.dv.coords) == set(self.ds.coords)
        for k, v in self.dv.coords.items():
            assert_array_equal(v, self.ds.coords[k])
        with pytest.raises(AttributeError):
            self.dv.dataset
        assert isinstance(self.ds["x"].to_index(), pd.Index)
        with pytest.raises(ValueError, match=r"must be 1-dimensional"):
            self.ds["foo"].to_index()
        with pytest.raises(AttributeError):
            self.dv.variable = self.v

    def test_data_property(self):
        array = DataArray(np.zeros((3, 4)))
        actual = array.copy()
        actual.values = np.ones((3, 4))
        assert_array_equal(np.ones((3, 4)), actual.values)
        actual.data = 2 * np.ones((3, 4))
        assert_array_equal(2 * np.ones((3, 4)), actual.data)
        assert_array_equal(actual.data, actual.values)

    def test_indexes(self):
        array = DataArray(np.zeros((2, 3)), [("x", [0, 1]), ("y", ["a", "b", "c"])])
        expected_indexes = {"x": pd.Index([0, 1]), "y": pd.Index(["a", "b", "c"])}
        expected_xindexes = {k: PandasIndex(idx) for k, idx in expected_indexes.items()}
        assert array.xindexes.keys() == expected_xindexes.keys()
        assert array.indexes.keys() == expected_indexes.keys()
        assert all([isinstance(idx, pd.Index) for idx in array.indexes.values()])
        assert all([isinstance(idx, Index) for idx in array.xindexes.values()])
        for k in expected_indexes:
            assert array.xindexes[k].equals(expected_xindexes[k])
            assert array.indexes[k].equals(expected_indexes[k])

    def test_get_index(self):
        array = DataArray(np.zeros((2, 3)), coords={"x": ["a", "b"]}, dims=["x", "y"])
        assert array.get_index("x").equals(pd.Index(["a", "b"]))
        assert array.get_index("y").equals(pd.Index([0, 1, 2]))
        with pytest.raises(KeyError):
            array.get_index("z")

    def test_get_index_size_zero(self):
        array = DataArray(np.zeros((0,)), dims=["x"])
        actual = array.get_index("x")
        expected = pd.Index([], dtype=np.int64)
        assert actual.equals(expected)
        assert actual.dtype == expected.dtype

    def test_struct_array_dims(self):
        """
        This test checks subraction of two DataArrays for the case
        when dimension is a structured array.
        """
        # GH837, GH861
        # checking array subtraction when dims are the same
        p_data = np.array(
            [("Abe", 180), ("Stacy", 150), ("Dick", 200)],
            dtype=[("name", "|S256"), ("height", object)],
        )
        weights_0 = DataArray(
            [80, 56, 120], dims=["participant"], coords={"participant": p_data}
        )
        weights_1 = DataArray(
            [81, 52, 115], dims=["participant"], coords={"participant": p_data}
        )
        actual = weights_1 - weights_0

        expected = DataArray(
            [1, -4, -5], dims=["participant"], coords={"participant": p_data}
        )

        assert_identical(actual, expected)

        # checking array subraction when dims are not the same
        p_data_alt = np.array(
            [("Abe", 180), ("Stacy", 151), ("Dick", 200)],
            dtype=[("name", "|S256"), ("height", object)],
        )
        weights_1 = DataArray(
            [81, 52, 115], dims=["participant"], coords={"participant": p_data_alt}
        )
        actual = weights_1 - weights_0

        expected = DataArray(
            [1, -5], dims=["participant"], coords={"participant": p_data[[0, 2]]}
        )

        assert_identical(actual, expected)

        # checking array subraction when dims are not the same and one
        # is np.nan
        p_data_nan = np.array(
            [("Abe", 180), ("Stacy", np.nan), ("Dick", 200)],
            dtype=[("name", "|S256"), ("height", object)],
        )
        weights_1 = DataArray(
            [81, 52, 115], dims=["participant"], coords={"participant": p_data_nan}
        )
        actual = weights_1 - weights_0

        expected = DataArray(
            [1, -5], dims=["participant"], coords={"participant": p_data[[0, 2]]}
        )

        assert_identical(actual, expected)

    def test_name(self):
        arr = self.dv
        assert arr.name == "foo"

        copied = arr.copy()
        arr.name = "bar"
        assert arr.name == "bar"
        assert_equal(copied, arr)

        actual = DataArray(IndexVariable("x", [3]))
        actual.name = "y"
        expected = DataArray([3], [("x", [3])], name="y")
        assert_identical(actual, expected)

    def test_dims(self):
        arr = self.dv
        assert arr.dims == ("x", "y")

        with pytest.raises(AttributeError, match=r"you cannot assign"):
            arr.dims = ("w", "z")

    def test_sizes(self):
        array = DataArray(np.zeros((3, 4)), dims=["x", "y"])
        assert array.sizes == {"x": 3, "y": 4}
        assert tuple(array.sizes) == array.dims
        with pytest.raises(TypeError):
            array.sizes["foo"] = 5

    def test_encoding(self):
        expected = {"foo": "bar"}
        self.dv.encoding["foo"] = "bar"
        assert expected == self.dv.encoding

        expected = {"baz": 0}
        self.dv.encoding = expected

        assert expected is not self.dv.encoding

    def test_constructor(self):
        data = np.random.random((2, 3))

        actual = DataArray(data)
        expected = Dataset({None: (["dim_0", "dim_1"], data)})[None]
        assert_identical(expected, actual)

        actual = DataArray(data, [["a", "b"], [-1, -2, -3]])
        expected = Dataset(
            {
                None: (["dim_0", "dim_1"], data),
                "dim_0": ("dim_0", ["a", "b"]),
                "dim_1": ("dim_1", [-1, -2, -3]),
            }
        )[None]
        assert_identical(expected, actual)

        actual = DataArray(
            data, [pd.Index(["a", "b"], name="x"), pd.Index([-1, -2, -3], name="y")]
        )
        expected = Dataset(
            {None: (["x", "y"], data), "x": ("x", ["a", "b"]), "y": ("y", [-1, -2, -3])}
        )[None]
        assert_identical(expected, actual)

        coords = [["a", "b"], [-1, -2, -3]]
        actual = DataArray(data, coords, ["x", "y"])
        assert_identical(expected, actual)

        coords = [pd.Index(["a", "b"], name="A"), pd.Index([-1, -2, -3], name="B")]
        actual = DataArray(data, coords, ["x", "y"])
        assert_identical(expected, actual)

        coords = {"x": ["a", "b"], "y": [-1, -2, -3]}
        actual = DataArray(data, coords, ["x", "y"])
        assert_identical(expected, actual)

        coords = [("x", ["a", "b"]), ("y", [-1, -2, -3])]
        actual = DataArray(data, coords)
        assert_identical(expected, actual)

        expected = Dataset({None: (["x", "y"], data), "x": ("x", ["a", "b"])})[None]
        actual = DataArray(data, {"x": ["a", "b"]}, ["x", "y"])
        assert_identical(expected, actual)

        actual = DataArray(data, dims=["x", "y"])
        expected = Dataset({None: (["x", "y"], data)})[None]
        assert_identical(expected, actual)

        actual = DataArray(data, dims=["x", "y"], name="foo")
        expected = Dataset({"foo": (["x", "y"], data)})["foo"]
        assert_identical(expected, actual)

        actual = DataArray(data, name="foo")
        expected = Dataset({"foo": (["dim_0", "dim_1"], data)})["foo"]
        assert_identical(expected, actual)

        actual = DataArray(data, dims=["x", "y"], attrs={"bar": 2})
        expected = Dataset({None: (["x", "y"], data, {"bar": 2})})[None]
        assert_identical(expected, actual)

        actual = DataArray(data, dims=["x", "y"])
        expected = Dataset({None: (["x", "y"], data, {}, {"bar": 2})})[None]
        assert_identical(expected, actual)

    def test_constructor_invalid(self):
        data = np.random.randn(3, 2)

        with pytest.raises(ValueError, match=r"coords is not dict-like"):
            DataArray(data, [[0, 1, 2]], ["x", "y"])

        with pytest.raises(ValueError, match=r"not a subset of the .* dim"):
            DataArray(data, {"x": [0, 1, 2]}, ["a", "b"])
        with pytest.raises(ValueError, match=r"not a subset of the .* dim"):
            DataArray(data, {"x": [0, 1, 2]})

        with pytest.raises(TypeError, match=r"is not a string"):
            DataArray(data, dims=["x", None])

        with pytest.raises(ValueError, match=r"conflicting sizes for dim"):
            DataArray([1, 2, 3], coords=[("x", [0, 1])])
        with pytest.raises(ValueError, match=r"conflicting sizes for dim"):
            DataArray([1, 2], coords={"x": [0, 1], "y": ("x", [1])}, dims="x")

        with pytest.raises(ValueError, match=r"conflicting MultiIndex"):
            DataArray(np.random.rand(4, 4), [("x", self.mindex), ("y", self.mindex)])
        with pytest.raises(ValueError, match=r"conflicting MultiIndex"):
            DataArray(np.random.rand(4, 4), [("x", self.mindex), ("level_1", range(4))])

        with pytest.raises(ValueError, match=r"matching the dimension size"):
            DataArray(data, coords={"x": 0}, dims=["x", "y"])

    def test_constructor_from_self_described(self):
        data = [[-0.1, 21], [0, 2]]
        expected = DataArray(
            data,
            coords={"x": ["a", "b"], "y": [-1, -2]},
            dims=["x", "y"],
            name="foobar",
            attrs={"bar": 2},
        )
        actual = DataArray(expected)
        assert_identical(expected, actual)

        actual = DataArray(expected.values, actual.coords)
        assert_equal(expected, actual)

        frame = pd.DataFrame(
            data,
            index=pd.Index(["a", "b"], name="x"),
            columns=pd.Index([-1, -2], name="y"),
        )
        actual = DataArray(frame)
        assert_equal(expected, actual)

        series = pd.Series(data[0], index=pd.Index([-1, -2], name="y"))
        actual = DataArray(series)
        assert_equal(expected[0].reset_coords("x", drop=True), actual)

        if LooseVersion(pd.__version__) < "0.25.0":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"\W*Panel is deprecated")
                panel = pd.Panel({0: frame})
            actual = DataArray(panel)
            expected = DataArray([data], expected.coords, ["dim_0", "x", "y"])
            expected["dim_0"] = [0]
            assert_identical(expected, actual)

        expected = DataArray(
            data,
            coords={"x": ["a", "b"], "y": [-1, -2], "a": 0, "z": ("x", [-0.5, 0.5])},
            dims=["x", "y"],
        )
        actual = DataArray(expected)
        assert_identical(expected, actual)

        actual = DataArray(expected.values, expected.coords)
        assert_identical(expected, actual)

        expected = Dataset({"foo": ("foo", ["a", "b"])})["foo"]
        actual = DataArray(pd.Index(["a", "b"], name="foo"))
        assert_identical(expected, actual)

        actual = DataArray(IndexVariable("foo", ["a", "b"]))
        assert_identical(expected, actual)

    def test_constructor_from_0d(self):
        expected = Dataset({None: ([], 0)})[None]
        actual = DataArray(0)
        assert_identical(expected, actual)

    @requires_dask
    def test_constructor_dask_coords(self):
        # regression test for GH1684
        import dask.array as da

        coord = da.arange(8, chunks=(4,))
        data = da.random.random((8, 8), chunks=(4, 4)) + 1
        actual = DataArray(data, coords={"x": coord, "y": coord}, dims=["x", "y"])

        ecoord = np.arange(8)
        expected = DataArray(data, coords={"x": ecoord, "y": ecoord}, dims=["x", "y"])
        assert_equal(actual, expected)

    def test_equals_and_identical(self):
        orig = DataArray(np.arange(5.0), {"a": 42}, dims="x")

        expected = orig
        actual = orig.copy()
        assert expected.equals(actual)
        assert expected.identical(actual)

        actual = expected.rename("baz")
        assert expected.equals(actual)
        assert not expected.identical(actual)

        actual = expected.rename({"x": "xxx"})
        assert not expected.equals(actual)
        assert not expected.identical(actual)

        actual = expected.copy()
        actual.attrs["foo"] = "bar"
        assert expected.equals(actual)
        assert not expected.identical(actual)

        actual = expected.copy()
        actual["x"] = ("x", -np.arange(5))
        assert not expected.equals(actual)
        assert not expected.identical(actual)

        actual = expected.reset_coords(drop=True)
        assert not expected.equals(actual)
        assert not expected.identical(actual)

        actual = orig.copy()
        actual[0] = np.nan
        expected = actual.copy()
        assert expected.equals(actual)
        assert expected.identical(actual)

        actual[:] = np.nan
        assert not expected.equals(actual)
        assert not expected.identical(actual)

        actual = expected.copy()
        actual["a"] = 100000
        assert not expected.equals(actual)
        assert not expected.identical(actual)

    def test_equals_failures(self):
        orig = DataArray(np.arange(5.0), {"a": 42}, dims="x")
        assert not orig.equals(np.arange(5))
        assert not orig.identical(123)
        assert not orig.broadcast_equals({1: 2})

    def test_broadcast_equals(self):
        a = DataArray([0, 0], {"y": 0}, dims="x")
        b = DataArray([0, 0], {"y": ("x", [0, 0])}, dims="x")
        assert a.broadcast_equals(b)
        assert b.broadcast_equals(a)
        assert not a.equals(b)
        assert not a.identical(b)

        c = DataArray([0], coords={"x": 0}, dims="y")
        assert not a.broadcast_equals(c)
        assert not c.broadcast_equals(a)

    def test_getitem(self):
        # strings pull out dataarrays
        assert_identical(self.dv, self.ds["foo"])
        x = self.dv["x"]
        y = self.dv["y"]
        assert_identical(self.ds["x"], x)
        assert_identical(self.ds["y"], y)

        arr = ReturnItem()
        for i in [
            arr[:],
            arr[...],
            arr[x.values],
            arr[x.variable],
            arr[x],
            arr[x, y],
            arr[x.values > -1],
            arr[x.variable > -1],
            arr[x > -1],
            arr[x > -1, y > -1],
        ]:
            assert_equal(self.dv, self.dv[i])
        for i in [
            arr[0],
            arr[:, 0],
            arr[:3, :2],
            arr[x.values[:3]],
            arr[x.variable[:3]],
            arr[x[:3]],
            arr[x[:3], y[:4]],
            arr[x.values > 3],
            arr[x.variable > 3],
            arr[x > 3],
            arr[x > 3, y > 3],
        ]:
            assert_array_equal(self.v[i], self.dv[i])

    def test_getitem_dict(self):
        actual = self.dv[{"x": slice(3), "y": 0}]
        expected = self.dv.isel(x=slice(3), y=0)
        assert_identical(expected, actual)

    def test_getitem_coords(self):
        orig = DataArray(
            [[10], [20]],
            {
                "x": [1, 2],
                "y": [3],
                "z": 4,
                "x2": ("x", ["a", "b"]),
                "y2": ("y", ["c"]),
                "xy": (["y", "x"], [["d", "e"]]),
            },
            dims=["x", "y"],
        )

        assert_identical(orig, orig[:])
        assert_identical(orig, orig[:, :])
        assert_identical(orig, orig[...])
        assert_identical(orig, orig[:2, :1])
        assert_identical(orig, orig[[0, 1], [0]])

        actual = orig[0, 0]
        expected = DataArray(
            10, {"x": 1, "y": 3, "z": 4, "x2": "a", "y2": "c", "xy": "d"}
        )
        assert_identical(expected, actual)

        actual = orig[0, :]
        expected = DataArray(
            [10],
            {
                "x": 1,
                "y": [3],
                "z": 4,
                "x2": "a",
                "y2": ("y", ["c"]),
                "xy": ("y", ["d"]),
            },
            dims="y",
        )
        assert_identical(expected, actual)

        actual = orig[:, 0]
        expected = DataArray(
            [10, 20],
            {
                "x": [1, 2],
                "y": 3,
                "z": 4,
                "x2": ("x", ["a", "b"]),
                "y2": "c",
                "xy": ("x", ["d", "e"]),
            },
            dims="x",
        )
        assert_identical(expected, actual)

    def test_getitem_dataarray(self):
        # It should not conflict
        da = DataArray(np.arange(12).reshape((3, 4)), dims=["x", "y"])
        ind = DataArray([[0, 1], [0, 1]], dims=["x", "z"])
        actual = da[ind]
        assert_array_equal(actual, da.values[[[0, 1], [0, 1]], :])

        da = DataArray(
            np.arange(12).reshape((3, 4)),
            dims=["x", "y"],
            coords={"x": [0, 1, 2], "y": ["a", "b", "c", "d"]},
        )
        ind = xr.DataArray([[0, 1], [0, 1]], dims=["X", "Y"])
        actual = da[ind]
        expected = da.values[[[0, 1], [0, 1]], :]
        assert_array_equal(actual, expected)
        assert actual.dims == ("X", "Y", "y")

        # boolean indexing
        ind = xr.DataArray([True, True, False], dims=["x"])
        assert_equal(da[ind], da[[0, 1], :])
        assert_equal(da[ind], da[[0, 1]])
        assert_equal(da[ind], da[ind.values])

    def test_getitem_empty_index(self):
        da = DataArray(np.arange(12).reshape((3, 4)), dims=["x", "y"])
        assert_identical(da[{"x": []}], DataArray(np.zeros((0, 4)), dims=["x", "y"]))
        assert_identical(
            da.loc[{"y": []}], DataArray(np.zeros((3, 0)), dims=["x", "y"])
        )
        assert_identical(da[[]], DataArray(np.zeros((0, 4)), dims=["x", "y"]))

    def test_setitem(self):
        # basic indexing should work as numpy's indexing
        tuples = [
            (0, 0),
            (0, slice(None, None)),
            (slice(None, None), slice(None, None)),
            (slice(None, None), 0),
            ([1, 0], slice(None, None)),
            (slice(None, None), [1, 0]),
        ]
        for t in tuples:
            expected = np.arange(6).reshape(3, 2)
            orig = DataArray(
                np.arange(6).reshape(3, 2),
                {
                    "x": [1, 2, 3],
                    "y": ["a", "b"],
                    "z": 4,
                    "x2": ("x", ["a", "b", "c"]),
                    "y2": ("y", ["d", "e"]),
                },
                dims=["x", "y"],
            )
            orig[t] = 1
            expected[t] = 1
            assert_array_equal(orig.values, expected)

    def test_setitem_fancy(self):
        # vectorized indexing
        da = DataArray(np.ones((3, 2)), dims=["x", "y"])
        ind = Variable(["a"], [0, 1])
        da[dict(x=ind, y=ind)] = 0
        expected = DataArray([[0, 1], [1, 0], [1, 1]], dims=["x", "y"])
        assert_identical(expected, da)
        # assign another 0d-variable
        da[dict(x=ind, y=ind)] = Variable((), 0)
        expected = DataArray([[0, 1], [1, 0], [1, 1]], dims=["x", "y"])
        assert_identical(expected, da)
        # assign another 1d-variable
        da[dict(x=ind, y=ind)] = Variable(["a"], [2, 3])
        expected = DataArray([[2, 1], [1, 3], [1, 1]], dims=["x", "y"])
        assert_identical(expected, da)

        # 2d-vectorized indexing
        da = DataArray(np.ones((3, 2)), dims=["x", "y"])
        ind_x = DataArray([[0, 1]], dims=["a", "b"])
        ind_y = DataArray([[1, 0]], dims=["a", "b"])
        da[dict(x=ind_x, y=ind_y)] = 0
        expected = DataArray([[1, 0], [0, 1], [1, 1]], dims=["x", "y"])
        assert_identical(expected, da)

        da = DataArray(np.ones((3, 2)), dims=["x", "y"])
        ind = Variable(["a"], [0, 1])
        da[ind] = 0
        expected = DataArray([[0, 0], [0, 0], [1, 1]], dims=["x", "y"])
        assert_identical(expected, da)

    def test_setitem_dataarray(self):
        def get_data():
            return DataArray(
                np.ones((4, 3, 2)),
                dims=["x", "y", "z"],
                coords={
                    "x": np.arange(4),
                    "y": ["a", "b", "c"],
                    "non-dim": ("x", [1, 3, 4, 2]),
                },
            )

        da = get_data()
        # indexer with inconsistent coordinates.
        ind = DataArray(np.arange(1, 4), dims=["x"], coords={"x": np.random.randn(3)})
        with pytest.raises(IndexError, match=r"dimension coordinate 'x'"):
            da[dict(x=ind)] = 0

        # indexer with consistent coordinates.
        ind = DataArray(np.arange(1, 4), dims=["x"], coords={"x": np.arange(1, 4)})
        da[dict(x=ind)] = 0  # should not raise
        assert np.allclose(da[dict(x=ind)].values, 0)
        assert_identical(da["x"], get_data()["x"])
        assert_identical(da["non-dim"], get_data()["non-dim"])

        da = get_data()
        # conflict in the assigning values
        value = xr.DataArray(
            np.zeros((3, 3, 2)),
            dims=["x", "y", "z"],
            coords={"x": [0, 1, 2], "non-dim": ("x", [0, 2, 4])},
        )
        with pytest.raises(IndexError, match=r"dimension coordinate 'x'"):
            da[dict(x=ind)] = value

        # consistent coordinate in the assigning values
        value = xr.DataArray(
            np.zeros((3, 3, 2)),
            dims=["x", "y", "z"],
            coords={"x": [1, 2, 3], "non-dim": ("x", [0, 2, 4])},
        )
        da[dict(x=ind)] = value
        assert np.allclose(da[dict(x=ind)].values, 0)
        assert_identical(da["x"], get_data()["x"])
        assert_identical(da["non-dim"], get_data()["non-dim"])

        # Conflict in the non-dimension coordinate
        value = xr.DataArray(
            np.zeros((3, 3, 2)),
            dims=["x", "y", "z"],
            coords={"x": [1, 2, 3], "non-dim": ("x", [0, 2, 4])},
        )
        da[dict(x=ind)] = value  # should not raise

        # conflict in the assigning values
        value = xr.DataArray(
            np.zeros((3, 3, 2)),
            dims=["x", "y", "z"],
            coords={"x": [0, 1, 2], "non-dim": ("x", [0, 2, 4])},
        )
        with pytest.raises(IndexError, match=r"dimension coordinate 'x'"):
            da[dict(x=ind)] = value

        # consistent coordinate in the assigning values
        value = xr.DataArray(
            np.zeros((3, 3, 2)),
            dims=["x", "y", "z"],
            coords={"x": [1, 2, 3], "non-dim": ("x", [0, 2, 4])},
        )
        da[dict(x=ind)] = value  # should not raise

    def test_contains(self):
        data_array = DataArray([1, 2])
        assert 1 in data_array
        assert 3 not in data_array

    def test_attr_sources_multiindex(self):
        # make sure attr-style access for multi-index levels
        # returns DataArray objects
        assert isinstance(self.mda.level_1, DataArray)

    def test_pickle(self):
        data = DataArray(np.random.random((3, 3)), dims=("id", "time"))
        roundtripped = pickle.loads(pickle.dumps(data))
        assert_identical(data, roundtripped)

    @requires_dask
    def test_chunk(self):
        unblocked = DataArray(np.ones((3, 4)))
        assert unblocked.chunks is None

        blocked = unblocked.chunk()
        assert blocked.chunks == ((3,), (4,))
        first_dask_name = blocked.data.name

        blocked = unblocked.chunk(chunks=((2, 1), (2, 2)))
        assert blocked.chunks == ((2, 1), (2, 2))
        assert blocked.data.name != first_dask_name

        blocked = unblocked.chunk(chunks=(3, 3))
        assert blocked.chunks == ((3,), (3, 1))
        assert blocked.data.name != first_dask_name

        # name doesn't change when rechunking by same amount
        # this fails if ReprObject doesn't have __dask_tokenize__ defined
        assert unblocked.chunk(2).data.name == unblocked.chunk(2).data.name

        assert blocked.load().chunks is None

        # Check that kwargs are passed
        import dask.array as da

        blocked = unblocked.chunk(name_prefix="testname_")
        assert isinstance(blocked.data, da.Array)
        assert "testname_" in blocked.data.name

    def test_isel(self):
        assert_identical(self.dv[0], self.dv.isel(x=0))
        assert_identical(self.dv, self.dv.isel(x=slice(None)))
        assert_identical(self.dv[:3], self.dv.isel(x=slice(3)))
        assert_identical(self.dv[:3, :5], self.dv.isel(x=slice(3), y=slice(5)))
        with pytest.raises(
            ValueError,
            match=r"Dimensions {'not_a_dim'} do not exist. Expected "
            r"one or more of \('x', 'y'\)",
        ):
            self.dv.isel(not_a_dim=0)
        with pytest.warns(
            UserWarning,
            match=r"Dimensions {'not_a_dim'} do not exist. "
            r"Expected one or more of \('x', 'y'\)",
        ):
            self.dv.isel(not_a_dim=0, missing_dims="warn")
        assert_identical(self.dv, self.dv.isel(not_a_dim=0, missing_dims="ignore"))

    def test_isel_types(self):
        # regression test for #1405
        da = DataArray([1, 2, 3], dims="x")
        # uint64
        assert_identical(
            da.isel(x=np.array([0], dtype="uint64")), da.isel(x=np.array([0]))
        )
        # uint32
        assert_identical(
            da.isel(x=np.array([0], dtype="uint32")), da.isel(x=np.array([0]))
        )
        # int64
        assert_identical(
            da.isel(x=np.array([0], dtype="int64")), da.isel(x=np.array([0]))
        )

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_isel_fancy(self):
        shape = (10, 7, 6)
        np_array = np.random.random(shape)
        da = DataArray(
            np_array, dims=["time", "y", "x"], coords={"time": np.arange(0, 100, 10)}
        )
        y = [1, 3]
        x = [3, 0]

        expected = da.values[:, y, x]

        actual = da.isel(y=(("test_coord",), y), x=(("test_coord",), x))
        assert actual.coords["test_coord"].shape == (len(y),)
        assert list(actual.coords) == ["time"]
        assert actual.dims == ("time", "test_coord")

        np.testing.assert_equal(actual, expected)

        # a few corner cases
        da.isel(
            time=(("points",), [1, 2]), x=(("points",), [2, 2]), y=(("points",), [3, 4])
        )
        np.testing.assert_allclose(
            da.isel(
                time=(("p",), [1]), x=(("p",), [2]), y=(("p",), [4])
            ).values.squeeze(),
            np_array[1, 4, 2].squeeze(),
        )
        da.isel(time=(("points",), [1, 2]))
        y = [-1, 0]
        x = [-2, 2]
        expected = da.values[:, y, x]
        actual = da.isel(x=(("points",), x), y=(("points",), y)).values
        np.testing.assert_equal(actual, expected)

        # test that the order of the indexers doesn't matter
        assert_identical(
            da.isel(y=(("points",), y), x=(("points",), x)),
            da.isel(x=(("points",), x), y=(("points",), y)),
        )

        # make sure we're raising errors in the right places
        with pytest.raises(IndexError, match=r"Dimensions of indexers mismatch"):
            da.isel(y=(("points",), [1, 2]), x=(("points",), [1, 2, 3]))

        # tests using index or DataArray as indexers
        stations = Dataset()
        stations["station"] = (("station",), ["A", "B", "C"])
        stations["dim1s"] = (("station",), [1, 2, 3])
        stations["dim2s"] = (("station",), [4, 5, 1])

        actual = da.isel(x=stations["dim1s"], y=stations["dim2s"])
        assert "station" in actual.coords
        assert "station" in actual.dims
        assert_identical(actual["station"], stations["station"])

        with pytest.raises(ValueError, match=r"conflicting values for "):
            da.isel(
                x=DataArray([0, 1, 2], dims="station", coords={"station": [0, 1, 2]}),
                y=DataArray([0, 1, 2], dims="station", coords={"station": [0, 1, 3]}),
            )

        # multi-dimensional selection
        stations = Dataset()
        stations["a"] = (("a",), ["A", "B", "C"])
        stations["b"] = (("b",), [0, 1])
        stations["dim1s"] = (("a", "b"), [[1, 2], [2, 3], [3, 4]])
        stations["dim2s"] = (("a",), [4, 5, 1])

        actual = da.isel(x=stations["dim1s"], y=stations["dim2s"])
        assert "a" in actual.coords
        assert "a" in actual.dims
        assert "b" in actual.coords
        assert "b" in actual.dims
        assert_identical(actual["a"], stations["a"])
        assert_identical(actual["b"], stations["b"])
        expected = da.variable[
            :, stations["dim2s"].variable, stations["dim1s"].variable
        ]
        assert_array_equal(actual, expected)

    def test_sel(self):
        self.ds["x"] = ("x", np.array(list("abcdefghij")))
        da = self.ds["foo"]
        assert_identical(da, da.sel(x=slice(None)))
        assert_identical(da[1], da.sel(x="b"))
        assert_identical(da[:3], da.sel(x=slice("c")))
        assert_identical(da[:3], da.sel(x=["a", "b", "c"]))
        assert_identical(da[:, :4], da.sel(y=(self.ds["y"] < 4)))
        # verify that indexing with a dataarray works
        b = DataArray("b")
        assert_identical(da[1], da.sel(x=b))
        assert_identical(da[[1]], da.sel(x=slice(b, b)))

    def test_sel_dataarray(self):
        # indexing with DataArray
        self.ds["x"] = ("x", np.array(list("abcdefghij")))
        da = self.ds["foo"]

        ind = DataArray(["a", "b", "c"], dims=["x"])
        actual = da.sel(x=ind)
        assert_identical(actual, da.isel(x=[0, 1, 2]))

        # along new dimension
        ind = DataArray(["a", "b", "c"], dims=["new_dim"])
        actual = da.sel(x=ind)
        assert_array_equal(actual, da.isel(x=[0, 1, 2]))
        assert "new_dim" in actual.dims

        # with coordinate
        ind = DataArray(
            ["a", "b", "c"], dims=["new_dim"], coords={"new_dim": [0, 1, 2]}
        )
        actual = da.sel(x=ind)
        assert_array_equal(actual, da.isel(x=[0, 1, 2]))
        assert "new_dim" in actual.dims
        assert "new_dim" in actual.coords
        assert_equal(actual["new_dim"].drop_vars("x"), ind["new_dim"])

    def test_sel_invalid_slice(self):
        array = DataArray(np.arange(10), [("x", np.arange(10))])
        with pytest.raises(ValueError, match=r"cannot use non-scalar arrays"):
            array.sel(x=slice(array.x))

    def test_sel_dataarray_datetime_slice(self):
        # regression test for GH1240
        times = pd.date_range("2000-01-01", freq="D", periods=365)
        array = DataArray(np.arange(365), [("time", times)])
        result = array.sel(time=slice(array.time[0], array.time[-1]))
        assert_equal(result, array)

        array = DataArray(np.arange(365), [("delta", times - times[0])])
        result = array.sel(delta=slice(array.delta[0], array.delta[-1]))
        assert_equal(result, array)

    def test_sel_float(self):
        data_values = np.arange(4)

        # case coords are float32 and label is list of floats
        float_values = [0.0, 0.111, 0.222, 0.333]
        coord_values = np.asarray(float_values, dtype="float32")
        array = DataArray(data_values, [("float32_coord", coord_values)])
        expected = DataArray(data_values[1:3], [("float32_coord", coord_values[1:3])])
        actual = array.sel(float32_coord=float_values[1:3])
        # case coords are float16 and label is list of floats
        coord_values_16 = np.asarray(float_values, dtype="float16")
        expected_16 = DataArray(
            data_values[1:3], [("float16_coord", coord_values_16[1:3])]
        )
        array_16 = DataArray(data_values, [("float16_coord", coord_values_16)])
        actual_16 = array_16.sel(float16_coord=float_values[1:3])

        # case coord, label are scalars
        expected_scalar = DataArray(
            data_values[2], coords={"float32_coord": coord_values[2]}
        )
        actual_scalar = array.sel(float32_coord=float_values[2])

        assert_equal(expected, actual)
        assert_equal(expected_scalar, actual_scalar)
        assert_equal(expected_16, actual_16)

    def test_sel_no_index(self):
        array = DataArray(np.arange(10), dims="x")
        assert_identical(array[0], array.sel(x=0))
        assert_identical(array[:5], array.sel(x=slice(5)))
        assert_identical(array[[0, -1]], array.sel(x=[0, -1]))
        assert_identical(array[array < 5], array.sel(x=(array < 5)))

    def test_sel_method(self):
        data = DataArray(np.random.randn(3, 4), [("x", [0, 1, 2]), ("y", list("abcd"))])

        expected = data.sel(y=["a", "b"])
        actual = data.sel(y=["ab", "ba"], method="pad")
        assert_identical(expected, actual)

        expected = data.sel(x=[1, 2])
        actual = data.sel(x=[0.9, 1.9], method="backfill", tolerance=1)
        assert_identical(expected, actual)

    def test_sel_drop(self):
        data = DataArray([1, 2, 3], [("x", [0, 1, 2])])
        expected = DataArray(1)
        selected = data.sel(x=0, drop=True)
        assert_identical(expected, selected)

        expected = DataArray(1, {"x": 0})
        selected = data.sel(x=0, drop=False)
        assert_identical(expected, selected)

        data = DataArray([1, 2, 3], dims=["x"])
        expected = DataArray(1)
        selected = data.sel(x=0, drop=True)
        assert_identical(expected, selected)

    def test_isel_drop(self):
        data = DataArray([1, 2, 3], [("x", [0, 1, 2])])
        expected = DataArray(1)
        selected = data.isel(x=0, drop=True)
        assert_identical(expected, selected)

        expected = DataArray(1, {"x": 0})
        selected = data.isel(x=0, drop=False)
        assert_identical(expected, selected)

    def test_head(self):
        assert_equal(self.dv.isel(x=slice(5)), self.dv.head(x=5))
        assert_equal(self.dv.isel(x=slice(0)), self.dv.head(x=0))
        assert_equal(
            self.dv.isel({dim: slice(6) for dim in self.dv.dims}), self.dv.head(6)
        )
        assert_equal(
            self.dv.isel({dim: slice(5) for dim in self.dv.dims}), self.dv.head()
        )
        with pytest.raises(TypeError, match=r"either dict-like or a single int"):
            self.dv.head([3])
        with pytest.raises(TypeError, match=r"expected integer type"):
            self.dv.head(x=3.1)
        with pytest.raises(ValueError, match=r"expected positive int"):
            self.dv.head(-3)

    def test_tail(self):
        assert_equal(self.dv.isel(x=slice(-5, None)), self.dv.tail(x=5))
        assert_equal(self.dv.isel(x=slice(0)), self.dv.tail(x=0))
        assert_equal(
            self.dv.isel({dim: slice(-6, None) for dim in self.dv.dims}),
            self.dv.tail(6),
        )
        assert_equal(
            self.dv.isel({dim: slice(-5, None) for dim in self.dv.dims}), self.dv.tail()
        )
        with pytest.raises(TypeError, match=r"either dict-like or a single int"):
            self.dv.tail([3])
        with pytest.raises(TypeError, match=r"expected integer type"):
            self.dv.tail(x=3.1)
        with pytest.raises(ValueError, match=r"expected positive int"):
            self.dv.tail(-3)

    def test_thin(self):
        assert_equal(self.dv.isel(x=slice(None, None, 5)), self.dv.thin(x=5))
        assert_equal(
            self.dv.isel({dim: slice(None, None, 6) for dim in self.dv.dims}),
            self.dv.thin(6),
        )
        with pytest.raises(TypeError, match=r"either dict-like or a single int"):
            self.dv.thin([3])
        with pytest.raises(TypeError, match=r"expected integer type"):
            self.dv.thin(x=3.1)
        with pytest.raises(ValueError, match=r"expected positive int"):
            self.dv.thin(-3)
        with pytest.raises(ValueError, match=r"cannot be zero"):
            self.dv.thin(time=0)

    def test_loc(self):
        self.ds["x"] = ("x", np.array(list("abcdefghij")))
        da = self.ds["foo"]
        assert_identical(da[:3], da.loc[:"c"])
        assert_identical(da[1], da.loc["b"])
        assert_identical(da[1], da.loc[{"x": "b"}])
        assert_identical(da[1], da.loc["b", ...])
        assert_identical(da[:3], da.loc[["a", "b", "c"]])
        assert_identical(da[:3, :4], da.loc[["a", "b", "c"], np.arange(4)])
        assert_identical(da[:, :4], da.loc[:, self.ds["y"] < 4])

    def test_loc_datetime64_value(self):
        # regression test for https://github.com/pydata/xarray/issues/4283
        t = np.array(["2017-09-05T12", "2017-09-05T15"], dtype="datetime64[ns]")
        array = DataArray(np.ones(t.shape), dims=("time",), coords=(t,))
        assert_identical(array.loc[{"time": t[0]}], array[0])

    def test_loc_assign(self):
        self.ds["x"] = ("x", np.array(list("abcdefghij")))
        da = self.ds["foo"]
        # assignment
        da.loc["a":"j"] = 0
        assert np.all(da.values == 0)
        da.loc[{"x": slice("a", "j")}] = 2
        assert np.all(da.values == 2)

        da.loc[{"x": slice("a", "j")}] = 2
        assert np.all(da.values == 2)

        # Multi dimensional case
        da = DataArray(np.arange(12).reshape(3, 4), dims=["x", "y"])
        da.loc[0, 0] = 0
        assert da.values[0, 0] == 0
        assert da.values[0, 1] != 0

        da = DataArray(np.arange(12).reshape(3, 4), dims=["x", "y"])
        da.loc[0] = 0
        assert np.all(da.values[0] == np.zeros(4))
        assert da.values[1, 0] != 0

    def test_loc_assign_dataarray(self):
        def get_data():
            return DataArray(
                np.ones((4, 3, 2)),
                dims=["x", "y", "z"],
                coords={
                    "x": np.arange(4),
                    "y": ["a", "b", "c"],
                    "non-dim": ("x", [1, 3, 4, 2]),
                },
            )

        da = get_data()
        # indexer with inconsistent coordinates.
        ind = DataArray(np.arange(1, 4), dims=["y"], coords={"y": np.random.randn(3)})
        with pytest.raises(IndexError, match=r"dimension coordinate 'y'"):
            da.loc[dict(x=ind)] = 0

        # indexer with consistent coordinates.
        ind = DataArray(np.arange(1, 4), dims=["x"], coords={"x": np.arange(1, 4)})
        da.loc[dict(x=ind)] = 0  # should not raise
        assert np.allclose(da[dict(x=ind)].values, 0)
        assert_identical(da["x"], get_data()["x"])
        assert_identical(da["non-dim"], get_data()["non-dim"])

        da = get_data()
        # conflict in the assigning values
        value = xr.DataArray(
            np.zeros((3, 3, 2)),
            dims=["x", "y", "z"],
            coords={"x": [0, 1, 2], "non-dim": ("x", [0, 2, 4])},
        )
        with pytest.raises(IndexError, match=r"dimension coordinate 'x'"):
            da.loc[dict(x=ind)] = value

        # consistent coordinate in the assigning values
        value = xr.DataArray(
            np.zeros((3, 3, 2)),
            dims=["x", "y", "z"],
            coords={"x": [1, 2, 3], "non-dim": ("x", [0, 2, 4])},
        )
        da.loc[dict(x=ind)] = value
        assert np.allclose(da[dict(x=ind)].values, 0)
        assert_identical(da["x"], get_data()["x"])
        assert_identical(da["non-dim"], get_data()["non-dim"])

    def test_loc_single_boolean(self):
        data = DataArray([0, 1], coords=[[True, False]])
        assert data.loc[True] == 0
        assert data.loc[False] == 1

    def test_loc_dim_name_collision_with_sel_params(self):
        da = xr.DataArray(
            [[0, 0], [1, 1]],
            dims=["dim1", "method"],
            coords={"dim1": ["x", "y"], "method": ["a", "b"]},
        )
        np.testing.assert_array_equal(
            da.loc[dict(dim1=["x", "y"], method=["a"])], [[0], [1]]
        )

    def test_selection_multiindex(self):
        mindex = pd.MultiIndex.from_product(
            [["a", "b"], [1, 2], [-1, -2]], names=("one", "two", "three")
        )
        mdata = DataArray(range(8), [("x", mindex)])

        def test_sel(lab_indexer, pos_indexer, replaced_idx=False, renamed_dim=None):
            da = mdata.sel(x=lab_indexer)
            expected_da = mdata.isel(x=pos_indexer)
            if not replaced_idx:
                assert_identical(da, expected_da)
            else:
                if renamed_dim:
                    assert da.dims[0] == renamed_dim
                    da = da.rename({renamed_dim: "x"})
                assert_identical(da.variable, expected_da.variable)
                assert not da["x"].equals(expected_da["x"])

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

        assert_identical(mdata.loc["a"], mdata.sel(x="a"))
        assert_identical(mdata.loc[("a", 1), ...], mdata.sel(x=("a", 1)))
        assert_identical(mdata.loc[{"one": "a"}, ...], mdata.sel(x={"one": "a"}))
        with pytest.raises(IndexError):
            mdata.loc[("a", 1)]

        assert_identical(mdata.sel(x={"one": "a", "two": 1}), mdata.sel(one="a", two=1))

    def test_selection_multiindex_remove_unused(self):
        # GH2619. For MultiIndex, we need to call remove_unused.
        ds = xr.DataArray(
            np.arange(40).reshape(8, 5),
            dims=["x", "y"],
            coords={"x": np.arange(8), "y": np.arange(5)},
        )
        ds = ds.stack(xy=["x", "y"])
        ds_isel = ds.isel(xy=ds["x"] < 4)
        with pytest.raises(KeyError):
            ds_isel.sel(x=5)

        actual = ds_isel.unstack()
        expected = ds.reset_index("xy").isel(xy=ds["x"] < 4)
        expected = expected.set_index(xy=["x", "y"]).unstack()
        assert_identical(expected, actual)

    def test_selection_multiindex_from_level(self):
        # GH: 3512
        da = DataArray([0, 1], dims=["x"], coords={"x": [0, 1], "y": "a"})
        db = DataArray([2, 3], dims=["x"], coords={"x": [0, 1], "y": "b"})
        data = xr.concat([da, db], dim="x").set_index(xy=["x", "y"])
        assert data.dims == ("xy",)
        actual = data.sel(y="a")
        expected = data.isel(xy=[0, 1]).unstack("xy").squeeze("y").drop_vars("y")
        assert_equal(actual, expected)

    def test_stack_groupby_unsorted_coord(self):
        data = [[0, 1], [2, 3]]
        data_flat = [0, 1, 2, 3]
        dims = ["x", "y"]
        y_vals = [2, 3]

        arr = xr.DataArray(data, dims=dims, coords={"y": y_vals})
        actual1 = arr.stack(z=dims).groupby("z").first()
        midx1 = pd.MultiIndex.from_product([[0, 1], [2, 3]], names=dims)
        expected1 = xr.DataArray(data_flat, dims=["z"], coords={"z": midx1})
        xr.testing.assert_equal(actual1, expected1)

        # GH: 3287.  Note that y coord values are not in sorted order.
        arr = xr.DataArray(data, dims=dims, coords={"y": y_vals[::-1]})
        actual2 = arr.stack(z=dims).groupby("z").first()
        midx2 = pd.MultiIndex.from_product([[0, 1], [3, 2]], names=dims)
        expected2 = xr.DataArray(data_flat, dims=["z"], coords={"z": midx2})
        xr.testing.assert_equal(actual2, expected2)

    def test_virtual_default_coords(self):
        array = DataArray(np.zeros((5,)), dims="x")
        expected = DataArray(range(5), dims="x", name="x")
        assert_identical(expected, array["x"])
        assert_identical(expected, array.coords["x"])

    def test_virtual_time_components(self):
        dates = pd.date_range("2000-01-01", periods=10)
        da = DataArray(np.arange(1, 11), [("time", dates)])

        assert_array_equal(da["time.dayofyear"], da.values)
        assert_array_equal(da.coords["time.dayofyear"], da.values)

    def test_coords(self):
        # use int64 to ensure repr() consistency on windows
        coords = [
            IndexVariable("x", np.array([-1, -2], "int64")),
            IndexVariable("y", np.array([0, 1, 2], "int64")),
        ]
        da = DataArray(np.random.randn(2, 3), coords, name="foo")

        assert 2 == len(da.coords)

        assert ["x", "y"] == list(da.coords)

        assert coords[0].identical(da.coords["x"])
        assert coords[1].identical(da.coords["y"])

        assert "x" in da.coords
        assert 0 not in da.coords
        assert "foo" not in da.coords

        with pytest.raises(KeyError):
            da.coords[0]
        with pytest.raises(KeyError):
            da.coords["foo"]

        expected = dedent(
            """\
        Coordinates:
          * x        (x) int64 -1 -2
          * y        (y) int64 0 1 2"""
        )
        actual = repr(da.coords)
        assert expected == actual

        del da.coords["x"]
        da._indexes = propagate_indexes(da._indexes, exclude="x")
        expected = DataArray(da.values, {"y": [0, 1, 2]}, dims=["x", "y"], name="foo")
        assert_identical(da, expected)

        with pytest.raises(ValueError, match=r"conflicting MultiIndex"):
            self.mda["level_1"] = np.arange(4)
            self.mda.coords["level_1"] = np.arange(4)

    def test_coords_to_index(self):
        da = DataArray(np.zeros((2, 3)), [("x", [1, 2]), ("y", list("abc"))])

        with pytest.raises(ValueError, match=r"no valid index"):
            da[0, 0].coords.to_index()

        expected = pd.Index(["a", "b", "c"], name="y")
        actual = da[0].coords.to_index()
        assert expected.equals(actual)

        expected = pd.MultiIndex.from_product(
            [[1, 2], ["a", "b", "c"]], names=["x", "y"]
        )
        actual = da.coords.to_index()
        assert expected.equals(actual)

        expected = pd.MultiIndex.from_product(
            [["a", "b", "c"], [1, 2]], names=["y", "x"]
        )
        actual = da.coords.to_index(["y", "x"])
        assert expected.equals(actual)

        with pytest.raises(ValueError, match=r"ordered_dims must match"):
            da.coords.to_index(["x"])

    def test_coord_coords(self):
        orig = DataArray(
            [10, 20], {"x": [1, 2], "x2": ("x", ["a", "b"]), "z": 4}, dims="x"
        )

        actual = orig.coords["x"]
        expected = DataArray(
            [1, 2], {"z": 4, "x2": ("x", ["a", "b"]), "x": [1, 2]}, dims="x", name="x"
        )
        assert_identical(expected, actual)

        del actual.coords["x2"]
        assert_identical(expected.reset_coords("x2", drop=True), actual)

        actual.coords["x3"] = ("x", ["a", "b"])
        expected = DataArray(
            [1, 2], {"z": 4, "x3": ("x", ["a", "b"]), "x": [1, 2]}, dims="x", name="x"
        )
        assert_identical(expected, actual)

    def test_reset_coords(self):
        data = DataArray(
            np.zeros((3, 4)),
            {"bar": ("x", ["a", "b", "c"]), "baz": ("y", range(4)), "y": range(4)},
            dims=["x", "y"],
            name="foo",
        )

        actual = data.reset_coords()
        expected = Dataset(
            {
                "foo": (["x", "y"], np.zeros((3, 4))),
                "bar": ("x", ["a", "b", "c"]),
                "baz": ("y", range(4)),
                "y": range(4),
            }
        )
        assert_identical(actual, expected)

        actual = data.reset_coords(["bar", "baz"])
        assert_identical(actual, expected)

        actual = data.reset_coords("bar")
        expected = Dataset(
            {"foo": (["x", "y"], np.zeros((3, 4))), "bar": ("x", ["a", "b", "c"])},
            {"baz": ("y", range(4)), "y": range(4)},
        )
        assert_identical(actual, expected)

        actual = data.reset_coords(["bar"])
        assert_identical(actual, expected)

        actual = data.reset_coords(drop=True)
        expected = DataArray(
            np.zeros((3, 4)), coords={"y": range(4)}, dims=["x", "y"], name="foo"
        )
        assert_identical(actual, expected)

        actual = data.copy()
        actual = actual.reset_coords(drop=True)
        assert_identical(actual, expected)

        actual = data.reset_coords("bar", drop=True)
        expected = DataArray(
            np.zeros((3, 4)),
            {"baz": ("y", range(4)), "y": range(4)},
            dims=["x", "y"],
            name="foo",
        )
        assert_identical(actual, expected)

        with pytest.raises(ValueError, match=r"cannot be found"):
            data.reset_coords("foo", drop=True)
        with pytest.raises(ValueError, match=r"cannot be found"):
            data.reset_coords("not_found")
        with pytest.raises(ValueError, match=r"cannot remove index"):
            data.reset_coords("y")

    def test_assign_coords(self):
        array = DataArray(10)
        actual = array.assign_coords(c=42)
        expected = DataArray(10, {"c": 42})
        assert_identical(actual, expected)

        array = DataArray([1, 2, 3, 4], {"c": ("x", [0, 0, 1, 1])}, dims="x")
        actual = array.groupby("c").assign_coords(d=lambda a: a.mean())
        expected = array.copy()
        expected.coords["d"] = ("x", [1.5, 1.5, 3.5, 3.5])
        assert_identical(actual, expected)

        with pytest.raises(ValueError, match=r"conflicting MultiIndex"):
            self.mda.assign_coords(level_1=range(4))

        # GH: 2112
        da = xr.DataArray([0, 1, 2], dims="x")
        with pytest.raises(ValueError):
            da["x"] = [0, 1, 2, 3]  # size conflict
        with pytest.raises(ValueError):
            da.coords["x"] = [0, 1, 2, 3]  # size conflict

    def test_coords_alignment(self):
        lhs = DataArray([1, 2, 3], [("x", [0, 1, 2])])
        rhs = DataArray([2, 3, 4], [("x", [1, 2, 3])])
        lhs.coords["rhs"] = rhs

        expected = DataArray(
            [1, 2, 3], coords={"rhs": ("x", [np.nan, 2, 3]), "x": [0, 1, 2]}, dims="x"
        )
        assert_identical(lhs, expected)

    def test_set_coords_update_index(self):
        actual = DataArray([1, 2, 3], [("x", [1, 2, 3])])
        actual.coords["x"] = ["a", "b", "c"]
        assert actual.xindexes["x"].equals(pd.Index(["a", "b", "c"]))

    def test_coords_replacement_alignment(self):
        # regression test for GH725
        arr = DataArray([0, 1, 2], dims=["abc"])
        new_coord = DataArray([1, 2, 3], dims=["abc"], coords=[[1, 2, 3]])
        arr["abc"] = new_coord
        expected = DataArray([0, 1, 2], coords=[("abc", [1, 2, 3])])
        assert_identical(arr, expected)

    def test_coords_non_string(self):
        arr = DataArray(0, coords={1: 2})
        actual = arr.coords[1]
        expected = DataArray(2, coords={1: 2}, name=1)
        assert_identical(actual, expected)

    def test_coords_delitem_delete_indexes(self):
        # regression test for GH3746
        arr = DataArray(np.ones((2,)), dims="x", coords={"x": [0, 1]})
        del arr.coords["x"]
        assert "x" not in arr.xindexes

    def test_broadcast_like(self):
        arr1 = DataArray(
            np.ones((2, 3)),
            dims=["x", "y"],
            coords={"x": ["a", "b"], "y": ["a", "b", "c"]},
        )
        arr2 = DataArray(
            np.ones((3, 2)),
            dims=["x", "y"],
            coords={"x": ["a", "b", "c"], "y": ["a", "b"]},
        )
        orig1, orig2 = broadcast(arr1, arr2)
        new1 = arr1.broadcast_like(arr2)
        new2 = arr2.broadcast_like(arr1)

        assert_identical(orig1, new1)
        assert_identical(orig2, new2)

        orig3 = DataArray(np.random.randn(5), [("x", range(5))])
        orig4 = DataArray(np.random.randn(6), [("y", range(6))])
        new3, new4 = broadcast(orig3, orig4)

        assert_identical(orig3.broadcast_like(orig4), new3.transpose("y", "x"))
        assert_identical(orig4.broadcast_like(orig3), new4)

    def test_reindex_like(self):
        foo = DataArray(np.random.randn(5, 6), [("x", range(5)), ("y", range(6))])
        bar = foo[:2, :2]
        assert_identical(foo.reindex_like(bar), bar)

        expected = foo.copy()
        expected[:] = np.nan
        expected[:2, :2] = bar
        assert_identical(bar.reindex_like(foo), expected)

    def test_reindex_like_no_index(self):
        foo = DataArray(np.random.randn(5, 6), dims=["x", "y"])
        assert_identical(foo, foo.reindex_like(foo))

        bar = foo[:4]
        with pytest.raises(ValueError, match=r"different size for unlabeled"):
            foo.reindex_like(bar)

    def test_reindex_regressions(self):
        da = DataArray(np.random.randn(5), coords=[("time", range(5))])
        time2 = DataArray(np.arange(5), dims="time2")
        with pytest.raises(ValueError):
            da.reindex(time=time2)

        # regression test for #736, reindex can not change complex nums dtype
        x = np.array([1, 2, 3], dtype=complex)
        x = DataArray(x, coords=[[0.1, 0.2, 0.3]])
        y = DataArray([2, 5, 6, 7, 8], coords=[[-1.1, 0.21, 0.31, 0.41, 0.51]])
        re_dtype = x.reindex_like(y, method="pad").dtype
        assert x.dtype == re_dtype

    def test_reindex_method(self):
        x = DataArray([10, 20], dims="y", coords={"y": [0, 1]})
        y = [-0.1, 0.5, 1.1]
        actual = x.reindex(y=y, method="backfill", tolerance=0.2)
        expected = DataArray([10, np.nan, np.nan], coords=[("y", y)])
        assert_identical(expected, actual)

        alt = Dataset({"y": y})
        actual = x.reindex_like(alt, method="backfill")
        expected = DataArray([10, 20, np.nan], coords=[("y", y)])
        assert_identical(expected, actual)

    @pytest.mark.parametrize("fill_value", [dtypes.NA, 2, 2.0, {None: 2, "u": 1}])
    def test_reindex_fill_value(self, fill_value):
        x = DataArray([10, 20], dims="y", coords={"y": [0, 1], "u": ("y", [1, 2])})
        y = [0, 1, 2]
        if fill_value == dtypes.NA:
            # if we supply the default, we expect the missing value for a
            # float array
            fill_value_var = fill_value_u = np.nan
        elif isinstance(fill_value, dict):
            fill_value_var = fill_value[None]
            fill_value_u = fill_value["u"]
        else:
            fill_value_var = fill_value_u = fill_value
        actual = x.reindex(y=y, fill_value=fill_value)
        expected = DataArray(
            [10, 20, fill_value_var],
            dims="y",
            coords={"y": y, "u": ("y", [1, 2, fill_value_u])},
        )
        assert_identical(expected, actual)

    @pytest.mark.parametrize("dtype", [str, bytes])
    def test_reindex_str_dtype(self, dtype):

        data = DataArray(
            [1, 2], dims="x", coords={"x": np.array(["a", "b"], dtype=dtype)}
        )

        actual = data.reindex(x=data.x)
        expected = data

        assert_identical(expected, actual)
        assert actual.dtype == expected.dtype

    def test_rename(self):
        renamed = self.dv.rename("bar")
        assert_identical(renamed.to_dataset(), self.ds.rename({"foo": "bar"}))
        assert renamed.name == "bar"

        renamed = self.dv.x.rename({"x": "z"}).rename("z")
        assert_identical(renamed, self.ds.rename({"x": "z"}).z)
        assert renamed.name == "z"
        assert renamed.dims == ("z",)

        renamed_kwargs = self.dv.x.rename(x="z").rename("z")
        assert_identical(renamed, renamed_kwargs)

    def test_init_value(self):
        expected = DataArray(
            np.full((3, 4), 3), dims=["x", "y"], coords=[range(3), range(4)]
        )
        actual = DataArray(3, dims=["x", "y"], coords=[range(3), range(4)])
        assert_identical(expected, actual)

        expected = DataArray(
            np.full((1, 10, 2), 0),
            dims=["w", "x", "y"],
            coords={"x": np.arange(10), "y": ["north", "south"]},
        )
        actual = DataArray(0, dims=expected.dims, coords=expected.coords)
        assert_identical(expected, actual)

        expected = DataArray(
            np.full((10, 2), np.nan), coords=[("x", np.arange(10)), ("y", ["a", "b"])]
        )
        actual = DataArray(coords=[("x", np.arange(10)), ("y", ["a", "b"])])
        assert_identical(expected, actual)

        with pytest.raises(ValueError, match=r"different number of dim"):
            DataArray(np.array(1), coords={"x": np.arange(10)}, dims=["x"])
        with pytest.raises(ValueError, match=r"does not match the 0 dim"):
            DataArray(np.array(1), coords=[("x", np.arange(10))])

    def test_swap_dims(self):
        array = DataArray(np.random.randn(3), {"y": ("x", list("abc"))}, "x")
        expected = DataArray(array.values, {"y": list("abc")}, dims="y")
        actual = array.swap_dims({"x": "y"})
        assert_identical(expected, actual)
        for dim_name in set().union(expected.xindexes.keys(), actual.xindexes.keys()):
            pd.testing.assert_index_equal(
                expected.xindexes[dim_name].array, actual.xindexes[dim_name].array
            )

        array = DataArray(np.random.randn(3), {"x": list("abc")}, "x")
        expected = DataArray(array.values, {"x": ("y", list("abc"))}, dims="y")
        actual = array.swap_dims({"x": "y"})
        assert_identical(expected, actual)
        for dim_name in set().union(expected.xindexes.keys(), actual.xindexes.keys()):
            pd.testing.assert_index_equal(
                expected.xindexes[dim_name].to_pandas_index(),
                actual.xindexes[dim_name].to_pandas_index(),
            )

        # as kwargs
        array = DataArray(np.random.randn(3), {"x": list("abc")}, "x")
        expected = DataArray(array.values, {"x": ("y", list("abc"))}, dims="y")
        actual = array.swap_dims(x="y")
        assert_identical(expected, actual)
        for dim_name in set().union(expected.xindexes.keys(), actual.xindexes.keys()):
            pd.testing.assert_index_equal(
                expected.xindexes[dim_name].to_pandas_index(),
                actual.xindexes[dim_name].to_pandas_index(),
            )

        # multiindex case
        idx = pd.MultiIndex.from_arrays([list("aab"), list("yzz")], names=["y1", "y2"])
        array = DataArray(np.random.randn(3), {"y": ("x", idx)}, "x")
        expected = DataArray(array.values, {"y": idx}, "y")
        actual = array.swap_dims({"x": "y"})
        assert_identical(expected, actual)
        for dim_name in set().union(expected.xindexes.keys(), actual.xindexes.keys()):
            pd.testing.assert_index_equal(
                expected.xindexes[dim_name].to_pandas_index(),
                actual.xindexes[dim_name].to_pandas_index(),
            )

    def test_expand_dims_error(self):
        array = DataArray(
            np.random.randn(3, 4),
            dims=["x", "dim_0"],
            coords={"x": np.linspace(0.0, 1.0, 3)},
            attrs={"key": "entry"},
        )

        with pytest.raises(TypeError, match=r"dim should be hashable or"):
            array.expand_dims(0)
        with pytest.raises(ValueError, match=r"lengths of dim and axis"):
            # dims and axis argument should be the same length
            array.expand_dims(dim=["a", "b"], axis=[1, 2, 3])
        with pytest.raises(ValueError, match=r"Dimension x already"):
            # Should not pass the already existing dimension.
            array.expand_dims(dim=["x"])
        # raise if duplicate
        with pytest.raises(ValueError, match=r"duplicate values"):
            array.expand_dims(dim=["y", "y"])
        with pytest.raises(ValueError, match=r"duplicate values"):
            array.expand_dims(dim=["y", "z"], axis=[1, 1])
        with pytest.raises(ValueError, match=r"duplicate values"):
            array.expand_dims(dim=["y", "z"], axis=[2, -2])

        # out of bounds error, axis must be in [-4, 3]
        with pytest.raises(IndexError):
            array.expand_dims(dim=["y", "z"], axis=[2, 4])
        with pytest.raises(IndexError):
            array.expand_dims(dim=["y", "z"], axis=[2, -5])
        # Does not raise an IndexError
        array.expand_dims(dim=["y", "z"], axis=[2, -4])
        array.expand_dims(dim=["y", "z"], axis=[2, 3])

        array = DataArray(
            np.random.randn(3, 4),
            dims=["x", "dim_0"],
            coords={"x": np.linspace(0.0, 1.0, 3)},
            attrs={"key": "entry"},
        )
        with pytest.raises(TypeError):
            array.expand_dims({"new_dim": 3.2})

        # Attempt to use both dim and kwargs
        with pytest.raises(ValueError):
            array.expand_dims({"d": 4}, e=4)

    def test_expand_dims(self):
        array = DataArray(
            np.random.randn(3, 4),
            dims=["x", "dim_0"],
            coords={"x": np.linspace(0.0, 1.0, 3)},
            attrs={"key": "entry"},
        )
        # pass only dim label
        actual = array.expand_dims(dim="y")
        expected = DataArray(
            np.expand_dims(array.values, 0),
            dims=["y", "x", "dim_0"],
            coords={"x": np.linspace(0.0, 1.0, 3)},
            attrs={"key": "entry"},
        )
        assert_identical(expected, actual)
        roundtripped = actual.squeeze("y", drop=True)
        assert_identical(array, roundtripped)

        # pass multiple dims
        actual = array.expand_dims(dim=["y", "z"])
        expected = DataArray(
            np.expand_dims(np.expand_dims(array.values, 0), 0),
            dims=["y", "z", "x", "dim_0"],
            coords={"x": np.linspace(0.0, 1.0, 3)},
            attrs={"key": "entry"},
        )
        assert_identical(expected, actual)
        roundtripped = actual.squeeze(["y", "z"], drop=True)
        assert_identical(array, roundtripped)

        # pass multiple dims and axis. Axis is out of order
        actual = array.expand_dims(dim=["z", "y"], axis=[2, 1])
        expected = DataArray(
            np.expand_dims(np.expand_dims(array.values, 1), 2),
            dims=["x", "y", "z", "dim_0"],
            coords={"x": np.linspace(0.0, 1.0, 3)},
            attrs={"key": "entry"},
        )
        assert_identical(expected, actual)
        # make sure the attrs are tracked
        assert actual.attrs["key"] == "entry"
        roundtripped = actual.squeeze(["z", "y"], drop=True)
        assert_identical(array, roundtripped)

        # Negative axis and they are out of order
        actual = array.expand_dims(dim=["y", "z"], axis=[-1, -2])
        expected = DataArray(
            np.expand_dims(np.expand_dims(array.values, -1), -1),
            dims=["x", "dim_0", "z", "y"],
            coords={"x": np.linspace(0.0, 1.0, 3)},
            attrs={"key": "entry"},
        )
        assert_identical(expected, actual)
        assert actual.attrs["key"] == "entry"
        roundtripped = actual.squeeze(["y", "z"], drop=True)
        assert_identical(array, roundtripped)

    def test_expand_dims_with_scalar_coordinate(self):
        array = DataArray(
            np.random.randn(3, 4),
            dims=["x", "dim_0"],
            coords={"x": np.linspace(0.0, 1.0, 3), "z": 1.0},
            attrs={"key": "entry"},
        )
        actual = array.expand_dims(dim="z")
        expected = DataArray(
            np.expand_dims(array.values, 0),
            dims=["z", "x", "dim_0"],
            coords={"x": np.linspace(0.0, 1.0, 3), "z": np.ones(1)},
            attrs={"key": "entry"},
        )
        assert_identical(expected, actual)
        roundtripped = actual.squeeze(["z"], drop=False)
        assert_identical(array, roundtripped)

    def test_expand_dims_with_greater_dim_size(self):
        array = DataArray(
            np.random.randn(3, 4),
            dims=["x", "dim_0"],
            coords={"x": np.linspace(0.0, 1.0, 3), "z": 1.0},
            attrs={"key": "entry"},
        )
        actual = array.expand_dims({"y": 2, "z": 1, "dim_1": ["a", "b", "c"]})

        expected_coords = {
            "y": [0, 1],
            "z": [1.0],
            "dim_1": ["a", "b", "c"],
            "x": np.linspace(0, 1, 3),
            "dim_0": range(4),
        }
        expected = DataArray(
            array.values * np.ones([2, 1, 3, 3, 4]),
            coords=expected_coords,
            dims=list(expected_coords.keys()),
            attrs={"key": "entry"},
        ).drop_vars(["y", "dim_0"])
        assert_identical(expected, actual)

        # Test with kwargs instead of passing dict to dim arg.

        other_way = array.expand_dims(dim_1=["a", "b", "c"])

        other_way_expected = DataArray(
            array.values * np.ones([3, 3, 4]),
            coords={
                "dim_1": ["a", "b", "c"],
                "x": np.linspace(0, 1, 3),
                "dim_0": range(4),
                "z": 1.0,
            },
            dims=["dim_1", "x", "dim_0"],
            attrs={"key": "entry"},
        ).drop_vars("dim_0")
        assert_identical(other_way_expected, other_way)

    def test_set_index(self):
        indexes = [self.mindex.get_level_values(n) for n in self.mindex.names]
        coords = {idx.name: ("x", idx) for idx in indexes}
        array = DataArray(self.mda.values, coords=coords, dims="x")
        expected = self.mda.copy()
        level_3 = ("x", [1, 2, 3, 4])
        array["level_3"] = level_3
        expected["level_3"] = level_3

        obj = array.set_index(x=self.mindex.names)
        assert_identical(obj, expected)

        obj = obj.set_index(x="level_3", append=True)
        expected = array.set_index(x=["level_1", "level_2", "level_3"])
        assert_identical(obj, expected)

        array = array.set_index(x=["level_1", "level_2", "level_3"])
        assert_identical(array, expected)

        array2d = DataArray(
            np.random.rand(2, 2),
            coords={"x": ("x", [0, 1]), "level": ("y", [1, 2])},
            dims=("x", "y"),
        )
        with pytest.raises(ValueError, match=r"dimension mismatch"):
            array2d.set_index(x="level")

        # Issue 3176: Ensure clear error message on key error.
        with pytest.raises(ValueError) as excinfo:
            obj.set_index(x="level_4")
        assert str(excinfo.value) == "level_4 is not the name of an existing variable."

    def test_reset_index(self):
        indexes = [self.mindex.get_level_values(n) for n in self.mindex.names]
        coords = {idx.name: ("x", idx) for idx in indexes}
        expected = DataArray(self.mda.values, coords=coords, dims="x")

        obj = self.mda.reset_index("x")
        assert_identical(obj, expected)
        obj = self.mda.reset_index(self.mindex.names)
        assert_identical(obj, expected)
        obj = self.mda.reset_index(["x", "level_1"])
        assert_identical(obj, expected)

        coords = {
            "x": ("x", self.mindex.droplevel("level_1")),
            "level_1": ("x", self.mindex.get_level_values("level_1")),
        }
        expected = DataArray(self.mda.values, coords=coords, dims="x")
        obj = self.mda.reset_index(["level_1"])
        assert_identical(obj, expected)

        expected = DataArray(self.mda.values, dims="x")
        obj = self.mda.reset_index("x", drop=True)
        assert_identical(obj, expected)

        array = self.mda.copy()
        array = array.reset_index(["x"], drop=True)
        assert_identical(array, expected)

        # single index
        array = DataArray([1, 2], coords={"x": ["a", "b"]}, dims="x")
        expected = DataArray([1, 2], coords={"x_": ("x", ["a", "b"])}, dims="x")
        assert_identical(array.reset_index("x"), expected)

    def test_reset_index_keep_attrs(self):
        coord_1 = DataArray([1, 2], dims=["coord_1"], attrs={"attrs": True})
        da = DataArray([1, 0], [coord_1])
        expected = DataArray([1, 0], {"coord_1_": coord_1}, dims=["coord_1"])
        obj = da.reset_index("coord_1")
        assert_identical(expected, obj)

    def test_reorder_levels(self):
        midx = self.mindex.reorder_levels(["level_2", "level_1"])
        expected = DataArray(self.mda.values, coords={"x": midx}, dims="x")

        obj = self.mda.reorder_levels(x=["level_2", "level_1"])
        assert_identical(obj, expected)

        array = DataArray([1, 2], dims="x")
        with pytest.raises(KeyError):
            array.reorder_levels(x=["level_1", "level_2"])

        array["x"] = [0, 1]
        with pytest.raises(ValueError, match=r"has no MultiIndex"):
            array.reorder_levels(x=["level_1", "level_2"])

    def test_dataset_getitem(self):
        dv = self.ds["foo"]
        assert_identical(dv, self.dv)

    def test_array_interface(self):
        assert_array_equal(np.asarray(self.dv), self.x)
        # test patched in methods
        assert_array_equal(self.dv.astype(float), self.v.astype(float))
        assert_array_equal(self.dv.argsort(), self.v.argsort())
        assert_array_equal(self.dv.clip(2, 3), self.v.clip(2, 3))
        # test ufuncs
        expected = deepcopy(self.ds)
        expected["foo"][:] = np.sin(self.x)
        assert_equal(expected["foo"], np.sin(self.dv))
        assert_array_equal(self.dv, np.maximum(self.v, self.dv))
        bar = Variable(["x", "y"], np.zeros((10, 20)))
        assert_equal(self.dv, np.maximum(self.dv, bar))

    def test_astype_attrs(self):
        for v in [self.va.copy(), self.mda.copy(), self.ds.copy()]:
            v.attrs["foo"] = "bar"
            assert v.attrs == v.astype(float).attrs
            assert not v.astype(float, keep_attrs=False).attrs

    def test_astype_dtype(self):
        original = DataArray([-1, 1, 2, 3, 1000])
        converted = original.astype(float)
        assert_array_equal(original, converted)
        assert np.issubdtype(original.dtype, np.integer)
        assert np.issubdtype(converted.dtype, np.floating)

    def test_astype_order(self):
        original = DataArray([[1, 2], [3, 4]])
        converted = original.astype("d", order="F")
        assert_equal(original, converted)
        assert original.values.flags["C_CONTIGUOUS"]
        assert converted.values.flags["F_CONTIGUOUS"]

    def test_astype_subok(self):
        class NdArraySubclass(np.ndarray):
            pass

        original = DataArray(NdArraySubclass(np.arange(3)))
        converted_not_subok = original.astype("d", subok=False)
        converted_subok = original.astype("d", subok=True)
        if not isinstance(original.data, NdArraySubclass):
            pytest.xfail("DataArray cannot be backed yet by a subclasses of np.ndarray")
        assert isinstance(converted_not_subok.data, np.ndarray)
        assert not isinstance(converted_not_subok.data, NdArraySubclass)
        assert isinstance(converted_subok.data, NdArraySubclass)

    def test_is_null(self):
        x = np.random.RandomState(42).randn(5, 6)
        x[x < 0] = np.nan
        original = DataArray(x, [-np.arange(5), np.arange(6)], ["x", "y"])
        expected = DataArray(pd.isnull(x), [-np.arange(5), np.arange(6)], ["x", "y"])
        assert_identical(expected, original.isnull())
        assert_identical(~expected, original.notnull())

    def test_math(self):
        x = self.x
        v = self.v
        a = self.dv
        # variable math was already tested extensively, so let's just make sure
        # that all types are properly converted here
        assert_equal(a, +a)
        assert_equal(a, a + 0)
        assert_equal(a, 0 + a)
        assert_equal(a, a + 0 * v)
        assert_equal(a, 0 * v + a)
        assert_equal(a, a + 0 * x)
        assert_equal(a, 0 * x + a)
        assert_equal(a, a + 0 * a)
        assert_equal(a, 0 * a + a)

    def test_math_automatic_alignment(self):
        a = DataArray(range(5), [("x", range(5))])
        b = DataArray(range(5), [("x", range(1, 6))])
        expected = DataArray(np.ones(4), [("x", [1, 2, 3, 4])])
        assert_identical(a - b, expected)

    def test_non_overlapping_dataarrays_return_empty_result(self):

        a = DataArray(range(5), [("x", range(5))])
        result = a.isel(x=slice(2)) + a.isel(x=slice(2, None))
        assert len(result["x"]) == 0

    def test_empty_dataarrays_return_empty_result(self):

        a = DataArray(data=[])
        result = a * a
        assert len(result["dim_0"]) == 0

    def test_inplace_math_basics(self):
        x = self.x
        a = self.dv
        v = a.variable
        b = a
        b += 1
        assert b is a
        assert b.variable is v
        assert_array_equal(b.values, x)
        assert source_ndarray(b.values) is x

    def test_inplace_math_automatic_alignment(self):
        a = DataArray(range(5), [("x", range(5))])
        b = DataArray(range(1, 6), [("x", range(1, 6))])
        with pytest.raises(xr.MergeError, match="Automatic alignment is not supported"):
            a += b
        with pytest.raises(xr.MergeError, match="Automatic alignment is not supported"):
            b += a

    def test_math_name(self):
        # Verify that name is preserved only when it can be done unambiguously.
        # The rule (copied from pandas.Series) is keep the current name only if
        # the other object has the same name or no name attribute and this
        # object isn't a coordinate; otherwise reset to None.
        a = self.dv
        assert (+a).name == "foo"
        assert (a + 0).name == "foo"
        assert (a + a.rename(None)).name is None
        assert (a + a.rename("bar")).name is None
        assert (a + a).name == "foo"
        assert (+a["x"]).name == "x"
        assert (a["x"] + 0).name == "x"
        assert (a + a["x"]).name is None

    def test_math_with_coords(self):
        coords = {
            "x": [-1, -2],
            "y": ["ab", "cd", "ef"],
            "lat": (["x", "y"], [[1, 2, 3], [-1, -2, -3]]),
            "c": -999,
        }
        orig = DataArray(np.random.randn(2, 3), coords, dims=["x", "y"])

        actual = orig + 1
        expected = DataArray(orig.values + 1, orig.coords)
        assert_identical(expected, actual)

        actual = 1 + orig
        assert_identical(expected, actual)

        actual = orig + orig[0, 0]
        exp_coords = {k: v for k, v in coords.items() if k != "lat"}
        expected = DataArray(
            orig.values + orig.values[0, 0], exp_coords, dims=["x", "y"]
        )
        assert_identical(expected, actual)

        actual = orig[0, 0] + orig
        assert_identical(expected, actual)

        actual = orig[0, 0] + orig[-1, -1]
        expected = DataArray(orig.values[0, 0] + orig.values[-1, -1], {"c": -999})
        assert_identical(expected, actual)

        actual = orig[:, 0] + orig[0, :]
        exp_values = orig[:, 0].values[:, None] + orig[0, :].values[None, :]
        expected = DataArray(exp_values, exp_coords, dims=["x", "y"])
        assert_identical(expected, actual)

        actual = orig[0, :] + orig[:, 0]
        assert_identical(expected.transpose(transpose_coords=True), actual)

        actual = orig - orig.transpose(transpose_coords=True)
        expected = DataArray(np.zeros((2, 3)), orig.coords)
        assert_identical(expected, actual)

        actual = orig.transpose(transpose_coords=True) - orig
        assert_identical(expected.transpose(transpose_coords=True), actual)

        alt = DataArray([1, 1], {"x": [-1, -2], "c": "foo", "d": 555}, "x")
        actual = orig + alt
        expected = orig + 1
        expected.coords["d"] = 555
        del expected.coords["c"]
        assert_identical(expected, actual)

        actual = alt + orig
        assert_identical(expected, actual)

    def test_index_math(self):
        orig = DataArray(range(3), dims="x", name="x")
        actual = orig + 1
        expected = DataArray(1 + np.arange(3), dims="x", name="x")
        assert_identical(expected, actual)

        # regression tests for #254
        actual = orig[0] < orig
        expected = DataArray([False, True, True], dims="x", name="x")
        assert_identical(expected, actual)

        actual = orig > orig[0]
        assert_identical(expected, actual)

    def test_dataset_math(self):
        # more comprehensive tests with multiple dataset variables
        obs = Dataset(
            {"tmin": ("x", np.arange(5)), "tmax": ("x", 10 + np.arange(5))},
            {"x": ("x", 0.5 * np.arange(5)), "loc": ("x", range(-2, 3))},
        )

        actual = 2 * obs["tmax"]
        expected = DataArray(2 * (10 + np.arange(5)), obs.coords, name="tmax")
        assert_identical(actual, expected)

        actual = obs["tmax"] - obs["tmin"]
        expected = DataArray(10 * np.ones(5), obs.coords)
        assert_identical(actual, expected)

        sim = Dataset(
            {
                "tmin": ("x", 1 + np.arange(5)),
                "tmax": ("x", 11 + np.arange(5)),
                # does *not* include 'loc' as a coordinate
                "x": ("x", 0.5 * np.arange(5)),
            }
        )

        actual = sim["tmin"] - obs["tmin"]
        expected = DataArray(np.ones(5), obs.coords, name="tmin")
        assert_identical(actual, expected)

        actual = -obs["tmin"] + sim["tmin"]
        assert_identical(actual, expected)

        actual = sim["tmin"].copy()
        actual -= obs["tmin"]
        assert_identical(actual, expected)

        actual = sim.copy()
        actual["tmin"] = sim["tmin"] - obs["tmin"]
        expected = Dataset(
            {"tmin": ("x", np.ones(5)), "tmax": ("x", sim["tmax"].values)}, obs.coords
        )
        assert_identical(actual, expected)

        actual = sim.copy()
        actual["tmin"] -= obs["tmin"]
        assert_identical(actual, expected)

    def test_stack_unstack(self):
        orig = DataArray([[0, 1], [2, 3]], dims=["x", "y"], attrs={"foo": 2})
        assert_identical(orig, orig.unstack())

        # test GH3000
        a = orig[:0, :1].stack(dim=("x", "y")).dim.to_index()
        if pd.__version__ < "0.24.0":
            b = pd.MultiIndex(
                levels=[pd.Int64Index([]), pd.Int64Index([0])],
                labels=[[], []],
                names=["x", "y"],
            )
        else:
            b = pd.MultiIndex(
                levels=[pd.Int64Index([]), pd.Int64Index([0])],
                codes=[[], []],
                names=["x", "y"],
            )
        pd.testing.assert_index_equal(a, b)

        actual = orig.stack(z=["x", "y"]).unstack("z").drop_vars(["x", "y"])
        assert_identical(orig, actual)

        actual = orig.stack(z=[...]).unstack("z").drop_vars(["x", "y"])
        assert_identical(orig, actual)

        dims = ["a", "b", "c", "d", "e"]
        orig = xr.DataArray(np.random.rand(1, 2, 3, 2, 1), dims=dims)
        stacked = orig.stack(ab=["a", "b"], cd=["c", "d"])

        unstacked = stacked.unstack(["ab", "cd"])
        roundtripped = unstacked.drop_vars(["a", "b", "c", "d"]).transpose(*dims)
        assert_identical(orig, roundtripped)

        unstacked = stacked.unstack()
        roundtripped = unstacked.drop_vars(["a", "b", "c", "d"]).transpose(*dims)
        assert_identical(orig, roundtripped)

    def test_stack_unstack_decreasing_coordinate(self):
        # regression test for GH980
        orig = DataArray(
            np.random.rand(3, 4),
            dims=("y", "x"),
            coords={"x": np.arange(4), "y": np.arange(3, 0, -1)},
        )
        stacked = orig.stack(allpoints=["y", "x"])
        actual = stacked.unstack("allpoints")
        assert_identical(orig, actual)

    def test_unstack_pandas_consistency(self):
        df = pd.DataFrame({"foo": range(3), "x": ["a", "b", "b"], "y": [0, 0, 1]})
        s = df.set_index(["x", "y"])["foo"]
        expected = DataArray(s.unstack(), name="foo")
        actual = DataArray(s, dims="z").unstack("z")
        assert_identical(expected, actual)

    def test_stack_nonunique_consistency(self):
        orig = DataArray(
            [[0, 1], [2, 3]], dims=["x", "y"], coords={"x": [0, 1], "y": [0, 0]}
        )
        actual = orig.stack(z=["x", "y"])
        expected = DataArray(orig.to_pandas().stack(), dims="z")
        assert_identical(expected, actual)

    def test_to_unstacked_dataset_raises_value_error(self):
        data = DataArray([0, 1], dims="x", coords={"x": [0, 1]})
        with pytest.raises(ValueError, match="'x' is not a stacked coordinate"):
            data.to_unstacked_dataset("x", 0)

    def test_transpose(self):
        da = DataArray(
            np.random.randn(3, 4, 5),
            dims=("x", "y", "z"),
            coords={
                "x": range(3),
                "y": range(4),
                "z": range(5),
                "xy": (("x", "y"), np.random.randn(3, 4)),
            },
        )

        actual = da.transpose(transpose_coords=False)
        expected = DataArray(da.values.T, dims=("z", "y", "x"), coords=da.coords)
        assert_equal(expected, actual)

        actual = da.transpose("z", "y", "x", transpose_coords=True)
        expected = DataArray(
            da.values.T,
            dims=("z", "y", "x"),
            coords={
                "x": da.x.values,
                "y": da.y.values,
                "z": da.z.values,
                "xy": (("y", "x"), da.xy.values.T),
            },
        )
        assert_equal(expected, actual)

        # same as previous but with ellipsis
        actual = da.transpose("z", ..., "x", transpose_coords=True)
        assert_equal(expected, actual)

        # same as previous but with a missing dimension
        actual = da.transpose(
            "z", "y", "x", "not_a_dim", transpose_coords=True, missing_dims="ignore"
        )
        assert_equal(expected, actual)

        with pytest.raises(ValueError):
            da.transpose("x", "y")

        with pytest.raises(ValueError):
            da.transpose("not_a_dim", "z", "x", ...)

        with pytest.warns(UserWarning):
            da.transpose("not_a_dim", "y", "x", ..., missing_dims="warn")

    def test_squeeze(self):
        assert_equal(self.dv.variable.squeeze(), self.dv.squeeze().variable)

    def test_squeeze_drop(self):
        array = DataArray([1], [("x", [0])])
        expected = DataArray(1)
        actual = array.squeeze(drop=True)
        assert_identical(expected, actual)

        expected = DataArray(1, {"x": 0})
        actual = array.squeeze(drop=False)
        assert_identical(expected, actual)

        array = DataArray([[[0.0, 1.0]]], dims=["dim_0", "dim_1", "dim_2"])
        expected = DataArray([[0.0, 1.0]], dims=["dim_1", "dim_2"])
        actual = array.squeeze(axis=0)
        assert_identical(expected, actual)

        array = DataArray([[[[0.0, 1.0]]]], dims=["dim_0", "dim_1", "dim_2", "dim_3"])
        expected = DataArray([[0.0, 1.0]], dims=["dim_1", "dim_3"])
        actual = array.squeeze(axis=(0, 2))
        assert_identical(expected, actual)

        array = DataArray([[[0.0, 1.0]]], dims=["dim_0", "dim_1", "dim_2"])
        with pytest.raises(ValueError):
            array.squeeze(axis=0, dim="dim_1")

    def test_drop_coordinates(self):
        expected = DataArray(np.random.randn(2, 3), dims=["x", "y"])
        arr = expected.copy()
        arr.coords["z"] = 2
        actual = arr.drop_vars("z")
        assert_identical(expected, actual)

        with pytest.raises(ValueError):
            arr.drop_vars("not found")

        actual = expected.drop_vars("not found", errors="ignore")
        assert_identical(actual, expected)

        with pytest.raises(ValueError, match=r"cannot be found"):
            arr.drop_vars("w")

        actual = expected.drop_vars("w", errors="ignore")
        assert_identical(actual, expected)

        renamed = arr.rename("foo")
        with pytest.raises(ValueError, match=r"cannot be found"):
            renamed.drop_vars("foo")

        actual = renamed.drop_vars("foo", errors="ignore")
        assert_identical(actual, renamed)

    def test_drop_index_labels(self):
        arr = DataArray(np.random.randn(2, 3), coords={"y": [0, 1, 2]}, dims=["x", "y"])
        actual = arr.drop_sel(y=[0, 1])
        expected = arr[:, 2:]
        assert_identical(actual, expected)

        with pytest.raises((KeyError, ValueError), match=r"not .* in axis"):
            actual = arr.drop_sel(y=[0, 1, 3])

        actual = arr.drop_sel(y=[0, 1, 3], errors="ignore")
        assert_identical(actual, expected)

        with pytest.warns(DeprecationWarning):
            arr.drop([0, 1, 3], dim="y", errors="ignore")

    def test_drop_index_positions(self):
        arr = DataArray(np.random.randn(2, 3), dims=["x", "y"])
        actual = arr.drop_isel(y=[0, 1])
        expected = arr[:, 2:]
        assert_identical(actual, expected)

    def test_dropna(self):
        x = np.random.randn(4, 4)
        x[::2, 0] = np.nan
        arr = DataArray(x, dims=["a", "b"])

        actual = arr.dropna("a")
        expected = arr[1::2]
        assert_identical(actual, expected)

        actual = arr.dropna("b", how="all")
        assert_identical(actual, arr)

        actual = arr.dropna("a", thresh=1)
        assert_identical(actual, arr)

        actual = arr.dropna("b", thresh=3)
        expected = arr[:, 1:]
        assert_identical(actual, expected)

    def test_where(self):
        arr = DataArray(np.arange(4), dims="x")
        expected = arr.sel(x=slice(2))
        actual = arr.where(arr.x < 2, drop=True)
        assert_identical(actual, expected)

    def test_where_lambda(self):
        arr = DataArray(np.arange(4), dims="y")
        expected = arr.sel(y=slice(2))
        actual = arr.where(lambda x: x.y < 2, drop=True)
        assert_identical(actual, expected)

    def test_where_string(self):
        array = DataArray(["a", "b"])
        expected = DataArray(np.array(["a", np.nan], dtype=object))
        actual = array.where([True, False])
        assert_identical(actual, expected)

    def test_cumops(self):
        coords = {
            "x": [-1, -2],
            "y": ["ab", "cd", "ef"],
            "lat": (["x", "y"], [[1, 2, 3], [-1, -2, -3]]),
            "c": -999,
        }
        orig = DataArray([[-1, 0, 1], [-3, 0, 3]], coords, dims=["x", "y"])

        actual = orig.cumsum()
        expected = DataArray([[-1, -1, 0], [-4, -4, 0]], coords, dims=["x", "y"])
        assert_identical(expected, actual)

        actual = orig.cumsum("x")
        expected = DataArray([[-1, 0, 1], [-4, 0, 4]], coords, dims=["x", "y"])
        assert_identical(expected, actual)

        actual = orig.cumsum("y")
        expected = DataArray([[-1, -1, 0], [-3, -3, 0]], coords, dims=["x", "y"])
        assert_identical(expected, actual)

        actual = orig.cumprod("x")
        expected = DataArray([[-1, 0, 1], [3, 0, 3]], coords, dims=["x", "y"])
        assert_identical(expected, actual)

        actual = orig.cumprod("y")
        expected = DataArray([[-1, 0, 0], [-3, 0, 0]], coords, dims=["x", "y"])
        assert_identical(expected, actual)

    def test_reduce(self):
        coords = {
            "x": [-1, -2],
            "y": ["ab", "cd", "ef"],
            "lat": (["x", "y"], [[1, 2, 3], [-1, -2, -3]]),
            "c": -999,
        }
        orig = DataArray([[-1, 0, 1], [-3, 0, 3]], coords, dims=["x", "y"])

        actual = orig.mean()
        expected = DataArray(0, {"c": -999})
        assert_identical(expected, actual)

        actual = orig.mean(["x", "y"])
        assert_identical(expected, actual)

        actual = orig.mean("x")
        expected = DataArray([-2, 0, 2], {"y": coords["y"], "c": -999}, "y")
        assert_identical(expected, actual)

        actual = orig.mean(["x"])
        assert_identical(expected, actual)

        actual = orig.mean("y")
        expected = DataArray([0, 0], {"x": coords["x"], "c": -999}, "x")
        assert_identical(expected, actual)

        assert_equal(self.dv.reduce(np.mean, "x").variable, self.v.reduce(np.mean, "x"))

        orig = DataArray([[1, 0, np.nan], [3, 0, 3]], coords, dims=["x", "y"])
        actual = orig.count()
        expected = DataArray(5, {"c": -999})
        assert_identical(expected, actual)

        # uint support
        orig = DataArray(np.arange(6).reshape(3, 2).astype("uint"), dims=["x", "y"])
        assert orig.dtype.kind == "u"
        actual = orig.mean(dim="x", skipna=True)
        expected = DataArray(orig.values.astype(int), dims=["x", "y"]).mean("x")
        assert_equal(actual, expected)

    def test_reduce_keepdims(self):
        coords = {
            "x": [-1, -2],
            "y": ["ab", "cd", "ef"],
            "lat": (["x", "y"], [[1, 2, 3], [-1, -2, -3]]),
            "c": -999,
        }
        orig = DataArray([[-1, 0, 1], [-3, 0, 3]], coords, dims=["x", "y"])

        # Mean on all axes loses non-constant coordinates
        actual = orig.mean(keepdims=True)
        expected = DataArray(
            orig.data.mean(keepdims=True),
            dims=orig.dims,
            coords={k: v for k, v in coords.items() if k in ["c"]},
        )
        assert_equal(actual, expected)

        assert actual.sizes["x"] == 1
        assert actual.sizes["y"] == 1

        # Mean on specific axes loses coordinates not involving that axis
        actual = orig.mean("y", keepdims=True)
        expected = DataArray(
            orig.data.mean(axis=1, keepdims=True),
            dims=orig.dims,
            coords={k: v for k, v in coords.items() if k not in ["y", "lat"]},
        )
        assert_equal(actual, expected)

    @requires_bottleneck
    def test_reduce_keepdims_bottleneck(self):
        import bottleneck

        coords = {
            "x": [-1, -2],
            "y": ["ab", "cd", "ef"],
            "lat": (["x", "y"], [[1, 2, 3], [-1, -2, -3]]),
            "c": -999,
        }
        orig = DataArray([[-1, 0, 1], [-3, 0, 3]], coords, dims=["x", "y"])

        # Bottleneck does not have its own keepdims implementation
        actual = orig.reduce(bottleneck.nanmean, keepdims=True)
        expected = orig.mean(keepdims=True)
        assert_equal(actual, expected)

    def test_reduce_dtype(self):
        coords = {
            "x": [-1, -2],
            "y": ["ab", "cd", "ef"],
            "lat": (["x", "y"], [[1, 2, 3], [-1, -2, -3]]),
            "c": -999,
        }
        orig = DataArray([[-1, 0, 1], [-3, 0, 3]], coords, dims=["x", "y"])

        for dtype in [np.float16, np.float32, np.float64]:
            assert orig.astype(float).mean(dtype=dtype).dtype == dtype

    def test_reduce_out(self):
        coords = {
            "x": [-1, -2],
            "y": ["ab", "cd", "ef"],
            "lat": (["x", "y"], [[1, 2, 3], [-1, -2, -3]]),
            "c": -999,
        }
        orig = DataArray([[-1, 0, 1], [-3, 0, 3]], coords, dims=["x", "y"])

        with pytest.raises(TypeError):
            orig.mean(out=np.ones(orig.shape))

    @pytest.mark.parametrize("skipna", [True, False])
    @pytest.mark.parametrize("q", [0.25, [0.50], [0.25, 0.75]])
    @pytest.mark.parametrize(
        "axis, dim", zip([None, 0, [0], [0, 1]], [None, "x", ["x"], ["x", "y"]])
    )
    def test_quantile(self, q, axis, dim, skipna):
        actual = DataArray(self.va).quantile(q, dim=dim, keep_attrs=True, skipna=skipna)
        _percentile_func = np.nanpercentile if skipna else np.percentile
        expected = _percentile_func(self.dv.values, np.array(q) * 100, axis=axis)
        np.testing.assert_allclose(actual.values, expected)
        if is_scalar(q):
            assert "quantile" not in actual.dims
        else:
            assert "quantile" in actual.dims

        assert actual.attrs == self.attrs

    def test_reduce_keep_attrs(self):
        # Test dropped attrs
        vm = self.va.mean()
        assert len(vm.attrs) == 0
        assert vm.attrs == {}

        # Test kept attrs
        vm = self.va.mean(keep_attrs=True)
        assert len(vm.attrs) == len(self.attrs)
        assert vm.attrs == self.attrs

    def test_assign_attrs(self):
        expected = DataArray([], attrs=dict(a=1, b=2))
        expected.attrs["a"] = 1
        expected.attrs["b"] = 2
        new = DataArray([])
        actual = DataArray([]).assign_attrs(a=1, b=2)
        assert_identical(actual, expected)
        assert new.attrs == {}

        expected.attrs["c"] = 3
        new_actual = actual.assign_attrs({"c": 3})
        assert_identical(new_actual, expected)
        assert actual.attrs == {"a": 1, "b": 2}

    @pytest.mark.parametrize(
        "func", [lambda x: x.clip(0, 1), lambda x: np.float64(1.0) * x, np.abs, abs]
    )
    def test_propagate_attrs(self, func):
        da = DataArray(self.va)

        # test defaults
        assert func(da).attrs == da.attrs

        with set_options(keep_attrs=False):
            assert func(da).attrs == {}

        with set_options(keep_attrs=True):
            assert func(da).attrs == da.attrs

    def test_fillna(self):
        a = DataArray([np.nan, 1, np.nan, 3], coords={"x": range(4)}, dims="x")
        actual = a.fillna(-1)
        expected = DataArray([-1, 1, -1, 3], coords={"x": range(4)}, dims="x")
        assert_identical(expected, actual)

        b = DataArray(range(4), coords={"x": range(4)}, dims="x")
        actual = a.fillna(b)
        expected = b.copy()
        assert_identical(expected, actual)

        actual = a.fillna(range(4))
        assert_identical(expected, actual)

        actual = a.fillna(b[:3])
        assert_identical(expected, actual)

        actual = a.fillna(b[:0])
        assert_identical(a, actual)

        with pytest.raises(TypeError, match=r"fillna on a DataArray"):
            a.fillna({0: 0})

        with pytest.raises(ValueError, match=r"broadcast"):
            a.fillna([1, 2])

        fill_value = DataArray([0, 1], dims="y")
        actual = a.fillna(fill_value)
        expected = DataArray(
            [[0, 1], [1, 1], [0, 1], [3, 3]], coords={"x": range(4)}, dims=("x", "y")
        )
        assert_identical(expected, actual)

        expected = b.copy()
        for target in [a, expected]:
            target.coords["b"] = ("x", [0, 0, 1, 1])
        actual = a.groupby("b").fillna(DataArray([0, 2], dims="b"))
        assert_identical(expected, actual)

    def test_groupby_iter(self):
        for ((act_x, act_dv), (exp_x, exp_ds)) in zip(
            self.dv.groupby("y"), self.ds.groupby("y")
        ):
            assert exp_x == act_x
            assert_identical(exp_ds["foo"], act_dv)
        for ((_, exp_dv), act_dv) in zip(self.dv.groupby("x"), self.dv):
            assert_identical(exp_dv, act_dv)

    def make_groupby_example_array(self):
        da = self.dv.copy()
        da.coords["abc"] = ("y", np.array(["a"] * 9 + ["c"] + ["b"] * 10))
        da.coords["y"] = 20 + 100 * da["y"]
        return da

    def test_groupby_properties(self):
        grouped = self.make_groupby_example_array().groupby("abc")
        expected_groups = {"a": range(0, 9), "c": [9], "b": range(10, 20)}
        assert expected_groups.keys() == grouped.groups.keys()
        for key in expected_groups:
            assert_array_equal(expected_groups[key], grouped.groups[key])
        assert 3 == len(grouped)

    def test_groupby_map_identity(self):
        expected = self.make_groupby_example_array()
        idx = expected.coords["y"]

        def identity(x):
            return x

        for g in ["x", "y", "abc", idx]:
            for shortcut in [False, True]:
                for squeeze in [False, True]:
                    grouped = expected.groupby(g, squeeze=squeeze)
                    actual = grouped.map(identity, shortcut=shortcut)
                    assert_identical(expected, actual)

    def test_groupby_sum(self):
        array = self.make_groupby_example_array()
        grouped = array.groupby("abc")

        expected_sum_all = Dataset(
            {
                "foo": Variable(
                    ["abc"],
                    np.array(
                        [
                            self.x[:, :9].sum(),
                            self.x[:, 10:].sum(),
                            self.x[:, 9:10].sum(),
                        ]
                    ).T,
                ),
                "abc": Variable(["abc"], np.array(["a", "b", "c"])),
            }
        )["foo"]
        assert_allclose(expected_sum_all, grouped.reduce(np.sum, dim=...))
        assert_allclose(expected_sum_all, grouped.sum(...))

        expected = DataArray(
            [
                array["y"].values[idx].sum()
                for idx in [slice(9), slice(10, None), slice(9, 10)]
            ],
            [["a", "b", "c"]],
            ["abc"],
        )
        actual = array["y"].groupby("abc").map(np.sum)
        assert_allclose(expected, actual)
        actual = array["y"].groupby("abc").sum(...)
        assert_allclose(expected, actual)

        expected_sum_axis1 = Dataset(
            {
                "foo": (
                    ["x", "abc"],
                    np.array(
                        [
                            self.x[:, :9].sum(1),
                            self.x[:, 10:].sum(1),
                            self.x[:, 9:10].sum(1),
                        ]
                    ).T,
                ),
                "abc": Variable(["abc"], np.array(["a", "b", "c"])),
            }
        )["foo"]
        assert_allclose(expected_sum_axis1, grouped.reduce(np.sum, "y"))
        assert_allclose(expected_sum_axis1, grouped.sum("y"))

    def test_groupby_sum_default(self):
        array = self.make_groupby_example_array()
        grouped = array.groupby("abc")

        expected_sum_all = Dataset(
            {
                "foo": Variable(
                    ["x", "abc"],
                    np.array(
                        [
                            self.x[:, :9].sum(axis=-1),
                            self.x[:, 10:].sum(axis=-1),
                            self.x[:, 9:10].sum(axis=-1),
                        ]
                    ).T,
                ),
                "abc": Variable(["abc"], np.array(["a", "b", "c"])),
            }
        )["foo"]

        assert_allclose(expected_sum_all, grouped.sum(dim="y"))

    def test_groupby_count(self):
        array = DataArray(
            [0, 0, np.nan, np.nan, 0, 0],
            coords={"cat": ("x", ["a", "b", "b", "c", "c", "c"])},
            dims="x",
        )
        actual = array.groupby("cat").count()
        expected = DataArray([1, 1, 2], coords=[("cat", ["a", "b", "c"])])
        assert_identical(actual, expected)

    @pytest.mark.skip("needs to be fixed for shortcut=False, keep_attrs=False")
    def test_groupby_reduce_attrs(self):
        array = self.make_groupby_example_array()
        array.attrs["foo"] = "bar"

        for shortcut in [True, False]:
            for keep_attrs in [True, False]:
                print(f"shortcut={shortcut}, keep_attrs={keep_attrs}")
                actual = array.groupby("abc").reduce(
                    np.mean, keep_attrs=keep_attrs, shortcut=shortcut
                )
                expected = array.groupby("abc").mean()
                if keep_attrs:
                    expected.attrs["foo"] = "bar"
                assert_identical(expected, actual)

    def test_groupby_map_center(self):
        def center(x):
            return x - np.mean(x)

        array = self.make_groupby_example_array()
        grouped = array.groupby("abc")

        expected_ds = array.to_dataset()
        exp_data = np.hstack(
            [center(self.x[:, :9]), center(self.x[:, 9:10]), center(self.x[:, 10:])]
        )
        expected_ds["foo"] = (["x", "y"], exp_data)
        expected_centered = expected_ds["foo"]
        assert_allclose(expected_centered, grouped.map(center))

    def test_groupby_map_ndarray(self):
        # regression test for #326
        array = self.make_groupby_example_array()
        grouped = array.groupby("abc")
        actual = grouped.map(np.asarray)
        assert_equal(array, actual)

    def test_groupby_map_changes_metadata(self):
        def change_metadata(x):
            x.coords["x"] = x.coords["x"] * 2
            x.attrs["fruit"] = "lemon"
            return x

        array = self.make_groupby_example_array()
        grouped = array.groupby("abc")
        actual = grouped.map(change_metadata)
        expected = array.copy()
        expected = change_metadata(expected)
        assert_equal(expected, actual)

    def test_groupby_math(self):
        array = self.make_groupby_example_array()
        for squeeze in [True, False]:
            grouped = array.groupby("x", squeeze=squeeze)

            expected = array + array.coords["x"]
            actual = grouped + array.coords["x"]
            assert_identical(expected, actual)

            actual = array.coords["x"] + grouped
            assert_identical(expected, actual)

            ds = array.coords["x"].to_dataset(name="X")
            expected = array + ds
            actual = grouped + ds
            assert_identical(expected, actual)

            actual = ds + grouped
            assert_identical(expected, actual)

        grouped = array.groupby("abc")
        expected_agg = (grouped.mean(...) - np.arange(3)).rename(None)
        actual = grouped - DataArray(range(3), [("abc", ["a", "b", "c"])])
        actual_agg = actual.groupby("abc").mean(...)
        assert_allclose(expected_agg, actual_agg)

        with pytest.raises(TypeError, match=r"only support binary ops"):
            grouped + 1
        with pytest.raises(TypeError, match=r"only support binary ops"):
            grouped + grouped
        with pytest.raises(TypeError, match=r"in-place operations"):
            array += grouped

    def test_groupby_math_not_aligned(self):
        array = DataArray(
            range(4), {"b": ("x", [0, 0, 1, 1]), "x": [0, 1, 2, 3]}, dims="x"
        )
        other = DataArray([10], coords={"b": [0]}, dims="b")
        actual = array.groupby("b") + other
        expected = DataArray([10, 11, np.nan, np.nan], array.coords)
        assert_identical(expected, actual)

        other = DataArray([10], coords={"c": 123, "b": [0]}, dims="b")
        actual = array.groupby("b") + other
        expected.coords["c"] = (["x"], [123] * 2 + [np.nan] * 2)
        assert_identical(expected, actual)

        other = Dataset({"a": ("b", [10])}, {"b": [0]})
        actual = array.groupby("b") + other
        expected = Dataset({"a": ("x", [10, 11, np.nan, np.nan])}, array.coords)
        assert_identical(expected, actual)

    def test_groupby_restore_dim_order(self):
        array = DataArray(
            np.random.randn(5, 3),
            coords={"a": ("x", range(5)), "b": ("y", range(3))},
            dims=["x", "y"],
        )
        for by, expected_dims in [
            ("x", ("x", "y")),
            ("y", ("x", "y")),
            ("a", ("a", "y")),
            ("b", ("x", "b")),
        ]:
            result = array.groupby(by).map(lambda x: x.squeeze())
            assert result.dims == expected_dims

    def test_groupby_restore_coord_dims(self):
        array = DataArray(
            np.random.randn(5, 3),
            coords={
                "a": ("x", range(5)),
                "b": ("y", range(3)),
                "c": (("x", "y"), np.random.randn(5, 3)),
            },
            dims=["x", "y"],
        )

        for by, expected_dims in [
            ("x", ("x", "y")),
            ("y", ("x", "y")),
            ("a", ("a", "y")),
            ("b", ("x", "b")),
        ]:
            result = array.groupby(by, restore_coord_dims=True).map(
                lambda x: x.squeeze()
            )["c"]
            assert result.dims == expected_dims

    def test_groupby_first_and_last(self):
        array = DataArray([1, 2, 3, 4, 5], dims="x")
        by = DataArray(["a"] * 2 + ["b"] * 3, dims="x", name="ab")

        expected = DataArray([1, 3], [("ab", ["a", "b"])])
        actual = array.groupby(by).first()
        assert_identical(expected, actual)

        expected = DataArray([2, 5], [("ab", ["a", "b"])])
        actual = array.groupby(by).last()
        assert_identical(expected, actual)

        array = DataArray(np.random.randn(5, 3), dims=["x", "y"])
        expected = DataArray(array[[0, 2]], {"ab": ["a", "b"]}, ["ab", "y"])
        actual = array.groupby(by).first()
        assert_identical(expected, actual)

        actual = array.groupby("x").first()
        expected = array  # should be a no-op
        assert_identical(expected, actual)

    def make_groupby_multidim_example_array(self):
        return DataArray(
            [[[0, 1], [2, 3]], [[5, 10], [15, 20]]],
            coords={
                "lon": (["ny", "nx"], [[30, 40], [40, 50]]),
                "lat": (["ny", "nx"], [[10, 10], [20, 20]]),
            },
            dims=["time", "ny", "nx"],
        )

    def test_groupby_multidim(self):
        array = self.make_groupby_multidim_example_array()
        for dim, expected_sum in [
            ("lon", DataArray([5, 28, 23], coords=[("lon", [30.0, 40.0, 50.0])])),
            ("lat", DataArray([16, 40], coords=[("lat", [10.0, 20.0])])),
        ]:
            actual_sum = array.groupby(dim).sum(...)
            assert_identical(expected_sum, actual_sum)

    def test_groupby_multidim_map(self):
        array = self.make_groupby_multidim_example_array()
        actual = array.groupby("lon").map(lambda x: x - x.mean())
        expected = DataArray(
            [[[-2.5, -6.0], [-5.0, -8.5]], [[2.5, 3.0], [8.0, 8.5]]],
            coords=array.coords,
            dims=array.dims,
        )
        assert_identical(expected, actual)

    def test_groupby_bins(self):
        array = DataArray(np.arange(4), dims="dim_0")
        # the first value should not be part of any group ("right" binning)
        array[0] = 99
        # bins follow conventions for pandas.cut
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
        bins = [0, 1.5, 5]
        bin_coords = pd.cut(array["dim_0"], bins).categories
        expected = DataArray(
            [1, 5], dims="dim_0_bins", coords={"dim_0_bins": bin_coords}
        )
        # the problem with this is that it overwrites the dimensions of array!
        # actual = array.groupby('dim_0', bins=bins).sum()
        actual = array.groupby_bins("dim_0", bins).map(lambda x: x.sum())
        assert_identical(expected, actual)
        # make sure original array dims are unchanged
        assert len(array.dim_0) == 4

    def test_groupby_bins_empty(self):
        array = DataArray(np.arange(4), [("x", range(4))])
        # one of these bins will be empty
        bins = [0, 4, 5]
        bin_coords = pd.cut(array["x"], bins).categories
        actual = array.groupby_bins("x", bins).sum()
        expected = DataArray([6, np.nan], dims="x_bins", coords={"x_bins": bin_coords})
        assert_identical(expected, actual)
        # make sure original array is unchanged
        # (was a problem in earlier versions)
        assert len(array.x) == 4

    def test_groupby_bins_multidim(self):
        array = self.make_groupby_multidim_example_array()
        bins = [0, 15, 20]
        bin_coords = pd.cut(array["lat"].values.flat, bins).categories
        expected = DataArray([16, 40], dims="lat_bins", coords={"lat_bins": bin_coords})
        actual = array.groupby_bins("lat", bins).map(lambda x: x.sum())
        assert_identical(expected, actual)
        # modify the array coordinates to be non-monotonic after unstacking
        array["lat"].data = np.array([[10.0, 20.0], [20.0, 10.0]])
        expected = DataArray([28, 28], dims="lat_bins", coords={"lat_bins": bin_coords})
        actual = array.groupby_bins("lat", bins).map(lambda x: x.sum())
        assert_identical(expected, actual)

    def test_groupby_bins_sort(self):
        data = xr.DataArray(
            np.arange(100), dims="x", coords={"x": np.linspace(-100, 100, num=100)}
        )
        binned_mean = data.groupby_bins("x", bins=11).mean()
        assert binned_mean.to_index().is_monotonic

    def test_resample(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        array = DataArray(np.arange(10), [("time", times)])

        actual = array.resample(time="24H").mean()
        expected = DataArray(array.to_series().resample("24H").mean())
        assert_identical(expected, actual)

        actual = array.resample(time="24H").reduce(np.mean)
        assert_identical(expected, actual)

        # Our use of `loffset` may change if we align our API with pandas' changes.
        # ref https://github.com/pydata/xarray/pull/4537
        actual = array.resample(time="24H", loffset="-12H").mean()
        expected_ = array.to_series().resample("24H").mean()
        expected_.index += to_offset("-12H")
        expected = DataArray.from_series(expected_)
        assert_identical(actual, expected)

        with pytest.raises(ValueError, match=r"index must be monotonic"):
            array[[2, 0, 1]].resample(time="1D")

    def test_da_resample_func_args(self):
        def func(arg1, arg2, arg3=0.0):
            return arg1.mean("time") + arg2 + arg3

        times = pd.date_range("2000", periods=3, freq="D")
        da = xr.DataArray([1.0, 1.0, 1.0], coords=[times], dims=["time"])
        expected = xr.DataArray([3.0, 3.0, 3.0], coords=[times], dims=["time"])
        actual = da.resample(time="D").map(func, args=(1.0,), arg3=1.0)
        assert_identical(actual, expected)

    def test_resample_first(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        array = DataArray(np.arange(10), [("time", times)])

        actual = array.resample(time="1D").first()
        expected = DataArray([0, 4, 8], [("time", times[::4])])
        assert_identical(expected, actual)

        # verify that labels don't use the first value
        actual = array.resample(time="24H").first()
        expected = DataArray(array.to_series().resample("24H").first())
        assert_identical(expected, actual)

        # missing values
        array = array.astype(float)
        array[:2] = np.nan
        actual = array.resample(time="1D").first()
        expected = DataArray([2, 4, 8], [("time", times[::4])])
        assert_identical(expected, actual)

        actual = array.resample(time="1D").first(skipna=False)
        expected = DataArray([np.nan, 4, 8], [("time", times[::4])])
        assert_identical(expected, actual)

        # regression test for http://stackoverflow.com/questions/33158558/
        array = Dataset({"time": times})["time"]
        actual = array.resample(time="1D").last()
        expected_times = pd.to_datetime(
            ["2000-01-01T18", "2000-01-02T18", "2000-01-03T06"]
        )
        expected = DataArray(expected_times, [("time", times[::4])], name="time")
        assert_identical(expected, actual)

    def test_resample_bad_resample_dim(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        array = DataArray(np.arange(10), [("__resample_dim__", times)])
        with pytest.raises(ValueError, match=r"Proxy resampling dimension"):
            array.resample(**{"__resample_dim__": "1D"}).first()

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

        # Select the data now, with the auxiliary coordinates in place
        array = ds["data"]

        # Re-sample
        actual = array.resample(time="12H", restore_coord_dims=True).mean("time")
        assert "tc" not in actual.coords

        # Up-sample - filling
        actual = array.resample(time="1H", restore_coord_dims=True).ffill()
        assert "tc" not in actual.coords

        # Up-sample - interpolation
        actual = array.resample(time="1H", restore_coord_dims=True).interpolate(
            "linear"
        )
        assert "tc" not in actual.coords

    def test_resample_keep_attrs(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        array = DataArray(np.ones(10), [("time", times)])
        array.attrs["meta"] = "data"

        result = array.resample(time="1D").mean(keep_attrs=True)
        expected = DataArray([1, 1, 1], [("time", times[::4])], attrs=array.attrs)
        assert_identical(result, expected)

        with pytest.warns(
            UserWarning, match="Passing ``keep_attrs`` to ``resample`` has no effect."
        ):
            array.resample(time="1D", keep_attrs=True)

    def test_resample_skipna(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        array = DataArray(np.ones(10), [("time", times)])
        array[1] = np.nan

        result = array.resample(time="1D").mean(skipna=False)
        expected = DataArray([np.nan, 1, 1], [("time", times[::4])])
        assert_identical(result, expected)

    def test_upsample(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=5)
        array = DataArray(np.arange(5), [("time", times)])

        # Forward-fill
        actual = array.resample(time="3H").ffill()
        expected = DataArray(array.to_series().resample("3H").ffill())
        assert_identical(expected, actual)

        # Backward-fill
        actual = array.resample(time="3H").bfill()
        expected = DataArray(array.to_series().resample("3H").bfill())
        assert_identical(expected, actual)

        # As frequency
        actual = array.resample(time="3H").asfreq()
        expected = DataArray(array.to_series().resample("3H").asfreq())
        assert_identical(expected, actual)

        # Pad
        actual = array.resample(time="3H").pad()
        expected = DataArray(array.to_series().resample("3H").pad())
        assert_identical(expected, actual)

        # Nearest
        rs = array.resample(time="3H")
        actual = rs.nearest()
        new_times = rs._full_index
        expected = DataArray(array.reindex(time=new_times, method="nearest"))
        assert_identical(expected, actual)

    def test_upsample_nd(self):
        # Same as before, but now we try on multi-dimensional DataArrays.
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range("2000-01-01", freq="6H", periods=5)
        data = np.tile(np.arange(5), (6, 3, 1))
        array = DataArray(data, {"time": times, "x": xs, "y": ys}, ("x", "y", "time"))

        # Forward-fill
        actual = array.resample(time="3H").ffill()
        expected_data = np.repeat(data, 2, axis=-1)
        expected_times = times.to_series().resample("3H").asfreq().index
        expected_data = expected_data[..., : len(expected_times)]
        expected = DataArray(
            expected_data,
            {"time": expected_times, "x": xs, "y": ys},
            ("x", "y", "time"),
        )
        assert_identical(expected, actual)

        # Backward-fill
        actual = array.resample(time="3H").ffill()
        expected_data = np.repeat(np.flipud(data.T).T, 2, axis=-1)
        expected_data = np.flipud(expected_data.T).T
        expected_times = times.to_series().resample("3H").asfreq().index
        expected_data = expected_data[..., : len(expected_times)]
        expected = DataArray(
            expected_data,
            {"time": expected_times, "x": xs, "y": ys},
            ("x", "y", "time"),
        )
        assert_identical(expected, actual)

        # As frequency
        actual = array.resample(time="3H").asfreq()
        expected_data = np.repeat(data, 2, axis=-1).astype(float)[..., :-1]
        expected_data[..., 1::2] = np.nan
        expected_times = times.to_series().resample("3H").asfreq().index
        expected = DataArray(
            expected_data,
            {"time": expected_times, "x": xs, "y": ys},
            ("x", "y", "time"),
        )
        assert_identical(expected, actual)

        # Pad
        actual = array.resample(time="3H").pad()
        expected_data = np.repeat(data, 2, axis=-1)
        expected_data[..., 1::2] = expected_data[..., ::2]
        expected_data = expected_data[..., :-1]
        expected_times = times.to_series().resample("3H").asfreq().index
        expected = DataArray(
            expected_data,
            {"time": expected_times, "x": xs, "y": ys},
            ("x", "y", "time"),
        )
        assert_identical(expected, actual)

    def test_upsample_tolerance(self):
        # Test tolerance keyword for upsample methods bfill, pad, nearest
        times = pd.date_range("2000-01-01", freq="1D", periods=2)
        times_upsampled = pd.date_range("2000-01-01", freq="6H", periods=5)
        array = DataArray(np.arange(2), [("time", times)])

        # Forward fill
        actual = array.resample(time="6H").ffill(tolerance="12H")
        expected = DataArray([0.0, 0.0, 0.0, np.nan, 1.0], [("time", times_upsampled)])
        assert_identical(expected, actual)

        # Backward fill
        actual = array.resample(time="6H").bfill(tolerance="12H")
        expected = DataArray([0.0, np.nan, 1.0, 1.0, 1.0], [("time", times_upsampled)])
        assert_identical(expected, actual)

        # Nearest
        actual = array.resample(time="6H").nearest(tolerance="6H")
        expected = DataArray([0, 0, np.nan, 1, 1], [("time", times_upsampled)])
        assert_identical(expected, actual)

    @requires_scipy
    def test_upsample_interpolate(self):
        from scipy.interpolate import interp1d

        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range("2000-01-01", freq="6H", periods=5)

        z = np.arange(5) ** 2
        data = np.tile(z, (6, 3, 1))
        array = DataArray(data, {"time": times, "x": xs, "y": ys}, ("x", "y", "time"))

        expected_times = times.to_series().resample("1H").asfreq().index
        # Split the times into equal sub-intervals to simulate the 6 hour
        # to 1 hour up-sampling
        new_times_idx = np.linspace(0, len(times) - 1, len(times) * 5)
        for kind in ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]:
            actual = array.resample(time="1H").interpolate(kind)
            f = interp1d(
                np.arange(len(times)),
                data,
                kind=kind,
                axis=-1,
                bounds_error=True,
                assume_sorted=True,
            )
            expected_data = f(new_times_idx)
            expected = DataArray(
                expected_data,
                {"time": expected_times, "x": xs, "y": ys},
                ("x", "y", "time"),
            )
            # Use AllClose because there are some small differences in how
            # we upsample timeseries versus the integer indexing as I've
            # done here due to floating point arithmetic
            assert_allclose(expected, actual, rtol=1e-16)

    @requires_scipy
    def test_upsample_interpolate_bug_2197(self):
        dates = pd.date_range("2007-02-01", "2007-03-01", freq="D")
        da = xr.DataArray(np.arange(len(dates)), [("time", dates)])
        result = da.resample(time="M").interpolate("linear")
        expected_times = np.array(
            [np.datetime64("2007-02-28"), np.datetime64("2007-03-31")]
        )
        expected = xr.DataArray([27.0, np.nan], [("time", expected_times)])
        assert_equal(result, expected)

    @requires_scipy
    def test_upsample_interpolate_regression_1605(self):
        dates = pd.date_range("2016-01-01", "2016-03-31", freq="1D")
        expected = xr.DataArray(
            np.random.random((len(dates), 2, 3)),
            dims=("time", "x", "y"),
            coords={"time": dates},
        )
        actual = expected.resample(time="1D").interpolate("linear")
        assert_allclose(actual, expected, rtol=1e-16)

    @requires_dask
    @requires_scipy
    @pytest.mark.parametrize("chunked_time", [True, False])
    def test_upsample_interpolate_dask(self, chunked_time):
        from scipy.interpolate import interp1d

        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range("2000-01-01", freq="6H", periods=5)

        z = np.arange(5) ** 2
        data = np.tile(z, (6, 3, 1))
        array = DataArray(data, {"time": times, "x": xs, "y": ys}, ("x", "y", "time"))
        chunks = {"x": 2, "y": 1}
        if chunked_time:
            chunks["time"] = 3

        expected_times = times.to_series().resample("1H").asfreq().index
        # Split the times into equal sub-intervals to simulate the 6 hour
        # to 1 hour up-sampling
        new_times_idx = np.linspace(0, len(times) - 1, len(times) * 5)
        for kind in ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]:
            actual = array.chunk(chunks).resample(time="1H").interpolate(kind)
            actual = actual.compute()
            f = interp1d(
                np.arange(len(times)),
                data,
                kind=kind,
                axis=-1,
                bounds_error=True,
                assume_sorted=True,
            )
            expected_data = f(new_times_idx)
            expected = DataArray(
                expected_data,
                {"time": expected_times, "x": xs, "y": ys},
                ("x", "y", "time"),
            )
            # Use AllClose because there are some small differences in how
            # we upsample timeseries versus the integer indexing as I've
            # done here due to floating point arithmetic
            assert_allclose(expected, actual, rtol=1e-16)

    def test_align(self):
        array = DataArray(
            np.random.random((6, 8)), coords={"x": list("abcdef")}, dims=["x", "y"]
        )
        array1, array2 = align(array, array[:5], join="inner")
        assert_identical(array1, array[:5])
        assert_identical(array2, array[:5])

    def test_align_dtype(self):
        # regression test for #264
        x1 = np.arange(30)
        x2 = np.arange(5, 35)
        a = DataArray(np.random.random((30,)).astype(np.float32), [("x", x1)])
        b = DataArray(np.random.random((30,)).astype(np.float32), [("x", x2)])
        c, d = align(a, b, join="outer")
        assert c.dtype == np.float32

    def test_align_copy(self):
        x = DataArray([1, 2, 3], coords=[("a", [1, 2, 3])])
        y = DataArray([1, 2], coords=[("a", [3, 1])])

        expected_x2 = x
        expected_y2 = DataArray([2, np.nan, 1], coords=[("a", [1, 2, 3])])

        x2, y2 = align(x, y, join="outer", copy=False)
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)
        assert source_ndarray(x2.data) is source_ndarray(x.data)

        x2, y2 = align(x, y, join="outer", copy=True)
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)
        assert source_ndarray(x2.data) is not source_ndarray(x.data)

        # Trivial align - 1 element
        x = DataArray([1, 2, 3], coords=[("a", [1, 2, 3])])
        (x2,) = align(x, copy=False)
        assert_identical(x, x2)
        assert source_ndarray(x2.data) is source_ndarray(x.data)

        (x2,) = align(x, copy=True)
        assert_identical(x, x2)
        assert source_ndarray(x2.data) is not source_ndarray(x.data)

    def test_align_override(self):
        left = DataArray([1, 2, 3], dims="x", coords={"x": [0, 1, 2]})
        right = DataArray(
            np.arange(9).reshape((3, 3)),
            dims=["x", "y"],
            coords={"x": [0.1, 1.1, 2.1], "y": [1, 2, 3]},
        )

        expected_right = DataArray(
            np.arange(9).reshape(3, 3),
            dims=["x", "y"],
            coords={"x": [0, 1, 2], "y": [1, 2, 3]},
        )

        new_left, new_right = align(left, right, join="override")
        assert_identical(left, new_left)
        assert_identical(new_right, expected_right)

        new_left, new_right = align(left, right, exclude="x", join="override")
        assert_identical(left, new_left)
        assert_identical(right, new_right)

        new_left, new_right = xr.align(
            left.isel(x=0, drop=True), right, exclude="x", join="override"
        )
        assert_identical(left.isel(x=0, drop=True), new_left)
        assert_identical(right, new_right)

        with pytest.raises(ValueError, match=r"Indexes along dimension 'x' don't have"):
            align(left.isel(x=0).expand_dims("x"), right, join="override")

    @pytest.mark.parametrize(
        "darrays",
        [
            [
                DataArray(0),
                DataArray([1], [("x", [1])]),
                DataArray([2, 3], [("x", [2, 3])]),
            ],
            [
                DataArray([2, 3], [("x", [2, 3])]),
                DataArray([1], [("x", [1])]),
                DataArray(0),
            ],
        ],
    )
    def test_align_override_error(self, darrays):
        with pytest.raises(ValueError, match=r"Indexes along dimension 'x' don't have"):
            xr.align(*darrays, join="override")

    def test_align_exclude(self):
        x = DataArray([[1, 2], [3, 4]], coords=[("a", [-1, -2]), ("b", [3, 4])])
        y = DataArray([[1, 2], [3, 4]], coords=[("a", [-1, 20]), ("b", [5, 6])])
        z = DataArray([1], dims=["a"], coords={"a": [20], "b": 7})

        x2, y2, z2 = align(x, y, z, join="outer", exclude=["b"])
        expected_x2 = DataArray(
            [[3, 4], [1, 2], [np.nan, np.nan]],
            coords=[("a", [-2, -1, 20]), ("b", [3, 4])],
        )
        expected_y2 = DataArray(
            [[np.nan, np.nan], [1, 2], [3, 4]],
            coords=[("a", [-2, -1, 20]), ("b", [5, 6])],
        )
        expected_z2 = DataArray(
            [np.nan, np.nan, 1], dims=["a"], coords={"a": [-2, -1, 20], "b": 7}
        )
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)
        assert_identical(expected_z2, z2)

    def test_align_indexes(self):
        x = DataArray([1, 2, 3], coords=[("a", [-1, 10, -2])])
        y = DataArray([1, 2], coords=[("a", [-2, -1])])

        x2, y2 = align(x, y, join="outer", indexes={"a": [10, -1, -2]})
        expected_x2 = DataArray([2, 1, 3], coords=[("a", [10, -1, -2])])
        expected_y2 = DataArray([np.nan, 2, 1], coords=[("a", [10, -1, -2])])
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)

        (x2,) = align(x, join="outer", indexes={"a": [-2, 7, 10, -1]})
        expected_x2 = DataArray([3, np.nan, 2, 1], coords=[("a", [-2, 7, 10, -1])])
        assert_identical(expected_x2, x2)

    def test_align_without_indexes_exclude(self):
        arrays = [DataArray([1, 2, 3], dims=["x"]), DataArray([1, 2], dims=["x"])]
        result0, result1 = align(*arrays, exclude=["x"])
        assert_identical(result0, arrays[0])
        assert_identical(result1, arrays[1])

    def test_align_mixed_indexes(self):
        array_no_coord = DataArray([1, 2], dims=["x"])
        array_with_coord = DataArray([1, 2], coords=[("x", ["a", "b"])])
        result0, result1 = align(array_no_coord, array_with_coord)
        assert_identical(result0, array_with_coord)
        assert_identical(result1, array_with_coord)

        result0, result1 = align(array_no_coord, array_with_coord, exclude=["x"])
        assert_identical(result0, array_no_coord)
        assert_identical(result1, array_with_coord)

    def test_align_without_indexes_errors(self):
        with pytest.raises(ValueError, match=r"cannot be aligned"):
            align(DataArray([1, 2, 3], dims=["x"]), DataArray([1, 2], dims=["x"]))

        with pytest.raises(ValueError, match=r"cannot be aligned"):
            align(
                DataArray([1, 2, 3], dims=["x"]),
                DataArray([1, 2], coords=[("x", [0, 1])]),
            )

    def test_align_str_dtype(self):

        a = DataArray([0, 1], dims=["x"], coords={"x": ["a", "b"]})
        b = DataArray([1, 2], dims=["x"], coords={"x": ["b", "c"]})

        expected_a = DataArray(
            [0, 1, np.NaN], dims=["x"], coords={"x": ["a", "b", "c"]}
        )
        expected_b = DataArray(
            [np.NaN, 1, 2], dims=["x"], coords={"x": ["a", "b", "c"]}
        )

        actual_a, actual_b = xr.align(a, b, join="outer")

        assert_identical(expected_a, actual_a)
        assert expected_a.x.dtype == actual_a.x.dtype

        assert_identical(expected_b, actual_b)
        assert expected_b.x.dtype == actual_b.x.dtype

    def test_broadcast_arrays(self):
        x = DataArray([1, 2], coords=[("a", [-1, -2])], name="x")
        y = DataArray([1, 2], coords=[("b", [3, 4])], name="y")
        x2, y2 = broadcast(x, y)
        expected_coords = [("a", [-1, -2]), ("b", [3, 4])]
        expected_x2 = DataArray([[1, 1], [2, 2]], expected_coords, name="x")
        expected_y2 = DataArray([[1, 2], [1, 2]], expected_coords, name="y")
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)

        x = DataArray(np.random.randn(2, 3), dims=["a", "b"])
        y = DataArray(np.random.randn(3, 2), dims=["b", "a"])
        x2, y2 = broadcast(x, y)
        expected_x2 = x
        expected_y2 = y.T
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)

    def test_broadcast_arrays_misaligned(self):
        # broadcast on misaligned coords must auto-align
        x = DataArray([[1, 2], [3, 4]], coords=[("a", [-1, -2]), ("b", [3, 4])])
        y = DataArray([1, 2], coords=[("a", [-1, 20])])
        expected_x2 = DataArray(
            [[3, 4], [1, 2], [np.nan, np.nan]],
            coords=[("a", [-2, -1, 20]), ("b", [3, 4])],
        )
        expected_y2 = DataArray(
            [[np.nan, np.nan], [1, 1], [2, 2]],
            coords=[("a", [-2, -1, 20]), ("b", [3, 4])],
        )
        x2, y2 = broadcast(x, y)
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)

    def test_broadcast_arrays_nocopy(self):
        # Test that input data is not copied over in case
        # no alteration is needed
        x = DataArray([1, 2], coords=[("a", [-1, -2])], name="x")
        y = DataArray(3, name="y")
        expected_x2 = DataArray([1, 2], coords=[("a", [-1, -2])], name="x")
        expected_y2 = DataArray([3, 3], coords=[("a", [-1, -2])], name="y")

        x2, y2 = broadcast(x, y)
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)
        assert source_ndarray(x2.data) is source_ndarray(x.data)

        # single-element broadcast (trivial case)
        (x2,) = broadcast(x)
        assert_identical(x, x2)
        assert source_ndarray(x2.data) is source_ndarray(x.data)

    def test_broadcast_arrays_exclude(self):
        x = DataArray([[1, 2], [3, 4]], coords=[("a", [-1, -2]), ("b", [3, 4])])
        y = DataArray([1, 2], coords=[("a", [-1, 20])])
        z = DataArray(5, coords={"b": 5})

        x2, y2, z2 = broadcast(x, y, z, exclude=["b"])
        expected_x2 = DataArray(
            [[3, 4], [1, 2], [np.nan, np.nan]],
            coords=[("a", [-2, -1, 20]), ("b", [3, 4])],
        )
        expected_y2 = DataArray([np.nan, 1, 2], coords=[("a", [-2, -1, 20])])
        expected_z2 = DataArray(
            [5, 5, 5], dims=["a"], coords={"a": [-2, -1, 20], "b": 5}
        )
        assert_identical(expected_x2, x2)
        assert_identical(expected_y2, y2)
        assert_identical(expected_z2, z2)

    def test_broadcast_coordinates(self):
        # regression test for GH649
        ds = Dataset({"a": (["x", "y"], np.ones((5, 6)))})
        x_bc, y_bc, a_bc = broadcast(ds.x, ds.y, ds.a)
        assert_identical(ds.a, a_bc)

        X, Y = np.meshgrid(np.arange(5), np.arange(6), indexing="ij")
        exp_x = DataArray(X, dims=["x", "y"], name="x")
        exp_y = DataArray(Y, dims=["x", "y"], name="y")
        assert_identical(exp_x, x_bc)
        assert_identical(exp_y, y_bc)

    def test_to_pandas(self):
        # 0d
        actual = DataArray(42).to_pandas()
        expected = np.array(42)
        assert_array_equal(actual, expected)

        # 1d
        values = np.random.randn(3)
        index = pd.Index(["a", "b", "c"], name="x")
        da = DataArray(values, coords=[index])
        actual = da.to_pandas()
        assert_array_equal(actual.values, values)
        assert_array_equal(actual.index, index)
        assert_array_equal(actual.index.name, "x")

        # 2d
        values = np.random.randn(3, 2)
        da = DataArray(
            values, coords=[("x", ["a", "b", "c"]), ("y", [0, 1])], name="foo"
        )
        actual = da.to_pandas()
        assert_array_equal(actual.values, values)
        assert_array_equal(actual.index, ["a", "b", "c"])
        assert_array_equal(actual.columns, [0, 1])

        # roundtrips
        for shape in [(3,), (3, 4)]:
            dims = list("abc")[: len(shape)]
            da = DataArray(np.random.randn(*shape), dims=dims)
            roundtripped = DataArray(da.to_pandas()).drop_vars(dims)
            assert_identical(da, roundtripped)

        with pytest.raises(ValueError, match=r"cannot convert"):
            DataArray(np.random.randn(1, 2, 3, 4, 5)).to_pandas()

    def test_to_dataframe(self):
        # regression test for #260
        arr_np = np.random.randn(3, 4)

        arr = DataArray(arr_np, [("B", [1, 2, 3]), ("A", list("cdef"))], name="foo")
        expected = arr.to_series()
        actual = arr.to_dataframe()["foo"]
        assert_array_equal(expected.values, actual.values)
        assert_array_equal(expected.name, actual.name)
        assert_array_equal(expected.index.values, actual.index.values)

        actual = arr.to_dataframe(dim_order=["A", "B"])["foo"]
        assert_array_equal(arr_np.transpose().reshape(-1), actual.values)

        # regression test for coords with different dimensions
        arr.coords["C"] = ("B", [-1, -2, -3])
        expected = arr.to_series().to_frame()
        expected["C"] = [-1] * 4 + [-2] * 4 + [-3] * 4
        expected = expected[["C", "foo"]]
        actual = arr.to_dataframe()
        assert_array_equal(expected.values, actual.values)
        assert_array_equal(expected.columns.values, actual.columns.values)
        assert_array_equal(expected.index.values, actual.index.values)

        with pytest.raises(ValueError, match="does not match the set of dimensions"):
            arr.to_dataframe(dim_order=["B", "A", "C"])

        with pytest.raises(ValueError, match=r"cannot convert a scalar"):
            arr.sel(A="c", B=2).to_dataframe()

        arr.name = None  # unnamed
        with pytest.raises(ValueError, match=r"unnamed"):
            arr.to_dataframe()

    def test_to_dataframe_multiindex(self):
        # regression test for #3008
        arr_np = np.random.randn(4, 3)

        mindex = pd.MultiIndex.from_product([[1, 2], list("ab")], names=["A", "B"])

        arr = DataArray(arr_np, [("MI", mindex), ("C", [5, 6, 7])], name="foo")

        actual = arr.to_dataframe()
        assert_array_equal(actual["foo"].values, arr_np.flatten())
        assert_array_equal(actual.index.names, list("ABC"))
        assert_array_equal(actual.index.levels[0], [1, 2])
        assert_array_equal(actual.index.levels[1], ["a", "b"])
        assert_array_equal(actual.index.levels[2], [5, 6, 7])

    def test_to_dataframe_0length(self):
        # regression test for #3008
        arr_np = np.random.randn(4, 0)

        mindex = pd.MultiIndex.from_product([[1, 2], list("ab")], names=["A", "B"])

        arr = DataArray(arr_np, [("MI", mindex), ("C", [])], name="foo")

        actual = arr.to_dataframe()
        assert len(actual) == 0
        assert_array_equal(actual.index.names, list("ABC"))

    def test_to_pandas_name_matches_coordinate(self):
        # coordinate with same name as array
        arr = DataArray([1, 2, 3], dims="x", name="x")
        series = arr.to_series()
        assert_array_equal([1, 2, 3], series.values)
        assert_array_equal([0, 1, 2], series.index.values)
        assert "x" == series.name
        assert "x" == series.index.name

        frame = arr.to_dataframe()
        expected = series.to_frame()
        assert expected.equals(frame)

    def test_to_and_from_series(self):
        expected = self.dv.to_dataframe()["foo"]
        actual = self.dv.to_series()
        assert_array_equal(expected.values, actual.values)
        assert_array_equal(expected.index.values, actual.index.values)
        assert "foo" == actual.name
        # test roundtrip
        assert_identical(self.dv, DataArray.from_series(actual).drop_vars(["x", "y"]))
        # test name is None
        actual.name = None
        expected_da = self.dv.rename(None)
        assert_identical(
            expected_da, DataArray.from_series(actual).drop_vars(["x", "y"])
        )

    def test_from_series_multiindex(self):
        # GH:3951
        df = pd.DataFrame({"B": [1, 2, 3], "A": [4, 5, 6]})
        df = df.rename_axis("num").rename_axis("alpha", axis=1)
        actual = df.stack("alpha").to_xarray()
        assert (actual.sel(alpha="B") == [1, 2, 3]).all()
        assert (actual.sel(alpha="A") == [4, 5, 6]).all()

    @requires_sparse
    def test_from_series_sparse(self):
        import sparse

        series = pd.Series([1, 2], index=[("a", 1), ("b", 2)])

        actual_sparse = DataArray.from_series(series, sparse=True)
        actual_dense = DataArray.from_series(series, sparse=False)

        assert isinstance(actual_sparse.data, sparse.COO)
        actual_sparse.data = actual_sparse.data.todense()
        assert_identical(actual_sparse, actual_dense)

    @requires_sparse
    def test_from_multiindex_series_sparse(self):
        # regression test for GH4019
        import sparse

        idx = pd.MultiIndex.from_product([np.arange(3), np.arange(5)], names=["a", "b"])
        series = pd.Series(np.random.RandomState(0).random(len(idx)), index=idx).sample(
            n=5, random_state=3
        )

        dense = DataArray.from_series(series, sparse=False)
        expected_coords = sparse.COO.from_numpy(dense.data, np.nan).coords

        actual_sparse = xr.DataArray.from_series(series, sparse=True)
        actual_coords = actual_sparse.data.coords

        np.testing.assert_equal(actual_coords, expected_coords)

    def test_to_and_from_empty_series(self):
        # GH697
        expected = pd.Series([], dtype=np.float64)
        da = DataArray.from_series(expected)
        assert len(da) == 0
        actual = da.to_series()
        assert len(actual) == 0
        assert expected.equals(actual)

    def test_series_categorical_index(self):
        # regression test for GH700
        if not hasattr(pd, "CategoricalIndex"):
            pytest.skip("requires pandas with CategoricalIndex")

        s = pd.Series(np.arange(5), index=pd.CategoricalIndex(list("aabbc")))
        arr = DataArray(s)
        assert "'a'" in repr(arr)  # should not error

    def test_to_and_from_dict(self):
        array = DataArray(
            np.random.randn(2, 3), {"x": ["a", "b"]}, ["x", "y"], name="foo"
        )
        expected = {
            "name": "foo",
            "dims": ("x", "y"),
            "data": array.values.tolist(),
            "attrs": {},
            "coords": {"x": {"dims": ("x",), "data": ["a", "b"], "attrs": {}}},
        }
        actual = array.to_dict()

        # check that they are identical
        assert expected == actual

        # check roundtrip
        assert_identical(array, DataArray.from_dict(actual))

        # a more bare bones representation still roundtrips
        d = {
            "name": "foo",
            "dims": ("x", "y"),
            "data": array.values.tolist(),
            "coords": {"x": {"dims": "x", "data": ["a", "b"]}},
        }
        assert_identical(array, DataArray.from_dict(d))

        # and the most bare bones representation still roundtrips
        d = {"name": "foo", "dims": ("x", "y"), "data": array.values}
        assert_identical(array.drop_vars("x"), DataArray.from_dict(d))

        # missing a dims in the coords
        d = {
            "dims": ("x", "y"),
            "data": array.values,
            "coords": {"x": {"data": ["a", "b"]}},
        }
        with pytest.raises(
            ValueError,
            match=r"cannot convert dict when coords are missing the key 'dims'",
        ):
            DataArray.from_dict(d)

        # this one is missing some necessary information
        d = {"dims": ("t")}
        with pytest.raises(
            ValueError, match=r"cannot convert dict without the key 'data'"
        ):
            DataArray.from_dict(d)

        # check the data=False option
        expected_no_data = expected.copy()
        del expected_no_data["data"]
        del expected_no_data["coords"]["x"]["data"]
        endiantype = "<U1" if sys.byteorder == "little" else ">U1"
        expected_no_data["coords"]["x"].update({"dtype": endiantype, "shape": (2,)})
        expected_no_data.update({"dtype": "float64", "shape": (2, 3)})
        actual_no_data = array.to_dict(data=False)
        assert expected_no_data == actual_no_data

    def test_to_and_from_dict_with_time_dim(self):
        x = np.random.randn(10, 3)
        t = pd.date_range("20130101", periods=10)
        lat = [77.7, 83.2, 76]
        da = DataArray(x, {"t": t, "lat": lat}, dims=["t", "lat"])
        roundtripped = DataArray.from_dict(da.to_dict())
        assert_identical(da, roundtripped)

    def test_to_and_from_dict_with_nan_nat(self):
        y = np.random.randn(10, 3)
        y[2] = np.nan
        t = pd.Series(pd.date_range("20130101", periods=10))
        t[2] = np.nan
        lat = [77.7, 83.2, 76]
        da = DataArray(y, {"t": t, "lat": lat}, dims=["t", "lat"])
        roundtripped = DataArray.from_dict(da.to_dict())
        assert_identical(da, roundtripped)

    def test_to_dict_with_numpy_attrs(self):
        # this doesn't need to roundtrip
        x = np.random.randn(10, 3)
        t = list("abcdefghij")
        lat = [77.7, 83.2, 76]
        attrs = {
            "created": np.float64(1998),
            "coords": np.array([37, -110.1, 100]),
            "maintainer": "bar",
        }
        da = DataArray(x, {"t": t, "lat": lat}, dims=["t", "lat"], attrs=attrs)
        expected_attrs = {
            "created": attrs["created"].item(),
            "coords": attrs["coords"].tolist(),
            "maintainer": "bar",
        }
        actual = da.to_dict()

        # check that they are identical
        assert expected_attrs == actual["attrs"]

    def test_to_masked_array(self):
        rs = np.random.RandomState(44)
        x = rs.random_sample(size=(10, 20))
        x_masked = np.ma.masked_where(x < 0.5, x)
        da = DataArray(x_masked)

        # Test round trip
        x_masked_2 = da.to_masked_array()
        da_2 = DataArray(x_masked_2)
        assert_array_equal(x_masked, x_masked_2)
        assert_equal(da, da_2)

        da_masked_array = da.to_masked_array(copy=True)
        assert isinstance(da_masked_array, np.ma.MaskedArray)
        # Test masks
        assert_array_equal(da_masked_array.mask, x_masked.mask)
        # Test that mask is unpacked correctly
        assert_array_equal(da.values, x_masked.filled(np.nan))
        # Test that the underlying data (including nans) hasn't changed
        assert_array_equal(da_masked_array, x_masked.filled(np.nan))

        # Test that copy=False gives access to values
        masked_array = da.to_masked_array(copy=False)
        masked_array[0, 0] = 10.0
        assert masked_array[0, 0] == 10.0
        assert da[0, 0].values == 10.0
        assert masked_array.base is da.values
        assert isinstance(masked_array, np.ma.MaskedArray)

        # Test with some odd arrays
        for v in [4, np.nan, True, "4", "four"]:
            da = DataArray(v)
            ma = da.to_masked_array()
            assert isinstance(ma, np.ma.MaskedArray)

        # Fix GH issue 684 - masked arrays mask should be an array not a scalar
        N = 4
        v = range(N)
        da = DataArray(v)
        ma = da.to_masked_array()
        assert len(ma.mask) == N

    def test_to_and_from_cdms2_classic(self):
        """Classic with 1D axes"""
        pytest.importorskip("cdms2")

        original = DataArray(
            np.arange(6).reshape(2, 3),
            [
                ("distance", [-2, 2], {"units": "meters"}),
                ("time", pd.date_range("2000-01-01", periods=3)),
            ],
            name="foo",
            attrs={"baz": 123},
        )
        expected_coords = [
            IndexVariable("distance", [-2, 2]),
            IndexVariable("time", [0, 1, 2]),
        ]
        actual = original.to_cdms2()
        assert_array_equal(actual.asma(), original)
        assert actual.id == original.name
        assert tuple(actual.getAxisIds()) == original.dims
        for axis, coord in zip(actual.getAxisList(), expected_coords):
            assert axis.id == coord.name
            assert_array_equal(axis, coord.values)
        assert actual.baz == original.attrs["baz"]

        component_times = actual.getAxis(1).asComponentTime()
        assert len(component_times) == 3
        assert str(component_times[0]) == "2000-1-1 0:0:0.0"

        roundtripped = DataArray.from_cdms2(actual)
        assert_identical(original, roundtripped)

        back = from_cdms2(actual)
        assert original.dims == back.dims
        assert original.coords.keys() == back.coords.keys()
        for coord_name in original.coords.keys():
            assert_array_equal(original.coords[coord_name], back.coords[coord_name])

    def test_to_and_from_cdms2_sgrid(self):
        """Curvilinear (structured) grid

        The rectangular grid case is covered by the classic case
        """
        pytest.importorskip("cdms2")

        lonlat = np.mgrid[:3, :4]
        lon = DataArray(lonlat[1], dims=["y", "x"], name="lon")
        lat = DataArray(lonlat[0], dims=["y", "x"], name="lat")
        x = DataArray(np.arange(lon.shape[1]), dims=["x"], name="x")
        y = DataArray(np.arange(lon.shape[0]), dims=["y"], name="y")
        original = DataArray(
            lonlat.sum(axis=0),
            dims=["y", "x"],
            coords=dict(x=x, y=y, lon=lon, lat=lat),
            name="sst",
        )
        actual = original.to_cdms2()
        assert tuple(actual.getAxisIds()) == original.dims
        assert_array_equal(original.coords["lon"], actual.getLongitude().asma())
        assert_array_equal(original.coords["lat"], actual.getLatitude().asma())

        back = from_cdms2(actual)
        assert original.dims == back.dims
        assert set(original.coords.keys()) == set(back.coords.keys())
        assert_array_equal(original.coords["lat"], back.coords["lat"])
        assert_array_equal(original.coords["lon"], back.coords["lon"])

    def test_to_and_from_cdms2_ugrid(self):
        """Unstructured grid"""
        pytest.importorskip("cdms2")

        lon = DataArray(np.random.uniform(size=5), dims=["cell"], name="lon")
        lat = DataArray(np.random.uniform(size=5), dims=["cell"], name="lat")
        cell = DataArray(np.arange(5), dims=["cell"], name="cell")
        original = DataArray(
            np.arange(5), dims=["cell"], coords={"lon": lon, "lat": lat, "cell": cell}
        )
        actual = original.to_cdms2()
        assert tuple(actual.getAxisIds()) == original.dims
        assert_array_equal(original.coords["lon"], actual.getLongitude().getValue())
        assert_array_equal(original.coords["lat"], actual.getLatitude().getValue())

        back = from_cdms2(actual)
        assert set(original.dims) == set(back.dims)
        assert set(original.coords.keys()) == set(back.coords.keys())
        assert_array_equal(original.coords["lat"], back.coords["lat"])
        assert_array_equal(original.coords["lon"], back.coords["lon"])

    def test_to_dataset_whole(self):
        unnamed = DataArray([1, 2], dims="x")
        with pytest.raises(ValueError, match=r"unable to convert unnamed"):
            unnamed.to_dataset()

        actual = unnamed.to_dataset(name="foo")
        expected = Dataset({"foo": ("x", [1, 2])})
        assert_identical(expected, actual)

        named = DataArray([1, 2], dims="x", name="foo", attrs={"y": "testattr"})
        actual = named.to_dataset()
        expected = Dataset({"foo": ("x", [1, 2], {"y": "testattr"})})
        assert_identical(expected, actual)

        # Test promoting attrs
        actual = named.to_dataset(promote_attrs=True)
        expected = Dataset(
            {"foo": ("x", [1, 2], {"y": "testattr"})}, attrs={"y": "testattr"}
        )
        assert_identical(expected, actual)

        with pytest.raises(TypeError):
            actual = named.to_dataset("bar")

    def test_to_dataset_split(self):
        array = DataArray([1, 2, 3], coords=[("x", list("abc"))], attrs={"a": 1})
        expected = Dataset({"a": 1, "b": 2, "c": 3}, attrs={"a": 1})
        actual = array.to_dataset("x")
        assert_identical(expected, actual)

        with pytest.raises(TypeError):
            array.to_dataset("x", name="foo")

        roundtripped = actual.to_array(dim="x")
        assert_identical(array, roundtripped)

        array = DataArray([1, 2, 3], dims="x")
        expected = Dataset({0: 1, 1: 2, 2: 3})
        actual = array.to_dataset("x")
        assert_identical(expected, actual)

    def test_to_dataset_retains_keys(self):

        # use dates as convenient non-str objects. Not a specific date test
        import datetime

        dates = [datetime.date(2000, 1, d) for d in range(1, 4)]

        array = DataArray([1, 2, 3], coords=[("x", dates)], attrs={"a": 1})

        # convert to dateset and back again
        result = array.to_dataset("x").to_array(dim="x")

        assert_equal(array, result)

    def test__title_for_slice(self):
        array = DataArray(
            np.ones((4, 3, 2)),
            dims=["a", "b", "c"],
            coords={"a": range(4), "b": range(3), "c": range(2)},
        )
        assert "" == array._title_for_slice()
        assert "c = 0" == array.isel(c=0)._title_for_slice()
        title = array.isel(b=1, c=0)._title_for_slice()
        assert "b = 1, c = 0" == title or "c = 0, b = 1" == title

        a2 = DataArray(np.ones((4, 1)), dims=["a", "b"])
        assert "" == a2._title_for_slice()

    def test__title_for_slice_truncate(self):
        array = DataArray(np.ones(4))
        array.coords["a"] = "a" * 100
        array.coords["b"] = "b" * 100

        nchar = 80
        title = array._title_for_slice(truncate=nchar)

        assert nchar == len(title)
        assert title.endswith("...")

    def test_dataarray_diff_n1(self):
        da = DataArray(np.random.randn(3, 4), dims=["x", "y"])
        actual = da.diff("y")
        expected = DataArray(np.diff(da.values, axis=1), dims=["x", "y"])
        assert_equal(expected, actual)

    def test_coordinate_diff(self):
        # regression test for GH634
        arr = DataArray(range(0, 20, 2), dims=["lon"], coords=[range(10)])
        lon = arr.coords["lon"]
        expected = DataArray([1] * 9, dims=["lon"], coords=[range(1, 10)], name="lon")
        actual = lon.diff("lon")
        assert_equal(expected, actual)

    @pytest.mark.parametrize("offset", [-5, 0, 1, 2])
    @pytest.mark.parametrize("fill_value, dtype", [(2, int), (dtypes.NA, float)])
    def test_shift(self, offset, fill_value, dtype):
        arr = DataArray([1, 2, 3], dims="x")
        actual = arr.shift(x=1, fill_value=fill_value)
        if fill_value == dtypes.NA:
            # if we supply the default, we expect the missing value for a
            # float array
            fill_value = np.nan
        expected = DataArray([fill_value, 1, 2], dims="x")
        assert_identical(expected, actual)
        assert actual.dtype == dtype

        arr = DataArray([1, 2, 3], [("x", ["a", "b", "c"])])
        expected = DataArray(arr.to_pandas().shift(offset))
        actual = arr.shift(x=offset)
        assert_identical(expected, actual)

    def test_roll_coords(self):
        arr = DataArray([1, 2, 3], coords={"x": range(3)}, dims="x")
        actual = arr.roll(x=1, roll_coords=True)
        expected = DataArray([3, 1, 2], coords=[("x", [2, 0, 1])])
        assert_identical(expected, actual)

    def test_roll_no_coords(self):
        arr = DataArray([1, 2, 3], coords={"x": range(3)}, dims="x")
        actual = arr.roll(x=1, roll_coords=False)
        expected = DataArray([3, 1, 2], coords=[("x", [0, 1, 2])])
        assert_identical(expected, actual)

    def test_roll_coords_none(self):
        arr = DataArray([1, 2, 3], coords={"x": range(3)}, dims="x")

        with pytest.warns(FutureWarning):
            actual = arr.roll(x=1, roll_coords=None)

        expected = DataArray([3, 1, 2], coords=[("x", [2, 0, 1])])
        assert_identical(expected, actual)

    def test_copy_with_data(self):
        orig = DataArray(
            np.random.random(size=(2, 2)),
            dims=("x", "y"),
            attrs={"attr1": "value1"},
            coords={"x": [4, 3]},
            name="helloworld",
        )
        new_data = np.arange(4).reshape(2, 2)
        actual = orig.copy(data=new_data)
        expected = orig.copy()
        expected.data = new_data
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
        dataarray deep copied from another."""
        da = xr.DataArray(
            np.ones([2, 2, 2]),
            coords={"a": [1, 2], "b": ["x", "y"], "c": [0, 1]},
            dims=["a", "b", "c"],
        )
        da_cp = da.copy(deep)
        da_cp["a"].data[0] = 999

        expected_cp = xr.DataArray(
            xr.IndexVariable("a", np.array([999, 2])),
            coords={"a": [999, 2]},
            dims=["a"],
        )
        assert_identical(da_cp["a"], expected_cp)

        assert_identical(da["a"], expected_orig)

    def test_real_and_imag(self):
        array = DataArray(1 + 2j)
        assert_identical(array.real, DataArray(1))
        assert_identical(array.imag, DataArray(2))

    def test_setattr_raises(self):
        array = DataArray(0, coords={"scalar": 1}, attrs={"foo": "bar"})
        with pytest.raises(AttributeError, match=r"cannot set attr"):
            array.scalar = 2
        with pytest.raises(AttributeError, match=r"cannot set attr"):
            array.foo = 2
        with pytest.raises(AttributeError, match=r"cannot set attr"):
            array.other = 2

    def test_full_like(self):
        # For more thorough tests, see test_variable.py
        da = DataArray(
            np.random.random(size=(2, 2)),
            dims=("x", "y"),
            attrs={"attr1": "value1"},
            coords={"x": [4, 3]},
            name="helloworld",
        )

        actual = full_like(da, 2)
        expect = da.copy(deep=True)
        expect.values = [[2.0, 2.0], [2.0, 2.0]]
        assert_identical(expect, actual)

        # override dtype
        actual = full_like(da, fill_value=True, dtype=bool)
        expect.values = [[True, True], [True, True]]
        assert expect.dtype == bool
        assert_identical(expect, actual)

        with pytest.raises(ValueError, match="'dtype' cannot be dict-like"):
            full_like(da, fill_value=True, dtype={"x": bool})

    def test_dot(self):
        x = np.linspace(-3, 3, 6)
        y = np.linspace(-3, 3, 5)
        z = range(4)
        da_vals = np.arange(6 * 5 * 4).reshape((6, 5, 4))
        da = DataArray(da_vals, coords=[x, y, z], dims=["x", "y", "z"])

        dm_vals = range(4)
        dm = DataArray(dm_vals, coords=[z], dims=["z"])

        # nd dot 1d
        actual = da.dot(dm)
        expected_vals = np.tensordot(da_vals, dm_vals, [2, 0])
        expected = DataArray(expected_vals, coords=[x, y], dims=["x", "y"])
        assert_equal(expected, actual)

        # all shared dims
        actual = da.dot(da)
        expected_vals = np.tensordot(da_vals, da_vals, axes=([0, 1, 2], [0, 1, 2]))
        expected = DataArray(expected_vals)
        assert_equal(expected, actual)

        # multiple shared dims
        dm_vals = np.arange(20 * 5 * 4).reshape((20, 5, 4))
        j = np.linspace(-3, 3, 20)
        dm = DataArray(dm_vals, coords=[j, y, z], dims=["j", "y", "z"])
        actual = da.dot(dm)
        expected_vals = np.tensordot(da_vals, dm_vals, axes=([1, 2], [1, 2]))
        expected = DataArray(expected_vals, coords=[x, j], dims=["x", "j"])
        assert_equal(expected, actual)

        # Ellipsis: all dims are shared
        actual = da.dot(da, dims=...)
        expected = da.dot(da)
        assert_equal(expected, actual)

        # Ellipsis: not all dims are shared
        actual = da.dot(dm, dims=...)
        expected = da.dot(dm, dims=("j", "x", "y", "z"))
        assert_equal(expected, actual)

        with pytest.raises(NotImplementedError):
            da.dot(dm.to_dataset(name="dm"))
        with pytest.raises(TypeError):
            da.dot(dm.values)

    def test_dot_align_coords(self):
        # GH 3694

        x = np.linspace(-3, 3, 6)
        y = np.linspace(-3, 3, 5)
        z_a = range(4)
        da_vals = np.arange(6 * 5 * 4).reshape((6, 5, 4))
        da = DataArray(da_vals, coords=[x, y, z_a], dims=["x", "y", "z"])

        z_m = range(2, 6)
        dm_vals = range(4)
        dm = DataArray(dm_vals, coords=[z_m], dims=["z"])

        with xr.set_options(arithmetic_join="exact"):
            with pytest.raises(ValueError, match=r"indexes along dimension"):
                da.dot(dm)

        da_aligned, dm_aligned = xr.align(da, dm, join="inner")

        # nd dot 1d
        actual = da.dot(dm)
        expected_vals = np.tensordot(da_aligned.values, dm_aligned.values, [2, 0])
        expected = DataArray(expected_vals, coords=[x, da_aligned.y], dims=["x", "y"])
        assert_equal(expected, actual)

        # multiple shared dims
        dm_vals = np.arange(20 * 5 * 4).reshape((20, 5, 4))
        j = np.linspace(-3, 3, 20)
        dm = DataArray(dm_vals, coords=[j, y, z_m], dims=["j", "y", "z"])
        da_aligned, dm_aligned = xr.align(da, dm, join="inner")
        actual = da.dot(dm)
        expected_vals = np.tensordot(
            da_aligned.values, dm_aligned.values, axes=([1, 2], [1, 2])
        )
        expected = DataArray(expected_vals, coords=[x, j], dims=["x", "j"])
        assert_equal(expected, actual)

    def test_matmul(self):

        # copied from above (could make a fixture)
        x = np.linspace(-3, 3, 6)
        y = np.linspace(-3, 3, 5)
        z = range(4)
        da_vals = np.arange(6 * 5 * 4).reshape((6, 5, 4))
        da = DataArray(da_vals, coords=[x, y, z], dims=["x", "y", "z"])

        result = da @ da
        expected = da.dot(da)
        assert_identical(result, expected)

    def test_matmul_align_coords(self):
        # GH 3694

        x_a = np.arange(6)
        x_b = np.arange(2, 8)
        da_vals = np.arange(6)
        da_a = DataArray(da_vals, coords=[x_a], dims=["x"])
        da_b = DataArray(da_vals, coords=[x_b], dims=["x"])

        # only test arithmetic_join="inner" (=default)
        result = da_a @ da_b
        expected = da_a.dot(da_b)
        assert_identical(result, expected)

        with xr.set_options(arithmetic_join="exact"):
            with pytest.raises(ValueError, match=r"indexes along dimension"):
                da_a @ da_b

    def test_binary_op_propagate_indexes(self):
        # regression test for GH2227
        self.dv["x"] = np.arange(self.dv.sizes["x"])
        expected = self.dv.xindexes["x"]

        actual = (self.dv * 10).xindexes["x"]
        assert expected is actual

        actual = (self.dv > 10).xindexes["x"]
        assert expected is actual

    def test_binary_op_join_setting(self):
        dim = "x"
        align_type = "outer"
        coords_l, coords_r = [0, 1, 2], [1, 2, 3]
        missing_3 = xr.DataArray(coords_l, [(dim, coords_l)])
        missing_0 = xr.DataArray(coords_r, [(dim, coords_r)])
        with xr.set_options(arithmetic_join=align_type):
            actual = missing_0 + missing_3
        missing_0_aligned, missing_3_aligned = xr.align(
            missing_0, missing_3, join=align_type
        )
        expected = xr.DataArray([np.nan, 2, 4, np.nan], [(dim, [0, 1, 2, 3])])
        assert_equal(actual, expected)

    def test_combine_first(self):
        ar0 = DataArray([[0, 0], [0, 0]], [("x", ["a", "b"]), ("y", [-1, 0])])
        ar1 = DataArray([[1, 1], [1, 1]], [("x", ["b", "c"]), ("y", [0, 1])])
        ar2 = DataArray([2], [("x", ["d"])])

        actual = ar0.combine_first(ar1)
        expected = DataArray(
            [[0, 0, np.nan], [0, 0, 1], [np.nan, 1, 1]],
            [("x", ["a", "b", "c"]), ("y", [-1, 0, 1])],
        )
        assert_equal(actual, expected)

        actual = ar1.combine_first(ar0)
        expected = DataArray(
            [[0, 0, np.nan], [0, 1, 1], [np.nan, 1, 1]],
            [("x", ["a", "b", "c"]), ("y", [-1, 0, 1])],
        )
        assert_equal(actual, expected)

        actual = ar0.combine_first(ar2)
        expected = DataArray(
            [[0, 0], [0, 0], [2, 2]], [("x", ["a", "b", "d"]), ("y", [-1, 0])]
        )
        assert_equal(actual, expected)

    def test_sortby(self):
        da = DataArray(
            [[1, 2], [3, 4], [5, 6]], [("x", ["c", "b", "a"]), ("y", [1, 0])]
        )

        sorted1d = DataArray(
            [[5, 6], [3, 4], [1, 2]], [("x", ["a", "b", "c"]), ("y", [1, 0])]
        )

        sorted2d = DataArray(
            [[6, 5], [4, 3], [2, 1]], [("x", ["a", "b", "c"]), ("y", [0, 1])]
        )

        expected = sorted1d
        dax = DataArray([100, 99, 98], [("x", ["c", "b", "a"])])
        actual = da.sortby(dax)
        assert_equal(actual, expected)

        # test descending order sort
        actual = da.sortby(dax, ascending=False)
        assert_equal(actual, da)

        # test alignment (fills in nan for 'c')
        dax_short = DataArray([98, 97], [("x", ["b", "a"])])
        actual = da.sortby(dax_short)
        assert_equal(actual, expected)

        # test multi-dim sort by 1D dataarray values
        expected = sorted2d
        dax = DataArray([100, 99, 98], [("x", ["c", "b", "a"])])
        day = DataArray([90, 80], [("y", [1, 0])])
        actual = da.sortby([day, dax])
        assert_equal(actual, expected)

        expected = sorted1d
        actual = da.sortby("x")
        assert_equal(actual, expected)

        expected = sorted2d
        actual = da.sortby(["x", "y"])
        assert_equal(actual, expected)

    @requires_bottleneck
    def test_rank(self):
        # floats
        ar = DataArray([[3, 4, np.nan, 1]])
        expect_0 = DataArray([[1, 1, np.nan, 1]])
        expect_1 = DataArray([[2, 3, np.nan, 1]])
        assert_equal(ar.rank("dim_0"), expect_0)
        assert_equal(ar.rank("dim_1"), expect_1)
        # int
        x = DataArray([3, 2, 1])
        assert_equal(x.rank("dim_0"), x)
        # str
        y = DataArray(["c", "b", "a"])
        assert_equal(y.rank("dim_0"), x)

        x = DataArray([3.0, 1.0, np.nan, 2.0, 4.0], dims=("z",))
        y = DataArray([0.75, 0.25, np.nan, 0.5, 1.0], dims=("z",))
        assert_equal(y.rank("z", pct=True), y)

    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("use_datetime", [True, False])
    def test_polyfit(self, use_dask, use_datetime):
        if use_dask and not has_dask:
            pytest.skip("requires dask")
        xcoord = xr.DataArray(
            pd.date_range("1970-01-01", freq="D", periods=10), dims=("x",), name="x"
        )
        x = xr.core.missing.get_clean_interp_index(xcoord, "x")
        if not use_datetime:
            xcoord = x

        da_raw = DataArray(
            np.stack(
                (10 + 1e-15 * x + 2e-28 * x ** 2, 30 + 2e-14 * x + 1e-29 * x ** 2)
            ),
            dims=("d", "x"),
            coords={"x": xcoord, "d": [0, 1]},
        )

        if use_dask:
            da = da_raw.chunk({"d": 1})
        else:
            da = da_raw

        out = da.polyfit("x", 2)
        expected = DataArray(
            [[2e-28, 1e-15, 10], [1e-29, 2e-14, 30]],
            dims=("d", "degree"),
            coords={"degree": [2, 1, 0], "d": [0, 1]},
        ).T
        assert_allclose(out.polyfit_coefficients, expected, rtol=1e-3)

        # Full output and deficient rank
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            out = da.polyfit("x", 12, full=True)
            assert out.polyfit_residuals.isnull().all()

        # With NaN
        da_raw[0, 1:3] = np.nan
        if use_dask:
            da = da_raw.chunk({"d": 1})
        else:
            da = da_raw
        out = da.polyfit("x", 2, skipna=True, cov=True)
        assert_allclose(out.polyfit_coefficients, expected, rtol=1e-3)
        assert "polyfit_covariance" in out

        # Skipna + Full output
        out = da.polyfit("x", 2, skipna=True, full=True)
        assert_allclose(out.polyfit_coefficients, expected, rtol=1e-3)
        assert out.x_matrix_rank == 3
        np.testing.assert_almost_equal(out.polyfit_residuals, [0, 0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            out = da.polyfit("x", 8, full=True)
            np.testing.assert_array_equal(out.polyfit_residuals.isnull(), [True, False])

    def test_pad_constant(self):
        ar = DataArray(np.arange(3 * 4 * 5).reshape(3, 4, 5))
        actual = ar.pad(dim_0=(1, 3))
        expected = DataArray(
            np.pad(
                np.arange(3 * 4 * 5).reshape(3, 4, 5).astype(np.float32),
                mode="constant",
                pad_width=((1, 3), (0, 0), (0, 0)),
                constant_values=np.nan,
            )
        )
        assert actual.shape == (7, 4, 5)
        assert_identical(actual, expected)

        ar = xr.DataArray([9], dims="x")

        actual = ar.pad(x=1)
        expected = xr.DataArray([np.NaN, 9, np.NaN], dims="x")
        assert_identical(actual, expected)

        actual = ar.pad(x=1, constant_values=1.23456)
        expected = xr.DataArray([1, 9, 1], dims="x")
        assert_identical(actual, expected)

        if LooseVersion(np.__version__) >= "1.20":
            with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
                ar.pad(x=1, constant_values=np.NaN)
        else:
            actual = ar.pad(x=1, constant_values=np.NaN)
            expected = xr.DataArray(
                [-9223372036854775808, 9, -9223372036854775808], dims="x"
            )
            assert_identical(actual, expected)

    def test_pad_coords(self):
        ar = DataArray(
            np.arange(3 * 4 * 5).reshape(3, 4, 5),
            [("x", np.arange(3)), ("y", np.arange(4)), ("z", np.arange(5))],
        )
        actual = ar.pad(x=(1, 3), constant_values=1)
        expected = DataArray(
            np.pad(
                np.arange(3 * 4 * 5).reshape(3, 4, 5),
                mode="constant",
                pad_width=((1, 3), (0, 0), (0, 0)),
                constant_values=1,
            ),
            [
                (
                    "x",
                    np.pad(
                        np.arange(3).astype(np.float32),
                        mode="constant",
                        pad_width=(1, 3),
                        constant_values=np.nan,
                    ),
                ),
                ("y", np.arange(4)),
                ("z", np.arange(5)),
            ],
        )
        assert_identical(actual, expected)

    @pytest.mark.parametrize("mode", ("minimum", "maximum", "mean", "median"))
    @pytest.mark.parametrize(
        "stat_length", (None, 3, (1, 3), {"dim_0": (2, 1), "dim_2": (4, 2)})
    )
    def test_pad_stat_length(self, mode, stat_length):
        ar = DataArray(np.arange(3 * 4 * 5).reshape(3, 4, 5))
        actual = ar.pad(dim_0=(1, 3), dim_2=(2, 2), mode=mode, stat_length=stat_length)
        if isinstance(stat_length, dict):
            stat_length = (stat_length["dim_0"], (4, 4), stat_length["dim_2"])
        expected = DataArray(
            np.pad(
                np.arange(3 * 4 * 5).reshape(3, 4, 5),
                pad_width=((1, 3), (0, 0), (2, 2)),
                mode=mode,
                stat_length=stat_length,
            )
        )
        assert actual.shape == (7, 4, 9)
        assert_identical(actual, expected)

    @pytest.mark.parametrize(
        "end_values", (None, 3, (3, 5), {"dim_0": (2, 1), "dim_2": (4, 2)})
    )
    def test_pad_linear_ramp(self, end_values):
        ar = DataArray(np.arange(3 * 4 * 5).reshape(3, 4, 5))
        actual = ar.pad(
            dim_0=(1, 3), dim_2=(2, 2), mode="linear_ramp", end_values=end_values
        )
        if end_values is None:
            end_values = 0
        elif isinstance(end_values, dict):
            end_values = (end_values["dim_0"], (4, 4), end_values["dim_2"])
        expected = DataArray(
            np.pad(
                np.arange(3 * 4 * 5).reshape(3, 4, 5),
                pad_width=((1, 3), (0, 0), (2, 2)),
                mode="linear_ramp",
                end_values=end_values,
            )
        )
        assert actual.shape == (7, 4, 9)
        assert_identical(actual, expected)

    @pytest.mark.parametrize("mode", ("reflect", "symmetric"))
    @pytest.mark.parametrize("reflect_type", (None, "even", "odd"))
    def test_pad_reflect(self, mode, reflect_type):

        ar = DataArray(np.arange(3 * 4 * 5).reshape(3, 4, 5))
        actual = ar.pad(
            dim_0=(1, 3), dim_2=(2, 2), mode=mode, reflect_type=reflect_type
        )
        np_kwargs = {
            "array": np.arange(3 * 4 * 5).reshape(3, 4, 5),
            "pad_width": ((1, 3), (0, 0), (2, 2)),
            "mode": mode,
        }
        # numpy does not support reflect_type=None
        if reflect_type is not None:
            np_kwargs["reflect_type"] = reflect_type
        expected = DataArray(np.pad(**np_kwargs))

        assert actual.shape == (7, 4, 9)
        assert_identical(actual, expected)

    @pytest.mark.parametrize("parser", ["pandas", "python"])
    @pytest.mark.parametrize(
        "engine", ["python", None, pytest.param("numexpr", marks=[requires_numexpr])]
    )
    @pytest.mark.parametrize(
        "backend", ["numpy", pytest.param("dask", marks=[requires_dask])]
    )
    def test_query(self, backend, engine, parser):
        """Test querying a dataset."""

        # setup test data
        np.random.seed(42)
        a = np.arange(0, 10, 1)
        b = np.random.randint(0, 100, size=10)
        c = np.linspace(0, 1, 20)
        d = np.random.choice(["foo", "bar", "baz"], size=30, replace=True).astype(
            object
        )
        if backend == "numpy":
            aa = DataArray(data=a, dims=["x"], name="a")
            bb = DataArray(data=b, dims=["x"], name="b")
            cc = DataArray(data=c, dims=["y"], name="c")
            dd = DataArray(data=d, dims=["z"], name="d")

        elif backend == "dask":
            import dask.array as da

            aa = DataArray(data=da.from_array(a, chunks=3), dims=["x"], name="a")
            bb = DataArray(data=da.from_array(b, chunks=3), dims=["x"], name="b")
            cc = DataArray(data=da.from_array(c, chunks=7), dims=["y"], name="c")
            dd = DataArray(data=da.from_array(d, chunks=12), dims=["z"], name="d")

        # query single dim, single variable
        actual = aa.query(x="a > 5", engine=engine, parser=parser)
        expect = aa.isel(x=(a > 5))
        assert_identical(expect, actual)

        # query single dim, single variable, via dict
        actual = aa.query(dict(x="a > 5"), engine=engine, parser=parser)
        expect = aa.isel(dict(x=(a > 5)))
        assert_identical(expect, actual)

        # query single dim, single variable
        actual = bb.query(x="b > 50", engine=engine, parser=parser)
        expect = bb.isel(x=(b > 50))
        assert_identical(expect, actual)

        # query single dim, single variable
        actual = cc.query(y="c < .5", engine=engine, parser=parser)
        expect = cc.isel(y=(c < 0.5))
        assert_identical(expect, actual)

        # query single dim, single string variable
        if parser == "pandas":
            # N.B., this query currently only works with the pandas parser
            # xref https://github.com/pandas-dev/pandas/issues/40436
            actual = dd.query(z='d == "bar"', engine=engine, parser=parser)
            expect = dd.isel(z=(d == "bar"))
            assert_identical(expect, actual)

        # test error handling
        with pytest.raises(ValueError):
            aa.query("a > 5")  # must be dict or kwargs
        with pytest.raises(ValueError):
            aa.query(x=(a > 5))  # must be query string
        with pytest.raises(UndefinedVariableError):
            aa.query(x="spam > 50")  # name not present

    @requires_scipy
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_curvefit(self, use_dask):
        if use_dask and not has_dask:
            pytest.skip("requires dask")

        def exp_decay(t, n0, tau=1):
            return n0 * np.exp(-t / tau)

        t = np.arange(0, 5, 0.5)
        da = DataArray(
            np.stack([exp_decay(t, 3, 3), exp_decay(t, 5, 4), np.nan * t], axis=-1),
            dims=("t", "x"),
            coords={"t": t, "x": [0, 1, 2]},
        )
        da[0, 0] = np.nan

        expected = DataArray(
            [[3, 3], [5, 4], [np.nan, np.nan]],
            dims=("x", "param"),
            coords={"x": [0, 1, 2], "param": ["n0", "tau"]},
        )

        if use_dask:
            da = da.chunk({"x": 1})

        fit = da.curvefit(
            coords=[da.t], func=exp_decay, p0={"n0": 4}, bounds={"tau": [2, 6]}
        )
        assert_allclose(fit.curvefit_coefficients, expected, rtol=1e-3)

        da = da.compute()
        fit = da.curvefit(coords="t", func=np.power, reduce_dims="x", param_names=["a"])
        assert "a" in fit.param
        assert "x" not in fit.dims

    def test_curvefit_helpers(self):
        def exp_decay(t, n0, tau=1):
            return n0 * np.exp(-t / tau)

        params, func_args = xr.core.dataset._get_func_args(exp_decay, [])
        assert params == ["n0", "tau"]
        param_defaults, bounds_defaults = xr.core.dataset._initialize_curvefit_params(
            params, {"n0": 4}, {"tau": [5, np.inf]}, func_args
        )
        assert param_defaults == {"n0": 4, "tau": 6}
        assert bounds_defaults == {"n0": (-np.inf, np.inf), "tau": (5, np.inf)}

        param_names = ["a"]
        params, func_args = xr.core.dataset._get_func_args(np.power, param_names)
        assert params == param_names
        with pytest.raises(ValueError):
            xr.core.dataset._get_func_args(np.power, [])


class TestReduce:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.attrs = {"attr1": "value1", "attr2": 2929}


@pytest.mark.parametrize(
    "x, minindex, maxindex, nanindex",
    [
        (np.array([0, 1, 2, 0, -2, -4, 2]), 5, 2, None),
        (np.array([0.0, 1.0, 2.0, 0.0, -2.0, -4.0, 2.0]), 5, 2, None),
        (np.array([1.0, np.NaN, 2.0, np.NaN, -2.0, -4.0, 2.0]), 5, 2, 1),
        (
            np.array([1.0, np.NaN, 2.0, np.NaN, -2.0, -4.0, 2.0]).astype("object"),
            5,
            2,
            1,
        ),
        (np.array([np.NaN, np.NaN]), np.NaN, np.NaN, 0),
        (
            np.array(
                ["2015-12-31", "2020-01-02", "2020-01-01", "2016-01-01"],
                dtype="datetime64[ns]",
            ),
            0,
            1,
            None,
        ),
    ],
)
class TestReduce1D(TestReduce):
    def test_min(self, x, minindex, maxindex, nanindex):
        ar = xr.DataArray(
            x, dims=["x"], coords={"x": np.arange(x.size) * 4}, attrs=self.attrs
        )

        if np.isnan(minindex):
            minindex = 0

        expected0 = ar.isel(x=minindex, drop=True)
        result0 = ar.min(keep_attrs=True)
        assert_identical(result0, expected0)

        result1 = ar.min()
        expected1 = expected0.copy()
        expected1.attrs = {}
        assert_identical(result1, expected1)

        result2 = ar.min(skipna=False)
        if nanindex is not None and ar.dtype.kind != "O":
            expected2 = ar.isel(x=nanindex, drop=True)
            expected2.attrs = {}
        else:
            expected2 = expected1

        assert_identical(result2, expected2)

    def test_max(self, x, minindex, maxindex, nanindex):
        ar = xr.DataArray(
            x, dims=["x"], coords={"x": np.arange(x.size) * 4}, attrs=self.attrs
        )

        if np.isnan(minindex):
            maxindex = 0

        expected0 = ar.isel(x=maxindex, drop=True)
        result0 = ar.max(keep_attrs=True)
        assert_identical(result0, expected0)

        result1 = ar.max()
        expected1 = expected0.copy()
        expected1.attrs = {}
        assert_identical(result1, expected1)

        result2 = ar.max(skipna=False)
        if nanindex is not None and ar.dtype.kind != "O":
            expected2 = ar.isel(x=nanindex, drop=True)
            expected2.attrs = {}
        else:
            expected2 = expected1

        assert_identical(result2, expected2)

    @pytest.mark.filterwarnings(
        "ignore:Behaviour of argmin/argmax with neither dim nor :DeprecationWarning"
    )
    def test_argmin(self, x, minindex, maxindex, nanindex):
        ar = xr.DataArray(
            x, dims=["x"], coords={"x": np.arange(x.size) * 4}, attrs=self.attrs
        )
        indarr = xr.DataArray(np.arange(x.size, dtype=np.intp), dims=["x"])

        if np.isnan(minindex):
            with pytest.raises(ValueError):
                ar.argmin()
            return

        expected0 = indarr[minindex]
        result0 = ar.argmin()
        assert_identical(result0, expected0)

        result1 = ar.argmin(keep_attrs=True)
        expected1 = expected0.copy()
        expected1.attrs = self.attrs
        assert_identical(result1, expected1)

        result2 = ar.argmin(skipna=False)
        if nanindex is not None and ar.dtype.kind != "O":
            expected2 = indarr.isel(x=nanindex, drop=True)
            expected2.attrs = {}
        else:
            expected2 = expected0

        assert_identical(result2, expected2)

    @pytest.mark.filterwarnings(
        "ignore:Behaviour of argmin/argmax with neither dim nor :DeprecationWarning"
    )
    def test_argmax(self, x, minindex, maxindex, nanindex):
        ar = xr.DataArray(
            x, dims=["x"], coords={"x": np.arange(x.size) * 4}, attrs=self.attrs
        )
        indarr = xr.DataArray(np.arange(x.size, dtype=np.intp), dims=["x"])

        if np.isnan(maxindex):
            with pytest.raises(ValueError):
                ar.argmax()
            return

        expected0 = indarr[maxindex]
        result0 = ar.argmax()
        assert_identical(result0, expected0)

        result1 = ar.argmax(keep_attrs=True)
        expected1 = expected0.copy()
        expected1.attrs = self.attrs
        assert_identical(result1, expected1)

        result2 = ar.argmax(skipna=False)
        if nanindex is not None and ar.dtype.kind != "O":
            expected2 = indarr.isel(x=nanindex, drop=True)
            expected2.attrs = {}
        else:
            expected2 = expected0

        assert_identical(result2, expected2)

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_idxmin(self, x, minindex, maxindex, nanindex, use_dask):
        if use_dask and not has_dask:
            pytest.skip("requires dask")
        if use_dask and x.dtype.kind == "M":
            pytest.xfail("dask operation 'argmin' breaks when dtype is datetime64 (M)")
        ar0_raw = xr.DataArray(
            x, dims=["x"], coords={"x": np.arange(x.size) * 4}, attrs=self.attrs
        )

        if use_dask:
            ar0 = ar0_raw.chunk({})
        else:
            ar0 = ar0_raw

        # dim doesn't exist
        with pytest.raises(KeyError):
            ar0.idxmin(dim="spam")

        # Scalar Dataarray
        with pytest.raises(ValueError):
            xr.DataArray(5).idxmin()

        coordarr0 = xr.DataArray(ar0.coords["x"], dims=["x"])
        coordarr1 = coordarr0.copy()

        hasna = np.isnan(minindex)
        if np.isnan(minindex):
            minindex = 0

        if hasna:
            coordarr1[...] = 1
            fill_value_0 = np.NaN
        else:
            fill_value_0 = 1

        expected0 = (
            (coordarr1 * fill_value_0).isel(x=minindex, drop=True).astype("float")
        )
        expected0.name = "x"

        # Default fill value (NaN)
        result0 = ar0.idxmin()
        assert_identical(result0, expected0)

        # Manually specify NaN fill_value
        result1 = ar0.idxmin(fill_value=np.NaN)
        assert_identical(result1, expected0)

        # keep_attrs
        result2 = ar0.idxmin(keep_attrs=True)
        expected2 = expected0.copy()
        expected2.attrs = self.attrs
        assert_identical(result2, expected2)

        # skipna=False
        if nanindex is not None and ar0.dtype.kind != "O":
            expected3 = coordarr0.isel(x=nanindex, drop=True).astype("float")
            expected3.name = "x"
            expected3.attrs = {}
        else:
            expected3 = expected0.copy()

        result3 = ar0.idxmin(skipna=False)
        assert_identical(result3, expected3)

        # fill_value should be ignored with skipna=False
        result4 = ar0.idxmin(skipna=False, fill_value=-100j)
        assert_identical(result4, expected3)

        # Float fill_value
        if hasna:
            fill_value_5 = -1.1
        else:
            fill_value_5 = 1

        expected5 = (coordarr1 * fill_value_5).isel(x=minindex, drop=True)
        expected5.name = "x"

        result5 = ar0.idxmin(fill_value=-1.1)
        assert_identical(result5, expected5)

        # Integer fill_value
        if hasna:
            fill_value_6 = -1
        else:
            fill_value_6 = 1

        expected6 = (coordarr1 * fill_value_6).isel(x=minindex, drop=True)
        expected6.name = "x"

        result6 = ar0.idxmin(fill_value=-1)
        assert_identical(result6, expected6)

        # Complex fill_value
        if hasna:
            fill_value_7 = -1j
        else:
            fill_value_7 = 1

        expected7 = (coordarr1 * fill_value_7).isel(x=minindex, drop=True)
        expected7.name = "x"

        result7 = ar0.idxmin(fill_value=-1j)
        assert_identical(result7, expected7)

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_idxmax(self, x, minindex, maxindex, nanindex, use_dask):
        if use_dask and not has_dask:
            pytest.skip("requires dask")
        if use_dask and x.dtype.kind == "M":
            pytest.xfail("dask operation 'argmax' breaks when dtype is datetime64 (M)")
        ar0_raw = xr.DataArray(
            x, dims=["x"], coords={"x": np.arange(x.size) * 4}, attrs=self.attrs
        )

        if use_dask:
            ar0 = ar0_raw.chunk({})
        else:
            ar0 = ar0_raw

        # dim doesn't exist
        with pytest.raises(KeyError):
            ar0.idxmax(dim="spam")

        # Scalar Dataarray
        with pytest.raises(ValueError):
            xr.DataArray(5).idxmax()

        coordarr0 = xr.DataArray(ar0.coords["x"], dims=["x"])
        coordarr1 = coordarr0.copy()

        hasna = np.isnan(maxindex)
        if np.isnan(maxindex):
            maxindex = 0

        if hasna:
            coordarr1[...] = 1
            fill_value_0 = np.NaN
        else:
            fill_value_0 = 1

        expected0 = (
            (coordarr1 * fill_value_0).isel(x=maxindex, drop=True).astype("float")
        )
        expected0.name = "x"

        # Default fill value (NaN)
        result0 = ar0.idxmax()
        assert_identical(result0, expected0)

        # Manually specify NaN fill_value
        result1 = ar0.idxmax(fill_value=np.NaN)
        assert_identical(result1, expected0)

        # keep_attrs
        result2 = ar0.idxmax(keep_attrs=True)
        expected2 = expected0.copy()
        expected2.attrs = self.attrs
        assert_identical(result2, expected2)

        # skipna=False
        if nanindex is not None and ar0.dtype.kind != "O":
            expected3 = coordarr0.isel(x=nanindex, drop=True).astype("float")
            expected3.name = "x"
            expected3.attrs = {}
        else:
            expected3 = expected0.copy()

        result3 = ar0.idxmax(skipna=False)
        assert_identical(result3, expected3)

        # fill_value should be ignored with skipna=False
        result4 = ar0.idxmax(skipna=False, fill_value=-100j)
        assert_identical(result4, expected3)

        # Float fill_value
        if hasna:
            fill_value_5 = -1.1
        else:
            fill_value_5 = 1

        expected5 = (coordarr1 * fill_value_5).isel(x=maxindex, drop=True)
        expected5.name = "x"

        result5 = ar0.idxmax(fill_value=-1.1)
        assert_identical(result5, expected5)

        # Integer fill_value
        if hasna:
            fill_value_6 = -1
        else:
            fill_value_6 = 1

        expected6 = (coordarr1 * fill_value_6).isel(x=maxindex, drop=True)
        expected6.name = "x"

        result6 = ar0.idxmax(fill_value=-1)
        assert_identical(result6, expected6)

        # Complex fill_value
        if hasna:
            fill_value_7 = -1j
        else:
            fill_value_7 = 1

        expected7 = (coordarr1 * fill_value_7).isel(x=maxindex, drop=True)
        expected7.name = "x"

        result7 = ar0.idxmax(fill_value=-1j)
        assert_identical(result7, expected7)

    @pytest.mark.filterwarnings(
        "ignore:Behaviour of argmin/argmax with neither dim nor :DeprecationWarning"
    )
    def test_argmin_dim(self, x, minindex, maxindex, nanindex):
        ar = xr.DataArray(
            x, dims=["x"], coords={"x": np.arange(x.size) * 4}, attrs=self.attrs
        )
        indarr = xr.DataArray(np.arange(x.size, dtype=np.intp), dims=["x"])

        if np.isnan(minindex):
            with pytest.raises(ValueError):
                ar.argmin()
            return

        expected0 = {"x": indarr[minindex]}
        result0 = ar.argmin(...)
        for key in expected0:
            assert_identical(result0[key], expected0[key])

        result1 = ar.argmin(..., keep_attrs=True)
        expected1 = deepcopy(expected0)
        for da in expected1.values():
            da.attrs = self.attrs
        for key in expected1:
            assert_identical(result1[key], expected1[key])

        result2 = ar.argmin(..., skipna=False)
        if nanindex is not None and ar.dtype.kind != "O":
            expected2 = {"x": indarr.isel(x=nanindex, drop=True)}
            expected2["x"].attrs = {}
        else:
            expected2 = expected0

        for key in expected2:
            assert_identical(result2[key], expected2[key])

    @pytest.mark.filterwarnings(
        "ignore:Behaviour of argmin/argmax with neither dim nor :DeprecationWarning"
    )
    def test_argmax_dim(self, x, minindex, maxindex, nanindex):
        ar = xr.DataArray(
            x, dims=["x"], coords={"x": np.arange(x.size) * 4}, attrs=self.attrs
        )
        indarr = xr.DataArray(np.arange(x.size, dtype=np.intp), dims=["x"])

        if np.isnan(maxindex):
            with pytest.raises(ValueError):
                ar.argmax()
            return

        expected0 = {"x": indarr[maxindex]}
        result0 = ar.argmax(...)
        for key in expected0:
            assert_identical(result0[key], expected0[key])

        result1 = ar.argmax(..., keep_attrs=True)
        expected1 = deepcopy(expected0)
        for da in expected1.values():
            da.attrs = self.attrs
        for key in expected1:
            assert_identical(result1[key], expected1[key])

        result2 = ar.argmax(..., skipna=False)
        if nanindex is not None and ar.dtype.kind != "O":
            expected2 = {"x": indarr.isel(x=nanindex, drop=True)}
            expected2["x"].attrs = {}
        else:
            expected2 = expected0

        for key in expected2:
            assert_identical(result2[key], expected2[key])


@pytest.mark.parametrize(
    "x, minindex, maxindex, nanindex",
    [
        (
            np.array(
                [
                    [0, 1, 2, 0, -2, -4, 2],
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, -10, 5, 20, 0, 0],
                ]
            ),
            [5, 0, 2],
            [2, 0, 4],
            [None, None, None],
        ),
        (
            np.array(
                [
                    [2.0, 1.0, 2.0, 0.0, -2.0, -4.0, 2.0],
                    [-4.0, np.NaN, 2.0, np.NaN, -2.0, -4.0, 2.0],
                    [np.NaN] * 7,
                ]
            ),
            [5, 0, np.NaN],
            [0, 2, np.NaN],
            [None, 1, 0],
        ),
        (
            np.array(
                [
                    [2.0, 1.0, 2.0, 0.0, -2.0, -4.0, 2.0],
                    [-4.0, np.NaN, 2.0, np.NaN, -2.0, -4.0, 2.0],
                    [np.NaN] * 7,
                ]
            ).astype("object"),
            [5, 0, np.NaN],
            [0, 2, np.NaN],
            [None, 1, 0],
        ),
        (
            np.array(
                [
                    ["2015-12-31", "2020-01-02", "2020-01-01", "2016-01-01"],
                    ["2020-01-02", "2020-01-02", "2020-01-02", "2020-01-02"],
                    ["1900-01-01", "1-02-03", "1900-01-02", "1-02-03"],
                ],
                dtype="datetime64[ns]",
            ),
            [0, 0, 1],
            [1, 0, 2],
            [None, None, None],
        ),
    ],
)
class TestReduce2D(TestReduce):
    def test_min(self, x, minindex, maxindex, nanindex):
        ar = xr.DataArray(
            x,
            dims=["y", "x"],
            coords={"x": np.arange(x.shape[1]) * 4, "y": 1 - np.arange(x.shape[0])},
            attrs=self.attrs,
        )

        minindex = [x if not np.isnan(x) else 0 for x in minindex]
        expected0 = [
            ar.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex)
        ]
        expected0 = xr.concat(expected0, dim="y")

        result0 = ar.min(dim="x", keep_attrs=True)
        assert_identical(result0, expected0)

        result1 = ar.min(dim="x")
        expected1 = expected0
        expected1.attrs = {}
        assert_identical(result1, expected1)

        result2 = ar.min(axis=1)
        assert_identical(result2, expected1)

        minindex = [
            x if y is None or ar.dtype.kind == "O" else y
            for x, y in zip(minindex, nanindex)
        ]
        expected2 = [
            ar.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex)
        ]
        expected2 = xr.concat(expected2, dim="y")
        expected2.attrs = {}

        result3 = ar.min(dim="x", skipna=False)

        assert_identical(result3, expected2)

    def test_max(self, x, minindex, maxindex, nanindex):
        ar = xr.DataArray(
            x,
            dims=["y", "x"],
            coords={"x": np.arange(x.shape[1]) * 4, "y": 1 - np.arange(x.shape[0])},
            attrs=self.attrs,
        )

        maxindex = [x if not np.isnan(x) else 0 for x in maxindex]
        expected0 = [
            ar.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex)
        ]
        expected0 = xr.concat(expected0, dim="y")

        result0 = ar.max(dim="x", keep_attrs=True)
        assert_identical(result0, expected0)

        result1 = ar.max(dim="x")
        expected1 = expected0.copy()
        expected1.attrs = {}
        assert_identical(result1, expected1)

        result2 = ar.max(axis=1)
        assert_identical(result2, expected1)

        maxindex = [
            x if y is None or ar.dtype.kind == "O" else y
            for x, y in zip(maxindex, nanindex)
        ]
        expected2 = [
            ar.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex)
        ]
        expected2 = xr.concat(expected2, dim="y")
        expected2.attrs = {}

        result3 = ar.max(dim="x", skipna=False)

        assert_identical(result3, expected2)

    def test_argmin(self, x, minindex, maxindex, nanindex):
        ar = xr.DataArray(
            x,
            dims=["y", "x"],
            coords={"x": np.arange(x.shape[1]) * 4, "y": 1 - np.arange(x.shape[0])},
            attrs=self.attrs,
        )
        indarr = np.tile(np.arange(x.shape[1], dtype=np.intp), [x.shape[0], 1])
        indarr = xr.DataArray(indarr, dims=ar.dims, coords=ar.coords)

        if np.isnan(minindex).any():
            with pytest.raises(ValueError):
                ar.argmin(dim="x")
            return

        expected0 = [
            indarr.isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(minindex)
        ]
        expected0 = xr.concat(expected0, dim="y")

        result0 = ar.argmin(dim="x")
        assert_identical(result0, expected0)

        result1 = ar.argmin(axis=1)
        assert_identical(result1, expected0)

        result2 = ar.argmin(dim="x", keep_attrs=True)
        expected1 = expected0.copy()
        expected1.attrs = self.attrs
        assert_identical(result2, expected1)

        minindex = [
            x if y is None or ar.dtype.kind == "O" else y
            for x, y in zip(minindex, nanindex)
        ]
        expected2 = [
            indarr.isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(minindex)
        ]
        expected2 = xr.concat(expected2, dim="y")
        expected2.attrs = {}

        result3 = ar.argmin(dim="x", skipna=False)

        assert_identical(result3, expected2)

    def test_argmax(self, x, minindex, maxindex, nanindex):
        ar = xr.DataArray(
            x,
            dims=["y", "x"],
            coords={"x": np.arange(x.shape[1]) * 4, "y": 1 - np.arange(x.shape[0])},
            attrs=self.attrs,
        )
        indarr = np.tile(np.arange(x.shape[1], dtype=np.intp), [x.shape[0], 1])
        indarr = xr.DataArray(indarr, dims=ar.dims, coords=ar.coords)

        if np.isnan(maxindex).any():
            with pytest.raises(ValueError):
                ar.argmax(dim="x")
            return

        expected0 = [
            indarr.isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(maxindex)
        ]
        expected0 = xr.concat(expected0, dim="y")

        result0 = ar.argmax(dim="x")
        assert_identical(result0, expected0)

        result1 = ar.argmax(axis=1)
        assert_identical(result1, expected0)

        result2 = ar.argmax(dim="x", keep_attrs=True)
        expected1 = expected0.copy()
        expected1.attrs = self.attrs
        assert_identical(result2, expected1)

        maxindex = [
            x if y is None or ar.dtype.kind == "O" else y
            for x, y in zip(maxindex, nanindex)
        ]
        expected2 = [
            indarr.isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(maxindex)
        ]
        expected2 = xr.concat(expected2, dim="y")
        expected2.attrs = {}

        result3 = ar.argmax(dim="x", skipna=False)

        assert_identical(result3, expected2)

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_idxmin(self, x, minindex, maxindex, nanindex, use_dask):
        if use_dask and not has_dask:
            pytest.skip("requires dask")
        if use_dask and x.dtype.kind == "M":
            pytest.xfail("dask operation 'argmin' breaks when dtype is datetime64 (M)")

        if x.dtype.kind == "O":
            # TODO: nanops._nan_argminmax_object computes once to check for all-NaN slices.
            max_computes = 1
        else:
            max_computes = 0

        ar0_raw = xr.DataArray(
            x,
            dims=["y", "x"],
            coords={"x": np.arange(x.shape[1]) * 4, "y": 1 - np.arange(x.shape[0])},
            attrs=self.attrs,
        )

        if use_dask:
            ar0 = ar0_raw.chunk({})
        else:
            ar0 = ar0_raw

        assert_identical(ar0, ar0)

        # No dimension specified
        with pytest.raises(ValueError):
            ar0.idxmin()

        # dim doesn't exist
        with pytest.raises(KeyError):
            ar0.idxmin(dim="Y")

        assert_identical(ar0, ar0)

        coordarr0 = xr.DataArray(
            np.tile(ar0.coords["x"], [x.shape[0], 1]), dims=ar0.dims, coords=ar0.coords
        )

        hasna = [np.isnan(x) for x in minindex]
        coordarr1 = coordarr0.copy()
        coordarr1[hasna, :] = 1
        minindex0 = [x if not np.isnan(x) else 0 for x in minindex]

        nan_mult_0 = np.array([np.NaN if x else 1 for x in hasna])[:, None]
        expected0 = [
            (coordarr1 * nan_mult_0).isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(minindex0)
        ]
        expected0 = xr.concat(expected0, dim="y")
        expected0.name = "x"

        # Default fill value (NaN)
        with raise_if_dask_computes(max_computes=max_computes):
            result0 = ar0.idxmin(dim="x")
        assert_identical(result0, expected0)

        # Manually specify NaN fill_value
        with raise_if_dask_computes(max_computes=max_computes):
            result1 = ar0.idxmin(dim="x", fill_value=np.NaN)
        assert_identical(result1, expected0)

        # keep_attrs
        with raise_if_dask_computes(max_computes=max_computes):
            result2 = ar0.idxmin(dim="x", keep_attrs=True)
        expected2 = expected0.copy()
        expected2.attrs = self.attrs
        assert_identical(result2, expected2)

        # skipna=False
        minindex3 = [
            x if y is None or ar0.dtype.kind == "O" else y
            for x, y in zip(minindex0, nanindex)
        ]
        expected3 = [
            coordarr0.isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(minindex3)
        ]
        expected3 = xr.concat(expected3, dim="y")
        expected3.name = "x"
        expected3.attrs = {}

        with raise_if_dask_computes(max_computes=max_computes):
            result3 = ar0.idxmin(dim="x", skipna=False)
        assert_identical(result3, expected3)

        # fill_value should be ignored with skipna=False
        with raise_if_dask_computes(max_computes=max_computes):
            result4 = ar0.idxmin(dim="x", skipna=False, fill_value=-100j)
        assert_identical(result4, expected3)

        # Float fill_value
        nan_mult_5 = np.array([-1.1 if x else 1 for x in hasna])[:, None]
        expected5 = [
            (coordarr1 * nan_mult_5).isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(minindex0)
        ]
        expected5 = xr.concat(expected5, dim="y")
        expected5.name = "x"

        with raise_if_dask_computes(max_computes=max_computes):
            result5 = ar0.idxmin(dim="x", fill_value=-1.1)
        assert_identical(result5, expected5)

        # Integer fill_value
        nan_mult_6 = np.array([-1 if x else 1 for x in hasna])[:, None]
        expected6 = [
            (coordarr1 * nan_mult_6).isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(minindex0)
        ]
        expected6 = xr.concat(expected6, dim="y")
        expected6.name = "x"

        with raise_if_dask_computes(max_computes=max_computes):
            result6 = ar0.idxmin(dim="x", fill_value=-1)
        assert_identical(result6, expected6)

        # Complex fill_value
        nan_mult_7 = np.array([-5j if x else 1 for x in hasna])[:, None]
        expected7 = [
            (coordarr1 * nan_mult_7).isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(minindex0)
        ]
        expected7 = xr.concat(expected7, dim="y")
        expected7.name = "x"

        with raise_if_dask_computes(max_computes=max_computes):
            result7 = ar0.idxmin(dim="x", fill_value=-5j)
        assert_identical(result7, expected7)

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_idxmax(self, x, minindex, maxindex, nanindex, use_dask):
        if use_dask and not has_dask:
            pytest.skip("requires dask")
        if use_dask and x.dtype.kind == "M":
            pytest.xfail("dask operation 'argmax' breaks when dtype is datetime64 (M)")

        if x.dtype.kind == "O":
            # TODO: nanops._nan_argminmax_object computes once to check for all-NaN slices.
            max_computes = 1
        else:
            max_computes = 0

        ar0_raw = xr.DataArray(
            x,
            dims=["y", "x"],
            coords={"x": np.arange(x.shape[1]) * 4, "y": 1 - np.arange(x.shape[0])},
            attrs=self.attrs,
        )

        if use_dask:
            ar0 = ar0_raw.chunk({})
        else:
            ar0 = ar0_raw

        # No dimension specified
        with pytest.raises(ValueError):
            ar0.idxmax()

        # dim doesn't exist
        with pytest.raises(KeyError):
            ar0.idxmax(dim="Y")

        ar1 = ar0.copy()
        del ar1.coords["y"]
        with pytest.raises(KeyError):
            ar1.idxmax(dim="y")

        coordarr0 = xr.DataArray(
            np.tile(ar0.coords["x"], [x.shape[0], 1]), dims=ar0.dims, coords=ar0.coords
        )

        hasna = [np.isnan(x) for x in maxindex]
        coordarr1 = coordarr0.copy()
        coordarr1[hasna, :] = 1
        maxindex0 = [x if not np.isnan(x) else 0 for x in maxindex]

        nan_mult_0 = np.array([np.NaN if x else 1 for x in hasna])[:, None]
        expected0 = [
            (coordarr1 * nan_mult_0).isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(maxindex0)
        ]
        expected0 = xr.concat(expected0, dim="y")
        expected0.name = "x"

        # Default fill value (NaN)
        with raise_if_dask_computes(max_computes=max_computes):
            result0 = ar0.idxmax(dim="x")
        assert_identical(result0, expected0)

        # Manually specify NaN fill_value
        with raise_if_dask_computes(max_computes=max_computes):
            result1 = ar0.idxmax(dim="x", fill_value=np.NaN)
        assert_identical(result1, expected0)

        # keep_attrs
        with raise_if_dask_computes(max_computes=max_computes):
            result2 = ar0.idxmax(dim="x", keep_attrs=True)
        expected2 = expected0.copy()
        expected2.attrs = self.attrs
        assert_identical(result2, expected2)

        # skipna=False
        maxindex3 = [
            x if y is None or ar0.dtype.kind == "O" else y
            for x, y in zip(maxindex0, nanindex)
        ]
        expected3 = [
            coordarr0.isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(maxindex3)
        ]
        expected3 = xr.concat(expected3, dim="y")
        expected3.name = "x"
        expected3.attrs = {}

        with raise_if_dask_computes(max_computes=max_computes):
            result3 = ar0.idxmax(dim="x", skipna=False)
        assert_identical(result3, expected3)

        # fill_value should be ignored with skipna=False
        with raise_if_dask_computes(max_computes=max_computes):
            result4 = ar0.idxmax(dim="x", skipna=False, fill_value=-100j)
        assert_identical(result4, expected3)

        # Float fill_value
        nan_mult_5 = np.array([-1.1 if x else 1 for x in hasna])[:, None]
        expected5 = [
            (coordarr1 * nan_mult_5).isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(maxindex0)
        ]
        expected5 = xr.concat(expected5, dim="y")
        expected5.name = "x"

        with raise_if_dask_computes(max_computes=max_computes):
            result5 = ar0.idxmax(dim="x", fill_value=-1.1)
        assert_identical(result5, expected5)

        # Integer fill_value
        nan_mult_6 = np.array([-1 if x else 1 for x in hasna])[:, None]
        expected6 = [
            (coordarr1 * nan_mult_6).isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(maxindex0)
        ]
        expected6 = xr.concat(expected6, dim="y")
        expected6.name = "x"

        with raise_if_dask_computes(max_computes=max_computes):
            result6 = ar0.idxmax(dim="x", fill_value=-1)
        assert_identical(result6, expected6)

        # Complex fill_value
        nan_mult_7 = np.array([-5j if x else 1 for x in hasna])[:, None]
        expected7 = [
            (coordarr1 * nan_mult_7).isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(maxindex0)
        ]
        expected7 = xr.concat(expected7, dim="y")
        expected7.name = "x"

        with raise_if_dask_computes(max_computes=max_computes):
            result7 = ar0.idxmax(dim="x", fill_value=-5j)
        assert_identical(result7, expected7)

    @pytest.mark.filterwarnings(
        "ignore:Behaviour of argmin/argmax with neither dim nor :DeprecationWarning"
    )
    def test_argmin_dim(self, x, minindex, maxindex, nanindex):
        ar = xr.DataArray(
            x,
            dims=["y", "x"],
            coords={"x": np.arange(x.shape[1]) * 4, "y": 1 - np.arange(x.shape[0])},
            attrs=self.attrs,
        )
        indarr = np.tile(np.arange(x.shape[1], dtype=np.intp), [x.shape[0], 1])
        indarr = xr.DataArray(indarr, dims=ar.dims, coords=ar.coords)

        if np.isnan(minindex).any():
            with pytest.raises(ValueError):
                ar.argmin(dim="x")
            return

        expected0 = [
            indarr.isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(minindex)
        ]
        expected0 = {"x": xr.concat(expected0, dim="y")}

        result0 = ar.argmin(dim=["x"])
        for key in expected0:
            assert_identical(result0[key], expected0[key])

        result1 = ar.argmin(dim=["x"], keep_attrs=True)
        expected1 = deepcopy(expected0)
        expected1["x"].attrs = self.attrs
        for key in expected1:
            assert_identical(result1[key], expected1[key])

        minindex = [
            x if y is None or ar.dtype.kind == "O" else y
            for x, y in zip(minindex, nanindex)
        ]
        expected2 = [
            indarr.isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(minindex)
        ]
        expected2 = {"x": xr.concat(expected2, dim="y")}
        expected2["x"].attrs = {}

        result2 = ar.argmin(dim=["x"], skipna=False)

        for key in expected2:
            assert_identical(result2[key], expected2[key])

        result3 = ar.argmin(...)
        min_xind = ar.isel(expected0).argmin()
        expected3 = {
            "y": DataArray(min_xind),
            "x": DataArray(minindex[min_xind.item()]),
        }

        for key in expected3:
            assert_identical(result3[key], expected3[key])

    @pytest.mark.filterwarnings(
        "ignore:Behaviour of argmin/argmax with neither dim nor :DeprecationWarning"
    )
    def test_argmax_dim(self, x, minindex, maxindex, nanindex):
        ar = xr.DataArray(
            x,
            dims=["y", "x"],
            coords={"x": np.arange(x.shape[1]) * 4, "y": 1 - np.arange(x.shape[0])},
            attrs=self.attrs,
        )
        indarr = np.tile(np.arange(x.shape[1], dtype=np.intp), [x.shape[0], 1])
        indarr = xr.DataArray(indarr, dims=ar.dims, coords=ar.coords)

        if np.isnan(maxindex).any():
            with pytest.raises(ValueError):
                ar.argmax(dim="x")
            return

        expected0 = [
            indarr.isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(maxindex)
        ]
        expected0 = {"x": xr.concat(expected0, dim="y")}

        result0 = ar.argmax(dim=["x"])
        for key in expected0:
            assert_identical(result0[key], expected0[key])

        result1 = ar.argmax(dim=["x"], keep_attrs=True)
        expected1 = deepcopy(expected0)
        expected1["x"].attrs = self.attrs
        for key in expected1:
            assert_identical(result1[key], expected1[key])

        maxindex = [
            x if y is None or ar.dtype.kind == "O" else y
            for x, y in zip(maxindex, nanindex)
        ]
        expected2 = [
            indarr.isel(y=yi).isel(x=indi, drop=True)
            for yi, indi in enumerate(maxindex)
        ]
        expected2 = {"x": xr.concat(expected2, dim="y")}
        expected2["x"].attrs = {}

        result2 = ar.argmax(dim=["x"], skipna=False)

        for key in expected2:
            assert_identical(result2[key], expected2[key])

        result3 = ar.argmax(...)
        max_xind = ar.isel(expected0).argmax()
        expected3 = {
            "y": DataArray(max_xind),
            "x": DataArray(maxindex[max_xind.item()]),
        }

        for key in expected3:
            assert_identical(result3[key], expected3[key])


@pytest.mark.parametrize(
    "x, minindices_x, minindices_y, minindices_z, minindices_xy, "
    "minindices_xz, minindices_yz, minindices_xyz, maxindices_x, "
    "maxindices_y, maxindices_z, maxindices_xy, maxindices_xz, maxindices_yz, "
    "maxindices_xyz, nanindices_x, nanindices_y, nanindices_z, nanindices_xy, "
    "nanindices_xz, nanindices_yz, nanindices_xyz",
    [
        (
            np.array(
                [
                    [[0, 1, 2, 0], [-2, -4, 2, 0]],
                    [[1, 1, 1, 1], [1, 1, 1, 1]],
                    [[0, 0, -10, 5], [20, 0, 0, 0]],
                ]
            ),
            {"x": np.array([[0, 2, 2, 0], [0, 0, 2, 0]])},
            {"y": np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])},
            {"z": np.array([[0, 1], [0, 0], [2, 1]])},
            {"x": np.array([0, 0, 2, 0]), "y": np.array([1, 1, 0, 0])},
            {"x": np.array([2, 0]), "z": np.array([2, 1])},
            {"y": np.array([1, 0, 0]), "z": np.array([1, 0, 2])},
            {"x": np.array(2), "y": np.array(0), "z": np.array(2)},
            {"x": np.array([[1, 0, 0, 2], [2, 1, 0, 1]])},
            {"y": np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0]])},
            {"z": np.array([[2, 2], [0, 0], [3, 0]])},
            {"x": np.array([2, 0, 0, 2]), "y": np.array([1, 0, 0, 0])},
            {"x": np.array([2, 2]), "z": np.array([3, 0])},
            {"y": np.array([0, 0, 1]), "z": np.array([2, 0, 0])},
            {"x": np.array(2), "y": np.array(1), "z": np.array(0)},
            {"x": np.array([[None, None, None, None], [None, None, None, None]])},
            {
                "y": np.array(
                    [
                        [None, None, None, None],
                        [None, None, None, None],
                        [None, None, None, None],
                    ]
                )
            },
            {"z": np.array([[None, None], [None, None], [None, None]])},
            {
                "x": np.array([None, None, None, None]),
                "y": np.array([None, None, None, None]),
            },
            {"x": np.array([None, None]), "z": np.array([None, None])},
            {"y": np.array([None, None, None]), "z": np.array([None, None, None])},
            {"x": np.array(None), "y": np.array(None), "z": np.array(None)},
        ),
        (
            np.array(
                [
                    [[2.0, 1.0, 2.0, 0.0], [-2.0, -4.0, 2.0, 0.0]],
                    [[-4.0, np.NaN, 2.0, np.NaN], [-2.0, -4.0, 2.0, 0.0]],
                    [[np.NaN] * 4, [np.NaN] * 4],
                ]
            ),
            {"x": np.array([[1, 0, 0, 0], [0, 0, 0, 0]])},
            {
                "y": np.array(
                    [[1, 1, 0, 0], [0, 1, 0, 1], [np.NaN, np.NaN, np.NaN, np.NaN]]
                )
            },
            {"z": np.array([[3, 1], [0, 1], [np.NaN, np.NaN]])},
            {"x": np.array([1, 0, 0, 0]), "y": np.array([0, 1, 0, 0])},
            {"x": np.array([1, 0]), "z": np.array([0, 1])},
            {"y": np.array([1, 0, np.NaN]), "z": np.array([1, 0, np.NaN])},
            {"x": np.array(0), "y": np.array(1), "z": np.array(1)},
            {"x": np.array([[0, 0, 0, 0], [0, 0, 0, 0]])},
            {
                "y": np.array(
                    [[0, 0, 0, 0], [1, 1, 0, 1], [np.NaN, np.NaN, np.NaN, np.NaN]]
                )
            },
            {"z": np.array([[0, 2], [2, 2], [np.NaN, np.NaN]])},
            {"x": np.array([0, 0, 0, 0]), "y": np.array([0, 0, 0, 0])},
            {"x": np.array([0, 0]), "z": np.array([2, 2])},
            {"y": np.array([0, 0, np.NaN]), "z": np.array([0, 2, np.NaN])},
            {"x": np.array(0), "y": np.array(0), "z": np.array(0)},
            {"x": np.array([[2, 1, 2, 1], [2, 2, 2, 2]])},
            {
                "y": np.array(
                    [[None, None, None, None], [None, 0, None, 0], [0, 0, 0, 0]]
                )
            },
            {"z": np.array([[None, None], [1, None], [0, 0]])},
            {"x": np.array([2, 1, 2, 1]), "y": np.array([0, 0, 0, 0])},
            {"x": np.array([1, 2]), "z": np.array([1, 0])},
            {"y": np.array([None, 0, 0]), "z": np.array([None, 1, 0])},
            {"x": np.array(1), "y": np.array(0), "z": np.array(1)},
        ),
        (
            np.array(
                [
                    [[2.0, 1.0, 2.0, 0.0], [-2.0, -4.0, 2.0, 0.0]],
                    [[-4.0, np.NaN, 2.0, np.NaN], [-2.0, -4.0, 2.0, 0.0]],
                    [[np.NaN] * 4, [np.NaN] * 4],
                ]
            ).astype("object"),
            {"x": np.array([[1, 0, 0, 0], [0, 0, 0, 0]])},
            {
                "y": np.array(
                    [[1, 1, 0, 0], [0, 1, 0, 1], [np.NaN, np.NaN, np.NaN, np.NaN]]
                )
            },
            {"z": np.array([[3, 1], [0, 1], [np.NaN, np.NaN]])},
            {"x": np.array([1, 0, 0, 0]), "y": np.array([0, 1, 0, 0])},
            {"x": np.array([1, 0]), "z": np.array([0, 1])},
            {"y": np.array([1, 0, np.NaN]), "z": np.array([1, 0, np.NaN])},
            {"x": np.array(0), "y": np.array(1), "z": np.array(1)},
            {"x": np.array([[0, 0, 0, 0], [0, 0, 0, 0]])},
            {
                "y": np.array(
                    [[0, 0, 0, 0], [1, 1, 0, 1], [np.NaN, np.NaN, np.NaN, np.NaN]]
                )
            },
            {"z": np.array([[0, 2], [2, 2], [np.NaN, np.NaN]])},
            {"x": np.array([0, 0, 0, 0]), "y": np.array([0, 0, 0, 0])},
            {"x": np.array([0, 0]), "z": np.array([2, 2])},
            {"y": np.array([0, 0, np.NaN]), "z": np.array([0, 2, np.NaN])},
            {"x": np.array(0), "y": np.array(0), "z": np.array(0)},
            {"x": np.array([[2, 1, 2, 1], [2, 2, 2, 2]])},
            {
                "y": np.array(
                    [[None, None, None, None], [None, 0, None, 0], [0, 0, 0, 0]]
                )
            },
            {"z": np.array([[None, None], [1, None], [0, 0]])},
            {"x": np.array([2, 1, 2, 1]), "y": np.array([0, 0, 0, 0])},
            {"x": np.array([1, 2]), "z": np.array([1, 0])},
            {"y": np.array([None, 0, 0]), "z": np.array([None, 1, 0])},
            {"x": np.array(1), "y": np.array(0), "z": np.array(1)},
        ),
        (
            np.array(
                [
                    [["2015-12-31", "2020-01-02"], ["2020-01-01", "2016-01-01"]],
                    [["2020-01-02", "2020-01-02"], ["2020-01-02", "2020-01-02"]],
                    [["1900-01-01", "1-02-03"], ["1900-01-02", "1-02-03"]],
                ],
                dtype="datetime64[ns]",
            ),
            {"x": np.array([[2, 2], [2, 2]])},
            {"y": np.array([[0, 1], [0, 0], [0, 0]])},
            {"z": np.array([[0, 1], [0, 0], [1, 1]])},
            {"x": np.array([2, 2]), "y": np.array([0, 0])},
            {"x": np.array([2, 2]), "z": np.array([1, 1])},
            {"y": np.array([0, 0, 0]), "z": np.array([0, 0, 1])},
            {"x": np.array(2), "y": np.array(0), "z": np.array(1)},
            {"x": np.array([[1, 0], [1, 1]])},
            {"y": np.array([[1, 0], [0, 0], [1, 0]])},
            {"z": np.array([[1, 0], [0, 0], [0, 0]])},
            {"x": np.array([1, 0]), "y": np.array([0, 0])},
            {"x": np.array([0, 1]), "z": np.array([1, 0])},
            {"y": np.array([0, 0, 1]), "z": np.array([1, 0, 0])},
            {"x": np.array(0), "y": np.array(0), "z": np.array(1)},
            {"x": np.array([[None, None], [None, None]])},
            {"y": np.array([[None, None], [None, None], [None, None]])},
            {"z": np.array([[None, None], [None, None], [None, None]])},
            {"x": np.array([None, None]), "y": np.array([None, None])},
            {"x": np.array([None, None]), "z": np.array([None, None])},
            {"y": np.array([None, None, None]), "z": np.array([None, None, None])},
            {"x": np.array(None), "y": np.array(None), "z": np.array(None)},
        ),
    ],
)
class TestReduce3D(TestReduce):
    def test_argmin_dim(
        self,
        x,
        minindices_x,
        minindices_y,
        minindices_z,
        minindices_xy,
        minindices_xz,
        minindices_yz,
        minindices_xyz,
        maxindices_x,
        maxindices_y,
        maxindices_z,
        maxindices_xy,
        maxindices_xz,
        maxindices_yz,
        maxindices_xyz,
        nanindices_x,
        nanindices_y,
        nanindices_z,
        nanindices_xy,
        nanindices_xz,
        nanindices_yz,
        nanindices_xyz,
    ):

        ar = xr.DataArray(
            x,
            dims=["x", "y", "z"],
            coords={
                "x": np.arange(x.shape[0]) * 4,
                "y": 1 - np.arange(x.shape[1]),
                "z": 2 + 3 * np.arange(x.shape[2]),
            },
            attrs=self.attrs,
        )
        xindarr = np.tile(
            np.arange(x.shape[0], dtype=np.intp)[:, np.newaxis, np.newaxis],
            [1, x.shape[1], x.shape[2]],
        )
        xindarr = xr.DataArray(xindarr, dims=ar.dims, coords=ar.coords)
        yindarr = np.tile(
            np.arange(x.shape[1], dtype=np.intp)[np.newaxis, :, np.newaxis],
            [x.shape[0], 1, x.shape[2]],
        )
        yindarr = xr.DataArray(yindarr, dims=ar.dims, coords=ar.coords)
        zindarr = np.tile(
            np.arange(x.shape[2], dtype=np.intp)[np.newaxis, np.newaxis, :],
            [x.shape[0], x.shape[1], 1],
        )
        zindarr = xr.DataArray(zindarr, dims=ar.dims, coords=ar.coords)

        for inds in [
            minindices_x,
            minindices_y,
            minindices_z,
            minindices_xy,
            minindices_xz,
            minindices_yz,
            minindices_xyz,
        ]:
            if np.array([np.isnan(i) for i in inds.values()]).any():
                with pytest.raises(ValueError):
                    ar.argmin(dim=[d for d in inds])
                return

        result0 = ar.argmin(dim=["x"])
        expected0 = {
            key: xr.DataArray(value, dims=("y", "z"))
            for key, value in minindices_x.items()
        }
        for key in expected0:
            assert_identical(result0[key].drop_vars(["y", "z"]), expected0[key])

        result1 = ar.argmin(dim=["y"])
        expected1 = {
            key: xr.DataArray(value, dims=("x", "z"))
            for key, value in minindices_y.items()
        }
        for key in expected1:
            assert_identical(result1[key].drop_vars(["x", "z"]), expected1[key])

        result2 = ar.argmin(dim=["z"])
        expected2 = {
            key: xr.DataArray(value, dims=("x", "y"))
            for key, value in minindices_z.items()
        }
        for key in expected2:
            assert_identical(result2[key].drop_vars(["x", "y"]), expected2[key])

        result3 = ar.argmin(dim=("x", "y"))
        expected3 = {
            key: xr.DataArray(value, dims=("z")) for key, value in minindices_xy.items()
        }
        for key in expected3:
            assert_identical(result3[key].drop_vars("z"), expected3[key])

        result4 = ar.argmin(dim=("x", "z"))
        expected4 = {
            key: xr.DataArray(value, dims=("y")) for key, value in minindices_xz.items()
        }
        for key in expected4:
            assert_identical(result4[key].drop_vars("y"), expected4[key])

        result5 = ar.argmin(dim=("y", "z"))
        expected5 = {
            key: xr.DataArray(value, dims=("x")) for key, value in minindices_yz.items()
        }
        for key in expected5:
            assert_identical(result5[key].drop_vars("x"), expected5[key])

        result6 = ar.argmin(...)
        expected6 = {key: xr.DataArray(value) for key, value in minindices_xyz.items()}
        for key in expected6:
            assert_identical(result6[key], expected6[key])

        minindices_x = {
            key: xr.where(
                nanindices_x[key] == None,  # noqa: E711
                minindices_x[key],
                nanindices_x[key],
            )
            for key in minindices_x
        }
        expected7 = {
            key: xr.DataArray(value, dims=("y", "z"))
            for key, value in minindices_x.items()
        }

        result7 = ar.argmin(dim=["x"], skipna=False)
        for key in expected7:
            assert_identical(result7[key].drop_vars(["y", "z"]), expected7[key])

        minindices_y = {
            key: xr.where(
                nanindices_y[key] == None,  # noqa: E711
                minindices_y[key],
                nanindices_y[key],
            )
            for key in minindices_y
        }
        expected8 = {
            key: xr.DataArray(value, dims=("x", "z"))
            for key, value in minindices_y.items()
        }

        result8 = ar.argmin(dim=["y"], skipna=False)
        for key in expected8:
            assert_identical(result8[key].drop_vars(["x", "z"]), expected8[key])

        minindices_z = {
            key: xr.where(
                nanindices_z[key] == None,  # noqa: E711
                minindices_z[key],
                nanindices_z[key],
            )
            for key in minindices_z
        }
        expected9 = {
            key: xr.DataArray(value, dims=("x", "y"))
            for key, value in minindices_z.items()
        }

        result9 = ar.argmin(dim=["z"], skipna=False)
        for key in expected9:
            assert_identical(result9[key].drop_vars(["x", "y"]), expected9[key])

        minindices_xy = {
            key: xr.where(
                nanindices_xy[key] == None,  # noqa: E711
                minindices_xy[key],
                nanindices_xy[key],
            )
            for key in minindices_xy
        }
        expected10 = {
            key: xr.DataArray(value, dims="z") for key, value in minindices_xy.items()
        }

        result10 = ar.argmin(dim=("x", "y"), skipna=False)
        for key in expected10:
            assert_identical(result10[key].drop_vars("z"), expected10[key])

        minindices_xz = {
            key: xr.where(
                nanindices_xz[key] == None,  # noqa: E711
                minindices_xz[key],
                nanindices_xz[key],
            )
            for key in minindices_xz
        }
        expected11 = {
            key: xr.DataArray(value, dims="y") for key, value in minindices_xz.items()
        }

        result11 = ar.argmin(dim=("x", "z"), skipna=False)
        for key in expected11:
            assert_identical(result11[key].drop_vars("y"), expected11[key])

        minindices_yz = {
            key: xr.where(
                nanindices_yz[key] == None,  # noqa: E711
                minindices_yz[key],
                nanindices_yz[key],
            )
            for key in minindices_yz
        }
        expected12 = {
            key: xr.DataArray(value, dims="x") for key, value in minindices_yz.items()
        }

        result12 = ar.argmin(dim=("y", "z"), skipna=False)
        for key in expected12:
            assert_identical(result12[key].drop_vars("x"), expected12[key])

        minindices_xyz = {
            key: xr.where(
                nanindices_xyz[key] == None,  # noqa: E711
                minindices_xyz[key],
                nanindices_xyz[key],
            )
            for key in minindices_xyz
        }
        expected13 = {key: xr.DataArray(value) for key, value in minindices_xyz.items()}

        result13 = ar.argmin(..., skipna=False)
        for key in expected13:
            assert_identical(result13[key], expected13[key])

    def test_argmax_dim(
        self,
        x,
        minindices_x,
        minindices_y,
        minindices_z,
        minindices_xy,
        minindices_xz,
        minindices_yz,
        minindices_xyz,
        maxindices_x,
        maxindices_y,
        maxindices_z,
        maxindices_xy,
        maxindices_xz,
        maxindices_yz,
        maxindices_xyz,
        nanindices_x,
        nanindices_y,
        nanindices_z,
        nanindices_xy,
        nanindices_xz,
        nanindices_yz,
        nanindices_xyz,
    ):

        ar = xr.DataArray(
            x,
            dims=["x", "y", "z"],
            coords={
                "x": np.arange(x.shape[0]) * 4,
                "y": 1 - np.arange(x.shape[1]),
                "z": 2 + 3 * np.arange(x.shape[2]),
            },
            attrs=self.attrs,
        )
        xindarr = np.tile(
            np.arange(x.shape[0], dtype=np.intp)[:, np.newaxis, np.newaxis],
            [1, x.shape[1], x.shape[2]],
        )
        xindarr = xr.DataArray(xindarr, dims=ar.dims, coords=ar.coords)
        yindarr = np.tile(
            np.arange(x.shape[1], dtype=np.intp)[np.newaxis, :, np.newaxis],
            [x.shape[0], 1, x.shape[2]],
        )
        yindarr = xr.DataArray(yindarr, dims=ar.dims, coords=ar.coords)
        zindarr = np.tile(
            np.arange(x.shape[2], dtype=np.intp)[np.newaxis, np.newaxis, :],
            [x.shape[0], x.shape[1], 1],
        )
        zindarr = xr.DataArray(zindarr, dims=ar.dims, coords=ar.coords)

        for inds in [
            maxindices_x,
            maxindices_y,
            maxindices_z,
            maxindices_xy,
            maxindices_xz,
            maxindices_yz,
            maxindices_xyz,
        ]:
            if np.array([np.isnan(i) for i in inds.values()]).any():
                with pytest.raises(ValueError):
                    ar.argmax(dim=[d for d in inds])
                return

        result0 = ar.argmax(dim=["x"])
        expected0 = {
            key: xr.DataArray(value, dims=("y", "z"))
            for key, value in maxindices_x.items()
        }
        for key in expected0:
            assert_identical(result0[key].drop_vars(["y", "z"]), expected0[key])

        result1 = ar.argmax(dim=["y"])
        expected1 = {
            key: xr.DataArray(value, dims=("x", "z"))
            for key, value in maxindices_y.items()
        }
        for key in expected1:
            assert_identical(result1[key].drop_vars(["x", "z"]), expected1[key])

        result2 = ar.argmax(dim=["z"])
        expected2 = {
            key: xr.DataArray(value, dims=("x", "y"))
            for key, value in maxindices_z.items()
        }
        for key in expected2:
            assert_identical(result2[key].drop_vars(["x", "y"]), expected2[key])

        result3 = ar.argmax(dim=("x", "y"))
        expected3 = {
            key: xr.DataArray(value, dims=("z")) for key, value in maxindices_xy.items()
        }
        for key in expected3:
            assert_identical(result3[key].drop_vars("z"), expected3[key])

        result4 = ar.argmax(dim=("x", "z"))
        expected4 = {
            key: xr.DataArray(value, dims=("y")) for key, value in maxindices_xz.items()
        }
        for key in expected4:
            assert_identical(result4[key].drop_vars("y"), expected4[key])

        result5 = ar.argmax(dim=("y", "z"))
        expected5 = {
            key: xr.DataArray(value, dims=("x")) for key, value in maxindices_yz.items()
        }
        for key in expected5:
            assert_identical(result5[key].drop_vars("x"), expected5[key])

        result6 = ar.argmax(...)
        expected6 = {key: xr.DataArray(value) for key, value in maxindices_xyz.items()}
        for key in expected6:
            assert_identical(result6[key], expected6[key])

        maxindices_x = {
            key: xr.where(
                nanindices_x[key] == None,  # noqa: E711
                maxindices_x[key],
                nanindices_x[key],
            )
            for key in maxindices_x
        }
        expected7 = {
            key: xr.DataArray(value, dims=("y", "z"))
            for key, value in maxindices_x.items()
        }

        result7 = ar.argmax(dim=["x"], skipna=False)
        for key in expected7:
            assert_identical(result7[key].drop_vars(["y", "z"]), expected7[key])

        maxindices_y = {
            key: xr.where(
                nanindices_y[key] == None,  # noqa: E711
                maxindices_y[key],
                nanindices_y[key],
            )
            for key in maxindices_y
        }
        expected8 = {
            key: xr.DataArray(value, dims=("x", "z"))
            for key, value in maxindices_y.items()
        }

        result8 = ar.argmax(dim=["y"], skipna=False)
        for key in expected8:
            assert_identical(result8[key].drop_vars(["x", "z"]), expected8[key])

        maxindices_z = {
            key: xr.where(
                nanindices_z[key] == None,  # noqa: E711
                maxindices_z[key],
                nanindices_z[key],
            )
            for key in maxindices_z
        }
        expected9 = {
            key: xr.DataArray(value, dims=("x", "y"))
            for key, value in maxindices_z.items()
        }

        result9 = ar.argmax(dim=["z"], skipna=False)
        for key in expected9:
            assert_identical(result9[key].drop_vars(["x", "y"]), expected9[key])

        maxindices_xy = {
            key: xr.where(
                nanindices_xy[key] == None,  # noqa: E711
                maxindices_xy[key],
                nanindices_xy[key],
            )
            for key in maxindices_xy
        }
        expected10 = {
            key: xr.DataArray(value, dims="z") for key, value in maxindices_xy.items()
        }

        result10 = ar.argmax(dim=("x", "y"), skipna=False)
        for key in expected10:
            assert_identical(result10[key].drop_vars("z"), expected10[key])

        maxindices_xz = {
            key: xr.where(
                nanindices_xz[key] == None,  # noqa: E711
                maxindices_xz[key],
                nanindices_xz[key],
            )
            for key in maxindices_xz
        }
        expected11 = {
            key: xr.DataArray(value, dims="y") for key, value in maxindices_xz.items()
        }

        result11 = ar.argmax(dim=("x", "z"), skipna=False)
        for key in expected11:
            assert_identical(result11[key].drop_vars("y"), expected11[key])

        maxindices_yz = {
            key: xr.where(
                nanindices_yz[key] == None,  # noqa: E711
                maxindices_yz[key],
                nanindices_yz[key],
            )
            for key in maxindices_yz
        }
        expected12 = {
            key: xr.DataArray(value, dims="x") for key, value in maxindices_yz.items()
        }

        result12 = ar.argmax(dim=("y", "z"), skipna=False)
        for key in expected12:
            assert_identical(result12[key].drop_vars("x"), expected12[key])

        maxindices_xyz = {
            key: xr.where(
                nanindices_xyz[key] == None,  # noqa: E711
                maxindices_xyz[key],
                nanindices_xyz[key],
            )
            for key in maxindices_xyz
        }
        expected13 = {key: xr.DataArray(value) for key, value in maxindices_xyz.items()}

        result13 = ar.argmax(..., skipna=False)
        for key in expected13:
            assert_identical(result13[key], expected13[key])


class TestReduceND(TestReduce):
    @pytest.mark.parametrize("op", ["idxmin", "idxmax"])
    @pytest.mark.parametrize("ndim", [3, 5])
    def test_idxminmax_dask(self, op, ndim):
        if not has_dask:
            pytest.skip("requires dask")

        ar0_raw = xr.DataArray(
            np.random.random_sample(size=[10] * ndim),
            dims=[i for i in "abcdefghij"[: ndim - 1]] + ["x"],
            coords={"x": np.arange(10)},
            attrs=self.attrs,
        )

        ar0_dsk = ar0_raw.chunk({})
        # Assert idx is the same with dask and without
        assert_equal(getattr(ar0_dsk, op)(dim="x"), getattr(ar0_raw, op)(dim="x"))


@pytest.fixture(params=["numpy", pytest.param("dask", marks=requires_dask)])
def backend(request):
    return request.param


@pytest.fixture(params=[1])
def da(request, backend):
    if request.param == 1:
        times = pd.date_range("2000-01-01", freq="1D", periods=21)
        da = DataArray(
            np.random.random((3, 21, 4)),
            dims=("a", "time", "x"),
            coords=dict(time=times),
        )

    if request.param == 2:
        da = DataArray([0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7], dims="time")

    if request.param == "repeating_ints":
        da = DataArray(
            np.tile(np.arange(12), 5).reshape(5, 4, 3),
            coords={"x": list("abc"), "y": list("defg")},
            dims=list("zyx"),
        )

    if backend == "dask":
        return da.chunk()
    elif backend == "numpy":
        return da
    else:
        raise ValueError


@pytest.fixture
def da_dask(seed=123):
    # TODO: if possible, use the `da` fixture parameterized with backends rather than
    # this.
    pytest.importorskip("dask.array")
    rs = np.random.RandomState(seed)
    times = pd.date_range("2000-01-01", freq="1D", periods=21)
    values = rs.normal(size=(1, 21, 1))
    da = DataArray(values, dims=("a", "time", "x")).chunk({"time": 7})
    da["time"] = times
    return da


@pytest.mark.parametrize("da", ("repeating_ints",), indirect=True)
def test_isin(da):
    expected = DataArray(
        np.asarray([[0, 0, 0], [1, 0, 0]]),
        dims=list("yx"),
        coords={"x": list("abc"), "y": list("de")},
    ).astype("bool")

    result = da.isin([3]).sel(y=list("de"), z=0)
    assert_equal(result, expected)

    expected = DataArray(
        np.asarray([[0, 0, 1], [1, 0, 0]]),
        dims=list("yx"),
        coords={"x": list("abc"), "y": list("de")},
    ).astype("bool")
    result = da.isin([2, 3]).sel(y=list("de"), z=0)
    assert_equal(result, expected)


@pytest.mark.parametrize(
    "funcname, argument",
    [
        ("reduce", (np.mean,)),
        ("mean", ()),
    ],
)
def test_coarsen_keep_attrs(funcname, argument):
    attrs_da = {"da_attr": "test"}
    attrs_coords = {"attrs_coords": "test"}

    data = np.linspace(10, 15, 100)
    coords = np.linspace(1, 10, 100)

    da = DataArray(
        data,
        dims=("coord"),
        coords={"coord": ("coord", coords, attrs_coords)},
        attrs=attrs_da,
        name="name",
    )

    # attrs are now kept per default
    func = getattr(da.coarsen(dim={"coord": 5}), funcname)
    result = func(*argument)
    assert result.attrs == attrs_da
    da.coord.attrs == attrs_coords
    assert result.name == "name"

    # discard attrs
    func = getattr(da.coarsen(dim={"coord": 5}), funcname)
    result = func(*argument, keep_attrs=False)
    assert result.attrs == {}
    da.coord.attrs == {}
    assert result.name == "name"

    # test discard attrs using global option
    func = getattr(da.coarsen(dim={"coord": 5}), funcname)
    with set_options(keep_attrs=False):
        result = func(*argument)
    assert result.attrs == {}
    da.coord.attrs == {}
    assert result.name == "name"

    # keyword takes precedence over global option
    func = getattr(da.coarsen(dim={"coord": 5}), funcname)
    with set_options(keep_attrs=False):
        result = func(*argument, keep_attrs=True)
    assert result.attrs == attrs_da
    da.coord.attrs == {}
    assert result.name == "name"

    func = getattr(da.coarsen(dim={"coord": 5}), funcname)
    with set_options(keep_attrs=True):
        result = func(*argument, keep_attrs=False)
    assert result.attrs == {}
    da.coord.attrs == {}
    assert result.name == "name"


def test_coarsen_keep_attrs_deprecated():
    attrs_da = {"da_attr": "test"}

    data = np.linspace(10, 15, 100)
    coords = np.linspace(1, 10, 100)

    da = DataArray(data, dims=("coord"), coords={"coord": coords}, attrs=attrs_da)

    # deprecated option
    with pytest.warns(
        FutureWarning, match="Passing ``keep_attrs`` to ``coarsen`` is deprecated"
    ):
        result = da.coarsen(dim={"coord": 5}, keep_attrs=False).mean()

    assert result.attrs == {}

    # the keep_attrs in the reduction function takes precedence
    with pytest.warns(
        FutureWarning, match="Passing ``keep_attrs`` to ``coarsen`` is deprecated"
    ):
        result = da.coarsen(dim={"coord": 5}, keep_attrs=True).mean(keep_attrs=False)

    assert result.attrs == {}


@pytest.mark.parametrize("da", (1, 2), indirect=True)
@pytest.mark.parametrize("window", (1, 2, 3, 4))
@pytest.mark.parametrize("name", ("sum", "mean", "std", "max"))
def test_coarsen_reduce(da, window, name):
    if da.isnull().sum() > 1 and window == 1:
        pytest.skip("These parameters lead to all-NaN slices")

    # Use boundary="trim" to accomodate all window sizes used in tests
    coarsen_obj = da.coarsen(time=window, boundary="trim")

    # add nan prefix to numpy methods to get similar # behavior as bottleneck
    actual = coarsen_obj.reduce(getattr(np, f"nan{name}"))
    expected = getattr(coarsen_obj, name)()
    assert_allclose(actual, expected)


@pytest.mark.parametrize("da", (1, 2), indirect=True)
def test_rolling_iter(da):
    rolling_obj = da.rolling(time=7)
    rolling_obj_mean = rolling_obj.mean()

    assert len(rolling_obj.window_labels) == len(da["time"])
    assert_identical(rolling_obj.window_labels, da["time"])

    for i, (label, window_da) in enumerate(rolling_obj):
        assert label == da["time"].isel(time=i)

        actual = rolling_obj_mean.isel(time=i)
        expected = window_da.mean("time")

        # TODO add assert_allclose_with_nan, which compares nan position
        # as well as the closeness of the values.
        assert_array_equal(actual.isnull(), expected.isnull())
        if (~actual.isnull()).sum() > 0:
            np.allclose(
                actual.values[actual.values.nonzero()],
                expected.values[expected.values.nonzero()],
            )


@pytest.mark.parametrize("da", (1,), indirect=True)
def test_rolling_repr(da):
    rolling_obj = da.rolling(time=7)
    assert repr(rolling_obj) == "DataArrayRolling [time->7]"
    rolling_obj = da.rolling(time=7, center=True)
    assert repr(rolling_obj) == "DataArrayRolling [time->7(center)]"
    rolling_obj = da.rolling(time=7, x=3, center=True)
    assert repr(rolling_obj) == "DataArrayRolling [time->7(center),x->3(center)]"


@requires_dask
def test_repeated_rolling_rechunks():

    # regression test for GH3277, GH2514
    dat = DataArray(np.random.rand(7653, 300), dims=("day", "item"))
    dat_chunk = dat.chunk({"item": 20})
    dat_chunk.rolling(day=10).mean().rolling(day=250).std()


def test_rolling_doc(da):
    rolling_obj = da.rolling(time=7)

    # argument substitution worked
    assert "`mean`" in rolling_obj.mean.__doc__


def test_rolling_properties(da):
    rolling_obj = da.rolling(time=4)

    assert rolling_obj.obj.get_axis_num("time") == 1

    # catching invalid args
    with pytest.raises(ValueError, match="window must be > 0"):
        da.rolling(time=-2)

    with pytest.raises(ValueError, match="min_periods must be greater than zero"):
        da.rolling(time=2, min_periods=0)


@pytest.mark.parametrize("name", ("sum", "mean", "std", "min", "max", "median"))
@pytest.mark.parametrize("center", (True, False, None))
@pytest.mark.parametrize("min_periods", (1, None))
@pytest.mark.parametrize("backend", ["numpy"], indirect=True)
def test_rolling_wrapped_bottleneck(da, name, center, min_periods):
    bn = pytest.importorskip("bottleneck", minversion="1.1")

    # Test all bottleneck functions
    rolling_obj = da.rolling(time=7, min_periods=min_periods)

    func_name = f"move_{name}"
    actual = getattr(rolling_obj, name)()
    expected = getattr(bn, func_name)(
        da.values, window=7, axis=1, min_count=min_periods
    )
    assert_array_equal(actual.values, expected)

    with pytest.warns(DeprecationWarning, match="Reductions are applied"):
        getattr(rolling_obj, name)(dim="time")

    # Test center
    rolling_obj = da.rolling(time=7, center=center)
    actual = getattr(rolling_obj, name)()["time"]
    assert_equal(actual, da["time"])


@requires_dask
@pytest.mark.parametrize("name", ("mean", "count"))
@pytest.mark.parametrize("center", (True, False, None))
@pytest.mark.parametrize("min_periods", (1, None))
@pytest.mark.parametrize("window", (7, 8))
def test_rolling_wrapped_dask(da_dask, name, center, min_periods, window):
    # dask version
    rolling_obj = da_dask.rolling(time=window, min_periods=min_periods, center=center)
    actual = getattr(rolling_obj, name)().load()
    if name != "count":
        with pytest.warns(DeprecationWarning, match="Reductions are applied"):
            getattr(rolling_obj, name)(dim="time")
    # numpy version
    rolling_obj = da_dask.load().rolling(
        time=window, min_periods=min_periods, center=center
    )
    expected = getattr(rolling_obj, name)()

    # using all-close because rolling over ghost cells introduces some
    # precision errors
    assert_allclose(actual, expected)

    # with zero chunked array GH:2113
    rolling_obj = da_dask.chunk().rolling(
        time=window, min_periods=min_periods, center=center
    )
    actual = getattr(rolling_obj, name)().load()
    assert_allclose(actual, expected)


@pytest.mark.parametrize("center", (True, None))
def test_rolling_wrapped_dask_nochunk(center):
    # GH:2113
    pytest.importorskip("dask.array")

    da_day_clim = xr.DataArray(
        np.arange(1, 367), coords=[np.arange(1, 367)], dims="dayofyear"
    )
    expected = da_day_clim.rolling(dayofyear=31, center=center).mean()
    actual = da_day_clim.chunk().rolling(dayofyear=31, center=center).mean()
    assert_allclose(actual, expected)


@pytest.mark.parametrize("center", (True, False))
@pytest.mark.parametrize("min_periods", (None, 1, 2, 3))
@pytest.mark.parametrize("window", (1, 2, 3, 4))
def test_rolling_pandas_compat(center, window, min_periods):
    s = pd.Series(np.arange(10))
    da = DataArray.from_series(s)

    if min_periods is not None and window < min_periods:
        min_periods = window

    s_rolling = s.rolling(window, center=center, min_periods=min_periods).mean()
    da_rolling = da.rolling(index=window, center=center, min_periods=min_periods).mean()
    da_rolling_np = da.rolling(
        index=window, center=center, min_periods=min_periods
    ).reduce(np.nanmean)

    np.testing.assert_allclose(s_rolling.values, da_rolling.values)
    np.testing.assert_allclose(s_rolling.index, da_rolling["index"])
    np.testing.assert_allclose(s_rolling.values, da_rolling_np.values)
    np.testing.assert_allclose(s_rolling.index, da_rolling_np["index"])


@pytest.mark.parametrize("center", (True, False))
@pytest.mark.parametrize("window", (1, 2, 3, 4))
def test_rolling_construct(center, window):
    s = pd.Series(np.arange(10))
    da = DataArray.from_series(s)

    s_rolling = s.rolling(window, center=center, min_periods=1).mean()
    da_rolling = da.rolling(index=window, center=center, min_periods=1)

    da_rolling_mean = da_rolling.construct("window").mean("window")
    np.testing.assert_allclose(s_rolling.values, da_rolling_mean.values)
    np.testing.assert_allclose(s_rolling.index, da_rolling_mean["index"])

    # with stride
    da_rolling_mean = da_rolling.construct("window", stride=2).mean("window")
    np.testing.assert_allclose(s_rolling.values[::2], da_rolling_mean.values)
    np.testing.assert_allclose(s_rolling.index[::2], da_rolling_mean["index"])

    # with fill_value
    da_rolling_mean = da_rolling.construct("window", stride=2, fill_value=0.0).mean(
        "window"
    )
    assert da_rolling_mean.isnull().sum() == 0
    assert (da_rolling_mean == 0.0).sum() >= 0


@pytest.mark.parametrize("da", (1, 2), indirect=True)
@pytest.mark.parametrize("center", (True, False))
@pytest.mark.parametrize("min_periods", (None, 1, 2, 3))
@pytest.mark.parametrize("window", (1, 2, 3, 4))
@pytest.mark.parametrize("name", ("sum", "mean", "std", "max"))
def test_rolling_reduce(da, center, min_periods, window, name):
    if min_periods is not None and window < min_periods:
        min_periods = window

    if da.isnull().sum() > 1 and window == 1:
        # this causes all nan slices
        window = 2

    rolling_obj = da.rolling(time=window, center=center, min_periods=min_periods)

    # add nan prefix to numpy methods to get similar # behavior as bottleneck
    actual = rolling_obj.reduce(getattr(np, "nan%s" % name))
    expected = getattr(rolling_obj, name)()
    assert_allclose(actual, expected)
    assert actual.dims == expected.dims


@pytest.mark.parametrize("center", (True, False))
@pytest.mark.parametrize("min_periods", (None, 1, 2, 3))
@pytest.mark.parametrize("window", (1, 2, 3, 4))
@pytest.mark.parametrize("name", ("sum", "max"))
def test_rolling_reduce_nonnumeric(center, min_periods, window, name):
    da = DataArray(
        [0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7], dims="time"
    ).isnull()

    if min_periods is not None and window < min_periods:
        min_periods = window

    rolling_obj = da.rolling(time=window, center=center, min_periods=min_periods)

    # add nan prefix to numpy methods to get similar behavior as bottleneck
    actual = rolling_obj.reduce(getattr(np, "nan%s" % name))
    expected = getattr(rolling_obj, name)()
    assert_allclose(actual, expected)
    assert actual.dims == expected.dims


def test_rolling_count_correct():
    da = DataArray([0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7], dims="time")

    kwargs = [
        {"time": 11, "min_periods": 1},
        {"time": 11, "min_periods": None},
        {"time": 7, "min_periods": 2},
    ]
    expecteds = [
        DataArray([1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8], dims="time"),
        DataArray(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            dims="time",
        ),
        DataArray([np.nan, np.nan, 2, 3, 3, 4, 5, 5, 5, 5, 5], dims="time"),
    ]

    for kwarg, expected in zip(kwargs, expecteds):
        result = da.rolling(**kwarg).count()
        assert_equal(result, expected)

        result = da.to_dataset(name="var1").rolling(**kwarg).count()["var1"]
        assert_equal(result, expected)


@pytest.mark.parametrize("da", (1,), indirect=True)
@pytest.mark.parametrize("center", (True, False))
@pytest.mark.parametrize("min_periods", (None, 1))
@pytest.mark.parametrize("name", ("sum", "mean", "max"))
def test_ndrolling_reduce(da, center, min_periods, name):
    rolling_obj = da.rolling(time=3, x=2, center=center, min_periods=min_periods)

    actual = getattr(rolling_obj, name)()
    expected = getattr(
        getattr(
            da.rolling(time=3, center=center, min_periods=min_periods), name
        )().rolling(x=2, center=center, min_periods=min_periods),
        name,
    )()

    assert_allclose(actual, expected)
    assert actual.dims == expected.dims

    if name in ["mean"]:
        # test our reimplementation of nanmean using np.nanmean
        expected = getattr(rolling_obj.construct({"time": "tw", "x": "xw"}), name)(
            ["tw", "xw"]
        )
        count = rolling_obj.count()
        if min_periods is None:
            min_periods = 1
        assert_allclose(actual, expected.where(count >= min_periods))


@pytest.mark.parametrize("center", (True, False, (True, False)))
@pytest.mark.parametrize("fill_value", (np.nan, 0.0))
def test_ndrolling_construct(center, fill_value):
    da = DataArray(
        np.arange(5 * 6 * 7).reshape(5, 6, 7).astype(float),
        dims=["x", "y", "z"],
        coords={"x": ["a", "b", "c", "d", "e"], "y": np.arange(6)},
    )
    actual = da.rolling(x=3, z=2, center=center).construct(
        x="x1", z="z1", fill_value=fill_value
    )
    if not isinstance(center, tuple):
        center = (center, center)
    expected = (
        da.rolling(x=3, center=center[0])
        .construct(x="x1", fill_value=fill_value)
        .rolling(z=2, center=center[1])
        .construct(z="z1", fill_value=fill_value)
    )
    assert_allclose(actual, expected)


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
    attrs_da = {"da_attr": "test"}

    data = np.linspace(10, 15, 100)
    coords = np.linspace(1, 10, 100)

    da = DataArray(
        data, dims=("coord"), coords={"coord": coords}, attrs=attrs_da, name="name"
    )

    # attrs are now kept per default
    func = getattr(da.rolling(dim={"coord": 5}), funcname)
    result = func(*argument)
    assert result.attrs == attrs_da
    assert result.name == "name"

    # discard attrs
    func = getattr(da.rolling(dim={"coord": 5}), funcname)
    result = func(*argument, keep_attrs=False)
    assert result.attrs == {}
    assert result.name == "name"

    # test discard attrs using global option
    func = getattr(da.rolling(dim={"coord": 5}), funcname)
    with set_options(keep_attrs=False):
        result = func(*argument)
    assert result.attrs == {}
    assert result.name == "name"

    # keyword takes precedence over global option
    func = getattr(da.rolling(dim={"coord": 5}), funcname)
    with set_options(keep_attrs=False):
        result = func(*argument, keep_attrs=True)
    assert result.attrs == attrs_da
    assert result.name == "name"

    func = getattr(da.rolling(dim={"coord": 5}), funcname)
    with set_options(keep_attrs=True):
        result = func(*argument, keep_attrs=False)
    assert result.attrs == {}
    assert result.name == "name"


def test_rolling_keep_attrs_deprecated():
    attrs_da = {"da_attr": "test"}

    data = np.linspace(10, 15, 100)
    coords = np.linspace(1, 10, 100)

    da = DataArray(data, dims=("coord"), coords={"coord": coords}, attrs=attrs_da)

    # deprecated option
    with pytest.warns(
        FutureWarning, match="Passing ``keep_attrs`` to ``rolling`` is deprecated"
    ):
        result = da.rolling(dim={"coord": 5}, keep_attrs=False).construct("window_dim")

    assert result.attrs == {}

    # the keep_attrs in the reduction function takes precedence
    with pytest.warns(
        FutureWarning, match="Passing ``keep_attrs`` to ``rolling`` is deprecated"
    ):
        result = da.rolling(dim={"coord": 5}, keep_attrs=True).construct(
            "window_dim", keep_attrs=False
        )

    assert result.attrs == {}


def test_raise_no_warning_for_nan_in_binary_ops():
    with pytest.warns(None) as record:
        xr.DataArray([1, 2, np.NaN]) > 0
    assert len(record) == 0


@pytest.mark.filterwarnings("error")
def test_no_warning_for_all_nan():
    _ = xr.DataArray([np.NaN, np.NaN]).mean()


def test_name_in_masking():
    name = "RingoStarr"
    da = xr.DataArray(range(10), coords=[("x", range(10))], name=name)
    assert da.where(da > 5).name == name
    assert da.where((da > 5).rename("YokoOno")).name == name
    assert da.where(da > 5, drop=True).name == name
    assert da.where((da > 5).rename("YokoOno"), drop=True).name == name


class TestIrisConversion:
    @requires_iris
    def test_to_and_from_iris(self):
        import cf_units  # iris requirement
        import iris

        # to iris
        coord_dict = {}
        coord_dict["distance"] = ("distance", [-2, 2], {"units": "meters"})
        coord_dict["time"] = ("time", pd.date_range("2000-01-01", periods=3))
        coord_dict["height"] = 10
        coord_dict["distance2"] = ("distance", [0, 1], {"foo": "bar"})
        coord_dict["time2"] = (("distance", "time"), [[0, 1, 2], [2, 3, 4]])

        original = DataArray(
            np.arange(6, dtype="float").reshape(2, 3),
            coord_dict,
            name="Temperature",
            attrs={
                "baz": 123,
                "units": "Kelvin",
                "standard_name": "fire_temperature",
                "long_name": "Fire Temperature",
            },
            dims=("distance", "time"),
        )

        # Set a bad value to test the masking logic
        original.data[0, 2] = np.NaN

        original.attrs["cell_methods"] = "height: mean (comment: A cell method)"
        actual = original.to_iris()
        assert_array_equal(actual.data, original.data)
        assert actual.var_name == original.name
        assert tuple(d.var_name for d in actual.dim_coords) == original.dims
        assert actual.cell_methods == (
            iris.coords.CellMethod(
                method="mean",
                coords=("height",),
                intervals=(),
                comments=("A cell method",),
            ),
        )

        for coord, orginal_key in zip((actual.coords()), original.coords):
            original_coord = original.coords[orginal_key]
            assert coord.var_name == original_coord.name
            assert_array_equal(
                coord.points, CFDatetimeCoder().encode(original_coord).values
            )
            assert actual.coord_dims(coord) == original.get_axis_num(
                original.coords[coord.var_name].dims
            )

        assert (
            actual.coord("distance2").attributes["foo"]
            == original.coords["distance2"].attrs["foo"]
        )
        assert actual.coord("distance").units == cf_units.Unit(
            original.coords["distance"].units
        )
        assert actual.attributes["baz"] == original.attrs["baz"]
        assert actual.standard_name == original.attrs["standard_name"]

        roundtripped = DataArray.from_iris(actual)
        assert_identical(original, roundtripped)

        actual.remove_coord("time")
        auto_time_dimension = DataArray.from_iris(actual)
        assert auto_time_dimension.dims == ("distance", "dim_1")

    @requires_iris
    @requires_dask
    def test_to_and_from_iris_dask(self):
        import cf_units  # iris requirement
        import dask.array as da
        import iris

        coord_dict = {}
        coord_dict["distance"] = ("distance", [-2, 2], {"units": "meters"})
        coord_dict["time"] = ("time", pd.date_range("2000-01-01", periods=3))
        coord_dict["height"] = 10
        coord_dict["distance2"] = ("distance", [0, 1], {"foo": "bar"})
        coord_dict["time2"] = (("distance", "time"), [[0, 1, 2], [2, 3, 4]])

        original = DataArray(
            da.from_array(np.arange(-1, 5, dtype="float").reshape(2, 3), 3),
            coord_dict,
            name="Temperature",
            attrs=dict(
                baz=123,
                units="Kelvin",
                standard_name="fire_temperature",
                long_name="Fire Temperature",
            ),
            dims=("distance", "time"),
        )

        # Set a bad value to test the masking logic
        original.data = da.ma.masked_less(original.data, 0)

        original.attrs["cell_methods"] = "height: mean (comment: A cell method)"
        actual = original.to_iris()

        # Be careful not to trigger the loading of the iris data
        actual_data = (
            actual.core_data() if hasattr(actual, "core_data") else actual.data
        )
        assert_array_equal(actual_data, original.data)
        assert actual.var_name == original.name
        assert tuple(d.var_name for d in actual.dim_coords) == original.dims
        assert actual.cell_methods == (
            iris.coords.CellMethod(
                method="mean",
                coords=("height",),
                intervals=(),
                comments=("A cell method",),
            ),
        )

        for coord, orginal_key in zip((actual.coords()), original.coords):
            original_coord = original.coords[orginal_key]
            assert coord.var_name == original_coord.name
            assert_array_equal(
                coord.points, CFDatetimeCoder().encode(original_coord).values
            )
            assert actual.coord_dims(coord) == original.get_axis_num(
                original.coords[coord.var_name].dims
            )

        assert (
            actual.coord("distance2").attributes["foo"]
            == original.coords["distance2"].attrs["foo"]
        )
        assert actual.coord("distance").units == cf_units.Unit(
            original.coords["distance"].units
        )
        assert actual.attributes["baz"] == original.attrs["baz"]
        assert actual.standard_name == original.attrs["standard_name"]

        roundtripped = DataArray.from_iris(actual)
        assert_identical(original, roundtripped)

        # If the Iris version supports it then we should have a dask array
        # at each stage of the conversion
        if hasattr(actual, "core_data"):
            assert isinstance(original.data, type(actual.core_data()))
            assert isinstance(original.data, type(roundtripped.data))

        actual.remove_coord("time")
        auto_time_dimension = DataArray.from_iris(actual)
        assert auto_time_dimension.dims == ("distance", "dim_1")

    @requires_iris
    @pytest.mark.parametrize(
        "var_name, std_name, long_name, name, attrs",
        [
            (
                "var_name",
                "height",
                "Height",
                "var_name",
                {"standard_name": "height", "long_name": "Height"},
            ),
            (
                None,
                "height",
                "Height",
                "height",
                {"standard_name": "height", "long_name": "Height"},
            ),
            (None, None, "Height", "Height", {"long_name": "Height"}),
            (None, None, None, None, {}),
        ],
    )
    def test_da_name_from_cube(self, std_name, long_name, var_name, name, attrs):
        from iris.cube import Cube

        data = []
        cube = Cube(
            data, var_name=var_name, standard_name=std_name, long_name=long_name
        )
        result = xr.DataArray.from_iris(cube)
        expected = xr.DataArray(data, name=name, attrs=attrs)
        xr.testing.assert_identical(result, expected)

    @requires_iris
    @pytest.mark.parametrize(
        "var_name, std_name, long_name, name, attrs",
        [
            (
                "var_name",
                "height",
                "Height",
                "var_name",
                {"standard_name": "height", "long_name": "Height"},
            ),
            (
                None,
                "height",
                "Height",
                "height",
                {"standard_name": "height", "long_name": "Height"},
            ),
            (None, None, "Height", "Height", {"long_name": "Height"}),
            (None, None, None, "unknown", {}),
        ],
    )
    def test_da_coord_name_from_cube(self, std_name, long_name, var_name, name, attrs):
        from iris.coords import DimCoord
        from iris.cube import Cube

        latitude = DimCoord(
            [-90, 0, 90], standard_name=std_name, var_name=var_name, long_name=long_name
        )
        data = [0, 0, 0]
        cube = Cube(data, dim_coords_and_dims=[(latitude, 0)])
        result = xr.DataArray.from_iris(cube)
        expected = xr.DataArray(data, coords=[(name, [-90, 0, 90], attrs)])
        xr.testing.assert_identical(result, expected)

    @requires_iris
    def test_prevent_duplicate_coord_names(self):
        from iris.coords import DimCoord
        from iris.cube import Cube

        # Iris enforces unique coordinate names. Because we use a different
        # name resolution order a valid iris Cube with coords that have the
        # same var_name would lead to duplicate dimension names in the
        # DataArray
        longitude = DimCoord([0, 360], standard_name="longitude", var_name="duplicate")
        latitude = DimCoord(
            [-90, 0, 90], standard_name="latitude", var_name="duplicate"
        )
        data = [[0, 0, 0], [0, 0, 0]]
        cube = Cube(data, dim_coords_and_dims=[(longitude, 0), (latitude, 1)])
        with pytest.raises(ValueError):
            xr.DataArray.from_iris(cube)

    @requires_iris
    @pytest.mark.parametrize(
        "coord_values",
        [["IA", "IL", "IN"], [0, 2, 1]],  # non-numeric values  # non-monotonic values
    )
    def test_fallback_to_iris_AuxCoord(self, coord_values):
        from iris.coords import AuxCoord
        from iris.cube import Cube

        data = [0, 0, 0]
        da = xr.DataArray(data, coords=[coord_values], dims=["space"])
        result = xr.DataArray.to_iris(da)
        expected = Cube(
            data, aux_coords_and_dims=[(AuxCoord(coord_values, var_name="space"), 0)]
        )
        assert result == expected


@requires_numbagg
@pytest.mark.parametrize("dim", ["time", "x"])
@pytest.mark.parametrize(
    "window_type, window", [["span", 5], ["alpha", 0.5], ["com", 0.5], ["halflife", 5]]
)
@pytest.mark.parametrize("backend", ["numpy"], indirect=True)
def test_rolling_exp(da, dim, window_type, window):
    da = da.isel(a=0)
    da = da.where(da > 0.2)

    result = da.rolling_exp(window_type=window_type, **{dim: window}).mean()
    assert isinstance(result, DataArray)

    pandas_array = da.to_pandas()
    assert pandas_array.index.name == "time"
    if dim == "x":
        pandas_array = pandas_array.T
    expected = xr.DataArray(pandas_array.ewm(**{window_type: window}).mean()).transpose(
        *da.dims
    )

    assert_allclose(expected.variable, result.variable)


@requires_numbagg
@pytest.mark.parametrize("backend", ["numpy"], indirect=True)
def test_rolling_exp_keep_attrs(da):
    attrs = {"attrs": "da"}
    da.attrs = attrs

    # attrs are kept per default
    result = da.rolling_exp(time=10).mean()
    assert result.attrs == attrs

    # discard attrs
    result = da.rolling_exp(time=10).mean(keep_attrs=False)
    assert result.attrs == {}

    # test discard attrs using global option
    with set_options(keep_attrs=False):
        result = da.rolling_exp(time=10).mean()
    assert result.attrs == {}

    # keyword takes precedence over global option
    with set_options(keep_attrs=False):
        result = da.rolling_exp(time=10).mean(keep_attrs=True)
    assert result.attrs == attrs

    with set_options(keep_attrs=True):
        result = da.rolling_exp(time=10).mean(keep_attrs=False)
    assert result.attrs == {}

    with pytest.warns(
        UserWarning, match="Passing ``keep_attrs`` to ``rolling_exp`` has no effect."
    ):
        da.rolling_exp(time=10, keep_attrs=True)


def test_no_dict():
    d = DataArray()
    with pytest.raises(AttributeError):
        d.__dict__


def test_subclass_slots():
    """Test that DataArray subclasses must explicitly define ``__slots__``.

    .. note::
       As of 0.13.0, this is actually mitigated into a FutureWarning for any class
       defined outside of the xarray package.
    """
    with pytest.raises(AttributeError) as e:

        class MyArray(DataArray):
            pass

    assert str(e.value) == "MyArray must explicitly define __slots__"


def test_weakref():
    """Classes with __slots__ are incompatible with the weakref module unless they
    explicitly state __weakref__ among their slots
    """
    from weakref import ref

    a = DataArray(1)
    r = ref(a)
    assert r() is a


def test_delete_coords():
    """Make sure that deleting a coordinate doesn't corrupt the DataArray.
    See issue #3899.

    Also test that deleting succeeds and produces the expected output.
    """
    a0 = DataArray(
        np.array([[1, 2, 3], [4, 5, 6]]),
        dims=["y", "x"],
        coords={"x": ["a", "b", "c"], "y": [-1, 1]},
    )
    assert_identical(a0, a0)

    a1 = a0.copy()
    del a1.coords["y"]

    # This test will detect certain sorts of corruption in the DataArray
    assert_identical(a0, a0)

    assert a0.dims == ("y", "x")
    assert a1.dims == ("y", "x")
    assert set(a0.coords.keys()) == {"x", "y"}
    assert set(a1.coords.keys()) == {"x"}


def test_deepcopy_obj_array():
    x0 = DataArray(np.array([object()]))
    x1 = deepcopy(x0)
    assert x0.values[0] is not x1.values[0]


def test_clip(da):
    with raise_if_dask_computes():
        result = da.clip(min=0.5)
    assert result.min(...) >= 0.5

    result = da.clip(max=0.5)
    assert result.max(...) <= 0.5

    result = da.clip(min=0.25, max=0.75)
    assert result.min(...) >= 0.25
    assert result.max(...) <= 0.75

    with raise_if_dask_computes():
        result = da.clip(min=da.mean("x"), max=da.mean("a"))
    assert result.dims == da.dims
    assert_array_equal(
        result.data,
        np.clip(da.data, da.mean("x").data[:, :, np.newaxis], da.mean("a").data),
    )

    with_nans = da.isel(time=[0, 1]).reindex_like(da)
    with raise_if_dask_computes():
        result = da.clip(min=da.mean("x"), max=da.mean("a"))
    result = da.clip(with_nans)
    # The values should be the same where there were NaNs.
    assert_array_equal(result.isel(time=[0, 1]), with_nans.isel(time=[0, 1]))

    # Unclear whether we want this work, OK to adjust the test when we have decided.
    with pytest.raises(ValueError, match="arguments without labels along dimension"):
        result = da.clip(min=da.mean("x"), max=da.mean("a").isel(x=[0, 1]))


@pytest.mark.parametrize("keep", ["first", "last", False])
def test_drop_duplicates(keep):
    ds = xr.DataArray(
        [0, 5, 6, 7], dims="time", coords={"time": [0, 0, 1, 2]}, name="test"
    )

    if keep == "first":
        data = [0, 6, 7]
        time = [0, 1, 2]
    elif keep == "last":
        data = [5, 6, 7]
        time = [0, 1, 2]
    else:
        data = [6, 7]
        time = [1, 2]

    expected = xr.DataArray(data, dims="time", coords={"time": time}, name="test")
    result = ds.drop_duplicates("time", keep=keep)
    assert_equal(expected, result)
