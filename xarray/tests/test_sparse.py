from collections import OrderedDict
from contextlib import suppress
from distutils.version import LooseVersion
from textwrap import dedent
import pickle
import numpy as np
import pandas as pd

from xarray import DataArray, Dataset, Variable
from xarray.tests import mock
from xarray.core.npcompat import IS_NEP18_ACTIVE
import xarray as xr
import xarray.ufuncs as xu

from . import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_frame_equal,
    assert_identical,
    raises_regex,
)

import pytest

param = pytest.param
xfail = pytest.mark.xfail

if not IS_NEP18_ACTIVE:
    pytest.skip(
        "NUMPY_EXPERIMENTAL_ARRAY_FUNCTION is not enabled", allow_module_level=True
    )

sparse = pytest.importorskip("sparse")
from sparse.utils import assert_eq as assert_sparse_eq  # noqa
from sparse import COO, SparseArray  # noqa


def make_ndarray(shape):
    return np.arange(np.prod(shape)).reshape(shape)


def make_sparray(shape):
    return sparse.random(shape, density=0.1, random_state=0)


def make_xrvar(dim_lengths):
    return xr.Variable(
        tuple(dim_lengths.keys()), make_sparray(shape=tuple(dim_lengths.values()))
    )


def make_xrarray(dim_lengths, coords=None, name="test"):
    if coords is None:
        coords = {d: np.arange(n) for d, n in dim_lengths.items()}
    return xr.DataArray(
        make_sparray(shape=tuple(dim_lengths.values())),
        dims=tuple(coords.keys()),
        coords=coords,
        name=name,
    )


class do:
    def __init__(self, meth, *args, **kwargs):
        self.meth = meth
        self.args = args
        self.kwargs = kwargs

    def __call__(self, obj):
        return getattr(obj, self.meth)(*self.args, **self.kwargs)

    def __repr__(self):
        return "obj.{}(*{}, **{})".format(self.meth, self.args, self.kwargs)


@pytest.mark.parametrize(
    "prop",
    [
        "chunks",
        "data",
        "dims",
        "dtype",
        "encoding",
        "imag",
        "nbytes",
        "ndim",
        param("values", marks=xfail(reason="Coercion to dense")),
    ],
)
def test_variable_property(prop):
    var = make_xrvar({"x": 10, "y": 5})
    getattr(var, prop)


@pytest.mark.parametrize(
    "func,sparse_output",
    [
        (do("all"), False),
        (do("any"), False),
        (do("astype", dtype=int), True),
        (do("broadcast_equals", make_xrvar({"x": 10, "y": 5})), False),
        (do("clip", min=0, max=1), True),
        (do("coarsen", windows={"x": 2}, func=np.sum), True),
        (do("compute"), True),
        (do("conj"), True),
        (do("copy"), True),
        (do("count"), False),
        (do("equals", make_xrvar({"x": 10, "y": 5})), False),
        (do("get_axis_num", dim="x"), False),
        (do("identical", other=make_xrvar({"x": 10, "y": 5})), False),
        (do("isel", x=slice(2, 4)), True),
        (do("isnull"), True),
        (do("load"), True),
        (do("mean"), False),
        (do("notnull"), True),
        (do("roll"), True),
        (do("round"), True),
        (do("set_dims", dims=("x", "y", "z")), True),
        (do("stack", dimensions={"flat": ("x", "y")}), True),
        (do("to_base_variable"), True),
        (do("transpose"), True),
        (do("unstack", dimensions={"x": {"x1": 5, "x2": 2}}), True),
        param(
            do("argmax"),
            True,
            marks=xfail(reason="Missing implementation for np.argmin"),
        ),
        param(
            do("argmin"),
            True,
            marks=xfail(reason="Missing implementation for np.argmax"),
        ),
        param(
            do("argsort"),
            True,
            marks=xfail(reason="'COO' object has no attribute 'argsort'"),
        ),
        param(do("chunk", chunks=(5, 5)), True, marks=xfail),
        param(
            do(
                "concat",
                variables=[
                    make_xrvar({"x": 10, "y": 5}),
                    make_xrvar({"x": 10, "y": 5}),
                ],
            ),
            True,
            marks=xfail(reason="Coercion to dense"),
        ),
        param(
            do("conjugate"),
            True,
            marks=xfail(reason="'COO' object has no attribute 'conjugate'"),
        ),
        param(
            do("cumprod"),
            True,
            marks=xfail(reason="Missing implementation for np.nancumprod"),
        ),
        param(
            do("cumsum"),
            True,
            marks=xfail(reason="Missing implementation for np.nancumsum"),
        ),
        param(
            do("fillna", 0),
            True,
            marks=xfail(reason="Missing implementation for np.result_type"),
        ),
        param(
            do("item", (1, 1)),
            False,
            marks=xfail(reason="'COO' object has no attribute 'item'"),
        ),
        param(do("max"), False, marks=xfail(reason="Coercion to dense via bottleneck")),
        param(
            do("median"), False, marks=xfail(reason="Coercion to dense via bottleneck")
        ),
        param(do("min"), False, marks=xfail(reason="Coercion to dense via bottleneck")),
        param(
            do("no_conflicts", other=make_xrvar({"x": 10, "y": 5})),
            True,
            marks=xfail(reason="mixed sparse-dense operation"),
        ),
        param(
            do("pad_with_fill_value", pad_widths={"x": (1, 1)}, fill_value=5),
            True,  # noqa
            marks=xfail(reason="Missing implementation for np.pad"),
        ),
        param(
            do("prod"),
            False,
            marks=xfail(reason="Missing implementation for np.result_type"),
        ),
        param(
            do("quantile", q=0.5),
            True,
            marks=xfail(reason="Missing implementation for np.nanpercentile"),
        ),
        param(
            do("rank", dim="x"),
            False,
            marks=xfail(reason="Coercion to dense via bottleneck"),
        ),
        param(
            do("reduce", func=np.sum, dim="x"),
            True,
            marks=xfail(reason="Coercion to dense"),
        ),
        param(
            do("rolling_window", dim="x", window=2, window_dim="x_win"),
            True,
            marks=xfail(reason="Missing implementation for np.pad"),
        ),
        param(
            do("shift", x=2), True, marks=xfail(reason="mixed sparse-dense operation")
        ),
        param(do("std"), False, marks=xfail(reason="Coercion to dense via bottleneck")),
        param(
            do("sum"),
            False,
            marks=xfail(reason="Missing implementation for np.result_type"),
        ),
        param(do("var"), False, marks=xfail(reason="Coercion to dense via bottleneck")),
        param(do("to_dict"), False, marks=xfail(reason="Coercion to dense")),
        param(
            do("where", cond=make_xrvar({"x": 10, "y": 5}) > 0.5),
            True,
            marks=xfail(reason="Coercion of dense to sparse when using sparse mask"),
        ),  # noqa
    ],
    ids=repr,
)
def test_variable_method(func, sparse_output):
    var_s = make_xrvar({"x": 10, "y": 5})
    var_d = xr.Variable(var_s.dims, var_s.data.todense())
    ret_s = func(var_s)
    ret_d = func(var_d)

    if sparse_output:
        assert isinstance(ret_s.data, SparseArray)
        assert np.allclose(ret_s.data.todense(), ret_d.data, equal_nan=True)
    else:
        assert np.allclose(ret_s, ret_d, equal_nan=True)


@pytest.mark.parametrize(
    "func,sparse_output",
    [
        (do("squeeze"), True),
        param(do("to_index"), False, marks=xfail(reason="Coercion to dense")),
        param(do("to_index_variable"), False, marks=xfail(reason="Coercion to dense")),
        param(
            do("searchsorted", 0.5),
            True,
            marks=xfail(reason="'COO' object has no attribute 'searchsorted'"),
        ),
    ],
)
def test_1d_variable_method(func, sparse_output):
    var_s = make_xrvar({"x": 10})
    var_d = xr.Variable(var_s.dims, var_s.data.todense())
    ret_s = func(var_s)
    ret_d = func(var_d)

    if sparse_output:
        assert isinstance(ret_s.data, SparseArray)
        assert np.allclose(ret_s.data.todense(), ret_d.data)
    else:
        assert np.allclose(ret_s, ret_d)


class TestSparseVariable:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.data = sparse.random((4, 6), random_state=0, density=0.5)
        self.var = xr.Variable(("x", "y"), self.data)

    def test_unary_op(self):
        assert_sparse_eq(-self.var.data, -self.data)
        assert_sparse_eq(abs(self.var).data, abs(self.data))
        assert_sparse_eq(self.var.round().data, self.data.round())

    def test_univariate_ufunc(self):
        assert_sparse_eq(np.sin(self.data), xu.sin(self.var).data)

    def test_bivariate_ufunc(self):
        assert_sparse_eq(np.maximum(self.data, 0), xu.maximum(self.var, 0).data)
        assert_sparse_eq(np.maximum(self.data, 0), xu.maximum(0, self.var).data)

    def test_repr(self):
        expected = dedent(
            """\
        <xarray.Variable (x: 4, y: 6)>
        <COO: shape=(4, 6), dtype=float64, nnz=12, fill_value=0.0>"""
        )
        assert expected == repr(self.var)

    def test_pickle(self):
        v1 = self.var
        v2 = pickle.loads(pickle.dumps(v1))
        assert_sparse_eq(v1.data, v2.data)

    @pytest.mark.xfail(reason="Missing implementation for np.result_type")
    def test_missing_values(self):
        a = np.array([0, 1, np.nan, 3])
        s = COO.from_numpy(a)
        var_s = Variable("x", s)
        assert np.all(var_s.fillna(2).data.todense() == np.arange(4))
        assert np.all(var_s.count() == 3)


@pytest.mark.parametrize(
    "prop",
    [
        "attrs",
        "chunks",
        "coords",
        "data",
        "dims",
        "dtype",
        "encoding",
        "imag",
        "indexes",
        "loc",
        "name",
        "nbytes",
        "ndim",
        "plot",
        "real",
        "shape",
        "size",
        "sizes",
        "str",
        "variable",
    ],
)
def test_dataarray_property(prop):
    arr = make_xrarray({"x": 10, "y": 5})
    getattr(arr, prop)


@pytest.mark.parametrize(
    "func,sparse_output",
    [
        (do("all"), False),
        (do("any"), False),
        (do("assign_attrs", {"foo": "bar"}), True),
        (do("assign_coords", x=make_xrarray({"x": 10}).x + 1), True),
        (do("astype", int), True),
        (do("broadcast_equals", make_xrarray({"x": 10, "y": 5})), False),
        (do("clip", min=0, max=1), True),
        (do("compute"), True),
        (do("conj"), True),
        (do("copy"), True),
        (do("count"), False),
        (do("diff", "x"), True),
        (do("drop", "x"), True),
        (do("equals", make_xrarray({"x": 10, "y": 5})), False),
        (do("expand_dims", {"z": 2}, axis=2), True),
        (do("get_axis_num", "x"), False),
        (do("get_index", "x"), False),
        (do("identical", make_xrarray({"x": 5, "y": 5})), False),
        (do("integrate", "x"), True),
        (do("isel", {"x": slice(0, 3), "y": slice(2, 4)}), True),
        (do("isnull"), True),
        (do("load"), True),
        (do("mean"), False),
        (do("persist"), True),
        (do("reindex", {"x": [1, 2, 3]}), True),
        (do("rename", "foo"), True),
        (do("reorder_levels"), True),
        (do("reset_coords", drop=True), True),
        (do("reset_index", "x"), True),
        (do("round"), True),
        (do("sel", x=[0, 1, 2]), True),
        (do("shift"), True),
        (do("sortby", "x", ascending=False), True),
        (do("stack", z={"x", "y"}), True),
        (do("transpose"), True),
        # TODO
        # isel_points
        # sel_points
        # set_index
        # swap_dims
        param(
            do("argmax"),
            True,
            marks=xfail(reason="Missing implementation for np.argmax"),
        ),
        param(
            do("argmin"),
            True,
            marks=xfail(reason="Missing implementation for np.argmin"),
        ),
        param(
            do("argsort"),
            True,
            marks=xfail(reason="'COO' object has no attribute 'argsort'"),
        ),
        param(
            do("bfill", dim="x"),
            False,
            marks=xfail(reason="Missing implementation for np.flip"),
        ),
        param(
            do("chunk", chunks=(5, 5)), False, marks=xfail(reason="Coercion to dense")
        ),
        param(
            do("combine_first", make_xrarray({"x": 10, "y": 5})),
            True,
            marks=xfail(reason="mixed sparse-dense operation"),
        ),
        param(
            do("conjugate"),
            False,
            marks=xfail(reason="'COO' object has no attribute 'conjugate'"),
        ),
        param(
            do("cumprod"),
            True,
            marks=xfail(reason="Missing implementation for np.nancumprod"),
        ),
        param(
            do("cumsum"),
            True,
            marks=xfail(reason="Missing implementation for np.nancumsum"),
        ),
        param(
            do("differentiate", "x"),
            False,
            marks=xfail(reason="Missing implementation for np.gradient"),
        ),
        param(
            do("dot", make_xrarray({"x": 10, "y": 5})),
            True,
            marks=xfail(reason="Missing implementation for np.einsum"),
        ),
        param(do("dropna", "x"), False, marks=xfail(reason="Coercion to dense")),
        param(
            do("ffill", "x"),
            False,
            marks=xfail(reason="Coercion to dense via bottleneck.push"),
        ),
        param(
            do("fillna", 0),
            True,
            marks=xfail(reason="Missing implementation for np.result_type"),
        ),
        param(
            do("interp", coords={"x": np.arange(10) + 0.5}),
            True,
            marks=xfail(reason="Coercion to dense"),
        ),
        param(
            do(
                "interp_like",
                make_xrarray(
                    {"x": 10, "y": 5},
                    coords={"x": np.arange(10) + 0.5, "y": np.arange(5) + 0.5},
                ),
            ),
            True,
            marks=xfail(reason="Indexing COO with more than one iterable index"),
        ),  # noqa
        param(do("interpolate_na", "x"), True, marks=xfail(reason="Coercion to dense")),
        param(
            do("isin", [1, 2, 3]),
            False,
            marks=xfail(reason="Missing implementation for np.isin"),
        ),
        param(
            do("item", (1, 1)),
            False,
            marks=xfail(reason="'COO' object has no attribute 'item'"),
        ),
        param(do("max"), False, marks=xfail(reason="Coercion to dense via bottleneck")),
        param(
            do("median"), False, marks=xfail(reason="Coercion to dense via bottleneck")
        ),
        param(do("min"), False, marks=xfail(reason="Coercion to dense via bottleneck")),
        param(
            do("notnull"),
            False,
            marks=xfail(reason="'COO' object has no attribute 'notnull'"),
        ),
        param(
            do("pipe", np.sum, axis=1),
            True,
            marks=xfail(reason="Missing implementation for np.result_type"),
        ),
        param(
            do("prod"),
            False,
            marks=xfail(reason="Missing implementation for np.result_type"),
        ),
        param(
            do("quantile", q=0.5),
            False,
            marks=xfail(reason="Missing implementation for np.nanpercentile"),
        ),
        param(
            do("rank", "x"),
            False,
            marks=xfail(reason="Coercion to dense via bottleneck"),
        ),
        param(
            do("reduce", np.sum, dim="x"),
            False,
            marks=xfail(reason="Coercion to dense"),
        ),
        param(
            do(
                "reindex_like",
                make_xrarray(
                    {"x": 10, "y": 5},
                    coords={"x": np.arange(10) + 0.5, "y": np.arange(5) + 0.5},
                ),
            ),
            True,
            marks=xfail(reason="Indexing COO with more than one iterable index"),
        ),  # noqa
        param(
            do("roll", x=2),
            True,
            marks=xfail(reason="Missing implementation for np.result_type"),
        ),
        param(
            do("sel", x=[0, 1, 2], y=[2, 3]),
            True,
            marks=xfail(reason="Indexing COO with more than one iterable index"),
        ),  # noqa
        param(do("std"), False, marks=xfail(reason="Coercion to dense via bottleneck")),
        param(
            do("sum"),
            False,
            marks=xfail(reason="Missing implementation for np.result_type"),
        ),
        param(do("var"), False, marks=xfail(reason="Coercion to dense via bottleneck")),
        param(
            do("where", make_xrarray({"x": 10, "y": 5}) > 0.5),
            False,
            marks=xfail(reason="Conversion of dense to sparse when using sparse mask"),
        ),  # noqa
    ],
    ids=repr,
)
def test_dataarray_method(func, sparse_output):
    arr_s = make_xrarray(
        {"x": 10, "y": 5}, coords={"x": np.arange(10), "y": np.arange(5)}
    )
    arr_d = xr.DataArray(arr_s.data.todense(), coords=arr_s.coords, dims=arr_s.dims)
    ret_s = func(arr_s)
    ret_d = func(arr_d)

    if sparse_output:
        assert isinstance(ret_s.data, SparseArray)
        assert np.allclose(ret_s.data.todense(), ret_d.data, equal_nan=True)
    else:
        assert np.allclose(ret_s, ret_d, equal_nan=True)


@pytest.mark.parametrize(
    "func,sparse_output",
    [
        (do("squeeze"), True),
        param(
            do("searchsorted", [1, 2, 3]),
            False,
            marks=xfail(reason="'COO' object has no attribute 'searchsorted'"),
        ),
    ],
)
def test_datarray_1d_method(func, sparse_output):
    arr_s = make_xrarray({"x": 10}, coords={"x": np.arange(10)})
    arr_d = xr.DataArray(arr_s.data.todense(), coords=arr_s.coords, dims=arr_s.dims)
    ret_s = func(arr_s)
    ret_d = func(arr_d)

    if sparse_output:
        assert isinstance(ret_s.data, SparseArray)
        assert np.allclose(ret_s.data.todense(), ret_d.data, equal_nan=True)
    else:
        assert np.allclose(ret_s, ret_d, equal_nan=True)


class TestSparseDataArrayAndDataset:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.sp_ar = sparse.random((4, 6), random_state=0, density=0.5)
        self.sp_xr = xr.DataArray(
            self.sp_ar, coords={"x": range(4)}, dims=("x", "y"), name="foo"
        )
        self.ds_ar = self.sp_ar.todense()
        self.ds_xr = xr.DataArray(
            self.ds_ar, coords={"x": range(4)}, dims=("x", "y"), name="foo"
        )

    @pytest.mark.xfail(reason="Missing implementation for np.result_type")
    def test_to_dataset_roundtrip(self):
        x = self.sp_xr
        assert_equal(x, x.to_dataset("x").to_array("x"))

    def test_align(self):
        a1 = xr.DataArray(
            COO.from_numpy(np.arange(4)), dims=["x"], coords={"x": ["a", "b", "c", "d"]}
        )
        b1 = xr.DataArray(
            COO.from_numpy(np.arange(4)), dims=["x"], coords={"x": ["a", "b", "d", "e"]}
        )
        a2, b2 = xr.align(a1, b1, join="inner")
        assert isinstance(a2.data, sparse.SparseArray)
        assert isinstance(b2.data, sparse.SparseArray)
        assert np.all(a2.coords["x"].data == ["a", "b", "d"])
        assert np.all(b2.coords["x"].data == ["a", "b", "d"])

    @pytest.mark.xfail(
        reason="COO objects currently do not accept more than one "
        "iterable index at a time"
    )
    def test_align_2d(self):
        A1 = xr.DataArray(
            self.sp_ar,
            dims=["x", "y"],
            coords={
                "x": np.arange(self.sp_ar.shape[0]),
                "y": np.arange(self.sp_ar.shape[1]),
            },
        )

        A2 = xr.DataArray(
            self.sp_ar,
            dims=["x", "y"],
            coords={
                "x": np.arange(1, self.sp_ar.shape[0] + 1),
                "y": np.arange(1, self.sp_ar.shape[1] + 1),
            },
        )

        B1, B2 = xr.align(A1, A2, join="inner")
        assert np.all(B1.coords["x"] == np.arange(1, self.sp_ar.shape[0]))
        assert np.all(B1.coords["y"] == np.arange(1, self.sp_ar.shape[0]))
        assert np.all(B1.coords["x"] == B2.coords["x"])
        assert np.all(B1.coords["y"] == B2.coords["y"])

    @pytest.mark.xfail(reason="fill value leads to sparse-dense operation")
    def test_align_outer(self):
        a1 = xr.DataArray(
            COO.from_numpy(np.arange(4)), dims=["x"], coords={"x": ["a", "b", "c", "d"]}
        )
        b1 = xr.DataArray(
            COO.from_numpy(np.arange(4)), dims=["x"], coords={"x": ["a", "b", "d", "e"]}
        )
        a2, b2 = xr.align(a1, b1, join="outer")
        assert isinstance(a2.data, sparse.SparseArray)
        assert isinstance(b2.data, sparse.SparseArray)
        assert np.all(a2.coords["x"].data == ["a", "b", "c", "d"])
        assert np.all(b2.coords["x"].data == ["a", "b", "c", "d"])

    @pytest.mark.xfail(reason="Missing implementation for np.result_type")
    def test_concat(self):
        ds1 = xr.Dataset(data_vars={"d": self.sp_xr})
        ds2 = xr.Dataset(data_vars={"d": self.sp_xr})
        ds3 = xr.Dataset(data_vars={"d": self.sp_xr})
        out = xr.concat([ds1, ds2, ds3], dim="x")
        assert_sparse_eq(
            out["d"].data,
            sparse.concatenate([self.sp_ar, self.sp_ar, self.sp_ar], axis=0),
        )

        out = xr.concat([self.sp_xr, self.sp_xr, self.sp_xr], dim="y")
        assert_sparse_eq(
            out.data, sparse.concatenate([self.sp_ar, self.sp_ar, self.sp_ar], axis=1)
        )

    def test_stack(self):
        arr = make_xrarray({"w": 2, "x": 3, "y": 4})
        stacked = arr.stack(z=("x", "y"))

        z = pd.MultiIndex.from_product([np.arange(3), np.arange(4)], names=["x", "y"])

        expected = xr.DataArray(
            arr.data.reshape((2, -1)), {"w": [0, 1], "z": z}, dims=["w", "z"]
        )

        assert_equal(expected, stacked)

        roundtripped = stacked.unstack()
        assert arr.identical(roundtripped)

    def test_ufuncs(self):
        x = self.sp_xr
        assert_equal(np.sin(x), xu.sin(x))

    def test_dataarray_repr(self):
        a = xr.DataArray(
            COO.from_numpy(np.ones(4)),
            dims=["x"],
            coords={"y": ("x", COO.from_numpy(np.arange(4)))},
        )
        expected = dedent(
            """\
        <xarray.DataArray (x: 4)>
        <COO: shape=(4,), dtype=float64, nnz=4, fill_value=0.0>
        Coordinates:
            y        (x) int64 ...
        Dimensions without coordinates: x"""
        )
        assert expected == repr(a)

    def test_dataset_repr(self):
        ds = xr.Dataset(
            data_vars={"a": ("x", COO.from_numpy(np.ones(4)))},
            coords={"y": ("x", COO.from_numpy(np.arange(4)))},
        )
        expected = dedent(
            """\
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
            y        (x) int64 ...
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 ..."""
        )
        assert expected == repr(ds)

    def test_dataarray_pickle(self):
        a1 = xr.DataArray(
            COO.from_numpy(np.ones(4)),
            dims=["x"],
            coords={"y": ("x", COO.from_numpy(np.arange(4)))},
        )
        a2 = pickle.loads(pickle.dumps(a1))
        assert_identical(a1, a2)

    def test_dataset_pickle(self):
        ds1 = xr.Dataset(
            data_vars={"a": ("x", COO.from_numpy(np.ones(4)))},
            coords={"y": ("x", COO.from_numpy(np.arange(4)))},
        )
        ds2 = pickle.loads(pickle.dumps(ds1))
        assert_identical(ds1, ds2)

    def test_coarsen(self):
        a1 = self.ds_xr
        a2 = self.sp_xr
        m1 = a1.coarsen(x=2, boundary="trim").mean()
        m2 = a2.coarsen(x=2, boundary="trim").mean()

        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail(reason="No implementation of np.pad")
    def test_rolling(self):
        a1 = self.ds_xr
        a2 = self.sp_xr
        m1 = a1.rolling(x=2, center=True).mean()
        m2 = a2.rolling(x=2, center=True).mean()

        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail(reason="Coercion to dense")
    def test_rolling_exp(self):
        a1 = self.ds_xr
        a2 = self.sp_xr
        m1 = a1.rolling_exp(x=2, center=True).mean()
        m2 = a2.rolling_exp(x=2, center=True).mean()

        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail(reason="No implementation of np.einsum")
    def test_dot(self):
        a1 = self.xp_xr.dot(self.xp_xr[0])
        a2 = self.sp_ar.dot(self.sp_ar[0])
        assert_equal(a1, a2)

    @pytest.mark.xfail(reason="Groupby reductions produce dense output")
    def test_groupby(self):
        x1 = self.ds_xr
        x2 = self.sp_xr
        m1 = x1.groupby("x").mean(xr.ALL_DIMS)
        m2 = x2.groupby("x").mean(xr.ALL_DIMS)
        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail(reason="Groupby reductions produce dense output")
    def test_groupby_first(self):
        x = self.sp_xr.copy()
        x.coords["ab"] = ("x", ["a", "a", "b", "b"])
        x.groupby("ab").first()
        x.groupby("ab").first(skipna=False)

    @pytest.mark.xfail(reason="Groupby reductions produce dense output")
    def test_groupby_bins(self):
        x1 = self.ds_xr
        x2 = self.sp_xr
        m1 = x1.groupby_bins("x", bins=[0, 3, 7, 10]).sum()
        m2 = x2.groupby_bins("x", bins=[0, 3, 7, 10]).sum()
        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail(reason="Resample produces dense output")
    def test_resample(self):
        t1 = xr.DataArray(
            np.linspace(0, 11, num=12),
            coords=[
                pd.date_range("15/12/1999", periods=12, freq=pd.DateOffset(months=1))
            ],
            dims="time",
        )
        t2 = t1.copy()
        t2.data = COO(t2.data)
        m1 = t1.resample(time="QS-DEC").mean()
        m2 = t2.resample(time="QS-DEC").mean()
        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail
    def test_reindex(self):
        x1 = self.ds_xr
        x2 = self.sp_xr
        for kwargs in [
            {"x": [2, 3, 4]},
            {"x": [1, 100, 2, 101, 3]},
            {"x": [2.5, 3, 3.5], "y": [2, 2.5, 3]},
        ]:
            m1 = x1.reindex(**kwargs)
            m2 = x2.reindex(**kwargs)
            assert np.allclose(m1, m2, equal_nan=True)

    @pytest.mark.xfail
    def test_merge(self):
        x = self.sp_xr
        y = xr.merge([x, x.rename("bar")]).to_array()
        assert isinstance(y, sparse.SparseArray)

    @pytest.mark.xfail
    def test_where(self):
        a = np.arange(10)
        cond = a > 3
        xr.DataArray(a).where(cond)

        s = COO.from_numpy(a)
        cond = s > 3
        xr.DataArray(s).where(cond)

        x = xr.DataArray(s)
        cond = x > 3
        x.where(cond)


class TestSparseCoords:
    @pytest.mark.xfail(reason="Coercion of coords to dense")
    def test_sparse_coords(self):
        xr.DataArray(
            COO.from_numpy(np.arange(4)),
            dims=["x"],
            coords={"x": COO.from_numpy([1, 2, 3, 4])},
        )
