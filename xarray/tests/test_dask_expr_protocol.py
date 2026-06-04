from __future__ import annotations

import numpy as np
import pytest

import xarray as xr
from xarray import DataArray, Dataset
from xarray.testing import assert_equal, assert_identical

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")


def test_standalone_dask_array_dataset_composite_expr_protocol():
    dask_array = pytest.importorskip("dask_array")
    from dask._expr import CompositeExpr

    x = dask_array.arange(6, chunks=(3,))
    ds = Dataset(
        {"foo": ("i", x + 1)},
        coords={"coord": ("i", x + 10)},
        attrs={"source": "test"},
    )
    ds["foo"].encoding["example"] = "kept"

    expr = dask.base.collections_to_expr(ds)

    assert isinstance(expr, CompositeExpr)
    assert len(expr.exprs) == 2

    expected = Dataset(
        {"foo": ("i", np.arange(6) + 1)},
        coords={"coord": ("i", np.arange(6) + 10)},
        attrs={"source": "test"},
    )
    computed = dask.compute(ds, scheduler="single-threaded")[0]
    assert_identical(computed, expected)
    assert computed["foo"].encoding["example"] == "kept"

    persisted = dask.persist(ds, scheduler="single-threaded")[0]
    assert hasattr(persisted["foo"].data, "expr")
    assert_identical(persisted.compute(), expected)
    assert persisted["foo"].encoding["example"] == "kept"

    optimized = dask.optimize(ds)[0]
    assert hasattr(optimized["foo"].data, "expr")
    assert_identical(optimized.compute(), expected)


def test_standalone_dask_array_dataset_variable_named_expr():
    dask_array = pytest.importorskip("dask_array")
    from dask._expr import CompositeExpr

    x = dask_array.arange(6, chunks=(3,))
    ds = Dataset({"expr": ("i", x + 1)})

    expr = dask.base.collections_to_expr(ds)

    assert isinstance(expr, CompositeExpr)
    assert_identical(
        dask.compute(ds, scheduler="single-threaded")[0],
        Dataset({"expr": ("i", np.arange(6) + 1)}),
    )


def test_standalone_dask_array_dataarray_composite_expr_protocol():
    dask_array = pytest.importorskip("dask_array")
    from dask._expr import CompositeExpr

    x = dask_array.arange(6, chunks=(3,))
    arr = DataArray(x + 1, dims=("i",), coords={"coord": ("i", x)}, name="z")

    expr = dask.base.collections_to_expr(arr)

    assert isinstance(expr, CompositeExpr)
    assert len(expr.exprs) == 2

    expected = DataArray(
        np.arange(6) + 1,
        dims=("i",),
        coords={"coord": ("i", np.arange(6))},
        name="z",
    )
    computed = dask.compute(arr, scheduler="single-threaded")[0]
    assert_identical(computed, expected)

    persisted = dask.persist(arr, scheduler="single-threaded")[0]
    assert hasattr(persisted.data, "expr")
    assert_identical(persisted.compute(), expected)


def test_standalone_dask_array_dataset_computes_with_raw_array():
    dask_array = pytest.importorskip("dask_array")

    x = dask_array.arange(6, chunks=(3,))
    ds = Dataset({"foo": ("i", x + 1)})

    computed_ds, computed_x = dask.compute(ds, x, scheduler="single-threaded")

    assert_identical(computed_ds, Dataset({"foo": ("i", np.arange(6) + 1)}))
    np.testing.assert_array_equal(computed_x, np.arange(6))


def test_standalone_dask_array_optimize_culls_child_graphs():
    dask_array = pytest.importorskip("dask_array")
    from dask.core import flatten

    x = dask_array.arange(6, chunks=(3,))
    ds = Dataset({"foo": ("i", x + 1), "bar": ("i", x + 2)})

    optimized = dask.optimize(ds)[0]

    foo_graph_keys = set(optimized["foo"].data.__dask_graph__())
    bar_output_keys = set(flatten(optimized["bar"].data.__dask_keys__()))
    assert not foo_graph_keys & bar_output_keys
    assert_identical(
        optimized.compute(),
        Dataset({"foo": ("i", np.arange(6) + 1), "bar": ("i", np.arange(6) + 2)}),
    )


def test_standalone_dask_array_mixed_legacy_falls_back_from_composite_expr():
    dask_array = pytest.importorskip("dask_array")

    ds = Dataset(
        {
            "expr": ("i", dask_array.arange(3, chunks=(1,))),
            "legacy": ("i", da.arange(3, chunks=(1,))),
        }
    )

    assert ds.__dask_exprs__() is None


def test_standalone_dask_array_mixed_legacy_map_blocks_fallback_computes():
    dask_array = pytest.importorskip("dask_array")

    ds = Dataset(
        {
            "expr": ("i", dask_array.arange(6, chunks=(3,))),
            "legacy": ("i", da.arange(6, chunks=(3,))),
        }
    )

    out = xr.map_blocks(lambda block: block + 1, ds)

    expected = Dataset(
        {"expr": ("i", np.arange(6) + 1), "legacy": ("i", np.arange(6) + 1)}
    )
    assert_identical(out.compute(scheduler="single-threaded"), expected)


def test_standalone_dask_array_open_mfdataset_uses_expressions(tmp_path):
    pytest.importorskip("dask_array")
    from dask._expr import CompositeExpr

    paths = []
    for i in range(2):
        path = tmp_path / f"part-{i}.nc"
        Dataset(
            {"x": ("t", np.arange(i * 3, i * 3 + 3))},
            coords={"t": np.arange(i * 3, i * 3 + 3)},
        ).to_netcdf(path)
        paths.append(path)

    ds = xr.open_mfdataset(paths, chunks={"t": 2}, combine="by_coords")
    try:
        expr = dask.base.collections_to_expr(ds)

        assert type(ds["x"].data).__module__.startswith("dask_array")
        assert isinstance(expr, CompositeExpr)

        expected = Dataset({"x": ("t", np.arange(6))}, coords={"t": np.arange(6)})
        assert_identical(ds.compute(scheduler="single-threaded"), expected)
        assert_identical(dask.compute(ds, scheduler="single-threaded")[0], expected)
        assert_identical(
            ds.persist(scheduler="single-threaded").compute(
                scheduler="single-threaded"
            ),
            expected,
        )
        assert_identical(
            dask.optimize(ds)[0].compute(scheduler="single-threaded"), expected
        )
    finally:
        ds.close()


def test_standalone_dask_array_shared_subexpressions_and_chunked_coords():
    dask_array = pytest.importorskip("dask_array")

    x = dask_array.arange(6, chunks=(3,))
    ds = Dataset(
        {"a": ("t", x + 1), "b": ("t", x * 2)},
        coords={"qc": ("t", x + 10)},
        attrs={"case": "shared"},
    )

    expected = Dataset(
        {"a": ("t", np.arange(6) + 1), "b": ("t", np.arange(6) * 2)},
        coords={"qc": ("t", np.arange(6) + 10)},
        attrs={"case": "shared"},
    )
    assert len(dask.base.collections_to_expr(ds).exprs) == 3
    assert_identical(ds.compute(scheduler="single-threaded"), expected)
    assert_identical(
        dask.persist(ds, scheduler="single-threaded")[0].compute(
            scheduler="single-threaded"
        ),
        expected,
    )
    assert_identical(
        dask.optimize(ds)[0].compute(scheduler="single-threaded"), expected
    )


def test_standalone_dask_array_rechunk_reduction_chain():
    dask_array = pytest.importorskip("dask_array")
    from dask._expr import CompositeExpr

    x = dask_array.arange(12, chunks=(4,)).reshape((3, 4))
    out = Dataset({"x": (("a", "b"), x)}).chunk({"a": 1, "b": 2}).sum("b") + 1

    assert isinstance(dask.base.collections_to_expr(out), CompositeExpr)
    expected = Dataset({"x": ("a", np.arange(12).reshape(3, 4).sum(axis=1) + 1)})
    assert_identical(out.compute(scheduler="single-threaded"), expected)
    assert_identical(
        dask.optimize(out)[0].compute(scheduler="single-threaded"), expected
    )


def test_standalone_dask_array_groupby_sum():
    dask_array = pytest.importorskip("dask_array")
    from dask._expr import CompositeExpr

    x = dask_array.arange(6, chunks=(3,))
    arr = DataArray(
        x,
        dims="t",
        coords={"label": ("t", np.array(["a", "a", "b", "b", "a", "b"]))},
        name="x",
    )
    out = arr.groupby("label").sum()

    assert isinstance(dask.base.collections_to_expr(out), CompositeExpr)
    expected = DataArray(
        [5, 10],
        dims="label",
        coords={"label": np.array(["a", "b"], dtype=object)},
        name="x",
    )
    assert_equal(out.compute(scheduler="single-threaded"), expected)
    assert_equal(dask.optimize(out)[0].compute(scheduler="single-threaded"), expected)


def test_standalone_dask_array_map_blocks_fallback_computes():
    dask_array = pytest.importorskip("dask_array")

    x = dask_array.arange(6, chunks=(3,))
    arr = DataArray(x, dims="t", name="x")
    out = xr.map_blocks(lambda block: block + 1, arr)

    expected = DataArray(np.arange(6) + 1, dims="t", name="x")
    assert_identical(out.compute(scheduler="single-threaded"), expected)
    assert_identical(
        dask.optimize(out)[0].compute(scheduler="single-threaded"), expected
    )


def test_standalone_dask_array_apply_ufunc_parallelized():
    dask_array = pytest.importorskip("dask_array")
    from dask._expr import CompositeExpr

    x = dask_array.arange(6, chunks=(3,))
    arr = DataArray(x, dims="t", name="x")
    out = xr.apply_ufunc(lambda z: z + 2, arr, dask="parallelized", output_dtypes=[int])

    assert isinstance(dask.base.collections_to_expr(out), CompositeExpr)
    assert_identical(
        out.compute(scheduler="single-threaded"),
        DataArray(np.arange(6) + 2, dims="t", name="x"),
    )
