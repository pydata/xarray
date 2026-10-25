"""Opening the groups of a zarr hierarchy concurrently in open_datatree."""

from __future__ import annotations

import asyncio
import threading
import time
import warnings
from unittest.mock import patch

import numpy as np
import pytest

import xarray as xr
from xarray.backends.api import _load_default_index_variables_async
from xarray.backends.store import StoreBackendEntrypoint
from xarray.backends.zarr import ZarrBackendEntrypoint
from xarray.testing import assert_equal
from xarray.tests import (
    has_zarr,
    parametrize_zarr_format,
    requires_netCDF4,
    requires_zarr,
)

if has_zarr:
    from zarr.storage import MemoryStore


def create_tree() -> xr.DataTree:
    """Build a nested tree whose groups each have their own dimension coordinate."""

    def node(n: int) -> xr.Dataset:
        return xr.Dataset(
            {"var": (("x", f"y{n}"), np.arange(3.0 * n).reshape(3, n))},
            coords={"x": [1, 2, 3], f"y{n}": np.arange(n)},
        )

    return xr.DataTree.from_dict(
        {
            "/": node(2),
            "/leaf_0": node(3),
            "/leaf_0/child": node(4),
            "/leaf-1": node(5),
            "/leaf_1": xr.Dataset(),
        }
    )


@pytest.fixture
def store(zarr_format):
    with MemoryStore() as store:
        create_tree().to_zarr(store, mode="w", zarr_format=zarr_format)
        yield store


def open_tree(store, **kwargs) -> xr.DataTree:
    return xr.open_datatree(store, engine="zarr", **kwargs)  # type: ignore[arg-type, unused-ignore]


def fail_in(groups: dict[str, float]):
    """Patch open_dataset to raise for some groups, after the given delays."""
    original = StoreBackendEntrypoint.open_dataset

    def open_dataset(self, store, **kwargs):
        path = store.zarr_group.path.rpartition("/")[2]
        if path in groups:
            time.sleep(groups[path])
            raise ValueError(f"cannot decode {path}")
        return original(self, store, **kwargs)

    return patch.object(StoreBackendEntrypoint, "open_dataset", open_dataset)


@requires_zarr
@parametrize_zarr_format
class TestConcurrentOpen:
    @pytest.mark.parametrize("max_concurrency", [None, 1, 3])
    def test_same_tree(self, store, max_concurrency) -> None:
        expected = create_tree()
        assert_equal(open_tree(store, max_concurrency=max_concurrency), expected)

    def test_groups_in_depth_first_order(self, store) -> None:
        groups = xr.open_groups(store, engine="zarr")  # type: ignore[arg-type, unused-ignore]
        assert list(groups) == ["/", "/leaf-1", "/leaf_0", "/leaf_0/child", "/leaf_1"]

    def test_runs_groups_concurrently(self, store) -> None:
        original = StoreBackendEntrypoint.open_dataset
        running: set[int] = set()
        seen_together = 0
        lock = threading.Lock()

        def open_dataset(self, store, **kwargs):
            nonlocal seen_together
            with lock:
                running.add(threading.get_ident())
                seen_together = max(seen_together, len(running))
            time.sleep(0.05)
            with lock:
                running.discard(threading.get_ident())
            return original(self, store, **kwargs)

        with patch.object(StoreBackendEntrypoint, "open_dataset", open_dataset):
            open_tree(store, max_concurrency=2)
        assert seen_together == 2

    def test_first_error_in_group_order(self, store) -> None:
        # "child" fails first in time, "leaf-1" comes first in depth-first order
        with (
            fail_in({"leaf-1": 0.2, "child": 0.0}),
            pytest.raises(ValueError, match="cannot decode leaf-1"),
        ):
            open_tree(store)

    def test_opened_groups_closed_after_error(self, store) -> None:
        original = StoreBackendEntrypoint.open_dataset
        opened: list[str] = []
        closed: list[str] = []

        def open_dataset(self, store, **kwargs):
            path = store.zarr_group.path or "/"
            if path == "leaf_1":
                raise ValueError("cannot decode leaf_1")
            ds = original(self, store, **kwargs)
            opened.append(path)
            ds.set_close(lambda: closed.append(path))
            return ds

        with (
            patch.object(StoreBackendEntrypoint, "open_dataset", open_dataset),
            pytest.raises(ValueError, match="cannot decode leaf_1"),
        ):
            open_tree(store)
        assert opened
        assert sorted(closed) == sorted(opened)

    def test_warnings_point_at_caller(self, store) -> None:
        original = StoreBackendEntrypoint.open_dataset

        def open_dataset(self, store, **kwargs):
            warnings.warn("decoding warning", UserWarning, stacklevel=1)
            return original(self, store, **kwargs)

        with (
            patch.object(StoreBackendEntrypoint, "open_dataset", open_dataset),
            pytest.warns(UserWarning, match="decoding warning") as record,
        ):
            open_tree(store)
        assert {w.filename for w in record} == {__file__}

    @pytest.mark.parametrize("max_concurrency", [0, -1, 1.5])
    def test_invalid_max_concurrency(self, store, max_concurrency) -> None:
        with pytest.raises(ValueError, match="max_concurrency must be a positive"):
            open_tree(store, max_concurrency=max_concurrency)

    @pytest.mark.parametrize("create_default_indexes", [True, False])
    def test_index_prefetch_hook(self, store, create_default_indexes) -> None:
        # also when the engine is given as the backend class
        with patch.object(
            ZarrBackendEntrypoint, "_run_async", autospec=True, return_value=True
        ) as run_async:
            xr.open_datatree(
                store,  # type: ignore[arg-type, unused-ignore]
                engine=ZarrBackendEntrypoint,
                create_default_indexes=create_default_indexes,
            )
        assert run_async.called == create_default_indexes
        if run_async.called:
            run_async.call_args.args[1].close()

    def test_load_default_index_variables(self, store) -> None:
        lazy = open_tree(store, create_default_indexes=False)
        asyncio.run(_load_default_index_variables_async(lazy, max_concurrency=2))
        for node in lazy.subtree:
            ds = node.to_dataset(inherit=False)
            for name in ds.coords:
                assert isinstance(ds.variables[name]._data, np.ndarray)
            if "var" in ds:
                assert not isinstance(ds.variables["var"]._data, np.ndarray)

    @pytest.mark.asyncio
    async def test_called_from_running_event_loop(self, store) -> None:
        # e.g. inside Jupyter
        assert_equal(open_tree(store), create_tree())


@requires_netCDF4
def test_max_concurrency_other_engine_raises(tmp_path) -> None:
    path = tmp_path / "tree.nc"
    xr.DataTree(xr.Dataset({"a": ("x", [1, 2])})).to_netcdf(path, engine="netcdf4")
    with pytest.raises(ValueError, match="not supported by the 'netcdf4' engine"):
        xr.open_datatree(path, engine="netcdf4", max_concurrency=2)
