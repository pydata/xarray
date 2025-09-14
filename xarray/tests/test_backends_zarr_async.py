"""Tests for asynchronous zarr group loading functionality."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import patch

import numpy as np
import pytest

import xarray as xr
from xarray.backends.api import _maybe_create_default_indexes_async, open_datatree_async
from xarray.backends.zarr import ZarrBackendEntrypoint
from xarray.testing import assert_equal
from xarray.tests import (
    has_zarr_v3,
    parametrize_zarr_format,
    requires_zarr,
    requires_zarr_v3,
)

if has_zarr_v3:
    from zarr.storage import MemoryStore


def create_dataset_with_coordinates(n_coords=5):
    """Create a dataset with coordinate variables to trigger indexing."""
    coords = {}
    for i in range(n_coords):
        coords[f"coord_{i}"] = (f"coord_{i}", np.arange(3))

    coord_names = list(coords.keys())
    data_vars = {}

    if len(coord_names) >= 2:
        data_vars["temperature"] = (coord_names[:2], np.random.random((3, 3)))
    if len(coord_names) >= 1:
        data_vars["pressure"] = (coord_names[:1], np.random.random(3))

    data_vars["simple"] = ([], np.array(42.0))

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return ds


def create_test_datatree(n_groups=3, coords_per_group=5):
    """Create a DataTree for testing with multiple groups."""
    root_ds = create_dataset_with_coordinates(coords_per_group)
    tree_dict = {"/": root_ds}

    for i in range(n_groups):
        group_name = f"/group_{i:03d}"
        group_ds = create_dataset_with_coordinates(n_coords=coords_per_group)
        tree_dict[group_name] = group_ds

    tree = xr.DataTree.from_dict(tree_dict)
    return tree


@requires_zarr
class TestAsyncZarrGroupLoading:
    """Tests for asynchronous zarr group loading functionality."""

    @contextlib.contextmanager
    def create_zarr_store(self):
        """Create a zarr target for testing."""
        if has_zarr_v3:
            with MemoryStore() as store:
                yield store
        else:
            from zarr.storage import MemoryStore as V2MemoryStore

            store = V2MemoryStore()
            yield store

    @parametrize_zarr_format
    def test_async_datatree_roundtrip(self, zarr_format):
        """Test that async datatree loading preserves data integrity."""

        dtree = create_test_datatree(n_groups=3, coords_per_group=4)

        with self.create_zarr_store() as store:
            dtree.to_zarr(store, mode="w", consolidated=False, zarr_format=zarr_format)

            async def load_async():
                return await open_datatree_async(
                    store,
                    consolidated=False,
                    zarr_format=zarr_format,
                    create_default_indexes=True,
                    engine="zarr",
                )

            dtree_async = asyncio.run(load_async())
            assert_equal(dtree, dtree_async)

    def test_async_error_handling_unsupported_engine(self):
        """Test that async functions properly handle unsupported engines."""

        async def test_unsupported_engine():
            with pytest.raises(
                ValueError, match="does not support asynchronous operations"
            ):
                await open_datatree_async("/fake/path", engine="netcdf4")

        asyncio.run(test_unsupported_engine())

    @pytest.mark.asyncio
    @requires_zarr_v3
    async def test_async_concurrent_loading(self):
        """Test that async loading uses concurrent calls for multiple groups."""
        import zarr

        dtree = create_test_datatree(n_groups=3, coords_per_group=4)

        with self.create_zarr_store() as store:
            dtree.to_zarr(store, mode="w", consolidated=False, zarr_format=3)

            target_class = zarr.AsyncGroup
            original_method = target_class.getitem

            with patch.object(
                target_class, "getitem", wraps=original_method, autospec=True
            ) as mocked_method:
                dtree_async = await open_datatree_async(
                    store,
                    consolidated=False,
                    zarr_format=3,
                    create_default_indexes=True,
                    engine="zarr",
                )

                assert_equal(dtree, dtree_async)

                assert mocked_method.call_count > 0
                mocked_method.assert_awaited()

    @pytest.mark.asyncio
    @parametrize_zarr_format
    async def test_async_root_only_datatree(self, zarr_format):
        """Test async loading of datatree with only root node (no child groups)."""

        root_ds = create_dataset_with_coordinates(3)
        dtree = xr.DataTree(root_ds)

        with self.create_zarr_store() as store:
            dtree.to_zarr(store, mode="w", consolidated=False, zarr_format=zarr_format)

            dtree_async = await open_datatree_async(
                store,
                consolidated=False,
                zarr_format=zarr_format,
                create_default_indexes=True,
                engine="zarr",
            )

            assert len(list(dtree_async.subtree)) == 1
            assert dtree_async.path == "/"
            assert dtree_async.ds is not None

    @pytest.mark.asyncio
    @parametrize_zarr_format
    @pytest.mark.parametrize("n_groups", [1, 3, 10])
    async def test_async_multiple_groups(self, zarr_format, n_groups):
        """Test async loading of datatree with varying numbers of groups."""
        dtree = create_test_datatree(n_groups=n_groups, coords_per_group=3)

        with self.create_zarr_store() as store:
            dtree.to_zarr(store, mode="w", consolidated=False, zarr_format=zarr_format)

            # Load asynchronously
            dtree_async = await open_datatree_async(
                store,
                consolidated=False,
                zarr_format=zarr_format,
                create_default_indexes=True,
                engine="zarr",
            )

            expected_groups = ["/"] + [f"/group_{i:03d}" for i in range(n_groups)]
            group_paths = [node.path for node in dtree_async.subtree]

            assert len(group_paths) == len(expected_groups)
            for expected_path in expected_groups:
                assert expected_path in group_paths

    @pytest.mark.asyncio
    @parametrize_zarr_format
    async def test_async_create_default_indexes_false(self, zarr_format):
        """Test that create_default_indexes=False prevents index creation."""
        dtree = create_test_datatree(n_groups=2, coords_per_group=3)

        with self.create_zarr_store() as store:
            dtree.to_zarr(store, mode="w", consolidated=False, zarr_format=zarr_format)

            dtree_async = await open_datatree_async(
                store,
                consolidated=False,
                zarr_format=zarr_format,
                create_default_indexes=False,
                engine="zarr",
            )

            assert len(list(dtree_async.subtree)) == 3

            for node in dtree_async.subtree:
                dataset = node.dataset
                if dataset is not None:
                    coord_names = [
                        name
                        for name, coord in dataset.coords.items()
                        if coord.dims == (name,)
                    ]
                    for coord_name in coord_names:
                        assert coord_name not in dataset.xindexes, (
                            f"Index should not exist for coordinate '{coord_name}' when create_default_indexes=False"
                        )

    def test_sync_vs_async_api_compatibility(self):
        """Test that sync and async APIs have compatible signatures."""
        import inspect

        from xarray.backends.api import open_datatree

        sync_sig = inspect.signature(open_datatree)
        async_sig = inspect.signature(open_datatree_async)

        sync_params = list(sync_sig.parameters.keys())
        async_params = list(async_sig.parameters.keys())

        for param in sync_params:
            assert param in async_params, (
                f"Parameter '{param}' missing from async version"
            )

    @pytest.mark.asyncio
    @requires_zarr
    @parametrize_zarr_format
    async def test_backend_open_groups_async_equivalence(self, zarr_format):
        """Backend async group opening returns the same groups and datasets as sync."""
        dtree = create_test_datatree(n_groups=3, coords_per_group=4)
        backend = ZarrBackendEntrypoint()

        with self.create_zarr_store() as store:
            dtree.to_zarr(store, mode="w", consolidated=False, zarr_format=zarr_format)

            groups_sync = backend.open_groups_as_dict(
                store,
                consolidated=False,
                zarr_format=zarr_format,
            )

            groups_async = await backend.open_groups_as_dict_async(
                store,
                consolidated=False,
                zarr_format=zarr_format,
            )

            assert set(groups_sync.keys()) == set(groups_async.keys())
            for k in list(groups_sync.keys())[:2]:
                assert_equal(groups_sync[k], groups_async[k])

    @pytest.mark.asyncio
    async def test_maybe_create_default_indexes_async_no_coords_needing_indexes(self):
        """Test _maybe_create_default_indexes_async with no coordinates needing indexes."""
        ds = xr.Dataset(
            {
                "temperature": (("x", "y"), np.random.random((3, 4))),
            }
        )

        result = await _maybe_create_default_indexes_async(ds)
        assert_equal(ds, result)
        assert len(result.xindexes) == 0

    @pytest.mark.asyncio
    async def test_maybe_create_default_indexes_async_creates_indexes(self):
        """Test _maybe_create_default_indexes_async creates indexes for coordinate variables."""
        coords = {"time": ("time", np.arange(5)), "x": ("x", np.arange(3))}
        data_vars = {
            "temperature": (("time", "x"), np.random.random((5, 3))),
        }
        ds = xr.Dataset(data_vars, coords)
        ds_no_indexes = ds.drop_indexes(["time", "x"])

        assert len(ds_no_indexes.xindexes) == 0

        result = await _maybe_create_default_indexes_async(ds_no_indexes)

        assert "time" in result.xindexes
        assert "x" in result.xindexes
        assert len(result.xindexes) == 2

    @pytest.mark.asyncio
    async def test_maybe_create_default_indexes_async_partial_indexes(self):
        """Test with mix of coords that need indexes and those that don't."""
        coords = {
            "time": ("time", np.arange(5)),
            "x": ("x", np.arange(3)),
        }
        data_vars = {
            "temperature": (("time", "x"), np.random.random((5, 3))),
        }
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds_partial = ds.drop_indexes(["x"])

        assert "time" in ds_partial.xindexes
        assert "x" not in ds_partial.xindexes

        result = await _maybe_create_default_indexes_async(ds_partial)

        assert "time" in result.xindexes
        assert "x" in result.xindexes

    @pytest.mark.asyncio
    async def test_maybe_create_default_indexes_async_all_indexes_exist(self):
        """Test that function returns original dataset when all coords already have indexes."""
        ds = create_dataset_with_coordinates(n_coords=2)

        assert len(ds.xindexes) > 0

        result = await _maybe_create_default_indexes_async(ds)
        assert result is ds  # Same object returned
