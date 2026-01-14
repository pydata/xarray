"""Tests for internal asynchronous zarr group loading functionality."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

import xarray as xr
from xarray.backends.api import _maybe_create_default_indexes_async
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
    """Tests for internal asynchronous zarr group loading functionality."""

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
    def test_sync_datatree_roundtrip_with_async_optimization(self, zarr_format):
        """Test that sync open_datatree with internal async optimization preserves data integrity."""
        dtree = create_test_datatree(n_groups=3, coords_per_group=4)

        with self.create_zarr_store() as store:
            dtree.to_zarr(store, mode="w", consolidated=False, zarr_format=zarr_format)

            # Use sync open_datatree which internally uses async for zarr v3
            dtree_loaded = xr.open_datatree(
                store,
                consolidated=False,
                zarr_format=zarr_format,
                create_default_indexes=True,
                engine="zarr",
            )
            assert_equal(dtree, dtree_loaded)

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

    @parametrize_zarr_format
    def test_sync_open_datatree_source_encoding(self, zarr_format):
        """Test that open_datatree sets source encoding correctly."""
        import os
        import tempfile

        dtree = create_test_datatree(n_groups=2, coords_per_group=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.zarr")
            dtree.to_zarr(
                store_path, mode="w", consolidated=False, zarr_format=zarr_format
            )

            dtree_loaded = xr.open_datatree(
                store_path,
                consolidated=False,
                zarr_format=zarr_format,
                engine="zarr",
            )
            assert "source" in dtree_loaded.encoding
            # Normalize paths for cross-platform comparison
            source = os.path.normpath(dtree_loaded.encoding["source"])
            expected = os.path.normpath(store_path)
            assert expected in source

    @requires_zarr_v3
    @parametrize_zarr_format
    def test_sync_open_datatree_uses_async_internally(self, zarr_format):
        """Test that sync open_datatree uses async index creation for zarr v3."""
        from unittest.mock import patch

        dtree = create_test_datatree(n_groups=2, coords_per_group=3)

        with self.create_zarr_store() as store:
            dtree.to_zarr(store, mode="w", consolidated=False, zarr_format=zarr_format)

            # Patch the async index creation function to verify it's called
            with patch(
                "xarray.backends.api._maybe_create_default_indexes_async",
                wraps=_maybe_create_default_indexes_async,
            ) as mock_async:
                dtree_loaded = xr.open_datatree(
                    store,
                    consolidated=False,
                    zarr_format=zarr_format,
                    create_default_indexes=True,
                    engine="zarr",
                )

                # For zarr v3, the async function should be called
                assert mock_async.call_count > 0
                assert_equal(dtree, dtree_loaded)
