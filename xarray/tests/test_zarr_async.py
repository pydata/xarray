from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

import xarray as xr
from xarray.backends.zarr import has_zarr_async_index
from xarray.testing import assert_identical

if TYPE_CHECKING:
    from zarr.core.common import JSON

zarr = pytest.importorskip("zarr", minversion="3.0")
pytestmark = pytest.mark.asyncio


@pytest.fixture
def forbid_sync(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("async opening must not start threads or call Zarr sync")

    import zarr.core.sync

    monkeypatch.setattr(zarr.core.sync, "sync", fail)
    monkeypatch.setattr(zarr.core.sync.SyncMixin, "_sync", fail)
    monkeypatch.setattr(zarr.core.sync.SyncMixin, "_sync_iter", fail)


@pytest.fixture
def forbid_threads(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("async opening must not use a thread executor")

    monkeypatch.setattr(threading.Thread, "start", fail)
    monkeypatch.setattr(asyncio, "to_thread", fail)
    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", fail)


async def make_store(zarr_format=3, consolidated=False):
    from zarr.api import asynchronous as za

    store = zarr.storage.MemoryStore()
    root = await za.open_group(store, mode="w", zarr_format=zarr_format)
    group = await root.create_group("nested", attributes={"title": "async"})
    variables: list[tuple[str, np.ndarray, tuple[str, ...], dict[str, JSON]]] = [
        ("x", np.arange(4), ("x",), {}),
        ("value", np.arange(4, dtype="int16"), ("x",), {"scale_factor": 2.0}),
        ("time", np.arange(4), ("x",), {"units": "days since 2000-01-01"}),
    ]
    for name, data, dims, attrs in variables:
        if zarr_format == 2:
            attrs = dict(attrs, _ARRAY_DIMENSIONS=list(dims))
        array = await group.create_array(
            name,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(2,),
            attributes=attrs,
            dimension_names=dims if zarr_format == 3 else None,
            compressors=None,
            fill_value=None if zarr_format == 2 else 0,
        )
        await array.setitem(slice(None), data)
    if consolidated:
        await za.consolidate_metadata(store)
    return store


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("consolidated", [False, True])
@pytest.mark.parametrize("indexes", [False, True])
async def test_open_and_load_without_sync(zarr_format, consolidated, indexes, request):
    store = await make_store(zarr_format, consolidated)
    expected = xr.open_zarr(
        store, group="nested", consolidated=consolidated, chunks=None
    ).load()
    request.getfixturevalue("forbid_sync")
    # Zarr's format-2 codec itself uses asyncio.to_thread. This is independent
    # of xarray's opening path; format-3 BytesCodec permits a thread-free test.
    if zarr_format == 3:
        request.getfixturevalue("forbid_threads")
    ds = await xr.open_zarr_async(
        store, group="nested", consolidated=consolidated, create_default_indexes=indexes
    )
    assert bool(ds.xindexes) is indexes
    assert not isinstance(ds["value"].variable._data, np.ndarray)
    subset = await ds.isel(x=slice(1, 3)).load_async()
    if not indexes:
        expected = expected.drop_indexes("x")
    assert_identical(subset, expected.isel(x=slice(1, 3)))
    ds.close()
    assert store._is_open  # A caller-owned store remains open.


async def test_raw_drop_and_missing(request):
    store = await make_store()
    request.getfixturevalue("forbid_sync")
    ds = await xr.open_zarr_async(
        store,
        group="nested",
        decode_cf=False,
        drop_variables="time",
        create_default_indexes=False,
    )
    assert "time" not in ds
    assert ds["value"].attrs["scale_factor"] == 2.0
    result = await ds.isel(x=slice(1, 3)).load_async()
    np.testing.assert_array_equal(result["value"], [1, 2])
    with pytest.raises((KeyError, FileNotFoundError)):
        await xr.open_zarr_async(store, group="missing")
    with pytest.raises(TypeError, match="Expected a Zarr group"):
        await xr.open_zarr_async(store, group="nested/value")


async def test_decode_times_false_keeps_time_lazy(request):
    store = await make_store()
    request.getfixturevalue("forbid_sync")
    ds = await xr.open_zarr_async(
        store, group="nested", decode_times=False, create_default_indexes=False
    )
    assert not isinstance(ds["time"].variable._data, np.ndarray)
    np.testing.assert_array_equal((await ds["time"].load_async()).values, np.arange(4))


async def test_store_reads_stay_on_callers_loop(request):
    from zarr.storage import WrapperStore

    store = await make_store()
    loop = asyncio.get_running_loop()
    reads = []

    class LoopBoundStore(WrapperStore):
        async def get(self, key, *args, **kwargs):
            assert asyncio.get_running_loop() is loop
            reads.append(key)
            return await self._store.get(key, *args, **kwargs)

    request.getfixturevalue("forbid_sync")
    request.getfixturevalue("forbid_threads")
    ds = await xr.open_zarr_async(
        LoopBoundStore(store.with_read_only(True)),
        group="nested",
        decode_times=False,
        create_default_indexes=False,
    )
    assert not any("/c/" in key for key in reads)
    await ds.isel(x=slice(0, 1)).load_async()
    assert any("/c/" in key for key in reads)


@pytest.mark.parametrize("cancel", [False, True])
async def test_owned_store_closed_on_failure(monkeypatch, cancel):
    from zarr.api import asynchronous as za

    store = await make_store()
    root = await za.open_group(store, mode="r")

    async def open_group(*args, **kwargs):
        return root

    async def getitem(*args, **kwargs):
        if cancel:
            raise asyncio.CancelledError
        raise KeyError("missing")

    monkeypatch.setattr(za, "open_group", open_group)
    monkeypatch.setattr(type(root), "getitem", getitem)
    error = asyncio.CancelledError if cancel else KeyError
    with pytest.raises(error):
        await xr.open_zarr_async("owned-store", group="missing")
    assert not root.store._is_open


@pytest.mark.parametrize("valid", [False, True])
async def test_nczarr_dimensions(valid, request):
    import json

    from zarr.api import asynchronous as za
    from zarr.core.buffer import default_buffer_prototype

    store = zarr.storage.MemoryStore()
    group = await za.open_group(store, mode="w", zarr_format=2)
    await group.create_array("value", shape=(2,), dtype="int32", compressors=None)
    prototype = default_buffer_prototype()
    raw = await store.get("value/.zarray", prototype=prototype)
    metadata = json.loads(raw.to_bytes())
    if valid:
        metadata["_NCZARR_ARRAY"] = {"dimrefs": ["/x"]}
        await store.set(
            "value/.zarray", prototype.buffer.from_bytes(json.dumps(metadata).encode())
        )
    request.getfixturevalue("forbid_sync")
    if valid:
        ds = await xr.open_zarr_async(store, consolidated=False)
        assert ds["value"].dims == ("x",)
        assert ds.sizes == {"x": 2}
    else:
        with pytest.raises(KeyError, match="missing dimension metadata"):
            await xr.open_zarr_async(store, consolidated=False)
    assert store._is_open


@pytest.mark.skipif(
    not has_zarr_async_index(),
    reason="Async orthogonal indexing requires Zarr >= 3.1.2",
)
async def test_async_fancy_indexing(request):
    store = await make_store()
    request.getfixturevalue("forbid_sync")
    request.getfixturevalue("forbid_threads")
    ds = await xr.open_zarr_async(
        store, group="nested", decode_cf=False, create_default_indexes=False
    )
    result = await ds.isel(x=[3, 1]).load_async()
    np.testing.assert_array_equal(result["value"], [3, 1])
