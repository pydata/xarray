from typing import Any

import numpy as np
import pytest

import xarray as xr
from xarray.core.types import T_NormalizedChunks
from xarray.namedarray._typing import T_Chunks, _Chunks
from xarray.namedarray.parallelcompat import (
    ChunkManagerEntrypoint,
    list_chunkmanagers,
)
from xarray.namedarray.pycompat import is_chunked_array
from xarray.tests import assert_identical, requires_zarr


class SimpleChunkedArray(np.ndarray):
    """
    A custom array-like structure that exposes chunks as a simple tuple
    instead of a tuple of tuples.
    """
    def __new__(cls, input_array, chunks=None):
        obj = np.asarray(input_array).view(cls)
        obj._chunks = chunks
        return obj

    @property
    def chunks(self):
        return self._chunks


class SimpleChunkManager(ChunkManagerEntrypoint):
    """Minimal ChunkManager for SimpleChunkedArray"""

    def __init__(self):
        self.array_cls = SimpleChunkedArray

    def is_chunked_array(self, data) -> bool:
        return isinstance(data, SimpleChunkedArray)

    def chunks(self, data: SimpleChunkedArray) -> T_NormalizedChunks:
        chunks = data.chunks
        if chunks and isinstance(chunks[0], int):
            return tuple((chunk,) for chunk in chunks)
        return chunks or ()

    def normalize_chunks(
        self,
        chunks: T_Chunks | T_NormalizedChunks,
        shape: tuple[int, ...] | None = None,
        limit: int | None = None,
        dtype: np.dtype | None = None,
        previous_chunks: T_NormalizedChunks | None = None,
    ) -> T_NormalizedChunks:
        if isinstance(chunks, tuple) and chunks and isinstance(chunks[0], int):
            return tuple((chunk,) for chunk in chunks)
        return chunks or ()

    def from_array(
        self, data, chunks: _Chunks, **kwargs
    ) -> SimpleChunkedArray:
        arr = np.asarray(data)
        return SimpleChunkedArray(arr, chunks=chunks)

    def rechunk(self, data: SimpleChunkedArray, chunks, **kwargs) -> SimpleChunkedArray:
        return SimpleChunkedArray(data, chunks=chunks)

    def compute(self, *data: SimpleChunkedArray, **kwargs) -> tuple[np.ndarray, ...]:
        return tuple(np.asarray(d) for d in data)

    def store(
        self,
        sources: SimpleChunkedArray | list[SimpleChunkedArray],
        targets: Any,
        **kwargs: dict[str, Any],
    ) -> Any:
        if isinstance(sources, list):
            for source, target in zip(sources, targets):
                arr = np.asarray(source)
                if hasattr(target, '__setitem__'):
                    regions = kwargs.get('regions', [None] * len(sources))
                    region = regions[sources.index(source)] if sources.index(source) < len(regions) else None
                    if region:
                        target[region] = arr
                    else:
                        target[...] = arr
        else:
            arr = np.asarray(sources)
            regions = kwargs.get('regions', [None])
            region = regions[0] if regions else None
            if region:
                targets[region] = arr
            else:
                targets[...] = arr
        return None

    def apply_gufunc(self, *args,  **kwargs):
        raise NotImplementedError("SimpleChunkManager does not support gufunc")


@pytest.fixture
def register_simple_chunkmanager(monkeypatch):
    """Register SimpleChunkManager for testing"""
    preregistered_chunkmanagers = list_chunkmanagers()
    monkeypatch.setattr(
        "xarray.namedarray.parallelcompat.list_chunkmanagers",
        lambda: {"simple": SimpleChunkManager()} | preregistered_chunkmanagers,
    )
    yield


@requires_zarr
def test_zarr_with_simple_chunks_array_class(tmp_path, register_simple_chunkmanager):
    arr = np.arange(250).reshape(10, 25)

    simple_chunked_arr = SimpleChunkedArray(arr, chunks=(2, 5))

    assert simple_chunked_arr.chunks == (2, 5)
    assert is_chunked_array(simple_chunked_arr)

    ds = xr.Dataset({"test_var": (("x", "y"), simple_chunked_arr)})

    assert ds["test_var"].variable.chunks == (2, 5)

    zarr_path = tmp_path / "test.zarr"

    ds.to_zarr(zarr_path)

    with xr.open_zarr(zarr_path) as loaded:
        assert_identical(ds.load(), loaded.load())
