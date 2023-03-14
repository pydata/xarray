from typing import Any, Optional

import numpy as np
import pytest

from xarray.core.parallelcompat import (
    CHUNK_MANAGERS,
    ChunkManager,
    DaskManager,
    T_Chunks,
    get_chunked_array_type,
    get_chunkmanager,
)

dask = pytest.importorskip("dask")


class DummyChunkedArray(np.ndarray):
    """
    Mock-up of a chunked array class.

    Adds a (non-functional) .chunks attribute by following this example in the numpy docs
    https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    """

    chunks: Optional[T_Chunks]

    def __new__(
        cls,
        shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        chunks=None,
    ):
        obj = super().__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.chunks = chunks
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.chunks = getattr(obj, "chunks", None)


class DummyChunkManager(ChunkManager):
    """Mock-up of ChunkManager class for DummyChunkedArray"""

    def __init__(self):
        self.array_cls = DummyChunkedArray

    def is_chunked_array(self, data: Any) -> bool:
        return isinstance(data, DummyChunkedArray)

    def chunks(self, data: DummyChunkedArray) -> T_Chunks:
        return data.chunks

    def from_array(
        self, data: np.ndarray, chunks: T_Chunks, **kwargs
    ) -> DummyChunkedArray:
        from dask import array as da

        return da.from_array(data, chunks, **kwargs)

    def rechunk(self, data: DummyChunkedArray, chunks, **kwargs) -> DummyChunkedArray:
        return data.rechunk(chunks, **kwargs)

    def compute(self, *data: DummyChunkedArray, **kwargs) -> np.ndarray:
        from dask.array import compute

        return compute(*data, **kwargs)

    def apply_gufunc(
        self,
        func,
        signature,
        *args,
        axes=None,
        axis=None,
        keepdims=False,
        output_dtypes=None,
        output_sizes=None,
        vectorize=None,
        allow_rechunk=False,
        meta=None,
        **kwargs,
    ):
        from dask.array.gufunc import apply_gufunc

        return apply_gufunc(
            func,
            signature,
            *args,
            axes=axes,
            axis=axis,
            keepdims=keepdims,
            output_dtypes=output_dtypes,
            output_sizes=output_sizes,
            vectorize=vectorize,
            allow_rechunk=allow_rechunk,
            meta=meta,
            **kwargs,
        )


class TestGetChunkManager:
    # TODO do these need setups and teardowns?

    def test_get_chunkmanger(self):
        CHUNK_MANAGERS["dummy"] = DummyChunkManager

        chunkmanager = get_chunkmanager("dummy")
        assert isinstance(chunkmanager, DummyChunkManager)

    def test_fail_on_nonexistent_chunkmanager(self):
        with pytest.raises(ImportError, match="nonsense has not been defined"):
            get_chunkmanager("nonsense")


class TestGetChunkedArrayType:
    def test_detect_chunked_arrays(self):
        CHUNK_MANAGERS["dummy"] = DummyChunkManager
        dummy_arr = DummyChunkedArray([1, 2, 3])

        chunk_manager = get_chunked_array_type(dummy_arr)
        assert isinstance(chunk_manager, DummyChunkManager)

    def test_ignore_inmemory_arrays(self):
        CHUNK_MANAGERS["dummy"] = DummyChunkManager
        dummy_arr = DummyChunkedArray([1, 2, 3])

        chunk_manager = get_chunked_array_type(*[dummy_arr, 1.0, np.array([5, 6])])
        assert isinstance(chunk_manager, DummyChunkManager)

        with pytest.raises(TypeError, match="Expected a chunked array"):
            get_chunked_array_type(5.0)

    def test_detect_dask_by_default(self):
        dask_arr = dask.array.from_array([1, 2, 3], chunks=(1,))

        chunk_manager = get_chunked_array_type(dask_arr)
        assert isinstance(chunk_manager, DaskManager)

    def test_raise_on_mixed_types(self):
        CHUNK_MANAGERS["dummy"] = DummyChunkManager
        dummy_arr = DummyChunkedArray([1, 2, 3])
        dask_arr = dask.array.from_array([1, 2, 3], chunks=(1,))

        with pytest.raises(TypeError, match="received multiple types"):
            get_chunked_array_type(*[dask_arr, dummy_arr])
