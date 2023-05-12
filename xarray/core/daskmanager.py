from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from xarray.core.duck_array_ops import dask_available
from xarray.core.indexing import ImplicitToExplicitIndexingAdapter
from xarray.core.parallelcompat import ChunkManagerEntrypoint, T_ChunkedArray
from xarray.core.pycompat import is_duck_dask_array

if TYPE_CHECKING:
    from xarray.core.types import DaskArray, T_Chunks, T_NormalizedChunks


class DaskManager(ChunkManagerEntrypoint["DaskArray"]):
    array_cls: type[DaskArray]
    available: bool = dask_available

    def __init__(self):
        # TODO can we replace this with a class attribute instead?

        from dask.array import Array

        self.array_cls = Array

    def is_chunked_array(self, data: Any) -> bool:
        return is_duck_dask_array(data)

    def chunks(self, data: DaskArray) -> T_NormalizedChunks:
        return data.chunks

    def normalize_chunks(
        self,
        chunks: T_Chunks,
        shape: tuple[int] | None = None,
        limit: int | None = None,
        dtype: np.dtype | None = None,
        previous_chunks: T_NormalizedChunks | None = None,
    ) -> T_NormalizedChunks:
        """Called by open_dataset"""
        from dask.array.core import normalize_chunks

        return normalize_chunks(
            chunks,
            shape=shape,
            limit=limit,
            dtype=dtype,
            previous_chunks=previous_chunks,
        )

    def from_array(self, data, chunks, **kwargs) -> DaskArray:
        import dask.array as da

        if isinstance(data, ImplicitToExplicitIndexingAdapter):
            # lazily loaded backend array classes should use NumPy array operations.
            kwargs["meta"] = np.ndarray

        return da.from_array(
            data,
            chunks,
            **kwargs,
        )

    def compute(self, *data: DaskArray, **kwargs) -> tuple[np.ndarray, ...]:
        from dask.array import compute

        return compute(*data, **kwargs)

    @property
    def array_api(self) -> Any:
        from dask import array as da

        return da

    def reduction(
        self,
        arr: T_ChunkedArray,
        func: Callable,
        combine_func: Callable | None = None,
        aggregate_func: Callable | None = None,
        axis: int | Sequence[int] | None = None,
        dtype: np.dtype | None = None,
        keepdims: bool = False,
    ) -> T_ChunkedArray:
        from dask.array import reduction

        return reduction(
            arr,
            chunk=func,
            combine=combine_func,
            aggregate=aggregate_func,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
        )

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

    def map_blocks(
        self,
        func,
        *args,
        dtype=None,
        chunks=None,
        drop_axis=[],
        new_axis=None,
        **kwargs,
    ):
        from dask.array import map_blocks

        # pass through name, meta, token as kwargs
        return map_blocks(
            func,
            *args,
            dtype=dtype,
            chunks=chunks,
            drop_axis=drop_axis,
            new_axis=new_axis,
            **kwargs,
        )

    def blockwise(
        self,
        func,
        out_ind,
        *args,
        name=None,
        token=None,
        dtype=None,
        adjust_chunks=None,
        new_axes=None,
        align_arrays=True,
        concatenate=None,
        meta=None,
        **kwargs,
    ):
        from dask.array import blockwise

        return blockwise(
            func,
            out_ind,
            *args,
            name=name,
            token=token,
            dtype=dtype,
            adjust_chunks=adjust_chunks,
            new_axes=new_axes,
            align_arrays=align_arrays,
            concatenate=concatenate,
            meta=meta,
            **kwargs,
        )

    def unify_chunks(
        self, *args, **kwargs
    ) -> tuple[dict[str, T_NormalizedChunks], list[DaskArray]]:
        from dask.array.core import unify_chunks

        return unify_chunks(*args, **kwargs)

    def store(
        self,
        sources: DaskArray | Sequence[DaskArray],
        targets: Any,
        **kwargs,
    ):
        from dask.array import store

        return store(
            sources=sources,
            targets=targets,
            **kwargs,
        )
