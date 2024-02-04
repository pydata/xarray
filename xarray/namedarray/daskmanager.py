from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from packaging.version import Version

from xarray.core.indexing import ImplicitToExplicitIndexingAdapter
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint, T_ChunkedArray
from xarray.namedarray.utils import is_duck_dask_array, module_available

if TYPE_CHECKING:
    from xarray.namedarray._typing import (
        T_Chunks,
        _DType_co,
        _NormalizedChunks,
        duckarray,
    )


dask_available = module_available("dask")


class DaskManager(ChunkManagerEntrypoint[Any]):
    array_cls: type[Any]
    available: bool = dask_available

    def __init__(self) -> None:
        # TODO can we replace this with a class attribute instead?

        from dask.array import Array

        self.array_cls = Array

    def is_chunked_array(self, data: duckarray[Any, Any]) -> bool:
        return is_duck_dask_array(data)

    def chunks(self, data: Any) -> _NormalizedChunks:
        return data.chunks  # type: ignore

    def normalize_chunks(
        self,
        chunks: T_Chunks | _NormalizedChunks,
        shape: tuple[int, ...] | None = None,
        limit: int | None = None,
        dtype: _DType_co | None = None,
        previous_chunks: _NormalizedChunks | None = None,
    ) -> Any:
        """Called by open_dataset"""
        from dask.array.core import normalize_chunks

        return normalize_chunks(
            chunks,
            shape=shape,
            limit=limit,
            dtype=dtype,
            previous_chunks=previous_chunks,
        )  # type: ignore

    def from_array(self, data: Any, chunks: Any, **kwargs: Any) -> Any:
        import dask.array as da

        if isinstance(data, ImplicitToExplicitIndexingAdapter):
            # lazily loaded backend array classes should use NumPy array operations.
            kwargs["meta"] = np.ndarray

        return da.from_array(
            data,
            chunks,
            **kwargs,
        )  # type: ignore

    def compute(self, *data: Any, **kwargs: Any) -> Any:
        from dask.array import compute

        return compute(*data, **kwargs)  # type: ignore

    @property
    def array_api(self) -> Any:
        from dask import array as da

        return da

    def reduction(
        self,
        arr: T_ChunkedArray,
        func: Callable[..., Any],
        combine_func: Callable[..., Any] | None = None,
        aggregate_func: Callable[..., Any] | None = None,
        axis: int | Sequence[int] | None = None,
        dtype: _DType_co | None = None,
        keepdims: bool = False,
    ) -> Any:
        from dask.array import reduction

        return reduction(
            arr,
            chunk=func,
            combine=combine_func,
            aggregate=aggregate_func,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
        )  # type: ignore

    def scan(
        self,
        func: Callable[..., Any],
        binop: Callable[..., Any],
        ident: float,
        arr: T_ChunkedArray,
        axis: int | None = None,
        dtype: _DType_co | None = None,
        **kwargs: Any,
    ) -> Any:
        from dask.array.reductions import cumreduction

        return cumreduction(
            func,
            binop,
            ident,
            arr,
            axis=axis,
            dtype=dtype,
            **kwargs,
        )  # type: ignore

    def apply_gufunc(
        self,
        func: Callable[..., Any],
        signature: str,
        *args: Any,
        axes: Sequence[tuple[int, ...]] | None = None,
        axis: int | None = None,
        keepdims: bool = False,
        output_dtypes: Sequence[_DType_co] | None = None,
        output_sizes: dict[str, int] | None = None,
        vectorize: bool | None = None,
        allow_rechunk: bool = False,
        meta: tuple[np.ndarray[Any, _DType_co], ...] | None = None,
        **kwargs: Any,
    ) -> Any:
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
        )  # type: ignore

    def map_blocks(
        self,
        func: Callable[..., Any],
        *args: Any,
        dtype: _DType_co | None = None,
        chunks: tuple[int, ...] | None = None,
        drop_axis: int | Sequence[int] | None = None,
        new_axis: int | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> Any:
        import dask
        from dask.array import map_blocks

        if drop_axis is None and Version(dask.__version__) < Version("2022.9.1"):
            # See https://github.com/pydata/xarray/pull/7019#discussion_r1196729489
            # TODO remove once dask minimum version >= 2022.9.1
            drop_axis = []

        # pass through name, meta, token as kwargs
        return map_blocks(
            func,
            *args,
            dtype=dtype,
            chunks=chunks,
            drop_axis=drop_axis,
            new_axis=new_axis,
            **kwargs,
        )  # type: ignore

    def blockwise(
        self,
        func: Callable[..., Any],
        out_ind: Iterable[Any],
        *args: Any,
        # can't type this as mypy assumes args are all same type, but dask blockwise args alternate types
        name: str | None = None,
        token: Any | None = None,
        dtype: _DType_co | None = None,
        adjust_chunks: dict[Any, Callable[..., Any]] | None = None,
        new_axes: dict[Any, int] | None = None,
        align_arrays: bool = True,
        concatenate: bool | None = None,
        meta: tuple[np.ndarray[Any, _DType_co], ...] | None = None,
        **kwargs: Any,
    ) -> Any:
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
        )  # type: ignore

    def unify_chunks(
        self,
        *args: Any,  # can't type this as mypy assumes args are all same type, but dask unify_chunks args alternate types
        **kwargs: Any,
    ) -> Any:
        from dask.array.core import unify_chunks

        return unify_chunks(*args, **kwargs)  # type: ignore

    def store(
        self,
        sources: Any | Sequence[Any],
        targets: Any,
        **kwargs: Any,
    ) -> Any:
        from dask.array import store

        return store(
            sources=sources,
            targets=targets,
            **kwargs,
        )
