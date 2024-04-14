from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload

import numpy as np
from packaging.version import Version

from xarray.core.indexing import ImplicitToExplicitIndexingAdapter
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint
from xarray.namedarray.utils import is_duck_dask_array, module_available

if TYPE_CHECKING:
    from xarray.namedarray._typing import (
        T_Chunks,
        _Chunks,
        _ChunksLike,
        _ChunksType,
        _dtype,
        _DType,
        _DType_co,
        _NormalizedChunks,
        _Shape,
        _ShapeType,
        chunkedduckarray,
        duckarray,
    )

    try:
        from dask.array.core import Array as DaskArray
    except ImportError:
        DaskArray = np.ndarray[Any, Any]  # type: ignore[assignment,  misc]


dask_available = module_available("dask")

_T_DaskArray = TypeVar("_T_DaskArray", bound="DaskArray")


class DaskManager(ChunkManagerEntrypoint):  # type: ignore[type-var]
    # array_cls: type[DaskArray]
    available: bool = dask_available

    def __init__(self) -> None:
        # TODO can we replace this with a class attribute instead?

        from dask.array.core import Array

        self.array_cls = Array

    def is_chunked_array(self, data: duckarray[Any, Any]) -> bool:
        return is_duck_dask_array(data)

    def chunks(self, data: chunkedduckarray[Any, Any]) -> _Chunks:
        return data.chunks  # type: ignore[no-any-return]

    def normalize_chunks(
        self,
        chunks: _ChunksLike,
        shape: _Shape | None = None,
        limit: int | None = None,
        dtype: _dtype | None = None,
        previous_chunks: _Chunks | None = None,
    ) -> _Chunks:
        """Called by open_dataset"""
        from dask.array.core import normalize_chunks

        return normalize_chunks(
            chunks,
            shape=shape,
            limit=limit,
            dtype=dtype,
            previous_chunks=previous_chunks,
        )  # type: ignore[no-untyped-call]

    def from_array(
        self, data: duckarray[Any, _DType], chunks: _ChunksLike, **kwargs: Any
    ) -> chunkedduckarray[Any, _DType]:
        from dask.array.core import from_array

        if isinstance(data, ImplicitToExplicitIndexingAdapter):
            # lazily loaded backend array classes should use NumPy array operations.
            kwargs["meta"] = np.ndarray

        return from_array(
            data,
            chunks,
            **kwargs,
        )  # type: ignore[no-untyped-call]

    def compute(
        self, *data: chunkedduckarray[Any, _DType] | Any, **kwargs: Any
    ) -> tuple[np.ndarray[Any, np.dtype[np.generic]], ...]:
        from dask.base import compute

        return compute(*data, **kwargs)  # type: ignore[no-untyped-call, no-any-return]

    @property
    def array_api(self) -> Any:
        from dask import array as da

        return da

    def reduction(
        self,
        arr: chunkedduckarray[Any, _DType],
        func: Callable[..., Any],
        combine_func: Callable[..., Any] | None = None,
        aggregate_func: Callable[..., Any] | None = None,
        axis: int | Sequence[int] | None = None,
        dtype: _DType | None = None,
        keepdims: bool = False,
    ) -> chunkedduckarray[Any, _DType]:
        from dask.array.reductions import reduction

        return reduction(
            arr,
            chunk=func,
            combine=combine_func,
            aggregate=aggregate_func,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
        )  # type: ignore[no-untyped-call]

    def scan(
        self,
        func: Callable[..., Any],
        binop: Callable[..., Any],
        ident: float,
        arr: chunkedduckarray[Any, _DType],
        axis: int | None = None,
        dtype: _DType | None = None,
        **kwargs: Any,
    ) -> chunkedduckarray[Any, _DType]:
        from dask.array.reductions import cumreduction

        return cumreduction(
            func,
            binop,
            ident,
            arr,
            axis=axis,
            dtype=dtype,
            **kwargs,
        )  # type: ignore[no-untyped-call]

    def apply_gufunc(
        self,
        func: Callable[..., Any],
        signature: str,
        *args: Any,
        axes: Sequence[tuple[int, ...]] | None = None,
        keepdims: bool = False,
        output_dtypes: Sequence[_DType] | None = None,
        vectorize: bool | None = None,
        axis: int | None = None,
        output_sizes: dict[str, int] | None = None,
        allow_rechunk: bool = False,
        meta: tuple[np.ndarray[Any, np.dtype[np.generic]], ...] | None = None,
        **kwargs: Any,
    ) -> chunkedduckarray[Any, _DType] | tuple[chunkedduckarray[Any, _DType], ...]:
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
        )  # type: ignore[no-untyped-call]

    def map_blocks(
        self,
        func: Callable[..., Any],
        *args: Any,
        dtype: _DType | None = None,
        chunks: _Chunks | None = None,
        drop_axis: int | Sequence[int] | None = None,
        new_axis: int | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> chunkedduckarray[Any, _DType]:
        import dask
        from dask.array.core import map_blocks

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
        )  # type: ignore[no-untyped-call]

    def blockwise(
        self,
        func: Callable[..., Any],
        out_ind: Iterable[Any],
        *args: Any,  # can't type this as mypy assumes args are all same type, but dask blockwise args alternate types
        adjust_chunks: dict[Any, Callable[..., Any]] | None = None,
        new_axes: dict[Any, int] | None = None,
        align_arrays: bool = True,
        name: str | None = None,
        token: Any | None = None,
        dtype: _DType | None = None,
        concatenate: bool | None = None,
        meta: tuple[np.ndarray[Any, np.dtype[np.generic]], ...] | None = None,
        **kwargs: Any,
    ) -> chunkedduckarray[Any, _DType]:
        from dask.array.blockwise import blockwise

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
        )  # type: ignore[no-untyped-call]

    def unify_chunks(
        self,
        *args: Any,  # can't type this as mypy assumes args are all same type, but dask unify_chunks args alternate types
        **kwargs: Any,
    ) -> tuple[dict[str, _Chunks], list[chunkedduckarray[Any, Any]]]:
        from dask.array.core import unify_chunks

        return unify_chunks(*args, **kwargs)  # type: ignore[no-any-return, no-untyped-call]

    def store(
        self,
        sources: Any | Sequence[Any],
        targets: Any,
        **kwargs: Any,
    ) -> Any:
        from dask.array.core import store

        return store(
            sources=sources,
            targets=targets,
            **kwargs,
        )
