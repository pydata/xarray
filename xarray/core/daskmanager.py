from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np

from xarray.core import utils
from xarray.core.duck_array_ops import dask_available
from xarray.core.parallelcompat import ChunkManagerEntrypoint, T_ChunkedArray, T_Chunks
from xarray.core.pycompat import is_duck_dask_array

if TYPE_CHECKING:
    from xarray.core.types import DaskArray


class DaskManager(ChunkManagerEntrypoint["DaskArray"]):
    array_cls: type["DaskArray"]
    available: bool = dask_available

    def __init__(self):
        # TODO can we replace this with a class attribute instead?

        from dask.array import Array

        self.array_cls = Array

    def is_chunked_array(self, data: Any) -> bool:
        return is_duck_dask_array(data)

    def chunks(self, data: "DaskArray") -> T_Chunks:
        return data.chunks

    def from_array(self, data, chunks, **kwargs) -> "DaskArray":
        import dask.array as da

        from xarray.core import indexing

        # dask-specific kwargs
        name = kwargs.pop("name", None)
        lock = kwargs.pop("lock", False)
        inline_array = kwargs.pop("inline_array", False)

        if is_duck_dask_array(data):
            data = self.rechunk(data, chunks)
        else:
            if isinstance(data, indexing.ExplicitlyIndexed):
                # Unambiguously handle array storage backends (like NetCDF4 and h5py)
                # that can't handle general array indexing. For example, in netCDF4 you
                # can do "outer" indexing along two dimensions independent, which works
                # differently from how NumPy handles it.
                # da.from_array works by using lazy indexing with a tuple of slices.
                # Using OuterIndexer is a pragmatic choice: dask does not yet handle
                # different indexing types in an explicit way:
                # https://github.com/dask/dask/issues/2883
                data = indexing.ImplicitToExplicitIndexingAdapter(
                    data, indexing.OuterIndexer
                )

                # All of our lazily loaded backend array classes should use NumPy
                # array operations.
                dask_kwargs = {"meta": np.ndarray}
            else:
                dask_kwargs = {}

            if utils.is_dict_like(chunks):
                chunks = tuple(chunks.get(n, s) for n, s in enumerate(data.shape))

            data = da.from_array(
                data,
                chunks,
                name=name,
                lock=lock,
                inline_array=inline_array,
                **dask_kwargs,
            )
        return data

    def compute(self, *data: "DaskArray", **kwargs) -> np.ndarray:
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
        combine_func: Optional[Callable] = None,
        aggregate_func: Optional[Callable] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        dtype: Optional[np.dtype] = None,
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
    ) -> tuple[dict[str, T_Chunks], list["DaskArray"]]:
        from dask.array.core import unify_chunks

        return unify_chunks(*args, **kwargs)

    def store(
        self,
        sources: Union["DaskArray", Sequence["DaskArray"]],
        targets: Any,
        **kwargs,
    ):
        from dask.array import store

        # TODO separate expected store kwargs from other compute kwargs?

        return store(
            sources=sources,
            targets=targets,
            **kwargs,
        )
