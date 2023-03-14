"""
The code in this module is an experiment in going from N=1 to N=2 parallel computing frameworks in xarray.
It could later be used as the basis for a public interface allowing any N frameworks to interoperate with xarray,
but for now it is just a private experiment.
"""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, Union

import numpy as np

from xarray.core import utils
from xarray.core.pycompat import DuckArrayModule, is_chunked_array, is_duck_dask_array
from xarray.core.types import T_Chunks

T_ChunkManager = TypeVar("T_ChunkManager", bound="ChunkManager")
T_ChunkedArray = TypeVar("T_ChunkedArray")

CHUNK_MANAGERS: dict[str, T_ChunkManager] = {}

if TYPE_CHECKING:
    try:
        from cubed import Array as CubedArray
    except ImportError:
        CubedArray = Any

    try:
        from zarr.core import Array as ZarrArray
    except ImportError:
        ZarrArray = Any


def get_chunkmanager(name: str) -> "ChunkManager":
    if name in CHUNK_MANAGERS:
        chunkmanager_cls = CHUNK_MANAGERS[name]
        return chunkmanager_cls()
    else:
        raise ImportError(f"ChunkManager {name} has not been defined")


def get_chunked_array_type(*args) -> "ChunkManager":
    """
    Detects which parallel backend should be used for given set of arrays.

    Also checks that all arrays are of same chunking type (i.e. not a mix of cubed and dask).
    """

    # TODO this list is probably redundant with something inside xarray.apply_ufunc
    ALLOWED_NON_CHUNKED_TYPES = {int, float, np.ndarray}

    chunked_arrays = [
        a
        for a in args
        if is_chunked_array(a) and type(a) not in ALLOWED_NON_CHUNKED_TYPES
    ]

    # Asserts all arrays are the same type (or numpy etc.)
    chunked_array_types = {type(a) for a in chunked_arrays}
    if len(chunked_array_types) > 1:
        raise TypeError(
            f"Mixing chunked array types is not supported, but received multiple types: {chunked_array_types}"
        )
    elif len(chunked_array_types) == 0:
        raise TypeError("Expected a chunked array but none were found")

    # iterate over defined chunk managers, seeing if each recognises this array type
    chunked_arr = chunked_arrays[0]
    for chunkmanager_cls in CHUNK_MANAGERS.values():
        chunkmanager = chunkmanager_cls()
        if chunkmanager.is_chunked_array(chunked_arr):
            return chunkmanager

    raise ChunkManagerNotFoundError(
        f"Could not find a Chunk Manager which recognises type {type(chunked_arr)}"
    )


class ChunkManagerNotFoundError(Exception):
    ...


class ChunkManager(ABC, Generic[T_ChunkedArray]):
    """
    Adapter between a particular parallel computing framework and xarray.

    Attributes
    ----------
    array_cls
        Type of the array class this parallel computing framework provides.

        Parallel frameworks need to provide an array class that supports the array API standard.
        Used for type checking.
    """

    array_cls: type[T_ChunkedArray]

    @abstractmethod
    def __init__(self):
        ...

    def is_chunked_array(self, data: Any) -> bool:
        return isinstance(data, self.array_cls)

    @abstractmethod
    def chunks(self, data: T_ChunkedArray) -> T_Chunks:
        ...

    @abstractmethod
    def from_array(
        self, data: np.ndarray, chunks: T_Chunks, **kwargs
    ) -> T_ChunkedArray:
        ...

    @abstractmethod
    def rechunk(
        self, data: T_ChunkedArray, chunks: T_Chunks, **kwargs
    ) -> T_ChunkedArray:
        ...

    @abstractmethod
    def compute(self, data: T_ChunkedArray, **kwargs) -> np.ndarray:
        ...

    @property
    def array_api(self) -> Any:
        """Return the array_api namespace following the python array API standard."""
        raise NotImplementedError()

    def reduction(
        self,
        arr: T_ChunkedArray,
        func: Callable,
        combine_func: Callable = None,
        aggregate_func: Callable = None,
        axis: Union[int, Sequence[int]] = None,
        dtype: np.dtype = None,
        keepdims: bool = False,
    ) -> T_ChunkedArray:
        """Used in some reductions like nanfirst, which is used by groupby.first"""
        raise NotImplementedError()

    @abstractmethod
    def apply_gufunc(
        self,
        func,
        signature,
        *args,
        axes=None,
        keepdims=False,
        output_dtypes=None,
        vectorize=None,
        **kwargs,
    ):
        """
        Called inside xarray.apply_ufunc, so must be supplied for vast majority of xarray computations to be supported.
        """
        ...

    def map_blocks(
        self,
        func,
        *args,
        dtype=None,
        **kwargs,
    ):
        """Currently only called in a couple of really niche places in xarray. Not even called in xarray.map_blocks."""
        raise NotImplementedError()

    def blockwise(
        self,
        func,
        out_ind,
        *args,
        adjust_chunks=None,
        new_axes=None,
        align_arrays=True,
        **kwargs,
    ):
        """Called by some niche functions in xarray."""
        raise NotImplementedError()

    def unify_chunks(
        self, *args, **kwargs
    ) -> tuple[dict[str, T_Chunks], list[T_ChunkedArray]]:
        """Called by xr.unify_chunks."""
        raise NotImplementedError()

    def store(
        self,
        sources: Union[T_ChunkedArray, Sequence[T_ChunkedArray]],
        targets: Any,
        **kwargs: dict[str, Any],
    ):
        """Used when writing to any backend."""
        raise NotImplementedError()


T_DaskArray = TypeVar("T_DaskArray", bound="dask.array.Array")


class DaskManager(ChunkManager[T_DaskArray]):
    array_cls: T_DaskArray

    def __init__(self):
        # TODO can we replace this with a class attribute instead?

        from dask.array import Array

        self.array_cls = Array

    def is_chunked_array(self, data: Any) -> bool:
        return is_duck_dask_array(data)

    def chunks(self, data: T_DaskArray) -> T_Chunks:
        return data.chunks

    def from_array(self, data: np.ndarray, chunks, **kwargs) -> T_DaskArray:
        import dask.array as da

        from xarray.core import indexing

        # dask-specific kwargs
        name = kwargs.pop("name", None)
        lock = kwargs.pop("lock", False)
        inline_array = kwargs.pop("inline_array", False)

        if is_duck_dask_array(data):
            data = self.rechunk(data, chunks)
        elif isinstance(data, DuckArrayModule("cubed").type):
            raise TypeError("Trying to rechunk a cubed array using dask")
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

    # TODO is simple method propagation like this necessary?
    def rechunk(self, data: T_DaskArray, chunks, **kwargs) -> T_DaskArray:
        return data.rechunk(chunks, **kwargs)

    def compute(self, *data: T_DaskArray, **kwargs) -> np.ndarray:
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
        combine_func: Callable = None,
        aggregate_func: Callable = None,
        axis: Union[int, Sequence[int]] = None,
        dtype: np.dtype = None,
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
        name=None,
        token=None,
        dtype=None,
        chunks=None,
        drop_axis=None,
        new_axis=None,
        enforce_ndim=False,
        meta=None,
        **kwargs,
    ):
        from dask.array import map_blocks

        return map_blocks(
            func,
            *args,
            name=name,
            token=token,
            dtype=dtype,
            chunks=chunks,
            drop_axis=drop_axis,
            new_axis=new_axis,
            enforce_ndim=enforce_ndim,
            meta=meta,
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
    ) -> tuple[dict[str, T_Chunks], list[T_DaskArray]]:
        from dask.array.core import unify_chunks

        return unify_chunks(*args, **kwargs)

    def store(
        self,
        sources: Union[T_DaskArray, Sequence[T_DaskArray]],
        targets: Any,
        **kwargs: dict[str, Any],
    ):
        from dask.array import store

        # TODO separate expected store kwargs from other compute kwargs?

        return store(
            sources=sources,
            targets=targets,
            **kwargs,
        )


try:
    import dask

    CHUNK_MANAGERS["dask"] = DaskManager
except ImportError:
    pass


class CubedManager(ChunkManager["CubedArray"]):
    def __init__(self):
        from cubed import Array

        self.array_cls = Array

    def chunks(self, data: "CubedArray") -> T_Chunks:
        return data.chunks

    def from_array(self, data: np.ndarray, chunks, **kwargs) -> "CubedArray":
        import cubed  # type: ignore

        spec = kwargs.pop("spec", None)

        if isinstance(data, cubed.Array):
            data = data.rechunk(chunks)
        elif is_duck_dask_array(data):
            raise TypeError("Trying to rechunk a dask array using cubed")
        else:
            data = cubed.from_array(
                data,
                chunks,
                spec=spec,
            )

        return data

    def rechunk(self, data: "CubedArray", chunks, **kwargs) -> "CubedArray":
        return data.rechunk(chunks, **kwargs)

    def compute(self, *data: "CubedArray", **kwargs) -> np.ndarray:
        from cubed import compute

        return compute(*data, **kwargs)

    @property
    def array_api(self) -> Any:
        from cubed import array_api

        return array_api

    def reduction(
        self,
        arr: T_ChunkedArray,
        func: Callable,
        combine_func: Callable = None,
        aggregate_func: Callable = None,
        axis: Union[int, Sequence[int]] = None,
        dtype: np.dtype = None,
        keepdims: bool = False,
    ) -> T_ChunkedArray:
        from cubed.core.ops import reduction

        return reduction(
            arr,
            func=func,
            combine_func=combine_func,
            aggegrate_func=aggregate_func,  # TODO fix the typo in argument name in cubed
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
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
        from cubed.core.ops import map_blocks

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
        *args: Any,
        # can't type this as mypy assumes args are all same type, but blockwise args alternate types
        dtype=None,
        adjust_chunks=None,
        new_axes=None,
        align_arrays=True,
        target_store=None,
        **kwargs,
    ):
        from cubed.core.ops import blockwise

        # TODO where to get the target_store kwarg from? Filter down from a blockwise call? Set as attribute on CubedManager?

        return blockwise(
            func,
            out_ind,
            *args,
            dtype=dtype,
            adjust_chunks=adjust_chunks,
            new_axes=new_axes,
            align_arrays=align_arrays,
            target_store=target_store,
            **kwargs,
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
        if allow_rechunk:
            raise NotImplementedError(
                "cubed.apply_gufunc doesn't support allow_rechunk"
            )
        if keepdims:
            raise NotImplementedError("cubed.apply_gufunc doesn't support keepdims")

        from cubed import apply_gufunc

        return apply_gufunc(
            func,
            signature,
            *args,
            axes=axes,
            axis=axis,
            output_dtypes=output_dtypes,
            output_sizes=output_sizes,
            vectorize=vectorize,
            **kwargs,
        )

    def unify_chunks(
        self, *args, **kwargs
    ) -> tuple[dict[str, T_Chunks], list["CubedArray"]]:
        from cubed.core import unify_chunks

        return unify_chunks(*args, **kwargs)

    def store(
        self,
        sources: Union["CubedArray", Sequence["CubedArray"]],
        targets: Union["ZarrArray", Sequence["ZarrArray"]],
        **kwargs: dict[str, Any],
    ):
        """Used when writing to any backend."""
        from cubed.core.ops import store

        return store(
            sources,
            targets,
            **kwargs,
        )


try:
    CHUNK_MANAGERS["cubed"] = CubedManager
except ImportError:
    pass
