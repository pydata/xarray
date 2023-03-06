"""
The code in this module is an experiment in going from N=1 to N=2 parallel computing frameworks in xarray.
It could later be used as the basis for a public interface allowing any N frameworks to interoperate with xarray,
but for now it is just a private experiment.
"""
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
from typing_extensions import TypeAlias

from xarray.core import indexing, utils
from xarray.core.pycompat import DuckArrayModule, is_chunked_array, is_duck_dask_array

T_ChunkManager = TypeVar("T_ChunkManager", bound="ChunkManager")
T_ChunkedArray = TypeVar("T_ChunkedArray")
T_Chunks: TypeAlias = tuple[tuple[int, ...], ...]

CHUNK_MANAGERS: dict[str, T_ChunkManager] = {}


def _get_chunk_manager(name: str) -> "ChunkManager":
    if name in CHUNK_MANAGERS:
        chunkmanager = CHUNK_MANAGERS[name]
        return chunkmanager
    else:
        raise ImportError(f"ChunkManager {name} has not been defined")


def _detect_parallel_array_type(*args) -> "ChunkManager":
    """
    Detects which parallel backend should be used for given set of arrays.

    Also checks that all arrays are of same chunking type (i.e. not a mix of cubed and dask).
    """

    # TODO this list is probably redundant with something inside xarray.apply_ufunc
    ALLOWED_NON_CHUNKED_TYPES = {int, float, np.ndarray}

    chunked_array_types_found = {
        type(a)
        for a in args
        if is_chunked_array(a) and type(a) not in ALLOWED_NON_CHUNKED_TYPES
    }

    # Asserts all arrays are the same type (or numpy etc.)
    if len(chunked_array_types_found) > 1:
        raise TypeError(
            f"Mixing chunked array types is not supported, but received types {chunked_array_types_found}"
        )

    (chunked_arr_type,) = chunked_array_types_found

    # iterate over defined chunk managers, seeing if each recognises this array type
    for chunkmanager in CHUNK_MANAGERS.values():
        if chunked_arr_type == chunkmanager.array_cls:
            return chunkmanager

    raise ChunkManagerNotFoundError(
        f"Could not find a Chunk Manager which recognises type {chunked_arr_type}"
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

    def is_array_type(self, data: Any) -> bool:
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


T_DaskArray = TypeVar("T_DaskArray", bound="dask.array.Array")


class DaskManager(ChunkManager[T_DaskArray]):
    array_cls: T_DaskArray

    def __init__(self):
        from dask.array import Array

        self.array_cls = Array

    def chunks(self, data: T_DaskArray) -> T_Chunks:
        return data.chunks

    def from_array(self, data: np.ndarray, chunks, **kwargs) -> T_DaskArray:
        import dask.array as da

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

    def rechunk(self, data: T_DaskArray, chunks, **kwargs) -> T_DaskArray:
        return data.rechunk(chunks, **kwargs)

    def compute(self, data: T_DaskArray, **kwargs) -> np.ndarray:
        return data.compute(**kwargs)

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


try:
    import dask

    CHUNK_MANAGERS["dask"] = DaskManager()
except ImportError:
    pass


T_CubedArray = TypeVar("T_CubedArray", bound="cubed.Array")


class CubedManager(ChunkManager[T_CubedArray]):
    def __init__(self):
        from cubed import Array

        self.array_cls = Array

    def from_array(self, data: np.ndarray, chunks, **kwargs) -> T_CubedArray:
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

    def rechunk(self, data: T_CubedArray, chunks, **kwargs) -> T_CubedArray:
        return data.rechunk(chunks, **kwargs)

    def compute(self, data: T_CubedArray, **kwargs) -> np.ndarray:
        return data.compute(**kwargs)

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


try:
    import cubed  # type: ignore

    CHUNK_MANAGERS["cubed"] = CubedManager()
except ImportError:
    pass
