"""
The code in this module is an experiment in going from N=1 to N=2 parallel computing frameworks in xarray.
It could later be used as the basis for a public interface allowing any N frameworks to interoperate with xarray,
but for now it is just a private experiment.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Tuple, Type, TypeVar

import numpy as np
from typing_extensions import TypeAlias

from xarray.core import indexing, utils
from xarray.core.pycompat import DuckArrayModule, is_duck_dask_array

T_ChunkManager = TypeVar("T_ChunkManager", bound="ChunkManager")
T_ChunkedArray = TypeVar("T_ChunkedArray")
T_Chunks: TypeAlias = Tuple[Tuple[int, ...], ...]

CHUNK_MANAGERS: Dict[str, T_ChunkManager] = {}


def _get_chunk_manager(name: str) -> "ChunkManager":
    if name in CHUNK_MANAGERS:
        chunkmanager = CHUNK_MANAGERS[name]
        return chunkmanager()
    else:
        raise ImportError(f"ChunkManager {name} has not been defined")


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

    array_cls: Type[T_ChunkedArray]

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
    def apply_gufunc(self):
        """
        Called inside xarray.apply_ufunc, so must be supplied for vast majority of xarray computations to be supported.
        """
        ...

    def map_blocks(self):
        raise NotImplementedError()

    def blockwise(self):
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

    def apply_gufunc(self):
        from dask.array.gufunc import apply_gufunc

        ...

    def map_blocks(self):
        from dask.array import map_blocks

        ...

    def blockwise(self):
        from dask.array import blockwise

        ...


try:
    import dask

    CHUNK_MANAGERS["dask"] = DaskManager
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


try:
    import cubed  # type: ignore

    CHUNK_MANAGERS["cubed"] = CubedManager
except ImportError:
    pass
