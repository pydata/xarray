"""
The code in this module is an experiment in going from N=1 to N=2 parallel computing frameworks in xarray.
It could later be used as the basis for a public interface allowing any N frameworks to interoperate with xarray,
but for now it is just a private experiment.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from . import indexing, utils
from .pycompat import DuckArrayModule, is_duck_dask_array

CHUNK_MANAGERS = {}


def _get_chunk_manager(name: str) -> "ChunkManager":
    if name in CHUNK_MANAGERS:
        chunkmanager = CHUNK_MANAGERS[name]
        return chunkmanager()
    else:
        raise ImportError(f"ChunkManager {name} has not been defined")


class ChunkManager(ABC):
    """
    Adapter between a particular parallel computing framework and xarray.

    Attributes
    ----------
    array_type
        Type of the array class this parallel computing framework provides.

        Parallel frameworks need to provide an array class that supports the array API standard.
        Used for type checking.
    """

    @abstractmethod
    def chunks(self, arr):
        ...

    @abstractmethod
    def from_array(self, data: np.ndarray, chunks, **kwargs):
        ...

    @abstractmethod
    def rechunk(self, data: Any, chunks, **kwargs):
        ...

    @abstractmethod
    def compute(self, arr, **kwargs) -> np.ndarray:
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


class DaskManager(ChunkManager):
    def __init__(self):
        from dask.array import Array

        self.array_type = Array

    def chunks(self, arr: "dask.array.Array"):
        return arr.chunks

    def chunk(self, data: Any, chunks, **kwargs):
        import dask.array as da

        # dask-specific kwargs
        name = kwargs.pop("name", None)
        lock = kwargs.pop("lock", False)
        inline_array = kwargs.pop("inline_array", False)

        if is_duck_dask_array(data):
            data = data.rechunk(chunks)
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

    def rechunk(self, chunks, **kwargs):
        ...

    def compute(self, arr, **kwargs):
        return arr.compute(**kwargs)

    def apply_ufunc(self):
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


class CubedManager(ChunkManager):
    def __init__(self):
        from cubed import Array

        self.array_type = Array

    def chunk(self, data: np.ndarray, chunks, **kwargs):
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
    import cubed

    CHUNK_MANAGERS["cubed"] = CubedManager
except ImportError:
    pass
