"""
The code in this module is an experiment in going from N=1 to N=2 parallel computing frameworks in xarray.
It could later be used as the basis for a public interface allowing any N frameworks to interoperate with xarray,
but for now it is just a private experiment.
"""
from __future__ import annotations

import functools
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from importlib.metadata import EntryPoint, entry_points
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    TypeVar,
)

import numpy as np

from xarray.core.pycompat import is_chunked_array

T_ChunkedArray = TypeVar("T_ChunkedArray")

if TYPE_CHECKING:
    from xarray.core.types import T_Chunks, T_NormalizedChunks


@functools.lru_cache(maxsize=1)
def list_chunkmanagers() -> dict[str, ChunkManagerEntrypoint]:
    """
    Return a dictionary of available chunk managers and their ChunkManagerEntrypoint subclass objects.

    Returns
    -------
    chunnkmanagers : dict
        Dictionary whose values are registered ChunkManagerEntrypoint subclass instances, and whose values
        are the strings under which they are registered.

    Notes
    -----
    # New selection mechanism introduced with Python 3.10. See GH6514.
    """
    if sys.version_info >= (3, 10):
        entrypoints = entry_points(group="xarray.chunkmanagers")
    else:
        entrypoints = entry_points().get("xarray.chunkmanagers", ())

    return load_chunkmanagers(entrypoints)


def load_chunkmanagers(
    entrypoints: Sequence[EntryPoint],
) -> dict[str, ChunkManagerEntrypoint]:
    """Load entrypoints and instantiate chunkmanagers only once."""

    loaded_entrypoints = {
        entrypoint.name: entrypoint.load() for entrypoint in entrypoints
    }

    available_chunkmanagers = {
        name: chunkmanager()
        for name, chunkmanager in loaded_entrypoints.items()
        if chunkmanager.available
    }
    return available_chunkmanagers


def guess_chunkmanager(
    manager: str | ChunkManagerEntrypoint | None,
) -> ChunkManagerEntrypoint:
    """
    Get namespace of chunk-handling methods, guessing from what's available.

    If the name of a specific ChunkManager is given (e.g. "dask"), then use that.
    Else use whatever is installed, defaulting to dask if there are multiple options.
    """

    chunkmanagers = list_chunkmanagers()

    if manager is None:
        if len(chunkmanagers) == 1:
            # use the only option available
            manager = next(iter(chunkmanagers.keys()))
        else:
            # default to trying to use dask
            manager = "dask"

    if isinstance(manager, str):
        if manager not in chunkmanagers:
            raise ValueError(
                f"unrecognized chunk manager {manager} - must be one of: {list(chunkmanagers)}"
            )

        return chunkmanagers[manager]
    elif isinstance(manager, ChunkManagerEntrypoint):
        # already a valid ChunkManager so just pass through
        return manager
    else:
        raise TypeError(
            f"manager must be a string or instance of ChunkManagerEntrypoint, but received type {type(manager)}"
        )


def get_chunked_array_type(*args) -> ChunkManagerEntrypoint:
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
    chunkmanagers = list_chunkmanagers()
    selected = [
        chunkmanager
        for chunkmanager in chunkmanagers.values()
        if chunkmanager.is_chunked_array(chunked_arr)
    ]
    if not selected:
        raise TypeError(
            f"Could not find a Chunk Manager which recognises type {type(chunked_arr)}"
        )
    elif len(selected) >= 2:
        raise TypeError(f"Multiple ChunkManagers recognise type {type(chunked_arr)}")
    else:
        return selected[0]


class ChunkManagerEntrypoint(ABC, Generic[T_ChunkedArray]):
    """
    Interface between a particular parallel computing framework and xarray.

    This abstract base class must be subclassed by libraries implementing chunked array types, and
    registered via the ``chunkmanagers`` entrypoint.

    Abstract methods on this class must be implemented, whereas non-abstract methods are only required in order to
    enable a subset of xarray functionality, and by default will raise a ``NotImplementedError`` if called.

    Attributes
    ----------
    array_cls
        Type of the array class this parallel computing framework provides.

        Parallel frameworks need to provide an array class that supports the array API standard.
        This attribute is used for array instance type checking at runtime.
    """

    array_cls: type[T_ChunkedArray]
    available: bool = True

    @abstractmethod
    def __init__(self) -> None:
        """Used to set the array_cls attribute at import time."""
        raise NotImplementedError()

    def is_chunked_array(self, data: Any) -> bool:
        """
        Check if the given object is an instance of this type of chunked array.

        Compares against the type stored in the array_cls attribute by default.
        """
        return isinstance(data, self.array_cls)

    @abstractmethod
    def chunks(self, data: T_ChunkedArray) -> T_NormalizedChunks:
        """
        Return the current chunks of the given array.

        Used internally by xarray objects' .chunks and .chunksizes properties.

        See Also
        --------
        dask.array.Array.chunks
        """
        raise NotImplementedError()

    @abstractmethod
    def normalize_chunks(
        self,
        chunks: T_Chunks | T_NormalizedChunks,
        shape: tuple[int, ...] | None = None,
        limit: int | None = None,
        dtype: np.dtype | None = None,
        previous_chunks: T_NormalizedChunks | None = None,
    ) -> T_NormalizedChunks:
        """
        Called internally by xarray.open_dataset.

        See Also
        --------
        dask.array.normalize_chunks
        """
        raise NotImplementedError()

    @abstractmethod
    def from_array(
        self, data: np.ndarray, chunks: T_Chunks, **kwargs
    ) -> T_ChunkedArray:
        """
        Creates a chunked array from a non-chunked numpy-like array.

        Called when the .chunk method is called on an xarray object that is not already chunked.
        Also called within open_dataset (when chunks is not None) to create a chunked array from
        an xarray lazily indexed array.

        See Also
        --------
        dask.Array.array.from_array
        """
        raise NotImplementedError()

    def rechunk(
        self,
        data: T_ChunkedArray,
        chunks: T_NormalizedChunks | tuple[int, ...] | T_Chunks,
        **kwargs,
    ) -> T_ChunkedArray:
        """
        Changes the chunking pattern of the given array.

        Called when the .chunk method is called on an xarray object that is already chunked.

        See Also
        --------
        dask.array.Array.rechunk
        """
        return data.rechunk(chunks, **kwargs)  # type: ignore[attr-defined]

    @abstractmethod
    def compute(self, *data: T_ChunkedArray, **kwargs) -> tuple[np.ndarray, ...]:
        """
        Computes one or more chunked arrays, returning them as eager numpy arrays.

        Called anytime something needs to computed, including multiple arrays at once.
        Used by `.compute`, `.persist`, `.values`.

        See Also
        --------
        dask.array.compute
        """
        raise NotImplementedError()

    @property
    def array_api(self) -> Any:
        """
        Return the array_api namespace following the python array API standard.

        See Also
        --------
        dask.array
        """
        raise NotImplementedError()

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
        """
        Used in some reductions like nanfirst, which is used by groupby.first.

        See Also
        --------
        dask.array.reduction
        """
        raise NotImplementedError()

    @abstractmethod
    def apply_gufunc(
        self,
        func: Callable,
        signature: str,
        *args: Any,
        axes: Sequence[tuple[int, ...]] | None = None,
        keepdims: bool = False,
        output_dtypes: Sequence[np.typing.DTypeLike] | None = None,
        vectorize: bool | None = None,
        **kwargs,
    ):
        """
        Called inside xarray.apply_ufunc, so must be supplied for vast majority of xarray computations to be supported.

        See Also
        --------
        dask.array.apply_gufunc
        """
        raise NotImplementedError()

    def map_blocks(
        self,
        func: Callable,
        *args: Any,
        dtype: np.typing.DTypeLike | None = None,
        chunks: tuple[int, ...] | None = None,
        drop_axis: int | Sequence[int] | None = None,
        new_axis: int | Sequence[int] | None = None,
        **kwargs,
    ):
        """
        Called in elementwise operations, but notably not called in xarray.map_blocks.

        See Also
        --------
        dask.array.map_blocks
        """
        raise NotImplementedError()

    def blockwise(
        self,
        func: Callable,
        out_ind: Iterable,
        *args: Any,  # can't type this as mypy assumes args are all same type, but dask blockwise args alternate types
        adjust_chunks: dict[Any, Callable] | None = None,
        new_axes: dict[Any, int] | None = None,
        align_arrays: bool = True,
        **kwargs,
    ):
        """
        Called by some niche functions in xarray.

        See Also
        --------
        dask.array.blockwise
        """
        raise NotImplementedError()

    def unify_chunks(
        self,
        *args: Any,  # can't type this as mypy assumes args are all same type, but dask unify_chunks args alternate types
        **kwargs,
    ) -> tuple[dict[str, T_NormalizedChunks], list[T_ChunkedArray]]:
        """
        Called by xarray.unify_chunks.

        See Also
        --------
        dask.array.unify_chunks
        """
        raise NotImplementedError()

    def store(
        self,
        sources: T_ChunkedArray | Sequence[T_ChunkedArray],
        targets: Any,
        **kwargs: dict[str, Any],
    ):
        """
        Used when writing to any backend.

        See Also
        --------
        dask.array.store
        """
        raise NotImplementedError()
