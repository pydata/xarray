from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union, cast

import numpy as np

# TODO: get rid of this after migrating this class to array API
from xarray.core import dtypes, formatting, formatting_html
from xarray.core.indexing import (
    ExplicitlyIndexed,
    ImplicitToExplicitIndexingAdapter,
    OuterIndexer,
)
from xarray.namedarray._aggregations import NamedArrayAggregations
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import array_type, is_chunked_array
from xarray.namedarray.utils import (
    Default,
    T_DuckArray,
    _default,
    astype,
    consolidate_dask_from_array_kwargs,
    either_dict_or_kwargs,
    is_chunked_duck_array,
    is_dict_like,
    is_duck_array,
    is_duck_dask_array,
    to_0d_object_array,
)

if TYPE_CHECKING:
    from xarray.core.types import Dims
    from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint
    from xarray.namedarray.utils import Self  # type: ignore[attr-defined]

    try:
        from dask.typing import (
            Graph,
            NestedKeys,
            PostComputeCallable,
            PostPersistCallable,
            SchedulerGetCallable,
        )
    except ImportError:
        Graph: Any  # type: ignore[no-redef]
        NestedKeys: Any  # type: ignore[no-redef]
        SchedulerGetCallable: Any  # type: ignore[no-redef]
        PostComputeCallable: Any  # type: ignore[no-redef]
        PostPersistCallable: Any  # type: ignore[no-redef]

    # T_NamedArray = TypeVar("T_NamedArray", bound="NamedArray[T_DuckArray]")
    DimsInput = Union[str, Iterable[Hashable]]
    DimsProperty = tuple[Hashable, ...]
    AttrsInput = Union[Mapping[Any, Any], None]


# TODO: Add tests!
def as_compatible_data(
    data: T_DuckArray | np.typing.ArrayLike, fastpath: bool = False
) -> T_DuckArray:
    if fastpath and getattr(data, "ndim", 0) > 0:
        # can't use fastpath (yet) for scalars
        return cast(T_DuckArray, data)

    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)  # type: ignore[no-untyped-call]
        if mask.any():
            # TODO: requires refactoring/vendoring xarray.core.dtypes and xarray.core.duck_array_ops
            raise NotImplementedError("MaskedArray is not supported yet")
        else:
            return cast(T_DuckArray, np.asarray(data))
    if is_duck_array(data):
        return data
    if isinstance(data, NamedArray):
        return cast(T_DuckArray, data.data)

    if isinstance(data, ExplicitlyIndexed):
        # TODO: better that is_duck_array(ExplicitlyIndexed) -> True
        return cast(T_DuckArray, data)

    if isinstance(data, tuple):
        data = to_0d_object_array(data)

    # validate whether the data is valid data types.
    return cast(T_DuckArray, np.asarray(data))


class NamedArray(NamedArrayAggregations, Generic[T_DuckArray]):

    """A lightweight wrapper around duck arrays with named dimensions and attributes which describe a single Array.
    Numeric operations on this object implement array broadcasting and dimension alignment based on dimension names,
    rather than axis order."""

    __slots__ = ("_data", "_dims", "_attrs")

    _data: T_DuckArray
    _dims: DimsProperty
    _attrs: dict[Any, Any] | None

    def __init__(
        self,
        dims: DimsInput,
        data: T_DuckArray | np.typing.ArrayLike,
        attrs: AttrsInput = None,
        fastpath: bool = False,
    ):
        """
        Parameters
        ----------
        dims : str or iterable of str
            Name(s) of the dimension(s).
        data : T_DuckArray or np.typing.ArrayLike
            The actual data that populates the array. Should match the shape specified by `dims`.
        attrs : dict, optional
            A dictionary containing any additional information or attributes you want to store with the array.
            Default is None, meaning no attributes will be stored.
        fastpath : bool, optional
            A flag to indicate if certain validations should be skipped for performance reasons.
            Should only be True if you are certain about the integrity of the input data.
            Default is False.

        Raises
        ------
        ValueError
            If the `dims` length does not match the number of data dimensions (ndim).


        """
        self._data = as_compatible_data(data, fastpath=fastpath)
        self._dims = self._parse_dimensions(dims)
        self._attrs = dict(attrs) if attrs else None

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions.

        See Also
        --------
        numpy.ndarray.ndim
        """
        return len(self.shape)

    @property
    def size(self) -> int:
        """
        Number of elements in the array.

        Equal to ``np.prod(a.shape)``, i.e., the product of the array’s dimensions.

        See Also
        --------
        numpy.ndarray.size
        """
        return math.prod(self.shape)

    def __len__(self) -> int:
        try:
            return self.shape[0]
        except Exception as exc:
            raise TypeError("len() of unsized object") from exc

    @property
    def dtype(self) -> np.dtype[Any]:
        """
        Data-type of the array’s elements.

        See Also
        --------
        ndarray.dtype
        numpy.dtype
        """
        return self._data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """


        Returns
        -------
        shape : tuple of ints
                Tuple of array dimensions.



        See Also
        --------
        numpy.ndarray.shape
        """
        return self._data.shape

    @property
    def nbytes(self) -> int:
        """
        Total bytes consumed by the elements of the data array.

        If the underlying data array does not include ``nbytes``, estimates
        the bytes consumed based on the ``size`` and ``dtype``.
        """
        if hasattr(self._data, "nbytes"):
            return self._data.nbytes  # type: ignore[no-any-return]
        else:
            return self.size * self.dtype.itemsize

    @property
    def dims(self) -> DimsProperty:
        """Tuple of dimension names with which this NamedArray is associated."""
        return self._dims

    @dims.setter
    def dims(self, value: DimsInput) -> None:
        self._dims = self._parse_dimensions(value)

    def _parse_dimensions(self, dims: DimsInput) -> DimsProperty:
        dims = (dims,) if isinstance(dims, str) else tuple(dims)
        if len(dims) != self.ndim:
            raise ValueError(
                f"dimensions {dims} must have the same length as the "
                f"number of data dimensions, ndim={self.ndim}"
            )
        return dims

    @property
    def attrs(self) -> dict[Any, Any]:
        """Dictionary of local attributes on this NamedArray."""
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Any, Any]) -> None:
        self._attrs = dict(value)

    def _check_shape(self, new_data: T_DuckArray) -> None:
        if new_data.shape != self.shape:
            raise ValueError(
                f"replacement data must match the {self.__class__.__name__}'s shape. "
                f"replacement data has shape {new_data.shape}; {self.__class__.__name__} has shape {self.shape}"
            )

    @property
    def data(self) -> T_DuckArray:
        """
        The NamedArray's data as an array. The underlying array type
        (e.g. dask, sparse, pint) is preserved.

        """

        return self._data

    @data.setter
    def data(self, data: T_DuckArray | np.typing.ArrayLike) -> None:
        data = as_compatible_data(data)
        self._check_shape(data)
        self._data = data

    def __dask_tokenize__(self) -> Hashable:
        # Use v.data, instead of v._data, in order to cope with the wrappers
        # around NetCDF and the like
        from dask.base import normalize_token

        s, d, a, attrs = type(self), self._dims, self.data, self.attrs
        return normalize_token((s, d, a, attrs))  # type: ignore[no-any-return]

    def __dask_graph__(self) -> Graph | None:
        if is_duck_dask_array(self._data):
            return self._data.__dask_graph__()
        else:
            # TODO: Should this method just raise instead?
            # raise NotImplementedError("Method requires self.data to be a dask array")
            return None

    def __dask_keys__(self) -> NestedKeys:
        if is_duck_dask_array(self._data):
            return self._data.__dask_keys__()
        else:
            raise AttributeError("Method requires self.data to be a dask array.")

    def __dask_layers__(self) -> Sequence[str]:
        if is_duck_dask_array(self._data):
            return self._data.__dask_layers__()
        else:
            raise AttributeError("Method requires self.data to be a dask array.")

    @property
    def __dask_optimize__(
        self,
    ) -> Callable[..., dict[Any, Any]]:
        if is_duck_dask_array(self._data):
            return self._data.__dask_optimize__  # type: ignore[no-any-return]
        else:
            raise AttributeError("Method requires self.data to be a dask array.")

    @property
    def __dask_scheduler__(self) -> SchedulerGetCallable:
        if is_duck_dask_array(self._data):
            return self._data.__dask_scheduler__
        else:
            raise AttributeError("Method requires self.data to be a dask array.")

    def __dask_postcompute__(
        self,
    ) -> tuple[PostComputeCallable, tuple[Any, ...]]:
        if is_duck_dask_array(self._data):
            array_func, array_args = self._data.__dask_postcompute__()  # type: ignore[no-untyped-call]
            return self._dask_finalize, (array_func,) + array_args
        else:
            raise AttributeError("Method requires self.data to be a dask array.")

    def __dask_postpersist__(
        self,
    ) -> tuple[
        Callable[
            [Graph, PostPersistCallable[Any], Any, Any],
            Self,
        ],
        tuple[Any, ...],
    ]:
        if is_duck_dask_array(self._data):
            a: tuple[PostPersistCallable[Any], tuple[Any, ...]]
            a = self._data.__dask_postpersist__()  # type: ignore[no-untyped-call]
            array_func, array_args = a

            return self._dask_finalize, (array_func,) + array_args
        else:
            raise AttributeError("Method requires self.data to be a dask array.")

    def _dask_finalize(
        self,
        results: Graph,
        array_func: PostPersistCallable[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        data = array_func(results, *args, **kwargs)
        return type(self)(self._dims, data, attrs=self._attrs)

    def load(self, **kwargs):
        """Manually trigger loading of this variable's data from disk or a
        remote source into memory and return this variable.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        if is_chunked_array(self._data):
            chunkmanager = get_chunked_array_type(self._data)
            loaded_data, *_ = chunkmanager.compute(self._data, **kwargs)
            self._data = as_compatible_data(loaded_data)
        elif isinstance(self._data, ExplicitlyIndexed):
            self._data = self._data.get_duck_array()
        elif not is_duck_array(self._data):
            self._data = np.asarray(self._data)
        return self

    def compute(self, **kwargs):
        """Manually trigger loading of this variable's data from disk or a
        remote source into memory and return a new variable. The original is
        left unaltered.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        new = self.copy(deep=False)
        return new.load(**kwargs)

    def get_axis_num(self, dim: Hashable | Iterable[Hashable]) -> int | tuple[int, ...]:
        """Return axis number(s) corresponding to dimension(s) in this array.

        Parameters
        ----------
        dim : str or iterable of str
            Dimension name(s) for which to lookup axes.

        Returns
        -------
        int or tuple of int
            Axis number or numbers corresponding to the given dimensions.
        """
        if not isinstance(dim, str) and isinstance(dim, Iterable):
            return tuple(self._get_axis_num(d) for d in dim)
        else:
            return self._get_axis_num(dim)

    def _get_axis_num(self: Any, dim: Hashable) -> int:
        try:
            return self.dims.index(dim)  # type: ignore[no-any-return]
        except ValueError:
            raise ValueError(f"{dim!r} not found in array dimensions {self.dims!r}")

    @property
    def chunks(self) -> tuple[tuple[int, ...], ...] | None:
        """
        Tuple of block lengths for this NamedArray's data, in order of dimensions, or None if
        the underlying data is not a dask array.

        See Also
        --------
        NamedArray.chunk
        NamedArray.chunksizes
        xarray.unify_chunks
        """
        data = self._data
        if is_chunked_duck_array(data):
            return data.chunks
        else:
            return None

    @property
    def chunksizes(
        self,
    ) -> Mapping[Any, tuple[int, ...]]:
        """
        Mapping from dimension names to block lengths for this namedArray's data, or None if
        the underlying data is not a dask array.
        Cannot be modified directly, but can be modified by calling .chunk().

        Differs from NamedArray.chunks because it returns a mapping of dimensions to chunk shapes
        instead of a tuple of chunk shapes.

        See Also
        --------
        NamedArray.chunk
        NamedArray.chunks
        xarray.unify_chunks
        """
        data = self._data
        if is_chunked_duck_array(data):
            return dict(zip(self.dims, data.chunks))
        else:
            return {}

    @property
    def sizes(self) -> dict[Hashable, int]:
        """Ordered mapping from dimension names to lengths."""
        return dict(zip(self.dims, self.shape))

    def _replace(
        self,
        dims: DimsInput | Default = _default,
        data: T_DuckArray | np.typing.ArrayLike | Default = _default,
        attrs: AttrsInput | Default = _default,
    ) -> Self:
        if dims is _default:
            dims = copy.copy(self._dims)
        if data is _default:
            data = copy.copy(self._data)
        if attrs is _default:
            attrs = copy.copy(self._attrs)
        return type(self)(dims, data, attrs)

    def _copy(
        self,
        deep: bool = True,
        data: T_DuckArray | np.typing.ArrayLike | None = None,
        memo: dict[int, Any] | None = None,
    ) -> Self:
        if data is None:
            ndata = self._data
            if deep:
                ndata = copy.deepcopy(ndata, memo=memo)
        else:
            ndata = as_compatible_data(data)
            self._check_shape(ndata)

        attrs = (
            copy.deepcopy(self._attrs, memo=memo) if deep else copy.copy(self._attrs)
        )

        return self._replace(data=ndata, attrs=attrs)

    def __copy__(self) -> Self:
        return self._copy(deep=False)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self._copy(deep=True, memo=memo)

    def copy(
        self,
        deep: bool = True,
        data: T_DuckArray | np.typing.ArrayLike | None = None,
    ) -> Self:
        """Returns a copy of this object.

        If `deep=True`, the data array is loaded into memory and copied onto
        the new object. Dimensions, attributes and encodings are always copied.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, default: True
            Whether the data array is loaded into memory and copied onto
            the new object. Default is True.
        data : array_like, optional
            Data to use in the new object. Must have same shape as original.
            When `data` is used, `deep` is ignored.

        Returns
        -------
        object : NamedArray
            New object with dimensions, attributes, and optionally
            data copied from original.


        """
        return self._copy(deep=deep, data=data)

    def chunk(
        self,
        chunks: (
            int
            | Literal["auto"]
            | tuple[int, ...]
            | tuple[tuple[int, ...], ...]
            | Mapping[Any, None | int | tuple[int, ...]]
        ) = {},
        name: str | None = None,
        lock: bool | None = None,
        inline_array: bool | None = None,
        chunked_array_type: str | ChunkManagerEntrypoint | None = None,
        from_array_kwargs=None,
        **chunks_kwargs: Any,
    ) -> Self:
        """Coerce this array's data into a dask array with the given chunks.

        If this variable is a non-dask array, it will be converted to dask
        array. If it's a dask array, it will be rechunked to the given chunk
        sizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

        Parameters
        ----------
        chunks : int, tuple or dict, optional
            Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
            ``{'x': 5, 'y': 5}``.
        name : str, optional
            Used to generate the name for this array in the internal dask
            graph. Does not need not be unique.
        lock : bool, default: False
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.
        inline_array : bool, default: False
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.
        chunked_array_type: str, optional
            Which chunked array type to coerce this datasets' arrays to.
            Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEntrypoint` system.
            Experimental API that should not be relied upon.
        from_array_kwargs: dict, optional
            Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
            chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
            For example, with dask as the default chunked array type, this method would pass additional kwargs
            to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
        **chunks_kwargs : {dim: chunks, ...}, optional
            The keyword arguments form of ``chunks``.
            One of chunks or chunks_kwargs must be provided.

        Returns
        -------
        chunked : xarray.Variable

        See Also
        --------
        Variable.chunks
        Variable.chunksizes
        xarray.unify_chunks
        dask.array.from_array
        """

        if chunks is None:
            warnings.warn(
                "None value for 'chunks' is deprecated. "
                "It will raise an error in the future. Use instead '{}'",
                category=FutureWarning,
            )
            chunks = {}

        if isinstance(chunks, (float, str, int, tuple, list)):
            # TODO we shouldn't assume here that other chunkmanagers can handle these types
            # TODO should we call normalize_chunks here?
            pass  # dask.array.from_array can handle these directly
        else:
            chunks = either_dict_or_kwargs(chunks, chunks_kwargs, "chunk")

        if is_dict_like(chunks):
            chunks = {self.get_axis_num(dim): chunk for dim, chunk in chunks.items()}

        chunkmanager = guess_chunkmanager(chunked_array_type)

        if from_array_kwargs is None:
            from_array_kwargs = {}

        # TODO deprecate passing these dask-specific arguments explicitly. In future just pass everything via from_array_kwargs
        _from_array_kwargs = consolidate_dask_from_array_kwargs(
            from_array_kwargs,
            name=name,
            lock=lock,
            inline_array=inline_array,
        )

        data_old = self._data
        if chunkmanager.is_chunked_array(data_old):
            data_chunked = chunkmanager.rechunk(data_old, chunks)
        else:
            if not isinstance(data_old, ExplicitlyIndexed):
                ndata = data_old
            else:
                # Unambiguously handle array storage backends (like NetCDF4 and h5py)
                # that can't handle general array indexing. For example, in netCDF4 you
                # can do "outer" indexing along two dimensions independent, which works
                # differently from how NumPy handles it.
                # da.from_array works by using lazy indexing with a tuple of slices.
                # Using OuterIndexer is a pragmatic choice: dask does not yet handle
                # different indexing types in an explicit way:
                # https://github.com/dask/dask/issues/2883
                ndata = ImplicitToExplicitIndexingAdapter(data_old, OuterIndexer)

            if is_dict_like(chunks):
                chunks = tuple(chunks.get(n, s) for n, s in enumerate(ndata.shape))

            data_chunked = chunkmanager.from_array(
                ndata,
                chunks,
                **_from_array_kwargs,
            )

        return self._replace(data=data_chunked)

    def to_numpy(self) -> np.ndarray:
        """Coerces wrapped data to numpy and returns a numpy.ndarray"""
        # TODO an entrypoint so array libraries can choose coercion method?
        data = self.data

        # TODO first attempt to call .to_numpy() once some libraries implement it
        if hasattr(data, "chunks"):
            chunkmanager = get_chunked_array_type(data)
            data, *_ = chunkmanager.compute(data)
        if isinstance(data, array_type("cupy")):
            data = data.get()
        # pint has to be imported dynamically as pint imports xarray
        if isinstance(data, array_type("pint")):
            data = data.magnitude
        if isinstance(data, array_type("sparse")):
            data = data.todense()
        data = np.asarray(data)

        return data

    def as_numpy(self) -> Self:
        """Coerces wrapped data into a numpy array, returning a Variable."""
        return self._replace(data=self.to_numpy())

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Dims = None,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : "...", str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`. By default `func` is
            applied over all dimensions.
        axis : int or Sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dim'
            and 'axis' arguments can be supplied. If neither are supplied, then
            the reduction is calculated over the flattened array (by calling
            `func(x)` without an axis argument).
        keepdims : bool, default: False
            If True, the dimensions which are reduced are left in the result
            as dimensions of size one
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim == ...:
            dim = None
        if dim is not None and axis is not None:
            raise ValueError("cannot supply both 'axis' and 'dim' arguments")

        if dim is not None:
            axis = self.get_axis_num(dim)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"Mean of empty slice", category=RuntimeWarning
            )
            if axis is not None:
                if isinstance(axis, tuple) and len(axis) == 1:
                    # unpack axis for the benefit of functions
                    # like np.argmin which can't handle tuple arguments
                    axis = axis[0]
                data = func(self.data, axis=axis, **kwargs)
            else:
                data = func(self.data, **kwargs)

        if getattr(data, "shape", ()) == self.shape:
            dims = self.dims
        else:
            removed_axes: Iterable[int]
            if axis is None:
                removed_axes = range(self.ndim)
            else:
                removed_axes = np.atleast_1d(axis) % self.ndim
            if keepdims:
                # Insert np.newaxis for removed dims
                slices = tuple(
                    np.newaxis if i in removed_axes else slice(None, None)
                    for i in range(self.ndim)
                )
                if getattr(data, "shape", None) is None:
                    # Reduce has produced a scalar value, not an array-like
                    data = np.asanyarray(data)[slices]
                else:
                    data = data[slices]
                dims = self.dims
            else:
                dims = tuple(
                    adim for n, adim in enumerate(self.dims) if n not in removed_axes
                )

        # Return NamedArray to handle IndexVariable when data is nD
        return NamedArray(dims, data, attrs=self._attrs)

    def _nonzero(self) -> tuple[Self, ...]:
        """Equivalent numpy's nonzero but returns a tuple of NamedArrays."""
        # TODO we should replace dask's native nonzero
        # after https://github.com/dask/dask/issues/1076 is implemented.
        nonzeros = np.nonzero(self.data)
        return tuple(type(self)((dim,), nz) for nz, dim in zip(nonzeros, self.dims))

    def __repr__(self) -> str:
        return formatting.array_repr(self)

    def _repr_html_(self) -> str:
        return formatting_html.array_repr(self)

    def _as_sparse(
        self,
        sparse_format: str | Default = _default,
        fill_value: np.typing.ArrayLike | Default = _default,
    ) -> Self:
        """
        use sparse-array as backend.
        """
        import sparse

        # TODO: what to do if dask-backended?
        if fill_value is _default:
            dtype, fill_value = dtypes.maybe_promote(self.dtype)
        else:
            dtype = dtypes.result_type(self.dtype, fill_value)

        if sparse_format is _default:
            sparse_format = "coo"
        try:
            as_sparse = getattr(sparse, f"as_{sparse_format.lower()}")
        except AttributeError as exc:
            raise ValueError(f"{sparse_format} is not a valid sparse format") from exc

        data = as_sparse(astype(self.data, dtype), fill_value=fill_value)
        return self._replace(data=data)

    def _to_dense(self) -> Self:
        """
        Change backend from sparse to np.array
        """
        if hasattr(self._data, "todense"):
            return self._replace(data=self._data.todense())
        return self.copy(deep=False)

    def _nonzero(self) -> tuple[Self, ...]:
        """Equivalent to numpy's nonzero but returns a tuple of NamedArrays."""
        # TODO we should replace dask's native nonzero
        # after https://github.com/dask/dask/issues/1076 is implemented.
        nonzeros = np.nonzero(self.data)
        return tuple(type(self)((dim,), nz) for nz, dim in zip(nonzeros, self.dims))
