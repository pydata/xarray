from __future__ import annotations

import copy
import math
from collections.abc import Hashable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)

import numpy as np

# TODO: get rid of this after migrating this class to array API
from xarray.core import dtypes
from xarray.namedarray.utils import (
    Default,
    T_DuckArray,
    _array,
    _AttrsLike,
    _Chunks,
    _default,
    _Dim,
    _Dims,
    _DimsLike,
    _IntOrUnknown,
    _Shape,
    _sparsearray,
    is_chunked_duck_array,
    is_duck_dask_array,
    to_0d_object_array,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from xarray.namedarray.utils import (
        Self,  # type: ignore[attr-defined]
        _Array,
    )

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

    T_NamedArray = TypeVar("T_NamedArray", bound="NamedArray")


def _replace_with_new_data_type(
    obj: T_NamedArray,
    dims: _DimsLike | Default = _default,
    data: _Array | Default = _default,
    attrs: _AttrsLike | Default = _default,
) -> T_NamedArray:
    if dims is _default:
        dims = copy.copy(obj._dims)
    if data is _default:
        data_ = copy.copy(obj._data)
    else:
        data_ = data
    if attrs is _default:
        attrs = copy.copy(obj._attrs)
    return type(obj)(dims, data_, attrs)


@overload
def from_array(
    dims: _DimsLike,
    data: T_DuckArray,
    attrs: _AttrsLike = None,
) -> NamedArray[T_DuckArray]:
    ...


@overload
def from_array(
    dims: _DimsLike,
    data: ArrayLike,
    attrs: _AttrsLike = None,
) -> NamedArray[NDArray[Any]]:
    ...


def from_array(
    dims: _DimsLike,
    data: T_DuckArray | ArrayLike,
    attrs: _AttrsLike = None,
) -> NamedArray[T_DuckArray] | NamedArray[NDArray[Any]]:
    """
    Create a Named array from an array-like object.

    Parameters
    ----------
    dims : str or iterable of str
        Name(s) of the dimension(s).
    data : T_DuckArray or ArrayLike
        The actual data that populates the array. Should match the
        shape specified by `dims`.
    attrs : dict, optional
        A dictionary containing any additional information or
        attributes you want to store with the array.
        Default is None, meaning no attributes will be stored.
    """
    if isinstance(data, NamedArray):
        raise ValueError(
            "Array is already a Named array. Use 'data.data' to retrieve the data array"
        )

    # TODO: dask.array.ma.masked_array also exists, better way?
    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)  # type: ignore[no-untyped-call]
        if mask.any():
            # TODO: requires refactoring/vendoring xarray.core.dtypes and
            # xarray.core.duck_array_ops
            raise NotImplementedError("MaskedArray is not supported yet")
        # TODO: cast is a mypy workaround for https://github.com/python/mypy/issues/10817
        # pyright doesn't need it.
        data_ = cast(T_DuckArray, data)
        return NamedArray(dims, data_, attrs)

    if isinstance(data, _array):
        # TODO: cast is a mypy workaround for https://github.com/python/mypy/issues/10817
        # pyright doesn't need it.
        data_ = cast(T_DuckArray, data)
        return NamedArray(dims, data_, attrs)
    else:
        if isinstance(data, tuple):
            return NamedArray(dims, to_0d_object_array(data), attrs)
        else:
            # validate whether the data is valid data types.
            return NamedArray(dims, np.asarray(data), attrs)


class NamedArray(Generic[T_DuckArray]):
    """
    A wrapper around duck arrays with named dimensions
    and attributes which describe a single Array.
    Numeric operations on this object implement array broadcasting and
    dimension alignment based on dimension names,
    rather than axis order.


    Parameters
    ----------
    dims : str or iterable of str
        Name(s) of the dimension(s).
    data : T_DuckArray
        The actual data that populates the array. Should match the
        shape specified by `dims`.
    attrs : dict, optional
        A dictionary containing any additional information or
        attributes you want to store with the array.
        Default is None, meaning no attributes will be stored.

    Raises
    ------
    ValueError
        If the `dims` length does not match the number of data dimensions (ndim).


    Examples
    --------
    >>> data = np.array([1.5, 2, 3], dtype=float)
    >>> narr = NamedArray(("x",), data, {"units": "m"})  # TODO: Better name than narr?
    """

    __slots__ = ("_data", "_dims", "_attrs")

    _data: T_DuckArray
    _dims: _Dims
    _attrs: dict[Any, Any] | None

    def __init__(
        self,
        dims: _DimsLike,
        data: T_DuckArray,
        attrs: _AttrsLike = None,
    ):
        self._data = data
        self._dims = self._parse_dimensions(dims)
        self._attrs = dict(attrs) if attrs else None

    @property
    def ndim(self) -> _IntOrUnknown:
        """
        Number of array dimensions.

        See Also
        --------
        numpy.ndarray.ndim
        """
        return len(self.shape)

    @property
    def size(self) -> _IntOrUnknown:
        """
        Number of elements in the array.

        Equal to ``np.prod(a.shape)``, i.e., the product of the array’s dimensions.

        See Also
        --------
        numpy.ndarray.size
        """
        return math.prod(self.shape)

    def __len__(self) -> _IntOrUnknown:
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
    def shape(self) -> _Shape:
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
    def nbytes(self) -> _IntOrUnknown:
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
    def dims(self) -> _Dims:
        """Tuple of dimension names with which this NamedArray is associated."""
        return self._dims

    @dims.setter
    def dims(self, value: _DimsLike) -> None:
        self._dims = self._parse_dimensions(value)

    def _parse_dimensions(self, dims: _DimsLike) -> _Dims:
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
    def data(self, data: T_DuckArray) -> None:
        self._check_shape(data)
        self._data = data

    @property
    def real(self) -> NamedArray:
        """
        The real part of the NamedArray.

        See Also
        --------
        numpy.ndarray.real
        """
        # return self._replace(data=self.data.real)
        return _replace_with_new_data_type(self, data=self.data.real)

    @property
    def imag(self) -> NamedArray:
        """
        The imaginary part of the NamedArray.

        See Also
        --------
        numpy.ndarray.imag
        """
        # return self._replace(data=self.data.imag)
        return _replace_with_new_data_type(self, data=self.data.imag)

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

    @property
    def chunks(self) -> _Chunks | None:
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
    ) -> Mapping[_Dim, _Shape]:
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
    def sizes(self) -> dict[_Dim, _IntOrUnknown]:
        """Ordered mapping from dimension names to lengths."""
        return dict(zip(self.dims, self.shape))

    def _replace(
        self,
        dims: _DimsLike | Default = _default,
        data: T_DuckArray | Default = _default,
        attrs: _AttrsLike | Default = _default,
    ) -> Self:
        """
        Create a new Named array with dims, data or attrs.

        The types for each argument cannot change,
        use _replace_with_new_data_type if that is a risk.
        """
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
        data: T_DuckArray | None = None,
        memo: dict[int, Any] | None = None,
    ) -> Self:
        if data is None:
            ndata = self._data
            if deep:
                ndata = copy.deepcopy(ndata, memo=memo)
        else:
            ndata = data
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
        data: T_DuckArray | None = None,
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

    def _nonzero(self) -> tuple[Self, ...]:
        """Equivalent numpy's nonzero but returns a tuple of NamedArrays."""
        # TODO: we should replace dask's native nonzero
        # after https://github.com/dask/dask/issues/1076 is implemented.
        # TODO: cast to ndarray and back to T_DuckArray is a workaround
        nonzeros = np.nonzero(cast("NDArray[np.generic]", self.data))
        return tuple(
            type(self)((dim,), cast(T_DuckArray, nz))
            for nz, dim in zip(nonzeros, self.dims)
        )

    def _as_sparse(
        self,
        sparse_format: Literal["coo"] | Default = _default,
        fill_value: ArrayLike | Default = _default,
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

        data = as_sparse(self.data.astype(dtype), fill_value=fill_value)
        return self._replace(data=data)

    def _to_dense(self) -> NamedArray:
        """
        Change backend from sparse to np.array
        """
        if isinstance(self._data, _sparsearray):
            # return self._replace(data=self._data.todense())
            return _replace_with_new_data_type(self, data=self._data.todense())
        else:
            raise TypeError("self.data is not a sparse array")
