from __future__ import annotations

import copy
import math
import typing
from collections.abc import Hashable, Iterable, Mapping

import numpy as np

# TODO: get rid of this after migrating this class to array API
from xarray.core import dtypes
from xarray.core.indexing import ExplicitlyIndexed
from xarray.core.utils import Default, _default
from xarray.namedarray.utils import (
    T_DuckArray,
    is_duck_array,
    is_duck_dask_array,
    to_0d_object_array,
)

if typing.TYPE_CHECKING:
    from xarray.namedarray.utils import Self

    # T_NamedArray = typing.TypeVar("T_NamedArray", bound="NamedArray")
    DimsInput = typing.Union[str, Iterable[Hashable]]
    Dims = tuple[Hashable, ...]


# TODO: Add tests!
def as_compatible_data(
    data: T_DuckArray | np.typing.ArrayLike, fastpath: bool = False
) -> T_DuckArray:
    if fastpath and getattr(data, "ndim", 0) > 0:
        # can't use fastpath (yet) for scalars
        return typing.cast(T_DuckArray, data)

    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)  # type: ignore[no-untyped-call]
        if mask.any():
            # TODO: requires refactoring/vendoring xarray.core.dtypes and xarray.core.duck_array_ops
            raise NotImplementedError("MaskedArray is not supported yet")
        else:
            return typing.cast(T_DuckArray, np.asarray(data))
    if is_duck_array(data):
        return data
    if isinstance(data, NamedArray):
        return typing.cast(T_DuckArray, data.data)

    if isinstance(data, ExplicitlyIndexed):
        # TODO: better that is_duck_array(ExplicitlyIndexed) -> True
        return typing.cast(T_DuckArray, data)

    if isinstance(data, tuple):
        data = to_0d_object_array(data)

    # validate whether the data is valid data types.
    return typing.cast(T_DuckArray, np.asarray(data))


class NamedArray(typing.Generic[T_DuckArray]):

    """A lightweight wrapper around duck arrays with named dimensions and attributes which describe a single Array.
    Numeric operations on this object implement array broadcasting and dimension alignment based on dimension names,
    rather than axis order."""

    __slots__ = ("_data", "_dims", "_attrs")

    _data: T_DuckArray
    _dims: Dims
    _attrs: dict[typing.Any, typing.Any] | None

    def __init__(
        self,
        dims: DimsInput,
        data: T_DuckArray | np.typing.ArrayLike,
        attrs: dict[typing.Any, typing.Any] | None = None,
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
    def dtype(self) -> np.dtype[typing.Any]:
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
            return self._data.nbytes
        else:
            return self.size * self.dtype.itemsize

    @property
    def dims(self) -> Dims:
        """Tuple of dimension names with which this NamedArray is associated."""
        return self._dims

    @dims.setter
    def dims(self, value: DimsInput) -> None:
        self._dims = self._parse_dimensions(value)

    def _parse_dimensions(self, dims: DimsInput) -> Dims:
        dims = (dims,) if isinstance(dims, str) else tuple(dims)
        if len(dims) != self.ndim:
            raise ValueError(
                f"dimensions {dims} must have the same length as the "
                f"number of data dimensions, ndim={self.ndim}"
            )
        return dims

    @property
    def attrs(self) -> dict[typing.Any, typing.Any]:
        """Dictionary of local attributes on this NamedArray."""
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[typing.Any, typing.Any]) -> None:
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

    @property
    def real(self) -> Self:
        """
        The real part of the NamedArray.

        See Also
        --------
        numpy.ndarray.real
        """
        return self._replace(data=self.data.real)

    @property
    def imag(self) -> Self:
        """
        The imaginary part of the NamedArray.

        See Also
        --------
        numpy.ndarray.imag
        """
        return self._replace(data=self.data.imag)

    def __dask_tokenize__(self):
        # Use v.data, instead of v._data, in order to cope with the wrappers
        # around NetCDF and the like
        from dask.base import normalize_token

        return normalize_token((type(self), self._dims, self.data, self.attrs))

    def __dask_graph__(self):
        return self._data.__dask_graph__() if is_duck_dask_array(self._data) else None

    def __dask_keys__(self):
        return self._data.__dask_keys__()

    def __dask_layers__(self):
        return self._data.__dask_layers__()

    @property
    def __dask_optimize__(self) -> typing.Callable:
        return self._data.__dask_optimize__

    @property
    def __dask_scheduler__(self) -> typing.Callable:
        return self._data.__dask_scheduler__

    def __dask_postcompute__(
        self,
    ) -> tuple[typing.Callable, tuple[typing.Any, ...]]:
        array_func, array_args = self._data.__dask_postcompute__()
        return self._dask_finalize, (array_func,) + array_args

    def __dask_postpersist__(
        self,
    ) -> tuple[typing.Callable, tuple[typing.Any, ...]]:
        array_func, array_args = self._data.__dask_postpersist__()
        return self._dask_finalize, (array_func,) + array_args

    def _dask_finalize(self, results, array_func, *args, **kwargs) -> Self:
        data = array_func(results, *args, **kwargs)
        return type(self)(self._dims, data, attrs=self._attrs)

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
        return getattr(self._data, "chunks", None)

    @property
    def chunksizes(
        self,
    ) -> typing.Mapping[typing.Any, tuple[int, ...]]:
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
        if hasattr(self._data, "chunks"):
            return dict(zip(self.dims, self.data.chunks))
        else:
            return {}

    @property
    def sizes(self) -> dict[Hashable, int]:
        """Ordered mapping from dimension names to lengths."""
        return dict(zip(self.dims, self.shape))

    def _replace(self, dims=_default, data=_default, attrs=_default) -> Self:
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
        memo: dict[int, typing.Any] | None = None,
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

    def __deepcopy__(self, memo: dict[int, typing.Any] | None = None) -> Self:
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

    def _nonzero(self) -> tuple[Self, ...]:
        """Equivalent numpy's nonzero but returns a tuple of NamedArrays."""
        # TODO we should replace dask's native nonzero
        # after https://github.com/dask/dask/issues/1076 is implemented.
        nonzeros = np.nonzero(self.data)
        return tuple(type(self)((dim,), nz) for nz, dim in zip(nonzeros, self.dims))

    def _as_sparse(
        self,
        sparse_format: str | Default = _default,
        fill_value=dtypes.NA,
    ) -> Self:
        """
        use sparse-array as backend.
        """
        import sparse

        # TODO: what to do if dask-backended?
        if fill_value is dtypes.NA:
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

    def _to_dense(self) -> Self:
        """
        Change backend from sparse to np.array
        """
        if hasattr(self._data, "todense"):
            return self._replace(data=self._data.todense())
        return self.copy(deep=False)
