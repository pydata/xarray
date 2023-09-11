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
    Frozen,
    is_duck_array,
    is_duck_dask_array,
    to_0d_object_array,
)

if typing.TYPE_CHECKING:
    from xarray.namedarray.utils import T_DuckArray

    T_NamedArray = typing.TypeVar("T_NamedArray", bound="NamedArray")


# TODO: Add tests!
def as_compatible_data(
    data: T_DuckArray | np.typing.ArrayLike, fastpath: bool = False
) -> T_DuckArray:
    if fastpath and getattr(data, "ndim", 0) > 0:
        # can't use fastpath (yet) for scalars
        return typing.cast(T_DuckArray, data)

    # TODO : check scalar
    if is_duck_array(data):
        return data
    if isinstance(data, NamedArray):
        raise ValueError
    if isinstance(data, np.ma.MaskedArray):
        raise ValueError
    if isinstance(data, ExplicitlyIndexed):
        # TODO: better that is_duck_array(ExplicitlyIndexed) -> True
        return typing.cast(T_DuckArray, data)

    if not isinstance(data, np.ndarray) and (
        hasattr(data, "__array_function__") or hasattr(data, "__array_namespace__")
    ):
        return typing.cast(T_DuckArray, data)
    if isinstance(data, tuple):
        data = to_0d_object_array(data)

    # validate whether the data is valid data types.
    return typing.cast(T_DuckArray, np.asarray(data))


class NamedArray:
    __slots__ = ("_dims", "_data", "_attrs")

    def __init__(
        self,
        dims: str | Iterable[Hashable],
        data: T_DuckArray | np.typing.ArrayLike,
        attrs: dict | None = None,
    ):
        self._data: T_DuckArray = as_compatible_data(data)
        self._dims: tuple[Hashable, ...] = self._parse_dimensions(dims)
        self._attrs: dict | None = dict(attrs) if attrs else None

    @property
    def ndim(self: T_NamedArray) -> int:
        """
        Number of array dimensions.

        See Also
        --------
        numpy.ndarray.ndim
        """
        return len(self.shape)

    @property
    def size(self: T_NamedArray) -> int:
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
    def dtype(self: T_NamedArray) -> np.dtype:
        """
        Data-type of the array’s elements.

        See Also
        --------
        ndarray.dtype
        numpy.dtype
        """
        return self._data.dtype

    @property
    def shape(self: T_NamedArray) -> tuple[int, ...]:
        """
        Tuple of array dimensions.

        See Also
        --------
        numpy.ndarray.shape
        """
        return self._data.shape

    @property
    def nbytes(self: T_NamedArray) -> int:
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
    def dims(self: T_NamedArray) -> tuple[Hashable, ...]:
        """Tuple of dimension names with which this variable is associated."""
        return self._dims

    @dims.setter
    def dims(self: T_NamedArray, value: str | Iterable[Hashable]) -> None:
        self._dims = self._parse_dimensions(value)

    def _parse_dimensions(
        self: T_NamedArray, dims: str | Iterable[Hashable]
    ) -> tuple[Hashable, ...]:
        dims = (dims,) if isinstance(dims, str) else tuple(dims)
        if len(dims) != self.ndim:
            raise ValueError(
                f"dimensions {dims} must have the same length as the "
                f"number of data dimensions, ndim={self.ndim}"
            )
        return dims

    @property
    def attrs(self: T_NamedArray) -> dict[typing.Any, typing.Any]:
        """Dictionary of local attributes on this variable."""
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self: T_NamedArray, value: Mapping) -> None:
        self._attrs = dict(value)

    def _check_shape(self, new_data: T_DuckArray) -> None:
        if new_data.shape != self.shape:
            raise ValueError(
                f"replacement data must match the Variable's shape. "
                f"replacement data has shape {new_data.shape}; Variable has shape {self.shape}"
            )

    @property
    def data(self: T_NamedArray):
        """
        The Variable's data as an array. The underlying array type
        (e.g. dask, sparse, pint) is preserved.

        See Also
        --------
        Variable.to_numpy
        Variable.as_numpy
        Variable.values
        """

        return self._data

    @data.setter
    def data(self: T_NamedArray, data: T_DuckArray | np.typing.ArrayLike) -> None:
        data = as_compatible_data(data)
        self._check_shape(data)
        self._data = data

    @property
    def real(self: T_NamedArray) -> T_NamedArray:
        """
        The real part of the variable.

        See Also
        --------
        numpy.ndarray.real
        """
        return self._replace(data=self.data.real)

    @property
    def imag(self: T_NamedArray) -> T_NamedArray:
        """
        The imaginary part of the variable.

        See Also
        --------
        numpy.ndarray.imag
        """
        return self._replace(data=self.data.imag)

    def __dask_tokenize__(self: T_NamedArray):
        # Use v.data, instead of v._data, in order to cope with the wrappers
        # around NetCDF and the like
        from dask.base import normalize_token

        return normalize_token((type(self), self._dims, self.data, self.attrs))

    def __dask_graph__(self: T_NamedArray):
        return self._data.__dask_graph__() if is_duck_dask_array(self._data) else None

    def __dask_keys__(self: T_NamedArray):
        return self._data.__dask_keys__()

    def __dask_layers__(self: T_NamedArray):
        return self._data.__dask_layers__()

    @property
    def __dask_optimize__(self: T_NamedArray) -> typing.Callable:
        return self._data.__dask_optimize__

    @property
    def __dask_scheduler__(self: T_NamedArray) -> typing.Callable:
        return self._data.__dask_scheduler__

    def __dask_postcompute__(
        self: T_NamedArray,
    ) -> tuple[typing.Callable, tuple[typing.Any, ...]]:
        array_func, array_args = self._data.__dask_postcompute__()
        return self._dask_finalize, (array_func,) + array_args

    def __dask_postpersist__(
        self: T_NamedArray,
    ) -> tuple[typing.Callable, tuple[typing.Any, ...]]:
        array_func, array_args = self._data.__dask_postpersist__()
        return self._dask_finalize, (array_func,) + array_args

    def _dask_finalize(
        self: T_NamedArray, results, array_func, *args, **kwargs
    ) -> T_NamedArray:
        data = array_func(results, *args, **kwargs)
        return type(self)(self._dims, data, attrs=self._attrs)

    @property
    def chunks(self: T_NamedArray) -> tuple[tuple[int, ...], ...] | None:
        """
        Tuple of block lengths for this dataarray's data, in order of dimensions, or None if
        the underlying data is not a dask array.

        See Also
        --------
        Variable.chunk
        Variable.chunksizes
        xarray.unify_chunks
        """
        return getattr(self._data, "chunks", None)

    @property
    def chunksizes(
        self: T_NamedArray,
    ) -> typing.Mapping[typing.Any, tuple[int, ...]]:
        """
        Mapping from dimension names to block lengths for this variable's data, or None if
        the underlying data is not a dask array.
        Cannot be modified directly, but can be modified by calling .chunk().

        Differs from variable.chunks because it returns a mapping of dimensions to chunk shapes
        instead of a tuple of chunk shapes.

        See Also
        --------
        Variable.chunk
        Variable.chunks
        xarray.unify_chunks
        """
        if hasattr(self._data, "chunks"):
            return Frozen(dict(zip(self.dims, self.data.chunks)))
        else:
            return {}

    def _replace(
        self: T_NamedArray, dims=_default, data=_default, attrs=_default
    ) -> T_NamedArray:
        if dims is _default:
            dims = copy.copy(self._dims)
        if data is _default:
            data = copy.copy(self._data)
        if attrs is _default:
            attrs = copy.copy(self._attrs)
        return type(self)(dims, data, attrs)

    def _copy(
        self: T_NamedArray,
        deep: bool = True,
        data: T_DuckArray | np.typing.ArrayLike | None = None,
        memo: dict[int, typing.Any] | None = None,
    ) -> T_NamedArray:
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

    def __copy__(self: T_NamedArray) -> T_NamedArray:
        return self._copy(deep=False)

    def __deepcopy__(
        self: T_NamedArray, memo: dict[int, typing.Any] | None = None
    ) -> T_NamedArray:
        return self._copy(deep=True, memo=memo)

    def copy(
        self: T_NamedArray,
        deep: bool = True,
        data: T_DuckArray | np.typing.ArrayLike | None = None,
    ) -> T_NamedArray:
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
        object : Variable
            New object with dimensions, attributes, encodings, and optionally
            data copied from original.

        Examples
        --------
        Shallow copy versus deep copy

        >>> var = xr.Variable(data=[1, 2, 3], dims="x")
        >>> var.copy()
        <xarray.Variable (x: 3)>
        array([1, 2, 3])
        >>> var_0 = var.copy(deep=False)
        >>> var_0[0] = 7
        >>> var_0
        <xarray.Variable (x: 3)>
        array([7, 2, 3])
        >>> var
        <xarray.Variable (x: 3)>
        array([7, 2, 3])

        Changing the data using the ``data`` argument maintains the
        structure of the original object, but with the new data. Original
        object is unaffected.

        >>> var.copy(data=[0.1, 0.2, 0.3])
        <xarray.Variable (x: 3)>
        array([0.1, 0.2, 0.3])
        >>> var
        <xarray.Variable (x: 3)>
        array([7, 2, 3])

        See Also
        --------
        pandas.DataFrame.copy
        """
        return self._copy(deep=deep, data=data)

    def _nonzero(self: T_NamedArray) -> tuple[T_NamedArray, ...]:
        """Equivalent numpy's nonzero but returns a tuple of Variables."""
        # TODO we should replace dask's native nonzero
        # after https://github.com/dask/dask/issues/1076 is implemented.
        nonzeros = np.nonzero(self.data)
        return tuple(type(self)((dim), nz) for nz, dim in zip(nonzeros, self.dims))

    def _as_sparse(
        self: T_NamedArray,
        sparse_format: str | Default = _default,
        fill_value=dtypes.NA,
    ) -> T_NamedArray:
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

    def _to_dense(self: T_NamedArray) -> T_NamedArray:
        """
        Change backend from sparse to np.array
        """
        if hasattr(self._data, "todense"):
            return self._replace(data=self._data.todense())
        return self.copy(deep=False)
