import copy
import itertools
import typing

import numpy as np
import numpy.typing as npt

from xarray.core.pycompat import is_duck_dask_array
from xarray.core.utils import Frozen, _default
from xarray.named_array.utils import NdimSizeLenMixin


class NamedArray(NdimSizeLenMixin):
    __slots__ = ("_dims", "_data", "_attrs")

    def __init__(
        self, dims, data: npt.ArrayLike, attrs: dict[typing.Any, typing.Any] = None
    ):
        self._dims = self._parse_dimensions(dims)
        self._data = data
        self._attrs = attrs or {}

    @property
    def dtype(self) -> np.dtype:
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
    def dims(self) -> tuple[typing.Hashable, ...]:
        """Tuple of dimension names with which this variable is associated."""
        return self._dims

    @dims.setter
    def dims(self, value: str | typing.Iterable[typing.Hashable]) -> None:
        self._dims = self._parse_dimensions(value)

    def _parse_dimensions(
        self, dims: str | typing.Iterable[typing.Hashable]
    ) -> tuple[typing.Hashable, ...]:
        dims = (dims,) if isinstance(dims, str) else tuple(dims)
        if len(dims) != self.ndim:
            raise ValueError(
                f"dimensions {dims} must have the same length as the "
                f"number of data dimensions, ndim={self.ndim}"
            )
        return dims

    @property
    def attrs(self) -> dict[typing.Any, typing.Any]:
        """Dictionary of local attributes on this variable."""
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: typing.Mapping[typing.Any, typing.Any]) -> None:
        self._attrs = dict(value)

    @property
    def data(self) -> typing.Any:
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
    def data(self, data):
        if data.shape != self.shape:
            raise ValueError(
                f"replacement data must match the Variable's shape. "
                f"replacement data has shape {data.shape}; Variable has shape {self.shape}"
            )
        self._data = data

    @property
    def real(self):
        """
        The real part of the variable.

        See Also
        --------
        numpy.ndarray.real
        """
        return self._replace(data=self.data.real)

    @property
    def imag(self):
        """
        The imaginary part of the variable.

        See Also
        --------
        numpy.ndarray.imag
        """
        return self._replace(data=self.data.imag)

    def __dask_tokenize__(self):
        # Use v.data, instead of v._data, in order to cope with the wrappers
        # around NetCDF and the like
        from dask.base import normalize_token

        return normalize_token((type(self), self._dims, self.data, self._attrs))

    def __dask_graph__(self):
        return self._data.__dask_graph__() if is_duck_dask_array(self._data) else None

    def __dask_keys__(self):
        return self._data.__dask_keys__()

    def __dask_layers__(self):
        return self._data.__dask_layers__()

    @property
    def __dask_optimize__(self):
        return self._data.__dask_optimize__

    @property
    def __dask_scheduler__(self):
        return self._data.__dask_scheduler__

    def __dask_postcompute__(self):
        array_func, array_args = self._data.__dask_postcompute__()
        return self._dask_finalize, (array_func,) + array_args

    def __dask_postpersist__(self):
        array_func, array_args = self._data.__dask_postpersist__()
        return self._dask_finalize, (array_func,) + array_args

    def _dask_finalize(self, results, array_func, *args, **kwargs):
        data = array_func(results, *args, **kwargs)
        return NamedArray(self._dims, data, attrs=self._attrs)

    @property
    def chunks(self) -> tuple[tuple[int, ...], ...] | None:
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
    def chunksizes(self) -> typing.Mapping[typing.Any, tuple[int, ...]]:
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

    def _replace(self, dims=_default, data=_default, attrs=_default):
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
        data: npt.ArrayLike | None = None,
        memo: dict[int, typing.Any] | None = None,
    ):
        if data is None:
            ndata = self._data

            if deep:
                ndata = copy.deepcopy(ndata, memo=memo)
        else:
            ndata = data
            if self.shape != ndata.shape:
                raise ValueError(
                    f"Data shape {ndata.shape} must match shape of object {self.shape}"
                )

        attrs = (
            copy.deepcopy(self._attrs, memo=memo) if deep else copy.copy(self._attrs)
        )

        return self._replace(data=ndata, attrs=attrs)

    def __copy__(self):
        return self._copy(deep=False)

    def __deepcopy__(self, memo: dict[int, typing.Any] | None = None):
        return self._copy(deep=True, memo=memo)

    def copy(self, deep: bool = True, data: npt.ArrayLike | None = None):
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

    # mutable objects should not be hashable
    # https://github.com/python/mypy/issues/4266
    __hash__ = None  # type: ignore[assignment]

    _array_counter = itertools.count()
