from __future__ import annotations

import copy
import math
import sys
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)

import numpy as np

# TODO: get rid of this after migrating this class to array API
from xarray.core import dtypes, formatting, formatting_html
from xarray.core.indexing import (
    ExplicitlyIndexed,
    ImplicitToExplicitIndexingAdapter,
    OuterIndexer,
)
from xarray.namedarray._aggregations import NamedArrayAggregations
from xarray.namedarray._typing import (
    Default,
    ErrorOptionsWithWarn,
    _arrayapi,
    _arrayfunction_or_api,
    _chunkedarray,
    _default,
    _dtype,
    _DType_co,
    _ScalarType_co,
    _ShapeType_co,
    _sparsearrayfunction_or_api,
    _SupportsImag,
    _SupportsReal,
    duckarray,
)
from xarray.namedarray.parallelcompat import guess_chunkmanager
from xarray.namedarray.pycompat import to_numpy
from xarray.namedarray.utils import (
    either_dict_or_kwargs,
    infix_dims,
    is_dict_like,
    is_duck_dask_array,
    to_0d_object_array,
)

if TYPE_CHECKING:
    from enum import IntEnum

    from numpy.typing import ArrayLike, NDArray

    from xarray.core.types import Dims, T_Chunks
    from xarray.namedarray._typing import (
        _AttrsLike,
        _Chunks,
        _Device,
        _Dim,
        _Dims,
        _DimsLike,
        _DType,
        _IndexKeyLike,
        _IntOrUnknown,
        _ScalarType,
        _Shape,
        _ShapeType,
    )
    from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint

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

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    T_NamedArray = TypeVar("T_NamedArray", bound="_NamedArray[Any]")
    T_NamedArrayInteger = TypeVar(
        "T_NamedArrayInteger", bound="_NamedArray[np.integer[Any]]"
    )


@overload
def _new(
    x: NamedArray[Any, _DType_co],
    dims: _DimsLike | Default = ...,
    data: duckarray[_ShapeType, _DType] = ...,
    attrs: _AttrsLike | Default = ...,
) -> NamedArray[_ShapeType, _DType]: ...


@overload
def _new(
    x: NamedArray[_ShapeType_co, _DType_co],
    dims: _DimsLike | Default = ...,
    data: Default = ...,
    attrs: _AttrsLike | Default = ...,
) -> NamedArray[_ShapeType_co, _DType_co]: ...


def _new(
    x: NamedArray[Any, _DType_co],
    dims: _DimsLike | Default = _default,
    data: duckarray[_ShapeType, _DType] | Default = _default,
    attrs: _AttrsLike | Default = _default,
) -> NamedArray[_ShapeType, _DType] | NamedArray[Any, _DType_co]:
    """
    Create a new array with new typing information.

    Parameters
    ----------
    x : NamedArray
        Array to create a new array from
    dims : Iterable of Hashable, optional
        Name(s) of the dimension(s).
        Will copy the dims from x by default.
    data : duckarray, optional
        The actual data that populates the array. Should match the
        shape specified by `dims`.
        Will copy the data from x by default.
    attrs : dict, optional
        A dictionary containing any additional information or
        attributes you want to store with the array.
        Will copy the attrs from x by default.
    """
    dims_ = copy.copy(x._dims) if isinstance(dims, Default) else dims

    attrs_: Mapping[Any, Any] | None
    if isinstance(attrs, Default):
        attrs_ = None if x._attrs is None else x._attrs.copy()
    else:
        attrs_ = attrs

    if isinstance(data, Default):
        return type(x)(dims_, copy.copy(x._data), attrs_)
    else:
        cls_ = cast("type[NamedArray[_ShapeType, _DType]]", type(x))
        return cls_(dims_, data, attrs_)


@overload
def from_array(
    dims: _DimsLike,
    data: duckarray[_ShapeType, _DType],
    attrs: _AttrsLike = ...,
) -> NamedArray[_ShapeType, _DType]: ...


@overload
def from_array(
    dims: _DimsLike,
    data: ArrayLike,
    attrs: _AttrsLike = ...,
) -> NamedArray[Any, Any]: ...


def from_array(
    dims: _DimsLike,
    data: duckarray[_ShapeType, _DType] | ArrayLike,
    attrs: _AttrsLike = None,
) -> NamedArray[_ShapeType, _DType] | NamedArray[Any, Any]:
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
        raise TypeError(
            "Array is already a Named array. Use 'data.data' to retrieve the data array"
        )

    # TODO: dask.array.ma.MaskedArray also exists, better way?
    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)  # type: ignore[no-untyped-call]
        if mask.any():
            # TODO: requires refactoring/vendoring xarray.core.dtypes and
            # xarray.core.duck_array_ops
            raise NotImplementedError("MaskedArray is not supported yet")

        return NamedArray(dims, data, attrs)

    if isinstance(data, _arrayfunction_or_api) and not isinstance(data, np.generic):
        return NamedArray(dims, data, attrs)

    if isinstance(data, tuple):
        return NamedArray(dims, to_0d_object_array(data), attrs)

    # validate whether the data is valid data types.
    return NamedArray(dims, np.asarray(data), attrs)


class NamedArray(NamedArrayAggregations, Generic[_ShapeType_co, _DType_co]):
    """
    A wrapper around duck arrays with named dimensions
    and attributes which describe a single Array.
    Numeric operations on this object implement array broadcasting and
    dimension alignment based on dimension names,
    rather than axis order.


    Parameters
    ----------
    dims : str or iterable of hashable
        Name(s) of the dimension(s).
    data : array-like or duck-array
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

    __slots__ = ("_attrs", "_data", "_dims")

    _data: duckarray[Any, _DType_co]
    _dims: _Dims
    _attrs: dict[Any, Any] | None

    def __init__(
        self,
        dims: _DimsLike,
        data: duckarray[Any, _DType_co],
        attrs: _AttrsLike = None,
    ):
        self._data = data
        self._dims = self._parse_dimensions(dims)
        self._attrs = dict(attrs) if attrs else None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if NamedArray in cls.__bases__ and (cls._new == NamedArray._new):
            # Type hinting does not work for subclasses unless _new is
            # overridden with the correct class.
            raise TypeError(
                "Subclasses of `NamedArray` must override the `_new` method."
            )
        super().__init_subclass__(**kwargs)

    @overload
    def _new(
        self,
        dims: _DimsLike | Default = ...,
        data: duckarray[_ShapeType, _DType] = ...,
        attrs: _AttrsLike | Default = ...,
    ) -> NamedArray[_ShapeType, _DType]: ...

    @overload
    def _new(
        self,
        dims: _DimsLike | Default = ...,
        data: Default = ...,
        attrs: _AttrsLike | Default = ...,
    ) -> NamedArray[_ShapeType_co, _DType_co]: ...

    def _new(
        self,
        dims: _DimsLike | Default = _default,
        data: duckarray[Any, _DType] | Default = _default,
        attrs: _AttrsLike | Default = _default,
    ) -> NamedArray[_ShapeType, _DType] | NamedArray[_ShapeType_co, _DType_co]:
        """
        Create a new array with new typing information.

        _new has to be reimplemented each time NamedArray is subclassed,
        otherwise type hints will not be correct. The same is likely true
        for methods that relied on _new.

        Parameters
        ----------
        dims : Iterable of Hashable, optional
            Name(s) of the dimension(s).
            Will copy the dims from x by default.
        data : duckarray, optional
            The actual data that populates the array. Should match the
            shape specified by `dims`.
            Will copy the data from x by default.
        attrs : dict, optional
            A dictionary containing any additional information or
            attributes you want to store with the array.
            Will copy the attrs from x by default.
        """
        return _new(self, dims, data, attrs)

    def _replace(
        self,
        dims: _DimsLike | Default = _default,
        data: duckarray[_ShapeType_co, _DType_co] | Default = _default,
        attrs: _AttrsLike | Default = _default,
    ) -> Self:
        """
        Create a new array with the same typing information.

        The types for each argument cannot change,
        use self._new if that is a risk.

        Parameters
        ----------
        dims : Iterable of Hashable, optional
            Name(s) of the dimension(s).
            Will copy the dims from x by default.
        data : duckarray, optional
            The actual data that populates the array. Should match the
            shape specified by `dims`.
            Will copy the data from x by default.
        attrs : dict, optional
            A dictionary containing any additional information or
            attributes you want to store with the array.
            Will copy the attrs from x by default.
        """
        return cast("Self", self._new(dims, data, attrs))

    def _copy(
        self,
        deep: bool = True,
        data: duckarray[_ShapeType_co, _DType_co] | None = None,
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
        data: duckarray[_ShapeType_co, _DType_co] | None = None,
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

    def __len__(self) -> _IntOrUnknown:
        try:
            return self.shape[0]
        except Exception as exc:
            raise TypeError("len() of unsized object") from exc

    # < Array api >

    # def _maybe_asarray(
    #     self, x: bool | int | float | complex | NamedArray[Any, Any]
    # ) -> NamedArray[Any, Any]:
    #     """
    #     If x is a scalar, use asarray with the same dtype as self.
    #     If it is namedarray already, respect the dtype and return it.

    #     Array API always promotes scalars to the same dtype as the other array.
    #     Arrays are promoted according to result_types.
    #     """
    #     from xarray.namedarray._array_api import asarray

    #     if isinstance(x, NamedArray):
    #         # x is proper array. Respect the chosen dtype.
    #         return x
    #     # x is a scalar. Use the same dtype as self.
    #     # TODO: Is this a good idea? x[Any, int] + 1.4 => int result then.
    #     return asarray(x, dtype=self.dtype)

    def _promote_scalar(
        self: NamedArray[Any, Any],
        scalar: bool | int | float | complex,
        dtype_category: str | None = None,
        func_name: str = "",
    ) -> NamedArray[Any, Any]:
        """
        Returns a promoted version of a Python scalar appropriate for use with
        operations on x.

        This may raise an OverflowError in cases where the scalar is an
        integer that is too large to fit in a NumPy integer dtype, or
        TypeError when the scalar type is incompatible with the dtype of x.

        Examples
        --------
        >>> import numpy as np
        >>> x = NamedArray((), np.array(-2, dtype=np.int8))
        >>> x._promote_scalar(3)
        <xarray.NamedArray ()> Size: 1B
        array(3, dtype=int8)
        """
        from xarray.namedarray._array_api import asarray, iinfo
        from xarray.namedarray._array_api._dtypes import (
            _boolean_dtypes,
            _dtype_categories,
            _floating_dtypes,
            _integer_dtypes,
        )

        if dtype_category is not None:
            _allowed_dtypes = _dtype_categories[dtype_category]
            if self.dtype not in _allowed_dtypes:
                raise TypeError(
                    f"Only {dtype_category} dtypes are allowed in {func_name}. "
                    "Got {self.dtype}."
                )

        # Note: Only Python scalar types that match the array dtype are
        # allowed.
        if isinstance(scalar, bool):
            if self.dtype not in _boolean_dtypes:
                raise TypeError(
                    "Python bool scalars can only be promoted with bool arrays"
                )
        elif isinstance(scalar, int):
            if self.dtype in _boolean_dtypes:
                raise TypeError(
                    "Python int scalars cannot be promoted with bool arrays"
                )
            if self.dtype in _integer_dtypes:
                info = iinfo(self.dtype)
                if not (info.min <= scalar <= info.max):
                    raise OverflowError(
                        "Python int scalars must be within the bounds of the dtype for integer arrays"
                    )
            # int + array(floating) is allowed
        elif isinstance(scalar, float):
            if self.dtype not in _floating_dtypes:
                raise TypeError(
                    "Python float scalars can only be promoted with floating-point arrays."
                )
        elif isinstance(scalar, complex):
            if self.dtype not in _floating_dtypes:
                raise TypeError(
                    "Python complex scalars can only be promoted with floating-point arrays."
                )
        else:
            raise TypeError("'scalar' must be a Python scalar")

        # Note: scalars are unconditionally cast to the same dtype as the
        # array.

        # Note: the spec only specifies integer-dtype/int promotion
        # behavior for integers within the bounds of the integer dtype.
        # Outside of those bounds we use the default NumPy behavior (either
        # cast or raise OverflowError).
        return asarray(scalar, dtype=self.dtype, device=self.device)

    # Required methods below:

    def __abs__(self, /) -> Self:
        from xarray.namedarray._array_api import abs

        return abs(self)

    def __add__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import add

        return add(self, other)

    def __and__(
        self, other: int | bool | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import bitwise_and

        return bitwise_and(self, other)

    def __array_namespace__(
        self, /, *, api_version: str | None = None
    ) -> NamedArray[Any, Any]:
        if api_version is not None and api_version not in (
            "2021.12",
            "2022.12",
            "2023.12",
            "2024.12",
        ):
            raise ValueError(f"Unrecognized array API version: {api_version!r}")
        import xarray.namedarray._array_api as array_api

        return array_api

    def __bool__(self, /) -> bool:
        return self._data.__bool__()

    def __complex__(self, /) -> complex:
        return self._data.__complex__()

    def __dlpack__(
        self,
        /,
        *,
        stream: int | Any | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[IntEnum, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        return self._data.__dlpack__(
            stream=stream, max_version=max_version, dl_device=dl_device, copy=copy
        )

    def __dlpack_device__(self, /) -> tuple[IntEnum, int]:
        return self._data.__dlpack_device__()

    def __eq__(self, other: int | float | bool | NamedArray[Any, Any], /) -> bool:
        from xarray.namedarray._array_api import equal

        return equal(self, other)

    def __float__(self, /) -> float:
        return self._data.__float__()

    def __floordiv__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import floor_divide

        return floor_divide(self, other)

    def __ge__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import greater_equal

        return greater_equal(self, other)

    def __getitem__(
        self, key: _IndexKeyLike | NamedArray[Any, Any]
    ) -> NamedArray[Any, Any]:
        """
        Returns self[key].

        Some rules:
        * Integers removes the dim.
        * Slices and ellipsis maintains same dim.
        * None adds a dim.
        * tuple follows above but on that specific axis.

        Examples
        --------

        1D

        >>> x = NamedArray(("x",), np.array([0, 1, 2]))
        >>> key = NamedArray(("x",), np.array([1, 0, 0], dtype=bool))
        >>> xm = x[key]
        >>> xm.dims, xm.shape
        (('x',), (1,))

        >>> x = NamedArray(("x",), np.array([0, 1, 2]))
        >>> key = NamedArray(("x",), np.array([0, 0, 0], dtype=bool))
        >>> xm = x[key]
        >>> xm.dims, xm.shape
        (('x',), (0,))

        Setup a ND array:

        >>> x = NamedArray(("x", "y"), np.arange(3 * 4).reshape((3, 4)))
        >>> xm = x[0]
        >>> xm.dims, xm.shape
        (('y',), (4,))
        >>> xm = x[slice(0)]
        >>> xm.dims, xm.shape
        (('x', 'y'), (0, 4))
        >>> xm = x[None]
        >>> xm.dims, xm.shape
        (('dim_2', 'x', 'y'), (1, 3, 4))

        >>> key = NamedArray(("x", "y"), np.ones((3, 4), dtype=bool))
        >>> xm = x[key]
        >>> xm.dims, xm.shape
        ((('x', 'y'),), (12,))

        0D

        >>> x = NamedArray((), np.array(False, dtype=np.bool))
        >>> key = NamedArray((), np.array(False, dtype=np.bool))
        >>> xm = x[key]
        >>> xm.dims, xm.shape
        (('dim_0',), (0,))
        """
        from xarray.namedarray._array_api._manipulation_functions import (
            _broadcast_arrays,
        )
        from xarray.namedarray._array_api._utils import (
            _atleast1d_dims,
            _dims_from_tuple_indexing,
            _flatten_dims,
        )

        if isinstance(key, NamedArray):
            self_new, key_new = _broadcast_arrays(self, key)
            _data = self_new._data[key_new._data]
            _dims = _flatten_dims(_atleast1d_dims(self_new.dims))
            return self._new(_dims, _data)
        # elif isinstance(key, int):
        #     return self._new(self.dims[1:], self._data[key])
        # elif isinstance(key, slice) or key is ...:
        #     return self._new(self.dims, self._data[key])
        # elif key is None:
        #     return expand_dims(self)
        # elif isinstance(key, tuple):
        #     _dims = _dims_from_tuple_indexing(self.dims, key)
        #     return self._new(_dims, self._data[key])

        elif isinstance(key, int | slice | tuple) or key is None or key is ...:
            # TODO: __getitem__ not always available, use expand_dims
            _data = self._data[key]
            _dims = _dims_from_tuple_indexing(
                self.dims, key if isinstance(key, tuple) else (key,)
            )
            return self._new(_dims, _data)
        else:
            raise NotImplementedError(f"{key=} is not supported")

    def __gt__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import greater

        return greater(self, other)

    def __index__(self, /) -> int:
        return self._data.__index__()

    def __int__(self, /) -> int:
        return self._data.__int__()

    def __invert__(self, /) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import bitwise_invert

        return bitwise_invert(self)

    def __iter__(self: NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import asarray

        # TODO: smarter way to retain dims, xarray?
        return (asarray(i) for i in self._data)

    def __le__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import less_equal

        return less_equal(self, other)

    def __lshift__(self, other: int | NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import bitwise_left_shift

        return bitwise_left_shift(self, other)

    def __lt__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import less

        return less(self, other)

    def __matmul__(self, other: NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import matmul

        return matmul(self, other)

    def __mod__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import remainder

        return remainder(self, other)

    def __mul__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import multiply

        return multiply(self, other)

    def __ne__(
        self, other: int | float | bool | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import not_equal

        return not_equal(self, other)

    def __neg__(self, /) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import negative

        return negative(self)

    def __or__(
        self, other: int | bool | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import bitwise_or

        return bitwise_or(self, other)

    def __pos__(self, /) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import positive

        return positive(self)

    def __pow__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import pow

        return pow(self, other)

    def __rshift__(self, other: int | NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import bitwise_right_shift

        return bitwise_right_shift(self, other)

    def __setitem__(
        self,
        key: _IndexKeyLike,
        value: int | float | bool | NamedArray[Any, Any],
        /,
    ) -> None:
        if isinstance(key, NamedArray):
            key = key._data
        self._data.__setitem__(key, self._maybe_asarray(value)._data)

    def __sub__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import subtract

        return subtract(self, other)

    def __truediv__(
        self, other: float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import divide

        return divide(self, other)

    def __xor__(
        self, other: int | bool | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import bitwise_xor

        return bitwise_xor(self, other)

    def __iadd__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        self._data += other._data
        return self

    def __radd__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import add

        return add(other, self)

    def __iand__(
        self, other: int | bool | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        self._data &= other._data
        return self

    def __rand__(
        self, other: int | bool | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import bitwise_and

        return bitwise_and(other, self)

    def __ifloordiv__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        self._data //= other._data
        return self

    def __rfloordiv__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import floor_divide

        return floor_divide(other, self)

    def __ilshift__(self, other: int | NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
        self._data <<= other._data
        return self

    def __rlshift__(self, other: int | NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import bitwise_left_shift

        return bitwise_left_shift(other, self)

    def __imatmul__(self, other: NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
        self._data @= other._data
        return self

    def __rmatmul__(self, other: NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import matmul

        return matmul(other, self)

    def __imod__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        self._data %= other._data
        return self

    def __rmod__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import remainder

        return remainder(other, self)

    def __imul__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        self._data *= other._data
        return self

    def __rmul__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import multiply

        return multiply(other, self)

    def __ior__(
        self, other: int | bool | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        self._data |= other._data

        return self

    def __ror__(
        self, other: int | bool | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import bitwise_or

        return bitwise_or(other, self)

    def __ipow__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        self._data **= other._data
        return self

    def __rpow__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import pow

        return pow(other, self)

    def __irshift__(self, other: int | NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
        self._data >>= other._data
        return self

    def __rrshift__(self, other: int | NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import bitwise_right_shift

        return bitwise_right_shift(other, self)

    def __isub__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        self._data -= other._data
        return self

    def __rsub__(
        self, other: int | float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import subtract

        return subtract(other, self)

    def __itruediv__(
        self, other: float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        self._data /= other._data
        return self

    def __rtruediv__(
        self, other: float | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import divide

        return divide(other, self)

    def __ixor__(
        self, other: int | bool | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        self._data ^= other._data
        return self

    def __rxor__(
        self, other: int | bool | NamedArray[Any, Any], /
    ) -> NamedArray[Any, Any]:
        from xarray.namedarray._array_api import bitwise_xor

        return bitwise_xor(other, self)

    def to_device(self, device: _Device, /, stream: None = None) -> Self:
        if isinstance(self._data, _arrayapi):
            return self._replace(data=self._data.to_device(device, stream=stream))
        else:
            raise NotImplementedError("Only array api are valid.")

    @property
    def dtype(self) -> _DType_co:
        """
        Data-type of the array’s elements.

        See Also
        --------
        ndarray.dtype
        numpy.dtype
        """
        return self._data.dtype

    @property
    def device(self) -> _Device:
        """
        Device of the array’s elements.

        See Also
        --------
        ndarray.device
        """
        if isinstance(self._data, _arrayapi):
            return self._data.device
        else:
            raise NotImplementedError("self._data missing device")

    @property
    def mT(self) -> NamedArray[Any, _DType_co]:
        if isinstance(self._data, _arrayapi):
            from xarray.namedarray._array_api._utils import _infer_dims

            _data = self._data.mT
            _dims = _infer_dims(_data.shape)
            return self._new(_dims, _data)
        else:
            raise NotImplementedError("self._data missing mT")

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
    def shape(self) -> _Shape:
        """
        Get the shape of the array.

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
    def size(self) -> _IntOrUnknown:
        """
        Number of elements in the array.

        Equal to ``np.prod(a.shape)``, i.e., the product of the array’s dimensions.

        See Also
        --------
        numpy.ndarray.size
        """
        return math.prod(self.shape)

    @property
    def T(self) -> NamedArray[Any, _DType_co]:
        """Return a new object with transposed dimensions."""
        if self.ndim != 2:
            raise ValueError(
                f"x.T requires x to have 2 dimensions, x has {self.dims}. "
                "Use x.permute_dims() to permute multiple dimensions."
            )

        return self.permute_dims()

    # </ Array api >

    @property
    def nbytes(self) -> _IntOrUnknown:
        """
        Total bytes consumed by the elements of the data array.

        If the underlying data array does not include ``nbytes``, estimates
        the bytes consumed based on the ``size`` and ``dtype``.
        """
        from xarray.namedarray._array_api._utils import _get_data_namespace

        if hasattr(self._data, "nbytes"):
            return self._data.nbytes  # type: ignore[no-any-return]

        if hasattr(self.dtype, "itemsize"):
            itemsize = self.dtype.itemsize
        elif isinstance(self._data, _arrayapi):
            xp = _get_data_namespace(self)

            if xp.isdtype(self.dtype, "bool"):
                itemsize = 1
            elif xp.isdtype(self.dtype, "integral"):
                itemsize = xp.iinfo(self.dtype).bits // 8
            else:
                itemsize = xp.finfo(self.dtype).bits // 8
        else:
            raise TypeError(
                "cannot compute the number of bytes (no array API nor nbytes / itemsize)"
            )

        return self.size * itemsize

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
        if len(set(dims)) < len(dims):
            repeated_dims = {d for d in dims if dims.count(d) > 1}
            warnings.warn(
                f"Duplicate dimension names present: dimensions {repeated_dims} appear more than once in dims={dims}. "
                "We do not yet support duplicate dimension names, but we do allow initial construction of the object. "
                "We recommend you rename the dims immediately to become distinct, as most xarray functionality is likely to fail silently if you do not. "
                "To rename the dimensions you will need to set the ``.dims`` attribute of each variable, ``e.g. var.dims=('x0', 'x1')``.",
                UserWarning,
                stacklevel=2,
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
        self._attrs = dict(value) if value else None

    def _check_shape(self, new_data: duckarray[Any, _DType_co]) -> None:
        if new_data.shape != self.shape:
            raise ValueError(
                f"replacement data must match the {self.__class__.__name__}'s shape. "
                f"replacement data has shape {new_data.shape}; {self.__class__.__name__} has shape {self.shape}"
            )

    @property
    def data(self) -> duckarray[Any, _DType_co]:
        """
        The NamedArray's data as an array. The underlying array type
        (e.g. dask, sparse, pint) is preserved.

        """

        return self._data

    @data.setter
    def data(self, data: duckarray[Any, _DType_co]) -> None:
        self._check_shape(data)
        self._data = data

    @property
    def imag(
        self: NamedArray[_ShapeType, np.dtype[_SupportsImag[_ScalarType]]],  # type: ignore[type-var]
    ) -> NamedArray[_ShapeType, _dtype[_ScalarType]]:
        """
        The imaginary part of the array.

        See Also
        --------
        numpy.ndarray.imag
        """
        if isinstance(self._data, _arrayapi):
            from xarray.namedarray._array_api import imag

            return imag(self)

        return self._new(data=self._data.imag)

    @property
    def real(
        self: NamedArray[_ShapeType, np.dtype[_SupportsReal[_ScalarType]]],  # type: ignore[type-var]
    ) -> NamedArray[_ShapeType, _dtype[_ScalarType]]:
        """
        The real part of the array.

        See Also
        --------
        numpy.ndarray.real
        """
        if isinstance(self._data, _arrayapi):
            from xarray.namedarray._array_api import real

            return real(self)
        return self._new(data=self._data.real)

    @overload
    def __array__(
        self, dtype: None = ..., /, *, copy: bool | None = ...
    ) -> np.ndarray[Any, _DType_co]: ...
    @overload
    def __array__(
        self, dtype: _DType, /, *, copy: bool | None = ...
    ) -> np.ndarray[Any, _DType]: ...
    def __array__(
        self, dtype: _DType | None = ..., /, *, copy: bool | None = ...
    ) -> np.ndarray[Any, _DType] | np.ndarray[Any, _DType_co]:
        return np.asarray(self._data)

    def __dask_tokenize__(self) -> object:
        # Use v.data, instead of v._data, in order to cope with the wrappers
        # around NetCDF and the like
        from dask.base import normalize_token

        return normalize_token((type(self), self._dims, self.data, self._attrs or None))

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

    @overload
    def get_axis_num(self, dim: str) -> int: ...  # type: ignore [overload-overlap]

    @overload
    def get_axis_num(self, dim: Iterable[Hashable]) -> tuple[int, ...]: ...

    @overload
    def get_axis_num(self, dim: Hashable) -> int: ...

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
        _raise_if_any_duplicate_dimensions(self.dims)
        try:
            return self.dims.index(dim)  # type: ignore[no-any-return]
        except ValueError as err:
            raise ValueError(
                f"{dim!r} not found in array dimensions {self.dims!r}"
            ) from err

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
        if isinstance(data, _chunkedarray):
            return data.chunks
        else:
            return None

    @property
    def chunksizes(
        self,
    ) -> Mapping[_Dim, _Shape]:
        """
        Mapping from dimension names to block lengths for this NamedArray's data.

        If this NamedArray does not contain chunked arrays, the mapping will be empty.

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
        if isinstance(data, _chunkedarray):
            return dict(zip(self.dims, data.chunks, strict=True))
        else:
            return {}

    @property
    def sizes(self) -> dict[_Dim, _IntOrUnknown]:
        """Ordered mapping from dimension names to lengths."""
        return dict(zip(self.dims, self.shape, strict=True))

    def chunk(
        self,
        chunks: T_Chunks = {},  # noqa: B006  # even though it's unsafe, it is being used intentionally here (#4667)
        chunked_array_type: str | ChunkManagerEntrypoint[Any] | None = None,
        from_array_kwargs: Any = None,
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

        if from_array_kwargs is None:
            from_array_kwargs = {}

        if chunks is None:
            warnings.warn(
                "None value for 'chunks' is deprecated. "
                "It will raise an error in the future. Use instead '{}'",
                category=FutureWarning,
                stacklevel=2,
            )
            chunks = {}

        if isinstance(chunks, float | str | int | tuple | list):
            # TODO we shouldn't assume here that other chunkmanagers can handle these types
            # TODO should we call normalize_chunks here?
            pass  # dask.array.from_array can handle these directly
        else:
            chunks = either_dict_or_kwargs(chunks, chunks_kwargs, "chunk")

        if is_dict_like(chunks):
            # This method of iteration allows for duplicated dimension names, GH8579
            chunks = {
                dim_number: chunks[dim]
                for dim_number, dim in enumerate(self.dims)
                if dim in chunks
            }

        chunkmanager = guess_chunkmanager(chunked_array_type)

        data_old = self._data
        if chunkmanager.is_chunked_array(data_old):
            data_chunked = chunkmanager.rechunk(data_old, chunks)  # type: ignore[arg-type]
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
                ndata = ImplicitToExplicitIndexingAdapter(data_old, OuterIndexer)  # type: ignore[assignment]

            if is_dict_like(chunks):
                chunks = tuple(chunks.get(n, s) for n, s in enumerate(ndata.shape))

            data_chunked = chunkmanager.from_array(ndata, chunks, **from_array_kwargs)  # type: ignore[arg-type]

        return self._replace(data=data_chunked)

    def to_numpy(self) -> np.ndarray[Any, Any]:
        """Coerces wrapped data to numpy and returns a numpy.ndarray"""
        # TODO an entrypoint so array libraries can choose coercion method?
        return to_numpy(self._data)

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
    ) -> NamedArray[Any, Any]:
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
        return from_array(dims, data, attrs=self._attrs)

    def _nonzero(self: T_NamedArrayInteger) -> tuple[T_NamedArrayInteger, ...]:
        """Equivalent numpy's nonzero but returns a tuple of NamedArrays."""
        # TODO: we should replace dask's native nonzero
        # after https://github.com/dask/dask/issues/1076 is implemented.
        # TODO: cast to ndarray and back to T_DuckArray is a workaround
        nonzeros = np.nonzero(cast("NDArray[np.integer[Any]]", self.data))
        _attrs = self.attrs
        return tuple(
            cast("T_NamedArrayInteger", self._new((dim,), nz, _attrs))
            for nz, dim in zip(nonzeros, self.dims, strict=True)
        )

    def __repr__(self) -> str:
        return formatting.array_repr(self)

    def _repr_html_(self) -> str:
        return formatting_html.array_repr(self)

    def _as_sparse(
        self,
        sparse_format: Literal["coo"] | Default = _default,
        fill_value: ArrayLike | Default = _default,
    ) -> NamedArray[Any, _DType_co]:
        """
        Use sparse-array as backend.
        """
        import sparse

        from xarray.namedarray._array_api import astype

        # TODO: what to do if dask-backended?
        if isinstance(fill_value, Default):
            dtype, fill_value = dtypes.maybe_promote(self.dtype)
        else:
            dtype = dtypes.result_type(self.dtype, fill_value)

        if isinstance(sparse_format, Default):
            sparse_format = "coo"
        try:
            as_sparse = getattr(sparse, f"as_{sparse_format.lower()}")
        except AttributeError as exc:
            raise ValueError(f"{sparse_format} is not a valid sparse format") from exc

        data = as_sparse(astype(self, dtype).data, fill_value=fill_value)
        return self._new(data=data)

    def _to_dense(self) -> NamedArray[Any, _DType_co]:
        """
        Change backend from sparse to np.array.
        """
        if isinstance(self._data, _sparsearrayfunction_or_api):
            data_dense: np.ndarray[Any, _DType_co] = self._data.todense()
            return self._new(data=data_dense)
        else:
            raise TypeError("self.data is not a sparse array")

    def permute_dims(
        self,
        *dim: Iterable[_Dim] | EllipsisType,
        missing_dims: ErrorOptionsWithWarn = "raise",
    ) -> NamedArray[Any, _DType_co]:
        """Return a new object with transposed dimensions.

        Parameters
        ----------
        *dim : Hashable, optional
            By default, reverse the order of the dimensions. Otherwise, reorder the
            dimensions to this order.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            NamedArray:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions

        Returns
        -------
        NamedArray
            The returned NamedArray has permuted dimensions and data with the
            same attributes as the original.


        See Also
        --------
        numpy.transpose
        """

        from xarray.namedarray._array_api import permute_dims

        if not dim:
            dims = self.dims[::-1]
        else:
            dims = tuple(infix_dims(dim, self.dims, missing_dims))  # type: ignore[arg-type]

        if len(dims) < 2 or dims == self.dims:
            # no need to transpose if only one dimension
            # or dims are in same order
            return self.copy(deep=False)

        axes = self.get_axis_num(dims)
        assert isinstance(axes, tuple)

        return permute_dims(self, axes)

    def broadcast_to(
        self, dim: Mapping[_Dim, int] | None = None, **dim_kwargs: Any
    ) -> NamedArray[Any, _DType_co]:
        """
        Broadcast the NamedArray to a new shape. New dimensions are not allowed.

        This method allows for the expansion of the array's dimensions to a specified shape.
        It handles both positional and keyword arguments for specifying the dimensions to broadcast.
        An error is raised if new dimensions are attempted to be added.

        Parameters
        ----------
        dim : dict, str, sequence of str, optional
            Dimensions to broadcast the array to. If a dict, keys are dimension names and values are the new sizes.
            If a string or sequence of strings, existing dimensions are matched with a size of 1.

        **dim_kwargs : Any
            Additional dimensions specified as keyword arguments. Each keyword argument specifies the name of an existing dimension and its size.

        Returns
        -------
        NamedArray
            A new NamedArray with the broadcasted dimensions.

        Examples
        --------
        >>> data = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        >>> array = NamedArray(("x", "y"), data)
        >>> array.sizes
        {'x': 2, 'y': 2}

        >>> broadcasted = array.broadcast_to(x=2, y=2)
        >>> broadcasted.sizes
        {'x': 2, 'y': 2}
        """

        from xarray.core import duck_array_ops

        combined_dims = either_dict_or_kwargs(dim, dim_kwargs, "broadcast_to")

        # Check that no new dimensions are added
        if new_dims := set(combined_dims) - set(self.dims):
            raise ValueError(
                f"Cannot add new dimensions: {new_dims}. Only existing dimensions are allowed. "
                "Use `expand_dims` method to add new dimensions."
            )

        # Create a dictionary of the current dimensions and their sizes
        current_shape = self.sizes

        # Update the current shape with the new dimensions, keeping the order of the original dimensions
        broadcast_shape = {d: current_shape.get(d, 1) for d in self.dims}
        broadcast_shape |= combined_dims

        # Ensure the dimensions are in the correct order
        ordered_dims = list(broadcast_shape.keys())
        ordered_shape = tuple(broadcast_shape[d] for d in ordered_dims)
        data = duck_array_ops.broadcast_to(self._data, ordered_shape)  # type: ignore[no-untyped-call]  # TODO: use array-api-compat function
        return self._new(data=data, dims=ordered_dims)

    def expand_dims(
        self,
        dim: _Dim | Default = _default,
    ) -> NamedArray[Any, _DType_co]:
        """
        Expand the dimensions of the NamedArray.

        This method adds new dimensions to the object. The new dimensions are added at the beginning of the array.

        Parameters
        ----------
        dim : Hashable, optional
            Dimension name to expand the array to. This dimension will be added at the beginning of the array.

        Returns
        -------
        NamedArray
            A new NamedArray with expanded dimensions.


        Examples
        --------
        >>> data = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        >>> x = NamedArray(("x", "y"), data)
        >>> expanded = x.expand_dims(dim="z")
        >>> expanded.dims
        ('z', 'x', 'y')
        """

        from xarray.namedarray._array_api import expand_dims

        return expand_dims(self, dim=dim)


_NamedArray = NamedArray[Any, np.dtype[_ScalarType_co]]


def _raise_if_any_duplicate_dimensions(
    dims: _Dims, err_context: str = "This function"
) -> None:
    if len(set(dims)) < len(dims):
        repeated_dims = {d for d in dims if dims.count(d) > 1}
        raise ValueError(
            f"{err_context} cannot handle duplicate dimensions, but dimensions {repeated_dims} appear more than once on this object's dims: {dims}"
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
