from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType
from typing import TYPE_CHECKING, Any

from xarray.namedarray._typing import (
    Default,
    _arrayapi,
    _Axis,
    _AxisLike,
    _Axes,
    _default,
    _Dim,
    _Dims,
    _DimsLike,
    _DType,
    _dtype,
    _Shape,
    duckarray,
)

if TYPE_CHECKING:
    from xarray.namedarray.core import NamedArray


def _maybe_default_namespace(xp: ModuleType | None = None) -> ModuleType:
    if xp is None:
        # import array_api_strict as xpd

        # import array_api_compat.numpy as xpd

        import numpy as xpd

        return xpd
    else:
        return xp


def _get_data_namespace(x: NamedArray[Any, Any]) -> ModuleType:
    if isinstance(x._data, _arrayapi):
        return x._data.__array_namespace__()

    return _maybe_default_namespace()


def _get_namespace_dtype(dtype: _dtype | None = None) -> ModuleType:
    if dtype is None:
        return _maybe_default_namespace()

    try:
        xp = __import__(dtype.__module__)
    except AttributeError:
        # TODO: Fix this.
        #         FAILED array_api_tests/test_searching_functions.py::test_searchsorted - AttributeError: 'numpy.dtypes.Float64DType' object has no attribute '__module__'. Did you mean: '__mul__'?
        # Falsifying example: test_searchsorted(
        #     data=data(...),
        # )
        return _maybe_default_namespace()
    return xp


def _infer_dims(
    shape: _Shape,
    dims: _DimsLike | Default = _default,
) -> _DimsLike:
    if dims is _default:
        return tuple(f"dim_{n}" for n in range(len(shape)))
    else:
        return dims


def _normalize_dimensions(dims: _DimsLike) -> _Dims:
    """
    Normalize dimensions.

    Examples
    --------
    >>> _normalize_dimensions(None)
    (None,)
    >>> _normalize_dimensions(1)
    (1,)
    >>> _normalize_dimensions("2")
    ('2',)
    >>> _normalize_dimensions(("time",))
    ('time',)
    >>> _normalize_dimensions(["time"])
    ('time',)
    >>> _normalize_dimensions([("time", "x", "y")])
    (('time', 'x', 'y'),)
    """
    if isinstance(dims, str) or not isinstance(dims, Iterable):
        return (dims,)

    return tuple(dims)


def _assert_either_dim_or_axis(
    dims: _Dim | _Dims | Default, axis: _AxisLike | None
) -> None:
    if dims is not _default and axis is not None:
        raise ValueError("cannot supply both 'axis' and 'dim(s)' arguments")


def _dims_to_axis(
    x: NamedArray[Any, Any], dims: _Dim | _Dims | Default, axis: _AxisLike | None
) -> _Axes | None:
    """
    Convert dims to axis indices.

    Examples
    --------
    >>> narr = NamedArray(("x", "y"), np.array([[1, 2, 3], [5, 6, 7]]))
    >>> _dims_to_axis(narr, ("y",), None)
    (1,)
    >>> _dims_to_axis(narr, _default, 0)
    (0,)
    >>> _dims_to_axis(narr, None, None)
    """
    _assert_either_dim_or_axis(dims, axis)

    if dims is not _default:
        axis = ()
        for dim in dims:
            try:
                axis = (x.dims.index(dim),)
            except ValueError:
                raise ValueError(f"{dim!r} not found in array dimensions {x.dims!r}")
        return axis

    if isinstance(axis, int):
        return (axis,)

    return axis


def _get_remaining_dims(
    x: NamedArray[Any, _DType],
    data: duckarray[Any, _DType],
    axis: _AxisLike | None,
    *,
    keepdims: bool,
) -> tuple[_Dims, duckarray[Any, _DType]]:
    """
    Get the reamining dims after a reduce operation.
    """
    if data.shape == x.shape:
        return x.dims, data

    removed_axes: tuple[int, ...]
    if axis is None:
        removed_axes = tuple(v for v in range(x.ndim))
    else:
        removed_axes = axis % x.ndim if isinstance(axis, tuple) else (axis % x.ndim,)

    if keepdims:
        # Insert None (aka newaxis) for removed dims
        slices = tuple(
            None if i in removed_axes else slice(None, None) for i in range(x.ndim)
        )
        data = data[slices]
        dims = x.dims
    else:
        dims = tuple(adim for n, adim in enumerate(x.dims) if n not in removed_axes)

    return dims, data


def _insert_dim(dims: _Dims, dim: _Dim | Default, axis: _Axis) -> _Dims:
    if dim is _default:
        dim = f"dim_{len(dims)}"
    d = list(dims)
    d.insert(axis, dim)
    return tuple(d)
