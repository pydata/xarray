from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import zip_longest
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from xarray.namedarray._typing import (
    Default,
    _arrayapi,
    _Axes,
    _Axis,
    _AxisLike,
    _default,
    _Dim,
    _Dims,
    _DimsLike2,
    _DType,
    _dtype,
    _Shape,
    duckarray,
)

from xarray.namedarray.core import NamedArray


def _maybe_default_namespace(xp: ModuleType | None = None) -> ModuleType:
    if xp is None:
        # import array_api_strict as xpd

        # import array_api_compat.numpy as xpd

        import numpy as xpd

        return xpd
    else:
        return xp


def _get_namespace(x: Any) -> ModuleType:
    if isinstance(x, _arrayapi):
        return x.__array_namespace__()

    return _maybe_default_namespace()


def _get_data_namespace(x: NamedArray[Any, Any]) -> ModuleType:
    return _get_namespace(x._data)


def _get_namespace_dtype(dtype: _dtype[Any] | None = None) -> ModuleType:
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


def _is_single_dim(dims: _DimsLike2) -> TypeGuard[_Dim]:
    # TODO: https://peps.python.org/pep-0742/
    return isinstance(dims, str) or not isinstance(dims, Iterable)


def _normalize_dimensions(dims: _DimsLike2) -> _Dims:
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
    if _is_single_dim(dims):
        return (dims,)
    else:
        return tuple(cast(_Dims, dims))


def _infer_dims(
    shape: _Shape,
    dims: _DimsLike2 | Default = _default,
) -> _Dims:
    """
    Create default dim names if no dims were supplied.

    Examples
    --------
    >>> _infer_dims(())
    ()
    >>> _infer_dims((1,))
    ('dim_0',)
    >>> _infer_dims((3, 1))
    ('dim_1', 'dim_0')

    >>> _infer_dims((1,), "x")
    ('x',)
    >>> _infer_dims((1,), None)
    (None,)
    >>> _infer_dims((1,), ("x",))
    ('x',)
    """
    if isinstance(dims, Default):
        ndim = len(shape)
        return tuple(f"dim_{ndim - 1 - n}" for n in range(ndim))
    else:
        return _normalize_dimensions(dims)


def _assert_either_dim_or_axis(
    dims: _DimsLike2 | Default, axis: _AxisLike | None
) -> None:
    if dims is not _default and axis is not None:
        raise ValueError("cannot supply both 'axis' and 'dim(s)' arguments")


# @overload
# def _dims_to_axis(x: NamedArray[Any, Any], dims: Default, axis: None) -> None: ...
# @overload
# def _dims_to_axis(x: NamedArray[Any, Any], dims: _DimsLike2, axis: None) -> _Axes: ...
# @overload
# def _dims_to_axis(x: NamedArray[Any, Any], dims: Default, axis: _AxisLike) -> _Axes: ...
def _dims_to_axis(
    x: NamedArray[Any, Any], dims: _DimsLike2 | Default, axis: _AxisLike | None
) -> _Axes | None:
    """
    Convert dims to axis indices.

    Examples
    --------

    Convert to dims to axis values

    >>> x = NamedArray(("x", "y"), np.array([[1, 2, 3], [5, 6, 7]]))
    >>> _dims_to_axis(x, ("y",), None)
    (1,)
    >>> _dims_to_axis(x, _default, 0)
    (0,)
    >>> _dims_to_axis(x, _default, None)

    Using Hashable dims

    >>> x = NamedArray(("x", None), np.array([[1, 2, 3], [5, 6, 7]]))
    >>> _dims_to_axis(x, None, None)
    (1,)

    Defining both dims and axis raises an error

    >>> _dims_to_axis(x, "x", 1)
    Traceback (most recent call last):
     ...
    ValueError: cannot supply both 'axis' and 'dim(s)' arguments
    """
    _assert_either_dim_or_axis(dims, axis)
    if not isinstance(dims, Default):
        _dims = _normalize_dimensions(dims)

        axis = ()
        for dim in _dims:
            try:
                axis = (x.dims.index(dim),)
            except ValueError:
                raise ValueError(f"{dim!r} not found in array dimensions {x.dims!r}")
        return axis

    if axis is None:
        return axis

    if isinstance(axis, tuple):
        return axis
    else:
        return (axis,)


def _dim_to_optional_axis(
    x: NamedArray[Any, Any], dim: _Dim | Default, axis: int | None
) -> int | None:
    a = _dims_to_axis(x, dim, axis)
    if a is None:
        return a

    return a[0]


def _dim_to_axis(x: NamedArray[Any, Any], dim: _Dim | Default, axis: int) -> int:
    _dim: _Dim = x.dims[axis] if isinstance(dim, Default) else dim
    _axis = _dim_to_optional_axis(x, _dim, None)
    assert _axis is not None  # Not supposed to happen.
    return _axis


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
    elif isinstance(axis, tuple):
        removed_axes = tuple(a % x.ndim for a in axis)
    else:
        removed_axes = (axis % x.ndim,)

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
    if isinstance(dim, Default):
        _dim: _Dim = f"dim_{len(dims)}"
    else:
        _dim = dim

    d = list(dims)
    d.insert(axis, _dim)
    return tuple(d)


def _raise_if_any_duplicate_dimensions(
    dims: _Dims, err_context: str = "This function"
) -> None:
    if len(set(dims)) < len(dims):
        repeated_dims = {d for d in dims if dims.count(d) > 1}
        raise ValueError(
            f"{err_context} cannot handle duplicate dimensions, "
            f"but dimensions {repeated_dims} appear more than once on this object's dims: {dims}"
        )


def _isnone(shape: _Shape) -> tuple[bool, ...]:
    # TODO: math.isnan should not be needed for array api, but dask still uses np.nan:
    return tuple(v is None and math.isnan(v) for v in shape)


def _get_broadcasted_dims(*arrays: NamedArray[Any, Any]) -> tuple[_Dims, _Shape]:
    """
    Get the expected broadcasted dims.

    Examples
    --------
    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> _get_broadcasted_dims(a)
    (('x', 'y', 'z'), (5, 3, 4))

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("y", "z"), np.zeros((3, 4)))
    >>> _get_broadcasted_dims(a, b)
    (('x', 'y', 'z'), (5, 3, 4))
    >>> _get_broadcasted_dims(b, a)
    (('x', 'y', 'z'), (5, 3, 4))

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("x", "y", "z"), np.zeros((0, 3, 4)))
    >>> _get_broadcasted_dims(a, b)
    (('x', 'y', 'z'), (5, 3, 4))

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("x", "y", "z"), np.zeros((1, 3, 4)))
    >>> _get_broadcasted_dims(a, b)
    (('x', 'y', 'z'), (5, 3, 4))

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> _get_broadcasted_dims(a, b)
    (('x', 'y', 'z'), (5, 3, 4))

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("x", "y", "z"), np.zeros((2, 3, 4)))
    >>> _get_broadcasted_dims(a, b)
    Traceback (most recent call last):
     ...
    ValueError: operands could not be broadcast together with dims = (('x', 'y', 'z'), ('x', 'y', 'z')) and shapes = ((5, 3, 4), (2, 3, 4))
    """
    dims = tuple(a.dims for a in arrays)
    shapes = tuple(a.shape for a in arrays)

    out_dims: _Dims = ()
    out_shape: _Shape = ()
    for d, sizes in zip(
        zip_longest(*map(reversed, dims), fillvalue=_default),
        zip_longest(*map(reversed, shapes), fillvalue=-1),
    ):
        _d = tuple(v for v in d if v is not _default)
        if any(_isnone(sizes)):
            # dim = None
            raise NotImplementedError("TODO: Handle None in shape, {shapes = }")
        else:
            dim = max(sizes)

        if any(i not in [-1, 0, 1, dim] for i in sizes) or len(_d) != 1:
            raise ValueError(
                f"operands could not be broadcast together with {dims = } and {shapes = }"
            )

        out_dims += (_d[0],)
        out_shape += (dim,)

    return out_dims[::-1], out_shape[::-1]
