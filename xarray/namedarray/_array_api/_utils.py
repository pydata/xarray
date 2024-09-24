from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import zip_longest
from types import ModuleType
from typing import Any, TypeGuard, cast

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
    _IndexKeys,
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

    >>> _infer_dims((), ())
    ()
    >>> _infer_dims((1,), "x")
    ('x',)
    >>> _infer_dims((1,), None)
    (None,)
    >>> _infer_dims((1,), ("x",))
    ('x',)
    >>> _infer_dims((1, 3), ("x",))
    ('dim_0', 'x')
    >>> _infer_dims((1, 1, 3), ("x",))
    ('dim_1', 'dim_0', 'x')
    """
    if isinstance(dims, Default):
        ndim = len(shape)
        return tuple(f"dim_{ndim - 1 - n}" for n in range(ndim))

    _dims = _normalize_dimensions(dims)
    diff = len(shape) - len(_dims)
    if diff > 0:
        # TODO: Leads to ('dim_0', 'x'), should it be ('dim_1', 'x')?
        return _infer_dims(shape[:diff], _default) + _dims
    else:
        return _dims


def _normalize_axis_index(axis: int, ndim: int) -> int:
    """
    Normalize axis index to positive values.

    Parameters
    ----------
    axis : int
        The un-normalized index of the axis. Can be negative
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against

    Returns
    -------
    normalized_axis : int
        The normalized axis index, such that `0 <= normalized_axis < ndim`


    Examples
    --------
    >>> _normalize_axis_index(0, ndim=3)
    0
    >>> _normalize_axis_index(1, ndim=3)
    1
    >>> _normalize_axis_index(2, ndim=3)
    2
    >>> _normalize_axis_index(-1, ndim=3)
    2
    >>> _normalize_axis_index(-2, ndim=3)
    1
    >>> _normalize_axis_index(-3, ndim=3)
    0

    Errors

    >>> _normalize_axis_index(3, ndim=3)
    Traceback (most recent call last):
     ...
    ValueError: axis 3 is out of bounds for array of dimension 3
    >>> _normalize_axis_index(-4, ndim=3)
    Traceback (most recent call last):
     ...
    ValueError: axis -4 is out of bounds for array of dimension 3
    """

    if -ndim > axis or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")

    return axis % ndim


def _normalize_axis_tuple(
    axis: _AxisLike,
    ndim: int,
    argname: str | None = None,
    allow_duplicate: bool = False,
) -> _Axes:
    """
    Normalizes an axis argument into a tuple of non-negative integer axes.

    This handles shorthands such as ``1`` and converts them to ``(1,)``,
    as well as performing the handling of negative indices covered by
    `normalize_axis_index`.

    By default, this forbids axes from being specified multiple times.


    Parameters
    ----------
    axis : int, iterable of int
        The un-normalized index or indices of the axis.
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against.
    argname : str, optional
        A prefix to put before the error message, typically the name of the
        argument.
    allow_duplicate : bool, optional
        If False, the default, disallow an axis from being specified twice.

    Returns
    -------
    normalized_axes : tuple of int
        The normalized axis index, such that `0 <= normalized_axis < ndim`

    Raises
    ------
    AxisError
        If any axis provided is out of range
    ValueError
        If an axis is repeated
    """
    if not isinstance(axis, tuple):
        _axis = (axis,)
    else:
        _axis = axis

    # Going via an iterator directly is slower than via list comprehension.
    _axis = tuple([_normalize_axis_index(ax, ndim) for ax in _axis])
    if not allow_duplicate and len(set(_axis)) != len(_axis):
        if argname:
            raise ValueError(f"repeated axis in `{argname}` argument")
        else:
            raise ValueError("repeated axis")
    return _axis


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
    >>> import numpy as np
    >>> x = NamedArray(("x", "y", "z"), np.zeros((1, 2, 3)))
    >>> _dims_to_axis(x, ("y", "x"), None)
    (1, 0)
    >>> _dims_to_axis(x, ("y",), None)
    (1,)
    >>> _dims_to_axis(x, _default, 0)
    (0,)
    >>> type(_dims_to_axis(x, _default, None))
    <class 'NoneType'>

    Normalizes negative integers

    >>> _dims_to_axis(x, _default, -1)
    (2,)
    >>> _dims_to_axis(x, _default, (-2, -1))
    (1, 2)

    Using Hashable dims

    >>> x = NamedArray(("x", None), np.zeros((1, 2)))
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
                axis += (x.dims.index(dim),)
            except ValueError as err:
                raise ValueError(
                    f"{dim!r} not found in array dimensions {x.dims!r}"
                ) from err
        return axis

    if axis is None:
        return axis

    return _normalize_axis_tuple(axis, x.ndim)


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


def _new_unique_dim_name(dims: _Dims, i=None) -> _Dim:
    """
    Get a new unique dimension name.

    Examples
    --------
    >>> _new_unique_dim_name(())
    'dim_0'
    >>> _new_unique_dim_name(("dim_0",))
    'dim_1'
    >>> _new_unique_dim_name(("dim_1", "dim_0"))
    'dim_2'
    >>> _new_unique_dim_name(("dim_0", "dim_2"))
    'dim_3'
    >>> _new_unique_dim_name(("dim_3", "dim_2"))
    'dim_4'
    """
    i = len(dims) if i is None else i
    _dim: _Dim = f"dim_{i}"
    return _new_unique_dim_name(dims, i=i + 1) if _dim in dims else _dim


def _insert_dim(dims: _Dims, dim: _Dim | Default, axis: _Axis) -> _Dims:
    if isinstance(dim, Default):
        _dim: _Dim = _new_unique_dim_name(dims)
    else:
        _dim = dim

    d = list(dims)
    d.insert(axis, _dim)
    return tuple(d)


def _dims_from_tuple_indexing(dims: _Dims, key: _IndexKeys) -> _Dims:
    """
    Get the expected dims when using tuples in __getitem__.

    Examples
    --------
    >>> _dims_from_tuple_indexing(("x", "y"), ())
    ('x', 'y')
    >>> _dims_from_tuple_indexing(("x", "y"), (0,))
    ('y',)
    >>> _dims_from_tuple_indexing(("x", "y"), (0, 0))
    ()
    >>> _dims_from_tuple_indexing(("x", "y"), (0, ...))
    ('y',)
    >>> _dims_from_tuple_indexing(("x", "y"), (0, slice(0)))
    ('y',)
    >>> _dims_from_tuple_indexing(("x", "y"), (None,))
    ('dim_2', 'x', 'y')
    >>> _dims_from_tuple_indexing(("x", "y"), (0, None, None, 0))
    ('dim_1', 'dim_2')
    """
    _dims = list(dims)
    j = 0
    for k in key:
        if k is None:
            # None adds 1 dimension:
            _dims.insert(j, _new_unique_dim_name(tuple(_dims)))
        elif isinstance(k, int):
            # Integer removes 1 dimension:
            _dims.pop(j)
            j -= 1
        else:
            # Slices and Ellipsis maintains same dimensions:
            pass
        j += 1
    return tuple(_dims)


def _flatten_dims(dims: _Dims) -> _Dims:
    """
    Flatten multidimensional dims to 1-dimensional.

    Examples
    --------
    >>> _flatten_dims(())
    ()
    >>> _flatten_dims(("x",))
    ('x',)
    >>> _flatten_dims(("x", "y"))
    (('x', 'y'),)
    """
    return (dims,) if len(dims) > 1 else dims




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
    >>> import numpy as np
    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> _get_broadcasted_dims(a)
    (('x', 'y', 'z'), (5, 3, 4))

    Broadcasting 0- and 1-sized dims

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("x", "y", "z"), np.zeros((0, 3, 4)))
    >>> _get_broadcasted_dims(a, b)
    (('x', 'y', 'z'), (5, 3, 4))
    >>> _get_broadcasted_dims(b, a)
    (('x', 'y', 'z'), (5, 3, 4))

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("x", "y", "z"), np.zeros((1, 3, 4)))
    >>> _get_broadcasted_dims(a, b)
    (('x', 'y', 'z'), (5, 3, 4))

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> _get_broadcasted_dims(a, b)
    (('x', 'y', 'z'), (5, 3, 4))

    Broadcasting different dims

    >>> a = NamedArray(("x",), np.zeros((5,)))
    >>> b = NamedArray(("y",), np.zeros((3,)))
    >>> _get_broadcasted_dims(a, b)
    (('x', 'y'), (5, 3))

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("y", "z"), np.zeros((3, 4)))
    >>> _get_broadcasted_dims(a, b)
    (('x', 'y', 'z'), (5, 3, 4))
    >>> _get_broadcasted_dims(b, a)
    (('x', 'y', 'z'), (5, 3, 4))


    # Errors

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("x", "y", "z"), np.zeros((2, 3, 4)))
    >>> _get_broadcasted_dims(a, b)
    Traceback (most recent call last):
     ...
    ValueError: operands could not be broadcast together with dims = (('x', 'y', 'z'), ('x', 'y', 'z')) and shapes = ((5, 3, 4), (2, 3, 4))
    """
    DEFAULT_SIZE = -1
    BROADCASTABLE_SIZES = (0, 1)
    BROADCASTABLE_SIZES_OR_DEFAULT = BROADCASTABLE_SIZES + (DEFAULT_SIZE,)

    arrays_dims = tuple(a.dims for a in arrays)
    arrays_shapes = tuple(a.shape for a in arrays)

    sizes: dict[Any, Any] = {}
    for dims, shape in zip(
        zip_longest(*map(reversed, arrays_dims), fillvalue=_default),
        zip_longest(*map(reversed, arrays_shapes), fillvalue=DEFAULT_SIZE),
        strict=False,
    ):
        for d, s in zip(reversed(dims), reversed(shape), strict=False):
            if isinstance(d, Default):
                continue

            if s is None:
                raise NotImplementedError("TODO: Handle None in shape, {shapes = }")

            s_prev = sizes.get(d, DEFAULT_SIZE)
            if not (
                s == s_prev
                or any(v in BROADCASTABLE_SIZES_OR_DEFAULT for v in (s, s_prev))
            ):
                raise ValueError(
                    "operands could not be broadcast together with "
                    f"dims = {arrays_dims} and shapes = {arrays_shapes}"
                )

            sizes[d] = max(s, s_prev)

    out_dims: _Dims = tuple(reversed(sizes.keys()))
    out_shape: _Shape = tuple(reversed(sizes.values()))
    return out_dims, out_shape


if __name__ == "__main__":
    import doctest

    doctest.testmod()
