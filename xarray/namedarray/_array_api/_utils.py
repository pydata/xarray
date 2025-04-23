from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Iterator
from itertools import zip_longest
from types import ModuleType
from typing import Any, TypeGuard, cast

from xarray.namedarray._typing import (
    _T,
    Default,
    _arrayapi,
    _Axes,
    _Axis,
    _AxisLike,
    _default,
    _Dim,
    _Dims,
    _DimsLike2,
    _dtype,
    _IndexKeys,
    _IndexKeysNoEllipsis,
    _Shape,
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
    """
    Attempt to get the namespace from dtype.

    Examples
    --------
    >>> import numpy as np
    >>> _get_namespace_dtype(None) is np
    True
    >>> _get_namespace_dtype(np.int64) is np
    True
    """
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
    if isinstance(axis, tuple):
        _axis = axis
    else:
        _axis = (axis,)

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


def _dim_to_axis(
    x: NamedArray[Any, Any],
    dim: _Dim | Default,
    axis: int,
    *,
    axis_default: int | None = None,
) -> int:
    """
    Convert dim to axis.

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x", "y", "z"), np.zeros((1, 2, 3)))
    >>> _dim_to_axis(x, _default, 0)
    0
    >>> _dim_to_axis(x, _default, -1)
    2
    >>> _dim_to_axis(x, "x", -1)
    0
    >>> _dim_to_axis(x, "z", 0)
    2

    Add a default value

    >>> _dim_to_axis(x, _default, 0, axis_default=-1)
    0
    >>> _dim_to_axis(x, "z", -1, axis_default=-1)
    2

    Defining both dims and axis value different to the default raises an error

    >>> _dim_to_axis(x, "x", 0, axis_default=-1)
    Traceback (most recent call last):
     ...
    ValueError: cannot supply both 'axis' and 'dim(s)' arguments
    """
    if axis_default is not None:
        a = None if axis is axis_default else axis
        _assert_either_dim_or_axis(dim, a)

    _dim: _Dim = x.dims[axis] if isinstance(dim, Default) else dim
    _axis = _dim_to_optional_axis(x, _dim, None)
    assert _axis is not None  # Not supposed to happen.
    return _axis


def _new_unique_dim_name(dims: _Dims, i: int | None = None) -> _Dim:
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


def _filter_next_false(
    predicate: Callable[..., bool], iterable: Iterable[_T]
) -> Iterator[_T]:
    """
    Make an iterator that filters elements from the iterable returning only those
    for which the predicate returns a false value for the second time.

    Variant on itertools.filterfalse but doesn't filter until the 2 second False.

    Examples
    --------
    >>> tuple(_filter_next_false(lambda x: x is not None, (1, None, 3, None, 4)))
    (1, None, 3, 4)
    """
    predicate_has_been_false = False
    for x in iterable:
        if not predicate(x):
            if predicate_has_been_false:
                continue
            predicate_has_been_false = True
        yield x


def _replace_ellipsis(key: _IndexKeys, ndim: int) -> _IndexKeysNoEllipsis:
    """
    Replace ... with slices, :, : ,:

    Examples
    --------
    >>> _replace_ellipsis((3, Ellipsis, 2), 4)
    (3, slice(None, None, None), slice(None, None, None), 2)
    >>> _replace_ellipsis((Ellipsis, None), 2)
    (slice(None, None, None), slice(None, None, None), None)
    >>> _replace_ellipsis((Ellipsis, None, Ellipsis), 2)
    (slice(None, None, None), slice(None, None, None), None)
    """
    # https://github.com/dask/dask/blob/569abf8e8048cbfb1d750900468dda0de7c56358/dask/array/slicing.py#L701
    key = tuple(_filter_next_false(lambda x: x is not Ellipsis, key))
    expanded_dims = sum(i is None for i in key)
    extra_dimensions = ndim - (len(key) - expanded_dims - 1)
    replaced_slices = (slice(None, None, None),) * extra_dimensions

    out: _IndexKeysNoEllipsis = ()
    for k in key:
        if k is Ellipsis:
            out += replaced_slices
        else:
            out += (k,)
    return out


def _check_indexing_dims(original_dims: _Dims, indexing_dims: _Dims) -> None:
    """
    Check if dims do not match. If it does not, something is likely wrong.

    Normally NamedArray should raise an error when it doesn't match but since indexing
    arrays is also a Array API feature we can only warn the user.
    """
    if original_dims != indexing_dims:
        warnings.warn(
            (
                "Dimension name of indexing array does not match.\n"
                f"{original_dims=} != {indexing_dims=}"
            ),
            stacklevel=2,
        )


def _dims_from_tuple_indexing(dims: _Dims, key: _IndexKeys) -> _Dims:
    """
    Get the expected dims when using tuples in __getitem__.

    Examples
    --------
    >>> _dims_from_tuple_indexing(("x", "y", "z"), ())
    ('x', 'y', 'z')
    >>> _dims_from_tuple_indexing(("x", "y", "z"), (0,))
    ('y', 'z')
    >>> _dims_from_tuple_indexing(("x", "y", "z"), (0, 0))
    ('z',)
    >>> _dims_from_tuple_indexing(("x", "y", "z"), (0, 0, 0))
    ()
    >>> _dims_from_tuple_indexing(("x", "y", "z"), (0, 0, 0, ...))
    ()
    >>> _dims_from_tuple_indexing(("x", "y", "z"), (0, ...))
    ('y', 'z')
    >>> _dims_from_tuple_indexing(("x", "y", "z"), (0, ..., 0))
    ('y',)
    >>> _dims_from_tuple_indexing(("x", "y", "z"), (0, slice(0)))
    ('y', 'z')
    >>> _dims_from_tuple_indexing(("x", "y"), (None,))
    ('dim_2', 'x', 'y')
    >>> _dims_from_tuple_indexing(("x", "y"), (0, None, None, 0))
    ('dim_1', 'dim_2')
    >>> _dims_from_tuple_indexing(("x",), (..., 0))
    ()

    Indexing array

    >>> import numpy as np
    >>> key = (0, NamedArray((), np.array(0, dtype=int)))
    >>> _dims_from_tuple_indexing(("x", "y", "z"), key)
    ('z',)
    >>> key = (0, NamedArray(("y",), np.array([0], dtype=int)))
    >>> _dims_from_tuple_indexing(("x", "y", "z"), key)
    ('y', 'z')
    """
    key_no_ellipsis = _replace_ellipsis(key, len(dims))

    new_dims = list(dims)
    j = 0  # keeps track of where the original dims are.
    for i, k in enumerate(key_no_ellipsis):
        if k is None:
            # None adds 1 dimension:
            new_dims.insert(j, _new_unique_dim_name(tuple(new_dims)))
            j += 1
        elif isinstance(k, int):
            # Integer removes 1 dimension:
            new_dims.pop(j)
        elif isinstance(k, slice):
            # Slice retains the dimension.
            j += 1
        elif isinstance(k, NamedArray):
            if len(k.dims) == 0:
                # if 0 dim, removes 1 dimension
                new_dims.pop(j)
            else:
                # same size retains the dimension:
                _check_indexing_dims(dims[i : i + 1], k.dims)
                j += 1

    return tuple(new_dims)


def _atleast1d_dims(dims: _Dims) -> _Dims:
    """
    Set dims atleast 1-dimensional.

    Examples
    --------
    >>> _atleast1d_dims(())
    ('dim_0',)
    >>> _atleast1d_dims(("x",))
    ('x',)
    >>> _atleast1d_dims(("x", "y"))
    ('x', 'y')
    """
    return (_new_unique_dim_name(dims),) if len(dims) < 1 else dims


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


def _move_dims(dims: _Dims, source: _AxisLike, destination: _AxisLike) -> _Dims:
    """
    Move dims position in source to the destination position.

    Examples
    --------
    >>> _move_dims(("x", "y", "z"), 0, -1)
    ('y', 'z', 'x')
    >>> _move_dims(("x", "y", "z"), -1, 0)
    ('z', 'x', 'y')
    >>> _move_dims(("x", "y", "z"), (0, 1), (-1, -2))
    ('z', 'y', 'x')
    >>> _move_dims(("x", "y", "z"), (0, 1, 2), (-1, -2, -3))
    ('z', 'y', 'x')
    >>> _move_dims(("x", "y", "z"), 0, 1)
    ('y', 'x', 'z')
    """
    _ndim = len(dims)
    _source = _normalize_axis_tuple(source, _ndim)
    _destination = _normalize_axis_tuple(destination, _ndim)
    order = [n for n in range(_ndim) if n not in _source]
    for dest, src in sorted(zip(_destination, _source, strict=True)):
        order.insert(dest, src)

    return tuple(dims[i] for i in order)


def _reduce_dims(dims: _Dims, *, axis: _AxisLike | None, keepdims: bool) -> _Dims:
    """
    Reduce dims according to axis.

    Examples
    --------
    >>> _reduce_dims(("x", "y", "z"), axis=None, keepdims=False)
    ()
    >>> _reduce_dims(("x", "y", "z"), axis=1, keepdims=False)
    ('x', 'z')
    >>> _reduce_dims(("x", "y", "z"), axis=-1, keepdims=False)
    ('x', 'y')

    keepdims retains the same dims

    >>> _reduce_dims(("x", "y", "z"), axis=-1, keepdims=True)
    ('x', 'y', 'z')
    """
    if keepdims:
        return dims

    ndim = len(dims)
    if axis is None:
        _axis = tuple(v for v in range(ndim))
    else:
        _axis = _normalize_axis_tuple(axis, ndim)

    k: _IndexKeys = (slice(None),) * ndim
    key = list(k)
    for v in _axis:
        key[v] = 0

    return _dims_from_tuple_indexing(dims, tuple(key))


def _squeeze_dims(dims: _Dims, shape: _Shape, axis: _AxisLike) -> _Dims:
    """
    Squeeze dims.

    Examples
    --------
    >>> _squeeze_dims(("x", "y", "z"), (0, 2, 1), (0, 2))
    ('y',)
    """
    sizes = dict(zip(dims, shape, strict=True))
    for a in _normalize_axis_tuple(axis, len(dims)):
        d = dims[a]
        if sizes[d] < 2:
            sizes.pop(d)

    return tuple(sizes.keys())


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
    """
    Check if each element has None.

    Examples
    --------
    >>> _isnone((1, 2, 3))
    (False, False, False)
    >>> _isnone((1, 2, None))
    (False, False, True)

    Dask uses np.nan and should be handled the same way:

    >>> import numpy as np
    >>> _isnone((1, 2, np.nan))
    (False, False, True)

    >>> _isnone((1, 2, math.nan))
    (False, False, True)
    """
    # TODO: math.isnan should not be needed for array api, but dask still uses np.nan:
    return tuple(v is None or math.isnan(v) for v in shape)


def _broadcast_dims(*arrays: NamedArray[Any, Any]) -> tuple[_Dims, _Shape]:
    """
    Get the expected broadcasted dims.

    Examples
    --------
    >>> import numpy as np
    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> _broadcast_dims(a)
    (('x', 'y', 'z'), (5, 3, 4))

    Broadcasting 0- and 1-sized dims

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("x", "y", "z"), np.zeros((0, 3, 4)))
    >>> _broadcast_dims(a, b)
    (('x', 'y', 'z'), (5, 3, 4))
    >>> _broadcast_dims(b, a)
    (('x', 'y', 'z'), (5, 3, 4))

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("x", "y", "z"), np.zeros((1, 3, 4)))
    >>> _broadcast_dims(a, b)
    (('x', 'y', 'z'), (5, 3, 4))

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> _broadcast_dims(a, b)
    (('x', 'y', 'z'), (5, 3, 4))

    Broadcasting different dims

    >>> a = NamedArray(("x",), np.zeros((5,)))
    >>> b = NamedArray(("y",), np.zeros((3,)))
    >>> _broadcast_dims(a, b)
    (('x', 'y'), (5, 3))

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("y", "z"), np.zeros((3, 4)))
    >>> _broadcast_dims(a, b)
    (('x', 'y', 'z'), (5, 3, 4))
    >>> _broadcast_dims(b, a)
    (('x', 'y', 'z'), (5, 3, 4))


    # Errors

    >>> a = NamedArray(("x", "y", "z"), np.zeros((5, 3, 4)))
    >>> b = NamedArray(("x", "y", "z"), np.zeros((2, 3, 4)))
    >>> _broadcast_dims(a, b)
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
