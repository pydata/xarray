from __future__ import annotations

import functools
from typing import Any

import numpy as np

from xarray.core import utils

# Use as a sentinel value to indicate a dtype appropriate NA value.
NA = utils.ReprObject("<NA>")


@functools.total_ordering
class AlwaysGreaterThan:
    def __gt__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))


@functools.total_ordering
class AlwaysLessThan:
    def __lt__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))


# Equivalence to np.inf (-np.inf) for object-type
INF = AlwaysGreaterThan()
NINF = AlwaysLessThan()


# Pairs of types that, if both found, should be promoted to object dtype
# instead of following NumPy's own type-promotion rules. These type promotion
# rules match pandas instead. For reference, see the NumPy type hierarchy:
# https://numpy.org/doc/stable/reference/arrays.scalars.html
PROMOTE_TO_OBJECT: tuple[tuple[type[np.generic], type[np.generic]], ...] = (
    (np.number, np.character),  # numpy promotes to character
    (np.bool_, np.character),  # numpy promotes to character
    (np.bytes_, np.str_),  # numpy promotes to unicode
)


def maybe_promote(dtype: np.dtype) -> tuple[np.dtype, Any]:
    """Simpler equivalent of pandas.core.common._maybe_promote

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    dtype : Promoted dtype that can hold missing values.
    fill_value : Valid missing value for the promoted dtype.
    """
    # N.B. these casting rules should match pandas
    dtype_: np.typing.DTypeLike
    fill_value: Any
    if np.issubdtype(dtype, np.floating):
        dtype_ = dtype
        fill_value = np.nan
    elif np.issubdtype(dtype, np.timedelta64):
        # See https://github.com/numpy/numpy/issues/10685
        # np.timedelta64 is a subclass of np.integer
        # Check np.timedelta64 before np.integer
        fill_value = np.timedelta64("NaT")
        dtype_ = dtype
    elif np.issubdtype(dtype, np.integer):
        dtype_ = np.float32 if dtype.itemsize <= 2 else np.float64
        fill_value = np.nan
    elif np.issubdtype(dtype, np.complexfloating):
        dtype_ = dtype
        fill_value = np.nan + np.nan * 1j
    elif np.issubdtype(dtype, np.datetime64):
        dtype_ = dtype
        fill_value = np.datetime64("NaT")
    else:
        dtype_ = object
        fill_value = np.nan

    dtype_out = np.dtype(dtype_)
    fill_value = dtype_out.type(fill_value)
    return dtype_out, fill_value


NAT_TYPES = {np.datetime64("NaT").dtype, np.timedelta64("NaT").dtype}


def get_fill_value(dtype):
    """Return an appropriate fill value for this dtype.

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    fill_value : Missing value corresponding to this dtype.
    """
    _, fill_value = maybe_promote(dtype)
    return fill_value


def get_pos_infinity(dtype, max_for_int=False):
    """Return an appropriate positive infinity for this dtype.

    Parameters
    ----------
    dtype : np.dtype
    max_for_int : bool
        Return np.iinfo(dtype).max instead of np.inf

    Returns
    -------
    fill_value : positive infinity value corresponding to this dtype.
    """
    if issubclass(dtype.type, np.floating):
        return np.inf

    if issubclass(dtype.type, np.integer):
        if max_for_int:
            return np.iinfo(dtype).max
        else:
            return np.inf

    if issubclass(dtype.type, np.complexfloating):
        return np.inf + 1j * np.inf

    return INF


def get_neg_infinity(dtype, min_for_int=False):
    """Return an appropriate positive infinity for this dtype.

    Parameters
    ----------
    dtype : np.dtype
    min_for_int : bool
        Return np.iinfo(dtype).min instead of -np.inf

    Returns
    -------
    fill_value : positive infinity value corresponding to this dtype.
    """
    if issubclass(dtype.type, np.floating):
        return -np.inf

    if issubclass(dtype.type, np.integer):
        if min_for_int:
            return np.iinfo(dtype).min
        else:
            return -np.inf

    if issubclass(dtype.type, np.complexfloating):
        return -np.inf - 1j * np.inf

    return NINF


def is_datetime_like(dtype):
    """Check if a dtype is a subclass of the numpy datetime types"""
    return np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)


def isdtype(dtype, kind, xp=None):
    array_api_names = {
        "bool": "b",
        "signed integer": "i",
        "unsigned integer": "u",
        "integral": "ui",
        "real floating": "f",
        "complex floating": "c",
        "numeric": "uifc",
    }
    numpy_names = {
        "object": "O",
        "character": "U",
        "string": "S",
    }
    long_names = array_api_names | numpy_names

    def issubdtype(dtype, kind):
        if isinstance(dtype, np.dtype):
            return np.issubdtype(dtype, kind)
        else:
            # TODO (keewis): find a better way to compare dtypes (like pandas extension dtypes)
            return dtype == kind

    def compare_dtype(dtype, kind):
        if isinstance(kind, np.dtype):
            return dtype == kind
        elif isinstance(kind, str):
            return dtype.kind in long_names.get(kind, kind)
        elif isinstance(kind, type) and issubclass(kind, (np.dtype, np.generic)):
            return issubdtype(dtype, kind)
        else:
            raise TypeError(f"unknown dtype kind: {kind}")

    def is_numpy_kind(kind):
        return (isinstance(kind, str) and kind in numpy_names) or (
            isinstance(kind, type) and issubclass(kind, (np.dtype, np.generic))
        )

    def split_numpy_kinds(kinds):
        if not isinstance(kinds, tuple):
            kinds = (kinds,)

        numpy_kinds = tuple(kind for kind in kinds if is_numpy_kind(kind))
        non_numpy_kinds = tuple(kind for kind in kinds if not is_numpy_kind(kind))

        return numpy_kinds, non_numpy_kinds

    numpy_kinds, non_numpy_kinds = split_numpy_kinds(kind)
    if xp in (None, np) or not hasattr(xp, "isdtype"):
        # need to take this path to allow checking for datetime/timedelta/strings
        return any(compare_dtype(dtype, k) for k in (numpy_kinds + non_numpy_kinds))
    elif non_numpy_kinds:
        return xp.isdtype(dtype, non_numpy_kinds)
    else:
        # can't compare numpy kinds with non-numpy dtypes
        return False


def result_type(
    *arrays_and_dtypes: np.typing.ArrayLike | np.typing.DTypeLike,
) -> np.dtype:
    """Like np.result_type, but with type promotion rules matching pandas.

    Examples of changed behavior:
    number + string -> object (not string)
    bytes + unicode -> object (not unicode)

    Parameters
    ----------
    *arrays_and_dtypes : list of arrays and dtypes
        The dtype is extracted from both numpy and dask arrays.

    Returns
    -------
    numpy.dtype for the result.
    """
    from xarray.core.duck_array_ops import get_array_namespace

    namespaces = {get_array_namespace(t) for t in arrays_and_dtypes}
    non_numpy = namespaces - {np}
    if non_numpy:
        [xp] = non_numpy
    else:
        xp = np

    types = {xp.result_type(t) for t in arrays_and_dtypes}

    if any(isinstance(t, np.dtype) for t in types):
        # only check if there's numpy dtypes – the array API does not
        # define the types we're checking for
        for left, right in PROMOTE_TO_OBJECT:
            if any(isdtype(t, left, xp=xp) for t in types) and any(
                isdtype(t, right, xp=xp) for t in types
            ):
                return xp.dtype(object)

    return xp.result_type(*arrays_and_dtypes)
