from __future__ import annotations

import functools
from typing import Any

import numpy as np
from pandas.api.types import is_extension_array_dtype

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
    if isdtype(dtype, "real floating"):
        dtype_ = dtype
        fill_value = np.nan
    elif isdtype(dtype, np.timedelta64):
        # See https://github.com/numpy/numpy/issues/10685
        # np.timedelta64 is a subclass of np.integer
        # Check np.timedelta64 before np.integer
        fill_value = np.timedelta64("NaT")
        dtype_ = dtype
    elif isdtype(dtype, "integral"):
        dtype_ = np.float32 if dtype.itemsize <= 2 else np.float64
        fill_value = np.nan
    elif isdtype(dtype, "complex floating"):
        dtype_ = dtype
        fill_value = np.nan + np.nan * 1j
    elif isdtype(dtype, np.datetime64):
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
    if isdtype(dtype, "real floating"):
        return np.inf

    if isdtype(dtype, "integral"):
        if max_for_int:
            return np.iinfo(dtype).max
        else:
            return np.inf

    if isdtype(dtype, "complex floating"):
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
    if isdtype(dtype, "real floating"):
        return -np.inf

    if isdtype(dtype, "integral"):
        if min_for_int:
            return np.iinfo(dtype).min
        else:
            return -np.inf

    if isdtype(dtype, "complex floating"):
        return -np.inf - 1j * np.inf

    return NINF


def is_datetime_like(dtype):
    """Check if a dtype is a subclass of the numpy datetime types"""
    return isdtype(dtype, (np.datetime64, np.timedelta64))


def isdtype(dtype, kind, xp=None):
    array_api_names = {
        "bool": np.bool_,
        "signed integer": np.signedinteger,
        "unsigned integer": np.unsignedinteger,
        "integral": np.integer,
        "real floating": np.floating,
        "complex floating": np.complexfloating,
        "numeric": np.number,
    }
    numpy_names = {
        "object": np.object_,
        "character": np.character,
        "string": np.str_,
    }
    long_names = array_api_names | numpy_names

    def is_numpy_kind(kind):
        return (isinstance(kind, str) and kind in numpy_names) or (
            isinstance(kind, type) and issubclass(kind, (np.dtype, np.generic))
        )

    def split_numpy_kinds(kinds):
        numpy_kinds = tuple(kind for kind in kinds if is_numpy_kind(kind))
        non_numpy_kinds = tuple(kind for kind in kinds if not is_numpy_kind(kind))

        return numpy_kinds, non_numpy_kinds

    def translate_kind(kind):
        if isinstance(kind, str):
            translated = long_names.get(kind)
            if translated is None:
                raise ValueError(f"unknown kind: {kind!r}")

            return translated
        elif isinstance(kind, type) and issubclass(kind, np.generic):
            return kind
        else:
            raise TypeError(f"invalid type of kind: {kind!r}")

    def numpy_isdtype(dtype, kinds):
        translated_kinds = [translate_kind(kind) for kind in kinds]
        if isinstance(dtype, np.generic):
            return any(isinstance(dtype, kind) for kind in translated_kinds)
        else:
            return any(np.issubdtype(dtype, kind) for kind in translated_kinds)

    def pandas_isdtype(dtype, kinds):
        return any(
            isinstance(dtype, kind) if isinstance(kind, type) else False
            for kind in kinds
        )

    if xp is None:
        xp = np

    if not isinstance(kind, tuple):
        kinds = (kind,)
    else:
        kinds = kind

    if isinstance(dtype, np.dtype):
        return numpy_isdtype(dtype, kinds)
    elif is_extension_array_dtype(dtype):
        return pandas_isdtype(dtype, kinds)
    else:
        numpy_kinds, non_numpy_kinds = split_numpy_kinds(kinds)

        if not non_numpy_kinds:
            return False

        return xp.isdtype(dtype, non_numpy_kinds)


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
