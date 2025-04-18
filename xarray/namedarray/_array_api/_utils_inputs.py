from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._data_type_functions import iinfo
from xarray.namedarray._array_api._dtypes import _dtype_categories
from xarray.namedarray.core import NamedArray

_py_scalars = (bool, int, float, complex)


def _maybe_normalize_py_scalars(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    dtype_category: str,
    func_name: str,
) -> tuple[NamedArray[Any, Any], NamedArray[Any, Any]]:
    _allowed_dtypes = _dtype_categories[dtype_category]

    if isinstance(x1, _py_scalars):
        if isinstance(x2, _py_scalars):
            raise TypeError(
                f"Two scalars not allowed, got {type(x1) = } and {type(x2) =}"
            )
        # x2 must be an array
        if x2.dtype not in _allowed_dtypes:
            raise TypeError(
                f"Only {dtype_category} dtypes are allowed {func_name}. Got {x2.dtype}."
            )
        x1 = x2._promote_scalar(x1)

    elif isinstance(x2, _py_scalars):
        # x1 must be an array
        if x1.dtype not in _allowed_dtypes:
            raise TypeError(
                f"Only {dtype_category} dtypes are allowed {func_name}. Got {x1.dtype}."
            )
        x2 = x1._promote_scalar(x2)
    else:
        if x1.dtype not in _allowed_dtypes or x2.dtype not in _allowed_dtypes:
            raise TypeError(
                f"Only {dtype_category} dtypes are allowed in {func_name}(...). "
                f"Got {x1.dtype} and {x2.dtype}."
            )
    return x1, x2


def _promote_scalar(x, scalar: _py_scalars) -> NamedArray[Any, Any]:
    """
    Returns a promoted version of a Python scalar appropriate for use with
    operations on x.

    This may raise an OverflowError in cases where the scalar is an
    integer that is too large to fit in a NumPy integer dtype, or
    TypeError when the scalar type is incompatible with the dtype of x.
    """

    target_dtype = x.dtype
    # Note: Only Python scalar types that match the array dtype are
    # allowed.
    if isinstance(scalar, bool):
        if x.dtype not in _boolean_dtypes:
            raise TypeError("Python bool scalars can only be promoted with bool arrays")
    elif isinstance(scalar, int):
        if x.dtype in _boolean_dtypes:
            raise TypeError("Python int scalars cannot be promoted with bool arrays")
        if x.dtype in _integer_dtypes:
            info = iinfo(x.dtype)
            if not (info.min <= scalar <= info.max):
                raise OverflowError(
                    "Python int scalars must be within the bounds of the dtype for integer arrays"
                )
        # int + array(floating) is allowed
    elif isinstance(scalar, float):
        if x.dtype not in _floating_dtypes:
            raise TypeError(
                "Python float scalars can only be promoted with floating-point arrays."
            )
    elif isinstance(scalar, complex):
        if x.dtype not in _floating_dtypes:
            raise TypeError(
                "Python complex scalars can only be promoted with floating-point arrays."
            )
        # 1j * array(floating) is allowed
        if x.dtype in _real_floating_dtypes:
            target_dtype = _real_to_complex_map[x.dtype]
    else:
        raise TypeError("'scalar' must be a Python scalar")

    # Note: scalars are unconditionally cast to the same dtype as the
    # array.

    # Note: the spec only specifies integer-dtype/int promotion
    # behavior for integers within the bounds of the integer dtype.
    # Outside of those bounds we use the default NumPy behavior (either
    # cast or raise OverflowError).
    return NamedArray(
        (), xp.array(scalar, dtype=target_dtype._np_dtype), device=x.device
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
