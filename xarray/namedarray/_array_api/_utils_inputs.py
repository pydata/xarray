from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._dtypes import (
    _dtype_categories,
)
from xarray.namedarray.core import NamedArray

_py_scalars = (bool, int, float, complex)


# def _check_args(
#     x1: NamedArray[Any, Any] | bool | int | float | complex,
#     x2: NamedArray[Any, Any] | bool | int | float | complex,
#     # dtype_category: str,
#     # func_name: str,
# ) -> tuple[NamedArray[Any, Any], NamedArray[Any, Any]]:
#     """

#     Examples
#     --------
#     >>> import numpy as np
#     >>> x1 = NamedArray(("x",), np.array([1, 2, 3]))
#     >>> x2 = 2
#     >>> _normalize_args(x1, x2)
#     """
#     if isinstance(x1, _py_scalars):
#         if isinstance(x2, _py_scalars):
#             raise TypeError(
#                 f"Two scalars not allowed, got {type(x1) = } and {type(x2) =}"
#             )
#         # x2 must be an array
#         if x2.dtype not in _allowed_dtypes:
#             raise TypeError(
#                 f"Only {dtype_category} dtypes are allowed {func_name}. Got {x2.dtype}."
#             )
#         x1 = _promote_scalar(x2, x1)

#     elif isinstance(x2, _py_scalars):
#         # x1 must be an array
#         if x1.dtype not in _allowed_dtypes:
#             raise TypeError(
#                 f"Only {dtype_category} dtypes are allowed {func_name}. Got {x1.dtype}."
#             )
#         x2 = _promote_scalar(x1, x2)
#     elif x1.dtype not in _allowed_dtypes or x2.dtype not in _allowed_dtypes:
#         raise TypeError(
#             f"Only {dtype_category} dtypes are allowed in {func_name}(...). "
#             f"Got {x1.dtype} and {x2.dtype}."
#         )
#     return x1, x2


# def _maybe_normalize_py_scalars(
#     x1: NamedArray[Any, Any] | bool | int | float | complex,
#     x2: NamedArray[Any, Any] | bool | int | float | complex,
#     dtype_category: str,
#     func_name: str,
# ) -> tuple[NamedArray[Any, Any], NamedArray[Any, Any]]:
#     _allowed_dtypes = _dtype_categories[dtype_category]

#     if isinstance(x1, _py_scalars):
#         if isinstance(x2, _py_scalars):
#             raise TypeError(
#                 f"Two scalars not allowed, got {type(x1) = } and {type(x2) =}"
#             )
#         # x2 must be an array
#         if x2.dtype not in _allowed_dtypes:
#             raise TypeError(
#                 f"Only {dtype_category} dtypes are allowed {func_name}. Got {x2.dtype}."
#             )
#         x1 = _promote_scalar(x2, x1)

#     elif isinstance(x2, _py_scalars):
#         # x1 must be an array
#         if x1.dtype not in _allowed_dtypes:
#             raise TypeError(
#                 f"Only {dtype_category} dtypes are allowed {func_name}. Got {x1.dtype}."
#             )
#         x2 = _promote_scalar(x1, x2)
#     elif x1.dtype not in _allowed_dtypes or x2.dtype not in _allowed_dtypes:
#         raise TypeError(
#             f"Only {dtype_category} dtypes are allowed in {func_name}(...). "
#             f"Got {x1.dtype} and {x2.dtype}."
#         )
#     return x1, x2


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


# def _split_args(
#     *arrays_dtypes_scalars: NamedArray[Any, Any]
#     | int
#     | float
#     | complex
#     | bool
#     | _dtype[Any],
# ):
#     arrays: list[NamedArray[Any, Any]] = []
#     dtypes: list[_dtype[Any]] = []
#     scalars: list[bool | int | float | complex] = []
#     for a in arrays_dtypes_scalars:
#         if isinstance(a, NamedArray):
#             arrays.append(a)
#         elif isinstance(a, (bool, int, float, complex)):
#             scalars.append(a)
#         else:
#             dtypes.append(a)

#     # if not dtypes:
#     #     # Need at least 1 array or dtype to retrieve namespace otherwise need to use
#     #     # the default namespace.
#     #     raise ValueError("at least one array or dtype is required")

#     return arrays, dtypes, scalars


if __name__ == "__main__":
    import doctest

    doctest.testmod()
