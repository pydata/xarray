from __future__ import annotations

import importlib
import sys
import typing

import numpy as np

if typing.TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

# temporary placeholder for indicating an array api compliant type.
# hopefully in the future we can narrow this down more
T_DuckArray = typing.TypeVar("T_DuckArray", bound=typing.Any)


def module_available(module: str) -> bool:
    """Checks whether a module is installed without importing it.

    Use this for a lightweight check and lazy imports.

    Parameters
    ----------
    module : str
        Name of the module.

    Returns
    -------
    available : bool
        Whether the module is installed.
    """
    return importlib.util.find_spec(module) is not None


def is_dask_collection(x: typing.Any) -> bool:
    if module_available("dask"):
        from dask.base import is_dask_collection

        return is_dask_collection(x)
    return False


def is_duck_array(value: typing.Any) -> TypeGuard[T_DuckArray]:
    if isinstance(value, np.ndarray):
        return True
    return (
        hasattr(value, "ndim")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and (
            (hasattr(value, "__array_function__") and hasattr(value, "__array_ufunc__"))
            or hasattr(value, "__array_namespace__")
        )
    )


def is_duck_dask_array(x: typing.Any) -> bool:
    return is_duck_array(x) and is_dask_collection(x)


def to_0d_object_array(value: typing.Any) -> np.ndarray[object]:
    """Given a value, wrap it in a 0-D numpy.ndarray with dtype=object."""
    result = np.empty((), dtype=object)
    result[()] = value
    return result
