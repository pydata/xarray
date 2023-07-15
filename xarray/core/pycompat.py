from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from packaging.version import Version

from xarray.core.utils import is_duck_array, is_scalar, module_available

integer_types = (int, np.integer)

if TYPE_CHECKING:
    ModType = Literal["dask", "pint", "cupy", "sparse", "cubed"]
    DuckArrayTypes = tuple[type[Any], ...]  # TODO: improve this? maybe Generic


class DuckArrayModule:
    """
    Solely for internal isinstance and version checks.

    Motivated by having to only import pint when required (as pint currently imports xarray)
    https://github.com/pydata/xarray/pull/5561#discussion_r664815718
    """

    module: ModuleType | None
    version: Version
    type: DuckArrayTypes
    available: bool

    def __init__(self, mod: ModType) -> None:
        duck_array_module: ModuleType | None
        duck_array_version: Version
        duck_array_type: DuckArrayTypes
        try:
            duck_array_module = import_module(mod)
            duck_array_version = Version(duck_array_module.__version__)

            if mod == "dask":
                duck_array_type = (import_module("dask.array").Array,)
            elif mod == "pint":
                duck_array_type = (duck_array_module.Quantity,)
            elif mod == "cupy":
                duck_array_type = (duck_array_module.ndarray,)
            elif mod == "sparse":
                duck_array_type = (duck_array_module.SparseArray,)
            elif mod == "cubed":
                duck_array_type = (duck_array_module.Array,)
            else:
                raise NotImplementedError

        except (ImportError, AttributeError):  # pragma: no cover
            duck_array_module = None
            duck_array_version = Version("0.0.0")
            duck_array_type = ()

        self.module = duck_array_module
        self.version = duck_array_version
        self.type = duck_array_type
        self.available = duck_array_module is not None


_cached_duck_array_modules: dict[ModType, DuckArrayModule] = {}


def _get_cached_duck_array_module(mod: ModType) -> DuckArrayModule:
    if mod not in _cached_duck_array_modules:
        duckmod = DuckArrayModule(mod)
        _cached_duck_array_modules[mod] = duckmod
        return duckmod
    else:
        return _cached_duck_array_modules[mod]


def array_type(mod: ModType) -> DuckArrayTypes:
    """Quick wrapper to get the array class of the module."""
    return _get_cached_duck_array_module(mod).type


def mod_version(mod: ModType) -> Version:
    """Quick wrapper to get the version of the module."""
    return _get_cached_duck_array_module(mod).version


def is_dask_collection(x):
    if module_available("dask"):
        from dask.base import is_dask_collection

        return is_dask_collection(x)
    return False


def is_duck_dask_array(x):
    return is_duck_array(x) and is_dask_collection(x)


def is_chunked_array(x) -> bool:
    return is_duck_dask_array(x) or (is_duck_array(x) and hasattr(x, "chunks"))


def is_0d_dask_array(x):
    return is_duck_dask_array(x) and is_scalar(x)
