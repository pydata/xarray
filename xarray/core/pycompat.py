from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, Tuple, Type

import numpy as np
from packaging.version import Version

from .utils import is_duck_array, module_available

integer_types = (int, np.integer)

if TYPE_CHECKING:
    ModType = Literal["dask", "pint", "cupy", "sparse"]
    DuckArrayTypes = Tuple[Type[Any], ...]  # TODO: improve this? maybe Generic


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
        duck_array_module: ModuleType | None = None
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
            else:
                raise NotImplementedError

        except ImportError:  # pragma: no cover
            duck_array_module = None
            duck_array_version = Version("0.0.0")
            duck_array_type = ()

        self.module = duck_array_module
        self.version = duck_array_version
        self.type = duck_array_type
        self.available = duck_array_module is not None


def array_type(mod: ModType) -> DuckArrayTypes:
    """Quick wrapper to get the array class of the module."""
    return DuckArrayModule(mod).type


def mod_version(mod: ModType) -> Version:
    """Quick wrapper to get the version of the module."""
    return DuckArrayModule(mod).version


def is_dask_collection(x):
    if module_available("dask"):
        from dask.base import is_dask_collection

        return is_dask_collection(x)
    return False


def is_duck_dask_array(x):
    return is_duck_array(x) and is_dask_collection(x)
