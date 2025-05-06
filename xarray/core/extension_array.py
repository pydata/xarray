from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, cast

import numpy as np
import pandas as pd
from packaging.version import Version
from pandas.api.types import is_extension_array_dtype

from xarray.core.types import DTypeLikeSave, T_ExtensionArray
from xarray.core.utils import NDArrayMixin

HANDLED_EXTENSION_ARRAY_FUNCTIONS: dict[Callable, Callable] = {}


def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""

    def decorator(func):
        HANDLED_EXTENSION_ARRAY_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.issubdtype)
def __extension_duck_array__issubdtype(
    extension_array_dtype: T_ExtensionArray, other_dtype: DTypeLikeSave
) -> bool:
    return False  # never want a function to think a pandas extension dtype is a subtype of numpy


@implements(np.broadcast_to)
def __extension_duck_array__broadcast(arr: T_ExtensionArray, shape: tuple):
    if shape[0] == len(arr) and len(shape) == 1:
        return arr
    raise NotImplementedError("Cannot broadcast 1d-only pandas extension array.")


@implements(np.stack)
def __extension_duck_array__stack(arr: T_ExtensionArray, axis: int):
    raise NotImplementedError("Cannot stack 1d-only pandas extension array.")


@implements(np.concatenate)
def __extension_duck_array__concatenate(
    arrays: Sequence[T_ExtensionArray], axis: int = 0, out=None
) -> T_ExtensionArray:
    return type(arrays[0])._concat_same_type(arrays)  # type: ignore[attr-defined]


@implements(np.where)
def __extension_duck_array__where(
    condition: np.ndarray, x: T_ExtensionArray, y: T_ExtensionArray
) -> T_ExtensionArray:
    if (
        isinstance(x, pd.Categorical)
        and isinstance(y, pd.Categorical)
        and x.dtype != y.dtype
    ):
        x = x.add_categories(set(y.categories).difference(set(x.categories)))  # type: ignore[assignment]
        y = y.add_categories(set(x.categories).difference(set(y.categories)))  # type: ignore[assignment]
    return cast(T_ExtensionArray, pd.Series(x).where(condition, pd.Series(y)).array)


@dataclass(frozen=True)
class PandasExtensionArray(Generic[T_ExtensionArray], NDArrayMixin):
    """NEP-18 compliant wrapper for pandas extension arrays.

    Parameters
    ----------
    array : T_ExtensionArray
        The array to be wrapped upon e.g,. :py:class:`xarray.Variable` creation.
    ```
    """

    array: T_ExtensionArray

    def __post_init__(self):
        if not isinstance(self.array, pd.api.extensions.ExtensionArray):
            raise TypeError(f"{self.array} is not an pandas ExtensionArray.")

    def __array_function__(self, func, types, args, kwargs):
        def replace_duck_with_extension_array(args) -> list:
            args_as_list = list(args)
            for index, value in enumerate(args_as_list):
                if isinstance(value, PandasExtensionArray):
                    args_as_list[index] = value.array
                elif isinstance(
                    value, tuple
                ):  # should handle more than just tuple? iterable?
                    args_as_list[index] = tuple(
                        replace_duck_with_extension_array(value)
                    )
                elif isinstance(value, list):
                    args_as_list[index] = replace_duck_with_extension_array(value)
            return args_as_list

        args = tuple(replace_duck_with_extension_array(args))
        if func not in HANDLED_EXTENSION_ARRAY_FUNCTIONS:
            return func(*args, **kwargs)
        res = HANDLED_EXTENSION_ARRAY_FUNCTIONS[func](*args, **kwargs)
        if is_extension_array_dtype(res):
            return type(self)[type(res)](res)
        return res

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return ufunc(*inputs, **kwargs)

    def __getitem__(self, key) -> PandasExtensionArray[T_ExtensionArray]:
        item = self.array[key]
        if is_extension_array_dtype(item):
            return PandasExtensionArray(item)
        if np.isscalar(item) or isinstance(key, int):
            return PandasExtensionArray(type(self.array)._from_sequence([item]))  # type: ignore[call-arg,attr-defined,unused-ignore]
        return PandasExtensionArray(item)

    def __setitem__(self, key, val):
        self.array[key] = val

    def __eq__(self, other):
        if isinstance(other, PandasExtensionArray):
            return self.array == other.array
        return self.array == other

    def __ne__(self, other):
        return ~(self == other)

    def __len__(self):
        return len(self.array)

    @property
    def ndim(self) -> int:
        return 1

    def __array__(
        self, dtype: np.typing.DTypeLike = None, /, *, copy: bool | None = None
    ) -> np.ndarray:
        if Version(np.__version__) >= Version("2.0.0"):
            return np.asarray(self.array, dtype=dtype, copy=copy)
        else:
            return np.asarray(self.array, dtype=dtype)
