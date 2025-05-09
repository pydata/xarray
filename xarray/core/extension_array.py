from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Generic, cast

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.api.types import is_extension_array_dtype
from pandas.api.types import is_scalar as pd_is_scalar
from pandas.core.dtypes.astype import astype_array_safe
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.concat import concat_compat

from xarray.core.types import DTypeLikeSave, T_ExtensionArray

HANDLED_EXTENSION_ARRAY_FUNCTIONS: dict[Callable, Callable] = {}


if TYPE_CHECKING:
    from pandas._typing import DtypeObj, Scalar


def is_scalar(value: object) -> bool:
    """Workaround: pandas is_scalar doesn't recognize Categorical nulls for some reason."""
    return value is pd.CategoricalDtype.na_value or pd_is_scalar(value)


def implements(numpy_function_or_name: Callable | str) -> Callable:
    """Register an __array_function__ implementation.

    Pass a function directly if it's guaranteed to exist in all supported numpy versions, or a
    string to first check for its existence.
    """

    def decorator(func):
        if isinstance(numpy_function_or_name, str):
            numpy_function = getattr(np, numpy_function_or_name, None)
        else:
            numpy_function = numpy_function_or_name

        if numpy_function:
            HANDLED_EXTENSION_ARRAY_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.issubdtype)
def __extension_duck_array__issubdtype(
    extension_array_dtype: T_ExtensionArray, other_dtype: DTypeLikeSave
) -> bool:
    return False  # never want a function to think a pandas extension dtype is a subtype of numpy


@implements("astype")  # np.astype was added in 2.1.0, but we only require >=1.24
def __extension_duck_array__astype(
    array_or_scalar: np.typing.ArrayLike,
    dtype: DTypeLikeSave,
    order: str = "K",
    casting: str = "unsafe",
    subok: bool = True,
    copy: bool = True,
    device: str = None,
) -> T_ExtensionArray:
    if (
        not (
            is_extension_array_dtype(array_or_scalar) or is_extension_array_dtype(dtype)
        )
        or casting != "unsafe"
        or not subok
        or order != "K"
    ):
        return NotImplemented

    return as_extension_array(array_or_scalar, dtype, copy=copy)


@implements(np.asarray)
def __extension_duck_array__asarray(
    array_or_scalar: np.typing.ArrayLike, dtype: DTypeLikeSave = None
) -> T_ExtensionArray:
    if not is_extension_array_dtype(dtype):
        return NotImplemented

    return as_extension_array(array_or_scalar, dtype)


def as_extension_array(
    array_or_scalar: np.typing.ArrayLike, dtype: ExtensionDtype, copy: bool = False
) -> T_ExtensionArray:
    if is_scalar(array_or_scalar):
        return dtype.construct_array_type()._from_sequence(
            [array_or_scalar], dtype=dtype
        )
    else:
        return astype_array_safe(array_or_scalar, dtype, copy=copy)


@implements(np.result_type)
def __extension_duck_array__result_type(
    *arrays_and_dtypes: np.typing.ArrayLike | np.typing.DTypeLike,
) -> DtypeObj:
    extension_arrays_and_dtypes = [
        x for x in arrays_and_dtypes if is_extension_array_dtype(x)
    ]
    if not extension_arrays_and_dtypes:
        return NotImplemented

    ea_dtypes: list[ExtensionDtype] = [
        getattr(x, "dtype", x) for x in extension_arrays_and_dtypes
    ]
    scalars: list[Scalar] = [x for x in arrays_and_dtypes if is_scalar(x)]
    # other_stuff could include:
    # - arrays such as pd.ABCSeries, np.ndarray, or other array-api duck arrays
    # - dtypes such as pd.DtypeObj, np.dtype, or other array-api duck dtypes
    other_stuff = [
        x
        for x in arrays_and_dtypes
        if not is_extension_array_dtype(x) and not is_scalar(x)
    ]

    # We implement one special case: when possible, preserve Categoricals (avoid promoting
    # to object) by merging the categories of all given Categoricals + scalars + NA.
    # Ideally this could be upstreamed into pandas find_result_type / find_common_type.
    if not other_stuff and all(
        isinstance(x, pd.CategoricalDtype) and not x.ordered for x in ea_dtypes
    ):
        return union_unordered_categorical_and_scalar(ea_dtypes, scalars)

    # In all other cases, we defer to pandas find_result_type, which is the only Pandas API
    # permissive enough to handle scalars + other_stuff.
    # Note that unlike find_common_type or np.result_type, it operates in pairs, where
    # the left side must be a DtypeObj.
    return functools.reduce(find_result_type, arrays_and_dtypes, ea_dtypes[0])


def union_unordered_categorical_and_scalar(
    categorical_dtypes: list[pd.CategoricalDtype], scalars: list[Scalar]
) -> pd.CategoricalDtype:
    scalars = [x for x in scalars if x is not pd.CategoricalDtype.na_value]
    all_categories = set().union(*(x.categories for x in categorical_dtypes))
    all_categories = all_categories.union(scalars)
    return pd.CategoricalDtype(categories=all_categories)


@implements(np.broadcast_to)
def __extension_duck_array__broadcast(arr: T_ExtensionArray, shape: tuple):
    if shape[0] == len(arr) and len(shape) == 1:
        return arr
    raise NotImplementedError("Cannot broadcast 1d-only pandas categorical array.")


@implements(np.stack)
def __extension_duck_array__stack(arr: T_ExtensionArray, axis: int):
    raise NotImplementedError("Cannot stack 1d-only pandas categorical array.")


@implements(np.concatenate)
def __extension_duck_array__concatenate(
    arrays: Sequence[T_ExtensionArray], axis: int = 0, out=None
) -> T_ExtensionArray:
    return concat_compat(arrays, ea_compat_axis=True)


@implements(np.where)
def __extension_duck_array__where(
    condition: T_ExtensionArray | np.ArrayLike,
    x: T_ExtensionArray,
    y: T_ExtensionArray | np.ArrayLike,
) -> T_ExtensionArray:
    return cast(T_ExtensionArray, pd.Series(x).where(condition, y).array)


def _replace_duck(args, replacer: Callable[[PandasExtensionArray]]) -> list:
    args_as_list = list(args)
    for index, value in enumerate(args_as_list):
        if isinstance(value, PandasExtensionArray):
            args_as_list[index] = replacer(value)
        elif isinstance(value, tuple):  # should handle more than just tuple? iterable?
            args_as_list[index] = tuple(_replace_duck(value, replacer))
        elif isinstance(value, list):
            args_as_list[index] = _replace_duck(value, replacer)
    return args_as_list


def replace_duck_with_extension_array(args) -> tuple:
    return tuple(_replace_duck(args, lambda duck: duck.array))


def replace_duck_with_series(args) -> tuple:
    return tuple(_replace_duck(args, lambda duck: pd.Series(duck.array)))


class PandasExtensionArray(Generic[T_ExtensionArray]):
    array: T_ExtensionArray

    def __init__(self, array: T_ExtensionArray):
        """NEP-18 compliant wrapper for pandas extension arrays.

        Parameters
        ----------
        array : T_ExtensionArray
            The array to be wrapped upon e.g,. :py:class:`xarray.Variable` creation.
        ```
        """
        if not isinstance(array, ExtensionArray):
            raise TypeError(f"{array} is not an pandas ExtensionArray.")
        self.array = array

        self._add_ops_dunders()

    def _add_ops_dunders(self):
        """Delegate all operators to pd.Series"""

        def create_dunder(name: str) -> Callable:
            def binary_dunder(self, other):
                self, other = replace_duck_with_series((self, other))
                res = getattr(pd.Series, name)(self, other)
                if isinstance(res, pd.Series):
                    res = PandasExtensionArray(res.array)
                return res

            return binary_dunder

        # see pandas.core.arraylike.OpsMixin
        binary_operators = [
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
            "__and__",
            "__rand__",
            "__or__",
            "__ror__",
            "__xor__",
            "__rxor__",
            "__add__",
            "__radd__",
            "__sub__",
            "__rsub__",
            "__mul__",
            "__rmul__",
            "__truediv__",
            "__rtruediv__",
            "__floordiv__",
            "__rfloordiv__",
            "__mod__",
            "__rmod__",
            "__divmod__",
            "__rdivmod__",
            "__pow__",
            "__rpow__",
        ]
        for method_name in binary_operators:
            setattr(self.__class__, method_name, create_dunder(method_name))

    def __array_function__(self, func, types, args, kwargs):
        args = replace_duck_with_extension_array(args)
        if func not in HANDLED_EXTENSION_ARRAY_FUNCTIONS:
            return func(*args, **kwargs)
        res = HANDLED_EXTENSION_ARRAY_FUNCTIONS[func](*args, **kwargs)
        if isinstance(res, ExtensionArray):
            return type(self)[type(res)](res)
        return res

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if first_ea := next(
            (x for x in inputs if isinstance(x, PandasExtensionArray)), None
        ):
            inputs = replace_duck_with_series(inputs)
            res = first_ea.__array_ufunc__(ufunc, method, *inputs, **kwargs)
            if isinstance(res, pd.Series):
                arr = res.array
                return type(self)[type(arr)](arr)
            return res

        return getattr(ufunc, method)(*inputs, **kwargs)

    def __repr__(self):
        return f"PandasExtensionArray(array={self.array!r})"

    def __getattr__(self, attr: str) -> object:
        return getattr(self.array, attr)

    def __getitem__(self, key) -> PandasExtensionArray[T_ExtensionArray]:
        item = self.array[key]
        if is_extension_array_dtype(item):
            return type(self)(item)
        if is_scalar(item):
            return type(self)(type(self.array)([item]))  # type: ignore[call-arg]  # only subclasses with proper __init__ allowed
        return item

    def __setitem__(self, key, val):
        self.array[key] = val

    def __len__(self):
        return len(self.array)
