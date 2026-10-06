"""Base classes implementing arithmetic for xarray objects."""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Self, overload

import numpy as np

from xarray.computation.ops import IncludeNumpySameMethods

# _typed_ops.py is a generated file
from xarray.core._typed_ops import (
    DataArrayGroupByOpsMixin,
    DataArrayOpsMixin,
    DatasetGroupByOpsMixin,
    DatasetOpsMixin,
    VariableOpsMixin,
)
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.namedarray.utils import is_duck_array

if TYPE_CHECKING:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.variable import Variable

# Operands besides xarray objects that `__array_ufunc__` handles, see
# SupportsArithmetic._HANDLED_TYPES. Not `ArrayLike`, since that includes xarray
# objects themselves (they implement `__array__`). Other duck arrays are allowed
# at runtime, but not typed.
type _UfuncOperand = complex | str | bytes | np.generic | np.ndarray[Any, Any]
type _UnsupportedUfuncMethod = Literal[
    "reduce", "reduceat", "accumulate", "outer", "at"
]


class SupportsArithmetic:
    """Base class for xarray types that support arithmetic.

    Used by Dataset, DataArray, Variable and GroupBy.
    """

    __slots__ = ()

    # TODO: implement special methods for arithmetic here rather than injecting
    # them in xarray/computation/ops.py. Ideally, do so by inheriting from
    # numpy.lib.mixins.NDArrayOperatorsMixin.

    # TODO: allow extending this with some sort of registration system
    _HANDLED_TYPES = (
        np.generic,
        numbers.Number,
        bytes,
        str,
    )

    # The subclasses add typed overloads of `__array_ufunc__`, so that NumPy's
    # `__array_ufunc__` protocols (e.g. `_CanUfuncCall1`, `_CanUfuncCall2L` and
    # `_CanUfuncCall2R` in numpy/_core/umath.pyi) infer the result type of e.g.
    # `np.exp(da)` or `np.add(da, ds)`. Type checkers only infer the result from
    # the first overload whose shape matches a protocol, therefore:
    # - There is a single binary overload, that covers the xarray object as left
    #   and right operand. It returns the type with the highest priority (Dataset >
    #   DataArray > Variable), and lists the operands this type wins against.
    #   Combinations with a higher priority type fail it, so that the type checker
    #   uses the `__array_ufunc__` of the other operand.
    # - The binary overload comes before the unary one.
    # - There is no catch-all overload for `method: str`.
    # The overloads are not compatible with this signature, as they only accept the
    # methods and numbers of inputs that are supported at runtime.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        from xarray.computation.apply_ufunc import apply_ufunc

        # See the docstring example for numpy.lib.mixins.NDArrayOperatorsMixin.
        out = kwargs.get("out", ())
        for x in inputs + out:
            if not is_duck_array(x) and not isinstance(
                x, self._HANDLED_TYPES + (SupportsArithmetic,)
            ):
                return NotImplemented

        if ufunc.signature is not None:
            raise NotImplementedError(
                f"{ufunc} not supported: xarray objects do not directly implement "
                "generalized ufuncs. Instead, use xarray.apply_ufunc or "
                "explicitly convert to xarray objects to NumPy arrays "
                "(e.g., with `.values`)."
            )

        if method != "__call__":
            # TODO: support other methods, e.g., reduce and accumulate.
            raise NotImplementedError(
                f"{method} method for ufunc {ufunc} is not implemented on xarray objects, "
                "which currently only support the __call__ method. As an "
                "alternative, consider explicitly converting xarray objects "
                "to NumPy arrays (e.g., with `.values`)."
            )

        if any(isinstance(o, SupportsArithmetic) for o in out):
            # TODO: implement this with logic like _inplace_binary_op. This
            # will be necessary to use NDArrayOperatorsMixin.
            raise NotImplementedError(
                "xarray objects are not yet supported in the `out` argument "
                "for ufuncs. As an alternative, consider explicitly "
                "converting xarray objects to NumPy arrays (e.g., with "
                "`.values`)."
            )

        join = dataset_join = OPTIONS["arithmetic_join"]

        return apply_ufunc(
            ufunc,
            *inputs,
            input_core_dims=((),) * ufunc.nin,
            output_core_dims=((),) * ufunc.nout,
            join=join,
            dataset_join=dataset_join,
            dataset_fill_value=np.nan,
            kwargs=kwargs,
            dask="allowed",
            keep_attrs=_get_keep_attrs(default=True),
        )


class VariableArithmetic(
    ImplementsArrayReduce,
    IncludeNumpySameMethods,
    SupportsArithmetic,
    VariableOpsMixin,
):
    __slots__ = ()
    # prioritize our operations over those of numpy.ndarray (priority=0)
    __array_priority__ = 50

    if TYPE_CHECKING:
        # override: only the methods and numbers of inputs that work at runtime
        @overload  # type: ignore[override]
        def __array_ufunc__(
            self,
            ufunc: np.ufunc,
            method: Literal["__call__"],
            lhs: Variable | _UfuncOperand,
            rhs: Variable | _UfuncOperand,
            /,
            **kwargs: Any,
        ) -> Self: ...
        @overload
        def __array_ufunc__(
            self,
            ufunc: np.ufunc,
            method: Literal["__call__"],
            x: Self,
            /,
            **kwargs: Any,
        ) -> Self: ...
        @overload
        def __array_ufunc__(
            self,
            ufunc: np.ufunc,
            method: _UnsupportedUfuncMethod,
            /,
            *inputs: Any,
            **kwargs: Any,
        ) -> NoReturn: ...
        def __array_ufunc__(
            self, ufunc: np.ufunc, method: str, /, *inputs: Any, **kwargs: Any
        ) -> Any: ...


class DatasetArithmetic(
    ImplementsDatasetReduce,
    SupportsArithmetic,
    DatasetOpsMixin,
):
    __slots__ = ()
    __array_priority__ = 50

    if TYPE_CHECKING:
        # override: only the methods and numbers of inputs that work at runtime
        @overload  # type: ignore[override]
        def __array_ufunc__(
            self,
            ufunc: np.ufunc,
            method: Literal["__call__"],
            lhs: Dataset | DataArray | Variable | _UfuncOperand,
            rhs: Dataset | DataArray | Variable | _UfuncOperand,
            /,
            **kwargs: Any,
        ) -> Self: ...
        @overload
        def __array_ufunc__(
            self,
            ufunc: np.ufunc,
            method: Literal["__call__"],
            x: Self,
            /,
            **kwargs: Any,
        ) -> Self: ...
        @overload
        def __array_ufunc__(
            self,
            ufunc: np.ufunc,
            method: _UnsupportedUfuncMethod,
            /,
            *inputs: Any,
            **kwargs: Any,
        ) -> NoReturn: ...
        def __array_ufunc__(
            self, ufunc: np.ufunc, method: str, /, *inputs: Any, **kwargs: Any
        ) -> Any: ...


class DataArrayArithmetic(
    ImplementsArrayReduce,
    IncludeNumpySameMethods,
    SupportsArithmetic,
    DataArrayOpsMixin,
):
    __slots__ = ()
    # priority must be higher than Variable to properly work with binary ufuncs
    __array_priority__ = 60

    if TYPE_CHECKING:
        # override: only the methods and numbers of inputs that work at runtime
        @overload  # type: ignore[override]
        def __array_ufunc__(
            self,
            ufunc: np.ufunc,
            method: Literal["__call__"],
            lhs: DataArray | Variable | _UfuncOperand,
            rhs: DataArray | Variable | _UfuncOperand,
            /,
            **kwargs: Any,
        ) -> Self: ...
        @overload
        def __array_ufunc__(
            self,
            ufunc: np.ufunc,
            method: Literal["__call__"],
            x: Self,
            /,
            **kwargs: Any,
        ) -> Self: ...
        @overload
        def __array_ufunc__(
            self,
            ufunc: np.ufunc,
            method: _UnsupportedUfuncMethod,
            /,
            *inputs: Any,
            **kwargs: Any,
        ) -> NoReturn: ...
        def __array_ufunc__(
            self, ufunc: np.ufunc, method: str, /, *inputs: Any, **kwargs: Any
        ) -> Any: ...


class DataArrayGroupbyArithmetic(
    SupportsArithmetic,
    DataArrayGroupByOpsMixin,
):
    __slots__ = ()


class DatasetGroupbyArithmetic(
    SupportsArithmetic,
    DatasetGroupByOpsMixin,
):
    __slots__ = ()
