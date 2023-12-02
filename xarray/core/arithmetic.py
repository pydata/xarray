"""Base classes implementing arithmetic for xarray objects."""
from __future__ import annotations

import inspect
import numbers

import numpy as np

# _typed_ops.py is a generated file
from xarray.core._typed_ops import (
    DataArrayGroupByOpsMixin,
    DataArrayOpsMixin,
    DatasetGroupByOpsMixin,
    DatasetOpsMixin,
    VariableOpsMixin,
)
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.ops import (
    IncludeNumpySameMethods,
    IncludeReduceMethods,
)
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.pycompat import is_duck_array


class SupportsArithmetic:
    """Base class for xarray types that support arithmetic.

    Used by Dataset, DataArray, Variable and GroupBy.
    """

    __slots__ = ()

    # TODO: implement special methods for arithmetic here rather than injecting
    # them in xarray/core/ops.py. Ideally, do so by inheriting from
    # numpy.lib.mixins.NDArrayOperatorsMixin.

    # TODO: allow extending this with some sort of registration system
    _HANDLED_TYPES = (
        np.generic,
        numbers.Number,
        bytes,
        str,
    )

    def __array_function__(self, func, types, args, kwargs):
        """
        When ``np.clip(da, foo, bar)`` is called, this forwards that call to
        ``da.clip(foo, bar)``. It's similiar to ``__array_ufunc__``, but for non-ufuncs.

        This has the advantage of preserving attributes / retaining chunks / etc.
        """
        if not all(issubclass(t, SupportsArithmetic) for t in types):
            return NotImplemented

        # Define the mapping for numpy functions to internal methods and argument
        # mappings — currently only `np.clip`
        func_mappings = {np.clip: (self.clip, {"a_min": "min", "a_max": "max"}, {"a"})}

        if func in func_mappings:
            internal_method, arg_mapping, special_args = func_mappings[func]

            # Inspect the signature of the internal method
            method_sig = inspect.signature(internal_method)
            method_params = set(method_sig.parameters.keys())

            # Bind args and kwargs for the numpy function
            func_sig = inspect.signature(func)
            bound_arguments = func_sig.bind(*args, **kwargs)
            bound_arguments.apply_defaults()

            # Prepare arguments for the internal method
            method_args = {}
            for arg_name, value in bound_arguments.arguments.items():
                if arg_name in arg_mapping:
                    # Map numpy function argument names to internal method argument names
                    method_args[arg_mapping[arg_name]] = value
                elif arg_name in method_params:
                    # Pass arguments that share the same name
                    method_args[arg_name] = value

            # Check for any unsupported keyword arguments
            allowed_kwargs = set(arg_mapping.keys()) | special_args
            unsupported_kwargs = set(kwargs.keys()) - allowed_kwargs - method_params
            if unsupported_kwargs:
                # There's some case for returning `NotImplemented` here, so another type
                # could attempt to handle the function call. But the error message would
                # be quite unclear, and it seems quite unlikely that we'd hit this. It
                # might also be possible to defer to `np.clip`, but that has similar
                # downsides.
                raise ValueError(
                    f"Unsupported keyword arguments for {internal_method.__name__}: {unsupported_kwargs}"
                )

            return internal_method(**method_args)

        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        from xarray.core.computation import apply_ufunc

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
                "{} method for ufunc {} is not implemented on xarray objects, "
                "which currently only support the __call__ method. As an "
                "alternative, consider explicitly converting xarray objects "
                "to NumPy arrays (e.g., with `.values`).".format(method, ufunc)
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


class DatasetArithmetic(
    ImplementsDatasetReduce,
    SupportsArithmetic,
    DatasetOpsMixin,
):
    __slots__ = ()
    __array_priority__ = 50


class DataArrayArithmetic(
    ImplementsArrayReduce,
    IncludeNumpySameMethods,
    SupportsArithmetic,
    DataArrayOpsMixin,
):
    __slots__ = ()
    # priority must be higher than Variable to properly work with binary ufuncs
    __array_priority__ = 60


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


class CoarsenArithmetic(IncludeReduceMethods):
    __slots__ = ()
