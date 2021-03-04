"""Generate stub file for arithmetic operators of various xarray classes.

For internal xarray development use only.

Usage:
    python -m xarray.util.stubgen_ops > xarray/core/_typed_ops.pyi
"""

from collections import defaultdict

unary_ops = ["__neg__", "__pos__", "__abs__", "__invert__"]

binary_ops = ["eq", "ne", "lt", "le", "gt", "ge"]
binary_ops += ["add", "sub", "mul", "pow", "truediv", "floordiv", "mod"]
binary_ops += ["radd", "rsub", "rmul", "rpow", "rtruediv", "rfloordiv", "rmod"]
binary_ops += ["and", "xor", "or"]
binary_ops += ["rand", "rxor", "ror"]
binary_ops = [f"__{op}__" for op in binary_ops]

stub_info = defaultdict(list)
METHOD_TEMPLATE_UNOPS = "    def {method}(self: T_Self) -> T_Self: ..."

method_template_binops = """\
    def {method}(self: T_Dataset, other: DsCompatible) -> T_Dataset: ...{override_misc}"""
stub_info["TypedDatasetOps"].append((METHOD_TEMPLATE_UNOPS, unary_ops))
stub_info["TypedDatasetOps"].append((method_template_binops, binary_ops))

# Note: in some of the overloads below the return value in reality is
# NotImplemented, which cannot accurately be expressed with type hints,
# e.g. Literal[NotImplemented] or type(NotImplemented) are not allowed and
# NoReturn has a different meaning.
# In such cases we are lending the type checkers a hand by specifying the
# return type of the corresponding reflexive method on the other argument
# which will be called in such instances.

# TypedDataArrayOps
method_template_binops = """\
    @overload{override}
    def {method}(self, other: T_Dataset) -> T_Dataset: ...{misc}
    @overload
    def {method}(self, other: DatasetGroupBy) -> Dataset: ...{misc}
    @overload
    def {method}(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...{misc}
    @overload
    def {method}(self: T_DataArray, other: DaCompatible) -> T_DataArray: ...{misc}"""
stub_info["TypedDataArrayOps"].append((METHOD_TEMPLATE_UNOPS, unary_ops))
stub_info["TypedDataArrayOps"].append((method_template_binops, binary_ops))

# TypedVariableOps
method_template_binops = """\
    @overload{override}
    def {method}(self, other: T_Dataset) -> T_Dataset: ...{misc}
    @overload
    def {method}(self, other: T_DataArray) -> T_DataArray: ...{misc}
    @overload
    def {method}(self: T_Variable, other: VarCompatible) -> T_Variable: ...{misc}"""
stub_info["TypedVariableOps"].append((METHOD_TEMPLATE_UNOPS, unary_ops))
stub_info["TypedVariableOps"].append((method_template_binops, binary_ops))

# TypedDatasetGroupByOps
method_template_binops = """\
    @overload{override}
    def {method}(self, other: T_Dataset) -> T_Dataset: ...{misc}
    @overload
    def {method}(self, other: DataArray) -> Dataset: ...{misc}
    @overload
    def {method}(self, other: GroupByIncompatible) -> NoReturn: ..."""
stub_info["TypedDatasetGroupByOps"].append((method_template_binops, binary_ops))

# TypedDataArrayGroupByOps
method_template_binops = """\
    @overload{override}
    def {method}(self, other: T_Dataset) -> T_Dataset: ...{misc}
    @overload
    def {method}(self, other: T_DataArray) -> T_DataArray: ...{misc}
    @overload
    def {method}(self, other: GroupByIncompatible) -> NoReturn: ..."""
stub_info["TypedDataArrayGroupByOps"].append((method_template_binops, binary_ops))


# For some methods override return type `bool` defined by base class `object`.
def override(method):
    if method in {"__eq__", "__ne__"}:
        return "  # type: ignore[override]"
    return ""


def override_misc(method):
    if method in {"__eq__", "__ne__"}:
        return "  # type: ignore[override, misc]"
    return "  # type: ignore[misc]"


def misc():
    return "  # type: ignore[misc]"


stubfile_preamble = '''\
"""Stub file for arithmetic operators of various xarray classes.

This file was generated using xarray.util.stubgen_ops. Do not edit manually."""

from typing import NoReturn, TypeVar, Union, overload

import numpy as np

from .dataarray import DataArray
from .dataset import Dataset
from .groupby import DataArrayGroupBy, DatasetGroupBy, GroupBy
from .variable import Variable

try:
    from dask.array import Array as DaskArray
except ImportError:
    DaskArray = np.ndarray

T_Dataset = TypeVar("T_Dataset", bound=Dataset)
T_DataArray = TypeVar("T_DataArray", bound=DataArray)
T_Variable = TypeVar("T_Variable", bound=Variable)
T_Self = TypeVar("T_Self")

# Note: ScalarOrArray (and types involving ScalarOrArray) is to be used last in overloads,
# since nd.ndarray is typed as Any for older versions of numpy.
ScalarOrArray = Union[complex, bytes, str, np.generic, np.ndarray, DaskArray]
DsCompatible = Union[Dataset, DataArray, Variable, GroupBy, ScalarOrArray]
DaCompatible = Union[DataArray, Variable, ScalarOrArray]
VarCompatible = Union[Variable, ScalarOrArray]
GroupByIncompatible = Union[Variable, GroupBy]'''


# Render stub file
if __name__ == "__main__":
    print(stubfile_preamble)

    for cls_name, method_blocks in stub_info.items():
        print()
        print(f"class {cls_name}:")
        for method_template, methods in method_blocks:
            for method in methods:
                print(
                    method_template.format(
                        method=method,
                        override=override(method),
                        override_misc=override_misc(method),
                        misc=misc(),
                    )
                )
