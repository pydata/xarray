"""Generate module and stub file for arithmetic operators of various xarray classes.

For internal xarray development use only.

Usage:
    python xarray/util/generate_ops.py --module > xarray/core/_typed_ops.py
    python xarray/util/generate_ops.py --stubs > xarray/core/_typed_ops.pyi

"""
# Note: the comments in https://github.com/pydata/xarray/pull/4904 provide some
# background to some of the design choices made here.

import sys

BINOPS_EQNE = (("__eq__", "nputils.array_eq"), ("__ne__", "nputils.array_ne"))
BINOPS_CMP = (
    ("__lt__", "operator.lt"),
    ("__le__", "operator.le"),
    ("__gt__", "operator.gt"),
    ("__ge__", "operator.ge"),
)
BINOPS_NUM = (
    ("__add__", "operator.add"),
    ("__sub__", "operator.sub"),
    ("__mul__", "operator.mul"),
    ("__pow__", "operator.pow"),
    ("__truediv__", "operator.truediv"),
    ("__floordiv__", "operator.floordiv"),
    ("__mod__", "operator.mod"),
    ("__and__", "operator.and_"),
    ("__xor__", "operator.xor"),
    ("__or__", "operator.or_"),
)
BINOPS_REFLEXIVE = (
    ("__radd__", "operator.add"),
    ("__rsub__", "operator.sub"),
    ("__rmul__", "operator.mul"),
    ("__rpow__", "operator.pow"),
    ("__rtruediv__", "operator.truediv"),
    ("__rfloordiv__", "operator.floordiv"),
    ("__rmod__", "operator.mod"),
    ("__rand__", "operator.and_"),
    ("__rxor__", "operator.xor"),
    ("__ror__", "operator.or_"),
)
BINOPS_INPLACE = (
    ("__iadd__", "operator.iadd"),
    ("__isub__", "operator.isub"),
    ("__imul__", "operator.imul"),
    ("__ipow__", "operator.ipow"),
    ("__itruediv__", "operator.itruediv"),
    ("__ifloordiv__", "operator.ifloordiv"),
    ("__imod__", "operator.imod"),
    ("__iand__", "operator.iand"),
    ("__ixor__", "operator.ixor"),
    ("__ior__", "operator.ior"),
)
UNARY_OPS = (
    ("__neg__", "operator.neg"),
    ("__pos__", "operator.pos"),
    ("__abs__", "operator.abs"),
    ("__invert__", "operator.invert"),
)
# round method and numpy/pandas unary methods which don't modify the data shape,
# so the result should still be wrapped in an Variable/DataArray/Dataset
OTHER_UNARY_METHODS = (
    ("round", "ops.round_"),
    ("argsort", "ops.argsort"),
    ("conj", "ops.conj"),
    ("conjugate", "ops.conjugate"),
)

template_binop = """
    def {method}(self, other):
        return self._binary_op(other, {func})"""
template_reflexive = """
    def {method}(self, other):
        return self._binary_op(other, {func}, reflexive=True)"""
template_inplace = """
    def {method}(self, other):
        return self._inplace_binary_op(other, {func})"""
template_unary = """
    def {method}(self):
        return self._unary_op({func})"""
template_other_unary = """
    def {method}(self, *args, **kwargs):
        return self._unary_op({func}, *args, **kwargs)"""
required_method_unary = """
    def _unary_op(self, f, *args, **kwargs):
        raise NotImplementedError"""
required_method_binary = """
    def _binary_op(self, other, f, reflexive=False):
        raise NotImplementedError"""
required_method_inplace = """
    def _inplace_binary_op(self, other, f):
        raise NotImplementedError"""

# For some methods we override return type `bool` defined by base class `object`.
OVERRIDE_TYPESHED = {"override": "  # type: ignore[override]"}
NO_OVERRIDE = {"override": ""}

# Note: in some of the overloads below the return value in reality is NotImplemented,
# which cannot accurately be expressed with type hints,e.g. Literal[NotImplemented]
# or type(NotImplemented) are not allowed and NoReturn has a different meaning.
# In such cases we are lending the type checkers a hand by specifying the return type
# of the corresponding reflexive method on `other` which will be called instead.
stub_ds = """\
    def {method}(self: T_Dataset, other: DsCompatible) -> T_Dataset: ...{override}"""
stub_da = """\
    @overload{override}
    def {method}(self, other: T_Dataset) -> T_Dataset: ...
    @overload
    def {method}(self, other: "DatasetGroupBy") -> "Dataset": ...
    @overload
    def {method}(self: T_DataArray, other: DaCompatible) -> T_DataArray: ..."""
stub_var = """\
    @overload{override}
    def {method}(self, other: T_Dataset) -> T_Dataset: ...
    @overload
    def {method}(self, other: T_DataArray) -> T_DataArray: ...
    @overload
    def {method}(self: T_Variable, other: VarCompatible) -> T_Variable: ..."""
stub_dsgb = """\
    @overload{override}
    def {method}(self, other: T_Dataset) -> T_Dataset: ...
    @overload
    def {method}(self, other: "DataArray") -> "Dataset": ...
    @overload
    def {method}(self, other: GroupByIncompatible) -> NoReturn: ..."""
stub_dagb = """\
    @overload{override}
    def {method}(self, other: T_Dataset) -> T_Dataset: ...
    @overload
    def {method}(self, other: T_DataArray) -> T_DataArray: ...
    @overload
    def {method}(self, other: GroupByIncompatible) -> NoReturn: ..."""
stub_unary = """\
    def {method}(self: {self_type}) -> {self_type}: ..."""
stub_other_unary = """\
    def {method}(self: {self_type}, *args, **kwargs) -> {self_type}: ..."""
stub_required_unary = """\
    def _unary_op(self, f, *args, **kwargs): ..."""
stub_required_binary = """\
    def _binary_op(self, other, f, reflexive=...): ..."""
stub_required_inplace = """\
    def _inplace_binary_op(self, other, f): ..."""


def unops(self_type):
    extra_context = {"self_type": self_type}
    return [
        ([(None, None)], required_method_unary, stub_required_unary, {}),
        (UNARY_OPS, template_unary, stub_unary, extra_context),
        (OTHER_UNARY_METHODS, template_other_unary, stub_other_unary, extra_context),
    ]


def binops(stub=""):
    return [
        ([(None, None)], required_method_binary, stub_required_binary, {}),
        (BINOPS_NUM + BINOPS_CMP, template_binop, stub, NO_OVERRIDE),
        (BINOPS_EQNE, template_binop, stub, OVERRIDE_TYPESHED),
        (BINOPS_REFLEXIVE, template_reflexive, stub, NO_OVERRIDE),
    ]


def inplace():
    return [
        ([(None, None)], required_method_inplace, stub_required_inplace, {}),
        (BINOPS_INPLACE, template_inplace, "", {}),
    ]


ops_info = {}
ops_info["DatasetOpsMixin"] = binops(stub_ds) + inplace() + unops("T_Dataset")
ops_info["DataArrayOpsMixin"] = binops(stub_da) + inplace() + unops("T_DataArray")
ops_info["VariableOpsMixin"] = binops(stub_var) + inplace() + unops("T_Variable")
ops_info["DatasetGroupByOpsMixin"] = binops(stub_dsgb)
ops_info["DataArrayGroupByOpsMixin"] = binops(stub_dagb)

MODULE_PREAMBLE = '''\
"""Mixin classes with arithmetic operators."""
# This file was generated using xarray.util.generate_ops. Do not edit manually.

import operator

from . import nputils, ops'''

STUBFILE_PREAMBLE = '''\
"""Stub file for mixin classes with arithmetic operators."""
# This file was generated using xarray.util.generate_ops. Do not edit manually.

from typing import NoReturn, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike

from .dataarray import DataArray
from .dataset import Dataset
from .groupby import DataArrayGroupBy, DatasetGroupBy, GroupBy
from .types import (
    DaCompatible,
    DsCompatible,
    GroupByIncompatible,
    ScalarOrArray,
    VarCompatible,
)
from .variable import Variable

try:
    from dask.array import Array as DaskArray
except ImportError:
    DaskArray = np.ndarray  # type: ignore

# DatasetOpsMixin etc. are parent classes of Dataset etc.
# Because of https://github.com/pydata/xarray/issues/5755, we redefine these. Generally
# we use the ones in `types`. (We're open to refining this, and potentially integrating
# the `py` & `pyi` files to simplify them.)
T_Dataset = TypeVar("T_Dataset", bound="DatasetOpsMixin")
T_DataArray = TypeVar("T_DataArray", bound="DataArrayOpsMixin")
T_Variable = TypeVar("T_Variable", bound="VariableOpsMixin")'''


CLASS_PREAMBLE = """{newline}
class {cls_name}:
    __slots__ = ()"""

COPY_DOCSTRING = """\
    {method}.__doc__ = {func}.__doc__"""


def render(ops_info, is_module):
    """Render the module or stub file."""
    yield MODULE_PREAMBLE if is_module else STUBFILE_PREAMBLE

    for cls_name, method_blocks in ops_info.items():
        yield CLASS_PREAMBLE.format(cls_name=cls_name, newline="\n" * is_module)
        yield from _render_classbody(method_blocks, is_module)


def _render_classbody(method_blocks, is_module):
    for method_func_pairs, method_template, stub_template, extra in method_blocks:
        template = method_template if is_module else stub_template
        if template:
            for method, func in method_func_pairs:
                yield template.format(method=method, func=func, **extra)

    if is_module:
        yield ""
        for method_func_pairs, *_ in method_blocks:
            for method, func in method_func_pairs:
                if method and func:
                    yield COPY_DOCSTRING.format(method=method, func=func)


if __name__ == "__main__":
    option = sys.argv[1].lower() if len(sys.argv) == 2 else None
    if option not in {"--module", "--stubs"}:
        raise SystemExit(f"Usage: {sys.argv[0]} --module | --stubs")
    is_module = option == "--module"

    for line in render(ops_info, is_module):
        print(line)
