"""Generate module and stub file for arithmetic operators of various xarray classes.

For internal xarray development use only. Requires that jinja2 is installed.

Usage:
    pixi run generate-ops

or, without pixi:
    python xarray/util/generate_ops.py > xarray/core/_typed_ops.py
    pre-commit run --files xarray/core/_typed_ops.py

"""

# Note: the comments in https://github.com/pydata/xarray/pull/4904 provide some
# background to some of the design choices made here.

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import jinja2

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
    ("__lshift__", "operator.lshift"),
    ("__rshift__", "operator.rshift"),
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
    ("__ilshift__", "operator.ilshift"),
    ("__irshift__", "operator.irshift"),
)
UNARY_OPS = (
    ("__neg__", "operator.neg"),
    ("__pos__", "operator.pos"),
    ("__abs__", "operator.abs"),
    ("__invert__", "operator.invert"),
)
# round method and numpy/pandas unary methods which don't modify the data shape,
# so the result should still be wrapped in a Variable/DataArray/Dataset
OTHER_UNARY_METHODS = (
    ("round", "ops.round_"),
    ("argsort", "ops.argsort"),
    ("conj", "ops.conj"),
    ("conjugate", "ops.conjugate"),
)


required_method_binary = """
    def _binary_op{{ type_params }}(
        self, other: {{ other_type }}, f: Callable, reflexive: bool = False
    ) -> {{ return_type }}:
        raise NotImplementedError"""
template_binop = """
    def {{ method }}{{ type_params }}(self, other: {{ other_type }}) -> {{ return_type }}:{{ type_ignore }}
        return self._binary_op(other, {{ func }})"""
template_binop_overload = """
{%- for overload_type in overload_types %}
    @overload{{ overload_type_ignore if loop.first else "" }}
{%- if overload_type in subclass_overload_types %}
    def {{ method }}[T: {{ overload_type }}](self, other: T) -> T: ...
{%- else %}
    def {{ method }}(self, other: {{ overload_type }}) -> {{ overload_type }}: ...
{%- endif %}
{% endfor %}
    @overload
    def {{method}}(self, other: {{ other_type }}) -> {{ return_type }}: ...

    def {{ method }}(self, other: {{ other_type }}) -> {{ return_type }} | {{ ' | '.join(overload_types) }}:{{ type_ignore }}
        return self._binary_op(other, {{ func }})"""
template_reflexive = """
    def {{ method }}{{ type_params }}(self, other: {{ other_type }}) -> {{ return_type }}:
        return self._binary_op(other, {{ func }}, reflexive=True)"""

required_method_inplace = """
    def _inplace_binary_op(self, other: {{ other_type }}, f: Callable) -> Self:
        raise NotImplementedError"""
template_inplace = """
    def {{ method }}(self, other: {{ other_type }}) -> Self:{{type_ignore}}
        return self._inplace_binary_op(other, {{ func }})"""

required_method_unary = """
    def _unary_op(self, f: Callable, *args: Any, **kwargs: Any) -> Self:
        raise NotImplementedError"""
template_unary = """
    def {{ method }}(self) -> Self:
        return self._unary_op({{ func }})"""
template_other_unary = """
    def {{ method }}(self, *args: Any, **kwargs: Any) -> Self:
        return self._unary_op({{ func }}, *args, **kwargs)"""
unhashable = """
    # When __eq__ is defined but __hash__ is not, then an object is unhashable,
    # and it should be declared as follows:
    __hash__: None  # type:ignore[assignment]"""

# __eq__ and __ne__ return an array instead of the `bool` of `object.__eq__`, so
# they need a "# type: ignore[override]". Keep an eye out for:
# https://discuss.python.org/t/make-type-hints-for-eq-of-primitives-less-strict/34240
#
# The overloads tell type checkers that e.g. Variable + DataArray = DataArray. At
# runtime, Variable.__add__ returns NotImplemented and Python calls
# DataArray.__radd__ instead. Type checkers would do the same, if Variable.__add__
# didn't accept a DataArray. But VarCompatible contains ArrayLike, which every
# object with an __array__ method matches, so it does accept DataArray.
#
# In-place operations keep the type of the left operand, e.g. Variable += DataArray
# is still a Variable. Mypy reports this as incompatible with the overloads of the
# corresponding binary operator, which requires a "# type: ignore[misc]".


def _type_ignore(ignore: str) -> str:
    return f"  # type:ignore[{ignore}]" if ignore else ""


FuncType = Sequence[tuple[str | None, str | None]]
OpsType = tuple[FuncType, str, dict[str, Any]]


def binops(
    other_type: str,
    return_type: str = "Self",
    type_ignore_eq: str = "override",
    type_params: str = "",
) -> list[OpsType]:
    extras = {
        "other_type": other_type,
        "return_type": return_type,
        "type_params": type_params,
    }
    return [
        ([(None, None)], required_method_binary, extras),
        (BINOPS_NUM + BINOPS_CMP, template_binop, extras | {"type_ignore": ""}),
        (
            BINOPS_EQNE,
            template_binop,
            extras | {"type_ignore": _type_ignore(type_ignore_eq)},
        ),
        ([(None, None)], unhashable, extras),
        (BINOPS_REFLEXIVE, template_reflexive, extras),
    ]


def binops_overload(
    other_type: str,
    overload_types: list[str],
    subclass_overload_types: Sequence[str] = (),
    return_type: str = "Self",
    type_ignore_eq: str = "override",
) -> list[OpsType]:
    """Binary operations with an overload for each of ``overload_types``.

    The overloads of ``subclass_overload_types`` return the type of ``other``, so
    that subclasses are kept.
    """
    extras = {
        "other_type": other_type,
        "return_type": return_type,
        "type_params": "",
        "subclass_overload_types": subclass_overload_types,
    }
    return [
        ([(None, None)], required_method_binary, extras),
        (
            BINOPS_NUM + BINOPS_CMP,
            template_binop_overload,
            extras
            | {
                "overload_types": overload_types,
                "type_ignore": "",
                "overload_type_ignore": "",
            },
        ),
        (
            BINOPS_EQNE,
            template_binop_overload,
            extras
            | {
                "overload_types": overload_types,
                "type_ignore": "",
                "overload_type_ignore": _type_ignore(type_ignore_eq),
            },
        ),
        ([(None, None)], unhashable, extras),
        (BINOPS_REFLEXIVE, template_reflexive, extras),
    ]


def inplace(other_type: str, type_ignore: str = "") -> list[OpsType]:
    extras = {"other_type": other_type}
    return [
        ([(None, None)], required_method_inplace, extras),
        (
            BINOPS_INPLACE,
            template_inplace,
            extras | {"type_ignore": _type_ignore(type_ignore)},
        ),
    ]


def unops() -> list[OpsType]:
    return [
        ([(None, None)], required_method_unary, {}),
        (UNARY_OPS, template_unary, {}),
        (OTHER_UNARY_METHODS, template_other_unary, {}),
    ]


ops_info = {
    # TODO add inplace ops for DataTree?
    "DataTreeOpsMixin": binops(other_type="DtCompatible") + unops(),
    "DatasetOpsMixin": (
        binops_overload(other_type="DsCompatible", overload_types=["DataTree"])
        + inplace(other_type="DsCompatible", type_ignore="misc")
        + unops()
    ),
    "DataArrayOpsMixin": (
        binops_overload(
            other_type="DaCompatible", overload_types=["Dataset", "DataTree"]
        )
        + inplace(other_type="DaCompatible", type_ignore="misc")
        + unops()
    ),
    "VariableOpsMixin": (
        binops_overload(
            other_type="VarCompatible",
            overload_types=["DataArray", "Dataset", "DataTree"],
            subclass_overload_types=["DataArray"],
        )
        + inplace(other_type="VarCompatible", type_ignore="misc")
        + unops()
    ),
    "DatasetGroupByOpsMixin": binops(
        other_type="Dataset | DataArray", return_type="Dataset"
    ),
    # the result has the type of the other operand
    "DataArrayGroupByOpsMixin": binops(
        other_type="T", return_type="T", type_params="[T: (DataArray, Dataset)]"
    ),
}

MODULE_PREAMBLE = '''\
"""Mixin classes with arithmetic operators."""

# This file was generated using xarray.util.generate_ops. Do not edit manually.

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self, overload

from xarray.computation import ops
from xarray.core import nputils
from xarray.core.types import DaCompatible, DsCompatible, DtCompatible, VarCompatible

if TYPE_CHECKING:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.datatree import DataTree'''


CLASS_PREAMBLE = """{newline}
class {cls_name}:
    __slots__ = ()"""

COPY_DOCSTRING = """\
    {method}.__doc__ = {func}.__doc__"""


def render(ops_info: dict[str, list[OpsType]]) -> Iterator[str]:
    """Render the module or stub file."""
    yield MODULE_PREAMBLE

    for cls_name, method_blocks in ops_info.items():
        yield CLASS_PREAMBLE.format(cls_name=cls_name, newline="\n")
        yield from _render_classbody(method_blocks)


def _render_classbody(method_blocks: list[OpsType]) -> Iterator[str]:
    environment = jinja2.Environment()

    for method_func_pairs, template, extra in method_blocks:
        if template:
            for method, func in method_func_pairs:
                yield environment.from_string(template).render(
                    method=method, func=func, **extra
                )

    yield ""
    for method_func_pairs, *_ in method_blocks:
        for method, func in method_func_pairs:
            if method and func:
                yield COPY_DOCSTRING.format(method=method, func=func)


if __name__ == "__main__":
    for line in render(ops_info):
        print(line)
