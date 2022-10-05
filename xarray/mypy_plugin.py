from __future__ import annotations

from functools import partial
from typing import Callable

from mypy.nodes import CallExpr, ClassDef, Expression, StrExpr, TypeInfo
from mypy.plugin import ClassDefContext, Plugin
from mypy.plugins.common import add_attribute_to_class
from mypy.types import Instance


def accessor_callback(
    ctx: ClassDefContext, accessor_cls: ClassDef, xr_cls: ClassDef
) -> None:
    name = _get_name_arg(ctx.reason)
    cls_typ = Instance(ctx.cls.info, [])
    # TODO: when changing code and running mypy again, this fails?
    attr_typ = Instance(accessor_cls.info, [cls_typ])
    add_attribute_to_class(api=ctx.api, cls=xr_cls, name=name, typ=attr_typ)


def _get_name_arg(reason: Expression) -> str:
    assert isinstance(reason, CallExpr)
    assert len(reason.args) == 1  # only a single "name" arg
    name_expr = reason.args[0]
    assert isinstance(name_expr, StrExpr)
    return name_expr.value


class XarrayPlugin(Plugin):
    def get_class_decorator_hook(
        self, fullname: str
    ) -> Callable[[ClassDefContext], None] | None:
        for x in ("DataArray", "Dataset"):
            if fullname == f"xarray.core.extensions.register_{x.lower()}_accessor":
                xr_cls = self._get_cls(f"{x.lower()}.{x}")
                ac_cls = self._get_cls("extensions._CachedAccessor")
                return partial(accessor_callback, accessor_cls=ac_cls, xr_cls=xr_cls)
        return None

    def _get_cls(self, typename: str) -> ClassDef:
        cls = self.lookup_fully_qualified("xarray.core." + typename)
        assert cls is not None
        node = cls.node
        assert isinstance(node, TypeInfo)
        return node.defn


def plugin(version: str) -> type[XarrayPlugin]:
    """An entry-point for mypy."""
    return XarrayPlugin
