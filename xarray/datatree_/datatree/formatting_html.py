from functools import partial
from html import escape
from typing import Any, Mapping

from xarray.core.formatting_html import (
    _mapping_section,
    _obj_repr,
    attr_section,
    coord_section,
    datavar_section,
    dim_section,
)
from xarray.core.options import OPTIONS

OPTIONS["display_expand_groups"] = "default"


def summarize_children(children: Mapping[str, Any]) -> str:
    children_li = "".join(
        f"<ul class='xr-sections'>{node_repr(n, c)}</ul>" for n, c in children.items()
    )

    return (
        "<ul class='xr-sections'>"
        f"<div style='padding-left:2rem;'>{children_li}<br></div>"
        "</ul>"
    )


children_section = partial(
    _mapping_section,
    name="Groups",
    details_func=summarize_children,
    max_items_collapse=1,
    expand_option_name="display_expand_groups",
)


def node_repr(group_title: str, dt: Any) -> str:
    header_components = [f"<div class='xr-obj-type'>{escape(group_title)}</div>"]

    ds = dt.ds

    sections = [
        children_section(dt.children),
        dim_section(ds),
        coord_section(ds.coords),
        datavar_section(ds.data_vars),
        attr_section(ds.attrs),
    ]

    return _obj_repr(ds, header_components, sections)


def datatree_repr(dt: Any) -> str:
    obj_type = f"datatree.{type(dt).__name__}"
    return node_repr(obj_type, dt)
