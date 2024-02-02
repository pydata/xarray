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
    N_CHILDREN = len(children) - 1

    # Get result from node_repr and wrap it
    lines_callback = lambda n, c, end: _wrap_repr(node_repr(n, c), end=end)

    children_html = "".join(
        lines_callback(n, c, end=False)  # Long lines
        if i < N_CHILDREN
        else lines_callback(n, c, end=True)  # Short lines
        for i, (n, c) in enumerate(children.items())
    )

    return "".join(
        [
            "<div style='display: inline-grid; grid-template-columns: 100%'>",
            children_html,
            "</div>",
        ]
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


def _wrap_repr(r: str, end: bool = False) -> str:
    """
    Wrap HTML representation with a tee to the left of it.

    Enclosing HTML tag is a <div> with :code:`display: inline-grid` style.

    Turns:
    [    title    ]
    |   details   |
    |_____________|

    into (A):
    |─ [    title    ]
    |  |   details   |
    |  |_____________|

    or (B):
    └─ [    title    ]
       |   details   |
       |_____________|

    Parameters
    ----------
    r: str
        HTML representation to wrap.
    end: bool
        Specify if the line on the left should continue or end.

        Default is True.

    Returns
    -------
    str
        Wrapped HTML representation.

        Tee color is set to the variable :code:`--xr-border-color`.
    """
    # height of line
    end = bool(end)
    height = "100%" if end is False else "1.2em"
    return "".join(
        [
            "<div style='display: inline-grid;'>",
            "<div style='",
            "grid-column-start: 1;",
            "border-right: 0.2em solid;",
            "border-color: var(--xr-border-color);",
            f"height: {height};",
            "width: 0px;",
            "'>",
            "</div>",
            "<div style='",
            "grid-column-start: 2;",
            "grid-row-start: 1;",
            "height: 1em;",
            "width: 20px;",
            "border-bottom: 0.2em solid;",
            "border-color: var(--xr-border-color);",
            "'>",
            "</div>",
            "<div style='",
            "grid-column-start: 3;",
            "'>",
            "<ul class='xr-sections'>",
            r,
            "</ul>" "</div>",
            "</div>",
        ]
    )


def datatree_repr(dt: Any) -> str:
    obj_type = f"datatree.{type(dt).__name__}"
    return node_repr(obj_type, dt)
