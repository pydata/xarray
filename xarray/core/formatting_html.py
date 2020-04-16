import uuid
from collections import OrderedDict
from functools import partial
from html import escape

import pkg_resources

from .formatting import inline_variable_array_repr, short_data_repr

CSS_FILE_PATH = "/".join(("static", "css", "style.css"))
CSS_STYLE = pkg_resources.resource_string("xarray", CSS_FILE_PATH).decode("utf8")


ICONS_SVG_PATH = "/".join(("static", "html", "icons-svg-inline.html"))
ICONS_SVG = pkg_resources.resource_string("xarray", ICONS_SVG_PATH).decode("utf8")


def short_data_repr_html(array):
    """Format "data" for DataArray and Variable."""
    internal_data = getattr(array, "variable", array)._data
    if hasattr(internal_data, "_repr_html_"):
        return internal_data._repr_html_()
    return escape(short_data_repr(array))


def format_dims(dims, coord_names):
    if not dims:
        return ""

    dim_css_map = {
        k: " class='xr-has-index'" if k in coord_names else "" for k, v in dims.items()
    }

    dims_li = "".join(
        f"<li><span{dim_css_map[dim]}>" f"{escape(dim)}</span>: {size}</li>"
        for dim, size in dims.items()
    )

    return f"<ul class='xr-dim-list'>{dims_li}</ul>"


def summarize_attrs(attrs):
    attrs_dl = "".join(
        f"<dt><span>{escape(k)} :</span></dt>" f"<dd>{escape(str(v))}</dd>"
        for k, v in attrs.items()
    )

    return f"<dl class='xr-attrs'>{attrs_dl}</dl>"


def _icon(icon_name):
    # icon_name should be defined in xarray/static/html/icon-svg-inline.html
    return (
        "<svg class='icon xr-{0}'>"
        "<use xlink:href='#{0}'>"
        "</use>"
        "</svg>".format(icon_name)
    )


def _summarize_coord_multiindex(name, coord):
    preview = f"({', '.join(escape(l) for l in coord.level_names)})"
    return summarize_variable(
        name, coord, is_index=True, dtype="MultiIndex", preview=preview
    )


def summarize_coord(name, var):
    is_index = name in var.dims
    if is_index:
        coord = var.variable.to_index_variable()
        if coord.level_names is not None:
            coords = {}
            coords[name] = _summarize_coord_multiindex(name, coord)
            for lname in coord.level_names:
                var = coord.get_level_variable(lname)
                coords[lname] = summarize_variable(lname, var)
            return coords

    return {name: summarize_variable(name, var, is_index)}


def summarize_coords(variables):
    coords = {}
    for k, v in variables.items():
        coords.update(**summarize_coord(k, v))

    vars_li = "".join(f"<li class='xr-var-item'>{v}</li>" for v in coords.values())

    return f"<ul class='xr-var-list'>{vars_li}</ul>"


def summarize_variable(name, var, is_index=False, dtype=None, preview=None):
    variable = var.variable if hasattr(var, "variable") else var

    cssclass_idx = " class='xr-has-index'" if is_index else ""
    dims_str = f"({', '.join(escape(dim) for dim in var.dims)})"
    name = escape(str(name))
    dtype = dtype or escape(str(var.dtype))

    # "unique" ids required to expand/collapse subsections
    attrs_id = "attrs-" + str(uuid.uuid4())
    data_id = "data-" + str(uuid.uuid4())
    disabled = "" if len(var.attrs) else "disabled"

    preview = preview or escape(inline_variable_array_repr(variable, 35))
    attrs_ul = summarize_attrs(var.attrs)
    data_repr = short_data_repr_html(variable)

    attrs_icon = _icon("icon-file-text2")
    data_icon = _icon("icon-database")

    return (
        f"<div class='xr-var-name'><span{cssclass_idx}>{name}</span></div>"
        f"<div class='xr-var-dims'>{dims_str}</div>"
        f"<div class='xr-var-dtype'>{dtype}</div>"
        f"<div class='xr-var-preview xr-preview'>{preview}</div>"
        f"<input id='{attrs_id}' class='xr-var-attrs-in' "
        f"type='checkbox' {disabled}>"
        f"<label for='{attrs_id}' title='Show/Hide attributes'>"
        f"{attrs_icon}</label>"
        f"<input id='{data_id}' class='xr-var-data-in' type='checkbox'>"
        f"<label for='{data_id}' title='Show/Hide data repr'>"
        f"{data_icon}</label>"
        f"<div class='xr-var-attrs'>{attrs_ul}</div>"
        f"<pre class='xr-var-data'>{data_repr}</pre>"
    )


def summarize_vars(variables):
    vars_li = "".join(
        f"<li class='xr-var-item'>{summarize_variable(k, v)}</li>"
        for k, v in variables.items()
    )

    return f"<ul class='xr-var-list'>{vars_li}</ul>"


def collapsible_section(
    name, inline_details="", details="", n_items=None, enabled=True, collapsed=False
):
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())

    has_items = n_items is not None and n_items
    n_items_span = "" if n_items is None else f" <span>({n_items})</span>"
    enabled = "" if enabled and has_items else "disabled"
    collapsed = "" if collapsed or not has_items else "checked"
    tip = " title='Expand/collapse section'" if enabled else ""

    return (
        f"<input id='{data_id}' class='xr-section-summary-in' "
        f"type='checkbox' {enabled} {collapsed}>"
        f"<label for='{data_id}' class='xr-section-summary' {tip}>"
        f"{name}:{n_items_span}</label>"
        f"<div class='xr-section-inline-details'>{inline_details}</div>"
        f"<div class='xr-section-details'>{details}</div>"
    )


def _mapping_section(mapping, name, details_func, max_items_collapse, enabled=True):
    n_items = len(mapping)
    collapsed = n_items >= max_items_collapse

    return collapsible_section(
        name,
        details=details_func(mapping),
        n_items=n_items,
        enabled=enabled,
        collapsed=collapsed,
    )


def dim_section(obj):
    dim_list = format_dims(obj.dims, list(obj.coords))

    return collapsible_section(
        "Dimensions", inline_details=dim_list, enabled=False, collapsed=True
    )


def array_section(obj):
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())
    collapsed = ""
    preview = escape(inline_variable_array_repr(obj.variable, max_width=70))
    data_repr = short_data_repr_html(obj)
    data_icon = _icon("icon-database")

    return (
        "<div class='xr-array-wrap'>"
        f"<input id='{data_id}' class='xr-array-in' type='checkbox' {collapsed}>"
        f"<label for='{data_id}' title='Show/hide data repr'>{data_icon}</label>"
        f"<div class='xr-array-preview xr-preview'><span>{preview}</span></div>"
        f"<pre class='xr-array-data'>{data_repr}</pre>"
        "</div>"
    )


coord_section = partial(
    _mapping_section,
    name="Coordinates",
    details_func=summarize_coords,
    max_items_collapse=25,
)


datavar_section = partial(
    _mapping_section,
    name="Data variables",
    details_func=summarize_vars,
    max_items_collapse=15,
)


attr_section = partial(
    _mapping_section,
    name="Attributes",
    details_func=summarize_attrs,
    max_items_collapse=10,
)


def _obj_repr(header_components, sections):
    header = f"<div class='xr-header'>{''.join(h for h in header_components)}</div>"
    sections = "".join(f"<li class='xr-section-item'>{s}</li>" for s in sections)

    return (
        "<div>"
        f"{ICONS_SVG}<style>{CSS_STYLE}</style>"
        "<div class='xr-wrap'>"
        f"{header}"
        f"<ul class='xr-sections'>{sections}</ul>"
        "</div>"
        "</div>"
    )


def array_repr(arr):
    dims = OrderedDict((k, v) for k, v in zip(arr.dims, arr.shape))

    obj_type = "xarray.{}".format(type(arr).__name__)
    arr_name = "'{}'".format(arr.name) if getattr(arr, "name", None) else ""
    coord_names = list(arr.coords) if hasattr(arr, "coords") else []

    header_components = [
        "<div class='xr-obj-type'>{}</div>".format(obj_type),
        "<div class='xr-array-name'>{}</div>".format(arr_name),
        format_dims(dims, coord_names),
    ]

    sections = [array_section(arr)]

    if hasattr(arr, "coords"):
        sections.append(coord_section(arr.coords))

    sections.append(attr_section(arr.attrs))

    return _obj_repr(header_components, sections)


def dataset_repr(ds):
    obj_type = "xarray.{}".format(type(ds).__name__)

    header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}</div>"]

    sections = [
        dim_section(ds),
        coord_section(ds.coords),
        datavar_section(ds.data_vars),
        attr_section(ds.attrs),
    ]

    return _obj_repr(header_components, sections)
