# coding: utf-8

import uuid
import pkg_resources
from functools import partial
from collections import OrderedDict

from .formatting import format_array_flat


CSS_FILE_PATH = '/'.join(('static', 'css', 'style-jupyterlab.css'))
CSS_STYLE = (pkg_resources
             .resource_string('xarray', CSS_FILE_PATH)
             .decode('utf8'))


ICONS_SVG_PATH = '/'.join(('static', 'html', 'icons-svg-inline.html'))
ICONS_SVG = (pkg_resources
             .resource_string('xarray', ICONS_SVG_PATH)
             .decode('utf8'))


def format_dims(dims, coord_names):
    if not dims:
        return ''

    dim_css_map = {k: " class='xr-has-index'" if k in coord_names else ''
                   for k, v in dims.items()}

    dims_li = "".join("<li><span{cssclass_idx}>{name}</span>: {size}</li>"
                      .format(cssclass_idx=dim_css_map[k], name=k, size=v)
                      for k, v in dims.items())

    return "<ul class='xr-dim-list'>{}</ul>".format(dims_li)


def format_values_preview(array, max_char=35):
    pprint_str = format_array_flat(array, max_char)

    return "".join("{} ".format(s) for s in pprint_str.split())


def summarize_attrs(attrs):
    attrs_li = "".join("<li>{} : {}</li>".format(k, v)
                       for k, v in attrs.items())

    return "<ul class='xr-attrs'>{}</ul>".format(attrs_li)


def _icon(icon_name):
    # icon_name should be defined in xarray/static/html/icon-svg-inline.html
    return ("<svg class='icon xr-{0}'>"
            "<use xlink:href='#{0}'>"
            "</use>"
            "</svg>"
            .format(icon_name))


def _summarize_coord_multiindex(name, coord, d):
    d['dtype'] = 'MultiIndex'
    d['preview'] = '(' +  ', '.join(l for l in coord.level_names) + ')'
    return summarize_variable(name, coord, d)


def summarize_coord(name, var):
    d = {}
    is_index = name in var.dims
    d['cssclass_idx'] = " class='xr-has-index'" if is_index else ""
    if is_index:
        coord = var.variable.to_index_variable()
        if coord.level_names is not None:
            coords = {}
            coords[name] = _summarize_coord_multiindex(name, coord, d)

            for lname in coord.level_names:
                var = coord.get_level_variable(lname)
                coords[lname] = summarize_variable(lname, var)

            return coords

    return {name: summarize_variable(name, var.variable, d)}


def summarize_coords(variables):
    coords = {}
    for k, v in variables.items():
        coords.update(**summarize_coord(k, v))

    vars_li = "".join("<li class='xr-var-item'>{}</li>"
                     .format(v) for v in coords.values())

    return "<ul class='xr-var-list'>{}</ul>".format(vars_li)


def summarize_variable(name, var, d=None):
    if d is None:
        d = {'cssclass_idx': ""}

    d['dims_str'] = '(' +  ', '.join(dim for dim in var.dims) + ')'
    d['name'] = name
    d['dtype'] = d.get('dtype', var.dtype)

    # "unique" ids required to expand/collapse subsections
    d['attrs_id'] = 'attrs-' + str(uuid.uuid4())
    d['data_id'] = 'data-' + str(uuid.uuid4())

    if len(var.attrs):
        d['disabled'] = ''
        d['attrs'] = summarize_attrs(var.attrs)
    else:
        d['disabled'] = 'disabled'
        d['attrs'] = ''

    # TODO: no value preview if not in memory
    d['preview'] = d.get('preview', format_values_preview(var))
    d['attrs_ul'] = summarize_attrs(var.attrs)
    d['data_repr'] = repr(var.data)

    d['attrs_icon'] = _icon('icon-file-text2')
    d['data_icon'] = _icon('icon-database')

    return (
        "<div class='xr-var-name'><span{cssclass_idx}>{name}</span></div>"
        "<div class='xr-var-dims'>{dims_str}</div>"
        "<div class='xr-var-dtype'>{dtype}</div>"
        "<div class='xr-var-preview xr-preview'>{preview}</div>"
        "<input id='{attrs_id}' class='xr-var-attrs-in' "
        "type='checkbox' {disabled}>"
        "<label for='{attrs_id}' title='Show/Hide attributes'>"
        "{attrs_icon}</label>"
        "<input id='{data_id}' class='xr-var-data-in' type='checkbox'>"
        "<label for='{data_id}' title='Show/Hide data repr'>"
        "{data_icon}</label>"
        "<div class='xr-var-attrs'>{attrs_ul}</div>"
        "<pre class='xr-var-data'>{data_repr}</pre>"
        .format(**d))


def summarize_vars(variables):
    vars_li = "".join("<li class='xr-var-item'>{}</li>"
                      .format(summarize_variable(k, v))
                      for k, v in variables.items())

    return "<ul class='xr-var-list'>{}</ul>".format(vars_li)


def collapsible_section(name, inline_details=None, details=None,
                        n_items=None, enabled=True, collapsed=False):
    d = {}

    # "unique" id to expand/collapse the section
    d['id'] = 'section-' + str(uuid.uuid4())

    if n_items is not None:
        n_items_span = " <span>({})</span>".format(n_items)
    else:
        n_items_span = ''

    d['title'] = "{}:{}".format(name, n_items_span)

    if n_items is not None and not n_items:
        collapsed = True

    d['inline_details'] = inline_details or ''
    d['details'] = details or ''

    d['enabled'] = '' if enabled else 'disabled'
    d['collapsed'] = '' if collapsed else 'checked'

    if enabled:
        d['tip'] = " title='Expand/collapse section'"
    else:
        d['tip'] = ""

    return (
        "<input id='{id}' class='xr-section-summary-in' "
        "type='checkbox' {enabled} {collapsed}>"
        "<label for='{id}' class='xr-section-summary' {tip}>{title}</label>"
        "<div class='xr-section-inline-details'>{inline_details}</div>"
        "<div class='xr-section-details'>{details}</div>"
        .format(**d))


def _mapping_section(mapping, name, details_func,
                     enabled=True, max_items_collapse=None):
    n_items = len(mapping)

    if max_items_collapse is not None and n_items <= max_items_collapse:
        collapsed = False
    else:
        collapsed = True

    return collapsible_section(
        name, details=details_func(mapping), n_items=n_items,
        enabled=enabled, collapsed=collapsed
    )


def dim_section(obj):
    dim_list = format_dims(obj.dims, list(obj.coords))

    return collapsible_section('Dimensions', inline_details=dim_list,
                               enabled=False, collapsed=True)


def array_section(obj):
    d = {}

    # "unique" id to expand/collapse the section
    d['id'] = 'section-' + str(uuid.uuid4())

    # TODO: no value preview if not in memory
    d['preview'] = format_values_preview(obj.values, max_char=70)

    d['data_repr'] = repr(obj.data)

    # TODO: maybe collapse section dep. on number of lines in data repr
    d['collapsed'] = ''

    d['tip'] = "Show/hide data repr"
    d['data_icon'] = _icon('icon-database')
    return (
        "<div class='xr-array-wrap'>"
        "<input id='{id}' class='xr-array-in' type='checkbox' {collapsed}>"
        "<label for='{id}' title='{tip}'>{data_icon}</label>"
        "<div class='xr-array-preview xr-preview'><span>{preview}</span></div>"
        "<pre class='xr-array-data'>{data_repr}</pre>"
        "</div>"
        .format(**d))


coord_section = partial(_mapping_section,
                        name='Coordinates', details_func=summarize_coords,
                        max_items_collapse=25)


datavar_section = partial(_mapping_section,
                          name='Data variables', details_func=summarize_vars,
                          max_items_collapse=15)


attr_section = partial(_mapping_section,
                       name='Attributes', details_func=summarize_attrs,
                       max_items_collapse=10)


def _obj_repr(header_components, sections):
    d = {}

    d['header'] = "<div class='xr-header'>{}</div>".format(
        "".join(comp for comp in header_components))

    d['icons'] = ICONS_SVG
    d['style'] = "<style>{}</style>".format(CSS_STYLE)

    d['sections'] = "".join("<li class='xr-section-item'>{}</li>".format(s)
                            for s in sections)

    return ("<div>"
            "{icons}{style}"
            "<div class='xr-wrap'>"
            "{header}"
            "<ul class='xr-sections'>{sections}</ul>"
            "</div>"
            "</div>"
            .format(**d))


def array_repr(arr):
    dims = OrderedDict((k, v) for k, v in zip(arr.dims, arr.shape))

    obj_type = "xarray.{}".format(type(arr).__name__)

    if hasattr(arr, 'name') and arr.name is not None:
        arr_name = "'{}'".format(arr.name)
    else:
        arr_name = ""

    if hasattr(arr, 'coords'):
        coord_names = list(arr.coords)
    else:
        coord_names = []

    header_components = [
        "<div class='xr-obj-type'>{}</div>".format(obj_type),
        "<div class='xr-array-name'>{}</div>".format(arr_name),
        format_dims(dims, coord_names)
    ]

    sections = []

    sections.append(array_section(arr))

    if hasattr(arr, 'coords'):
        sections.append(coord_section(arr.coords))

    sections.append(attr_section(arr.attrs))

    return _obj_repr(header_components, sections)


def dataset_repr(ds):
    obj_type = "xarray.{}".format(type(ds).__name__)

    header_components = ["<div class='xr-obj-type'>{}</div>".format(obj_type)]

    sections = [dim_section(ds),
                coord_section(ds.coords),
                datavar_section(ds.data_vars),
                attr_section(ds.attrs)]

    return _obj_repr(header_components, sections)
