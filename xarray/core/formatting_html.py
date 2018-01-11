# coding: utf-8

import uuid
import pkg_resources
from functools import partial
from collections import OrderedDict

from .formatting import format_array_flat


CSS_FILE_PATH = '/'.join(('static', 'css', 'style-jupyterlab.css'))
CSS_STYLE = pkg_resources.resource_string('xarray', CSS_FILE_PATH)


def format_dims(dims, coord_names):
    if not dims:
        return ''

    dim_css_map = {k: " class='xr-has-index'" if k in coord_names else ''
                   for k, v in dims.items()}

    dims_li = "".join("<li><span{cssclass}>{name}</span>: {size}</li>"
                      .format(cssclass=dim_css_map[k], name=k, size=v)
                      for k, v in dims.items())

    return "<ul class='xr-dim-list'>{}</ul>".format(dims_li)


def format_values_preview(array, max_char=35):
    pprint_str = format_array_flat(array, max_char)

    return "".join("<span>{} </span>".format(s)
                   for s in pprint_str.split())


def summarize_attrs(attrs):
    attrs_li = "".join("<li>{} : {}</li>".format(k, v)
                       for k, v in attrs.items())

    return "<ul class='xr-attr-list'>{}</ul>".format(attrs_li)


def summarize_variable(name, var):
    d = {}

    d['dims_str'] = '(' +  ', '.join(dim for dim in var.dims) + ')'

    d['name'] = name
    d['cssclass_varname'] = 'xr-varname'
    if name in var.dims:
        d['cssclass_varname'] += ' xr-has-index'

    d['dtype'] = var.dtype

    # "unique" ids required to expand/collapse subsections
    d['attrs_id'] = 'attrs-' + str(uuid.uuid4())
    d['values_id'] = 'values-' + str(uuid.uuid4())

    if len(var.attrs):
        d['disabled'] = ''
        d['attrs'] = summarize_attrs(var.attrs)
    else:
        d['disabled'] = 'disabled'
        d['attrs'] = ''

    # TODO: no value preview if not in memory
    d['values_preview'] = format_values_preview(var)
    d['attrs_subsection'] = summarize_attrs(var.attrs)
    d['data_repr_subsection'] = repr(var.data)

    return (
        "<input id='{attrs_id}' class='xr-varname-in' "
        "type='checkbox' {disabled}>"
        "<label class='{cssclass_varname}' for='{attrs_id}'>{name}</label>"
        "<span class='xr-dims'>{dims_str}</span>"
        "<span class='xr-dtype'>{dtype}</span>"
        "<input id='{values_id}' class='xr-values-in' type='checkbox'>"
        "<label for='{values_id}' class='xr-values'>{values_preview}</label>"
        "{attrs_subsection}"
        "<pre class='xr-data-repr'>{data_repr_subsection}</pre>"
        .format(**d))


def summarize_vars(variables):
    vars_li = "".join("<li>{}</li>".format(summarize_variable(k, v))
                      for k, v in variables.items())

    return "<ul class='xr-var-list'>{}</ul>".format(vars_li)


def collapsible_section(name, body, n_items=None,
                        enabled=True, collapsed=False,
                        input_cssclass='xr-section-in'):
    d = {}

    d['input_cssclass'] = input_cssclass

    # "unique" id to expand/collapse the section
    d['section_id'] = 'section-' + str(uuid.uuid4())

    if name is not None:
        if n_items is not None:
            n_items_span = " <span>({})</span>".format(n_items)
        else:
            n_items_span = ''

        d['title'] = "{}:{}".format(name, n_items_span)

    else:
        d['title'] = ""

    if n_items is not None and not n_items:
        collapsed = True

    d['body'] = body

    d['enabled'] = '' if enabled else 'disabled'
    d['collapsed'] = '' if collapsed else 'checked'

    return (
        "<input id='{section_id}' class='{input_cssclass}' "
        "type='checkbox' {enabled} {collapsed}>"
        "<label for='{section_id}'>{title}</label>"
        "{body}"
        .format(**d))


def _mapping_section(mapping, name, body_func,
                     enabled=True, max_items_collapse=None):
    n_items = len(mapping)

    if max_items_collapse is not None and n_items <= max_items_collapse:
        collapsed = False
    else:
        collapsed = True

    return collapsible_section(
        name, body_func(mapping), n_items=n_items,
        enabled=enabled, collapsed=collapsed
    )


def dim_section(obj):
    body = format_dims(obj.dims, list(obj.coords))

    return collapsible_section('Dimensions', body,
                               enabled=False, collapsed=True)


def array_section(obj):
    # TODO: no value preview if not in memory
    values_preview_div = "<div>{}</div>".format(
        format_values_preview(obj.values, max_char=70))

    data_repr_pre = "<pre>{}</pre>".format(repr(obj.data))

    body = values_preview_div + data_repr_pre

    # TODO: maybe collapse section dep. on number of lines in <pre>
    return collapsible_section(None, body, input_cssclass='xr-dataarray-in')


coord_section = partial(_mapping_section,
                        name='Coordinates', body_func=summarize_vars,
                        max_items_collapse=25)


datavar_section = partial(_mapping_section,
                          name='Data variables', body_func=summarize_vars,
                          max_items_collapse=15)


attr_section = partial(_mapping_section,
                       name='Attributes', body_func=summarize_attrs,
                       max_items_collapse=10)


def _obj_repr(header_components, sections):
    d = {}

    d['header'] = "<div class='xr-header'>{}</div>".format(
        "".join(comp for comp in header_components))

    d['style'] = "<style>{}</style>".format(CSS_STYLE)

    d['sections'] = "".join("<li>{}</li>".format(s)
                            for s in sections)

    return ("<div>{style}<div class='xr-wrap'>"
            "{header}<ul class='xr-sections'>{sections}</ul>"
            "</div></div>"
            .format(**d))


def array_repr(arr):
    dims = OrderedDict((k, v) for k, v in zip(arr.dims, arr.shape))

    arr_type = "xarray.{}".format(type(arr).__name__)

    if hasattr(arr, 'name') and arr.name is not None:
        arr_name = "'{}'".format(arr.name)
    else:
        arr_name = ""

    if hasattr(arr, 'coords'):
        coord_names = list(arr.coords)
    else:
        coord_names = []

    header_components = [
        "<div class='xr-dataarray-cls'>{}</div>".format(arr_type),
        "<div class='xr-dataarray-name'>{}</div>".format(arr_name),
        format_dims(dims, coord_names)
    ]

    sections = []

    sections.append(array_section(arr))

    if hasattr(arr, 'coords'):
        sections.append(coord_section(arr.coords))

    sections.append(attr_section(arr.attrs))

    return _obj_repr(header_components, sections)


def dataset_repr(ds):
    header_components = ["xarray.{}".format(type(ds).__name__)]

    sections = [dim_section(ds),
                coord_section(ds.coords),
                datavar_section(ds.data_vars),
                attr_section(ds.attrs)]

    return _obj_repr(header_components, sections)
