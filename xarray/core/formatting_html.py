# coding: utf-8

import uuid
from functools import partial

from .formatting import format_array_flat


XR_REPR_STYLE = """
.xr-wrap {
  width: 540px;
  font-size: 13px;
  line-height: 1.5;
  background-color: #fff;
}

.xr-wrap ul {
  padding: 0;
}

.xr-header {
  padding: 6px 0 6px 3px;
  border-bottom-width: 1px;
  border-bottom-style: solid;
  border-bottom-color: #777;
  color: #555;;
}

ul.xr-sections {
  list-style: none !important;
  padding: 3px !important;
  margin: 0 !important;
}

input.xr-section-in {
  display: none;
}

input.xr-section-in + label {
  display: inline-block;
  width: 140px;
  color: #555;
  font-weight: 500;
  padding: 4px 0 2px 0;
}

input.xr-section-in:enabled + label {
  cursor: pointer;
}

input.xr-section-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

input.xr-section-in:checked + label:before {
  content: '▼';
}

input.xr-section-in:disabled + label:before {
  color: #777;
}

input.xr-section-in + label > span {
  display: inline-block;
  margin-left: 4px;
}

input.xr-section-in:checked + label > span {
  display: none;
}

input.xr-section-in ~ ul {
  display: none;
}

input.xr-section-in:checked ~ ul {
  display: block;
}

.xr-sections summary > div {
  display: inline-block;
  cursor: pointer;
  width: 140px;
  color: #555;
  font-weight: 500;
  padding: 4px 0 2px 0;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
}

.xr-dim-list li {
  display: inline-block;
  font-size: 13px !important;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  text-decoration: underline;
}

ul.xr-var-list {
  list-style: none !important;
  padding: 0 !important;
  margin: 0 !important;
}

.xr-var-list > li {
  background-color: #fcfcfc;
  overflow: hidden;
}

.xr-var-list > li:nth-child(odd) {
  background-color: #efefef;
}

.xr-var-list li:hover {
  background-color: rgba(3, 169, 244, .2);
}

.xr-var-list li > span {
  display: inline-block;
}

.xr-var-list li input {
  display: none;
}

.xr-var-list li input:enabled + label {
  cursor: pointer;
}

input.xr-varname-in + label {
  display: inline-block;
  width: 140px;
  padding-left: 0;
}

input.xr-varname-in + label:before {
  content: ' ';
  display: inline-block;
  font-size: 11px;
  width: 15px;
  padding-left: 20px;
  padding-right: 5px;
  text-align: center;
  color: #aaa;
  text-decoration: none !important;
}

input.xr-varname-in ~ ul {
  display: none;
}

input.xr-varname-in:checked ~ ul {
  display: block;
}

input.xr-varname-in:enabled + label:before {
  content: 'a';
}

input.xr-varname-in:enabled + label:hover:before {
  color: #000;
}

input.xr-varname-in:checked + label:before {
  color: #ccc;
}

.xr-dims {
  width: 80px;
}

.xr-dtype {
  width: 96px;
  padding-right: 4px;
  text-align: right;
  color: #555;
}

.xr-values {
  width: 200px;
  text-align: left;
  color: #888;
  white-space: nowrap;
  font-size: 12px;
}

.xr-values > span:nth-child(odd) {
  color: rgba(0, 0, 0, .65);
}

input.xr-values-in + label:hover > span {
  color: #000;
}

input.xr-values-in:checked + label > span {
  color: #ccc;
}

input.xr-values-in ~ pre {
  display: none;
}

input.xr-values-in:checked ~ pre {
  display: block;
}

input.xr-values-in:checked + label > span {
  color: #ccc;
}

.xr-data-repr {
  font-size: 11px !important;
  background-color: #fff;
  padding: 4px 0 6px 40px !important;
  margin: 0 !important;
}

.xr-attr-list {
  list-style: none !important;
  background-color: #fff;
  padding: 0 0 6px 40px !important;
  color: #555;
}

.xr-attr-list li,
.xr-attr-list li:hover {
  background-color: #fff;
}
"""


def format_dims(dims, coord_names):
    dim_css_map = {k: " class='xr-has-index'" if k in coord_names else ''
                   for k, v in dims.items()}

    dims_li = "".join("<li><span{cssclass}>{name}</span>: {size}</li>"
                      .format(cssclass=dim_css_map[k], name=k, size=v)
                      for k, v in dims.items())

    return "<ul class='xr-dim-list'>{}</ul>".format(dims_li)


def format_values_preview(var):
    pprint_str = format_array_flat(var, 35)

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
        d['attrs'] = format_attrs(var.attrs)
    else:
        d['disabled'] = 'disabled'
        d['attrs'] = ''

    d['values_preview'] = format_values_preview(var)
    d['attrs_subsection'] = summarize_attrs(var.attrs)
    d['data_repr_subsection'] = repr(var.data)

    return (
        "<input id='{attrs_id}' class='xr-varname-in' type='checkbox' {disabled}>"
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
                        enabled=True, collapsed=False):
    d = {}

    # "unique" id to expand/collapse the section
    d['section_id'] = 'section-' + str(uuid.uuid4())

    if n_items is not None:
        n_items_span = " <span>({})</span>".format(n_items)
    else:
        n_items_span = ''

    d['title'] = "{}:{}".format(name, n_items_span)

    d['body'] = body

    d['enabled'] = '' if enabled else 'disabled'
    d['collapsed'] = '' if collapsed else 'checked'

    return (
        "<input id='{section_id}' class='xr-section-in' type='checkbox' {enabled} {collapsed}>"
        "<label for='{section_id}'>{title}</label>"
        "{body}"
        .format(**d))


def _generic_section(mapping, name, body_func,
                     enabled=True, nmax_items_collapse=None):
    n_items = len(mapping)

    if nmax_items_collapse is not None and n_items <= nmax_items_collapse:
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


coord_section = partial(_generic_section,
                        name='Coordinates', body_func=summarize_vars,
                        nmax_items_collapse=25)


datavar_section = partial(_generic_section,
                          name='Data variables', body_func=summarize_vars,
                          nmax_items_collapse=15)


attr_section = partial(_generic_section,
                       name='Attributes', body_func=summarize_attrs,
                       nmax_items_collapse=10)


def dataset_repr(ds):
    d = {}

    d['header'] = "<div class='xr-header'>xarray.Dataset</div>"
    d['style'] = "<style>{}</style>".format(XR_REPR_STYLE)

    sections = [dim_section(ds),
                coord_section(ds.coords),
                datavar_section(ds.data_vars),
                attr_section(ds.attrs)]

    d['sections'] = "".join("<li>{}</li>".format(s)
                            for s in sections)

    return ("<div>{style}<div class='xr-wrap'>"
            "{header}<ul class='xr-sections'>{sections}</ul>"
            "</div></div>"
            .format(**d))
