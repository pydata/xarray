from datetime import datetime
import functools
import itertools

import numpy as np
import pandas as pd

from .pycompat import (OrderedDict, iteritems, itervalues, unicode_type,
                       bytes_type)


def pretty_print(x, numchars):
    """Given an object `x`, call `str(x)` and format the returned string so
    that it is numchars long, padding with trailing spaces or truncating with
    ellipses as necessary
    """
    s = str(x)
    if len(s) > numchars:
        return s[:(numchars - 3)] + '...'
    else:
        return s + ' ' * (numchars - len(s))


def wrap_indent(text, start='', length=None):
    if length is None:
        length = len(start)
    indent = '\n' + ' ' * length
    return start + indent.join(x for x in text.splitlines())


def _get_indexer_at_least_n_items(shape, n_desired):
    assert 0 < n_desired <= np.prod(shape)
    cum_items = np.cumprod(shape[::-1])
    n_steps = np.argmax(cum_items >= n_desired)
    stop = int(np.ceil(float(n_desired) / np.r_[1, cum_items][n_steps]))
    indexer = ((0,) * (len(shape) - 1 - n_steps) + (slice(stop),)
               + (slice(None),) * n_steps)
    return indexer


def first_n_items(x, n_desired):
    """Returns the first n_desired items of an array"""
    # Unfortunately, we can't just do x.flat[:n_desired] here because x might
    # not be a numpy.ndarray. Moreover, access to elements of x could be very
    # expensive (e.g. if it's only available over DAP), so go out of our way to
    # get them in a single call to __getitem__ using only slices.
    if n_desired < 1:
        raise ValueError('must request at least one item')

    if x.size == 0:
        # work around for https://github.com/numpy/numpy/issues/5195
        return []

    if n_desired < x.size:
        indexer = _get_indexer_at_least_n_items(x.shape, n_desired)
        x = x[indexer]
    return np.asarray(x).flat[:n_desired]


def format_timestamp(t):
    """Cast given object to a Timestamp and return a nicely formatted string"""
    datetime_str = str(pd.Timestamp(t))
    try:
        date_str, time_str = datetime_str.split()
    except ValueError:
        # catch NaT and others that don't split nicely
        return datetime_str
    else:
        if time_str == '00:00:00':
            return date_str
        else:
            return '%sT%s' % (date_str, time_str)


def format_item(x):
    """Returns a succinct summary of an object as a string"""
    if isinstance(x, (np.datetime64, datetime)):
        return format_timestamp(x)
    elif isinstance(x, (unicode_type, bytes_type)):
        return repr(x)
    elif isinstance(x, (float, np.float)):
        return '{0:.4}'.format(x)
    else:
        return str(x)


def format_array_flat(items_ndarray, max_width):
    """Return a formatted string for as many items in the flattened version of
    items_ndarray that will fit within max_width characters
    """
    # every item will take up at least two characters
    max_possibly_relevant = int(np.ceil(max_width / 2.0))
    relevant_items = first_n_items(items_ndarray, max_possibly_relevant)
    pprint_items = list(map(format_item, relevant_items))

    end_padding = ' ...'

    cum_len = np.cumsum([len(s) + 1 for s in pprint_items])
    gt_max_width = cum_len > (max_width - len(end_padding))
    if not gt_max_width.any():
        num_to_print = len(pprint_items)
    else:
        num_to_print = max(np.argmax(gt_max_width) - 1, 1)

    pprint_str = ' '.join(itertools.islice(pprint_items, int(num_to_print)))
    remaining_chars = max_width - len(pprint_str) - len(end_padding)
    if remaining_chars > 0 and num_to_print < items_ndarray.size:
        pprint_str += end_padding
    return pprint_str


def _summarize_var_or_coord(name, var, col_width, show_values=True,
                            marker=' ', max_width=100):
    first_col = pretty_print('  %s %s ' % (marker, name), col_width)
    dims_str = '(%s) ' % ', '.join(map(str, var.dims)) if var.dims else ''
    front_str = first_col + dims_str + ('%s ' % var.dtype)
    if show_values:
        values_str = format_array_flat(var, max_width - len(front_str))
    else:
        values_str = '...'
    return front_str + values_str


def _not_remote(var):
    """Helper function to identify if array is positively identifiable as
    coming from a remote source.
    """
    source = var.encoding.get('source')
    if source and source.startswith('http') and not var._in_memory:
        return False
    return True


def summarize_var(name, var, col_width):
    show_values = _not_remote(var)
    return _summarize_var_or_coord(name, var, col_width, show_values)


def summarize_coord(name, var, col_width):
    is_index = name in var.dims
    show_values = is_index or _not_remote(var)
    marker = '*' if is_index else ' '
    return _summarize_var_or_coord(name, var, col_width, show_values, marker)


def _maybe_truncate(obj, maxlen=500):
    s = str(obj)
    if len(s) > maxlen:
        s = s[:(maxlen - 3)] + '...'
    return s


def summarize_attr(key, value, col_width=None):
    # ignore col_width for now to more clearly distinguish attributes
    return '    %s: %s' % (key, _maybe_truncate(value))


EMPTY_REPR = '    *empty*'


def _calculate_col_width(mapping):
    max_name_length = max(len(str(k)) for k in mapping) if mapping else 0
    col_width = max(max_name_length, 7) + 6
    return col_width


def _mapping_repr(mapping, title, summarizer, col_width=None):
    if col_width is None:
        col_width = _calculate_col_width(mapping)
    summary = ['%s:' % title]
    if mapping:
        summary += [summarizer(k, v, col_width) for k, v in mapping.items()]
    else:
        summary += [EMPTY_REPR]
    return '\n'.join(summary)


coords_repr = functools.partial(_mapping_repr, title='Coordinates',
                                summarizer=summarize_coord)


vars_repr = functools.partial(_mapping_repr, title='Data variables',
                              summarizer=summarize_var)


attrs_repr = functools.partial(_mapping_repr, title='Attributes',
                               summarizer=summarize_attr)


def indexes_repr(indexes):
    summary = []
    for k, v in indexes.items():
        summary.append(wrap_indent(repr(v), '%s: ' % k))
    return '\n'.join(summary)


def array_repr(arr):
    # used for DataArray, Variable and Coordinate
    if hasattr(arr, 'name') and arr.name is not None:
        name_str = '%r ' % arr.name
    else:
        name_str = ''
    dim_summary = ', '.join('%s: %s' % (k, v) for k, v
                            in zip(arr.dims, arr.shape))

    summary = ['<xray.%s %s(%s)>'% (type(arr).__name__, name_str, dim_summary)]

    if arr.size < 1e5 or arr._in_memory:
        summary.append(repr(arr.values))
    else:
        summary.append('[%s values with dtype=%s]' % (arr.size, arr.dtype))

    if hasattr(arr, 'coords'):
        if arr.coords:
            summary.append(repr(arr.coords))

    if arr.attrs:
        summary.append(attrs_repr(arr.attrs))

    return '\n'.join(summary)


def dataset_repr(ds):
    summary = ['<xray.%s>' % type(ds).__name__]

    col_width = _calculate_col_width(ds)

    dims_start = pretty_print('Dimensions:', col_width)
    all_dim_strings = ['%s: %s' % (k, v) for k, v in iteritems(ds.dims)]
    summary.append('%s(%s)' % (dims_start, ', '.join(all_dim_strings)))

    summary.append(coords_repr(ds.coords, col_width=col_width))
    summary.append(vars_repr(ds.data_vars, col_width=col_width))
    if ds.attrs:
        summary.append(attrs_repr(ds.attrs))

    return '\n'.join(summary)
