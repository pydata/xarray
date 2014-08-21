from datetime import datetime
import itertools

import numpy as np
import pandas as pd

from .pycompat import iteritems, itervalues, unicode_type, bytes_type


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
    if n_desired < x.size:
        indexer = _get_indexer_at_least_n_items(x.shape, n_desired)
        x = x[indexer]
    return np.asarray(x).flat[:n_desired]


def format_item(x):
    """Returns a succinct summary of an object as a string"""
    if isinstance(x, (np.datetime64, datetime)):
        date_str, time_str = str(pd.Timestamp(x)).split()
        if time_str == '00:00:00':
            return date_str
        else:
            return '%sT%s' % (date_str, time_str)
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


def summarize_var(name, var, first_col_width, max_width=100, show_values=True):
    first_col = pretty_print('    %s ' % name, first_col_width)
    dims_str = '(%s) ' % ', '.join(map(str, var.dims)) if var.dims else ''
    front_str = first_col + dims_str + ('%s ' % var.dtype)
    if show_values:
        # print '%s: showing values' % name
        values_str = format_array_flat(var, max_width - len(front_str))
    else:
        values_str = '...'
    return front_str + values_str


def coords_repr(coords):
    col_width = (max(len(str(k)) for k in coords) if coords else 0) + 5
    summary = ['Coordinates:']
    summary.extend(summarize_var(k, v, col_width) for k, v in coords.items())
    return '\n'.join(summary)


def _summarize_attributes(data, indent='    '):
    if data.attrs:
        attr_summary = '\n'.join('%s%s: %s' % (indent, k, v) for k, v
                                 in iteritems(data.attrs))
    else:
        attr_summary = indent + 'Empty'
    return attr_summary


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
        other_vars = [k for k in arr.dataset
                      if k not in arr.coords and k != arr.name]
        if other_vars:
            summary.append('Linked dataset variables:')
            summary.append('    ' + ', '.join(other_vars))
    summary.append('Attributes:\n%s' % _summarize_attributes(arr))
    return '\n'.join(summary)


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


def dataset_repr(ds, preview_all_values=False):
    summary = ['<xray.%s>' % type(ds).__name__]

    max_name_length = max(len(str(k)) for k in ds.variables) if ds else 0
    first_col_width = max(5 + max_name_length, 16)
    coords_str = pretty_print('Dimensions:', first_col_width)
    all_dim_strings = ['%s: %s' % (k, v) for k, v in iteritems(ds.dims)]
    summary.append('%s(%s)' % (coords_str, ', '.join(all_dim_strings)))

    def summarize_variables(variables, always_show_values):
        return ([summarize_var(v.name, v, first_col_width,
                               show_values=(always_show_values or v._in_memory))
                 for v in itervalues(variables)]
                or ['    Empty'])

    summary.append('Coordinates:')
    summary.extend(summarize_variables(ds.coords, always_show_values=True))

    summary.append('Noncoordinates:')
    summary.extend(summarize_variables(
        ds.noncoords, always_show_values=preview_all_values))

    summary.append('Attributes:\n%s' % _summarize_attributes(ds, '    '))

    return '\n'.join(summary)
