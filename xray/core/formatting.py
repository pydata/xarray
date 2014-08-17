from .pycompat import iteritems


def _summarize_attributes(data):
    if data.attrs:
        attr_summary = '\n'.join('    %s: %s' % (k, v) for k, v
                                 in iteritems(data.attrs))
    else:
        attr_summary = '    Empty'
    return attr_summary


def wrap_indent(text, start='', length=None):
    if length is None:
        length = len(start)
    indent = '\n' + ' ' * length
    return start + indent.join(x for x in text.splitlines())


def array_repr(arr):
    if hasattr(arr, 'name') and arr.name is not None:
        name_str = '%r ' % arr.name
    else:
        name_str = ''
    dim_summary = ', '.join('%s: %s' % (k, v) for k, v
                            in zip(arr.dims, arr.shape))
    summary = ['<xray.%s %s(%s)>'% (type(arr).__name__, name_str, dim_summary)]
    if arr.size < 1e5 or arr._in_memory():
        summary.append(repr(arr.values))
    else:
        summary.append('[%s values with dtype=%s]' % (arr.size, arr.dtype))
    if hasattr(arr, 'dataset'):
        if arr.coords:
            summary.append('Coordinates:')
            summary.append(wrap_indent(repr(arr.coords), '    '))
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


def dataset_repr(ds):
    summary = ['<xray.%s>' % type(ds).__name__]

    max_name_length = max(len(k) for k in ds.variables) if ds else 0
    first_col_width = max(4 + max_name_length, 16)
    coords_str = pretty_print('Dimensions:', first_col_width)
    all_dim_strings = ['%s: %s' % (k, v) for k, v in iteritems(ds.dims)]
    summary.append('%s(%s)' % (coords_str, ', '.join(all_dim_strings)))

    def summarize_var(k, not_found=' ', found=int):
        v = ds.variables[k]
        dim_strs = []
        for n, d in enumerate(ds.dims):
            length = len(all_dim_strings[n])
            prepend = ' ' * (length // 2)
            if d in v.dims:
                if found is int:
                    indicator = str(v.dims.index(d))
                else:
                    indicator = found
            else:
                indicator = not_found
            dim_strs.append(pretty_print(prepend + indicator, length))
        string = pretty_print('    ' + k, first_col_width) + ' '
        string += '  '.join(dim_strs)
        return string

    def summarize_variables(variables, not_found=' ', found=int):
        if variables:
            return [summarize_var(k, not_found, found) for k in variables]
        else:
            return ['    None']

    summary.append('Coordinates:')
    summary.extend(summarize_variables(ds.coords, ' ', 'X'))

    summary.append('Noncoordinates:')
    summary.extend(summarize_variables(ds.noncoords, ' ', int))

    summary.append('Attributes:\n%s' % _summarize_attributes(ds))

    return '\n'.join(summary)
