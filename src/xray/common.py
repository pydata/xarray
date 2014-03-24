class ImplementsReduce(object):
    @classmethod
    def _reduce_method(cls, f, name=None, module=None):
        def func(self, dimension=cls._reduce_dimension_default,
                 axis=cls._reduce_axis_default, **kwargs):
            return self.reduce(f, dimension, axis, **kwargs)
        if name is None:
            name = f.__name__
        func.__name__ = name
        func.__doc__ = cls._reduce_method_docstring.format(
            name=('' if module is None else module + '.') + name,
            cls=cls.__name__)
        return func


class AbstractArray(ImplementsReduce):
    def __nonzero__(self):
        return bool(self.data)

    def __float__(self):
        return float(self.data)

    def __int__(self):
        return int(self.data)

    def __complex__(self):
        return complex(self.data)

    def __long__(self):
        return long(self.data)

    def __array__(self, dtype=None):
        return self.data

    def __repr__(self):
        return array_repr(self)

    @property
    def T(self):
        return self.transpose()

    _reduce_method_docstring = \
        """Reduce this {cls}'s data' by applying `{name}` along some
        dimension(s).

        Parameters
        ----------
        dimension : str or sequence of str, optional
            Dimension(s) over which to repeatedly apply `{name}`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `{name}`. Only one of the
            'dimension' and 'axis' arguments can be supplied. If neither are
            supplied, then `{name}` is calculated over the flattened array
            (by calling `{name}(x)` without an axis argument).
        **kwargs : dict
            Additional keyword arguments passed on to `{name}`.

        Notes
        -----
        If this method is called with multiple dimensions (or axes, which are
        converted into dimensions), then `{name}` is performed repeatedly along
        each dimension in turn from left to right.

        Returns
        -------
        reduced : {cls}
            New {cls} object with `{name}` applied to its data and the
            indicated dimension(s) removed.
        """

    _reduce_dimension_default = None
    _reduce_axis_default = None


def _summarize_attributes(data):
    if data.attributes:
        attr_summary = '\n'.join('    %s: %s' % (k, v) for k, v
                                 in data.attributes.iteritems())
    else:
        attr_summary = '    Empty'
    return attr_summary


def array_repr(arr):
    focus_str = ('%r ' % arr.focus) if hasattr(arr, 'focus') else ''
    dim_summary = ', '.join('%s: %s' % (k, v) for k, v
                            in zip(arr.dimensions, arr.shape))
    summary = ['<xray.%s %s(%s)>'% (type(arr).__name__, focus_str,
                                    dim_summary)]
    if arr.size < 1e5 or arr.in_memory():
        summary.append(repr(arr.data))
    else:
        summary.append('[%s values with dtype=%s]' % (arr.size, arr.dtype))
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

    first_col_width = max(4 + max(len(k) for k in ds.variables), 17)
    coords_str = pretty_print('Coordinates:', first_col_width)
    all_dim_strings = ['%s: %s' % (k, v) for k, v in ds.dimensions.iteritems()]
    summary.append('%s(%s)' % (coords_str, ', '.join(all_dim_strings)))

    def summarize_var(k):
        v = ds.variables[k]
        dim_strs = []
        for n, d in enumerate(ds.dimensions):
            length = len(all_dim_strings[n])
            prepend = ' ' * (length // 2)
            indicator = 'X' if d in v.dimensions else '-'
            dim_strs.append(pretty_print(prepend + indicator, length))
        string = pretty_print('    ' + k, first_col_width) + ' '
        string += '  '.join(dim_strs)
        return string

    summary.append('Non-coordinates:')
    if ds.noncoordinates:
        summary.extend(summarize_var(k) for k in ds.noncoordinates)
    else:
        summary.append('    None')
    summary.append('Attributes:\n%s' % _summarize_attributes(ds))
    return '\n'.join(summary)
