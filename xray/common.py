from collections import Mapping

import numpy as np

from .pycompat import basestring, iteritems
from . import utils


class ImplementsArrayReduce(object):
    @classmethod
    def _reduce_method(cls, func):
        def wrapped_func(self, dim=None, axis=None, keep_attrs=False,
                         **kwargs):
            return self.reduce(func, dim, axis, keep_attrs, **kwargs)
        return wrapped_func

    _reduce_extra_args_docstring = \
        """dim : str or sequence of str, optional
            Dimension(s) over which to apply `{name}`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `{name}`. Only one of the 'dim'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `{name}` is calculated over axes.\n"""


class ImplementsDatasetReduce(object):
    @classmethod
    def _reduce_method(cls, func):
        def wrapped_func(self, dim=None, keep_attrs=False, **kwargs):
            return self.reduce(func, dim, keep_attrs, **kwargs)
        return wrapped_func

    _reduce_extra_args_docstring = \
        """dim : str or sequence of str, optional
            Dimension(s) over which to apply `func`.  By default `func` is
            applied over all dimensions.\n"""


class AbstractArray(ImplementsArrayReduce):
    def __nonzero__(self):
        return bool(self.values)

    # Python 3 uses __bool__, Python 2 uses __nonzero__
    __bool__ = __nonzero__

    def __float__(self):
        return float(self.values)

    def __int__(self):
        return int(self.values)

    def __complex__(self):
        return complex(self.values)

    def __long__(self):
        return long(self.values)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def __repr__(self):
        return array_repr(self)

    def _iter(self):
        for n in range(len(self)):
            yield self[n]

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError('iteration over a 0-d array')
        return self._iter()

    @property
    def T(self):
        return self.transpose()

    def get_axis_num(self, dim):
        """Return axis number(s) corresponding to dimension(s) in this array.

        Parameters
        ----------
        dim : str or iterable of str
            Dimension name(s) for which to lookup axes.

        Returns
        -------
        int or tuple of int
            Axis number or numbers corresponding to the given dimensions.
        """
        if isinstance(dim, basestring):
            return self._get_axis_num(dim)
        else:
            return tuple(self._get_axis_num(d) for d in dim)

    def _get_axis_num(self, dim):
        try:
            return self.dims.index(dim)
        except ValueError:
            raise ValueError("%r not found in array dimensions %r" %
                             (dim, self.dims))


class AbstractCoordinates(Mapping):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        return iter(self._data.dims)

    def __len__(self):
        return len(self._data.dims)

    def __contains__(self, key):
        return key in self._data.dims

    def __repr__(self):
        return '\n'.join(_wrap_indent(repr(v.to_index()), '%s: ' % k)
                         for k, v in self.items())

    @staticmethod
    def _convert_to_coord(key, value, expected_size=None):
        from .variable import Coordinate, as_variable

        if not isinstance(value, AbstractArray):
            value = Coordinate(key, value)
        coord = as_variable(value).to_coord()

        if expected_size is not None and coord.size != expected_size:
            raise ValueError('new coordinate has size %s but the existing '
                             'coordinate has size %s'
                             % (coord.size, expected_size))
        return coord


def _summarize_attributes(data):
    if data.attrs:
        attr_summary = '\n'.join('    %s: %s' % (k, v) for k, v
                                 in iteritems(data.attrs))
    else:
        attr_summary = '    Empty'
    return attr_summary


def _wrap_indent(text, start='', length=None):
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
            summary.append(_wrap_indent(repr(arr.coords), '    '))
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
