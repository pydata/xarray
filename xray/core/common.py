import functools

import numpy as np
import pandas as pd

from .pycompat import basestring, iteritems
from . import formatting


class ImplementsArrayReduce(object):
    @classmethod
    def _reduce_method(cls, func, include_skipna, numeric_only):
        if include_skipna:
            def wrapped_func(self, dim=None, axis=None, skipna=None,
                             keep_attrs=False, **kwargs):
                return self.reduce(func, dim, axis, keep_attrs, skipna=skipna,
                                   **kwargs)
        else:
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
            `{name}` is calculated over axes."""


class ImplementsDatasetReduce(object):
    @classmethod
    def _reduce_method(cls, func, include_skipna, numeric_only):
        if include_skipna:
            def wrapped_func(self, dim=None, keep_attrs=False, skipna=None,
                             **kwargs):
                return self.reduce(func, dim, keep_attrs, skipna=skipna,
                                   numeric_only=numeric_only, **kwargs)
        else:
            def wrapped_func(self, dim=None, keep_attrs=False, **kwargs):
                return self.reduce(func, dim, keep_attrs,
                                   numeric_only=numeric_only, **kwargs)
        return wrapped_func

    _reduce_extra_args_docstring = \
        """dim : str or sequence of str, optional
            Dimension(s) over which to apply `func`.  By default `func` is
            applied over all dimensions."""


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
        return formatting.array_repr(self)

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


class AttrAccessMixin(object):
    """Mixin class that allow getting keys with attribute access
    """
    @property
    def __attr_sources__(self):
        """List of places to look-up items for attribute-style access"""
        return [self, self.attrs]

    def __getattr__(self, name):
        for source in self.__attr_sources__:
            try:
                return source[name]
            except KeyError:
                pass
        raise AttributeError("%r object has no attribute %r" %
                             (type(self).__name__, name))

    def __dir__(self):
        """Provide method name lookup and completion. Only provide 'public'
        methods.
        """
        extra_attrs = [item for sublist in self.__attr_sources__
                       for item in sublist]
        return sorted(set(dir(type(self)) + extra_attrs))


def squeeze(xray_obj, dims, dim=None):
    """Squeeze the dims of an xray object."""
    if dim is None:
        dim = [d for d, s in iteritems(dims) if s == 1]
    else:
        if isinstance(dim, basestring):
            dim = [dim]
        if any(dims[k] > 1 for k in dim):
            raise ValueError('cannot select a dimension to squeeze out '
                             'which has length greater than one')
    return xray_obj.isel(**dict((d, 0) for d in dim))


def _maybe_promote(dtype):
    """Simpler equivalent of pandas.core.common._maybe_promote"""
    # N.B. these casting rules should match pandas
    if np.issubdtype(dtype, float):
        fill_value = np.nan
    elif np.issubdtype(dtype, int):
        # convert to floating point so NaN is valid
        dtype = float
        fill_value = np.nan
    elif np.issubdtype(dtype, np.datetime64):
        fill_value = np.datetime64('NaT')
    else:
        dtype = object
        fill_value = np.nan
    return dtype, fill_value


def _possibly_convert_objects(values):
    """Convert arrays of datetime.datetime and datetime.timedelta objects into
    datetime64 and timedelta64
    """
    try:
        converter = functools.partial(pd.core.common._possibly_convert_objects,
                                      convert_numeric=False)
    except AttributeError:
        # our fault for using a private pandas API that has gone missing
        # this should do the same coercion (though it will be slower)
        converter = lambda x: np.asarray(pd.Series(x))
    return converter(values.ravel()).reshape(values.shape)
