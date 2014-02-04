import copy
import functools
import warnings
from collections import OrderedDict

import numpy as np

import conventions
import data
import dataview
import ops
import utils
from common import _DataWrapperMixin


def _as_compatible_data(data):
    """If data does not have the necessary attributes to be the private _data
    attribute, convert it to a np.ndarray and raise an warning
    """
    # don't check for __len__ or __iter__ so as not to warn if data is a numpy
    # numeric type like np.float32
    required = ['dtype', 'shape', 'size', 'ndim']
    if not all(hasattr(data, attr) for attr in required):
        warnings.warn('converting data to np.ndarray because %s lacks some of '
                      'the necesssary attributes for lazy use'
                      % type(data).__name__, RuntimeWarning, stacklevel=3)
        data = np.asarray(data)
    return data


class Variable(_DataWrapperMixin):
    """
    A netcdf-like variable consisting of dimensions, data and attributes
    which describe a single varRiable.  A single variable object is not
    fully described outside the context of its parent Dataset.
    """
    def __init__(self, dims, data, attributes=None):
        data = _as_compatible_data(data)
        if len(dims) != data.ndim:
            raise ValueError('data must have same shape as the number of '
                             'dimensions')
        self._dimensions = tuple(dims)
        self._data = data
        if attributes is None:
            attributes = {}
        self._attributes = OrderedDict(attributes)

    @property
    def dimensions(self):
        return self._dimensions

    def _remap_indexer(self, key):
        """Converts an orthogonal indexer into a fully expanded key (of the
        same length as dimensions) suitable for indexing `_data`

        See Also
        --------
        utils.expanded_indexer
        utils.orthogonal_indexer
        """
        key = utils.expanded_indexer(key, self.ndim)
        if any(not isinstance(k, (int, slice)) for k in key):
            # key would trigger fancy indexing
            key = utils.orthogonal_indexer(key, self.shape)
        return key

    def __getitem__(self, key):
        """Return a new Variable object whose contents are consistent with
        getting the provided key from the underlying data

        NB. __getitem__ and __setitem__ implement "orthogonal indexing" like
        netCDF4-python, where the key can only include integers, slices
        (including `Ellipsis`) and 1d arrays, each of which are applied
        orthogonally along their respective dimensions.

        The difference not matter in most cases unless you are using numpy's
        "fancy indexing," which can otherwise result in data arrays
        with shapes is inconsistent (or just uninterpretable with) with the
        variable's dimensions.

        If you really want to do indexing like `x[x > 0]`, manipulate the numpy
        array `x.data` directly.
        """
        key = self._remap_indexer(key)
        dimensions = [dim for k, dim in zip(key, self.dimensions)
                      if not isinstance(k, int)]
        new_data = self._data[key]
        # orthogonal indexing should ensure the dimensionality is consistent
        assert new_data.ndim == len(dimensions)
        # always return a Variable, because Variable subtypes may have
        # different constructors and may not make sense without an attached
        # datastore
        return Variable(dimensions, new_data, self.attributes)

    def __setitem__(self, key, value):
        """__setitem__ is overloaded to access the underlying numpy data with
        orthogonal indexing (see __getitem__ for more details)
        """
        self.data[self._remap_indexer(key)] = value

    def __iter__(self):
        for n in range(len(self)):
            yield self[n]

    @property
    def attributes(self):
        return self._attributes

    def copy(self):
        """
        Returns a shallow copy of the current object.
        """
        return self.__copy__()

    def _copy(self, deepcopy=False):
        # deepcopies should always be of a numpy view of the data, not the data
        # itself, because non-memory backends don't necessarily have deepcopy
        # defined sensibly (this is a problem for netCDF4 variables)
        data = copy.deepcopy(self.data) if deepcopy else self._data
        # note:
        # dimensions is already an immutable tuple
        # attributes will be copied when the new Variable is created
        return Variable(self.dimensions, data, self.attributes)

    def __copy__(self):
        """
        Returns a shallow copy of the current object.
        """
        return self._copy(deepcopy=False)

    def __deepcopy__(self, memo=None):
        """
        Returns a deep copy of the current object.

        memo does nothing but is required for compatability with copy.deepcopy
        """
        return self._copy(deepcopy=True)

    # mutable objects should not be hashable
    __hash__ = None

    def __str__(self):
        """Create a ncdump-like summary of the object"""
        summary = ["dimensions:"]
        # prints dims that look like:
        #    dimension = length
        dim_print = lambda d, l : "\t%s : %s" % (conventions.pretty_print(d, 30),
                                                 conventions.pretty_print(l, 10))
        # add each dimension to the summary
        summary.extend([dim_print(d, l) for d, l in zip(self.dimensions, self.shape)])
        summary.append("dtype : %s" % (conventions.pretty_print(self.dtype, 8)))
        summary.append("attributes:")
        #    attribute:value
        summary.extend(["\t%s:%s" % (conventions.pretty_print(att, 30),
                                     conventions.pretty_print(val, 30))
                        for att, val in self.attributes.iteritems()])
        # create the actual summary
        return '\n'.join(summary).replace('\t', ' ' * 4)

    def __repr__(self):
        if self.ndim > 0:
            dim_summary = ', '.join('%s: %s' % (k, v) for k, v
                                    in zip(self.dimensions, self.shape))
            contents = ' (%s): %s' % (dim_summary, self.dtype)
        else:
            contents = ': %s' % self.data
        return '<scidata.%s%s>' % (type(self).__name__, contents)

    def views(self, slicers):
        """Return a new Variable object whose contents are a view of the object
        sliced along a specified dimension.

        Parameters
        ----------
        slicers : {dim: slice, ...}
            A dictionary mapping from dim to slice, dim represents
            the dimension to slice along slice represents the range of the
            values to extract.

        Returns
        -------
        obj : Variable object
            The returned object has the same attributes and dimensions
            as the original. Data contents are taken along the
            specified dimension.  Care must be taken since modifying (most)
            values in the returned object will result in modification to the
            parent object.

        See Also
        --------
        view
        take
        """
        slices = [slice(None)] * self.data.ndim
        for i, dim in enumerate(self.dimensions):
            if dim in slicers:
                slices[i] = slicers[dim]
        return self[tuple(slices)]

    def view(self, s, dim):
        """Return a new Variable object whose contents are a view of the object
        sliced along a specified dimension.

        Parameters
        ----------
        s : slice
            The slice representing the range of the values to extract.
        dim : string
            The dimension to slice along.

        Returns
        -------
        obj : Variable object
            The returned object has the same attributes and dimensions
            as the original. Data contents are taken along the
            specified dimension.  Care must be taken since modifying (most)
            values in the returned object will result in modification to the
            parent object.

        See Also
        --------
        take
        """
        return self.views({dim: s})

    def transpose(self, *dimensions):
        """Return a new Variable object with transposed dimensions

        Note: Although this operation returns a view of this variable's data,
        it is not lazy -- the data will be fully loaded.

        Parameters
        ----------
        *dimensions : str, optional
            By default, reverse the dimensions. Otherwise, reorder the
            dimensions to this order.

        Returns
        -------
        transposed : Variable
            The returned object has transposed data and dimensions with the
            same attributes as the original.

        See Also
        --------
        numpy.transpose
        """
        if len(dimensions) == 0:
            dimensions = self.dimensions[::-1]
        axes = [dimensions.index(dim) for dim in self.dimensions]
        data = self.data.transpose(*axes)
        return Variable(dimensions, data, self.attributes)

    def collapsed(self, f, dimension=None, axis=None, **kwargs):
        """Collapse this variable by applying `f` along some dimension(s)

        Parameters
        ----------
        f : function
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of collapsing an
            np.ndarray over an integer valued axis.
        dimension : str or sequence of str, optional
            Dimension(s) over which to repeatedly apply `f`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `f`. Only one of the
            'dimension' and 'axis' arguments can be supplied. If neither are
            supplied, then the collapse is calculated over the flattened array
            (by calling `f(x)` without an axis argument).
        **kwargs : dict
            Additional keyword arguments passed on to `f`.

        Note
        ----
        If `collapsed` is called with multiple dimensions (or axes, which
        are converted into dimensions), then the collapse operation is
        performed repeatedly along each dimension in turn from left to right.

        Returns
        -------
        collapsed : Variable
            Variable with summarized data and the indicated dimension(s)
            removed.
        """
        if dimension is not None and axis is not None:
            raise ValueError("cannot supply both 'axis' and 'dimension' "
                             "arguments")

        if axis is not None:
            # determine dimensions
            if isinstance(axis, int):
                axis = [axis]
            dimension = [self.dimensions[i] for i in axis]

        if dimension is not None:
            if isinstance(dimension, basestring):
                dimension = [dimension]
            var = self
            for dim in dimension:
                var = var._collapsed(f, dim, **kwargs)
        else:
            attr = self._attributes_with_added_cell_method(
                ': '.join(self.dimensions) + ': ' + f.__name__)
            var = Variable([], f(self.data, **kwargs), attr)
        return var

    def _attributes_with_added_cell_method(self, string):
        attr = OrderedDict(self.attributes)
        if 'cell_methods' in attr:
            base = attr['cell_methods'] + ' '
        else:
            base = ''
        attr['cell_methods'] = base + string
        return attr

    def _collapsed(self, f, dim, **kwargs):
        """Collapse a single dimension"""
        axis = self.dimensions.index(dim)
        dims = tuple(dim for i, dim in enumerate(self.dimensions)
                     if axis not in [i, i - self.ndim])
        data = f(self.data, axis=axis, **kwargs)
        attr = self._attributes_with_added_cell_method(
            self.dimensions[axis] + ': ' + f.__name__)
        return Variable(dims, data, attr)

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self):
            return Variable(self.dimensions, f(self.data), self.attributes)
        return func

    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, dataview.DataView):
                return NotImplemented
            self_data, other_data, new_dims = _broadcast_var_data(self, other)
            new_data = (f(self_data, other_data)
                        if not reflexive
                        else f(other_data, self_data))
            new_attr = utils.safe_merge(_math_safe_attributes(self),
                                        _math_safe_attributes(other))
            return Variable(new_dims, new_data, new_attr)
        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            self_data, other_data, dims = _broadcast_var_data(self, other)
            if dims != self.dimensions:
                raise ValueError('dimensions cannot change for in-place '
                                 'operations')
            self.data = f(self_data, other_data)
            utils.safe_update(self.attributes, _math_safe_attributes(other))
            return self
        return func

    # @staticmethod
    # def collapse_method(f):
    #     @functools.wraps(f)
    #     def func(self, dimension=None, axis=None):
    #         return self.collapsed(f, dimension, axis)
    #     return func

ops.inject_special_operations(Variable)


def _broadcast_var_data(self, other):
    self_data = self.data
    if isinstance(other, data.Dataset):
        raise TypeError('datasets do not support mathematical operations')
    elif all(hasattr(other, attr) for attr in ['dimensions', 'data', 'shape']):
        # validate dimensions
        dim_lengths = dict(zip(self.dimensions, self.shape))
        for k, v in zip(other.dimensions, other.shape):
            if k in dim_lengths and dim_lengths[k] != v:
                raise ValueError('operands could not be broadcast together '
                                 'with mismatched lengths for dimension %r: %s'
                                 % (k, (dim_lengths[k], v)))
        for dimensions in [self.dimensions, other.dimensions]:
            if len(set(dimensions)) < len(dimensions):
                raise ValueError('broadcasting requires that neither operand '
                                 'has duplicate dimensions: %r'
                                 % list(dimensions))

        # build dimensions for new Variable
        other_only_dims = [dim for dim in other.dimensions
                           if dim not in self.dimensions]
        dimensions = list(self.dimensions) + other_only_dims

        # expand self_data's dimensions so it's broadcast compatible after
        # adding other's dimensions to the end
        for _ in xrange(len(other_only_dims)):
            self_data = np.expand_dims(self_data, axis=-1)

        # expand and reorder other_data so the dimensions line up
        self_only_dims = [dim for dim in dimensions
                          if dim not in other.dimensions]
        other_data = other.data
        for _ in xrange(len(self_only_dims)):
            other_data = np.expand_dims(other_data, axis=-1)
        other_dims = list(other.dimensions) + self_only_dims
        axes = [other_dims.index(dim) for dim in dimensions]
        other_data = other_data.transpose(axes)
    else:
        # rely on numpy broadcasting rules
        other_data = other
        dimensions = self.dimensions
    return self_data, other_data, dimensions


def _math_safe_attributes(v):
    """Given a variable, return the variables's attributes that are safe for
    mathematical operations (e.g., all those except for 'units')
    """
    try:
        attr = v.attributes
    except AttributeError:
        return {}
    else:
        return OrderedDict((k, v) for k, v in attr.items() if k != 'units')
