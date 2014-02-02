import copy
import numpy as np

import warnings
from collections import OrderedDict

import conventions
import data
import dataview
from common import _DataWrapperMixin
from utils import expanded_indexer, safe_merge, safe_update
from ops import inject_special_operations


class AttributesDict(OrderedDict):
    """A subclass of OrderedDict whose __setitem__ method automatically
    checks and converts values to be valid netCDF attributes
    """
    def __init__(self, *args, **kwds):
        OrderedDict.__init__(self, *args, **kwds)

    def __setitem__(self, key, value):
        if not conventions.is_valid_name(key):
            raise ValueError("Not a valid attribute name")
        # Strings get special handling because netCDF treats them as
        # character arrays. Everything else gets coerced to a numpy
        # vector. netCDF treats scalars as 1-element vectors. Arrays of
        # non-numeric type are not allowed.
        if isinstance(value, basestring):
            # netcdf attributes should be unicode
            value = unicode(value)
        else:
            try:
                value = conventions.coerce_type(np.atleast_1d(np.asarray(value)))
            except:
                raise ValueError("Not a valid value for a netCDF attribute")
            if value.ndim > 1:
                raise ValueError("netCDF attributes must be vectors " +
                        "(1-dimensional)")
            value = conventions.coerce_type(value)
            if str(value.dtype) not in conventions.TYPEMAP:
                # A plain string attribute is okay, but an array of
                # string objects is not okay!
                raise ValueError("Can not convert to a valid netCDF type")
        OrderedDict.__setitem__(self, key, value)

    def copy(self):
        """The copy method of the superclass simply calls the constructor,
        which in turn calls the update method, which in turns calls
        __setitem__. This subclass implementation bypasses the expensive
        validation in __setitem__ for a substantial speedup."""
        obj = self.__class__()
        for (attr, value) in self.iteritems():
            OrderedDict.__setitem__(obj, attr, copy.copy(value))
        return obj

    def __deepcopy__(self, memo=None):
        """
        Returns a deep copy of the current object.

        memo does nothing but is required for compatability with copy.deepcopy
        """
        return self.copy()

    def update(self, *other, **kwargs):
        """Set multiple attributes with a mapping object or an iterable of
        key/value pairs"""
        # Capture arguments in an OrderedDict
        args_dict = OrderedDict(*other, **kwargs)
        try:
            # Attempt __setitem__
            for (attr, value) in args_dict.iteritems():
                self.__setitem__(attr, value)
        except:
            # A plain string attribute is okay, but an array of
            # string objects is not okay!
            raise ValueError("Can not convert to a valid netCDF type")
            # Clean up so that we don't end up in a partial state
            for (attr, value) in args_dict.iteritems():
                if self.__contains__(attr):
                    self.__delitem__(attr)
            # Re-raise
            raise

    def __eq__(self, other):
        if not set(self.keys()) == set(other.keys()):
            return False
        for (key, value) in self.iteritems():
            if value.__class__ != other[key].__class__:
                return False
            if isinstance(value, basestring):
                if value != other[key]:
                    return False
            else:
                if value.tostring() != other[key].tostring():
                    return False
        return True


def _as_compatible_data(data):
    """If data does not have the necessary attributes to be the private _data
    attribute, convert it to a np.ndarray and raise an warning
    """
    # don't check for __len__ or __iter__ so as not to warn if data is a numpy
    # numeric type like np.float32
    required = ['dtype', 'shape', 'size', 'ndim']
    if not all(hasattr(data, attr) for attr in required):
        warnings.warn('converting data to np.ndarray because it lacks some of '
                      'the necesssary attributes for lazy use', RuntimeWarning,
                      stacklevel=3)
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
        self._attributes = AttributesDict(attributes)

    @property
    def dimensions(self):
        return self._dimensions

    def __getitem__(self, key):
        """
        Return a new Variable object whose contents are consistent with getting
        the provided key from the underlying data
        """
        key = expanded_indexer(key, self.ndim)
        dimensions = [dim for k, dim in zip(key, self.dimensions)
                      if not isinstance(k, int)]
        #TODO: wrap _data in a biggus array or use np.ix_ so fancy indexing
        # always slices axes independently (as in the python-netcdf4 package)
        new_data = self._data[key]
        if new_data.ndim != len(dimensions):
            raise ValueError('indexing results in an array of shape %s, '
                             'which has inconsistent length with the '
                             'expected dimensions %s (if you really want to '
                             'do this sort of indexing, index the `data` '
                             'attribute directly)'
                             % (new_data.shape, dimensions))
        # always return a Variable, because Variable subtypes may have
        # different constructors and may not make sense without an attached
        # datastore
        return Variable(dimensions, new_data, self.attributes)

    def __setitem__(self, key, value):
        """__setitem__ is overloaded to access the underlying numpy data"""
        self.data[key] = value

    def __iter__(self):
        """
        Iterate over the contents of this Variable
        """
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

    def take(self, indices, dim):
        """Return a new Variable object whose contents are sliced from
        the current object along a specified dimension

        Parameters
        ----------
        indices : array_like
            The indices of the values to extract. indices must be compatible
            with the ndarray.take() method.
        dim : string
            The dimension to slice along. If multiple dimensions equal
            dim (e.g. a correlation matrix), then the slicing is done
            only along the first matching dimension.

        Returns
        -------
        obj : Variable object
            The returned object has the same attributes and dimensions
            as the original. Data contents are taken along the
            specified dimension.

        See Also
        --------
        numpy.take
        """
        indices = np.asarray(indices)
        if indices.ndim != 1:
            raise ValueError('indices should have a single dimension')
        # When dim appears repeatedly in self.dimensions, using the index()
        # method gives us only the first one, which is the desired behavior
        axis = self.dimensions.index(dim)
        # take only works on actual numpy arrays
        data = self.data.take(indices, axis=axis)
        return Variable(self.dimensions, data, self.attributes)

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
        obj : Variable object
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


def broadcast_var_data(self, other):
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


def unary_op(f):
    def func(self):
        return Variable(self.dimensions, f(self.data), self.attributes)
    return func


def binary_op(f, reflexive=False):
    def func(self, other):
        if isinstance(other, dataview.DataView):
            return NotImplemented
        self_data, other_data, new_dims = broadcast_var_data(self, other)
        new_data = (f(self_data, other_data)
                    if not reflexive
                    else f(other_data, self_data))
        new_attr = safe_merge(_math_safe_attributes(self),
                              _math_safe_attributes(other))
        return Variable(new_dims, new_data, new_attr)
    return func


def inplace_binary_op(f):
    def func(self, other):
        self_data, other_data, dimensions = broadcast_var_data(self, other)
        if dimensions != self.dimensions:
            raise ValueError('dimensions cannot change for in-place operations')
        self.data = f(self_data, other_data)
        safe_update(self.attributes, _math_safe_attributes(other))
        return self
    return func


inject_special_operations(Variable, unary_op, binary_op, inplace_binary_op)
