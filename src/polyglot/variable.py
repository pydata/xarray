import copy
import numpy as np

from collections import OrderedDict

import conventions

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


class Variable(object):
    """
    A netcdf-like variable consisting of dimensions, data and attributes
    which describe a single varRiable.  A single variable object is not
    fully described outside the context of its parent Dataset.
    """
    def __init__(self, dims, data, attributes=None):
        object.__setattr__(self, 'dimensions', dims)
        object.__setattr__(self, 'data', data)
        if attributes is None:
            attributes = {}
        object.__setattr__(self, 'attributes', AttributesDict(attributes))

    def _allocate(self):
        return self.__class__(dims=(), data=0)

    def __getattribute__(self, key):
        """
        Here we give some of the attributes of self.data preference over
        attributes in the object instelf.
        """
        if key in ['dtype', 'shape', 'size', 'ndim', 'nbytes',
                'flat', '__iter__', 'view']:
            return getattr(self.data, key)
        else:
            return object.__getattribute__(self, key)

    def __setattr__(self, attr, value):
        """"__setattr__ is overloaded to prevent operations that could
        cause loss of data consistency. If you really intend to update
        dir(self), use the self.__dict__.update method or the
        super(type(a), self).__setattr__ method to bypass."""
        raise AttributeError, "Object is tamper-proof"

    def __delattr__(self, attr):
        raise AttributeError, "Object is tamper-proof"

    def __getitem__(self, index):
        """__getitem__ is overloaded to access the underlying numpy data"""
        return self.data[index]

    def __setitem__(self, index, data):
        """__setitem__ is overloaded to access the underlying numpy data"""
        self.data[index] = data

    def __hash__(self):
        """__hash__ is overloaded to guarantee that two variables with the same
        attributes and np.data values have the same hash (the converse is not true)"""
        return hash((self.dimensions,
                     frozenset((k,v.tostring()) if isinstance(v,np.ndarray) else (k,v)
                               for (k,v) in self.attributes.items()),
                     self.data.tostring()))

    def __len__(self):
        """__len__ is overloaded to access the underlying numpy data"""
        return self.data.__len__()

    def __copy__(self):
        """
        Returns a shallow copy of the current object.
        """
        # Create the simplest possible dummy object and then overwrite it
        obj = self._allocate()
        object.__setattr__(obj, 'dimensions', self.dimensions)
        object.__setattr__(obj, 'data', self.data)
        object.__setattr__(obj, 'attributes', self.attributes)
        return obj

    def __deepcopy__(self, memo=None):
        """
        Returns a deep copy of the current object.

        memo does nothing but is required for compatability with copy.deepcopy
        """
        # Create the simplest possible dummy object and then overwrite it
        obj = self._allocate()
        # tuples are immutable
        object.__setattr__(obj, 'dimensions', self.dimensions)
        object.__setattr__(obj, 'data', self.data[:].copy())
        object.__setattr__(obj, 'attributes', self.attributes.copy())
        return obj

    def __eq__(self, other):
        if self.dimensions != other.dimensions or \
           (self.data.tostring() != other.data.tostring()):
            return False
        if not self.attributes == other.attributes:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        """Create a ncdump-like summary of the object"""
        summary = ["dimensions:"]
        # prints dims that look like:
        #    dimension = length
        dim_print = lambda d, l : "\t%s : %s" % (conventions.pretty_print(d, 30),
                                                 conventions.pretty_print(l, 10))
        # add each dimension to the summary
        summary.extend([dim_print(d, l) for d, l in zip(self.dimensions, self.shape)])
        summary.append("\ndtype : %s" % (conventions.pretty_print(self.dtype, 8)))
        summary.append("\nattributes:")
        #    attribute:value
        summary.extend(["\t%s:%s" % (conventions.pretty_print(att, 30),
                                     conventions.pretty_print(val, 30))
                        for att, val in self.attributes.iteritems()])
        # create the actual summary
        return '\n'.join(summary)

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
        # Shallow copy
        obj = copy.copy(self)
        object.__setattr__(obj, 'data', self.data[slices])
        return obj

    def view(self, s, dim):
        """Return a new Variable object whose contents are a view of the object
        sliced along a specified dimension.

        Parameters
        ----------
        s : slice
            The slice representing the range of the values to extract.
        dim : string
            The dimension to slice along. If multiple dimensions equal
            dim (e.g. a correlation matrix), then the slicing is done
            only along the first matching dimension.

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
        return self.views({dim : s})

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
        axis = list(self.dimensions).index(dim)
        # Deep copy
        obj = copy.deepcopy(self)
        # In case data is lazy we need to slice out all the data before taking.
        object.__setattr__(obj, 'data', self.data[:].take(indices, axis=axis))
        return obj

class LazyVariableData(object):
    """
    This object wraps around a Variable object (though
    it only really makes sense to use it with a class that
    extends variable.Variable).  The result mascarades as
    variable data, but doesn't actually try accessing the
    data until indexing is attempted.

    For example, imagine you have some variable that was
    derived from an opendap dataset, 'nc'.

        var = nc['massive_variable']

    if you wanted to check the data type of var:

        var.data.dtype

    you would find that it might involve downloading all
    of the actual data, then inspecting the resulting
    numpy array.  But with this wrapper calling:

        nc['large_variable'].data.someattribute

    will first inspect the Variable object to see if it has
    the desired attribute and only then will it suck down the
    actual numpy array and request 'someattribute'.
    """
    def __init__(self, lazy_variable):
        self.lazyvar = lazy_variable

    def __eq__(self, other):
        return self.lazyvar[:] == other

    def __ne__(self, other):
        return self.lazyvar[:] != other

    def __getitem__(self, key):
        return self.lazyvar[key]

    def __setitem__(self, key, value):
        if not isinstance(self.lazyvar, Variable):
            self.lazyvar = Variable(self.lazyvar.dimensions,
                                        data = self.lazyvar[:],
                                        dtype = self.lazyvar.dtype,
                                        shape = self.lazyvar.shape,
                                        attributes = self.lazyvar.attributes)
        self.lazyvar.__setitem__(key, value)

    def __getattr__(self, attr):
        """__getattr__ is overloaded to selectively expose some of the
        attributes of the underlying lazy variable"""
        if hasattr(self.lazyvar, attr):
            return getattr(self.lazyvar, attr)
        else:
            return getattr(self.lazyvar[:], attr)