# TODO Use various backend data stores. pytable, ncdf4, scipy.io, iris, memory

import os
import copy
import numpy as np
import netCDF4 as nc4

from operator import or_
from scipy.io import netcdf
from cStringIO import StringIO
from collections import OrderedDict

import conventions, backends, variable

date2num = nc4.date2num
num2date = nc4.num2date

class Dataset(object):
    """
    A netcdf-like data object consisting of dimensions, variables and
    attributes which together form a self describing data set.
    """
    def __init__(self, nc = None, store = None, *args, **kwdargs):

        if store is None:
            store = backends.InMemoryDataStore()
        object.__setattr__(self, 'store', store)

        if isinstance(nc, basestring) and not nc.startswith('CDF'):
            """
            If the initialization nc is a string and it doesn't
            appear to be the contents of a netcdf file we load
            it using the netCDF4 package
            """
            self._load_netcdf4(nc, *args, **kwdargs)
        elif nc is not None:
            """
            If nc is a file-like object we read it using
            the scipy.io.netcdf package
            """
            self._load_scipy(nc)

    def _unchecked_set_dimensions(self, *args, **kwdargs):
        self.store.unchecked_set_dimensions(*args, **kwdargs)

    def _unchecked_set_attributes(self, *args, **kwdargs):
        self.store.unchecked_set_attributes(*args, **kwdargs)

    def _unchecked_set_variables(self, *args, **kwdargs):
        self.store.unchecked_set_variables(*args, **kwdargs)

    def _unchecked_create_dimension(self, *args, **kwdargs):
        self.store.unchecked_create_dimension(*args, **kwdargs)

    def _unchecked_add_variable(self, *args, **kwdargs):
        self.store.unchecked_add_variable(*args, **kwdargs)

    def _unchecked_create_variable(self, name, dims, data, attributes):
        """Creates a variable without checks"""
        v = variable.Variable(dims=dims, data=data,
                              attributes=attributes)
        self._unchecked_add_variable(name, v)
        return v

    def _unchecked_create_coordinate(self, name, data, attributes):
        """Creates a coordinate (dim and var) without checks"""
        self._unchecked_create_dimension(name, data.size)
        return self._unchecked_create_variable(name, (name,), data, attributes)

    def sync(self):
        return self.store.sync()

    @property
    def variables(self):
        return self.store.variables

    @property
    def attributes(self):
        return self.store.attributes

    @property
    def dimensions(self):
        return self.store.dimensions

    def _allocate(self):
        return self.__class__()

    def __copy__(self):
        """
        Returns a shallow copy of the current object.
        """
        obj = self._allocate()
        self.translate(obj)
        return obj

    def __deepcopy__(self, memo=None):
        """
        Returns a deep copy of the current object.

        memo does nothing but is required for compatability with copy.deepcopy
        """
        # Create the simplest possible dummy object and then overwrite it
        obj = self._allocate()
        obj._unchecked_set_dimensions(copy.deepcopy(self.dimensions))
        for vn, v in self.variables.iteritems():
            obj._unchecked_add_variable(vn, copy.deepcopy(v))
        obj._unchecked_set_attributes(copy.deepcopy(self.attributes))
        return obj

    def _load_scipy(self, scipy_nc, *args, **kwdargs):
        """
        Interprets a netcdf file-like object using scipy.io.netcdf.
        The contents of the netcdf object are loaded into memory.
        """
        try:
            nc = netcdf.netcdf_file(scipy_nc, mode='r', *args, **kwdargs)
        except:
            scipy_nc = StringIO(scipy_nc)
            scipy_nc.seek(0)
            nc = netcdf.netcdf_file(scipy_nc, mode='r', *args, **kwdargs)

        self.attributes.update(nc._attributes)
        for k, d in nc.dimensions.iteritems():
            self._unchecked_create_dimension(k, d)
        for vn, sci_var in nc.variables.iteritems():
            self._unchecked_create_variable(vn,
                                           dims = sci_var.dimensions,
                                           data = sci_var.data,
                                           attributes = sci_var._attributes)

    def _load_netcdf4(self, netcdf_path, *args, **kwdargs):
        """
        Interprets the contents of netcdf_path using the netCDF4
        package.
        """
        nc = nc4.Dataset(netcdf_path, *args, **kwdargs)

        self.attributes.update(dict((k.encode(), nc.getncattr(k)) for k in nc.ncattrs()))

        for k, d in nc.dimensions.iteritems():
            self._unchecked_create_dimension(k.encode(), len(d))

        for vn, v in nc.variables.iteritems():
            attributes = dict((k, v.getncattr(k)) for k in v.ncattrs())
            self._unchecked_create_variable(vn,
                            dims = tuple(v.dimensions),
                            # TODO : this variable copy is lazy and
                            # might cause issues in the future.
                            data = v,
                            attributes = attributes)

    def __setattr__(self, attr, value):
        """"__setattr__ is overloaded to prevent operations that could
        cause loss of data consistency. If you really intend to update
        dir(self), use the self.__dict__.update method or the
        super(type(a), self).__setattr__ method to bypass."""
        raise AttributeError("__setattr__ is disabled")

    def __contains__(self, key):
        """
        The 'in' operator will return true or false depending on
        whether 'key' is a varibale in the data object or not.
        """
        return key in self.variables

    def __eq__(self, other):
        if not isinstance(other, Dataset):
            return False
        if dict(self.dimensions) != dict(other.dimensions):
            return False
        if not dict(self.variables) == dict(other.variables):
            return False
        if not self.attributes == other.attributes:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def coordinates(self):
        # A coordinate variable is a 1-dimensional variable with the
        # same name as its dimension
        return OrderedDict([(dim, length)
                for (dim, length) in self.dimensions.iteritems()
                if (dim in self.variables) and
                (self.variables[dim].data.ndim == 1) and
                (self.variables[dim].dimensions == (dim,))
                ])

    @property
    def noncoordinates(self):
        # A coordinate variable is a 1-dimensional variable with the
        # same name as its dimension
        return OrderedDict([(name, v)
                for (name, v) in self.variables.iteritems()
                if name not in self.coordinates])

    def translate(self, target):
        target.store.unchecked_set_dimensions(self.dimensions)
        target.store.unchecked_set_variables(self.variables)
        target.store.unchecked_set_attributes((self.attributes))
        target.store.sync()

    def dump(self, filepath, *args, **kwdargs):
        """
        Dump the contents to a location on disk using
        the netCDF4 package
        """
        nc4_store = backends.NetCDF4DataStore(filepath, mode='w',
                                              *args, **kwdargs)
        out = Dataset(store=nc4_store)
        self.translate(out)

    def dumps(self):
        """
        Serialize the contents to a string.  The serialization
        creates an in memory netcdf version 3 string using
        the scipy.io.netcdf package.
        """
        fobj = StringIO()
        scipy_store = backends.ScipyDataStore(fobj, mode='w')
        out = Dataset(store=scipy_store)
        self.translate(out)
        return fobj.getvalue()

    def __str__(self):
        """Create a ncdump-like summary of the object"""
        summary = ["dimensions:"]
        # prints dims that look like:
        #    dimension = length
        dim_print = lambda d, l : "\t%s = %s" % (conventions.pretty_print(d, 30),
                                                 conventions.pretty_print(l, 10))
        # add each dimension to the summary
        summary.extend([dim_print(d, l) for d, l in self.dimensions.iteritems()])

        # Print variables
        summary.append("\nvariables:")
        for vname, var in self.variables.iteritems():
            # this looks like:
            #    dtype name(dim1, dim2)
            summary.append("\t%s %s(%s)" % (conventions.pretty_print(var.dtype, 8),
                                            conventions.pretty_print(vname, 20),
                                            conventions.pretty_print(', '.join(var.dimensions), 45)))
            #        attribute:value
            summary.extend(["\t\t%s:%s" % (conventions.pretty_print(att, 30),
                                           conventions.pretty_print(val, 30))
                            for att, val in var.attributes.iteritems()])

        summary.append("\nattributes:")
        #    attribute:value
        summary.extend(["\t%s:%s" % (conventions.pretty_print(att, 30),
                                     conventions.pretty_print(val, 30))
                        for att, val in self.attributes.iteritems()])
        # create the actual summary
        return '\n'.join(summary)

    def __getitem__(self, key):
        if key in self.variables:
            return self.variables[key]
        else:
            raise ValueError("%s is not a variable" % key)

    def create_dimension(self, name, length):
        """Adds a dimension with name dim and length to the object

        Parameters
        ----------
        name : string
            The name of the new dimension. An exception will be raised if the
            object already has a dimension with this name.
        length : int or None
            The length of the new dimension; must be non-negative and
            representable as a signed 32-bit integer.
        """
        if name in self.dimensions:
            raise ValueError("Dimension named '%s' already exists" % name)
        if length is None:
            # unlimted dimensions aren't allowed yet
            raise ValueError(" unlimited dimensions are not allowed")
        else:
            if not isinstance(length, int):
                raise TypeError("Dimension length must be int")
            assert length >= 0
        self._unchecked_create_dimension(name, length)

    def create_variable(self, name, dims, data, attributes=None):
        """Create a new variable.

        Parameters
        ----------
        name : string
            The name of the new variable. An exception will be raised
            if the object already has a variable with this name. name
            must satisfy netCDF-3 naming rules. If name equals the name
            of a dimension, then the new variable is treated as a
            coordinate variable and must be 1-dimensional.
        dims : tuple
            The dimensions of the new variable. Elements must be dimensions of
            the object.
        data : numpy.ndarray or None, optional
            Data to populate the new variable. If None (default), then
            an empty numpy array is allocated with the appropriate
            shape and dtype. If data contains int64 integers, it will
            be coerced to int32 (for the sake of netCDF compatibility),
            and an exception will be raised if this coercion is not
            safe.
        attributes : dict_like or None, optional
            Attributes to assign to the new variable. Attribute names
            must be unique and must satisfy netCDF-3 naming rules. If
            None (default), an empty attribute dictionary is
            initialized.

        Returns
        -------
        var : Variable
            Reference to the newly created variable.
        """
        if name in self.variables:
            raise ValueError("Variable named '%s' already exists" % (name))

        if not all([(d in self.dimensions) for d in dims]):
            bad = [d for d in dims if (d not in self.dimensions)]
            raise ValueError("the following dim(s) are not valid " +
                    "dimensions of this object: %s" % bad)

        data = np.asarray(data)
        for axis, cdim in enumerate(dims):
            if (not (data.shape[axis] == self.dimensions[cdim])):
                raise ValueError("data shape does not match dimensions: " +
                                 "axis %d (dims '%s'). " %
                                 (axis, cdim) +
                                 "expected length %d, got %d." %
                                 (self.dimensions[cdim],
                                  data.shape[axis]))
        if (name in self.dimensions) and (data.ndim != 1):
            raise ValueError("A coordinate variable must be defined with " +
                             "1-dimensional data")
        return self._unchecked_create_variable(name, dims, data, attributes)

    def create_coordinate(self, name, data, attributes=None):
        """Create a new dimension and a corresponding coordinate variable.

        This method combines the create_dimension and create_variable methods
        for the common case when the variable is a 1-dimensional coordinate
        variable with the same name as the dimension.

        Parameters
        ----------
        name : string
            The name of the new dimension and variable. An exception
            will be raised if the object already has a dimension or
            variable with this name. name must satisfy netCDF-3 naming
            rules.
        data : array_like
            The coordinate values along this dimension; must be
            1-dimensional.  The dtype of data is the dtype of the new
            coordinate variable, and the size of data is the length of
            the new dimension. If data contains int64 integers, it will
            be coerced to int32 (for the sake of netCDF compatibility),
            and an exception will be raised if this coercion is not
            safe.
        attributes : dict_like or None, optional
            Attributes to assign to the new variable. Attribute names
            must be unique and must satisfy netCDF-3 naming rules. If
            None (default), an empty attribute dictionary is
            initialized.

        Returns
        -------
        var : Variable
            Reference to the newly created coordinate variable.
        """
        data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError("data must be 1-dimensional (vector)")
        # We need to be cleanly roll back the effects of
        # create_dimension if create_variable fails, otherwise we will
        # end up in a partial state.
        if data.ndim != 1:
            raise ValueError("coordinate must have ndim==1")
        return self._unchecked_create_coordinate(name, data, attributes)

    def add_variable(self, name, variable):
        """A convenience function for adding a variable from one object to
        another.

        Parameters:
        name : string - The name under which the variable will be added
        variable : core.Variable - The variable to be added. If the desired
            action is to add a copy of the variable be sure to do so before
            passing it to this function.
        """
        # any error checking should be taken care of by create_variable
        return self.create_variable(name,
                                    dims=variable.dimensions,
                                    data=variable.data,
                                    attributes=variable.attributes)

    def delete_variable(self, name):
        """Delete a variable. Dimensions on which the variable is
        defined are not affected.

        Parameters
        ----------
        name : string
            The name of the variable to be deleted. An exception will
            be raised if there is no variable with this name.
        """
        if name not in self.variables:
            raise ValueError("Object does not have a variable '%s'" %
                    (str(name)))
        else:

            super(type(self.variables), self.variables).__delitem__(name)

    def views(self, slicers):
        """Return a new object whose contents are a view of a slice from the
        current object along a specified dimension

        Parameters
        ----------
        slicers : {dim: slice, ...}
            A dictionary mapping from a dimension to a slice object.

        Returns
        -------
        obj : Data object
            The returned object has the same attributes, dimensions,
            variable names and variable attributes as the original.
            Variables that are not defined along the specified
            dimensions are viewed in their entirety. Variables that are
            defined along the specified dimension have their data
            contents taken along the specified dimension.

            Care must be taken since modifying (most) values in the returned
            object will result in modification to the parent object.

        See Also
        --------
        view
        numpy.take
        Variable.take
        """
        if not all([isinstance(sl, slice) for sl in slicers.values()]):
            raise ValueError("view expects a dict whose values are slice objects")
        if not all([k in self.dimensions for k in slicers.keys()]):
            invalid = [k for k in slicers.keys() if not k in self.dimensions]
            raise KeyError("dimensions %s don't exist" % ', '.join(map(str, invalid)))
        # Create a new object
        obj = self._allocate()
        # Create views onto the variables and infer the new dimension length
        new_dims = dict(self.dimensions.iteritems())
        for (name, var) in self.variables.iteritems():
            var_slicers = dict((k, v) for k, v in slicers.iteritems() if k in var.dimensions)
            if len(var_slicers):
                obj.store.unchecked_add_variable(name, var.views(var_slicers))
                new_dims.update(dict(zip(obj[name].dimensions, obj[name].shape)))
            else:
                obj.store.unchecked_add_variable(name, var)
        # Hard write the dimensions, skipping validation
        obj.store.unchecked_set_dimensions(new_dims)
        # Reference to the attributes, this intentionally does not copy.
        obj.store.unchecked_set_attributes(self.attributes)
        return obj

    def view(self, s, dim=None):
        """Return a new object whose contents are a view of a slice from the
        current object along a specified dimension

        Parameters
        ----------
        s : slice
            The slice representing the range of the values to extract.
        dim : string, optional
            The dimension to slice along. If multiple dimensions of a
            variable equal dim (e.g. a correlation matrix), then that
            variable is sliced only along both dimensions.  Without
            this behavior the resulting data object would have
            inconsistent dimensions.

        Returns
        -------
        obj : Data object
            The returned object has the same attributes, dimensions,
            variable names and variable attributes as the original.
            Variables that are not defined along the specified
            dimensions are viewed in their entirety. Variables that are
            defined along the specified dimension have their data
            contents taken along the specified dimension.

            Care must be taken since modifying (most) values in the returned
            object will result in modification to the parent object.

        See Also
        --------
        views
        numpy.take
        Variable.take
        """
        obj = self.views({dim : s})
        if obj.dimensions[dim] == 0:
            raise IndexError("view results in a dimension of length zero")
        return obj

    def take(self, indices, dim=None):
        """Return a new object whose contents are taken from the
        current object along a specified dimension

        Parameters
        ----------
        indices : array_like
            The indices of the values to extract. indices must be compatible
            with the ndarray.take() method.
        dim : string, optional
            The dimension to slice along. If multiple dimensions of a
            variable equal dim (e.g. a correlation matrix), then that
            variable is sliced only along its first matching dimension.
            If None (default), then the object is sliced along its
            unlimited dimension; an exception is raised if the object
            does not have an unlimited dimension.

        Returns
        -------
        obj : Data object
            The returned object has the same attributes, dimensions,
            variable names and variable attributes as the original.
            Variables that are not defined along the specified
            dimensions are copied in their entirety. Variables that are
            defined along the specified dimension have their data
            contents taken along the specified dimension.

        See Also
        --------
        numpy.take
        Variable.take
        """
        if dim is None:
            raise ValueError("dim cannot be None")
        # Create a new object
        obj = self._allocate()
        # Create fancy-indexed variables and infer the new dimension length
        new_length = self.dimensions[dim]
        for (name, var) in self.variables.iteritems():
            if dim in var.dimensions:
                obj.store.unchecked_add_variable(name, var.take(indices, dim))
                new_length = obj.variables[name].data.shape[
                    list(var.dimensions).index(dim)]
            else:
                obj.store.unchecked_add_variable(name, copy.deepcopy(var))
        # Hard write the dimensions, skipping validation
        for d, l in self.dimensions.iteritems():
            if d == dim:
                l = new_length
            obj.store.unchecked_create_dimension(d, l)
        if obj.dimensions[dim] == 0:
            raise IndexError(
                "take would result in a dimension of length zero")
        # Copy attributes
        self._unchecked_set_attributes(self.attributes.copy())
        return obj

    def renamed(self, name_dict):
        """
        Returns a copy of the current object with variables and dimensions
        reanmed according to the arguments passed via **kwds

        Parameters
        ----------
        name_dict : dict-like
            Dictionary-like object whose keys are current variable
            names and whose values are new names.
        """
        for name in self.dimensions.iterkeys():
            if name in self.variables and not name in self.coordinates:
                raise ValueError("Renaming assumes that only coordinates " +
                                 "have both a dimension and variable under " +
                                 "the same name.  In this case it appears %s " +
                                 "has a dim and var but is not a coordinate"
                                 % name)

        new_names = dict((name, name)
                for name, _ in self.dimensions.iteritems())
        new_names.update(dict((name, name)
                for name, _ in self.variables.iteritems()))

        for k, v in name_dict.iteritems():
            if not k in new_names:
                raise ValueError("Cannot rename %s because it does not exist" % k)
        new_names.update(name_dict)

        obj = self._allocate()
        # if a dimension is a new one it gets added, if the dimension already
        # exists we confirm that they are identical (or throw an exception)
        for (name, length) in self.dimensions.iteritems():
            obj.create_dimension(new_names[name], length)
        # a variable is only added if it doesn't currently exist, otherwise
        # and exception is thrown
        for (name, v) in self.variables.iteritems():
            obj.create_variable(new_names[name],
                                tuple([new_names[d] for d in v.dimensions]),
                                data=v.data.copy(),
                                attributes=v.attributes.copy())
        # update the root attributes
        self._unchecked_set_attributes(self.attributes.copy())
        return obj

    def update(self, other):
        """
        An update method (simular to dict.update) for data objects whereby each
        dimension, variable and attribute from 'other' is updated in the current
        object.  Note however that because Data object attributes are often
        write protected an exception will be raised if an attempt to overwrite
        any variables is made.
        """
        # if a dimension is a new one it gets added, if the dimension already
        # exists we confirm that they are identical (or throw an exception)
        for (name, length) in other.dimensions.iteritems():
            if (name == other.record_dimension and
                    name != self.record_dimension):
                raise ValueError(
                    ("record dimensions do not match: "
                     "self: %s, other: %s") %
                    (self.record_dimension, other.record_dimension))
            if not name in self.dimensions:
                self.create_dimension(name, length)
            else:
                cur_length = self.dimensions[name]
                if cur_length is None:
                    cur_length = self[self.record_dimension].data.size
                if length != cur_length:
                    raise ValueError("inconsistent dimension lengths for " +
                                     "dim: %s , %s != %s" %
                                     (name, length, cur_length))
        # a variable is only added if it doesn't currently exist, otherwise
        # and exception is thrown
        for (name, v) in other.variables.iteritems():
            if not name in self.variables:
                self.create_variable(name,
                                     v.dimensions,
                                     data=v.data.copy(),
                                     attributes=v.attributes.copy())
            else:
                if self[name].dimensions != other[name].dimensions:
                    raise ValueError("%s has different dimensions cur:%s new:%s"
                                     % (name, str(self[name].dimensions),
                                        str(other[name].dimensions)))
                if (self.variables[name].data.tostring() !=
                    other.variables[name].data.tostring()):
                    raise ValueError("%s has different data" % name)
                self[name].attributes.update(other[name].attributes)
        # update the root attributes
        self.attributes.update(other.attributes)

    def select(self, var):
        """Return a new object that contains the specified variables,
        along with the dimensions on which those variables are defined
        and corresponding coordinate variables.

        Parameters
        ----------
        var : bounded sequence of strings
            The variables to include in the returned object.

        Returns
        -------
        obj : Data object
            The returned object has the same attributes as the
            original. A dimension is included if at least one of the
            specified variables is defined along that dimension.
            Coordinate variables (1-dimensional variables with the same
            name as a dimension) that correspond to an included
            dimension are also included. All other variables are
            dropped.
        """
        if isinstance(var, basestring):
            var = [var]
        if not (hasattr(var, '__iter__') and hasattr(var, '__len__')):
            raise TypeError("var must be a bounded sequence")
        if not all((v in self.variables for v in var)):
            raise KeyError(
                "One or more of the specified variables does not exist")
        # Create a new Data instance
        obj = self._allocate()
        # Copy relevant dimensions
        dim = reduce(or_, [set(self.variables[v].dimensions) for v in var])
        # Create dimensions in the same order as they appear in self.dimension
        for d in dim:
            obj.store.unchecked_create_dimension(d, self.dimensions[d])
        # Also include any coordinate variables defined on the relevant
        # dimensions
        for (name, v) in self.variables.iteritems():
            if (name in var) or ((name in dim) and (v.dimensions == (name,))):
                obj._unchecked_create_variable(name,
                            dims=v.dimensions,
                            data=v.data.copy(),
                            attributes=v.attributes.copy())
        obj.store.unchecked_set_attributes(self.attributes.copy())
        return obj

    def iterator(self, dim=None, views=False):
        """Iterator along a data dimension

        Return an iterator yielding (coordinate, data_object) pairs
        that are singleton along the specified dimension

        Parameters
        ----------
        dim : string, optional
            The dimension along which you want to iterate. If None
            (default), then the iterator operates along the record
            dimension; if there is no record dimension, an exception
            will be raised.
        views : boolean, optional
            If True, the iterator will give views of the data along
            the dimension, otherwise copies.

        Returns
        -------
        it : iterator
            The returned iterator yields pairs of scalar-valued
            coordinate variables and data objects. The yielded data
            objects contain *copies* onto the underlying numpy arrays of
            the original data object. If the data object does not have
            a coordinate variable with the same name as the specified
            dimension, then the returned coordinate value is None. If
            multiple dimensions of a variable equal dim (e.g. a
            correlation matrix), then that variable is iterated along
            the first matching dimension.

        Examples
        --------
        >>> d = Data()
        >>> d.create_coordinate(name='x', data=numpy.arange(10))
        >>> d.create_coordinate(name='y', data=numpy.arange(20))
        >>> print d

        dimensions:
          name            | length
         ===========================
          x               | 10
          y               | 20

        variables:
          name            | dtype   | shape           | dimensions
         =====================================================================
          x               | int32   | (10,)           | ('x',)
          y               | int32   | (20,)           | ('y',)

        attributes:
          None

        >>> i = d.iterator(dim='x')
        >>> (a, b) = i.next()
        >>> print a

        dtype:
          int32

        dimensions:
          name            | length
         ===========================
          x               | 1

        attributes:
          None

        >>> print b

        dimensions:
          name            | length
         ===========================
          x               | 1
          y               | 20

        variables:
          name            | dtype   | shape           | dimensions
         =====================================================================
          x               | int32   | (1,)            | ('x',)
          y               | int32   | (20,)           | ('y',)

        attributes:
          None

        """
        # Determine the size of the dim we're about to iterate over
        n = self.dimensions[dim]
        # Iterate over the object
        if dim in self.coordinates:
            coord = self.variables[dim]
            if views:
                for i in xrange(n):
                    s = slice(i, i + 1)
                    yield (coord.view(s, dim=dim),
                           self.view(s, dim=dim))
            else:
                for i in xrange(n):
                    indices = np.array([i])
                    yield (coord.take(indices, dim=dim),
                           self.take(indices, dim=dim))
        else:
            if views:
                for i in xrange(n):
                    yield (None, self.view(slice(i, i + 1), dim=dim))
            else:
                for i in xrange(n):
                    yield (None, self.take(np.array([i]), dim=dim))

    def iterarray(self, var, dim=None):
        """Iterator along a data dimension returning the corresponding slices
        of the underlying data of a varaible.

        Return an iterator yielding (scalar, ndarray) pairs that are singleton
        along the specified dimension.  While iterator is more general, this
        method has less overhead and in turn should be considerably faster.

        Parameters
        ----------
        var : string
            The variable over which you want to iterate.

        dim : string, optional
            The dimension along which you want to iterate. If None
            (default), then the iterator operates along the record
            dimension; if there is no record dimension, an exception
            will be raised.

        Returns
        -------
        it : iterator
            The returned iterator yields pairs of scalar-valued
            and ndarray objects. The yielded data objects contain *views*
            onto the underlying numpy arrays of the original data object.

        Examples
        --------
        >>> d = Data()
        >>> d.create_coordinate(name='t', data=numpy.arange(5))
        >>> d.create_dimension(name='h', length=3)
        >>> d.create_variable(name='x', dim=('t', 'h'),\
        ...     data=numpy.random.random((10, 3,)))
        >>> print d['x'].data
        [[ 0.33499995  0.47606901  0.41334325]
         [ 0.20229308  0.73693437  0.97451746]
         [ 0.40020704  0.29763575  0.85588908]
         [ 0.44114434  0.79233816  0.59115313]
         [ 0.18583972  0.55084889  0.95478946]]
        >>> i = d.iterarray(var='x', dim='t')
        >>> (a, b) = i.next()
        >>> print a
        0
        >>> print b
        [[ 0.33499995  0.47606901  0.41334325]]
        """
        # Get a reference to the underlying ndarray for the desired variable
        # and build a list of slice objects
        data = self.variables[var].data
        axis = list(self.variables[var].dimensions).index(dim)
        slicer = [slice(None)] * data.ndim
        # Determine the size of the dim we're about to iterate over
        n = self.dimensions[dim]
        # Iterate over dim returning views of the variable.
        if dim in self.coordinates:
            coord = self.variables[dim].data
            for i in xrange(n):
                slicer[axis] = slice(i, i + 1)
                yield (coord[i], data[slicer])
        else:
            for i in xrange(n):
                slicer[axis] = slice(i, i + 1)
                yield (None, data[slicer])

    def squeeze(self, dimension):
        """
        Squeezes a dimension of length 1, returning a copy of the object
        with that dimension removed.
        """
        if self.dimensions[dimension] != 1:
            raise ValueError(("Can only squeeze along dimensions with" +
                             "length one, %s has length %d") %
                             (dimension, self.dimensions[dimension]))
        # Create a new Data instance
        obj = self._allocate()
        # Copy dimensions
        for (name, length) in self.dimensions.iteritems():
            if not name == dimension:
                obj.create_dimension(name, length)
        # Copy variables
        for (name, var) in self.variables.iteritems():
            if not name == dimension:
                dims = list(var.dimensions)
                data = var.data.copy()
                if dimension in dims:
                    shape = list(var.data.shape)
                    index = dims.index(dimension)
                    shape.pop(index)
                    dims.pop(index)
                    data = data.reshape(shape)
                obj.create_variable(name=name,
                        dims=tuple(dims),
                        data=data,
                        attributes=var.attributes.copy())
        obj.store.unchecked_set_attributes(self.attributes.copy())
        return obj

if __name__ == "__main__":
    """
    A bunch of regression tests.
    """
    base_dir = os.path.dirname(__file__)
    test_dir = os.path.join(base_dir, '..', '..', 'test', )
    write_test_path = os.path.join(test_dir, 'test_output.nc')
    ecmwf_netcdf = os.path.join(test_dir, 'ECMWF_ERA-40_subset.nc')

    import time
    st = time.time()
    nc = Dataset(ecmwf_netcdf)
    print "Seconds to read from filepath : ", time.time() - st

    st = time.time()
    nc.dump(write_test_path)
    print "Seconds to write : ", time.time() - st

    st = time.time()
    nc_string = nc.dumps()
    print "Seconds to serialize : ", time.time() - st

    st = time.time()
    nc = Dataset(nc_string)
    print "Seconds to deserialize : ", time.time() - st

    st = time.time()
    with open(ecmwf_netcdf, 'r') as f:
        nc = Dataset(f)
    print "Seconds to read from fobj : ", time.time() - st

