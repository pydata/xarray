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


def construct_dimensions(variables):
    """
    Given a dictionary of variables, construct a dimensions mapping

    Parameters
    ----------
    variables : mapping
        Mapping from variable names to Variable objects.

    Returns
    -------
    dimensions : mapping
        Mapping from dimension names to lengths.

    Raises
    ------
    ValueError if variable dimensions are inconsistent.
    """
    dimensions = OrderedDict()
    for k, var in variables.iteritems():
        for dim, length in zip(var.dimensions, var.shape):
            if dim not in dimensions:
                dimensions[dim] = length
            elif dimensions[dim] != length:
                raise ValueError('dimension %r on variable %r has length %s '
                                 'but already is saved with length %s' %
                                 (dim, k, length, dimensions[dim]))
    return dimensions


def check_dims_and_vars_consistency(dimensions, variables):
    """
    Validate dimensions and variables are consistent

    Parameters
    ----------
    dimensions : mapping
        Mapping from dimension names to lengths.
    variables : mapping
        Mapping from variable names to Variable objects.

    Raises
    ------
    ValueError if variable dimensions are inconsistent with the provided
    dimensions.
    """
    for k, var in variables.iteritems():
        if k in dimensions and var.ndim != 1:
            raise ValueError('a coordinate variable must be defined with '
                             '1-dimensional data')
        for dim, length in zip(var.dimensions, var.shape):
            if dim not in dimensions:
                raise ValueError('dimension %r on variable %r is not one '
                                 'of the dataset dimensions %r' %
                                 (dim, k, list(dimensions)))
            elif dimensions[dim] != length:
                raise ValueError('dimension %r on variable %r has length '
                                 '%s but in on the dataset has length %s' %
                                 (dim, k, length, dimensions[dim]))


def open_dataset(nc, *args, **kwargs):
    #TODO: add tests for this function
    # move this to a classmethod Dataset.open?
    if isinstance(nc, basestring) and not nc.startswith('CDF'):
        # If the initialization nc is a string and it doesn't
        # appear to be the contents of a netcdf file we load
        # it using the netCDF4 package
        store = backends.NetCDF4DataStore(nc, *args, **kwargs)
    else:
        # If nc is a file-like object we read it using
        # the scipy.io.netcdf package
        store = backends.ScipyDataStore(nc, *args, **kwargs)
    return Dataset(store=store)


class Dataset(object):
    """
    A netcdf-like data object consisting of dimensions, variables and
    attributes which together form a self describing data set

    Dataset objects can also be treated as a mapping from variable names to
    Variable objects.

    They should be modified by using methods, not by directly changing any of
    the attributes listed below:
    TODO: change this!

    Attributes
    ----------
    dimensions : {name: length, ...}
    variables : {name: variable, ...}
    coordinates : {name: variable, ...}
        Coordinates are simply variables that are also dimensions. They must
        all have dimension 1.
    noncoordinates : {name: variable, ...}
        Variables that are not coordinates.
    attributes : dict-like
    store : baackends.*DataStore
    """
    def __init__(self, variables=None, dimensions=None, attributes=None,
                 store=None, check_consistency=True):
        """
        If dimensions are not provided, they are inferred from the variables.

        Otherwise, variables and dimensions are only checked for consistency
        if check_dimensions=True.
        """
        # TODO: fill out this docstring
        if store is None:
            store = backends.InMemoryDataStore()
        object.__setattr__(self, 'store', store)

        if attributes is not None:
            self._unchecked_set_attributes(attributes)

        if dimensions is not None:
            self._unchecked_set_dimensions(dimensions)

        if variables is not None:
            if dimensions is None:
                self._unchecked_set_dimensions(construct_dimensions(variables))
            elif check_consistency:
                check_dims_and_vars_consistency(dimensions, variables)
            self._unchecked_set_variables(variables)

    def _unchecked_set_dimensions(self, *args, **kwdargs):
        self.store.unchecked_set_dimensions(*args, **kwdargs)

    def _unchecked_set_attributes(self, *args, **kwdargs):
        self.store.unchecked_set_attributes(*args, **kwdargs)

    def _unchecked_set_variables(self, *args, **kwdargs):
        self.store.unchecked_set_variables(*args, **kwdargs)

    def _unchecked_create_dimension(self, *args, **kwdargs):
        self.store.unchecked_create_dimension(*args, **kwdargs)

    def _unchecked_add_variable(self, *args, **kwdargs):
        return self.store.unchecked_add_variable(*args, **kwdargs)

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

    def copy(self):
        """
        Returns a shallow copy of the current object.
        """
        return self.__copy__()

    def __copy__(self):
        """
        Returns a shallow copy of the current object.
        """
        return Dataset(self.variables, self.dimensions, self.attributes,
                       check_consistency=False)

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

    def __iter__(self):
        return iter(self.variables)

    def __getitem__(self, key):
        return self.variables[key]

    def __setitem__(self, key, value):
        return self.add_variable(key, value)

    def __delitem__(self, key):
        # does deleting variables make sense for all backends?
        raise NotImplementedError

    def __eq__(self, other):
        try:
            # some stores (e.g., scipy) do not seem to preserve order, so don't
            # require matching order for equality
            return (dict(self.dimensions) == dict(other.dimensions)
                    and dict(self.variables) == dict(other.variables)
                    and self.attributes == other.attributes)
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    @property
    def coordinates(self):
        # A coordinate variable is a 1-dimensional variable with the
        # same name as its dimension
        return OrderedDict([(dim, self.variables[dim])
                for dim in self.dimensions
                if dim in self.variables and
                self.variables[dim].data.ndim == 1 and
                self.variables[dim].dimensions == (dim,)])

    @property
    def noncoordinates(self):
        # A coordinate variable is a 1-dimensional variable with the
        # same name as its dimension
        return OrderedDict([(name, v)
                for (name, v) in self.variables.iteritems()
                if name not in self.coordinates])

    def dump_to(self, store):
        """
        Dump dataset contents to a backends.*DataStore object
        """
        target = Dataset(self.variables, self.dimensions, self.attributes,
                         store=store, check_consistency=False)
        target.store.sync()

    def dump(self, filepath, *args, **kwdargs):
        """
        Dump dataset contents to a location on disk using
        the netCDF4 package
        """
        nc4_store = backends.NetCDF4DataStore(filepath, mode='w',
                                              *args, **kwdargs)
        self.dump_to(nc4_store)

    def dumps(self):
        """
        Serialize dataset contents to a string.  The serialization
        creates an in memory netcdf version 3 string using
        the scipy.io.netcdf package.
        """
        fobj = StringIO()
        scipy_store = backends.ScipyDataStore(fobj, mode='w')
        self.dump_to(scipy_store)
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

    def create_dimension(self, name, length):
        """Adds a dimension with name dim and length to the object

        Parameters
        ----------
        name : string
            The name of the new dimension. An exception will be raised if the
            object already has a dimension with this name.
        length : int
            The length of the new dimension; must a be non-negative integer.
        """
        if name in self.dimensions:
            raise ValueError('dimension named %r already exists' % name)
        elif not isinstance(length, int):
            raise TypeError('length must be an integer')
        elif length < 0:
            raise ValueError('length must be non-negative')
        self._unchecked_create_dimension(name, int(length))

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
        data : numpy.ndarray
            Data to populate the new variable.
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
        # any error checking should be taken care of by add_variable
        v = variable.Variable(dims, np.asarray(data), attributes)
        return self.add_variable(name, v)

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
            the new dimension.
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
        # We need to be cleanly roll back the effects of
        # create_dimension if create_variable fails, otherwise we will
        # end up in a partial state.
        if name in self.dimensions:
            raise ValueError("dimension named '%s' already exists" % name)
        var = variable.Variable((name,), np.asarray(data), attributes)
        if var.ndim != 1:
            raise ValueError("coordinate data must be 1-dimensional (vector)")
        self._unchecked_create_dimension(name, var.size)
        return self._unchecked_add_variable(name, var)

    def add_variable(self, name, var):
        """Add a variable to the dataset

        Parameters
        ----------
        name : string
            The name under which the variable will be added
        variable : variable.Variable
            The variable to be added. If the desired action is to add a copy of
            the variable be sure to do so before passing it to this function.

        Returns
        -------
        variable
            The variable object in the underlying datastore
        """
        if name in self.variables:
            raise ValueError("Variable named %r already exists" % name)
        check_dims_and_vars_consistency(self.dimensions, {name: var})
        return self._unchecked_add_variable(name, var)

    def views(self, slicers):
        """Return a new object whose contents are a view of a slice from the
        current object along a specified dimension

        Parameters
        ----------
        slicers : {dim: slice, ...}
            A dictionary mapping from dimensions to integers or slice objects.

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
        if not all(k in self.dimensions for k in slicers):
            invalid = [k for k in slicers if not k in self.dimensions]
            raise KeyError("dimensions %r don't exist" % invalid)

        # slice all variables
        variables = OrderedDict()
        for (name, var) in self.variables.iteritems():
            var_slicers = dict((k, v) for k, v in slicers.iteritems()
                               if k in var.dimensions)
            variables[name] = var.views(var_slicers)

        def search_dim_len(dim, variables):
            # loop through the variables to find the dimension length, or if
            # the dimension is not found, return None
            for var in variables.values():
                if dim in var.dimensions:
                    return int(var.shape[var.dimensions.index(dim)])
            return None

        # update dimensions
        dimensions = OrderedDict()
        for dim in self.dimensions:
            new_len = search_dim_len(dim, variables)
            if new_len is not None:
                # dimension length is defined by a new dataset variable
                dimensions[dim] = new_len
            elif search_dim_len(dim, self.variables) is None:
                # dimension length is also not defined by old dataset variables
                # note: dimensions only defined in old dataset variables are be
                # dropped
                if dim not in slicers:
                    dimensions[dim] = self.dimensions[dim]
                else:
                    # figure it by slicing temporary coordinate data
                    temp_data = np.arange(self.dimensions[dim])
                    temp_data_sliced = temp_data[slicers[dim]]
                    new_len = temp_data_sliced.size
                    if new_len > 0 and temp_data_sliced.ndim > 0:
                        # drop the dimension if the result of getitem is an
                        # integer (dimension 0)
                        dimensions[dim] = new_len

        return type(self)(variables, dimensions, self.attributes,
                          check_consistency=False)

    def view(self, s, dim):
        """Return a new object whose contents are a view of a slice from the
        current object along a specified dimension

        Parameters
        ----------
        s : slice
            The slice representing the range of the values to extract.
        dim : string, optional
            The dimension to slice along.

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
        return self.views({dim: s})

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
        obj = type(self)()
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
        Returns a new object with renamed variables and dimensions

        Parameters
        ----------
        name_dict : dict-like
            Dictionary-like object whose keys are current variable or dimension
            names and whose values are new names.
        """
        for k in name_dict:
            if k not in self.dimensions and k not in self.variables:
                raise ValueError("Cannot rename %r because it is not a "
                                 "variable or dimension in this dataset" % k)
        variables = OrderedDict()
        for k, v in self.variables.iteritems():
            name = name_dict.get(k, k)
            dims = tuple(name_dict.get(dim, dim) for dim in v.dimensions)
            #TODO: public interface for renaming a variable without loading
            # data
            variables[name] = variable.Variable(dims, v._data, v.attributes)

        dimensions = OrderedDict((name_dict.get(k, k), v)
                                 for k, v in self.dimensions.iteritems())

        return type(self)(variables, dimensions, self.attributes,
                          check_consistency=False)

    def _update(self, other, override_attributes=True):
        """
        An update method (simular to dict.update) for data objects whereby each
        dimension, variable and attribute from 'other' is updated in the current
        object.  Note however that because Data object attributes are often
        write protected an exception will be raised if an attempt to overwrite
        any variables is made.
        """
        if not override_attributes:
            #TODO: add options to hard fail instead of overriding attributes
            raise NotImplementedError

        # if a dimension is a new one it gets added, if the dimension already
        # exists we confirm that they are identical (or throw an exception)
        for (name, length) in other.dimensions.iteritems():
            if not name in self.dimensions:
                self.create_dimension(name, length)
            else:
                cur_length = self.dimensions[name]
                if length != cur_length:
                    raise ValueError("inconsistent dimension lengths for "
                                     "dim: %s, %s != %s" %
                                     (name, length, cur_length))
        # a variable is only added if it doesn't currently exist, otherwise
        # it is confirmed to be identical (except for attributes)
        for (name, v1) in other.variables.iteritems():
            if not name in self.variables:
                self.add_variable(name, v1)
            else:
                v0 = self.variables[name]
                if v0.dimensions != v1.dimensions:
                    raise ValueError("%r has different dimensions cur:%r new:%r"
                                     % (name, v0.dimensions, v1.dimensions))
                elif v0._data is not v1._data and np.any(v0.data != v1.data):
                    raise ValueError("%s has different data" % name)
                v0.attributes.update(v1.attributes)
        # update the root attributes
        self._unchecked_set_attributes(other.attributes)

    def join(self, other, override_attributes=True):
        """
        Join two datasets into a single new dataset

        Raises ValueError if any variables or dimensions do not match.
        """
        obj = self.copy()
        obj._update(other, override_attributes=override_attributes)
        return obj

    def select(self, names):
        """Return a new object that contains the specified namesiables,
        along with the dimensions on which those variables are defined
        and corresponding coordinate variables.

        Parameters
        ----------
        names : bounded sequence of strings
            Names of the variables to include in the returned object.

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
        if isinstance(names, basestring):
            names = [names]
        if not (hasattr(names, '__iter__') and hasattr(names, '__len__')):
            raise TypeError("names must be a bounded sequence")
        if not all(k in self.variables for k in names):
            raise KeyError(
                "One or more of the specified variables does not exist")

        dim_names = [set(self.variables[k].dimensions) for k in names]
        names = set(names).union(*dim_names)

        variables = OrderedDict((k, v) for k, v in self.variables.iteritems()
                                if k in names)
        dimensions = OrderedDict((k, v) for k, v in self.dimensions.iteritems()
                                 if k in names)
        return type(self)(variables, dimensions, self.attributes,
                          check_consistency=False)

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
        obj = type(self)()
        # Copy dimensions
        for (name, length) in self.dimensions.iteritems():
            if not name == dimension:
                obj.create_dimension(name, length)
        # Copy variables
        for (name, var) in self.variables.iteritems():
            if not name == dimension:
                dims = list(var.dimensions)
                data = var.data
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

