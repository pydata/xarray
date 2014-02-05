# TODO Use various backend data stores. pytable, ncdf4, scipy.io, iris, memory
import os
import numpy as np
import netCDF4 as nc4
import pandas as pd

from cStringIO import StringIO
from collections import OrderedDict, MutableMapping

from dataview import DataView
from utils import FrozenOrderedDict
from variable import Variable
import backends, conventions, utils

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
    return Dataset.load_store(store)


class _IndicesCache(MutableMapping):
    """Cache for Dataset indices"""
    # MutableMapping subclasses should implement:
    # __getitem__, __setitem__, __delitem__, __iter__, __len__
    def __init__(self, dataset, cache=None):
        self.dataset = dataset
        self.cache = {} if cache is None else dict(cache)
        # for performance reasons, we could remove this:
        self.sync()

    def build_index(self, key):
        """Cache the index for the dimension 'key'"""
        self.cache[key] = self.dataset._create_index(key)

    def sync(self):
        """Cache indices for all dimensions in this dataset"""
        for key in self.dataset.dimensions:
            self.build_index(key)

    def __getitem__(self, key):
        if not key in self.cache:
            assert key in self.dataset.dimensions
            self.build_index(key)
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __delitem__(self, key):
        del self.cache[key]

    def __iter__(self):
        return iter(self.dataset.dimensions)

    def __len__(self):
        return len(self.dataset.dimensions)

    def __contains__(self, key):
        return key in self.cache

    def __repr__(self):
        contents = '\n'.join("'%s': %s" %
                             (k, str(v).replace(
                                  '\n', '\n' + ' ' * (len(k) + 4)))
                             for k, v in self.items())
        return ("<class 'scidata.data.%s' (dict-like)>\n%s"
                % (type(self).__name__, contents))


class Dataset(object):
    """
    A netcdf-like data object consisting of dimensions, variables and
    attributes which together form a self describing data set

    Datasets are containers of variable name. Getting an item from a Dataset
    returns a DataView focused on that variable.

    Attributes
    ----------
    dimensions : {name: length, ...}
    variables : {name: variable, ...}
    coordinates : {name: variable, ...}
        Coordinates are simply variables that are also dimensions. They must
        all have dimension 1.
    noncoordinates : {name: variable, ...}
        Variables that are not coordinates.
    attributes : {key: value, ...}
    indices : {dimension: index, ...}
        Mapping from dimensions to pandas.Index objects.
    store : backends.*DataStore
    """
    def __init__(self, variables=None, dimensions=None, attributes=None,
                 indices=None, store=None):
        """
        If dimensions are not provided, they are inferred from the variables.

        Only set a store if you want to Dataset operations to modify stored
        data in-place. Otherwise, load data from a store using the
        `open_dataset` function or the `from_store` class method.
        """
        # TODO: fill out this docstring
        if store is None:
            store = backends.InMemoryDataStore()
        self.store = store

        if attributes is not None:
            store.set_attributes(attributes)

        if dimensions is not None:
            store.set_dimensions(dimensions)

        if variables is not None:
            if dimensions is None:
                store.set_dimensions(construct_dimensions(variables))
            else:
                check_dims_and_vars_consistency(dimensions, variables)
            store.set_variables(variables)

        if indices is None:
            indices = {}
        else:
            for k, v in indices.iteritems():
                if k not in self.dimensions or v.size != self.dimensions[k]:
                    raise ValueError('inconsistent index %r' % k)
        self._indices = _IndicesCache(self, indices)

    @classmethod
    def load_store(cls, store):
        return cls(store.variables, store.dimensions, store.attributes)

    def _create_index(self, dim):
        if dim in self.variables:
            var = self.variables[dim]
            data = var.data
            attr = var.attributes
            if 'units' in attr and 'since' in attr['units']:
                index = utils.num2datetimeindex(data, attr['units'],
                                                attr.get('calendar'))
            else:
                index = pd.Index(data)
        elif dim in self.dimensions:
            index = pd.Index(np.arange(self.dimensions[dim]))
        else:
            raise ValueError('cannot find index %r in dataset' % dim)
        return index

    @property
    def indices(self):
        return self._indices

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
        return type(self)(self.variables, self.dimensions, self.attributes,
                          indices=self.indices.cache)

    def __contains__(self, key):
        """
        The 'in' operator will return true or false depending on
        whether 'key' is a varibale in the data object or not.
        """
        return key in self.variables

    def __iter__(self):
        return iter(self.variables)

    def __getitem__(self, key):
        return DataView(self.select(key), key)

    #TODO: add keys, items, and values methods (and the iter versions) to
    # complete the dict analogy?

    # mutable objects should not be hashable
    __hash__ = None

    def __eq__(self, other):
        try:
            # some stores (e.g., scipy) do not seem to preserve order, so don't
            # require matching dimension or variable order for equality
            return (sorted(self.dimensions.items())
                     == sorted(other.dimensions.items())
                    and sorted(self.attributes.items())
                        == sorted(other.attributes.items())
                    and all(k1 == k2 and utils.variable_equal(v1, v2)
                            for (k1, v1), (k2, v2)
                            in zip(sorted(self.variables.items()),
                                   sorted(other.variables.items()))))
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    @property
    def coordinates(self):
        """Coordinates are variables with names that match dimensions"""
        return FrozenOrderedDict([(dim, self.variables[dim])
                for dim in self.dimensions
                if dim in self.variables and
                self.variables[dim].data.ndim == 1 and
                self.variables[dim].dimensions == (dim,)])

    @property
    def noncoordinates(self):
        """Non-coordinates are variables with names that do not match
        dimensions
        """
        return FrozenOrderedDict([(name, v)
                for (name, v) in self.variables.iteritems()
                if name not in self.coordinates])

    def dump_to_store(self, store):
        """Store dataset contents to a backends.*DataStore object"""
        target = type(self)(self.variables, self.dimensions, self.attributes,
                            store=store, indices=self.indices.cache)
        target.store.sync()
        return target

    def dump(self, filepath, *args, **kwdargs):
        """Dump dataset contents to a location on disk using the netCDF4
        package
        """
        nc4_store = backends.NetCDF4DataStore(filepath, mode='w',
                                              *args, **kwdargs)
        self.dump_to_store(nc4_store)

    def dumps(self):
        """Serialize dataset contents to a string. The serialization creates an
        in memory netcdf version 3 string using the scipy.io.netcdf package.
        """
        fobj = StringIO()
        scipy_store = backends.ScipyDataStore(fobj, mode='w')
        self.dump_to_store(scipy_store)
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
        summary.append("variables:")
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

        summary.append("attributes:")
        #    attribute:value
        summary.extend(["\t%s:%s" % (conventions.pretty_print(att, 30),
                                     conventions.pretty_print(val, 30))
                        for att, val in self.attributes.iteritems()])
        # create the actual summary
        return '\n'.join(summary).replace('\t', ' ' * 4)

    def __repr__(self):
        dim_summary = ', '.join('%s%s: %s' % ('@' if k in self else '', k, v)
                                for k, v in self.dimensions.iteritems())
        return '<scidata.%s (%s): %s>' % (type(self).__name__,
                                          dim_summary,
                                          ' '.join(self.noncoordinates))

    def create_variable(self, name, dims, data, attributes=None):
        """Create a new variable and add it to this dataset

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
            Attributes to assign to the new variable. If None (default), an
            empty attribute dictionary is initialized.

        Returns
        -------
        var : Variable
            Reference to the newly created variable.
        """
        # any error checking should be taken care of by add_variable
        v = Variable(dims, np.asarray(data), attributes)
        return self.add_variable(name, v)

    def create_coordinate(self, name, data, attributes=None):
        """Create a new dimension and a corresponding coordinate variable

        This method combines the create_dimension and create_variable methods
        for the common case when the variable is a 1-dimensional coordinate
        variable with the same name as the dimension.

        If the dimension already exists, this function proceeds unless there is
        already a corresponding variable or if the lengths disagree.

        Parameters
        ----------
        name : string
            The name of the new dimension and variable. An exception will be
            raised if the object already has a dimension or variable with this
            name.
        data : array_like
            The coordinate values along this dimension; must be 1-dimensional.
            The size of data is the length of the new dimension.
        attributes : dict_like or None, optional
            Attributes to assign to the new variable. If None (default), an
            empty attribute dictionary is initialized.

        Returns
        -------
        var : Variable
            Reference to the newly created coordinate variable.
        """
        # any error checking should be taken care of by add_coordinate
        v = Variable((name,), np.asarray(data), attributes)
        return self.add_coordinate(v)

    def add_dimension(self, name, length):
        """Add a dimension to this dataset

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
        length = int(length)
        if length < 0:
            raise ValueError('length must be non-negative')
        self.store.set_dimension(name, length)

    def add_variable(self, name, var):
        """Add a variable to the dataset

        Parameters
        ----------
        name : string
            The name under which the variable will be added.
        variable : Variable
            The variable to be added. If the desired action is to add a copy of
            the variable be sure to do so before passing it to this function.

        Returns
        -------
        variable
            The variable object in the underlying datastore.
        """
        if name in self.variables:
            raise ValueError("Variable named %r already exists" % name)
        return self.set_variable(name, var)

    def add_coordinate(self, var):
        """Add a coordinate variable to the dataset

        Parameters
        ----------
        variable : Variable
            The coordinate variable to be added. Coordinate variables must be
            1D, and will be added under the same name as their sole dimension.

        Returns
        -------
        variable
            The variable object in the underlying datastore.
        """
        # We need to be cleanly roll back the effects of
        # create_dimension if create_variable fails, otherwise we will
        # end up in a partial state.
        name = var.dimensions[0]
        if name in self.coordinates:
            raise ValueError("coordinate named '%s' already exists" % name)
        if var.ndim != 1:
            raise ValueError("coordinate data must be 1-dimensional (vector)")
        if name not in self.dimensions:
            self.store.set_dimension(name, var.size)
        elif self.dimensions[name] != var.size:
            raise ValueError('dimension already exists with different length')
        return self.store.set_variable(name, var)

    def set_variable(self, name, var):
        """Set a variable in the dataset

        Unlike `add_variable`, this function allows for overriding existing
        variables.

        Parameters
        ----------
        name : string
            The name under which the variable will be added.
        variable : Variable
            The variable to be added. If the desired action is to add a copy of
            the variable be sure to do so before passing it to this function.

        Returns
        -------
        variable
            The variable object in the underlying datastore.
        """
        check_dims_and_vars_consistency(self.dimensions, {name: var})
        new_var = self.store.set_variable(name, var)
        if name in self.indices:
            self.indices.build_index(name)
        return new_var

    def views(self, **slicers):
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
        invalid = [k for k in slicers if not k in self.dimensions]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        # all slicers should be int, slice or np.ndarrays
        slicers = {k: np.asarray(v) if not isinstance(v, (int, slice)) else v
                   for k, v in slicers.iteritems()}

        variables = OrderedDict()
        for name, var in self.variables.iteritems():
            var_slicers = {k: v for k, v in slicers.iteritems()
                           if k in var.dimensions}
            variables[name] = var.views(**var_slicers)

        indices = {k: (v[slicers[k]] if k in slicers else v)
                   for k, v in self.indices.iteritems()}
        # filter out non-indices (indices for which one value was selected)
        indices = {k: v for k, v in indices.iteritems()
                   if isinstance(v, pd.Index)}
        dimensions = OrderedDict((k, indices[k].size) for k in self.dimensions
                                 if k in indices)
        return type(self)(variables, dimensions, self.attributes,
                          indices=indices)

    def _loc_to_int_indexer(self, dim, locations):
        index = self.indices[dim]
        if isinstance(locations, slice):
            tmp_slice = index.slice_indexer(locations.start, locations.stop)
            # assume step-size is valid unchanged
            indexer = slice(tmp_slice.start, tmp_slice.stop, locations.step)
        else:
            try:
                indexer = index.get_loc(locations)
            except TypeError:
                # value is an list or array
                new_index, indexer = index.reindex(np.asarray(locations))
                if np.any(indexer < 0):
                    raise ValueError('not all values found in index %r' % dim)
                # FIXME: don't throw away new_index (we'll need to recreate it
                # later)
        return indexer

    def loc_views(self, **slicers):
        return self.views(**{k: self._loc_to_int_indexer(k, v)
                             for k, v in slicers.iteritems()})

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
            variables[name] = Variable(dims, v._data, v.attributes)

        dimensions = OrderedDict((name_dict.get(k, k), v)
                                 for k, v in self.dimensions.iteritems())
        indices = {name_dict.get(k, k): v
                   for k, v in self.indices.cache.items()}
        return type(self)(variables, dimensions, self.attributes,
                          indices=indices)

    def merge(self, other, inplace=False):
        """Merge two datasets into a single new dataset

        This method generally not allow for overriding data. Variables,
        dimensions and indices are checked for conflicts. However, conflicting
        attributes are removed.

        Parameters
        ----------
        other : Dataset
            Dataset to merge with this dataset.
        inplace : bool, optional
            If True, merge the other dataset into this dataset in-place.

        Returns
        -------
        merged : Dataset
            Merged dataset.

        Raises
        ------
        ValueError
            If any variables or dimensions conflict. Conflicting attributes
            are silently dropped.
        """
        # check for conflicts
        utils.update_safety_check(self.variables, other.variables,
                                  compat=utils.variable_equal)
        utils.update_safety_check(self.dimensions, other.dimensions)
        utils.update_safety_check(self.indices.cache, other.indices.cache,
                                  compat=np.array_equal)
        # update contents
        obj = self if inplace else self.copy()
        obj.store.set_variables(other.variables)
        obj.store.set_dimensions(other.dimensions)
        obj._indices.update(other.indices.cache)
        # remove conflicting attributes
        for k, v in other.attributes.iteritems():
            if k in self.attributes and not v != self.attributes[k]:
                obj.store.del_attribute(k)
        return obj

    def select(self, *names):
        """Returns a new dataset that contains the named variables

        Dimensions on which those variables are defined are also included, as
        well as the corresponding coordinate variables, and any variables
        listed under the 'coordinates' attribute of the named variables.

        Parameters
        ----------
        *names : str
            Names of the variables to include in the returned object.

        Returns
        -------
        Dataset
            The returned object has the same attributes as the original. A
            dimension is included if at least one of the specified variables is
            defined along that dimension. Coordinate variables (1-dimensional
            variables with the same name as a dimension) that correspond to an
            included dimension are also included. All other variables are
            dropped.
        """
        if not all(k in self.variables for k in names):
            raise ValueError(
                "One or more of the specified variables does not exist")

        def get_aux_names(var):
            names = set(var.dimensions)
            if 'coordinates' in var.attributes:
                coords = var.attributes['coordinates']
                if coords != '':
                    names |= set(coords.split(' '))
            return names

        aux_names = [get_aux_names(self.variables[k]) for k in names]
        names = set(names).union(*aux_names)

        variables = OrderedDict((k, v) for k, v in self.variables.iteritems()
                                if k in names)
        dimensions = OrderedDict((k, v) for k, v in self.dimensions.iteritems()
                                 if k in names)
        indices = {k: v for k, v in self.indices.cache.items() if k in names}
        return type(self)(variables, dimensions, self.attributes,
                          indices=indices)

    def unselect(self, *names, **kwargs):
        """Returns a new dataset without the named variables

        Parameters
        ----------
        *names : str
            Names of the variables to omit from the returned object.
        omit_dimensions : bool, optional (default True)
            Whether or not to also omit dimensions with the given names.

        Returns
        -------
        Dataset
            New dataset based on this dataset. Only the named variables
            /dimensions are removed.
        """
        if any(k not in self.variables and k not in self.dimensions
               for k in names):
            raise ValueError('One or more of the specified variable/dimension '
                             'names does not exist on this dataset')
        variables = OrderedDict((k, v) for k, v in self.variables.iteritems()
                                if k not in names)
        if kwargs.get('omit_dimensions', True):
            dimensions = OrderedDict((k, v) for k, v
                                     in self.dimensions.iteritems()
                                     if k not in names)
            indices = {k: v for k, v in self.indices.cache.items()
                       if k not in names}
        else:
            dimensions = self.dimensions
            indices = self.indices
        return type(self)(variables, dimensions, self.attributes,
                          indices=indices)

    def replace(self, name, variable):
        """Returns a new dataset with the variable 'name' replaced with
        'variable'

        Parameters
        ----------
        name : str
            Name of the variable to replace in this object.
        variable : Variable
            Replacement variable.

        Returns
        -------
        Dataset
            New dataset based on this dataset. Dimensions are unchanged.
        """
        ds = self.unselect(name, omit_dimensions=False)
        ds.add_variable(name, variable)
        return ds

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
        of the underlying data of a variable.

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

