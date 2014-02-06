# TODO Use various backend data stores. pytable, ncdf4, scipy.io, iris, memory
import os
import numpy as np
import netCDF4 as nc4
import pandas as pd

from cStringIO import StringIO
from collections import OrderedDict, Mapping, MutableMapping

from dataview import DataView
from utils import FrozenOrderedDict, Frozen
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
        if k in var.dimensions and var.ndim != 1:
            raise ValueError('a coordinate variable must be defined with '
                             '1-dimensional data')
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
                                 '%s but on the dataset has length %s' %
                                 (dim, k, length, dimensions[dim]))


def open_dataset(nc, *args, **kwargs):
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
        return key in self.dataset.dimensions

    def __repr__(self):
        contents = '\n'.join("'%s': %s" %
                             (k, str(v).replace(
                                  '\n', '\n' + ' ' * (len(k) + 4)))
                             for k, v in self.items())
        return ("<class 'scidata.data.%s' (dict-like)>\n%s"
                % (type(self).__name__, contents))


# list of attributes of pd.DatetimeIndex that are ndarrays of time info
_DATETIMEINDEX_COMPONENTS = ['year', 'month', 'day', 'hour', 'minute',
                             'second', 'microsecond', 'nanosecond', 'date',
                             'time', 'dayofyear', 'weekofyear', 'dayofweek',
                             'quarter']


class Dataset(Mapping):
    """A netcdf-like data object consisting of dimensions, variables and
    attributes which together form a self describing data set

    Datasets are mappings from variable names to dataviews focused on those
    variable.

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
        Don't modify the store directly unless you want to avoid all validation
        checks.
    """
    def __init__(self, variables=None, dimensions=None, attributes=None,
                 indices=None, store=None):
        """
        If dimensions are not provided, they are inferred from the variables.

        In general, load data from a store using the `open_dataset` function or
        the `from_store` class method. The `store` argument should only be used
        if you want to Dataset operations to modify stored data in-place.
        Note, however, that modifying datasets in-place is not entirely
        implemented and thus may lead to unexpected behavior.
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
        return Frozen(self.store.variables)

    @property
    def attributes(self):
        return Frozen(self.store.attributes)

    @property
    def dimensions(self):
        return Frozen(self.store.dimensions)

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

    def __len__(self):
        return len(self.variable)

    def __iter__(self):
        return iter(self.variables)

    @property
    def _datetimeindices(self):
        return [k for k, v in self.indices.iteritems()
                if isinstance(v, pd.DatetimeIndex)]

    def _get_virtual_variable(self, key):
        if key in self.indices:
            return Variable([key], self.indices[key].values)
        split_key = key.split('.')
        if len(split_key) == 2:
            var, suffix = split_key
            if var in self._datetimeindices:
                if suffix in _DATETIMEINDEX_COMPONENTS:
                    return Variable([var], getattr(self.indices[var], suffix))
                elif suffix == 'season':
                    # seasons = np.array(['DJF', 'MAM', 'JJA', 'SON'])
                    month = self.indices[var].month
                    return Variable([var], (month // 3) % 4 + 1)
        raise ValueError('virtual variable %r not found' % key)

    def _get_virtual_dataview(self, key):
        virtual_var = self._get_virtual_variable(key)
        new_vars = OrderedDict(self.variables.items() + [(key, virtual_var)])
        ds = type(self)(new_vars, self.dimensions, self.attributes,
                        indices=self.indices.cache)
        return DataView(ds, key)

    @property
    def virtual_variables(self):
        """Variables that don't exist in this dataset but for which dataviews
        could be created on demand (because they can be calculated from other
        dataset variables or dimensions)
        """
        possible_vars = list(self.dimensions)
        for k in self._datetimeindices:
            for suffix in _DATETIMEINDEX_COMPONENTS + ['season']:
                possible_vars.append('%s.%s' % (k, suffix))
        return tuple(k for k in possible_vars if k not in self)

    def __getitem__(self, key):
        if key not in self.variables:
            try:
                return self._get_virtual_dataview(key)
            except ValueError:
                raise KeyError('dataset contains no variable with name %r '
                               % key)
        else:
            return DataView(self.select(key), key)

    def __setitem__(self, key, value):
        # TODO: allow this operation to be destructive, overriding existing
        # variables? If so, we may want to implement __delitem__, too.
        # (We would need to change DataView.__setitem__ in that case, because
        # we definitely don't want to override focus variables.)
        if isinstance(value, DataView):
            self.merge(value.renamed(key).dataset, inplace=True)
        elif isinstance(value, Variable):
            self.set_variable(key, value)
        else:
            raise TypeError('only DataViews and Variables can be added to '
                            'datasets via `__setitem__`')

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
            The name of the new variable. An exception will be raised if the
            object already has a variable with this name. If name equals the
            name of a dimension, then the new variable is treated as a
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
        # check old + new dimensions for consistency checks
        new_dims = OrderedDict()
        for dim, length in zip(var.dimensions, var.shape):
            if dim not in self.dimensions:
                new_dims[dim] = length
        check_dims_and_vars_consistency(
            dict(self.dimensions.items() + new_dims.items()),
            {name: var})
        # now set the new dimensions and variables, and rebuild the indices
        self.store.set_dimensions(new_dims)
        new_var = self.store.set_variable(name, var)
        if name in list(self.indices) + list(new_dims):
            self.indices.build_index(name)
        return new_var

    def indexed_by(self, **indexers):
        """Return a new dataset with each variable indexed along the specified
        dimension(s)

        This method selects values from each variable using its `__getitem__`
        method, except this method does not require knowing the order of
        each variable's dimensions.

        Parameters
        ----------
        **indexers : {dim: indexer, ...}
            Keyword arguments with names matching dimensions and values given
            by integers, slice objects or arrays.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            variable and dimension is indexed by the appropriate indexers. In
            general, each variable's data will be a view of the variable's data
            in this dataset, unless numpy fancy indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        See Also
        --------
        Dataset.labeled_by
        Dataset.indexed_by
        Variable.indexed_by
        """
        invalid = [k for k in indexers if not k in self.dimensions]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        # all indexers should be int, slice or np.ndarrays
        indexers = {k: np.asarray(v) if not isinstance(v, (int, slice)) else v
                   for k, v in indexers.iteritems()}

        variables = OrderedDict()
        for name, var in self.variables.iteritems():
            var_indexers = {k: v for k, v in indexers.iteritems()
                           if k in var.dimensions}
            variables[name] = var.indexed_by(**var_indexers)

        indices = {k: (v[indexers[k]] if k in indexers else v)
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
            indexer = index.slice_indexer(locations.start, locations.stop,
                                          locations.step)
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

    def labeled_by(self, **indexers):
        """Return a new dataset with each variable indexed by coordinate labels
        along the specified dimension(s)

        In contrast to `Dataset.indexed_by`, indexers for this method should
        use coordinate values instead of integers.

        Under the hood, this method is powered by using Panda's powerful Index
        objects. This makes label based indexing essentially just as fast as
        using integer indexing.

        It also means this method uses pandas's (well documented) logic for
        indexing. This means you can use string shortcuts for datetime indexes
        (e.g., '2000-01' to select all values in January 2000). It also means
        that slices are treated as inclusive of both the start and stop values,
        unlike normal Python indexing.

        Parameters
        ----------
        **indexers : {dim: indexer, ...}
            Keyword arguments with names matching dimensions and values given
            by individual, slices or arrays of coordinate values.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            variable and dimension is indexed by the appropriate indexers. In
            general, each variable's data will be a view of the variable's data
            in this dataset, unless numpy fancy indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        See Also
        --------
        Dataset.labeled_by
        Dataset.indexed_by
        Variable.indexed_by
        """
        return self.indexed_by(**{k: self._loc_to_int_indexer(k, v)
                                  for k, v in indexers.iteritems()})

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
        utils.update_safety_check(self.noncoordinates, other.noncoordinates,
                                  compat=utils.variable_equal)
        utils.update_safety_check(self.dimensions, other.dimensions)
        # note: coordinates are checked by comparing indices instead of
        # variables, which lets us merge two datasets even if they have
        # different time units
        utils.update_safety_check(self.indices, other.indices,
                                  compat=np.array_equal)
        # update contents
        obj = self if inplace else self.copy()
        obj.store.set_variables(OrderedDict((k, v) for k, v
                                            in other.variables.iteritems()
                                            if k not in obj.variables))
        obj.store.set_dimensions(OrderedDict((k, v) for k, v
                                             in other.dimensions.iteritems()
                                             if k not in obj.dimensions))
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

    def iterator(self, dimension):
        """Iterate along a data dimension

        Returns an iterator yielding (coordinate, dataset) pairs for each
        coordinate value along the specified dimension.

        Parameters
        ----------
        dimension : string
            The dimension along which to iterate.

        Returns
        -------
        it : iterator
            The returned iterator yields pairs of scalar-valued coordinate
            variables and Dataset objects.
        """
        coord = self.variables[dimension]
        for i in xrange(self.dimensions[dimension]):
            yield (coord[i], self.indexed_by(**{dimension: i}))


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

