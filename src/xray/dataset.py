import numpy as np
import netCDF4 as nc4
import pandas as pd

from cStringIO import StringIO
from collections import OrderedDict, Mapping

import xarray
import backends
import conventions
import common
import groupby
import utils
from dataset_array import DataArray
from utils import (FrozenOrderedDict, Frozen, remap_loc_indexers,
                   multi_index_from_product)

date2num = nc4.date2num
num2date = nc4.num2date


def open_dataset(nc, decode_cf=True, *args, **kwargs):
    """Open the dataset given the object or path `nc`.

    *args and **kwargs provide format specific options
    """
    # move this to a classmethod Dataset.open?
    # TODO: this check has the unfortunate side-effect that
    # paths to files cannot start with 'CDF'.
    if isinstance(nc, basestring) and not nc.startswith('CDF'):
        # If the initialization nc is a string and it doesn't
        # appear to be the contents of a netcdf file we load
        # it using the netCDF4 package
        store = backends.NetCDF4DataStore(nc, *args, **kwargs)
    else:
        # If nc is a file-like object we read it using
        # the scipy.io.netcdf package
        store = backends.ScipyDataStore(nc, *args, **kwargs)
    return Dataset.load_store(store, decode_cf=decode_cf)


# list of attributes of pd.DatetimeIndex that are ndarrays of time info
_DATETIMEINDEX_COMPONENTS = ['year', 'month', 'day', 'hour', 'minute',
                             'second', 'microsecond', 'nanosecond', 'date',
                             'time', 'dayofyear', 'weekofyear', 'dayofweek',
                             'quarter']


class _VariablesDict(OrderedDict):
    """_VariablesDict is an OrderedDict subclass that also implements "virtual"
    variables that are created from other variables on demand

    Currently, virtual variables are restricted to attributes of
    pandas.DatetimeIndex objects (e.g., 'year', 'month', 'day', etc., plus
    'season' for climatological season), which are accessed by getting the item
    'time.year'.
    """
    def _datetimeindices(self):
        return [k for k, v in self.iteritems()
                if np.issubdtype(v.dtype, np.datetime64) and v.ndim == 1
                and isinstance(v.index, pd.DatetimeIndex)]

    @property
    def virtual(self):
        """Variables that don't exist in this dataset but for which could be
        created on demand (because they can be calculated from other dataset
        variables)
        """
        virtual_vars = []
        for k in self._datetimeindices():
            for suffix in _DATETIMEINDEX_COMPONENTS + ['season']:
                name = '%s.%s' % (k, suffix)
                if name not in self:
                    virtual_vars.append(name)
        return virtual_vars

    def _get_virtual_variable(self, key):
        split_key = key.split('.')
        if len(split_key) == 2:
            ref_var, suffix = split_key
            if ref_var in self._datetimeindices():
                if suffix == 'season':
                    # seasons = np.array(['DJF', 'MAM', 'JJA', 'SON'])
                    month = self[ref_var].index.month
                    data = (month // 3) % 4 + 1
                else:
                    data = getattr(self[ref_var].index, suffix)
                return xarray.XArray(self[ref_var].dimensions, data)
        raise KeyError('virtual variable %r not found' % key)

    def __getitem__(self, key):
        if key in self:
            return OrderedDict.__getitem__(self, key)
        elif key in self.virtual:
            return self._get_virtual_variable(key)
        else:
            raise KeyError(repr(key))


class Dataset(Mapping):
    """A netcdf-like data object consisting of variables and attributes which
    together form a self describing dataset.

    Dataset implements the mapping interface with keys given by variable names
    and values given by DataArray objects for each variable name.

    Note: the size of dimensions in a dataset cannot be changed.
    """
    def __init__(self, variables=None, attributes=None, decode_cf=False):
        """To load data from a file or file-like object, use the `open_dataset`
        function.

        Parameters
        ----------
        variables : dict-like, optional
            A mapping from variable names to `XArray` objects or sequences of
            the form `(dimensions, data[, attributes])` which can be used as
            arguments to create a new `XArray`. Each dimension must have the
            same length in all variables in which it appears. One dimensional
            variables with name equal to their dimension are coordinate
            variables, which means they are saved in the dataset as
            `pandas.Index` objects.
        attributes : dict-like, optional
            Global attributes to save on this dataset.
        decode_cf : bool, optional
            Whether to decode these variables according to CF conventions.
        """
        self._variables = _VariablesDict()
        self._dimensions = OrderedDict()
        if variables is not None:
            self.set_variables(variables, decode_cf=decode_cf)
        if attributes is None:
            attributes = {}
        self._attributes = OrderedDict(attributes)

    def _as_variable(self, name, var, decode_cf=False):
        try:
            var = xarray.as_xarray(var)
        except TypeError:
            raise TypeError('Dataset variables must be of type '
                            'DataArray or XArray, or a sequence of the '
                            'form (dimensions, data[, attributes, encoding])')
        # this will unmask and rescale the data as well as convert
        # time variables to datetime indices.
        if decode_cf:
            var = conventions.decode_cf_variable(var)
        if name in var.dimensions:
            # convert the coordinate into a pandas.Index
            if var.ndim != 1:
                raise ValueError('a coordinate variable must be defined with '
                                 '1-dimensional data')
            var = var.to_coord()
        return var

    def set_variables(self, variables, decode_cf=False):
        """Set a mapping of variables and update dimensions.

        Parameters
        ----------
        variables : dict-like, optional
            A mapping from variable names to `XArray` objects or sequences of
            the form `(dimensions, data[, attributes])` which can be used as
            arguments to create a new `XArray`. Each dimension must have the
            same length in all variables in which it appears. One dimensional
            variables with name equal to their dimension are coordinate
            variables, which means they are saved in the dataset as
            `pandas.Index` objects.
        decode_cf : bool, optional
            Whether to decode these variables according to CF conventions.

        Returns
        -------
        None
        """
        # save new variables into a temporary list so all the error checking
        # can be done before updating _variables
        new_variables = []
        for k, var in variables.iteritems():
            var = self._as_variable(k, var, decode_cf=decode_cf)
            for dim, size in zip(var.dimensions, var.shape):
                if dim not in self._dimensions:
                    self._dimensions[dim] = size
                    if dim not in variables and dim not in self._variables:
                        coord = self._as_variable(dim, (dim, np.arange(size)))
                        new_variables.append((dim, coord))
                elif self._dimensions[dim] != size:
                    raise ValueError('dimension %r on variable %r has size %s '
                                     'but already is saved with size %s' %
                                     (dim, k, size, self._dimensions[dim]))
            new_variables.append((k, var))
        self._variables.update(new_variables)

    @classmethod
    def load_store(cls, store, decode_cf=True):
        return cls(store.variables, store.attributes, decode_cf=decode_cf)

    @property
    def variables(self):
        """Dictionary of XArray objects contained in this dataset.

        This dictionary is frozen to prevent it from being modified in ways
        that would cause invalid dataset metadata (e.g., by setting variables with
        inconsistent dimensions). Instead, add or remove variables by
        acccessing the dataset directly (e.g., `dataset['foo'] = bar` or
        `del dataset['foo']`).
        """
        return Frozen(self._variables)

    @property
    def attributes(self):
        """Dictionary of global attributes on this dataset
        """
        return self._attributes

    @attributes.setter
    def attributes(self, value):
        self._attributes = OrderedDict(value)

    @property
    def dimensions(self):
        """Mapping from dimension names to lengths.

        This dictionary cannot be modified directly, but is updated when adding
        new variables.
        """
        return Frozen(self._dimensions)

    def copy(self, deep=False):
        """Returns a copy of this dataset.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy is made, so each variable in the new dataset
        is also a variable in the original dataset.
        """
        if deep:
            variables = OrderedDict((k, v.copy(deep=True))
                                    for k, v in self.variables.iteritems())
        else:
            variables = self.variables
        return type(self)(variables, self.attributes)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        # memo does nothing but is required for compatability with
        # copy.deepcopy
        return self.copy(deep=True)

    def __contains__(self, key):
        """The 'in' operator will return true or false depending on whether
        'key' is a variable in the dataset or not.
        """
        return key in self.variables

    def __len__(self):
        return len(self.variables)

    def __iter__(self):
        return iter(self.variables)

    @property
    def virtual_variables(self):
        """Variables that do not exist in this dataset but could be created on
        demand.

        These variables can be derived by performing simple operations on
        existing dataset variables. Currently, the only implemented virtual
        variables are time/date components [1_] such as "time.month" or
        "time.dayofyear", where "time" is the name of a coordinate whose data
        is a `pandas.DatetimeIndex` object. The virtual variable "time.season"
        (for climatological season, starting with 1 for "DJF") is the only such
        variable which is not directly implemented in pandas.

        References
        ----------
        .. [1] http://pandas.pydata.org/pandas-docs/stable/api.html#time-date-components
        """
        return self._variables.virtual

    def __getitem__(self, key):
        """Access the given variable name in this dataset as a `DataArray`.
        """
        return DataArray(self, key)

    def __setitem__(self, key, value):
        """Add an array to this dataset.

        If value is a `DataArray`, call its `select()` method, rename it to
        `key` and merge the contents of the resulting dataset into this
        dataset.

        If value is an `XArray` object (or tuple of form
        `(dimensions, data[, attributes])`), add it to this dataset as a new
        variable.
        """
        if isinstance(value, DataArray):
            self.merge(value.rename(key).select().dataset, inplace=True,
                       overwrite_vars=[key])
        else:
            self.set_variables({key: value})

    def __delitem__(self, key):
        """Remove a variable from this dataset.

        If this variable is a dimension, all variables containing this
        dimension are also removed.
        """
        if key in self._dimensions:
            del self._dimensions[key]
        del self._variables[key]
        also_delete = [k for k, v in self._variables.iteritems()
                       if key in v.dimensions]
        for key in also_delete:
            del self._variables[key]

    # mutable objects should not be hashable
    __hash__ = None

    def __eq__(self, other):
        """Two Datasets are equal if they have equal variables and global
        attributes.
        """
        try:
            # some stores (e.g., scipy) do not seem to preserve order, so don't
            # require matching dimension or variable order for equality
            return (utils.dict_equal(self.attributes, other.attributes)
                    and all(k1 == k2 and utils.xarray_equal(v1, v2)
                            for (k1, v1), (k2, v2)
                            in zip(sorted(self.variables.items()),
                                   sorted(other.variables.items()))))
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    @property
    def coordinates(self):
        """Coordinates are variables with names that match dimensions.

        They are always stored internally as `XArray` objects with data that is
        a `pandas.Index` object.
        """
        return FrozenOrderedDict([(dim, self.variables[dim])
                                  for dim in self.dimensions])

    @property
    def noncoordinates(self):
        """Non-coordinates are variables with names that do not match
        dimensions.
        """
        return FrozenOrderedDict([(name, v)
                for (name, v) in self.variables.iteritems()
                if name not in self.dimensions])

    def dump_to_store(self, store):
        """Store dataset contents to a backends.*DataStore object."""
        store.set_variables(self.variables)
        store.set_attributes(self.attributes)
        store.sync()

    def dump(self, filepath, **kwdargs):
        """Dump dataset contents to a location on disk using the netCDF4
        package.
        """
        nc4_store = backends.NetCDF4DataStore(filepath, mode='w', **kwdargs)
        self.dump_to_store(nc4_store)

    def dumps(self, **kwargs):
        """Serialize dataset contents to a string. The serialization creates an
        in memory netcdf version 3 string using the scipy.io.netcdf package.
        """
        fobj = StringIO()
        scipy_store = backends.ScipyDataStore(fobj, mode='w', **kwargs)
        self.dump_to_store(scipy_store)
        return fobj.getvalue()

    def __repr__(self):
        return common.dataset_repr(self)

    def indexed_by(self, **indexers):
        """Return a new dataset with each array indexed along the specified
        dimension(s).

        This method selects values from each array using its `__getitem__`
        method, except this method does not require knowing the order of
        each array's dimensions.

        Parameters
        ----------
        **indexers : {dim: indexer, ...}
            Keyword arguments with names matching dimensions and values given
            by integers, slice objects or arrays.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array and dimension is indexed by the appropriate indexers. In
            general, each array's data will be a view of the array's data
            in this dataset, unless numpy fancy indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        See Also
        --------
        Dataset.labeled_by
        Dataset.indexed_by
        Array.indexed_by
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
            new_var = var.indexed_by(**var_indexers)
            if new_var.ndim > 0:
                # filter out variables reduced to numbers
                variables[name] = new_var

        return type(self)(variables, self.attributes)

    def labeled_by(self, **indexers):
        """Return a new dataset with each variable indexed by coordinate labels
        along the specified dimension(s).

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
        Array.indexed_by
        """
        return self.indexed_by(**remap_loc_indexers(self, indexers))

    def reindex_like(self, other, copy=True):
        """Conform this object onto the coordinates of another object, filling
        in missing values with NaN.

        Parameters
        ----------
        other : Dataset or DatasetArray
            Object with a coordinates attribute giving a mapping from dimension
            names to xray.XArray objects, which provides coordinates upon which
            to index the variables in this dataset. The coordinates on this
            other object need not be the same as the coordinates on this
            dataset. Any mis-matched coordinates values will be filled in with
            NaN, and any mis-matched coordinate names will simply be ignored.
        copy : bool, optional
            If `copy=True`, the returned dataset contains only copied
            variables. If `copy=False` and no reindexing is required then
            original variables from this dataset are returned.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with coordinates replaced from the other object.

        See Also
        --------
        Dataset.reindex
        align
        """
        return self.reindex(copy=copy, **other.coordinates)

    def reindex(self, copy=True, **coordinates):
        """Conform this object onto a new set of coordinates or pandas.Index
        objects, filling in missing values with NaN.

        Parameters
        ----------
        copy : bool, optional
            If `copy=True`, the returned dataset contains only copied
            variables. If `copy=False` and no reindexing is required then
            original variables from this dataset are returned.
        **coordinates : dict
            Dictionary with keys given by dimension names and values given by
            arrays of coordinate labels. Any mis-matched coordinates values
            will be filled in with NaN, and any mis-matched coordinate names
            will simply be ignored.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with replaced coordinates.

        See Also
        --------
        Dataset.reindex_like
        align
        """
        if not coordinates:
            # shortcut
            return self.copy(deep=True) if copy else self

        # build up indexers for assignment along each coordinate
        to_indexers = {}
        from_indexers = {}
        for name, coord in self.coordinates.iteritems():
            if name in coordinates:
                index = coordinates[name]
                if hasattr(index, 'index'):
                    index = index.index
                indexer = coord.index.get_indexer(index)

                # Note pandas uses negative values from get_indexer to signify
                # values that are missing in the index
                # The non-negative values thus indicate the non-missing values
                to_indexers[name] = indexer >= 0
                if to_indexers[name].all():
                    # If an indexer includes no negative values, then the
                    # assignment can be to a full-slice (which is much faster,
                    # and means we won't need to fill in any missing values)
                    to_indexers[name] = slice(None)

                from_indexers[name] = indexer[to_indexers[name]]
                if np.array_equal(from_indexers[name], np.arange(coord.size)):
                    # If the indexer is equal to the original index, use a full
                    # slice object to speed up selection and so we can avoid
                    # unnecessary copies
                    from_indexers[name] = slice(None)

        def is_full_slice(idx):
            return isinstance(idx, slice) and idx == slice(None)

        def any_not_full_slices(indexers):
            return any(not is_full_slice(idx) for idx in indexers)

        def var_indexers(var, indexers):
            return tuple(indexers.get(d, slice(None)) for d in var.dimensions)

        def get_fill_value_and_dtype(dtype):
            if np.issubdtype(dtype, np.datetime64):
                fill_value = np.datetime64('NaT')
            elif any(np.issubdtype(dtype, t) for t in (int, float)):
                # convert to floating point so NaN is valid
                dtype = float
                fill_value = np.nan
            elif any(np.issubdtype(dtype, t) for t in (str, unicode)):
                # TODO: consider eliminating this case to better align behavior
                # with pandas (which upcasts strings to object arrays and
                # inserts NaN for missing values)
                fill_value = 'NA'
            else:
                dtype = object
                fill_value = np.nan
            return fill_value, dtype

        # create variables for the new dataset
        variables = OrderedDict()
        for name, var in self.variables.iteritems():
            if name in coordinates:
                new_var = coordinates[name]
                if not (hasattr(new_var, 'dimensions') and
                        hasattr(new_var, 'data')):
                    new_var = xarray.CoordXArray(var.dimensions, new_var,
                                                 var.attributes, var.encoding)
                elif copy:
                    new_var = xarray.as_xarray(new_var).copy()
            else:
                # TODO: Resolve edge cases where variables are not found in
                # coordinates, but actually take on coordinates values
                # associated with a variable in the second dataset. An example
                # would be latitude and longitude if the dataset has some non-
                # Mercator projection. For these variables, we probably want to
                # replace them with the other coordinates unchanged (if
                # possible) instead of filling these coordinate values in with
                # NaN.

                assign_to = var_indexers(var, to_indexers)
                assign_from = var_indexers(var, from_indexers)

                if any_not_full_slices(assign_to):
                    # there are missing values to in-fill
                    fill_value, dtype = get_fill_value_and_dtype(var.dtype)
                    shape = tuple(length if is_full_slice(idx) else idx.size
                                  for idx, length in zip(assign_to, var.shape))
                    data = np.empty(shape, dtype=dtype)
                    data[:] = fill_value
                    # create a new XArray so we can use orthogonal indexing
                    new_var = xarray.XArray(var.dimensions, data, var.attributes)
                    new_var[assign_to] = var[assign_from].data
                elif any_not_full_slices(assign_from):
                    # type coercion is not necessary as there are no missing
                    # values
                    new_var = var[assign_from]
                else:
                    # no reindexing is necessary
                    # here we need to manually deal with copying data, since
                    # we neither created a new ndarray nor used fancy indexing
                    new_var = var.copy() if copy else var
            variables[name] = new_var
        return type(self)(variables, self.attributes)

    def rename(self, name_dict):
        """Returns a new object with renamed variables and dimensions.

        Parameters
        ----------
        name_dict : dict-like
            Dictionary whose keys are current variable or dimension names and
            whose values are new names.

        Returns
        -------
        renamed : Dataset
            Dataset with renamed variables and dimensions.
        """
        for k in name_dict:
            if k not in self.variables:
                raise ValueError("Cannot rename %r because it is not a "
                                 "variable in this dataset" % k)
        variables = OrderedDict()
        for k, v in self.variables.iteritems():
            name = name_dict.get(k, k)
            dims = tuple(name_dict.get(dim, dim) for dim in v.dimensions)
            var = v.copy(deep=False)
            var.dimensions = dims
            variables[name] = var
        return type(self)(variables, self.attributes)

    def merge(self, other, inplace=False, overwrite_vars=None,
              attribute_conflicts='ignore'):
        """Merge two datasets into a single new dataset.

        This method generally not allow for overriding data. Arrays,
        dimensions and indices are checked for conflicts. However, conflicting
        attributes are removed.

        Parameters
        ----------
        other : Dataset
            Dataset to merge with this dataset.
        inplace : bool, optional
            If True, merge the other dataset into this dataset in-place.
        overwrite_vars : list, optional
            If provided, update variables of these names without checking for
            conflicts in this dataset.
        attribute_conflicts : str, optional
            How to handle attribute conflicts on datasets and variables. The
            only currently supported option is 'ignore'.

        Returns
        -------
        merged : Dataset
            Merged dataset.

        Raises
        ------
        ValueError
            If any variables or dimensions conflict. Conflicting attributes
            are silently dropped.

        Warning
        -------
        The current interface and defaults for handling for conflicting
        attributes is not ideal and very preliminary. Expect this behavior to
        change in future pre-release versions of xray. See the discussion
        on GitHub: https://github.com/akleeman/xray/issues/25
        """
        if attribute_conflicts != 'ignore':
            raise NotImplementedError

        # check for conflicts
        if overwrite_vars is None:
            overwrite_vars = {}
        for k, v in other.variables.iteritems():
            if (k in self and k not in overwrite_vars
                    and not utils.xarray_equal(v, self[k],
                                               check_attributes=False)):
                raise ValueError('unsafe to merge datasets; '
                                 'conflicting variable %r' % k)
        # update contents
        obj = self if inplace else self.copy()
        obj.set_variables(OrderedDict((k, v) for k, v
                                      in other.variables.iteritems()
                                      if k not in obj.variables
                                      or k in overwrite_vars))
        # remove conflicting attributes
        for k, v in other.attributes.iteritems():
            if k in self.attributes and v != self.attributes[k]:
                del self.attributes[k]
        return obj

    def select(self, *names):
        """Returns a new dataset that contains the named variables.

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
            The returned object has the same attributes as the original.
            Variables are included (recursively) if at least one of the
            specified variables refers to that variable in its dimensions or
            "coordinates" attribute. All other variables are dropped.
        """
        possible_vars = set(self) | set(self.virtual_variables)
        if not set(names) <= possible_vars:
            raise ValueError(
                "One or more of the specified variables does not exist")

        def get_all_associated_names(name):
            yield name
            if name in possible_vars:
                var = self.variables[name]
                for dim in var.dimensions:
                    yield dim
                if 'coordinates' in var.attributes:
                    coords = var.attributes['coordinates']
                    if coords != '':
                        for coord in coords.split(' '):
                            yield coord

        queue = set(names)
        selected_names = set()
        while queue:
            name = queue.pop()
            new_names = set(get_all_associated_names(name))
            queue |= new_names - selected_names
            selected_names |= new_names

        variables = OrderedDict((k, self.variables[k])
                                for k in list(self) + self.virtual_variables
                                if k in selected_names)
        return type(self)(variables, self.attributes)

    def unselect(self, *names):
        """Returns a new dataset without the named variables

        Parameters
        ----------
        *names : str
            Names of the variables to omit from the returned object.

        Returns
        -------
        Dataset
            New dataset based on this dataset. Only the named variables are
            removed.
        """
        if any(k not in self.variables and k not in self.virtual_variables
               for k in names):
            raise ValueError('One or more of the specified variable '
                             'names does not exist on this dataset')
        drop = set(names)
        drop |= {k for k, v in self.variables.iteritems()
                 if any(name in v.dimensions for name in names)}
        variables = OrderedDict((k, v) for k, v in self.variables.iteritems()
                                if k not in drop)
        return type(self)(variables, self.attributes)

    def replace(self, name, variable):
        """Returns a new dataset with the variable 'name' replaced with
        'variable'.

        Parameters
        ----------
        name : str
            Name of the variable to replace in this object.
        variable : Array
            Replacement variable.

        Returns
        -------
        Dataset
            New dataset based on this dataset. Dimensions are unchanged.
        """
        ds = self.copy()
        ds[name] = variable
        return ds

    def groupby(self, group, squeeze=True):
        """Group this dataset by unique values of the indicated group.

        Parameters
        ----------
        group : str or DataArray
            Array whose unique values should be used to group this array. If a
            string, must be the name of a variable contained in this dataset.
        squeeze : boolean, optional
            If "group" is a coordinate of this array, `squeeze` controls
            whether the subarrays have a dimension of length 1 along that
            coordinate or if the dimension is squeezed out.

        Returns
        -------
        grouped : GroupBy
            A `GroupBy` object patterned after `pandas.GroupBy` that can be
            iterated over in the form of `(unique_value, grouped_array)` pairs.
        """
        if isinstance(group, basestring):
            # merge in the group's dataset to allow group to be a virtual
            # variable in this dataset
            ds = self.merge(self[group].dataset)
            group = DataArray(ds, group)
        return groupby.GroupBy(self, group.name, group, squeeze=squeeze)

    def squeeze(self, dimension=None):
        """Return a new dataset with squeezed data.

        Parameters
        ----------
        dimension : None or str or tuple of str, optional
            Selects a subset of the length one dimensions. If a dimension is
            selected with length greater than one, an error is raised.  If
            None, all length one dimensions are squeezed.

        Returns
        -------
        squeezed : Dataset
            This dataset, but with with all or a subset of the dimensions of
            length 1 removed.

        Notes
        -----
        Although this operation returns a view of each variable's data, it is
        not lazy -- all variable data will be fully loaded.

        See Also
        --------
        numpy.squeeze
        """
        if dimension is None:
            dimension = [d for d, s in self.dimensions.iteritems() if s == 1]
        else:
            if isinstance(dimension, basestring):
                dimension = [dimension]
            if any(self.dimensions[k] > 1 for k in dimension):
                raise ValueError('cannot select a dimension to squeeze out '
                                 'which has length greater than one')
        return self.indexed_by(**{dim: 0 for dim in dimension})

    def _add_dimension(self, dimension):
        """Given a dimension (string or Array), add the dimension to this
        dataset (is it is an array) and return its name.

        N.B. This function modifies the dataset in-place.
        """
        if isinstance(dimension, basestring):
            dim_name = dimension
        else:
            dim_name, = dimension.dimensions
            if isinstance(dimension, DataArray):
                self.merge(dimension._unselect_nonfocus_dims().dataset,
                           inplace=True)
            else:
                self[dim_name] = dimension
        return dim_name

    @classmethod
    def concat(cls, datasets, dimension='concat_dimension', indexers=None,
               concat_over=None):
        """Concatenate datasets along a new or existing dimension.

        Parameters
        ----------
        datasets : iterable of Dataset
            Datasets to stack together. Each dataset is expected to have
            matching attributes, and all variables except those along the
            stacked dimension (those that contain "dimension" as a dimension or
            are listed in "concat_over") are expected to be equal.
        dimension : str or Array, optional
            Name of the dimension to stack along. If dimension is provided as
            an XArray or DataArray, the focus of the dataset array or the
            singleton dimension of the xarray is used as the stacking dimension
            and the array is added to the returned dataset.
        indexers : None or iterable of indexers, optional
            Iterable of indexers of the same length as variables which
            specifies how to assign variables from each dataset along the given
            dimension. If not supplied, indexers is inferred from the length of
            each variable along the dimension, and the variables are stacked in
            the given order.
        concat_over : None or str or iterable of str, optional
            Names of additional variables to concatenate, in which "dimension"
            does not already appear as a dimension.

        Returns
        -------
        concatenated : Dataset
            Concatenated dataset formed by concatenating dataset variables.
        """
        # don't bother trying to work with datasets as a generator instead of a
        # list; the gains would be minimal
        datasets = list(datasets)
        if not datasets:
            raise ValueError('datasets cannot be empty')
        # create the new dataset and add non-concatenated variables
        concatenated = cls({}, datasets[0].attributes)
        dim_name = concatenated._add_dimension(dimension)
        if concat_over is None:
            concat_over = set()
        else:
            if isinstance(concat_over, basestring):
                concat_over = {concat_over}
            concat_over = set(concat_over)
            if any(v not in datasets[0] for v in concat_over):
                raise ValueError('not all elements in concat_over %r found '
                                 'in the first dataset %r'
                                 % (concat_over, datasets[0]))
        for k, v in datasets[0].variables.iteritems():
            if k == dim_name or dim_name in v.dimensions:
                if k not in concatenated:
                    # don't concat over dim_name if it was provided as an array
                    concat_over.add(k)
            elif k not in concat_over:
                concatenated[k] = v
        # check that global attributes and non-concatenated variables are fixed
        # across all datasets
        for ds in datasets[1:]:
            if not utils.dict_equal(ds.attributes, concatenated.attributes):
                raise ValueError('dataset global attributes not equal')
            for k, v in ds.variables.iteritems():
                if k not in concatenated and k not in concat_over:
                    raise ValueError('encountered unexpected variable %r' % k)
                elif (k in concatenated and k != dim_name and
                          not utils.xarray_equal(v, concatenated[k])):
                    raise ValueError(
                        'variable %r not equal across datasets' % k)
        # stack up each variable to fill-out the dataset
        for k in concat_over:
            concatenated[k] = xarray.XArray.concat(
                [ds[k] for ds in datasets], dim_name, indexers)
        return concatenated

    def to_dataframe(self):
        """Convert this dataset into a pandas.DataFrame.

        Non-coordinate variables in this dataset form the columns of the
        DataFrame. The DataFrame is be indexed by the Cartesian product of
        this dataset's indices.
        """
        columns = self.noncoordinates.keys()
        data = []
        # we need a template to broadcast all dataset variables against
        # using stride_tricks lets us make the ndarray for broadcasting without
        # having to allocate memory
        shape = tuple(self.dimensions.values())
        empty_data = np.lib.stride_tricks.as_strided(np.array(0), shape=shape,
                                                     strides=[0] * len(shape))
        template = xarray.XArray(self.dimensions.keys(), empty_data)
        for k in columns:
            _, var = xarray.broadcast_xarrays(template, self.variables[k])
            _, var_data = np.broadcast_arrays(template.data, var.data)
            data.append(var_data.reshape(-1))

        index = multi_index_from_product(self.coordinates.values(),
                                         names=self.coordinates.keys())
        return pd.DataFrame(OrderedDict(zip(columns, data)), index=index)

    @classmethod
    def from_dataframe(cls, dataframe):
        """Convert a pandas.DataFrame into an xray.Dataset

        Each column will be converted into an independent variable in the
        Dataset. If the dataframe's index is a MultiIndex, it will be expanded
        into a tensor product of one-dimensional indices (filling in missing
        values with NaN). This method will produce a Dataset very similar to
        that on which the 'to_dataframe' method was called, except with
        possibly redundant dimensions (since all dataset variables will have
        the same dimensionality).
        """
        # TODO: Add an option to remove dimensions along which the variables
        # are constant, to enable consistent serialization to/from a dataframe,
        # even if some variables have different dimensionality.

        idx = dataframe.index
        obj = cls()

        if hasattr(idx, 'levels'):
            # it's a multi-index
            # expand the DataFrame to include the product of all levels
            full_idx = multi_index_from_product(idx.levels, idx.names)
            dataframe = dataframe.reindex(full_idx)
            dimensions = [name if name is not None else 'level_%i' % n
                          for n, name in enumerate(idx.names)]
            for dim, lev in zip(dimensions, idx.levels):
                obj[dim] = (dim, lev)
            shape = [lev.size for lev in idx.levels]
        else:
            dimensions = (idx.name if idx.name is not None else 'index',)
            obj[dimensions[0]] = (dimensions, idx)
            shape = -1

        for name, series in dataframe.iteritems():
            data = series.values.reshape(shape)
            obj[name] = (dimensions, data)
        return obj
