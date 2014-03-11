import numpy as np
import netCDF4 as nc4
import pandas as pd

from cStringIO import StringIO
from collections import OrderedDict, Mapping

import xarray
import backends
import conventions
import groupby
import utils
from dataset_array import DatasetArray
from utils import FrozenOrderedDict, Frozen, remap_loc_indexers

date2num = nc4.date2num
num2date = nc4.num2date


def open_dataset(nc, decode_cf=True, *args, **kwargs):
    """Open the dataset given the object or path `nc`.

    *args and **kwargs provide format specific options
    """
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
                if np.issubdtype(v.dtype, np.datetime64)
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
    and values given by DatasetArray objects focused on each variable name.

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
        if isinstance(var, DatasetArray):
            var = xarray.as_xarray(var)
        elif not isinstance(var, xarray.XArray):
            try:
                var = xarray.XArray(*var)
            except TypeError:
                raise TypeError('Dataset variables must be of type '
                                'DatasetArray or XArray, or a sequence of the '
                                'form (dimensions, data[, attributes, '
                                'encoding])')
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

    def copy(self):
        """
        Returns a shallow copy of the current object.
        """
        return self.__copy__()

    def __copy__(self):
        """
        Returns a shallow copy of the current object.
        """
        return type(self)(self.variables, self.attributes)

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
        """Access the DatasetArray focused on the given variable name.
        """
        return DatasetArray(self, key)

    def __setitem__(self, key, value):
        """Add an array to this dataset.

        If value is a `DatasetArray`, merge its contents into this dataset.

        If value is an `XArray` object (or tuple of form
        `(dimensions, data[, attributes])`), add it to this dataset as a new
        variable.
        """
        if isinstance(value, DatasetArray):
            self.merge(value.renamed(key).dataset, inplace=True,
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

    def __str__(self):
        """Create a ncdump-like summary of the object."""
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
        dim_summary = ', '.join('%s: %s' % (k, v) for k, v
                                in self.dimensions.iteritems())
        return '<xray.%s (%s): %s>' % (type(self).__name__, dim_summary,
                                       ' '.join(self.noncoordinates))

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

    def renamed(self, name_dict):
        """Returns a new object with renamed variables and dimensions.

        Parameters
        ----------
        name_dict : dict-like
            Dictionary-like object whose keys are current variable or dimension
            names and whose values are new names.
        """
        for k in name_dict:
            if k not in self.variables:
                raise ValueError("Cannot rename %r because it is not a "
                                 "variable in this dataset" % k)
        variables = OrderedDict()
        for k, v in self.variables.iteritems():
            name = name_dict.get(k, k)
            dims = tuple(name_dict.get(dim, dim) for dim in v.dimensions)
            #TODO: public interface for renaming a variable without loading
            # data?
            variables[name] = xarray.XArray(dims, v._data, v.attributes,
                                            v.encoding, v._indexing_mode)

        return type(self)(variables, self.attributes)

    def replace(self, variables, decode_cf=False):
        """Returns a new dataset with some variables replaced or removed.

        Parameters
        ----------
        variables : dict-like, optional
            A mapping from variable names to `XArray` objects, sequences of
            the form `(dimensions, data[, attributes])` which can be used as
            arguments to create a new `XArray`, or `None`, which indicates that
            the variable by this name should be omited if it already exists
            in this dataset.
        decode_cf : bool, optional
            Whether to decode these variables according to CF conventions.

        Returns
        -------
        replaced: Dataset
            New dataset based on this dataset.


        Notes
        -----

        As long as dimensions in the resulting dataset are consistent, replace
        can alter or replace existing dimensinos.
        """
        all_variables = OrderedDict(
            [(k, v) for k, v in self.variables.iteritems()
             if k not in variables]
            + [(k, v) for k, v in variables.iteritems() if v is not None])
        return type(self)(all_variables, self.attributes, decode_cf=decode_cf)

    def merge(self, other, inplace=False, overwrite_vars=None):
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
        if overwrite_vars is None:
            overwrite_vars = {}
        for k, v in other.variables.iteritems():
            if (k in self and k not in overwrite_vars
                    and not utils.xarray_equal(v, self[k])):
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

    def groupby(self, group, squeeze=True):
        """Group this dataset by unique values of the indicated group.

        Parameters
        ----------
        group : str or DatasetArray
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
            group = DatasetArray(ds, group)
        return groupby.GroupBy(self, group.focus, group, squeeze=squeeze)

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

    @classmethod
    def concat(cls, datasets, dimension, indexers=None, concat_over=None):
        """Concatenate datasets along a new or existing dimension.

        Parameters
        ----------
        datasets : iterable of Dataset
            Datasets to stack together. Each dataset is expected to have
            matching attributes, and all variables except those along the
            stacked dimension (those that contain "dimension" as a dimension or
            are listed in "concat_over") are expected to be equal.
        dimension : str
            Name of the dimension to stack along.
        indexers : None or iterable of indexers, optional
            Iterable of indexers of the same length as variables which
            specifies how to assign variables from each dataset along the given
            dimension. If not supplied, indexers is inferred from the length of
            each variable along the dimension, and the variables are stacked in
            the given order.
        concat_over : None or iterable of str, optional
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
        if not isinstance(dimension, basestring):
            raise ValueError('dimension currently must be a string')
        # create the new dataset and add non-concatenated variables
        concatenated = cls({}, datasets[0].attributes)
        if concat_over is None:
            concat_over = set()
        else:
            concat_over = set(concat_over)
            if any(v not in datasets[0] for v in concat_over):
                raise ValueError('not all elements in concat_over %r found '
                                 'in the first dataset %r'
                                 % (concat_over, datasets[0]))
        for k, v in datasets[0].variables.iteritems():
            if k == dimension or dimension in v.dimensions:
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
                elif (k in concatenated and
                          not utils.xarray_equal(v, concatenated[k])):
                    raise ValueError(
                        'variable %r not equal across datasets' % k)
        # stack up each variable in turn
        for k in concat_over:
            concatenated[k] = xarray.XArray.concat(
                [ds[k] for ds in datasets], dimension, indexers)
        # finally, reorder the concatenated dataset's variables to the order they
        # were encountered in the first dataset
        reordered = cls(OrderedDict((k, concatenated[k]) for k in datasets[0]),
                        concatenated.attributes)
        return reordered

    def to_dataframe(self):
        """Convert this dataset into a pandas.DataFrame.

        Non-coordinate variables in this dataset form the columns of the
        DataFrame. The DataFrame is be indexed by the Cartesian product of
        this dataset's indices.
        """
        index_names = self.coordinates.keys()
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
        # note: pd.MultiIndex.from_product is new in pandas-0.13.1
        # np.asarray is necessary to work around a pandas bug:
        # https://github.com/pydata/pandas/issues/6439
        coords = [np.asarray(v) for v in self.coordinates.values()]
        index = pd.MultiIndex.from_product(coords, names=index_names)
        return pd.DataFrame(OrderedDict(zip(columns, data)), index=index)
