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
                if isinstance(v._data, pd.DatetimeIndex)]

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
                    month = self[ref_var].data.month
                    data = (month // 3) % 4 + 1
                else:
                    data = getattr(self[ref_var].data, suffix)
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
    together form a self describing data set

    Dataset implements the mapping interface with keys given by variable names
    and values given by DatasetArray objects focused on each variable name.

    Note: the size of dimensions in a dataset cannot be changed.

    Attributes
    ----------
    variables : {name: variable, ...}
    attributes : {key: value, ...}
    dimensions : {name: length, ...}
    coordinates : {name: variable, ...}
    noncoordinates : {name: variable, ...}
    virtual_variables : list
    """
    def __init__(self, variables=None, attributes=None):
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
        """
        self._variables = _VariablesDict()
        self._dimensions = OrderedDict()
        if variables is not None:
            self._set_variables(variables)
        if attributes is None:
            attributes = {}
        self._attributes = OrderedDict(attributes)

    def _as_variable(self, name, var):
        if not isinstance(var, xarray.XArray):
            try:
                var = xarray.XArray(*var)
            except TypeError:
                raise TypeError('Dataset variables must be of type '
                                'DatasetArray or XArray, or a sequence of the '
                                'form (dimensions, data[, attributes])')

        if name in var.dimensions:
            # convert the coordinate into a pandas.Index
            if var.ndim != 1:
                raise ValueError('a coordinate variable must be defined with '
                                 '1-dimensional data')
            attr = var.attributes
            if 'units' in attr and 'since' in attr['units']:
                units = attr.pop('units')
                var.data = utils.num2datetimeindex(var.data, units,
                                                   attr.pop('calendar', None))
                attr['cf_units'] = units
            else:
                var.data = pd.Index(var.data)
        return var

    def _set_variables(self, variables):
        """Set a mapping of variables and update dimensions"""
        # save new variables into a temporary list so all the error checking
        # can be done before updating _variables
        new_variables = []
        for k, var in variables.iteritems():
            var = self._as_variable(k, var)
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
    def load_store(cls, store):
        return cls(store.variables, store.attributes)

    @property
    def variables(self):
        return Frozen(self._variables)

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, value):
        self._attributes = OrderedDict(value)

    @property
    def dimensions(self):
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
        """
        The 'in' operator will return true or false depending on
        whether 'key' is a varibale in the data object or not.
        """
        return key in self.variables

    def __len__(self):
        return len(self.variables)

    def __iter__(self):
        return iter(self.variables)

    @property
    def virtual_variables(self):
        """Arrays that don't exist in this dataset but for which dataviews
        could be created on demand (because they can be calculated from other
        dataset variables or dimensions)
        """
        return self._variables.virtual

    def __getitem__(self, key):
        return DatasetArray(self.select(key), key)

    def __setitem__(self, key, value):
        if isinstance(value, DatasetArray):
            self.merge(value.renamed(key).dataset, inplace=True)
        else:
            self._set_variables({key: value})

    def __delitem__(self, key):
        del self._variables[key]
        dims = set().union(v.dimensions for v in self._variables.itervalues())
        for dim in self._dimensions:
            if dim not in dims:
                del self._dimensions[dim]

    # mutable objects should not be hashable
    __hash__ = None

    def __eq__(self, other):
        try:
            # some stores (e.g., scipy) do not seem to preserve order, so don't
            # require matching dimension or variable order for equality
            return (sorted(self.attributes.items())
                        == sorted(other.attributes.items())
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
        """Coordinates are variables with names that match dimensions

        They are always stored internally as arrays with data that is a
        pandas.Index object
        """
        return FrozenOrderedDict([(dim, self.variables[dim])
                                  for dim in self.dimensions])

    @property
    def noncoordinates(self):
        """Non-coordinates are variables with names that do not match
        dimensions
        """
        return FrozenOrderedDict([(name, v)
                for (name, v) in self.variables.iteritems()
                if name not in self.dimensions])

    def dump_to_store(self, store):
        """Store dataset contents to a backends.*DataStore object"""
        store.set_dimensions(self.dimensions)
        store.set_variables(self.variables)
        store.set_attributes(self.attributes)
        store.sync()

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
        dim_summary = ', '.join('%s: %s' % (k, v) for k, v
                                in self.dimensions.iteritems())
        return '<xray.%s (%s): %s>' % (type(self).__name__, dim_summary,
                                       ' '.join(self.noncoordinates))

    def indexed_by(self, **indexers):
        """Return a new dataset with each array indexed along the specified
        dimension(s)

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
        Array.indexed_by
        """
        return self.indexed_by(**remap_loc_indexers(self.variables, indexers))

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
                                            v._indexing_mode)

        return type(self)(variables, self.attributes)

    def merge(self, other, inplace=False):
        """Merge two datasets into a single new dataset

        This method generally not allow for overriding data. Arrays,
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
                                  compat=utils.xarray_equal)
        # update contents
        obj = self if inplace else self.copy()
        obj._set_variables(OrderedDict((k, v) for k, v
                                       in other.variables.iteritems()
                                       if k not in obj.variables))
        # remove conflicting attributes
        for k, v in other.attributes.iteritems():
            if k in self.attributes and v != self.attributes[k]:
                del self.attributes[k]
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
        if any(k not in self.variables and k not in self.dimensions
               for k in names):
            raise ValueError('One or more of the specified variable/dimension '
                             'names does not exist on this dataset')
        variables = OrderedDict((k, v) for k, v in self.variables.iteritems()
                                if k not in names)
        return type(self)(variables, self.attributes)

    def replace(self, name, variable):
        """Returns a new dataset with the variable 'name' replaced with
        'variable'

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
        ds = self.unselect(name)
        ds[name] = variable
        return ds

    def groupby(self, group, squeeze=True):
        """Group this dataset by unique values of the indicated group

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

    def to_dataframe(self):
        """Convert this dataset into a pandas.DataFrame

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
