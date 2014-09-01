from collections import Mapping
from io import BytesIO
import warnings

import numpy as np
import pandas as pd

from .. import backends, conventions
from . import alignment
from . import common
from . import formatting
from . import groupby
from . import indexing
from . import variable
from . import utils
from . import ops
from .coordinates import DatasetCoordinates
from .utils import (FrozenOrderedDict, Frozen, SortedKeysDict, ChainMap,
                    multi_index_from_product)
from .pycompat import iteritems, itervalues, basestring, OrderedDict


def open_dataset(nc, decode_cf=True, mask_and_scale=True, decode_times=True,
                 concat_characters=True, *args, **kwargs):
    """Load a dataset from a file or file-like object.

    Parameters
    ----------
    nc : str or file
        Path to a netCDF4 file or an OpenDAP URL (opened with python-netCDF4)
        or a file object or string serialization of a netCDF3 file (opened with
        scipy.io.netcdf).
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    *args, **kwargs : optional
        Format specific loading options passed on to the datastore.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.
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
    return Dataset.load_store(store, decode_cf=decode_cf,
                              mask_and_scale=mask_and_scale,
                              decode_times=decode_times,
                              concat_characters=concat_characters)


# list of attributes of pd.DatetimeIndex that are ndarrays of time info
_DATETIMEINDEX_COMPONENTS = ['year', 'month', 'day', 'hour', 'minute',
                             'second', 'microsecond', 'nanosecond', 'date',
                             'time', 'dayofyear', 'weekofyear', 'dayofweek',
                             'quarter']


class VariablesDict(OrderedDict):
    """VariablesDict is an OrderedDict subclass that also implements "virtual"
    variables that are created from other variables on demand

    Currently, virtual variables are restricted to attributes of
    pandas.DatetimeIndex objects (e.g., 'year', 'month', 'day', etc., plus
    'season' for climatological season), which are accessed by getting the item
    'time.year'.
    """
    @property
    def virtual(self):
        """A frozenset of variable names that don't exist in this dataset but
        for which could be created on demand (because they can be calculated
        from other dataset variables)
        """
        def _castable_to_timestamp(obj):
            try:
                pd.Timestamp(obj)
            except:
                return False
            else:
                return True

        virtual_vars = []
        for k, v in iteritems(self):
            if ((v.dtype.kind == 'M' and isinstance(v, variable.Coordinate))
                    or (v.ndim == 0 and _castable_to_timestamp(v.values))):
                # nb. dtype.kind == 'M' is datetime64
                for suffix in _DATETIMEINDEX_COMPONENTS + ['season']:
                    name = '%s.%s' % (k, suffix)
                    if name not in self:
                        virtual_vars.append(name)
        return frozenset(virtual_vars)

    def __missing__(self, key):
        """Fall back to returning a virtual variable, if possible
        """
        if not isinstance(key, basestring):
            raise KeyError(repr(key))

        split_key = key.split('.')
        if len(split_key) != 2:
            raise KeyError(repr(key))

        ref_var_name, suffix = split_key
        ref_var = self[ref_var_name]
        if isinstance(ref_var, variable.Coordinate):
            date = ref_var.to_index()
        elif ref_var.ndim == 0:
            date = pd.Timestamp(ref_var.values)

        if suffix == 'season':
            # seasons = np.array(['DJF', 'MAM', 'JJA', 'SON'])
            month = date.month
            data = (month // 3) % 4 + 1
        else:
            data = getattr(date, suffix)
        return variable.Variable(ref_var.dims, data)


def _as_dataset_variable(name, var):
    """Prepare a variable for adding it to a Dataset
    """
    try:
        var = variable.as_variable(var, key=name)
    except TypeError:
        raise TypeError('Dataset variables must be an arrays or a tuple of '
                        'the form (dims, data[, attrs, encoding])')
    if name in var.dims:
        # convert the into an Index
        if var.ndim != 1:
            raise ValueError('an index variable must be defined with '
                             '1-dimensional data')
        var = var.to_coord()
    return var


def _expand_variables(raw_variables, old_variables={}, compat='identical'):
    """Expand a dictionary of variables.

    Returns a dictionary of Variable objects suitable for inserting into a
    Dataset._variables dictionary.

    This includes converting tuples (dims, data) into Variable objects,
    converting coordinate variables into Coordinate objects and expanding
    DataArray objects into Variables plus coordinates.

    Raises ValueError if any conflicting values are found, between any of the
    new or old variables.
    """
    new_variables = OrderedDict()
    new_coord_names = set()
    variables = ChainMap(new_variables, old_variables)

    def add_variable(name, var):
        if name not in variables:
            variables[name] = _as_dataset_variable(name, var)
        elif not getattr(variables[name], compat)(var):
            raise ValueError('conflicting value for variable %s:\n'
                             'first value: %r\nsecond value: %r'
                             % (name, variables[name], var))

    for name, var in iteritems(raw_variables):
        if hasattr(var, 'coords'):
            # it's a DataArray
            new_coord_names.update(var.coords)
            for dim, coord in iteritems(var.coords):
                if dim != name:
                    add_variable(dim, coord)
        add_variable(name, var)
    return new_variables, new_coord_names


def _calculate_dims(variables):
    """Calculate the dimensions corresponding to a set of variables.

    Returns dictionary mapping from dimension names to sizes. Raises ValueError
    if any of the dimension sizes conflict.
    """
    dims = SortedKeysDict()
    last_used = {}
    scalar_vars = set(k for k, v in iteritems(variables) if not v.dims)
    for k, var in iteritems(variables):
        for dim, size in zip(var.dims, var.shape):
            if dim in scalar_vars:
                raise ValueError('dimension %s already exists as a scalar '
                                 'variable' % dim)
            if dim not in dims:
                dims[dim] = size
                last_used[dim] = k
            elif dims[dim] != size:
                raise ValueError('conflicting sizes for dimension %r: '
                                 'length %s on %r and length %s on %r' %
                                 (dim, size, k, dims[dim], last_used[dim]))
    return dims


class _DatasetLike(object):
    """A Dataset-like object that only contains a few private attributes

    Like `as_dataset`, handles DataArrays, Datasets and dictionaries of
    variables. The difference is that this method never creates a new Dataset
    object, and hence is much more lightweight, avoiding any consistency
    checks on the variables (that should be handled later).
    """
    def __init__(self, obj):
        obj = getattr(obj, '_dataset', obj)
        self._variables = getattr(obj, '_variables', obj)
        self._coord_names = getattr(obj, '_coord_names', set())
        self.attrs = getattr(obj, 'attrs', {})


def _assert_compat_valid(compat):
    if compat not in ['equals', 'identical']:
        raise ValueError("compat=%r invalid: must be 'equals' or "
                         "'identical'" % compat)


def _item0_str(items):
    """Key function for use in sorted on a list of variables.

    This is useful because None is not comparable to strings in Python 3.
    """
    return str(items[0])


def _assert_empty(args, msg='%s'):
    if args:
        raise ValueError(msg % args)


def as_dataset(obj):
    """Cast the given object to a Dataset.

    Handles DataArrays, Datasets and dictionaries of variables. A new Dataset
    object is only created in the last case.
    """
    obj = getattr(obj, '_dataset', obj)
    if not isinstance(obj, Dataset):
        obj = Dataset(obj)
    return obj


class Dataset(Mapping, common.ImplementsDatasetReduce):
    """A netcdf-like data object consisting of variables and attributes which
    together form a self describing dataset.

    Dataset implements the mapping interface with keys given by variable names
    and values given by DataArray objects for each variable name.

    One dimensional variables with name equal to their dimension are
    coordinates, which means they are saved in the dataset as
    :py:class:`~xray.Coordinate` objects.
    """
    def __init__(self, variables=None, coords=None, attrs=None):
        """To load data from a file or file-like object, use the `open_dataset`
        function.

        Parameters
        ----------
        variables : dict-like, optional
            A mapping from variable names to :py:class:`~xray.DataArray`
            objects, :py:class:`~xray.Variable` objects or tuples of the
            form ``(dims, data[, attrs])`` which can be used as arguments to
            create a new ``Variable``. Each dimension must have the same length
            in all variables in which it appears.
        coords : dict-like, optional
            Do not use: not yet implemented!
        attrs : dict-like, optional
            Global attributes to save on this dataset.

        .. warning:: For now, if you wish to specify ``attrs``, you *must* use
        a keyword argument: ``xray.Dataset(variables, attrs=attrs)``. The
        ``coords`` argument is reserved for specifying coordinates
        independently of other variables for use in a future version of xray.
        For now, coordinates will extracted automatically from variables.
        """
        self._variables = VariablesDict()
        self._coord_names = set()
        self._dims = SortedKeysDict()
        self._attrs = OrderedDict()
        self._file_obj = None
        if variables is None:
            variables = {}
        if coords is None:
            coords = set()
        if variables or coords:
            self._set_init_vars_and_dims(variables, coords)
        if attrs is not None:
            self._attrs.update(attrs)

    def _add_missing_coords(self):
        """Add missing coordinates IN-PLACE to _variables
        """
        for dim, size in iteritems(self._dims):
            if dim not in self._variables:
                coord = variable.Coordinate(dim, np.arange(size))
                self._variables[dim] = coord

    def _update_vars_and_coords(self, new_variables, new_coord_names={},
                                needs_copy=True, check_coord_names=True):
        """Add a dictionary of new variables to this dataset.

        Raises a ValueError if any dimensions have conflicting lengths in the
        new dataset. Otherwise will update this dataset's _variables and
        _dims attributes in-place.

        Set `needs_copy=False` only if this dataset is brand-new and hence
        can be thrown away if this method fails.
        """
        # default to creating another copy of variables so can unroll if we end
        # up with inconsistent dimensions
        variables = self._variables.copy() if needs_copy else self._variables

        if check_coord_names:
            _assert_empty([k for k in self.noncoords if k in new_coord_names],
                          'coordinates already exist as variables: %s')

        variables.update(new_variables)
        dims = _calculate_dims(variables)
        # all checks are complete: it's safe to update
        self._variables = variables
        self._dims = dims
        self._add_missing_coords()
        self._coord_names.update(dims)
        self._coord_names.update(new_coord_names)

    def _set_init_vars_and_dims(self, variables, coords):
        """Set the initial value of Dataset variables and dimensions
        """
        _assert_empty([k for k in variables if k in coords],
                      'redundant variables and coordinates: %s')
        variables = ChainMap(variables, coords)

        new_variables, new_coord_names = _expand_variables(variables)
        _assert_empty([k for k in new_coord_names if k not in new_variables],
                      'no matching variables exist for some coordinates: %s')

        new_coord_names.update(coords)
        self._update_vars_and_coords(new_variables, new_coord_names,
                                     needs_copy=False,
                                     check_coord_names=False)

    @classmethod
    def load_store(cls, store, decode_cf=True, mask_and_scale=True,
                   decode_times=True, concat_characters=True):
        """Create a new dataset from the contents of a backends.*DataStore
        object
        """
        variables = store.variables
        if decode_cf:
            variables = conventions.decode_cf_variables(
                store.variables, mask_and_scale=mask_and_scale,
                decode_times=decode_times, concat_characters=concat_characters)
        obj = cls(variables, attrs=store.attrs)
        obj._file_obj = store
        return obj

    def close(self):
        """Close any datastores linked to this dataset
        """
        if self._file_obj is not None:
            self._file_obj.close()
        self._file_obj = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getstate__(self):
        """Always load data in-memory before pickling"""
        self.load_data()
        # self.__dict__ is the default pickle object, we don't need to
        # implement our own __setstate__ method to make pickle work
        state = self.__dict__.copy()
        # throw away any references to datastores in the pickle
        state['_file_obj'] = None
        return state

    @property
    def variables(self):
        """Dictionary of Variable objects contained in this dataset.

        This is a low-level interface into the contents of a dataset. The
        dictionary is frozen to prevent it from being modified in ways that
        could create an inconsistent dataset (e.g., by setting variables with
        inconsistent dimensions).

        In general, to access and modify dataset contents, you should use
        dictionary methods on the dataset itself instead of the variables
        dictionary.
        """
        return Frozen(self._variables)

    @property
    def attributes(self):
        utils.alias_warning('attributes', 'attrs', 3)
        return self._attrs

    @attributes.setter
    def attributes(self, value):
        utils.alias_warning('attributes', 'attrs', 3)
        self._attrs = OrderedDict(value)

    @property
    def attrs(self):
        """Dictionary of global attributes on this dataset
        """
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        self._attrs = OrderedDict(value)

    @property
    def dims(self):
        """Mapping from dimension names to lengths.

        This dictionary cannot be modified directly, but is updated when adding
        new variables.
        """
        return Frozen(self._dims)

    @property
    def dimensions(self):
        utils.alias_warning('dimensions', 'dims')
        return self.dims

    def load_data(self):
        """Manually trigger loading of this dataset's data from disk or a
        remote source into memory and return this dataset.

        Normally, it should not be necessary to call this method in user code,
        because all xray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.
        """
        for v in itervalues(self._variables):
            v.load_data()
        return self

    @classmethod
    def _construct_direct(cls, variables, coord_names, dims, attrs,
                          file_obj=None):
        """Shortcut around __init__ for internal use when we want to skip
        costly validation
        """
        obj = object.__new__(cls)
        obj._variables = variables
        obj._coord_names = coord_names
        obj._dims = dims
        obj._attrs = attrs
        obj._file_obj = file_obj
        return obj

    def copy(self, deep=False):
        """Returns a copy of this dataset.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy is made, so each variable in the new dataset
        is also a variable in the original dataset.
        """
        if deep:
            variables = VariablesDict((k, v.copy(deep=True))
                                      for k, v in iteritems(self.variables))
        else:
            variables = self._variables.copy()
        # skip __init__ to avoid costly validation
        return self._construct_direct(variables, self._coord_names.copy(),
                                      self._dims.copy(), self._attrs.copy())

    def _copy_listed(self, names):
        """Create a new Dataset with the listed variables from this dataset and
        the all relevant coordinates. Skips all validation.
        """
        variables = VariablesDict((k, self._variables[k]) for k in names)
        needed_dims = set()
        for v in variables.values():
            needed_dims.update(v.dims)
        coord_names = set()
        for k in self._coord_names:
            if set(self._variables[k].dims) <= needed_dims:
                variables[k] = self._variables[k]
                coord_names.add(k)
        dims = SortedKeysDict(dict((k, self._dims[k]) for k in needed_dims))
        return self._construct_direct(variables, coord_names, dims,
                                      self._attrs.copy())

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
        """A frozenset of variable names that don't exist in this dataset but
        for which could be created on demand.

        These variables can be derived by performing simple operations on
        existing dataset variables. Currently, the only implemented virtual
        variables are time/date components [1_] such as "time.month" or
        "time.dayofyear", where "time" is the name of a index whose data
        is a `pandas.DatetimeIndex` object. The virtual variable "time.season"
        (for climatological season, starting with 1 for "DJF") is the only such
        variable which is not directly implemented in pandas.

        References
        ----------
        .. [1] http://pandas.pydata.org/pandas-docs/stable/api.html#time-date-components
        """
        return self._variables.virtual

    def __getitem__(self, key):
        """Access the given variable name in this dataset as a `DataArray`, or
        the given variable names as another Dataset.
        """
        from .dataarray import DataArray

        if isinstance(key, basestring) or key is None:
            return DataArray._new_from_dataset(self, key)
        else:
            return self._copy_listed(key)

    def __setitem__(self, key, value):
        """Add an array to this dataset.

        If value is a `DataArray`, call its `select_vars()` method, rename it
        to `key` and merge the contents of the resulting dataset into this
        dataset.

        If value is an `Variable` object (or tuple of form
        ``(dims, data[, attrs])``), add it to this dataset as a new
        variable.
        """
        self.merge({key: value}, inplace=True, overwrite_vars=[key])

    def __delitem__(self, key):
        """Remove a variable from this dataset.

        If this variable is a dimension, all variables containing this
        dimension are also removed.
        """
        def remove(k):
            del self._variables[k]
            self._coord_names.discard(k)

        if key in self._dims:
            del self._dims[key]
        remove(key)
        also_delete = [k for k, v in iteritems(self._variables)
                       if key in v.dims]
        for key in also_delete:
            remove(key)

    # mutable objects should not be hashable
    __hash__ = None

    def equals(self, other):
        """Two Datasets are equal if they have the same variables and all
        variables are equal.
        """
        # use equals as an alias for __eq__ to support a common interface with
        # DataArray
        try:
            # some stores (e.g., scipy) do not seem to preserve order, so don't
            # require matching order for equality
            return (len(self) == len(other)
                    and all(k1 == k2 and v1.equals(v2)
                            for (k1, v1), (k2, v2)
                            in zip(sorted(self.variables.items(),
                                          key=_item0_str),
                                   sorted(other.variables.items(),
                                          key=_item0_str))))
        except (TypeError, AttributeError):
            return False

    def __eq__(self, other):
        raise TypeError('__eq__ no longer defined for Dataset; in a future '
                        'version of xray, it will be reenabled but applied '
                        'elementwise')

    def __ne__(self, other):
        raise TypeError('__ne__ no longer defined for Dataset; in a future '
                        'version of xray, it will be reenabled but applied '
                        'elementwise')

    def identical(self, other):
        """Two Datasets are identical if they have the same variables and all
        variables are identical (with the same attributes), and they also have
        the same global attributes.
        """
        try:
            return (utils.dict_equiv(self.attrs, other.attrs)
                    and len(self) == len(other)
                    and all(k1 == k2 and v1.identical(v2)
                            for (k1, v1), (k2, v2)
                            in zip(sorted(self.variables.items(),
                                          key=_item0_str),
                                   sorted(other.variables.items(),
                                          key=_item0_str))))
        except (TypeError, AttributeError):
            return False

    @property
    def indexes(self):
        return utils.FrozenOrderedDict(
            (k, self[k].to_index()) for k in self.dims)

    @property
    def nonindexes(self):
        return utils.FrozenOrderedDict(
            (k, self[k]) for k in self.variables if k not in self.dims)

    @property
    def coords(self):
        """Dictionary of xray.Coordinate objects used for label based indexing.
        """
        return DatasetCoordinates(self)

    @property
    def coordinates(self):
        utils.alias_warning('coordinates', 'coords')
        return self.coords

    @property
    def noncoords(self):
        """Dictionary of DataArrays whose names do not match dimensions.
        """
        return FrozenOrderedDict((name, self[name]) for name in self
                                 if name not in self.coords)

    @property
    def noncoordinates(self):
        """Dictionary of DataArrays whose names do not match dimensions.
        """
        utils.alias_warning('noncoordinates', 'noncoords')
        return self.noncoords

    def set_coords(self, names, inplace=False):
        """Given names of one or more variables, set them as coordinates

        Parameters
        ----------
        names : str or list of str
            Name(s) of variables in this dataset to convert into coordinates.
        inplace : bool, optional
            If True, modify this dataset inplace. Otherwise, create a new
            object.

        Returns
        -------
        Dataset
        """
        # TODO: allow inserting new coordinates with this method, like
        # DataFrame.set_index?
        # nb. check in self._variables, not self.noncoords to insure that the
        # operation is idempotent
        if isinstance(names, basestring):
            names = [names]
        self._assert_all_in_dataset(names)
        obj = self if inplace else self.copy()
        obj._coord_names.update(names)
        return obj

    def reset_coords(self, names=None, drop=False, inplace=False):
        """Given names of coordinates, reset them to become variables

        Parameters
        ----------
        names : str or list of str, optional
            Name(s) of coordinates in this dataset to reset into variables. By
            default, all coordinates are reset.
        drop : bool, optional
            If True, remove coordinates instead of converting them into
            variables.
        inplace : bool, optional
            If True, modify this dataset inplace. Otherwise, create a new
            object.

        Returns
        -------
        Dataset
        """
        if names is None:
            names = self._coord_names - set(self.dims)
        else:
            if isinstance(names, basestring):
                names = [names]
            self._assert_all_in_dataset(names)
            _assert_empty(
                set(names) & set(self.dims),
                'cannot remove index coordinates with reset_coords: %s')
        obj = self if inplace else self.copy()
        obj._coord_names.difference_update(names)
        if drop:
            for name in names:
                del obj._variables[name]
        return obj

    def dump_to_store(self, store):
        """Store dataset contents to a backends.*DataStore object."""
        store.set_variables(self.variables)
        store.set_attributes(self.attrs)
        store.sync()

    def to_netcdf(self, filepath, **kwdargs):
        """Dump dataset contents to a location on disk using the netCDF4
        package.
        """
        with backends.NetCDF4DataStore(filepath, mode='w', **kwdargs) as store:
            self.dump_to_store(store)

    dump = to_netcdf

    def dumps(self, **kwargs):
        """Serialize dataset contents to a string. The serialization creates an
        in memory netcdf version 3 string using the scipy.io.netcdf package.
        """
        fobj = BytesIO()
        store = backends.ScipyDataStore(fobj, mode='w', **kwargs)
        self.dump_to_store(store)
        return fobj.getvalue()

    def __repr__(self):
        return formatting.dataset_repr(self)

    def isel(self, **indexers):
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
        Dataset.sel
        DataArray.isel
        DataArray.sel
        """
        invalid = [k for k in indexers if not k in self.dims]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        # all indexers should be int, slice or np.ndarrays
        indexers = [(k, (np.asarray(v)
                         if not isinstance(v, (int, np.integer, slice))
                         else v))
                    for k, v in iteritems(indexers)]

        variables = OrderedDict()
        for name, var in iteritems(self.variables):
            var_indexers = dict((k, v) for k, v in indexers if k in var.dims)
            variables[name] = var.isel(**var_indexers)
        obj = type(self)(variables, attrs=self.attrs)
        obj._coord_names.update(self.coords)
        return obj

    indexed = utils.function_alias(isel, 'indexed')

    def sel(self, **indexers):
        """Return a new dataset with each variable indexed by tick labels
        along the specified dimension(s).

        In contrast to `Dataset.isel`, indexers for this method should use
        labels instead of integers.

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
            by individual, slices or arrays of tick labels.

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
        Dataset.isel
        DataArray.isel
        DataArray.sel
        """
        return self.isel(**indexing.remap_label_indexers(self, indexers))

    labeled = utils.function_alias(sel, 'labeled')

    def reindex_like(self, other, copy=True):
        """Conform this object onto the indexes of another object, filling
        in missing values with NaN.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to pandas.Index objects, which provides coordinates upon
            which to index the variables in this dataset. The indexes on this
            other object need not be the same as the indexes on this
            dataset. Any mis-matched index values will be filled in with
            NaN, and any mis-matched dimension names will simply be ignored.
        copy : bool, optional
            If `copy=True`, the returned dataset contains only copied
            variables. If `copy=False` and no reindexing is required then
            original variables from this dataset are returned.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with this dataset's data but coordinates from the
            other object.

        See Also
        --------
        Dataset.reindex
        align
        """
        return self.reindex(copy=copy, **other.indexes)

    def reindex(self, copy=True, **indexers):
        """Conform this object onto a new set of indexes, filling in
        missing values with NaN.

        Parameters
        ----------
        copy : bool, optional
            If `copy=True`, the returned dataset contains only copied
            variables. If `copy=False` and no reindexing is required then
            original variables from this dataset are returned.
        **indexers : dict
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate values
            will be filled in with NaN, and any mis-matched dimension names will
            simply be ignored.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.reindex_like
        align
        """
        if not indexers:
            # shortcut
            return self.copy(deep=True) if copy else self

        variables = alignment.reindex_variables(
            self.variables, self.indexes, indexers, copy=copy)
        obj = type(self)(variables, attrs=self.attrs)
        obj._coord_names.update(self.coords)
        return obj

    def rename(self, name_dict, inplace=False):
        """Returns a new object with renamed variables and dimensions.

        Parameters
        ----------
        name_dict : dict-like
            Dictionary whose keys are current variable or dimension names and
            whose values are new names.
        inplace : bool, optional
            If True, rename variables and dimensions in-place. Otherwise,
            return a new dataset object.

        Returns
        -------
        renamed : Dataset
            Dataset with renamed variables and dimensions.
        """
        for k in name_dict:
            if k not in self.variables:
                raise ValueError("cannot rename %r because it is not a "
                                 "variable in this dataset" % k)
        variables = VariablesDict()
        coord_names = set()
        for k, v in iteritems(self.variables):
            name = name_dict.get(k, k)
            dims = tuple(name_dict.get(dim, dim) for dim in v.dims)
            var = v.copy(deep=False)
            var.dims = dims
            variables[name] = var
            if k in self._coord_names:
                coord_names.add(name)

        if inplace:
            self._dims = _calculate_dims(variables)
            self._variables = variables
            obj = self
        else:
            obj = type(self)(variables, attrs=self.attrs)
        obj._coord_names = coord_names
        return obj

    def update(self, other, inplace=True):
        """Update this dataset's variables and attributes with those from
        another dataset.

        Parameters
        ----------
        other : Dataset or castable to Dataset
            Dataset or variables with which to update this dataset.
        inplace : bool, optional
            If True, merge the other dataset into this dataset in-place.
            Otherwise, return a new dataset object.

        Returns
        -------
        updated : Dataset
            Updated dataset.

        Raises
        ------
        ValueError
            If any dimensions would have inconsistent sizes in the updated
            dataset.
        """
        other = _DatasetLike(other)
        new_variables, _ = _expand_variables(other._variables)

        obj = self if inplace else self.copy()
        obj._update_vars_and_coords(new_variables, other._coord_names,
                                    needs_copy=inplace)
        obj.attrs.update(other.attrs)
        return obj

    def merge(self, other, inplace=False, overwrite_vars=set(),
              compat='equals'):
        """Merge the variables of two datasets into a single new dataset.

        This method generally not allow for overriding data, with the exception
        of attributes, which are ignored on the second dataset. Variables with
        the same name are checked for conflicts via the equals or identical
        methods.

        Parameters
        ----------
        other : Dataset or castable to Dataset
            Dataset or variables to merge with this dataset.
        inplace : bool, optional
            If True, merge the other dataset into this dataset in-place.
            Otherwise, return a new dataset object.
        overwrite_vars : str or sequence, optional
            If provided, update variables of these name(s) without checking for
            conflicts in this dataset.
        compat : {'equals', 'identical'}, optional
            String indicating how to compare variables of the same name for
            potential conflicts. 'equals' means that all values and dimensions
            must be the same; 'identical' means attributes must also be equal.

        Returns
        -------
        merged : Dataset
            Merged dataset.

        Raises
        ------
        ValueError
            If any variables conflict (see ``compat``).
        """
        _assert_compat_valid(compat)
        other = _DatasetLike(other)

        # determine variables to check for conflicts
        if not overwrite_vars:
            potential_conflicts = self.variables
        else:
            if isinstance(overwrite_vars, basestring):
                overwrite_vars = set([overwrite_vars])
            else:
                overwrite_vars = set(overwrite_vars)
            potential_conflicts = dict((k, v) for k, v in iteritems(self.variables)
                                       if k not in overwrite_vars)

        # update variables
        new_variables, _ = _expand_variables(other._variables,
                                             potential_conflicts, compat)
        obj = self if inplace else self.copy()
        obj._update_vars_and_coords(new_variables, other._coord_names,
                                    needs_copy=inplace)
        return obj

    def _assert_all_in_dataset(self, names, virtual_okay=False):
        bad_names = set(names) - set(self._variables)
        if virtual_okay:
            bad_names -= self.virtual_variables
        if bad_names:
            raise ValueError('One or more of the specified variables '
                             'cannot be found in this dataset')

    def select_vars(self, *names):
        """Returns a new dataset that contains only the named variables and
        their coordinates.

        Parameters
        ----------
        *names : str
            Names of the variables to include in the returned object.

        Returns
        -------
        Dataset
            The returned object has the same attributes as the original. Only
            the named variables and their coordinates are included.
        """
        self._assert_all_in_dataset(names, virtual_okay=True)
        variables = OrderedDict((k, self[k]) for k in names)
        return type(self)(variables, attrs=self.attrs)

    select = utils.function_alias(select_vars, 'select')

    def drop_vars(self, *names):
        """Returns a new dataset without the named variables.

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
        self._assert_all_in_dataset(names)
        drop = set(names)
        drop |= set(k for k, v in iteritems(self.variables)
                    if any(name in v.dims for name in names))
        variables = OrderedDict((k, v) for k, v in iteritems(self.variables)
                                if k not in drop)
        coord_names = set(k for k in self._coord_names if k in variables)
        obj = type(self)(variables, attrs=self.attrs)
        obj._coord_names = coord_names
        return obj

    unselect = utils.function_alias(drop_vars, 'unselect')

    def groupby(self, group, squeeze=True):
        """Group this dataset by unique values of the indicated group.

        Parameters
        ----------
        group : str, DataArray or Coordinate
            Array whose unique values should be used to group this array. If a
            string, must be the name of a variable contained in this dataset.
        squeeze : boolean, optional
            If "group" is a dimension of any arrays in this dataset, `squeeze`
            controls whether the subarrays have a dimension of length 1 along
            that dimension or if the dimension is squeezed out.

        Returns
        -------
        grouped : GroupBy
            A `GroupBy` object patterned after `pandas.GroupBy` that can be
            iterated over in the form of `(unique_value, grouped_array)` pairs.
        """
        if isinstance(group, basestring):
            group = self[group]
        return groupby.DatasetGroupBy(self, group, squeeze=squeeze)

    def squeeze(self, dim=None):
        """Return a new dataset with squeezed data.

        Parameters
        ----------
        dim : None or str or tuple of str, optional
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
        return common.squeeze(self, self.dims, dim)

    def reduce(self, func, dim=None, keep_attrs=False, **kwargs):
        """Reduce this dataset by applying `func` along some dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `func`.  By default `func` is
            applied over all dimensions.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Dataset
            Dataset with this object's DataArrays replaced with new DataArrays
            of summarized data and the indicated dimension(s) removed.
        """
        if 'dimension' in kwargs and dim is None:
            dim = kwargs.pop('dimension')
            utils.alias_warning('dimension', 'dim')

        if isinstance(dim, basestring):
            dims = set([dim])
        elif dim is None:
            dims = set(self.dims)
        else:
            dims = set(dim)

        _assert_empty([dim for dim in dims if dim not in self.dims],
                      'Dataset does not contain the dimensions: %s')

        variables = OrderedDict()
        for name, var in iteritems(self.variables):
            reduce_dims = [dim for dim in var.dims if dim in dims]
            if reduce_dims:
                if name not in self.coords:
                    if len(reduce_dims) == 1:
                        # unpack dimensions for the benefit of functions like
                        # np.argmin which can't handle tuple arguments
                        reduce_dims, = reduce_dims
                    try:
                        variables[name] = var.reduce(func,
                                                     dim=reduce_dims,
                                                     **kwargs)
                    except TypeError:
                        # array (e.g., string) does not support this reduction,
                        # so skip it
                        # TODO: rethink silently passing, because the problem
                        # may be the dimensions and kwargs arguments, not the
                        # dtype of the array
                        pass
            else:
                variables[name] = var

        coord_names = set(k for k in self.coords if k in variables)
        attrs = self.attrs if keep_attrs else {}
        obj = type(self)(variables, attrs=attrs)
        obj._coord_names = coord_names
        return obj

    def apply(self, func, keep_attrs=False, **kwargs):
        """Apply a function over noncoordinate variables in this dataset.

        Parameters
        ----------
        func : function
            Function which can be called in the form `f(x, **kwargs)` to
            transform each DataArray `x` in this dataset into another
            DataArray.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one. If False, the new object will
            be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        applied : Dataset
            Resulting dataset from applying over each noncoordinate.
            Coordinates which are no longer used as the dimension of a
            noncoordinate are dropped.
        """
        variables = OrderedDict((k, func(v, **kwargs))
                                for k, v in iteritems(self.noncoords))
        attrs = self.attrs if keep_attrs else {}
        return type(self)(variables, attrs=attrs)

    @classmethod
    def concat(cls, *args, **kwargs):
        """Deprecated; use xray.concat instead"""
        warnings.warn('xray.Dataset.concat has been deprecated; use '
                      'xray.concat instead', FutureWarning, stacklevel=2)
        return cls._concat(*args, **kwargs)

    @classmethod
    def _concat(cls, datasets, dim='concat_dim', indexers=None,
                mode='different', concat_over=None, compat='equals'):
        _assert_compat_valid(compat)

        # don't bother trying to work with datasets as a generator instead of a
        # list; the gains would be minimal
        datasets = list(map(as_dataset, datasets))
        dim_name = getattr(dim, 'name', dim)

        # figure out variables to concatenate over
        if concat_over is None:
            concat_over = set()
        elif isinstance(concat_over, basestring):
            concat_over = set([concat_over])
        else:
            concat_over = set(concat_over)

        # add variables to concat_over depending on the mode
        if mode == 'different':
            def differs(vname, v):
                # simple helper function which compares a variable
                # across all datasets and indicates whether that
                # variable differs or not.
                return any(not ds[vname].variable.equals(v)
                           for ds in datasets[1:])
            non_indexes = iteritems(datasets[0].nonindexes)
            # all nonindexes that are not the same in each dataset
            concat_over.update(k for k, v in non_indexes if differs(k, v))
        elif mode == 'all':
            # concatenate all nonindexes
            concat_over.update(set(datasets[0].nonindexes))
        elif mode == 'minimal':
            # only concatenate variables in which 'dim' already
            # appears. These variables are added later.
            pass
        else:
            raise ValueError("Unexpected value for mode: %s" % mode)

        if any(v not in datasets[0] for v in concat_over):
            raise ValueError('not all elements in concat_over %r found '
                             'in the first dataset %r'
                             % (concat_over, datasets[0]))

        # automatically concatenate over variables along the dimension
        auto_concat_dims = set([dim_name])
        if hasattr(dim, 'dims'):
            auto_concat_dims |= set(dim.dims)
        for k, v in iteritems(datasets[0]._variables):
            if k == dim_name or auto_concat_dims.intersection(v.dims):
                concat_over.add(k)

        # create the new dataset and add constant variables
        concatenated = cls({}, attrs=datasets[0].attrs)
        for k, v in iteritems(datasets[0]._variables):
            if k not in concat_over:
                concatenated[k] = v

        # check that global attributes and non-concatenated variables are fixed
        # across all datasets
        for ds in datasets[1:]:
            if (compat == 'identical'
                    and not utils.dict_equiv(ds.attrs, concatenated.attrs)):
                raise ValueError('dataset global attributes not equal')
            for k, v in iteritems(ds._variables):
                if k not in concatenated and k not in concat_over:
                    raise ValueError('encountered unexpected variable %r' % k)
                elif (k in concatenated and k != dim_name and
                          not getattr(v, compat)(concatenated[k])):
                    verb = 'equal' if compat == 'equals' else compat
                    raise ValueError(
                        'variable %r not %s across datasets' % (k, verb))

        # stack up each variable to fill-out the dataset
        for k in concat_over:
            concatenated[k] = variable.Variable.concat(
                [ds[k] for ds in datasets], dim, indexers)

        if not isinstance(dim, basestring):
            # add dimension last to ensure that its in the final Dataset
            concatenated[dim_name] = dim

        concatenated._coord_names.update(datasets[0].coords)

        return concatenated

    def to_dataframe(self):
        """Convert this dataset into a pandas.DataFrame.

        Non-index variables in this dataset form the columns of the
        DataFrame. The DataFrame is be indexed by the Cartesian product of
        this dataset's indices.
        """
        columns = self.noncoords.keys()
        data = []
        # we need a template to broadcast all dataset variables against
        # using stride_tricks lets us make the ndarray for broadcasting without
        # having to allocate memory
        shape = tuple(self.dims.values())
        empty_data = np.lib.stride_tricks.as_strided(np.array(0), shape=shape,
                                                     strides=[0] * len(shape))
        template = variable.Variable(self.dims.keys(), empty_data)
        for k in columns:
            _, var = variable.broadcast_variables(template, self.variables[k])
            _, var_data = np.broadcast_arrays(template.values, var.values)
            data.append(var_data.reshape(-1))

        index = self.coords.to_index()
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
            dims = [name if name is not None else 'level_%i' % n
                    for n, name in enumerate(idx.names)]
            for dim, lev in zip(dims, idx.levels):
                obj[dim] = (dim, lev)
            shape = [lev.size for lev in idx.levels]
        else:
            dims = (idx.name if idx.name is not None else 'index',)
            obj[dims[0]] = (dims, idx)
            shape = -1

        for name, series in iteritems(dataframe):
            data = series.values.reshape(shape)
            obj[name] = (dims, data)
        return obj

ops.inject_reduce_methods(Dataset)
