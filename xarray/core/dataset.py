import functools
import warnings
from collections import Mapping
from numbers import Number

import numpy as np
import pandas as pd

from . import ops
from . import utils
from . import common
from . import groupby
from . import indexing
from . import alignment
from . import formatting
from .. import conventions
from .alignment import align, align_variables
from .coordinates import DatasetCoordinates, Indexes
from .common import ImplementsDatasetReduce, BaseDataObject
from .merge import merge_datasets, expand_variables
from .utils import Frozen, SortedKeysDict, ChainMap, maybe_wrap_array, hashable
from .variable import (as_variable, Variable, Coordinate, broadcast_variables,
                       default_index_coordinate)
from .pycompat import (iteritems, basestring, OrderedDict,
                       dask_array_type)
from .combine import concat


# list of attributes of pd.DatetimeIndex that are ndarrays of time info
_DATETIMEINDEX_COMPONENTS = ['year', 'month', 'day', 'hour', 'minute',
                             'second', 'microsecond', 'nanosecond', 'date',
                             'time', 'dayofyear', 'weekofyear', 'dayofweek',
                             'quarter']


def _get_virtual_variable(variables, key):
    """Get a virtual variable (e.g., 'time.year') from a dict of
    xarray.Variable objects (if possible)
    """
    if not isinstance(key, basestring):
        raise KeyError(key)

    split_key = key.split('.', 1)
    if len(split_key) != 2:
        raise KeyError(key)

    ref_name, var_name = split_key
    ref_var = variables[ref_name]
    if ref_var.ndim == 1:
        date = ref_var.to_index()
    elif ref_var.ndim == 0:
        date = pd.Timestamp(ref_var.values)
    else:
        raise KeyError(key)

    if var_name == 'season':
        # TODO: move 'season' into pandas itself
        seasons = np.array(['DJF', 'MAM', 'JJA', 'SON'])
        month = date.month
        data = seasons[(month // 3) % 4]
    else:
        data = getattr(date, var_name)
    return ref_name, var_name, Variable(ref_var.dims, data)


def _calculate_dims(variables):
    """Calculate the dimensions corresponding to a set of variables.

    Returns dictionary mapping from dimension names to sizes. Raises ValueError
    if any of the dimension sizes conflict.
    """
    dims = {}
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


def _assert_empty(args, msg='%s'):
    if args:
        raise ValueError(msg % args)


def as_dataset(obj):
    """Cast the given object to a Dataset.

    Handles Datasets, DataArrays and dictionaries of variables. A new Dataset
    object is only created if the provided object is not already one.
    """
    if hasattr(obj, 'to_dataset'):
        obj = obj.to_dataset()
    if not isinstance(obj, Dataset):
        obj = Dataset(obj)
    return obj


class DataVariables(Mapping):
    def __init__(self, dataset):
        self._dataset = dataset

    def __iter__(self):
        return (key for key in self._dataset._variables
                if key not in self._dataset._coord_names)

    def __len__(self):
        return len(self._dataset._variables) - len(self._dataset._coord_names)

    def __contains__(self, key):
        return (key in self._dataset._variables and
                key not in self._dataset._coord_names)

    def __getitem__(self, key):
        if key not in self._dataset._coord_names:
            return self._dataset[key]
        else:
            raise KeyError(key)

    def __repr__(self):
        return formatting.vars_repr(self)


class _LocIndexer(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, key):
        if not utils.is_dict_like(key):
            raise TypeError('can only lookup dictionaries from Dataset.loc')
        return self.dataset.sel(**key)


class Dataset(Mapping, ImplementsDatasetReduce, BaseDataObject):
    """A multi-dimensional, in memory, array database.

    A dataset resembles an in-memory representation of a NetCDF file, and
    consists of variables, coordinates and attributes which together form a
    self describing dataset.

    Dataset implements the mapping interface with keys given by variable names
    and values given by DataArray objects for each variable name.

    One dimensional variables with name equal to their dimension are index
    coordinates used for label based indexing.
    """
    groupby_cls = groupby.DatasetGroupBy

    def __init__(self, data_vars=None, coords=None, attrs=None,
                 compat='broadcast_equals', **kwargs):
        """To load data from a file or file-like object, use the `open_dataset`
        function.

        Parameters
        ----------
        data_vars : dict-like, optional
            A mapping from variable names to :py:class:`~xarray.DataArray`
            objects, :py:class:`~xarray.Variable` objects or tuples of the
            form ``(dims, data[, attrs])`` which can be used as arguments to
            create a new ``Variable``. Each dimension must have the same length
            in all variables in which it appears.
        coords : dict-like, optional
            Another mapping in the same form as the `variables` argument,
            except the each item is saved on the dataset as a "coordinate".
            These variables have an associated meaning: they describe
            constant/fixed/independent quantities, unlike the
            varying/measured/dependent quantities that belong in `variables`.
            Coordinates values may be given by 1-dimensional arrays or scalars,
            in which case `dims` do not need to be supplied: 1D arrays will be
            assumed to give index values along the dimension with the same
            name.
        attrs : dict-like, optional
            Global attributes to save on this dataset.
        compat : {'broadcast_equals', 'equals', 'identical'}, optional
            String indicating how to compare variables of the same name for
            potential conflicts:

            - 'broadcast_equals': all values must be equal when variables are
              broadcast against each other to ensure common dimensions.
            - 'equals': all values and dimensions must be the same.
            - 'identical': all values, dimensions and attributes must be the
              same.
        """
        self._variables = OrderedDict()
        self._coord_names = set()
        self._dims = {}
        self._attrs = None
        self._file_obj = None
        if kwargs:
            if 'variables' in kwargs:
                data_vars = kwargs.pop('variables')
                warnings.warn('Variables kwarg is deprecated. Use data_vars', stacklevel=2)
            if kwargs:
                raise TypeError('{0} are not valid kwargs'.format(kwargs.keys()))
        if data_vars is None:
            data_vars = {}
        if coords is None:
            coords = set()
        if data_vars is not None or coords is not None:
            self._set_init_vars_and_dims(data_vars, coords, compat)
        if attrs is not None:
            self.attrs = attrs
        self._initialized = True

    def _add_missing_coords_inplace(self):
        """Add missing coordinates to self._variables
        """
        for dim, size in iteritems(self.dims):
            if dim not in self._variables:
                self._variables[dim] = default_index_coordinate(dim, size)

    def _update_vars_and_coords(self, new_variables, new_coord_names=None,
                                needs_copy=True, check_coord_names=True):
        """Add a dictionary of new variables to this dataset.

        Raises a ValueError if any dimensions have conflicting lengths in the
        new dataset. Otherwise will update this dataset's _variables and
        _dims attributes in-place.

        Set `needs_copy=False` only if this dataset is brand-new and hence
        can be thrown away if this method fails.
        """
        if new_coord_names is None:
            new_coord_names = {}
        # default to creating another copy of variables so can unroll if we end
        # up with inconsistent dimensions
        variables = self._variables.copy() if needs_copy else self._variables

        if check_coord_names:
            _assert_empty([k for k in self.data_vars if k in new_coord_names],
                          'coordinates with these names already exist as '
                          'variables: %s')

        variables.update(new_variables)
        dims = _calculate_dims(variables)
        # all checks are complete: it's safe to update
        self._variables = variables
        self._dims = dims
        self._add_missing_coords_inplace()
        self._coord_names.update(new_coord_names)

    def _set_init_vars_and_dims(self, vars, coords, compat):
        """Set the initial value of Dataset variables and dimensions
        """
        _assert_empty([k for k in vars if k in coords],
                      'redundant variables and coordinates: %s')

        # Need to coerce Series to a dict as it doesn't implement __iter__ like a mapping
        # GH470
        vars = dict(vars) if isinstance(vars, pd.Series) else vars

        variables = ChainMap(vars, coords)

        aligned = align_variables(variables)
        new_variables, new_coord_names = expand_variables(aligned,
                                                          compat=compat)

        new_coord_names.update(coords)
        self._update_vars_and_coords(new_variables, new_coord_names,
                                     needs_copy=False, check_coord_names=False)

    @classmethod
    def load_store(cls, store, decoder=None):
        """Create a new dataset from the contents of a backends.*DataStore
        object
        """
        variables, attributes = store.load()
        if decoder:
            variables, attributes = decoder(variables, attributes)
        obj = cls(variables, attrs=attributes)
        obj._file_obj = store
        return obj

    def close(self):
        """Close any files linked to this dataset
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
        self.load()
        # self.__dict__ is the default pickle object, we don't need to
        # implement our own __setstate__ method to make pickle work
        state = self.__dict__.copy()
        # throw away any references to datastores in the pickle
        state['_file_obj'] = None
        return state

    @property
    def variables(self):
        """Frozen dictionary of xarray.Variable objects constituting this
        dataset's data
        """
        return Frozen(self._variables)

    def _attrs_copy(self):
        return None if self._attrs is None else OrderedDict(self._attrs)

    @property
    def attrs(self):
        """Dictionary of global attributes on this dataset
        """
        if self._attrs is None:
            self._attrs = OrderedDict()
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
        return Frozen(SortedKeysDict(self._dims))

    def load(self):
        """Manually trigger loading of this dataset's data from disk or a
        remote source into memory and return this dataset.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.
        """
        # access .data to coerce everything to numpy or dask arrays
        all_data = dict((k, v.data) for k, v in self.variables.items())
        lazy_data = dict((k, v) for k, v in all_data.items()
                         if isinstance(v, dask_array_type))
        if lazy_data:
            import dask.array as da

            # evaluate all the dask arrays simultaneously
            evaluated_data = da.compute(*lazy_data.values())

            for k, data in zip(lazy_data, evaluated_data):
                self.variables[k].data = data

        return self

    def load_data(self):  # pragma: no cover
        warnings.warn('the Dataset method `load_data` has been deprecated; '
                      'use `load` instead',
                      FutureWarning, stacklevel=2)
        return self.load()

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
        obj._initialized = True
        return obj

    __default_attrs = object()

    def _replace_vars_and_dims(self, variables, coord_names=None,
                               attrs=__default_attrs, inplace=False):
        """Fastpath constructor for internal use.

        Preserves coord names and attributes; dimensions are recalculated from
        the supplied variables.

        The arguments are *not* copied when placed on the new dataset. It is up
        to the caller to ensure that they have the right type and are not used
        elsewhere.

        Parameters
        ----------
        variables : OrderedDict
        coord_names : set or None, optional
        attrs : OrderedDict or None, optional

        Returns
        -------
        new : Dataset
        """
        dims = _calculate_dims(variables)
        if inplace:
            self._dims = dims
            self._variables = variables
            if coord_names is not None:
                self._coord_names = coord_names
            if attrs is not self.__default_attrs:
                self._attrs = attrs
            obj = self
        else:
            if coord_names is None:
                coord_names = self._coord_names.copy()
            if attrs is self.__default_attrs:
                attrs = self._attrs_copy()
            obj = self._construct_direct(variables, coord_names, dims, attrs)
        return obj

    def copy(self, deep=False):
        """Returns a copy of this dataset.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy is made, so each variable in the new dataset
        is also a variable in the original dataset.
        """
        if deep:
            variables = OrderedDict((k, v.copy(deep=True))
                                    for k, v in iteritems(self._variables))
        else:
            variables = self._variables.copy()
        # skip __init__ to avoid costly validation
        return self._construct_direct(variables, self._coord_names.copy(),
                                      self._dims.copy(), self._attrs_copy())

    def _subset_with_all_valid_coords(self, variables, coord_names, attrs):
        needed_dims = set()
        for v in variables.values():
            needed_dims.update(v.dims)
        for k in self._coord_names:
            if set(self.variables[k].dims) <= needed_dims:
                variables[k] = self._variables[k]
                coord_names.add(k)
        dims = dict((k, self._dims[k]) for k in needed_dims)

        return self._construct_direct(variables, coord_names, dims, attrs)

    def _copy_listed(self, names):
        """Create a new Dataset with the listed variables from this dataset and
        the all relevant coordinates. Skips all validation.
        """
        variables = OrderedDict()
        coord_names = set()

        for name in names:
            try:
                variables[name] = self._variables[name]
            except KeyError:
                ref_name, var_name, var = _get_virtual_variable(
                    self._variables, name)
                variables[var_name] = var
                if ref_name in self._coord_names:
                    coord_names.add(var_name)

        return self._subset_with_all_valid_coords(variables, coord_names,
                                                  attrs=self.attrs.copy())

    def _construct_dataarray(self, name):
        """Construct a DataArray by indexing this dataset
        """
        from .dataarray import DataArray

        try:
            variable = self._variables[name]
        except KeyError:
            _, name, variable = _get_virtual_variable(self._variables, name)

        coords = OrderedDict()
        needed_dims = set(variable.dims)
        for k in self.coords:
            if set(self.variables[k].dims) <= needed_dims:
                coords[k] = self.variables[k]

        return DataArray(variable, coords, name=name, fastpath=True)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        # memo does nothing but is required for compatibility with
        # copy.deepcopy
        return self.copy(deep=True)

    def __contains__(self, key):
        """The 'in' operator will return true or false depending on whether
        'key' is an array in the dataset or not.
        """
        return key in self._variables

    def __len__(self):
        return len(self._variables)

    def __iter__(self):
        return iter(self._variables)

    @property
    def nbytes(self):
        return sum(v.nbytes for v in self.variables.values())

    @property
    def loc(self):
        """Attribute for location based indexing. Only supports __getitem__,
        and only when the key is a dict of the form {dim: labels}.
        """
        return _LocIndexer(self)

    def __getitem__(self, key):
        """Access variables or coordinates this dataset as a
        :py:class:`~xarray.DataArray`.

        Indexing with a list of names will return a new ``Dataset`` object.
        """
        if utils.is_dict_like(key):
            return self.isel(**key)

        if hashable(key):
            return self._construct_dataarray(key)
        else:
            return self._copy_listed(np.asarray(key))

    def __setitem__(self, key, value):
        """Add an array to this dataset.

        If value is a `DataArray`, call its `select_vars()` method, rename it
        to `key` and merge the contents of the resulting dataset into this
        dataset.

        If value is an `Variable` object (or tuple of form
        ``(dims, data[, attrs])``), add it to this dataset as a new
        variable.
        """
        if utils.is_dict_like(key):
            raise NotImplementedError('cannot yet use a dictionary as a key '
                                      'to set Dataset values')
        self.update({key: value})

    def __delitem__(self, key):
        """Remove a variable from this dataset.

        If this variable is a dimension, all variables containing this
        dimension are also removed.
        """
        def remove(k):
            del self._variables[k]
            self._coord_names.discard(k)

        remove(key)

        if key in self._dims:
            del self._dims[key]
            also_delete = [k for k, v in iteritems(self._variables)
                           if key in v.dims]
            for key in also_delete:
                remove(key)

    # mutable objects should not be hashable
    __hash__ = None

    def _all_compat(self, other, compat_str):
        """Helper function for equals and identical"""
        # some stores (e.g., scipy) do not seem to preserve order, so don't
        # require matching order for equality
        def compat(x, y):
            return getattr(x, compat_str)(y)
        return (self._coord_names == other._coord_names and
                utils.dict_equiv(self._variables, other._variables,
                                 compat=compat))

    def broadcast_equals(self, other):
        """Two Datasets are broadcast equal if they are equal after
        broadcasting all variables against each other.

        For example, variables that are scalar in one dataset but non-scalar in
        the other dataset can still be broadcast equal if the the non-scalar
        variable is a constant.

        See Also
        --------
        Dataset.equals
        Dataset.identical
        """
        try:
            return self._all_compat(other, 'broadcast_equals')
        except (TypeError, AttributeError):
            return False

    def equals(self, other):
        """Two Datasets are equal if they have matching variables and
        coordinates, all of which are equal.

        Datasets can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for ``Dataset``
        does element-wise comparisons (like numpy.ndarrays).

        See Also
        --------
        Dataset.broadcast_equals
        Dataset.identical
        """
        try:
            return self._all_compat(other, 'equals')
        except (TypeError, AttributeError):
            return False

    def identical(self, other):
        """Like equals, but also checks all dataset attributes and the
        attributes on all variables and coordinates.

        See Also
        --------
        Dataset.broadcast_equals
        Dataset.equals
        """
        try:
            return (utils.dict_equiv(self.attrs, other.attrs) and
                    self._all_compat(other, 'identical'))
        except (TypeError, AttributeError):
            return False

    @property
    def indexes(self):
        """OrderedDict of pandas.Index objects used for label based indexing
        """
        return Indexes(self)

    @property
    def coords(self):
        """Dictionary of xarray.DataArray objects corresponding to coordinate
        variables
        """
        return DatasetCoordinates(self)

    @property
    def data_vars(self):
        """Dictionary of xarray.DataArray objects corresponding to data variables
        """
        return DataVariables(self)

    @property
    def vars(self):  # pragma: no cover
        warnings.warn('the Dataset property `vars` has been deprecated; '
                      'use `data_vars` instead',
                      FutureWarning, stacklevel=2)
        return self.data_vars

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
        # nb. check in self._variables, not self.data_vars to insure that the
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
            Name(s) of non-index coordinates in this dataset to reset into
            variables. By default, all non-index coordinates are reset.
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

    def dump_to_store(self, store, encoder=None, sync=True, encoding=None):
        """Store dataset contents to a backends.*DataStore object."""
        if encoding is None:
            encoding = {}
        variables, attrs = conventions.encode_dataset_coordinates(self)

        check_encoding = set()
        for k, enc in encoding.items():
            # no need to shallow copy the variable again; that already happened
            # in encode_dataset_coordinates
            variables[k].encoding = enc
            check_encoding.add(k)

        if encoder:
            variables, attrs = encoder(variables, attrs)

        store.store(variables, attrs, check_encoding)
        if sync:
            store.sync()

    def to_netcdf(self, path=None, mode='w', format=None, group=None,
                  engine=None, encoding=None):
        """Write dataset contents to a netCDF file.

        Parameters
        ----------
        path : str, optional
            Path to which to save this dataset. If no path is provided, this
            function returns the resulting netCDF file as a bytes object; in
            this case, we need to use scipy.io.netcdf, which does not support
            netCDF version 4 (the default format becomes NETCDF3_64BIT).
        mode : {'w', 'a'}, optional
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            this location will be overwritten.
        format : {'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT', 'NETCDF3_CLASSIC'}, optional
            File format for the resulting netCDF file:

            * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API
              features.
            * NETCDF4_CLASSIC: Data is stored in an HDF5 file, using only
              netCDF 3 compatible API features.
            * NETCDF3_64BIT: 64-bit offset version of the netCDF 3 file format,
              which fully supports 2+ GB files, but is only compatible with
              clients linked against netCDF version 3.6.0 or later.
            * NETCDF3_CLASSIC: The classic netCDF 3 file format. It does not
              handle 2+ GB files very well.

            All formats are supported by the netCDF4-python library.
            scipy.io.netcdf only supports the last two formats.

            The default format is NETCDF4 if you are saving a file to disk and
            have the netCDF4-python library available. Otherwise, xarray falls
            back to using scipy to write netCDF files and defaults to the
            NETCDF3_64BIT format (scipy does not support netCDF4).
        group : str, optional
            Path to the netCDF4 group in the given file to open (only works for
            format='NETCDF4'). The group(s) will be created if necessary.
        engine : {'netcdf4', 'scipy', 'h5netcdf'}, optional
            Engine to use when writing netCDF files. If not provided, the
            default engine is chosen based on available dependencies, with a
            preference for 'netcdf4' if writing to a file on disk.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{'my_variable': {'dtype': 'int16', 'scale_factor': 0.1, 'zlib': True}, ...}``
        """
        if encoding is None:
            encoding = {}
        from ..backends.api import to_netcdf
        return to_netcdf(self, path, mode, format=format, group=group,
                         engine=engine, encoding=encoding)

    dump = utils.function_alias(to_netcdf, 'dump')
    dumps = utils.function_alias(to_netcdf, 'dumps')

    def __repr__(self):
        return formatting.dataset_repr(self)

    @property
    def chunks(self):
        """Block dimensions for this dataset's data or None if it's not a dask
        array.
        """
        chunks = {}
        for v in self.variables.values():
            if v.chunks is not None:
                new_chunks = list(zip(v.dims, v.chunks))
                if any(chunk != chunks[d] for d, chunk in new_chunks
                       if d in chunks):
                    raise ValueError('inconsistent chunks')
                chunks.update(new_chunks)
        return Frozen(SortedKeysDict(chunks))

    def chunk(self, chunks=None, name_prefix='xarray-', token=None,
              lock=False):
        """Coerce all arrays in this dataset into dask arrays with the given
        chunks.

        Non-dask arrays in this dataset will be converted to dask arrays. Dask
        arrays will be rechunked to the given chunk sizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

        Parameters
        ----------
        chunks : int or dict, optional
            Chunk sizes along each dimension, e.g., ``5`` or
            ``{'x': 5, 'y': 5}``.
        name_prefix : str, optional
            Prefix for the name of any new dask arrays.
        token : str, optional
            Token uniquely identifying this dataset.
        lock : optional
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.

        Returns
        -------
        chunked : xarray.Dataset
        """
        try:
            from dask.base import tokenize
        except ImportError:
            import dask  # raise the usual error if dask is entirely missing
            raise ImportError('xarray requires dask version 0.6 or newer')

        if isinstance(chunks, Number):
            chunks = dict.fromkeys(self.dims, chunks)

        if chunks is not None:
            bad_dims = [d for d in chunks if d not in self.dims]
            if bad_dims:
                raise ValueError('some chunks keys are not dimensions on this '
                                 'object: %s' % bad_dims)

        def selkeys(dict_, keys):
            if dict_ is None:
                return None
            return dict((d, dict_[d]) for d in keys if d in dict_)

        def maybe_chunk(name, var, chunks):
            chunks = selkeys(chunks, var.dims)
            if not chunks:
                chunks = None
            if var.ndim > 0:
                token2 = tokenize(name, token if token else var._data)
                name2 = '%s%s-%s' % (name_prefix, name, token2)
                return var.chunk(chunks, name=name2, lock=lock)
            else:
                return var

        variables = OrderedDict([(k, maybe_chunk(k, v, chunks))
                                 for k, v in self.variables.items()])
        return self._replace_vars_and_dims(variables)

    def isel(self, **indexers):
        """Returns a new dataset with each array indexed along the specified
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
        Dataset.sel_points
        Dataset.isel_points
        DataArray.isel
        """
        invalid = [k for k in indexers if k not in self.dims]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        # all indexers should be int, slice or np.ndarrays
        indexers = [(k, (np.asarray(v)
                         if not isinstance(v, (int, np.integer, slice))
                         else v))
                    for k, v in iteritems(indexers)]

        variables = OrderedDict()
        for name, var in iteritems(self._variables):
            var_indexers = dict((k, v) for k, v in indexers if k in var.dims)
            variables[name] = var.isel(**var_indexers)
        return self._replace_vars_and_dims(variables)

    def sel(self, method=None, tolerance=None, **indexers):
        """Returns a new dataset with each array indexed by tick labels
        along the specified dimension(s).

        In contrast to `Dataset.isel`, indexers for this method should use
        labels instead of integers.

        Under the hood, this method is powered by using pandas's powerful Index
        objects. This makes label based indexing essentially just as fast as
        using integer indexing.

        It also means this method uses pandas's (well documented) logic for
        indexing. This means you can use string shortcuts for datetime indexes
        (e.g., '2000-01' to select all values in January 2000). It also means
        that slices are treated as inclusive of both the start and stop values,
        unlike normal Python indexing.

        Parameters
        ----------
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for inexact matches (requires pandas>=0.16):

            * None (default): only exact matches
            * pad / ffill: propgate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations most
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Requires pandas>=0.17.
        **indexers : {dim: indexer, ...}
            Keyword arguments with names matching dimensions and values given
            by scalars, slices or arrays of tick labels.

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
        Dataset.sel_points
        Dataset.isel_points
        DataArray.sel
        """
        return self.isel(**indexing.remap_label_indexers(
            self, indexers, method=method, tolerance=tolerance))

    def isel_points(self, dim='points', **indexers):
        """Returns a new dataset with each array indexed pointwise along the
        specified dimension(s).

        This method selects pointwise values from each array and is akin to
        the NumPy indexing behavior of `arr[[0, 1], [0, 1]]`, except this
        method does not require knowing the order of each array's dimensions.

        Parameters
        ----------
        dim : str or DataArray or pandas.Index or other list-like object, optional
            Name of the dimension to concatenate along. If dim is provided as a
            string, it must be a new dimension name, in which case it is added
            along axis=0. If dim is provided as a DataArray or Index or
            list-like object, its name, which must not be present in the
            dataset, is used as the dimension to concatenate along and the
            values are added as a coordinate.
        **indexers : {dim: indexer, ...}
            Keyword arguments with names matching dimensions and values given
            by array-like objects. All indexers must be the same length and
            1 dimensional.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array and dimension is indexed by the appropriate indexers. With
            pointwise indexing, the new Dataset will always be a copy of the
            original.

        See Also
        --------
        Dataset.sel
        Dataset.isel
        Dataset.sel_points
        DataArray.isel_points
        """
        indexer_dims = set(indexers)

        def relevant_keys(mapping):
            return [k for k, v in mapping.items()
                    if any(d in indexer_dims for d in v.dims)]

        data_vars = relevant_keys(self.data_vars)
        coords = relevant_keys(self.coords)

        # all the indexers should be iterables
        keys = indexers.keys()
        indexers = [(k, np.asarray(v)) for k, v in iteritems(indexers)]
        # Check that indexers are valid dims, integers, and 1D
        for k, v in indexers:
            if k not in self.dims:
                raise ValueError("dimension %s does not exist" % k)
            if v.dtype.kind != 'i':
                raise TypeError('Indexers must be integers')
            if v.ndim != 1:
                raise ValueError('Indexers must be 1 dimensional')

        # all the indexers should have the same length
        lengths = set(len(v) for k, v in indexers)
        if len(lengths) > 1:
            raise ValueError('All indexers must be the same length')

        # Existing dimensions are not valid choices for the dim argument
        if isinstance(dim, basestring):
            if dim in self.dims:
                # dim is an invalid string
                raise ValueError('Existing dimension names are not valid '
                                 'choices for the dim argument in sel_points')
        elif hasattr(dim, 'dims'):
            # dim is a DataArray or Coordinate
            if dim.name in self.dims:
                # dim already exists
                raise ValueError('Existing dimensions are not valid choices '
                                 'for the dim argument in sel_points')
        else:
            # try to cast dim to DataArray with name = points
            from .dataarray import DataArray
            dim = DataArray(dim, dims='points', name='points')

        # TODO: This would be sped up with vectorized indexing. This will
        # require dask to support pointwise indexing as well.
        return concat([self.isel(**d) for d in
                       [dict(zip(keys, inds)) for inds in
                        zip(*[v for k, v in indexers])]],
                      dim=dim, coords=coords, data_vars=data_vars)

    def sel_points(self, dim='points', method=None, tolerance=None,
                   **indexers):
        """Returns a new dataset with each array indexed pointwise by tick
        labels along the specified dimension(s).

        In contrast to `Dataset.isel_points`, indexers for this method should
        use labels instead of integers.

        In contrast to `Dataset.sel`, this method selects points along the
        diagonal of multi-dimensional arrays, not the intersection.

        Parameters
        ----------
        dim : str or DataArray or pandas.Index or other list-like object, optional
            Name of the dimension to concatenate along. If dim is provided as a
            string, it must be a new dimension name, in which case it is added
            along axis=0. If dim is provided as a DataArray or Index or
            list-like object, its name, which must not be present in the
            dataset, is used as the dimension to concatenate along and the
            values are added as a coordinate.
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for inexact matches (requires pandas>=0.16):

            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations most
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Requires pandas>=0.17.
        **indexers : {dim: indexer, ...}
            Keyword arguments with names matching dimensions and values given
            by array-like objects. All indexers must be the same length and
            1 dimensional.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array and dimension is indexed by the appropriate indexers. With
            pointwise indexing, the new Dataset will always be a copy of the
            original.

        See Also
        --------
        Dataset.sel
        Dataset.isel
        Dataset.isel_points
        DataArray.sel_points
        """
        pos_indexers = indexing.remap_label_indexers(
            self, indexers, method=method, tolerance=tolerance)
        return self.isel_points(dim=dim, **pos_indexers)

    def reindex_like(self, other, method=None, tolerance=None, copy=True):
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
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for filling index values from other not found in this
            dataset:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value (requires pandas>=0.16)
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations most
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Requires pandas>=0.17.
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
        indexers = dict((k, v) for k, v in other.indexes.items()
                        if k in self.dims)
        return self.reindex(method=method, copy=copy, tolerance=tolerance,
                            **indexers)

    def reindex(self, indexers=None, method=None, tolerance=None, copy=True, **kw_indexers):
        """Conform this object onto a new set of indexes, filling in
        missing values with NaN.

        Parameters
        ----------
        indexers : dict. optional
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate values
            will be filled in with NaN, and any mis-matched dimension names will
            simply be ignored.
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for filling index values in ``indexers`` not found in
            this dataset:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value (requires pandas>=0.16)
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations most
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Requires pandas>=0.17.
        copy : bool, optional
            If `copy=True`, the returned dataset contains only copied
            variables. If `copy=False` and no reindexing is required then
            original variables from this dataset are returned.
        **kw_indexers : optional
            Keyword arguments in the same form as ``indexers``.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.reindex_like
        align
        pandas.Index.get_indexer
        """
        indexers = utils.combine_pos_and_kw_args(indexers, kw_indexers,
                                                 'reindex')
        if not indexers:
            # shortcut
            return self.copy(deep=True) if copy else self

        bad_dims = [d for d in indexers if d not in self.dims]
        if bad_dims:
            raise ValueError('invalid reindex dimensions: %s' % bad_dims)

        variables = alignment.reindex_variables(
            self.variables, self.indexes, indexers, method, tolerance, copy=copy)
        return self._replace_vars_and_dims(variables)

    def rename(self, name_dict, inplace=False):
        """Returns a new object with renamed variables and dimensions.

        Parameters
        ----------
        name_dict : dict-like
            Dictionary whose keys are current variable or dimension names and
            whose values are the desired names.
        inplace : bool, optional
            If True, rename variables and dimensions in-place. Otherwise,
            return a new dataset object.

        Returns
        -------
        renamed : Dataset
            Dataset with renamed variables and dimensions.

        See Also
        --------

        Dataset.swap_dims
        DataArray.rename
        """
        for k, v in name_dict.items():
            if k not in self:
                raise ValueError("cannot rename %r because it is not a "
                                 "variable in this dataset" % k)
            if v in self and k != v:
                raise ValueError('the new name %r already exists' % v)

        variables = OrderedDict()
        coord_names = set()
        for k, v in iteritems(self._variables):
            name = name_dict.get(k, k)
            dims = tuple(name_dict.get(dim, dim) for dim in v.dims)
            var = v.copy(deep=False)
            var.dims = dims
            variables[name] = var
            if k in self._coord_names:
                coord_names.add(name)

        return self._replace_vars_and_dims(variables, coord_names,
                                           inplace=inplace)

    def swap_dims(self, dims_dict, inplace=False):
        """Returns a new object with swapped dimensions.

        Parameters
        ----------
        dims_dict : dict-like
            Dictionary whose keys are current dimension names and whose values
            are new names. Each value must already be a variable in the
            dataset.
        inplace : bool, optional
            If True, swap dimensions in-place. Otherwise, return a new dataset
            object.

        Returns
        -------
        renamed : Dataset
            Dataset with swapped dimensions.

        See Also
        --------

        Dataset.rename
        DataArray.swap_dims
        """
        for k, v in dims_dict.items():
            if k not in self.dims:
                raise ValueError('cannot swap from dimension %r because it is '
                                 'not an existing dimension' % k)
            if self.variables[v].dims != (k,):
                raise ValueError('replacement dimension %r is not a 1D '
                                 'variable along the old dimension %r'
                                 % (v, k))

        result_dims = set(dims_dict.get(dim, dim) for dim in self.dims)

        variables = OrderedDict()

        coord_names = self._coord_names.copy()
        coord_names.update(dims_dict.values())

        for k, v in iteritems(self.variables):
            dims = tuple(dims_dict.get(dim, dim) for dim in v.dims)
            var = v.to_coord() if k in result_dims else v.to_variable()
            var.dims = dims
            variables[k] = var

        return self._replace_vars_and_dims(variables, coord_names,
                                           inplace=inplace)

    def _stack_once(self, dims, new_dim):
        variables = OrderedDict()
        for name, var in self.variables.items():
            if name not in dims:
                if any(d in var.dims for d in dims):
                    add_dims = [d for d in dims if d not in var.dims]
                    vdims = list(var.dims) + add_dims
                    shape = [self.dims[d] for d in vdims]
                    exp_var = var.expand_dims(vdims, shape)
                    stacked_var = exp_var.stack(**{new_dim: dims})
                    variables[name] = stacked_var
                else:
                    variables[name] = var.copy(deep=False)

        idx = pd.MultiIndex.from_product([self.indexes[d] for d in dims],
                                         names=dims)
        variables[new_dim] = Coordinate(new_dim, idx)

        coord_names = set(self._coord_names) - set(dims) | set([new_dim])

        return self._replace_vars_and_dims(variables, coord_names)

    def stack(self, **dimensions):
        """
        Stack any number of existing dimensions into a single new dimension.

        New dimensions will be added at the end, and the corresponding
        coordinate variables will be combined into a MultiIndex.

        Parameters
        ----------
        **dimensions : keyword arguments of the form new_name=(dim1, dim2, ...)
            Names of new dimensions, and the existing dimensions that they
            replace.

        Returns
        -------
        stacked : Dataset
            Dataset with stacked data.

        See also
        --------
        Dataset.unstack
        """
        result = self
        for new_dim, dims in dimensions.items():
            result = result._stack_once(dims, new_dim)
        return result

    def unstack(self, dim):
        """
        Unstack an existing dimension corresponding to a MultiIndex into
        multiple new dimensions.

        New dimensions will be added at the end.

        Parameters
        ----------
        dim : str
            Name of the existing dimension to unstack.

        Returns
        -------
        unstacked : Dataset
            Dataset with unstacked data.

        See also
        --------
        Dataset.stack
        """
        if dim not in self.dims:
            raise ValueError('invalid dimension: %s' % dim)

        index = self.indexes[dim]
        if not isinstance(index, pd.MultiIndex):
            raise ValueError('cannot unstack a dimension that does not have '
                             'a MultiIndex')

        full_idx = pd.MultiIndex.from_product(index.levels, names=index.names)
        obj = self.reindex(copy=False, **{dim: full_idx})

        new_dim_names = index.names
        if any(name is None for name in new_dim_names):
            raise ValueError('cannot unstack dimension with unnamed levels')

        new_dim_sizes = [lev.size for lev in index.levels]

        variables = OrderedDict()
        for name, var in obj.variables.items():
            if name != dim:
                if dim in var.dims:
                    new_dims = OrderedDict(zip(new_dim_names, new_dim_sizes))
                    variables[name] = var.unstack(**{dim: new_dims})
                else:
                    variables[name] = var

        for name, lev in zip(new_dim_names, index.levels):
            variables[name] = Coordinate(name, lev)

        coord_names = set(self._coord_names) - set([dim]) | set(new_dim_names)

        return self._replace_vars_and_dims(variables, coord_names)

    def update(self, other, inplace=True):
        """Update this dataset's variables with those from another dataset.

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
        return self.merge(
            other, inplace=inplace, overwrite_vars=list(other), join='left')

    def merge(self, other, inplace=False, overwrite_vars=set(),
              compat='broadcast_equals', join='outer'):
        """Merge the arrays of two datasets into a single dataset.

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
        compat : {'broadcast_equals', 'equals', 'identical'}, optional
            String indicating how to compare variables of the same name for
            potential conflicts:

            - 'broadcast_equals': all values must be equal when variables are
              broadcast against each other to ensure common dimensions.
            - 'equals': all values and dimensions must be the same.
            - 'identical': all values, dimensions and attributes must be the
              same.
        join : {'outer', 'inner', 'left', 'right'}, optional
            Method for joining ``self`` and ``other`` along shared dimensions:

            - 'outer': use the union of the indexes
            - 'inner': use the intersection of the indexes
            - 'left': use indexes from ``self``
            - 'right': use indexes from ``other``

        Returns
        -------
        merged : Dataset
            Merged dataset.

        Raises
        ------
        ValueError
            If any variables conflict (see ``compat``).
        """
        replace_vars, new_coord_names = merge_datasets(
            self, other, overwrite_vars, compat=compat, join=join)
        obj = self if inplace else self.copy()
        obj._update_vars_and_coords(replace_vars, new_coord_names)
        return obj

    def _assert_all_in_dataset(self, names, virtual_okay=False):
        bad_names = set(names) - set(self._variables)
        if virtual_okay:
            bad_names -= self.virtual_variables
        if bad_names:
            raise ValueError('One or more of the specified variables '
                             'cannot be found in this dataset')

    def drop(self, labels, dim=None):
        """Drop variables or index labels from this dataset.

        If a variable corresponding to a dimension is dropped, all variables
        that use that dimension are also dropped.

        Parameters
        ----------
        labels : str
            Names of variables or index labels to drop.
        dim : None or str, optional
            Dimension along which to drop index labels. By default (if
            ``dim is None``), drops variables rather than index labels.

        Returns
        -------
        dropped : Dataset
        """
        if utils.is_scalar(labels):
            labels = [labels]
        if dim is None:
            return self._drop_vars(labels)
        else:
            new_index = self.indexes[dim].drop(labels)
            return self.loc[{dim: new_index}]

    def _drop_vars(self, names):
        self._assert_all_in_dataset(names)
        drop = set(names)
        drop |= set(k for k, v in iteritems(self._variables)
                    if any(name in v.dims for name in names))
        variables = OrderedDict((k, v) for k, v in iteritems(self._variables)
                                if k not in drop)
        coord_names = set(k for k in self._coord_names if k in variables)
        return self._replace_vars_and_dims(variables, coord_names)

    def drop_vars(self, *names):  # pragma: no cover
        warnings.warn('the Dataset method `drop_vars` has been deprecated; '
                      'use `drop` instead',
                      FutureWarning, stacklevel=2)
        return self.drop(names)

    def transpose(self, *dims):
        """Return a new Dataset object with all array dimensions transposed.

        Although the order of dimensions on each array will change, the dataset
        dimensions themselves will remain in fixed (sorted) order.

        Parameters
        ----------
        *dims : str, optional
            By default, reverse the dimensions on each array. Otherwise,
            reorder the dimensions to this order.

        Returns
        -------
        transposed : Dataset
            Each array in the dataset (including) coordinates will be
            transposed to the given order.

        Notes
        -----
        Although this operation returns a view of each array's data, it
        is not lazy -- the data will be fully loaded into memory.

        See Also
        --------
        numpy.transpose
        DataArray.transpose
        """
        if dims:
            if set(dims) ^ set(self.dims):
                raise ValueError('arguments to transpose (%s) must be '
                                 'permuted dataset dimensions (%s)'
                                 % (dims, tuple(self.dims)))
        ds = self.copy()
        for name, var in iteritems(self._variables):
            var_dims = tuple(dim for dim in dims if dim in var.dims)
            ds._variables[name] = var.transpose(*var_dims)
        return ds

    @property
    def T(self):
        return self.transpose()

    def squeeze(self, dim=None):
        """Returns a new dataset with squeezed data.

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

    def dropna(self, dim, how='any', thresh=None, subset=None):
        """Returns a new dataset with dropped labels for missing values along
        the provided dimension.

        Parameters
        ----------
        dim : str
            Dimension along which to drop missing values. Dropping along
            multiple dimensions simultaneously is not yet supported.
        how : {'any', 'all'}, optional
            * any : if any NA values are present, drop that label
            * all : if all values are NA, drop that label
        thresh : int, default None
            If supplied, require this many non-NA values.
        subset : sequence, optional
            Subset of variables to check for missing values. By default, all
            variables in the dataset are checked.

        Returns
        -------
        Dataset
        """
        # TODO: consider supporting multiple dimensions? Or not, given that
        # there are some ugly edge cases, e.g., pandas's dropna differs
        # depending on the order of the supplied axes.

        if dim not in self.dims:
            raise ValueError('%s must be a single dataset dimension' % dim)

        if subset is None:
            subset = list(self.data_vars)

        count = np.zeros(self.dims[dim], dtype=np.int64)
        size = 0

        for k in subset:
            array = self._variables[k]
            if dim in array.dims:
                dims = [d for d in array.dims if d != dim]
                count += array.count(dims)
                size += np.prod([self.dims[d] for d in dims])

        if thresh is not None:
            mask = count >= thresh
        elif how == 'any':
            mask = count == size
        elif how == 'all':
            mask = count > 0
        elif how is not None:
            raise ValueError('invalid how option: %s' % how)
        else:
            raise TypeError('must specify how or thresh')

        return self.isel(**{dim: mask})

    def fillna(self, value):
        """Fill missing values in this object.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic, except the result is aligned to this
        object (``join='left'``) instead of aligned to the intersection of
        index coordinates (``join='inner'``).

        Parameters
        ----------
        value : scalar, ndarray, DataArray, dict or Dataset
            Used to fill all matching missing values in this dataset's data
            variables. Scalars, ndarrays or DataArrays arguments are used to
            fill all data with aligned coordinates (for DataArrays).
            Dictionaries or datasets match data variables and then align
            coordinates if necessary.

        Returns
        -------
        Dataset
        """
        return self._fillna(value)

    def reduce(self, func, dim=None, keep_attrs=False, numeric_only=False,
               allow_lazy=False, **kwargs):
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
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        numeric_only : bool, optional
            If True, only apply ``func`` to variables with a numeric dtype.
        **kwargs : dict
            Additional keyword arguments passed on to ``func``.

        Returns
        -------
        reduced : Dataset
            Dataset with this object's DataArrays replaced with new DataArrays
            of summarized data and the indicated dimension(s) removed.
        """
        if isinstance(dim, basestring):
            dims = set([dim])
        elif dim is None:
            dims = set(self.dims)
        else:
            dims = set(dim)

        _assert_empty([dim for dim in dims if dim not in self.dims],
                      'Dataset does not contain the dimensions: %s')

        variables = OrderedDict()
        for name, var in iteritems(self._variables):
            reduce_dims = [dim for dim in var.dims if dim in dims]
            if reduce_dims or not var.dims:
                if name not in self.coords:
                    if (not numeric_only or
                        np.issubdtype(var.dtype, np.number) or
                            var.dtype == np.bool_):
                        if len(reduce_dims) == 1:
                            # unpack dimensions for the benefit of functions
                            # like np.argmin which can't handle tuple arguments
                            reduce_dims, = reduce_dims
                        elif len(reduce_dims) == var.ndim:
                            # prefer to aggregate over axis=None rather than
                            # axis=(0, 1) if they will be equivalent, because
                            # the former is often more efficient
                            reduce_dims = None
                        variables[name] = var.reduce(func, dim=reduce_dims,
                                                     keep_attrs=keep_attrs,
                                                     allow_lazy=allow_lazy,
                                                     **kwargs)
            else:
                variables[name] = var

        coord_names = set(k for k in self.coords if k in variables)
        attrs = self.attrs if keep_attrs else None
        return self._replace_vars_and_dims(variables, coord_names, attrs)

    def apply(self, func, keep_attrs=False, args=(), **kwargs):
        """Apply a function over the data variables in this dataset.

        Parameters
        ----------
        func : function
            Function which can be called in the form `f(x, **kwargs)` to
            transform each DataArray `x` in this dataset into another
            DataArray.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one. If False, the new object will
            be returned without attributes.
        args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : dict
            Keyword arguments passed on to `func`.

        Returns
        -------
        applied : Dataset
            Resulting dataset from applying ``func`` over each data variable.
        """
        variables = OrderedDict(
            (k, maybe_wrap_array(v, func(v, *args, **kwargs)))
            for k, v in iteritems(self.data_vars))
        attrs = self.attrs if keep_attrs else None
        return type(self)(variables, attrs=attrs)

    def assign(self, **kwargs):
        """Assign new data variables to a Dataset, returning a new object
        with all the original variables in addition to the new ones.

        Parameters
        ----------
        kwargs : keyword, value pairs
            keywords are the variables names. If the values are callable, they
            are computed on the Dataset and assigned to new data variables. If
            the values are not callable, (e.g. a DataArray, scalar, or array),
            they are simply assigned.

        Returns
        -------
        ds : Dataset
            A new Dataset with the new variables in addition to all the
            existing variables.

        Notes
        -----
        Since ``kwargs`` is a dictionary, the order of your arguments may not
        be preserved, and so the order of the new variables is not well
        defined. Assigning multiple variables within the same ``assign`` is
        possible, but you cannot reference other variables created within the
        same ``assign`` call.

        See Also
        --------
        pandas.DataFrame.assign
        """
        data = self.copy()
        # do all calculations first...
        results = data._calc_assign_results(kwargs)
        # ... and then assign
        data.update(results)
        return data

    def to_array(self, dim='variable', name=None):
        """Convert this dataset into an xarray.DataArray

        The data variables of this dataset will be broadcast against each other
        and stacked along the first axis of the new array. All coordinates of
        this dataset will remain coordinates.

        Parameters
        ----------
        dim : str, optional
            Name of the new dimension.
        name : str, optional
            Name of the new data array.

        Returns
        -------
        array : xarray.DataArray
        """
        from .dataarray import DataArray

        data_vars = [self.variables[k] for k in self.data_vars]
        broadcast_vars = broadcast_variables(*data_vars)
        data = ops.stack([b.data for b in broadcast_vars], axis=0)

        coords = dict(self.coords)
        coords[dim] = list(self.data_vars)

        dims = (dim,) + broadcast_vars[0].dims

        return DataArray(data, coords, dims, attrs=self.attrs, name=name)

    def _to_dataframe(self, ordered_dims):
        columns = [k for k in self if k not in self.dims]
        data = [self._variables[k].expand_dims(ordered_dims).values.reshape(-1)
                for k in columns]
        index = self.coords.to_index(ordered_dims)
        return pd.DataFrame(OrderedDict(zip(columns, data)), index=index)

    def to_dataframe(self):
        """Convert this dataset into a pandas.DataFrame.

        Non-index variables in this dataset form the columns of the
        DataFrame. The DataFrame is be indexed by the Cartesian product of
        this dataset's indices.
        """
        return self._to_dataframe(self.dims)

    @classmethod
    def from_dataframe(cls, dataframe):
        """Convert a pandas.DataFrame into an xarray.Dataset

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

        if not dataframe.columns.is_unique:
            raise ValueError(
                'cannot convert DataFrame with non-unique columns')

        idx = dataframe.index
        obj = cls()

        if hasattr(idx, 'levels'):
            # it's a multi-index
            # expand the DataFrame to include the product of all levels
            full_idx = pd.MultiIndex.from_product(idx.levels, names=idx.names)
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
            data = np.asarray(series).reshape(shape)
            obj[name] = (dims, data)
        return obj

    @staticmethod
    def _unary_op(f, keep_attrs=False):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            ds = self.coords.to_dataset()
            for k in self.data_vars:
                ds._variables[k] = f(self._variables[k], *args, **kwargs)
            if keep_attrs:
                ds._attrs = self._attrs
            return ds
        return func

    @staticmethod
    def _binary_op(f, reflexive=False, join='inner', fillna=False):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, groupby.GroupBy):
                return NotImplemented
            if hasattr(other, 'indexes'):
                self, other = align(self, other, join=join, copy=False)
            g = f if not reflexive else lambda x, y: f(y, x)
            ds = self._calculate_binary_op(g, other, fillna=fillna)
            return ds
        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, groupby.GroupBy):
                raise TypeError('in-place operations between a Dataset and '
                                'a grouped object are not permitted')
            if hasattr(other, 'indexes'):
                other = other.reindex_like(self, copy=False)
            # we don't want to actually modify arrays in-place
            g = ops.inplace_to_noninplace_op(f)
            ds = self._calculate_binary_op(g, other, inplace=True)
            self._replace_vars_and_dims(ds._variables, ds._coord_names,
                                        ds._attrs, inplace=True)
            return self
        return func

    def _calculate_binary_op(self, f, other, inplace=False, fillna=False):

        def apply_over_both(lhs_data_vars, rhs_data_vars, lhs_vars, rhs_vars):
            if fillna and not set(rhs_data_vars) <= set(lhs_data_vars):
                raise ValueError('all variables in the argument to `fillna` '
                                 'must be contained in the original dataset')

            dest_vars = OrderedDict()
            for k in lhs_data_vars:
                if k in rhs_data_vars:
                    dest_vars[k] = f(lhs_vars[k], rhs_vars[k])
                elif inplace:
                    raise ValueError(
                        'datasets must have the same data variables '
                        'for in-place arithmetic operations: %s, %s'
                        % (list(lhs_data_vars), list(rhs_data_vars)))
                elif fillna:
                    # this shortcuts left alignment of variables for fillna
                    dest_vars[k] = lhs_vars[k]
            return dest_vars

        if utils.is_dict_like(other) and not isinstance(other, Dataset):
            # can't use our shortcut of doing the binary operation with
            # Variable objects, so apply over our data vars instead.
            new_data_vars = apply_over_both(self.data_vars, other,
                                            self.data_vars, other)
            return Dataset(new_data_vars)

        other_coords = getattr(other, 'coords', None)
        ds = self.coords.merge(other_coords)

        if isinstance(other, Dataset):
            new_vars = apply_over_both(self.data_vars, other.data_vars,
                                       self.variables, other.variables)
        else:
            other_variable = getattr(other, 'variable', other)
            new_vars = OrderedDict((k, f(self.variables[k], other_variable))
                                   for k in self.data_vars)

        ds._variables.update(new_vars)
        return ds

    def diff(self, dim, n=1, label='upper'):
        """Calculate the n-th order discrete difference along given axis.

        Parameters
        ----------
        dim : str, optional
            Dimension over which to calculate the finite difference.
        n : int, optional
            The number of times values are differenced.
        label : str, optional
            The new coordinate in dimension ``dim`` will have the
            values of either the minuend's or subtrahend's coordinate
            for values 'upper' and 'lower', respectively.  Other
            values are not supported.

        Returns
        -------
        difference : same type as caller
            The n-th order finite difference of this object.

        Examples
        --------
        >>> ds = xr.Dataset({'foo': ('x', [5, 5, 6, 6])})
        >>> ds.diff('x')
        <xarray.Dataset>
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 1 2 3
        Data variables:
            foo      (x) int64 0 1 0
        >>> ds.diff('x', 2)
        <xarray.Dataset>
        Dimensions:  (x: 2)
        Coordinates:
        * x        (x) int64 2 3
        Data variables:
        foo      (x) int64 1 -1

        """
        if n == 0:
            return self
        if n < 0:
            raise ValueError('order `n` must be non-negative but got {0}'
                             ''.format(n))

        # prepare slices
        kwargs_start = {dim: slice(None, -1)}
        kwargs_end = {dim: slice(1, None)}

        # prepare new coordinate
        if label == 'upper':
            kwargs_new = kwargs_end
        elif label == 'lower':
            kwargs_new = kwargs_start
        else:
            raise ValueError('The \'label\' argument has to be either '
                             '\'upper\' or \'lower\'')

        variables = OrderedDict()

        for name, var in iteritems(self.variables):
            if dim in var.dims:
                if name in self.data_vars:
                    variables[name] = (var.isel(**kwargs_end) -
                                       var.isel(**kwargs_start))
                else:
                    variables[name] = var.isel(**kwargs_new)
            else:
                variables[name] = var

        difference = self._replace_vars_and_dims(variables)

        if n > 1:
            return difference.diff(dim, n - 1)
        else:
            return difference

    def shift(self, **shifts):
        """Shift this dataset by an offset along one or more dimensions.

        Only data variables are moved; coordinates stay in place. This is
        consistent with the behavior of ``shift`` in pandas.

        Parameters
        ----------
        **shifts : keyword arguments of the form {dim: offset}
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.

        Returns
        -------
        shifted : Dataset
            Dataset with the same coordinates and attributes but shifted data
            variables.

        See also
        --------
        roll

        Examples
        --------

        >>> ds = xr.Dataset({'foo': ('x', list('abcde'))})
        >>> ds.shift(x=2)
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 0 1 2 3 4
        Data variables:
            foo      (x) object nan nan 'a' 'b' 'c'
        """
        invalid = [k for k in shifts if k not in self.dims]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        variables = OrderedDict()
        for name, var in iteritems(self.variables):
            if name in self.data_vars:
                var_shifts = dict((k, v) for k, v in shifts.items()
                                  if k in var.dims)
                variables[name] = var.shift(**var_shifts)
            else:
                variables[name] = var

        return self._replace_vars_and_dims(variables)

    def roll(self, **shifts):
        """Roll this dataset by an offset along one or more dimensions.

        Unlike shift, roll rotates all variables, including coordinates. The
        direction of rotation is consistent with :py:func:`numpy.roll`.

        Parameters
        ----------
        **shifts : keyword arguments of the form {dim: offset}
            Integer offset to rotate each of the given dimensions. Positive
            offsets roll to the right; negative offsets roll to the left.

        Returns
        -------
        rolled : Dataset
            Dataset with the same coordinates and attributes but rolled
            variables.

        See also
        --------
        shift

        Examples
        --------

        >>> ds = xr.Dataset({'foo': ('x', list('abcde'))})
        >>> ds.roll(x=2)
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 3 4 0 1 2
        Data variables:
            foo      (x) object 'd' 'e' 'a' 'b' 'c'
        """
        invalid = [k for k in shifts if k not in self.dims]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        variables = OrderedDict()
        for name, var in iteritems(self.variables):
            var_shifts = dict((k, v) for k, v in shifts.items()
                              if k in var.dims)
            variables[name] = var.roll(**var_shifts)

        return self._replace_vars_and_dims(variables)

    @property
    def real(self):
        return self._unary_op(lambda x: x.real, keep_attrs=True)(self)

    @property
    def imag(self):
        return self._unary_op(lambda x: x.imag, keep_attrs=True)(self)


ops.inject_all_ops_and_reduce_methods(Dataset, array_only=False)
