import contextlib
import functools
import operator
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

import xray
from . import indexing
from . import groupby
from . import ops
from . import utils
from . import variable
from .common import AbstractArray, AbstractCoordinates
from .utils import multi_index_from_product
from .pycompat import iteritems, basestring, OrderedDict


def _is_dict_like(value):
    return hasattr(value, '__getitem__') and hasattr(value, 'keys')


def _infer_coords_and_dims(shape, coords, dims):
    """All the logic for creating a new DataArray"""

    if isinstance(dims, basestring):
        dims = [dims]

    if _is_dict_like(coords):
        if dims is None:
            dims = list(coords.keys())
        else:
            bad_coords = [dim for dim in coords if dim not in dims]
            if bad_coords:
                raise ValueError('coordinates %r are not array dimensions'
                                 % bad_coords)
        coords = [coords.get(d, None) for d in dims]
    elif coords is not None and len(coords) != len(shape):
        raise ValueError('%s coordinates supplied but data has ndim=%s'
                         % (len(coords), len(shape)))

    if dims is None:
        dims = ['dim_%s' % n for n in range(len(shape))]
        if coords is not None:
            for n, idx in enumerate(coords):
                if hasattr(idx, 'name') and idx.name is not None:
                    dims[n] = idx.name
    else:
        for d in dims:
            if not isinstance(d, basestring):
                raise TypeError('dimension %s is not a string' % d)

    if coords is None:
        coords = [None] * len(shape)
    coords = [idx if isinstance(idx, AbstractArray) else
              variable.Coordinate(dims[n], idx) if idx is not None else
              variable.Coordinate(dims[n], np.arange(shape[n]))
              for n, idx in enumerate(coords)]

    return coords, dims


class _LocIndexer(object):
    def __init__(self, data_array):
        self.data_array = data_array

    def _remap_key(self, key):
        label_indexers = self.data_array._key_to_indexers(key)
        indexers = []
        for dim, label in iteritems(label_indexers):
            index = self.data_array.coords[dim]
            indexers.append(indexing.convert_label_indexer(index, label))
        return tuple(indexers)

    def __getitem__(self, key):
        return self.data_array[self._remap_key(key)]

    def __setitem__(self, key, value):
        self.data_array[self._remap_key(key)] = value


def _assert_coordinates_same_size(orig, new):
    if not new.size == orig.size:
        raise ValueError('new coordinate has size %s but the existing '
                         'coordinate has size %s' % (new.size, orig.size))


class DataArrayCoordinates(AbstractCoordinates):
    """Dictionary like container for DataArray coordinates.

    Essentially an immutable OrderedDict with keys given by the array's
    dimensions and the values given by the corresponding xray.Coordinate
    objects.
    """
    def __getitem__(self, key):
        if key in self._data.dims:
            return self._data.dataset.variables[key]
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError('%s is not an existing coordinate')

        coord = self._convert_to_coord(key, value, self[key].size)
        with self._data._set_new_dataset() as ds:
            ds._variables[key] = coord


class DataArray(AbstractArray):
    """N-dimensional array with labeled coordinates and dimensions.

    DataArray provides a wrapper around numpy ndarrays that uses labeled
    dimensions and coordinates to support metadata aware operations. The API is
    similar to that for the pandas Series or DataFrame, but DataArray objects
    can have any number of dimensions, and their contents have fixed data
    types.

    Additional features over raw numpy arrays:

    - Apply operations over dimensions by name: ``x.sum('time')``.
    - Select or assign values by integer location (like numpy): ``x[:10]``
      or by label (like pandas): ``x.loc['2014-01-01']`` or
      ``x.labeled(time='2014-01-01')``.
    - Mathematical operations (e.g., ``x - y``) vectorize across multiple
      dimensions (known in numpy as "broadcasting") based on dimension names,
      regardless of their original order.
    - Keep track of arbitrary metadata in the form of a Python dictionary:
      ``x.attrs``
    - Convert to a pandas Series: ``x.to_series()``.

    Getting items from or doing mathematical operations with a DataArray
    always returns another DataArray.

    Under the covers, a DataArray is a thin wrapper around an xray Dataset,
    and is uniquely defined by its `dataset` and `name` parameters.

    Attributes
    ----------
    dims : tuple
        Dimension names associated with this array.
    values : np.ndarray
        Access or modify DataArray values as a numpy array.
    coords : dict-like
        Dictionary of Coordinate objects that label values along each dimension.
    """
    def __init__(self, data=None, coords=None, dims=None, name=None,
                 attrs=None, encoding=None):
        """
        Parameters
        ----------
        data : array_like, optional
            Values for this array. Must be a ``numpy.ndarray``, ndarray like,
            or castable to an ``ndarray``. If a self-described xray or pandas
            object, attempst are made to use this array's metadata to fill in
            other unspecified arguments. This argument is required unless the
            'dataset' argument is provided.
        coords : sequence or dict of array_like objects, optional
            Coordinates (tick labels) to use for indexing along each dimension.
            If dict-like, should be a mapping from dimension names to the
            corresponding coordinates.
        dims : str or sequence of str, optional
            Name(s) of the the data dimension(s). Must be either a string (only
            for 1D data) or a sequence of strings with length equal to the
            number of dimensions. If this argument is omited, dimension names
            are taken from ``coords`` (if possible) and otherwise default to
            ``['dim_0', ... 'dim_n']``.
        name : str or None, optional
            Name of this array.
        attrs : dict_like or None, optional
            Attributes to assign to the new variable. By default, an empty
            attribute dictionary is initialized.
        encoding : dict_like or None, optional
            Dictionary specifying how to encode this array's data into a
            serialized format like netCDF4. Currently used keys (for netCDF)
            include '_FillValue', 'scale_factor', 'add_offset', 'dtype',
            'units' and 'calendar' (the later two only for datetime arrays).
            Unrecognized keys are ignored.
        """
        # try to fill in arguments from data if they weren't supplied
        if coords is None:
            coords = getattr(data, 'coords', None)
            if isinstance(data, pd.Series):
                coords = [data.index]
            elif isinstance(data, pd.DataFrame):
                coords = [data.index, data.columns]
            elif isinstance(data, (pd.Index, variable.Coordinate)):
                coords = [data]
            elif isinstance(data, pd.Panel):
                coords = [data.items, data.major_axis, data.minor_axis]
        if dims is None:
            dims = getattr(data, 'dims', None)
        if name is None:
            name = getattr(data, 'name', None)
        if attrs is None:
            attrs = getattr(data, 'attrs', None)
        if encoding is None:
            encoding = getattr(data, 'encoding', None)

        data = variable._as_compatible_data(data)
        coords, dims = _infer_coords_and_dims(data.shape, coords, dims)
        variables = OrderedDict((var.name, var) for var in coords)
        variables[name] = variable.Variable(dims, data, attrs, encoding)
        dataset = xray.Dataset(variables)

        self._dataset = dataset
        self._name = name

    @classmethod
    def _new_from_dataset(cls, dataset, name):
        """Private constructor for the benefit Dataset.__getitem__ (skips all
        validation)
        """
        obj = object.__new__(cls)
        obj._dataset = dataset
        obj._name = name
        return obj

    @property
    def dataset(self):
        """The dataset with which this DataArray is associated.
        """
        return self._dataset

    def to_dataset(self, name=None):
        """Convert a DataArray to a Dataset

        Parameters
        ----------
        name : str, optional
            Name to substitute for this array's name (if it has one).

        Returns
        -------
        dataset : Dataset
        """
        if name is None:
            return self._dataset.copy()
        else:
            return self.rename(name)._dataset

    @property
    def name(self):
        """The name of the variable in `dataset` to which array operations
        are applied.
        """
        return self._name

    @contextlib.contextmanager
    def _set_new_dataset(self):
        """Context manager to use for modifying _dataset, in a manner that
        can be safely rolled back if an error is encountered.
        """
        ds = self.dataset.copy(deep=False)
        yield ds
        self._dataset = ds

    @name.setter
    def name(self, value):
        with self._set_new_dataset() as ds:
            ds.rename({self.name: value}, inplace=True)
        self._name = value

    @property
    def variable(self):
        return self.dataset.variables[self.name]

    @property
    def dtype(self):
        return self.variable.dtype

    @property
    def shape(self):
        return self.variable.shape

    @property
    def size(self):
        return self.variable.size

    @property
    def ndim(self):
        return self.variable.ndim

    def __len__(self):
        return len(self.variable)

    @property
    def values(self):
        """The variables's data as a numpy.ndarray"""
        return self.variable.values

    @values.setter
    def values(self, value):
        self.variable.values = value

    def _in_memory(self):
        return self.variable._in_memory()

    @property
    def as_index(self):
        utils.alias_warning('as_index', 'to_index()')
        return self.to_index()

    def to_index(self):
        """Convert this variable to a pandas.Index. Only possible for 1D
        arrays.
        """
        return self.variable.to_index()

    @property
    def dims(self):
        return self.variable.dims

    @dims.setter
    def dims(self, value):
        with self._set_new_dataset() as ds:
            if not len(value) == self.ndim:
                raise ValueError('%s dimensions supplied but data has ndim=%s'
                                 % (len(value), self.ndim))
            name_map = dict(zip(self.dims, value))
            ds.rename(name_map, inplace=True)
        if self.name in name_map:
            self._name = name_map[self.name]

    @property
    def dimensions(self):
        utils.alias_warning('dimensions', 'dims')
        return self.dims

    def _key_to_indexers(self, key):
        return OrderedDict(
            zip(self.dims, indexing.expanded_indexer(key, self.ndim)))

    def __getitem__(self, key):
        if isinstance(key, basestring):
            # grab another dataset array from the dataset
            return self.dataset[key]
        else:
            # orthogonal array indexing
            return self.isel(**self._key_to_indexers(key))

    def __setitem__(self, key, value):
        if isinstance(key, basestring):
            # add an array to the dataset
            self.dataset[key] = value
        else:
            # orthogonal array indexing
            self.variable[key] = value

    def __delitem__(self, key):
        del self.dataset[key]

    def __contains__(self, key):
        return key in self.dataset

    @property
    def loc(self):
        """Attribute for location based indexing like pandas..
        """
        return _LocIndexer(self)

    @property
    def attributes(self):
        utils.alias_warning('attributes', 'attrs')
        return self.variable.attrs

    @attributes.setter
    def attributes(self, value):
        utils.alias_warning('attributes', 'attrs')
        self.variable.attrs = value

    @property
    def attrs(self):
        """Dictionary storing arbitrary metadata with this array."""
        return self.variable.attrs

    @attrs.setter
    def attrs(self, value):
        self.variable.attrs = value

    @property
    def encoding(self):
        """Dictionary of format-specific settings for how this array should be
        serialized."""
        return self.variable.encoding

    @encoding.setter
    def encoding(self, value):
        self.variable.encoding = value

    @property
    def indexes(self):
        return self.coords

    @property
    def coords(self):
        """Dictionary-like container of xray.Coordinate objects used for label based
        indexing.

        Keys are given by the dimensions, but list-like (integer based)
        indexing is also supported.
        """
        return DataArrayCoordinates(self)

    @coords.setter
    def coords(self, value):
        if not len(value) == self.ndim:
            raise ValueError('%s coordinates supplied but data has ndim=%s'
                             % (len(value), self.ndim))
        with self._set_new_dataset() as ds:
            # TODO: allow setting to dict-like objects other than
            # DataArrayCoordinates?
            if isinstance(value, DataArrayCoordinates):
                # yes, this is regretably complex and probably slow
                name_map = dict(zip(self.dims, value.keys()))
                ds.rename(name_map, inplace=True)
                name = name_map.get(self.name, self.name)
                dims = ds[name].dims
                value = value.values()
            else:
                name = self.name
                dims = self.dims

            for k, v in zip(dims, value):
                coord = DataArrayCoordinates._convert_to_coord(
                    k, v, expected_size=ds.coords[k].size)
                ds[k] = coord
        self._name = name

    @property
    def coordinates(self):
        utils.alias_warning('coordinates', 'coords')
        return self.coords

    def load_data(self):
        """Manually trigger loading of this array's data from disk or a
        remote source into memory and return this array.

        Normally, it should not be necessary to call this method in user code,
        because all xray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.
        """
        self.variable.load_data()
        for coord in self.coords.values():
            coord.load_data()
        return self

    def copy(self, deep=True):
        """Returns a copy of this array.

        If `deep=True`, a deep copy is made of all variables in the underlying
        dataset. Otherwise, a shallow copy is made, so each variable in the new
        array's dataset is also a variable in this array's dataset.
        """
        ds = self.dataset.copy(deep=deep)
        return ds[self.name]

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        # memo does nothing but is required for compatability with
        # copy.deepcopy
        return self.copy(deep=True)

    # mutable objects should not be hashable
    __hash__ = None

    def isel(self, **indexers):
        """Return a new DataArray whose dataset is given by integer indexing
        along the specified dimension(s).

        See Also
        --------
        Dataset.isel
        DataArray.sel
        """
        ds = self.dataset.isel(**indexers)
        return ds[self.name]

    indexed = utils.function_alias(isel, 'indexed')

    def sel(self, **indexers):
        """Return a new DataArray whose dataset is given by selecting
        index labels along the specified dimension(s).

        See Also
        --------
        Dataset.sel
        DataArray.isel
        """
        return self.isel(**indexing.remap_label_indexers(self, indexers))

    labeled = utils.function_alias(sel, 'labeled')

    def reindex_like(self, other, copy=True):
        """Conform this object onto the coordinates of another object, filling
        in missing values with NaN.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with a coordinates attribute giving a mapping from dimension
            names to xray.Coordinate objects, which provides indexes upon
            which to index the variables in this dataset. The coordinates on
            this other object need not be the same as the coordinates on this
            dataset. Any mis-matched coordinate values will be filled in with
            NaN, and any mis-matched dimension names will simply be ignored.
        copy : bool, optional
            If `copy=True`, the returned array's dataset contains only copied
            variables. If `copy=False` and no reindexing is required then
            original variables from this array's dataset are returned.

        Returns
        -------
        reindexed : DataArray
            Another dataset array, with this array's data but coordinates from
            the other object.

        See Also
        --------
        DataArray.reindex
        align
        """
        return self.reindex(copy=copy, **other.coords)

    def reindex(self, copy=True, **indexers):
        """Conform this object onto a new set of coordinates, filling in
        missing values with NaN.

        Parameters
        ----------
        copy : bool, optional
            If `copy=True`, the returned array's dataset contains only copied
            variables. If `copy=False` and no reindexing is required then
            original variables from this array's dataset are returned.
        **indexers : dict
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate values
            will be filled in with NaN, and any mis-matched dimension names will
            simply be ignored.

        Returns
        -------
        reindexed : DataArray
            Another dataset array, with this array's data but replaced
            coordinates.

        See Also
        --------
        DataArray.reindex_like
        align
        """
        ds = self.select_vars().dataset
        reindexed_ds = ds.reindex(copy=copy, **indexers)
        return reindexed_ds[self.name]

    def rename(self, new_name_or_name_dict):
        """Returns a new DataArray with renamed variables.

        If the argument is dict-like, it it used as a mapping from old names to
        new names for dataset variables. Otherwise, use the argument as the new
        name for this array.

        See Also
        --------
        Dataset.rename
        """
        if _is_dict_like(new_name_or_name_dict):
            name_dict = new_name_or_name_dict
            new_name = name_dict.get(self.name, self.name)
        else:
            new_name = new_name_or_name_dict
            name_dict = {self.name: new_name}
        renamed_dataset = self.dataset.rename(name_dict)
        return renamed_dataset[new_name]

    def select_vars(self, *names):
        """Returns a new DataArray with only the named variables, as well
        as this DataArray's array variable (and all associated coordinates).

        See Also
        --------
        Dataset.select_vars
        """
        names = names + (self.name,)
        ds = self.dataset.select_vars(*names)
        return ds[self.name]

    select = utils.function_alias(select_vars, 'select')

    def drop_vars(self, *names):
        """Returns a new DataArray without the named variables.

        See Also
        --------
        Dataset.drop_vars
        """
        if self.name in names:
            raise ValueError('cannot drop the name of a DataArray with '
                             'drop_vars. Use the `drop_vars` method of '
                             'the dataset instead.')
        if any(name in self.dims for name in names):
            raise ValueError('cannot drop a coordinate variable from a '
                             'DataArray. Use the `drop_vars` method of '
                             'the dataset instead.')
        ds = self.dataset.drop_vars(*names)
        return ds[self.name]

    unselect = utils.function_alias(drop_vars, 'unselect')

    def groupby(self, group, squeeze=True):
        """Group this dataset by unique values of the indicated group.

        Parameters
        ----------
        group : str, DataArray or Coordinate
            Array whose unique values should be used to group this array. If a
            string, must be the name of a variable contained in this dataset.
        squeeze : boolean, optional
            If "group" is a diension of this array, `squeeze` controls
            whether the subarrays have a dimension of length 1 along that
            dimension or if the dimension is squeezed out.

        Returns
        -------
        grouped : GroupBy
            A `GroupBy` object patterned after `pandas.GroupBy` that can be
            iterated over in the form of `(unique_value, grouped_array)` pairs
            or over which grouped operations can be applied with the `apply`
            and `reduce` methods (and the associated aliases `mean`, `sum`,
            `std`, etc.).
        """
        if isinstance(group, basestring):
            group = self.dataset[group]
        return groupby.ArrayGroupBy(self, group, squeeze=squeeze)

    def transpose(self, *dims):
        """Return a new DataArray object with transposed dimensions.

        Note: Although this operation returns a view of this array's data, it
        is not lazy -- the data will be fully loaded.

        Parameters
        ----------
        *dims : str, optional
            By default, reverse the dimensions. Otherwise, reorder the
            dimensions to this order.

        Returns
        -------
        transposed : DataArray
            The returned DataArray's array is transposed.

        Notes
        -----
        Although this operation returns a view of this array's data, it is
        not lazy -- the data will be fully loaded.

        See Also
        --------
        numpy.transpose
        Array.transpose
        """
        ds = self.dataset.copy()
        ds[self.name] = self.variable.transpose(*dims)
        return ds[self.name]

    def squeeze(self, dim=None):
        """Return a new DataArray object with squeezed data.

        Parameters
        ----------
        dim : None or str or tuple of str, optional
            Selects a subset of the length one dimensions. If a dimension is
            selected with length greater than one, an error is raised. If
            None, all length one dimensions are squeezed.

        Returns
        -------
        squeezed : DataArray
            This array, but with with all or a subset of the dimensions of
            length 1 removed.

        Notes
        -----
        Although this operation returns a view of this array's data, it is
        not lazy -- the data will be fully loaded.

        See Also
        --------
        numpy.squeeze
        """
        ds = self.dataset.squeeze(dim)
        return ds[self.name]

    def reduce(self, func, dim=None, axis=None, keep_attrs=False, **kwargs):
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `func`. Only one of the
            'dim' and 'axis' arguments can be supplied. If neither are
            supplied, then the reduction is calculated over the flattened array
            (by calling `f(x)` without an axis argument).
        keep_attrs : bool, optional
            If True, the variable's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            DataArray with this object's array replaced with an array with
            summarized data and the indicated dimension(s) removed.
        """
        if 'dimension' in kwargs and dim is None:
            dim = kwargs.pop('dimension')
            utils.alias_warning('dimension', 'dim')

        var = self.variable.reduce(func, dim, axis, keep_attrs, **kwargs)
        drop = set(self.dims) - set(var.dims)
        # For now, take an aggressive strategy of removing all variables
        # associated with any dropped dimensions
        # TODO: save some summary (mean? bounds?) of dropped variables
        drop |= set(k for k, v in iteritems(self.dataset.variables)
                    if any(dim in drop for dim in v.dims))
        ds = self.dataset.drop_vars(*drop)
        ds[self.name] = var

        if keep_attrs:
            ds.attrs = self.dataset.attrs

        return ds[self.name]

    @classmethod
    def concat(cls, arrays, dim='concat_dim', indexers=None,
               concat_over=None):
        """Stack arrays along a new or existing dimension to form a new
        DataArray.

        Parameters
        ----------
        arrays : iterable of DataArray
            Arrays to stack together. Each variable is expected to have
            matching dimensions and shape except for along the concatenated
            dimension.
        dim : str or Array, optional
            Name of the dimension to stack along. This can either be a new
            dimension name, in which case it is added along axis=0, or an
            existing dimension name, in which case the location of the
            dimension is unchanged. Where to insert the new dimension is
            determined by whether it is found in the first array. If dimension
            is provided as an Variable or DataArray, the name of the dataset
            array or the singleton dimension of the variable is used as the
            stacking dimension and the array is added to the returned dataset.
        indexers : iterable of indexers, optional
            Iterable of indexers of the same length as variables which
            specifies how to assign variables along the given dimension. If
            not supplied, indexers is inferred from the length of each
            variable along the dimension, and the variables are concatenated in
            the given order.
        concat_over : None or str or iterable of str, optional
            Names of additional variables to concatenate (other than the given
            arrays variables), in which "dimension" does not already appear as
            a dimension.

        Returns
        -------
        concatenated : DataArray
            Concatenated DataArray formed by concatenated all the supplied
            variables along the new dimension.

        See also
        --------
        Dataset.concat
        """
        # TODO: call select() on each DataArray and get rid of the confusing
        # concat_over kwarg?
        datasets = []
        for n, arr in enumerate(arrays):
            if n == 0:
                name = arr.name
            elif name != arr.name:
                arr = arr.rename(name)
            datasets.append(arr.dataset)
        if concat_over is None:
            concat_over = set()
        elif isinstance(concat_over, basestring):
            concat_over = set([concat_over])
        concat_over = set(concat_over) | set([name])
        ds = xray.Dataset.concat(datasets, dim, indexers,
                                 concat_over=concat_over)
        return ds[name]

    def to_dataframe(self):
        """Convert this array into a pandas.DataFrame.

        Non-coordinate variables in this array's dataset (which include this
        array's data) form the columns of the DataFrame. The DataFrame is be
        indexed by the Cartesian product of the dataset's coordintaes.
        """
        return self.dataset.to_dataframe()

    def to_series(self):
        """Convert this array into a pandas.Series.

        The Series is indexed by the Cartesian product of the coordinates.
        Unlike `to_dataframe`, only this array is including in the returned
        series; the other non-coordinate variables in the dataset are not.
        """
        index = multi_index_from_product(self.coords.values(),
                                         names=self.coords.keys())
        return pd.Series(self.values.reshape(-1), index=index, name=self.name)

    @classmethod
    def from_series(cls, series):
        """Convert a pandas.Series into an xray.DataArray

        If the series's index is a MultiIndex, it will be expanded into a
        tensor product of one-dimensional coordinates (filling in missing values
        with NaN). Thus this operation should be the inverse of the `to_series`
        method.
        """
        df = pd.DataFrame({series.name: series})
        ds = xray.Dataset.from_dataframe(df)
        return ds[series.name]

    def equals(self, other):
        """True if two DataArrays have the same dimensions, coordinates and
        values; otherwise False.

        DataArrays can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for DataArrays
        does element-wise comparisions (like numpy.ndarrays).
        """
        try:
            return (all(k1 == k2 and v1.equals(v2)
                        for (k1, v1), (k2, v2)
                        in zip(self.coords.items(),
                               other.coords.items()))
                    and self.variable.equals(other.variable))
        except AttributeError:
            return False

    def identical(self, other):
        """Like equals, but also checks DataArray names and attributes, and
        attributes on their coordinates.
        """
        try:
            return (self.name == other.name
                    and all(k1 == k2 and v1.identical(v2)
                            for (k1, v1), (k2, v2)
                            in zip(self.coords.items(),
                                   other.coords.items()))
                    and self.variable.identical(other.variable))
        except AttributeError:
            return False

    def _select_coords(self):
        return xray.Dataset(self.coords)

    def __array_wrap__(self, obj, context=None):
        new_var = self.variable.__array_wrap__(obj, context)
        ds = self._select_coords()
        if (self.name,) == self.dims:
            # use a new name for coordinate variables
            name = None
        else:
            name = self.name
        ds[name] = new_var
        return ds[name]

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            return self.__array_wrap__(f(self.values, *args, **kwargs))
        return func

    def _check_coords_compat(self, other):
        # TODO: possibly automatically select index intersection instead?
        if hasattr(other, 'coords'):
            for k, v in iteritems(self.coords):
                if (k in other.coords
                        and not v.equals(other.coords[k])):
                    raise ValueError('coordinate %r is not aligned' % k)

    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            # TODO: automatically group by other variable dimensions to allow
            # for broadcasting dimensions like 'dayofyear' against 'time'
            self._check_coords_compat(other)
            ds = self._select_coords()
            if hasattr(other, 'coords'):
                ds.merge(other.coords, inplace=True)
            other_array = getattr(other, 'variable', other)
            if hasattr(other, 'name') or (self.name,) == self.dims:
                name = None
            else:
                name = self.name
            ds[name] = (f(self.variable, other_array)
                        if not reflexive
                        else f(other_array, self.variable))
            return ds[name]
        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            self._check_coords_compat(other)
            other_array = getattr(other, 'variable', other)
            f(self.variable, other_array)
            if hasattr(other, 'coords'):
                self.dataset.merge(other.coords, inplace=True)
            return self
        return func

ops.inject_special_operations(DataArray, priority=60)


def align(*objects, **kwargs):
    """align(*objects, join='inner', copy=True)

    Given any number of Dataset and/or DataArray objects, returns new
    objects with aligned coordinates.

    Array from the aligned objects are suitable as input to mathematical
    operators, because along each dimension they are indexed by the same
    coordinates.

    Missing values (if ``join != 'inner'``) are filled with NaN.

    Parameters
    ----------
    *objects : Dataset or DataArray
        Objects to align.
    join : {'outer', 'inner', 'left', 'right'}, optional
        Method for joining the coordinates of the passed objects along each
        dimension:
         - 'outer': use the union of object coordinates
         - 'outer': use the intersection of object coordinates
         - 'left': use coordinates from the first object with each dimension
         - 'right': use coordinates from the last object with each dimension
    copy : bool, optional
        If `copy=True`, the returned objects contain all new variables. If
        `copy=False` and no reindexing is required then the aligned objects
        will include original variables.

    Returns
    -------
    aligned : same as *objects
        Tuple of objects with aligned coordinates.
    """
    # TODO: automatically align when doing math with dataset arrays?
    # TODO: change this to default to join='outer' like pandas?
    if 'join' not in kwargs:
        warnings.warn('using align without setting explicitly setting the '
                      "'join' keyword argument. In future versions of xray, "
                      "the default will likely change from join='inner' to "
                      "join='outer', to match pandas.",
                      FutureWarning, stacklevel=2)

    join = kwargs.pop('join', 'inner')
    copy = kwargs.pop('copy', True)

    if join == 'outer':
        join_indices = functools.partial(functools.reduce, operator.or_)
    elif join == 'inner':
        join_indices = functools.partial(functools.reduce, operator.and_)
    elif join == 'left':
        join_indices = operator.itemgetter(0)
    elif join == 'right':
        join_indices = operator.itemgetter(-1)

    all_indexes = defaultdict(list)
    for obj in objects:
        for k, v in iteritems(obj.coords):
            all_indexes[k].append(v.to_index())

    # Exclude dimensions with all equal indices to avoid unnecessary reindexing
    # work.
    joined_indexes = dict((k, join_indices(v)) for k, v in iteritems(all_indexes)
                          if any(not v[0].equals(idx) for idx in v[1:]))

    return tuple(obj.reindex(copy=copy, **joined_indexes) for obj in objects)
