from __future__ import absolute_import, division, print_function

import functools
import warnings

import numpy as np
import pandas as pd

from . import computation, groupby, indexing, ops, resample, rolling, utils
from ..plot.plot import _PlotMethods
from .accessors import DatetimeAccessor
from .alignment import align, reindex_like_indexers
from .common import AbstractArray, DataWithCoords
from .coordinates import (
    DataArrayCoordinates, Indexes, LevelCoordinatesSource,
    assert_coordinate_consistent, remap_label_indexers)
from .dataset import Dataset, merge_indexes, split_indexes
from .formatting import format_item
from .options import OPTIONS
from .pycompat import OrderedDict, basestring, iteritems, range, zip
from .utils import (
    decode_numpy_dict_values, either_dict_or_kwargs, ensure_us_time_resolution)
from .variable import (
    IndexVariable, Variable, as_compatible_data, as_variable,
    assert_unique_multiindex_level_names)


def _infer_coords_and_dims(shape, coords, dims):
    """All the logic for creating a new DataArray"""

    if (coords is not None and not utils.is_dict_like(coords) and
            len(coords) != len(shape)):
        raise ValueError('coords is not dict-like, but it has %s items, '
                         'which does not match the %s dimensions of the '
                         'data' % (len(coords), len(shape)))

    if isinstance(dims, basestring):
        dims = (dims,)

    if dims is None:
        dims = ['dim_%s' % n for n in range(len(shape))]
        if coords is not None and len(coords) == len(shape):
            # try to infer dimensions from coords
            if utils.is_dict_like(coords):
                # deprecated in GH993, removed in GH1539
                raise ValueError('inferring DataArray dimensions from '
                                 'dictionary like ``coords`` is no longer '
                                 'supported. Use an explicit list of '
                                 '``dims`` instead.')
            for n, (dim, coord) in enumerate(zip(dims, coords)):
                coord = as_variable(coord,
                                    name=dims[n]).to_index_variable()
                dims[n] = coord.name
        dims = tuple(dims)
    else:
        for d in dims:
            if not isinstance(d, basestring):
                raise TypeError('dimension %s is not a string' % d)

    new_coords = OrderedDict()

    if utils.is_dict_like(coords):
        for k, v in coords.items():
            new_coords[k] = as_variable(v, name=k)
    elif coords is not None:
        for dim, coord in zip(dims, coords):
            var = as_variable(coord, name=dim)
            var.dims = (dim,)
            new_coords[dim] = var

    sizes = dict(zip(dims, shape))
    for k, v in new_coords.items():
        if any(d not in dims for d in v.dims):
            raise ValueError('coordinate %s has dimensions %s, but these '
                             'are not a subset of the DataArray '
                             'dimensions %s' % (k, v.dims, dims))

        for d, s in zip(v.dims, v.shape):
            if s != sizes[d]:
                raise ValueError('conflicting sizes for dimension %r: '
                                 'length %s on the data but length %s on '
                                 'coordinate %r' % (d, sizes[d], s, k))

        if k in sizes and v.shape != (sizes[k],):
            raise ValueError('coordinate %r is a DataArray dimension, but '
                             'it has shape %r rather than expected shape %r '
                             'matching the dimension size'
                             % (k, v.shape, (sizes[k],)))

    assert_unique_multiindex_level_names(new_coords)

    return new_coords, dims


class _LocIndexer(object):
    def __init__(self, data_array):
        self.data_array = data_array

    def __getitem__(self, key):
        if not utils.is_dict_like(key):
            # expand the indexer so we can handle Ellipsis
            labels = indexing.expanded_indexer(key, self.data_array.ndim)
            key = dict(zip(self.data_array.dims, labels))
        return self.data_array.sel(**key)

    def __setitem__(self, key, value):
        if not utils.is_dict_like(key):
            # expand the indexer so we can handle Ellipsis
            labels = indexing.expanded_indexer(key, self.data_array.ndim)
            key = dict(zip(self.data_array.dims, labels))

        pos_indexers, _ = remap_label_indexers(self.data_array, **key)
        self.data_array[pos_indexers] = value


# Used as the key corresponding to a DataArray's variable when converting
# arbitrary DataArray objects to datasets
_THIS_ARRAY = utils.ReprObject('<this-array>')


class DataArray(AbstractArray, DataWithCoords):
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
      ``x.sel(time='2014-01-01')``.
    - Mathematical operations (e.g., ``x - y``) vectorize across multiple
      dimensions (known in numpy as "broadcasting") based on dimension names,
      regardless of their original order.
    - Keep track of arbitrary metadata in the form of a Python dictionary:
      ``x.attrs``
    - Convert to a pandas Series: ``x.to_series()``.

    Getting items from or doing mathematical operations with a DataArray
    always returns another DataArray.

    Attributes
    ----------
    dims : tuple
        Dimension names associated with this array.
    values : np.ndarray
        Access or modify DataArray values as a numpy array.
    coords : dict-like
        Dictionary of DataArray objects that label values along each dimension.
    name : str or None
        Name of this array.
    attrs : OrderedDict
        Dictionary for holding arbitrary metadata.
    """
    _groupby_cls = groupby.DataArrayGroupBy
    _rolling_cls = rolling.DataArrayRolling
    _resample_cls = resample.DataArrayResample

    dt = property(DatetimeAccessor)

    def __init__(self, data, coords=None, dims=None, name=None,
                 attrs=None, encoding=None, fastpath=False):
        """
        Parameters
        ----------
        data : array_like
            Values for this array. Must be an ``numpy.ndarray``, ndarray like,
            or castable to an ``ndarray``. If a self-described xarray or pandas
            object, attempts are made to use this array's metadata to fill in
            other unspecified arguments. A view of the array's data is used
            instead of a copy if possible.
        coords : sequence or dict of array_like objects, optional
            Coordinates (tick labels) to use for indexing along each dimension.
            If dict-like, should be a mapping from dimension names to the
            corresponding coordinates. If sequence-like, should be a sequence
            of tuples where the first element is the dimension name and the
            second element is the corresponding coordinate array_like object.
        dims : str or sequence of str, optional
            Name(s) of the data dimension(s). Must be either a string (only
            for 1D data) or a sequence of strings with length equal to the
            number of dimensions. If this argument is omitted, dimension names
            are taken from ``coords`` (if possible) and otherwise default to
            ``['dim_0', ... 'dim_n']``.
        name : str or None, optional
            Name of this array.
        attrs : dict_like or None, optional
            Attributes to assign to the new instance. By default, an empty
            attribute dictionary is initialized.
        encoding : dict_like or None, optional
            Dictionary specifying how to encode this array's data into a
            serialized format like netCDF4. Currently used keys (for netCDF)
            include '_FillValue', 'scale_factor', 'add_offset', 'dtype',
            'units' and 'calendar' (the later two only for datetime arrays).
            Unrecognized keys are ignored.
        """
        if fastpath:
            variable = data
            assert dims is None
            assert attrs is None
            assert encoding is None
        else:
            # try to fill in arguments from data if they weren't supplied
            if coords is None:
                coords = getattr(data, 'coords', None)
                if isinstance(data, pd.Series):
                    coords = [data.index]
                elif isinstance(data, pd.DataFrame):
                    coords = [data.index, data.columns]
                elif isinstance(data, (pd.Index, IndexVariable)):
                    coords = [data]
                elif isinstance(data, pd.Panel):
                    coords = [data.items, data.major_axis, data.minor_axis]
            if dims is None:
                dims = getattr(data, 'dims', getattr(coords, 'dims', None))
            if name is None:
                name = getattr(data, 'name', None)
            if attrs is None:
                attrs = getattr(data, 'attrs', None)
            if encoding is None:
                encoding = getattr(data, 'encoding', None)

            data = as_compatible_data(data)
            coords, dims = _infer_coords_and_dims(data.shape, coords, dims)
            variable = Variable(dims, data, attrs, encoding, fastpath=True)

        # uncomment for a useful consistency check:
        # assert all(isinstance(v, Variable) for v in coords.values())

        # These fully describe a DataArray
        self._variable = variable
        self._coords = coords
        self._name = name

        self._file_obj = None

        self._initialized = True

    __default = object()

    def _replace(self, variable=None, coords=None, name=__default):
        if variable is None:
            variable = self.variable
        if coords is None:
            coords = self._coords
        if name is self.__default:
            name = self.name
        return type(self)(variable, coords, name=name, fastpath=True)

    def _replace_maybe_drop_dims(self, variable, name=__default):
        if variable.dims == self.dims:
            coords = self._coords.copy()
        else:
            allowed_dims = set(variable.dims)
            coords = OrderedDict((k, v) for k, v in self._coords.items()
                                 if set(v.dims) <= allowed_dims)
        return self._replace(variable, coords, name)

    def _replace_indexes(self, indexes):
        if not len(indexes):
            return self
        coords = self._coords.copy()
        for name, idx in indexes.items():
            coords[name] = IndexVariable(name, idx)
        obj = self._replace(coords=coords)

        # switch from dimension to level names, if necessary
        dim_names = {}
        for dim, idx in indexes.items():
            if not isinstance(idx, pd.MultiIndex) and idx.name != dim:
                dim_names[dim] = idx.name
        if dim_names:
            obj = obj.rename(dim_names)
        return obj

    def _to_temp_dataset(self):
        return self._to_dataset_whole(name=_THIS_ARRAY,
                                      shallow_copy=False)

    def _from_temp_dataset(self, dataset, name=__default):
        variable = dataset._variables.pop(_THIS_ARRAY)
        coords = dataset._variables
        return self._replace(variable, coords, name)

    def _to_dataset_split(self, dim):
        def subset(dim, label):
            array = self.loc[{dim: label}]
            if dim in array.coords:
                del array.coords[dim]
            array.attrs = {}
            return array

        variables = OrderedDict([(label, subset(dim, label))
                                 for label in self.get_index(dim)])
        coords = self.coords.to_dataset()
        if dim in coords:
            del coords[dim]
        return Dataset(variables, coords, self.attrs)

    def _to_dataset_whole(self, name=None, shallow_copy=True):
        if name is None:
            name = self.name
        if name is None:
            raise ValueError('unable to convert unnamed DataArray to a '
                             'Dataset without providing an explicit name')
        if name in self.coords:
            raise ValueError('cannot create a Dataset from a DataArray with '
                             'the same name as one of its coordinates')
        # use private APIs for speed: this is called by _to_temp_dataset(),
        # which is used in the guts of a lot of operations (e.g., reindex)
        variables = self._coords.copy()
        variables[name] = self.variable
        if shallow_copy:
            for k in variables:
                variables[k] = variables[k].copy(deep=False)
        coord_names = set(self._coords)
        dataset = Dataset._from_vars_and_coord_names(variables, coord_names)
        return dataset

    def to_dataset(self, dim=None, name=None):
        """Convert a DataArray to a Dataset.

        Parameters
        ----------
        dim : str, optional
            Name of the dimension on this array along which to split this array
            into separate variables. If not provided, this array is converted
            into a Dataset of one variable.
        name : str, optional
            Name to substitute for this array's name. Only valid if ``dim`` is
            not provided.

        Returns
        -------
        dataset : Dataset
        """
        if dim is not None and dim not in self.dims:
            warnings.warn('the order of the arguments on DataArray.to_dataset '
                          'has changed; you now need to supply ``name`` as '
                          'a keyword argument',
                          FutureWarning, stacklevel=2)
            name = dim
            dim = None

        if dim is not None:
            if name is not None:
                raise TypeError('cannot supply both dim and name arguments')
            return self._to_dataset_split(dim)
        else:
            return self._to_dataset_whole(name)

    @property
    def name(self):
        """The name of this array.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def variable(self):
        """Low level interface to the Variable object for this DataArray."""
        return self._variable

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
    def nbytes(self):
        return self.variable.nbytes

    @property
    def ndim(self):
        return self.variable.ndim

    def __len__(self):
        return len(self.variable)

    @property
    def data(self):
        """The array's data as a dask or numpy array"""
        return self.variable.data

    @data.setter
    def data(self, value):
        self.variable.data = value

    @property
    def values(self):
        """The array's data as a numpy.ndarray"""
        return self.variable.values

    @values.setter
    def values(self, value):
        self.variable.values = value

    @property
    def _in_memory(self):
        return self.variable._in_memory

    def to_index(self):
        """Convert this variable to a pandas.Index. Only possible for 1D
        arrays.
        """
        return self.variable.to_index()

    @property
    def dims(self):
        """Tuple of dimension names associated with this array.

        Note that the type of this property is inconsistent with
        `Dataset.dims`.  See `Dataset.sizes` and `DataArray.sizes` for
        consistently named properties.
        """
        return self.variable.dims

    @dims.setter
    def dims(self, value):
        raise AttributeError('you cannot assign dims on a DataArray. Use '
                             '.rename() or .swap_dims() instead.')

    def _item_key_to_dict(self, key):
        if utils.is_dict_like(key):
            return key
        else:
            key = indexing.expanded_indexer(key, self.ndim)
            return dict(zip(self.dims, key))

    @property
    def _level_coords(self):
        """Return a mapping of all MultiIndex levels and their corresponding
        coordinate name.
        """
        level_coords = OrderedDict()
        for cname, var in self._coords.items():
            if var.ndim == 1 and isinstance(var, IndexVariable):
                level_names = var.level_names
                if level_names is not None:
                    dim, = var.dims
                    level_coords.update({lname: dim for lname in level_names})
        return level_coords

    def _getitem_coord(self, key):
        from .dataset import _get_virtual_variable

        try:
            var = self._coords[key]
        except KeyError:
            dim_sizes = dict(zip(self.dims, self.shape))
            _, key, var = _get_virtual_variable(
                self._coords, key, self._level_coords, dim_sizes)

        return self._replace_maybe_drop_dims(var, name=key)

    def __getitem__(self, key):
        if isinstance(key, basestring):
            return self._getitem_coord(key)
        else:
            # xarray-style array indexing
            return self.isel(indexers=self._item_key_to_dict(key))

    def __setitem__(self, key, value):
        if isinstance(key, basestring):
            self.coords[key] = value
        else:
            # Coordinates in key, value and self[key] should be consistent.
            # TODO Coordinate consistency in key is checked here, but it
            # causes unnecessary indexing. It should be optimized.
            obj = self[key]
            if isinstance(value, DataArray):
                assert_coordinate_consistent(value, obj.coords.variables)
            # DataArray key -> Variable key
            key = {k: v.variable if isinstance(v, DataArray) else v
                   for k, v in self._item_key_to_dict(key).items()}
            self.variable[key] = value

    def __delitem__(self, key):
        del self.coords[key]

    @property
    def _attr_sources(self):
        """List of places to look-up items for attribute-style access"""
        return self._item_sources + [self.attrs]

    @property
    def _item_sources(self):
        """List of places to look-up items for key-completion"""
        return [self.coords, {d: self.coords[d] for d in self.dims},
                LevelCoordinatesSource(self)]

    def __contains__(self, key):
        warnings.warn(
            'xarray.DataArray.__contains__ currently checks membership in '
            'DataArray.coords, but in xarray v0.11 will change to check '
            'membership in array values.', FutureWarning, stacklevel=2)
        return key in self._coords

    @property
    def loc(self):
        """Attribute for location based indexing like pandas.
        """
        return _LocIndexer(self)

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
        """OrderedDict of pandas.Index objects used for label based indexing
        """
        return Indexes(self._coords, self.sizes)

    @property
    def coords(self):
        """Dictionary-like container of coordinate arrays.
        """
        return DataArrayCoordinates(self)

    def reset_coords(self, names=None, drop=False, inplace=False):
        """Given names of coordinates, reset them to become variables.

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
        Dataset, or DataArray if ``drop == True``
        """
        if inplace and not drop:
            raise ValueError('cannot reset coordinates in-place on a '
                             'DataArray without ``drop == True``')
        if names is None:
            names = set(self.coords) - set(self.dims)
        dataset = self.coords.to_dataset().reset_coords(names, drop)
        if drop:
            if inplace:
                self._coords = dataset._variables
            else:
                return self._replace(coords=dataset._variables)
        else:
            if self.name is None:
                raise ValueError('cannot reset_coords with drop=False '
                                 'on an unnamed DataArrray')
            dataset[self.name] = self.variable
            return dataset

    def __dask_graph__(self):
        return self._to_temp_dataset().__dask_graph__()

    def __dask_keys__(self):
        return self._to_temp_dataset().__dask_keys__()

    @property
    def __dask_optimize__(self):
        return self._to_temp_dataset().__dask_optimize__

    @property
    def __dask_scheduler__(self):
        return self._to_temp_dataset().__dask_scheduler__

    def __dask_postcompute__(self):
        func, args = self._to_temp_dataset().__dask_postcompute__()
        return self._dask_finalize, (func, args, self.name)

    def __dask_postpersist__(self):
        func, args = self._to_temp_dataset().__dask_postpersist__()
        return self._dask_finalize, (func, args, self.name)

    @staticmethod
    def _dask_finalize(results, func, args, name):
        ds = func(results, *args)
        variable = ds._variables.pop(_THIS_ARRAY)
        coords = ds._variables
        return DataArray(variable, coords, name=name, fastpath=True)

    def load(self, **kwargs):
        """Manually trigger loading of this array's data from disk or a
        remote source into memory and return this array.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        ds = self._to_temp_dataset().load(**kwargs)
        new = self._from_temp_dataset(ds)
        self._variable = new._variable
        self._coords = new._coords
        return self

    def compute(self, **kwargs):
        """Manually trigger loading of this array's data from disk or a
        remote source into memory and return a new array. The original is
        left unaltered.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        new = self.copy(deep=False)
        return new.load(**kwargs)

    def persist(self, **kwargs):
        """ Trigger computation in constituent dask arrays

        This keeps them as dask arrays but encourages them to keep data in
        memory.  This is particularly useful when on a distributed machine.
        When on a single machine consider using ``.compute()`` instead.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.persist``.

        See Also
        --------
        dask.persist
        """
        ds = self._to_temp_dataset().persist(**kwargs)
        return self._from_temp_dataset(ds)

    def copy(self, deep=True):
        """Returns a copy of this array.

        If `deep=True`, a deep copy is made of all variables in the underlying
        dataset. Otherwise, a shallow copy is made, so each variable in the new
        array's dataset is also a variable in this array's dataset.
        """
        variable = self.variable.copy(deep=deep)
        coords = OrderedDict((k, v.copy(deep=deep))
                             for k, v in self._coords.items())
        return self._replace(variable, coords)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        # memo does nothing but is required for compatibility with
        # copy.deepcopy
        return self.copy(deep=True)

    # mutable objects should not be hashable
    __hash__ = None

    @property
    def chunks(self):
        """Block dimensions for this array's data or None if it's not a dask
        array.
        """
        return self.variable.chunks

    def chunk(self, chunks=None, name_prefix='xarray-', token=None,
              lock=False):
        """Coerce this array's data into a dask arrays with the given chunks.

        If this variable is a non-dask array, it will be converted to dask
        array. If it's a dask array, it will be rechunked to the given chunk
        sizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

        Parameters
        ----------
        chunks : int, tuple or dict, optional
            Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
            ``{'x': 5, 'y': 5}``.
        name_prefix : str, optional
            Prefix for the name of the new dask array.
        token : str, optional
            Token uniquely identifying this array.
        lock : optional
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.

        Returns
        -------
        chunked : xarray.DataArray
        """
        if isinstance(chunks, (list, tuple)):
            chunks = dict(zip(self.dims, chunks))

        ds = self._to_temp_dataset().chunk(chunks, name_prefix=name_prefix,
                                           token=token, lock=lock)
        return self._from_temp_dataset(ds)

    def isel(self, indexers=None, drop=False, **indexers_kwargs):
        """Return a new DataArray whose dataset is given by integer indexing
        along the specified dimension(s).

        See Also
        --------
        Dataset.isel
        DataArray.sel
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'isel')
        ds = self._to_temp_dataset().isel(drop=drop, indexers=indexers)
        return self._from_temp_dataset(ds)

    def sel(self, indexers=None, method=None, tolerance=None, drop=False,
            **indexers_kwargs):
        """Return a new DataArray whose dataset is given by selecting
        index labels along the specified dimension(s).

        .. warning::

          Do not try to assign values when using any of the indexing methods
          ``isel`` or ``sel``::

            da = xr.DataArray([0, 1, 2, 3], dims=['x'])
            # DO NOT do this
            da.isel(x=[0, 1, 2])[1] = -1

          Assigning values with the chained indexing using ``.sel`` or
          ``.isel`` fails silently.

        See Also
        --------
        Dataset.sel
        DataArray.isel

        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'sel')
        ds = self._to_temp_dataset().sel(
            indexers=indexers, drop=drop, method=method, tolerance=tolerance)
        return self._from_temp_dataset(ds)

    def isel_points(self, dim='points', **indexers):
        """Return a new DataArray whose dataset is given by pointwise integer
        indexing along the specified dimension(s).

        See Also
        --------
        Dataset.isel_points
        """
        ds = self._to_temp_dataset().isel_points(dim=dim, **indexers)
        return self._from_temp_dataset(ds)

    def sel_points(self, dim='points', method=None, tolerance=None,
                   **indexers):
        """Return a new DataArray whose dataset is given by pointwise selection
        of index labels along the specified dimension(s).

        See Also
        --------
        Dataset.sel_points
        """
        ds = self._to_temp_dataset().sel_points(
            dim=dim, method=method, tolerance=tolerance, **indexers)
        return self._from_temp_dataset(ds)

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
            Method to use for filling index values from other not found on this
            data array:

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
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.

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
        indexers = reindex_like_indexers(self, other)
        return self.reindex(method=method, tolerance=tolerance, copy=copy,
                            **indexers)

    def reindex(self, indexers=None, method=None, tolerance=None, copy=True,
                **indexers_kwargs):
        """Conform this object onto a new set of indexes, filling in
        missing values with NaN.

        Parameters
        ----------
        indexers : dict, optional
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate
            values will be filled in with NaN, and any mis-matched dimension
            names will simply be ignored.
            One of indexers or indexers_kwargs must be provided.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for filling index values in ``indexers`` not found on
            this data array:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value (requires pandas>=0.16)
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations most
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        **indexers_kwarg : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

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
        indexers = either_dict_or_kwargs(
            indexers, indexers_kwargs, 'reindex')
        ds = self._to_temp_dataset().reindex(
            indexers=indexers, method=method, tolerance=tolerance, copy=copy)
        return self._from_temp_dataset(ds)

    def interp(self, coords=None, method='linear', assume_sorted=False,
               kwargs={}, **coords_kwargs):
        """ Multidimensional interpolation of variables.

        coords : dict, optional
            Mapping from dimension names to the new coordinates.
            new coordinate can be an scalar, array-like or DataArray.
            If DataArrays are passed as new coordates, their dimensions are
            used for the broadcasting.
        method: {'linear', 'nearest'} for multidimensional array,
            {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'}
            for 1-dimensional array.
        assume_sorted: boolean, optional
            If False, values of x can be in any order and they are sorted
            first. If True, x has to be an array of monotonically increasing
            values.
        kwargs: dictionary
            Additional keyword passed to scipy's interpolator.
        **coords_kwarg : {dim: coordinate, ...}, optional
            The keyword arguments form of ``coords``.
            One of coords or coords_kwargs must be provided.

        Returns
        -------
        interpolated: xr.DataArray
            New dataarray on the new coordinates.

        Note
        ----
        scipy is required.

        See Also
        --------
        scipy.interpolate.interp1d
        scipy.interpolate.interpn
        """
        if self.dtype.kind not in 'uifc':
            raise TypeError('interp only works for a numeric type array. '
                            'Given {}.'.format(self.dtype))

        ds = self._to_temp_dataset().interp(
            coords, method=method, kwargs=kwargs, assume_sorted=assume_sorted,
            **coords_kwargs)
        return self._from_temp_dataset(ds)

    def interp_like(self, other, method='linear', assume_sorted=False,
                    kwargs={}):
        """Interpolate this object onto the coordinates of another object,
        filling out of range values with NaN.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to an 1d array-like, which provides coordinates upon
            which to index the variables in this dataset.
        method: string, optional.
            {'linear', 'nearest'} for multidimensional array,
            {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'}
            for 1-dimensional array. 'linear' is used by default.
        assume_sorted: boolean, optional
            If False, values of coordinates that are interpolated over can be
            in any order and they are sorted first. If True, interpolated
            coordinates are assumed to be an array of monotonically increasing
            values.
        kwargs: dictionary, optional
            Additional keyword passed to scipy's interpolator.

        Returns
        -------
        interpolated: xr.DataArray
            Another dataarray by interpolating this dataarray's data along the
            coordinates of the other object.

        Note
        ----
        scipy is required.
        If the dataarray has object-type coordinates, reindex is used for these
        coordinates instead of the interpolation.

        See Also
        --------
        DataArray.interp
        DataArray.reindex_like
        """
        if self.dtype.kind not in 'uifc':
            raise TypeError('interp only works for a numeric type array. '
                            'Given {}.'.format(self.dtype))

        ds = self._to_temp_dataset().interp_like(
            other, method=method, kwargs=kwargs, assume_sorted=assume_sorted)
        return self._from_temp_dataset(ds)

    def rename(self, new_name_or_name_dict=None, **names):
        """Returns a new DataArray with renamed coordinates or a new name.

        Parameters
        ----------
        new_name_or_name_dict : str or dict-like, optional
            If the argument is dict-like, it it used as a mapping from old
            names to new names for coordinates. Otherwise, use the argument
            as the new name for this array.
        **names, optional
            The keyword arguments form of a mapping from old names to
            new names for coordinates.
            One of new_name_or_name_dict or names must be provided.


        Returns
        -------
        renamed : DataArray
            Renamed array or array with renamed coordinates.

        See Also
        --------
        Dataset.rename
        DataArray.swap_dims
        """
        if names or utils.is_dict_like(new_name_or_name_dict):
            name_dict = either_dict_or_kwargs(
                new_name_or_name_dict, names, 'rename')
            dataset = self._to_temp_dataset().rename(name_dict)
            return self._from_temp_dataset(dataset)
        else:
            return self._replace(name=new_name_or_name_dict)

    def swap_dims(self, dims_dict):
        """Returns a new DataArray with swapped dimensions.

        Parameters
        ----------
        dims_dict : dict-like
            Dictionary whose keys are current dimension names and whose values
            are new names. Each value must already be a coordinate on this
            array.

        Returns
        -------
        renamed : Dataset
            DataArray with swapped dimensions.

        See Also
        --------

        DataArray.rename
        Dataset.swap_dims
        """
        ds = self._to_temp_dataset().swap_dims(dims_dict)
        return self._from_temp_dataset(ds)

    def expand_dims(self, dim, axis=None):
        """Return a new object with an additional axis (or axes) inserted at
        the corresponding position in the array shape.

        If dim is already a scalar coordinate, it will be promoted to a 1D
        coordinate consisting of a single value.

        Parameters
        ----------
        dim : str or sequence of str.
            Dimensions to include on the new variable.
            dimensions are inserted with length 1.
        axis : integer, list (or tuple) of integers, or None
            Axis position(s) where new axis is to be inserted (position(s) on
            the result array). If a list (or tuple) of integers is passed,
            multiple axes are inserted. In this case, dim arguments should be
            same length list. If axis=None is passed, all the axes will be
            inserted to the start of the result array.

        Returns
        -------
        expanded : same type as caller
            This object, but with an additional dimension(s).
        """
        ds = self._to_temp_dataset().expand_dims(dim, axis)
        return self._from_temp_dataset(ds)

    def set_index(self, append=False, inplace=False, **indexes):
        """Set DataArray (multi-)indexes using one or more existing
        coordinates.

        Parameters
        ----------
        append : bool, optional
            If True, append the supplied index(es) to the existing index(es).
            Otherwise replace the existing index(es) (default).
        inplace : bool, optional
            If True, set new index(es) in-place. Otherwise, return a new
            DataArray object.
        **indexes : {dim: index, ...}
            Keyword arguments with names matching dimensions and values given
            by (lists of) the names of existing coordinates or variables to set
            as new (multi-)index.

        Returns
        -------
        obj : DataArray
            Another dataarray, with this data but replaced coordinates.

        See Also
        --------
        DataArray.reset_index
        """
        coords, _ = merge_indexes(indexes, self._coords, set(), append=append)
        if inplace:
            self._coords = coords
        else:
            return self._replace(coords=coords)

    def reset_index(self, dims_or_levels, drop=False, inplace=False):
        """Reset the specified index(es) or multi-index level(s).

        Parameters
        ----------
        dims_or_levels : str or list
            Name(s) of the dimension(s) and/or multi-index level(s) that will
            be reset.
        drop : bool, optional
            If True, remove the specified indexes and/or multi-index levels
            instead of extracting them as new coordinates (default: False).
        inplace : bool, optional
            If True, modify the dataarray in-place. Otherwise, return a new
            DataArray object.

        Returns
        -------
        obj : DataArray
            Another dataarray, with this dataarray's data but replaced
            coordinates.

        See Also
        --------
        DataArray.set_index
        """
        coords, _ = split_indexes(dims_or_levels, self._coords, set(),
                                  self._level_coords, drop=drop)
        if inplace:
            self._coords = coords
        else:
            return self._replace(coords=coords)

    def reorder_levels(self, inplace=False, **dim_order):
        """Rearrange index levels using input order.

        Parameters
        ----------
        inplace : bool, optional
            If True, modify the dataarray in-place. Otherwise, return a new
            DataArray object.
        **dim_order : optional
            Keyword arguments with names matching dimensions and values given
            by lists representing new level orders. Every given dimension
            must have a multi-index.

        Returns
        -------
        obj : DataArray
            Another dataarray, with this dataarray's data but replaced
            coordinates.
        """
        replace_coords = {}
        for dim, order in dim_order.items():
            coord = self._coords[dim]
            index = coord.to_index()
            if not isinstance(index, pd.MultiIndex):
                raise ValueError("coordinate %r has no MultiIndex" % dim)
            replace_coords[dim] = IndexVariable(coord.dims,
                                                index.reorder_levels(order))
        coords = self._coords.copy()
        coords.update(replace_coords)
        if inplace:
            self._coords = coords
        else:
            return self._replace(coords=coords)

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
        stacked : DataArray
            DataArray with stacked data.

        Examples
        --------

        >>> arr = DataArray(np.arange(6).reshape(2, 3),
        ...                 coords=[('x', ['a', 'b']), ('y', [0, 1, 2])])
        >>> arr
        <xarray.DataArray (x: 2, y: 3)>
        array([[0, 1, 2],
               [3, 4, 5]])
        Coordinates:
          * x        (x) |S1 'a' 'b'
          * y        (y) int64 0 1 2
        >>> stacked = arr.stack(z=('x', 'y'))
        >>> stacked.indexes['z']
        MultiIndex(levels=[[u'a', u'b'], [0, 1, 2]],
                   labels=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
                   names=[u'x', u'y'])

        See also
        --------
        DataArray.unstack
        """
        ds = self._to_temp_dataset().stack(**dimensions)
        return self._from_temp_dataset(ds)

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
        unstacked : DataArray
            Array with unstacked data.

        See also
        --------
        DataArray.stack
        """
        ds = self._to_temp_dataset().unstack(dim)
        return self._from_temp_dataset(ds)

    def transpose(self, *dims):
        """Return a new DataArray object with transposed dimensions.

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
        Dataset.transpose
        """
        variable = self.variable.transpose(*dims)
        return self._replace(variable)

    def drop(self, labels, dim=None):
        """Drop coordinates or index labels from this DataArray.

        Parameters
        ----------
        labels : scalar or list of scalars
            Name(s) of coordinate variables or index labels to drop.
        dim : str, optional
            Dimension along which to drop index labels. By default (if
            ``dim is None``), drops coordinates rather than index labels.

        Returns
        -------
        dropped : DataArray
        """
        if utils.is_scalar(labels):
            labels = [labels]
        ds = self._to_temp_dataset().drop(labels, dim)
        return self._from_temp_dataset(ds)

    def dropna(self, dim, how='any', thresh=None):
        """Returns a new array with dropped labels for missing values along
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

        Returns
        -------
        DataArray
        """
        ds = self._to_temp_dataset().dropna(dim, how=how, thresh=thresh)
        return self._from_temp_dataset(ds)

    def fillna(self, value):
        """Fill missing values in this object.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic, except the result is aligned to this
        object (``join='left'``) instead of aligned to the intersection of
        index coordinates (``join='inner'``).

        Parameters
        ----------
        value : scalar, ndarray or DataArray
            Used to fill all matching missing values in this array. If the
            argument is a DataArray, it is first aligned with (reindexed to)
            this array.

        Returns
        -------
        DataArray
        """
        if utils.is_dict_like(value):
            raise TypeError('cannot provide fill value as a dictionary with '
                            'fillna on a DataArray')
        out = ops.fillna(self, value)
        return out

    def interpolate_na(self, dim=None, method='linear', limit=None,
                       use_coordinate=True,
                       **kwargs):
        """Interpolate values according to different methods.

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to interpolate.
        method : {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
                  'polynomial', 'barycentric', 'krog', 'pchip',
                  'spline', 'akima'}, optional
            String indicating which method to use for interpolation:

            - 'linear': linear interpolation (Default). Additional keyword
              arguments are passed to ``numpy.interp``
            - 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
              'polynomial': are passed to ``scipy.interpolate.interp1d``. If
              method=='polynomial', the ``order`` keyword argument must also be
              provided.
            - 'barycentric', 'krog', 'pchip', 'spline', and `akima`: use their
              respective``scipy.interpolate`` classes.
        use_coordinate : boolean or str, default True
            Specifies which index to use as the x values in the interpolation
            formulated as `y = f(x)`. If False, values are treated as if
            eqaully-spaced along `dim`. If True, the IndexVariable `dim` is
            used. If use_coordinate is a string, it specifies the name of a
            coordinate variariable to use as the index.
        limit : int, default None
            Maximum number of consecutive NaNs to fill. Must be greater than 0
            or None for no limit.

        Returns
        -------
        DataArray

        See also
        --------
        numpy.interp
        scipy.interpolate
        """
        from .missing import interp_na
        return interp_na(self, dim=dim, method=method, limit=limit,
                         use_coordinate=use_coordinate, **kwargs)

    def ffill(self, dim, limit=None):
        '''Fill NaN values by propogating values forward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default None
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        DataArray
        '''
        from .missing import ffill
        return ffill(self, dim, limit=limit)

    def bfill(self, dim, limit=None):
        '''Fill NaN values by propogating values backward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default None
            The maximum number of consecutive NaN values to backward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        DataArray
        '''
        from .missing import bfill
        return bfill(self, dim, limit=limit)

    def combine_first(self, other):
        """Combine two DataArray objects, with union of coordinates.

        This operation follows the normal broadcasting and alignment rules of
        ``join='outer'``.  Default to non-null values of array calling the
        method.  Use np.nan to fill in vacant cells after alignment.

        Parameters
        ----------
        other : DataArray
            Used to fill all matching missing values in this array.

        Returns
        -------
        DataArray
        """
        return ops.fillna(self, other, join="outer")

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
        var = self.variable.reduce(func, dim, axis, keep_attrs, **kwargs)
        return self._replace_maybe_drop_dims(var)

    def to_pandas(self):
        """Convert this array into a pandas object with the same shape.

        The type of the returned object depends on the number of DataArray
        dimensions:

        * 1D -> `pandas.Series`
        * 2D -> `pandas.DataFrame`
        * 3D -> `pandas.Panel`

        Only works for arrays with 3 or fewer dimensions.

        The DataArray constructor performs the inverse transformation.
        """
        # TODO: consolidate the info about pandas constructors and the
        # attributes that correspond to their indexes into a separate module?
        constructors = {0: lambda x: x,
                        1: pd.Series,
                        2: pd.DataFrame,
                        3: pd.Panel}
        try:
            constructor = constructors[self.ndim]
        except KeyError:
            raise ValueError('cannot convert arrays with %s dimensions into '
                             'pandas objects' % self.ndim)
        indexes = [self.get_index(dim) for dim in self.dims]
        return constructor(self.values, *indexes)

    def to_dataframe(self, name=None):
        """Convert this array and its coordinates into a tidy pandas.DataFrame.

        The DataFrame is indexed by the Cartesian product of index coordinates
        (in the form of a :py:class:`pandas.MultiIndex`).

        Other coordinates are included as columns in the DataFrame.
        """
        if name is None:
            name = self.name
        if name is None:
            raise ValueError('cannot convert an unnamed DataArray to a '
                             'DataFrame: use the ``name`` parameter')

        dims = OrderedDict(zip(self.dims, self.shape))
        # By using a unique name, we can convert a DataArray into a DataFrame
        # even if it shares a name with one of its coordinates.
        # I would normally use unique_name = object() but that results in a
        # dataframe with columns in the wrong order, for reasons I have not
        # been able to debug (possibly a pandas bug?).
        unique_name = '__unique_name_identifier_z98xfz98xugfg73ho__'
        ds = self._to_dataset_whole(name=unique_name)
        df = ds._to_dataframe(dims)
        df.columns = [name if c == unique_name else c
                      for c in df.columns]
        return df

    def to_series(self):
        """Convert this array into a pandas.Series.

        The Series is indexed by the Cartesian product of index coordinates
        (in the form of a :py:class:`pandas.MultiIndex`).
        """
        index = self.coords.to_index()
        return pd.Series(self.values.reshape(-1), index=index, name=self.name)

    def to_masked_array(self, copy=True):
        """Convert this array into a numpy.ma.MaskedArray

        Parameters
        ----------
        copy : bool
            If True (default) make a copy of the array in the result. If False,
            a MaskedArray view of DataArray.values is returned.

        Returns
        -------
        result : MaskedArray
            Masked where invalid values (nan or inf) occur.
        """
        isnull = pd.isnull(self.values)
        return np.ma.MaskedArray(data=self.values, mask=isnull, copy=copy)

    def to_netcdf(self, *args, **kwargs):
        """Write DataArray contents to a netCDF file.

        Parameters
        ----------
        path : str or Path, optional
            Path to which to save this dataset. If no path is provided, this
            function returns the resulting netCDF file as a bytes object; in
            this case, we need to use scipy.io.netcdf, which does not support
            netCDF version 4 (the default format becomes NETCDF3_64BIT).
        mode : {'w', 'a'}, optional
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            this location will be overwritten.
        format : {'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT',
                  'NETCDF3_CLASSIC'}, optional
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
            ``{'my_variable': {'dtype': 'int16', 'scale_factor': 0.1,
               'zlib': True}, ...}``

        Notes
        -----
        Only xarray.Dataset objects can be written to netCDF files, so
        the xarray.DataArray is converted to a xarray.Dataset object
        containing a single variable. If the DataArray has no name, or if the
        name is the same as a co-ordinate name, then it is given the name
        '__xarray_dataarray_variable__'.

        All parameters are passed directly to `xarray.Dataset.to_netcdf`.
        """
        from ..backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE

        if self.name is None:
            # If no name is set then use a generic xarray name
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
        elif self.name in self.coords or self.name in self.dims:
            # The name is the same as one of the coords names, which netCDF
            # doesn't support, so rename it but keep track of the old name
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
            dataset.attrs[DATAARRAY_NAME] = self.name
        else:
            # No problems with the name - so we're fine!
            dataset = self.to_dataset()

        return dataset.to_netcdf(*args, **kwargs)

    def to_dict(self):
        """
        Convert this xarray.DataArray into a dictionary following xarray
        naming conventions.

        Converts all variables and attributes to native Python objects.
        Useful for coverting to json. To avoid datetime incompatibility
        use decode_times=False kwarg in xarrray.open_dataset.

        See also
        --------
        DataArray.from_dict
        """
        d = {'coords': {}, 'attrs': decode_numpy_dict_values(self.attrs),
             'dims': self.dims}

        for k in self.coords:
            data = ensure_us_time_resolution(self[k].values).tolist()
            d['coords'].update({
                k: {'data': data,
                    'dims': self[k].dims,
                    'attrs': decode_numpy_dict_values(self[k].attrs)}})

        d.update({'data': ensure_us_time_resolution(self.values).tolist(),
                  'name': self.name})
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Convert a dictionary into an xarray.DataArray

        Input dict can take several forms::

            d = {'dims': ('t'), 'data': x}

            d = {'coords': {'t': {'dims': 't', 'data': t,
                                  'attrs': {'units':'s'}}},
                 'attrs': {'title': 'air temperature'},
                 'dims': 't',
                 'data': x,
                 'name': 'a'}

        where 't' is the name of the dimesion, 'a' is the name of the array,
        and  x and t are lists, numpy.arrays, or pandas objects.

        Parameters
        ----------
        d : dict, with a minimum structure of {'dims': [..], 'data': [..]}

        Returns
        -------
        obj : xarray.DataArray

        See also
        --------
        DataArray.to_dict
        Dataset.from_dict
        """
        coords = None
        if 'coords' in d:
            try:
                coords = OrderedDict([(k, (v['dims'],
                                           v['data'],
                                           v.get('attrs')))
                                      for k, v in d['coords'].items()])
            except KeyError as e:
                raise ValueError(
                    "cannot convert dict when coords are missing the key "
                    "'{dims_data}'".format(dims_data=str(e.args[0])))
        try:
            data = d['data']
        except KeyError:
            raise ValueError("cannot convert dict without the key 'data''")
        else:
            obj = cls(data, coords, d.get('dims'), d.get('name'),
                      d.get('attrs'))
        return obj

    @classmethod
    def from_series(cls, series):
        """Convert a pandas.Series into an xarray.DataArray.

        If the series's index is a MultiIndex, it will be expanded into a
        tensor product of one-dimensional coordinates (filling in missing
        values with NaN). Thus this operation should be the inverse of the
        `to_series` method.
        """
        # TODO: add a 'name' parameter
        name = series.name
        df = pd.DataFrame({name: series})
        ds = Dataset.from_dataframe(df)
        return ds[name]

    def to_cdms2(self):
        """Convert this array into a cdms2.Variable
        """
        from ..convert import to_cdms2
        return to_cdms2(self)

    @classmethod
    def from_cdms2(cls, variable):
        """Convert a cdms2.Variable into an xarray.DataArray
        """
        from ..convert import from_cdms2
        return from_cdms2(variable)

    def to_iris(self):
        """Convert this array into a iris.cube.Cube
        """
        from ..convert import to_iris
        return to_iris(self)

    @classmethod
    def from_iris(cls, cube):
        """Convert a iris.cube.Cube into an xarray.DataArray
        """
        from ..convert import from_iris
        return from_iris(cube)

    def _all_compat(self, other, compat_str):
        """Helper function for equals and identical"""

        def compat(x, y):
            return getattr(x.variable, compat_str)(y.variable)

        return (utils.dict_equiv(self.coords, other.coords, compat=compat) and
                compat(self, other))

    def broadcast_equals(self, other):
        """Two DataArrays are broadcast equal if they are equal after
        broadcasting them against each other such that they have the same
        dimensions.

        See Also
        --------
        DataArray.equals
        DataArray.identical
        """
        try:
            return self._all_compat(other, 'broadcast_equals')
        except (TypeError, AttributeError):
            return False

    def equals(self, other):
        """True if two DataArrays have the same dimensions, coordinates and
        values; otherwise False.

        DataArrays can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for ``DataArray``
        does element-wise comparisons (like numpy.ndarrays).

        See Also
        --------
        DataArray.broadcast_equals
        DataArray.identical
        """
        try:
            return self._all_compat(other, 'equals')
        except (TypeError, AttributeError):
            return False

    def identical(self, other):
        """Like equals, but also checks the array name and attributes, and
        attributes on all coordinates.

        See Also
        --------
        DataArray.broadcast_equals
        DataArray.equal
        """
        try:
            return (self.name == other.name and
                    self._all_compat(other, 'identical'))
        except (TypeError, AttributeError):
            return False

    __default_name = object()

    def _result_name(self, other=None):
        # use the same naming heuristics as pandas:
        # https://github.com/ContinuumIO/blaze/issues/458#issuecomment-51936356
        other_name = getattr(other, 'name', self.__default_name)
        if other_name is self.__default_name or other_name == self.name:
            return self.name
        else:
            return None

    def __array_wrap__(self, obj, context=None):
        new_var = self.variable.__array_wrap__(obj, context)
        return self._replace(new_var)

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            with np.errstate(all='ignore'):
                return self.__array_wrap__(f(self.variable.data, *args,
                                             **kwargs))

        return func

    @staticmethod
    def _binary_op(f, reflexive=False, join=None, **ignored_kwargs):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, (Dataset, groupby.GroupBy)):
                return NotImplemented
            if hasattr(other, 'indexes'):
                align_type = (OPTIONS['arithmetic_join']
                              if join is None else join)
                self, other = align(self, other, join=align_type, copy=False)
            other_variable = getattr(other, 'variable', other)
            other_coords = getattr(other, 'coords', None)

            variable = (f(self.variable, other_variable)
                        if not reflexive
                        else f(other_variable, self.variable))
            coords = self.coords._merge_raw(other_coords)
            name = self._result_name(other)

            return self._replace(variable, coords, name)

        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, groupby.GroupBy):
                raise TypeError('in-place operations between a DataArray and '
                                'a grouped object are not permitted')
            # n.b. we can't align other to self (with other.reindex_like(self))
            # because `other` may be converted into floats, which would cause
            # in-place arithmetic to fail unpredictably. Instead, we simply
            # don't support automatic alignment with in-place arithmetic.
            other_coords = getattr(other, 'coords', None)
            other_variable = getattr(other, 'variable', other)
            with self.coords._merge_inplace(other_coords):
                f(self.variable, other_variable)
            return self

        return func

    def _copy_attrs_from(self, other):
        self.attrs = other.attrs

    @property
    def plot(self):
        """
        Access plotting functions

        >>> d = DataArray([[1, 2], [3, 4]])

        For convenience just call this directly
        >>> d.plot()

        Or use it as a namespace to use xarray.plot functions as
        DataArray methods
        >>> d.plot.imshow()  # equivalent to xarray.plot.imshow(d)

        """
        return _PlotMethods(self)

    def _title_for_slice(self, truncate=50):
        """
        If the dataarray has 1 dimensional coordinates or comes from a slice
        we can show that info in the title

        Parameters
        ----------
        truncate : integer
            maximum number of characters for title

        Returns
        -------
        title : string
            Can be used for plot titles

        """
        one_dims = []
        for dim, coord in iteritems(self.coords):
            if coord.size == 1:
                one_dims.append('{dim} = {v}'.format(
                    dim=dim, v=format_item(coord.values)))

        title = ', '.join(one_dims)
        if len(title) > truncate:
            title = title[:(truncate - 3)] + '...'

        return title

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
        >>> arr = xr.DataArray([5, 5, 6, 6], [[1, 2, 3, 4]], ['x'])
        >>> arr.diff('x')
        <xarray.DataArray (x: 3)>
        array([0, 1, 0])
        Coordinates:
        * x        (x) int64 2 3 4
        >>> arr.diff('x', 2)
        <xarray.DataArray (x: 2)>
        array([ 1, -1])
        Coordinates:
        * x        (x) int64 3 4

        """
        ds = self._to_temp_dataset().diff(n=n, dim=dim, label=label)
        return self._from_temp_dataset(ds)

    def shift(self, **shifts):
        """Shift this array by an offset along one or more dimensions.

        Only the data is moved; coordinates stay in place. Values shifted from
        beyond array bounds are replaced by NaN. This is consistent with the
        behavior of ``shift`` in pandas.

        Parameters
        ----------
        **shifts : keyword arguments of the form {dim: offset}
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.

        Returns
        -------
        shifted : DataArray
            DataArray with the same coordinates and attributes but shifted
            data.

        See also
        --------
        roll

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], dims='x')
        >>> arr.shift(x=1)
        <xarray.DataArray (x: 3)>
        array([ nan,   5.,   6.])
        Coordinates:
          * x        (x) int64 0 1 2
        """
        variable = self.variable.shift(**shifts)
        return self._replace(variable)

    def roll(self, **shifts):
        """Roll this array by an offset along one or more dimensions.

        Unlike shift, roll rotates all variables, including coordinates. The
        direction of rotation is consistent with :py:func:`numpy.roll`.

        Parameters
        ----------
        **shifts : keyword arguments of the form {dim: offset}
            Integer offset to rotate each of the given dimensions. Positive
            offsets roll to the right; negative offsets roll to the left.

        Returns
        -------
        rolled : DataArray
            DataArray with the same attributes but rolled data and coordinates.

        See also
        --------
        shift

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], dims='x')
        >>> arr.roll(x=1)
        <xarray.DataArray (x: 3)>
        array([7, 5, 6])
        Coordinates:
          * x        (x) int64 2 0 1
        """
        ds = self._to_temp_dataset().roll(**shifts)
        return self._from_temp_dataset(ds)

    @property
    def real(self):
        return self._replace(self.variable.real)

    @property
    def imag(self):
        return self._replace(self.variable.imag)

    def dot(self, other, dims=None):
        """Perform dot product of two DataArrays along their shared dims.

        Equivalent to taking taking tensordot over all shared dims.

        Parameters
        ----------
        other : DataArray
            The other array with which the dot product is performed.
        dims: list of strings, optional
            Along which dimensions to be summed over. Default all the common
            dimensions are summed over.

        Returns
        -------
        result : DataArray
            Array resulting from the dot product over all shared dimensions.

        See also
        --------
        dot
        numpy.tensordot

        Examples
        --------

        >>> da_vals = np.arange(6 * 5 * 4).reshape((6, 5, 4))
        >>> da = DataArray(da_vals, dims=['x', 'y', 'z'])
        >>> dm_vals = np.arange(4)
        >>> dm = DataArray(dm_vals, dims=['z'])

        >>> dm.dims
        ('z')
        >>> da.dims
        ('x', 'y', 'z')

        >>> dot_result = da.dot(dm)
        >>> dot_result.dims
        ('x', 'y')
        """
        if isinstance(other, Dataset):
            raise NotImplementedError('dot products are not yet supported '
                                      'with Dataset objects.')
        if not isinstance(other, DataArray):
            raise TypeError('dot only operates on DataArrays.')

        return computation.dot(self, other, dims=dims)

    def sortby(self, variables, ascending=True):
        """
        Sort object by labels or values (along an axis).

        Sorts the dataarray, either along specified dimensions,
        or according to values of 1-D dataarrays that share dimension
        with calling object.

        If the input variables are dataarrays, then the dataarrays are aligned
        (via left-join) to the calling object prior to sorting by cell values.
        NaNs are sorted to the end, following Numpy convention.

        If multiple sorts along the same dimension is
        given, numpy's lexsort is performed along that dimension:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.lexsort.html
        and the FIRST key in the sequence is used as the primary sort key,
        followed by the 2nd key, etc.

        Parameters
        ----------
        variables: str, DataArray, or list of either
            1D DataArray objects or name(s) of 1D variable(s) in
            coords whose values are used to sort this array.
        ascending: boolean, optional
            Whether to sort by ascending or descending order.

        Returns
        -------
        sorted: DataArray
            A new dataarray where all the specified dims are sorted by dim
            labels.

        Examples
        --------

        >>> da = xr.DataArray(np.random.rand(5),
        ...                   coords=[pd.date_range('1/1/2000', periods=5)],
        ...                   dims='time')
        >>> da
        <xarray.DataArray (time: 5)>
        array([ 0.965471,  0.615637,  0.26532 ,  0.270962,  0.552878])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03 ...

        >>> da.sortby(da)
        <xarray.DataArray (time: 5)>
        array([ 0.26532 ,  0.270962,  0.552878,  0.615637,  0.965471])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-03 2000-01-04 2000-01-05 ...
        """
        ds = self._to_temp_dataset().sortby(variables, ascending=ascending)
        return self._from_temp_dataset(ds)

    def quantile(self, q, dim=None, interpolation='linear', keep_attrs=False):
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements.

        Parameters
        ----------
        q : float in range of [0,1] (or sequence of floats)
            Quantile to compute, which must be between 0 and 1 inclusive.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply quantile.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to
            use when the desired quantile lies between two data points
            ``i < j``:

                * linear: ``i + (j - i) * fraction``, where ``fraction`` is
                  the fractional part of the index surrounded by ``i`` and
                  ``j``.
                * lower: ``i``.
                * higher: ``j``.
                * nearest: ``i`` or ``j``, whichever is nearest.
                * midpoint: ``(i + j) / 2``.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

        Returns
        -------
        quantiles : DataArray
            If `q` is a single quantile, then the result
            is a scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile and a quantile dimension
            is added to the return array. The other dimensions are the
             dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanpercentile, pandas.Series.quantile, Dataset.quantile
        """

        ds = self._to_temp_dataset().quantile(
            q, dim=dim, keep_attrs=keep_attrs, interpolation=interpolation)
        return self._from_temp_dataset(ds)

    def rank(self, dim, pct=False, keep_attrs=False):
        """Ranks the data.

        Equal values are assigned a rank that is the average of the ranks that
        would have been otherwise assigned to all of the values within that
        set.  Ranks begin at 1, not 0. If pct, computes percentage ranks.

        NaNs in the input array are returned as NaNs.

        The `bottleneck` library is required.

        Parameters
        ----------
        dim : str
            Dimension over which to compute rank.
        pct : bool, optional
            If True, compute percentage ranks, otherwise compute integer ranks.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

        Returns
        -------
        ranked : DataArray
            DataArray with the same coordinates and dtype 'float64'.

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], dims='x')
        >>> arr.rank('x')
        <xarray.DataArray (x: 3)>
        array([ 1.,   2.,   3.])
        Dimensions without coordinates: x
        """
        ds = self._to_temp_dataset().rank(dim, pct=pct, keep_attrs=keep_attrs)
        return self._from_temp_dataset(ds)


# priority most be higher than Variable to properly work with binary ufuncs
ops.inject_all_ops_and_reduce_methods(DataArray, priority=60)
