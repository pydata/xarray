import contextlib
import functools
import warnings

import pandas as pd

from . import indexing
from . import groupby
from . import ops
from . import utils
from . import variable
from .alignment import align
from .common import AbstractArray, BaseDataObject
from .coordinates import DataArrayCoordinates, Indexes
from .dataset import Dataset
from .pycompat import iteritems, basestring, OrderedDict, zip
from .utils import FrozenOrderedDict
from .variable import as_variable, _as_compatible_data, Coordinate


def _infer_coords_and_dims(shape, coords, dims):
    """All the logic for creating a new DataArray"""

    if (coords is not None and not utils.is_dict_like(coords)
            and len(coords) != len(shape)):
        raise ValueError('coords is not dict-like, but it has %s items, '
                         'which does not match the %s dimensions of the '
                         'data' % (len(coords), len(shape)))

    if isinstance(dims, basestring):
        dims = [dims]

    if dims is None:
        dims = ['dim_%s' % n for n in range(len(shape))]
        if coords is not None and len(coords) == len(shape):
            # try to infer dimensions from coords
            if utils.is_dict_like(coords):
                dims = list(coords.keys())
            else:
                for n, (dim, coord) in enumerate(zip(dims, coords)):
                    if getattr(coord, 'name', None) is None:
                        coord = as_variable(coord, key=dim).to_coord()
                    dims[n] = coord.name
    else:
        for d in dims:
            if not isinstance(d, basestring):
                raise TypeError('dimension %s is not a string' % d)
        if coords is not None and not utils.is_dict_like(coords):
            # ensure coordinates have the right dimensions
            coords = [Coordinate(dim, coord, getattr(coord, 'attrs', {}))
                      for dim, coord in zip(dims, coords)]

    if coords is None:
        coords = {}
    elif not utils.is_dict_like(coords):
        coords = OrderedDict(zip(dims, coords))

    return coords, dims


class _LocIndexer(object):
    def __init__(self, data_array):
        self.data_array = data_array

    def _remap_key(self, key):
        def lookup_positions(dim, labels):
            index = self.data_array.indexes[dim]
            return indexing.convert_label_indexer(index, labels)

        if utils.is_dict_like(key):
            return dict((dim, lookup_positions(dim, labels))
                        for dim, labels in iteritems(key))
        else:
            # expand the indexer so we can handle Ellipsis
            key = indexing.expanded_indexer(key, self.data_array.ndim)
            return tuple(lookup_positions(dim, labels) for dim, labels
                         in zip(self.data_array.dims, key))

    def __getitem__(self, key):
        return self.data_array[self._remap_key(key)]

    def __setitem__(self, key, value):
        self.data_array[self._remap_key(key)] = value


class DataArray(AbstractArray, BaseDataObject):
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
        Dictionary of Coordinate objects that label values along each dimension.
    name : str or None
        Name of this array.
    attrs : OrderedDict
        Dictionary for holding arbitrary metadata.
    """
    groupby_cls = groupby.DataArrayGroupBy

    def __init__(self, data, coords=None, dims=None, name=None,
                 attrs=None, encoding=None):
        """
        Parameters
        ----------
        data : array_like
            Values for this array. Must be an ``numpy.ndarray``, ndarray like,
            or castable to an ``ndarray``. If a self-described xray or pandas
            object, attempts are made to use this array's metadata to fill in
            other unspecified arguments. A view of the array's data is used
            instead of a copy if possible.
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
            dims = getattr(data, 'dims', getattr(coords, 'dims', None))
        if name is None:
            name = getattr(data, 'name', None)
        if attrs is None:
            attrs = getattr(data, 'attrs', None)
        if encoding is None:
            encoding = getattr(data, 'encoding', None)

        data = _as_compatible_data(data)
        coords, dims = _infer_coords_and_dims(data.shape, coords, dims)
        dataset = Dataset(coords=coords)
        # insert data afterwards in case of redundant coords/data
        dataset[name] = (dims, data, attrs, encoding)

        for k, v in iteritems(dataset.coords):
            if any(d not in dims for d in v.dims):
                raise ValueError('coordinate %s has dimensions %s, but these '
                                 'are not a subset of the DataArray '
                                 'dimensions %s' % (k, v.dims, dims))

        # these fully describe a DataArray:
        self._dataset = dataset
        self._name = name

    @classmethod
    def _new_from_dataset(cls, dataset, name):
        """Private constructor for the benefit of Dataset.__getitem__ (skips
        all validation)
        """
        obj = object.__new__(cls)
        obj._dataset = dataset._copy_listed([name], keep_attrs=False)
        if name not in obj._dataset:
            # handle virtual variables
            try:
                _, name = name.split('.', 1)
            except Exception:
                raise KeyError(name)
        obj._name = name
        if name not in dataset._dims:
            obj._dataset._coord_names.discard(name)
        return obj

    @classmethod
    def _new_from_dataset_no_copy(cls, dataset, name):
        obj = object.__new__(cls)
        obj._dataset = dataset
        obj._name = name
        return obj

    def _with_replaced_dataset(self, dataset):
        obj = object.__new__(type(self))
        obj._name = self.name
        obj._dataset = dataset
        return obj

    def _to_dataset_split(self, dim):
        def subset(dim, label):
            array = self.loc[{dim: label}].drop(dim)
            array.attrs = {}
            return array

        variables = OrderedDict([(str(label), subset(dim, label))
                                 for label in self.indexes[dim]])
        coords = self.coords.to_dataset()
        del coords[dim]
        return Dataset(variables, coords, self.attrs)

    def _to_dataset_whole(self, name):
        if name is None:
            return self._dataset.copy()
        else:
            return self.rename(name)._dataset

    def to_dataset(self, dim=None, name=None):
        """Convert a DataArray to a Dataset.

        Parameters
        ----------
        dim : str, optional
            Name of the dimension on this array along which to split this array
            into separate variables. If not provided, this array is converted
            into a Dataset of one variable.
        name : str, optional
            Name to substitute for this array's name. Only valid is ``dim`` is
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

    @contextlib.contextmanager
    def _set_new_dataset(self):
        """Context manager to use for modifying _dataset, in a manner that
        can be safely rolled back if an error is encountered.
        """
        ds = self._dataset.copy(deep=False)
        yield ds
        self._dataset = ds

    @name.setter
    def name(self, value):
        with self._set_new_dataset() as ds:
            ds.rename({self.name: value}, inplace=True)
        self._name = value

    @property
    def variable(self):
        return self._dataset._variables[self.name]

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
        """Dimension names associated with this array."""
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

    def __getitem__(self, key):
        if isinstance(key, basestring):
            return self.coords[key]
        else:
            # orthogonal array indexing
            return self.isel(**self._item_key_to_dict(key))

    def __setitem__(self, key, value):
        if isinstance(key, basestring):
            self.coords[key] = value
        else:
            # orthogonal array indexing
            self.variable[key] = value

    def __delitem__(self, key):
        del self._dataset[key]

    @property
    def __attr_sources__(self):
        """List of places to look-up items for attribute-style access"""
        return [self.coords, self.attrs]

    def __contains__(self, key):
        return key in self._dataset

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
        return Indexes(self)

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
            names = (self._dataset._coord_names - set(self.dims)
                     - set([self.name]))
        ds = self._dataset.reset_coords(names, drop, inplace)
        return ds[self.name] if drop else ds

    def load(self):
        """Manually trigger loading of this array's data from disk or a
        remote source into memory and return this array.

        Normally, it should not be necessary to call this method in user code,
        because all xray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.
        """
        self._dataset.load()
        return self

    def load_data(self):  # pragma: no cover
        warnings.warn('the DataArray method `load_data` has been deprecated; '
                      'use `load` instead',
                      FutureWarning, stacklevel=2)
        return self.load()

    def copy(self, deep=True):
        """Returns a copy of this array.

        If `deep=True`, a deep copy is made of all variables in the underlying
        dataset. Otherwise, a shallow copy is made, so each variable in the new
        array's dataset is also a variable in this array's dataset.
        """
        ds = self._dataset.copy(deep=deep)
        return self._with_replaced_dataset(ds)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        # memo does nothing but is required for compatability with
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

    def chunk(self, chunks=None):
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

        Returns
        -------
        chunked : xray.DataArray
        """
        if isinstance(chunks, (list, tuple)):
            chunks = dict(zip(self.dims, chunks))

        ds = self._dataset.chunk(chunks)
        return self._with_replaced_dataset(ds)

    def isel(self, **indexers):
        """Return a new DataArray whose dataset is given by integer indexing
        along the specified dimension(s).

        See Also
        --------
        Dataset.isel
        DataArray.sel
        """
        ds = self._dataset.isel(**indexers)
        return self._with_replaced_dataset(ds)

    def sel(self, method=None, **indexers):
        """Return a new DataArray whose dataset is given by selecting
        index labels along the specified dimension(s).

        See Also
        --------
        Dataset.sel
        DataArray.isel
        """
        return self.isel(**indexing.remap_label_indexers(self, indexers,
                                                         method=method))

    def reindex_like(self, other, method=None, copy=True):
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

            * default: don't fill gaps
            * pad / ffill: propgate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value (requires pandas>=0.16)
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
        return self.reindex(method=method, copy=copy, **other.indexes)

    def reindex(self, method=None, copy=True, **indexers):
        """Conform this object onto a new set of indexes, filling in
        missing values with NaN.

        Parameters
        ----------
        copy : bool, optional
            If `copy=True`, the returned array's dataset contains only copied
            variables. If `copy=False` and no reindexing is required then
            original variables from this array's dataset are returned.
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for filling index values in ``indexers`` not found on
            this data array:

            * default: don't fill gaps
            * pad / ffill: propgate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value (requires pandas>=0.16)
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
        ds = self._dataset.reindex(method=method, copy=copy, **indexers)
        return self._with_replaced_dataset(ds)

    def rename(self, new_name_or_name_dict):
        """Returns a new DataArray with renamed coordinates and/or a new name.


        Parameters
        ----------
        new_name_or_name_dict : str or dict-like
            If the argument is dict-like, it it used as a mapping from old
            names to new names for coordinates (and/or this array itself).
            Otherwise, use the argument as the new name for this array.

        Returns
        -------
        renamed : DataArray
            Renamed array or array with renamed coordinates.

        See Also
        --------
        Dataset.rename
        DataArray.swap_dims
        """
        if utils.is_dict_like(new_name_or_name_dict):
            name_dict = new_name_or_name_dict
            new_name = name_dict.get(self.name, self.name)
        else:
            new_name = new_name_or_name_dict
            name_dict = {self.name: new_name}
        renamed_dataset = self._dataset.rename(name_dict)
        return renamed_dataset[new_name]

    def swap_dims(self, dims_dict):
        """Returns a new DataArray with swapped dimensions.

        Parameters
        ----------
        dims_dict : dict-like
            Dictionary whose keys are current dimension names and whose values
            are new names. Each value must already be a coordinate on this
            array.
        inplace : bool, optional
            If True, swap dimensions in-place. Otherwise, return a new object.

        Returns
        -------
        renamed : Dataset
            DataArray with swapped dimensions.

        See Also
        --------

        DataArray.rename
        Dataset.swap_dims
        """
        ds = self._dataset.swap_dims(dims_dict)
        return self._with_replaced_dataset(ds)

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
        ds = self._dataset.copy()
        ds[self.name] = self.variable.transpose(*dims)
        return self._with_replaced_dataset(ds)

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
        ds = self._dataset.squeeze(dim)
        return self._with_replaced_dataset(ds)

    def drop(self, labels, dim=None):
        """Drop coordinates or index labels from this DataArray.

        Parameters
        ----------
        labels : str
            Names of coordinate variables or index labels to drop.
        dim : str, optional
            Dimension along which to drop index labels. By default (if
            ``dim is None``), drops coordinates rather than index labels.

        Returns
        -------
        dropped : DataArray
        """
        if utils.is_scalar(labels):
            labels = [labels]
        if dim is None and self.name in labels:
            raise ValueError('cannot drop this DataArray from itself')
        ds = self._dataset.drop(labels, dim)
        return self._with_replaced_dataset(ds)

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
        ds = self._dataset.dropna(dim, how=how, thresh=thresh)
        return self._with_replaced_dataset(ds)

    def fillna(self, value):
        """Fill missing values in this object.

        This operation follows the normal broadcasting and alignment rules that
        xray uses for binary arithmetic, except the result is aligned to this
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
        return self._fillna(value)

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
        ds = self._dataset.drop(set(self.dims) - set(var.dims))
        ds[self.name] = var
        return self._with_replaced_dataset(ds)

    @classmethod
    def _concat(cls, arrays, dim='concat_dim', indexers=None,
                mode='different', concat_over=None, compat='equals'):
        datasets = []
        for n, arr in enumerate(arrays):
            if n == 0:
                name = arr.name
            elif name != arr.name:
                if compat == 'identical':
                    raise ValueError('array names not identical')
                else:
                    arr = arr.rename(name)
            datasets.append(arr._dataset)

        if concat_over is None:
            concat_over = set()
        elif isinstance(concat_over, basestring):
            concat_over = set([concat_over])
        concat_over = set(concat_over) | set([name])

        ds = Dataset._concat(datasets, dim, indexers, concat_over=concat_over)
        return cls._new_from_dataset_no_copy(ds, name)

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
        return constructor(self.values, *self.indexes.values())

    def to_dataframe(self):
        """Convert this array and its coordinates into a tidy pandas.DataFrame.

        The DataFrame is indexed by the Cartesian product of index coordinates
        (in the form of a :py:class:`pandas.MultiIndex`).

        Other coordinates are included as columns in the DataFrame.
        """
        # TODO: add a 'name' parameter
        dims = OrderedDict(zip(self.dims, self.shape))
        return self._dataset._to_dataframe(dims)

    def to_series(self):
        """Convert this array into a pandas.Series.

        The Series is indexed by the Cartesian product of index coordinates
        (in the form of a :py:class:`pandas.MultiIndex`).
        """
        index = self.coords.to_index()
        return pd.Series(self.values.reshape(-1), index=index, name=self.name)

    @classmethod
    def from_series(cls, series):
        """Convert a pandas.Series into an xray.DataArray.

        If the series's index is a MultiIndex, it will be expanded into a
        tensor product of one-dimensional coordinates (filling in missing values
        with NaN). Thus this operation should be the inverse of the `to_series`
        method.
        """
        # TODO: add a 'name' parameter
        df = pd.DataFrame({series.name: series})
        ds = Dataset.from_dataframe(df)
        return cls._new_from_dataset_no_copy(ds, series.name)

    def to_cdms2(self):
        """Convert this array into a cdms2.Variable
        """
        from ..convert import to_cdms2
        return to_cdms2(self)

    @classmethod
    def from_cdms2(cls, variable):
        """Convert a cdms2.Variable into an xray.DataArray
        """
        from ..convert import from_cdms2
        return from_cdms2(variable)

    def _all_compat(self, other, compat_str):
        """Helper function for equals and identical"""
        compat = lambda x, y: getattr(x.variable, compat_str)(y.variable)
        return (utils.dict_equiv(self.coords, other.coords, compat=compat)
                and compat(self, other))

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
        does element-wise comparisions (like numpy.ndarrays).

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
            return (self.name == other.name
                    and self._all_compat(other, 'identical'))
        except (TypeError, AttributeError):
            return False

    __default_name = object()

    def _result_name(self, other=None):

        if self.name in self.dims:
            # these names match dimension, so if we preserve them we will also
            # rename indexes
            return None

        if other is None:
            # shortcut
            return self.name

        other_name = getattr(other, 'name', self.__default_name)
        other_dims = getattr(other, 'dims', ())

        if other_name in other_dims:
            # same trouble as above
            return None

        # use the same naming heuristics as pandas:
        # https://github.com/ContinuumIO/blaze/issues/458#issuecomment-51936356
        if other_name is self.__default_name or other_name == self.name:
            return self.name

        return None

    def __array_wrap__(self, obj, context=None):
        new_var = self.variable.__array_wrap__(obj, context)
        ds = self.coords.to_dataset()
        name = self._result_name()
        ds[name] = new_var
        return self._new_from_dataset_no_copy(ds, name)

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            return self.__array_wrap__(f(self.variable.data, *args, **kwargs))
        return func

    @staticmethod
    def _binary_op(f, reflexive=False, join='inner', **ignored_kwargs):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, (Dataset, groupby.GroupBy)):
                return NotImplemented
            if hasattr(other, 'indexes'):
                self, other = align(self, other, join=join, copy=False)
                empty_indexes = [d for d, s in zip(self.dims, self.shape)
                                 if s == 0]
                if empty_indexes:
                    raise ValueError('no overlapping labels for some '
                                     'dimensions: %s' % empty_indexes)
            other_coords = getattr(other, 'coords', None)
            other_variable = getattr(other, 'variable', other)
            ds = self.coords.merge(other_coords)
            name = self._result_name(other)
            ds[name] = (f(self.variable, other_variable)
                        if not reflexive
                        else f(other_variable, self.variable))
            result = self._new_from_dataset_no_copy(ds, name)
            return result
        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, groupby.GroupBy):
                raise TypeError('in-place operations between a DataArray and '
                                'a grouped object are not permitted')
            other_coords = getattr(other, 'coords', None)
            other_variable = getattr(other, 'variable', other)
            with self.coords._merge_inplace(other_coords):
                f(self.variable, other_variable)
            return self
        return func


# priority most be higher than Variable to properly work with binary ufuncs
ops.inject_all_ops_and_reduce_methods(DataArray, priority=60)
