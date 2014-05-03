import functools
import operator
import warnings
from collections import defaultdict, OrderedDict

import pandas as pd

import dataset as dataset_
import indexing
import groupby
import ops
import utils
import variable
from common import AbstractArray
from utils import FrozenOrderedDict, multi_index_from_product


class _LocIndexer(object):
    def __init__(self, data_array):
        self.data_array = data_array

    def _remap_key(self, key):
        label_indexers = self.data_array._key_to_indexers(key)
        indexers = []
        for dim, label in label_indexers.iteritems():
            index = self.data_array.coordinates[dim]
            indexers.append(indexing.convert_label_indexer(index, label))
        return tuple(indexers)

    def __getitem__(self, key):
        return self.data_array[self._remap_key(key)]

    def __setitem__(self, key, value):
        self.data_array[self._remap_key(key)] = value


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
    dimensions : tuple
        Dimension names associated with this array.
    values : np.ndarray
        Access or modify DataArray values as a numpy array.
    coordinates : OrderedDict
        Dictionary of DataArray objects that label values along each dimension.
    """
    def __new__(cls, dataset, name):
        warnings.warn('the constructor for DataArray objects will change in a '
                      'future version of xray. For now, create new '
                      'DataArray objects by getting items from a Dataset '
                      'instead of calling the constructor directly.',
                      FutureWarning, stacklevel=2)
        return cls._constructor(dataset, name)

    @classmethod
    def _constructor(cls, dataset, name):
        """
        Parameters
        ----------
        dataset : xray.Dataset
            The dataset in which to find this array.
        name : str
            The name of the variable in `dataset` to which array operations
            should be applied.
        """
        obj = object.__new__(cls)
        if name not in dataset and name not in dataset.virtual_variables:
            raise ValueError('name %r must be a variable in dataset %r'
                             % (name, dataset))
        obj._dataset = dataset
        obj._name = name
        return obj

    @property
    def dataset(self):
        """The dataset with which this DataArray is associated.
        """
        return self._dataset

    @property
    def name(self):
        """The name of the variable in `dataset` to which array operations
        are applied.
        """
        return self._name

    @name.setter
    def name(self, value):
        raise AttributeError('cannot modify the name of a %s inplace; use the '
                             "'rename' method instead" % type(self).__name__)

    @property
    def variable(self):
        return self.dataset.variables[self.name]

    @variable.setter
    def variable(self, value):
        self.dataset[self.name] = value

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
        """The variable's data as a pandas.Index. Only possible for 1D arrays.
        """
        return self.variable.to_coord().as_index

    @property
    def dimensions(self):
        return self.variable.dimensions

    def _key_to_indexers(self, key):
        return OrderedDict(
            zip(self.dimensions, indexing.expanded_indexer(key, self.ndim)))

    def __getitem__(self, key):
        if isinstance(key, basestring):
            # grab another dataset array from the dataset
            return self.dataset[key]
        else:
            # orthogonal array indexing
            return self.indexed(**self._key_to_indexers(key))

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
        utils.alias_warning('attributes', 'attrs', 3)
        return self.variable.attrs

    @attributes.setter
    def attributes(self, value):
        utils.alias_warning('attributes', 'attrs', 3)
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

    @property
    def coordinates(self):
        """Dictionary of CoordVaraibles used for label based indexing.
        """
        return FrozenOrderedDict((dim, self.dataset.variables[dim])
                                 for dim in self.dimensions)

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

    def indexed(self, **indexers):
        """Return a new DataArray whose dataset is given by indexing along
        the specified dimension(s).

        See Also
        --------
        Dataset.indexed
        DataArray.labeled
        """
        ds = self.dataset.indexed(**indexers)
        return ds[self.name]

    def labeled(self, **indexers):
        """Return a new DataArray whose dataset is given by selecting
        coordinate labels along the specified dimension(s).

        See Also
        --------
        Dataset.labeled
        DataArray.indexed
        """
        return self.indexed(**indexing.remap_label_indexers(self, indexers))

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
            If `copy=True`, the returned array's dataset contains only copied
            variables. If `copy=False` and no reindexing is required then
            original variables from this array's dataset are returned.

        Returns
        -------
        reindexed : DatasetArray
            Another dataset array, with coordinates replaced from the other
            object.

        See Also
        --------
        DatasetArray.reindex
        align
        """
        return self.reindex(copy=copy, **other.coordinates)

    def reindex(self, copy=True, **coordinates):
        """Conform this object onto a new set of coordinates or pandas.Index
        objects, filling in missing values with NaN.

        Parameters
        ----------
        copy : bool, optional
            If `copy=True`, the returned array's dataset contains only copied
            variables. If `copy=False` and no reindexing is required then
            original variables from this array's dataset are returned.
        **coordinates : dict
            Dictionary with keys given by dimension names and values given by
            arrays of coordinate labels. Any mis-matched coordinates values
            will be filled in with NaN, and any mis-matched coordinate names
            will simply be ignored.

        Returns
        -------
        reindexed : DatasetArray
            Another dataset array, with replaced coordinates.

        See Also
        --------
        DatasetArray.reindex_like
        align
        """
        reindexed_ds = self.select().dataset.reindex(copy=copy, **coordinates)
        return reindexed_ds[self.name]

    def rename(self, new_name_or_name_dict):
        """Returns a new DataArray with renamed variables.

        If the argument is a string, rename this DataArray's arary variable.
        Otherwise, the argument is assumed to be a mapping from old names to
        new names for dataset variables.

        See Also
        --------
        Dataset.rename
        """
        if isinstance(new_name_or_name_dict, basestring):
            new_name = new_name_or_name_dict
            name_dict = {self.name: new_name}
        else:
            name_dict = new_name_or_name_dict
            new_name = name_dict.get(self.name, self.name)
        renamed_dataset = self.dataset.rename(name_dict)
        return renamed_dataset[new_name]

    def select(self, *names):
        """Returns a new DataArray with only the named variables, as well
        as this DataArray's array variable (and all associated coordinates).

        See Also
        --------
        Dataset.select
        """
        names = names + (self.name,)
        ds = self.dataset.select(*names)
        return ds[self.name]

    def unselect(self, *names):
        """Returns a new DataArray without the named variables.

        See Also
        --------
        Dataset.unselect
        """
        if self.name in names:
            raise ValueError('cannot unselect the name of a DataArray with '
                             'unselect. Use the `unselect` method of the '
                             'dataset instead.')
        ds = self.dataset.unselect(*names)
        return ds[self.name]

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
            iterated over in the form of `(unique_value, grouped_array)` pairs
            or over which grouped operations can be applied with the `apply`
            and `reduce` methods (and the associated aliases `mean`, `sum`,
            `std`, etc.).
        """
        if isinstance(group, basestring):
            group = self[group]
        return groupby.ArrayGroupBy(self, group, squeeze=squeeze)

    def transpose(self, *dimensions):
        """Return a new DataArray object with transposed dimensions.

        Note: Although this operation returns a view of this array's data, it
        is not lazy -- the data will be fully loaded.

        Parameters
        ----------
        *dimensions : str, optional
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
        ds[self.name] = self.variable.transpose(*dimensions)
        return ds[self.name]

    def squeeze(self, dimension=None):
        """Return a new DataArray object with squeezed data.

        Parameters
        ----------
        dimensions : None or str or tuple of str, optional
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
        ds = self.dataset.squeeze(dimension)
        return ds[self.name]

    def reduce(self, func, dimension=None, axis=None, **kwargs):
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dimension : str or sequence of str, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `func`. Only one of the
            'dimension' and 'axis' arguments can be supplied. If neither are
            supplied, then the reduction is calculated over the flattened array
            (by calling `f(x)` without an axis argument).
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            DataArray with this object's array replaced with an array with
            summarized data and the indicated dimension(s) removed.
        """
        var = self.variable.reduce(func, dimension, axis, **kwargs)
        drop = set(self.dimensions) - set(var.dimensions)
        # For now, take an aggressive strategy of removing all variables
        # associated with any dropped dimensions
        # TODO: save some summary (mean? bounds?) of dropped variables
        drop |= {k for k, v in self.dataset.variables.iteritems()
                 if any(dim in drop for dim in v.dimensions)}
        ds = self.dataset.unselect(*drop)
        ds[self.name] = var
        return ds[self.name]

    @classmethod
    def concat(cls, arrays, dimension='concat_dimension', indexers=None,
               concat_over=None):
        """Stack arrays along a new or existing dimension to form a new
        DataArray.

        Parameters
        ----------
        arrays : iterable of DataArray
            Arrays to stack together. Each variable is expected to have
            matching dimensions and shape except for along the concatenated
            dimension.
        dimension : str or Array, optional
            Name of the dimension to stack along. This can either be a new
            dimension name, in which case it is added along axis=0, or an
            existing dimension name, in which case the location of the
            dimension is unchanged. Where to insert the new dimension is
            determined by whether it is found in the first array. If dimension
            is provided as an XArray or DataArray, the name of the dataset
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
        concat_over = set(concat_over) | {name}
        ds = dataset_.Dataset.concat(datasets, dimension, indexers,
                                     concat_over=concat_over)
        return ds[name]

    def to_dataframe(self):
        """Convert this array into a pandas.DataFrame.

        Non-coordinate variables in this array's dataset (which include this
        array's data) form the columns of the DataFrame. The DataFrame is be
        indexed by the Cartesian product of the dataset's coordinates.
        """
        return self.dataset.to_dataframe()

    def to_series(self):
        """Convert this array into a pandas.Series.

        The Series is indexed by the Cartesian product of the coordinates.
        Unlike `to_dataframe`, only this array is including in the returned
        series; the other non-coordinates variables in the dataset are not.
        """
        index = multi_index_from_product(self.coordinates.values(),
                                         names=self.coordinates.keys())
        return pd.Series(self.values.reshape(-1), index=index, name=self.name)

    @classmethod
    def from_series(cls, series):
        """Convert a pandas.Series into an xray.DatasetArray

        If the series's index is a MultiIndex, it will be expanded into a
        tensor product of one-dimensional indices  (filling in missing values
        with NaN). Thus this operation should be the inverse of the `to_series`
        method.
        """
        name = series.name if series.name is not None else 'values'
        df = pd.DataFrame({name: series})
        ds = dataset_.Dataset.from_dataframe(df)
        return ds[name]

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
                        in zip(self.coordinates.items(),
                               other.coordinates.items()))
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
                            in zip(self.coordinates.items(),
                                   other.coordinates.items()))
                    and self.variable.identical(other.variable))
        except AttributeError:
            return False

    def _select_coordinates(self):
        return dataset_.Dataset(self.coordinates)

    def _refocus(self, new_var, name=None):
        """Returns a copy of this DataArray's dataset with this
        DataArray's focus variable replaced by `new_var`.

        If `new_var` is a DataArray, its contents will be merged in.
        """
        if not hasattr(new_var, 'dimensions'):
            new_var = variable.Variable(self.variable.dimensions, new_var)
        ds = self._select_coordinates()
        if name is None:
            name = self.name + '_'
        ds[name] = new_var
        return ds[name]

    def __array_wrap__(self, obj, context=None):
        return self._refocus(self.variable.__array_wrap__(obj, context))

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            return self._refocus(f(self.variable, *args, **kwargs),
                                 self.name + '_' + f.__name__)
        return func

    def _check_coordinates_compat(self, other):
        # TODO: possibly automatically select index intersection instead?
        if hasattr(other, 'coordinates'):
            for k, v in self.coordinates.iteritems():
                if (k in other.coordinates
                        and not v.equals(other.coordinates[k])):
                    raise ValueError('coordinate %r is not aligned' % k)

    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            # TODO: automatically group by other variable dimensions to allow
            # for broadcasting dimensions like 'dayofyear' against 'time'
            self._check_coordinates_compat(other)
            ds = self._select_coordinates()
            if hasattr(other, 'coordinates'):
                ds.merge(other.coordinates, inplace=True)
            other_array = getattr(other, 'variable', other)
            other_name = getattr(other, 'name', 'other')
            name = self.name + '_' + f.__name__ + '_' + other_name
            ds[name] = (f(self.variable, other_array)
                         if not reflexive
                         else f(other_array, self.variable))
            return ds[name]
        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            self._check_coordinates_compat(other)
            other_array = getattr(other, 'variable', other)
            self.variable = f(self.variable, other_array)
            if hasattr(other, 'coordinates'):
                self.dataset.merge(other.coordinates, inplace=True)
            return self
        return func

ops.inject_special_operations(DataArray, priority=60)


def align(*objects, **kwargs):
    """align(*objects, join='inner', copy=True)

    Given any number of Dataset and/or DataArray objects, returns new
    objects with aligned coordinates.

    Array from the aligned objects are suitable as input to mathematical
    operators, because along each dimension they have the same coordinates.

    Missing values (if ``join != 'inner'``) are filled with NaN.

    Parameters
    ----------
    *objects : Dataset or DatasetArray
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
        join_indices = functools.partial(reduce, operator.or_)
    elif join == 'inner':
        join_indices = functools.partial(reduce, operator.and_)
    elif join == 'left':
        join_indices = operator.itemgetter(0)
    elif join == 'right':
        join_indices = operator.itemgetter(-1)

    all_coords = defaultdict(list)
    for obj in objects:
        for k, v in obj.coordinates.iteritems():
            all_coords[k].append(v.as_index)

    # Exclude dimensions with all equal indices to avoid unnecessary reindexing
    # work.
    joined_coords = {k: join_indices(v) for k, v in all_coords.iteritems()
                     if any(not v[0].equals(idx) for idx in v[1:])}

    return tuple(obj.reindex(copy=copy, **joined_coords) for obj in objects)
