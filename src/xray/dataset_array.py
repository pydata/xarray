# TODO: replace aggregate and iterator methods by a 'groupby' method/object
# like pandas
import functools
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

import xarray
import dataset as dataset_
import groupby
import ops
from common import AbstractArray
from utils import expanded_indexer, FrozenOrderedDict, remap_loc_indexers


class _LocIndexer(object):
    def __init__(self, ds_array):
        self.ds_array = ds_array

    def _remap_key(self, key):
        indexers = remap_loc_indexers(self.ds_array.dataset.variables,
                                      self.ds_array._key_to_indexers(key))
        return tuple(indexers.values())

    def __getitem__(self, key):
        return self.ds_array[self._remap_key(key)]

    def __setitem__(self, key, value):
        self.ds_array[self._remap_key(key)] = value


class DataArray(AbstractArray):
    """Hybrid between Dataset and Array.

    DataArrays are the primary way to do computations with Dataset
    variables. They are designed to make it easy to manipulate arrays in the
    context of an intact Dataset object. Indeed, the contents of a DataArray
    are uniquely defined by its `dataset` and `name` parameters.

    Getting items from or doing mathematical operations with a DataArray
    returns another DataArray.

    The design of DataArray is strongly inspired by the Iris Cube. However,
    DataArrays are much lighter weight than cubes. They are simply aligned,
    labeled datasets and do not explicitly guarantee or rely on the CF model.
    """
    def __init__(self, dataset, name):
        """
        Parameters
        ----------
        dataset : xray.Dataset
            The dataset in which to find this array.
        name : str
            The name of the variable in `dataset` to which array operations
            should be applied.
        """
        if not isinstance(dataset, dataset_.Dataset):
            dataset = dataset_.Dataset(dataset)
        if name not in dataset and name not in dataset.virtual_variables:
            raise ValueError('name %r is not a variable in dataset %r'
                             % (name, dataset))
        self._dataset = dataset
        self._name = name

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
    def data(self):
        """The variables's data as a numpy.ndarray"""
        return self.variable.data
    @data.setter
    def data(self, value):
        self.variable.data = value

    def in_memory(self):
        return self.variable.in_memory()

    @property
    def index(self):
        """The variable's data as a pandas.Index"""
        return self.variable.index

    def is_coord(self):
        return isinstance(self.variable, xarray.CoordXArray)

    @property
    def dimensions(self):
        return self.variable.dimensions

    def _key_to_indexers(self, key):
        return OrderedDict(
            zip(self.dimensions, expanded_indexer(key, self.ndim)))

    def __getitem__(self, key):
        if isinstance(key, basestring):
            # grab another dataset array from the dataset
            return self.dataset[key]
        else:
            # orthogonal array indexing
            return self.indexed_by(**self._key_to_indexers(key))

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
        """Attribute for location based indexing like pandas.
        """
        return _LocIndexer(self)

    def __iter__(self):
        for n in range(len(self)):
            yield self[n]

    @property
    def attributes(self):
        return self.variable.attributes

    @property
    def encoding(self):
        return self.variable.encoding

    @property
    def variables(self):
        return self.dataset.variables

    @property
    def coordinates(self):
        return FrozenOrderedDict((k, self.dataset.variables[k])
                                 for k in self.dimensions)

    def copy(self, deep=True):
        """Returns a copy of this array.

        If `deep=True`, a deep copy is made of all variables in the underlying
        dataset. Otherwise, a shallow copy is made, so each variable in the new
        array's dataset is also a variable in this array's dataset.
        """
        return type(self)(self.dataset.copy(deep=deep), self.name)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        # memo does nothing but is required for compatability with
        # copy.deepcopy
        return self.copy(deep=True)

    # mutable objects should not be hashable
    __hash__ = None

    def indexed_by(self, **indexers):
        """Return a new dat array whose dataset is given by indexing along
        the specified dimension(s).

        See Also
        --------
        Dataset.indexed_by
        """
        ds = self.dataset.indexed_by(**indexers)
        if self.name not in ds and self.name in self.dataset:
            # always keep the array variable in the dataset, even if it was
            # unselected because indexing made it a scaler
            # don't add back in virtual variables (not found in the dataset)
            ds[self.name] = self.variable.indexed_by(**indexers)
        return type(self)(ds, self.name)

    def labeled_by(self, **indexers):
        """Return a new DataArray whose dataset is given by selecting
        coordinate labels along the specified dimension(s).

        See Also
        --------
        Dataset.labeled_by
        """
        return self.indexed_by(**remap_loc_indexers(self.dataset.variables,
                                                    indexers))

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
        return type(self)(renamed_dataset, new_name)

    def select(self, *names):
        """Returns a new DataArray with only the named variables, as well
        as this DataArray's array variable (and all associated coordinates).

        See Also
        --------
        Dataset.select
        """
        names = names + (self.name,)
        return type(self)(self.dataset.select(*names), self.name)

    def unselect(self, *names):
        """Returns a new DataArray without the named variables.

        See Also
        --------
        Dataset.unselect
        """
        if self.name in names:
            raise ValueError('cannot unselect the array variable of a '
                             'DataArray with unselect. Use the `unselected`'
                             'method or the `unselect` method of the dataset.')
        return type(self)(self.dataset.unselect(*names), self.name)

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
        return groupby.ArrayGroupBy(self, group.name, group, squeeze=squeeze)

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
        ds = self.copy()
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
        return type(self)(self.dataset.squeeze(dimension), self.name)

    def reduce(self, func, dimension=None, axis=None, **kwargs):
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dimension : str or sequence of str, optional
            Dimension(s) over which to repeatedly apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `func`. Only one of the
            'dimension' and 'axis' arguments can be supplied. If neither are
            supplied, then the reduction is calculated over the flattened array
            (by calling `f(x)` without an axis argument).
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Notes
        -----
        If `reduce` is called with multiple dimensions (or axes, which
        are converted into dimensions), then the reduce operation is
        performed repeatedly along each dimension in turn from left to right.

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
        return type(self)(ds, self.name)

    def _unselect_nonfocus_dims(self):
        """Unselect all dimensions found in this array's dataset that aren't
        also found in the dimensions of the array. Returns either a modified
        copy or this DataArray if there were no dimensions to remove.
        """
        other_dims = [k for k in self.dataset.dimensions
                      if k not in self.dimensions]
        if other_dims:
            self = self.unselect(*other_dims)
        return self

    @classmethod
    def concat(cls, arrays, dimension='concat_dimension', indexers=None,
               length=None, template=None):
        """Stack arrays along a new or existing dimension to form a new
        DataArray.

        Parameters
        ----------
        arrays : iterable of Array
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
            array or the singleton dimension of the xarray is used as the
            stacking dimension and the array is added to the returned dataset.
        indexers : iterable of indexers, optional
            Iterable of indexers of the same length as variables which
            specifies how to assign variables along the given dimension. If
            not supplied, indexers is inferred from the length of each
            variable along the dimension, and the variables are concatenated in
            the given order.
        length : int, optional
            Length of the new dimension. This is used to allocate the new data
            array for the concatenated variable data before iterating over all
            items, which is thus more memory efficient and a bit faster. If
            dimension is provided as an array, length is calculated
            automatically.
        template : DataArray, optional
            This option is used internally to speed-up groupby operations. The
            template's attributes and variables that are not along the
            concatenated dimension are added to the returned array.
            Furthermore, if a template is given, some checks of internal
            consistency between arrays to stack are skipped.

        Returns
        -------
        concatenated : DataArray
            Concatenated DataArray formed by concatenated all the supplied
            variables along the new dimension.

        Notes
        -----
        This function has subtly different logic from Dataset.concat. It is not
        entirely clear that it is necessary or useful in the public API.

        The current implementation will fail hard if Datasets with more than
        one variable to concatenate are supplied.

        See also
        --------
        Dataset.concat
        XArray.concat
        """
        ds = dataset_.Dataset()
        dim_name = ds._add_dimension(dimension)

        if template is not None:
            # use metadata from the template data array
            name = template.name
            old_dim_name, = template[dim_name].dimensions
            ds.merge(template.dataset.unselect(old_dim_name), inplace=True)
        else:
            # figure out metadata by inspecting each array
            name = 'concat_variable'
            arrays = list(arrays)
            for array in arrays:
                if isinstance(array, cls):
                    drop = [array.name] + [k for k in array.dataset.dimensions
                                           if k == dim_name]
                    ds.merge(array.dataset.unselect(*drop), inplace=True)
                    if name is 'concat_variable':
                        name = array.name

        ds[name] = xarray.XArray.concat(
            arrays, dimension, indexers, length, template)
        return cls(ds, name)._unselect_nonfocus_dims()

    def to_dataframe(self):
        """Convert this array into a pandas.DataFrame.

        Non-coordinate variables in this array's dataset (which include this
        array's data) form the columns of the DataFrame. The DataFrame is be
        indexed by the Cartesian product of the dataset's coordinates.
        """
        return self.dataset.to_dataframe()

    def to_series(self):
        """Conver this array into a pandas.Series.

        The Series is indexed by the Cartesian product of the coordinates.
        Unlike `to_dataframe`, only this array is including in the returned
        series; the other non-coordinates variables in the dataset are not.
        """
        # np.asarray is necessary to work around a pandas bug:
        # https://github.com/pydata/pandas/issues/6439
        coords = [np.asarray(v) for v in self.coordinates.values()]
        index_names = self.coordinates.keys()
        index = pd.MultiIndex.from_product(coords, names=index_names)
        return pd.Series(self.data.reshape(-1), index=index, name=self.name)

    def _select_coordinates(self):
        dataset = self.select().dataset
        if not self.is_coord():
            del dataset[self.name]
        return dataset

    def _refocus(self, new_var, name=None):
        """Returns a copy of this DataArray's dataset with this
        DataArray's focus variable replaced by `new_var`.

        If `new_var` is a DataArray, its contents will be merged in.
        """
        if not hasattr(new_var, 'dimensions'):
            new_var = type(self.variable)(self.variable.dimensions, new_var)
        ds = self._select_coordinates()
        if name is None:
            name = self.name + '_'
        ds[name] = new_var
        return type(self)(ds, name)

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
                        and not np.array_equal(v, other.coordinates[k])):
                    raise ValueError('coordinate %r is not aligned' % k)

    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            # TODO: automatically group by other variable dimensions to allow
            # for broadcasting dimensions like 'dayofyear' against 'time'
            self._check_coordinates_compat(other)
            ds = self._select_coordinates()
            if hasattr(other, '_select_coordinates'):
                ds.merge(other._select_coordinates(), inplace=True)
            other_array = getattr(other, 'variable', other)
            other_name = getattr(other, 'name', 'other')
            name = self.name + '_' + f.__name__ + '_' + other_name
            ds[name] = (f(self.variable, other_array)
                         if not reflexive
                         else f(other_array, self.variable))
            return type(self)(ds, name)
        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            self._check_coordinates_compat(other)
            other_array = getattr(other, 'variable', other)
            self.variable = f(self.variable, other_array)
            if hasattr(other, '_select_coordinates'):
                self.dataset.merge(other._select_coordinates(), inplace=True)
            return self
        return func

ops.inject_special_operations(DataArray, priority=60)


def align(array1, array2):
    """Given two Dataset or DataArray objects, returns two new objects where
    all coordinates found on both datasets are replaced by their intersection,
    and thus are aligned for performing mathematical operations.
    """
    # TODO: automatically align when doing math with arrays, or better yet
    # calculate the union of the indices and fill in the mis-aligned data with
    # NaN.
    # TODO: generalize this function to any number of arguments
    overlapping_coords = {k: (array1.coordinates[k].index
                              & array2.coordinates[k].index)
                          for k in array1.coordinates
                          if k in array2.coordinates}
    return tuple(ar.labeled_by(**overlapping_coords)
                 for ar in [array1, array2])
