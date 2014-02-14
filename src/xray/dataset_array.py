# TODO: replace aggregate and iterator methods by a 'groupby' method/object
# like pandas
import functools
import re
from collections import OrderedDict

import numpy as np

import array_
import dataset
import groupby
import ops
from common import AbstractArray
from utils import expanded_indexer, FrozenOrderedDict, remap_loc_indexers


class _LocIndexer(object):
    def __init__(self, array):
        self.array = array

    def _remap_key(self, key):
        indexers = remap_loc_indexers(self.array.indices,
                                      self.array._key_to_indexers(key))
        return tuple(indexers.values())

    def __getitem__(self, key):
        return self.array[self._remap_key(key)]

    def __setitem__(self, key, value):
        self.array[self._remap_key(key)] = value


class DatasetArray(AbstractArray):
    """A Dataset wrapper oriented around a single Array

    Dataviews are the primary way to do computations with Dataset variables.
    They are designed to make it easy to manipulate variables in the context of
    an intact Dataset object. Getting items from or doing mathematical
    operations with a dataset array returns another dataset array.

    The design of DatasetArray is strongly inspired by the Iris Cube. However,
    dataset arrays are much lighter weight than cubes. They are simply aligned,
    labeled datasets and do not explicitly guarantee or rely on the CF model.
    """
    def __init__(self, dataset, focus):
        """
        Parameters
        ----------
        dataset : xray.Dataset
            The dataset on which to build this data view.
        focus : str
            The name of the "focus variable" in `dataset` on which this object
            is oriented.
        """
        if not focus in dataset:
            raise ValueError('focus %r is not a variable in dataset %r'
                             % (focus, dataset))
        self.dataset = dataset
        self.focus = focus

    @classmethod
    def create(cls, focus, dimensions, data):
        ds = dataset.Dataset()
        ds.create_variable(focus, dimensions, data)
        return ds[focus]

    @property
    def variable(self):
        return self.dataset.variables[self.focus]
    @variable.setter
    def variable(self, value):
        self.dataset.set_variable(self.focus, value)

    # _data is necessary for AbstractArray
    @property
    def _data(self):
        return self.variable._data

    @property
    def data(self):
        """The dataset array's data as a numpy.ndarray"""
        return self.variable.data
    @data.setter
    def data(self, value):
        self.variable.data = value

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

    def __contains__(self, key):
        return key in self.dataset

    @property
    def loc(self):
        """Attribute for location based indexing with pandas
        """
        return _LocIndexer(self)

    def __iter__(self):
        for n in range(len(self)):
            yield self[n]

    @property
    def attributes(self):
        return self.variable.attributes

    @property
    def indices(self):
        return FrozenOrderedDict((k, v) for k, v
                                 in self.dataset.indices.iteritems()
                                 if k in self.dimensions)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        # shallow copy the underlying dataset
        return DatasetArray(self.dataset.copy(), self.focus)

    # mutable objects should not be hashable
    __hash__ = None

    def __str__(self):
        #TODO: make this less hacky
        return re.sub(' {4}(%s\s+%s)' % (self.dtype, self.focus),
                      r'--> \1', str(self.dataset))

    def __repr__(self):
        if self.ndim > 0:
            dim_summary = ', '.join('%s%s: %s' %
                                    ('@' if k in self.dataset else '', k, v)
                                    for k, v in zip(self.dimensions,
                                                    self.shape))
            contents = ' (%s): %s' % (dim_summary, self.dtype)
        else:
            contents = ': %s' % self.data
        return '<xray.%s %r%s>' % (type(self).__name__, self.focus, contents)

    def indexed_by(self, **indexers):
        """Return a new dataset array whose dataset is given by indexing along
        the specified dimension(s)

        See Also
        --------
        Dataset.indexed_by
        """
        ds = self.dataset.indexed_by(**indexers)
        if self.focus not in ds:
            # always keep focus variable in the dataset, even if it was
            # unselected because indexing made it a scaler
            ds[self.focus] = self.variable.indexed_by(**indexers)
        return type(self)(ds, self.focus)

    def labeled_by(self, **indexers):
        """Return a new dataset array whose dataset is given by selecting
        coordinate labels along the specified dimension(s)

        See Also
        --------
        Dataset.labeled_by
        """
        return self.indexed_by(**remap_loc_indexers(self.indices, indexers))

    def renamed(self, new_name):
        """Returns a new DatasetArray with this DatasetArray's focus variable
        renamed
        """
        renamed_dataset = self.dataset.renamed({self.focus: new_name})
        return type(self)(renamed_dataset, new_name)

    def unselected(self):
        """Returns a copy of this DatasetArray's dataset with this
        DatasetArray's focus variable removed
        """
        return self.dataset.unselect(self.focus)

    def unselect(self, *names):
        if self.focus in names:
            raise ValueError('cannot unselect the focus variable of a '
                             'DatasetArray with unselect. Use the `unselected`'
                             'method or the `unselect` method of the dataset.')
        return type(self)(self.dataset.unselect(*names), self.focus)

    def refocus(self, new_var):
        """Returns a copy of this DatasetArray's dataset with this
        DatasetArray's focus variable replaced by `new_var`

        If `new_var` is a dataview, its contents will be merged in.
        """
        if not hasattr(new_var, 'dimensions'):
            new_var = type(self.variable)(self.variable.dimensions, new_var)
        ds = self.unselected()
        ds[self.focus] = new_var
        return type(self)(ds, self.focus)

    def iterator(self, dimension):
        """Iterate along a data dimension

        Returns an iterator yielding (coordinate, dataview) pairs for each
        coordinate value along the specified dimension.

        Parameters
        ----------
        dimension : string
            The dimension along which to iterate.

        Returns
        -------
        it : iterator
            The returned iterator yields pairs of scalar-valued coordinate
            arrays and DatasetArray objects.
        """
        for (x, ds) in self.dataset.iterator(dimension):
            yield (x, type(self)(ds, self.focus))

    def groupby(self, group, squeeze=True):
        if isinstance(group, basestring):
            # merge in the group's dataset to allow group to be a virtual
            # variable in this dataset
            ds = self.dataset.merge(self.dataset[group].dataset)
            group = DatasetArray(ds, group)
        return groupby.GroupBy(self, group.focus, group, squeeze=squeeze)

    def transpose(self, *dimensions):
        """Return a new DatasetArray object with transposed dimensions

        Note: Although this operation returns a view of this array's data, it
        is not lazy -- the data will be fully loaded.

        Parameters
        ----------
        *dimensions : str, optional
            By default, reverse the dimensions. Otherwise, reorder the
            dimensions to this order.

        Returns
        -------
        transposed : DatasetArray
            The returned DatasetArray's variable is transposed.

        See Also
        --------
        numpy.transpose
        Array.transpose
        """
        return self.refocus(self.variable.transpose(*dimensions))

    def collapse(self, func, dimension=None, axis=None, **kwargs):
        """Collapse this array by applying `func` along some dimension(s)

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of collapsing an
            np.ndarray over an integer valued axis.
        dimension : str or sequence of str, optional
            Dimension(s) over which to repeatedly apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `func`. Only one of the
            'dimension' and 'axis' arguments can be supplied. If neither are
            supplied, then the collapse is calculated over the flattened array
            (by calling `f(x)` without an axis argument).
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Note
        ----
        If `collapse` is called with multiple dimensions (or axes, which
        are converted into dimensions), then the collapse operation is
        performed repeatedly along each dimension in turn from left to right.

        Returns
        -------
        collapsed : DatasetArray
            DatasetArray with this object's array replaced with an array with
            summarized data and the indicated dimension(s) removed.
        """
        var = self.variable.collapse(func, dimension, axis, **kwargs)
        drop = set(self.dimensions) - set(var.dimensions)
        # For now, take an aggressive strategy of removing all variables
        # associated with any dropped dimensions
        # TODO: save some summary (mean? bounds?) of dropped variables
        drop |= {k for k, v in self.dataset.variables.iteritems()
                 if any(dim in drop for dim in v.dimensions)}
        ds = self.dataset.unselect(*drop)
        ds.add_variable(self.focus, var)
        return type(self)(ds, self.focus)

    def aggregate(self, func, new_dim, **kwargs):
        """Aggregate this array by applying `func` to grouped elements

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to reduce an np.ndarray over an
            integer valued axis.
        new_dim : str or DatasetArray
            Name of a variable in this array's dataset or DatasetArray by which
            to group variable elements. The dimension along which this variable
            exists will be replaced by this name. The array must be one-
            dimensional.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        aggregated : DatasetArray
            DatasetArray with aggregated data and the new dimension `new_dim`.
        """
        if isinstance(new_dim, basestring):
            new_dim = self.dataset[new_dim]
        unique, aggregated = self.variable.aggregate(
            func, new_dim.focus, new_dim, **kwargs)
        # TODO: add options for how to summarize variables along aggregated
        # dimensions instead of just dropping them?
        drop = {k for k, v in self.dataset.variables.iteritems()
                if any(dim in new_dim.dimensions for dim in v.dimensions)}
        ds = self.dataset.unselect(*drop)
        ds.add_coordinate(unique)
        ds.add_variable(self.focus, aggregated)
        return type(self)(ds, self.focus)

    @classmethod
    def from_stack(cls, arrays, dimension='stacked_dimension',
                   stacked_indexers=None, length=None, template=None):
        """Stack arrays along a new or existing dimension to form a new
        dataview

        Parameters
        ----------
        arrays : iterable of Array
            Arrays to stack together. Each variable is expected to have
            matching dimensions and shape except for along the stacked
            dimension.
        dimension : str or Array, optional
            Name of the dimension to stack along. This can either be a new
            dimension name, in which case it is added along axis=0, or an
            existing dimension name, in which case the location of the
            dimension is unchanged. Where to insert the new dimension is
            determined by whether it is found in the first array.
        stacked_indexers : optional
        length : optional
        template : optional

        Returns
        -------
        stacked : DatasetArray
            Stacked dataset array formed by stacking all the supplied variables
            along the new dimension.
        """
        # create an empty dataset in which to stack variables
        # start by putting in the dimension variable
        ds = dataset.Dataset()
        if isinstance(dimension, basestring):
            dim_name = dimension
        else:
            dim_name, = dimension.dimensions
            if hasattr(dimension, 'focus'):
                ds[dimension.focus] = dimension

        if template is not None:
            # use metadata from the template dataset array
            focus = template.focus
            drop = {k for k, v in template.dataset.variables.iteritems()
                    if k in [focus, dim_name]}
            ds.merge(template.dataset.unselect(*drop), inplace=True)
        else:
            # figure out metadata by inspecting each array
            focus = None
            arrays = list(arrays)
            for array in arrays:
                if isinstance(array, cls):
                    unselected = array.unselected()
                    if dim_name in unselected:
                        unselected = unselected.unselect(dim_name)
                    ds.merge(unselected, inplace=True)
                    if focus is None:
                        focus = array.focus
                    elif focus != array.focus:
                        raise ValueError('DatasetArray.from_stack requires '
                                         'that all stacked views have the '
                                         'same focus')
            if focus is None:
                focus = 'stacked_variable'

        # finally, merge in the stacked variables
        ds[focus] = array_.Array.from_stack(arrays, dimension,
                                            stacked_indexers, length, template)
        stacked = cls(ds, focus)

        if template is not None:
            drop = set(template.dataset.dimensions) - set(stacked.dimensions)
            drop |= {k for k, v in ds.variables.iteritems()
                     if any(dim in drop for dim in v.dimensions)}
            stacked = stacked.unselect(*drop)
        return stacked

    def to_dataframe(self):
        """Convert this array into a pandas.DataFrame

        Non-coordinate variables in this array's dataset (which include the
        view's data) form the columns of the DataFrame. The DataFrame is be
        indexed by the Cartesian product of the dataset's indices.
        """
        return self.dataset.to_dataframe()

    def __array_wrap__(self, result):
        return self.refocus(self.variable.__array_wrap__(result))

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            return self.refocus(f(self.variable, *args, **kwargs))
        return func

    def _check_indices_compat(self, other):
        # TODO: possibly automatically select index intersection instead?
        if hasattr(other, 'indices'):
            for k, v in self.indices.iteritems():
                if (k in other.indices
                        and not np.array_equal(v, other.indices[k])):
                    raise ValueError('index %r is not aligned' % k)

    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            # TODO: automatically group by other variable dimensions
            self._check_indices_compat(other)
            ds = self.unselected()
            if hasattr(other, 'unselected'):
                ds.merge(other.unselected(), inplace=True)
            other_variable = getattr(other, 'variable', other)
            ds[self.focus] = (f(self.variable, other_variable)
                              if not reflexive
                              else f(other_variable, self.variable))
            return ds[self.focus]
        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            self._check_indices_compat(other)
            other_variable = getattr(other, 'variable', other)
            self.variable = f(self.variable, other_variable)
            if hasattr(other, 'unselected'):
                self.dataset.merge(other.unselected(), inplace=True)
            return self
        return func

ops.inject_special_operations(DatasetArray, priority=60)


def intersection(array1, array2):
    """Given two dataset array objects, returns two new dataset arrays where
    all indices found on both arrays are replaced by their intersection
    """
    # TODO: automatically calculate the intersection when doing math with
    # arrays, or better yet calculate the union of the indices and fill in
    # the mis-aligned data with NaN.
    overlapping_indices = {k: array1.indices[k] & array2.indices[k]
                           for k in array1.indices if k in array2.indices}
    return tuple(dv.labeled_by(**overlapping_indices)
                 for dv in [array1, array2])
