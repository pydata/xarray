# TODO: replace aggregate and iterator methods by a 'groupby' method/object
# like pandas
import functools
import re

import numpy as np

import dataset
import ops
import variable
from common import _DataWrapperMixin
from utils import expanded_indexer, FrozenOrderedDict


class _LocIndexer(object):
    def __init__(self, dataview):
        self.dataview = dataview

    def _remap_key(self, key):
        return tuple(self.dataview.dataset._loc_to_int_indexer(k, v)
                     for k, v in self.dataview._key_to_indexers(key))

    def __getitem__(self, key):
        return self.dataview[self._remap_key(key)]

    def __setitem__(self, key, value):
        self.dataview[self._remap_key(key)] = value


class DataView(_DataWrapperMixin):
    """
    A Dataset wrapper oriented around a single Variable

    Dataviews are the primary way to do computations with Dataset variables.
    They are designed to make it easy to manipulate variables in the context of
    an intact Dataset object. Getting items from or doing mathematical
    operations with a dataview returns another dataview.

    The design of dataviews is strongly inspired by the Iris Cube. However,
    dataviews are much lighter weight than cubes. They are simply aligned,
    labeled datasets and do not explicitly guarantee or rely on the CF model.
    """
    def __init__(self, dataset, focus):
        """
        Parameters
        ----------
        dataset : scidata.Dataset
            The dataset on which to build this data view.
        focus : str
            The name of the "focus variable" in dataset on which this view is
            oriented.
        """
        if not focus in dataset:
            raise ValueError('focus %r is not a variable in dataset %r'
                             % (focus, dataset))
        self.dataset = dataset
        self.focus = focus

    @property
    def variable(self):
        return self.dataset.variables[self.focus]
    @variable.setter
    def variable(self, value):
        self.dataset.set_variable(self.focus, value)

    # _data is necessary for _DataWrapperMixin
    @property
    def _data(self):
        return self.variable._data

    @property
    def data(self):
        """The dataview's data as a numpy.ndarray"""
        return self.variable.data
    @data.setter
    def data(self, value):
        self.variable.data = value

    @property
    def dimensions(self):
        return self.variable.dimensions

    def _key_to_indexers(self, key):
        return zip(self.dimensions, expanded_indexer(key, self.ndim))

    def __getitem__(self, key):
        if isinstance(key, basestring):
            # grab another dataview from the dataset
            return self.dataset[key]
        else:
            # orthogonal array indexing
            return self.indexed_by(**dict(self._key_to_indexers(key)))

    def __setitem__(self, key, value):
        if isinstance(key, basestring):
            # add a variable or dataview to the dataset
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
        return DataView(self.dataset.copy(), self.focus)

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
        return '<scidata.%s %r%s>' % (type(self).__name__, self.focus, contents)

    def indexed_by(self, **indexers):
        """Return a new dataview whose dataset is given by indexing along the
        specified dimension(s)

        See Also
        --------
        Dataset.indexed_by
        """
        return type(self)(self.dataset.indexed_by(**indexers), self.focus)

    def labeled_by(self, **indexers):
        """Return a new dataview whose dataset is given by selecting coordinate
        labels along the specified dimension(s)

        See Also
        --------
        Dataset.labeled_by
        """
        return type(self)(self.dataset.labeled_by(**indexers), self.focus)

    def renamed(self, new_name):
        """Returns a new DataView with this DataView's focus variable renamed
        """
        renamed_dataset = self.dataset.renamed({self.focus: new_name})
        return type(self)(renamed_dataset, new_name)

    def unselected(self):
        """Returns a copy of this DataView's dataset with this DataView's
        focus variable removed
        """
        return self.dataset.unselect(self.focus)

    def refocus(self, new_var):
        """Returns a copy of this DataView's dataset with this DataView's
        focus variable replaced by 'new_var'
        """
        if not hasattr(new_var, 'dimensions'):
            new_var = type(self.variable)(self.variable.dimensions, new_var)
        ds = self.dataset.replace(self.focus, new_var)
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
            variables and DataView objects.
        """
        for (x, dataset) in self.dataset.iterator(dimension):
            yield (x, type(self)(dataset, self.focus))

    def transpose(self, *dimensions):
        """Return a new DataView object with transposed dimensions

        Note: Although this operation returns a view of this dataview's
        variable's data, it is not lazy -- the data will be fully loaded.

        Parameters
        ----------
        *dimensions : str, optional
            By default, reverse the dimensions. Otherwise, reorder the
            dimensions to this order.

        Returns
        -------
        transposed : DataView
            The returned DataView's variable is transposed.

        See Also
        --------
        numpy.transpose
        Variable.tranpose
        """
        return self.refocus(self.variable.transpose(*dimensions))

    def collapse(self, func, dimension=None, axis=None, **kwargs):
        """Collapse this variable by applying `func` along some dimension(s)

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
        collapsed : DataView
            DataView with this dataview's variable replaced with a variable
            with summarized data and the indicated dimension(s) removed.
        """
        var = self.variable.collapse(func, dimension, axis, **kwargs)
        dropped_dims = set(self.dimensions) - set(var.dimensions)
        # For now, take an aggressive strategy of removing all variables
        # associated with any dropped dimensions
        # TODO: save some summary (mean? bounds?) of dropped variables
        drop = ({self.focus} | dropped_dims |
                {k for k, v in self.dataset.variables.iteritems()
                if any(dim in dropped_dims for dim in v.dimensions)})
        ds = self.dataset.unselect(*drop)
        ds.add_variable(self.focus, var)
        return type(self)(ds, self.focus)

    def aggregate(self, func, new_dim, **kwargs):
        """Aggregate this dataview by applying `func` to grouped elements

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to reduce an np.ndarray over an
            integer valued axis.
        new_dim : str or DataView
            Name of a variable in this dataview's dataset or DataView by which
            to group variable elements. The dimension along which this variable
            exists will be replaced by this name. The variable or dataview must
            be one-dimensional.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        aggregated : DataView
            DataView with aggregated data and the new dimension `new_dim`.
        """
        if isinstance(new_dim, basestring):
            new_dim = self.dataset[new_dim]
        unique, aggregated = self.variable.aggregate(
            func, new_dim.focus, new_dim, **kwargs)
        # TODO: add options for how to summarize variables along aggregated
        # dimensions instead of just dropping them?
        drop = ({self.focus} |
                ({new_dim.focus} if new_dim.focus in self.dataset else set()) |
                {k for k, v in self.dataset.variables.iteritems()
                 if any(dim in new_dim.dimensions for dim in v.dimensions)})
        ds = self.dataset.unselect(*drop)
        ds.add_coordinate(unique)
        ds.add_variable(self.focus, aggregated)
        return type(self)(ds, self.focus)

    @classmethod
    def from_stack(cls, dataviews, new_dim_name='stacked_dimension'):
        """Stack dataviews along a new dimension to form a new dataview

        Parameters
        ----------
        dataviews : iterable of Variable and/or DataView
            Variables and/or DataView objects to stack together.
        dim : str, optional
            Name of the new dimension.

        Returns
        -------
        stacked : DataView
            Stacked dataview formed by stacking all the supplied variables
            along the new dimension. The new dimension will be the first
            dimension in the stacked dataview.
        """
        views = list(dataviews)
        if not views:
            raise ValueError('DataView.from_stack was supplied with an '
                             'empty argument')
        ds = dataset.Dataset()
        focus = default_focus = 'stacked_variable'
        for view in views:
            if isinstance(view, cls):
                ds.merge(view.unselected(), inplace=True)
                if focus == default_focus:
                    focus = view.focus
                elif focus != view.focus:
                    raise ValueError('DataView.from_stack requires that all '
                                     'stacked views have the same focus')
        ds[focus] = variable.Variable.from_stack(dataviews, new_dim_name)
        return cls(ds, focus)

    def to_dataframe(self):
        """Convert this dataview into a pandas.DataFrame

        Non-coordinate variables in this dataview's dataset (which include the
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
            self._check_indices_compat(other)
            other_variable = getattr(other, 'variable', other)
            dv = self.refocus(f(self.variable, other_variable)
                                    if not reflexive
                                    else f(other_variable, self.variable))
            if hasattr(other, 'unselected'):
                dv.dataset.merge(other.unselected(), inplace=True)
            return dv
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

ops.inject_special_operations(DataView, priority=60)


def intersection(dataview1, dataview2):
    """Given two dataview objects, returns two new dataviews where all indices
    found on both dataviews are replaced by their intersection
    """
    # TODO: automatically calculate the intersection when doing math with
    # dataviews, or better yet calculate the union of the indices and fill in
    # the mis-aligned data with NaN.
    overlapping_indices = {k: dataview1.indices[k] & dataview2.indices[k]
                           for k in dataview1.indices
                           if k in dataview2.indices}
    return tuple(dv.labeled_by(**overlapping_indices)
                 for dv in [dataview1, dataview2])

