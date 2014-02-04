import functools
import re

import numpy as np

import ops
from common import _DataWrapperMixin
from utils import expanded_indexer, FrozenOrderedDict


class _LocIndexer(object):
    def __init__(self, dataview):
        self.dataview = dataview

    def _remap_key(self, key):
        return tuple(self.dataview.dataset._loc_to_int_indexer(k, v)
                     for k, v in self.dataview._key_to_slicers(key))

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
    def __init__(self, dataset, name):
        """
        Parameters
        ----------
        dataset : scidata.Dataset
            The dataset on which to build this data view.
        name : str
            The name of the "focus variable" in dataset on which this view is
            oriented.
        """
        if not name in dataset:
            raise ValueError('name %r is not a variable in dataset %r'
                             % (name, dataset))
        self.dataset = dataset
        self.name = name

    @property
    def variable(self):
        return self.dataset.variables[self.name]
    @variable.setter
    def variable(self, value):
        self.dataset.set_variable(self.name, value)

    # _data and _data.setter are necessary for _DataWrapperMixin
    @property
    def _data(self):
        return self.variable._data
    @_data.setter
    def _data(self, value):
        self.variable._data = value

    @property
    def dimensions(self):
        return self.variable.dimensions

    def _key_to_slicers(self, key):
        key = expanded_indexer(key, self.ndim)
        return zip(self.dimensions, key)

    def __getitem__(self, key):
        slicers = dict(self._key_to_slicers(key))
        return type(self)(self.dataset.views(slicers), self.name)

    def __setitem__(self, key, value):
        self.variable[key] = value

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
        return DataView(self.dataset.copy(), self.name)

    # mutable objects should not be hashable
    __hash__ = None

    def __str__(self):
        #TODO: make this less hacky
        return re.sub(' {4}(%s\s+%s)' % (self.dtype, self.name),
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
        return '<scidata.%s %r%s>' % (type(self).__name__, self.name, contents)

    def renamed(self, new_name):
        """Returns a new DataView with this DataView's focus variable renamed
        """
        renamed_dataset = self.dataset.renamed({self.name: new_name})
        return type(self)(renamed_dataset, new_name)

    def unselected(self):
        """Returns a copy of this DataView's dataset with this DataView's
        focus variable removed
        """
        return self.dataset.unselect(self.name)

    def replace_focus(self, new_var):
        """Returns a copy of this DataView's dataset with this DataView's
        focus variable replaced by 'new_var'
        """
        ds = self.dataset.replace(self.name, new_var)
        return type(self)(ds, self.name)

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
        return self.replace_focus(self.variable.transpose(*dimensions))

    def collapsed(self, f, dimension=None, axis=None, **kwargs):
        """Collapse this variable by applying `f` along some dimension(s)

        Parameters
        ----------
        f : function
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of collapsing an
            np.ndarray over an integer valued axis.
        dimension : str or sequence of str, optional
            Dimension(s) over which to repeatedly apply `f`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `f`. Only one of the
            'dimension' and 'axis' arguments can be supplied. If neither are
            supplied, then the collapse is calculated over the flattened array
            (by calling `f(x)` without an axis argument).
        **kwargs : dict
            Additional keyword arguments passed on to `f`.

        Note
        ----
        If `collapsed` is called with multiple dimensions (or axes, which
        are converted into dimensions), then the collapse operation is
        performed repeatedly along each dimension in turn from left to right.

        Returns
        -------
        collapsed : DataView
            DataView with this dataview's variable replaced with a variable
            with summarized data and the indicated dimension(s) removed.
        """
        var = self.variable.collapsed(f, dimension, axis, **kwargs)
        dropped_dims = set(self.dimensions) - set(var.dimensions)
        # For now, take an aggressive strategy of removing all variables
        # associated with any dropped dimensions
        # TODO: save some summary (mean? bounds?) of dropped variables
        drop = ({self.name} | dropped_dims |
                {k for k, v in self.dataset.variables.iteritems()
                if any(dim in dropped_dims for dim in v.dimensions)})
        ds = self.dataset.unselect(*drop)
        ds.add_variable(self.name, var)
        return type(self)(ds, self.name)

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self):
            return self.replace_focus(f(self.variable))
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
            dv = self.replace_focus(f(self.variable, other_variable)
                                    if not reflexive
                                    else f(other_variable, self.variable))
            if hasattr(other, 'unselected'):
                dv.dataset.update(other.unselected())
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
                self.dataset.update(other.unselected())
            return self
        return func


ops.inject_special_operations(DataView)
