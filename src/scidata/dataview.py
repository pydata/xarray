import re

import variable
from common import _DataWrapperMixin
from ops import inject_special_operations
from utils import expanded_indexer


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
        return self.dataset[self.name]

    @variable.setter
    def variable(self, value):
        # TODO: remove this line, so we change DataView's dataset in-place
        # (if supported by the underlying store)
        self.dataset = self.unselected()
        self.dataset[self.name] = value

    @property
    def _data(self):
        # necessary for _DataWrapperMixin
        return self.variable._data

    @property
    def dimensions(self):
        return self.variable.dimensions

    def __getitem__(self, key):
        key = expanded_indexer(key, self.ndim)
        slicers = dict(zip(self.dimensions, key))
        return type(self)(self.dataset.views(slicers), self.name)

    def __setitem__(self, key, value):
        self.variable[key] = value

    def __iter__(self):
        for n in range(len(self)):
            yield self[n]

    @property
    def attributes(self):
        return self.variable.attributes

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

    def transpose(self, *dimensions):
        ds = self.unselected()
        ds[self.name] = self.variable.transpose(*dimensions)
        return DataView(ds, self.name)


def unary_op(f):
    def func(self):
        ds = self.unselected()
        ds[self.name] = f(self.variable)
        return DataView(ds, self.name)
    return func


def binary_op(f, reflexive=False):
    def func(self, other):
        ds = self.unselected()
        other_variable = getattr(other, 'variable', other)
        ds[self.name] = (f(self.variable, other_variable)
                         if not reflexive
                         else f(other_variable, self.variable))
        if hasattr(other, 'unselected'):
            ds.update(other.unselected())
        return DataView(ds, self.name)
    return func


def inplace_binary_op(f):
    def func(self, other):
        other_variable = getattr(other, 'variable', other)
        self.variable = f(self.variable, other_variable)
        if hasattr(other, 'unselected'):
            self.dataset.update(other.unselected())
        return self
    return func


inject_special_operations(DataView, unary_op, binary_op, inplace_binary_op)

DataView.tranpose = unary_op(variable.Variable.transpose)

