import functools
import warnings
from collections import OrderedDict
from itertools import izip

import numpy as np

import conventions
import dataset
import dataset_array
import groupby
import ops
import utils
from common import AbstractArray


def _as_compatible_data(data):
    """If data does not have the necessary attributes to be the private _data
    attribute, convert it to a np.ndarray and raise an warning
    """
    # don't check for __len__ or __iter__ so as not to warn if data is a numpy
    # numeric type like np.float32
    required = ['dtype', 'shape', 'size', 'ndim']
    if not all(hasattr(data, attr) for attr in required):
        data = np.asarray(data)
        if data.ndim == 0:
            # unpack 0d data
            data = data[()]
    elif isinstance(data, AbstractArray):
        # we don't want nested Array objects
        data = data.data
    return data


class Array(AbstractArray):
    """A netcdf-like variable consisting of dimensions, data and attributes
    which describe a single Array. A single Array object is not fully described
    outside the context of its parent Dataset (if you want such a fully
    described object, use a DatasetArray instead).
    """
    def __init__(self, dims, data, attributes=None, indexing_mode='numpy'):
        """
        Parameters
        ----------
        dims : str or sequence of str
            Name(s) of the the data dimension(s). Must be either a string (only
            for 1D data) or a sequence of strings with length equal to the
            number of dimensions.
        data : array_like
            Data array which supports numpy-like data access.
        attributes : dict_like or None, optional
            Attributes to assign to the new variable. If None (default), an
            empty attribute dictionary is initialized.
        indexing_mode : {'numpy', 'orthogonal'}
            String indicating how the data parameter handles fancy indexing
            (with arrays). Two modes are supported: 'numpy' (fancy indexing
            like numpy.ndarray objects) and 'orthogonal' (array indexing
            accesses different dimensions independently, like netCDF4
            variables). Accessing data from a Array always uses orthogonal
            indexing, so `indexing_mode` tells the variable whether index
            lookups need to be internally converted to numpy-style indexing.
        """
        if isinstance(dims, basestring):
            dims = (dims,)
        self._dimensions = tuple(dims)
        self._data = _as_compatible_data(data)
        if len(dims) != self.ndim:
            raise ValueError('data and dimensions must have the same '
                             'dimensionality')
        if attributes is None:
            attributes = {}
        self._attributes = OrderedDict(attributes)
        self._indexing_mode = indexing_mode

    @property
    def data(self):
        """The variable's data as a numpy.ndarray"""
        if not isinstance(self._data, (np.ndarray, np.string_)):
            self._data = np.asarray(self._data[...])
            self._indexing_mode = 'numpy'
        return self._data

    @data.setter
    def data(self, value):
        # allow any array to support pandas.Index objects
        value = np.asanyarray(value)
        if value.shape != self.shape:
            raise ValueError("replacement data must match the Array's "
                             "shape")
        self._data = value
        self._indexing_mode = 'numpy'

    @property
    def dimensions(self):
        return self._dimensions

    def _convert_indexer(self, key, indexing_mode=None):
        """Converts an orthogonal indexer into a fully expanded key (of the
        same length as dimensions) suitable for indexing a data array with the
        given indexing_mode.

        See Also
        --------
        utils.expanded_indexer
        utils.orthogonal_indexer
        """
        if indexing_mode is None:
            indexing_mode = self._indexing_mode
        key = utils.expanded_indexer(key, self.ndim)
        if (indexing_mode == 'numpy'
                and any(not isinstance(k, (int, slice)) for k in key)):
            # key would trigger fancy indexing
            key = utils.orthogonal_indexer(key, self.shape)
        return key

    def __getitem__(self, key):
        """Return a new Array object whose contents are consistent with
        getting the provided key from the underlying data

        NB. __getitem__ and __setitem__ implement "orthogonal indexing" like
        netCDF4-python, where the key can only include integers, slices
        (including `Ellipsis`) and 1d arrays, each of which are applied
        orthogonally along their respective dimensions.

        The difference not matter in most cases unless you are using numpy's
        "fancy indexing," which can otherwise result in data arrays
        with shapes is inconsistent (or just uninterpretable with) with the
        variable's dimensions.

        If you really want to do indexing like `x[x > 0]`, manipulate the numpy
        array `x.data` directly.
        """
        key = self._convert_indexer(key)
        dimensions = [dim for k, dim in zip(key, self.dimensions)
                      if not isinstance(k, int)]
        if len(key) == 1:
            # unpack key so it can index a pandas.Index object (pandas.Index
            # objects don't like tuples)
            key, = key
        # do location based indexing if supported by _data
        new_data = getattr(self._data, 'iloc', self._data)[key]
        # orthogonal indexing should ensure the dimensionality is consistent
        if hasattr(new_data, 'ndim'):
            assert new_data.ndim == len(dimensions)
        else:
            assert len(dimensions) == 0
        # return a variable with the same indexing_mode, because data should
        # still be the same type as _data
        return type(self)(dimensions, new_data, self.attributes,
                          indexing_mode=self._indexing_mode)

    def __setitem__(self, key, value):
        """__setitem__ is overloaded to access the underlying numpy data with
        orthogonal indexing (see __getitem__ for more details)
        """
        self.data[self._convert_indexer(key, indexing_mode='numpy')] = value

    def __iter__(self):
        for n in range(len(self)):
            yield self[n]

    @property
    def attributes(self):
        return self._attributes

    def copy(self):
        """Returns a shallow copy of the current object. The data array is
        always loaded into memory.
        """
        return self.__copy__()

    def _copy(self, deepcopy=False):
        # np.array always makes a copy
        data = np.array(self._data) if deepcopy else self.data
        # note:
        # dimensions is already an immutable tuple
        # attributes will be copied when the new Array is created
        return type(self)(self.dimensions, data, self.attributes)

    def __copy__(self):
        return self._copy(deepcopy=False)

    def __deepcopy__(self, memo=None):
        # memo does nothing but is required for compatability with
        # copy.deepcopy
        return self._copy(deepcopy=True)

    # mutable objects should not be hashable
    __hash__ = None

    def __str__(self):
        """Create a ncdump-like summary of the object"""
        summary = ["dimensions:"]
        # prints dims that look like:
        #    dimension = length
        dim_print = lambda d, l : "\t%s : %s" % (conventions.pretty_print(d, 30),
                                                 conventions.pretty_print(l, 10))
        # add each dimension to the summary
        summary.extend([dim_print(d, l) for d, l in zip(self.dimensions, self.shape)])
        summary.append("dtype : %s" % (conventions.pretty_print(self.dtype, 8)))
        summary.append("attributes:")
        #    attribute:value
        summary.extend(["\t%s:%s" % (conventions.pretty_print(att, 30),
                                     conventions.pretty_print(val, 30))
                        for att, val in self.attributes.iteritems()])
        # create the actual summary
        return '\n'.join(summary).replace('\t', ' ' * 4)

    def __repr__(self):
        if self.ndim > 0:
            dim_summary = ', '.join('%s: %s' % (k, v) for k, v
                                    in zip(self.dimensions, self.shape))
            contents = ' (%s): %s' % (dim_summary, self.dtype)
        else:
            contents = ': %s' % self.data
        return '<xray.%s%s>' % (type(self).__name__, contents)

    def indexed_by(self, **indexers):
        """Return a new array indexed along the specified dimension(s)

        Parameters
        ----------
        **indexers : {dim: indexer, ...}
            Keyword arguments with names matching dimensions and values given
            by integers, slice objects or arrays.

        Returns
        -------
        obj : Array object
            A new Array with the selected data and dimensions. In general,
            the new variable's data will be a view of this variable's data,
            unless numpy fancy indexing was triggered by using an array
            indexer, in which case the data will be a copy.
        """
        invalid = [k for k in indexers if not k in self.dimensions]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        key = [slice(None)] * self.data.ndim
        for i, dim in enumerate(self.dimensions):
            if dim in indexers:
                key[i] = indexers[dim]
        return self[tuple(key)]

    def transpose(self, *dimensions):
        """Return a new Array object with transposed dimensions

        Note: Although this operation returns a view of this variable's data,
        it is not lazy -- the data will be fully loaded.

        Parameters
        ----------
        *dimensions : str, optional
            By default, reverse the dimensions. Otherwise, reorder the
            dimensions to this order.

        Returns
        -------
        transposed : Array
            The returned object has transposed data and dimensions with the
            same attributes as the original.

        See Also
        --------
        numpy.transpose
        """
        if len(dimensions) == 0:
            dimensions = self.dimensions[::-1]
        axes = [self.dimensions.index(dim) for dim in dimensions]
        data = self.data.transpose(*axes)
        return type(self)(dimensions, data, self.attributes)

    def reduce(self, func, dimension=None, axis=None, **kwargs):
        """Reduce this array by applying `func` along some dimension(s)

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dimension : str or sequence of str, optional
            Dimension(s) over which to repeatedly apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `func`. Only one of the
            'dimension' and 'axis' arguments can be supplied. If neither are
            supplied, then the reduction is calculated over the flattened array
            (by calling `func(x)` without an axis argument).
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Note
        ----
        If `reduce` is called with multiple dimensions (or axes, which
        are converted into dimensions), then the reduce operation is
        performed repeatedly along each dimension in turn from left to right.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dimension is not None and axis is not None:
            raise ValueError("cannot supply both 'axis' and 'dimension' "
                             "arguments")

        if axis is not None:
            # determine dimensions
            if isinstance(axis, int):
                axis = [axis]
            dimension = [self.dimensions[i] for i in axis]

        if dimension is not None:
            if isinstance(dimension, basestring):
                dimension = [dimension]
            var = self
            for dim in dimension:
                var = var._reduce(func, dim, **kwargs)
        else:
            var = type(self)([], func(self.data, **kwargs), self.attributes)
            var._append_to_cell_methods(': '.join(self.dimensions)
                                        + ': ' + func.__name__)
        return var

    def _append_to_cell_methods(self, string):
        if 'cell_methods' in self.attributes:
            base = self.attributes['cell_methods'] + ' '
        else:
            base = ''
        self.attributes['cell_methods'] = base + string

    def _reduce(self, f, dim, **kwargs):
        """Reduce a single dimension"""
        axis = self.dimensions.index(dim)
        dims = tuple(dim for i, dim in enumerate(self.dimensions)
                     if axis not in [i, i - self.ndim])
        data = f(self.data, axis=axis, **kwargs)
        new_var = type(self)(dims, data, self.attributes)
        new_var._append_to_cell_methods(self.dimensions[axis]
                                        + ': ' + f.__name__)
        return new_var

    def groupby(self, group_name, group_array, squeeze=True):
        """Group this dataset by unique values of the indicated group

        Parameters
        ----------
        group_name : str
            Name of the group array.
        group_array : Array
            Array whose unique values should be used to group this array.
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
        return groupby.ArrayGroupBy(
            self, group_name, group_array, squeeze=squeeze)

    @classmethod
    def from_stack(cls, variables, dimension='stacked_dimension',
                   stacked_indexers=None, length=None, template=None):
        """Stack variables along a new or existing dimension to form a new
        variable

        Parameters
        ----------
        variables : iterable of Array
            Arrays to stack together. Each variable is expected to have
            matching dimensions and shape except for along the stacked
            dimension.
        dimension : str or DatasetArray, optional
            Name of the dimension to stack along. This can either be a new
            dimension name, in which case it is added along axis=0, or an
            existing dimension name, in which case the location of the
            dimension is unchanged. Where to insert the new dimension is
            determined by the first variable.
        stacked_indexers : iterable of indexers, optional
        length : int, optional
            Length of the new dimension. This is used to allocate the new data
            array for the stacked variable data before iterating over all
            items, which is thus more memory efficient and a bit faster.

        Returns
        -------
        stacked : Array
            Stacked variable formed by stacking all the supplied variables
            along the new dimension.
        """
        if not isinstance(dimension, basestring):
            length = dimension.size
            dimension, = dimension.dimensions

        if length is None or stacked_indexers is None:
            # so much for lazy evaluation! we need to look at all the variables
            # to figure out the indexers and/or dimensions of the stacked
            # variable
            variables = list(variables)
            steps = [var.shape[var.dimensions.index(dimension)]
                     if dimension in var.dimensions else 1
                     for var in variables]
            if length is None:
                length = sum(steps)
            if stacked_indexers is None:
                stacked_indexers = []
                i = 0
                for step in steps:
                    stacked_indexers.append(slice(i, i + step))
                    i += step
                if i != length:
                    raise ValueError('actual length of stacked variables '
                                     'along %s is %r but expected length was '
                                     '%s' % (dimension, i, length))

        # initialize the stacked variable with empty data
        first_var, variables = groupby.peek_at(variables)
        if dimension in first_var.dimensions:
            axis = first_var.dimensions.index(dimension)
            shape = tuple(length if n == axis else s
                          for n, s in enumerate(first_var.shape))
            dims = first_var.dimensions
        else:
            axis = 0
            shape = (length,) + first_var.shape
            dims = (dimension,) + first_var.dimensions
        attr = OrderedDict() if template is None else template.attributes

        stacked = cls(dims, np.empty(shape, dtype=first_var.dtype), attr)
        stacked.attributes.update(first_var.attributes)

        alt_dims = tuple(d for d in dims if d != dimension)

        # copy in the data from the variables
        for var, indexer in izip(variables, stacked_indexers):
            if template is None:
                # do sanity checks if we don't have a template
                if dimension in var.dimensions:
                    # transpose verifies that the dimensions are equivalent
                    if var.dimensions != stacked.dimensions:
                        var = var.transpose(*stacked.dimensions)
                elif var.dimensions != alt_dims:
                    raise ValueError('inconsistent dimensions')
                utils.remove_incompatible_items(stacked.attributes,
                                                var.attributes)

            key = tuple(indexer if n == axis else slice(None)
                        for n in range(stacked.ndim))
            stacked.data[tuple(key)] = var.data

        return stacked

    def __array_wrap__(self, result):
        return type(self)(self.dimensions, result, self.attributes)

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            return type(self)(self.dimensions, f(self.data, *args, **kwargs),
                              self.attributes)
        return func

    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, dataset_array.DatasetArray):
                return NotImplemented
            self_data, other_data, dims = _broadcast_variable_data(self, other)
            new_data = (f(self_data, other_data)
                        if not reflexive
                        else f(other_data, self_data))
            if hasattr(other, 'attributes'):
                new_attr = utils.ordered_dict_intersection(self.attributes,
                                                           other.attributes)
            else:
                new_attr = self.attributes
            return type(self)(dims, new_data, new_attr)
        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            self_data, other_data, dims = _broadcast_variable_data(self, other)
            if dims != self.dimensions:
                raise ValueError('dimensions cannot change for in-place '
                                 'operations')
            self.data = f(self_data, other_data)
            if hasattr(other, 'attributes'):
                utils.remove_incompatible_items(self.attributes, other)
            return self
        return func

ops.inject_special_operations(Array)


def broadcast_variables(first, second):
    """Given two arrays, return two arrays with matching dimensions and numpy
    broadcast compatible data

    Parameters
    ----------
    first, second : Array
        Array objects to broadcast.

    Returns
    -------
    first_broadcast, second_broadcast : Array
        Broadcast arrays. The data on each variable will be a view of the
        data on the corresponding original arrays, but dimensions will be
        reordered and inserted so that both broadcast arrays have the same
        dimensions. The new dimensions are sorted in order of appearence in the
        first variable's dimensions followed by the second variable's
        dimensions.
    """
    # TODO: add unit tests specifically for this function
    # validate dimensions
    dim_lengths = dict(zip(first.dimensions, first.shape))
    for k, v in zip(second.dimensions, second.shape):
        if k in dim_lengths and dim_lengths[k] != v:
            raise ValueError('operands could not be broadcast together '
                             'with mismatched lengths for dimension %r: %s'
                             % (k, (dim_lengths[k], v)))
    for dimensions in [first.dimensions, second.dimensions]:
        if len(set(dimensions)) < len(dimensions):
            raise ValueError('broadcasting requires that neither operand '
                             'has duplicate dimensions: %r'
                             % list(dimensions))

    # build dimensions for new Array
    second_only_dims = [d for d in second.dimensions
                        if d not in first.dimensions]
    dimensions = list(first.dimensions) + second_only_dims

    # expand first_data's dimensions so it's broadcast compatible after
    # adding second's dimensions at the end
    first_data = first.data[(Ellipsis,) + (None,) * len(second_only_dims)]
    new_first = Array(dimensions, first_data)
    # expand and reorder second_data so the dimensions line up
    first_only_dims = [d for d in dimensions if d not in second.dimensions]
    second_dims = list(second.dimensions) + first_only_dims
    second_data = second.data[(Ellipsis,) + (None,) * len(first_only_dims)]
    new_second = Array(second_dims, second_data).transpose(*dimensions)
    return new_first, new_second


def _broadcast_variable_data(self, other):
    if isinstance(other, dataset.Dataset):
        raise TypeError('datasets do not support mathematical operations')
    elif all(hasattr(other, attr) for attr in ['dimensions', 'data', 'shape']):
        # `other` satisfies the xray.Array API
        new_self, new_other = broadcast_variables(self, other)
        self_data = new_self.data
        other_data = new_other.data
        dimensions = new_self.dimensions
    else:
        # rely on numpy broadcasting rules
        self_data = self.data
        other_data = other
        dimensions = self.dimensions
    return self_data, other_data, dimensions
