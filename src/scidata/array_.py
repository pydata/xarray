import functools
import warnings
from collections import OrderedDict

import numpy as np

import conventions
import dataset
import dataset_array
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
        warnings.warn('converting data to np.ndarray because %s lacks some of '
                      'the necesssary attributes for direct use'
                      % type(data).__name__, RuntimeWarning, stacklevel=3)
        data = np.asarray(data)
    elif isinstance(data, AbstractArray):
        # we don't want nested Array objects
        data = np.asarray(data)
    return data


def unique_value_groups(ar):
    """Group an array by its unique values

    Parameters
    ----------
    ar : array_like
        Input array. This will be flattened if it is not already 1-D.

    Returns
    -------
    values : np.ndarray
        Sorted, unique values as returned by `np.unique`.
    indices : list of lists of int
        Each element provides the integer indices in `ar` with values given by
        the corresponding value in `unique_values`.
    """
    values, inverse = np.unique(ar, return_inverse=True)
    groups = [[] for _ in range(len(values))]
    for n, g in enumerate(inverse):
        groups[g].append(n)
    return values, groups


class Array(AbstractArray):
    """
    A netcdf-like variable consisting of dimensions, data and attributes
    which describe a single Array.  A single variable object is not
    fully described outside the context of its parent Dataset.
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
            dims = [dims]
        data = _as_compatible_data(data)
        if len(dims) != data.ndim:
            raise ValueError('data and dimensions must have the same '
                             'dimensionality')
        self._dimensions = tuple(dims)
        self._data = data
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
        value = np.asarray(value)
        if value.shape != self.shape:
            raise ValueError("replacement data must match the Array's "
                             "shape")
        self._data = value

    @property
    def dimensions(self):
        return self._dimensions

    def _remap_indexer(self, key):
        """Converts an orthogonal indexer into a fully expanded key (of the
        same length as dimensions) suitable for indexing `_data`

        See Also
        --------
        utils.expanded_indexer
        utils.orthogonal_indexer
        """
        key = utils.expanded_indexer(key, self.ndim)
        if (self._indexing_mode == 'numpy'
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
        key = self._remap_indexer(key)
        dimensions = [dim for k, dim in zip(key, self.dimensions)
                      if not isinstance(k, int)]
        new_data = self._data[key]
        # orthogonal indexing should ensure the dimensionality is consistent
        assert new_data.ndim == len(dimensions)
        # return a variable with the same indexing_mode, because data should
        # still be the same type as _data
        return type(self)(dimensions, new_data, self.attributes,
                          indexing_mode=self._indexing_mode)

    def __setitem__(self, key, value):
        """__setitem__ is overloaded to access the underlying numpy data with
        orthogonal indexing (see __getitem__ for more details)
        """
        self._data[self._remap_indexer(key)] = value

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
        return '<scidata.%s%s>' % (type(self).__name__, contents)

    def indexed_by(self, **indexers):
        """Return a new variable indexed along the specified dimension(s)

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

    def collapse(self, func, dimension=None, axis=None, **kwargs):
        """Collapse this variable by applying `func` along some dimension(s)

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing an
            np.ndarray over an integer valued axis.
        dimension : str or sequence of str, optional
            Dimension(s) over which to repeatedly apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `func`. Only one of the
            'dimension' and 'axis' arguments can be supplied. If neither are
            supplied, then the collapse is calculated over the flattened array
            (by calling `func(x)` without an axis argument).
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Note
        ----
        If `collapse` is called with multiple dimensions (or axes, which
        are converted into dimensions), then the collapse operation is
        performed repeatedly along each dimension in turn from left to right.

        Returns
        -------
        collapsed : Array
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
                var = var._collapse(func, dim, **kwargs)
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

    def _collapse(self, f, dim, **kwargs):
        """Collapse a single dimension"""
        axis = self.dimensions.index(dim)
        dims = tuple(dim for i, dim in enumerate(self.dimensions)
                     if axis not in [i, i - self.ndim])
        data = f(self.data, axis=axis, **kwargs)
        new_var = type(self)(dims, data, self.attributes)
        new_var._append_to_cell_methods(self.dimensions[axis]
                                        + ': ' + f.__name__)
        return new_var

    def aggregate(self, func, new_dim_name, group_by, **kwargs):
        """Aggregate this variable by applying `func` to grouped elements

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to reduce an np.ndarray over an
            integer valued axis.
        new_dim_name : str or sequence of str, optional
            Name of the new dimension to create.
        group_by : Array
            1D variable which contains the values by which to group.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        unique : Array
            1D variable of unique values in group, along the dimension given by
            `new_dim_name`.
        aggregated : Array
            Array with aggregated data and the original dimension from
            `group_by` replaced by `new_dim_name`.
        """
        if group_by.ndim != 1:
            # TODO: remove this limitation?
            raise ValueError('group variables must be 1 dimensional')
        dim = group_by.dimensions[0]
        axis = self.dimensions.index(dim)
        if group_by.size != self.shape[axis]:
            raise ValueError('the group variable\'s length does not '
                             'match the length of this variable along its '
                             'dimension')
        unique_values, group_indices = unique_value_groups(group_by.data)
        aggregated = (self.indexed_by(**{dim: indices}).collapse(
                          func, dim, axis=None, **kwargs)
                      for indices in group_indices)
        stacked = type(self).from_stack(aggregated, new_dim_name,
                                        length=unique_values.size)
        ordered_dims = [new_dim_name if d == dim else d for d in self.dimensions]
        unique = type(self)([new_dim_name], unique_values)
        return unique, stacked.transpose(*ordered_dims)

    @classmethod
    def from_stack(cls, variables, dimension='stacked_dimension',
                   length=None):
        """Stack variables along a new or existing dimension to form a new
        variable

        Parameters
        ----------
        variables : iterable of Array
            Arrays to stack together. Each variable is expected to have
            matching dimensions and shape except for along the stacked
            dimension.
        dimension : str, optional
            Name of the dimension to stack along. This can either be a new
            dimension name, in which case it is added along axis=0, or an
            existing dimension name, in which case the location of the
            dimension is unchanged. Where to insert the new dimension is
            determined by the first variable.
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
        if length is None:
            # so much for lazy evaluation! we need to look at all the variables
            # to figure out the dimensions of the stacked variable
            variables = list(variables)
            length = 0
            for var in variables:
                if dimension in var.dimensions:
                    axis = var.dimensions.index(dimension)
                    length += var.shape[axis]
                else:
                    length += 1

        # manually keep track of progress along
        i = 0
        for var in variables:
            if i == 0:
                # initialize the stacked variable with empty data
                if dimension not in var.dimensions:
                    shape = (length,) + var.shape
                    dims = (dimension,) + var.dimensions
                else:
                    shape = tuple(length if d == dimension else s
                                  for d, s in zip(var.dimensions, var.shape))
                    dims = var.dimensions
                stacked = cls(dims, np.empty(shape, dtype=var.dtype),
                              var.attributes)
                # required dimensions (including order) if we have any N - 1
                # dimensional variables
                alt_dims = tuple(d for d in dims if d != dimension)

            if dimension in var.dimensions:
                # transpose requires that the dimensions are equivalent
                var = var.transpose(*stacked.dimensions)
                axis = var.dimensions.index(dimension)
                step = var.shape[axis]
            elif var.dimensions == alt_dims:
                step = 1
            else:
                raise ValueError('inconsistent dimensions')

            if i + step > length:
                raise ValueError('actual length of stacked variables along %s '
                                 'is greater than expected length %s'
                                 % (dimension, length))

            indexer = tuple((slice(i, i + step) if step > 1 else i)
                            if d == dimension else slice(None)
                            for d in stacked.dimensions)
            # by-pass variable indexing for possible speedup
            stacked.data[indexer] = var.data
            utils.remove_incompatible_items(stacked.attributes, var.attributes)
            i += step

        if i != length:
            raise ValueError('actual length of stacked variables along %s is '
                             '%s but expected length was %s'
                             % (dimension, i, length))

        return stacked

    def apply(self, func, *args, **kwargs):
        """Apply `func` with *args and **kwargs to this variable's data and
        return the result as a new variable with the same dimensions
        """
        data = np.asarray(func(self.data, *args, **kwargs))
        return type(self)(self.dimensions, data, self.attributes)

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
        # `other` satisfies the scidata.Array API
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
