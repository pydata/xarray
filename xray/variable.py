import functools
import numpy as np
import pandas as pd

from itertools import izip
from collections import OrderedDict

import ops
import utils
import dataset
import groupby
import data_array

from common import AbstractArray


def as_variable(obj, strict=True):
    """Convert an object into an Variable

    - If the object is already an `Variable`, return it.
    - If the object is a `DataArray`, return it if `strict=False` or return
      its variable if `strict=True`.
    - Otherwise, if the object has 'dimensions' and 'data' attributes, convert
      it into a new `Variable`.
    - If all else fails, attempt to convert the object into an `Variable` by
      unpacking it into the arguments for `Variable.__init__`.
    """
    # TODO: consider extending this method to automatically handle Iris and
    # pandas objects.
    if strict and hasattr(obj, 'variable'):
        # extract the primary Variable from DataArrays
        obj = obj.variable
    if not isinstance(obj, (Variable, data_array.DataArray)):
        if hasattr(obj, 'dimensions') and hasattr(obj, 'values'):
            obj = Variable(obj.dimensions, obj.values,
                         getattr(obj, 'attributes', None),
                         getattr(obj, 'encoding', None))
        else:
            if isinstance(obj, np.ndarray):
                raise TypeError('cannot convert numpy.ndarray objects into '
                                'Variable objects without supplying '
                                'dimensions')
            try:
                obj = Variable(*obj)
            except TypeError:
                raise TypeError('cannot convert argument into an Variable')
    return obj


def _as_compatible_data(data):
    """If data does not have the necessary attributes to be the private _data
    attribute, convert it to a np.ndarray and raise an warning
    """
    # don't check for __len__ or __iter__ so as not to warn if data is a numpy
    # numeric type like np.float32
    required = ['dtype', 'shape', 'size', 'ndim']
    if (not all(hasattr(data, attr) for attr in required)
            or isinstance(data, np.string_)):
        data = np.asarray(data)
    elif hasattr(data, 'values') and not isinstance(data, pd.Index):
        # we don't want nested self-described arrays
        data = data.values
    return data


class Variable(AbstractArray):
    """A netcdf-like variable consisting of dimensions, data and attributes
    which describe a single Array. A single Variable object is not fully
    described outside the context of its parent Dataset (if you want such a
    fully described object, use a DataArray instead).
    """
    def __init__(self, dims, data, attributes=None, encoding=None,
                 indexing_mode='numpy'):
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
        encoding : dict_like or None, optional
            Dictionary specifying how to encode this array's data into a
            serialized format like netCDF4. Currently used keys (for netCDF)
            include '_FillValue', 'scale_factor', 'add_offset' and 'dtype'.
            Well behaviored code to serialize an Variable should ignore
        indexing_mode : {'numpy', 'orthogonal'}
            String indicating how the data parameter handles fancy indexing
            (with arrays). Two modes are supported: 'numpy' (fancy indexing
            like numpy.ndarray objects) and 'orthogonal' (array indexing
            accesses different dimensions independently, like netCDF4
            variables). Accessing data from an Variable always uses orthogonal
            indexing, so `indexing_mode` tells the variable whether index
            lookups need to be internally converted to numpy-style indexing.
            unrecognized keys in this dictionary.
        """
        self._data = _as_compatible_data(data)
        self._dimensions = self._parse_dimensions(dims)
        if attributes is None:
            attributes = {}
        self._attributes = OrderedDict(attributes)
        self.encoding = dict({} if encoding is None else encoding)
        self._indexing_mode = indexing_mode

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def ndim(self):
        return self._data.ndim

    def __len__(self):
        return len(self._data)

    def in_memory(self):
        return isinstance(self._data, (np.ndarray, pd.Index))

    @property
    def values(self):
        """The variable's data as a numpy.ndarray"""
        self._data = np.asarray(self._data)
        self._indexing_mode = 'numpy'
        values = self._data
        if values.ndim == 0 and values.dtype.kind == 'O':
            # unpack 0d object arrays to be consistent with numpy
            values = values.item()
        return values

    @values.setter
    def values(self, values):
        values = np.asarray(values)
        if values.shape != self.shape:
            raise ValueError(
                "replacement values must match the Variable's shape")
        self._data = values
        self._indexing_mode = 'numpy'

    @property
    def data(self):
        utils.alias_warning('data', 'values', stacklevel=3)
        return self.values

    @data.setter
    def data(self, value):
        utils.alias_warning('data', 'values')
        self.values = value

    @property
    def index(self):
        """The variable's data as a pandas.Index"""
        if self.ndim != 1:
            raise ValueError('can only access 1-d arrays as an index')
        if isinstance(self._data, pd.Index):
            index = self._data
        else:
            index = utils.safe_cast_to_index(self.values)
        return index

    def to_coord(self):
        """Return this variable as an CoordVariable"""
        return CoordVariable(self.dimensions, self._data, self.attributes,
                             encoding=self.encoding,
                             indexing_mode=self._indexing_mode,
                             dtype=self.dtype)

    @property
    def dimensions(self):
        """Tuple of dimension names with which this variable is associated.
        """
        return self._dimensions

    def _parse_dimensions(self, dims):
        if isinstance(dims, basestring):
            dims = (dims,)
        dims = tuple(dims)
        if len(dims) != self.ndim:
            raise ValueError('dimensions %s must have the same length as the '
                             'number of data dimensions, ndim=%s'
                             % (dims, self.ndim))
        return dims

    @dimensions.setter
    def dimensions(self, value):
        self._dimensions = self._parse_dimensions(value)

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

    def _get_values(self, key):
        """Internal method for getting data from _data, given a key already
        converted to a suitable type (via _convert_indexer)"""
        if len(key) == 1:
            # unpack key so it can index a pandas.Index object (pandas.Index
            # objects don't like tuples)
            key, = key
        return self._data[key]

    def __getitem__(self, key):
        """Return a new Array object whose contents are consistent with
        getting the provided key from the underlying data.

        NB. __getitem__ and __setitem__ implement "orthogonal indexing" like
        netCDF4-python, where the key can only include integers, slices
        (including `Ellipsis`) and 1d arrays, each of which are applied
        orthogonally along their respective dimensions.

        The difference not matter in most cases unless you are using numpy's
        "fancy indexing," which can otherwise result in data arrays
        with shapes is inconsistent (or just uninterpretable with) with the
        variable's dimensions.

        If you really want to do indexing like `x[x > 0]`, manipulate the numpy
        array `x.values` directly.
        """
        key = self._convert_indexer(key)
        dimensions = [dim for k, dim in zip(key, self.dimensions)
                      if not isinstance(k, int)]
        values = self._get_values(key)
        # orthogonal indexing should ensure the dimensionality is consistent
        if hasattr(values, 'ndim'):
            assert values.ndim == len(dimensions)
        else:
            assert len(dimensions) == 0
        # don't keep indexing_mode, because values should now be an ndarray
        return type(self)(dimensions, values, self.attributes, self.encoding)

    def __setitem__(self, key, value):
        """__setitem__ is overloaded to access the underlying numpy values with
        orthogonal indexing.

        See __getitem__ for more details.
        """
        self.values[self._convert_indexer(key, indexing_mode='numpy')] = value

    def __iter__(self):
        for n in range(len(self)):
            yield self[n]

    @property
    def attributes(self):
        """Dictionary of local attributes on this variable.
        """
        return self._attributes

    def copy(self, deep=True):
        """Returns a copy of this object.

        If `deep=True`, the data array is loaded into memory and copied onto
        the new object. Dimensions, attributes and encodings are always copied.
        """
        data = self.values.copy() if deep else self._data
        # note:
        # dimensions is already an immutable tuple
        # attributes and encoding will be copied when the new Array is created
        return type(self)(self.dimensions, data, self.attributes,
                          self.encoding, self._indexing_mode)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        # memo does nothing but is required for compatability with
        # copy.deepcopy
        return self.copy(deep=True)

    # mutable objects should not be hashable
    __hash__ = None

    def indexed_by(self, **indexers):
        """Return a new array indexed along the specified dimension(s).

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

        key = [slice(None)] * self.ndim
        for i, dim in enumerate(self.dimensions):
            if dim in indexers:
                key[i] = indexers[dim]
        return self[tuple(key)]

    def transpose(self, *dimensions):
        """Return a new Variable object with transposed dimensions.

        Parameters
        ----------
        *dimensions : str, optional
            By default, reverse the dimensions. Otherwise, reorder the
            dimensions to this order.

        Returns
        -------
        transposed : Variable
            The returned object has transposed data and dimensions with the
            same attributes as the original.

        Notes
        -----
        Although this operation returns a view of this variable's data, it is
        not lazy -- the data will be fully loaded.

        See Also
        --------
        numpy.transpose
        """
        if len(dimensions) == 0:
            dimensions = self.dimensions[::-1]
        axes = [self.dimensions.index(dim) for dim in dimensions]
        data = self.values.transpose(*axes)
        return type(self)(dimensions, data, self.attributes, self.encoding)

    def squeeze(self, dimension=None):
        """Return a new Variable object with squeezed data.

        Parameters
        ----------
        dimensions : None or str or tuple of str, optional
            Selects a subset of the length one dimensions. If a dimension is
            selected with length greater than one, an error is raised. If
            None, all length one dimensions are squeezed.

        Returns
        -------
        squeezed : Variable
            This array, but with with all or a subset of the dimensions of
            length 1 removed.

        Notes
        -----
        Although this operation returns a view of this variable's data, it is
        not lazy -- the data will be fully loaded.

        See Also
        --------
        numpy.squeeze
        """
        dimensions = dict(zip(self.dimensions, self.shape))
        if dimension is None:
            dimension = [d for d, s in dimensions.iteritems() if s == 1]
        else:
            if isinstance(dimension, basestring):
                dimension = [dimension]
            if any(dimensions[k] > 1 for k in dimension):
                raise ValueError('cannot select a dimension to squeeze out '
                                 'which has length greater than one')
        return self.indexed_by(**{dim: 0 for dim in dimension})

    def reduce(self, func, dimension=None, axis=None, **kwargs):
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dimension : str or sequence of str, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            the reduction is calculated over the flattened array (by calling
            `func(x)` without an axis argument).
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dimension is not None and axis is not None:
            raise ValueError("cannot supply both 'axis' and 'dimension' "
                             "arguments")

        # reduce the data
        if dimension is not None:
            axis = self.get_axis_num(dimension)
        data = func(self.values, axis=axis, **kwargs)

        # construct the new Variable
        removed_axes = (range(self.ndim) if axis is None
                        else np.atleast_1d(axis) % self.ndim)
        dims = [dim for axis, dim in enumerate(self.dimensions)
                if axis not in removed_axes]
        var = Variable(dims, data, _math_safe_attributes(self.attributes))

        # update 'cell_methods' according to CF conventions
        removed_dims = [dim for axis, dim in enumerate(self.dimensions)
                        if axis in removed_axes]
        summary = '%s: %s' % (': '.join(removed_dims), func.__name__)
        if 'cell_methods' in var.attributes:
            var.attributes['cell_methods'] += ' ' + summary
        else:
            var.attributes['cell_methods'] = summary

        return var

    @classmethod
    def concat(cls, variables, dimension='stacked_dimension',
               indexers=None, length=None, shortcut=False):
        """Concatenate variables along a new or existing dimension.

        Parameters
        ----------
        variables : iterable of Array
            Arrays to stack together. Each variable is expected to have
            matching dimensions and shape except for along the stacked
            dimension.
        dimension : str or DataArray, optional
            Name of the dimension to stack along. This can either be a new
            dimension name, in which case it is added along axis=0, or an
            existing dimension name, in which case the location of the
            dimension is unchanged. Where to insert the new dimension is
            determined by the first variable.
        indexers : iterable of indexers, optional
            Iterable of indexers of the same length as variables which
            specifies how to assign variables along the given dimension. If
            not supplied, indexers is inferred from the length of each
            variable along the dimension, and the variables are stacked in the
            given order.
        length : int, optional
            Length of the new dimension. This is used to allocate the new data
            array for the stacked variable data before iterating over all
            items, which is thus more memory efficient and a bit faster. If
            dimension is provided as a DataArray, length is calculated
            automatically.
        shortcut : bool, optional
            This option is used internally to speed-up groupby operations.
            If `shortcut` is True, some checks of internal consistency between
            arrays to concatenate are skipped.

        Returns
        -------
        stacked : Variable
            Concatenated Variable formed by stacking all the supplied variables
            along the given dimension.
        """
        if not isinstance(dimension, basestring):
            length = dimension.size
            dimension, = dimension.dimensions

        if length is None or indexers is None:
            # so much for lazy evaluation! we need to look at all the variables
            # to figure out the indexers and/or dimensions of the stacked
            # variable
            variables = list(variables)
            steps = [var.shape[var.dimensions.index(dimension)]
                     if dimension in var.dimensions else 1
                     for var in variables]
            if length is None:
                length = sum(steps)
            if indexers is None:
                indexers = []
                i = 0
                for step in steps:
                    indexers.append(slice(i, i + step))
                    i += step
                if i != length:
                    raise ValueError('actual length of stacked variables '
                                     'along %s is %r but expected length was '
                                     '%s' % (dimension, i, length))

        # initialize the stacked variable with empty data
        first_var, variables = groupby.peek_at(variables)
        if dimension in first_var.dimensions:
            axis = first_var.get_axis_num(dimension)
            shape = tuple(length if n == axis else s
                          for n, s in enumerate(first_var.shape))
            dims = first_var.dimensions
        else:
            axis = 0
            shape = (length,) + first_var.shape
            dims = (dimension,) + first_var.dimensions

        concatenated = cls(dims, np.empty(shape, dtype=first_var.dtype))
        concatenated.attributes.update(first_var.attributes)

        alt_dims = tuple(d for d in dims if d != dimension)

        # copy in the data from the variables
        for var, indexer in izip(variables, indexers):
            if not shortcut:
                # do sanity checks & attributes clean-up
                if dimension in var.dimensions:
                    # transpose verifies that the dimensions are equivalent
                    if var.dimensions != concatenated.dimensions:
                        var = var.transpose(*concatenated.dimensions)
                elif var.dimensions != alt_dims:
                    raise ValueError('inconsistent dimensions')
                utils.remove_incompatible_items(concatenated.attributes,
                                                var.attributes)

            key = tuple(indexer if n == axis else slice(None)
                        for n in range(concatenated.ndim))
            concatenated.values[key] = var.values

        return concatenated

    def __array_wrap__(self, obj, context=None):
        return Variable(self.dimensions, obj, self.attributes)

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            return Variable(self.dimensions, f(self.values, *args, **kwargs),
                          _math_safe_attributes(self.attributes))
        return func

    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, data_array.DataArray):
                return NotImplemented
            self_data, other_data, dims = _broadcast_variable_data(self, other)
            new_data = (f(self_data, other_data)
                        if not reflexive
                        else f(other_data, self_data))
            new_attr = _math_safe_attributes(self.attributes)
            # TODO: reconsider handling of conflicting attributes
            if hasattr(other, 'attributes'):
                new_attr = utils.ordered_dict_intersection(
                    new_attr, _math_safe_attributes(other.attributes))
            return Variable(dims, new_data, new_attr)
        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            self_data, other_data, dims = _broadcast_variable_data(self, other)
            if dims != self.dimensions:
                raise ValueError('dimensions cannot change for in-place '
                                 'operations')
            self.values = f(self_data, other_data)
            if hasattr(other, 'attributes'):
                utils.remove_incompatible_items(
                    self.attributes, _math_safe_attributes(other.attributes))
            return self
        return func

ops.inject_special_operations(Variable)


class CoordVariable(Variable):
    """Subclass of Variable which caches its data as a pandas.Index instead of
    a numpy.ndarray

    CoordVariables must always be 1-dimensional.
    """
    def __init__(self, dims, data, attributes=None, encoding=None,
                 indexing_mode='numpy', dtype=None):
        """
        Parameters
        ----------
        dtype : np.dtype, optional
            Numpy dtype for the values in data. It is useful to keep track of
            this separately because data converted into a pandas.Index does not
            necessarily faithfully maintain the data type (many types are
            converted into object arrays).
        """
        super(CoordVariable, self).__init__(dims, data, attributes, encoding,
                                          indexing_mode)
        if self.ndim != 1:
            raise ValueError('%s objects must be 1-dimensional' %
                             type(self).__name__)
        if dtype is None:
            dtype = self._data.dtype
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def values(self):
        """The variable's values as a numpy.ndarray"""
        values = np.asarray(self._data).astype(self.dtype)
        if not isinstance(self._data, pd.Index):
            # always cache values as a pandas index
            self._data = utils.safe_cast_to_index(values)
            self._indexing_mode = 'numpy'
        return values

    @values.setter
    def values(self, value):
        raise TypeError('%s values cannot be modified' % type(self).__name__)

    def __getitem__(self, key):
        values = self._get_values(self._convert_indexer(key))
        if not hasattr(values, 'ndim') or values.ndim == 0:
            values = np.asarray(values).astype(self.dtype)
            return Variable((), values, self.attributes, self.encoding)
        else:
            return type(self)(self.dimensions, values, self.attributes,
                              self.encoding, dtype=self.dtype)

    def __setitem__(self, key, value):
        raise TypeError('%s values cannot be modified' % type(self).__name__)

    def copy(self, deep=True):
        """Returns a copy of this object.

        If `deep=True`, the values array is loaded into memory and copied onto
        the new object. Dimensions, attributes and encodings are always copied.
        """
        # there is no need to copy the index values here even if deep=True
        # since pandas.Index objects are immutable
        data = self.index if deep else self._data
        return type(self)(self.dimensions, data, self.attributes,
                          self.encoding, self._indexing_mode, self.dtype)

    def to_coord(self):
        """Return this variable as an CoordVariable"""
        return self


def _math_safe_attributes(attributes):
    return OrderedDict((k, v) for k, v in attributes.iteritems()
                       if k not in ['units'])


def broadcast_variables(first, second):
    """Given two Variables, return two Variables with matching dimensions and
    numpy broadcast compatible data.

    Parameters
    ----------
    first, second : Variable
        Variable objects to broadcast.

    Returns
    -------
    first_broadcast, second_broadcast : Variable
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
    first_data = first.values[(Ellipsis,) + (None,) * len(second_only_dims)]
    new_first = Variable(dimensions, first_data, first.attributes,
                       first.encoding)
    # expand and reorder second_data so the dimensions line up
    first_only_dims = [d for d in dimensions if d not in second.dimensions]
    second_dims = list(second.dimensions) + first_only_dims
    second_data = second.values[(Ellipsis,) + (None,) * len(first_only_dims)]
    new_second = Variable(second_dims, second_data, first.attributes,
                        second.encoding).transpose(*dimensions)
    return new_first, new_second


def _broadcast_variable_data(self, other):
    if isinstance(other, dataset.Dataset):
        raise TypeError('datasets do not support mathematical operations')
    elif all(hasattr(other, attr) for attr
             in ['dimensions', 'values', 'shape', 'encoding']):
        # `other` satisfies the necessary Variable API for broadcast_variables
        new_self, new_other = broadcast_variables(self, other)
        self_data = new_self.values
        other_data = new_other.values
        dimensions = new_self.dimensions
    else:
        # rely on numpy broadcasting rules
        self_data = self.values
        other_data = other
        dimensions = self.dimensions
    return self_data, other_data, dimensions
