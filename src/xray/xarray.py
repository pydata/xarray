import functools
import numpy as np
import pandas as pd

from itertools import izip
from collections import OrderedDict

import ops
import utils
import dataset
import groupby
import conventions
import dataset_array

from common import AbstractArray


def as_xarray(obj, strict=True):
    """Convert an object into an XArray

    - If the object is already an `XArray`, return it.
    - If the object is a `DatasetArray`, return it if `strict=False` or return
      its variable if `strict=True`.
    - Otherwise, if the object has 'dimensions' and 'data' attributes, convert
      it into a new `XArray`.
    - If all else fails, attempt to convert the object into an `XArray` by
      unpacking it into the arguments for `XArray.__init__`.
    """
    # TODO: consider extending this method to automatically handle Iris and
    # pandas objects.
    if strict and hasattr(obj, 'variable'):
        # extract the focus XArray from DatasetArrays
        obj = obj.variable
    if not isinstance(obj, (XArray, dataset_array.DatasetArray)):
        if hasattr(obj, 'dimensions') and hasattr(obj, 'data'):
            obj = XArray(obj.dimensions, obj.data,
                         getattr(obj, 'attributes', None),
                         getattr(obj, 'encoding', None))
        else:
            try:
                obj = XArray(*obj)
            except TypeError:
                raise TypeError('cannot convert argument into an XArray')
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
    elif isinstance(data, AbstractArray):
        # we don't want nested Array objects
        data = data.data
    return data


class XArray(AbstractArray):
    """A netcdf-like variable consisting of dimensions, data and attributes
    which describe a single Array. A single XArray object is not fully described
    outside the context of its parent Dataset (if you want such a fully
    described object, use a DatasetArray instead).
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
            Well behaviored code to serialize an XArray should ignore
        indexing_mode : {'numpy', 'orthogonal'}
            String indicating how the data parameter handles fancy indexing
            (with arrays). Two modes are supported: 'numpy' (fancy indexing
            like numpy.ndarray objects) and 'orthogonal' (array indexing
            accesses different dimensions independently, like netCDF4
            variables). Accessing data from an XArray always uses orthogonal
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

    def _data_as_ndarray(self):
        if isinstance(self._data, pd.Index):
            # pandas does automatic type conversion when an index is accessed
            # like index[...], so use index.values instead
            data = self._data.values
        else:
            data = np.asarray(self._data[...])
        return data

    @property
    def data(self):
        """The variable's data as a numpy.ndarray"""
        self._data = self._data_as_ndarray()
        self._indexing_mode = 'numpy'
        data = self._data
        if data.ndim == 0 and data.dtype.kind == 'O':
            # unpack 0d object arrays to be consistent with numpy
            data = data.item()
        return data

    @data.setter
    def data(self, value):
        value = np.asarray(value)
        if value.shape != self.shape:
            raise ValueError("replacement data must match the XArray's shape")
        self._data = value
        self._indexing_mode = 'numpy'

    @property
    def index(self):
        """The variable's data as a pandas.Index"""
        if self.ndim != 1:
            raise ValueError('can only access 1-d arrays as an index')
        if isinstance(self._data, pd.Index):
            index = self._data
        else:
            index = utils.safe_cast_to_index(self.data)
        return index

    def to_coord(self):
        """Return this array as an CoordXArray"""
        return CoordXArray(self.dimensions, self._data, self.attributes,
                           encoding=self.encoding,
                           indexing_mode=self._indexing_mode, dtype=self.dtype)

    @property
    def dimensions(self):
        """Tuple of dimension names with which this array is associated.
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

    def _get_data(self, key):
        """Internal method for getting data from _data, given a key already
        converted to a suitable type (via _convert_indexer)"""
        if len(key) == 1:
            # unpack key so it can index a pandas.Index object (pandas.Index
            # objects don't like tuples)
            key, = key
        # do integer based indexing if supported by _data (i.e., if _data is
        # a pandas object)
        return getattr(self._data, 'iloc', self._data)[key]

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
        array `x.data` directly.
        """
        key = self._convert_indexer(key)
        dimensions = [dim for k, dim in zip(key, self.dimensions)
                      if not isinstance(k, int)]
        data = self._get_data(key)
        # orthogonal indexing should ensure the dimensionality is consistent
        if hasattr(data, 'ndim'):
            assert data.ndim == len(dimensions)
        else:
            assert len(dimensions) == 0
        # don't keep indexing_mode, because data should now be an ndarray
        return type(self)(dimensions, data, self.attributes, self.encoding)

    def __setitem__(self, key, value):
        """__setitem__ is overloaded to access the underlying numpy data with
        orthogonal indexing.

        See __getitem__ for more details.
        """
        self.data[self._convert_indexer(key, indexing_mode='numpy')] = value

    def __iter__(self):
        for n in range(len(self)):
            yield self[n]

    @property
    def attributes(self):
        """Dictionary of local attributes on this array.
        """
        return self._attributes

    def copy(self, deep=True):
        """Returns a copy of this object.

        If `deep=True`, the data array is loaded into memory and copied onto
        the new object. Dimensions, attributes and encodings are always copied.
        """
        data = self.data.copy() if deep else self._data
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
        """Return a new XArray object with transposed dimensions.

        Parameters
        ----------
        *dimensions : str, optional
            By default, reverse the dimensions. Otherwise, reorder the
            dimensions to this order.

        Returns
        -------
        transposed : XArray
            The returned object has transposed data and dimensions with the
            same attributes as the original.

        Notes
        -----
        Although this operation returns a view of this array's data, it is
        not lazy -- the data will be fully loaded.

        See Also
        --------
        numpy.transpose
        """
        if len(dimensions) == 0:
            dimensions = self.dimensions[::-1]
        axes = [self.dimensions.index(dim) for dim in dimensions]
        data = self.data.transpose(*axes)
        return type(self)(dimensions, data, self.attributes, self.encoding)

    def squeeze(self, dimension=None):
        """Return a new XArray object with squeezed data.

        Parameters
        ----------
        dimensions : None or str or tuple of str, optional
            Selects a subset of the length one dimensions. If a dimension is
            selected with length greater than one, an error is raised. If
            None, all length one dimensions are squeezed.

        Returns
        -------
        squeezed : XArray
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
            Dimension(s) over which to repeatedly apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `func`. Only one of the
            'dimension' and 'axis' arguments can be supplied. If neither are
            supplied, then the reduction is calculated over the flattened array
            (by calling `func(x)` without an axis argument).
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Notes
        -----
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
            var = type(self)([], func(self.data, **kwargs),
                             _math_safe_attributes(self.attributes))
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
        new_var = type(self)(dims, data,
                             _math_safe_attributes(self.attributes))
        new_var._append_to_cell_methods(self.dimensions[axis]
                                        + ': ' + f.__name__)
        return new_var

    def groupby(self, group_name, group_array, squeeze=True):
        """Group this dataset by unique values of the indicated group.

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
    def concat(cls, variables, dimension='stacked_dimension',
               indexers=None, length=None, template=None):
        """Concatenate variables along a new or existing dimension.

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
            dimension is provided as an array, length is calculated
            automatically.
        template : XArray, optional
            This option is used internally to speed-up groupby operations. The
            template's attributes are added to the returned array's attributes.
            Furthermore, if a template is given, some checks of internal
            consistency between arrays to stack are skipped.

        Returns
        -------
        stacked : XArray
            Concatenated XArray formed by stacking all the supplied variables
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
            axis = first_var.dimensions.index(dimension)
            shape = tuple(length if n == axis else s
                          for n, s in enumerate(first_var.shape))
            dims = first_var.dimensions
        else:
            axis = 0
            shape = (length,) + first_var.shape
            dims = (dimension,) + first_var.dimensions
        attr = OrderedDict() if template is None else template.attributes

        concatenated = cls(dims, np.empty(shape, dtype=first_var.dtype), attr)
        concatenated.attributes.update(first_var.attributes)

        alt_dims = tuple(d for d in dims if d != dimension)

        # copy in the data from the variables
        for var, indexer in izip(variables, indexers):
            if template is None:
                # do sanity checks if we don't have a template
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
            concatenated.data[tuple(key)] = var.data

        return concatenated

    def __array_wrap__(self, obj, context=None):
        return XArray(self.dimensions, obj, self.attributes)

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            return XArray(self.dimensions, f(self.data, *args, **kwargs),
                          _math_safe_attributes(self.attributes))
        return func

    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, dataset_array.DatasetArray):
                return NotImplemented
            self_data, other_data, dims = _broadcast_xarray_data(self, other)
            new_data = (f(self_data, other_data)
                        if not reflexive
                        else f(other_data, self_data))
            new_attr = _math_safe_attributes(self.attributes)
            # TODO: reconsider handling of conflicting attributes
            if hasattr(other, 'attributes'):
                new_attr = utils.ordered_dict_intersection(
                    new_attr, _math_safe_attributes(other.attributes))
            return XArray(dims, new_data, new_attr)
        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            self_data, other_data, dims = _broadcast_xarray_data(self, other)
            if dims != self.dimensions:
                raise ValueError('dimensions cannot change for in-place '
                                 'operations')
            self.data = f(self_data, other_data)
            if hasattr(other, 'attributes'):
                utils.remove_incompatible_items(
                    self.attributes, _math_safe_attributes(other.attributes))
            return self
        return func

ops.inject_special_operations(XArray)


class CoordXArray(XArray):
    """Subclass of XArray which caches its data as a pandas.Index instead of
    a numpy.ndarray

    CoordXArrays must always be 1-dimensional.
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
        super(CoordXArray, self).__init__(dims, data, attributes, encoding,
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
    def data(self):
        """The variable's data as a numpy.ndarray"""
        data = self._data_as_ndarray().astype(self.dtype)
        if not isinstance(self._data, pd.Index):
            # always cache data as a pandas index
            self._data = utils.safe_cast_to_index(data)
            self._indexing_mode = 'numpy'
        return data

    @data.setter
    def data(self, value):
        raise TypeError('%s data cannot be modified' % type(self).__name__)

    def __getitem__(self, key):
        data = self._get_data(self._convert_indexer(key))
        if not hasattr(data, 'ndim') or data.ndim == 0:
            data = np.asarray(data).astype(self.dtype)
            return XArray((), data, self.attributes, self.encoding)
        else:
            return type(self)(self.dimensions, data, self.attributes,
                              self.encoding, dtype=self.dtype)

    def __setitem__(self, key, value):
        raise TypeError('%s data cannot be modified' % type(self).__name__)

    def copy(self, deep=True):
        """Returns a copy of this object.

        If `deep=True`, the data array is loaded into memory and copied onto
        the new object. Dimensions, attributes and encodings are always copied.
        """
        # there is no need to copy the index data here even if deep=True since
        # pandas.Index objects are immutable
        data = self.index if deep else self._data
        return type(self)(self.dimensions, data, self.attributes,
                          self.encoding, self._indexing_mode, self.dtype)

    def to_coord(self):
        """Return this array as an CoordXArray"""
        return self


def _math_safe_attributes(attributes):
    return OrderedDict((k, v) for k, v in attributes.iteritems()
                       if k not in ['units'])


def broadcast_xarrays(first, second):
    """Given two XArrays, return two XArrays with matching dimensions and numpy
    broadcast compatible data.

    Parameters
    ----------
    first, second : XArray
        XArray objects to broadcast.

    Returns
    -------
    first_broadcast, second_broadcast : XArray
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
    new_first = XArray(dimensions, first_data, first.attributes,
                       first.encoding)
    # expand and reorder second_data so the dimensions line up
    first_only_dims = [d for d in dimensions if d not in second.dimensions]
    second_dims = list(second.dimensions) + first_only_dims
    second_data = second.data[(Ellipsis,) + (None,) * len(first_only_dims)]
    new_second = XArray(second_dims, second_data, first.attributes,
                        second.encoding).transpose(*dimensions)
    return new_first, new_second


def _broadcast_xarray_data(self, other):
    if isinstance(other, dataset.Dataset):
        raise TypeError('datasets do not support mathematical operations')
    elif all(hasattr(other, attr) for attr
             in ['dimensions', 'data', 'shape', 'encoding']):
        # `other` satisfies the necessary xray.Array API for broadcast_xarrays
        new_self, new_other = broadcast_xarrays(self, other)
        self_data = new_self.data
        other_data = new_other.data
        dimensions = new_self.dimensions
    else:
        # rely on numpy broadcasting rules
        self_data = self.data
        other_data = other
        dimensions = self.dimensions
    return self_data, other_data, dimensions
