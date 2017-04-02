from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import timedelta
from collections import defaultdict
import functools
import itertools
from distutils.version import LooseVersion

import numpy as np
import pandas as pd

from . import common
from . import indexing
from . import ops
from . import utils
from . import nputils
from .pycompat import (basestring, OrderedDict, zip, integer_types,
                       dask_array_type)
from .indexing import (PandasIndexAdapter, orthogonally_indexable)

import xarray as xr  # only for Dataset and DataArray

try:
    import dask.array as da
except ImportError:
    pass


def as_variable(obj, name=None, copy=False):
    """Convert an object into an Variable.

    - If the object is already an `Variable`, return a shallow copy.
    - Otherwise, if the object has 'dims' and 'data' attributes, convert
      it into a new `Variable`.
    - If all else fails, attempt to convert the object into an `Variable` by
      unpacking it into the arguments for `Variable.__init__`.
    """
    # TODO: consider extending this method to automatically handle Iris and
    # pandas objects.
    if hasattr(obj, 'variable'):
        # extract the primary Variable from DataArrays
        obj = obj.variable

    if isinstance(obj, Variable):
        obj = obj.copy(deep=False)
    elif hasattr(obj, 'dims') and (hasattr(obj, 'data') or
                                   hasattr(obj, 'values')):
        obj = Variable(obj.dims, getattr(obj, 'data', obj.values),
                       getattr(obj, 'attrs', None),
                       getattr(obj, 'encoding', None))
    elif isinstance(obj, tuple):
        try:
            obj = Variable(*obj)
        except TypeError:
            # use .format() instead of % because it handles tuples consistently
            raise TypeError('tuples to convert into variables must be of the '
                            'form (dims, data[, attrs, encoding]): '
                            '{}'.format(obj))
    elif utils.is_scalar(obj):
        obj = Variable([], obj)
    elif getattr(obj, 'name', None) is not None:
        obj = Variable(obj.name, obj)
    elif name is not None:
        data = as_compatible_data(obj)
        if data.ndim != 1:
            raise ValueError(
                'cannot set variable %r with %r-dimensional data '
                'without explicit dimension names. Pass a tuple of '
                '(dims, data) instead.' % (name, data.ndim))
        obj = Variable(name, obj, fastpath=True)
    else:
        raise TypeError('unable to convert object into a variable without an '
                        'explicit list of dimensions: %r' % obj)

    if name is not None and name in obj.dims:
        # convert the into an Index
        if obj.ndim != 1:
            raise ValueError(
                '%r has more than 1-dimension and the same name as one of its '
                'dimensions %r. xarray disallows such variables because they '
                'conflict with the coordinates used to label dimensions.'
                % (name, obj.dims))
        obj = obj.to_index_variable()

    return obj


def _maybe_wrap_data(data):
    """
    Put pandas.Index and numpy.ndarray arguments in adapter objects to ensure
    they can be indexed properly.

    NumpyArrayAdapter, PandasIndexAdapter and LazilyIndexedArray should
    all pass through unmodified.
    """
    if isinstance(data, pd.Index):
        return PandasIndexAdapter(data)
    return data


def as_compatible_data(data, fastpath=False):
    """Prepare and wrap data to put in a Variable.

    - If data does not have the necessary attributes, convert it to ndarray.
    - If data has dtype=datetime64, ensure that it has ns precision. If it's a
      pandas.Timestamp, convert it to datetime64.
    - If data is already a pandas or xarray object (other than an Index), just
      use the values.

    Finally, wrap it up with an adapter if necessary.
    """
    if fastpath and getattr(data, 'ndim', 0) > 0:
        # can't use fastpath (yet) for scalars
        return _maybe_wrap_data(data)

    if isinstance(data, Variable):
        return data.data

    # add a custom fast-path for dask.array to avoid expensive checks for the
    # dtype attribute
    if isinstance(data, dask_array_type):
        return data

    if isinstance(data, pd.Index):
        return _maybe_wrap_data(data)

    if isinstance(data, tuple):
        data = utils.to_0d_object_array(data)

    if isinstance(data, pd.Timestamp):
        # TODO: convert, handle datetime objects, too
        data = np.datetime64(data.value, 'ns')

    if isinstance(data, timedelta):
        data = np.timedelta64(getattr(data, 'value', data), 'ns')

    if (not hasattr(data, 'dtype') or not isinstance(data.dtype, np.dtype) or
            not hasattr(data, 'shape') or
            isinstance(data, (np.string_, np.unicode_,
                              np.datetime64, np.timedelta64))):
        # data must be ndarray-like
        # don't allow non-numpy dtypes (e.g., categories)
        data = np.asarray(data)

    # we don't want nested self-described arrays
    data = getattr(data, 'values', data)

    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)
        if mask.any():
            dtype, fill_value = common._maybe_promote(data.dtype)
            data = np.asarray(data, dtype=dtype)
            data[mask] = fill_value
        else:
            data = np.asarray(data)

    if isinstance(data, np.ndarray):
        if data.dtype.kind == 'O':
            data = common._possibly_convert_objects(data)
        elif data.dtype.kind == 'M':
            data = np.asarray(data, 'datetime64[ns]')
        elif data.dtype.kind == 'm':
            data = np.asarray(data, 'timedelta64[ns]')

    return _maybe_wrap_data(data)


def _as_array_or_item(data):
    """Return the given values as a numpy array, or as an individual item if
    it's a 0d datetime64 or timedelta64 array.

    Importantly, this function does not copy data if it is already an ndarray -
    otherwise, it will not be possible to update Variable values in place.

    This function mostly exists because 0-dimensional ndarrays with
    dtype=datetime64 are broken :(
    https://github.com/numpy/numpy/issues/4337
    https://github.com/numpy/numpy/issues/7619

    TODO: remove this (replace with np.asarray) once these issues are fixed
    """
    data = np.asarray(data)
    if data.ndim == 0:
        if data.dtype.kind == 'M':
            data = np.datetime64(data, 'ns')
        elif data.dtype.kind == 'm':
            data = np.timedelta64(data, 'ns')
    return data


class Variable(common.AbstractArray, utils.NdimSizeLenMixin):

    """A netcdf-like variable consisting of dimensions, data and attributes
    which describe a single Array. A single Variable object is not fully
    described outside the context of its parent Dataset (if you want such a
    fully described object, use a DataArray instead).

    The main functional difference between Variables and numpy arrays is that
    numerical operations on Variables implement array broadcasting by dimension
    name. For example, adding an Variable with dimensions `('time',)` to
    another Variable with dimensions `('space',)` results in a new Variable
    with dimensions `('time', 'space')`. Furthermore, numpy reduce operations
    like ``mean`` or ``sum`` are overwritten to take a "dimension" argument
    instead of an "axis".

    Variables are light-weight objects used as the building block for datasets.
    They are more primitive objects, so operations with them provide marginally
    higher performance than using DataArrays. However, manipulating data in the
    form of a Dataset or DataArray should almost always be preferred, because
    they can use more complete metadata in context of coordinate labels.
    """

    def __init__(self, dims, data, attrs=None, encoding=None, fastpath=False):
        """
        Parameters
        ----------
        dims : str or sequence of str
            Name(s) of the the data dimension(s). Must be either a string (only
            for 1D data) or a sequence of strings with length equal to the
            number of dimensions.
        data : array_like
            Data array which supports numpy-like data access.
        attrs : dict_like or None, optional
            Attributes to assign to the new variable. If None (default), an
            empty attribute dictionary is initialized.
        encoding : dict_like or None, optional
            Dictionary specifying how to encode this array's data into a
            serialized format like netCDF4. Currently used keys (for netCDF)
            include '_FillValue', 'scale_factor', 'add_offset' and 'dtype'.
            Well-behaved code to serialize a Variable should ignore
            unrecognized encoding items.
        """
        self._data = as_compatible_data(data, fastpath=fastpath)
        self._dims = self._parse_dimensions(dims)
        self._attrs = None
        self._encoding = None
        if attrs is not None:
            self.attrs = attrs
        if encoding is not None:
            self.encoding = encoding

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def _in_memory(self):
        return (isinstance(self._data, (np.ndarray, PandasIndexAdapter)) or
                (isinstance(self._data, indexing.MemoryCachedArray) and
                 isinstance(self._data.array, np.ndarray)))

    @property
    def data(self):
        if isinstance(self._data, dask_array_type):
            return self._data
        else:
            return self.values

    @data.setter
    def data(self, data):
        data = as_compatible_data(data)
        if data.shape != self.shape:
            raise ValueError(
                "replacement data must match the Variable's shape")
        self._data = data

    @property
    def _indexable_data(self):
        return orthogonally_indexable(self._data)

    def load(self):
        """Manually trigger loading of this variable's data from disk or a
        remote source into memory and return this variable.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically.
        """
        if not isinstance(self._data, np.ndarray):
            self._data = np.asarray(self._data)
        return self

    def compute(self):
        """Manually trigger loading of this variable's data from disk or a
        remote source into memory and return a new variable. The original is
        left unaltered.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically.
        """
        new = self.copy(deep=False)
        return new.load()

    @property
    def values(self):
        """The variable's data as a numpy.ndarray"""
        return _as_array_or_item(self._data)

    @values.setter
    def values(self, values):
        self.data = values

    def to_base_variable(self):
        """Return this variable as a base xarray.Variable"""
        return Variable(self.dims, self._data, self._attrs,
                        encoding=self._encoding, fastpath=True)

    to_variable = utils.alias(to_base_variable, 'to_variable')

    def to_index_variable(self):
        """Return this variable as an xarray.IndexVariable"""
        return IndexVariable(self.dims, self._data, self._attrs,
                             encoding=self._encoding, fastpath=True)

    to_coord = utils.alias(to_index_variable, 'to_coord')

    def to_index(self):
        """Convert this variable to a pandas.Index"""
        return self.to_index_variable().to_index()

    @property
    def dims(self):
        """Tuple of dimension names with which this variable is associated.
        """
        return self._dims

    def _parse_dimensions(self, dims):
        if isinstance(dims, basestring):
            dims = (dims,)
        dims = tuple(dims)
        if len(dims) != self.ndim:
            raise ValueError('dimensions %s must have the same length as the '
                             'number of data dimensions, ndim=%s'
                             % (dims, self.ndim))
        return dims

    @dims.setter
    def dims(self, value):
        self._dims = self._parse_dimensions(value)

    def _item_key_to_tuple(self, key):
        if utils.is_dict_like(key):
            return tuple(key.get(dim, slice(None)) for dim in self.dims)
        else:
            return key

    def __getitem__(self, key):
        """Return a new Array object whose contents are consistent with
        getting the provided key from the underlying data.

        NB. __getitem__ and __setitem__ implement "orthogonal indexing" like
        netCDF4-python, where the key can only include integers, slices
        (including `Ellipsis`) and 1d arrays, each of which are applied
        orthogonally along their respective dimensions.

        The difference does not matter in most cases unless you are using
        numpy's "fancy indexing," which can otherwise result in data arrays
        whose shapes is inconsistent (or just uninterpretable with) with the
        variable's dimensions.

        If you really want to do indexing like `x[x > 0]`, manipulate the numpy
        array `x.values` directly.
        """
        key = self._item_key_to_tuple(key)
        key = indexing.expanded_indexer(key, self.ndim)
        dims = tuple(dim for k, dim in zip(key, self.dims)
                     if not isinstance(k, integer_types))
        values = self._indexable_data[key]
        # orthogonal indexing should ensure the dimensionality is consistent
        if hasattr(values, 'ndim'):
            assert values.ndim == len(dims), (values.ndim, len(dims))
        else:
            assert len(dims) == 0, len(dims)
        return type(self)(dims, values, self._attrs, self._encoding,
                          fastpath=True)

    def __setitem__(self, key, value):
        """__setitem__ is overloaded to access the underlying numpy values with
        orthogonal indexing.

        See __getitem__ for more details.
        """
        key = self._item_key_to_tuple(key)
        if isinstance(self._data, dask_array_type):
            raise TypeError("this variable's data is stored in a dask array, "
                            'which does not support item assignment. To '
                            'assign to this variable, you must first load it '
                            'into memory explicitly using the .load_data() '
                            'method or accessing its .values attribute.')
        data = orthogonally_indexable(self._data)
        data[key] = value

    @property
    def attrs(self):
        """Dictionary of local attributes on this variable.
        """
        if self._attrs is None:
            self._attrs = OrderedDict()
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        self._attrs = OrderedDict(value)

    @property
    def encoding(self):
        """Dictionary of encodings on this variable.
        """
        if self._encoding is None:
            self._encoding = {}
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        try:
            self._encoding = dict(value)
        except ValueError:
            raise ValueError('encoding must be castable to a dictionary')

    def copy(self, deep=True):
        """Returns a copy of this object.

        If `deep=True`, the data array is loaded into memory and copied onto
        the new object. Dimensions, attributes and encodings are always copied.
        """
        data = self._data

        if isinstance(data, indexing.MemoryCachedArray):
            # don't share caching between copies
            data = indexing.MemoryCachedArray(data.array)

        if deep:
            if isinstance(data, dask_array_type):
                data = data.copy()
            elif not isinstance(data, PandasIndexAdapter):
                # pandas.Index is immutable
                data = np.array(data)

        # note:
        # dims is already an immutable tuple
        # attributes and encoding will be copied when the new Array is created
        return type(self)(self.dims, data, self._attrs, self._encoding,
                          fastpath=True)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        # memo does nothing but is required for compatibility with
        # copy.deepcopy
        return self.copy(deep=True)

    # mutable objects should not be hashable
    __hash__ = None

    @property
    def chunks(self):
        """Block dimensions for this array's data or None if it's not a dask
        array.
        """
        return getattr(self._data, 'chunks', None)

    _array_counter = itertools.count()

    def chunk(self, chunks=None, name=None, lock=False):
        """Coerce this array's data into a dask arrays with the given chunks.

        If this variable is a non-dask array, it will be converted to dask
        array. If it's a dask array, it will be rechunked to the given chunk
        sizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

        Parameters
        ----------
        chunks : int, tuple or dict, optional
            Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
            ``{'x': 5, 'y': 5}``.
        name : str, optional
            Used to generate the name for this array in the internal dask
            graph. Does not need not be unique.
        lock : optional
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.

        Returns
        -------
        chunked : xarray.Variable
        """
        import dask.array as da

        if utils.is_dict_like(chunks):
            chunks = dict((self.get_axis_num(dim), chunk)
                          for dim, chunk in chunks.items())

        if chunks is None:
            chunks = self.chunks or self.shape

        data = self._data
        if isinstance(data, da.Array):
            data = data.rechunk(chunks)
        else:
            if utils.is_dict_like(chunks):
                chunks = tuple(chunks.get(n, s)
                               for n, s in enumerate(self.shape))
            data = da.from_array(data, chunks, name=name, lock=lock)

        return type(self)(self.dims, data, self._attrs, self._encoding,
                          fastpath=True)

    def isel(self, **indexers):
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
        invalid = [k for k in indexers if k not in self.dims]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        key = [slice(None)] * self.ndim
        for i, dim in enumerate(self.dims):
            if dim in indexers:
                key[i] = indexers[dim]
        return self[tuple(key)]

    def squeeze(self, dim=None):
        """Return a new object with squeezed data.

        Parameters
        ----------
        dim : None or str or tuple of str, optional
            Selects a subset of the length one dimensions. If a dimension is
            selected with length greater than one, an error is raised. If
            None, all length one dimensions are squeezed.

        Returns
        -------
        squeezed : same type as caller
            This object, but with with all or a subset of the dimensions of
            length 1 removed.

        See Also
        --------
        numpy.squeeze
        """
        dims = common.get_squeeze_dims(self, dim)
        return self.isel(**{d: 0 for d in dims})

    def _shift_one_dim(self, dim, count):
        axis = self.get_axis_num(dim)

        if count > 0:
            keep = slice(None, -count)
        elif count < 0:
            keep = slice(-count, None)
        else:
            keep = slice(None)

        trimmed_data = self[(slice(None),) * axis + (keep,)].data
        dtype, fill_value = common._maybe_promote(self.dtype)

        shape = list(self.shape)
        shape[axis] = min(abs(count), shape[axis])

        if isinstance(trimmed_data, dask_array_type):
            chunks = list(trimmed_data.chunks)
            chunks[axis] = (shape[axis],)
            full = functools.partial(da.full, chunks=chunks)
        else:
            full = np.full

        nans = full(shape, fill_value, dtype=dtype)

        if count > 0:
            arrays = [nans, trimmed_data]
        else:
            arrays = [trimmed_data, nans]

        data = ops.concatenate(arrays, axis)

        if isinstance(data, dask_array_type):
            # chunked data should come out with the same chunks; this makes
            # it feasible to combine shifted and unshifted data
            # TODO: remove this once dask.array automatically aligns chunks
            data = data.rechunk(self.data.chunks)

        return type(self)(self.dims, data, self._attrs, fastpath=True)

    def shift(self, **shifts):
        """
        Return a new Variable with shifted data.

        Parameters
        ----------
        **shifts : keyword arguments of the form {dim: offset}
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.

        Returns
        -------
        shifted : Variable
            Variable with the same dimensions and attributes but shifted data.
        """
        result = self
        for dim, count in shifts.items():
            result = result._shift_one_dim(dim, count)
        return result

    def _roll_one_dim(self, dim, count):
        axis = self.get_axis_num(dim)

        count %= self.shape[axis]
        if count != 0:
            indices = [slice(-count, None), slice(None, -count)]
        else:
            indices = [slice(None)]

        arrays = [self[(slice(None),) * axis + (idx,)].data
                  for idx in indices]

        data = ops.concatenate(arrays, axis)

        if isinstance(data, dask_array_type):
            # chunked data should come out with the same chunks; this makes
            # it feasible to combine shifted and unshifted data
            # TODO: remove this once dask.array automatically aligns chunks
            data = data.rechunk(self.data.chunks)

        return type(self)(self.dims, data, self._attrs, fastpath=True)

    def roll(self, **shifts):
        """
        Return a new Variable with rolld data.

        Parameters
        ----------
        **shifts : keyword arguments of the form {dim: offset}
            Integer offset to roll along each of the given dimensions.
            Positive offsets roll to the right; negative offsets roll to the
            left.

        Returns
        -------
        shifted : Variable
            Variable with the same dimensions and attributes but rolled data.
        """
        result = self
        for dim, count in shifts.items():
            result = result._roll_one_dim(dim, count)
        return result

    def transpose(self, *dims):
        """Return a new Variable object with transposed dimensions.

        Parameters
        ----------
        *dims : str, optional
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
        if len(dims) == 0:
            dims = self.dims[::-1]
        axes = self.get_axis_num(dims)
        if len(dims) < 2:  # no need to transpose if only one dimension
            return self.copy(deep=False)
        data = ops.transpose(self.data, axes)
        return type(self)(dims, data, self._attrs, self._encoding, fastpath=True)

    def expand_dims(self, dims, shape=None):
        """Return a new variable with expanded dimensions.

        When possible, this operation does not copy this variable's data.

        Parameters
        ----------
        dims : str or sequence of str or dict
            Dimensions to include on the new variable. If a dict, values are
            used to provide the sizes of new dimensions; otherwise, new
            dimensions are inserted with length 1.

        Returns
        -------
        Variable
        """
        if isinstance(dims, basestring):
            dims = [dims]

        if shape is None and utils.is_dict_like(dims):
            shape = dims.values()

        missing_dims = set(self.dims) - set(dims)
        if missing_dims:
            raise ValueError('new dimensions must be a superset of existing '
                             'dimensions')

        self_dims = set(self.dims)
        expanded_dims = tuple(
            d for d in dims if d not in self_dims) + self.dims

        if self.dims == expanded_dims:
            # don't use broadcast_to unless necessary so the result remains
            # writeable if possible
            expanded_data = self.data
        elif shape is not None:
            dims_map = dict(zip(dims, shape))
            tmp_shape = tuple(dims_map[d] for d in expanded_dims)
            expanded_data = ops.broadcast_to(self.data, tmp_shape)
        else:
            expanded_data = self.data[
                (None,) * (len(expanded_dims) - self.ndim)]

        expanded_var = Variable(expanded_dims, expanded_data, self._attrs,
                                self._encoding, fastpath=True)
        return expanded_var.transpose(*dims)

    def _stack_once(self, dims, new_dim):
        if not set(dims) <= set(self.dims):
            raise ValueError('invalid existing dimensions: %s' % dims)

        if new_dim in self.dims:
            raise ValueError('cannot create a new dimension with the same '
                             'name as an existing dimension')

        if len(dims) == 0:
            # don't stack
            return self.copy(deep=False)

        other_dims = [d for d in self.dims if d not in dims]
        dim_order = other_dims + list(dims)
        reordered = self.transpose(*dim_order)

        new_shape = reordered.shape[:len(other_dims)] + (-1,)
        new_data = reordered.data.reshape(new_shape)
        new_dims = reordered.dims[:len(other_dims)] + (new_dim,)

        return Variable(new_dims, new_data, self._attrs, self._encoding,
                        fastpath=True)

    def stack(self, **dimensions):
        """
        Stack any number of existing dimensions into a single new dimension.

        New dimensions will be added at the end, and the order of the data
        along each new dimension will be in contiguous (C) order.

        Parameters
        ----------
        **dimensions : keyword arguments of the form new_name=(dim1, dim2, ...)
            Names of new dimensions, and the existing dimensions that they
            replace.

        Returns
        -------
        stacked : Variable
            Variable with the same attributes but stacked data.

        See also
        --------
        Variable.unstack
        """
        result = self
        for new_dim, dims in dimensions.items():
            result = result._stack_once(dims, new_dim)
        return result

    def _unstack_once(self, dims, old_dim):
        new_dim_names = tuple(dims.keys())
        new_dim_sizes = tuple(dims.values())

        if old_dim not in self.dims:
            raise ValueError('invalid existing dimension: %s' % old_dim)

        if set(new_dim_names).intersection(self.dims):
            raise ValueError('cannot create a new dimension with the same '
                             'name as an existing dimension')

        if np.prod(new_dim_sizes) != self.sizes[old_dim]:
            raise ValueError('the product of the new dimension sizes must '
                             'equal the size of the old dimension')

        other_dims = [d for d in self.dims if d != old_dim]
        dim_order = other_dims + [old_dim]
        reordered = self.transpose(*dim_order)

        new_shape = reordered.shape[:len(other_dims)] + new_dim_sizes
        new_data = reordered.data.reshape(new_shape)
        new_dims = reordered.dims[:len(other_dims)] + new_dim_names

        return Variable(new_dims, new_data, self._attrs, self._encoding,
                        fastpath=True)

    def unstack(self, **dimensions):
        """
        Unstack an existing dimension into multiple new dimensions.

        New dimensions will be added at the end, and the order of the data
        along each new dimension will be in contiguous (C) order.

        Parameters
        ----------
        **dimensions : keyword arguments of the form old_dim={dim1: size1, ...}
            Names of existing dimensions, and the new dimensions and sizes that they
            map to.

        Returns
        -------
        unstacked : Variable
            Variable with the same attributes but unstacked data.

        See also
        --------
        Variable.stack
        """
        result = self
        for old_dim, dims in dimensions.items():
            result = result._unstack_once(dims, old_dim)
        return result

    def fillna(self, value):
        return ops.fillna(self, value)

    def where(self, cond):
        return self._where(cond)

    def reduce(self, func, dim=None, axis=None, keep_attrs=False,
               allow_lazy=False, **kwargs):
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dim'
            and 'axis' arguments can be supplied. If neither are supplied, then
            the reduction is calculated over the flattened array (by calling
            `func(x)` without an axis argument).
        keep_attrs : bool, optional
            If True, the variable's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is not None and axis is not None:
            raise ValueError("cannot supply both 'axis' and 'dim' arguments")

        if getattr(func, 'keep_dims', False):
            if dim is None and axis is None:
                raise ValueError("must supply either single 'dim' or 'axis' "
                                 "argument to %s" % (func.__name__))

        if dim is not None:
            axis = self.get_axis_num(dim)
        data = func(self.data if allow_lazy else self.values,
                    axis=axis, **kwargs)

        if getattr(data, 'shape', ()) == self.shape:
            dims = self.dims
        else:
            removed_axes = (range(self.ndim) if axis is None
                            else np.atleast_1d(axis) % self.ndim)
            dims = [adim for n, adim in enumerate(self.dims)
                    if n not in removed_axes]

        attrs = self._attrs if keep_attrs else None

        return Variable(dims, data, attrs=attrs)

    @classmethod
    def concat(cls, variables, dim='concat_dim', positions=None,
               shortcut=False):
        """Concatenate variables along a new or existing dimension.

        Parameters
        ----------
        variables : iterable of Array
            Arrays to stack together. Each variable is expected to have
            matching dimensions and shape except for along the stacked
            dimension.
        dim : str or DataArray, optional
            Name of the dimension to stack along. This can either be a new
            dimension name, in which case it is added along axis=0, or an
            existing dimension name, in which case the location of the
            dimension is unchanged. Where to insert the new dimension is
            determined by the first variable.
        positions : None or list of integer arrays, optional
            List of integer arrays which specifies the integer positions to which
            to assign each dataset along the concatenated dimension. If not
            supplied, objects are concatenated in the provided order.
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
        if not isinstance(dim, basestring):
            dim, = dim.dims

        # can't do this lazily: we need to loop through variables at least
        # twice
        variables = list(variables)
        first_var = variables[0]

        arrays = [v.data for v in variables]

        # TODO: use our own type promotion rules to ensure that
        # [str, float] -> object, not str like numpy
        if dim in first_var.dims:
            axis = first_var.get_axis_num(dim)
            dims = first_var.dims
            data = ops.concatenate(arrays, axis=axis)
            if positions is not None:
                # TODO: deprecate this option -- we don't need it for groupby
                # any more.
                indices = nputils.inverse_permutation(np.concatenate(positions))
                data = ops.take(data, indices, axis=axis)
        else:
            axis = 0
            dims = (dim,) + first_var.dims
            data = ops.stack(arrays, axis=axis)

        attrs = OrderedDict(first_var.attrs)
        if not shortcut:
            for var in variables:
                if var.dims != first_var.dims:
                    raise ValueError('inconsistent dimensions')
                utils.remove_incompatible_items(attrs, var.attrs)

        return cls(dims, data, attrs)

    def equals(self, other, equiv=ops.array_equiv):
        """True if two Variables have the same dimensions and values;
        otherwise False.

        Variables can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for Variables
        does element-wise comparisons (like numpy.ndarrays).
        """
        other = getattr(other, 'variable', other)
        try:
            return (self.dims == other.dims and
                    (self._data is other._data or
                     equiv(self.data, other.data)))
        except (TypeError, AttributeError):
            return False

    def broadcast_equals(self, other, equiv=ops.array_equiv):
        """True if two Variables have the values after being broadcast against
        each other; otherwise False.

        Variables can still be equal (like pandas objects) if they have NaN
        values in the same locations.
        """
        try:
            self, other = broadcast_variables(self, other)
        except (ValueError, AttributeError):
            return False
        return self.equals(other, equiv=equiv)

    def identical(self, other):
        """Like equals, but also checks attributes.
        """
        try:
            return (utils.dict_equiv(self.attrs, other.attrs) and
                    self.equals(other))
        except (TypeError, AttributeError):
            return False

    def no_conflicts(self, other):
        """True if the intersection of two Variable's non-null data is
        equal; otherwise false.

        Variables can thus still be equal if there are locations where either,
        or both, contain NaN values.
        """
        return self.broadcast_equals(other, equiv=ops.array_notnull_equiv)

    def quantile(self, q, dim=None, interpolation='linear'):
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements.

        Parameters
        ----------
        q : float in range of [0,1] (or sequence of floats)
            Quantile to compute, which must be between 0 and 1
            inclusive.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply quantile.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to
            use when the desired quantile lies between two data points
            ``i < j``:
                * linear: ``i + (j - i) * fraction``, where ``fraction`` is
                  the fractional part of the index surrounded by ``i`` and
                  ``j``.
                * lower: ``i``.
                * higher: ``j``.
                * nearest: ``i`` or ``j``, whichever is nearest.
                * midpoint: ``(i + j) / 2``.

        Returns
        -------
        quantiles : Variable
            If `q` is a single quantile, then the result
            is a scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile and a quantile dimension
            is added to the return array. The other dimensions are the
             dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanpercentile, pandas.Series.quantile, Dataset.quantile,
        DataArray.quantile
        """

        if isinstance(self.data, dask_array_type):
            TypeError("quantile does not work for arrays stored as dask "
                      "arrays. Load the data via .compute() or .load() prior "
                      "to calling this method.")
        if LooseVersion(np.__version__) < LooseVersion('1.10.0'):
            raise NotImplementedError(
                'quantile requres numpy version 1.10.0 or later')

        q = np.asarray(q, dtype=np.float64)

        new_dims = list(self.dims)
        if dim is not None:
            axis = self.get_axis_num(dim)
            if utils.is_scalar(dim):
                new_dims.remove(dim)
            else:
                for d in dim:
                    new_dims.remove(d)
        else:
            axis = None
            new_dims = []

        # only add the quantile dimension if q is array like
        if q.ndim != 0:
            new_dims = ['quantile'] + new_dims

        qs = np.nanpercentile(self.data, q * 100., axis=axis,
                              interpolation=interpolation)
        return Variable(new_dims, qs)

    @property
    def real(self):
        return type(self)(self.dims, self.data.real, self._attrs)

    @property
    def imag(self):
        return type(self)(self.dims, self.data.imag, self._attrs)

    def __array_wrap__(self, obj, context=None):
        return Variable(self.dims, obj)

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            return self.__array_wrap__(f(self.data, *args, **kwargs))
        return func

    @staticmethod
    def _binary_op(f, reflexive=False, **ignored_kwargs):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, (xr.DataArray, xr.Dataset)):
                return NotImplemented
            self_data, other_data, dims = _broadcast_compat_data(self, other)
            new_data = (f(self_data, other_data)
                        if not reflexive
                        else f(other_data, self_data))
            result = Variable(dims, new_data)
            return result
        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, xr.Dataset):
                raise TypeError('cannot add a Dataset to a Variable in-place')
            self_data, other_data, dims = _broadcast_compat_data(self, other)
            if dims != self.dims:
                raise ValueError('dimensions cannot change for in-place '
                                 'operations')
            self.values = f(self_data, other_data)
            return self
        return func

ops.inject_all_ops_and_reduce_methods(Variable)


class IndexVariable(Variable):
    """Wrapper for accommodating a pandas.Index in an xarray.Variable.

    IndexVariable preserve loaded values in the form of a pandas.Index instead
    of a NumPy array. Hence, their values are immutable and must always be one-
    dimensional.

    They also have a name property, which is the name of their sole dimension
    unless another name is given.
    """

    def __init__(self, dims, data, attrs=None, encoding=None, fastpath=False):
        super(IndexVariable, self).__init__(dims, data, attrs, encoding,
                                            fastpath)
        if self.ndim != 1:
            raise ValueError('%s objects must be 1-dimensional' %
                             type(self).__name__)

        # Unlike in Variable, always eagerly load values into memory
        if not isinstance(self._data, PandasIndexAdapter):
            self._data = PandasIndexAdapter(self._data)

    def load(self):
        # data is already loaded into memory for IndexVariable
        return self

    @Variable.data.setter
    def data(self, data):
        Variable.data.fset(self, data)
        if not isinstance(self._data, PandasIndexAdapter):
            self._data = PandasIndexAdapter(self._data)

    def chunk(self, chunks=None, name=None, lock=False):
        # Dummy - do not chunk. This method is invoked e.g. by Dataset.chunk()
        return self.copy(deep=False)

    def __getitem__(self, key):
        key = self._item_key_to_tuple(key)
        values = self._indexable_data[key]
        if not hasattr(values, 'ndim') or values.ndim == 0:
            return Variable((), values, self._attrs, self._encoding)
        else:
            return type(self)(self.dims, values, self._attrs,
                              self._encoding, fastpath=True)

    def __setitem__(self, key, value):
        raise TypeError('%s values cannot be modified' % type(self).__name__)

    @classmethod
    def concat(cls, variables, dim='concat_dim', positions=None,
               shortcut=False):
        """Specialized version of Variable.concat for IndexVariable objects.

        This exists because we want to avoid converting Index objects to NumPy
        arrays, if possible.
        """
        if not isinstance(dim, basestring):
            dim, = dim.dims

        variables = list(variables)
        first_var = variables[0]

        if any(not isinstance(v, cls) for v in variables):
            raise TypeError('IndexVariable.concat requires that all input '
                            'variables be IndexVariable objects')

        indexes = [v._data.array for v in variables]

        if not indexes:
            data = []
        else:
            data = indexes[0].append(indexes[1:])

            if positions is not None:
                indices = nputils.inverse_permutation(
                    np.concatenate(positions))
                data = data.take(indices)

        attrs = OrderedDict(first_var.attrs)
        if not shortcut:
            for var in variables:
                if var.dims != first_var.dims:
                    raise ValueError('inconsistent dimensions')
                utils.remove_incompatible_items(attrs, var.attrs)

        return cls(first_var.dims, data, attrs)

    def copy(self, deep=True):
        """Returns a copy of this object.

        `deep` is ignored since data is stored in the form of pandas.Index,
        which is already immutable. Dimensions, attributes and encodings are
        always copied.
        """
        return type(self)(self.dims, self._data, self._attrs,
                          self._encoding, fastpath=True)

    def equals(self, other, equiv=None):
        # if equiv is specified, super up
        if equiv is not None:
            return super(IndexVariable, self).equals(other, equiv)

        # otherwise use the native index equals, rather than looking at _data
        other = getattr(other, 'variable', other)
        try:
            return (self.dims == other.dims and
                    self._data_equals(other))
        except (TypeError, AttributeError):
            return False

    def _data_equals(self, other):
        return self.to_index().equals(other.to_index())

    def to_index_variable(self):
        """Return this variable as an xarray.IndexVariable"""
        return self

    to_coord = utils.alias(to_index_variable, 'to_coord')

    def to_index(self):
        """Convert this variable to a pandas.Index"""
        # n.b. creating a new pandas.Index from an old pandas.Index is
        # basically free as pandas.Index objects are immutable
        assert self.ndim == 1
        index = self._data.array
        if isinstance(index, pd.MultiIndex):
            # set default names for multi-index unnamed levels so that
            # we can safely rename dimension / coordinate later
            valid_level_names = [name or '{}_level_{}'.format(self.dims[0], i)
                                 for i, name in enumerate(index.names)]
            index = index.set_names(valid_level_names)
        else:
            index = index.set_names(self.name)
        return index

    @property
    def level_names(self):
        """Return MultiIndex level names or None if this IndexVariable has no
        MultiIndex.
        """
        index = self.to_index()
        if isinstance(index, pd.MultiIndex):
            return index.names
        else:
            return None

    def get_level_variable(self, level):
        """Return a new IndexVariable from a given MultiIndex level."""
        if self.level_names is None:
            raise ValueError("IndexVariable %r has no MultiIndex" % self.name)
        index = self.to_index()
        return type(self)(self.dims, index.get_level_values(level))

    @property
    def name(self):
        return self.dims[0]

    @name.setter
    def name(self, value):
        raise AttributeError('cannot modify name of IndexVariable in-place')

# for backwards compatibility
Coordinate = utils.alias(IndexVariable, 'Coordinate')


def _unified_dims(variables):
    # validate dimensions
    all_dims = OrderedDict()
    for var in variables:
        var_dims = var.dims
        if len(set(var_dims)) < len(var_dims):
            raise ValueError('broadcasting cannot handle duplicate '
                             'dimensions: %r' % list(var_dims))
        for d, s in zip(var_dims, var.shape):
            if d not in all_dims:
                all_dims[d] = s
            elif all_dims[d] != s:
                raise ValueError('operands cannot be broadcast together '
                                 'with mismatched lengths for dimension %r: %s'
                                 % (d, (all_dims[d], s)))
    return all_dims


def _broadcast_compat_variables(*variables):
    dims = tuple(_unified_dims(variables))
    return tuple(var.expand_dims(dims) if var.dims != dims else var
                 for var in variables)


def broadcast_variables(*variables):
    """Given any number of variables, return variables with matching dimensions
    and broadcast data.

    The data on the returned variables will be a view of the data on the
    corresponding original arrays, but dimensions will be reordered and
    inserted so that both broadcast arrays have the same dimensions. The new
    dimensions are sorted in order of appearance in the first variable's
    dimensions followed by the second variable's dimensions.
    """
    dims_map = _unified_dims(variables)
    dims_tuple = tuple(dims_map)
    return tuple(var.expand_dims(dims_map) if var.dims != dims_tuple else var
                 for var in variables)


def _broadcast_compat_data(self, other):
    if all(hasattr(other, attr) for attr
            in ['dims', 'data', 'shape', 'encoding']):
        # `other` satisfies the necessary Variable API for broadcast_variables
        new_self, new_other = _broadcast_compat_variables(self, other)
        self_data = new_self.data
        other_data = new_other.data
        dims = new_self.dims
    else:
        # rely on numpy broadcasting rules
        self_data = self.data
        other_data = other
        dims = self.dims
    return self_data, other_data, dims


def concat(variables, dim='concat_dim', positions=None, shortcut=False):
    """Concatenate variables along a new or existing dimension.

    Parameters
    ----------
    variables : iterable of Array
        Arrays to stack together. Each variable is expected to have
        matching dimensions and shape except for along the stacked
        dimension.
    dim : str or DataArray, optional
        Name of the dimension to stack along. This can either be a new
        dimension name, in which case it is added along axis=0, or an
        existing dimension name, in which case the location of the
        dimension is unchanged. Where to insert the new dimension is
        determined by the first variable.
    positions : None or list of integer arrays, optional
        List of integer arrays which specifies the integer positions to which
        to assign each dataset along the concatenated dimension. If not
        supplied, objects are concatenated in the provided order.
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
    variables = list(variables)
    if all(isinstance(v, IndexVariable) for v in variables):
        return IndexVariable.concat(variables, dim, positions, shortcut)
    else:
        return Variable.concat(variables, dim, positions, shortcut)


def assert_unique_multiindex_level_names(variables):
    """Check for uniqueness of MultiIndex level names in all given
    variables.

    Not public API. Used for checking consistency of DataArray and Dataset
    objects.
    """
    level_names = defaultdict(list)
    for var_name, var in variables.items():
        if isinstance(var._data, PandasIndexAdapter):
            idx_level_names = var.to_index_variable().level_names
            if idx_level_names is not None:
                for n in idx_level_names:
                    level_names[n].append('%r (%s)' % (n, var_name))

    for k, v in level_names.items():
        if k in variables:
            v.append('(%s)' % k)

    duplicate_names = [v for v in level_names.values() if len(v) > 1]
    if duplicate_names:
        conflict_str = '\n'.join([', '.join(v) for v in duplicate_names])
        raise ValueError('conflicting MultiIndex level name(s):\n%s'
                         % conflict_str)
