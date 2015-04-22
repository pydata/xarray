from datetime import timedelta
import functools
import itertools

import numpy as np
import pandas as pd

from . import common
from . import indexing
from . import ops
from . import utils
from .pycompat import basestring, OrderedDict, zip, reduce, dask_array_type
from .indexing import (PandasIndexAdapter, LazilyIndexedArray,
                       orthogonally_indexable)

import xray # only for Dataset and DataArray


def as_variable(obj, key=None, strict=True):
    """Convert an object into an Variable

    - If the object is already an `Variable`, return it.
    - If the object is a `DataArray`, return it if `strict=False` or return
      its variable if `strict=True`.
    - Otherwise, if the object has 'dims' and 'data' attributes, convert
      it into a new `Variable`.
    - If all else fails, attempt to convert the object into an `Variable` by
      unpacking it into the arguments for `Variable.__init__`.
    """
    # TODO: consider extending this method to automatically handle Iris and
    # pandas objects.
    if strict and hasattr(obj, 'variable'):
        # extract the primary Variable from DataArrays
        obj = obj.variable
    if not isinstance(obj, (Variable, xray.DataArray)):
        if hasattr(obj, 'dims') and (hasattr(obj, 'data')
                                     or hasattr(obj, 'values')):
            obj = Variable(obj.dims, getattr(obj, 'data', obj.values),
                           getattr(obj, 'attrs', None),
                           getattr(obj, 'encoding', None))
        elif isinstance(obj, tuple):
            try:
                obj = Variable(*obj)
            except TypeError:
                raise TypeError('cannot convert argument into an Variable')
        elif utils.is_scalar(obj):
            obj = Variable([], obj)
        elif getattr(obj, 'name', None) is not None:
            obj = Variable(obj.name, obj)
        elif key is not None:
            obj = Variable(key, obj)
        else:
            raise TypeError('cannot infer Variable dimensions')
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


def _as_compatible_data(data, fastpath=False):
    """Prepare and wrap data to put in a Variable.

    - If data does not have the necessary attributes, convert it to ndarray.
    - If data has dtype=datetime64, ensure that it has ns precision. If it's a
      pandas.Timestamp, convert it to datetime64.
    - If data is already a pandas or xray object (other than an Index), just
      use the values.

    Finally, wrap it up with an adapter if necessary.
    """
    if fastpath and getattr(data, 'ndim', 0) > 0:
        # can't use fastpath (yet) for scalars
        return _maybe_wrap_data(data)

    # add a custom fast-path for dask.array to avoid expensive checks for the
    # dtype attribute
    if isinstance(data, dask_array_type):
        return data

    if isinstance(data, pd.Index):
        if isinstance(data, pd.MultiIndex):
            raise NotImplementedError(
                'no support yet for using a pandas.MultiIndex in an '
                'xray.Coordinate')
        return _maybe_wrap_data(data)

    if isinstance(data, pd.Timestamp):
        # TODO: convert, handle datetime objects, too
        data = np.datetime64(data.value, 'ns')
    if isinstance(data, timedelta):
        data = np.timedelta64(getattr(data, 'value', data), 'ns')

    if (not hasattr(data, 'dtype') or not hasattr(data, 'shape')
            or isinstance(data, (np.string_, np.datetime64, np.timedelta64))):
        # data must be ndarray-like
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
        data = common._possibly_convert_objects(data)
        if data.dtype.kind == 'M':
            # TODO: automatically cast arrays of datetime objects as well
            data = np.asarray(data, 'datetime64[ns]')
        if data.dtype.kind == 'm':
            data = np.asarray(data, 'timedelta64[ns]')

    return _maybe_wrap_data(data)


def _as_array_or_item(data):
    """Return the given values as a numpy array, or as an individual item if
    it's a 0-dimensional object array or datetime64.

    Importantly, this function does not copy data if it is already an ndarray -
    otherwise, it will not be possible to update Variable values in place.
    """
    data = np.asarray(data)
    if data.ndim == 0:
        if data.dtype.kind == 'O':
            # unpack 0d object arrays to be consistent with numpy
            data = data.item()
        elif data.dtype.kind == 'M':
            # convert to a np.datetime64 object, because 0-dimensional ndarrays
            # with dtype=datetime64 are broken :(
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
            Well behaviored code to serialize a Variable should ignore
            unrecognized encoding items.
        """
        self._data = _as_compatible_data(data, fastpath=fastpath)
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
        return isinstance(self._data, (np.ndarray, PandasIndexAdapter))

    @property
    def data(self):
        if isinstance(self._data, dask_array_type):
            return self._data
        else:
            return self.values

    @data.setter
    def data(self, data):
        data = _as_compatible_data(data)
        if data.shape != self.shape:
            raise ValueError(
                "replacement data must match the Variable's shape")
        self._data = data

    def _data_cached(self):
        if not isinstance(self._data, np.ndarray):
            self._data = np.asarray(self._data)
        return self._data

    @property
    def _indexable_data(self):
        return orthogonally_indexable(self._data)

    def load(self):
        """Manually trigger loading of this variable's data from disk or a
        remote source into memory and return this variable.

        Normally, it should not be necessary to call this method in user code,
        because all xray functions should either work on deferred data or
        load data automatically.
        """
        self._data_cached()
        return self

    def load_data(self):  # pragma: no cover
        warnings.warn('the Variable method `load_data` has been deprecated; '
                      'use `load` instead',
                      FutureWarning, stacklevel=2)
        return self.load()

    def __getstate__(self):
        """Always cache data as an in-memory array before pickling"""
        self._data_cached()
        # self.__dict__ is the default pickle object, we don't need to
        # implement our own __setstate__ method to make pickle work
        return self.__dict__

    @property
    def values(self):
        """The variable's data as a numpy.ndarray"""
        return _as_array_or_item(self._data_cached())

    @values.setter
    def values(self, values):
        self.data = values

    def to_variable(self):
        """Return this variable as a base xray.Variable"""
        return Variable(self.dims, self._data, self._attrs,
                        encoding=self._encoding, fastpath=True)

    def to_coord(self):
        """Return this variable as an xray.Coordinate"""
        return Coordinate(self.dims, self._data, self._attrs,
                          encoding=self._encoding, fastpath=True)

    def to_index(self):
        """Convert this variable to a pandas.Index"""
        return self.to_coord().to_index()

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
                     if not isinstance(k, (int, np.integer)))
        values = self._indexable_data[key]
        # orthogonal indexing should ensure the dimensionality is consistent
        if hasattr(values, 'ndim'):
            assert values.ndim == len(dims), (values.ndim, len(dims))
        else:
            assert len(dims) == 0, len(dims)
        return type(self)(dims, values, self._attrs, fastpath=True)

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
        data = orthogonally_indexable(self._data_cached())
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
        self._encoding = dict(value)

    def copy(self, deep=True):
        """Returns a copy of this object.

        If `deep=True`, the data array is loaded into memory and copied onto
        the new object. Dimensions, attributes and encodings are always copied.
        """
        data = self.values.copy() if deep else self._data
        # note:
        # dims is already an immutable tuple
        # attributes and encoding will be copied when the new Array is created
        return type(self)(self.dims, data, self._attrs, self._encoding,
                          fastpath=True)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        # memo does nothing but is required for compatability with
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

    def chunk(self, chunks=None, name=''):
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

        Returns
        -------
        chunked : xray.Variable
        """
        import dask.array as da

        if utils.is_dict_like(chunks):
            chunks = dict((self.get_axis_num(dim), chunk)
                          for dim, chunk in chunks.items())

        if chunks is None:
            chunks = self.chunks or self.shape

        data = self._data
        if isinstance(data, dask_array_type):
            data = data.rechunk(chunks)
        else:
            if name:
                name += '_'
            name = 'xray_%s%s' % (name, next(self._array_counter))

            if utils.is_dict_like(chunks):
                chunks = tuple(chunks.get(n, s)
                               for n, s in enumerate(self.shape))

            data = da.from_array(data, chunks, name=name)

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
        invalid = [k for k in indexers if not k in self.dims]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        key = [slice(None)] * self.ndim
        for i, dim in enumerate(self.dims):
            if dim in indexers:
                key[i] = indexers[dim]
        return self[tuple(key)]

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
        data = ops.transpose(self.data, axes)
        return type(self)(dims, data, self._attrs, self._encoding, fastpath=True)

    def squeeze(self, dim=None):
        """Return a new Variable object with squeezed data.

        Parameters
        ----------
        dim : None or str or tuple of str, optional
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
        dims = dict(zip(self.dims, self.shape))
        return common.squeeze(self, dims, dim)

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
        expanded_dims = tuple(d for d in dims if d not in self_dims) + self.dims
        if shape is not None:
            dims_map = dict(zip(dims, shape))
            tmp_shape = [dims_map[d] for d in expanded_dims]
            expanded_data = ops.broadcast_to(self.data, tmp_shape)
        else:
            expanded_data = self.data[(None,) * (len(expanded_dims) - self.ndim)]
        expanded_var = Variable(expanded_dims, expanded_data, self._attrs,
                                self._encoding, fastpath=True)
        return expanded_var.transpose(*dims)

    def fillna(self, value):
        return self._fillna(value)

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

        if dim is not None:
            axis = self.get_axis_num(dim)
        data = func(self.data if allow_lazy else self.values,
                    axis=axis, **kwargs)

        removed_axes = (range(self.ndim) if axis is None
                        else np.atleast_1d(axis) % self.ndim)
        dims = [dim for n, dim in enumerate(self.dims)
                if n not in removed_axes]

        attrs = self._attrs if keep_attrs else None

        return Variable(dims, data, attrs=attrs)

    @classmethod
    def concat(cls, variables, dim='concat_dim', indexers=None,
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
        indexers : iterable of indexers, optional
            Iterable of indexers of the same length as variables which
            specifies how to assign variables along the given dimension. If
            not supplied, indexers is inferred from the length of each
            variable along the dimension, and the variables are stacked in the
            given order.
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
            if indexers is None:
                data = ops.concatenate(arrays, axis=axis)
            else:
                data = ops.interleaved_concat(arrays, indexers, axis=axis)
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

    def _data_equals(self, other):
        return (self._data is other._data
                or ops.array_equiv(self.data, other.data))

    def equals(self, other):
        """True if two Variables have the same dimensions and values;
        otherwise False.

        Variables can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for Variables
        does element-wise comparisions (like numpy.ndarrays).
        """
        other = getattr(other, 'variable', other)
        try:
            return (self.dims == other.dims
                    and self._data_equals(other))
        except (TypeError, AttributeError):
            return False

    def broadcast_equals(self, other):
        """True if two Variables have the values after being broadcast against
        each other; otherwise False.

        Variables can still be equal (like pandas objects) if they have NaN
        values in the same locations.
        """
        try:
            self, other = broadcast_variables(self, other)
        except (ValueError, AttributeError):
            return False
        return self.equals(other)

    def identical(self, other):
        """Like equals, but also checks attributes.
        """
        try:
            return (utils.dict_equiv(self.attrs, other.attrs)
                    and self.equals(other))
        except (TypeError, AttributeError):
            return False

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
            if isinstance(other, (xray.DataArray, xray.Dataset)):
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
            if isinstance(other, xray.Dataset):
                raise TypeError('cannot add a Dataset to a Variable in-place')
            self_data, other_data, dims = _broadcast_compat_data(self, other)
            if dims != self.dims:
                raise ValueError('dimensions cannot change for in-place '
                                 'operations')
            self.values = f(self_data, other_data)
            return self
        return func

ops.inject_all_ops_and_reduce_methods(Variable)


class Coordinate(Variable):
    """Wrapper around pandas.Index that adds xray specific functionality.

    The most important difference is that Coordinate objects must always have a
    name, which is the dimension along which they index values.

    Coordinates must always be 1-dimensional. In addition to Variable methods
    and properties (attributes, encoding, broadcasting), they support some
    pandas.Index methods directly (e.g., get_indexer), even though pandas does
    not (yet) support duck-typing for indexes.
    """
    def __init__(self, name, data, attrs=None, encoding=None, fastpath=False):
        super(Coordinate, self).__init__(name, data, attrs, encoding, fastpath)
        if self.ndim != 1:
            raise ValueError('%s objects must be 1-dimensional' %
                             type(self).__name__)

    def _data_cached(self):
        if not isinstance(self._data, PandasIndexAdapter):
            self._data = PandasIndexAdapter(self._data)
        return self._data

    def __getitem__(self, key):
        key = self._item_key_to_tuple(key)
        values = self._indexable_data[key]
        if not hasattr(values, 'ndim') or values.ndim == 0:
            return Variable((), values, self._attrs, self._encoding)
        else:
            return type(self)(self.dims, values, self._attrs, self._encoding,
                              fastpath=True)

    def __setitem__(self, key, value):
        raise TypeError('%s values cannot be modified' % type(self).__name__)

    def copy(self, deep=True):
        """Returns a copy of this object.

        If `deep=True`, the values array is loaded into memory and copied onto
        the new object. Dimensions, attributes and encodings are always copied.
        """
        # there is no need to copy the index values here even if deep=True
        # since pandas.Index objects are immutable
        data = PandasIndexAdapter(self) if deep else self._data
        return type(self)(self.dims, data, self._attrs, self._encoding,
                          fastpath=True)

    def _data_equals(self, other):
        return self.to_index().equals(other.to_index())

    def to_coord(self):
        """Return this variable as an xray.Coordinate"""
        return self

    def to_index(self):
        """Convert this variable to a pandas.Index"""
        # n.b. creating a new pandas.Index from an old pandas.Index is
        # basically free as pandas.Index objects are immutable
        assert self.ndim == 1
        return pd.Index(self._data_cached().array, name=self.dims[0])

    # pandas.Index like properties:

    @property
    def name(self):
        return self.dims[0]

    @name.setter
    def name(self, value):
        raise AttributeError('cannot modify name of Coordinate in-place')

    def get_indexer(self, label):
        return self.to_index().get_indexer(label)

    def slice_indexer(self, start=None, stop=None, step=None):
        return self.to_index().slice_indexer(start, stop, step)

    def slice_locs(self, start=None, stop=None):
        return self.to_index().slice_locs(start, stop)

    def get_loc(self, label):
        return self.to_index().get_loc(label)

    @property
    def is_monotonic(self):
        return self.to_index().is_monotonic

    def is_numeric(self):
        return self.to_index().is_numeric()


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
    dimensions are sorted in order of appearence in the first variable's
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
