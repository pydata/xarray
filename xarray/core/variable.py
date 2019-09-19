import functools
import itertools
from collections import OrderedDict, defaultdict
from datetime import timedelta
from distutils.version import LooseVersion
from typing import Any, Hashable, Mapping, Union

import numpy as np
import pandas as pd

import xarray as xr  # only for Dataset and DataArray

from . import arithmetic, common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from .indexing import (
    BasicIndexer,
    OuterIndexer,
    PandasIndexAdapter,
    VectorizedIndexer,
    as_indexable,
)
from .npcompat import IS_NEP18_ACTIVE
from .options import _get_keep_attrs
from .pycompat import dask_array_type, integer_types
from .utils import (
    OrderedSet,
    decode_numpy_dict_values,
    either_dict_or_kwargs,
    ensure_us_time_resolution,
)

try:
    import dask.array as da
except ImportError:
    pass


NON_NUMPY_SUPPORTED_ARRAY_TYPES = (
    indexing.ExplicitlyIndexed,
    pd.Index,
) + dask_array_type
# https://github.com/python/mypy/issues/224
BASIC_INDEXING_TYPES = integer_types + (slice,)  # type: ignore


class MissingDimensionsError(ValueError):
    """Error class used when we can't safely guess a dimension name.
    """

    # inherits from ValueError for backward compatibility
    # TODO: move this to an xarray.exceptions module?


def as_variable(obj, name=None) -> "Union[Variable, IndexVariable]":
    """Convert an object into a Variable.

    Parameters
    ----------
    obj : object
        Object to convert into a Variable.

        - If the object is already a Variable, return a shallow copy.
        - Otherwise, if the object has 'dims' and 'data' attributes, convert
          it into a new Variable.
        - If all else fails, attempt to convert the object into a Variable by
          unpacking it into the arguments for creating a new Variable.
    name : str, optional
        If provided:

        - `obj` can be a 1D array, which is assumed to label coordinate values
          along a dimension of this given name.
        - Variables with name matching one of their dimensions are converted
          into `IndexVariable` objects.

    Returns
    -------
    var : Variable
        The newly created variable.

    """
    from .dataarray import DataArray

    # TODO: consider extending this method to automatically handle Iris and
    if isinstance(obj, DataArray):
        # extract the primary Variable from DataArrays
        obj = obj.variable

    if isinstance(obj, Variable):
        obj = obj.copy(deep=False)
    elif isinstance(obj, tuple):
        try:
            obj = Variable(*obj)
        except (TypeError, ValueError) as error:
            # use .format() instead of % because it handles tuples consistently
            raise error.__class__(
                "Could not convert tuple of form "
                "(dims, data[, attrs, encoding]): "
                "{} to Variable.".format(obj)
            )
    elif utils.is_scalar(obj):
        obj = Variable([], obj)
    elif isinstance(obj, (pd.Index, IndexVariable)) and obj.name is not None:
        obj = Variable(obj.name, obj)
    elif isinstance(obj, (set, dict)):
        raise TypeError("variable %r has invalid type %r" % (name, type(obj)))
    elif name is not None:
        data = as_compatible_data(obj)
        if data.ndim != 1:
            raise MissingDimensionsError(
                "cannot set variable %r with %r-dimensional data "
                "without explicit dimension names. Pass a tuple of "
                "(dims, data) instead." % (name, data.ndim)
            )
        obj = Variable(name, data, fastpath=True)
    else:
        raise TypeError(
            "unable to convert object into a variable without an "
            "explicit list of dimensions: %r" % obj
        )

    if name is not None and name in obj.dims:
        # convert the Variable into an Index
        if obj.ndim != 1:
            raise MissingDimensionsError(
                "%r has more than 1-dimension and the same name as one of its "
                "dimensions %r. xarray disallows such variables because they "
                "conflict with the coordinates used to label "
                "dimensions." % (name, obj.dims)
            )
        obj = obj.to_index_variable()

    return obj


def _maybe_wrap_data(data):
    """
    Put pandas.Index and numpy.ndarray arguments in adapter objects to ensure
    they can be indexed properly.

    NumpyArrayAdapter, PandasIndexAdapter and LazilyOuterIndexedArray should
    all pass through unmodified.
    """
    if isinstance(data, pd.Index):
        return PandasIndexAdapter(data)
    return data


def _possibly_convert_objects(values):
    """Convert arrays of datetime.datetime and datetime.timedelta objects into
    datetime64 and timedelta64, according to the pandas convention.
    """
    return np.asarray(pd.Series(values.ravel())).reshape(values.shape)


def as_compatible_data(data, fastpath=False):
    """Prepare and wrap data to put in a Variable.

    - If data does not have the necessary attributes, convert it to ndarray.
    - If data has dtype=datetime64, ensure that it has ns precision. If it's a
      pandas.Timestamp, convert it to datetime64.
    - If data is already a pandas or xarray object (other than an Index), just
      use the values.

    Finally, wrap it up with an adapter if necessary.
    """
    if fastpath and getattr(data, "ndim", 0) > 0:
        # can't use fastpath (yet) for scalars
        return _maybe_wrap_data(data)

    if isinstance(data, Variable):
        return data.data

    if isinstance(data, NON_NUMPY_SUPPORTED_ARRAY_TYPES):
        return _maybe_wrap_data(data)

    if isinstance(data, tuple):
        data = utils.to_0d_object_array(data)

    if isinstance(data, pd.Timestamp):
        # TODO: convert, handle datetime objects, too
        data = np.datetime64(data.value, "ns")

    if isinstance(data, timedelta):
        data = np.timedelta64(getattr(data, "value", data), "ns")

    # we don't want nested self-described arrays
    data = getattr(data, "values", data)

    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)
        if mask.any():
            dtype, fill_value = dtypes.maybe_promote(data.dtype)
            data = np.asarray(data, dtype=dtype)
            data[mask] = fill_value
        else:
            data = np.asarray(data)

    if not isinstance(data, np.ndarray):
        if hasattr(data, "__array_function__"):
            if IS_NEP18_ACTIVE:
                return data
            else:
                raise TypeError(
                    "Got an NumPy-like array type providing the "
                    "__array_function__ protocol but NEP18 is not enabled. "
                    "Check that numpy >= v1.16 and that the environment "
                    'variable "NUMPY_EXPERIMENTAL_ARRAY_FUNCTION" is set to '
                    '"1"'
                )

    # validate whether the data is valid data types
    data = np.asarray(data)

    if isinstance(data, np.ndarray):
        if data.dtype.kind == "O":
            data = _possibly_convert_objects(data)
        elif data.dtype.kind == "M":
            data = np.asarray(data, "datetime64[ns]")
        elif data.dtype.kind == "m":
            data = np.asarray(data, "timedelta64[ns]")

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
        if data.dtype.kind == "M":
            data = np.datetime64(data, "ns")
        elif data.dtype.kind == "m":
            data = np.timedelta64(data, "ns")
    return data


class Variable(
    common.AbstractArray, arithmetic.SupportsArithmetic, utils.NdimSizeLenMixin
):
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

    __slots__ = ("_dims", "_data", "_attrs", "_encoding")

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
        return isinstance(self._data, (np.ndarray, np.number, PandasIndexAdapter)) or (
            isinstance(self._data, indexing.MemoryCachedArray)
            and isinstance(self._data.array, indexing.NumpyIndexingAdapter)
        )

    @property
    def data(self):
        if hasattr(self._data, "__array_function__") or isinstance(
            self._data, dask_array_type
        ):
            return self._data
        else:
            return self.values

    @data.setter
    def data(self, data):
        data = as_compatible_data(data)
        if data.shape != self.shape:
            raise ValueError("replacement data must match the Variable's shape")
        self._data = data

    def load(self, **kwargs):
        """Manually trigger loading of this variable's data from disk or a
        remote source into memory and return this variable.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        if isinstance(self._data, dask_array_type):
            self._data = as_compatible_data(self._data.compute(**kwargs))
        elif not hasattr(self._data, "__array_function__"):
            self._data = np.asarray(self._data)
        return self

    def compute(self, **kwargs):
        """Manually trigger loading of this variable's data from disk or a
        remote source into memory and return a new variable. The original is
        left unaltered.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        new = self.copy(deep=False)
        return new.load(**kwargs)

    def __dask_graph__(self):
        if isinstance(self._data, dask_array_type):
            return self._data.__dask_graph__()
        else:
            return None

    def __dask_keys__(self):
        return self._data.__dask_keys__()

    def __dask_layers__(self):
        return self._data.__dask_layers__()

    @property
    def __dask_optimize__(self):
        return self._data.__dask_optimize__

    @property
    def __dask_scheduler__(self):
        return self._data.__dask_scheduler__

    def __dask_postcompute__(self):
        array_func, array_args = self._data.__dask_postcompute__()
        return (
            self._dask_finalize,
            (array_func, array_args, self._dims, self._attrs, self._encoding),
        )

    def __dask_postpersist__(self):
        array_func, array_args = self._data.__dask_postpersist__()
        return (
            self._dask_finalize,
            (array_func, array_args, self._dims, self._attrs, self._encoding),
        )

    @staticmethod
    def _dask_finalize(results, array_func, array_args, dims, attrs, encoding):
        if isinstance(results, dict):  # persist case
            name = array_args[0]
            results = {k: v for k, v in results.items() if k[0] == name}
        data = array_func(results, *array_args)
        return Variable(dims, data, attrs=attrs, encoding=encoding)

    @property
    def values(self):
        """The variable's data as a numpy.ndarray"""
        return _as_array_or_item(self._data)

    @values.setter
    def values(self, values):
        self.data = values

    def to_base_variable(self):
        """Return this variable as a base xarray.Variable"""
        return Variable(
            self.dims, self._data, self._attrs, encoding=self._encoding, fastpath=True
        )

    to_variable = utils.alias(to_base_variable, "to_variable")

    def to_index_variable(self):
        """Return this variable as an xarray.IndexVariable"""
        return IndexVariable(
            self.dims, self._data, self._attrs, encoding=self._encoding, fastpath=True
        )

    to_coord = utils.alias(to_index_variable, "to_coord")

    def to_index(self):
        """Convert this variable to a pandas.Index"""
        return self.to_index_variable().to_index()

    def to_dict(self, data=True):
        """Dictionary representation of variable."""
        item = {"dims": self.dims, "attrs": decode_numpy_dict_values(self.attrs)}
        if data:
            item["data"] = ensure_us_time_resolution(self.values).tolist()
        else:
            item.update({"dtype": str(self.dtype), "shape": self.shape})
        return item

    @property
    def dims(self):
        """Tuple of dimension names with which this variable is associated.
        """
        return self._dims

    @dims.setter
    def dims(self, value):
        self._dims = self._parse_dimensions(value)

    def _parse_dimensions(self, dims):
        if isinstance(dims, str):
            dims = (dims,)
        dims = tuple(dims)
        if len(dims) != self.ndim:
            raise ValueError(
                "dimensions %s must have the same length as the "
                "number of data dimensions, ndim=%s" % (dims, self.ndim)
            )
        return dims

    def _item_key_to_tuple(self, key):
        if utils.is_dict_like(key):
            return tuple(key.get(dim, slice(None)) for dim in self.dims)
        else:
            return key

    def _broadcast_indexes(self, key):
        """Prepare an indexing key for an indexing operation.

        Parameters
        -----------
        key: int, slice, array, dict or tuple of integer, slices and arrays
            Any valid input for indexing.

        Returns
        -------
        dims: tuple
            Dimension of the resultant variable.
        indexers: IndexingTuple subclass
            Tuple of integer, array-like, or slices to use when indexing
            self._data. The type of this argument indicates the type of
            indexing to perform, either basic, outer or vectorized.
        new_order : Optional[Sequence[int]]
            Optional reordering to do on the result of indexing. If not None,
            the first len(new_order) indexing should be moved to these
            positions.
        """
        key = self._item_key_to_tuple(key)  # key is a tuple
        # key is a tuple of full size
        key = indexing.expanded_indexer(key, self.ndim)
        # Convert a scalar Variable to an integer
        key = tuple(
            k.data.item() if isinstance(k, Variable) and k.ndim == 0 else k for k in key
        )
        # Convert a 0d-array to an integer
        key = tuple(
            k.item() if isinstance(k, np.ndarray) and k.ndim == 0 else k for k in key
        )

        if all(isinstance(k, BASIC_INDEXING_TYPES) for k in key):
            return self._broadcast_indexes_basic(key)

        self._validate_indexers(key)
        # Detect it can be mapped as an outer indexer
        # If all key is unlabeled, or
        # key can be mapped as an OuterIndexer.
        if all(not isinstance(k, Variable) for k in key):
            return self._broadcast_indexes_outer(key)

        # If all key is 1-dimensional and there are no duplicate labels,
        # key can be mapped as an OuterIndexer.
        dims = []
        for k, d in zip(key, self.dims):
            if isinstance(k, Variable):
                if len(k.dims) > 1:
                    return self._broadcast_indexes_vectorized(key)
                dims.append(k.dims[0])
            elif not isinstance(k, integer_types):
                dims.append(d)
        if len(set(dims)) == len(dims):
            return self._broadcast_indexes_outer(key)

        return self._broadcast_indexes_vectorized(key)

    def _broadcast_indexes_basic(self, key):
        dims = tuple(
            dim for k, dim in zip(key, self.dims) if not isinstance(k, integer_types)
        )
        return dims, BasicIndexer(key), None

    def _validate_indexers(self, key):
        """ Make sanity checks """
        for dim, k in zip(self.dims, key):
            if isinstance(k, BASIC_INDEXING_TYPES):
                pass
            else:
                if not isinstance(k, Variable):
                    k = np.asarray(k)
                    if k.ndim > 1:
                        raise IndexError(
                            "Unlabeled multi-dimensional array cannot be "
                            "used for indexing: {}".format(k)
                        )
                if k.dtype.kind == "b":
                    if self.shape[self.get_axis_num(dim)] != len(k):
                        raise IndexError(
                            "Boolean array size {:d} is used to index array "
                            "with shape {:s}.".format(len(k), str(self.shape))
                        )
                    if k.ndim > 1:
                        raise IndexError(
                            "{}-dimensional boolean indexing is "
                            "not supported. ".format(k.ndim)
                        )
                    if getattr(k, "dims", (dim,)) != (dim,):
                        raise IndexError(
                            "Boolean indexer should be unlabeled or on the "
                            "same dimension to the indexed array. Indexer is "
                            "on {:s} but the target dimension is {:s}.".format(
                                str(k.dims), dim
                            )
                        )

    def _broadcast_indexes_outer(self, key):
        dims = tuple(
            k.dims[0] if isinstance(k, Variable) else dim
            for k, dim in zip(key, self.dims)
            if not isinstance(k, integer_types)
        )

        new_key = []
        for k in key:
            if isinstance(k, Variable):
                k = k.data
            if not isinstance(k, BASIC_INDEXING_TYPES):
                k = np.asarray(k)
                if k.dtype.kind == "b":
                    (k,) = np.nonzero(k)
            new_key.append(k)

        return dims, OuterIndexer(tuple(new_key)), None

    def _nonzero(self):
        """ Equivalent numpy's nonzero but returns a tuple of Varibles. """
        # TODO we should replace dask's native nonzero
        # after https://github.com/dask/dask/issues/1076 is implemented.
        nonzeros = np.nonzero(self.data)
        return tuple(Variable((dim), nz) for nz, dim in zip(nonzeros, self.dims))

    def _broadcast_indexes_vectorized(self, key):
        variables = []
        out_dims_set = OrderedSet()
        for dim, value in zip(self.dims, key):
            if isinstance(value, slice):
                out_dims_set.add(dim)
            else:
                variable = (
                    value
                    if isinstance(value, Variable)
                    else as_variable(value, name=dim)
                )
                if variable.dtype.kind == "b":  # boolean indexing case
                    (variable,) = variable._nonzero()

                variables.append(variable)
                out_dims_set.update(variable.dims)

        variable_dims = set()
        for variable in variables:
            variable_dims.update(variable.dims)

        slices = []
        for i, (dim, value) in enumerate(zip(self.dims, key)):
            if isinstance(value, slice):
                if dim in variable_dims:
                    # We only convert slice objects to variables if they share
                    # a dimension with at least one other variable. Otherwise,
                    # we can equivalently leave them as slices aknd transpose
                    # the result. This is significantly faster/more efficient
                    # for most array backends.
                    values = np.arange(*value.indices(self.sizes[dim]))
                    variables.insert(i - len(slices), Variable((dim,), values))
                else:
                    slices.append((i, value))

        try:
            variables = _broadcast_compat_variables(*variables)
        except ValueError:
            raise IndexError("Dimensions of indexers mismatch: {}".format(key))

        out_key = [variable.data for variable in variables]
        out_dims = tuple(out_dims_set)
        slice_positions = set()
        for i, value in slices:
            out_key.insert(i, value)
            new_position = out_dims.index(self.dims[i])
            slice_positions.add(new_position)

        if slice_positions:
            new_order = [i for i in range(len(out_dims)) if i not in slice_positions]
        else:
            new_order = None

        return out_dims, VectorizedIndexer(tuple(out_key)), new_order

    def __getitem__(self, key):
        """Return a new Array object whose contents are consistent with
        getting the provided key from the underlying data.

        NB. __getitem__ and __setitem__ implement xarray-style indexing,
        where if keys are unlabeled arrays, we index the array orthogonally
        with them. If keys are labeled array (such as Variables), they are
        broadcasted with our usual scheme and then the array is indexed with
        the broadcasted key, like numpy's fancy indexing.

        If you really want to do indexing like `x[x > 0]`, manipulate the numpy
        array `x.values` directly.
        """
        dims, indexer, new_order = self._broadcast_indexes(key)
        data = as_indexable(self._data)[indexer]
        if new_order:
            data = duck_array_ops.moveaxis(data, range(len(new_order)), new_order)
        return self._finalize_indexing_result(dims, data)

    def _finalize_indexing_result(self, dims, data):
        """Used by IndexVariable to return IndexVariable objects when possible.
        """
        return type(self)(dims, data, self._attrs, self._encoding, fastpath=True)

    def _getitem_with_mask(self, key, fill_value=dtypes.NA):
        """Index this Variable with -1 remapped to fill_value."""
        # TODO(shoyer): expose this method in public API somewhere (isel?) and
        # use it for reindex.
        # TODO(shoyer): add a sanity check that all other integers are
        # non-negative
        # TODO(shoyer): add an optimization, remapping -1 to an adjacent value
        # that is actually indexed rather than mapping it to the last value
        # along each axis.

        if fill_value is dtypes.NA:
            fill_value = dtypes.get_fill_value(self.dtype)

        dims, indexer, new_order = self._broadcast_indexes(key)

        if self.size:
            if isinstance(self._data, dask_array_type):
                # dask's indexing is faster this way; also vindex does not
                # support negative indices yet:
                # https://github.com/dask/dask/pull/2967
                actual_indexer = indexing.posify_mask_indexer(indexer)
            else:
                actual_indexer = indexer

            data = as_indexable(self._data)[actual_indexer]
            mask = indexing.create_mask(indexer, self.shape, data)
            data = duck_array_ops.where(mask, fill_value, data)
        else:
            # array cannot be indexed along dimensions of size 0, so just
            # build the mask directly instead.
            mask = indexing.create_mask(indexer, self.shape)
            data = np.broadcast_to(fill_value, getattr(mask, "shape", ()))

        if new_order:
            data = duck_array_ops.moveaxis(data, range(len(new_order)), new_order)
        return self._finalize_indexing_result(dims, data)

    def __setitem__(self, key, value):
        """__setitem__ is overloaded to access the underlying numpy values with
        orthogonal indexing.

        See __getitem__ for more details.
        """
        dims, index_tuple, new_order = self._broadcast_indexes(key)

        if not isinstance(value, Variable):
            value = as_compatible_data(value)
            if value.ndim > len(dims):
                raise ValueError(
                    "shape mismatch: value array of shape %s could not be "
                    "broadcast to indexing result with %s dimensions"
                    % (value.shape, len(dims))
                )
            if value.ndim == 0:
                value = Variable((), value)
            else:
                value = Variable(dims[-value.ndim :], value)
        # broadcast to become assignable
        value = value.set_dims(dims).data

        if new_order:
            value = duck_array_ops.asarray(value)
            value = value[(len(dims) - value.ndim) * (np.newaxis,) + (Ellipsis,)]
            value = duck_array_ops.moveaxis(value, new_order, range(len(new_order)))

        indexable = as_indexable(self._data)
        indexable[index_tuple] = value

    @property
    def attrs(self) -> "OrderedDict[Any, Any]":
        """Dictionary of local attributes on this variable.
        """
        if self._attrs is None:
            self._attrs = OrderedDict()
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
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
            raise ValueError("encoding must be castable to a dictionary")

    def copy(self, deep=True, data=None):
        """Returns a copy of this object.

        If `deep=True`, the data array is loaded into memory and copied onto
        the new object. Dimensions, attributes and encodings are always copied.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, optional
            Whether the data array is loaded into memory and copied onto
            the new object. Default is True.
        data : array_like, optional
            Data to use in the new object. Must have same shape as original.
            When `data` is used, `deep` is ignored.

        Returns
        -------
        object : Variable
            New object with dimensions, attributes, encodings, and optionally
            data copied from original.

        Examples
        --------

        Shallow copy versus deep copy

        >>> var = xr.Variable(data=[1, 2, 3], dims='x')
        >>> var.copy()
        <xarray.Variable (x: 3)>
        array([1, 2, 3])
        >>> var_0 = var.copy(deep=False)
        >>> var_0[0] = 7
        >>> var_0
        <xarray.Variable (x: 3)>
        array([7, 2, 3])
        >>> var
        <xarray.Variable (x: 3)>
        array([7, 2, 3])

        Changing the data using the ``data`` argument maintains the
        structure of the original object, but with the new data. Original
        object is unaffected.

        >>> var.copy(data=[0.1, 0.2, 0.3])
        <xarray.Variable (x: 3)>
        array([ 0.1,  0.2,  0.3])
        >>> var
        <xarray.Variable (x: 3)>
        array([7, 2, 3])

        See Also
        --------
        pandas.DataFrame.copy
        """
        if data is None:
            data = self._data

            if isinstance(data, indexing.MemoryCachedArray):
                # don't share caching between copies
                data = indexing.MemoryCachedArray(data.array)

            if deep:
                if hasattr(data, "__array_function__") or isinstance(
                    data, dask_array_type
                ):
                    data = data.copy()
                elif not isinstance(data, PandasIndexAdapter):
                    # pandas.Index is immutable
                    data = np.array(data)
        else:
            data = as_compatible_data(data)
            if self.shape != data.shape:
                raise ValueError(
                    "Data shape {} must match shape of object {}".format(
                        data.shape, self.shape
                    )
                )

        # note:
        # dims is already an immutable tuple
        # attributes and encoding will be copied when the new Array is created
        return type(self)(self.dims, data, self._attrs, self._encoding, fastpath=True)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        # memo does nothing but is required for compatibility with
        # copy.deepcopy
        return self.copy(deep=True)

    # mutable objects should not be hashable
    # https://github.com/python/mypy/issues/4266
    __hash__ = None  # type: ignore

    @property
    def chunks(self):
        """Block dimensions for this array's data or None if it's not a dask
        array.
        """
        return getattr(self._data, "chunks", None)

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
        import dask
        import dask.array as da

        if utils.is_dict_like(chunks):
            chunks = {self.get_axis_num(dim): chunk for dim, chunk in chunks.items()}

        if chunks is None:
            chunks = self.chunks or self.shape

        data = self._data
        if isinstance(data, da.Array):
            data = data.rechunk(chunks)
        else:
            if isinstance(data, indexing.ExplicitlyIndexed):
                # Unambiguously handle array storage backends (like NetCDF4 and h5py)
                # that can't handle general array indexing. For example, in netCDF4 you
                # can do "outer" indexing along two dimensions independent, which works
                # differently from how NumPy handles it.
                # da.from_array works by using lazy indexing with a tuple of slices.
                # Using OuterIndexer is a pragmatic choice: dask does not yet handle
                # different indexing types in an explicit way:
                # https://github.com/dask/dask/issues/2883
                data = indexing.ImplicitToExplicitIndexingAdapter(
                    data, indexing.OuterIndexer
                )
                if LooseVersion(dask.__version__) < "2.0.0":
                    kwargs = {}
                else:
                    # All of our lazily loaded backend array classes should use NumPy
                    # array operations.
                    kwargs = {"meta": np.ndarray}
            else:
                kwargs = {}

            if utils.is_dict_like(chunks):
                chunks = tuple(chunks.get(n, s) for n, s in enumerate(self.shape))

            data = da.from_array(data, chunks, name=name, lock=lock, **kwargs)

        return type(self)(self.dims, data, self._attrs, self._encoding, fastpath=True)

    def isel(self, indexers=None, drop=False, **indexers_kwargs):
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
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")

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
        return self.isel({d: 0 for d in dims})

    def _shift_one_dim(self, dim, count, fill_value=dtypes.NA):
        axis = self.get_axis_num(dim)

        if count > 0:
            keep = slice(None, -count)
        elif count < 0:
            keep = slice(-count, None)
        else:
            keep = slice(None)

        trimmed_data = self[(slice(None),) * axis + (keep,)].data

        if fill_value is dtypes.NA:
            dtype, fill_value = dtypes.maybe_promote(self.dtype)
        else:
            dtype = self.dtype

        shape = list(self.shape)
        shape[axis] = min(abs(count), shape[axis])

        if isinstance(trimmed_data, dask_array_type):
            chunks = list(trimmed_data.chunks)
            chunks[axis] = (shape[axis],)
            full = functools.partial(da.full, chunks=chunks)
        else:
            full = np.full

        filler = full(shape, fill_value, dtype=dtype)

        if count > 0:
            arrays = [filler, trimmed_data]
        else:
            arrays = [trimmed_data, filler]

        data = duck_array_ops.concatenate(arrays, axis)

        if isinstance(data, dask_array_type):
            # chunked data should come out with the same chunks; this makes
            # it feasible to combine shifted and unshifted data
            # TODO: remove this once dask.array automatically aligns chunks
            data = data.rechunk(self.data.chunks)

        return type(self)(self.dims, data, self._attrs, fastpath=True)

    def shift(self, shifts=None, fill_value=dtypes.NA, **shifts_kwargs):
        """
        Return a new Variable with shifted data.

        Parameters
        ----------
        shifts : mapping of the form {dim: offset}
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.
        fill_value: scalar, optional
            Value to use for newly missing values
        **shifts_kwargs:
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwarg must be provided.

        Returns
        -------
        shifted : Variable
            Variable with the same dimensions and attributes but shifted data.
        """
        shifts = either_dict_or_kwargs(shifts, shifts_kwargs, "shift")
        result = self
        for dim, count in shifts.items():
            result = result._shift_one_dim(dim, count, fill_value=fill_value)
        return result

    def pad_with_fill_value(
        self, pad_widths=None, fill_value=dtypes.NA, **pad_widths_kwargs
    ):
        """
        Return a new Variable with paddings.

        Parameters
        ----------
        pad_width: Mapping of the form {dim: (before, after)}
            Number of values padded to the edges of each dimension.
        **pad_widths_kwargs:
            Keyword argument for pad_widths
        """
        pad_widths = either_dict_or_kwargs(pad_widths, pad_widths_kwargs, "pad")

        if fill_value is dtypes.NA:
            dtype, fill_value = dtypes.maybe_promote(self.dtype)
        else:
            dtype = self.dtype

        if isinstance(self.data, dask_array_type):
            array = self.data

            # Dask does not yet support pad. We manually implement it.
            # https://github.com/dask/dask/issues/1926
            for d, pad in pad_widths.items():
                axis = self.get_axis_num(d)
                before_shape = list(array.shape)
                before_shape[axis] = pad[0]
                before_chunks = list(array.chunks)
                before_chunks[axis] = (pad[0],)
                after_shape = list(array.shape)
                after_shape[axis] = pad[1]
                after_chunks = list(array.chunks)
                after_chunks[axis] = (pad[1],)

                arrays = []
                if pad[0] > 0:
                    arrays.append(
                        da.full(
                            before_shape, fill_value, dtype=dtype, chunks=before_chunks
                        )
                    )
                arrays.append(array)
                if pad[1] > 0:
                    arrays.append(
                        da.full(
                            after_shape, fill_value, dtype=dtype, chunks=after_chunks
                        )
                    )
                if len(arrays) > 1:
                    array = da.concatenate(arrays, axis=axis)
        else:
            pads = [(0, 0) if d not in pad_widths else pad_widths[d] for d in self.dims]
            array = np.pad(
                self.data.astype(dtype, copy=False),
                pads,
                mode="constant",
                constant_values=fill_value,
            )
        return type(self)(self.dims, array)

    def _roll_one_dim(self, dim, count):
        axis = self.get_axis_num(dim)

        count %= self.shape[axis]
        if count != 0:
            indices = [slice(-count, None), slice(None, -count)]
        else:
            indices = [slice(None)]

        arrays = [self[(slice(None),) * axis + (idx,)].data for idx in indices]

        data = duck_array_ops.concatenate(arrays, axis)

        if isinstance(data, dask_array_type):
            # chunked data should come out with the same chunks; this makes
            # it feasible to combine shifted and unshifted data
            # TODO: remove this once dask.array automatically aligns chunks
            data = data.rechunk(self.data.chunks)

        return type(self)(self.dims, data, self._attrs, fastpath=True)

    def roll(self, shifts=None, **shifts_kwargs):
        """
        Return a new Variable with rolld data.

        Parameters
        ----------
        shifts : mapping of the form {dim: offset}
            Integer offset to roll along each of the given dimensions.
            Positive offsets roll to the right; negative offsets roll to the
            left.
        **shifts_kwargs:
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwarg must be provided.

        Returns
        -------
        shifted : Variable
            Variable with the same dimensions and attributes but rolled data.
        """
        shifts = either_dict_or_kwargs(shifts, shifts_kwargs, "roll")

        result = self
        for dim, count in shifts.items():
            result = result._roll_one_dim(dim, count)
        return result

    def transpose(self, *dims) -> "Variable":
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
        This operation returns a view of this variable's data. It is
        lazy for dask-backed Variables but not for numpy-backed Variables.

        See Also
        --------
        numpy.transpose
        """
        if len(dims) == 0:
            dims = self.dims[::-1]
        axes = self.get_axis_num(dims)
        if len(dims) < 2:  # no need to transpose if only one dimension
            return self.copy(deep=False)

        data = as_indexable(self._data).transpose(axes)
        return type(self)(dims, data, self._attrs, self._encoding, fastpath=True)

    @property
    def T(self) -> "Variable":
        return self.transpose()

    def set_dims(self, dims, shape=None):
        """Return a new variable with given set of dimensions.
        This method might be used to attach new dimension(s) to variable.

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
        if isinstance(dims, str):
            dims = [dims]

        if shape is None and utils.is_dict_like(dims):
            shape = dims.values()

        missing_dims = set(self.dims) - set(dims)
        if missing_dims:
            raise ValueError(
                "new dimensions %r must be a superset of "
                "existing dimensions %r" % (dims, self.dims)
            )

        self_dims = set(self.dims)
        expanded_dims = tuple(d for d in dims if d not in self_dims) + self.dims

        if self.dims == expanded_dims:
            # don't use broadcast_to unless necessary so the result remains
            # writeable if possible
            expanded_data = self.data
        elif shape is not None:
            dims_map = dict(zip(dims, shape))
            tmp_shape = tuple(dims_map[d] for d in expanded_dims)
            expanded_data = duck_array_ops.broadcast_to(self.data, tmp_shape)
        else:
            expanded_data = self.data[(None,) * (len(expanded_dims) - self.ndim)]

        expanded_var = Variable(
            expanded_dims, expanded_data, self._attrs, self._encoding, fastpath=True
        )
        return expanded_var.transpose(*dims)

    def _stack_once(self, dims, new_dim):
        if not set(dims) <= set(self.dims):
            raise ValueError("invalid existing dimensions: %s" % dims)

        if new_dim in self.dims:
            raise ValueError(
                "cannot create a new dimension with the same "
                "name as an existing dimension"
            )

        if len(dims) == 0:
            # don't stack
            return self.copy(deep=False)

        other_dims = [d for d in self.dims if d not in dims]
        dim_order = other_dims + list(dims)
        reordered = self.transpose(*dim_order)

        new_shape = reordered.shape[: len(other_dims)] + (-1,)
        new_data = reordered.data.reshape(new_shape)
        new_dims = reordered.dims[: len(other_dims)] + (new_dim,)

        return Variable(new_dims, new_data, self._attrs, self._encoding, fastpath=True)

    def stack(self, dimensions=None, **dimensions_kwargs):
        """
        Stack any number of existing dimensions into a single new dimension.

        New dimensions will be added at the end, and the order of the data
        along each new dimension will be in contiguous (C) order.

        Parameters
        ----------
        dimensions : Mapping of form new_name=(dim1, dim2, ...)
            Names of new dimensions, and the existing dimensions that they
            replace.
        **dimensions_kwargs:
            The keyword arguments form of ``dimensions``.
            One of dimensions or dimensions_kwargs must be provided.

        Returns
        -------
        stacked : Variable
            Variable with the same attributes but stacked data.

        See also
        --------
        Variable.unstack
        """
        dimensions = either_dict_or_kwargs(dimensions, dimensions_kwargs, "stack")
        result = self
        for new_dim, dims in dimensions.items():
            result = result._stack_once(dims, new_dim)
        return result

    def _unstack_once(self, dims, old_dim):
        new_dim_names = tuple(dims.keys())
        new_dim_sizes = tuple(dims.values())

        if old_dim not in self.dims:
            raise ValueError("invalid existing dimension: %s" % old_dim)

        if set(new_dim_names).intersection(self.dims):
            raise ValueError(
                "cannot create a new dimension with the same "
                "name as an existing dimension"
            )

        if np.prod(new_dim_sizes) != self.sizes[old_dim]:
            raise ValueError(
                "the product of the new dimension sizes must "
                "equal the size of the old dimension"
            )

        other_dims = [d for d in self.dims if d != old_dim]
        dim_order = other_dims + [old_dim]
        reordered = self.transpose(*dim_order)

        new_shape = reordered.shape[: len(other_dims)] + new_dim_sizes
        new_data = reordered.data.reshape(new_shape)
        new_dims = reordered.dims[: len(other_dims)] + new_dim_names

        return Variable(new_dims, new_data, self._attrs, self._encoding, fastpath=True)

    def unstack(self, dimensions=None, **dimensions_kwargs):
        """
        Unstack an existing dimension into multiple new dimensions.

        New dimensions will be added at the end, and the order of the data
        along each new dimension will be in contiguous (C) order.

        Parameters
        ----------
        dimensions : mapping of the form old_dim={dim1: size1, ...}
            Names of existing dimensions, and the new dimensions and sizes
            that they map to.
        **dimensions_kwargs:
            The keyword arguments form of ``dimensions``.
            One of dimensions or dimensions_kwargs must be provided.

        Returns
        -------
        unstacked : Variable
            Variable with the same attributes but unstacked data.

        See also
        --------
        Variable.stack
        """
        dimensions = either_dict_or_kwargs(dimensions, dimensions_kwargs, "unstack")
        result = self
        for old_dim, dims in dimensions.items():
            result = result._unstack_once(dims, old_dim)
        return result

    def fillna(self, value):
        return ops.fillna(self, value)

    def where(self, cond, other=dtypes.NA):
        return ops.where_method(self, cond, other)

    def reduce(
        self,
        func,
        dim=None,
        axis=None,
        keep_attrs=None,
        keepdims=False,
        allow_lazy=False,
        **kwargs
    ):
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
        keepdims : bool, default False
            If True, the dimensions which are reduced are left in the result
            as dimensions of size one
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is common.ALL_DIMS:
            dim = None
        if dim is not None and axis is not None:
            raise ValueError("cannot supply both 'axis' and 'dim' arguments")

        if dim is not None:
            axis = self.get_axis_num(dim)
        input_data = self.data if allow_lazy else self.values
        if axis is not None:
            data = func(input_data, axis=axis, **kwargs)
        else:
            data = func(input_data, **kwargs)

        if getattr(data, "shape", ()) == self.shape:
            dims = self.dims
        else:
            removed_axes = (
                range(self.ndim) if axis is None else np.atleast_1d(axis) % self.ndim
            )
            if keepdims:
                # Insert np.newaxis for removed dims
                slices = tuple(
                    np.newaxis if i in removed_axes else slice(None, None)
                    for i in range(self.ndim)
                )
                if getattr(data, "shape", None) is None:
                    # Reduce has produced a scalar value, not an array-like
                    data = np.asanyarray(data)[slices]
                else:
                    data = data[slices]
                dims = self.dims
            else:
                dims = [
                    adim for n, adim in enumerate(self.dims) if n not in removed_axes
                ]

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self._attrs if keep_attrs else None

        return Variable(dims, data, attrs=attrs)

    @classmethod
    def concat(cls, variables, dim="concat_dim", positions=None, shortcut=False):
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
            List of integer arrays which specifies the integer positions to
            which to assign each dataset along the concatenated dimension.
            If not supplied, objects are concatenated in the provided order.
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
        if not isinstance(dim, str):
            dim, = dim.dims

        # can't do this lazily: we need to loop through variables at least
        # twice
        variables = list(variables)
        first_var = variables[0]

        arrays = [v.data for v in variables]

        if dim in first_var.dims:
            axis = first_var.get_axis_num(dim)
            dims = first_var.dims
            data = duck_array_ops.concatenate(arrays, axis=axis)
            if positions is not None:
                # TODO: deprecate this option -- we don't need it for groupby
                # any more.
                indices = nputils.inverse_permutation(np.concatenate(positions))
                data = duck_array_ops.take(data, indices, axis=axis)
        else:
            axis = 0
            dims = (dim,) + first_var.dims
            data = duck_array_ops.stack(arrays, axis=axis)

        attrs = OrderedDict(first_var.attrs)
        encoding = OrderedDict(first_var.encoding)
        if not shortcut:
            for var in variables:
                if var.dims != first_var.dims:
                    raise ValueError("inconsistent dimensions")
                utils.remove_incompatible_items(attrs, var.attrs)

        return cls(dims, data, attrs, encoding)

    def equals(self, other, equiv=duck_array_ops.array_equiv):
        """True if two Variables have the same dimensions and values;
        otherwise False.

        Variables can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for Variables
        does element-wise comparisons (like numpy.ndarrays).
        """
        other = getattr(other, "variable", other)
        try:
            return self.dims == other.dims and (
                self._data is other._data or equiv(self.data, other.data)
            )
        except (TypeError, AttributeError):
            return False

    def broadcast_equals(self, other, equiv=duck_array_ops.array_equiv):
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
            return utils.dict_equiv(self.attrs, other.attrs) and self.equals(other)
        except (TypeError, AttributeError):
            return False

    def no_conflicts(self, other):
        """True if the intersection of two Variable's non-null data is
        equal; otherwise false.

        Variables can thus still be equal if there are locations where either,
        or both, contain NaN values.
        """
        return self.broadcast_equals(other, equiv=duck_array_ops.array_notnull_equiv)

    def quantile(self, q, dim=None, interpolation="linear", keep_attrs=None):
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
        keep_attrs : bool, optional
            If True, the variable's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

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
            raise TypeError(
                "quantile does not work for arrays stored as dask "
                "arrays. Load the data via .compute() or .load() "
                "prior to calling this method."
            )

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

        # Only add the quantile dimension if q is array-like
        if q.ndim != 0:
            new_dims = ["quantile"] + new_dims

        qs = np.nanpercentile(
            self.data, q * 100.0, axis=axis, interpolation=interpolation
        )

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self._attrs if keep_attrs else None

        return Variable(new_dims, qs, attrs)

    def rank(self, dim, pct=False):
        """Ranks the data.

        Equal values are assigned a rank that is the average of the ranks that
        would have been otherwise assigned to all of the values within that
        set.  Ranks begin at 1, not 0. If `pct`, computes percentage ranks.

        NaNs in the input array are returned as NaNs.

        The `bottleneck` library is required.

        Parameters
        ----------
        dim : str
            Dimension over which to compute rank.
        pct : bool, optional
            If True, compute percentage ranks, otherwise compute integer ranks.

        Returns
        -------
        ranked : Variable

        See Also
        --------
        Dataset.rank, DataArray.rank
        """
        import bottleneck as bn

        data = self.data

        if isinstance(data, dask_array_type):
            raise TypeError(
                "rank does not work for arrays stored as dask "
                "arrays. Load the data via .compute() or .load() "
                "prior to calling this method."
            )
        elif not isinstance(data, np.ndarray):
            raise TypeError(
                "rank is not implemented for {} objects.".format(type(data))
            )

        axis = self.get_axis_num(dim)
        func = bn.nanrankdata if self.dtype.kind == "f" else bn.rankdata
        ranked = func(data, axis=axis)
        if pct:
            count = np.sum(~np.isnan(data), axis=axis, keepdims=True)
            ranked /= count
        return Variable(self.dims, ranked)

    def rolling_window(
        self, dim, window, window_dim, center=False, fill_value=dtypes.NA
    ):
        """
        Make a rolling_window along dim and add a new_dim to the last place.

        Parameters
        ----------
        dim: str
            Dimension over which to compute rolling_window
        window: int
            Window size of the rolling
        window_dim: str
            New name of the window dimension.
        center: boolean. default False.
            If True, pad fill_value for both ends. Otherwise, pad in the head
            of the axis.
        fill_value:
            value to be filled.

        Returns
        -------
        Variable that is a view of the original array with a added dimension of
        size w.
        The return dim: self.dims + (window_dim, )
        The return shape: self.shape + (window, )

        Examples
        --------
        >>> v=Variable(('a', 'b'), np.arange(8).reshape((2,4)))
        >>> v.rolling_window(x, 'b', 3, 'window_dim')
        <xarray.Variable (a: 2, b: 4, window_dim: 3)>
        array([[[nan, nan, 0], [nan, 0, 1], [0, 1, 2], [1, 2, 3]],
               [[nan, nan, 4], [nan, 4, 5], [4, 5, 6], [5, 6, 7]]])

        >>> v.rolling_window(x, 'b', 3, 'window_dim', center=True)
        <xarray.Variable (a: 2, b: 4, window_dim: 3)>
        array([[[nan, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, nan]],
               [[nan, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, nan]]])
        """
        if fill_value is dtypes.NA:  # np.nan is passed
            dtype, fill_value = dtypes.maybe_promote(self.dtype)
            array = self.astype(dtype, copy=False).data
        else:
            dtype = self.dtype
            array = self.data

        new_dims = self.dims + (window_dim,)
        return Variable(
            new_dims,
            duck_array_ops.rolling_window(
                array,
                axis=self.get_axis_num(dim),
                window=window,
                center=center,
                fill_value=fill_value,
            ),
        )

    def coarsen(self, windows, func, boundary="exact", side="left"):
        """
        Apply
        """
        windows = {k: v for k, v in windows.items() if k in self.dims}
        if not windows:
            return self.copy()

        reshaped, axes = self._coarsen_reshape(windows, boundary, side)
        if isinstance(func, str):
            name = func
            func = getattr(duck_array_ops, name, None)
            if func is None:
                raise NameError("{} is not a valid method.".format(name))
        return type(self)(self.dims, func(reshaped, axis=axes), self._attrs)

    def _coarsen_reshape(self, windows, boundary, side):
        """
        Construct a reshaped-array for corsen
        """
        if not utils.is_dict_like(boundary):
            boundary = {d: boundary for d in windows.keys()}

        if not utils.is_dict_like(side):
            side = {d: side for d in windows.keys()}

        # remove unrelated dimensions
        boundary = {k: v for k, v in boundary.items() if k in windows}
        side = {k: v for k, v in side.items() if k in windows}

        for d, window in windows.items():
            if window <= 0:
                raise ValueError("window must be > 0. Given {}".format(window))

        variable = self
        for d, window in windows.items():
            # trim or pad the object
            size = variable.shape[self._get_axis_num(d)]
            n = int(size / window)
            if boundary[d] == "exact":
                if n * window != size:
                    raise ValueError(
                        "Could not coarsen a dimension of size {} with "
                        "window {}".format(size, window)
                    )
            elif boundary[d] == "trim":
                if side[d] == "left":
                    variable = variable.isel({d: slice(0, window * n)})
                else:
                    excess = size - window * n
                    variable = variable.isel({d: slice(excess, None)})
            elif boundary[d] == "pad":  # pad
                pad = window * n - size
                if pad < 0:
                    pad += window
                if side[d] == "left":
                    pad_widths = {d: (0, pad)}
                else:
                    pad_widths = {d: (pad, 0)}
                variable = variable.pad_with_fill_value(pad_widths)
            else:
                raise TypeError(
                    "{} is invalid for boundary. Valid option is 'exact', "
                    "'trim' and 'pad'".format(boundary[d])
                )

        shape = []
        axes = []
        axis_count = 0
        for i, d in enumerate(variable.dims):
            if d in windows:
                size = variable.shape[i]
                shape.append(int(size / windows[d]))
                shape.append(windows[d])
                axis_count += 1
                axes.append(i + axis_count)
            else:
                shape.append(variable.shape[i])

        return variable.data.reshape(shape), tuple(axes)

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
            with np.errstate(all="ignore"):
                return self.__array_wrap__(f(self.data, *args, **kwargs))

        return func

    @staticmethod
    def _binary_op(f, reflexive=False, **ignored_kwargs):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, (xr.DataArray, xr.Dataset)):
                return NotImplemented
            self_data, other_data, dims = _broadcast_compat_data(self, other)
            keep_attrs = _get_keep_attrs(default=False)
            attrs = self._attrs if keep_attrs else None
            with np.errstate(all="ignore"):
                new_data = (
                    f(self_data, other_data)
                    if not reflexive
                    else f(other_data, self_data)
                )
            result = Variable(dims, new_data, attrs=attrs)
            return result

        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, xr.Dataset):
                raise TypeError("cannot add a Dataset to a Variable in-place")
            self_data, other_data, dims = _broadcast_compat_data(self, other)
            if dims != self.dims:
                raise ValueError("dimensions cannot change for in-place " "operations")
            with np.errstate(all="ignore"):
                self.values = f(self_data, other_data)
            return self

        return func

    def _to_numeric(self, offset=None, datetime_unit=None, dtype=float):
        """ A (private) method to convert datetime array to numeric dtype
        See duck_array_ops.datetime_to_numeric
        """
        numeric_array = duck_array_ops.datetime_to_numeric(
            self.data, offset, datetime_unit, dtype
        )
        return type(self)(self.dims, numeric_array, self._attrs)


ops.inject_all_ops_and_reduce_methods(Variable)


class IndexVariable(Variable):
    """Wrapper for accommodating a pandas.Index in an xarray.Variable.

    IndexVariable preserve loaded values in the form of a pandas.Index instead
    of a NumPy array. Hence, their values are immutable and must always be one-
    dimensional.

    They also have a name property, which is the name of their sole dimension
    unless another name is given.
    """

    __slots__ = ()

    def __init__(self, dims, data, attrs=None, encoding=None, fastpath=False):
        super().__init__(dims, data, attrs, encoding, fastpath)
        if self.ndim != 1:
            raise ValueError("%s objects must be 1-dimensional" % type(self).__name__)

        # Unlike in Variable, always eagerly load values into memory
        if not isinstance(self._data, PandasIndexAdapter):
            self._data = PandasIndexAdapter(self._data)

    def load(self):
        # data is already loaded into memory for IndexVariable
        return self

    # https://github.com/python/mypy/issues/1465
    @Variable.data.setter  # type: ignore
    def data(self, data):
        Variable.data.fset(self, data)
        if not isinstance(self._data, PandasIndexAdapter):
            self._data = PandasIndexAdapter(self._data)

    def chunk(self, chunks=None, name=None, lock=False):
        # Dummy - do not chunk. This method is invoked e.g. by Dataset.chunk()
        return self.copy(deep=False)

    def _finalize_indexing_result(self, dims, data):
        if getattr(data, "ndim", 0) != 1:
            # returns Variable rather than IndexVariable if multi-dimensional
            return Variable(dims, data, self._attrs, self._encoding)
        else:
            return type(self)(dims, data, self._attrs, self._encoding, fastpath=True)

    def __setitem__(self, key, value):
        raise TypeError("%s values cannot be modified" % type(self).__name__)

    @classmethod
    def concat(cls, variables, dim="concat_dim", positions=None, shortcut=False):
        """Specialized version of Variable.concat for IndexVariable objects.

        This exists because we want to avoid converting Index objects to NumPy
        arrays, if possible.
        """
        if not isinstance(dim, str):
            dim, = dim.dims

        variables = list(variables)
        first_var = variables[0]

        if any(not isinstance(v, cls) for v in variables):
            raise TypeError(
                "IndexVariable.concat requires that all input "
                "variables be IndexVariable objects"
            )

        indexes = [v._data.array for v in variables]

        if not indexes:
            data = []
        else:
            data = indexes[0].append(indexes[1:])

            if positions is not None:
                indices = nputils.inverse_permutation(np.concatenate(positions))
                data = data.take(indices)

        attrs = OrderedDict(first_var.attrs)
        if not shortcut:
            for var in variables:
                if var.dims != first_var.dims:
                    raise ValueError("inconsistent dimensions")
                utils.remove_incompatible_items(attrs, var.attrs)

        return cls(first_var.dims, data, attrs)

    def copy(self, deep=True, data=None):
        """Returns a copy of this object.

        `deep` is ignored since data is stored in the form of
        pandas.Index, which is already immutable. Dimensions, attributes
        and encodings are always copied.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, optional
            Deep is ignored when data is given. Whether the data array is
            loaded into memory and copied onto the new object. Default is True.
        data : array_like, optional
            Data to use in the new object. Must have same shape as original.

        Returns
        -------
        object : Variable
            New object with dimensions, attributes, encodings, and optionally
            data copied from original.
        """
        if data is None:
            data = self._data.copy(deep=deep)
        else:
            data = as_compatible_data(data)
            if self.shape != data.shape:
                raise ValueError(
                    "Data shape {} must match shape of object {}".format(
                        data.shape, self.shape
                    )
                )
        return type(self)(self.dims, data, self._attrs, self._encoding, fastpath=True)

    def equals(self, other, equiv=None):
        # if equiv is specified, super up
        if equiv is not None:
            return super().equals(other, equiv)

        # otherwise use the native index equals, rather than looking at _data
        other = getattr(other, "variable", other)
        try:
            return self.dims == other.dims and self._data_equals(other)
        except (TypeError, AttributeError):
            return False

    def _data_equals(self, other):
        return self.to_index().equals(other.to_index())

    def to_index_variable(self):
        """Return this variable as an xarray.IndexVariable"""
        return self

    to_coord = utils.alias(to_index_variable, "to_coord")

    def to_index(self):
        """Convert this variable to a pandas.Index"""
        # n.b. creating a new pandas.Index from an old pandas.Index is
        # basically free as pandas.Index objects are immutable
        assert self.ndim == 1
        index = self._data.array
        if isinstance(index, pd.MultiIndex):
            # set default names for multi-index unnamed levels so that
            # we can safely rename dimension / coordinate later
            valid_level_names = [
                name or "{}_level_{}".format(self.dims[0], i)
                for i, name in enumerate(index.names)
            ]
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
        raise AttributeError("cannot modify name of IndexVariable in-place")


# for backwards compatibility
Coordinate = utils.alias(IndexVariable, "Coordinate")


def _unified_dims(variables):
    # validate dimensions
    all_dims = OrderedDict()
    for var in variables:
        var_dims = var.dims
        if len(set(var_dims)) < len(var_dims):
            raise ValueError(
                "broadcasting cannot handle duplicate "
                "dimensions: %r" % list(var_dims)
            )
        for d, s in zip(var_dims, var.shape):
            if d not in all_dims:
                all_dims[d] = s
            elif all_dims[d] != s:
                raise ValueError(
                    "operands cannot be broadcast together "
                    "with mismatched lengths for dimension %r: %s"
                    % (d, (all_dims[d], s))
                )
    return all_dims


def _broadcast_compat_variables(*variables):
    """Create broadcast compatible variables, with the same dimensions.

    Unlike the result of broadcast_variables(), some variables may have
    dimensions of size 1 instead of the the size of the broadcast dimension.
    """
    dims = tuple(_unified_dims(variables))
    return tuple(var.set_dims(dims) if var.dims != dims else var for var in variables)


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
    return tuple(
        var.set_dims(dims_map) if var.dims != dims_tuple else var for var in variables
    )


def _broadcast_compat_data(self, other):
    if all(hasattr(other, attr) for attr in ["dims", "data", "shape", "encoding"]):
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


def concat(variables, dim="concat_dim", positions=None, shortcut=False):
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
    all_level_names = set()
    for var_name, var in variables.items():
        if isinstance(var._data, PandasIndexAdapter):
            idx_level_names = var.to_index_variable().level_names
            if idx_level_names is not None:
                for n in idx_level_names:
                    level_names[n].append("%r (%s)" % (n, var_name))
            if idx_level_names:
                all_level_names.update(idx_level_names)

    for k, v in level_names.items():
        if k in variables:
            v.append("(%s)" % k)

    duplicate_names = [v for v in level_names.values() if len(v) > 1]
    if duplicate_names:
        conflict_str = "\n".join([", ".join(v) for v in duplicate_names])
        raise ValueError("conflicting MultiIndex level name(s):\n%s" % conflict_str)
    # Check confliction between level names and dimensions GH:2299
    for k, v in variables.items():
        for d in v.dims:
            if d in all_level_names:
                raise ValueError(
                    "conflicting level / dimension names. {} "
                    "already exists as a level name.".format(d)
                )
