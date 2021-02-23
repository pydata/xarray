import copy
import functools
import itertools
import numbers
import warnings
from collections import defaultdict
from datetime import timedelta
from distutils.version import LooseVersion
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

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
from .pycompat import (
    cupy_array_type,
    dask_array_type,
    integer_types,
    is_duck_dask_array,
)
from .utils import (
    OrderedSet,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    ensure_us_time_resolution,
    infix_dims,
    is_duck_array,
    maybe_coerce_to_str,
)

NON_NUMPY_SUPPORTED_ARRAY_TYPES = (
    (
        indexing.ExplicitlyIndexed,
        pd.Index,
    )
    + dask_array_type
    + cupy_array_type
)
# https://github.com/python/mypy/issues/224
BASIC_INDEXING_TYPES = integer_types + (slice,)  # type: ignore

VariableType = TypeVar("VariableType", bound="Variable")
"""Type annotation to be used when methods of Variable return self or a copy of self.
When called from an instance of a subclass, e.g. IndexVariable, mypy identifies the
output as an instance of the subclass.

Usage::

   class Variable:
       def f(self: VariableType, ...) -> VariableType:
           ...
"""


class MissingDimensionsError(ValueError):
    """Error class used when we can't safely guess a dimension name."""

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
        if isinstance(obj[1], DataArray):
            # TODO: change into TypeError
            warnings.warn(
                (
                    "Using a DataArray object to construct a variable is"
                    " ambiguous, please extract the data using the .data property."
                    " This will raise a TypeError in 0.19.0."
                ),
                DeprecationWarning,
            )
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
        raise TypeError("variable {!r} has invalid type {!r}".format(name, type(obj)))
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
    datetime64 and timedelta64, according to the pandas convention. Also used for
    validating that datetime64 and timedelta64 objects are within the valid date
    range for ns precision, as pandas will raise an error if they are not.
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
    if isinstance(data, (pd.Series, pd.Index, pd.DataFrame)):
        data = data.values

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

    # validate whether the data is valid data types.
    data = np.asarray(data)

    if isinstance(data, np.ndarray):
        if data.dtype.kind == "O":
            data = _possibly_convert_objects(data)
        elif data.dtype.kind == "M":
            data = _possibly_convert_objects(data)
        elif data.dtype.kind == "m":
            data = _possibly_convert_objects(data)

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
    if isinstance(data, cupy_array_type):
        data = data.get()
    else:
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
        if is_duck_array(self._data):
            return self._data
        else:
            return self.values

    @data.setter
    def data(self, data):
        data = as_compatible_data(data)
        if data.shape != self.shape:
            raise ValueError(
                f"replacement data must match the Variable's shape. "
                f"replacement data has shape {data.shape}; Variable has shape {self.shape}"
            )
        self._data = data

    def astype(
        self: VariableType,
        dtype,
        *,
        order=None,
        casting=None,
        subok=None,
        copy=None,
        keep_attrs=True,
    ) -> VariableType:
        """
        Copy of the Variable object, with data cast to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        order : {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout order of the result. ‘C’ means C order,
            ‘F’ means Fortran order, ‘A’ means ‘F’ order if all the arrays are
            Fortran contiguous, ‘C’ order otherwise, and ‘K’ means as close to
            the order the array elements appear in memory as possible.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Controls what kind of data casting may occur.

            * 'no' means the data types should not be cast at all.
            * 'equiv' means only byte-order changes are allowed.
            * 'safe' means only casts which can preserve values are allowed.
            * 'same_kind' means only safe casts or casts within a kind,
              like float64 to float32, are allowed.
            * 'unsafe' means any data conversions may be done.
        subok : bool, optional
            If True, then sub-classes will be passed-through, otherwise the
            returned array will be forced to be a base-class array.
        copy : bool, optional
            By default, astype always returns a newly allocated array. If this
            is set to False and the `dtype` requirement is satisfied, the input
            array is returned instead of a copy.
        keep_attrs : bool, optional
            By default, astype keeps attributes. Set to False to remove
            attributes in the returned object.

        Returns
        -------
        out : same as object
            New object with data cast to the specified type.

        Notes
        -----
        The ``order``, ``casting``, ``subok`` and ``copy`` arguments are only passed
        through to the ``astype`` method of the underlying array when a value
        different than ``None`` is supplied.
        Make sure to only supply these arguments if the underlying array class
        supports them.

        See Also
        --------
        numpy.ndarray.astype
        dask.array.Array.astype
        sparse.COO.astype
        """
        from .computation import apply_ufunc

        kwargs = dict(order=order, casting=casting, subok=subok, copy=copy)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return apply_ufunc(
            duck_array_ops.astype,
            self,
            dtype,
            kwargs=kwargs,
            keep_attrs=keep_attrs,
            dask="allowed",
        )

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
        if is_duck_dask_array(self._data):
            self._data = as_compatible_data(self._data.compute(**kwargs))
        elif not is_duck_array(self._data):
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

    def __dask_tokenize__(self):
        # Use v.data, instead of v._data, in order to cope with the wrappers
        # around NetCDF and the like
        from dask.base import normalize_token

        return normalize_token((type(self), self._dims, self.data, self._attrs))

    def __dask_graph__(self):
        if is_duck_dask_array(self._data):
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
        """Tuple of dimension names with which this variable is associated."""
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
        ----------
        key : int, slice, array-like, dict or tuple of integer, slice and array-like
            Any valid input for indexing.

        Returns
        -------
        dims : tuple
            Dimension of the resultant variable.
        indexers : IndexingTuple subclass
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
                if k.size == 0:
                    # Slice by empty list; numpy could not infer the dtype
                    k = k.astype(int)
                elif k.dtype.kind == "b":
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
            raise IndexError(f"Dimensions of indexers mismatch: {key}")

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

    def __getitem__(self: VariableType, key) -> VariableType:
        """Return a new Variable object whose contents are consistent with
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

    def _finalize_indexing_result(self: VariableType, dims, data) -> VariableType:
        """Used by IndexVariable to return IndexVariable objects when possible."""
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
            if is_duck_dask_array(self._data):
                # dask's indexing is faster this way; also vindex does not
                # support negative indices yet:
                # https://github.com/dask/dask/pull/2967
                actual_indexer = indexing.posify_mask_indexer(indexer)
            else:
                actual_indexer = indexer

            data = as_indexable(self._data)[actual_indexer]
            mask = indexing.create_mask(indexer, self.shape, data)
            # we need to invert the mask in order to pass data first. This helps
            # pint to choose the correct unit
            # TODO: revert after https://github.com/hgrecco/pint/issues/1019 is fixed
            data = duck_array_ops.where(np.logical_not(mask), data, fill_value)
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
    def attrs(self) -> Dict[Hashable, Any]:
        """Dictionary of local attributes on this variable."""
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        self._attrs = dict(value)

    @property
    def encoding(self):
        """Dictionary of encodings on this variable."""
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

        >>> var = xr.Variable(data=[1, 2, 3], dims="x")
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
        array([0.1, 0.2, 0.3])
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
                data = copy.deepcopy(data)

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
        return self._replace(data=data)

    def _replace(
        self, dims=_default, data=_default, attrs=_default, encoding=_default
    ) -> "Variable":
        if dims is _default:
            dims = copy.copy(self._dims)
        if data is _default:
            data = copy.copy(self.data)
        if attrs is _default:
            attrs = copy.copy(self._attrs)
        if encoding is _default:
            encoding = copy.copy(self._encoding)
        return type(self)(dims, data, attrs, encoding, fastpath=True)

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

    def chunk(self, chunks={}, name=None, lock=False):
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

        if chunks is None:
            warnings.warn(
                "None value for 'chunks' is deprecated. "
                "It will raise an error in the future. Use instead '{}'",
                category=FutureWarning,
            )
            chunks = {}

        if utils.is_dict_like(chunks):
            chunks = {self.get_axis_num(dim): chunk for dim, chunk in chunks.items()}

        data = self._data
        if is_duck_dask_array(data):
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

    def _as_sparse(self, sparse_format=_default, fill_value=dtypes.NA):
        """
        use sparse-array as backend.
        """
        import sparse

        # TODO: what to do if dask-backended?
        if fill_value is dtypes.NA:
            dtype, fill_value = dtypes.maybe_promote(self.dtype)
        else:
            dtype = dtypes.result_type(self.dtype, fill_value)

        if sparse_format is _default:
            sparse_format = "coo"
        try:
            as_sparse = getattr(sparse, f"as_{sparse_format.lower()}")
        except AttributeError:
            raise ValueError(f"{sparse_format} is not a valid sparse format")

        data = as_sparse(self.data.astype(dtype), fill_value=fill_value)
        return self._replace(data=data)

    def _to_dense(self):
        """
        Change backend from sparse to np.array
        """
        if hasattr(self._data, "todense"):
            return self._replace(data=self._data.todense())
        return self.copy(deep=False)

    def isel(
        self: VariableType,
        indexers: Mapping[Hashable, Any] = None,
        missing_dims: str = "raise",
        **indexers_kwargs: Any,
    ) -> VariableType:
        """Return a new array indexed along the specified dimension(s).

        Parameters
        ----------
        **indexers : {dim: indexer, ...}
            Keyword arguments with names matching dimensions and values given
            by integers, slice objects or arrays.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            DataArray:
            - "raise": raise an exception
            - "warning": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions

        Returns
        -------
        obj : Array object
            A new Array with the selected data and dimensions. In general,
            the new variable's data will be a view of this variable's data,
            unless numpy fancy indexing was triggered by using an array
            indexer, in which case the data will be a copy.
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")

        indexers = drop_dims_from_indexers(indexers, self.dims, missing_dims)

        key = tuple(indexers.get(dim, slice(None)) for dim in self.dims)
        return self[key]

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

        width = min(abs(count), self.shape[axis])
        dim_pad = (width, 0) if count >= 0 else (0, width)
        pads = [(0, 0) if d != dim else dim_pad for d in self.dims]

        data = duck_array_ops.pad(
            trimmed_data.astype(dtype),
            pads,
            mode="constant",
            constant_values=fill_value,
        )

        if is_duck_dask_array(data):
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
        fill_value : scalar, optional
            Value to use for newly missing values
        **shifts_kwargs
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

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

    def _pad_options_dim_to_index(
        self,
        pad_option: Mapping[Hashable, Union[int, Tuple[int, int]]],
        fill_with_shape=False,
    ):
        if fill_with_shape:
            return [
                (n, n) if d not in pad_option else pad_option[d]
                for d, n in zip(self.dims, self.data.shape)
            ]
        return [(0, 0) if d not in pad_option else pad_option[d] for d in self.dims]

    def pad(
        self,
        pad_width: Mapping[Hashable, Union[int, Tuple[int, int]]] = None,
        mode: str = "constant",
        stat_length: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        constant_values: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        end_values: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        reflect_type: str = None,
        **pad_width_kwargs: Any,
    ):
        """
        Return a new Variable with padded data.

        Parameters
        ----------
        pad_width : mapping of hashable to tuple of int
            Mapping with the form of {dim: (pad_before, pad_after)}
            describing the number of values padded along each dimension.
            {dim: pad} is a shortcut for pad_before = pad_after = pad
        mode : str, default: "constant"
            See numpy / Dask docs
        stat_length : int, tuple or mapping of hashable to tuple
            Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
            values at edge of each axis used to calculate the statistic value.
        constant_values : scalar, tuple or mapping of hashable to tuple
            Used in 'constant'.  The values to set the padded values for each
            axis.
        end_values : scalar, tuple or mapping of hashable to tuple
            Used in 'linear_ramp'.  The values used for the ending value of the
            linear_ramp and that will form the edge of the padded array.
        reflect_type : {"even", "odd"}, optional
            Used in "reflect", and "symmetric".  The "even" style is the
            default with an unaltered reflection around the edge value.  For
            the "odd" style, the extended part of the array is created by
            subtracting the reflected values from two times the edge value.
        **pad_width_kwargs
            One of pad_width or pad_width_kwargs must be provided.

        Returns
        -------
        padded : Variable
            Variable with the same dimensions and attributes but padded data.
        """
        pad_width = either_dict_or_kwargs(pad_width, pad_width_kwargs, "pad")

        # change default behaviour of pad with mode constant
        if mode == "constant" and (
            constant_values is None or constant_values is dtypes.NA
        ):
            dtype, constant_values = dtypes.maybe_promote(self.dtype)
        else:
            dtype = self.dtype

        # create pad_options_kwargs, numpy requires only relevant kwargs to be nonempty
        if isinstance(stat_length, dict):
            stat_length = self._pad_options_dim_to_index(
                stat_length, fill_with_shape=True
            )
        if isinstance(constant_values, dict):
            constant_values = self._pad_options_dim_to_index(constant_values)
        if isinstance(end_values, dict):
            end_values = self._pad_options_dim_to_index(end_values)

        # workaround for bug in Dask's default value of stat_length https://github.com/dask/dask/issues/5303
        if stat_length is None and mode in ["maximum", "mean", "median", "minimum"]:
            stat_length = [(n, n) for n in self.data.shape]  # type: ignore

        # change integer values to a tuple of two of those values and change pad_width to index
        for k, v in pad_width.items():
            if isinstance(v, numbers.Number):
                pad_width[k] = (v, v)
        pad_width_by_index = self._pad_options_dim_to_index(pad_width)

        # create pad_options_kwargs, numpy/dask requires only relevant kwargs to be nonempty
        pad_option_kwargs = {}
        if stat_length is not None:
            pad_option_kwargs["stat_length"] = stat_length
        if constant_values is not None:
            pad_option_kwargs["constant_values"] = constant_values
        if end_values is not None:
            pad_option_kwargs["end_values"] = end_values
        if reflect_type is not None:
            pad_option_kwargs["reflect_type"] = reflect_type  # type: ignore

        array = duck_array_ops.pad(
            self.data.astype(dtype, copy=False),
            pad_width_by_index,
            mode=mode,
            **pad_option_kwargs,
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

        if is_duck_dask_array(data):
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
        shifts : mapping of hashable to int
            Integer offset to roll along each of the given dimensions.
            Positive offsets roll to the right; negative offsets roll to the
            left.
        **shifts_kwargs
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

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
        dims = tuple(infix_dims(dims, self.dims))
        axes = self.get_axis_num(dims)
        if len(dims) < 2 or dims == self.dims:
            # no need to transpose if only one dimension
            # or dims are in same order
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

    def _stack_once(self, dims: List[Hashable], new_dim: Hashable):
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
        dimensions : mapping of hashable to tuple of hashable
            Mapping of form new_name=(dim1, dim2, ...) describing the
            names of new dimensions, and the existing dimensions that
            they replace.
        **dimensions_kwargs
            The keyword arguments form of ``dimensions``.
            One of dimensions or dimensions_kwargs must be provided.

        Returns
        -------
        stacked : Variable
            Variable with the same attributes but stacked data.

        See Also
        --------
        Variable.unstack
        """
        dimensions = either_dict_or_kwargs(dimensions, dimensions_kwargs, "stack")
        result = self
        for new_dim, dims in dimensions.items():
            result = result._stack_once(dims, new_dim)
        return result

    def _unstack_once_full(
        self, dims: Mapping[Hashable, int], old_dim: Hashable
    ) -> "Variable":
        """
        Unstacks the variable without needing an index.

        Unlike `_unstack_once`, this function requires the existing dimension to
        contain the full product of the new dimensions.
        """
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

    def _unstack_once(
        self,
        index: pd.MultiIndex,
        dim: Hashable,
        fill_value=dtypes.NA,
    ) -> "Variable":
        """
        Unstacks this variable given an index to unstack and the name of the
        dimension to which the index refers.
        """

        reordered = self.transpose(..., dim)

        new_dim_sizes = [lev.size for lev in index.levels]
        new_dim_names = index.names
        indexer = index.codes

        # Potentially we could replace `len(other_dims)` with just `-1`
        other_dims = [d for d in self.dims if d != dim]
        new_shape = list(reordered.shape[: len(other_dims)]) + new_dim_sizes
        new_dims = reordered.dims[: len(other_dims)] + new_dim_names

        if fill_value is dtypes.NA:
            is_missing_values = np.prod(new_shape) > np.prod(self.shape)
            if is_missing_values:
                dtype, fill_value = dtypes.maybe_promote(self.dtype)
            else:
                dtype = self.dtype
                fill_value = dtypes.get_fill_value(dtype)
        else:
            dtype = self.dtype

        # Currently fails on sparse due to https://github.com/pydata/sparse/issues/422
        data = np.full_like(
            self.data,
            fill_value=fill_value,
            shape=new_shape,
            dtype=dtype,
        )

        # Indexer is a list of lists of locations. Each list is the locations
        # on the new dimension. This is robust to the data being sparse; in that
        # case the destinations will be NaN / zero.
        data[(..., *indexer)] = reordered

        return self._replace(dims=new_dims, data=data)

    def unstack(self, dimensions=None, **dimensions_kwargs):
        """
        Unstack an existing dimension into multiple new dimensions.

        New dimensions will be added at the end, and the order of the data
        along each new dimension will be in contiguous (C) order.

        Note that unlike ``DataArray.unstack`` and ``Dataset.unstack``, this
        method requires the existing dimension to contain the full product of
        the new dimensions.

        Parameters
        ----------
        dimensions : mapping of hashable to mapping of hashable to int
            Mapping of the form old_dim={dim1: size1, ...} describing the
            names of existing dimensions, and the new dimensions and sizes
            that they map to.
        **dimensions_kwargs
            The keyword arguments form of ``dimensions``.
            One of dimensions or dimensions_kwargs must be provided.

        Returns
        -------
        unstacked : Variable
            Variable with the same attributes but unstacked data.

        See Also
        --------
        Variable.stack
        DataArray.unstack
        Dataset.unstack
        """
        dimensions = either_dict_or_kwargs(dimensions, dimensions_kwargs, "unstack")
        result = self
        for old_dim, dims in dimensions.items():
            result = result._unstack_once_full(dims, old_dim)
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
        **kwargs,
    ):
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : callable
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
        keepdims : bool, default: False
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
        if dim == ...:
            dim = None
        if dim is not None and axis is not None:
            raise ValueError("cannot supply both 'axis' and 'dim' arguments")

        if dim is not None:
            axis = self.get_axis_num(dim)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"Mean of empty slice", category=RuntimeWarning
            )
            if axis is not None:
                data = func(self.data, axis=axis, **kwargs)
            else:
                data = func(self.data, **kwargs)

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
        variables : iterable of Variable
            Arrays to stack together. Each variable is expected to have
            matching dimensions and shape except for along the stacked
            dimension.
        dim : str or DataArray, optional
            Name of the dimension to stack along. This can either be a new
            dimension name, in which case it is added along axis=0, or an
            existing dimension name, in which case the location of the
            dimension is unchanged. Where to insert the new dimension is
            determined by the first variable.
        positions : None or list of array-like, optional
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
            (dim,) = dim.dims

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

        attrs = dict(first_var.attrs)
        encoding = dict(first_var.encoding)
        if not shortcut:
            for var in variables:
                if var.dims != first_var.dims:
                    raise ValueError(
                        f"Variable has dimensions {list(var.dims)} but first Variable has dimensions {list(first_var.dims)}"
                    )

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

    def identical(self, other, equiv=duck_array_ops.array_equiv):
        """Like equals, but also checks attributes."""
        try:
            return utils.dict_equiv(self.attrs, other.attrs) and self.equals(
                other, equiv=equiv
            )
        except (TypeError, AttributeError):
            return False

    def no_conflicts(self, other, equiv=duck_array_ops.array_notnull_equiv):
        """True if the intersection of two Variable's non-null data is
        equal; otherwise false.

        Variables can thus still be equal if there are locations where either,
        or both, contain NaN values.
        """
        return self.broadcast_equals(other, equiv=equiv)

    def quantile(
        self, q, dim=None, interpolation="linear", keep_attrs=None, skipna=True
    ):
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements.

        Parameters
        ----------
        q : float or sequence of float
            Quantile to compute, which must be between 0 and 1
            inclusive.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply quantile.
        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}, default: "linear"
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
        numpy.nanquantile, pandas.Series.quantile, Dataset.quantile
        DataArray.quantile
        """

        from .computation import apply_ufunc

        _quantile_func = np.nanquantile if skipna else np.quantile

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)

        scalar = utils.is_scalar(q)
        q = np.atleast_1d(np.asarray(q, dtype=np.float64))

        if dim is None:
            dim = self.dims

        if utils.is_scalar(dim):
            dim = [dim]

        def _wrapper(npa, **kwargs):
            # move quantile axis to end. required for apply_ufunc
            return np.moveaxis(_quantile_func(npa, **kwargs), 0, -1)

        axis = np.arange(-1, -1 * len(dim) - 1, -1)
        result = apply_ufunc(
            _wrapper,
            self,
            input_core_dims=[dim],
            exclude_dims=set(dim),
            output_core_dims=[["quantile"]],
            output_dtypes=[np.float64],
            dask_gufunc_kwargs=dict(output_sizes={"quantile": len(q)}),
            dask="parallelized",
            kwargs={"q": q, "axis": axis, "interpolation": interpolation},
        )

        # for backward compatibility
        result = result.transpose("quantile", ...)
        if scalar:
            result = result.squeeze("quantile")
        if keep_attrs:
            result.attrs = self._attrs
        return result

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

        if is_duck_dask_array(data):
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
        dim : str
            Dimension over which to compute rolling_window.
            For nd-rolling, should be list of dimensions.
        window : int
            Window size of the rolling
            For nd-rolling, should be list of integers.
        window_dim : str
            New name of the window dimension.
            For nd-rolling, should be list of integers.
        center : bool, default: False
            If True, pad fill_value for both ends. Otherwise, pad in the head
            of the axis.
        fill_value
            value to be filled.

        Returns
        -------
        Variable that is a view of the original array with a added dimension of
        size w.
        The return dim: self.dims + (window_dim, )
        The return shape: self.shape + (window, )

        Examples
        --------
        >>> v = Variable(("a", "b"), np.arange(8).reshape((2, 4)))
        >>> v.rolling_window("b", 3, "window_dim")
        <xarray.Variable (a: 2, b: 4, window_dim: 3)>
        array([[[nan, nan,  0.],
                [nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.]],
        <BLANKLINE>
               [[nan, nan,  4.],
                [nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.]]])

        >>> v.rolling_window("b", 3, "window_dim", center=True)
        <xarray.Variable (a: 2, b: 4, window_dim: 3)>
        array([[[nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.],
                [ 2.,  3., nan]],
        <BLANKLINE>
               [[nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.],
                [ 6.,  7., nan]]])
        """
        if fill_value is dtypes.NA:  # np.nan is passed
            dtype, fill_value = dtypes.maybe_promote(self.dtype)
            array = self.astype(dtype, copy=False).data
        else:
            dtype = self.dtype
            array = self.data

        if isinstance(dim, list):
            assert len(dim) == len(window)
            assert len(dim) == len(window_dim)
            assert len(dim) == len(center)
        else:
            dim = [dim]
            window = [window]
            window_dim = [window_dim]
            center = [center]
        axis = [self.get_axis_num(d) for d in dim]
        new_dims = self.dims + tuple(window_dim)
        return Variable(
            new_dims,
            duck_array_ops.rolling_window(
                array, axis=axis, window=window, center=center, fill_value=fill_value
            ),
        )

    def coarsen(
        self, windows, func, boundary="exact", side="left", keep_attrs=None, **kwargs
    ):
        """
        Apply reduction function.
        """
        windows = {k: v for k, v in windows.items() if k in self.dims}
        if not windows:
            return self.copy()

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)

        if keep_attrs:
            _attrs = self.attrs
        else:
            _attrs = None

        reshaped, axes = self._coarsen_reshape(windows, boundary, side)
        if isinstance(func, str):
            name = func
            func = getattr(duck_array_ops, name, None)
            if func is None:
                raise NameError(f"{name} is not a valid method.")

        return self._replace(data=func(reshaped, axis=axes, **kwargs), attrs=_attrs)

    def _coarsen_reshape(self, windows, boundary, side):
        """
        Construct a reshaped-array for coarsen
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
                raise ValueError(f"window must be > 0. Given {window}")

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
                    pad_width = {d: (0, pad)}
                else:
                    pad_width = {d: (pad, 0)}
                variable = variable.pad(pad_width, mode="constant")
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

    def isnull(self, keep_attrs: bool = None):
        """Test each value in the array for whether it is a missing value.

        Returns
        -------
        isnull : Variable
            Same type and shape as object, but the dtype of the data is bool.

        See Also
        --------
        pandas.isnull

        Examples
        --------
        >>> var = xr.Variable("x", [1, np.nan, 3])
        >>> var
        <xarray.Variable (x: 3)>
        array([ 1., nan,  3.])
        >>> var.isnull()
        <xarray.Variable (x: 3)>
        array([False,  True, False])
        """
        from .computation import apply_ufunc

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)

        return apply_ufunc(
            duck_array_ops.isnull,
            self,
            dask="allowed",
            keep_attrs=keep_attrs,
        )

    def notnull(self, keep_attrs: bool = None):
        """Test each value in the array for whether it is not a missing value.

        Returns
        -------
        notnull : Variable
            Same type and shape as object, but the dtype of the data is bool.

        See Also
        --------
        pandas.notnull

        Examples
        --------
        >>> var = xr.Variable("x", [1, np.nan, 3])
        >>> var
        <xarray.Variable (x: 3)>
        array([ 1., nan,  3.])
        >>> var.notnull()
        <xarray.Variable (x: 3)>
        array([ True, False,  True])
        """
        from .computation import apply_ufunc

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)

        return apply_ufunc(
            duck_array_ops.notnull,
            self,
            dask="allowed",
            keep_attrs=keep_attrs,
        )

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
            keep_attrs = kwargs.pop("keep_attrs", None)
            if keep_attrs is None:
                keep_attrs = _get_keep_attrs(default=True)
            with np.errstate(all="ignore"):
                result = self.__array_wrap__(f(self.data, *args, **kwargs))
                if keep_attrs:
                    result.attrs = self.attrs
                return result

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
                raise ValueError("dimensions cannot change for in-place operations")
            with np.errstate(all="ignore"):
                self.values = f(self_data, other_data)
            return self

        return func

    def _to_numeric(self, offset=None, datetime_unit=None, dtype=float):
        """A (private) method to convert datetime array to numeric dtype
        See duck_array_ops.datetime_to_numeric
        """
        numeric_array = duck_array_ops.datetime_to_numeric(
            self.data, offset, datetime_unit, dtype
        )
        return type(self)(self.dims, numeric_array, self._attrs)

    def _unravel_argminmax(
        self,
        argminmax: str,
        dim: Union[Hashable, Sequence[Hashable], None],
        axis: Union[int, None],
        keep_attrs: Optional[bool],
        skipna: Optional[bool],
    ) -> Union["Variable", Dict[Hashable, "Variable"]]:
        """Apply argmin or argmax over one or more dimensions, returning the result as a
        dict of DataArray that can be passed directly to isel.
        """
        if dim is None and axis is None:
            warnings.warn(
                "Behaviour of argmin/argmax with neither dim nor axis argument will "
                "change to return a dict of indices of each dimension. To get a "
                "single, flat index, please use np.argmin(da.data) or "
                "np.argmax(da.data) instead of da.argmin() or da.argmax().",
                DeprecationWarning,
                stacklevel=3,
            )

        argminmax_func = getattr(duck_array_ops, argminmax)

        if dim is ...:
            # In future, should do this also when (dim is None and axis is None)
            dim = self.dims
        if (
            dim is None
            or axis is not None
            or not isinstance(dim, Sequence)
            or isinstance(dim, str)
        ):
            # Return int index if single dimension is passed, and is not part of a
            # sequence
            return self.reduce(
                argminmax_func, dim=dim, axis=axis, keep_attrs=keep_attrs, skipna=skipna
            )

        # Get a name for the new dimension that does not conflict with any existing
        # dimension
        newdimname = "_unravel_argminmax_dim_0"
        count = 1
        while newdimname in self.dims:
            newdimname = f"_unravel_argminmax_dim_{count}"
            count += 1

        stacked = self.stack({newdimname: dim})

        result_dims = stacked.dims[:-1]
        reduce_shape = tuple(self.sizes[d] for d in dim)

        result_flat_indices = stacked.reduce(argminmax_func, axis=-1, skipna=skipna)

        result_unravelled_indices = duck_array_ops.unravel_index(
            result_flat_indices.data, reduce_shape
        )

        result = {
            d: Variable(dims=result_dims, data=i)
            for d, i in zip(dim, result_unravelled_indices)
        }

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        if keep_attrs:
            for v in result.values():
                v.attrs = self.attrs

        return result

    def argmin(
        self,
        dim: Union[Hashable, Sequence[Hashable]] = None,
        axis: int = None,
        keep_attrs: bool = None,
        skipna: bool = None,
    ) -> Union["Variable", Dict[Hashable, "Variable"]]:
        """Index or indices of the minimum of the Variable over one or more dimensions.
        If a sequence is passed to 'dim', then result returned as dict of Variables,
        which can be passed directly to isel(). If a single str is passed to 'dim' then
        returns a Variable with dtype int.

        If there are multiple minima, the indices of the first one found will be
        returned.

        Parameters
        ----------
        dim : hashable, sequence of hashable or ..., optional
            The dimensions over which to find the minimum. By default, finds minimum over
            all dimensions - for now returning an int for backward compatibility, but
            this is deprecated, in future will return a dict with indices for all
            dimensions; to return a dict with all dimensions now, pass '...'.
        axis : int, optional
            Axis over which to apply `argmin`. Only one of the 'dim' and 'axis' arguments
            can be supplied.
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        result : Variable or dict of Variable

        See Also
        --------
        DataArray.argmin, DataArray.idxmin
        """
        return self._unravel_argminmax("argmin", dim, axis, keep_attrs, skipna)

    def argmax(
        self,
        dim: Union[Hashable, Sequence[Hashable]] = None,
        axis: int = None,
        keep_attrs: bool = None,
        skipna: bool = None,
    ) -> Union["Variable", Dict[Hashable, "Variable"]]:
        """Index or indices of the maximum of the Variable over one or more dimensions.
        If a sequence is passed to 'dim', then result returned as dict of Variables,
        which can be passed directly to isel(). If a single str is passed to 'dim' then
        returns a Variable with dtype int.

        If there are multiple maxima, the indices of the first one found will be
        returned.

        Parameters
        ----------
        dim : hashable, sequence of hashable or ..., optional
            The dimensions over which to find the maximum. By default, finds maximum over
            all dimensions - for now returning an int for backward compatibility, but
            this is deprecated, in future will return a dict with indices for all
            dimensions; to return a dict with all dimensions now, pass '...'.
        axis : int, optional
            Axis over which to apply `argmin`. Only one of the 'dim' and 'axis' arguments
            can be supplied.
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        result : Variable or dict of Variable

        See Also
        --------
        DataArray.argmax, DataArray.idxmax
        """
        return self._unravel_argminmax("argmax", dim, axis, keep_attrs, skipna)


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

    def __dask_tokenize__(self):
        from dask.base import normalize_token

        # Don't waste time converting pd.Index to np.ndarray
        return normalize_token((type(self), self._dims, self._data.array, self._attrs))

    def load(self):
        # data is already loaded into memory for IndexVariable
        return self

    # https://github.com/python/mypy/issues/1465
    @Variable.data.setter  # type: ignore
    def data(self, data):
        raise ValueError(
            f"Cannot assign to the .data attribute of dimension coordinate a.k.a IndexVariable {self.name!r}. "
            f"Please use DataArray.assign_coords, Dataset.assign_coords or Dataset.assign as appropriate."
        )

    @Variable.values.setter  # type: ignore
    def values(self, values):
        raise ValueError(
            f"Cannot assign to the .values attribute of dimension coordinate a.k.a IndexVariable {self.name!r}. "
            f"Please use DataArray.assign_coords, Dataset.assign_coords or Dataset.assign as appropriate."
        )

    def chunk(self, chunks={}, name=None, lock=False):
        # Dummy - do not chunk. This method is invoked e.g. by Dataset.chunk()
        return self.copy(deep=False)

    def _as_sparse(self, sparse_format=_default, fill_value=_default):
        # Dummy
        return self.copy(deep=False)

    def _to_dense(self):
        # Dummy
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
            (dim,) = dim.dims

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

        # keep as str if possible as pandas.Index uses object (converts to numpy array)
        data = maybe_coerce_to_str(data, variables)

        attrs = dict(first_var.attrs)
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
    all_dims = {}
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
    variables : iterable of Variable
        Arrays to stack together. Each variable is expected to have
        matching dimensions and shape except for along the stacked
        dimension.
    dim : str or DataArray, optional
        Name of the dimension to stack along. This can either be a new
        dimension name, in which case it is added along axis=0, or an
        existing dimension name, in which case the location of the
        dimension is unchanged. Where to insert the new dimension is
        determined by the first variable.
    positions : None or list of array-like, optional
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
                    level_names[n].append(f"{n!r} ({var_name})")
            if idx_level_names:
                all_level_names.update(idx_level_names)

    for k, v in level_names.items():
        if k in variables:
            v.append("(%s)" % k)

    duplicate_names = [v for v in level_names.values() if len(v) > 1]
    if duplicate_names:
        conflict_str = "\n".join(", ".join(v) for v in duplicate_names)
        raise ValueError("conflicting MultiIndex level name(s):\n%s" % conflict_str)
    # Check confliction between level names and dimensions GH:2299
    for k, v in variables.items():
        for d in v.dims:
            if d in all_level_names:
                raise ValueError(
                    "conflicting level / dimension names. {} "
                    "already exists as a level name.".format(d)
                )
