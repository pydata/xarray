import operator

import numpy as np

from xarray.namedarray.utils import (
    OrderedSet,
    is_0d_dask_array,
    is_dict_like,
    is_duck_array,
    is_duck_dask_array,
)

integer_types = (int, np.integer)
# https://github.com/python/mypy/issues/224
BASIC_INDEXING_TYPES = integer_types + (slice,)


def _unified_dims(named_arrays):
    all_dims = {}
    for namedarray in named_arrays:
        namedarray_dims = namedarray.dims
        if len(set(namedarray_dims)) < len(namedarray_dims):
            raise ValueError(
                "broadcasting cannot handle duplicate "
                f"dimensions: {list(namedarray_dims)!r}"
            )
        for d, s in zip(namedarray_dims, namedarray.shape):
            if d not in all_dims:
                all_dims[d] = s
            elif all_dims[d] != s:
                raise ValueError(
                    "operands cannot be broadcast together "
                    f"with mismatched lengths for dimension {d!r}: {(all_dims[d], s)}"
                )

    return all_dims


def _broadcast_compat_namedarrays(*named_arrays):
    dims = tuple(_unified_dims(named_arrays))
    return tuple(
        namedarray.set_dims(dims) if namedarray.dims != dims else namedarray
        for namedarray in named_arrays
    )


def as_integer_or_none(value):
    return None if value is None else operator.index(value)


def as_integer_slice(value):
    start = as_integer_or_none(value.start)
    stop = as_integer_or_none(value.stop)
    step = as_integer_or_none(value.step)
    return slice(start, stop, step)


def _item_key_to_tuple(*, key, dims):
    if is_dict_like(key):
        return tuple(key.get(dim, slice(None)) for dim in dims)
    else:
        return key


def expanded_indexer(*, key, ndim):
    """Given a key for indexing an ndarray, return an equivalent key which is a
    tuple with length equal to the number of dimensions.

    The expansion is done by replacing all `Ellipsis` items with the right
    number of full slices and then padding the key with full slices so that it
    reaches the appropriate dimensionality.
    """
    if not isinstance(key, tuple):
        # numpy treats non-tuple keys equivalent to tuples of length 1
        key = (key,)
    new_key = []
    # handling Ellipsis right is a little tricky, see:
    # https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing
    found_ellipsis = False
    for k in key:
        if k is Ellipsis:
            if not found_ellipsis:
                new_key.extend((ndim + 1 - len(key)) * [slice(None)])
                found_ellipsis = True
            else:
                new_key.append(slice(None))
        else:
            new_key.append(k)
    if len(new_key) > ndim:
        raise IndexError("too many indices")
    new_key.extend((ndim - len(new_key)) * [slice(None)])
    return tuple(new_key)


def _broadcast_indexes_basic(*, key, dims):
    dims = tuple(dim for k, dim in zip(key, dims) if not isinstance(k, integer_types))
    return dims, OIndexer(key), None


def _broadcast_indexes_outer(*, key, dims):
    from xarray.namedarray.core import NamedArray

    # drop dim if k is integer or if k is a 0d dask array
    dims = tuple(
        k.dims[0] if isinstance(k, NamedArray) else dim
        for k, dim in zip(key, dims)
        if (not isinstance(k, integer_types) and not is_0d_dask_array(k))
    )

    new_key = []
    for k in key:
        if isinstance(k, NamedArray):
            k = k.data
        if not isinstance(k, BASIC_INDEXING_TYPES):
            if not is_duck_array(k):
                k = np.asarray(k)
            if k.size == 0:
                # Slice by empty list; numpy could not infer the dtype
                k = k.astype(int)
            elif k.dtype.kind == "b":
                (k,) = np.nonzero(k)
        new_key.append(k)

    return dims, OIndexer(tuple(new_key)), None


def _broadcast_indexes_vectorized(*, key, dims, sizes):
    from xarray.namedarray.core import NamedArray

    named_arrays = []
    out_dims_set = OrderedSet()
    for dim, value in zip(dims, key):
        if isinstance(value, slice):
            out_dims_set.add(dim)
        else:
            named_array = (
                value
                if isinstance(value, NamedArray)
                else NamedArray._new(dims=dim, data=value)
            )
            if named_array.dtype.kind == "b":  # boolean indexing case
                (named_array,) = named_array._nonzero()

            named_arrays.append(named_array)
            out_dims_set.update(named_array.dims)

    named_array_dims = set()
    for named_array in named_arrays:
        named_array_dims.update(named_array.dims)

    slices = []
    for i, (dim, value) in enumerate(zip(dims, key)):
        if isinstance(value, slice):
            if dim in named_array_dims:
                # We only convert slice objects to variables if they share
                # a dimension with at least one other variable. Otherwise,
                # we can equivalently leave them as slices aknd transpose
                # the result. This is significantly faster/more efficient
                # for most array backends.
                values = np.arange(*value.indices(sizes[dim]))
                named_array.insert(i - len(slices), NamedArray((dim,), values))
            else:
                slices.append((i, value))
    try:
        named_arrays = _broadcast_compat_namedarrays(*named_arrays)
    except ValueError:
        raise IndexError(f"Dimensions of indexers mismatch: {key}")

    out_key = [named_array.data for named_array in named_arrays]
    out_dims = tuple(out_dims_set)
    slice_positions = set()
    for i, value in slices:
        out_key.insert(i, value)
        new_position = out_dims.index(dims[i])
        slice_positions.add(new_position)

    if slice_positions:
        new_order = [i for i in range(len(out_dims)) if i not in slice_positions]

    else:
        new_order = None

    return out_dims, VIndexer(tuple(out_key)), new_order


def _validate_indexers(*, key, dims, shape):
    from xarray.namedarray.core import NamedArray, _get_axis_num

    for dim, k in zip(dims, key):
        if not isinstance(k, BASIC_INDEXING_TYPES):
            if not isinstance(k, NamedArray):
                if not is_duck_array(k):
                    k = np.asarray(k)
                if k.ndim > 1:
                    raise IndexError(
                        "Unlabeled multi-dimensional array cannot be "
                        f"used for indexing: {k}"
                    )
            if k.dtype.kind == "b":
                if shape[_get_axis_num(dims, dim)] != len(k):
                    raise IndexError(
                        f"Boolean array size {len(k):d} is used to index array "
                        f"with shape {str(shape):s}."
                    )
                if k.ndim > 1:
                    raise IndexError(
                        f"{k.ndim}-dimensional boolean indexing is " "not supported. "
                    )
                if is_duck_dask_array(k.data):
                    raise IndexError(
                        "Boolean indexer should be unlabeled or on the "
                        "same dimension to the indexed array. Indexer is "
                        f"on {str(k.dims):s} but the target dimension is {dim:s}."
                    )


def broadcast_indexes(*, key, dims, ndim, shape, sizes):
    from xarray.namedarray.core import NamedArray

    key = _item_key_to_tuple(key=key, dims=dims)
    # key is a tuple of full size
    key = expanded_indexer(key=key, ndim=ndim)
    # convert a scalar namedArray to a 0d-array
    key = tuple(k.data if isinstance(k, NamedArray) and k.ndim == 0 else k for k in key)
    # Convert a 0d numpy arrays to an integer
    # dask 0d arrays are passed through
    key = tuple(
        k.item() if isinstance(k, np.ndarray) and k.ndim == 0 else k for k in key
    )

    if all(isinstance(k, BASIC_INDEXING_TYPES) for k in key):
        return _broadcast_indexes_basic(key=key, dims=dims)

    _validate_indexers(key=key, dims=dims, shape=shape)

    # Detect it can be mapped as an outer indexer
    # If all key is unlabeled, or
    # key can be mapped as an OuterIndexer.
    if all(not isinstance(k, NamedArray) for k in key):
        return _broadcast_indexes_outer(key=key, dims=dims)

    # If all key is 1-dimensional and there are no duplicate labels,
    # key can be mapped as an OuterIndexer.
    dims = []
    for k, d in zip(key, dims):
        if isinstance(k, NamedArray):
            if len(k.dims) > 1:
                return _broadcast_indexes_vectorized(key=key, dims=dims, sizes=sizes)
            dims.append(k.dims[0])
        elif not isinstance(k, integer_types):
            dims.append(d)
    if len(set(dims)) == len(dims):
        return _broadcast_indexes_outer(key=key, dims=dims)

    return _broadcast_indexes_vectorized(key=key, dims=dims, sizes=sizes)


class OIndexer:
    __slots__ = ("_key",)

    def __init__(self, key):
        if not isinstance(key, tuple):
            raise TypeError(f"key must be a tuple: {key!r}")

        new_key = []
        for k in key:
            if isinstance(k, integer_types):
                k = int(k)
            elif isinstance(k, slice):
                k = as_integer_slice(k)
            elif is_duck_array(k):
                if not np.issubdtype(k.dtype, np.integer):
                    raise TypeError(
                        f"invalid indexer array, does not have integer dtype: {k!r}"
                    )
                if k.ndim > 1:
                    raise TypeError(
                        f"invalid indexer array; must be scalar or have 1 dimension: {k!r}"
                    )
                k = k.astype(np.int64)
            else:
                raise TypeError(f"unexpected indexer type: {k!r}")

            new_key.append(k)

        self._key = new_key

    @property
    def tuple(self):
        return self._key

    def __repr__(self):
        return f"{type(self).__name__}({self.tuple})"


class VIndexer:
    __slots__ = ("_key",)

    def __init__(self, key):
        if not isinstance(key, tuple):
            raise TypeError(f"key must be a tuple: {key!r}")

        new_key = []
        ndim = None
        for k in key:
            if isinstance(k, slice):
                k = as_integer_slice(k)
            elif is_duck_dask_array(k):
                raise ValueError(
                    "Vectorized indexing with Dask arrays is not supported. "
                    "Please pass a numpy array by calling ``.compute``. "
                    "See https://github.com/dask/dask/issues/8958."
                )
            elif is_duck_array(k):
                if not np.issubdtype(k.dtype, np.integer):
                    raise TypeError(
                        f"invalid indexer array, does not have integer dtype: {k!r}"
                    )
                if ndim is None:
                    ndim = k.ndim
                elif ndim != k.ndim:
                    ndims = [k.ndim for k in key if isinstance(k, np.ndarray)]
                    raise ValueError(
                        "invalid indexer key: ndarray arguments "
                        f"have different numbers of dimensions: {ndims}"
                    )
                k = k.astype(np.int64)
            else:
                raise TypeError(
                    f"unexpected indexer type for {type(self).__name__}: {k!r}"
                )
            new_key.append(k)

        self._key = new_key

    @property
    def tuple(self):
        return self._key

    def __repr__(self):
        return f"{type(self).__name__}({self.tuple})"


class IndexCallable:
    __slots__ = ("fn",)

    def __init__(self, fn) -> None:
        self.fn = fn

    def __getitem__(self, key):
        return self.fn(key)


def _vindex(x, *indices):
    return x


def _oindex(x, *indices):
    return x
