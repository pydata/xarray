from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Union

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np

import xarray as xr
from xarray.core.utils import is_dict_like

from . import utils

all_dtypes: st.SearchStrategy[np.dtype] = (
    npst.integer_dtypes()
    | npst.unsigned_integer_dtypes()
    | npst.floating_dtypes()
    | npst.complex_number_dtypes()
)


def elements(dtype) -> st.SearchStrategy[Any]:
    max_value = 100
    min_value = 0 if dtype.kind == "u" else -max_value

    return npst.from_dtype(
        dtype, allow_infinity=False, min_value=min_value, max_value=max_value
    )


def numpy_array(shape, dtypes=None) -> st.SearchStrategy[np.ndarray]:
    if dtypes is None:
        dtypes = all_dtypes

    return dtypes.flatmap(
        lambda dtype: npst.arrays(dtype=dtype, shape=shape, elements=elements(dtype))
    )


def dimension_sizes(
    min_dims, max_dims, min_size, max_size
) -> st.SearchStrategy[List[Tuple[str, int]]]:
    sizes = st.lists(
        elements=st.tuples(st.text(min_size=1), st.integers(min_size, max_size)),
        min_size=min_dims,
        max_size=max_dims,
        unique_by=lambda x: x[0],
    )
    return sizes


@st.composite
def variables(
    draw: st.DrawFn,
    create_data: Callable,
    *,
    sizes=None,
    min_size=1,
    max_size=3,
    min_dims=1,
    max_dims=3,
    dtypes=None,
) -> st.SearchStrategy[xr.Variable]:

    if sizes is None:
        sizes = draw(
            dimension_sizes(
                min_size=min_size,
                max_size=max_size,
                min_dims=min_dims,
                max_dims=max_dims,
            )
        )

    if not sizes:
        dims = ()
        shape = ()
    else:
        dims, shape = zip(*sizes)
    data = create_data(shape, dtypes)

    return xr.Variable(dims, draw(data))


@st.composite
def dataarrays(
    draw: st.DrawFn,
    create_data: Callable,
    *,
    min_dims=1,
    max_dims=3,
    min_size=1,
    max_size=3,
    dtypes=None,
) -> st.SearchStrategy[xr.DataArray]:

    name = draw(st.none() | st.text(min_size=1))
    if dtypes is None:
        dtypes = all_dtypes

    sizes = st.lists(
        elements=st.tuples(st.text(min_size=1), st.integers(min_size, max_size)),
        min_size=min_dims,
        max_size=max_dims,
        unique_by=lambda x: x[0],
    )
    drawn_sizes = draw(sizes)
    dims, shape = zip(*drawn_sizes)

    data = draw(create_data(shape, dtypes))

    return xr.DataArray(
        data=data,
        name=name,
        dims=dims,
    )


@st.composite
def datasets(
    draw: st.DrawFn,
    create_data: Callable,
    *,
    min_dims=1,
    max_dims=3,
    min_size=1,
    max_size=3,
    min_vars=1,
    max_vars=3,
) -> st.SearchStrategy[xr.Dataset]:

    dtypes = st.just(draw(all_dtypes))
    names = st.text(min_size=1)
    sizes = dimension_sizes(
        min_size=min_size, max_size=max_size, min_dims=min_dims, max_dims=max_dims
    )

    data_vars = sizes.flatmap(
        lambda s: st.dictionaries(
            keys=names.filter(lambda n: n not in dict(s)),
            values=variables(create_data, sizes=s, dtypes=dtypes),
            min_size=min_vars,
            max_size=max_vars,
        )
    )

    return xr.Dataset(data_vars=draw(data_vars))


def valid_axis(ndim) -> st.SearchStrategy[Union[None, int]]:
    if ndim == 0:
        return st.none() | st.just(0)
    return st.none() | st.integers(-ndim, ndim - 1)


def valid_axes(ndim) -> st.SearchStrategy[Union[None, int, Tuple[int, ...]]]:
    return valid_axis(ndim) | npst.valid_tuple_axes(ndim, min_size=1)


def valid_dim(dims) -> st.SearchStrategy[str]:
    if not isinstance(dims, list):
        dims = [dims]

    ndim = len(dims)
    axis = valid_axis(ndim)
    return axis.map(lambda axes: utils.valid_dims_from_axes(dims, axes))


def valid_dims(dims) -> st.SearchStrategy[xr.DataArray]:
    if is_dict_like(dims):
        dims = list(dims.keys())
    elif isinstance(dims, tuple):
        dims = list(dims)
    elif not isinstance(dims, list):
        dims = [dims]

    ndim = len(dims)
    axes = valid_axes(ndim)
    return axes.map(lambda axes: utils.valid_dims_from_axes(dims, axes))


@st.composite
def block_lengths(
    draw: st.DrawFn,
    ax_length: int,
    min_chunk_length: int = 1,
    max_chunk_length: Optional[int] = None,
) -> st.SearchStrategy[Tuple[int, ...]]:
    """Generate different chunking patterns along one dimension of an array."""

    chunks = []
    remaining_length = ax_length
    while remaining_length > 0:
        _max_chunk_length = (
            min(remaining_length, max_chunk_length)
            if max_chunk_length
            else remaining_length
        )

        if min_chunk_length > _max_chunk_length:
            # if we are at the end of the array we have no choice but to use a smaller chunk
            chunk = remaining_length
        else:
            chunk = draw(
                st.integers(min_value=min_chunk_length, max_value=_max_chunk_length)
            )

        chunks.append(chunk)
        remaining_length = remaining_length - chunk

    return tuple(chunks)


# TODO we could remove this once dask/9374 is merged upstream
@st.composite
def chunks(
    draw: st.DrawFn,
    shape: Tuple[int, ...],
    axes: Optional[Union[int, Tuple[int, ...]]] = None,
    min_chunk_length: int = 1,
    max_chunk_length: Optional[int] = None,
) -> st.SearchStrategy[Tuple[Tuple[int, ...], ...]]:
    """
    Generates different chunking patterns for an N-D array with a given shape.

    Returns chunking structure as a tuple of tuples of ints, with each inner tuple containing
    the block lengths along one dimension of the array.

    You can limit chunking to specific axes using the `axes` kwarg, and specify minimum and
    maximum block lengths.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the array for which you want to generate a chunking pattern.
    axes : None or int or tuple of ints, optional
        ...
    min_chunk_length : int, default is 1
        Minimum chunk length to use along all axes.
    max_chunk_length: int, optional
        Maximum chunk length to use along all axes.
        Default is that the chunk can be as long as the length of the array along that axis.

    Examples
    --------
    Chunking along all axes by default

    >>> chunks(shape=(2, 3)).example()
    ((1, 1), (1, 2))

    Chunking only along the second axis

    >>> chunks(shape=(2, 3), axis=1).example()
    ((2,), (1, 1, 1))

    Minimum size chunks of length 2 along all axes

    >>> chunks(shape=(2, 3), min_chunk_length=2).example()
    ((2,), (2, 1))

    Smallest possible chunks along all axes

    >>> chunks(shape=(2, 3), max_chunk_length=1).example()
    ((1, 1), (1, 1, 1))

    Maximum size chunks along all axes

    >>> chunks(shape=(2, 3), axes=()).example()
    ((2,), (3,))

    See Also
    --------
    testing.strategies.chunks
    DataArray.chunk
    DataArray.chunks
    """

    if min_chunk_length < 1 or not isinstance(min_chunk_length, int):
        raise ValueError("min_chunk_length must be an integer >= 1")

    if max_chunk_length:
        if max_chunk_length < 1 or not isinstance(min_chunk_length, int):
            raise ValueError("max_chunk_length must be an integer >= 1")

    if axes is None:
        axes = tuple(range(len(shape)))
    elif isinstance(axes, int):
        axes = (axes,)

    chunks = []
    for axis, ax_length in enumerate(shape):

        _max_chunk_length = (
            min(max_chunk_length, ax_length) if max_chunk_length else ax_length
        )

        if axes is not None and axis in axes:
            block_lengths_along_ax = draw(
                block_lengths(
                    ax_length,
                    min_chunk_length=min_chunk_length,
                    max_chunk_length=_max_chunk_length,
                )
            )
        else:
            # don't chunk along this dimension
            block_lengths_along_ax = (ax_length,)

        chunks.append(block_lengths_along_ax)

    return tuple(chunks)


@st.composite
def chunksizes(
    draw: st.DrawFn,
    sizes: Mapping[str, int],
    dims: Set[str] = None,
    min_chunk_length: int = 1,
    max_chunk_length: int = None,
) -> st.SearchStrategy[Mapping[str, Tuple[int, ...]]]:
    """
    Generate different chunking patterns for an xarray object with given sizes.

    Returns chunking structure as a mapping of dimension names to tuples of ints,
    with each tuple containing the block lengths along one dimension of the object.

    You can limit chunking to specific dimensions given by the `dim` kwarg.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    sizes : mapping of dimension names to ints
        Size of the object for which you want to generate a chunking pattern.
    dims : set of str, optional
        Dimensions to chunk along. Default is to chunk along all dimensions.
    min_chunk_length : int, default is 1
        Minimum chunk length to use along all dimensions.
    max_chunk_length: int, optional
        Maximum chunk length to use along all dimensions.
        Default is that the chunk can be as long as the length of the array along that dimension.

    See Also
    --------
    testing.strategies.chunks
    DataArray.chunk
    DataArray.chunksizes
    DataArray.sizes
    """
    shape = tuple(sizes.values())
    axes = tuple(list(sizes.keys()).index(d) for d in dims) if dims else None
    _chunks = draw(
        chunks(
            shape=shape,
            axes=axes,
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
        )
    )

    return {d: c for d, c in zip(list(sizes.keys()), _chunks)}
