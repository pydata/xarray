from typing import Any, Callable, List, Tuple, Union

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
