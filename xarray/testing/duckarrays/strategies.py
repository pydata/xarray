from typing import Any

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st

import xarray as xr
from xarray.core.utils import is_dict_like

from . import utils

all_dtypes = (
    npst.integer_dtypes()
    | npst.unsigned_integer_dtypes()
    | npst.floating_dtypes()
    | npst.complex_number_dtypes()
)


def numpy_array(shape, dtypes=None):
    if dtypes is None:
        dtypes = all_dtypes

    def elements(dtype):
        max_value = 100
        min_value = 0 if dtype.kind == "u" else -max_value

        return npst.from_dtype(
            dtype, allow_infinity=False, min_value=min_value, max_value=max_value
        )

    return dtypes.flatmap(
        lambda dtype: npst.arrays(dtype=dtype, shape=shape, elements=elements(dtype))
    )


def dimension_sizes(min_dims, max_dims, min_size, max_size):
    sizes = st.lists(
        elements=st.tuples(st.text(min_size=1), st.integers(min_size, max_size)),
        min_size=min_dims,
        max_size=max_dims,
        unique_by=lambda x: x[0],
    )
    return sizes


# Is there a way to do this in general?
# Could make a Protocol...
T_DuckArray = Any


@st.composite
def duckarray(
    draw,
    create_data,
    *,
    sizes=None,
    min_size=1,
    max_size=3,
    min_dims=1,
    max_dims=3,
    dtypes=None,
) -> st.SearchStrategy[T_DuckArray]:
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
        shape = ()
    else:
        _, shape = zip(*sizes)
    data = create_data(shape, dtypes)

    return draw(data)


@st.composite
def variable(
    draw,
    create_data,
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
def data_array(
    draw, create_data, *, min_dims=1, max_dims=3, min_size=1, max_size=3, dtypes=None
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
def dataset(
    draw,
    create_data,
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
            values=variable(create_data, sizes=s, dtypes=dtypes),
            min_size=min_vars,
            max_size=max_vars,
        )
    )

    return xr.Dataset(data_vars=draw(data_vars))


def valid_axis(ndim):
    if ndim == 0:
        return st.none() | st.just(0)
    return st.none() | st.integers(-ndim, ndim - 1)


def valid_axes(ndim):
    return valid_axis(ndim) | npst.valid_tuple_axes(ndim, min_size=1)


def valid_dim(dims):
    if not isinstance(dims, list):
        dims = [dims]

    ndim = len(dims)
    axis = valid_axis(ndim)
    return axis.map(lambda axes: utils.valid_dims_from_axes(dims, axes))


def valid_dims(dims):
    if is_dict_like(dims):
        dims = list(dims.keys())
    elif isinstance(dims, tuple):
        dims = list(dims)
    elif not isinstance(dims, list):
        dims = [dims]

    ndim = len(dims)
    axes = valid_axes(ndim)
    return axes.map(lambda axes: utils.valid_dims_from_axes(dims, axes))
