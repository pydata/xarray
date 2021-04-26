import hypothesis.extra.numpy as npst
import hypothesis.strategies as st

import xarray as xr

dtypes = (
    npst.integer_dtypes()
    | npst.unsigned_integer_dtypes()
    | npst.floating_dtypes()
    | npst.complex_number_dtypes()
)


def numpy_array(shape):
    return npst.arrays(dtype=dtypes, shape=shape)


def dimension_sizes(min_dims, max_dims, min_size, max_size):
    sizes = st.lists(
        elements=st.tuples(st.text(min_size=1), st.integers(min_size, max_size)),
        min_size=min_dims,
        max_size=max_dims,
        unique_by=lambda x: x[0],
    )
    return sizes


@st.composite
def variable(
    draw, create_data, *, sizes=None, min_size=1, max_size=5, min_dims=1, max_dims=4
):
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
    data = create_data(shape)

    return xr.Variable(dims, draw(data))


@st.composite
def data_array(draw, create_data, *, min_dims=1, max_dims=4, min_size=1, max_size=5):
    name = draw(st.none() | st.text(min_size=1))

    sizes = st.lists(
        elements=st.tuples(st.text(min_size=1), st.integers(min_size, max_size)),
        min_size=min_dims,
        max_size=max_dims,
        unique_by=lambda x: x[0],
    )
    drawn_sizes = draw(sizes)
    dims, shape = zip(*drawn_sizes)

    data = create_data(shape)

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
    max_dims=4,
    min_size=2,
    max_size=5,
    min_vars=1,
    max_vars=5,
):
    names = st.text(min_size=1)
    sizes = dimension_sizes(
        min_size=min_size, max_size=max_size, min_dims=min_dims, max_dims=max_dims
    )

    data_vars = sizes.flatmap(
        lambda s: st.dictionaries(
            keys=names.filter(lambda n: n not in dict(s)),
            values=variable(create_data, sizes=s),
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
    return valid_axis(ndim) | npst.valid_tuple_axes(ndim)
