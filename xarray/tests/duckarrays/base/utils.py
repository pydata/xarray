import hypothesis.extra.numpy as npst
import hypothesis.strategies as st

shapes = npst.array_shapes()
dtypes = (
    npst.floating_dtypes()
    | npst.integer_dtypes()
    | npst.unsigned_integer_dtypes()
    | npst.complex_number_dtypes()
)


numpy_array = npst.arrays(dtype=dtypes, shape=shapes)


def create_dimension_names(ndim):
    return [f"dim_{n}" for n in range(ndim)]


def valid_axes(ndim):
    return st.none() | st.integers(-ndim, ndim - 1) | npst.valid_tuple_axes(ndim)


def valid_dims_from_axes(dims, axes):
    if axes is None:
        return None

    if isinstance(axes, int):
        return dims[axes]

    return type(axes)(dims[axis] for axis in axes)
