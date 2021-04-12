import warnings
from contextlib import contextmanager

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st

shapes = npst.array_shapes()
dtypes = (
    npst.integer_dtypes()
    | npst.unsigned_integer_dtypes()
    | npst.floating_dtypes()
    | npst.complex_number_dtypes()
)


@contextmanager
def suppress_warning(category, message=""):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=category, message=message)

        yield


numpy_array = npst.arrays(dtype=dtypes, shape=shapes)


def create_dimension_names(ndim):
    return [f"dim_{n}" for n in range(ndim)]


def valid_axis(ndim):
    return st.none() | st.integers(-ndim, ndim - 1)


def valid_axes(ndim):
    return valid_axis(ndim) | npst.valid_tuple_axes(ndim)


def valid_dims_from_axes(dims, axes):
    if axes is None:
        return None

    if isinstance(axes, int):
        return dims[axes]

    return [dims[axis] for axis in axes]


def valid_axes_from_dims(all_dims, dims):
    if dims is None:
        return None
    elif isinstance(dims, list):
        return [all_dims.index(dim) for dim in dims]
    else:
        return all_dims.index(dims)
