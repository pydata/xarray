import warnings
from contextlib import contextmanager


@contextmanager
def suppress_warning(category, message=""):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=category, message=message)

        yield


def create_dimension_names(ndim):
    return [f"dim_{n}" for n in range(ndim)]


def valid_dims_from_axes(dims, axes):
    if axes is None:
        return None

    if axes == 0 and len(dims) == 0:
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
