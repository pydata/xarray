import numpy as np

from xarray.core import array_api_compat, duck_array_ops


def reshape(array, shape, **kwargs):
    return type(array)(duck_array_ops.reshape(array.array, shape=shape, **kwargs))


def concatenate(arrays, axis):
    return type(arrays[0])(
        duck_array_ops.concatenate([a.array for a in arrays], axis=axis)
    )


def result_type(*arrays_and_dtypes):
    parsed = [a.array if hasattr(a, "array") else a for a in arrays_and_dtypes]
    return array_api_compat.result_type(*parsed, xp=np)


def astype(array, dtype, **kwargs):
    return type(array)(duck_array_ops.astype(array.array, dtype=dtype, **kwargs))


def isnan(array):
    return type(array)(duck_array_ops.isnull(array.array))


def any(array, *args, **kwargs):  # TODO: keepdims
    return duck_array_ops.array_any(array.array, *args, **kwargs)
