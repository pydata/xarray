from __future__ import annotations

from typing import NamedTuple

from xarray.namedarray._array_api._utils import (
    _get_data_namespace,
    _infer_dims,
)
from xarray.namedarray.core import NamedArray


class UniqueAllResult(NamedTuple):
    values: NamedArray
    indices: NamedArray
    inverse_indices: NamedArray
    counts: NamedArray


class UniqueCountsResult(NamedTuple):
    values: NamedArray
    counts: NamedArray


class UniqueInverseResult(NamedTuple):
    values: NamedArray
    inverse_indices: NamedArray


def unique_all(x: NamedArray, /) -> UniqueAllResult:
    xp = _get_data_namespace(x)
    values, indices, inverse_indices, counts = xp.unique_all(x._data)
    _dims_values = _infer_dims(values.shape)  # TODO: Fix
    _dims_indices = _infer_dims(indices.shape)  # TODO: Fix dims
    _dims_inverse_indices = _infer_dims(inverse_indices.shape)  # TODO: Fix dims
    _dims_counts = _infer_dims(counts.shape)  # TODO: Fix dims
    return UniqueAllResult(
        NamedArray(_dims_values, values),
        NamedArray(_dims_indices, indices),
        NamedArray(_dims_inverse_indices, inverse_indices),
        NamedArray(_dims_counts, counts),
    )


def unique_counts(x: NamedArray, /) -> UniqueCountsResult:
    xp = _get_data_namespace(x)
    values, counts = xp.unique_counts(x._data)
    _dims_values = _infer_dims(values.shape)  # TODO:  Fix dims
    _dims_counts = _infer_dims(counts.shape)  # TODO: Fix dims
    return UniqueCountsResult(
        NamedArray(_dims_values, values),
        NamedArray(_dims_counts, counts),
    )


def unique_inverse(x: NamedArray, /) -> UniqueInverseResult:
    xp = _get_data_namespace(x)
    values, inverse_indices = xp.unique_inverse(x._data)
    _dims_values = _infer_dims(values.shape)  # TODO: Fix
    _dims_inverse_indices = _infer_dims(inverse_indices.shape)  # TODO: Fix dims
    return UniqueInverseResult(
        NamedArray(_dims_values, values),
        NamedArray(_dims_inverse_indices, inverse_indices),
    )


def unique_values(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.unique_values(x._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix
    return x._new(_dims, _data)
