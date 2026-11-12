"""Xarray integration with Apache Arrow."""

from __future__ import annotations

import datetime
import json
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, override

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa

    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.variable import Variable


XARRAY_SCHEMA_VERSION = b"v1"


class XarrayArrowMetadata(TypedDict):
    name: NotRequired[Hashable | None]
    dims: list[Hashable]
    shape: list[int]
    attrs: dict[Hashable, Any]
    coords: dict[str, dict[str, Any]]


class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy values in Arrow schema metadata."""

    @override
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, datetime.date | datetime.time):
            return o.isoformat()
        if isinstance(o, np.datetime64):
            return str(o)
        return super().default(o)


def _import_pyarrow():
    try:
        import pyarrow as pa
    except ImportError:
        raise ImportError(
            "pyarrow is required to export via the Arrow PyCapsule Interface."
        ) from None
    return pa


def _get_schema_from_dataarray(data_array: DataArray) -> pa.Schema:
    """Get xarray pyarrow schema for a DataArray.

    Returns
    -------
    pyarrow.Schema
    """
    pa = _import_pyarrow()

    fields = [
        pa.field(str(name), pa.from_numpy_dtype(coord.dtype))
        for name, coord in data_array._coords.items()
    ]
    if data_array.name is None:
        raise ValueError("cannot convert an unnamed DataArray to Arrow")
    values_column = data_array.name
    fields.append(
        pa.field(str(values_column), pa.from_numpy_dtype(data_array._variable.dtype))
    )

    xarray_metadata: XarrayArrowMetadata = {
        "name": data_array.name,
        "dims": list(data_array._variable.dims),
        "shape": list(data_array._variable.shape),
        "attrs": data_array.attrs,
        "coords": {
            str(name): variable.to_dict(data=False)
            for name, variable in data_array._coords.items()
        },
    }
    schema_metadata = {
        b"xarray:arrow_schema_version": XARRAY_SCHEMA_VERSION,
        b"xarray": json.dumps(xarray_metadata, cls=_NumpyEncoder).encode(),
    }
    return pa.schema(fields, metadata=schema_metadata)


def _get_schema_from_dataset(dataset: Dataset) -> pa.Schema:
    """Get xarray pyarrow schema for a Dataset.

    Returns
    -------
    pyarrow.Schema
    """
    pa = _import_pyarrow()

    fields = [
        pa.field(str(name), pa.from_numpy_dtype(variable.dtype))
        for name, variable in dataset.variables.items()
    ]
    xarray_metadata = {
        "dims": list(dataset.dims),
        "shape": [dataset.sizes[dim] for dim in dataset.dims],
        "attrs": dataset.attrs,
        "coords": {
            str(name): coord.to_dict(data=False)
            for name, coord in dataset.coords.variables.items()
        },
        "variables": {
            str(name): variable.to_dict(data=False)
            for name, variable in dataset.data_vars.variables.items()
        },
    }
    schema_metadata = {
        b"xarray:arrow_schema_version": XARRAY_SCHEMA_VERSION,
        b"xarray": json.dumps(xarray_metadata, cls=_NumpyEncoder).encode(),
    }
    return pa.schema(fields, metadata=schema_metadata)


def _broadcast_to_dims(
    variable: Variable,
    dims: tuple[Hashable, ...],
    shape: tuple[int, ...],
) -> np.ndarray:
    """Flatten and broadcast `variable` to `shape`, as a flat numpy array.

    Returns
    -------
    numpy.ndarray
        C-contiguous, flattened array.

    Examples
    --------
    >>> coord = xr.Variable("y", np.array([10, 20, 30]))
    >>> _broadcast_to_dims(coord, dims=("x", "y"), shape=(2, 3))
    array([10, 20, 30, 10, 20, 30])
    """
    if len(dims) != len(shape):
        raise ValueError(
            f"dims and shape must have the same length, got dims={dims!r} "
            f"(length {len(dims)}) and shape={shape!r} (length {len(shape)})"
        )
    if not set(variable.dims) <= set(dims):
        raise ValueError(
            f"variable.dims must be a subset of dims, got variable.dims="
            f"{variable.dims!r} and dims={dims!r}"
        )

    coord_values = variable.values
    if (variable.dims, variable.shape) == (dims, shape):
        return coord_values.ravel()

    dim_order = tuple(variable.dims.index(dim) for dim in dims if dim in variable.dims)
    ordered_coords = coord_values.transpose(dim_order)
    indexer = tuple(slice(None) if dim in variable.dims else np.newaxis for dim in dims)
    expanded_coords = ordered_coords[indexer]
    return np.broadcast_to(expanded_coords, shape).ravel()
