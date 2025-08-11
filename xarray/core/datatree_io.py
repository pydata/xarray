from __future__ import annotations

import io
from collections.abc import Mapping
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal, get_args

from xarray.core.datatree import DataTree
from xarray.core.types import NetcdfWriteModes, ZarrWriteModes

T_DataTreeNetcdfEngine = Literal["netcdf4", "h5netcdf", "pydap"]
T_DataTreeNetcdfTypes = Literal["NETCDF4"]

if TYPE_CHECKING:
    from xarray.core.types import ZarrStoreLike


def _datatree_to_netcdf(
    dt: DataTree,
    filepath: str | PathLike | io.IOBase | None = None,
    mode: NetcdfWriteModes = "w",
    encoding: Mapping[str, Any] | None = None,
    unlimited_dims: Mapping | None = None,
    format: T_DataTreeNetcdfTypes | None = None,
    engine: T_DataTreeNetcdfEngine | None = None,
    group: str | None = None,
    write_inherited_coords: bool = False,
    compute: bool = True,
    **kwargs,
) -> None | memoryview:
    """Implementation of `DataTree.to_netcdf`."""

    if format not in [None, *get_args(T_DataTreeNetcdfTypes)]:
        raise ValueError("DataTree.to_netcdf only supports the NETCDF4 format")

    if engine not in [None, *get_args(T_DataTreeNetcdfEngine)]:
        raise ValueError(
            "DataTree.to_netcdf only supports the netcdf4 and h5netcdf engines"
        )

    if engine is None:
        engine = "h5netcdf"

    if group is not None:
        raise NotImplementedError(
            "specifying a root group for the tree has not been implemented"
        )

    if not compute:
        raise NotImplementedError("compute=False has not been implemented yet")

    if encoding is None:
        encoding = {}

    # In the future, we may want to expand this check to insure all the provided encoding
    # options are valid. For now, this simply checks that all provided encoding keys are
    # groups in the datatree.
    if set(encoding) - set(dt.groups):
        raise ValueError(
            f"unexpected encoding group name(s) provided: {set(encoding) - set(dt.groups)}"
        )

    if filepath is None:
        # No need to use BytesIOProxy here because the legacy scipy backend
        # cannot write netCDF files with groups
        target = io.BytesIO()
    else:
        target = filepath  # type: ignore[assignment]

    if unlimited_dims is None:
        unlimited_dims = {}

    for node in dt.subtree:
        at_root = node is dt
        ds = node.to_dataset(inherit=write_inherited_coords or at_root)
        group_path = None if at_root else "/" + node.relative_to(dt)
        ds.to_netcdf(
            target,
            group=group_path,
            mode=mode,
            encoding=encoding.get(node.path),
            unlimited_dims=unlimited_dims.get(node.path),
            engine=engine,
            format=format,
            compute=compute,
            **kwargs,
        )
        mode = "a"

    if filepath is None:
        assert isinstance(target, io.BytesIO)
        return target.getbuffer()

    return None


def _datatree_to_zarr(
    dt: DataTree,
    store: ZarrStoreLike,
    mode: ZarrWriteModes = "w-",
    encoding: Mapping[str, Any] | None = None,
    consolidated: bool = True,
    group: str | None = None,
    write_inherited_coords: bool = False,
    compute: bool = True,
    **kwargs,
):
    """Implementation of `DataTree.to_zarr`."""

    from zarr import consolidate_metadata

    if group is not None:
        raise NotImplementedError(
            "specifying a root group for the tree has not been implemented"
        )

    if "append_dim" in kwargs:
        raise NotImplementedError(
            "specifying ``append_dim`` with ``DataTree.to_zarr`` has not been implemented"
        )

    if encoding is None:
        encoding = {}

    # In the future, we may want to expand this check to insure all the provided encoding
    # options are valid. For now, this simply checks that all provided encoding keys are
    # groups in the datatree.
    if set(encoding) - set(dt.groups):
        raise ValueError(
            f"unexpected encoding group name(s) provided: {set(encoding) - set(dt.groups)}"
        )

    for node in dt.subtree:
        at_root = node is dt
        ds = node.to_dataset(inherit=write_inherited_coords or at_root)
        group_path = None if at_root else "/" + node.relative_to(dt)
        ds.to_zarr(
            store,
            group=group_path,
            mode=mode,
            encoding=encoding.get(node.path),
            consolidated=False,
            compute=compute,
            **kwargs,
        )
        if "w" in mode:
            mode = "a"

    if consolidated:
        consolidate_metadata(store)
