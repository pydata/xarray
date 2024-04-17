from xarray.core.datatree import DataTree


def _get_nc_dataset_class(engine):
    if engine == "netcdf4":
        from netCDF4 import Dataset
    elif engine == "h5netcdf":
        from h5netcdf.legacyapi import Dataset
    elif engine is None:
        try:
            from netCDF4 import Dataset
        except ImportError:
            from h5netcdf.legacyapi import Dataset
    else:
        raise ValueError(f"unsupported engine: {engine}")
    return Dataset


def _create_empty_netcdf_group(filename, group, mode, engine):
    ncDataset = _get_nc_dataset_class(engine)

    with ncDataset(filename, mode=mode) as rootgrp:
        rootgrp.createGroup(group)


def _datatree_to_netcdf(
    dt: DataTree,
    filepath,
    mode: str = "w",
    encoding=None,
    unlimited_dims=None,
    **kwargs,
):
    if kwargs.get("format", None) not in [None, "NETCDF4"]:
        raise ValueError("to_netcdf only supports the NETCDF4 format")

    engine = kwargs.get("engine", None)
    if engine not in [None, "netcdf4", "h5netcdf"]:
        raise ValueError("to_netcdf only supports the netcdf4 and h5netcdf engines")

    if kwargs.get("group", None) is not None:
        raise NotImplementedError(
            "specifying a root group for the tree has not been implemented"
        )

    if not kwargs.get("compute", True):
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

    if unlimited_dims is None:
        unlimited_dims = {}

    for node in dt.subtree:
        ds = node.ds
        group_path = node.path
        if ds is None:
            _create_empty_netcdf_group(filepath, group_path, mode, engine)
        else:
            ds.to_netcdf(
                filepath,
                group=group_path,
                mode=mode,
                encoding=encoding.get(node.path),
                unlimited_dims=unlimited_dims.get(node.path),
                **kwargs,
            )
        mode = "r+"


def _create_empty_zarr_group(store, group, mode):
    import zarr

    root = zarr.open_group(store, mode=mode)
    root.create_group(group, overwrite=True)


def _datatree_to_zarr(
    dt: DataTree,
    store,
    mode: str = "w-",
    encoding=None,
    consolidated: bool = True,
    **kwargs,
):
    from zarr.convenience import consolidate_metadata

    if kwargs.get("group", None) is not None:
        raise NotImplementedError(
            "specifying a root group for the tree has not been implemented"
        )

    if not kwargs.get("compute", True):
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

    for node in dt.subtree:
        ds = node.ds
        group_path = node.path
        if ds is None:
            _create_empty_zarr_group(store, group_path, mode)
        else:
            ds.to_zarr(
                store,
                group=group_path,
                mode=mode,
                encoding=encoding.get(node.path),
                consolidated=False,
                **kwargs,
            )
        if "w" in mode:
            mode = "a"

    if consolidated:
        consolidate_metadata(store)
