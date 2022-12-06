from xarray import Dataset, open_dataset

from .datatree import DataTree, NodePath


def _iter_zarr_groups(root, parent="/"):
    parent = NodePath(parent)
    for path, group in root.groups():
        gpath = parent / path
        yield str(gpath)
        yield from _iter_zarr_groups(group, parent=gpath)


def _iter_nc_groups(root, parent="/"):
    parent = NodePath(parent)
    for path, group in root.groups.items():
        gpath = parent / path
        yield str(gpath)
        yield from _iter_nc_groups(group, parent=gpath)


def _get_nc_dataset_class(engine):
    if engine == "netcdf4":
        from netCDF4 import Dataset  # type: ignore
    elif engine == "h5netcdf":
        from h5netcdf.legacyapi import Dataset  # type: ignore
    elif engine is None:
        try:
            from netCDF4 import Dataset
        except ImportError:
            from h5netcdf.legacyapi import Dataset  # type: ignore
    else:
        raise ValueError(f"unsupported engine: {engine}")
    return Dataset


def open_datatree(filename_or_obj, engine=None, **kwargs) -> DataTree:
    """
    Open and decode a dataset from a file or file-like object, creating one Tree node for each group in the file.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like, or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file or Zarr store.
    engine : str, optional
        Xarray backend engine to us. Valid options include `{"netcdf4", "h5netcdf", "zarr"}`.
    kwargs :
        Additional keyword arguments passed to ``xarray.open_dataset`` for each group.

    Returns
    -------
    DataTree
    """

    if engine == "zarr":
        return _open_datatree_zarr(filename_or_obj, **kwargs)
    elif engine in [None, "netcdf4", "h5netcdf"]:
        return _open_datatree_netcdf(filename_or_obj, engine=engine, **kwargs)
    else:
        raise ValueError("Unsupported engine")


def _open_datatree_netcdf(filename: str, **kwargs) -> DataTree:
    ncDataset = _get_nc_dataset_class(kwargs.get("engine", None))

    ds = open_dataset(filename, **kwargs)
    tree_root = DataTree.from_dict({"/": ds})
    with ncDataset(filename, mode="r") as ncds:
        for path in _iter_nc_groups(ncds):
            subgroup_ds = open_dataset(filename, group=path, **kwargs)

            # TODO refactor to use __setitem__ once creation of new nodes by assigning Dataset works again
            node_name = NodePath(path).name
            new_node: DataTree = DataTree(name=node_name, data=subgroup_ds)
            tree_root._set_item(
                path,
                new_node,
                allow_overwrite=False,
                new_nodes_along_path=True,
            )
    return tree_root


def _open_datatree_zarr(store, **kwargs) -> DataTree:
    import zarr  # type: ignore

    zds = zarr.open_group(store, mode="r")
    ds = open_dataset(store, engine="zarr", **kwargs)
    tree_root = DataTree.from_dict({"/": ds})
    for path in _iter_zarr_groups(zds):
        try:
            subgroup_ds = open_dataset(store, engine="zarr", group=path, **kwargs)
        except zarr.errors.PathNotFoundError:
            subgroup_ds = Dataset()

        # TODO refactor to use __setitem__ once creation of new nodes by assigning Dataset works again
        node_name = NodePath(path).name
        new_node: DataTree = DataTree(name=node_name, data=subgroup_ds)
        tree_root._set_item(
            path,
            new_node,
            allow_overwrite=False,
            new_nodes_along_path=True,
        )
    return tree_root


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
    import zarr  # type: ignore

    root = zarr.open_group(store, mode=mode)
    root.create_group(group, overwrite=True)


def _datatree_to_zarr(
    dt: DataTree,
    store,
    mode: str = "w",
    encoding=None,
    consolidated: bool = True,
    **kwargs,
):

    from zarr.convenience import consolidate_metadata  # type: ignore

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
