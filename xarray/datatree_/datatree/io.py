import pathlib
from typing import Sequence

from xarray import open_dataset

from .datatree import DataTree, PathType


def _ds_or_none(ds):
    """return none if ds is empty"""
    if any(ds.coords) or any(ds.variables) or any(ds.attrs):
        return ds
    return None


def _iter_zarr_groups(root, parrent=""):
    parrent = pathlib.Path(parrent)
    for path, group in root.groups():
        gpath = parrent / path
        yield str(gpath)
        yield from _iter_zarr_groups(group, parrent=gpath)


def _iter_nc_groups(root, parrent=""):
    parrent = pathlib.Path(parrent)
    for path, group in root.groups.items():
        gpath = parrent / path
        yield str(gpath)
        yield from _iter_nc_groups(group, parrent=gpath)


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


def _open_datatree_netcdf(filename: str, **kwargs) -> DataTree:
    ncDataset = _get_nc_dataset_class(kwargs.get("engine", None))

    with ncDataset(filename, mode="r") as ncds:
        ds = open_dataset(filename, **kwargs).pipe(_ds_or_none)
        tree_root = DataTree.from_dict(data_objects={"root": ds})
        for key in _iter_nc_groups(ncds):
            tree_root[key] = open_dataset(filename, group=key, **kwargs).pipe(
                _ds_or_none
            )
    return tree_root


def _open_datatree_zarr(store, **kwargs) -> DataTree:
    import zarr

    with zarr.open_group(store, mode="r") as zds:
        ds = open_dataset(store, engine="zarr", **kwargs).pipe(_ds_or_none)
        tree_root = DataTree.from_dict(data_objects={"root": ds})
        for key in _iter_zarr_groups(zds):
            try:
                tree_root[key] = open_dataset(
                    store, engine="zarr", group=key, **kwargs
                ).pipe(_ds_or_none)
            except zarr.errors.PathNotFoundError:
                tree_root[key] = None
    return tree_root


def open_mfdatatree(
    filepaths, rootnames: Sequence[PathType] = None, chunks=None, **kwargs
) -> DataTree:
    """
    Open multiple files as a single DataTree.

    Groups found in each file will be merged at the root level, unless rootnames are specified,
    which will then be used to organise the Tree instead.
    """
    if rootnames is None:
        rootnames = ["/" for _ in filepaths]
    elif len(rootnames) != len(filepaths):
        raise ValueError

    full_tree = DataTree()

    for file, root in zip(filepaths, rootnames):
        dt = open_datatree(file, chunks=chunks, **kwargs)
        full_tree.set_node(
            path=root, node=dt, new_nodes_along_path=True, allow_overwrite=False
        )

    return full_tree


def _maybe_extract_group_kwargs(enc, group):
    try:
        return enc[group]
    except KeyError:
        return None


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

    if unlimited_dims is None:
        unlimited_dims = {}

    for node in dt.subtree:
        ds = node.ds
        group_path = node.pathstr.replace(dt.root.pathstr, "")
        if ds is None:
            _create_empty_netcdf_group(filepath, group_path, mode, engine)
        else:

            ds.to_netcdf(
                filepath,
                group=group_path,
                mode=mode,
                encoding=_maybe_extract_group_kwargs(encoding, dt.pathstr),
                unlimited_dims=_maybe_extract_group_kwargs(unlimited_dims, dt.pathstr),
                **kwargs,
            )
        mode = "a"


def _create_empty_zarr_group(store, group, mode):
    import zarr

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

    from zarr.convenience import consolidate_metadata

    if kwargs.get("group", None) is not None:
        raise NotImplementedError(
            "specifying a root group for the tree has not been implemented"
        )

    if not kwargs.get("compute", True):
        raise NotImplementedError("compute=False has not been implemented yet")

    if encoding is None:
        encoding = {}

    for node in dt.subtree:
        ds = node.ds
        group_path = node.pathstr.replace(dt.root.pathstr, "")
        if ds is None:
            _create_empty_zarr_group(store, group_path, mode)
        else:
            ds.to_zarr(
                store,
                group=group_path,
                mode=mode,
                encoding=_maybe_extract_group_kwargs(encoding, dt.pathstr),
                consolidated=False,
                **kwargs,
            )
        if "w" in mode:
            mode = "a"

    if consolidated:
        consolidate_metadata(store)
