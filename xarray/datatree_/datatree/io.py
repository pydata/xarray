import os
from typing import Dict, Sequence

import netCDF4
from xarray import open_dataset

from .datatree import DataNode, DataTree, PathType


def _ds_or_none(ds):
    """return none if ds is empty"""
    if any(ds.coords) or any(ds.variables) or any(ds.attrs):
        return ds
    return None


def _open_group_children_recursively(filename, node, ncgroup, chunks, **kwargs):
    for g in ncgroup.groups.values():

        # Open and add this node's dataset to the tree
        name = os.path.basename(g.path)
        ds = open_dataset(filename, group=g.path, chunks=chunks, **kwargs)
        ds = _ds_or_none(ds)
        child_node = DataNode(name, ds)
        node.add_child(child_node)

        _open_group_children_recursively(filename, node[name], g, chunks, **kwargs)


def open_datatree(filename: str, chunks: Dict = None, **kwargs) -> DataTree:
    """
    Open and decode a dataset from a file or file-like object, creating one Tree node for each group in the file.

    Parameters
    ----------
    filename
    chunks

    Returns
    -------
    DataTree
    """

    with netCDF4.Dataset(filename, mode="r") as ncfile:
        ds = open_dataset(filename, chunks=chunks, **kwargs)
        tree_root = DataTree(data_objects={"root": ds})
        _open_group_children_recursively(filename, tree_root, ncfile, chunks, **kwargs)
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


def _create_empty_group(filename, group, mode):
    with netCDF4.Dataset(filename, mode=mode) as rootgrp:
        rootgrp.createGroup(group)


def _datatree_to_netcdf(
    dt: DataTree,
    filepath,
    mode: str = "w",
    encoding=None,
    unlimited_dims=None,
    **kwargs
):

    if kwargs.get("format", None) not in [None, "NETCDF4"]:
        raise ValueError("to_netcdf only supports the NETCDF4 format")

    if kwargs.get("engine", None) not in [None, "netcdf4", "h5netcdf"]:
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
            _create_empty_group(filepath, group_path, mode)
        else:
            ds.to_netcdf(
                filepath,
                group=group_path,
                mode=mode,
                encoding=_maybe_extract_group_kwargs(encoding, dt.pathstr),
                unlimited_dims=_maybe_extract_group_kwargs(unlimited_dims, dt.pathstr),
                **kwargs
            )
        mode = "a"
