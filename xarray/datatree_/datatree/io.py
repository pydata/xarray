from typing import Sequence, Dict
import os

import netCDF4

from xarray import open_dataset

from .datatree import DataTree, DataNode, PathType


def _open_group_children_recursively(filename, node, ncgroup, chunks, **kwargs):
    for g in ncgroup.groups.values():

        # Open and add this node's dataset to the tree
        name = os.path.basename(g.path)
        ds = open_dataset(filename, group=g.path, chunks=chunks, **kwargs)
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

    with netCDF4.Dataset(filename, mode='r') as ncfile:
        ds = open_dataset(filename, chunks=chunks, **kwargs)
        tree_root = DataTree(data_objects={'root': ds})
        _open_group_children_recursively(filename, tree_root, ncfile, chunks, **kwargs)
    return tree_root


def open_mfdatatree(filepaths, rootnames: Sequence[PathType] = None, chunks=None, **kwargs) -> DataTree:
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
        full_tree.set_node(path=root, node=dt, new_nodes_along_path=True, allow_overwrite=False)

    return full_tree


def _datatree_to_netcdf(dt: DataTree, filepath: str):
    raise NotImplementedError
