from typing import Sequence

from netCDF4 import Dataset as nc_dataset

from xarray import open_dataset

from .datatree import DataTree, PathType


def _get_group_names(file):
    rootgrp = nc_dataset("test.nc", "r", format="NETCDF4")

    def walktree(top):
        yield top.groups.values()
        for value in top.groups.values():
            yield from walktree(value)

    groups = []
    for children in walktree(rootgrp):
        for child in children:
            # TODO include parents in saved path
            groups.append(child.name)

    rootgrp.close()
    return groups


def open_datatree(filename_or_obj, engine=None, chunks=None, **kwargs) -> DataTree:
    """
    Open and decode a dataset from a file or file-like object, creating one DataTree node
    for each group in the file.
    """

    # TODO find all the netCDF groups in the file
    file_groups = _get_group_names(filename_or_obj)

    # Populate the DataTree with the groups
    groups_and_datasets = {group_path: open_dataset(engine=engine, chunks=chunks, **kwargs)
                           for group_path in file_groups}
    return DataTree(data_objects=groups_and_datasets)


def open_mfdatatree(filepaths, rootnames: Sequence[PathType] = None, engine=None, chunks=None, **kwargs) -> DataTree:
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
        dt = open_datatree(file, engine=engine, chunks=chunks, **kwargs)
        full_tree._set_item(path=root, value=dt, new_nodes_along_path=True, allow_overwrites=False)

    return full_tree


def _datatree_to_netcdf(dt: DataTree, path_or_file: str):
    raise NotImplementedError
