from __future__ import annotations
import functools
import textwrap

from typing import Mapping, Hashable, Union, List, Any, Callable, Iterable, Dict

import anytree

from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray
from xarray.core.variable import Variable
from xarray.core.combine import merge
from xarray.core import dtypes, utils

from .treenode import TreeNode, PathType

"""
The structure of a populated Datatree looks like this in terms of classes: 

DataTree("root name")
|-- DatasetNode("weather")
|   |-- DatasetNode("temperature")
|   |   |-- DataArrayNode("sea_surface_temperature")
|   |   |-- DataArrayNode("dew_point_temperature")
|   |-- DataArrayNode("wind_speed")
|   |-- DataArrayNode("pressure")
|-- DatasetNode("satellite image")
|   |-- DatasetNode("infrared")
|   |   |-- DataArrayNode("near_infrared")
|   |   |-- DataArrayNode("far_infrared")
|   |-- DataArrayNode("true_colour")
|-- DataTreeNode("topography")
|   |-- DatasetNode("elevation")
|   |   |-- DataArrayNode("height_above_sea_level")
|-- DataArrayNode("population")
"""


def _map_over_subtree(tree, func, *args, **kwargs):
    """Internal function which maps func over every node in tree, returning a tree of the results."""

    out_tree = DataTree(name=tree.name, data_objects={})

    for node in tree.subtree_nodes:
        relative_path = tree.path.replace(node.path, '')

        if node.has_data:
            result = func(node.ds, *args, **kwargs)
        else:
            result = None

        out_tree[relative_path] = DatasetNode(name=node.name, data=result)

    return out_tree


def map_over_subtree(func):
    """
    Decorator which turns a function which acts on (and returns) single Datasets into one which acts on DataTrees.

    Applies a function to every dataset in this subtree, returning a new tree which stores the results.

    The function will be applied to any dataset stored in this node, as well as any dataset stored in any of the
    descendant nodes. The returned tree will have the same structure as the original subtree.

    func needs to return a Dataset in order to rebuild the subtree.

    Parameters
    ----------
    func : callable
        Function to apply to datasets with signature:
        `func(node.ds, *args, **kwargs) -> Dataset`.

        Function will not be applied to any nodes without datasets.
    *args : tuple, optional
        Positional arguments passed on to `func`.
    **kwargs : Any
        Keyword arguments passed on to `func`.

    Returns
    -------
    mapped : callable
        Wrapped function which returns tree created from results of applying ``func`` to the dataset at each node.

    See also
    --------
    DataTree.map_over_subtree
    DataTree.map_over_subtree_inplace
    """
    return functools.wraps(func)(_map_over_subtree)


class DatasetNode(TreeNode):
    """
    A tree node, but optionally containing data in the form of an xarray.Dataset.

    Attempts to present the API of xarray.Dataset, but methods are wrapped to also update all the tree's child nodes.
    """

    # TODO should this instead be a subclass of Dataset?

    # TODO add any other properties (maybe dask ones?)
    _DS_PROPERTIES = ['variables', 'attrs', 'encoding', 'dims', 'sizes']

    # TODO add all the other methods to dispatch
    _DS_METHODS_TO_MAP_OVER_SUBTREES = ['isel', 'sel', 'min', 'max', 'mean', '__array_ufunc__']
    _MAPPED_DOCSTRING_ADDENDUM = textwrap.fill("This method was copied from xarray.Dataset, but has been altered to "
                                               "call the method on the Datasets stored in every node of the subtree. "
                                               "See the datatree.map_over_subtree decorator for more details.",
                                               width=117)

    # TODO currently allows self.ds = None, should we instead always store at least an empty Dataset?

    def __init__(
        self,
        name: Hashable,
        data: Dataset = None,
        parent: TreeNode = None,
        children: List[TreeNode] = None,
    ):
        super().__init__(name=name, parent=parent, children=children)
        self.ds = data

        # Expose properties of wrapped Dataset
        # TODO if self.ds = None what will happen?
        for property_name in self._DS_PROPERTIES:
            ds_property = getattr(Dataset, property_name)
            setattr(self, property_name, ds_property)

        # Enable dataset API methods
        for method_name in self._DS_METHODS_TO_MAP_OVER_SUBTREES:
            # Expose Dataset method, but wrapped to map over whole subtree
            ds_method = getattr(Dataset, method_name)
            setattr(self, method_name, map_over_subtree(ds_method))

            # Add a line to the method's docstring explaining how it's been mapped
            ds_method_docstring = getattr(Dataset, f'{method_name}').__doc__
            if ds_method_docstring is not None:
                updated_method_docstring = ds_method_docstring.replace('\n', self._MAPPED_DOCSTRING_ADDENDUM, 1)
                setattr(self, f'{method_name}.__doc__', updated_method_docstring)

    @property
    def ds(self) -> Dataset:
        return self._ds

    @ds.setter
    def ds(self, data: Union[Dataset, DataArray] = None):
        if not isinstance(data, (Dataset, DataArray)) and data is not None:
            raise TypeError(f"{type(data)} object is not an xarray Dataset, DataArray, or None")
        if isinstance(data, DataArray):
            data = data.to_dataset()
        self._ds = data

    @property
    def has_data(self):
        return self.ds is not None

    def __getitem__(self, key: Union[PathType, Hashable, Mapping, Any]) -> Union[TreeNode, Dataset, DataArray]:
        """
        Access either child nodes, or variables or coordinates stored in this node.

        Variable or coordinates of the contained dataset will be returned as a :py:class:`~xarray.DataArray`.
        Indexing with a list of names will return a new ``Dataset`` object.

        Parameters
        ----------
        key :
            If a path to child node then names can be given as unix-like paths, or as tuples of strings
            (where each string is known as a single "tag").

        """
        # Either:
        if utils.is_dict_like(key):
            # dict-like to variables
            return self.ds[key]
        elif utils.hashable(key):
            print(self.has_data)
            if self.has_data and key in self.ds.data_vars:
                # hashable variable
                return self.ds[key]
            else:
                # hashable child name (or path-like)
                return self.get_node(key)
        else:
            # iterable of hashables
            first_key, *_ = key
            if first_key in self.children:
                # iterable of child tags
                return self.get_node(key)
            else:
                # iterable of variable names
                return self.ds[key]

    def __setitem__(
        self,
        key: Union[Hashable, List[Hashable], Mapping, PathType],
        value: Union[TreeNode, Dataset, DataArray, Variable]
    ) -> None:
        """
        Add either a child node or an array to this node.

        Parameters
        ----------
        key
            Either a path-like address for a new node, or the name of a new variable.
        value
            If a node class or a Dataset, it will be added as a new child node.
            If an single array (i.e. DataArray, Variable), it will be added to the underlying Dataset.
        """
        if utils.is_dict_like(key):
            # TODO xarray.Dataset accepts other possibilities, how do we exactly replicate the behaviour?
            raise NotImplementedError
        else:
            if isinstance(value, (DataArray, Variable)):
                self.ds[key] = value
            elif isinstance(value, TreeNode):
                self.set_node(path=key, node=value)
            elif isinstance(value, Dataset):
                # TODO fix this splitting up of path
                *path_to_new_node, node_name = key
                new_node = DatasetNode(name=node_name, data=value, parent=self)
                self.set_node(path=key, node=new_node)
            else:
                raise TypeError("Can only assign values of type TreeNode, Dataset, DataArray, or Variable, "
                                f"not {type(value)}")

    def map_over_subtree(
            self,
            func: Callable,
            *args: Iterable[Any],
            **kwargs: Any,
    ) -> DataTree:
        """
        Apply a function to every dataset in this subtree, returning a new tree which stores the results.

        The function will be applied to any dataset stored in this node, as well as any dataset stored in any of the
        descendant nodes. The returned tree will have the same structure as the original subtree.

        func needs to return a Dataset in order to rebuild the subtree.

        Parameters
        ----------
        func : callable
            Function to apply to datasets with signature:
            `func(node.ds, *args, **kwargs) -> Dataset`.

            Function will not be applied to any nodes without datasets.
        *args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.

        Returns
        -------
        subtree : DataTree
            Subtree containing results from applying ``func`` to the dataset at each node.
        """
        # TODO this signature means that func has no way to know which node it is being called upon - change?

        return _map_over_subtree(self, func, *args, **kwargs)

    def map_over_subtree_inplace(
        self,
        func: Callable,
        *args: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        """
        Apply a function to every dataset in this subtree, updating data in place.

        Parameters
        ----------
        func : callable
            Function to apply to datasets with signature:
            `func(node.ds, *args, **kwargs) -> Dataset`.

            Function will not be applied to any nodes without datasets,
        *args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.
        """

        # TODO if func fails on some node then the previous nodes will still have been updated...

        for node in self.subtree_nodes:
            if node.has_data:
                node.ds = func(node.ds, *args, **kwargs)

    # TODO map applied ufuncs over all leaves

    def __str__(self):
        return f"DatasetNode('{self.name}', data={type(self.ds)})"

    def __repr__(self):
        # TODO update this to indent nicely
        return f"TreeNode(\n" \
               f"    name='{self.name}',\n" \
               f"    data={str(self.ds)},\n" \
               f"    parent={str(self.parent)},\n" \
               f"    children={tuple(str(c) for c in self.children)}\n" \
               f")"

    def render(self):
        """Print tree structure, including any data stored at each node."""
        for pre, fill, node in anytree.RenderTree(self):
            print(f"{pre}DatasetNode('{self.name}')")
            for ds_line in repr(node.ds)[1:]:
                print(f"{fill}{ds_line}")

    # TODO re-implement using anytree findall function?
    def get_all(self, *tags: Hashable) -> DataTree:
        """
        Return a DataTree containing the stored objects whose path contains all of the given tags,
        where the tags can be present in any order.
        """
        matching_children = {c.tags: c.get_node(tags) for c in self.descendants
                             if all(tag in c.tags for tag in tags)}
        return DataTree(data_objects=matching_children)

    # TODO re-implement using anytree find function?
    def get_any(self, *tags: Hashable) -> DataTree:
        """
        Return a DataTree containing the stored objects whose path contains any of the given tags.
        """
        matching_children = {c.tags: c.get_node(tags) for c in self.descendants
                             if any(tag in c.tags for tag in tags)}
        return DataTree(data_objects=matching_children)


class DataTree(DatasetNode):
    """
    A tree-like hierarchical collection of xarray objects.

    Parameters
    ----------
    data_objects : dict-like, optional
        A mapping from path names to xarray.Dataset, xarray.DataArray, or xtree.DataTree objects.

        Path names can be given as unix-like paths, or as tuples of strings (where each string
        is known as a single "tag"). If path names containing more than one tag are given, new
        tree nodes will be constructed as necessary.

        To assign data to the root node of the tree {name} as the path.
    name : Hashable, optional
        Name for the root node of the tree. Default is "root"
    """

    # TODO Add attrs dict

    # TODO attribute-like access for both vars and child nodes (by inheriting from xarray.core.common.AttrsAccessMixin?)

    # TODO ipython autocomplete for child nodes

    # TODO Some way of sorting children by depth

    # TODO Consistency in copying vs updating objects

    def __init__(
        self,
        data_objects: Dict[PathType, Union[Dataset, DataArray]] = None,
        name: Hashable = "root",
    ):
        if data_objects is not None:
            root_data = data_objects.pop(name, None)
        else:
            root_data = None
        super().__init__(name=name, data=root_data, parent=None, children=None)

        # TODO re-implement using anytree.DictImporter?
        if data_objects:
            # Populate tree with children determined from data_objects mapping
            for path, data in data_objects.items():
                # Determine name of new node
                path = self._tuple_or_path_to_path(path)
                if self.separator in path:
                    node_path, node_name = path.rsplit(self.separator, maxsplit=1)
                else:
                    node_path, node_name = '/', path

                # Create and set new node
                new_node = DatasetNode(name=node_name, data=data)
                self.set_node(node_path, new_node, allow_overwrite=False, new_nodes_along_path=True)
                new_node = self.get_node(path)
                new_node[path] = data

    # TODO do we need a watch out for if methods intended only for root nodes are calle on non-root nodes?

    @property
    def chunks(self):
        raise NotImplementedError

    def chunk(self):
        raise NotImplementedError

    def merge(self, datatree: DataTree) -> DataTree:
        """Merge all the leaves of a second DataTree into this one."""
        raise NotImplementedError

    def merge_child_nodes(self, *paths, new_path: PathType) -> DataTree:
        """Merge a set of child nodes into a single new node."""
        raise NotImplementedError

    def merge_child_datasets(
        self,
        *paths: PathType,
        compat: str = "no_conflicts",
        join: str = "outer",
        fill_value: Any = dtypes.NA,
        combine_attrs: str = "override",
    ) -> Dataset:
        """Merge the datasets at a set of child nodes and return as a single Dataset."""
        datasets = [self.get(path).ds for path in paths]
        return merge(datasets, compat=compat, join=join, fill_value=fill_value, combine_attrs=combine_attrs)

    def as_dataarray(self) -> DataArray:
        return self.ds.as_dataarray()

    @property
    def groups(self):
        """Return all netCDF4 groups in the tree, given as a tuple of path-like strings."""
        return tuple(node.path for node in self.subtree_nodes)

    def to_netcdf(self, filename: str):
        from .io import _datatree_to_netcdf

        _datatree_to_netcdf(self, filename)

    def plot(self):
        raise NotImplementedError
