from __future__ import annotations
import functools

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


class DatasetNode(TreeNode):
    """
    A tree node, but optionally containing data in the form of an xarray.Dataset.

    Attempts to present all of the API of xarray.Dataset, but methods are wrapped to also update all child nodes.
    """

    # TODO should this instead be a subclass of Dataset?

    # TODO add any other properties (maybe dask ones?)
    _DS_PROPERTIES = ['variables', 'attrs', 'encoding', 'dims', 'sizes']

    # TODO add all the other methods to dispatch
    _DS_METHODS_TO_DISPATCH = ['isel', 'sel', 'min', 'max', '__array_ufunc__']

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
        for property_name in self._DS_PROPERTIES:
            ds_property = getattr(self.ds, property_name)
            setattr(self, property_name, ds_property)

        # Enable dataset API methods
        for method_name in self._DS_METHODS_TO_DISPATCH:
            ds_method = getattr(Dataset, method_name)
            self._dispatch_to_children(ds_method)

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
        return self.ds is None

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
            if key in self.ds:
                # hashable variable
                return self.ds[key]
            else:
                # hashable child name (or path-like)
                return self.get(key)
        else:
            # iterable of hashables
            first_key, *_ = key
            if first_key in self.children:
                # iterable of child tags
                return self.get(key)
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
                self.set(path=key, value=value)
            elif isinstance(value, Dataset):
                # TODO fix this splitting up of path
                *path_to_new_node, node_name = key
                new_node = DatasetNode(name=node_name, data=value, parent=self)
                self.set(path=key, value=new_node)
            else:
                raise TypeError("Can only assign values of type TreeNode, Dataset, DataArray, or Variable, "
                                f"not {type(value)}")

    def map_inplace(
        self,
        func: Callable,
        *args: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        """
        Apply a function to the dataset at each child node in the tree, updating data in place.

        Parameters
        ----------
        func : callable
            Function to apply to datasets with signature:
            `func(node.name, node.dataset, *args, **kwargs) -> Dataset`.

            Function will still be applied to any nodes without datasets,
            in which cases the `dataset` argument to `func` will be `None`.
        *args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.
        """

        # TODO if func fails on some node then the previous nodes will still have been updated...

        for node in self._walk_children():
            new_ds = func(node.name, node.ds, *args, **kwargs)
            node.dataset = new_ds

    def map_over_descendants(
        self,
        func: Callable,
        *args: Iterable[Any],
        **kwargs: Any,
    ) -> Iterable[Any]:
        """
        Apply a function to the dataset at each node in the tree, returning a generator
        of all the results.

        Parameters
        ----------
        func : callable
            Function to apply to datasets with signature:
            `func(node.name, node.dataset, *args, **kwargs) -> None or return value`.

            Function will still be applied to any nodes without datasets,
            in which cases the `dataset` argument to `func` will be `None`.
        *args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.

        Returns
        -------
        applied : Iterable[Any]
            Generator of results from applying ``func`` to the dataset at each node.
        """
        for node in self._walk_children():
            yield func(node.name, node.ds, *args, **kwargs)

    # TODO map applied ufuncs over all leaves

    # TODO make this public API so that it could be used in a future @register_datatree_accessor example?
    @classmethod
    def _dispatch_to_children(cls, method: Callable) -> None:
        """Wrap such that when method is called on this instance it is also called on children."""
        _dispatching_method = functools.partial(cls.map_inplace, func=method)
        # TODO update method docstrings accordingly
        setattr(cls, method.__name__, _dispatching_method)

    def __str__(self):
        return f"DatasetNode('{self.name}', data={self.ds})"

    def __repr__(self):
        return f"TreeNode(name='{self.name}', data={str(self.ds)}, parent={str(self.parent)}, children={[str(c) for c in self.children]})"

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

        To assign data to the root node of the tree use an empty string as the path.
    name : Hashable, optional
        Name for the root node of the tree. Default is "root"
    """

    # TODO Add attrs dict by inheriting from xarray.core.common.AttrsAccessMixin

    # TODO Some way of sorting children by depth

    # TODO Consistency in copying vs updating objects

    # TODO ipython autocomplete for child nodes

    def __init__(
        self,
        data_objects: Dict[PathType, Union[Dataset, DataArray, DatasetNode, None]] = None,
        name: Hashable = "root",
    ):
        root_data = data_objects.pop("", None)
        super().__init__(name=name, data=root_data, parent=None, children=None)

        # TODO re-implement using anytree.DictImporter?
        if data_objects:
            # Populate tree with children determined from data_objects mapping
            for path in sorted(data_objects):
                self._set_item(path, data_objects[path], allow_overwrite=False, new_nodes_along_path=True)

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
