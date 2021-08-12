from __future__ import annotations

import functools

from typing import Sequence, Tuple, Mapping, Hashable, Union, List, Any, Callable, Iterable

import anytree

from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray
from xarray.core.combine import merge
from xarray.core import dtypes


PathType = Union[Hashable, Sequence[Hashable]]


class TreeNode(anytree.NodeMixin):
    """
    Base class representing a node of a tree, with methods for traversing and altering the tree.

    Depends on the anytree library for all tree traversal methods, but the parent class is fairly small
    so could be easily reimplemented to avoid a hard dependency.
    """

    _resolver = anytree.Resolver('name')

    def __init__(
        self,
        name: Hashable,
        parent: TreeNode = None,
        children: Iterable[TreeNode] = None,
    ):

        self.name = name
        self.parent = parent
        if children:
            self.children = children

    def __str__(self):
        return f"TreeNode('{self.name}')"

    def __repr__(self):
        return f"TreeNode(name='{self.name}', parent={str(self.parent)}, children={[str(c) for c in self.children]})"

    def _pre_attach(self, parent: TreeNode) -> None:
        """
        Method which super NodeMixin class calls before setting parent,
        here used to prevent children with duplicate names.
        """
        if self.name in list(c.name for c in parent.children):
            raise KeyError(f"parent {str(parent)} already has a child named {self.name}")

    def _pre_attach_children(self, children: Iterable[TreeNode]) -> None:
        """
        Method which super NodeMixin class calls before setting children,
        here used to prevent children with duplicate names.
        """
        # TODO test this
        childrens_names = (c.name for c in children)
        if len(set(childrens_names)) < len(list(childrens_names)):
            raise KeyError(f"Cannot add multiple children with the same name to parent {str(self)}")

    def add_child(self, child: TreeNode) -> None:
        """Add a single child node below this node, without replacement."""
        if child.name not in list(c.name for c in self.children):
            child.parent = self
        else:
            raise KeyError(f"Node already has a child named {child.name}")

    @classmethod
    def _tuple_or_path_to_path(cls, address: PathType) -> str:
        if isinstance(address, str):
            return address
        elif isinstance(address, tuple):
            return cls.separator.join(tag for tag in address)
        else:
            raise ValueError(f"{address} is not a valid form of path")

    def get(self, path: PathType) -> TreeNode:
        """
        Access node of the tree lying at the given path.

        Raises a KeyError if not found.

        Parameters
        ----------
        path :
            Path names can be given as unix-like paths, or as tuples of strings
            (where each string is known as a single "tag").

        Returns
        -------
        node
        """

        p = self._tuple_or_path_to_path(path)

        return anytree.Resolver('name').get(self, p)

    def set(self, path: PathType, value: Union[TreeNode, Dataset, DataArray]) -> None:
        """
        Set a node on the tree, overwriting anything already present at that path.

        The new value can be an array or a DataTree, in which case it forms a new node of the tree.

        Paths are specified relative to the node on which this method was called.

        Parameters
        ----------
        path : Union[Hashable, Sequence[Hashable]]
            Path names can be given as unix-like paths, or as tuples of strings (where each string
            is known as a single "tag").
        value : Union[TreeNOde, Dataset, DataArray, None]
        """
        self._set_item(path=path, value=value, new_nodes_along_path=True, allow_overwrite=True)

    def _set_item(self, path: PathType, value: Union[TreeNode, Dataset, DataArray, None],
                  new_nodes_along_path: bool, allow_overwrite: bool) -> None:

        p = self._tuple_or_path_to_path(path)

        # TODO: Check that dimensions/coordinates are compatible with adjacent nodes?

        if not isinstance(value, (TreeNode, Dataset, DataArray)):
            raise TypeError("Can only set new nodes to TreeNode, Dataset, or DataArray instances, not "
                            f"{type(value.__name__)}")

        # Walk to location of new node, creating node objects as we go if necessary
        path = self._tuple_or_path_to_path(path)
        *tags, last_tag = path.split(self.separator)
        parent = self
        for tag in tags:
            # TODO will this mutation within a for loop actually work?
            if tag not in parent.children:
                if new_nodes_along_path:
                    self.add_child(TreeNode(name=tag, parent=parent))
                else:
                    # TODO Should this also be before we walk?
                    raise KeyError(f"Cannot reach new node at path {path}: "
                                   f"parent {parent} has no child {tag}")
            parent = list(self.children)[tag]

        # Deal with anything existing at this location
        if last_tag in parent.children:
            if allow_overwrite:
                child = list(parent.children)[last_tag]
                child.parent = None
                del child
            else:
                # TODO should this be before we walk to the new node?
                raise KeyError(f"Cannot set item at {path} whilst that path already points to a "
                               f"{type(parent.get(last_tag))} object")

        # Create new child node and set at this location
        if value is None:
            new_child = TreeNode(name=last_tag, parent=parent)
        elif isinstance(value, (Dataset, DataArray)):
            new_child = TreeNode(name=last_tag, parent=parent)
            new_child.ds = value
        elif isinstance(value, TreeNode):
            new_child = value
            new_child.parent = parent
        else:
            raise TypeError

    def glob(self, path: str):
        return self._resolver.glob(self, path)

    @property
    def tags(self) -> Tuple[Hashable]:
        """All tags, returned in order starting from the root node"""
        return tuple(self.path.split(self.separator))

    @tags.setter
    def tags(self, value):
        raise AttributeError(f"tags cannot be set, except via changing the children and/or parent of a node.")


class DatasetNode(TreeNode):
    """
    A tree node, but optionally containing data in the form of an xarray.Dataset.

    Also implements xarray.Dataset methods, but wrapped to update all child nodes too.
    """

    # TODO add all the other methods to dispatch
    _DS_METHODS_TO_DISPATCH = ['isel', 'sel', 'min', 'max', '__array_ufunc__']

    def __init__(
        self,
        data: Dataset = None,
        name: Hashable = None,
        parent: TreeNode = None,
        children: List[TreeNode] = None,
    ):
        super().__init__(name=name, parent=parent, children=children)
        self.ds = data

        # Enable dataset API methods
        for method_name in self._DS_METHODS_TO_DISPATCH:
            ds_method = getattr(Dataset, method_name)
            self._dispatch_to_children(ds_method)

    @property
    def ds(self) -> Dataset:
        return self._ds

    @ds.setter
    def ds(self, data: Union[Dataset, DataArray] = None):
        if not isinstance(data, (Dataset, DataArray)) or data is not None:
            raise TypeError(f"{type(data)} object is not an xarray Dataset or DataArray")
        if isinstance(data, DataArray):
            data = data.to_dataset()
        self._ds = data

    def has_data(self):
        return self.ds is None

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
        for node in self._walk_children():
            new_ds = func(node.name, node.ds, *args, **kwargs)
            node.dataset = new_ds

    def map(
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

    def _dispatch_to_children(self, method: Callable) -> None:
        """Wrap such that when method is called on this instance it is also called on children."""
        _dispatching_method = functools.partial(self.map_inplace, func=method)
        # TODO update method docstrings accordingly
        setattr(self, method.__name__, _dispatching_method)

    def _node_repr(self, indent_depth: int) -> str:
        indent_str = "|" + indent_depth * "    |" + "-- "
        node_repr = "\n" + indent_str + str(self.name)

        if self.ds is not None:
            # TODO indent every line properly?
            node_repr += "\n" + indent_str + f"{repr(self.ds)[17:]}"

        for child in self.children:
            node_repr += child._node_repr(indent_depth+1)

        return node_repr


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

        To assign data to the root node of the tree use "/" or "" as the path.
    """

    # TODO Add attrs dict by inheriting from xarray.core.common.AttrsAccessMixin

    # TODO Some way of sorting children by depth

    # TODO Consistency in copying vs updating objects

    # TODO ipython autocomplete for child nodes

    def __init__(
        self,
        data_objects: Mapping[PathType, Union[Dataset, DataArray, DatasetNode]] = None,
    ):
        super().__init__(ds=None, name=None, parent=None, children=[])

        # TODO implement using anytree.DictImporter

        # Populate tree with children determined from data_objects mapping
        for path, obj in data_objects.items():
            self._set_item(path, obj, allow_overwrites=False, new_nodes_along_path=True)

    def __repr__(self) -> str:
        type_str = "<xtree.DataTree>"
        tree_str = self._node_repr(indent_depth=0)
        # TODO add attrs dict to the repr
        return type_str + tree_str

    def get_all(self, *tags: Hashable) -> DataTree:
        """
        Return a DataTree containing the stored objects whose path contains all of the given tags,
        where the tags can be present in any order.
        """
        matching_children = {c.tags: c.get(tags) for c in self._walk_children()
                             if all(tag in c.tags for tag in tags)}
        return DataTree(data_objects=matching_children)

    def get_any(self, *tags: Hashable) -> DataTree:
        """
        Return a DataTree containing the stored objects whose path contains any of the given tags.
        """
        matching_children = {c.tags: c.get(tags) for c in self._walk_children()
                             if any(tag in c.tags for tag in tags)}
        return DataTree(data_objects=matching_children)

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

    def to_netcdf(self, filename: str):
        from .io import _datatree_to_netcdf

        _datatree_to_netcdf(self, filename)

    def plot(self):
        raise NotImplementedError
