from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
import functools

from typing import Sequence, Tuple, Mapping, Hashable, Union, List, Any, Callable, Iterable

from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray
from xarray.core.combine import merge
from xarray.core import dtypes


PathType = Union[Hashable, Sequence[Hashable]]


def _path_to_tuple(path: PathType) -> Tuple[Hashable]:
    if isinstance(path, str):
        return path
    else:
        return tuple(Path(path).parts)


class TreeNode(MutableMapping):
    """Base class representing a node of a tree, with methods for traversing the tree."""

    def __init__(
        self,
        name: Hashable,
        parent: TreeNode = None,
        children: List[TreeNode] = None,
    ):

        if children is None:
            children = []

        self._name = name
        self.children = children
        self._parent = None
        self.parent = parent

    @property
    def name(self) -> Hashable:
        """Name tag for this node."""
        return self._name

    @property
    def parent(self) -> Union[TreeNode, None]:
        return self._parent

    @parent.setter
    def parent(self, parent: TreeNode):
        if parent is not None:
            if not isinstance(parent, TreeNode):
                raise TypeError(f"{type(parent)} object is not a valid parent")

            if self._name in [c.name for c in parent._children]:
                raise KeyError(f"Cannot set parent: parent node {parent._name} "
                               f"already has a child node named {self._name}")
            else:
                # If there was an original parent they can no longer have custody
                if self.parent is not None:
                    self.parent.children.remove(self)

                # New parent needs to know it now has a child
                parent.children = parent.children + [self]

        self._parent = parent

    @property
    def children(self) -> List[TreeNode]:
        return self._children

    @children.setter
    def children(self, children: List[TreeNode]):
        if not all(isinstance(c, TreeNode) for c in children):
            raise TypeError(f"children must all be valid tree nodes")
        self._children = children

    def _walk_parents(self) -> DataTree:
        """Walk through this node and its parents."""
        yield self
        node = self._parent
        while node is not None:
            yield node
            node = node._parent

    def root_node(self) -> DataTree:
        """Return the root node in the tree."""
        for node in self._walk_parents():
            pass
        return node

    def _walk_children(self) -> DataTree:
        """Recursively walk through this node and all its child nodes."""
        yield self
        for child in self._children:
            for node in child._walk_children():
                yield node

    def __repr__(self):
        return f"TreeNode(name={self._name}, parent={self._parent}, children={self.children})"

    def add_node(self, name: Hashable, data: Union[DataTree, Dataset, DataArray] = None) -> DataTree:
        """Add a child node immediately below this node, and return the new child node."""
        if isinstance(data, DataTree):
            data.parent = self
            self._children.append(data)
            return data
        else:
            return self._construct(name=name, parent=self, data=data)

    @staticmethod
    def _get_node_depth1(node: DataTree, key: Hashable) -> DataTree:
        if node is None:
            return None
        if key == '..':
            return node._parent
        if key == '.':
            return node
        for child in node._children:
            if key == child._name:
                return child
        return None

    def __delitem__(self, path: PathType):
        for child in self._walk_children():
            del child

    def __iter__(self):
        return iter(c.name for c in self._children)

    def __len__(self):
        return len(self._children)

    def get(self, path: str, default: DataTree = None) -> TreeNode:
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
        node : DataTree
        """

        """Return a node given any relative or absolute UNIX-like path."""
        # TODO rewrite using pathlib?
        if path == '/':
            return self.root_node()
        elif path.startswith('/'):
            node = self.root_node()
            slash, path = path
        else:
            node = self

        for key in path.split('/'):
            node = self._get_node_depth1(node, key)
        if node is None:
            node = default

        return node

    def __getitem__(self, path: PathType) -> DataTree:
        node = self.get(path)
        if node is None:
            raise KeyError(f"Node {path} not found")
        return node

    def set(self, path: PathType, value: Union[TreeNode, Dataset, DataArray]) -> None:
        """
        Add a leaf to the tree, overwriting anything already present at that path.

        The new value can be an array or a DataTree, in which case it forms a new node of the tree.

        Parameters
        ----------
        path : Union[Hashable, Sequence[Hashable]]
            Path names can be given as unix-like paths, or as tuples of strings (where each string
            is known as a single "tag").
        value : Union[DataTree, Dataset, DataArray]
        """
        self._set_item(path=path, value=value, new_nodes_along_path=True, allow_overwrites=True)

    def _set_item(self, path: PathType, value: Union[DataTree, Dataset, DataArray],
                  new_nodes_along_path: bool, allow_overwrites: bool) -> None:
        # TODO: Check that dimensions/coordinates are compatible with adjacent nodes?

        # This check is redundant with checks called in `add_node`, but if we don't do it here
        # then a failed __setitem__ might create a trail of new nodes all the way down
        if not isinstance(value, (DataTree, Dataset)):
            raise TypeError("Can only set new nodes to DataTree or Dataset instances, not "
                             f"{type(value.__name__)}")

        # Walk to location of new node, creating DataTree objects as we go if necessary
        *tags, last_tag = _path_to_tuple(path)
        parent = self
        for tag in tags:
            if tag not in parent.children:
                if new_nodes_along_path:
                    parent = self.add_node(tag)
                else:
                    # TODO Should this also be before we walk?
                    raise KeyError(f"Cannot reach new node at path {path}: "
                                     f"parent {parent} has no child {tag}")
            parent = self._get_node_depth1(parent, tag)

        if last_tag in parent.children:
            if not allow_overwrites:
                # TODO should this be before we walk to the new node?
                raise KeyError(f"Cannot set item at {path} whilst that path already points to a "
                               f"{type(parent.get(last_tag))} object")
            else:
                # TODO Delete any newly-orphaned children
                ...

        parent.add_node(last_tag, data=value)

    def __setitem__(self, path: PathType, value: Union[DataTree, Dataset, DataArray]) -> None:
        """
        Add a leaf to the DataTree, overwriting anything already present at that path.

        The new value can be an array or a DataTree, in which case it forms a new node of the tree.

        Parameters
        ----------
        path : Union[Hashable, Sequence[Hashable]]
            Path names can be given as unix-like paths, or as tuples of strings (where each string
            is known as a single "tag").
        value : Union[DataTree, Dataset, DataArray]
        """
        self._set_item(path=path, value=value, new_nodes_along_path=True, allow_overwrites=True)

    @property
    def tags(self) -> Tuple[Hashable]:
        """All tags, returned in order starting from the root node"""
        return tuple(reversed([node.name for node in self._walk_parents()]))

    @property
    def path(self) -> str:
        """Full path to this node, given as a UNIX-like path."""
        if self._parent is None:
            return '/'
        else:
            return '/'.join(self.tags[-1::-1])


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
