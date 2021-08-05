from __future__ import annotations

from collections import MutableMapping
from pathlib import Path

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


class DataTree(MutableMapping):
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
        data_objects: Mapping[PathType, Union[Dataset, DataArray]] = None,
    ):
        self._name = None
        self._parent = None
        self._dataset = None
        self._children = []

        # Populate tree with children determined from data_objects mapping
        for path, obj in data_objects.items():
            self._set_item(path, obj, allow_overwrites=False, new_nodes_along_path=True)

    @classmethod
    def _construct(
        cls,
        name: Hashable = None,
        parent: DataTree = None,
        children: List[DataTree] = None,
        data: Union[Dataset, DataArray] = None,
    ) -> DataTree:
        """Alternative to __init__ allowing direct creation of a non-root node."""

        if children is None:
            children = []

        node = cls.__new__(cls)

        node._name = name
        node._children = children
        node.parent = parent
        node.dataset = data

        return node

    @property
    def name(self) -> Hashable:
        """Name tag for this node."""
        return self._name

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @dataset.setter
    def dataset(self, data: Union[Dataset, DataArray] = None):
        if not isinstance(data, (Dataset, DataArray)) or data is not None:
            raise TypeError(f"{type(data)} object is not an xarray Dataset or DataArray")
        if isinstance(data, DataArray):
            data = data.to_dataset()
        self._dataset = data

    @property
    def parent(self) -> Union[DataTree, None]:
        return self._parent

    @parent.setter
    def parent(self, parent: DataTree):
        if parent is not None:
            if not isinstance(parent, DataTree):
                raise TypeError(f"{type(parent.__name__)} object is not a node of a DataTree")

            if self._name in [c.name for c in parent._children]:
                raise KeyError(f"Cannot set parent: parent node {parent._name} "
                                 f"already has a child node named {self._name}")
            else:
                # Parent needs to know it now has a child
                parent.children = parent.children + [self]
        self._parent = parent

    @property
    def children(self) -> List[DataTree]:
        return self._children

    @children.setter
    def children(self, children: List[DataTree]):
        if not all(isinstance(c, DataTree) for c in children):
            raise TypeError(f"children must all be of type DataTree")
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

    def get(self, path: str, default: DataTree = None) -> DataTree:
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
        """
        Access node of the tree lying at the given path.

        Raises a KeyError if not found.

        Parameters
        ----------
        path :
            Path names can be given as unix-like paths, or as tuples of strings (where each string
            is known as a single "tag").

        Returns
        -------
        node : DataTree
        """
        node = self.get(path)
        if node is None:
            raise KeyError(f"Node {path} not found")
        return node

    def _set_item(self, path: PathType, value: Union[DataTree, Dataset, DataArray],
                  new_nodes_along_path: bool,
                  allow_overwrites: bool) -> None:
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

    def update_node(self, path: PathType, value: Union[DataTree, Dataset, DataArray]) -> None:
        """Overwrite the data at a specific node."""
        self._set_item(path=path, value=value, new_nodes_along_path=False, allow_overwrites=True)

    def __delitem__(self, path: PathType):
        for child in self._walk_children():
            del child

    def __iter__(self):
        return iter(c.name for c in self._children)

    def __len__(self):
        return len(self._children)

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

    def __repr__(self) -> str:
        type_str = "<xtree.DataTree>"
        tree_str = self._node_repr(indent_depth=0)
        # TODO add attrs dict to the repr
        return type_str + tree_str

    def _node_repr(self, indent_depth: int) -> str:
        indent_str = "|" + indent_depth * "    |" + "-- "
        node_repr = "\n" + indent_str + str(self.name)

        if self.dataset is not None:
            # TODO indent every line properly?
            node_repr += "\n" + indent_str + f"{repr(self.dataset)[17:]}"

        for child in self.children:
            node_repr += child._node_repr(indent_depth+1)

        return node_repr

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
            yield func(node.name, node.dataset, *args, **kwargs)

    def map_inplace(
        self,
        func: Callable,
        *args: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        """
        Apply a function to the dataset at each node in the tree, updating each node in place.

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
            new_ds = func(node.name, node.dataset, *args, **kwargs)
            node.update_node(node.path, value=new_ds)

    # TODO map applied ufuncs over all leaves
    # TODO map applied dataset/dataarray methods over all leaves

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
        datasets = [self.get(path).dataset for path in paths]
        return merge(datasets, compat=compat, join=join, fill_value=fill_value, combine_attrs=combine_attrs)

    def as_dataarray(self) -> DataArray:
        return self.dataset.as_dataarray()

    def to_netcdf(self, filename: str):
        from .io import _datatree_to_netcdf

        _datatree_to_netcdf(self, filename)

    def plot(self):
        raise NotImplementedError
