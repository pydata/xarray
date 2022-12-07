from __future__ import annotations

from collections import OrderedDict
from pathlib import PurePosixPath
from typing import (
    TYPE_CHECKING,
    Generic,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from xarray.core.utils import Frozen, is_dict_like

if TYPE_CHECKING:
    from xarray.core.types import T_DataArray


class TreeError(Exception):
    """Exception type raised when user attempts to create an invalid tree in some way."""

    ...


class NodePath(PurePosixPath):
    """Represents a path from one node to another within a tree."""

    def __new__(cls, *args: str | "NodePath") -> "NodePath":
        obj = super().__new__(cls, *args)

        if obj.drive:
            raise ValueError("NodePaths cannot have drives")

        if obj.root not in ["/", ""]:
            raise ValueError(
                'Root of NodePath can only be either "/" or "", with "" meaning the path is relative.'
            )

        # TODO should we also forbid suffixes to avoid node names with dots in them?

        return obj


Tree = TypeVar("Tree", bound="TreeNode")


class TreeNode(Generic[Tree]):
    """
    Base class representing a node of a tree, with methods for traversing and altering the tree.

    This class stores no data, it has only parents and children attributes, and various methods.

    Stores child nodes in an Ordered Dictionary, which is necessary to ensure that equality checks between two trees
    also check that the order of child nodes is the same.

    Nodes themselves are intrinsically unnamed (do not possess a ._name attribute), but if the node has a parent you can
    find the key it is stored under via the .name property.

    The .parent attribute is read-only: to replace the parent using public API you must set this node as the child of a
    new parent using `new_parent.children[name] = child_node`, or to instead detach from the current parent use
    `child_node.orphan()`.

    This class is intended to be subclassed by DataTree, which will overwrite some of the inherited behaviour,
    in particular to make names an inherent attribute, and allow setting parents directly. The intention is to mirror
    the class structure of xarray.Variable & xarray.DataArray, where Variable is unnamed but DataArray is (optionally)
    named.

    Also allows access to any other node in the tree via unix-like paths, including upwards referencing via '../'.

    (This class is heavily inspired by the anytree library's NodeMixin class.)
    """

    _parent: Optional[Tree]
    _children: OrderedDict[str, Tree]

    def __init__(self, children: Optional[Mapping[str, Tree]] = None):
        """Create a parentless node."""
        self._parent = None
        self._children = OrderedDict()
        if children is not None:
            self.children = children

    @property
    def parent(self) -> Tree | None:
        """Parent of this node."""
        return self._parent

    def _set_parent(
        self, new_parent: Tree | None, child_name: Optional[str] = None
    ) -> None:
        # TODO is it possible to refactor in a way that removes this private method?

        if new_parent is not None and not isinstance(new_parent, TreeNode):
            raise TypeError(
                "Parent nodes must be of type DataTree or None, "
                f"not type {type(new_parent)}"
            )

        old_parent = self._parent
        if new_parent is not old_parent:
            self._check_loop(new_parent)
            self._detach(old_parent)
            self._attach(new_parent, child_name)

    def _check_loop(self, new_parent: Tree | None) -> None:
        """Checks that assignment of this new parent will not create a cycle."""
        if new_parent is not None:
            if new_parent is self:
                raise TreeError(
                    f"Cannot set parent, as node {self} cannot be a parent of itself."
                )

            if self._is_descendant_of(new_parent):
                raise TreeError(
                    "Cannot set parent, as intended parent is already a descendant of this node."
                )

    def _is_descendant_of(self, node: Tree) -> bool:
        _self, *lineage = list(node.lineage)
        return any(n is self for n in lineage)

    def _detach(self, parent: Tree | None) -> None:
        if parent is not None:
            self._pre_detach(parent)
            parents_children = parent.children
            parent._children = OrderedDict(
                {
                    name: child
                    for name, child in parents_children.items()
                    if child is not self
                }
            )
            self._parent = None
            self._post_detach(parent)

    def _attach(self, parent: Tree | None, child_name: Optional[str] = None) -> None:
        if parent is not None:
            if child_name is None:
                raise ValueError(
                    "To directly set parent, child needs a name, but child is unnamed"
                )

            self._pre_attach(parent)
            parentchildren = parent._children
            assert not any(
                child is self for child in parentchildren
            ), "Tree is corrupt."
            parentchildren[child_name] = self
            self._parent = parent
            self._post_attach(parent)
        else:
            self._parent = None

    def orphan(self) -> None:
        """Detach this node from its parent."""
        self._set_parent(new_parent=None)

    @property
    def children(self: Tree) -> Mapping[str, Tree]:
        """Child nodes of this node, stored under a mapping via their names."""
        return Frozen(self._children)

    @children.setter
    def children(self: Tree, children: Mapping[str, Tree]) -> None:
        self._check_children(children)
        children = OrderedDict(children)

        old_children = self.children
        del self.children
        try:
            self._pre_attach_children(children)
            for name, child in children.items():
                child._set_parent(new_parent=self, child_name=name)
            self._post_attach_children(children)
            assert len(self.children) == len(children)
        except Exception:
            # if something goes wrong then revert to previous children
            self.children = old_children
            raise

    @children.deleter
    def children(self) -> None:
        # TODO this just detaches all the children, it doesn't actually delete them...
        children = self.children
        self._pre_detach_children(children)
        for child in self.children.values():
            child.orphan()
        assert len(self.children) == 0
        self._post_detach_children(children)

    @staticmethod
    def _check_children(children: Mapping[str, Tree]) -> None:
        """Check children for correct types and for any duplicates."""
        if not is_dict_like(children):
            raise TypeError(
                "children must be a dict-like mapping from names to node objects"
            )

        seen = set()
        for name, child in children.items():
            if not isinstance(child, TreeNode):
                raise TypeError(
                    f"Cannot add object {name}. It is of type {type(child)}, "
                    "but can only add children of type DataTree"
                )

            childid = id(child)
            if childid not in seen:
                seen.add(childid)
            else:
                raise TreeError(
                    f"Cannot add same node {name} multiple times as different children."
                )

    def __repr__(self) -> str:
        return f"TreeNode(children={dict(self._children)})"

    def _pre_detach_children(self: Tree, children: Mapping[str, Tree]) -> None:
        """Method call before detaching `children`."""
        pass

    def _post_detach_children(self: Tree, children: Mapping[str, Tree]) -> None:
        """Method call after detaching `children`."""
        pass

    def _pre_attach_children(self: Tree, children: Mapping[str, Tree]) -> None:
        """Method call before attaching `children`."""
        pass

    def _post_attach_children(self: Tree, children: Mapping[str, Tree]) -> None:
        """Method call after attaching `children`."""
        pass

    def iter_lineage(self: Tree) -> Iterator[Tree]:
        """Iterate up the tree, starting from the current node."""
        # TODO should this instead return an OrderedDict, so as to include node names?
        node: Tree | None = self
        while node is not None:
            yield node
            node = node.parent

    @property
    def lineage(self: Tree) -> Tuple[Tree, ...]:
        """All parent nodes and their parent nodes, starting with the closest."""
        return tuple(self.iter_lineage())

    @property
    def ancestors(self: Tree) -> Tuple[Tree, ...]:
        """All parent nodes and their parent nodes, starting with the most distant."""
        if self.parent is None:
            return (self,)
        else:
            ancestors = tuple(reversed(list(self.lineage)))
            return ancestors

    @property
    def root(self: Tree) -> Tree:
        """Root node of the tree"""
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    @property
    def is_root(self) -> bool:
        """Whether or not this node is the tree root."""
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        """Whether or not this node is a leaf node."""
        return self.children == {}

    @property
    def siblings(self: Tree) -> OrderedDict[str, Tree]:
        """
        Nodes with the same parent as this node.
        """
        if self.parent:
            return OrderedDict(
                {
                    name: child
                    for name, child in self.parent.children.items()
                    if child is not self
                }
            )
        else:
            return OrderedDict()

    @property
    def subtree(self: Tree) -> Iterator[Tree]:
        """
        An iterator over all nodes in this tree, including both self and all descendants.

        Iterates depth-first.
        """
        from . import iterators

        return iterators.PreOrderIter(self)

    def _pre_detach(self: Tree, parent: Tree) -> None:
        """Method call before detaching from `parent`."""
        pass

    def _post_detach(self: Tree, parent: Tree) -> None:
        """Method call after detaching from `parent`."""
        pass

    def _pre_attach(self: Tree, parent: Tree) -> None:
        """Method call before attaching to `parent`."""
        pass

    def _post_attach(self: Tree, parent: Tree) -> None:
        """Method call after attaching to `parent`."""
        pass

    def get(self: Tree, key: str, default: Optional[Tree] = None) -> Optional[Tree]:
        """
        Return the child node with the specified key.

        Only looks for the node within the immediate children of this node,
        not in other nodes of the tree.
        """
        if key in self.children:
            return self.children[key]
        else:
            return default

    # TODO `._walk` method to be called by both `_get_item` and `_set_item`

    def _get_item(self: Tree, path: str | NodePath) -> Union[Tree, T_DataArray]:
        """
        Returns the object lying at the given path.

        Raises a KeyError if there is no object at the given path.
        """
        if isinstance(path, str):
            path = NodePath(path)

        if path.root:
            current_node = self.root
            root, *parts = list(path.parts)
        else:
            current_node = self
            parts = list(path.parts)

        for part in parts:
            if part == "..":
                if current_node.parent is None:
                    raise KeyError(f"Could not find node at {path}")
                else:
                    current_node = current_node.parent
            elif part in ("", "."):
                pass
            else:
                if current_node.get(part) is None:
                    raise KeyError(f"Could not find node at {path}")
                else:
                    current_node = current_node.get(part)
        return current_node

    def _set(self: Tree, key: str, val: Tree) -> None:
        """
        Set the child node with the specified key to value.

        Counterpart to the public .get method, and also only works on the immediate node, not other nodes in the tree.
        """
        new_children = {**self.children, key: val}
        self.children = new_children

    def _set_item(
        self: Tree,
        path: str | NodePath,
        item: Union[Tree, T_DataArray],
        new_nodes_along_path: bool = False,
        allow_overwrite: bool = True,
    ) -> None:
        """
        Set a new item in the tree, overwriting anything already present at that path.

        The given value either forms a new node of the tree or overwrites an existing item at that location.

        Parameters
        ----------
        path
        item
        new_nodes_along_path : bool
            If true, then if necessary new nodes will be created along the given path, until the tree can reach the
            specified location.
        allow_overwrite : bool
            Whether or not to overwrite any existing node at the location given by path.

        Raises
        ------
        KeyError
            If node cannot be reached, and new_nodes_along_path=False.
            Or if a node already exists at the specified path, and allow_overwrite=False.
        """
        if isinstance(path, str):
            path = NodePath(path)

        if not path.name:
            raise ValueError("Can't set an item under a path which has no name")

        if path.root:
            # absolute path
            current_node = self.root
            root, *parts, name = path.parts
        else:
            # relative path
            current_node = self
            *parts, name = path.parts

        if parts:
            # Walk to location of new node, creating intermediate node objects as we go if necessary
            for part in parts:
                if part == "..":
                    if current_node.parent is None:
                        # We can't create a parent if `new_nodes_along_path=True` as we wouldn't know what to name it
                        raise KeyError(f"Could not reach node at path {path}")
                    else:
                        current_node = current_node.parent
                elif part in ("", "."):
                    pass
                else:
                    if part in current_node.children:
                        current_node = current_node.children[part]
                    elif new_nodes_along_path:
                        # Want child classes (i.e. DataTree) to populate tree with their own types
                        new_node = type(self)()
                        current_node._set(part, new_node)
                        current_node = current_node.children[part]
                    else:
                        raise KeyError(f"Could not reach node at path {path}")

        if name in current_node.children:
            # Deal with anything already existing at this location
            if allow_overwrite:
                current_node._set(name, item)
            else:
                raise KeyError(f"Already a node object at path {path}")
        else:
            current_node._set(name, item)

    def __delitem__(self: Tree, key: str):
        """Remove a child node from this tree object."""
        if key in self.children:
            child = self._children[key]
            del self._children[key]
            child.orphan()
        else:
            raise KeyError("Cannot delete")

    def update(self: Tree, other: Mapping[str, Tree]) -> None:
        """
        Update this node's children.

        Just like `dict.update` this is an in-place operation.
        """
        new_children = {**self.children, **other}
        self.children = new_children

    def same_tree(self, other: Tree) -> bool:
        """True if other node is in the same tree as this node."""
        return self.root is other.root

    def find_common_ancestor(self, other: Tree) -> Tree:
        """
        Find the first common ancestor of two nodes in the same tree.

        Raise ValueError if they are not in the same tree.
        """
        common_ancestor = None
        for node in other.iter_lineage():
            if node in self.ancestors:
                common_ancestor = node
                break

        if not common_ancestor:
            raise ValueError(
                "Cannot find relative path because nodes do not lie within the same tree"
            )

        return common_ancestor

    def _path_to_ancestor(self, ancestor: Tree) -> NodePath:
        generation_gap = list(self.lineage).index(ancestor)
        path_upwards = "../" * generation_gap if generation_gap > 0 else "/"
        return NodePath(path_upwards)


class NamedNode(TreeNode, Generic[Tree]):
    """
    A TreeNode which knows its own name.

    Implements path-like relationships to other nodes in its tree.
    """

    _name: Optional[str]
    _parent: Optional[Tree]
    _children: OrderedDict[str, Tree]

    def __init__(self, name=None, children=None):
        super().__init__(children=children)
        self._name = None
        self.name = name

    @property
    def name(self) -> str | None:
        """The name of this node."""
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("node name must be a string or None")
            if "/" in name:
                raise ValueError("node names cannot contain forward slashes")
        self._name = name

    def __str__(self) -> str:
        return f"NamedNode({self.name})" if self.name else "NamedNode()"

    def _post_attach(self: NamedNode, parent: NamedNode) -> None:
        """Ensures child has name attribute corresponding to key under which it has been stored."""
        key = next(k for k, v in parent.children.items() if v is self)
        self.name = key

    @property
    def path(self) -> str:
        """Return the file-like path from the root to this node."""
        if self.is_root:
            return "/"
        else:
            root, *ancestors = self.ancestors
            # don't include name of root because (a) root might not have a name & (b) we want path relative to root.
            names = [node.name for node in ancestors]
            return "/" + "/".join(names)

    def relative_to(self: NamedNode, other: NamedNode) -> str:
        """
        Compute the relative path from this node to node `other`.

        If other is not in this tree, or it's otherwise impossible, raise a ValueError.
        """
        if not self.same_tree(other):
            raise ValueError(
                "Cannot find relative path because nodes do not lie within the same tree"
            )

        this_path = NodePath(self.path)
        if other in self.lineage:
            return str(this_path.relative_to(other.path))
        else:
            common_ancestor = self.find_common_ancestor(other)
            path_to_common_ancestor = other._path_to_ancestor(common_ancestor)
            return str(
                path_to_common_ancestor / this_path.relative_to(common_ancestor.path)
            )
