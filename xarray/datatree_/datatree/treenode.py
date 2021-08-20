from __future__ import annotations

from typing import Sequence, Tuple, Hashable, Union, Iterable

import anytree


PathType = Union[Hashable, Sequence[Hashable]]


class TreeNode(anytree.NodeMixin):
    """
    Base class representing a node of a tree, with methods for traversing and altering the tree.

    Depends on the anytree library for basic tree structure, but the parent class is fairly small
    so could be easily reimplemented to avoid a hard dependency.

    Adds restrictions preventing children with the same name, a method to set new nodes at arbitrary depth,
    and access via unix-like paths or tuples of tags. Does not yet store anything in the nodes of the tree.
    """

    # TODO remove anytree dependency
    # TODO allow for loops via symbolic links?

    # TODO store children with their names in an OrderedDict instead of a tuple like anytree does?
    # TODO do nodes even need names? Or can they just be referred to by the tags their parents store them under?
    # TODO nodes should have names but they should be optional. Getting and setting should be down without reference to
    # the names of stored objects, only their tags (i.e. position in the family tree)
    # Ultimately you either need a list of named children, or a dictionary of unnamed children

    _resolver = anytree.Resolver('name')

    def __init__(
        self,
        name: Hashable,
        parent: TreeNode = None,
        children: Iterable[TreeNode] = None,
    ):
        if not isinstance(name, str) or '/' in name:
            raise ValueError(f"invalid name {name}")
        self.name = name

        self.parent = parent
        if children:
            self.children = children

    def __str__(self):
        return f"TreeNode('{self.name}')"

    def __repr__(self):
        return f"TreeNode(name='{self.name}', parent={str(self.parent)}, children={[str(c) for c in self.children]})"

    def render(self):
        """Print tree structure, with only node names displayed."""
        # TODO should be rewritten to reflect names of children rather than names of nodes, probably like anytree.node
        # TODO add option to suppress dataset information beyond just variable names
        #for pre, _, node in anytree.RenderTree(self):
        #    print(f"{pre}{node}")
        args = ["%r" % self.separator.join([""] + [str(node.name) for node in self.path])]
        print(anytree.node.util._repr(self, args=args, nameblacklist=["name"]))

    def _pre_attach(self, parent: TreeNode) -> None:
        """
        Method which superclass calls before setting parent, here used to prevent having two
        children with duplicate names.
        """
        if self.name in list(c.name for c in parent.children):
            raise KeyError(f"parent {str(parent)} already has a child named {self.name}")

    def add_child(self, child: TreeNode) -> None:
        """Add a single child node below this node, without replacement."""
        if child.name in list(c.name for c in self.children):
            raise KeyError(f"Node already has a child named {child.name}")
        else:
            child.parent = self

    @classmethod
    def _tuple_or_path_to_path(cls, address: PathType) -> str:
        if isinstance(address, str):
            return address
        elif isinstance(address, tuple):
            return cls.separator.join(tag for tag in address)
        else:
            raise ValueError(f"{address} is not a valid form of path")

    def get_node(self, path: PathType) -> TreeNode:
        """
        Access node of the tree lying at the given path.

        Raises a TreeError if not found.

        Parameters
        ----------
        path :
            Paths can be given as unix-like paths, or as tuples of strings
            (where each string is known as a single "tag"). Path includes the name of the target node.

        Returns
        -------
        node
        """
        p = self._tuple_or_path_to_path(path)
        return anytree.Resolver('name').get(self, p)

    def set_node(
        self,
        path: PathType = '/',
        node: TreeNode = None,
        new_nodes_along_path: bool = True,
        allow_overwrite: bool = True,
    ) -> None:
        """
        Set a node on the tree, overwriting anything already present at that path.

        The given value either forms a new node of the tree or overwrites an existing node at that location.

        Paths are specified relative to the node on which this method was called, and the name of the node forms the
        last part of the path. (i.e. `.set_node(path='', TreeNode('a'))` is equivalent to `.add_child(TreeNode('a'))`.

        Parameters
        ----------
        path : Union[Hashable, Sequence[Hashable]]
            Path names can be given as unix-like paths, or as tuples of strings (where each string
            is known as a single "tag"). Default is '/'.
        node : TreeNode
        new_nodes_along_path : bool
            If true, then if necessary new nodes will be created along the given path, until the tree can reach the
            specified location. If false then an error is thrown instead of creating intermediate nodes alang the path.
        allow_overwrite : bool
            Whether or not to overwrite any existing node at the location given by path. Default is True.

        Raises
        ------
        KeyError
            If a node already exists at the given path
        """

        # Determine full path of new object
        path = self._tuple_or_path_to_path(path)

        if not isinstance(node, TreeNode):
            raise ValueError(f"Can only set nodes to be subclasses of TreeNode, but node is of type {type(node)}")
        node_name = node.name

        # Walk to location of new node, creating intermediate node objects as we go if necessary
        parent = self
        tags = [tag for tag in path.split(self.separator) if tag not in [self.separator, '']]
        for tag in tags:
            # TODO will this mutation within a for loop actually work?
            if tag not in [child.name for child in parent.children]:
                if new_nodes_along_path:
                    # TODO prevent this from leaving a trail of nodes if the assignment fails somehow
                    parent.add_child(TreeNode(name=tag))
                else:
                    raise KeyError(f"Cannot reach new node at path {path}: "
                                   f"parent {parent} has no child {tag}")
            parent = parent.get_node(tag)

        # Deal with anything already existing at this location
        if node_name in [child.name for child in parent.children]:
            if allow_overwrite:
                child = parent.get_node(node_name)
                child.parent = None
                del child
            else:
                # TODO should this be before we walk to the new node?
                raise KeyError(f"Cannot set item at {path} whilst that path already points to a "
                               f"{type(parent.get_node(node_name))} object")

        # Place new child node at this location
        parent.add_child(node)

    def glob(self, path: str):
        return self._resolver.glob(self, path)

    @property
    def tags(self) -> Tuple[Hashable]:
        """All tags, returned in order starting from the root node"""
        return tuple(self.path.split(self.separator))

    @tags.setter
    def tags(self, value):
        raise AttributeError(f"tags cannot be set, except via changing the children and/or parent of a node.")

    @property
    def subtree_nodes(self):
        """An iterator over all nodes in this tree, including both self and descendants."""
        return anytree.iterators.PreOrderIter(self)
