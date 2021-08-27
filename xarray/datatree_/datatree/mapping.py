import functools

from anytree.iterators import LevelOrderIter

from .treenode import TreeNode


class TreeIsomorphismError(ValueError):
    """Error raised if two tree objects are not isomorphic to one another when they need to be."""

    pass


def _check_isomorphic(subtree_a, subtree_b, require_names_equal=False):
    """
    Check that two trees have the same structure, raising an error if not.

    Does not check the actual data in the nodes, but it does check that if one node does/doesn't have data then its
    counterpart in the other tree also does/doesn't have data.

    Also does not check that the root nodes of each tree have the same parent - so this function checks that subtrees
    are isomorphic, not the entire tree above (if it exists).

    Can optionally check if respective nodes should have the same name.

    Parameters
    ----------
    subtree_a : DataTree
    subtree_b : DataTree
    require_names_equal : Bool, optional
        Whether or not to also check that each node has the same name as its counterpart. Default is False.

    Raises
    ------
    TypeError
        If either subtree_a or subtree_b are not tree objects.
    TreeIsomorphismError
        If subtree_a and subtree_b are tree objects, but are not isomorphic to one another, or one contains data at a
        location the other does not. Also optionally raised if their structure is isomorphic, but the names of any two
        respective nodes are not equal.
    """
    # TODO turn this into a public function called assert_isomorphic

    if not isinstance(subtree_a, TreeNode):
        raise TypeError(
            f"Argument `subtree_a is not a tree, it is of type {type(subtree_a)}"
        )
    if not isinstance(subtree_b, TreeNode):
        raise TypeError(
            f"Argument `subtree_b is not a tree, it is of type {type(subtree_b)}"
        )

    # Walking nodes in "level-order" fashion means walking down from the root breadth-first.
    # Checking by walking in this way implicitly assumes that the tree is an ordered tree (which it is so long as
    # children are stored in a tuple or list rather than in a set).
    for node_a, node_b in zip(LevelOrderIter(subtree_a), LevelOrderIter(subtree_b)):
        path_a, path_b = node_a.pathstr, node_b.pathstr

        if require_names_equal:
            if node_a.name != node_b.name:
                raise TreeIsomorphismError(
                    f"Trees are not isomorphic because node '{path_a}' in the first tree has "
                    f"name '{node_a.name}', whereas its counterpart node '{path_b}' in the "
                    f"second tree has name '{node_b.name}'."
                )

        if node_a.has_data != node_b.has_data:
            dat_a = "no " if not node_a.has_data else ""
            dat_b = "no " if not node_b.has_data else ""
            raise TreeIsomorphismError(
                f"Trees are not isomorphic because node '{path_a}' in the first tree has "
                f"{dat_a}data, whereas its counterpart node '{path_b}' in the second tree "
                f"has {dat_b}data."
            )

        if len(node_a.children) != len(node_b.children):
            raise TreeIsomorphismError(
                f"Trees are not isomorphic because node '{path_a}' in the first tree has "
                f"{len(node_a.children)} children, whereas its counterpart node '{path_b}' in "
                f"the second tree has {len(node_b.children)} children."
            )


def map_over_subtree(func):
    """
    Decorator which turns a function which acts on (and returns) single Datasets into one which acts on DataTrees.

    Applies a function to every dataset in this subtree, returning a new tree which stores the results.

    The function will be applied to any dataset stored in this node, as well as any dataset stored in any of the
    descendant nodes. The returned tree will have the same structure as the original subtree.

    func needs to return a Dataset, DataArray, or None in order to be able to rebuild the subtree after mapping, as each
    result will be assigned to its respective node of new tree via `DataTree.__setitem__`.

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

    @functools.wraps(func)
    def _map_over_subtree(tree, *args, **kwargs):
        """Internal function which maps func over every node in tree, returning a tree of the results."""

        # Recreate and act on root node
        from .datatree import DataNode

        out_tree = DataNode(name=tree.name, data=tree.ds)
        if out_tree.has_data:
            out_tree.ds = func(out_tree.ds, *args, **kwargs)

        # Act on every other node in the tree, and rebuild from results
        for node in tree.descendants:
            # TODO make a proper relative_path method
            relative_path = node.pathstr.replace(tree.pathstr, "")
            result = func(node.ds, *args, **kwargs) if node.has_data else None
            out_tree[relative_path] = result

        return out_tree

    return _map_over_subtree
