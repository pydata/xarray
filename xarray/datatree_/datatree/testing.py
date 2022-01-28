from xarray.testing import ensure_warnings

from .datatree import DataTree
from .formatting import diff_tree_repr


@ensure_warnings
def assert_isomorphic(a: DataTree, b: DataTree, from_root: bool = False):
    """
    Two DataTrees are considered isomorphic if every node has the same number of children.

    Nothing about the data in each node is checked.

    Isomorphism is a necessary condition for two trees to be used in a nodewise binary operation,
    such as tree1 + tree2.

    By default this function does not check any part of the tree above the given node.
    Therefore this function can be used as default to check that two subtrees are isomorphic.

    Parameters
    ----------
    a : DataTree
        The first object to compare.
    b : DataTree
        The second object to compare.
    from_root : bool, optional, default is False
        Whether or not to first traverse to the root of the trees before checking for isomorphism.
        If a & b have no parents then this has no effect.

    See Also
    --------
    DataTree.isomorphic
    assert_equals
    assert_identical
    """
    __tracebackhide__ = True
    assert type(a) == type(b)

    if isinstance(a, DataTree):
        if from_root:
            a = a.root
            b = b.root

        assert a.isomorphic(b, from_root=from_root), diff_tree_repr(a, b, "isomorphic")
    else:
        raise TypeError(f"{type(a)} not of type DataTree")


@ensure_warnings
def assert_equal(a: DataTree, b: DataTree, from_root: bool = True):
    """
    Two DataTrees are equal if they have isomorphic node structures, with matching node names,
    and if they have matching variables and coordinates, all of which are equal.

    By default this method will check the whole tree above the given node.

    Parameters
    ----------
    a : DataTree
        The first object to compare.
    b : DataTree
        The second object to compare.
    from_root : bool, optional, default is True
        Whether or not to first traverse to the root of the trees before checking for isomorphism.
        If a & b have no parents then this has no effect.

    See Also
    --------
    DataTree.equals
    assert_isomorphic
    assert_identical
    """
    __tracebackhide__ = True
    assert type(a) == type(b)

    if isinstance(a, DataTree):
        if from_root:
            a = a.root
            b = b.root

        assert a.equals(b, from_root=from_root), diff_tree_repr(a, b, "equals")
    else:
        raise TypeError(f"{type(a)} not of type DataTree")


@ensure_warnings
def assert_identical(a: DataTree, b: DataTree, from_root: bool = True):
    """
    Like assert_equals, but will also check all dataset attributes and the attributes on
    all variables and coordinates.

    By default this method will check the whole tree above the given node.

    Parameters
    ----------
    a : xarray.DataTree
        The first object to compare.
    b : xarray.DataTree
        The second object to compare.
    from_root : bool, optional, default is True
        Whether or not to first traverse to the root of the trees before checking for isomorphism.
        If a & b have no parents then this has no effect.

    See Also
    --------
    DataTree.identical
    assert_isomorphic
    assert_equal
    """

    __tracebackhide__ = True
    assert type(a) == type(b)
    if isinstance(a, DataTree):
        if from_root:
            a = a.root
            b = b.root

        assert a.identical(b, from_root=from_root), diff_tree_repr(a, b, "identical")
    else:
        raise TypeError(f"{type(a)} not of type DataTree")
