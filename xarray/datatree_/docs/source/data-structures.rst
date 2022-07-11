.. _data structures:

Data Structures
===============

.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr
    import datatree

    np.random.seed(123456)
    np.set_printoptions(threshold=10)

.. note::

    This page builds on the information given in xarray's main page on
    `data structures <https://docs.xarray.dev/en/stable/user-guide/data-structures.html>`_, so it is suggested that you
    are familiar with those first.

DataTree
--------

:py:class:``DataTree`` is xarray's highest-level data structure, able to organise heterogeneous data which
could not be stored inside a single ``Dataset`` object. This includes representing the recursive structure of multiple
`groups`_ within a netCDF file or `Zarr Store`_.

.. _groups: https://www.unidata.ucar.edu/software/netcdf/workshops/2011/groups-types/GroupsIntro.html
.. _Zarr Store: https://zarr.readthedocs.io/en/stable/tutorial.html#groups

Each ``DataTree`` object (or "node") contains the same data that a single ``xarray.Dataset`` would (i.e. ``DataArray`` objects
stored under hashable keys), and so has the same key properties:

- ``dims``: a dictionary mapping of dimension names to lengths, for the variables in this node,
- ``data_vars``: a dict-like container of DataArrays corresponding to variables in this node,
- ``coords``: another dict-like container of DataArrays, corresponding to coordinate variables in this node,
- ``attrs``: dict to hold arbitary metadata relevant to data in this node.

A single ``DataTree`` object acts much like a single ``Dataset`` object, and has a similar set of dict-like methods
defined upon it. However, ``DataTree``'s can also contain other ``DataTree`` objects, so they can be thought of as nested dict-like
containers of both ``xarray.DataArray``'s and ``DataTree``'s.

A single datatree object is known as a "node", and its position relative to other nodes is defined by two more key
properties:

- ``children``: An ordered dictionary mapping from names to other ``DataTree`` objects, known as its' "child nodes".
- ``parent``: The single ``DataTree`` object whose children this datatree is a member of, known as its' "parent node".

Each child automatically knows about its parent node, and a node without a parent is known as a "root" node
(represented by the ``parent`` attribute pointing to ``None``).
Nodes can have multiple children, but as each child node has at most one parent, there can only ever be one root node in a given tree.

The overall structure is technically a `connected acyclic undirected rooted graph`, otherwise known as a
`"Tree" <https://en.wikipedia.org/wiki/Tree_(graph_theory)>`_.

.. note::

    Technically a ``DataTree`` with more than one child node forms an `"Ordered Tree" <https://en.wikipedia.org/wiki/Tree_(graph_theory)#Ordered_tree>`_,
    because the children are stored in an Ordered Dictionary. However, this distinction only really matters for a few
    edge cases involving operations on multiple trees simultaneously, and can safely be ignored by most users.


``DataTree`` objects can also optionally have a ``name`` as well as ``attrs``, just like a ``DataArray``.
Again these are not normally used unless explicitly accessed by the user.


Creating a DataTree
~~~~~~~~~~~~~~~~~~~

There are two ways to create a ``DataTree`` from scratch. The first is to create each node individually,
specifying the nodes' relationship to one another as you create each one.

The ``DataTree`` constructor takes:

- ``data``: The data that will be stored in this node, represented by a single ``xarray.Dataset``, or a named ``xarray.DataArray``.
- ``parent``: The parent node (if there is one), given as a ``DataTree`` object.
- ``children``: The various child nodes (if there are any), given as a mapping from string keys to ``DataTree`` objects.
- ``name``: A string to use as the name of this node.

Let's make a datatree node without anything in it:

.. ipython:: python

    from datatree import DataTree

    # create root node
    node1 = DataTree(name="Oak")

    node1

At this point our node is also the root node, as every tree has a root node.

We can add a second node to this tree either by referring to the first node in the constructor of the second:

.. ipython:: python

    # add a child by referring to the parent node
    node2 = DataTree(name="Bonsai", parent=node1)

or by dynamically updating the attributes of one node to refer to another:

.. ipython:: python

    # add a grandparent by updating the .parent property of an existing node
    node0 = DataTree(name="General Sherman")
    node1.parent = node0

Our tree now has three nodes within it, and one of the two new nodes has become the new root:

.. ipython:: python

    node0

Is is at tree construction time that consistency checks are enforced. For instance, if we try to create a `cycle` the constructor will raise an error:

.. ipython:: python
    :okexcept:

    node0.parent = node2

The second way is to build the tree from a dictionary of filesystem-like paths and corresponding ``xarray.Dataset`` objects.

This relies on a syntax inspired by unix-like filesystems, where the "path" to a node is specified by the keys of each intermediate node in sequence,
separated by forward slashes. The root node is referred to by ``"/"``, so the path from our current root node to its grand-child would be ``"/Oak/Bonsai"``.
A path specified from the root (as opposed to being specified relative to an arbitrary node in the tree) is sometimes also referred to as a
`"fully qualified name" <https://www.unidata.ucar.edu/blogs/developer/en/entry/netcdf-zarr-data-model-specification#nczarr_fqn>`_.

If we have a dictionary where each key is a valid path, and each value is either valid data or ``None``,
we can construct a complex tree quickly using the alternative constructor ``:py:func::DataTree.from_dict``:

.. ipython:: python

    d = {
        "/": xr.Dataset({"foo": "orange"}),
        "/a": xr.Dataset({"bar": 0}, coords={"y": ("y", [0, 1, 2])}),
        "/a/b": xr.Dataset({"zed": np.NaN}),
        "a/c/d": None,
    }
    dt = DataTree.from_dict(d)
    dt

Notice that this method will also create any intermediate empty node necessary to reach the end of the specified path
(i.e. the node labelled `"c"` in this case.)

Finally if you have a file containing data on disk (such as a netCDF file or a Zarr Store), you can also create a datatree by opening the
file using ``:py:func::~datatree.open_datatree``.


DataTree Contents
~~~~~~~~~~~~~~~~~

Like ``xarray.Dataset``, ``DataTree`` implements the python mapping interface, but with values given by either ``xarray.DataArray`` objects or other ``DataTree`` objects.

.. ipython:: python

    dt["a"]
    dt["foo"]

Iterating over keys will iterate over both the names of variables and child nodes.

We can also access all the data in a single node through a dataset-like view

.. ipython:: python

    dt["a"].ds

This demonstrates the fact that the data in any one node is equivalent to the contents of a single ``xarray.Dataset`` object.
The ``DataTree.ds`` property returns an immutable view, but we can instead extract the node's data contents as a new (and mutable)
``xarray.Dataset`` object via ``.to_dataset()``:

.. ipython:: python

    dt["a"].to_dataset()

Like with ``Dataset``, you can access the data and coordinate variables of a node separately via the ``data_vars`` and ``coords`` attributes:

.. ipython:: python

    dt["a"].data_vars
    dt["a"].coords


Dictionary-like methods
~~~~~~~~~~~~~~~~~~~~~~~

We can update the contents of the tree in-place using a dictionary-like syntax.

We can update a datatree in-place using Python's standard dictionary syntax, similar to how we can for Dataset objects.
For example, to create this example datatree from scratch, we could have written:

# TODO update this example using ``.coords`` and ``.data_vars`` as setters,

.. ipython:: python

    dt = DataTree()
    dt["foo"] = "orange"
    dt["a"] = DataTree(data=xr.Dataset({"bar": 0}, coords={"y": ("y", [0, 1, 2])}))
    dt["a/b/zed"] = np.NaN
    dt["a/c/d"] = DataTree()
    dt

To change the variables in a node of a ``DataTree``, you can use all the standard dictionary
methods, including ``values``, ``items``, ``__delitem__``, ``get`` and
:py:meth:`~xarray.DataTree.update`.
Note that assigning a ``DataArray`` object to a ``DataTree`` variable using ``__setitem__`` or ``update`` will
:ref:`automatically align<update>` the array(s) to the original node's indexes.

If you copy a ``DataTree`` using the ``:py:func::copy`` function or the :py:meth:`~xarray.DataTree.copy` it will copy the entire tree,
including all parents and children.
Like for ``Dataset``, this copy is shallow by default, but you can copy all the data by calling ``dt.copy(deep=True)``.
