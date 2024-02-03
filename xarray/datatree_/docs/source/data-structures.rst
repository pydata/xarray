.. currentmodule:: datatree

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

    %xmode minimal

.. note::

    This page builds on the information given in xarray's main page on
    `data structures <https://docs.xarray.dev/en/stable/user-guide/data-structures.html>`_, so it is suggested that you
    are familiar with those first.

DataTree
--------

:py:class:`DataTree` is xarray's highest-level data structure, able to organise heterogeneous data which
could not be stored inside a single :py:class:`Dataset` object. This includes representing the recursive structure of multiple
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


.. _creating a datatree:

Creating a DataTree
~~~~~~~~~~~~~~~~~~~

One way to create a ``DataTree`` from scratch is to create each node individually,
specifying the nodes' relationship to one another as you create each one.

The ``DataTree`` constructor takes:

- ``data``: The data that will be stored in this node, represented by a single ``xarray.Dataset``, or a named ``xarray.DataArray``.
- ``parent``: The parent node (if there is one), given as a ``DataTree`` object.
- ``children``: The various child nodes (if there are any), given as a mapping from string keys to ``DataTree`` objects.
- ``name``: A string to use as the name of this node.

Let's make a single datatree node with some example data in it:

.. ipython:: python

    from datatree import DataTree

    ds1 = xr.Dataset({"foo": "orange"})
    dt = DataTree(name="root", data=ds1)  # create root node

    dt

At this point our node is also the root node, as every tree has a root node.

We can add a second node to this tree either by referring to the first node in the constructor of the second:

.. ipython:: python

    ds2 = xr.Dataset({"bar": 0}, coords={"y": ("y", [0, 1, 2])})
    # add a child by referring to the parent node
    node2 = DataTree(name="a", parent=dt, data=ds2)

or by dynamically updating the attributes of one node to refer to another:

.. ipython:: python

    # add a second child by first creating a new node ...
    ds3 = xr.Dataset({"zed": np.NaN})
    node3 = DataTree(name="b", data=ds3)
    # ... then updating its .parent property
    node3.parent = dt

Our tree now has three nodes within it:

.. ipython:: python

    dt

It is at tree construction time that consistency checks are enforced. For instance, if we try to create a `cycle` the constructor will raise an error:

.. ipython:: python
    :okexcept:

    dt.parent = node3

Alternatively you can also create a ``DataTree`` object from

- An ``xarray.Dataset`` using ``Dataset.to_node()`` (not yet implemented),
- A dictionary mapping directory-like paths to either ``DataTree`` nodes or data, using :py:meth:`DataTree.from_dict()`,
- A netCDF or Zarr file on disk with :py:func:`open_datatree()`. See :ref:`reading and writing files <io>`.


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
``xarray.Dataset`` object via :py:meth:`DataTree.to_dataset()`:

.. ipython:: python

    dt["a"].to_dataset()

Like with ``Dataset``, you can access the data and coordinate variables of a node separately via the ``data_vars`` and ``coords`` attributes:

.. ipython:: python

    dt["a"].data_vars
    dt["a"].coords


Dictionary-like methods
~~~~~~~~~~~~~~~~~~~~~~~

We can update a datatree in-place using Python's standard dictionary syntax, similar to how we can for Dataset objects.
For example, to create this example datatree from scratch, we could have written:

# TODO update this example using ``.coords`` and ``.data_vars`` as setters,

.. ipython:: python

    dt = DataTree(name="root")
    dt["foo"] = "orange"
    dt["a"] = DataTree(data=xr.Dataset({"bar": 0}, coords={"y": ("y", [0, 1, 2])}))
    dt["a/b/zed"] = np.NaN
    dt

To change the variables in a node of a ``DataTree``, you can use all the standard dictionary
methods, including ``values``, ``items``, ``__delitem__``, ``get`` and
:py:meth:`DataTree.update`.
Note that assigning a ``DataArray`` object to a ``DataTree`` variable using ``__setitem__`` or ``update`` will
:ref:`automatically align <update>` the array(s) to the original node's indexes.

If you copy a ``DataTree`` using the :py:func:`copy` function or the :py:meth:`DataTree.copy` method it will copy the subtree,
meaning that node and children below it, but no parents above it.
Like for ``Dataset``, this copy is shallow by default, but you can copy all the underlying data arrays by calling ``dt.copy(deep=True)``.
