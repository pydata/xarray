.. currentmodule:: datatree

.. _hierarchical-data:

Working With Hierarchical Data
==============================

.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr
    from datatree import DataTree

    np.random.seed(123456)
    np.set_printoptions(threshold=10)

    %xmode minimal

Why Hierarchical Data?
----------------------

Many real-world datasets are composed of multiple differing components,
and it can often be be useful to think of these in terms of a hierarchy of related groups of data.
Examples of data which one might want organise in a grouped or hierarchical manner include:

- Simulation data at multiple resolutions,
- Observational data about the same system but from multiple different types of sensors,
- Mixed experimental and theoretical data,
- A systematic study recording the same experiment but with different parameters,
- Heterogenous data, such as demographic and metereological data,

or even any combination of the above.

Often datasets like this cannot easily fit into a single :py:class:`xarray.Dataset` object,
or are more usefully thought of as groups of related ``xarray.Dataset`` objects.
For this purpose we provide the :py:class:`DataTree` class.

This page explains in detail how to understand and use the different features of the :py:class:`DataTree` class for your own hierarchical data needs.

.. _node relationships:

Node Relationships
------------------

.. _creating a family tree:

Creating a Family Tree
~~~~~~~~~~~~~~~~~~~~~~

The three main ways of creating a ``DataTree`` object are described briefly in :ref:`creating a datatree`.
Here we go into more detail about how to create a tree node-by-node, using a famous family tree from the Simpsons cartoon as an example.

Let's start by defining nodes representing the two siblings, Bart and Lisa Simpson:

.. ipython:: python

    bart = DataTree(name="Bart")
    lisa = DataTree(name="Lisa")

Each of these node objects knows their own :py:class:`~DataTree.name`, but they currently have no relationship to one another.
We can connect them by creating another node representing a common parent, Homer Simpson:

.. ipython:: python

    homer = DataTree(name="Homer", children={"Bart": bart, "Lisa": lisa})

Here we set the children of Homer in the node's constructor.
We now have a small family tree

.. ipython:: python

    homer

where we can see how these individual Simpson family members are related to one another.
The nodes representing Bart and Lisa are now connected - we can confirm their sibling rivalry by examining the :py:class:`~DataTree.siblings` property:

.. ipython:: python

    list(bart.siblings)

But oops, we forgot Homer's third daughter, Maggie! Let's add her by updating Homer's :py:class:`~DataTree.children` property to include her:

.. ipython:: python

    maggie = DataTree(name="Maggie")
    homer.children = {"Bart": bart, "Lisa": lisa, "Maggie": maggie}
    homer

Let's check that Maggie knows who her Dad is:

.. ipython:: python

    maggie.parent.name

That's good - updating the properties of our nodes does not break the internal consistency of our tree, as changes of parentage are automatically reflected on both nodes.

    These children obviously have another parent, Marge Simpson, but ``DataTree`` nodes can only have a maximum of one parent.
    Genealogical `family trees are not even technically trees <https://en.wikipedia.org/wiki/Family_tree#Graph_theory>`_ in the mathematical sense -
    the fact that distant relatives can mate makes it a directed acyclic graph.
    Trees of ``DataTree`` objects cannot represent this.

Homer is currently listed as having no parent (the so-called "root node" of this tree), but we can update his :py:class:`~DataTree.parent` property:

.. ipython:: python

    abe = DataTree(name="Abe")
    homer.parent = abe

Abe is now the "root" of this tree, which we can see by examining the :py:class:`~DataTree.root` property of any node in the tree

.. ipython:: python

    maggie.root.name

We can see the whole tree by printing Abe's node or just part of the tree by printing Homer's node:

.. ipython:: python

    abe
    homer

We can see that Homer is aware of his parentage, and we say that Homer and his children form a "subtree" of the larger Simpson family tree.

In episode 28, Abe Simpson reveals that he had another son, Herbert "Herb" Simpson.
We can add Herbert to the family tree without displacing Homer by :py:meth:`~DataTree.assign`-ing another child to Abe:

.. ipython:: python

    herbert = DataTree(name="Herb")
    abe.assign({"Herbert": herbert})

.. note::
   This example shows a minor subtlety - the returned tree has Homer's brother listed as ``"Herbert"``,
   but the original node was named "Herbert". Not only are names overriden when stored as keys like this,
   but the new node is a copy, so that the original node that was reference is unchanged (i.e. ``herbert.name == "Herb"`` still).
   In other words, nodes are copied into trees, not inserted into them.
   This is intentional, and mirrors the behaviour when storing named ``xarray.DataArray`` objects inside datasets.

Certain manipulations of our tree are forbidden, if they would create an inconsistent result.
In episode 51 of the show Futurama, Philip J. Fry travels back in time and accidentally becomes his own Grandfather.
If we try similar time-travelling hijinks with Homer, we get a :py:class:`InvalidTreeError` raised:

.. ipython:: python
    :okexcept:

    abe.parent = homer

.. _evolutionary tree:

Ancestry in an Evolutionary Tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's use a different example of a tree to discuss more complex relationships between nodes - the phylogenetic tree, or tree of life.

.. ipython:: python

    vertebrates = DataTree.from_dict(
        name="Vertebrae",
        d={
            "/Sharks": None,
            "/Bony Skeleton/Ray-finned Fish": None,
            "/Bony Skeleton/Four Limbs/Amphibians": None,
            "/Bony Skeleton/Four Limbs/Amniotic Egg/Hair/Primates": None,
            "/Bony Skeleton/Four Limbs/Amniotic Egg/Hair/Rodents & Rabbits": None,
            "/Bony Skeleton/Four Limbs/Amniotic Egg/Two Fenestrae/Dinosaurs": None,
            "/Bony Skeleton/Four Limbs/Amniotic Egg/Two Fenestrae/Birds": None,
        },
    )

    primates = vertebrates["/Bony Skeleton/Four Limbs/Amniotic Egg/Hair/Primates"]
    dinosaurs = vertebrates[
        "/Bony Skeleton/Four Limbs/Amniotic Egg/Two Fenestrae/Dinosaurs"
    ]

We have used the :py:meth:`~DataTree.from_dict` constructor method as an alternate way to quickly create a whole tree,
and :ref:`filesystem-like syntax <filesystem paths>`_ (to be explained shortly) to select two nodes of interest.

.. ipython:: python

    vertebrates

This tree shows various families of species, grouped by their common features (making it technically a `"Cladogram" <https://en.wikipedia.org/wiki/Cladogram>`_,
rather than an evolutionary tree).

Here both the species and the features used to group them are represented by ``DataTree`` node objects - there is no distinction in types of node.
We can however get a list of only the nodes we used to represent species by using the fact that all those nodes have no children - they are "leaf nodes".
We can check if a node is a leaf with :py:meth:`~DataTree.is_leaf`, and get a list of all leaves with the :py:class:`~DataTree.leaves` property:

.. ipython:: python

    primates.is_leaf
    [node.name for node in vertebrates.leaves]

Pretending that this is a true evolutionary tree for a moment, we can find the features of the evolutionary ancestors (so-called "ancestor" nodes),
the distinguishing feature of the common ancestor of all vertebrate life (the root node),
and even the distinguishing feature of the common ancestor of any two species (the common ancestor of two nodes):

.. ipython:: python

    [node.name for node in primates.ancestors]
    primates.root.name
    primates.find_common_ancestor(dinosaurs).name

We can only find a common ancestor between two nodes that lie in the same tree.
If we try to find the common evolutionary ancestor between primates and an Alien species that has no relationship to Earth's evolutionary tree,
an error will be raised.

.. ipython:: python
    :okexcept:

    alien = DataTree(name="Xenomorph")
    primates.find_common_ancestor(alien)


.. _navigating trees:

Navigating Trees
----------------

There are various ways to access the different nodes in a tree.

Properties
~~~~~~~~~~

We can navigate trees using the :py:class:`~DataTree.parent` and :py:class:`~DataTree.children` properties of each node, for example:

.. ipython:: python

    lisa.parent.children["Bart"].name

but there are also more convenient ways to access nodes.

Dictionary-like interface
~~~~~~~~~~~~~~~~~~~~~~~~~

Children are stored on each node as a key-value mapping from name to child node.
They can be accessed and altered via the :py:class:`~DataTree.__getitem__` and :py:class:`~DataTree.__setitem__` syntax.
In general :py:class:`~DataTree.DataTree` objects support almost the entire set of dict-like methods,
including :py:meth:`~DataTree.keys`, :py:class:`~DataTree.values`, :py:class:`~DataTree.items`,
:py:meth:`~DataTree.__delitem__` and :py:meth:`~DataTree.update`.

.. ipython:: python

    vertebrates["Bony Skeleton"]["Ray-finned Fish"]

Note that the dict-like interface combines access to child ``DataTree`` nodes and stored ``DataArrays``,
so if we have a node that contains both children and data, calling :py:meth:`~DataTree.keys` will list both names of child nodes and
names of data variables:

.. ipython:: python

    dt = DataTree(
        data=xr.Dataset({"foo": 0, "bar": 1}),
        children={"a": DataTree(), "b": DataTree()},
    )
    print(dt)
    list(dt.keys())

This also means that the names of variables and of child nodes must be different to one another.

Attribute-like access
~~~~~~~~~~~~~~~~~~~~~

You can also select both variables and child nodes through dot indexing

.. ipython:: python

    dt.foo
    dt.a

.. _filesystem paths:

Filesystem-like Paths
~~~~~~~~~~~~~~~~~~~~~

Hierarchical trees can be thought of as analogous to file systems.
Each node is like a directory, and each directory can contain both more sub-directories and data.

.. note::

    You can even make the filesystem analogy concrete by using :py:func:`~DataTree.open_mfdatatree` or :py:func:`~DataTree.save_mfdatatree` # TODO not yet implemented - see GH issue 51

Datatree objects support a syntax inspired by unix-like filesystems,
where the "path" to a node is specified by the keys of each intermediate node in sequence,
separated by forward slashes.
This is an extension of the conventional dictionary ``__getitem__`` syntax to allow navigation across multiple levels of the tree.

Like with filepaths, paths within the tree can either be relative to the current node, e.g.

.. ipython:: python

    abe["Homer/Bart"].name
    abe["./Homer/Bart"].name  # alternative syntax

or relative to the root node.
A path specified from the root (as opposed to being specified relative to an arbitrary node in the tree) is sometimes also referred to as a
`"fully qualified name" <https://www.unidata.ucar.edu/blogs/developer/en/entry/netcdf-zarr-data-model-specification#nczarr_fqn>`_,
or as an "absolute path".
The root node is referred to by ``"/"``, so the path from the root node to its grand-child would be ``"/child/grandchild"``, e.g.

.. ipython:: python

    # absolute path will start from root node
    lisa["/Homer/Bart"].name

Relative paths between nodes also support the ``"../"`` syntax to mean the parent of the current node.
We can use this with ``__setitem__`` to add a missing entry to our evolutionary tree, but add it relative to a more familiar node of interest:

.. ipython:: python

    primates["../../Two Fenestrae/Crocodiles"] = DataTree()
    print(vertebrates)

Given two nodes in a tree, we can also find their relative path:

.. ipython:: python

    bart.relative_to(lisa)

You can use this filepath feature to build a nested tree from a dictionary of filesystem-like paths and corresponding ``xarray.Dataset`` objects in a single step.
If we have a dictionary where each key is a valid path, and each value is either valid data or ``None``,
we can construct a complex tree quickly using the alternative constructor :py:meth:`DataTree.from_dict()`:

.. ipython:: python

    d = {
        "/": xr.Dataset({"foo": "orange"}),
        "/a": xr.Dataset({"bar": 0}, coords={"y": ("y", [0, 1, 2])}),
        "/a/b": xr.Dataset({"zed": np.NaN}),
        "a/c/d": None,
    }
    dt = DataTree.from_dict(d)
    dt

.. note::

    Notice that using the path-like syntax will also create any intermediate empty nodes necessary to reach the end of the specified path
    (i.e. the node labelled `"c"` in this case.)
    This is to help avoid lots of redundant entries when creating deeply-nested trees using :py:meth:`DataTree.from_dict`.
