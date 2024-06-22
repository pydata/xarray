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
and :ref:`filesystem paths` (to be explained shortly) to select two nodes of interest.

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

.. _iterating over trees:

Iterating over trees
~~~~~~~~~~~~~~~~~~~~

You can iterate over every node in a tree using the subtree :py:class:`~DataTree.subtree` property.
This returns an iterable of nodes, which yields them in depth-first order.

.. ipython:: python

    for node in vertebrates.subtree:
        print(node.path)

A very useful pattern is to use :py:class:`~DataTree.subtree` conjunction with the :py:class:`~DataTree.path` property to manipulate the nodes however you wish,
then rebuild a new tree using :py:meth:`DataTree.from_dict()`.

For example, we could keep only the nodes containing data by looping over all nodes,
checking if they contain any data using :py:class:`~DataTree.has_data`,
then rebuilding a new tree using only the paths of those nodes:

.. ipython:: python

    non_empty_nodes = {node.path: node.ds for node in dt.subtree if node.has_data}
    DataTree.from_dict(non_empty_nodes)

You can see this tree is similar to the ``dt`` object above, except that it is missing the empty nodes ``a/c`` and ``a/c/d``.

(If you want to keep the name of the root node, you will need to add the ``name`` kwarg to :py:class:`from_dict`, i.e. ``DataTree.from_dict(non_empty_nodes, name=dt.root.name)``.)

.. _manipulating trees:

Manipulating Trees
------------------

Subsetting Tree Nodes
~~~~~~~~~~~~~~~~~~~~~

We can subset our tree to select only nodes of interest in various ways.

Similarly to on a real filesystem, matching nodes by common patterns in their paths is often useful.
We can use :py:meth:`DataTree.match` for this:

.. ipython:: python

    dt = DataTree.from_dict(
        {
            "/a/A": None,
            "/a/B": None,
            "/b/A": None,
            "/b/B": None,
        }
    )
    result = dt.match("*/B")
    result

We can also subset trees by the contents of the nodes.
:py:meth:`DataTree.filter` retains only the nodes of a tree that meet a certain condition.
For example, we could recreate the Simpson's family tree with the ages of each individual, then filter for only the adults:
First lets recreate the tree but with an `age` data variable in every node:

.. ipython:: python

    simpsons = DataTree.from_dict(
        d={
            "/": xr.Dataset({"age": 83}),
            "/Herbert": xr.Dataset({"age": 40}),
            "/Homer": xr.Dataset({"age": 39}),
            "/Homer/Bart": xr.Dataset({"age": 10}),
            "/Homer/Lisa": xr.Dataset({"age": 8}),
            "/Homer/Maggie": xr.Dataset({"age": 1}),
        },
        name="Abe",
    )
    simpsons

Now let's filter out the minors:

.. ipython:: python

    simpsons.filter(lambda node: node["age"] > 18)

The result is a new tree, containing only the nodes matching the condition.

(Yes, under the hood :py:meth:`~DataTree.filter` is just syntactic sugar for the pattern we showed you in :ref:`iterating over trees` !)

.. _Tree Contents:

Tree Contents
-------------

Hollow Trees
~~~~~~~~~~~~

A concept that can sometimes be useful is that of a "Hollow Tree", which means a tree with data stored only at the leaf nodes.
This is useful because certain useful tree manipulation operations only make sense for hollow trees.

You can check if a tree is a hollow tree by using the :py:class:`~DataTree.is_hollow` property.
We can see that the Simpson's family is not hollow because the data variable ``"age"`` is present at some nodes which
have children (i.e. Abe and Homer).

.. ipython:: python

    simpsons.is_hollow

.. _tree computation:

Computation
-----------

`DataTree` objects are also useful for performing computations, not just for organizing data.

Operations and Methods on Trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To show how applying operations across a whole tree at once can be useful,
let's first create a example scientific dataset.

.. ipython:: python

    def time_stamps(n_samples, T):
        """Create an array of evenly-spaced time stamps"""
        return xr.DataArray(
            data=np.linspace(0, 2 * np.pi * T, n_samples), dims=["time"]
        )


    def signal_generator(t, f, A, phase):
        """Generate an example electrical-like waveform"""
        return A * np.sin(f * t.data + phase)


    time_stamps1 = time_stamps(n_samples=15, T=1.5)
    time_stamps2 = time_stamps(n_samples=10, T=1.0)

    voltages = DataTree.from_dict(
        {
            "/oscilloscope1": xr.Dataset(
                {
                    "potential": (
                        "time",
                        signal_generator(time_stamps1, f=2, A=1.2, phase=0.5),
                    ),
                    "current": (
                        "time",
                        signal_generator(time_stamps1, f=2, A=1.2, phase=1),
                    ),
                },
                coords={"time": time_stamps1},
            ),
            "/oscilloscope2": xr.Dataset(
                {
                    "potential": (
                        "time",
                        signal_generator(time_stamps2, f=1.6, A=1.6, phase=0.2),
                    ),
                    "current": (
                        "time",
                        signal_generator(time_stamps2, f=1.6, A=1.6, phase=0.7),
                    ),
                },
                coords={"time": time_stamps2},
            ),
        }
    )
    voltages

Most xarray computation methods also exist as methods on datatree objects,
so you can for example take the mean value of these two timeseries at once:

.. ipython:: python

    voltages.mean(dim="time")

This works by mapping the standard :py:meth:`xarray.Dataset.mean()` method over the dataset stored in each node of the
tree one-by-one.

The arguments passed to the method are used for every node, so the values of the arguments you pass might be valid for one node and invalid for another

.. ipython:: python
    :okexcept:

    voltages.isel(time=12)

Notice that the error raised helpfully indicates which node of the tree the operation failed on.

Arithmetic Methods on Trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Arithmetic methods are also implemented, so you can e.g. add a scalar to every dataset in the tree at once.
For example, we can advance the timeline of the Simpsons by a decade just by

.. ipython:: python

    simpsons + 10

See that the same change (fast-forwarding by adding 10 years to the age of each character) has been applied to every node.

Mapping Custom Functions Over Trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can map custom computation over each node in a tree using :py:meth:`DataTree.map_over_subtree`.
You can map any function, so long as it takes `xarray.Dataset` objects as one (or more) of the input arguments,
and returns one (or more) xarray datasets.

.. note::

    Functions passed to :py:func:`map_over_subtree` cannot alter nodes in-place.
    Instead they must return new `xarray.Dataset` objects.

For example, we can define a function to calculate the Root Mean Square of a timeseries

.. ipython:: python

    def rms(signal):
        return np.sqrt(np.mean(signal**2))

Then calculate the RMS value of these signals:

.. ipython:: python

    voltages.map_over_subtree(rms)

.. _multiple trees:

We can also use the :py:func:`map_over_subtree` decorator to promote a function which accepts datasets into one which
accepts datatrees.

Operating on Multiple Trees
---------------------------

The examples so far have involved mapping functions or methods over the nodes of a single tree,
but we can generalize this to mapping functions over multiple trees at once.

Comparing Trees for Isomorphism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For it to make sense to map a single non-unary function over the nodes of multiple trees at once,
each tree needs to have the same structure. Specifically two trees can only be considered similar, or "isomorphic",
if they have the same number of nodes, and each corresponding node has the same number of children.
We can check if any two trees are isomorphic using the :py:meth:`DataTree.isomorphic` method.

.. ipython:: python
    :okexcept:

    dt1 = DataTree.from_dict({"a": None, "a/b": None})
    dt2 = DataTree.from_dict({"a": None})
    dt1.isomorphic(dt2)

    dt3 = DataTree.from_dict({"a": None, "b": None})
    dt1.isomorphic(dt3)

    dt4 = DataTree.from_dict({"A": None, "A/B": xr.Dataset({"foo": 1})})
    dt1.isomorphic(dt4)

If the trees are not isomorphic a :py:class:`~TreeIsomorphismError` will be raised.
Notice that corresponding tree nodes do not need to have the same name or contain the same data in order to be considered isomorphic.

Arithmetic Between Multiple Trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Arithmetic operations like multiplication are binary operations, so as long as we have two isomorphic trees,
we can do arithmetic between them.

.. ipython:: python

    currents = DataTree.from_dict(
        {
            "/oscilloscope1": xr.Dataset(
                {
                    "current": (
                        "time",
                        signal_generator(time_stamps1, f=2, A=1.2, phase=1),
                    ),
                },
                coords={"time": time_stamps1},
            ),
            "/oscilloscope2": xr.Dataset(
                {
                    "current": (
                        "time",
                        signal_generator(time_stamps2, f=1.6, A=1.6, phase=0.7),
                    ),
                },
                coords={"time": time_stamps2},
            ),
        }
    )
    currents

    currents.isomorphic(voltages)

We could use this feature to quickly calculate the electrical power in our signal, P=IV.

.. ipython:: python

    power = currents * voltages
    power
