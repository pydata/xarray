.. _userguide.hierarchical-data:

Hierarchical data
=================

.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)
    np.set_printoptions(threshold=10)

    %xmode minimal

.. _why:

Why Hierarchical Data?
----------------------

Many real-world datasets are composed of multiple differing components,
and it can often be useful to think of these in terms of a hierarchy of related groups of data.
Examples of data which one might want organise in a grouped or hierarchical manner include:

- Simulation data at multiple resolutions,
- Observational data about the same system but from multiple different types of sensors,
- Mixed experimental and theoretical data,
- A systematic study recording the same experiment but with different parameters,
- Heterogeneous data, such as demographic and metereological data,

or even any combination of the above.

Often datasets like this cannot easily fit into a single :py:class:`~xarray.Dataset` object,
or are more usefully thought of as groups of related :py:class:`~xarray.Dataset` objects.
For this purpose we provide the :py:class:`xarray.DataTree` class.

This page explains in detail how to understand and use the different features
of the :py:class:`~xarray.DataTree` class for your own hierarchical data needs.

.. _node relationships:

Node Relationships
------------------

.. _creating a family tree:

Creating a Family Tree
~~~~~~~~~~~~~~~~~~~~~~

The three main ways of creating a :py:class:`~xarray.DataTree` object are described briefly in :ref:`creating a datatree`.
Here we go into more detail about how to create a tree node-by-node, using a famous family tree from the Simpsons cartoon as an example.

Let's start by defining nodes representing the two siblings, Bart and Lisa Simpson:

.. ipython:: python

    bart = xr.DataTree(name="Bart")
    lisa = xr.DataTree(name="Lisa")

Each of these node objects knows their own :py:class:`~xarray.DataTree.name`, but they currently have no relationship to one another.
We can connect them by creating another node representing a common parent, Homer Simpson:

.. ipython:: python

    homer = xr.DataTree(name="Homer", children={"Bart": bart, "Lisa": lisa})

Here we set the children of Homer in the node's constructor.
We now have a small family tree

.. ipython:: python

    homer

where we can see how these individual Simpson family members are related to one another.
The nodes representing Bart and Lisa are now connected - we can confirm their sibling rivalry by examining the :py:class:`~xarray.DataTree.siblings` property:

.. ipython:: python

    list(homer["Bart"].siblings)

But oops, we forgot Homer's third daughter, Maggie! Let's add her by updating Homer's :py:class:`~xarray.DataTree.children` property to include her:

.. ipython:: python

    maggie = xr.DataTree(name="Maggie")
    homer.children = {"Bart": bart, "Lisa": lisa, "Maggie": maggie}
    homer

Let's check that Maggie knows who her Dad is:

.. ipython:: python

    maggie.parent.name

That's good - updating the properties of our nodes does not break the internal consistency of our tree, as changes of parentage are automatically reflected on both nodes.

    These children obviously have another parent, Marge Simpson, but :py:class:`~xarray.DataTree` nodes can only have a maximum of one parent.
    Genealogical `family trees are not even technically trees <https://en.wikipedia.org/wiki/Family_tree#Graph_theory>`_ in the mathematical sense -
    the fact that distant relatives can mate makes them directed acyclic graphs.
    Trees of :py:class:`~xarray.DataTree` objects cannot represent this.

Homer is currently listed as having no parent (the so-called "root node" of this tree), but we can update his :py:class:`~xarray.DataTree.parent` property:

.. ipython:: python

    abe = xr.DataTree(name="Abe")
    abe.children = {"Homer": homer}

Abe is now the "root" of this tree, which we can see by examining the :py:class:`~xarray.DataTree.root` property of any node in the tree

.. ipython:: python

    maggie.root.name

We can see the whole tree by printing Abe's node or just part of the tree by printing Homer's node:

.. ipython:: python

    abe
    abe["Homer"]


In episode 28, Abe Simpson reveals that he had another son, Herbert "Herb" Simpson.
We can add Herbert to the family tree without displacing Homer by :py:meth:`~xarray.DataTree.assign`-ing another child to Abe:

.. ipython:: python

    herbert = xr.DataTree(name="Herb")
    abe = abe.assign({"Herbert": herbert})
    abe

    abe["Herbert"].name
    herbert.name

.. note::
   This example shows a subtlety - the returned tree has Homer's brother listed as ``"Herbert"``,
   but the original node was named "Herb". Not only are names overridden when stored as keys like this,
   but the new node is a copy, so that the original node that was referenced is unchanged (i.e. ``herbert.name == "Herb"`` still).
   In other words, nodes are copied into trees, not inserted into them.
   This is intentional, and mirrors the behaviour when storing named :py:class:`~xarray.DataArray` objects inside datasets.

Certain manipulations of our tree are forbidden, if they would create an inconsistent result.
In episode 51 of the show Futurama, Philip J. Fry travels back in time and accidentally becomes his own Grandfather.
If we try similar time-travelling hijinks with Homer, we get a :py:class:`~xarray.InvalidTreeError` raised:

.. ipython:: python
    :okexcept:

    abe["Homer"].children = {"Abe": abe}

.. _evolutionary tree:

Ancestry in an Evolutionary Tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's use a different example of a tree to discuss more complex relationships between nodes - the phylogenetic tree, or tree of life.

.. ipython:: python

    vertebrates = xr.DataTree.from_dict(
        {
            "/Sharks": None,
            "/Bony Skeleton/Ray-finned Fish": None,
            "/Bony Skeleton/Four Limbs/Amphibians": None,
            "/Bony Skeleton/Four Limbs/Amniotic Egg/Hair/Primates": None,
            "/Bony Skeleton/Four Limbs/Amniotic Egg/Hair/Rodents & Rabbits": None,
            "/Bony Skeleton/Four Limbs/Amniotic Egg/Two Fenestrae/Dinosaurs": None,
            "/Bony Skeleton/Four Limbs/Amniotic Egg/Two Fenestrae/Birds": None,
        },
        name="Vertebrae",
    )

    primates = vertebrates["/Bony Skeleton/Four Limbs/Amniotic Egg/Hair/Primates"]
    dinosaurs = vertebrates[
        "/Bony Skeleton/Four Limbs/Amniotic Egg/Two Fenestrae/Dinosaurs"
    ]

We have used the :py:meth:`~xarray.DataTree.from_dict` constructor method as a preferred way to quickly create a whole tree,
and :ref:`filesystem paths` (to be explained shortly) to select two nodes of interest.

.. ipython:: python

    vertebrates

This tree shows various families of species, grouped by their common features (making it technically a `"Cladogram" <https://en.wikipedia.org/wiki/Cladogram>`_,
rather than an evolutionary tree).

Here both the species and the features used to group them are represented by :py:class:`~xarray.DataTree` node objects - there is no distinction in types of node.
We can however get a list of only the nodes we used to represent species by using the fact that all those nodes have no children - they are "leaf nodes".
We can check if a node is a leaf with :py:meth:`~xarray.DataTree.is_leaf`, and get a list of all leaves with the :py:class:`~xarray.DataTree.leaves` property:

.. ipython:: python

    primates.is_leaf
    [node.name for node in vertebrates.leaves]

Pretending that this is a true evolutionary tree for a moment, we can find the features of the evolutionary ancestors (so-called "ancestor" nodes),
the distinguishing feature of the common ancestor of all vertebrate life (the root node),
and even the distinguishing feature of the common ancestor of any two species (the common ancestor of two nodes):

.. ipython:: python

    [node.name for node in reversed(primates.parents)]
    primates.root.name
    primates.find_common_ancestor(dinosaurs).name

We can only find a common ancestor between two nodes that lie in the same tree.
If we try to find the common evolutionary ancestor between primates and an Alien species that has no relationship to Earth's evolutionary tree,
an error will be raised.

.. ipython:: python
    :okexcept:

    alien = xr.DataTree(name="Xenomorph")
    primates.find_common_ancestor(alien)


.. _navigating trees:

Navigating Trees
----------------

There are various ways to access the different nodes in a tree.

Properties
~~~~~~~~~~

We can navigate trees using the :py:class:`~xarray.DataTree.parent` and :py:class:`~xarray.DataTree.children` properties of each node, for example:

.. ipython:: python

    lisa.parent.children["Bart"].name

but there are also more convenient ways to access nodes.

Dictionary-like interface
~~~~~~~~~~~~~~~~~~~~~~~~~

Children are stored on each node as a key-value mapping from name to child node.
They can be accessed and altered via the :py:class:`~xarray.DataTree.__getitem__` and :py:class:`~xarray.DataTree.__setitem__` syntax.
In general :py:class:`~xarray.DataTree.DataTree` objects support almost the entire set of dict-like methods,
including :py:meth:`~xarray.DataTree.keys`, :py:class:`~xarray.DataTree.values`, :py:class:`~xarray.DataTree.items`,
:py:meth:`~xarray.DataTree.__delitem__` and :py:meth:`~xarray.DataTree.update`.

.. ipython:: python

    vertebrates["Bony Skeleton"]["Ray-finned Fish"]

Note that the dict-like interface combines access to child :py:class:`~xarray.DataTree` nodes and stored :py:class:`~xarray.DataArrays`,
so if we have a node that contains both children and data, calling :py:meth:`~xarray.DataTree.keys` will list both names of child nodes and
names of data variables:

.. ipython:: python

    dt = xr.DataTree(
        dataset=xr.Dataset({"foo": 0, "bar": 1}),
        children={"a": xr.DataTree(), "b": xr.DataTree()},
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

    Future development will allow you to make the filesystem analogy concrete by
    using :py:func:`~xarray.DataTree.open_mfdatatree` or
    :py:func:`~xarray.DataTree.save_mfdatatree`.
    (`See related issue in GitHub <https://github.com/xarray-contrib/datatree/issues/55>`_)

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

    # access lisa's sibling by a relative path.
    lisa["../Bart"]
    # or from absolute path
    lisa["/Homer/Bart"]


Relative paths between nodes also support the ``"../"`` syntax to mean the parent of the current node.
We can use this with ``__setitem__`` to add a missing entry to our evolutionary tree, but add it relative to a more familiar node of interest:

.. ipython:: python

    primates["../../Two Fenestrae/Crocodiles"] = xr.DataTree()
    print(vertebrates)

Given two nodes in a tree, we can also find their relative path:

.. ipython:: python

    bart.relative_to(lisa)

You can use this filepath feature to build a nested tree from a dictionary of filesystem-like paths and corresponding :py:class:`~xarray.Dataset` objects in a single step.
If we have a dictionary where each key is a valid path, and each value is either valid data or ``None``,
we can construct a complex tree quickly using the alternative constructor :py:meth:`~xarray.DataTree.from_dict()`:

.. ipython:: python

    d = {
        "/": xr.Dataset({"foo": "orange"}),
        "/a": xr.Dataset({"bar": 0}, coords={"y": ("y", [0, 1, 2])}),
        "/a/b": xr.Dataset({"zed": np.nan}),
        "a/c/d": None,
    }
    dt = xr.DataTree.from_dict(d)
    dt

.. note::

    Notice that using the path-like syntax will also create any intermediate empty nodes necessary to reach the end of the specified path
    (i.e. the node labelled ``"/a/c"`` in this case.)
    This is to help avoid lots of redundant entries when creating deeply-nested trees using :py:meth:`xarray.DataTree.from_dict`.

.. _iterating over trees:

Iterating over trees
~~~~~~~~~~~~~~~~~~~~

You can iterate over every node in a tree using the subtree :py:class:`~xarray.DataTree.subtree` property.
This returns an iterable of nodes, which yields them in depth-first order.

.. ipython:: python

    for node in vertebrates.subtree:
        print(node.path)

Similarly, :py:class:`~xarray.DataTree.subtree_with_keys` returns an iterable of
relative paths and corresponding nodes.

A very useful pattern is to iterate over :py:class:`~xarray.DataTree.subtree_with_keys`
to manipulate nodes however you wish, then rebuild a new tree using
:py:meth:`xarray.DataTree.from_dict()`.
For example, we could keep only the nodes containing data by looping over all nodes,
checking if they contain any data using :py:class:`~xarray.DataTree.has_data`,
then rebuilding a new tree using only the paths of those nodes:

.. ipython:: python

    non_empty_nodes = {
        path: node.dataset for path, node in dt.subtree_with_keys if node.has_data
    }
    xr.DataTree.from_dict(non_empty_nodes)

You can see this tree is similar to the ``dt`` object above, except that it is missing the empty nodes ``a/c`` and ``a/c/d``.

(If you want to keep the name of the root node, you will need to add the ``name`` kwarg to :py:class:`~xarray.DataTree.from_dict`, i.e. ``DataTree.from_dict(non_empty_nodes, name=dt.name)``.)

.. _manipulating trees:

Manipulating Trees
------------------

Subsetting Tree Nodes
~~~~~~~~~~~~~~~~~~~~~

We can subset our tree to select only nodes of interest in various ways.

Similarly to on a real filesystem, matching nodes by common patterns in their paths is often useful.
We can use :py:meth:`xarray.DataTree.match` for this:

.. ipython:: python

    dt = xr.DataTree.from_dict(
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
:py:meth:`xarray.DataTree.filter` retains only the nodes of a tree that meet a certain condition.
For example, we could recreate the Simpson's family tree with the ages of each individual, then filter for only the adults:
First lets recreate the tree but with an ``age`` data variable in every node:

.. ipython:: python

    simpsons = xr.DataTree.from_dict(
        {
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

(Yes, under the hood :py:meth:`~xarray.DataTree.filter` is just syntactic sugar for the pattern we showed you in :ref:`iterating over trees` !)

.. _Tree Contents:

Tree Contents
-------------

Hollow Trees
~~~~~~~~~~~~

A concept that can sometimes be useful is that of a "Hollow Tree", which means a tree with data stored only at the leaf nodes.
This is useful because certain useful tree manipulation operations only make sense for hollow trees.

You can check if a tree is a hollow tree by using the :py:class:`~xarray.DataTree.is_hollow` property.
We can see that the Simpson's family is not hollow because the data variable ``"age"`` is present at some nodes which
have children (i.e. Abe and Homer).

.. ipython:: python

    simpsons.is_hollow

.. _tree computation:

Computation
-----------

:py:class:`~xarray.DataTree` objects are also useful for performing computations, not just for organizing data.

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

    voltages = xr.DataTree.from_dict(
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

You can map custom computation over each node in a tree using :py:meth:`xarray.DataTree.map_over_datasets`.
You can map any function, so long as it takes :py:class:`xarray.Dataset` objects as one (or more) of the input arguments,
and returns one (or more) xarray datasets.

.. note::

    Functions passed to :py:func:`~xarray.DataTree.map_over_datasets` cannot alter nodes in-place.
    Instead they must return new :py:class:`xarray.Dataset` objects.

For example, we can define a function to calculate the Root Mean Square of a timeseries

.. ipython:: python

    def rms(signal):
        return np.sqrt(np.mean(signal**2))

Then calculate the RMS value of these signals:

.. ipython:: python

    voltages.map_over_datasets(rms)

.. _multiple trees:

We can also use :py:func:`~xarray.map_over_datasets` to apply a function over
the data in multiple trees, by passing the trees as positional arguments.

Operating on Multiple Trees
---------------------------

The examples so far have involved mapping functions or methods over the nodes of a single tree,
but we can generalize this to mapping functions over multiple trees at once.

Iterating Over Multiple Trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To iterate over the corresponding nodes in multiple trees, use
:py:func:`~xarray.group_subtrees` instead of
:py:class:`~xarray.DataTree.subtree_with_keys`. This combines well with
:py:meth:`xarray.DataTree.from_dict()` to build a new tree:

.. ipython:: python

    dt1 = xr.DataTree.from_dict({"a": xr.Dataset({"x": 1}), "b": xr.Dataset({"x": 2})})
    dt2 = xr.DataTree.from_dict(
        {"a": xr.Dataset({"x": 10}), "b": xr.Dataset({"x": 20})}
    )
    result = {}
    for path, (node1, node2) in xr.group_subtrees(dt1, dt2):
        result[path] = node1.dataset + node2.dataset
    xr.DataTree.from_dict(result)

Alternatively, you apply a function directly to paired datasets at every node
using :py:func:`xarray.map_over_datasets`:

.. ipython:: python

    xr.map_over_datasets(lambda x, y: x + y, dt1, dt2)

Comparing Trees for Isomorphism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For it to make sense to map a single non-unary function over the nodes of multiple trees at once,
each tree needs to have the same structure. Specifically two trees can only be considered similar,
or "isomorphic", if the full paths to all of their descendent nodes are the same.

Applying :py:func:`~xarray.group_subtrees` to trees with different structures
raises :py:class:`~xarray.TreeIsomorphismError`:

.. ipython:: python
    :okexcept:

    tree = xr.DataTree.from_dict({"a": None, "a/b": None, "a/c": None})
    simple_tree = xr.DataTree.from_dict({"a": None})
    for _ in xr.group_subtrees(tree, simple_tree):
        ...

We can explicitly also check if any two trees are isomorphic using the :py:meth:`~xarray.DataTree.isomorphic` method:

.. ipython:: python

    tree.isomorphic(simple_tree)

Corresponding tree nodes do not need to have the same data in order to be considered isomorphic:

.. ipython:: python

    tree_with_data = xr.DataTree.from_dict({"a": xr.Dataset({"foo": 1})})
    simple_tree.isomorphic(tree_with_data)

They also do not need to define child nodes in the same order:

.. ipython:: python

    reordered_tree = xr.DataTree.from_dict({"a": None, "a/c": None, "a/b": None})
    tree.isomorphic(reordered_tree)

Arithmetic Between Multiple Trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Arithmetic operations like multiplication are binary operations, so as long as we have two isomorphic trees,
we can do arithmetic between them.

.. ipython:: python

    currents = xr.DataTree.from_dict(
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

.. _hierarchical-data.alignment-and-coordinate-inheritance:

Alignment and Coordinate Inheritance
------------------------------------

.. _data-alignment:

Data Alignment
~~~~~~~~~~~~~~

The data in different datatree nodes are not totally independent. In particular dimensions (and indexes) in child nodes must be exactly aligned with those in their parent nodes.
Exact alignment means that shared dimensions must be the same length, and indexes along those dimensions must be equal.

.. note::
    If you were a previous user of the prototype `xarray-contrib/datatree <https://github.com/xarray-contrib/datatree>`_ package, this is different from what you're used to!
    In that package the data model was that the data stored in each node actually was completely unrelated. The data model is now slightly stricter.
    This allows us to provide features like :ref:`coordinate-inheritance`.

To demonstrate, let's first generate some example datasets which are not aligned with one another:

.. ipython:: python

    # (drop the attributes just to make the printed representation shorter)
    ds = xr.tutorial.open_dataset("air_temperature").drop_attrs()

    ds_daily = ds.resample(time="D").mean("time")
    ds_weekly = ds.resample(time="W").mean("time")
    ds_monthly = ds.resample(time="ME").mean("time")

These datasets have different lengths along the ``time`` dimension, and are therefore not aligned along that dimension.

.. ipython:: python

    ds_daily.sizes
    ds_weekly.sizes
    ds_monthly.sizes

We cannot store these non-alignable variables on a single :py:class:`~xarray.Dataset` object, because they do not exactly align:

.. ipython:: python
    :okexcept:

    xr.align(ds_daily, ds_weekly, ds_monthly, join="exact")

But we :ref:`previously said <why>` that multi-resolution data is a good use case for :py:class:`~xarray.DataTree`, so surely we should be able to store these in a single :py:class:`~xarray.DataTree`?
If we first try to create a :py:class:`~xarray.DataTree` with these different-length time dimensions present in both parents and children, we will still get an alignment error:

.. ipython:: python
    :okexcept:

    xr.DataTree.from_dict({"daily": ds_daily, "daily/weekly": ds_weekly})

This is because DataTree checks that data in child nodes align exactly with their parents.

.. note::
    This requirement of aligned dimensions is similar to netCDF's concept of `inherited dimensions <https://www.unidata.ucar.edu/software/netcdf/workshops/2007/groups-types/Introduction.html>`_, as in netCDF-4 files dimensions are `visible to all child groups <https://docs.unidata.ucar.edu/netcdf-c/current/groups.html>`_.

This alignment check is performed up through the tree, all the way to the root, and so is therefore equivalent to requiring that this :py:func:`~xarray.align` command succeeds:

.. code:: python

    xr.align(child.dataset, *(parent.dataset for parent in child.parents), join="exact")

To represent our unalignable data in a single :py:class:`~xarray.DataTree`, we must instead place all variables which are a function of these different-length dimensions into nodes that are not direct descendents of one another, e.g. organize them as siblings.

.. ipython:: python

    dt = xr.DataTree.from_dict(
        {"daily": ds_daily, "weekly": ds_weekly, "monthly": ds_monthly}
    )
    dt

Now we have a valid :py:class:`~xarray.DataTree` structure which contains all the data at each different time frequency, stored in a separate group.

This is a useful way to organise our data because we can still operate on all the groups at once.
For example we can extract all three timeseries at a specific lat-lon location:

.. ipython:: python

    dt.sel(lat=75, lon=300)

or compute the standard deviation of each timeseries to find out how it varies with sampling frequency:

.. ipython:: python

    dt.std(dim="time")

.. _coordinate-inheritance:

Coordinate Inheritance
~~~~~~~~~~~~~~~~~~~~~~

Notice that in the trees we constructed above there is some redundancy - the ``lat`` and ``lon`` variables appear in each sibling group, but are identical across the groups.

.. ipython:: python

    dt

We can use "Coordinate Inheritance" to define them only once in a parent group and remove this redundancy, whilst still being able to access those coordinate variables from the child groups.

.. note::
    This is also a new feature relative to the prototype `xarray-contrib/datatree <https://github.com/xarray-contrib/datatree>`_ package.

Let's instead place only the time-dependent variables in the child groups, and put the non-time-dependent ``lat`` and ``lon`` variables in the parent (root) group:

.. ipython:: python

    dt = xr.DataTree.from_dict(
        {
            "/": ds.drop_dims("time"),
            "daily": ds_daily.drop_vars(["lat", "lon"]),
            "weekly": ds_weekly.drop_vars(["lat", "lon"]),
            "monthly": ds_monthly.drop_vars(["lat", "lon"]),
        }
    )
    dt

This is preferred to the previous representation because it now makes it clear that all of these datasets share common spatial grid coordinates.
Defining the common coordinates just once also ensures that the spatial coordinates for each group cannot become out of sync with one another during operations.

We can still access the coordinates defined in the parent groups from any of the child groups as if they were actually present on the child groups:

.. ipython:: python

    dt.daily.coords
    dt["daily/lat"]

As we can still access them, we say that the ``lat`` and ``lon`` coordinates in the child groups have been "inherited" from their common parent group.

If we print just one of the child nodes, it will still display inherited coordinates, but explicitly mark them as such:

.. ipython:: python

    print(dt["/daily"])

This helps to differentiate which variables are defined on the datatree node that you are currently looking at, and which were defined somewhere above it.

We can also still perform all the same operations on the whole tree:

.. ipython:: python

    dt.sel(lat=[75], lon=[300])

    dt.std(dim="time")
