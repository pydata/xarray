.. currentmodule:: datatree

##############
Quick overview
##############

DataTrees
---------

:py:class:`DataTree` is a tree-like container of :py:class:`xarray.DataArray` objects, organised into multiple mutually alignable groups.
You can think of it like a (recursive) ``dict`` of :py:class:`xarray.Dataset` objects.

Let's first make some example xarray datasets (following on from xarray's
`quick overview <https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html>`_ page):

.. ipython:: python

    import numpy as np
    import xarray as xr

    data = xr.DataArray(np.random.randn(2, 3), dims=("x", "y"), coords={"x": [10, 20]})
    ds = xr.Dataset(dict(foo=data, bar=("x", [1, 2]), baz=np.pi))
    ds

    ds2 = ds.interp(coords={"x": [10, 12, 14, 16, 18, 20]})
    ds2

    ds3 = xr.Dataset(
        dict(people=["alice", "bob"], heights=("people", [1.57, 1.82])),
        coords={"species": "human"},
    )
    ds3

Now we'll put this data into a multi-group tree:

.. ipython:: python

    from datatree import DataTree

    dt = DataTree.from_dict({"simulation/coarse": ds, "simulation/fine": ds2, "/": ds3})
    dt

This creates a datatree with various groups. We have one root group, containing information about individual people.
(This root group can be named, but here is unnamed, so is referred to with ``"/"``, same as the root of a unix-like filesystem.)
The root group then has one subgroup ``simulation``, which contains no data itself but does contain another two subgroups,
named ``fine`` and ``coarse``.

The (sub-)sub-groups ``fine`` and ``coarse`` contain two very similar datasets.
They both have an ``"x"`` dimension, but the dimension is of different lengths in each group, which makes the data in each group unalignable.
In the root group we placed some completely unrelated information, showing how we can use a tree to store heterogenous data.

The constraints on each group are therefore the same as the constraint on dataarrays within a single dataset.

We created the sub-groups using a filesystem-like syntax, and accessing groups works the same way.
We can access individual dataarrays in a similar fashion

.. ipython:: python

    dt["simulation/coarse/foo"]

and we can also pull out the data in a particular group as a ``Dataset`` object using ``.ds``:

.. ipython:: python

    dt["simulation/coarse"].ds

Operations map over subtrees, so we can take a mean over the ``x`` dimension of both the ``fine`` and ``coarse`` groups just by

.. ipython:: python

    avg = dt["simulation"].mean(dim="x")
    avg

Here the ``"x"`` dimension used is always the one local to that sub-group.

You can do almost everything you can do with ``Dataset`` objects with ``DataTree`` objects
(including indexing and arithmetic), as operations will be mapped over every sub-group in the tree.
This allows you to work with multiple groups of non-alignable variables at once.

.. note::

    If all of your variables are mutually alignable
    (i.e. they live on the same grid, such that every common dimension name maps to the same length),
    then you probably don't need :py:class:`DataTree`, and should consider just sticking with ``xarray.Dataset``.
