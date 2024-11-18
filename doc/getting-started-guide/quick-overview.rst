##############
Quick overview
##############

Here are some quick examples of what you can do with :py:class:`xarray.DataArray`
objects. Everything is explained in much more detail in the rest of the
documentation.

To begin, import numpy, pandas and xarray using their customary abbreviations:

.. ipython:: python

    import numpy as np
    import pandas as pd
    import xarray as xr

Create a DataArray
------------------

You can make a DataArray from scratch by supplying data in the form of a numpy
array or list, with optional *dimensions* and *coordinates*:

.. ipython:: python

    data = xr.DataArray(np.random.randn(2, 3), dims=("x", "y"), coords={"x": [10, 20]})
    data

In this case, we have generated a 2D array, assigned the names *x* and *y* to the two dimensions respectively and associated two *coordinate labels* '10' and '20' with the two locations along the x dimension. If you supply a pandas :py:class:`~pandas.Series` or :py:class:`~pandas.DataFrame`, metadata is copied directly:

.. ipython:: python

    xr.DataArray(pd.Series(range(3), index=list("abc"), name="foo"))

Here are the key properties for a ``DataArray``:

.. ipython:: python

    # like in pandas, values is a numpy array that you can modify in-place
    data.values
    data.dims
    data.coords
    # you can use this dictionary to store arbitrary metadata
    data.attrs


Indexing
--------

Xarray supports four kinds of indexing. Since we have assigned coordinate labels to the x dimension we can use label-based indexing along that dimension just like pandas. The four examples below all yield the same result (the value at ``x=10``) but at varying levels of convenience and intuitiveness.

.. ipython:: python

    # positional and by integer label, like numpy
    data[0, :]

    # loc or "location": positional and coordinate label, like pandas
    data.loc[10]

    # isel or "integer select":  by dimension name and integer label
    data.isel(x=0)

    # sel or "select": by dimension name and coordinate label
    data.sel(x=10)


Unlike positional indexing, label-based indexing frees us from having to know how our array is organized. All we need to know are the dimension name and the label we wish to index i.e. ``data.sel(x=10)`` works regardless of whether ``x`` is the first or second dimension of the array and regardless of whether ``10`` is the first or second element of ``x``. We have already told xarray that x is the first dimension when we created ``data``: xarray keeps track of this so we don't have to. For more, see :ref:`indexing`.


Attributes
----------

While you're setting up your DataArray, it's often a good idea to set metadata attributes. A useful choice is to set ``data.attrs['long_name']`` and ``data.attrs['units']`` since xarray will use these, if present, to automatically label your plots. These special names were chosen following the `NetCDF Climate and Forecast (CF) Metadata Conventions <https://cfconventions.org/cf-conventions/cf-conventions.html>`_. ``attrs`` is just a Python dictionary, so you can assign anything you wish.

.. ipython:: python

    data.attrs["long_name"] = "random velocity"
    data.attrs["units"] = "metres/sec"
    data.attrs["description"] = "A random variable created as an example."
    data.attrs["random_attribute"] = 123
    data.attrs
    # you can add metadata to coordinates too
    data.x.attrs["units"] = "x units"


Computation
-----------

Data arrays work very similarly to numpy ndarrays:

.. ipython:: python

    data + 10
    np.sin(data)
    # transpose
    data.T
    data.sum()

However, aggregation operations can use dimension names instead of axis
numbers:

.. ipython:: python

    data.mean(dim="x")

Arithmetic operations broadcast based on dimension name. This means you don't
need to insert dummy dimensions for alignment:

.. ipython:: python

    a = xr.DataArray(np.random.randn(3), [data.coords["y"]])
    b = xr.DataArray(np.random.randn(4), dims="z")

    a
    b

    a + b

It also means that in most cases you do not need to worry about the order of
dimensions:

.. ipython:: python

    data - data.T

Operations also align based on index labels:

.. ipython:: python

    data[:-1] - data[:1]

For more, see :ref:`comput`.

GroupBy
-------

Xarray supports grouped operations using a very similar API to pandas (see :ref:`groupby`):

.. ipython:: python

    labels = xr.DataArray(["E", "F", "E"], [data.coords["y"]], name="labels")
    labels
    data.groupby(labels).mean("y")
    data.groupby(labels).map(lambda x: x - x.min())

Plotting
--------

Visualizing your datasets is quick and convenient:

.. ipython:: python

    @savefig plotting_quick_overview.png
    data.plot()

Note the automatic labeling with names and units. Our effort in adding metadata attributes has paid off! Many aspects of these figures are customizable: see :ref:`plotting`.

pandas
------

Xarray objects can be easily converted to and from pandas objects using the :py:meth:`~xarray.DataArray.to_series`, :py:meth:`~xarray.DataArray.to_dataframe` and :py:meth:`~pandas.DataFrame.to_xarray` methods:

.. ipython:: python

    series = data.to_series()
    series

    # convert back
    series.to_xarray()

Datasets
--------

:py:class:`xarray.Dataset` is a dict-like container of aligned ``DataArray``
objects. You can think of it as a multi-dimensional generalization of the
:py:class:`pandas.DataFrame`:

.. ipython:: python

    ds = xr.Dataset(dict(foo=data, bar=("x", [1, 2]), baz=np.pi))
    ds


This creates a dataset with three DataArrays named ``foo``, ``bar`` and ``baz``. Use dictionary or dot indexing to pull out ``Dataset`` variables as ``DataArray`` objects but note that assignment only works with dictionary indexing:

.. ipython:: python

    ds["foo"]
    ds.foo


When creating ``ds``, we specified that ``foo`` is identical to ``data`` created earlier, ``bar`` is one-dimensional with single dimension ``x`` and associated values '1' and '2', and ``baz`` is a scalar not associated with any dimension in ``ds``. Variables in datasets can have different ``dtype`` and even different dimensions, but all dimensions are assumed to refer to points in the same shared coordinate system i.e. if two variables have dimension ``x``, that dimension must be identical in both variables.

For example, when creating ``ds`` xarray automatically *aligns* ``bar`` with ``DataArray`` ``foo``, i.e., they share the same coordinate system so that ``ds.bar['x'] == ds.foo['x'] == ds['x']``. Consequently, the following works without explicitly specifying the coordinate ``x`` when creating ``ds['bar']``:

.. ipython:: python

    ds.bar.sel(x=10)



You can do almost everything you can do with ``DataArray`` objects with
``Dataset`` objects (including indexing and arithmetic) if you prefer to work
with multiple variables at once.

Read & write netCDF files
-------------------------

NetCDF is the recommended file format for xarray objects. Users
from the geosciences will recognize that the :py:class:`~xarray.Dataset` data
model looks very similar to a netCDF file (which, in fact, inspired it).

You can directly read and write xarray objects to disk using :py:meth:`~xarray.Dataset.to_netcdf`, :py:func:`~xarray.open_dataset` and
:py:func:`~xarray.open_dataarray`:

.. ipython:: python

    ds.to_netcdf("example.nc")
    reopened = xr.open_dataset("example.nc")
    reopened

.. ipython:: python
    :suppress:

    import os

    reopened.close()
    os.remove("example.nc")


It is common for datasets to be distributed across multiple files (commonly one file per timestep). Xarray supports this use-case by providing the :py:meth:`~xarray.open_mfdataset` and the :py:meth:`~xarray.save_mfdataset` methods. For more, see :ref:`io`.


.. _quick-overview-datatrees:

DataTrees
---------

:py:class:`xarray.DataTree` is a tree-like container of :py:class:`~xarray.DataArray` objects, organised into multiple mutually alignable groups. You can think of it like a (recursive) ``dict`` of :py:class:`~xarray.Dataset` objects, where coordinate variables and their indexes are inherited down to children.

Let's first make some example xarray datasets:

.. ipython:: python

    import numpy as np
    import xarray as xr

    data = xr.DataArray(np.random.randn(2, 3), dims=("x", "y"), coords={"x": [10, 20]})
    ds = xr.Dataset({"foo": data, "bar": ("x", [1, 2]), "baz": np.pi})
    ds

    ds2 = ds.interp(coords={"x": [10, 12, 14, 16, 18, 20]})
    ds2

    ds3 = xr.Dataset(
        {"people": ["alice", "bob"], "heights": ("people", [1.57, 1.82])},
        coords={"species": "human"},
    )
    ds3

Now we'll put these datasets into a hierarchical DataTree:

.. ipython:: python

    dt = xr.DataTree.from_dict(
        {"simulation/coarse": ds, "simulation/fine": ds2, "/": ds3}
    )
    dt

This created a DataTree with nested groups. We have one root group, containing information about individual
people.  This root group can be named, but here it is unnamed, and is referenced with ``"/"``. This structure is similar to a
unix-like filesystem.  The root group then has one subgroup ``simulation``, which contains no data itself but does
contain another two subgroups, named ``fine`` and ``coarse``.

The (sub)subgroups ``fine`` and ``coarse`` contain two very similar datasets.  They both have an ``"x"``
dimension, but the dimension is of different lengths in each group, which makes the data in each group
unalignable.  In the root group we placed some completely unrelated information, in order to show how a tree can
store heterogeneous data.

Remember to keep unalignable dimensions in sibling groups because a DataTree inherits coordinates down through its
child nodes.  You can see this inheritance in the above representation of the DataTree.  The coordinates
``people`` and ``species`` defined in the root ``/`` node are shown in the child nodes both
``/simulation/coarse`` and ``/simulation/fine``.  All coordinates in parent-descendent lineage must be
alignable to form a DataTree.  If your input data is not aligned, you can still get a nested ``dict`` of
:py:class:`~xarray.Dataset` objects with :py:func:`~xarray.open_groups` and then apply any required changes to ensure alignment
before converting to a :py:class:`~xarray.DataTree`.

The constraints on each group are the same as the constraint on DataArrays within a single dataset with the
addition of requiring parent-descendent coordinate agreement.

We created the subgroups using a filesystem-like syntax, and accessing groups works the same way.  We can access
individual DataArrays in a similar fashion.

.. ipython:: python

    dt["simulation/coarse/foo"]

We can also view the data in a particular group as a read-only :py:class:`~xarray.Datatree.DatasetView` using :py:attr:`xarray.Datatree.dataset`:

.. ipython:: python

    dt["simulation/coarse"].dataset

We can get a copy of the :py:class:`~xarray.Dataset` including the inherited coordinates by calling the :py:class:`~xarray.datatree.to_dataset` method:

.. ipython:: python

    ds_inherited = dt["simulation/coarse"].to_dataset()
    ds_inherited

And you can get a copy of just the node local values of :py:class:`~xarray.Dataset` by setting the ``inherit`` keyword to ``False``:

.. ipython:: python

    ds_node_local = dt["simulation/coarse"].to_dataset(inherit=False)
    ds_node_local

.. note::

    We intend to eventually implement most :py:class:`~xarray.Dataset` methods
    (indexing, aggregation, arithmetic, etc) on :py:class:`~xarray.DataTree`
    objects, but many methods have not been implemented yet.

.. Operations map over subtrees, so we can take a mean over the ``x`` dimension of both the ``fine`` and ``coarse`` groups just by:

.. .. ipython:: python

..     avg = dt["simulation"].mean(dim="x")
..     avg

.. Here the ``"x"`` dimension used is always the one local to that subgroup.


.. You can do almost everything you can do with :py:class:`~xarray.Dataset` objects with :py:class:`~xarray.DataTree` objects
.. (including indexing and arithmetic), as operations will be mapped over every subgroup in the tree.
.. This allows you to work with multiple groups of non-alignable variables at once.

.. tip::

    If all of your variables are mutually alignable (i.e., they live on the same
    grid, such that every common dimension name maps to the same length), then
    you probably don't need :py:class:`xarray.DataTree`, and should consider
    just sticking with :py:class:`xarray.Dataset`.
