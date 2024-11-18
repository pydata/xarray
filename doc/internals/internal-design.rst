.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)
    np.set_printoptions(threshold=20)

.. _internal design:

Internal Design
===============

This page gives an overview of the internal design of xarray.

In totality, the Xarray project defines 4 key data structures.
In order of increasing complexity, they are:

- :py:class:`xarray.Variable`,
- :py:class:`xarray.DataArray`,
- :py:class:`xarray.Dataset`,
- :py:class:`xarray.DataTree`.

The user guide lists only :py:class:`xarray.DataArray` and :py:class:`xarray.Dataset`,
but :py:class:`~xarray.Variable` is the fundamental object internally,
and :py:class:`~xarray.DataTree` is a natural generalisation of :py:class:`xarray.Dataset`.

.. note::

    Our :ref:`roadmap` includes plans to document :py:class:`~xarray.Variable` as fully public API.

Internally private :ref:`lazy indexing classes <internal design.lazy indexing>` are used to avoid loading more data than necessary,
and flexible indexes classes (derived from :py:class:`~xarray.indexes.Index`) provide performant label-based lookups.


.. _internal design.data structures:

Data Structures
---------------

The :ref:`data structures` page in the user guide explains the basics and concentrates on user-facing behavior,
whereas this section explains how xarray's data structure classes actually work internally.


.. _internal design.data structures.variable:

Variable Objects
~~~~~~~~~~~~~~~~

The core internal data structure in xarray is the :py:class:`~xarray.Variable`,
which is used as the basic building block behind xarray's
:py:class:`~xarray.Dataset`, :py:class:`~xarray.DataArray` types. A
:py:class:`~xarray.Variable` consists of:

- ``dims``: A tuple of dimension names.
- ``data``: The N-dimensional array (typically a NumPy or Dask array) storing
  the Variable's data. It must have the same number of dimensions as the length
  of ``dims``.
- ``attrs``: A dictionary of metadata associated with this array. By
  convention, xarray's built-in operations never use this metadata.
- ``encoding``: Another dictionary used to store information about how
  these variable's data is represented on disk. See :ref:`io.encoding` for more
  details.

:py:class:`~xarray.Variable` has an interface similar to NumPy arrays, but extended to make use
of named dimensions. For example, it uses ``dim`` in preference to an ``axis``
argument for methods like ``mean``, and supports :ref:`compute.broadcasting`.

However, unlike ``Dataset`` and ``DataArray``, the basic ``Variable`` does not
include coordinate labels along each axis.

:py:class:`~xarray.Variable` is public API, but because of its incomplete support for labeled
data, it is mostly intended for advanced uses, such as in xarray itself, for
writing new backends, or when creating custom indexes.
You can access the variable objects that correspond to xarray objects via the (readonly)
:py:attr:`Dataset.variables <xarray.Dataset.variables>` and
:py:attr:`DataArray.variable <xarray.DataArray.variable>` attributes.


.. _internal design.dataarray:

DataArray Objects
~~~~~~~~~~~~~~~~~

The simplest data structure used by most users is :py:class:`~xarray.DataArray`.
A :py:class:`~xarray.DataArray` is a composite object consisting of multiple
:py:class:`~xarray.core.variable.Variable` objects which store related data.

A single :py:class:`~xarray.core.Variable` is referred to as the "data variable", and stored under the :py:attr:`~xarray.DataArray.variable`` attribute.
A :py:class:`~xarray.DataArray` inherits all of the properties of this data variable, i.e. ``dims``, ``data``, ``attrs`` and ``encoding``,
all of which are implemented by forwarding on to the underlying ``Variable`` object.

In addition, a :py:class:`~xarray.DataArray` stores additional ``Variable`` objects stored in a dict under the private ``_coords`` attribute,
each of which is referred to as a "Coordinate Variable". These coordinate variable objects are only allowed to have ``dims`` that are a subset of the data variable's ``dims``,
and each dim has a specific length. This means that the full :py:attr:`~xarray.DataArray.size` of the dataarray can be represented by a dictionary mapping dimension names to integer sizes.
The underlying data variable has this exact same size, and the attached coordinate variables have sizes which are some subset of the size of the data variable.
Another way of saying this is that all coordinate variables must be "alignable" with the data variable.

When a coordinate is accessed by the user (e.g. via the dict-like :py:class:`~xarray.DataArray.__getitem__` syntax),
then a new ``DataArray`` is constructed by finding all coordinate variables that have compatible dimensions and re-attaching them before the result is returned.
This is why most users never see the ``Variable`` class underlying each coordinate variable - it is always promoted to a ``DataArray`` before returning.

Lookups are performed by special :py:class:`~xarray.indexes.Index` objects, which are stored in a dict under the private ``_indexes`` attribute.
Indexes must be associated with one or more coordinates, and essentially act by translating a query given in physical coordinate space
(typically via the :py:meth:`~xarray.DataArray.sel` method) into a set of integer indices in array index space that can be used to index the underlying n-dimensional array-like ``data``.
Indexing in array index space (typically performed via the :py:meth:`~xarray.DataArray.isel` method) does not require consulting an ``Index`` object.

Finally a :py:class:`~xarray.DataArray` defines a :py:attr:`~xarray.DataArray.name` attribute, which refers to its data
variable but is stored on the wrapping ``DataArray`` class.
The ``name`` attribute is primarily used when one or more :py:class:`~xarray.DataArray` objects are promoted into a :py:class:`~xarray.Dataset`
(e.g. via :py:meth:`~xarray.DataArray.to_dataset`).
Note that the underlying :py:class:`~xarray.core.Variable` objects are all unnamed, so they can always be referred to uniquely via a
dict-like mapping.

.. _internal design.dataset:

Dataset Objects
~~~~~~~~~~~~~~~

The :py:class:`~xarray.Dataset` class is a generalization of the :py:class:`~xarray.DataArray` class that can hold multiple data variables.
Internally all data variables and coordinate variables are stored under a single ``variables`` dict, and coordinates are
specified by storing their names in a private ``_coord_names`` dict.

The dataset's ``dims`` are the set of all dims present across any variable, but (similar to in dataarrays) coordinate
variables cannot have a dimension that is not present on any data variable.

When a data variable or coordinate variable is accessed, a new ``DataArray`` is again constructed from all compatible
coordinates before returning.

.. _internal design.subclassing:

.. note::

    The way that selecting a variable from a ``DataArray`` or ``Dataset`` actually involves internally wrapping the
    ``Variable`` object back up into a ``DataArray``/``Dataset`` is the primary reason :ref:`we recommend against subclassing <internals.accessors.composition>`
    Xarray objects. The main problem it creates is that we currently cannot easily guarantee that for example selecting
    a coordinate variable from your ``SubclassedDataArray`` would return an instance of ``SubclassedDataArray`` instead
    of just an :py:class:`xarray.DataArray`. See `GH issue <https://github.com/pydata/xarray/issues/3980>`_ for more details.

.. _internal design.lazy indexing:

Lazy Indexing Classes
---------------------

Lazy Loading
~~~~~~~~~~~~

If we open a ``Variable`` object from disk using :py:func:`~xarray.open_dataset` we can see that the actual values of
the array wrapped by the data variable are not displayed.

.. ipython:: python

    da = xr.tutorial.open_dataset("air_temperature")["air"]
    var = da.variable
    var

We can see the size, and the dtype of the underlying array, but not the actual values.
This is because the values have not yet been loaded.

If we look at the private attribute :py:meth:`~xarray.Variable._data` containing the underlying array object, we see
something interesting:

.. ipython:: python

    var._data

You're looking at one of xarray's internal Lazy Indexing Classes. These powerful classes are hidden from the user,
but provide important functionality.

Calling the public :py:attr:`~xarray.Variable.data` property loads the underlying array into memory.

.. ipython:: python

    var.data

This array is now cached, which we can see by accessing the private attribute again:

.. ipython:: python

    var._data

Lazy Indexing
~~~~~~~~~~~~~

The purpose of these lazy indexing classes is to prevent more data being loaded into memory than is necessary for the
subsequent analysis, by deferring loading data until after indexing is performed.

Let's open the data from disk again.

.. ipython:: python

    da = xr.tutorial.open_dataset("air_temperature")["air"]
    var = da.variable

Now, notice how even after subsetting the data has does not get loaded:

.. ipython:: python

    var.isel(time=0)

The shape has changed, but the values are still not shown.

Looking at the private attribute again shows how this indexing information was propagated via the hidden lazy indexing classes:

.. ipython:: python

    var.isel(time=0)._data

.. note::

    Currently only certain indexing operations are lazy, not all array operations. For discussion of making all array
    operations lazy see `GH issue #5081 <https://github.com/pydata/xarray/issues/5081>`_.


Lazy Dask Arrays
~~~~~~~~~~~~~~~~

Note that xarray's implementation of Lazy Indexing classes is completely separate from how :py:class:`dask.array.Array`
objects evaluate lazily. Dask-backed xarray objects delay almost all operations until :py:meth:`~xarray.DataArray.compute`
is called (either explicitly or implicitly via :py:meth:`~xarray.DataArray.plot` for example). The exceptions to this
laziness are operations whose output shape is data-dependent, such as when calling :py:meth:`~xarray.DataArray.where`.
