.. _interoperability:

Interoperability of Xarray
==========================

Xarray is designed to be extremely interoperable, in many orthogonal ways.
Making xarray as flexible as possible is the common theme of most of the goals on our development :ref:`roadmap`.

This interoperability comes via a set of flexible abstractions into which the user can plug in. The current full list is:

- :ref:`Custom file backends <add_a_backend>` via the :py:class:`~xarray.backends.BackendEntrypoint` system,
- Numpy-like :ref:`"duck" array wrapping <internals.duckarrays>`, which supports the `Python Array API Standard <https://data-apis.org/array-api/latest/>`_,
- :ref:`Chunked distributed array computation <internals.chunkedarrays>` via the :py:class:`~xarray.core.parallelcompat.ChunkManagerEntrypoint` system,
- Custom :py:class:`~xarray.indexes.Index` objects for :ref:`flexible label-based lookups <internals.custom indexes>`,
- Extending xarray objects with domain-specific methods via :ref:`custom accessors <internals.accessors>`.

.. warning::

    One obvious way in which xarray could be more flexible is that whilst subclassing xarray objects is possible, we
    generally advise against it, instead recommending composition over inheritance. See the
    :ref:`internal design page <internal design.subclassing>` and `GH issue <https://github.com/pydata/xarray/issues/3980>`_
    for more details.

.. note::

    If you think there is another way in which xarray could become more generically flexible then please
    tell us your ideas by `raising an issue to request the feature <https://github.com/pydata/xarray/issues/new/choose>`_!


Whilst xarray was originally designed specifically to open ``netCDF4`` files as ``numpy.ndarray``s labelled by ``pandas.Index`` objects,
it is entirely possible today to:

- lazily open an xarray object directly from a custom binary file format (e.g. using ``open_dataset(path, engine='my_custom_format')``,
- handle the data as any API-compliant numpy-like array type (e.g. sparse or GPU-backed),
- distribute out-of-core computation across that array type in parallel (e.g. via :py:class:`dask.array`),
- track the physical units of the data through computations (e.g via ``pint``),
- query the data via custom index logic optimized for specific applications (e.g. an ``Index`` object backed by a KDTree structure),
- attach domain-specific logic via accessor methods (e.g. to understand geographic Coordinate Reference System metadata),
- organize hierarchical groups of xarray data in a :py:class:`~datatree.DataTree` (e.g. to treat heterogenous simulation and observational data together during analysis).

All of these features can be provided simultaneously, using libaries compatible with the rest of the scientific python ecosystem.
In this situation xarray would be essentially a thin wrapper acting as pure-python framework, providing a common interface and
separation of concerns via various domain-agnostic abstractions.

Most of the remaining pages in the documentation of xarray's internals describe these various types of interoperability in more detail.
