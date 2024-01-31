.. currentmodule:: datatree

.. _io:

Reading and Writing Files
=========================

.. note::

    This page builds on the information given in xarray's main page on
    `reading and writing files <https://docs.xarray.dev/en/stable/user-guide/io.html>`_,
    so it is suggested that you are familiar with those first.


netCDF
------

Groups
~~~~~~

Whilst netCDF groups can only be loaded individually as Dataset objects, a whole file of many nested groups can be loaded
as a single :py:class:`DataTree` object.
To open a whole netCDF file as a tree of groups use the :py:func:`open_datatree` function.
To save a DataTree object as a netCDF file containing many groups, use the :py:meth:`DataTree.to_netcdf` method.


.. _netcdf.group.warning:

.. warning::
    ``DataTree`` objects do not follow the exact same data model as netCDF files, which means that perfect round-tripping
    is not always possible.

    In particular in the netCDF data model dimensions are entities that can exist regardless of whether any variable possesses them.
    This is in contrast to `xarray's data model <https://docs.xarray.dev/en/stable/user-guide/data-structures.html>`_
    (and hence :ref:`datatree's data model <data structures>`) in which the dimensions of a (Dataset/Tree)
    object are simply the set of dimensions present across all variables in that dataset.

    This means that if a netCDF file contains dimensions but no variables which possess those dimensions,
    these dimensions will not be present when that file is opened as a DataTree object.
    Saving this DataTree object to file will therefore not preserve these "unused" dimensions.

Zarr
----

Groups
~~~~~~

Nested groups in zarr stores can be represented by loading the store as a :py:class:`DataTree` object, similarly to netCDF.
To open a whole zarr store as a tree of groups use the :py:func:`open_datatree` function.
To save a DataTree object as a zarr store containing many groups, use the :py:meth:`DataTree.to_zarr()` method.

.. note::
    Note that perfect round-tripping should always be possible with a zarr store (:ref:`unlike for netCDF files <netcdf.group.warning>`),
    as zarr does not support "unused" dimensions.
