.. currentmodule:: xarray
.. _io.zarr:

Zarr
====

.. jupyter-execute::
    :hide-code:

    import os

    import iris
    import ncdata.iris_xarray
    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)

`Zarr`_ is a Python package that provides an implementation of chunked, compressed,
N-dimensional arrays.
Zarr has the ability to store arrays in a range of ways, including in memory,
in files, and in cloud-based object storage such as `Amazon S3`_ and
`Google Cloud Storage`_.
Xarray's Zarr backend allows xarray to leverage these capabilities, including
the ability to store and analyze datasets far too large fit onto disk
(particularly :ref:`in combination with dask <dask>`).

Xarray can't open just any zarr dataset, because xarray requires special
metadata (attributes) describing the dataset dimensions and coordinates.
At this time, xarray can only open zarr datasets with these special attributes,
such as zarr datasets written by xarray,
`netCDF <https://docs.unidata.ucar.edu/nug/current/nczarr_head.html>`_,
or `GDAL <https://gdal.org/drivers/raster/zarr.html>`_.
For implementation details, see :ref:`zarr_encoding`.

To write a dataset with zarr, we use the :py:meth:`Dataset.to_zarr` method.

To write to a local directory, we pass a path to a directory:

.. jupyter-execute::
    :hide-code:

    ! rm -rf /tmp/path/directory.zarr

.. jupyter-execute::
    :stderr:

    ds = xr.Dataset(
        {"foo": (("x", "y"), np.random.rand(4, 5))},
        coords={
            "x": [10, 20, 30, 40],
            "y": pd.date_range("2000-01-01", periods=5),
            "z": ("x", list("abcd")),
        },
    )
    ds.to_zarr("/tmp/path/directory.zarr", zarr_format=2, consolidated=False)

(The suffix ``.zarr`` is optional--just a reminder that a zarr store lives
there.) If the directory does not exist, it will be created. If a zarr
store is already present at that path, an error will be raised, preventing it
from being overwritten. To override this behavior and overwrite an existing
store, add ``mode='w'`` when invoking :py:meth:`~Dataset.to_zarr`.

DataArrays can also be saved to disk using the :py:meth:`DataArray.to_zarr` method,
and loaded from disk using the :py:func:`open_dataarray` function with ``engine='zarr'``.
Similar to :py:meth:`DataArray.to_netcdf`, :py:meth:`DataArray.to_zarr` will
convert the ``DataArray`` to a ``Dataset`` before saving, and then convert back
when loading, ensuring that the ``DataArray`` that is loaded is always exactly
the same as the one that was saved.

.. note::

    xarray does not write `NCZarr <https://docs.unidata.ucar.edu/nug/current/nczarr_head.html>`_ attributes.
    Therefore, NCZarr data must be opened in read-only mode.

To store variable length strings, convert them to object arrays first with
``dtype=object``.

To read back a zarr dataset that has been created this way, we use the
:py:func:`open_zarr` method:

.. jupyter-execute::

    ds_zarr = xr.open_zarr("/tmp/path/directory.zarr", consolidated=False)
    ds_zarr

Cloud Storage Buckets
~~~~~~~~~~~~~~~~~~~~~~

It is possible to read and write xarray datasets directly from / to cloud
storage buckets using zarr. This example uses the `gcsfs`_ package to provide
an interface to `Google Cloud Storage`_.

General `fsspec`_ URLs, those that begin with ``s3://`` or ``gcs://`` for example,
are parsed and the store set up for you automatically when reading.
You should include any arguments to the storage backend as the
key ```storage_options``, part of ``backend_kwargs``.

.. code:: python

    ds_gcs = xr.open_dataset(
        "gcs://<bucket-name>/path.zarr",
        backend_kwargs={
            "storage_options": {"project": "<project-name>", "token": None}
        },
        engine="zarr",
    )


This also works with ``open_mfdataset``, allowing you to pass a list of paths or
a URL to be interpreted as a glob string.

For writing, you may either specify a bucket URL or explicitly set up a
``zarr.abc.store.Store`` instance, as follows:

.. tab:: URL

    .. code:: python

        # write to the bucket via GCS URL
        ds.to_zarr("gs://<bucket/path/to/data.zarr>")
        # read it back
        ds_gcs = xr.open_zarr("gs://<bucket/path/to/data.zarr>")

.. tab:: fsspec

    .. code:: python

        import gcsfs
        import zarr

        # manually manage the cloud filesystem connection -- useful, for example,
        # when you need to manage permissions to cloud resources
        fs = gcsfs.GCSFileSystem(project="<project-name>", token=None)
        zstore = zarr.storage.FsspecStore(fs, path="<bucket/path/to/data.zarr>")

        # write to the bucket
        ds.to_zarr(store=zstore)
        # read it back
        ds_gcs = xr.open_zarr(zstore)

.. tab:: obstore

    .. code:: python

        import obstore
        import zarr

        # alternatively, obstore offers a modern, performant interface for
        # cloud buckets
        gcsstore = obstore.store.GCSStore(
            "<bucket>", prefix="<path/to/data.zarr>", skip_signature=True
        )
        zstore = zarr.store.ObjectStore(gcsstore)

        # write to the bucket
        ds.to_zarr(store=zstore)
        # read it back
        ds_gcs = xr.open_zarr(zstore)


.. _fsspec: https://filesystem-spec.readthedocs.io/en/latest/
.. _obstore: https://developmentseed.org/obstore/latest/
.. _Zarr: https://zarr.readthedocs.io/
.. _Amazon S3: https://aws.amazon.com/s3/
.. _Google Cloud Storage: https://cloud.google.com/storage/
.. _gcsfs: https://github.com/fsspec/gcsfs

.. _io.zarr.distributed_writes:

Distributed writes
~~~~~~~~~~~~~~~~~~

Xarray will natively use dask to write in parallel to a zarr store, which should
satisfy most moderately sized datasets. For more flexible parallelization, we
can use ``region`` to write to limited regions of arrays in an existing Zarr
store.

To scale this up to writing large datasets, first create an initial Zarr store
without writing all of its array data. This can be done by first creating a
``Dataset`` with dummy values stored in :ref:`dask <dask>`, and then calling
``to_zarr`` with ``compute=False`` to write only metadata (including ``attrs``)
to Zarr:

.. jupyter-execute::
    :hide-code:

    ! rm -rf /tmp/directory.zarr /tmp/foo.zarr

.. jupyter-execute::

    import dask.array

    # The values of this dask array are entirely irrelevant; only the dtype,
    # shape and chunks are used
    dummies = dask.array.zeros(30, chunks=10)
    ds = xr.Dataset({"foo": ("x", dummies)}, coords={"x": np.arange(30)})
    path = "/tmp/directory.zarr"
    # Now we write the metadata without computing any array values
    ds.to_zarr(path, compute=False, consolidated=False)

Now, a Zarr store with the correct variable shapes and attributes exists that
can be filled out by subsequent calls to ``to_zarr``.
Setting ``region="auto"`` will open the existing store and determine the
correct alignment of the new data with the existing dimensions, or as an
explicit mapping from dimension names to Python ``slice`` objects indicating
where the data should be written (in index space, not label space), e.g.,

.. jupyter-execute::

    # For convenience, we'll slice a single dataset, but in the real use-case
    # we would create them separately possibly even from separate processes.
    ds = xr.Dataset({"foo": ("x", np.arange(30))}, coords={"x": np.arange(30)})
    # Any of the following region specifications are valid
    ds.isel(x=slice(0, 10)).to_zarr(path, region="auto", consolidated=False)
    ds.isel(x=slice(10, 20)).to_zarr(path, region={"x": "auto"}, consolidated=False)
    ds.isel(x=slice(20, 30)).to_zarr(path, region={"x": slice(20, 30)}, consolidated=False)

Concurrent writes with ``region`` are safe as long as they modify distinct
chunks in the underlying Zarr arrays (or use an appropriate ``lock``).

As a safety check to make it harder to inadvertently override existing values,
if you set ``region`` then *all* variables included in a Dataset must have
dimensions included in ``region``. Other variables (typically coordinates)
need to be explicitly dropped and/or written in a separate calls to ``to_zarr``
with ``mode='a'``.

Zarr Compressors and Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are many different `options for compression and filtering possible with
zarr <https://zarr.readthedocs.io/en/stable/user-guide/arrays.html#compressors>`_.

These options can be passed to the ``to_zarr`` method as variable encoding.
For example:

.. jupyter-execute::

    import zarr
    from zarr.codecs import BloscCodec

    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
    ds.to_zarr("/tmp/foo.zarr", consolidated=False, encoding={"foo": {"compressors": [compressor]}})

.. note::

    Not all native zarr compression and filtering options have been tested with
    xarray.

.. _io.zarr.appending:

Modifying existing Zarr stores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Xarray supports several ways of incrementally writing variables to a Zarr
store. These options are useful for scenarios when it is infeasible or
undesirable to write your entire dataset at once.

1. Use ``mode='a'`` to add or overwrite entire variables,
2. Use ``append_dim`` to resize and append to existing variables, and
3. Use ``region`` to write to limited regions of existing arrays.

.. tip::

    For ``Dataset`` objects containing dask arrays, a
    single call to ``to_zarr()`` will write all of your data in parallel.

.. warning::

    Alignment of coordinates is currently not checked when modifying an
    existing Zarr store. It is up to the user to ensure that coordinates are
    consistent.

To add or overwrite entire variables, simply call :py:meth:`~Dataset.to_zarr`
with ``mode='a'`` on a Dataset containing the new variables, passing in an
existing Zarr store or path to a Zarr store.

To resize and then append values along an existing dimension in a store, set
``append_dim``. This is a good option if data always arrives in a particular
order, e.g., for time-stepping a simulation:

.. jupyter-execute::
    :hide-code:

    ! rm -rf /tmp/path/directory.zarr

.. jupyter-execute::

    ds1 = xr.Dataset(
        {"foo": (("x", "y", "t"), np.random.rand(4, 5, 2))},
        coords={
            "x": [10, 20, 30, 40],
            "y": [1, 2, 3, 4, 5],
            "t": pd.date_range("2001-01-01", periods=2),
        },
    )
    ds1.to_zarr("/tmp/path/directory.zarr", consolidated=False)

.. jupyter-execute::

    ds2 = xr.Dataset(
        {"foo": (("x", "y", "t"), np.random.rand(4, 5, 2))},
        coords={
            "x": [10, 20, 30, 40],
            "y": [1, 2, 3, 4, 5],
            "t": pd.date_range("2001-01-03", periods=2),
        },
    )
    ds2.to_zarr("/tmp/path/directory.zarr", append_dim="t", consolidated=False)

.. _io.zarr.writing_chunks:

Specifying chunks in a zarr store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chunk sizes may be specified in one of three ways when writing to a zarr store:

1. Manual chunk sizing through the use of the ``encoding`` argument in :py:meth:`Dataset.to_zarr`:
2. Automatic chunking based on chunks in dask arrays
3. Default chunk behavior determined by the zarr library

The resulting chunks will be determined based on the order of the above list; dask
chunks will be overridden by manually-specified chunks in the encoding argument,
and the presence of either dask chunks or chunks in the ``encoding`` attribute will
supersede the default chunking heuristics in zarr.

Importantly, this logic applies to every array in the zarr store individually,
including coordinate arrays. Therefore, if a dataset contains one or more dask
arrays, it may still be desirable to specify a chunk size for the coordinate arrays
(for example, with a chunk size of ``-1`` to include the full coordinate).

To specify chunks manually using the ``encoding`` argument, provide a nested
dictionary with the structure ``{'variable_or_coord_name': {'chunks': chunks_tuple}}``.

.. note::

    The positional ordering of the chunks in the encoding argument must match the
    positional ordering of the dimensions in each array. Watch out for arrays with
    differently-ordered dimensions within a single Dataset.

For example, let's say we're working with a dataset with dimensions
``('time', 'x', 'y')``, a variable ``Tair`` which is chunked in ``x`` and ``y``,
and two multi-dimensional coordinates ``xc`` and ``yc``:

.. jupyter-execute::

    ds = xr.tutorial.open_dataset("rasm")

    ds["Tair"] = ds["Tair"].chunk({"x": 100, "y": 100})

    ds

These multi-dimensional coordinates are only two-dimensional and take up very little
space on disk or in memory, yet when writing to disk the default zarr behavior is to
split them into chunks:

.. jupyter-execute::

    ds.to_zarr("/tmp/path/directory.zarr", consolidated=False, mode="w")
    !tree -I zarr.json /tmp/path/directory.zarr


This may cause unwanted overhead on some systems, such as when reading from a cloud
storage provider. To disable this chunking, we can specify a chunk size equal to the
shape of each coordinate array in the ``encoding`` argument:

.. jupyter-execute::

    ds.to_zarr(
        "/tmp/path/directory.zarr",
        encoding={"xc": {"chunks": ds.xc.shape}, "yc": {"chunks": ds.yc.shape}},
        consolidated=False,
        mode="w",
    )
    !tree -I zarr.json /tmp/path/directory.zarr


The number of chunks on Tair matches our dask chunks, while there is now only a single
chunk in the directory stores of each coordinate.

Groups
~~~~~~

Nested groups in zarr stores can be represented by loading the store as a
:py:class:`xarray.DataTree` object, similarly to netCDF. To open a whole zarr store as
a tree of groups use the :py:func:`open_datatree` function. To save a
``DataTree`` object as a zarr store containing many groups, use the
:py:meth:`xarray.DataTree.to_zarr()` method.

.. note::
    Note that perfect round-tripping should always be possible with a zarr
    store (:ref:`unlike for netCDF files <netcdf.group.warning>`), as zarr does
    not support "unused" dimensions.

    For the root group the same restrictions (:ref:`as for netCDF files <netcdf.root_group.note>`) apply.
    Due to file format specifications the on-disk root group name is always ``"/"``
    overriding any given ``DataTree`` root node name.


.. _io.zarr.consolidated_metadata:

Consolidated Metadata
~~~~~~~~~~~~~~~~~~~~~


Xarray needs to read all of the zarr metadata when it opens a dataset.
In some storage mediums, such as with cloud object storage (e.g. `Amazon S3`_),
this can introduce significant overhead, because two separate HTTP calls to the
object store must be made for each variable in the dataset.
By default Xarray uses a feature called
*consolidated metadata*, storing all metadata for the entire dataset with a
single key (by default called ``.zmetadata``). This typically drastically speeds
up opening the store. (For more information on this feature, consult the
`zarr docs on consolidating metadata <https://zarr.readthedocs.io/en/latest/user-guide/consolidated_metadata.html>`_.)

By default, xarray writes consolidated metadata and attempts to read stores
with consolidated metadata, falling back to use non-consolidated metadata for
reads. Because this fall-back option is so much slower, xarray issues a
``RuntimeWarning`` with guidance when reading with consolidated metadata fails:

    Failed to open Zarr store with consolidated metadata, falling back to try
    reading non-consolidated metadata. This is typically much slower for
    opening a dataset. To silence this warning, consider:

    1. Consolidating metadata in this existing store with
       :py:func:`zarr.consolidate_metadata`.
    2. Explicitly setting ``consolidated=False``, to avoid trying to read
       consolidate metadata.
    3. Explicitly setting ``consolidated=True``, to raise an error in this case
       instead of falling back to try reading non-consolidated metadata.


Fill Values
~~~~~~~~~~~

Zarr arrays have a ``fill_value`` that is used for chunks that were never written to disk.
For the Zarr version 2 format, Xarray will set ``fill_value`` to be equal to the CF/NetCDF ``"_FillValue"``.
This is ``np.nan`` by default for floats, and unset otherwise. Note that the Zarr library will set a
default ``fill_value`` if not specified (usually ``0``).

For the Zarr version 3 format, ``_FillValue`` and ```fill_value`` are decoupled.
So you can set ``fill_value`` in ``encoding`` as usual.

Note that at read-time, you can control whether ``_FillValue`` is masked using the
``mask_and_scale`` kwarg; and whether Zarr's ``fill_value`` is treated as synonymous
with ``_FillValue`` using the ``use_zarr_fill_value_as_mask`` kwarg to :py:func:`xarray.open_zarr`.
