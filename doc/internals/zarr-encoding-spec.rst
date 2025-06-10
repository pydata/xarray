.. currentmodule:: xarray

.. _zarr_encoding:

Zarr Encoding Specification
============================

In implementing support for the `Zarr <https://zarr.dev>`_ storage
format, Xarray developers made some *ad hoc* choices about how to store
NetCDF data in Zarr.
Future versions of the Zarr spec will likely include a more formal convention
for the storage of the NetCDF data model in Zarr; see
`Zarr spec repo <https://github.com/zarr-developers/zarr-specs>`_ for ongoing
discussion.

First, Xarray can only read and write Zarr groups. There is currently no support
for reading / writing individual Zarr arrays. Zarr groups are mapped to
Xarray ``Dataset`` objects.

Second, from Xarray's point of view, the key difference between
NetCDF and Zarr is that all NetCDF arrays have *dimension names* while Zarr
arrays do not. Therefore, in order to store NetCDF data in Zarr, Xarray must
somehow encode and decode the name of each array's dimensions.

To accomplish this, Xarray developers decided to define a special Zarr array
attribute: ``_ARRAY_DIMENSIONS``. The value of this attribute is a list of
dimension names (strings), for example ``["time", "lon", "lat"]``. When writing
data to Zarr, Xarray sets this attribute on all variables based on the variable
dimensions. When reading a Zarr group, Xarray looks for this attribute on all
arrays, raising an error if it can't be found. The attribute is used to define
the variable dimension names and then removed from the attributes dictionary
returned to the user.

Because of these choices, Xarray cannot read arbitrary array data, but only
Zarr data with valid ``_ARRAY_DIMENSIONS`` or
`NCZarr <https://docs.unidata.ucar.edu/nug/current/nczarr_head.html>`_ attributes
on each array (NCZarr dimension names are defined in the ``.zarray`` file).

After decoding the ``_ARRAY_DIMENSIONS`` or NCZarr attribute and assigning the variable
dimensions, Xarray proceeds to [optionally] decode each variable using its
standard CF decoding machinery used for NetCDF data (see :py:func:`decode_cf`).

Finally, it's worth noting that Xarray writes (and attempts to read)
"consolidated metadata" by default (the ``.zmetadata`` file), which is another
non-standard Zarr extension, albeit one implemented upstream in Zarr-Python.
You do not need to write consolidated metadata to make Zarr stores readable in
Xarray, but because Xarray can open these stores much faster, users will see a
warning about poor performance when reading non-consolidated stores unless they
explicitly set ``consolidated=False``. See :ref:`io.zarr.consolidated_metadata`
for more details.

As a concrete example, here we write a tutorial dataset to Zarr and then
re-open it directly with Zarr:

.. jupyter-execute::

    import os
    import xarray as xr
    import zarr

    ds = xr.tutorial.load_dataset("rasm")
    ds.to_zarr("rasm.zarr", mode="w", consolidated=False)
    os.listdir("rasm.zarr")

.. jupyter-execute::

    zgroup = zarr.open("rasm.zarr")
    zgroup.tree()

.. jupyter-execute::

    dict(zgroup["Tair"].attrs)

.. jupyter-execute::
    :hide-code:

    import shutil

    shutil.rmtree("rasm.zarr")

Chunk Key Encoding
------------------

When writing data to Zarr stores, Xarray supports customizing how chunk keys are encoded
through the ``chunk_key_encoding`` parameter in the variable's encoding dictionary. This
is particularly useful when working with Zarr V2 arrays and you need to control the
dimension separator in chunk keys.

For example, to specify a custom separator for chunk keys:

.. jupyter-execute::

    import xarray as xr
    import numpy as np
    from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding

    # Create a custom chunk key encoding with "/" as separator
    enc = V2ChunkKeyEncoding(separator="/").to_dict()

    # Create and write a dataset with custom chunk key encoding
    arr = np.ones((42, 100))
    ds = xr.DataArray(arr, name="var1").to_dataset()
    ds.to_zarr(
        "example.zarr",
        zarr_format=2,
        mode="w",
        encoding={"var1": {"chunks": (42, 50), "chunk_key_encoding": enc}},
    )

The ``chunk_key_encoding`` option accepts a dictionary that specifies the encoding
configuration. For Zarr V2 arrays, you can use the ``V2ChunkKeyEncoding`` class from
``zarr.core.chunk_key_encodings`` to generate this configuration. This is particularly
useful when you need to ensure compatibility with specific Zarr V2 storage layouts or
when working with tools that expect a particular chunk key format.

.. note::
    The ``chunk_key_encoding`` option is only relevant when writing to Zarr stores.
    When reading Zarr arrays, Xarray automatically detects and uses the appropriate
    chunk key encoding based on the store's format and configuration.

.. jupyter-execute::
    :hide-code:

    import shutil

    shutil.rmtree("example.zarr")
