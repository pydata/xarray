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

.. ipython:: python
    :okwarning:

    import os
    import xarray as xr
    import zarr

    ds = xr.tutorial.load_dataset("rasm")
    ds.to_zarr("rasm.zarr", mode="w")

    zgroup = zarr.open("rasm.zarr")
    print(os.listdir("rasm.zarr"))
    print(zgroup.tree())
    dict(zgroup["Tair"].attrs)

.. ipython:: python
    :suppress:

    import shutil

    shutil.rmtree("rasm.zarr")
