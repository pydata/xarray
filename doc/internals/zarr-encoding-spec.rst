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
arrays do not. In Zarr v2, Xarray uses an ad-hoc convention to encode and decode
the name of each array's dimensions. However, starting with Zarr v3, the
``dimension_names`` attribute provides a formal convention for storing the
NetCDF data model in Zarr.

Dimension Encoding in Zarr Formats
-----------------------------------

Xarray encodes array dimensions differently depending on the Zarr format version:

**Zarr V2 Format:**
Xarray uses a special Zarr array attribute: ``_ARRAY_DIMENSIONS``. The value of this
attribute is a list of dimension names (strings), for example ``["time", "lon", "lat"]``.
When writing data to Zarr V2, Xarray sets this attribute on all variables based on the
variable dimensions. This attribute is visible when accessing arrays directly with
zarr-python.

**Zarr V3 Format:**
Xarray uses the native ``dimension_names`` field in the array metadata. This is part
of the official Zarr V3 specification and is not stored as a regular attribute.
When accessing arrays with zarr-python, this information is available in the array's
metadata but not in the attributes dictionary.

When reading a Zarr group, Xarray looks for dimension information in the appropriate
location based on the format version, raising an error if it can't be found. The
dimension information is used to define the variable dimension names and then
(for Zarr V2) removed from the attributes dictionary returned to the user.

CF Conventions
--------------

Xarray uses its standard CF encoding/decoding functionality for handling metadata
(see :py:func:`decode_cf`). This includes encoding concepts such as dimensions and
coordinates. The ``coordinates`` attribute, which lists coordinate variables
(e.g., ``"yc xc"`` for spatial coordinates), is one part of the broader CF conventions
used to describe metadata in NetCDF and Zarr.

Compatibility and Reading
-------------------------

Because of these encoding choices, Xarray cannot read arbitrary Zarr arrays, but only
Zarr data with valid dimension metadata. Xarray supports:

- Zarr V2 arrays with ``_ARRAY_DIMENSIONS`` attributes
- Zarr V3 arrays with ``dimension_names`` metadata
- `NCZarr <https://docs.unidata.ucar.edu/nug/current/nczarr_head.html>`_ format
  (dimension names are defined in the ``.zarray`` file)

After decoding the dimension information and assigning the variable dimensions,
Xarray proceeds to [optionally] decode each variable using its standard CF decoding
machinery used for NetCDF data.

Finally, it's worth noting that Xarray writes (and attempts to read)
"consolidated metadata" by default (the ``.zmetadata`` file), which is another
non-standard Zarr extension, albeit one implemented upstream in Zarr-Python.
You do not need to write consolidated metadata to make Zarr stores readable in
Xarray, but because Xarray can open these stores much faster, users will see a
warning about poor performance when reading non-consolidated stores unless they
explicitly set ``consolidated=False``. See :ref:`io.zarr.consolidated_metadata`
for more details.

Examples: Zarr Format Differences
----------------------------------

The following examples demonstrate how dimension and coordinate encoding differs
between Zarr format versions. We'll use the same tutorial dataset but write it
in different formats to show what users will see when accessing the files directly
with zarr-python.

**Example 1: Zarr V2 Format**

.. jupyter-execute::

    import os
    import xarray as xr
    import zarr

    # Load tutorial dataset and write as Zarr V2
    ds = xr.tutorial.load_dataset("rasm")
    ds.to_zarr("rasm_v2.zarr", mode="w", consolidated=False, zarr_format=2)

    # Open with zarr-python and examine attributes
    zgroup = zarr.open("rasm_v2.zarr")
    print("Zarr V2 - Tair attributes:")
    tair_attrs = dict(zgroup["Tair"].attrs)
    for key, value in tair_attrs.items():
        print(f"  '{key}': {repr(value)}")

.. jupyter-execute::
    :hide-code:

    import shutil
    shutil.rmtree("rasm_v2.zarr")

**Example 2: Zarr V3 Format**

.. jupyter-execute::

    # Write the same dataset as Zarr V3
    ds.to_zarr("rasm_v3.zarr", mode="w", consolidated=False, zarr_format=3)

    # Open with zarr-python and examine attributes
    zgroup = zarr.open("rasm_v3.zarr")
    print("Zarr V3 - Tair attributes:")
    tair_attrs = dict(zgroup["Tair"].attrs)
    for key, value in tair_attrs.items():
        print(f"  '{key}': {repr(value)}")

    # For Zarr V3, dimension information is in metadata
    tair_array = zgroup["Tair"]
    print(f"\nZarr V3 - dimension_names in metadata: {tair_array.metadata.dimension_names}")

.. jupyter-execute::
    :hide-code:

    import shutil
    shutil.rmtree("rasm_v3.zarr")


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
