.. currentmodule:: xarray
.. _io:

Reading and Writing Files
==========================

Xarray supports direct serialization and IO to several file formats, from
simple :ref:`io.pickle` files to the more flexible :ref:`io.netcdf`
format (recommended).

.. jupyter-execute::
    :hide-code:

    import os

    import iris
    import ncdata.iris_xarray
    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)

You can read different types of files in ``xr.open_dataset`` by specifying the engine to be used:

.. code:: python

    xr.open_dataset("example.nc", engine="netcdf4")

The "engine" provides a set of instructions that tells xarray how
to read the data and pack them into a ``Dataset`` (or ``Dataarray``).
These instructions are stored in an underlying "backend".

Xarray comes with several backends that cover many common data formats.
Many more backends are available via external libraries, or you can `write your own <https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html>`_.
This diagram aims to help you determine - based on the format of the file you'd like to read -
which type of backend you're using and how to use it.

Text and boxes are clickable for more information.
Following the diagram is detailed information on many popular backends.
You can learn more about using and developing backends in the
`Xarray tutorial JupyterBook <https://tutorial.xarray.dev/advanced/backends/backends.html>`_.

..
   _comment: mermaid Flowcharg "link" text gets secondary color background, SVG icon fill gets primary color

.. raw:: html

    <style>
      /* Ensure PST link colors don't override mermaid text colors */
      .mermaid a {
        color: white;
      }
      .mermaid a:hover {
        color: magenta;
        text-decoration-color: magenta;
      }
      .mermaid a:visited {
        color: white;
        text-decoration-color: white;
      }
    </style>

.. mermaid::
    :config: {"theme":"base","themeVariables":{"fontSize":"20px","primaryColor":"#fff","primaryTextColor":"#fff","primaryBorderColor":"#59c7d6","lineColor":"#e28126","secondaryColor":"#767985"}}
    :alt: Flowchart illustrating how to choose the right backend engine to read your data

    flowchart LR
        built-in-eng["`**Is your data stored in one of these formats?**
            - netCDF4
            - netCDF3
            - Zarr
            - DODS/OPeNDAP
            - HDF5
            `"]

        built-in("`**You're in luck!** Xarray bundles a backend to automatically read these formats.
            Open data using <code>xr.open_dataset()</code>. We recommend
            explicitly setting engine='xxxx' for faster loading.`")

        installed-eng["""<b>One of these formats?</b>
            - <a href='https://github.com/ecmwf/cfgrib'>GRIB</a>
            - <a href='https://tiledb-inc.github.io/TileDB-CF-Py/documentation'>TileDB</a>
            - <a href='https://corteva.github.io/rioxarray/stable/getting_started/getting_started.html#rioxarray'>GeoTIFF, JPEG-2000, etc. (via GDAL)</a>
            - <a href='https://www.bopen.eu/xarray-sentinel-open-source-library/'>Sentinel-1 SAFE</a>
            """]

        installed("""Install the linked backend library and use it with
            <code>xr.open_dataset(file, engine='xxxx')</code>.""")

        other["`**Options:**
            - Look around to see if someone has created an Xarray backend for your format!
            - <a href='https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html'>Create your own backend</a>
            - Convert your data to a supported format
            `"]

        built-in-eng -->|Yes| built-in
        built-in-eng -->|No| installed-eng

        installed-eng -->|Yes| installed
        installed-eng -->|No| other

        click built-in-eng "https://docs.xarray.dev/en/stable/get-help/faq.html#how-do-i-open-format-x-file-as-an-xarray-dataset"


        classDef quesNodefmt font-size:12pt,fill:#0e4666,stroke:#59c7d6,stroke-width:3
        class built-in-eng,installed-eng quesNodefmt

        classDef ansNodefmt font-size:12pt,fill:#4a4a4a,stroke:#17afb4,stroke-width:3
        class built-in,installed,other ansNodefmt

        linkStyle default font-size:18pt,stroke-width:4

Subsections
-----------

.. toctree::
   :maxdepth: 2

   io-netcdf-hdf
   io-zarr
   io-other-formats
