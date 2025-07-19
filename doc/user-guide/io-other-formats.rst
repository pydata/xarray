.. currentmodule:: xarray
.. _io.other-formats:

Other File Formats
===================

This page covers additional file formats and data sources supported by xarray beyond netCDF/HDF5 and Zarr.

.. jupyter-execute::
    :hide-code:

    import os

    import iris
    import ncdata.iris_xarray
    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)

.. _io.kerchunk:

Kerchunk
--------

`Kerchunk <https://fsspec.github.io/kerchunk>`_ is a Python library
that allows you to access chunked and compressed data formats (such as NetCDF3, NetCDF4, HDF5, GRIB2, TIFF & FITS),
many of which are primary data formats for many data archives, by viewing the
whole archive as an ephemeral `Zarr`_ dataset which allows for parallel, chunk-specific access.

Instead of creating a new copy of the dataset in the Zarr spec/format or
downloading the files locally, Kerchunk reads through the data archive and extracts the
byte range and compression information of each chunk and saves as a ``reference``.
These references are then saved as ``json`` files or ``parquet`` (more efficient)
for later use. You can view some of these stored in the ``references``
directory `here <https://github.com/pydata/xarray-data>`_.


.. note::
    These references follow this `specification <https://fsspec.github.io/kerchunk/spec.html>`_.
    Packages like `kerchunk`_ and `virtualizarr <https://github.com/zarr-developers/VirtualiZarr>`_
    help in creating and reading these references.


Reading these data archives becomes really easy with ``kerchunk`` in combination
with ``xarray``, especially when these archives are large in size. A single combined
reference can refer to thousands of the original data files present in these archives.
You can view the whole dataset with from this combined reference using the above packages.

The following example shows opening a single ``json`` reference to the ``saved_on_disk.h5`` file created above.
If the file were instead stored remotely (e.g. ``s3://saved_on_disk.h5``) you can use ``storage_options``
that are used to `configure fsspec <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.reference.ReferenceFileSystem.__init__>`_:

.. jupyter-execute::

    ds_kerchunked = xr.open_dataset(
        "./combined.json",
        engine="kerchunk",
        storage_options={},
    )

    ds_kerchunked

.. note::

    You can refer to the `project pythia kerchunk cookbook <https://projectpythia.org/kerchunk-cookbook/README.html>`_
    and the `pangeo guide on kerchunk <https://guide.cloudnativegeo.org/kerchunk/intro.html>`_ for more information.


.. _io.iris:

Iris
----

The Iris_ tool allows easy reading of common meteorological and climate model formats
(including GRIB and UK MetOffice PP files) into ``Cube`` objects which are in many ways very
similar to ``DataArray`` objects, while enforcing a CF-compliant data model.

DataArray ``to_iris`` and ``from_iris``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If iris is installed, xarray can convert a ``DataArray`` into a ``Cube`` using
:py:meth:`DataArray.to_iris`:

.. jupyter-execute::

    da = xr.DataArray(
        np.random.rand(4, 5),
        dims=["x", "y"],
        coords=dict(x=[10, 20, 30, 40], y=pd.date_range("2000-01-01", periods=5)),
    )

    cube = da.to_iris()
    print(cube)

Conversely, we can create a new ``DataArray`` object from a ``Cube`` using
:py:meth:`DataArray.from_iris`:

.. jupyter-execute::

    da_cube = xr.DataArray.from_iris(cube)
    da_cube

Ncdata
~~~~~~
Ncdata_ provides more sophisticated means of transferring data, including entire
datasets.  It uses the file saving and loading functions in both projects to provide a
more "correct" translation between them, but still with very low overhead and not
using actual disk files.

Here we load an xarray dataset and convert it to Iris cubes:

.. jupyter-execute::
    :stderr:

    ds = xr.tutorial.open_dataset("air_temperature_gradient")
    cubes = ncdata.iris_xarray.cubes_from_xarray(ds)
    print(cubes)

.. jupyter-execute::

    print(cubes[1])

And we can convert the cubes back to an xarray dataset:

.. jupyter-execute::

    # ensure dataset-level and variable-level attributes loaded correctly
    iris.FUTURE.save_split_attrs = True

    ds = ncdata.iris_xarray.cubes_to_xarray(cubes)
    ds

Ncdata can also adjust file data within load and save operations, to fix data loading
problems or provide exact save formatting without needing to modify files on disk.
See for example : `ncdata usage examples`_

.. _Iris: https://scitools.org.uk/iris
.. _Ncdata: https://ncdata.readthedocs.io/en/latest/index.html
.. _ncdata usage examples: https://github.com/pp-mo/ncdata/tree/v0.1.2?tab=readme-ov-file#correct-a-miscoded-attribute-in-iris-input

OPeNDAP
-------

Xarray includes support for `OPeNDAP`__ (via the netCDF4 library or Pydap), which
lets us access large datasets over HTTP.

__ https://www.opendap.org/

For example, we can open a connection to GBs of weather data produced by the
`PRISM`__ project, and hosted by `IRI`__ at Columbia:

__ https://www.prism.oregonstate.edu/
__ https://iri.columbia.edu/


.. jupyter-input::

    remote_data = xr.open_dataset(
        "http://iridl.ldeo.columbia.edu/SOURCES/.OSU/.PRISM/.monthly/dods",
        decode_times=False,
        )
    remote_data

.. jupyter-output::

    <xarray.Dataset>
    Dimensions:  (T: 1422, X: 1405, Y: 621)
    Coordinates:
      * X        (X) float32 -125.0 -124.958 -124.917 -124.875 -124.833 -124.792 -124.75 ...
      * T        (T) float32 -779.5 -778.5 -777.5 -776.5 -775.5 -774.5 -773.5 -772.5 -771.5 ...
      * Y        (Y) float32 49.9167 49.875 49.8333 49.7917 49.75 49.7083 49.6667 49.625 ...
    Data variables:
        ppt      (T, Y, X) float64 ...
        tdmean   (T, Y, X) float64 ...
        tmax     (T, Y, X) float64 ...
        tmin     (T, Y, X) float64 ...
    Attributes:
        Conventions: IRIDL
        expires: 1375315200

.. TODO: update this example to show off decode_cf?

.. note::

    Like many real-world datasets, this dataset does not entirely follow
    `CF conventions`_. Unexpected formats will usually cause xarray's automatic
    decoding to fail. The way to work around this is to either set
    ``decode_cf=False`` in ``open_dataset`` to turn off all use of CF
    conventions, or by only disabling the troublesome parser.
    In this case, we set ``decode_times=False`` because the time axis here
    provides the calendar attribute in a format that xarray does not expect
    (the integer ``360`` instead of a string like ``'360_day'``).

We can select and slice this data any number of times, and nothing is loaded
over the network until we look at particular values:

.. jupyter-input::

    tmax = remote_data["tmax"][:500, ::3, ::3]
    tmax

.. jupyter-output::

    <xarray.DataArray 'tmax' (T: 500, Y: 207, X: 469)>
    [48541500 values with dtype=float64]
    Coordinates:
      * Y        (Y) float32 49.9167 49.7917 49.6667 49.5417 49.4167 49.2917 ...
      * X        (X) float32 -125.0 -124.875 -124.75 -124.625 -124.5 -124.375 ...
      * T        (T) float32 -779.5 -778.5 -777.5 -776.5 -775.5 -774.5 -773.5 ...
    Attributes:
        pointwidth: 120
        standard_name: air_temperature
        units: Celsius_scale
        expires: 1443657600

.. jupyter-input::

    # the data is downloaded automatically when we make the plot
    tmax[0].plot()

.. image:: ../_static/opendap-prism-tmax.png

Some servers require authentication before we can access the data. Pydap uses
a `Requests`__ session object (which the user can pre-define), and this
session object can recover `authentication`__` credentials from a locally stored
``.netrc`` file. For example, to connect to a server that requires NASA's
URS authentication, with the username/password credentials stored on a locally
accessible ``.netrc``, access to OPeNDAP data should be as simple as this::

    import xarray as xr
    import requests

    my_session = requests.Session()

    ds_url = 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax/example.nc'

    ds = xr.open_dataset(ds_url, session=my_session, engine="pydap")

Moreover, a bearer token header can be included in a `Requests`__ session
object, allowing for token-based authentication which  OPeNDAP servers can use
to avoid some redirects.


Lastly, OPeNDAP servers may provide endpoint URLs for different OPeNDAP protocols,
DAP2 and DAP4. To specify which protocol between the two options to use, you can
replace the scheme of the url with the name of the protocol. For example::

    # dap2 url
    ds_url = 'dap2://gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax/example.nc'

    # dap4 url
    ds_url = 'dap4://gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax/example.nc'

While most OPeNDAP servers implement DAP2, not all servers implement DAP4. It
is recommended to check if the URL you are using `supports DAP4`__ by checking the
URL on a browser.

__ https://docs.python-requests.org
__ https://pydap.github.io/pydap/en/notebooks/Authentication.html
__ https://pydap.github.io/pydap/en/faqs/dap2_or_dap4_url.html

.. _io.pickle:

Pickle
------

The simplest way to serialize an xarray object is to use Python's built-in pickle
module:

.. jupyter-execute::

    import pickle

    # use the highest protocol (-1) because it is way faster than the default
    # text based pickle format
    pkl = pickle.dumps(ds, protocol=-1)

    pickle.loads(pkl)

Pickling is important because it doesn't require any external libraries
and lets you use xarray objects with Python modules like
:py:mod:`multiprocessing` or :ref:`Dask <dask>`. However, pickling is
**not recommended for long-term storage**.

Restoring a pickle requires that the internal structure of the types for the
pickled data remain unchanged. Because the internal design of xarray is still
being refined, we make no guarantees (at this point) that objects pickled with
this version of xarray will work in future versions.

.. note::

  When pickling an object opened from a NetCDF file, the pickle file will
  contain a reference to the file on disk. If you want to store the actual
  array values, load it into memory first with :py:meth:`Dataset.load`
  or :py:meth:`Dataset.compute`.

.. _dictionary io:

Dictionary
----------

We can convert a ``Dataset`` (or a ``DataArray``) to a dict using
:py:meth:`Dataset.to_dict`:

.. jupyter-execute::

    ds = xr.Dataset({"foo": ("x", np.arange(30))})
    d = ds.to_dict()
    d

We can create a new xarray object from a dict using
:py:meth:`Dataset.from_dict`:

.. jupyter-execute::

    ds_dict = xr.Dataset.from_dict(d)
    ds_dict

Dictionary support allows for flexible use of xarray objects. It doesn't
require external libraries and dicts can easily be pickled, or converted to
json, or geojson. All the values are converted to lists, so dicts might
be quite large.

To export just the dataset schema without the data itself, use the
``data=False`` option:

.. jupyter-execute::

    ds.to_dict(data=False)

.. jupyter-execute::
    :hide-code:

    # We're now done with the dataset named `ds`.  Although the `with` statement closed
    # the dataset, displaying the unpickled pickle of `ds` re-opened "saved_on_disk.nc".
    # However, `ds` (rather than the unpickled dataset) refers to the open file.  Delete
    # `ds` to close the file.
    del ds

    for f in ["saved_on_disk.nc", "saved_on_disk.h5"]:
        if os.path.exists(f):
            os.remove(f)

This can be useful for generating indices of dataset contents to expose to
search indices or other automated data discovery tools.

.. _io.rasterio:

Rasterio
--------

GDAL readable raster data using `rasterio`_  such as GeoTIFFs can be opened using the `rioxarray`_ extension.
`rioxarray`_ can also handle geospatial related tasks such as re-projecting and clipping.

.. jupyter-input::

    import rioxarray

    rds = rioxarray.open_rasterio("RGB.byte.tif")
    rds

.. jupyter-output::

    <xarray.DataArray (band: 3, y: 718, x: 791)>
    [1703814 values with dtype=uint8]
    Coordinates:
      * band         (band) int64 1 2 3
      * y            (y) float64 2.827e+06 2.826e+06 ... 2.612e+06 2.612e+06
      * x            (x) float64 1.021e+05 1.024e+05 ... 3.389e+05 3.392e+05
        spatial_ref  int64 0
    Attributes:
        STATISTICS_MAXIMUM:  255
        STATISTICS_MEAN:     29.947726688477
        STATISTICS_MINIMUM:  0
        STATISTICS_STDDEV:   52.340921626611
        transform:           (300.0379266750948, 0.0, 101985.0, 0.0, -300.0417827...
        _FillValue:          0.0
        scale_factor:        1.0
        add_offset:          0.0
        grid_mapping:        spatial_ref

.. jupyter-input::

    rds.rio.crs
    # CRS.from_epsg(32618)

    rds4326 = rds.rio.reproject("epsg:4326")

    rds4326.rio.crs
    # CRS.from_epsg(4326)

    rds4326.rio.to_raster("RGB.byte.4326.tif")


.. _rasterio: https://rasterio.readthedocs.io/en/latest/
.. _rioxarray: https://corteva.github.io/rioxarray/stable/
.. _test files: https://github.com/rasterio/rasterio/blob/master/tests/data/RGB.byte.tif
.. _pyproj: https://github.com/pyproj4/pyproj

.. _io.cfgrib:

GRIB format via cfgrib
----------------------

Xarray supports reading GRIB files via ECMWF cfgrib_ python driver,
if it is installed. To open a GRIB file supply ``engine='cfgrib'``
to :py:func:`open_dataset` after installing cfgrib_:

.. jupyter-input::

    ds_grib = xr.open_dataset("example.grib", engine="cfgrib")

We recommend installing cfgrib via conda::

    conda install -c conda-forge cfgrib

.. _cfgrib: https://github.com/ecmwf/cfgrib


CSV and other formats supported by pandas
-----------------------------------------

For more options (tabular formats and CSV files in particular), consider
exporting your objects to pandas and using its broad range of `IO tools`_.
For CSV files, one might also consider `xarray_extras`_.

.. _xarray_extras: https://xarray-extras.readthedocs.io/en/latest/api/csv.html

.. _IO tools: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html


Third party libraries
---------------------

More formats are supported by extension libraries:

- `xarray-mongodb <https://xarray-mongodb.readthedocs.io/en/latest/>`_: Store xarray objects on MongoDB

.. _Zarr: https://zarr.readthedocs.io/
.. _CF conventions: https://cfconventions.org/
.. _kerchunk: https://fsspec.github.io/kerchunk
