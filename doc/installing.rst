.. _installing:

Installation
============

Required dependencies
---------------------

- Python (3.7 or later)
- setuptools (40.4 or later)
- `numpy <http://www.numpy.org/>`__ (1.15 or later)
- `pandas <http://pandas.pydata.org/>`__ (0.25 or later)

.. _optional-dependencies:

Optional dependencies
---------------------

.. note::

  If you are using pip to install xarray, optional dependencies can be installed by
  specifying *extras*. :ref:`installation-instructions` for both pip and conda
  are given below.

For netCDF and IO
~~~~~~~~~~~~~~~~~

- `netCDF4 <https://github.com/Unidata/netcdf4-python>`__: recommended if you
  want to use xarray for reading or writing netCDF files
- `scipy <http://scipy.org/>`__: used as a fallback for reading/writing netCDF3
- `pydap <http://www.pydap.org/>`__: used as a fallback for accessing OPeNDAP
- `h5netcdf <https://github.com/shoyer/h5netcdf>`__: an alternative library for
  reading and writing netCDF4 files that does not use the netCDF-C libraries
- `PyNIO <https://www.pyngl.ucar.edu/Nio.shtml>`__: for reading GRIB and other
  geoscience specific file formats. Note that PyNIO is not available for Windows and
  that the PyNIO backend may be moved outside of xarray in the future.
- `zarr <http://zarr.readthedocs.io/>`__: for chunked, compressed, N-dimensional arrays.
- `cftime <https://unidata.github.io/cftime>`__: recommended if you
  want to encode/decode datetimes for non-standard calendars or dates before
  year 1678 or after year 2262.
- `PseudoNetCDF <http://github.com/barronh/pseudonetcdf/>`__: recommended
  for accessing CAMx, GEOS-Chem (bpch), NOAA ARL files, ICARTT files
  (ffi1001) and many other.
- `rasterio <https://github.com/mapbox/rasterio>`__: for reading GeoTiffs and
  other gridded raster datasets.
- `iris <https://github.com/scitools/iris>`__: for conversion to and from iris'
  Cube objects
- `cfgrib <https://github.com/ecmwf/cfgrib>`__: for reading GRIB files via the
  *ECMWF ecCodes* library.

For accelerating xarray
~~~~~~~~~~~~~~~~~~~~~~~

- `scipy <http://scipy.org/>`__: necessary to enable the interpolation features for
  xarray objects
- `bottleneck <https://github.com/pydata/bottleneck>`__: speeds up
  NaN-skipping and rolling window aggregations by a large factor
- `numbagg <https://github.com/shoyer/numbagg>`_: for exponential rolling
  window operations

For parallel computing
~~~~~~~~~~~~~~~~~~~~~~

- `dask.array <http://dask.pydata.org>`__: required for :ref:`dask`.

For plotting
~~~~~~~~~~~~

- `matplotlib <http://matplotlib.org/>`__: required for :ref:`plotting`
- `cartopy <http://scitools.org.uk/cartopy/>`__: recommended for :ref:`plot-maps`
- `seaborn <http://seaborn.pydata.org/>`__: for better
  color palettes
- `nc-time-axis <https://github.com/SciTools/nc-time-axis>`__: for plotting
  cftime.datetime objects

Alternative data containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `sparse <https://sparse.pydata.org/>`_: for sparse arrays
- `pint <https://pint.readthedocs.io/>`_: for units of measure

  .. note::

    At the moment of writing, xarray requires a `highly experimental version of pint
    <https://github.com/andrewgsavage/pint/pull/6>`_ (install with
    ``pip install git+https://github.com/andrewgsavage/pint.git@refs/pull/6/head)``.
    Even with it, interaction with non-numpy array libraries, e.g. dask or sparse, is broken.

- Any numpy-like objects that support
  `NEP-18 <https://numpy.org/neps/nep-0018-array-function-protocol.html>`_.
  Note that while such libraries theoretically should work, they are untested.
  Integration tests are in the process of being written for individual libraries.


.. _mindeps_policy:

Minimum dependency versions
---------------------------
xarray adopts a rolling policy regarding the minimum supported version of its
dependencies:

- **Python:** 24 months
  (`NEP-29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_)
- **setuptools:** 42 months (but no older than 40.4)
- **numpy:** 18 months
  (`NEP-29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_)
- **dask and dask.distributed:** 12 months (but no older than 2.9)
- **sparse, pint** and other libraries that rely on
  `NEP-18 <https://numpy.org/neps/nep-0018-array-function-protocol.html>`_
  for integration: very latest available versions only, until the technology will have
  matured. This extends to dask when used in conjunction with any of these libraries.
  numpy >=1.17.
- **all other libraries:** 12 months

This means the latest minor (X.Y) version from N months prior. Patch versions (x.y.Z)
are not pinned, and only the latest available at the moment of publishing the xarray
release is guaranteed to work.

You can see the actual minimum tested versions:

- `For NEP-18 libraries
  <https://github.com/pydata/xarray/blob/master/ci/requirements/py37-min-nep18.yml>`_
- `For everything else
  <https://github.com/pydata/xarray/blob/master/ci/requirements/py37-min-all-deps.yml>`_

.. _installation-instructions:

Instructions
------------

xarray itself is a pure Python package, but its dependencies are not. The
easiest way to get everything installed is to use conda_. To install xarray
with its recommended dependencies using the conda command line tool::

    $ conda install -c conda-forge xarray dask netCDF4 bottleneck

.. _conda: http://conda.io/

If you require other :ref:`optional-dependencies` add them to the line above.

We recommend using the community maintained `conda-forge <https://conda-forge.github.io/>`__ channel,
as some of the dependencies are difficult to build. New releases may also appear in conda-forge before
being updated in the default channel.

If you don't use conda, be sure you have the required dependencies (numpy and
pandas) installed first. Then, install xarray with pip::

    $ pip install xarray

We also maintain other dependency sets for different subsets of functionality::

    $ pip install "xarray[io]"        # Install optional dependencies for handling I/O
    $ pip install "xarray[accel]"     # Install optional dependencies for accelerating xarray
    $ pip install "xarray[parallel]"  # Install optional dependencies for dask arrays
    $ pip install "xarray[viz]"       # Install optional dependencies for visualization
    $ pip install "xarray[complete]"  # Install all the above

The above commands should install most of the `optional dependencies`_. However,
some packages which are either not listed on PyPI or require extra
installation steps are excluded. To know which dependencies would be
installed, take a look at the ``[options.extras_require]`` section in
``setup.cfg``:

.. literalinclude:: ../setup.cfg
   :language: ini
   :start-at: [options.extras_require]
   :end-before: [options.package_data]


Testing
-------

To run the test suite after installing xarray, install (via pypi or conda) `py.test
<https://pytest.org>`__ and run ``pytest`` in the root directory of the xarray
repository.


Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

A fixed-point performance monitoring of (a part of) our codes can be seen on
`this page <https://tomaugspurger.github.io/asv-collection/xarray/>`__.

To run these benchmark tests in a local machine, first install

- `airspeed-velocity <https://asv.readthedocs.io/en/latest/>`__: a tool for benchmarking
  Python packages over their lifetime.

and run
``asv run  # this will install some conda environments in ./.asv/envs``
