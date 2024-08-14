.. _installing:

Installation
============

Required dependencies
---------------------

- Python (3.10 or later)
- `numpy <https://www.numpy.org/>`__ (1.23 or later)
- `packaging <https://packaging.pypa.io/en/latest/#>`__ (23.1 or later)
- `pandas <https://pandas.pydata.org/>`__ (2.0 or later)

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
- `scipy <https://scipy.org>`__: used as a fallback for reading/writing netCDF3
- `pydap <https://www.pydap.org>`__: used as a fallback for accessing OPeNDAP
- `h5netcdf <https://github.com/h5netcdf/h5netcdf>`__: an alternative library for
  reading and writing netCDF4 files that does not use the netCDF-C libraries
- `zarr <https://zarr.readthedocs.io>`__: for chunked, compressed, N-dimensional arrays.
- `cftime <https://unidata.github.io/cftime>`__: recommended if you
  want to encode/decode datetimes for non-standard calendars or dates before
  year 1678 or after year 2262.
- `iris <https://github.com/scitools/iris>`__: for conversion to and from iris'
  Cube objects

For accelerating xarray
~~~~~~~~~~~~~~~~~~~~~~~

- `scipy <https://scipy.org/>`__: necessary to enable the interpolation features for
  xarray objects
- `bottleneck <https://github.com/pydata/bottleneck>`__: speeds up
  NaN-skipping and rolling window aggregations by a large factor
- `numbagg <https://github.com/numbagg/numbagg>`_: for exponential rolling
  window operations

For parallel computing
~~~~~~~~~~~~~~~~~~~~~~

- `dask.array <https://docs.dask.org>`__: required for :ref:`dask`.

For plotting
~~~~~~~~~~~~

- `matplotlib <https://matplotlib.org>`__: required for :ref:`plotting`
- `cartopy <https://scitools.org.uk/cartopy>`__: recommended for :ref:`plot-maps`
- `seaborn <https://seaborn.pydata.org>`__: for better
  color palettes
- `nc-time-axis <https://nc-time-axis.readthedocs.io>`__: for plotting
  cftime.datetime objects

Alternative data containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `sparse <https://sparse.pydata.org/>`_: for sparse arrays
- `pint <https://pint.readthedocs.io/>`_: for units of measure
- Any numpy-like objects that support
  `NEP-18 <https://numpy.org/neps/nep-0018-array-function-protocol.html>`_.
  Note that while such libraries theoretically should work, they are untested.
  Integration tests are in the process of being written for individual libraries.


.. _mindeps_policy:

Minimum dependency versions
---------------------------
Xarray adopts a rolling policy regarding the minimum supported version of its
dependencies:

- **Python:** 30 months
  (`NEP-29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_)
- **numpy:** 18 months
  (`NEP-29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_)
- **all other libraries:** 12 months

This means the latest minor (X.Y) version from N months prior. Patch versions (x.y.Z)
are not pinned, and only the latest available at the moment of publishing the xarray
release is guaranteed to work.

You can see the actual minimum tested versions:

`<https://github.com/pydata/xarray/blob/main/ci/requirements/min-all-deps.yml>`_

.. _installation-instructions:

Instructions
------------

Xarray itself is a pure Python package, but its dependencies are not. The
easiest way to get everything installed is to use conda_. To install xarray
with its recommended dependencies using the conda command line tool::

    $ conda install -c conda-forge xarray dask netCDF4 bottleneck

.. _conda: https://docs.conda.io

If you require other :ref:`optional-dependencies` add them to the line above.

We recommend using the community maintained `conda-forge <https://conda-forge.org>`__ channel,
as some of the dependencies are difficult to build. New releases may also appear in conda-forge before
being updated in the default channel.

If you don't use conda, be sure you have the required dependencies (numpy and
pandas) installed first. Then, install xarray with pip::

    $ python -m pip install xarray

We also maintain other dependency sets for different subsets of functionality::

    $ python -m pip install "xarray[io]"        # Install optional dependencies for handling I/O
    $ python -m pip install "xarray[accel]"     # Install optional dependencies for accelerating xarray
    $ python -m pip install "xarray[parallel]"  # Install optional dependencies for dask arrays
    $ python -m pip install "xarray[viz]"       # Install optional dependencies for visualization
    $ python -m pip install "xarray[complete]"  # Install all the above

The above commands should install most of the `optional dependencies`_. However,
some packages which are either not listed on PyPI or require extra
installation steps are excluded. To know which dependencies would be
installed, take a look at the ``[project.optional-dependencies]`` section in
``pyproject.toml``:

.. literalinclude:: ../../pyproject.toml
   :language: toml
   :start-at: [project.optional-dependencies]
   :end-before: [build-system]

Development versions
--------------------
To install the most recent development version, install from github::

     $ python -m pip install git+https://github.com/pydata/xarray.git

or from TestPyPI::

     $ python -m pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple --pre xarray

Testing
-------

To run the test suite after installing xarray, install (via pypi or conda) `py.test
<https://pytest.org>`__ and run ``pytest`` in the root directory of the xarray
repository.


Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

..
   TODO: uncomment once we have a working setup
         see https://github.com/pydata/xarray/pull/5066

   A fixed-point performance monitoring of (a part of) our code can be seen on
   `this page <https://pandas.pydata.org/speed/xarray/>`__.

To run these benchmark tests in a local machine, first install

- `airspeed-velocity <https://asv.readthedocs.io/en/latest/>`__: a tool for benchmarking
  Python packages over their lifetime.

and run
``asv run  # this will install some conda environments in ./.asv/envs``
