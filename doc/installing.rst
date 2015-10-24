Installation
============

Required dependencies
---------------------

- Python 2.6, 2.7, 3.3 or 3.4
- `numpy <http://www.numpy.org/>`__ (1.7 or later)
- `pandas <http://pandas.pydata.org/>`__ (0.15.0 or later)

Optional dependencies
---------------------

For netCDF and IO
~~~~~~~~~~~~~~~~~

- `netCDF4 <https://github.com/Unidata/netcdf4-python>`__: recommended if you
  want to use xray for reading or writing files
- `scipy <http://scipy.org/>`__: used as a fallback for reading/writing netCDF3
- `pydap <http://www.pydap.org/>`__: used as a fallback for accessing OPeNDAP
- `h5netcdf <https://github.com/shoyer/h5netcdf>`__: an alternative library for
  reading and writing netCDF4 files that does not use the netCDF-C libraries

For accelerating xray
~~~~~~~~~~~~~~~~~~~~~

- `bottleneck <https://github.com/kwgoodman/bottleneck>`__: speeds up
  NaN-skipping aggregations by a large factor
- `cyordereddict <https://github.com/shoyer/cyordereddict>`__: speeds up most
  internal operations with xray data structures

For parallel computing
~~~~~~~~~~~~~~~~~~~~~~

- `dask.array <http://dask.pydata.org>`__: required for :ref:`dask`.

For plotting
~~~~~~~~~~~~

- `matplotlib <http://matplotlib.org/>`__: required for :ref:`plotting`.


Instructions
------------

xray itself is a pure Python package, but its dependencies are not. The
easiest way to get them installed is to use conda_. You can then install xray
with its recommended dependencies with the conda command line tool::

    $ conda install xray dask netCDF4 bottleneck

.. _conda: http://conda.io/

If you don't use conda, be sure you have the required dependencies (numpy and
pandas) installed first. Then, install xray with pip::

    $ pip install xray

To run the test suite after installing xray, install
`py.test <https://pytest.org>`__ and run ``py.test xray``.
