Installation
============

Required dependencies:

- Python 2.6, 2.7, 3.3 or 3.4
- `numpy <http://www.numpy.org/>`__ (1.7 or later)
- `pandas <http://pandas.pydata.org/>`__ (0.15.0 or later)

Optional dependencies:

- `netCDF4 <https://github.com/Unidata/netcdf4-python>`__: recommended if you
  want to use xray for reading or writing files
- `scipy <http://scipy.org/>`__: used as a fallback for reading/writing netCDF3
- `pydap <http://www.pydap.org/>`__: used as a fallback for accessing OPeNDAP
- `cyordereddict <https://github.com/shoyer/cyordereddict>`__: speeds up most
  internal operations

Before you install xray, be sure you have the required dependencies (numpy and
pandas) installed. The easiest way to do so is to use the
`Anaconda python distribution <https://store.continuum.io/cshop/anaconda/>`__.

To install xray, use pip::

    pip install xray

To run the test suite after installing xray, install
`nose <https://nose.readthedocs.org>`__ and run ``nosetests xray``.
