Installing xray
===============

Required dependencies:

- Python 2.6, 2.7, 3.3 or 3.4
- `numpy <http://www.numpy.org/>`__ (1.7 or later)
- `pandas <http://pandas.pydata.org/>`__ (0.13.1 or later)

Optional dependencies:

- `netCDF4 <https://github.com/Unidata/netcdf4-python>`__ (recommended)
- `pydap <http://www.pydap.org/>`__
- `scipy <http://scipy.org/>`__

The easiest way to get all these dependencies installed is to use the
`Anaconda python distribution <https://store.continuum.io/cshop/anaconda/>`__.

To install xray, use pip::

    pip install xray

.. warning::

    If you don't already have recent versions of numpy and pandas installed,
    installing xray will attempt to automatically update them. This may or may
    not succeed: you probably want to ensure you have an up-to-date installs
    of numpy and pandas before attempting to install xray.
