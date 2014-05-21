Installing xray
===============

xray runs on Python 2.7 and Python 3.3 (Python 3.4 has not yet been tested).
It also requires
`numpy <http://www.numpy.org/>`__ (1.7 or later) and
`pandas <http://pandas.pydata.org/>`__ (0.13.1 or later).
`netCDF4-python <https://github.com/Unidata/netcdf4-python>`__,
`pydap <http://www.pydap.org/>`__ and `scipy <http://scipy.org/>`__ are
optional: they add support for reading and writing netCDF files and/or
accessing OpenDAP datasets.

The easiest way to get all these dependencies installed is to use the
`Anaconda python distribution <https://store.continuum.io/cshop/anaconda/>`__.

To install xray, use pip:

::

    pip install xray

.. warning::

    If you don't already have recent versions of numpy and pandas installed,
    installing xray will automatically update them.
