xray: extended arrays for working with scientific datasets in Python
====================================================================

**xray** is a Python package for working with aligned sets of
homogeneous, n-dimensional arrays. It implements flexible array
operations and dataset manipulation for in-memory datasets within the
`Common Data
Model <http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/CDM/>`__
widely used for self-describing scientific data (e.g., the NetCDF file
format).

Why xray?
---------

Adding dimensions names and coordinate values to numpy's
`ndarray <http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`__
makes many powerful array operations possible:

-  Apply operations over dimensions by name: ``x.sum('time')``.
-  Select values by label instead of integer location:
   ``x.loc['2014-01-01']`` or ``x.labeled(time='2014-01-01')``.
-  Mathematical operations (e.g., ``x - y``) vectorize across multiple
   dimensions (known in numpy as "broadcasting") based on dimension
   names, regardless of their original order.
-  Flexible split-apply-combine operations with groupby:
   ``x.groupby('time.dayofyear').apply(lambda y: y - y.mean())``.
-  Database like aligment based on coordinate labels that smoothly
   handles missing values: ``x, y = xray.align(x, y, join='outer')``.
-  Keep track of arbitrary metadata in the form of a Python dictionary:
   ``x.attrs``.

**xray** aims to provide a data analysis toolkit as powerful as
`pandas <http://pandas.pydata.org/>`__ but designed for working with
homogeneous N-dimensional arrays instead of tabular data. Indeed, much
of its design and internal functionality (in particular, fast indexing)
is shamelessly borrowed from pandas.

Because **xray** implements the same data model as the NetCDF file
format, xray datasets have a natural and portable serialization format.
But it's also easy to robustly convert an xray ``DataArray`` to and from
a numpy ``ndarray`` or a pandas ``DataFrame`` or ``Series``, providing
compatibility with the full `PyData ecosystem <http://pydata.org/>`__.

For a longer introduction to **xray** and its design goals, see
`the project's GitHub page <https://github.com/akleeman/xray>`__.

Contents
--------

.. toctree::
   :maxdepth: 1

   installing
   tutorial
   data-structures
   api
