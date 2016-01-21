Overview: Why xarray?
=====================

Features
--------

Adding dimensions names and coordinate indexes to numpy's ndarray_ makes many
powerful array operations possible:

-  Apply operations over dimensions by name: ``x.sum('time')``.
-  Select values by label instead of integer location:
   ``x.loc['2014-01-01']`` or ``x.sel(time='2014-01-01')``.
-  Mathematical operations (e.g., ``x - y``) vectorize across multiple
   dimensions (array broadcasting) based on dimension names, not shape.
-  Flexible split-apply-combine operations with groupby:
   ``x.groupby('time.dayofyear').mean()``.
-  Database like alignment based on coordinate labels that smoothly
   handles missing values: ``x, y = xr.align(x, y, join='outer')``.
-  Keep track of arbitrary metadata in the form of a Python dictionary:
   ``x.attrs``.

pandas_ provides many of these features, but it does not make use of dimension
names, and its core data structures are fixed dimensional arrays.

The N-dimensional nature of xarray's data structures makes it suitable for dealing
with multi-dimensional scientific data, and its use of dimension names
instead of axis labels (``dim='time'`` instead of ``axis=0``) makes such
arrays much more manageable than the raw numpy ndarray: with xarray, you don't
need to keep track of the order of arrays dimensions or insert dummy dimensions
(e.g., ``np.newaxis``) to align arrays.

Core data structures
--------------------

xarray has two core data structures. Both are fundamentally N-dimensional:

- :py:class:`~xarray.DataArray` is our implementation of a labeled, N-dimensional
  array. It is an N-D generalization of a :py:class:`pandas.Series`. The name
  ``DataArray`` itself is borrowed from Fernando Perez's datarray_ project,
  which prototyped a similar data structure.
- :py:class:`~xarray.Dataset` is a multi-dimensional, in-memory array database.
  It is a dict-like container of ``DataArray`` objects aligned along any number of
  shared dimensions, and serves a similar purpose in xarray to the
  :py:class:`pandas.DataFrame`.

.. _datarray: https://github.com/fperez/datarray

The value of attaching labels to numpy's :py:class:`numpy.ndarray` may be
fairly obvious, but the dataset may need more motivation.

The power of the dataset over a plain dictionary is that, in addition to
pulling out arrays by name, it is possible to select or combine data along a
dimension across all arrays simultaneously. Like a
:py:class:`~pandas.DataFrame`, datasets facilitate array operations with
heterogeneous data -- the difference is that the arrays in a dataset can not
only have different data types, but can also have different numbers of
dimensions.

This data model is borrowed from the netCDF_ file format, which also provides
xarray with a natural and portable serialization format. NetCDF is very popular
in the geosciences, and there are existing libraries for reading and writing
netCDF in many programming languages, including Python.

xarray distinguishes itself from many tools for working with netCDF data
in-so-far as it provides data structures for in-memory analytics that both
utilize and preserve labels. You only need to do the tedious work of adding
metadata once, not every time you save a file.

Goals and aspirations
---------------------

pandas_ excels at working with tabular data. That suffices for many statistical
analyses, but physical scientists rely on N-dimensional arrays -- which is
where xarray comes in.

xarray aims to provide a data analysis toolkit as powerful as pandas_ but
designed for working with homogeneous N-dimensional arrays
instead of tabular data. When possible, we copy the pandas API and rely on
pandas's highly optimized internals (in particular, for fast indexing).

Importantly, xarray has robust support for converting its objects to and
from a numpy ``ndarray`` or a pandas ``DataFrame`` or ``Series``, providing
compatibility with the full `PyData ecosystem <http://pydata.org/>`__.

Our target audience is anyone who needs N-dimensional labeled arrays, but we
are particularly focused on the data analysis needs of physical scientists --
especially geoscientists who already know and love netCDF_.

.. _ndarray: http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf
.. _pandas: http://pandas.pydata.org
