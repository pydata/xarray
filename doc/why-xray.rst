Why xray?
=========

pandas_ excels at working with tabular data. That suffices for many statistical
analyses, but physical scientists rely on N-dimensional arrays -- which is
where xray comes in.

xray aims to provide a data analysis toolkit as powerful as pandas_ but
designed for working with homogeneous N-dimensional arrays
instead of tabular data. When possible, we copy the pandas API and rely on
pandas's highly optimized internals (in particular, for fast indexing).

Importantly, xray has robust support for converting its objects to and
from a numpy ``ndarray`` or a pandas ``DataFrame`` or ``Series``, providing
compatibility with the full `PyData ecosystem <http://pydata.org/>`__.

Our target audience is anyone who needs N-dimensional labeled arrays, but we
are particularly focused on the data analysis needs of physical scientists --
especially geoscientists who already know and love netCDF_.

.. _ndarray: http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf
.. _pandas: http://pandas.pydata.org

.. warning::

    xray is a relatively new project and is still under heavy development.
    Although we will make a best effort to maintain compatibility with the
    current API, the API will change in future versions as xray matures.
    Already anticipated changes are called out in the :doc:`tutorial`.


Why ``DataArray``?
------------------

Adding dimensions names and coordinate indexes to numpy's ndarray_ makes many
powerful array operations possible:

-  Apply operations over dimensions by name: ``x.sum('time')``.
-  Select values by label instead of integer location:
   ``x.loc['2014-01-01']`` or ``x.sel(time='2014-01-01')``.
-  Mathematical operations (e.g., ``x - y``) vectorize across multiple
   dimensions (array broadcasting) based on dimension names, not shape.
-  Flexible split-apply-combine operations with groupby:
   ``x.groupby('time.dayofyear').mean()``.
-  Database like aligment based on coordinate labels that smoothly
   handles missing values: ``x, y = xray.align(x, y, join='outer')``.
-  Keep track of arbitrary metadata in the form of a Python dictionary:
   ``x.attrs``.

pandas_ provides many of these features, but it does not make use of dimension
names, and its core data structures are fixed dimensional arrays.

The N-dimensional nature of the ``DataArray`` makes it suitable for dealing
with multi-dimensional scientific data, and its use of dimension names
instead of axis labels (``dim='time'`` instead of ``axis=0``) makes such
arrays much more manageable than the raw numpy ndarray: with xray, you don't
need to keep track of the order of arrays dimensions or insert dummy dimensions
(e.g., ``np.newaxis``) to align arrays.

The support for storing arbitrary metadata along with an array is also a
welcome feature for scientific users.

The name ``DataArray`` is borrowed from Fernando Perez's datarray_ project,
which prototyped a similar data structure.

.. _datarray: https://github.com/fperez/datarray


Why ``Dataset``?
----------------

The ``Dataset`` is a multi-dimensional, in-memory, array database. It is a
dict-like container of ``DataArray`` objects aligned along any number of
shared dimensions.

The power of the dataset over a plain dictionary is that, in addition to
pulling out arrays by name, it is possible to select or combine data along a
dimension across all arrays simultaneously. In this fashion, it is similar to
a multi-dimensional :py:class:`~pandas.DataFrame`.

Here is an illustrative example:

.. image:: _static/dataset-diagram.png


- The dimensions ``x``, ``y`` and ``z`` each have a fixed size across all
  arrays.
- ``temperature`` and ``pressure`` are 3D arrays with the dimensions ``x``,
  ``y`` and ``z``.
- ``latitude`` and ``longitude`` are 2D arrays with the dimensions ``x`` and
  ``y``.
- ``x``, ``y`` and ``z`` are 1D arrays (`coordinates`) of tick marks that
  label each point in the other arrays.
- ``time`` is a scalar value (a 0-dimensional array).

This data model is borrowed from the netCDF_ file format, which also provides
xray with a natural and portable serialization format. NetCDF is very popular
in the geosciences, and there are existing libraries for reading and writing
netCDF in many programming languages, including Python.

xray distinguishes itself from most other tools for working with netCDF data
in-so-far as it provides data structures for in-memory analytics that both
utilize and preserve labels. You only need to do the tedious work of adding
metadata once, not every time you save a file.

``Dataset`` supports all of the features listed above for an individual
``DataArray``, except it does not (yet) directly support arithmetic operators.
