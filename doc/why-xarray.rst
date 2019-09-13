Overview: Why xarray?
=====================

What labels enable
------------------

Multi-dimensional (a.k.a. N-dimensional, ND) arrays (sometimes called
"tensors") are an essential part of computational science.
They are encountered in a wide range of fields, including physics, astronomy,
geoscience, bioinformatics, engineering, finance, and deep learning.
In Python, NumPy_ provides the fundamental data structure and API for
working with raw ND arrays.
However, real-world datasets are usually more than just raw numbers;
they have labels which encode information about how the array values map
to locations in space, time, etc.

Xarray doesn't just keep track of labels on arrays -- it uses them to provide a
powerful and concise interface. For example:

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

The N-dimensional nature of xarray's data structures makes it suitable for dealing
with multi-dimensional scientific data, and its use of dimension names
instead of axis labels (``dim='time'`` instead of ``axis=0``) makes such
arrays much more manageable than the raw numpy ndarray: with xarray, you don't
need to keep track of the order of arrays dimensions or insert dummy dimensions
(e.g., ``np.newaxis``) to align arrays.

The immediate payoff of using xarray is that you'll write less code. The
long-term payoff is that you'll understand what you were thinking when you come
back to look at it weeks or months later.

Core data structures
--------------------

xarray has two core data structures, which build upon and extend the core
strengths of  NumPy_ and pandas_. Both are fundamentally N-dimensional:

- :py:class:`~xarray.DataArray` is our implementation of a labeled, N-dimensional
  array. It is an N-D generalization of a :py:class:`pandas.Series`. The name
  ``DataArray`` itself is borrowed from Fernando Perez's datarray_ project,
  which prototyped a similar data structure.
- :py:class:`~xarray.Dataset` is a multi-dimensional, in-memory array database.
  It is a dict-like container of ``DataArray`` objects aligned along any number of
  shared dimensions, and serves a similar purpose in xarray to the
  :py:class:`pandas.DataFrame`.

The value of attaching labels to numpy's :py:class:`numpy.ndarray` may be
fairly obvious, but the dataset may need more motivation.

The power of the dataset over a plain dictionary is that, in addition to
pulling out arrays by name, it is possible to select or combine data along a
dimension across all arrays simultaneously. Like a
:py:class:`~pandas.DataFrame`, datasets facilitate array operations with
heterogeneous data -- the difference is that the arrays in a dataset can have 
not only different data types, but also different numbers of dimensions.

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

Xarray contributes domain-agnostic data-structures and tools for labeled
multi-dimensional arrays to Python's SciPy_ ecosystem for numerical computing.
In particular, xarray builds upon and integrates with NumPy_ and pandas_:

- Our user-facing interfaces aim to be more explicit verisons of those found in
  NumPy/pandas.
- Compatibility with the broader ecosystem is a major goal: it should be easy
  to get your data in and out.
- We try to keep a tight focus on functionality and interfaces related to
  labeled data, and leverage other Python libraries for everything else, e.g.,
  NumPy/pandas for fast arrays/indexing (xarray itself contains no compiled
  code), Dask_ for parallel computing, matplotlib_ for plotting, etc.

Xarray is a collaborative and community driven project, run entirely on
volunteer effort (see :ref:`contributing`).
Our target audience is anyone who needs N-dimensional labeled arrays in Python.
Originally, development was driven by the data analysis needs of physical
scientists (especially geoscientists who already know and love
netCDF_), but it has become a much more broadly useful tool, and is still
under active development.
See our technical :ref:`roadmap` for more details, and feel free to reach out
with questions about whether xarray is the right tool for your needs.

.. _datarray: https://github.com/fperez/datarray
.. _Dask: http://dask.org
.. _matplotlib: http://matplotlib.org
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf
.. _NumPy: http://www.numpy.org
.. _pandas: http://pandas.pydata.org
.. _SciPy: http://www.scipy.org
