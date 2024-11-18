.. _faq:

Frequently Asked Questions
==========================

.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)


Your documentation keeps mentioning pandas. What is pandas?
-----------------------------------------------------------

pandas_ is a very popular data analysis package in Python
with wide usage in many fields. Our API is heavily inspired by pandas —
this is why there are so many references to pandas.

.. _pandas: https://pandas.pydata.org


Do I need to know pandas to use xarray?
---------------------------------------

No! Our API is heavily inspired by pandas so while knowing pandas will let you
become productive more quickly, knowledge of pandas is not necessary to use xarray.


Should I use xarray instead of pandas?
--------------------------------------

It's not an either/or choice! xarray provides robust support for converting
back and forth between the tabular data-structures of pandas and its own
multi-dimensional data-structures.

That said, you should only bother with xarray if some aspect of data is
fundamentally multi-dimensional. If your data is unstructured or
one-dimensional, pandas is usually the right choice: it has better performance
for common operations such as ``groupby`` and you'll find far more usage
examples online.


Why is pandas not enough?
-------------------------

pandas is a fantastic library for analysis of low-dimensional labelled data -
if it can be sensibly described as "rows and columns", pandas is probably the
right choice.  However, sometimes we want to use higher dimensional arrays
(`ndim > 2`), or arrays for which the order of dimensions (e.g., columns vs
rows) shouldn't really matter. For example, the images of a movie can be
natively represented as an array with four dimensions: time, row, column and
color.

pandas has historically supported N-dimensional panels, but deprecated them in
version 0.20 in favor of xarray data structures. There are now built-in methods
on both sides to convert between pandas and xarray, allowing for more focused
development effort. Xarray objects have a much richer model of dimensionality -
if you were using Panels:

- You need to create a new factory type for each dimensionality.
- You can't do math between NDPanels with different dimensionality.
- Each dimension in a NDPanel has a name (e.g., 'labels', 'items',
  'major_axis', etc.) but the dimension names refer to order, not their
  meaning. You can't specify an operation as to be applied along the "time"
  axis.
- You often have to manually convert collections of pandas arrays
  (Series, DataFrames, etc) to have the same number of dimensions.
  In contrast, this sort of data structure fits very naturally in an
  xarray ``Dataset``.

You can :ref:`read about switching from Panels to xarray here <panel transition>`.
pandas gets a lot of things right, but many science, engineering and complex
analytics use cases need fully multi-dimensional data structures.

How do xarray data structures differ from those found in pandas?
----------------------------------------------------------------

The main distinguishing feature of xarray's ``DataArray`` over labeled arrays in
pandas is that dimensions can have names (e.g., "time", "latitude",
"longitude"). Names are much easier to keep track of than axis numbers, and
xarray uses dimension names for indexing, aggregation and broadcasting. Not only
can you write ``x.sel(time='2000-01-01')`` and  ``x.mean(dim='time')``, but
operations like ``x - x.mean(dim='time')`` always work, no matter the order
of the "time" dimension. You never need to reshape arrays (e.g., with
``np.newaxis``) to align them for arithmetic operations in xarray.


Why don't aggregations return Python scalars?
---------------------------------------------

Xarray tries hard to be self-consistent: operations on a ``DataArray`` (resp.
``Dataset``) return another ``DataArray`` (resp. ``Dataset``) object. In
particular, operations returning scalar values (e.g. indexing or aggregations
like ``mean`` or ``sum`` applied to all axes) will also return xarray objects.

Unfortunately, this means we sometimes have to explicitly cast our results from
xarray when using them in other libraries. As an illustration, the following
code fragment

.. ipython:: python

    arr = xr.DataArray([1, 2, 3])
    pd.Series({"x": arr[0], "mean": arr.mean(), "std": arr.std()})

does not yield the pandas DataFrame we expected. We need to specify the type
conversion ourselves:

.. ipython:: python

    pd.Series({"x": arr[0], "mean": arr.mean(), "std": arr.std()}, dtype=float)

Alternatively, we could use the ``item`` method or the ``float`` constructor to
convert values one at a time

.. ipython:: python

    pd.Series({"x": arr[0].item(), "mean": float(arr.mean())})


.. _approach to metadata:

What is your approach to metadata?
----------------------------------

We are firm believers in the power of labeled data! In addition to dimensions
and coordinates, xarray supports arbitrary metadata in the form of global
(Dataset) and variable specific (DataArray) attributes (``attrs``).

Automatic interpretation of labels is powerful but also reduces flexibility.
With xarray, we draw a firm line between labels that the library understands
(``dims`` and ``coords``) and labels for users and user code (``attrs``). For
example, we do not automatically interpret and enforce units or `CF
conventions`_. (An exception is serialization to and from netCDF files.)

.. _CF conventions: https://cfconventions.org/latest.html

An implication of this choice is that we do not propagate ``attrs`` through
most operations unless explicitly flagged (some methods have a ``keep_attrs``
option, and there is a global flag, accessible with :py:func:`xarray.set_options`,
for setting this to be always True or False). Similarly, xarray does not check
for conflicts between ``attrs`` when combining arrays and datasets, unless
explicitly requested with the option ``compat='identical'``. The guiding
principle is that metadata should not be allowed to get in the way.

In general xarray uses the capabilities of the backends for reading and writing
attributes. That has some implications on roundtripping. One example for such inconsistency is that size-1 lists will roundtrip as single element (for netcdf4 backends).

What other netCDF related Python libraries should I know about?
---------------------------------------------------------------

`netCDF4-python`__ provides a lower level interface for working with
netCDF and OpenDAP datasets in Python. We use netCDF4-python internally in
xarray, and have contributed a number of improvements and fixes upstream. Xarray
does not yet support all of netCDF4-python's features, such as modifying files
on-disk.

__ https://unidata.github.io/netcdf4-python/

Iris_ (supported by the UK Met office) provides similar tools for in-
memory manipulation of labeled arrays, aimed specifically at weather and
climate data needs. Indeed, the Iris :py:class:`~iris.cube.Cube` was direct
inspiration for xarray's :py:class:`~xarray.DataArray`. Xarray and Iris take very
different approaches to handling metadata: Iris strictly interprets
`CF conventions`_. Iris particularly shines at mapping, thanks to its
integration with Cartopy_.

.. _Iris: https://scitools-iris.readthedocs.io/en/stable/
.. _Cartopy: https://scitools.org.uk/cartopy/docs/latest/

We think the design decisions we have made for xarray (namely, basing it on
pandas) make it a faster and more flexible data analysis tool. That said, Iris
has some great domain specific functionality, and xarray includes
methods for converting back and forth between xarray and Iris. See
:py:meth:`~xarray.DataArray.to_iris` for more details.

What other projects leverage xarray?
------------------------------------

See section :ref:`ecosystem`.

How do I open format X file as an xarray dataset?
-------------------------------------------------

To open format X file in xarray, you need to know the `format of the data <https://docs.xarray.dev/en/stable/user-guide/io.html#csv-and-other-formats-supported-by-pandas/>`_ you want to read. If the format is supported, you can use the appropriate function provided by xarray. The following table provides functions used for different file formats in xarray, as well as links to other packages that can be used:

.. csv-table::
   :header: "File Format", "Open via", " Related Packages"
   :widths: 15, 45, 15

   "NetCDF (.nc, .nc4, .cdf)","``open_dataset()`` OR ``open_mfdataset()``", "`netCDF4 <https://pypi.org/project/netCDF4/>`_, `netcdf <https://pypi.org/project/netcdf/>`_ , `cdms2 <https://cdms.readthedocs.io/en/latest/cdms2.html>`_"
   "HDF5 (.h5, .hdf5)","``open_dataset()`` OR ``open_mfdataset()``", "`h5py <https://www.h5py.org/>`_, `pytables <https://www.pytables.org/>`_ "
   "GRIB (.grb, .grib)", "``open_dataset()``", "`cfgrib <https://pypi.org/project/cfgrib/>`_, `pygrib <https://pypi.org/project/pygrib/>`_"
   "CSV (.csv)","``open_dataset()``", "`pandas`_ , `dask <https://www.dask.org/>`_"
   "Zarr (.zarr)","``open_dataset()`` OR ``open_mfdataset()``", "`zarr <https://pypi.org/project/zarr/>`_ , `dask <https://www.dask.org/>`_ "

.. _pandas: https://pandas.pydata.org

If you are unable to open a file in xarray:

- You should check that you are having all necessary dependencies installed, including any optional dependencies (like scipy, h5netcdf, cfgrib etc as mentioned below) that may be required for the specific use case.

- If all necessary dependencies are installed but the file still cannot be opened, you must check if there are any specialized backends available for the specific file format you are working with. You can consult the xarray documentation or the documentation for the file format to determine if a specialized backend is required, and if so, how to install and use it with xarray.

- If the file format is not supported by xarray or any of its available backends, the user may need to use a different library or tool to work with the file. You can consult the documentation for the file format to determine which tools are recommended for working with it.

Xarray provides a default engine to read files, which is usually determined by the file extension or type. If you don't specify the engine, xarray will try to guess it based on the file extension or type, and may fall back to a different engine if it cannot determine the correct one.

Therefore, it's good practice to always specify the engine explicitly, to ensure that the correct backend is used and especially when working with complex data formats or non-standard file extensions.

:py:func:`xarray.backends.list_engines` is a function in xarray that returns a dictionary of available engines and their BackendEntrypoint objects.

You can use the ``engine`` argument to specify the backend when calling ``open_dataset()`` or other reading functions in xarray, as shown below:

NetCDF
~~~~~~
If you are reading a netCDF file with a ".nc" extension, the default engine is ``netcdf4``. However if you have files with non-standard extensions or if the file format is ambiguous. Specify the engine explicitly, to ensure that the correct backend is used.

Use :py:func:`~xarray.open_dataset` to open a NetCDF file and return an xarray Dataset object.

.. code:: python

    import xarray as xr

    # use xarray to open the file and return an xarray.Dataset object using netcdf4 engine

    ds = xr.open_dataset("/path/to/my/file.nc", engine="netcdf4")

    # Print Dataset object

    print(ds)

    # use xarray to open the file and return an xarray.Dataset object using scipy engine

    ds = xr.open_dataset("/path/to/my/file.nc", engine="scipy")

We recommend installing ``scipy`` via conda using the below given code:

::

    conda install scipy

HDF5
~~~~
Use :py:func:`~xarray.open_dataset` to open an HDF5 file and return an xarray Dataset object.

You should specify the ``engine`` keyword argument when reading HDF5 files with xarray, as there are multiple backends that can be used to read HDF5 files, and xarray may not always be able to automatically detect the correct one based on the file extension or file format.

To read HDF5 files with xarray, you can use the :py:func:`~xarray.open_dataset` function from the ``h5netcdf`` backend, as follows:

.. code:: python

    import xarray as xr

    # Open HDF5 file as an xarray Dataset

    ds = xr.open_dataset("path/to/hdf5/file.hdf5", engine="h5netcdf")

    # Print Dataset object

    print(ds)

We recommend you to install ``h5netcdf`` library using the below given code:

::

    conda install -c conda-forge h5netcdf

If you want to use the ``netCDF4`` backend to read a file with a ".h5" extension (which is typically associated with HDF5 file format), you can specify the engine argument as follows:

.. code:: python

    ds = xr.open_dataset("path/to/file.h5", engine="netcdf4")

GRIB
~~~~
You should specify the ``engine`` keyword argument when reading GRIB files with xarray, as there are multiple backends that can be used to read GRIB files, and xarray may not always be able to automatically detect the correct one based on the file extension or file format.

Use the :py:func:`~xarray.open_dataset` function from the ``cfgrib`` package to open a GRIB file as an xarray Dataset.

.. code:: python

    import xarray as xr

    # define the path to your GRIB file and the engine you want to use to open the file
    # use ``open_dataset()`` to open the file with the specified engine and return an xarray.Dataset object

    ds = xr.open_dataset("path/to/your/file.grib", engine="cfgrib")

    # Print Dataset object

    print(ds)

We recommend installing ``cfgrib`` via conda using the below given code:

::

    conda install -c conda-forge cfgrib

CSV
~~~
By default, xarray uses the built-in ``pandas`` library to read CSV files. In general, you don't need to specify the engine keyword argument when reading CSV files with xarray, as the default ``pandas`` engine is usually sufficient for most use cases. If you are working with very large CSV files or if you need to perform certain types of data processing that are not supported by the default ``pandas`` engine, you may want to use a different backend.
In such cases, you can specify the engine argument when reading the CSV file with xarray.

To read CSV files with xarray, use the :py:func:`~xarray.open_dataset` function and specify the path to the CSV file as follows:

.. code:: python

    import xarray as xr
    import pandas as pd

    # Load CSV file into pandas DataFrame using the "c" engine

    df = pd.read_csv("your_file.csv", engine="c")

    # Convert `:py:func:pandas` DataFrame to xarray.Dataset

    ds = xr.Dataset.from_dataframe(df)

    # Prints the resulting xarray dataset

    print(ds)

Zarr
~~~~
When opening a Zarr dataset with xarray, the ``engine`` is automatically detected based on the file extension or the type of input provided. If the dataset is stored in a directory with a ".zarr" extension, xarray will automatically use the "zarr" engine.

To read zarr files with xarray, use the :py:func:`~xarray.open_dataset` function and specify the path to the zarr file as follows:

.. code:: python

    import xarray as xr

    # use xarray to open the file and return an xarray.Dataset object using zarr engine

    ds = xr.open_dataset("path/to/your/file.zarr", engine="zarr")

    # Print Dataset object

    print(ds)

We recommend installing ``zarr`` via conda using the below given code:

::

    conda install -c conda-forge zarr

There may be situations where you need to specify the engine manually using the ``engine`` keyword argument. For example, if you have a Zarr dataset stored in a file with a different extension (e.g., ".npy"), you will need to specify the engine as "zarr" explicitly when opening the dataset.

Some packages may have additional functionality beyond what is shown here. You can refer to the documentation for each package for more information.

How does xarray handle missing values?
--------------------------------------

**xarray can handle missing values using ``np.nan``**

- ``np.nan`` is  used to represent missing values in labeled arrays and datasets. It is a commonly used standard for representing missing or undefined numerical data in scientific computing. ``np.nan`` is a constant value in NumPy that represents "Not a Number" or missing values.

- Most of xarray's computation methods are designed to automatically handle missing values appropriately.

  For example, when performing operations like addition or multiplication on arrays that contain missing values, xarray will automatically ignore the missing values and only perform the operation on the valid data. This makes it easy to work with data that may contain missing or undefined values without having to worry about handling them explicitly.

- Many of xarray's `aggregation methods <https://docs.xarray.dev/en/stable/user-guide/computation.html#aggregation>`_, such as ``sum()``, ``mean()``, ``min()``, ``max()``, and others, have a skipna argument that controls whether missing values (represented by NaN) should be skipped (True) or treated as NaN (False) when performing the calculation.

  By default, ``skipna`` is set to ``True``, so missing values are ignored when computing the result. However, you can set ``skipna`` to ``False`` if you want missing values to be treated as NaN and included in the calculation.

- On `plotting <https://docs.xarray.dev/en/stable/user-guide/plotting.html#missing-values>`_ an xarray dataset or array that contains missing values, xarray will simply leave the missing values as blank spaces in the plot.

- We have a set of `methods <https://docs.xarray.dev/en/stable/user-guide/computation.html#missing-values>`_ for manipulating missing and filling values.

How should I cite xarray?
-------------------------

If you are using xarray and would like to cite it in academic publication, we
would certainly appreciate it. We recommend two citations.

  1. At a minimum, we recommend citing the xarray overview journal article,
     published in the Journal of Open Research Software.

     - Hoyer, S. & Hamman, J., (2017). xarray: N-D labeled Arrays and
       Datasets in Python. Journal of Open Research Software. 5(1), p.10.
       DOI: https://doi.org/10.5334/jors.148

       Here’s an example of a BibTeX entry::

           @article{hoyer2017xarray,
             title     = {xarray: {N-D} labeled arrays and datasets in {Python}},
             author    = {Hoyer, S. and J. Hamman},
             journal   = {Journal of Open Research Software},
             volume    = {5},
             number    = {1},
             year      = {2017},
             publisher = {Ubiquity Press},
             doi       = {10.5334/jors.148},
             url       = {https://doi.org/10.5334/jors.148}
           }

  2. You may also want to cite a specific version of the xarray package. We
     provide a `Zenodo citation and DOI <https://doi.org/10.5281/zenodo.598201>`_
     for this purpose:

        .. image:: https://zenodo.org/badge/doi/10.5281/zenodo.598201.svg
           :target: https://doi.org/10.5281/zenodo.598201

       An example BibTeX entry::

           @misc{xarray_v0_8_0,
                 author = {Stephan Hoyer and Clark Fitzgerald and Joe Hamman and others},
                 title  = {xarray: v0.8.0},
                 month  = aug,
                 year   = 2016,
                 doi    = {10.5281/zenodo.59499},
                 url    = {https://doi.org/10.5281/zenodo.59499}
                }

.. _public api:

What parts of xarray are considered public API?
-----------------------------------------------

As a rule, only functions/methods documented in our :ref:`api` are considered
part of xarray's public API. Everything else (in particular, everything in
``xarray.core`` that is not also exposed in the top level ``xarray`` namespace)
is considered a private implementation detail that may change at any time.

Objects that exist to facilitate xarray's fluent interface on ``DataArray`` and
``Dataset`` objects are a special case. For convenience, we document them in
the API docs, but only their methods and the ``DataArray``/``Dataset``
methods/properties to construct them (e.g., ``.plot()``, ``.groupby()``,
``.str``) are considered public API. Constructors and other details of the
internal classes used to implemented them (i.e.,
``xarray.plot.plotting._PlotMethods``, ``xarray.core.groupby.DataArrayGroupBy``,
``xarray.core.accessor_str.StringAccessor``) are not.
