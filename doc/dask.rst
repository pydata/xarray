.. _dask:

Parallel computing with Dask
============================

xarray integrates with `Dask <http://dask.pydata.org/>`__ to support parallel
computations and streaming computation on datasets that don't fit into memory.
Currently, Dask is an entirely optional feature for xarray. However, the
benefits of using Dask are sufficiently strong that Dask may become a required
dependency in a future version of xarray.

For a full example of how to use xarray's Dask integration, read the
`blog post introducing xarray and Dask`_. More up-to-date examples
may be found at the `Pangeo project's use-cases <http://pangeo.io/use_cases/index.html>`_
and at the `Dask examples website <https://examples.dask.org/xarray.html>`_.

.. _blog post introducing xarray and Dask: http://stephanhoyer.com/2015/06/11/xray-dask-out-of-core-labeled-arrays/

What is a Dask array?
---------------------

.. image:: _static/dask_array.png
   :width: 40 %
   :align: right
   :alt: A Dask array

Dask divides arrays into many small pieces, called *chunks*, each of which is
presumed to be small enough to fit into memory.

Unlike NumPy, which has eager evaluation, operations on Dask arrays are lazy.
Operations queue up a series of tasks mapped over blocks, and no computation is
performed until you actually ask values to be computed (e.g., to print results
to your screen or write to disk). At that point, data is loaded into memory
and computation proceeds in a streaming fashion, block-by-block.

The actual computation is controlled by a multi-processing or thread pool,
which allows Dask to take full advantage of multiple processors available on
most modern computers.

For more details on Dask, read `its documentation <http://dask.pydata.org/>`__.
Note that xarray only makes use of ``dask.array`` and ``dask.delayed``.

.. _dask.io:

Reading and writing data
------------------------

The usual way to create a ``Dataset`` filled with Dask arrays is to load the
data from a netCDF file or files. You can do this by supplying a ``chunks``
argument to :py:func:`~xarray.open_dataset` or using the
:py:func:`~xarray.open_mfdataset` function.

.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr
    np.random.seed(123456)
    np.set_printoptions(precision=3, linewidth=100, threshold=100, edgeitems=3)

    ds = xr.Dataset({'temperature': (('time', 'latitude', 'longitude'),
                                     np.random.randn(30, 180, 180)),
                     'time': pd.date_range('2015-01-01', periods=30),
                     'longitude': np.arange(180),
                     'latitude': np.arange(89.5, -90.5, -1)})
    ds.to_netcdf('example-data.nc')

.. ipython:: python

    ds = xr.open_dataset('example-data.nc', chunks={'time': 10})
    ds

In this example ``latitude`` and ``longitude`` do not appear in the ``chunks``
dict, so only one chunk will be used along those dimensions.  It is also
entirely equivalent to opening a dataset using :py:meth:`~xarray.open_dataset`
and then chunking the data using the ``chunk`` method, e.g.,
``xr.open_dataset('example-data.nc').chunk({'time': 10})``.

To open multiple files simultaneously in parallel using Dask delayed,
use :py:func:`~xarray.open_mfdataset`::

    xr.open_mfdataset('my/files/*.nc', parallel=True)

This function will automatically concatenate and merge datasets into one in
the simple cases that it understands (see :py:func:`~xarray.auto_combine`
for the full disclaimer). By default, :py:meth:`~xarray.open_mfdataset` will chunk each
netCDF file into a single Dask array; again, supply the ``chunks`` argument to
control the size of the resulting Dask arrays. In more complex cases, you can
open each file individually using :py:meth:`~xarray.open_dataset` and merge the result, as
described in :ref:`combining data`. Passing the keyword argument ``parallel=True`` to :py:meth:`~xarray.open_mfdataset` will speed up the reading of large multi-file datasets by
executing those read tasks in parallel using ``dask.delayed``.

You'll notice that printing a dataset still shows a preview of array values,
even if they are actually Dask arrays. We can do this quickly with Dask because
we only need to compute the first few values (typically from the first block).
To reveal the true nature of an array, print a DataArray:

.. ipython:: python

    ds.temperature

Once you've manipulated a Dask array, you can still write a dataset too big to
fit into memory back to disk by using :py:meth:`~xarray.Dataset.to_netcdf` in the
usual way.

.. ipython:: python

    ds.to_netcdf('manipulated-example-data.nc')

By setting the ``compute`` argument to ``False``, :py:meth:`~xarray.Dataset.to_netcdf`
will return a ``dask.delayed`` object that can be computed later.

.. ipython:: python

    from dask.diagnostics import ProgressBar
    # or distributed.progress when using the distributed scheduler
    delayed_obj = ds.to_netcdf('manipulated-example-data.nc', compute=False)
    with ProgressBar():
        results = delayed_obj.compute()

.. note::

    When using Dask's distributed scheduler to write NETCDF4 files,
    it may be necessary to set the environment variable `HDF5_USE_FILE_LOCKING=FALSE`
    to avoid competing locks within the HDF5 SWMR file locking scheme. Note that
    writing netCDF files with Dask's distributed scheduler is only supported for
    the `netcdf4` backend.

A dataset can also be converted to a Dask DataFrame using :py:meth:`~xarray.Dataset.to_dask_dataframe`.

.. ipython:: python

    df = ds.to_dask_dataframe()
    df

Dask DataFrames do not support multi-indexes so the coordinate variables from the dataset are included as columns in the Dask DataFrame.

.. ipython:: python
    :suppress:

    import os
    os.remove('example-data.nc')
    os.remove('manipulated-example-data.nc')

Using Dask with xarray
----------------------

Nearly all existing xarray methods (including those for indexing, computation,
concatenating and grouped operations) have been extended to work automatically
with Dask arrays. When you load data as a Dask array in an xarray data
structure, almost all xarray operations will keep it as a Dask array; when this
is not possible, they will raise an exception rather than unexpectedly loading
data into memory. Converting a Dask array into memory generally requires an
explicit conversion step. One notable exception is indexing operations: to
enable label based indexing, xarray will automatically load coordinate labels
into memory.

.. tip::

   By default, dask uses its multi-threaded scheduler, which distributes work across
   multiple cores and allows for processing some datasets that do not fit into memory.
   For running across a cluster, `setup the distributed scheduler <https://docs.dask.org/en/latest/setup.html>`_.

The easiest way to convert an xarray data structure from lazy Dask arrays into
*eager*, in-memory NumPy arrays is to use the :py:meth:`~xarray.Dataset.load` method:

.. ipython:: python

    ds.load()

You can also access :py:attr:`~xarray.DataArray.values`, which will always be a
NumPy array:

.. ipython::
    :verbatim:

    In [5]: ds.temperature.values
    Out[5]:
    array([[[  4.691e-01,  -2.829e-01, ...,  -5.577e-01,   3.814e-01],
            [  1.337e+00,  -1.531e+00, ...,   8.726e-01,  -1.538e+00],
            ...
    # truncated for brevity

Explicit conversion by wrapping a DataArray with ``np.asarray`` also works:

.. ipython::
    :verbatim:

    In [5]: np.asarray(ds.temperature)
    Out[5]:
    array([[[  4.691e-01,  -2.829e-01, ...,  -5.577e-01,   3.814e-01],
            [  1.337e+00,  -1.531e+00, ...,   8.726e-01,  -1.538e+00],
            ...

Alternatively you can load the data into memory but keep the arrays as
Dask arrays using the :py:meth:`~xarray.Dataset.persist` method:

.. ipython:: python

   ds = ds.persist()

:py:meth:`~xarray.Dataset.persist` is particularly useful when using a
distributed cluster because the data will be loaded into distributed memory
across your machines and be much faster to use than reading repeatedly from
disk.

.. warning::

   On a single machine :py:meth:`~xarray.Dataset.persist` will try to load all of
   your data into memory. You should make sure that your dataset is not larger than
   available memory.

.. note::
   For more on the differences between :py:meth:`~xarray.Dataset.persist` and
   :py:meth:`~xarray.Dataset.compute` see this `Stack Overflow answer <https://stackoverflow.com/questions/41806850/dask-difference-between-client-persist-and-client-compute>`_ and the `Dask documentation <https://distributed.readthedocs.io/en/latest/manage-computation.html#dask-collections-to-futures>`_.

For performance you may wish to consider chunk sizes.  The correct choice of
chunk size depends both on your data and on the operations you want to perform.
With xarray, both converting data to a Dask arrays and converting the chunk
sizes of Dask arrays is done with the :py:meth:`~xarray.Dataset.chunk` method:

.. ipython:: python
    :suppress:

    ds = ds.chunk({'time': 10})

.. ipython:: python

    rechunked = ds.chunk({'latitude': 100, 'longitude': 100})

You can view the size of existing chunks on an array by viewing the
:py:attr:`~xarray.Dataset.chunks` attribute:

.. ipython:: python

    rechunked.chunks

If there are not consistent chunksizes between all the arrays in a dataset
along a particular dimension, an exception is raised when you try to access
``.chunks``.

.. note::

    In the future, we would like to enable automatic alignment of Dask
    chunksizes (but not the other way around). We might also require that all
    arrays in a dataset share the same chunking alignment. Neither of these
    are currently done.

NumPy ufuncs like ``np.sin`` currently only work on eagerly evaluated arrays
(this will change with the next major NumPy release). We have provided
replacements that also work on all xarray objects, including those that store
lazy Dask arrays, in the :ref:`xarray.ufuncs <api.ufuncs>` module:

.. ipython:: python

    import xarray.ufuncs as xu
    xu.sin(rechunked)

To access Dask arrays directly, use the new
:py:attr:`DataArray.data <xarray.DataArray.data>` attribute. This attribute exposes
array data either as a Dask array or as a NumPy array, depending on whether it has been
loaded into Dask or not:

.. ipython:: python

    ds.temperature.data

.. note::

    In the future, we may extend ``.data`` to support other "computable" array
    backends beyond Dask and NumPy (e.g., to support sparse arrays).

.. _dask.automatic-parallelization:

Automatic parallelization
-------------------------

Almost all of xarray's built-in operations work on Dask arrays. If you want to
use a function that isn't wrapped by xarray, one option is to extract Dask
arrays from xarray objects (``.data``) and use Dask directly.

Another option is to use xarray's :py:func:`~xarray.apply_ufunc`, which can
automate `embarrassingly parallel
<https://en.wikipedia.org/wiki/Embarrassingly_parallel>`__ "map" type operations
where a function written for processing NumPy arrays should be repeatedly
applied to xarray objects containing Dask arrays. It works similarly to
:py:func:`dask.array.map_blocks` and :py:func:`dask.array.atop`, but without
requiring an intermediate layer of abstraction.

For the best performance when using Dask's multi-threaded scheduler, wrap a
function that already releases the global interpreter lock, which fortunately
already includes most NumPy and Scipy functions. Here we show an example
using NumPy operations and a fast function from
`bottleneck <https://github.com/kwgoodman/bottleneck>`__, which
we use to calculate `Spearman's rank-correlation coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`__:

.. code-block:: python

    import numpy as np
    import xarray as xr
    import bottleneck

    def covariance_gufunc(x, y):
        return ((x - x.mean(axis=-1, keepdims=True))
                * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)

    def pearson_correlation_gufunc(x, y):
        return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))

    def spearman_correlation_gufunc(x, y):
        x_ranks = bottleneck.rankdata(x, axis=-1)
        y_ranks = bottleneck.rankdata(y, axis=-1)
        return pearson_correlation_gufunc(x_ranks, y_ranks)

    def spearman_correlation(x, y, dim):
        return xr.apply_ufunc(
            spearman_correlation_gufunc, x, y,
            input_core_dims=[[dim], [dim]],
            dask='parallelized',
            output_dtypes=[float])

The only aspect of this example that is different from standard usage of
``apply_ufunc()`` is that we needed to supply the ``output_dtypes`` arguments.
(Read up on :ref:`comput.wrapping-custom` for an explanation of the
"core dimensions" listed in ``input_core_dims``.)

Our new ``spearman_correlation()`` function achieves near linear speedup
when run on large arrays across the four cores on my laptop. It would also
work as a streaming operation, when run on arrays loaded from disk:

.. ipython::
    :verbatim:

    In [56]: rs = np.random.RandomState(0)

    In [57]: array1 = xr.DataArray(rs.randn(1000, 100000), dims=['place', 'time'])  # 800MB

    In [58]: array2 = array1 + 0.5 * rs.randn(1000, 100000)

    # using one core, on NumPy arrays
    In [61]: %time _ = spearman_correlation(array1, array2, 'time')
    CPU times: user 21.6 s, sys: 2.84 s, total: 24.5 s
    Wall time: 24.9 s

    In [8]: chunked1 = array1.chunk({'place': 10})

    In [9]: chunked2 = array2.chunk({'place': 10})

    # using all my laptop's cores, with Dask
    In [63]: r = spearman_correlation(chunked1, chunked2, 'time').compute()

    In [64]: %time _ = r.compute()
    CPU times: user 30.9 s, sys: 1.74 s, total: 32.6 s
    Wall time: 4.59 s

One limitation of ``apply_ufunc()`` is that it cannot be applied to arrays with
multiple chunks along a core dimension:

.. ipython::
    :verbatim:

    In [63]: spearman_correlation(chunked1, chunked2, 'place')
    ValueError: dimension 'place' on 0th function argument to apply_ufunc with
    dask='parallelized' consists of multiple chunks, but is also a core
    dimension. To fix, rechunk into a single Dask array chunk along this
    dimension, i.e., ``.rechunk({'place': -1})``, but beware that this may
    significantly increase memory usage.

This reflects the nature of core dimensions, in contrast to broadcast (non-core)
dimensions that allow operations to be split into arbitrary chunks for
application.

.. tip::

    For the majority of NumPy functions that are already wrapped by Dask, it's
    usually a better idea to use the pre-existing ``dask.array`` function, by
    using either a pre-existing xarray methods or
    :py:func:`~xarray.apply_ufunc()` with ``dask='allowed'``. Dask can often
    have a more efficient implementation that makes use of the specialized
    structure of a problem, unlike the generic speedups offered by
    ``dask='parallelized'``.

Chunking and performance
------------------------

The ``chunks`` parameter has critical performance implications when using Dask
arrays. If your chunks are too small, queueing up operations will be extremely
slow, because Dask will translate each operation into a huge number of
operations mapped across chunks. Computation on Dask arrays with small chunks
can also be slow, because each operation on a chunk has some fixed overhead from
the Python interpreter and the Dask task executor.

Conversely, if your chunks are too big, some of your computation may be wasted,
because Dask only computes results one chunk at a time.

A good rule of thumb is to create arrays with a minimum chunksize of at least
one million elements (e.g., a 1000x1000 matrix). With large arrays (10+ GB), the
cost of queueing up Dask operations can be noticeable, and you may need even
larger chunksizes.

.. tip::

   Check out the dask documentation on `chunks <https://docs.dask.org/en/latest/array-chunks.html>`_.


Optimization Tips
-----------------

With analysis pipelines involving both spatial subsetting and temporal resampling, Dask performance can become very slow in certain cases. Here are some optimization tips we have found through experience:

1. Do your spatial and temporal indexing (e.g. ``.sel()`` or ``.isel()``) early in the pipeline, especially before calling ``resample()`` or ``groupby()``. Grouping and resampling triggers some computation on all the blocks, which in theory should commute with indexing, but this optimization hasn't been implemented in Dask yet. (See `Dask issue #746 <https://github.com/dask/dask/issues/746>`_).

2. Save intermediate results to disk as a netCDF files (using ``to_netcdf()``) and then load them again with ``open_dataset()`` for further computations. For example, if subtracting temporal mean from a dataset, save the temporal mean to disk before subtracting. Again, in theory, Dask should be able to do the computation in a streaming fashion, but in practice this is a fail case for the Dask scheduler, because it tries to keep every chunk of an array that it computes in memory. (See `Dask issue #874 <https://github.com/dask/dask/issues/874>`_)

3. Specify smaller chunks across space when using :py:meth:`~xarray.open_mfdataset` (e.g., ``chunks={'latitude': 10, 'longitude': 10}``). This makes spatial subsetting easier, because there's no risk you will load chunks of data referring to different chunks (probably not necessary if you follow suggestion 1).

4. Using the h5netcdf package by passing ``engine='h5netcdf'`` to :py:meth:`~xarray.open_mfdataset`
   can be quicker than the default ``engine='netcdf4'`` that uses the netCDF4 package.

5. Some dask-specific tips may be found `here <https://docs.dask.org/en/latest/array-best-practices.html>`_.

6. The dask `diagnostics <https://docs.dask.org/en/latest/understanding-performance.html>`_ can be
   useful in identifying performance bottlenecks.
