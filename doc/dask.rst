.. _dask:

Parallel computing with dask
============================

xarray integrates with `dask <http://dask.pydata.org/>`__ to support parallel
computations and streaming computation on datasets that don't fit into memory.

Currently, dask is an entirely optional feature for xarray. However, the
benefits of using dask are sufficiently strong that dask may become a required
dependency in a future version of xarray.

For a full example of how to use xarray's dask integration, read the
`blog post introducing xarray and dask`_.

.. _blog post introducing xarray and dask: https://www.anaconda.com/blog/developer-blog/xray-dask-out-core-labeled-arrays-python/

What is a dask array?
---------------------

.. image:: _static/dask_array.png
   :width: 40 %
   :align: right
   :alt: A dask array

Dask divides arrays into many small pieces, called *chunks*, each of which is
presumed to be small enough to fit into memory.

Unlike NumPy, which has eager evaluation, operations on dask arrays are lazy.
Operations queue up a series of tasks mapped over blocks, and no computation is
performed until you actually ask values to be computed (e.g., to print results
to your screen or write to disk). At that point, data is loaded into memory
and computation proceeds in a streaming fashion, block-by-block.

The actual computation is controlled by a multi-processing or thread pool,
which allows dask to take full advantage of multiple processors available on
most modern computers.

For more details on dask, read `its documentation <http://dask.pydata.org/>`__.

.. _dask.io:

Reading and writing data
------------------------

The usual way to create a dataset filled with dask arrays is to load the
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
                                     np.random.randn(365, 180, 360)),
                     'time': pd.date_range('2015-01-01', periods=365),
                     'longitude': np.arange(360),
                     'latitude': np.arange(89.5, -90.5, -1)})
    ds.to_netcdf('example-data.nc')

.. ipython:: python

    ds = xr.open_dataset('example-data.nc', chunks={'time': 10})
    ds

In this example ``latitude`` and ``longitude`` do not appear in the
``chunks`` dict, so only one chunk will be used along those dimensions.  It
is also entirely equivalent to open a dataset using ``open_dataset`` and
then chunk the data use the ``chunk`` method, e.g.,
``xr.open_dataset('example-data.nc').chunk({'time': 10})``.

To open multiple files simultaneously, use :py:func:`~xarray.open_mfdataset`::

    xr.open_mfdataset('my/files/*.nc')

This function will automatically concatenate and merge dataset into one in
the simple cases that it understands (see :py:func:`~xarray.auto_combine`
for the full disclaimer). By default, ``open_mfdataset`` will chunk each
netCDF file into a single dask array; again, supply the ``chunks`` argument to
control the size of the resulting dask arrays. In more complex cases, you can
open each file individually using ``open_dataset`` and merge the result, as
described in :ref:`combining data`.

You'll notice that printing a dataset still shows a preview of array values,
even if they are actually dask arrays. We can do this quickly
with dask because we only need to the compute the first few values (typically
from the first block). To reveal the true nature of an array, print a DataArray:

.. ipython:: python

    ds.temperature

Once you've manipulated a dask array, you can still write a dataset too big to
fit into memory back to disk by using :py:meth:`~xarray.Dataset.to_netcdf` in the
usual way.

A dataset can also be converted to a dask DataFrame using :py:meth:`~xarray.Dataset.to_dask_dataframe`.

.. ipython:: python

    df = ds.to_dask_dataframe()
    df

Dask DataFrames do not support multi-indexes so the coordinate variables from the dataset are included as columns in the dask DataFrame.

Using dask with xarray
----------------------

Nearly all existing xarray methods (including those for indexing, computation,
concatenating and grouped operations) have been extended to work automatically
with dask arrays. When you load data as a dask array in an xarray data
structure, almost all xarray operations will keep it as a dask array; when this
is not possible, they will raise an exception rather than unexpectedly loading
data into memory. Converting a dask array into memory generally requires an
explicit conversion step. One noteable exception is indexing operations: to
enable label based indexing, xarray will automatically load coordinate labels
into memory.

The easiest way to convert an xarray data structure from lazy dask arrays into
eager, in-memory numpy arrays is to use the :py:meth:`~xarray.Dataset.load` method:

.. ipython:: python

    ds.load()

You can also access :py:attr:`~xarray.DataArray.values`, which will always be a
numpy array:

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
dask arrays using the :py:meth:`~xarray.Dataset.persist` method:

.. ipython::

   ds = ds.persist()

This is particularly useful when using a distributed cluster because the data
will be loaded into distributed memory across your machines and be much faster
to use than reading repeatedly from disk.  Warning that on a single machine
this operation will try to load all of your data into memory.  You should make
sure that your dataset is not larger than available memory.

For performance you may wish to consider chunk sizes.  The correct choice of
chunk size depends both on your data and on the operations you want to perform.
With xarray, both converting data to a dask arrays and converting the chunk
sizes of dask arrays is done with the :py:meth:`~xarray.Dataset.chunk` method:

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

    In the future, we would like to enable automatic alignment of dask
    chunksizes (but not the other way around). We might also require that all
    arrays in a dataset share the same chunking alignment. Neither of these
    are currently done.

NumPy ufuncs like ``np.sin`` currently only work on eagerly evaluated arrays
(this will change with the next major NumPy release). We have provided
replacements that also work on all xarray objects, including those that store
lazy dask arrays, in the :ref:`xarray.ufuncs <api.ufuncs>` module:

.. ipython:: python

    import xarray.ufuncs as xu
    xu.sin(rechunked)

To access dask arrays directly, use the new
:py:attr:`DataArray.data <xarray.DataArray.data>` attribute. This attribute exposes
array data either as a dask array or as a numpy array, depending on whether it has been
loaded into dask or not:

.. ipython:: python

    ds.temperature.data

.. note::

    In the future, we may extend ``.data`` to support other "computable" array
    backends beyond dask and numpy (e.g., to support sparse arrays).

.. _dask.automatic-parallelization:

Automatic parallelization
-------------------------

Almost all of xarray's built-in operations work on dask arrays. If you want to
use a function that isn't wrapped by xarray, one option is to extract dask
arrays from xarray objects (``.data``) and use dask directly.

Another option is to use xarray's :py:func:`~xarray.apply_ufunc`, which can
automate `embarrassingly parallel <https://en.wikipedia.org/wiki/Embarrassingly_parallel>`__
"map" type operations where a functions written for processing NumPy arrays
should be repeatedly applied to xarray objects containing dask arrays. It works
similarly to :py:func:`dask.array.map_blocks` and :py:func:`dask.array.atop`,
but without requiring an intermediate layer of abstraction.

For the best performance when using dask's multi-threaded scheduler, wrap a
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

    # using one core, on numpy arrays
    In [61]: %time _ = spearman_correlation(array1, array2, 'time')
    CPU times: user 21.6 s, sys: 2.84 s, total: 24.5 s
    Wall time: 24.9 s

    In [8]: chunked1 = array1.chunk({'place': 10})

    In [9]: chunked2 = array2.chunk({'place': 10})

    # using all my laptop's cores, with dask
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
    dimension. To fix, rechunk into a single dask array chunk along this
    dimension, i.e., ``.rechunk({'place': -1})``, but beware that this may
    significantly increase memory usage.

The reflects the nature of core dimensions, in contrast to broadcast
(non-core) dimensions that allow operations to be split into arbitrary chunks
for application.

.. tip::

    For the majority of NumPy functions that are already wrapped by dask, it's
    usually a better idea to use the pre-existing ``dask.array`` function, by
    using either a pre-existing xarray methods or
    :py:func:`~xarray.apply_ufunc()` with ``dask='allowed'``. Dask can often
    have a more efficient implementation that makes use of the specialized
    structure of a problem, unlike the generic speedups offered by
    ``dask='parallelized'``.

Chunking and performance
------------------------

The ``chunks`` parameter has critical performance implications when using dask
arrays. If your chunks are too small, queueing up operations will be extremely
slow, because dask will translates each operation into a huge number of
operations mapped across chunks. Computation on dask arrays with small chunks
can also be slow, because each operation on a chunk has some fixed overhead
from the Python interpreter and the dask task executor.

Conversely, if your chunks are too big, some of your computation may be wasted,
because dask only computes results one chunk at a time.

A good rule of thumb to create arrays with a minimum chunksize of at least one
million elements (e.g., a 1000x1000 matrix). With large arrays (10+ GB), the
cost of queueing up dask operations can be noticeable, and you may need even
larger chunksizes.

.. ipython:: python
    :suppress:

    import os
    os.remove('example-data.nc')

Optimization Tips
-----------------

With analysis pipelines involving both spatial subsetting and temporal resampling, dask performance can become very slow in certain cases. Here are some optimization tips we have found through experience:

1. Do your spatial and temporal indexing (e.g. ``.sel()`` or ``.isel()``) early in the pipeline, especially before calling ``resample()`` or ``groupby()``. Grouping and rasampling triggers some computation on all the blocks, which in theory should commute with indexing, but this optimization hasn't been implemented in dask yet. (See `dask issue #746 <https://github.com/dask/dask/issues/746>`_).

2. Save intermediate results to disk as a netCDF files (using ``to_netcdf()``) and then load them again with ``open_dataset()`` for further computations. For example, if subtracting temporal mean from a dataset, save the temporal mean to disk before subtracting. Again, in theory, dask should be able to do the computation in a streaming fashion, but in practice this is a fail case for the dask scheduler, because it tries to keep every chunk of an array that it computes in memory. (See `dask issue #874 <https://github.com/dask/dask/issues/874>`_)

3. Specify smaller chunks across space when using ``open_mfdataset()`` (e.g., ``chunks={'latitude': 10, 'longitude': 10}``). This makes spatial subsetting easier, because there's no risk you will load chunks of data referring to different chunks (probably not necessary if you follow suggestion 1).
