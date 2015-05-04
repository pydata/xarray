.. _dask:

Out of core computation with dask
=================================

xray v0.5 includes experimental integration with `dask <http://dask.pydata.org/>`__
to support streaming computation on datasets that don't fit into memory.

Currently, dask is an entirely optional feature for xray. However, the
benefits of using dask are sufficiently strong that we expect that dask may
become a requirement for a future version of xray.

What is a dask array?
---------------------

Dask divides arrays into many small pieces, called *chunks*, each of which is
presumed to be small enough to fit into memory. Unlike NumPy, which has eager
evaluation, operations on dask arrays are lazy. Operations queue up a series of
taks mapped over blocks, and no computation is performed until you
actually ask values to be computed (e.g., to print results to your screen or
write to disk). At that point, data is loaded into memory and computation proceeds
in a streaming fashion, block-by-block. The actual computation is controlled
by a multi-processing or thread pool, which allows dask to take full advantage
of multiple processers available on most modern computers.

Reading and writing data
------------------------

The usual way to create a dataset filled with dask arrays is to load the data
from a netCDF file or files, by supplying a ``chunks`` argument to
:py:func:`~xray.open_dataset` or using the :py:func:`~xray.open_mfdataset`
function.

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xray
    np.random.seed(123456)
    np.set_printoptions(precision=3, linewidth=100, threshold=100, edgeitems=3)

    ds = xray.Dataset({'temperature': (('time', 'latitude', 'longitude'),
                                       np.random.randn(365, 180, 360)),
                       'time': pd.date_range('2015-01-01', periods=365),
                       'longitude': np.arange(360),
                       'latitude': np.arange(89.5, -90.5, -1)})
    ds.to_netcdf('example-data.nc')


.. ipython:: python

    ds = xray.open_dataset('example-data.nc', chunks={'time': 10})
    ds

You'll notice that printing a dataset still shows a preview of array values,
even if they are actually dask arrays. We can do this quickly
with dask because we only need to the compute the first few values (typically
from the first block). To reveal the true nature of an array, print a DataArray:

.. ipython:: python

    ds.temperature

Once you've manipulated a dask array, you can still write a dataset too big to
fit into memory back to disk by using :py:meth:`~xray.Dataset.to_netcdf` in the
usual way.

Using dask with xray
--------------------

Nearly all existing xray methods (including those for indexing, computation,
concatenating and grouped operations) have been extended to work automatically
with dask arrays. When you load data as a dask array in an xray data
structure, almost all xray operations will keep it as a dask array; when this
is not possible, they will raise an exception rather than unexpectedly loading
data into memory. Converting a dask array into memory generally requires an
explicit conversion step. One noteable exception is indexing operations: to
enable label based indexing, xray will automatically load coordinate labels
into memory.

The easiest way to convert an xray data structure from lazy dask arrays into
eager, in-memory numpy arrays is to use the :py:meth:`~xray.Dataset.load` method:

.. ipython:: python

    ds.load()

You can also access :py:attr:`~xray.DataArray.values`, which will always be a
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

With the current versions of xray and dask, there is no automatic conversion
of eager numpy arrays to dask arrays, nor automatic alignment of chunks when
performing operations between dask arrays with different chunk sizes. You will
need to explicitly chunk each array to ensure compatibility. With xray, both
converting data to a dask arrays and converting the chunk sizes of dask arrays
is done with the :py:meth:`~xray.Dataset.chunk` method:

.. ipython:: python
    :suppress:

    ds = ds.chunk({'time': 10})

.. ipython:: python

    rechunked = ds.chunk({'latitude': 100, 'longitude': 100})

You can view the size of existing chunks on an array by viewing the
:py:attr:`~xray.Dataset.chunks` attribute:

.. ipython:: python

    rechunked.chunks

If there are not consistent chunksizes between all the ararys in a dataset
along a particular dimension, an exception is raised when you try to access
``.chunks``.

.. note::

    In the future, we would like to enable automatic alignment of dask
    chunksizes and automatic conversion of numpy arrays to dask (but not the
    other way around). We might also require that all arrays in a dataset
    share the same chunking alignment. None of these are currently done.

NumPy ufuncs like ``np.sin`` currently only work on eagerly evaluated arrays
(this will change with the next major NumPy release). We have provided
replacements that also work on all xray objects, including those that store
lazy dask arrays, in the ``xray.ufuncs`` module:

.. ipython:: python

    import xray.ufuncs as xu
    xu.sin(rechunked)

To access dask arrays directly, use the new
:py:attr:`DataArray.data <xray.DataArray.data>` attribute. This attribute exposes
array data either as a dask array or as a numpy array, depending on whether it has been
loaded into dask or not:

.. ipython:: python

    ds.temperature.data

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
