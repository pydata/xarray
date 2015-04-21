.. _dask:

Out of core computation with dask
=================================

xray v0.5 includes experimental integration with `dask <http://dask.pydata.org/>`__
to support streaming computation on datasets that don't fit into memory.

TOOD: briefly summarize the dask data model.

To create a dataset filled with dask arrays, either use :py:func:`~xray.open_mfdataset`
to open a collection of files from disk or the :py:meth:`~xray.Dataset.chunk_data`
method to convert an existing dataset.

In the dask computation work, actual calculations are only performed on
demand, when computed data is requested. Printing a ``Dataset`` will show the
first few computed values. To load a dataset from dask array entirely into memory as
numpy arrays, use the :py:meth:`~xray.Dataset.load_data` method. You can also write
datasets too big to fit into memory directly to disk with
:py:meth:`~xray.Dataset.to_netcdf`.

To access dask arrays directly, use the new
:py:attr:`DataArray.data <~xray.DataArray.data>` attribute. This attribute exposes
array data either as a dask array or as a numpy array, if it hasn't been loaded into
dask.

Nearly all existing xray methods (including those for indexing, computation,
concatenating and grouped operations) have been extended to work automatically
with dask arrays. However, there are a few caveats:

1. Operations between dask arrays with different ``chunks`` or between dask arrays
   and numpy arrays are not currently supported by dask. You'll need to use ``chunk_data``
   to manual coerce input into dask arrays with the same ``chunks``.
2. NumPy ufuncs like ``np.sin`` currently only work on eagerly evaluated arrays. We've
   provided replacements that also work on all xray objects, including those that
   store lazy dask arrays, in the ``xray.ufuncs`` module::

	   import xray.ufuncs as xu
	   xu.sin(my_lazy_dataset)
