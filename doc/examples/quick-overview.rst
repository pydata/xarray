##############
Quick overview
##############

Here are some quick examples of what you can do with :py:class:`xarray.DataArray`
objects. Everything is explained in much more detail in the rest of the
documentation.

To begin, import numpy, pandas and xarray using their customary abbreviations:

.. ipython:: python

    import numpy as np
    import pandas as pd
    import xarray as xr

Create a DataArray
------------------

You can make a DataArray from scratch by supplying data in the form of a numpy
array or list, with optional *dimensions* and *coordinates*:

.. ipython:: python

    xr.DataArray(np.random.randn(2, 3))
    data = xr.DataArray(np.random.randn(2, 3), [('x', ['a', 'b']), ('y', [-2, 0, 2])])
    data

If you supply a pandas :py:class:`~pandas.Series` or
:py:class:`~pandas.DataFrame`, metadata is copied directly:

.. ipython:: python

    xr.DataArray(pd.Series(range(3), index=list('abc'), name='foo'))

Here are the key properties for a ``DataArray``:

.. ipython:: python

    # like in pandas, values is a numpy array that you can modify in-place
    data.values
    data.dims
    data.coords
    # you can use this dictionary to store arbitrary metadata
    data.attrs

Indexing
--------

xarray supports four kind of indexing. These operations are just as fast as in
pandas, because we borrow pandas' indexing machinery.

.. ipython:: python

    # positional and by integer label, like numpy
    data[[0, 1]]

    # positional and by coordinate label, like pandas
    data.loc['a':'b']

    # by dimension name and integer label
    data.isel(x=slice(2))

    # by dimension name and coordinate label
    data.sel(x=['a', 'b'])

Computation
-----------

Data arrays work very similarly to numpy ndarrays:

.. ipython:: python

    data + 10
    np.sin(data)
    data.T
    data.sum()

However, aggregation operations can use dimension names instead of axis
numbers:

.. ipython:: python

    data.mean(dim='x')

Arithmetic operations broadcast based on dimension name. This means you don't
need to insert dummy dimensions for alignment:

.. ipython:: python

    a = xr.DataArray(np.random.randn(3), [data.coords['y']])
    b = xr.DataArray(np.random.randn(4), dims='z')

    a
    b

    a + b

It also means that in most cases you do not need to worry about the order of
dimensions:

.. ipython:: python

    data - data.T

Operations also align based on index labels:

.. ipython:: python

    data[:-1] - data[:1]

GroupBy
-------

xarray supports grouped operations using a very similar API to pandas:

.. ipython:: python

    labels = xr.DataArray(['E', 'F', 'E'], [data.coords['y']], name='labels')
    labels
    data.groupby(labels).mean('y')
    data.groupby(labels).apply(lambda x: x - x.min())

Convert to pandas
-----------------

A key feature of xarray is robust conversion to and from pandas objects:

.. ipython:: python

    data.to_series()
    data.to_pandas()

Datasets and NetCDF
-------------------

:py:class:`xarray.Dataset` is a dict-like container of ``DataArray`` objects that share
index labels and dimensions. It looks a lot like a netCDF file:

.. ipython:: python

    ds = data.to_dataset(name='foo')
    ds

You can do almost everything you can do with ``DataArray`` objects with
``Dataset`` objects if you prefer to work with multiple variables at once.

Datasets also let you easily read and write netCDF files:

.. ipython:: python

    ds.to_netcdf('example.nc')
    xr.open_dataset('example.nc')

.. ipython:: python
   :suppress:

    import os
    os.remove('example.nc')
