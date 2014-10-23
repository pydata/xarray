##############
Quick overview
##############

Here are some quick examples of what you can do with xray's
:py:class:`~xray.DataArray` object. Everything is explained in much more
detail in the rest of the documentation.

To begin, import numpy, pandas and xray:

.. ipython:: python

    import numpy as np
    import pandas as pd
    import xray

Create a DataArray
------------------

You can make a DataArray from scratch by supplying data in the form of a numpy
array or list, with optional *dimensions* and *coordinates*:

.. ipython:: python

   xray.DataArray(np.random.randn(2, 3))
   xray.DataArray(np.random.randn(2, 3), [('x', ['a', 'b']), ('y', [-2, 0, 2])])

You can also pass in pandas data structures directly:

.. ipython:: python

    df = pd.DataFrame(np.random.randn(2, 3), index=['a', 'b'], columns=[-2, 0, 2])
    df.index.name = 'x'
    df.columns.name = 'y'
    foo = xray.DataArray(df, name='foo')
    foo

Here are the key properties for a ``DataArray``:

.. ipython:: python

    # like in pandas, values is a numpy array that you can modify in-place
    foo.values
    foo.dims
    foo.coords['y']
    # you can use this dictionary to store arbitrary metadata
    foo.attrs

Indexing
--------

xray supports four kind of indexing. These operations are just as fast as in
pandas, because we borrow pandas' indexing machinery.

.. ipython:: python

    # positional and by integer label, like numpy
    foo[[0, 1]]

    # positional and by coordinate label, like pandas
    foo.loc['a':'b']

    # by dimension name and integer label
    foo.isel(x=slice(2))

    # by dimension name and coordinate label
    foo.sel(x=['a', 'b'])

Computation
-----------

Data arrays work very similarly to numpy ndarrays:

.. ipython:: python

    foo + 10
    np.sin(foo)
    foo.T
    foo.sum()

However, aggregation operations can use dimension names instead of axis
numbers:

.. ipython:: python

    foo.mean(dim='x')

Arithmetic operations broadcast based on dimension name, so you don't need to
insert dummy dimensions for alignment:

.. ipython:: python

    bar = xray.DataArray(np.random.randn(3), [foo.coords['y']])
    zzz = xray.DataArray(np.random.randn(4), dims='z')

    bar
    zzz

    bar + zzz

GroupBy
-------

xray supports grouped operations using a very similar API to pandas:

.. ipython:: python

    labels = xray.DataArray(['E', 'F', 'E'], [foo.coords['y']], name='labels')
    labels
    foo.groupby(labels).mean('y')
    foo.groupby(labels).apply(lambda x: x - x.min())

Convert to pandas
-----------------

A key feature of xray is robust conversion to and from pandas objects:

.. ipython:: python

    foo.to_series()
    foo.to_pandas()
