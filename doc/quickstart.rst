##########
Quickstart
##########

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

For more details, see :ref:`data structures`.

From scratch
~~~~~~~~~~~~

.. ipython:: python

   xray.DataArray(np.random.randn(2, 3))
   xray.DataArray(np.random.randn(2, 3), dims=['x', 'y'])
   xray.DataArray(np.random.randn(2, 3), [('x', ['a', 'b']), ('y', [-2, 0, 2])])

From pandas
~~~~~~~~~~~

.. ipython:: python

    df = pd.DataFrame(np.random.randn(2, 3), index=['a', 'b'], columns=[-2, 0, 2])
    df.index.name = 'x'
    df.columns.name = 'y'
    df
    foo = xray.DataArray(df, name='foo')
    foo

Properties
----------

.. ipython:: python

    foo.values
    foo.dims
    foo.coords['y']
    foo.attrs

Indexing
--------

For more details, see :ref:`indexing`.

Like numpy
~~~~~~~~~~

.. ipython:: python

    foo[[0, 1], 0]

Like pandas
~~~~~~~~~~~

.. ipython:: python

    foo.loc['a':'b', -2]

By dimension name and integer label
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python

    foo.isel(x=slice(2))

By dimension name and coordinate label
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python

    foo.sel(x=['a', 'b'])

Computation
-----------

For more details, see :ref:`comput`.

Unary operations
~~~~~~~~~~~~~~~~

.. ipython:: python

    foo.sum()
    foo.mean(dim=['x'])
    foo + 10
    np.sin(10)
    foo.T

Binary operations
~~~~~~~~~~~~~~~~~

.. ipython:: python

    bar = xray.DataArray(np.random.randn(3), [foo.coords['y']])
    zzz = xray.DataArray(np.random.randn(4), dims='z')

    bar
    zzz

    bar + zzz
    foo / bar

GroupBy
-------

For more details, see :ref:`groupby`.

.. ipython:: python

    labels = xray.DataArray(['E', 'F', 'E'], [foo.coords['y']], name='labels')
    labels
    foo.groupby(labels).mean('y')
    foo.groupby(labels).apply(lambda x: x.max() - x.min())

Convert to pandas
-----------------

For more details, see :ref:`pandas`.

.. ipython:: python

    foo.to_dataframe()
    foo.to_series()
