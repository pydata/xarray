.. _pandas:

===================
Working with pandas
===================

One of the most important features of xarray is the ability to convert to and
from :py:mod:`pandas` objects to interact with the rest of the PyData
ecosystem. For example, for plotting labeled data, we highly recommend
using the visualization `built in to pandas itself`__ or provided by the pandas
aware libraries such as `Seaborn`__.

__ http://pandas.pydata.org/pandas-docs/stable/visualization.html
__ http://stanford.edu/~mwaskom/software/seaborn/

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr
    np.random.seed(123456)

Hierarchical and tidy data
~~~~~~~~~~~~~~~~~~~~~~~~~~

Tabular data is easiest to work with when it meets the criteria for
`tidy data`__:

* Each column holds a different variable.
* Each rows holds a different observation.

__ http://www.jstatsoft.org/v59/i10/

In this "tidy data" format, we can represent any :py:class:`~xarray.Dataset` and
:py:class:`~xarray.DataArray` in terms of :py:class:`pandas.DataFrame` and
:py:class:`pandas.Series`, respectively (and vice-versa). The representation
works by flattening non-coordinates to 1D, and turning the tensor product of
coordinate indexes into a :py:class:`pandas.MultiIndex`.

Dataset and DataFrame
---------------------

To convert any dataset to a ``DataFrame`` in tidy form, use the
:py:meth:`Dataset.to_dataframe() <xarray.Dataset.to_dataframe>` method:

.. ipython:: python

    ds = xr.Dataset({'foo': (('x', 'y'), np.random.randn(2, 3))},
                     coords={'x': [10, 20], 'y': ['a', 'b', 'c'],
                             'along_x': ('x', np.random.randn(2)),
                             'scalar': 123})
    ds
    df = ds.to_dataframe()
    df

We see that each variable and coordinate in the Dataset is now a column in the
DataFrame, with the exception of indexes which are in the index.
To convert the ``DataFrame`` to any other convenient representation,
use ``DataFrame`` methods like :py:meth:`~pandas.DataFrame.reset_index`,
:py:meth:`~pandas.DataFrame.stack` and :py:meth:`~pandas.DataFrame.unstack`.

To create a ``Dataset`` from a ``DataFrame``, use the
:py:meth:`~xarray.Dataset.from_dataframe` class method:

.. ipython:: python

    xr.Dataset.from_dataframe(df)

Notice that that dimensions of variables in the ``Dataset`` have now
expanded after the round-trip conversion to a ``DataFrame``. This is because
every object in a ``DataFrame`` must have the same indices, so we need to
broadcast the data of each array to the full size of the new ``MultiIndex``.

Likewise, all the coordinates (other than indexes) ended up as variables,
because pandas does not distinguish non-index coordinates.

DataArray and Series
--------------------

``DataArray`` objects have a complementary representation in terms of a
:py:class:`pandas.Series`. Using a Series preserves the ``Dataset`` to
``DataArray`` relationship, because ``DataFrames`` are dict-like containers
of ``Series``. The methods are very similar to those for working with
DataFrames:

.. ipython:: python

    s = ds['foo'].to_series()
    s

    xr.DataArray.from_series(s)

Both the ``from_series`` and ``from_dataframe`` methods use reindexing, so they
work even if not the hierarchical index is not a full tensor product:

.. ipython:: python

    s[::2]
    xr.DataArray.from_series(s[::2])

Multi-dimensional data
~~~~~~~~~~~~~~~~~~~~~~

:py:meth:`DataArray.to_pandas() <xarray.DataArray.to_pandas>` is a shortcut that
lets you convert a DataArray directly into a pandas object with the same
dimensionality (i.e., a 1D array is converted to a :py:class:`~pandas.Series`,
2D to :py:class:`~pandas.DataFrame` and 3D to :py:class:`~pandas.Panel`):

.. ipython:: python

    arr = xr.DataArray(np.random.randn(2, 3),
                       coords=[('x', [10, 20]), ('y', ['a', 'b', 'c'])])
    df = arr.to_pandas()
    df

To perform the inverse operation of converting any pandas objects into a data
array with the same shape, simply use the ``DataArray`` constructor:

.. ipython:: python

    xr.DataArray(df)

xarray objects do not yet support hierarchical indexes, so if your data has
a hierarchical index, you will either need to unstack it first or use the
:py:meth:`~xarray.DataArray.from_series` or
:py:meth:`~xarray.Dataset.from_dataframe` constructors described above.
