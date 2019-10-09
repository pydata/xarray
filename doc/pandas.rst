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

For datasets containing dask arrays where the data should be lazily loaded, see the
:py:meth:`Dataset.to_dask_dataframe() <xarray.Dataset.to_dask_dataframe>` method.

To create a ``Dataset`` from a ``DataFrame``, use the
:py:meth:`~xarray.Dataset.from_dataframe` class method or the equivalent
:py:meth:`pandas.DataFrame.to_xarray <DataFrame.to_xarray>` method:

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
    # or equivalently, with Series.to_xarray()
    xr.DataArray.from_series(s)

Both the ``from_series`` and ``from_dataframe`` methods use reindexing, so they
work even if not the hierarchical index is not a full tensor product:

.. ipython:: python

    s[::2]
    s[::2].to_xarray()

Multi-dimensional data
~~~~~~~~~~~~~~~~~~~~~~

Tidy data is great, but it sometimes you want to preserve dimensions instead of
automatically stacking them into a ``MultiIndex``.

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
array with the same shape, simply use the :py:class:`~xarray.DataArray`
constructor:

.. ipython:: python

    xr.DataArray(df)

Both the ``DataArray`` and ``Dataset`` constructors directly convert pandas
objects into xarray objects with the same shape. This means that they
preserve all use of multi-indexes:

.. ipython:: python

    index = pd.MultiIndex.from_arrays([['a', 'a', 'b'], [0, 1, 2]],
                                      names=['one', 'two'])
    df = pd.DataFrame({'x': 1, 'y': 2}, index=index)
    ds = xr.Dataset(df)
    ds

However, you will need to set dimension names explicitly, either with the
``dims`` argument on in the ``DataArray`` constructor or by calling
:py:class:`~xarray.Dataset.rename` on the new object.

.. _panel transition:

Transitioning from pandas.Panel to xarray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Panel``, pandas' data structure for 3D arrays, has always
been a second class data structure compared to the Series and DataFrame. To
allow pandas developers to focus more on its core functionality built around
the DataFrame, pandas has deprecated ``Panel``. It will be removed in pandas
0.25.

xarray has most of ``Panel``'s features, a more explicit API (particularly around
indexing), and the ability to scale to >3 dimensions with the same interface.

As discussed :ref:`elsewhere <data structures>` in the docs, there are two primary data structures in
xarray: ``DataArray`` and ``Dataset``. You can imagine a ``DataArray`` as a
n-dimensional pandas ``Series`` (i.e. a single typed array), and a ``Dataset``
as the ``DataFrame`` equivalent (i.e. a dict of aligned ``DataArray`` objects).

So you can represent a Panel, in two ways:

- As a 3-dimensional ``DataArray``,
- Or as a ``Dataset`` containing a number of 2-dimensional DataArray objects.

Let's take a look:

.. ipython:: python

    data = np.random.RandomState(0).rand(2, 3, 4)
    items = list('ab')
    major_axis = list('mno')
    minor_axis = pd.date_range(start='2000', periods=4, name='date')

With old versions of pandas (prior to 0.25), this could stored in a ``Panel``:

.. ipython::
    :verbatim:

    In [1]: pd.Panel(data, items, major_axis, minor_axis)
    Out[1]:
    <class 'pandas.core.panel.Panel'>
    Dimensions: 2 (items) x 3 (major_axis) x 4 (minor_axis)
    Items axis: a to b
    Major_axis axis: m to o
    Minor_axis axis: 2000-01-01 00:00:00 to 2000-01-04 00:00:00

To put this data in a ``DataArray``, write:

.. ipython:: python

    array = xr.DataArray(data, [items, major_axis, minor_axis])
    array

As you can see, there are three dimensions (each is also a coordinate). Two of
the axes of were unnamed, so have been assigned ``dim_0`` and ``dim_1``
respectively, while the third retains its name ``date``.

You can also easily convert this data into ``Dataset``:

.. ipython:: python

    array.to_dataset(dim='dim_0')

Here, there are two data variables, each representing a DataFrame on panel's
``items`` axis, and labelled as such. Each variable is a 2D array of the
respective values along the ``items`` dimension.

While the xarray docs are relatively complete, a few items stand out for Panel users:

- A DataArray's data is stored as a numpy array, and so can only contain a single
  type. As a result, a Panel that contains :py:class:`~pandas.DataFrame` objects
  with multiple types will be converted to ``dtype=object``. A ``Dataset`` of
  multiple ``DataArray`` objects each with its own dtype will allow original
  types to be preserved.
- :ref:`Indexing <indexing>` is similar to pandas, but more explicit and
  leverages xarray's naming of dimensions.
- Because of those features, making much higher dimensional data is very
  practical.
- Variables in ``Dataset`` objects can use a subset of its dimensions. For
  example, you can have one dataset with Person x Score x Time, and another with
  Person x Score.
- You can use coordinates are used for both dimensions and for variables which
  _label_ the data variables, so you could have a coordinate Age, that labelled
  the Person dimension of a Dataset of Person x Score x Time.

While xarray may take some getting used to, it's worth it! If anything is unclear,
please post an issue on `GitHub <https://github.com/pydata/xarray>`__ or
`StackOverflow <http://stackoverflow.com/questions/tagged/python-xarray>`__,
and we'll endeavor to respond to the specific case or improve the general docs.
