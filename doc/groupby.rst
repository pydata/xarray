.. _groupby:

GroupBy: split-apply-combine
----------------------------

xarray supports `"group by"`__ operations with the same API as pandas to
implement the `split-apply-combine`__ strategy:

__ http://pandas.pydata.org/pandas-docs/stable/groupby.html
__ http://www.jstatsoft.org/v40/i01/paper

- Split your data into multiple independent groups.
- Apply some function to each group.
- Combine your groups back into a single data object.

Group by operations work on both :py:class:`~xarray.Dataset` and
:py:class:`~xarray.DataArray` objects. Currently, you can only group by a single
one-dimensional variable (eventually, we hope to remove this limitation). Also,
note that for one-dimensional data, it is usually faster to rely on pandas'
implementation of the same pipeline.

Split
~~~~~

Let's create a simple example dataset:

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr
    np.random.seed(123456)

.. ipython:: python

    ds = xr.Dataset({'foo': (('x', 'y'), np.random.rand(4, 3))},
                    coords={'x': [10, 20, 30, 40],
                            'letters': ('x', list('abba'))})
    arr = ds['foo']
    ds

If we groupby the name of a variable or coordinate in a dataset (we can also
use a DataArray directly), we get back a ``GroupBy`` object:

.. ipython:: python

    ds.groupby('letters')

This object works very similarly to a pandas GroupBy object. You can view
the group indices with the ``groups`` attribute:

.. ipython:: python

    ds.groupby('letters').groups

You can also iterate over over groups in ``(label, group)`` pairs:

.. ipython:: python

    list(ds.groupby('letters'))

Just like in pandas, creating a GroupBy object is cheap: it does not actually
split the data until you access particular values.

Apply
~~~~~

To apply a function to each group, you can use the flexible
:py:meth:`~xarray.DatasetGroupBy.apply` method. The resulting objects are automatically
concatenated back together along the group axis:

.. ipython:: python

    def standardize(x):
        return (x - x.mean()) / x.std()

    arr.groupby('letters').apply(standardize)

GroupBy objects also have a :py:meth:`~xarray.DatasetGroupBy.reduce` method and
methods like :py:meth:`~xarray.DatasetGroupBy.mean` as shortcuts for applying an
aggregation function:

.. ipython:: python

    arr.groupby('letters').mean(dim='x')

Using a groupby is thus also a convenient shortcut for aggregating over all
dimensions *other than* the provided one:

.. ipython:: python

    ds.groupby('x').std()

First and last
~~~~~~~~~~~~~~

There are two special aggregation operations that are currently only found on
groupby objects: first and last. These provide the first or last example of
values for group along the grouped dimension:

.. ipython:: python

    ds.groupby('letters').first()

By default, they skip missing values (control this with ``skipna``).

Grouped arithmetic
~~~~~~~~~~~~~~~~~~

GroupBy objects also support a limited set of binary arithmetic operations, as
a shortcut for mapping over all unique labels. Binary arithmetic is supported
for ``(GroupBy, Dataset)`` and ``(GroupBy, DataArray)`` pairs, as long as the
dataset or data array uses the unique grouped values as one of its index
coordinates. For example:

.. ipython:: python

    alt = arr.groupby('letters').mean()
    alt
    ds.groupby('letters') - alt

This last line is roughly equivalent to the following::

    results = []
    for label, group in ds.groupby('letters'):
        results.append(group - alt.sel(x=label))
    xr.concat(results, dim='x')

Squeezing
~~~~~~~~~

When grouping over a dimension, you can control whether the dimension is
squeezed out or if it should remain with length one on each group by using
the ``squeeze`` parameter:

.. ipython:: python

    next(iter(arr.groupby('x')))

.. ipython:: python

    next(iter(arr.groupby('x', squeeze=False)))

Although xarray will attempt to automatically
:py:attr:`~xarray.DataArray.transpose` dimensions back into their original order
when you use apply, it is sometimes useful to set ``squeeze=False`` to
guarantee that all original dimensions remain unchanged.

You can always squeeze explicitly later with the Dataset or DataArray
:py:meth:`~xarray.DataArray.squeeze` methods.
