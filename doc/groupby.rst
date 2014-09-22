.. _groupby:

GroupBy: split-apply-combine
----------------------------

xray supports `"group by"`__ operations with the same API as pandas to
implement the `split-apply-combine`__ strategy:

__ http://pandas.pydata.org/pandas-docs/stable/groupby.html
__ http://www.jstatsoft.org/v40/i01/paper

- Split your data into multiple independent groups.
- Apply some function to each group.
- Combine your groups back into a single data object.

Group by operations work on both :py:class:`~xray.Dataset` and
:py:class:`~xray.DataArray` objects. Currently, you can only group by a single
one-dimensional variable (eventually, we hope to remove this limitation).

Split
~~~~~

Let's create a simple example dataset:

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xray
    np.random.seed(123456)

.. ipython:: python

    ds = xray.Dataset({'foo': (('x', 'y'), np.random.rand(4, 3))},
                      coords={'x': [10, 20, 30, 40],
                              'letters': ('x', list('abba'))})
    arr = ds['foo']
    ds

If we groupby the name of a variable or coordinate in a dataset (we can also
use a DataArray directly), we get back a :py:class:`xray.GroupBy` object:

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
:py:meth:`xray.GroupBy.apply` method. The resulting objects are automatically
concatenated back together along the group axis:

.. ipython:: python

    def standardize(x):
        return (x - x.mean()) / x.std()

    arr.groupby('letters').apply(standardize)

GroupBy objects also have a :py:meth:`~xray.GroupBy.reduce` method and
methods like :py:meth:`~xray.GroupBy.mean` as shortcuts for applying an
aggregation function:

.. ipython:: python

    arr.groupby('letters').mean(dim='x')

Using a groupby is thus also a convenient shortcut for aggregating over all
dimensions *other than* the provided one:

.. ipython:: python

    ds.groupby('x').reduce(np.nanmean)

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
        results.append(group - alt.sel(label))
    xray.concat(results, dim='letters')

Squeezing
~~~~~~~~~

When grouping over a dimension, you can control whether the dimension is
squeezed out or if it should remain with length one on each group by using
the ``squeeze`` parameter:

.. ipython:: python

    next(iter(arr.groupby('x')))

.. ipython:: python

    next(iter(arr.groupby('x', squeeze=False)))

Although xray will attempt to automatically
:py:attr:`~xray.DataArray.transpose` dimensions back into their original order
when you use apply, it is sometimes useful to set ``squeeze=False`` to
guarantee that all original dimensions remain unchanged.

You can always squeeze explicitly later with the Dataset or DataArray
:py:meth:`~xray.DataArray.squeeze` methods.
