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
:py:class:`~xarray.DataArray` objects. Most of the examples focus on grouping by
a single one-dimensional variable, although support for grouping
over a multi-dimensional variable has recently been implemented. Note that for
one-dimensional data, it is usually faster to rely on pandas' implementation of
the same pipeline.

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

You can also iterate over groups in ``(label, group)`` pairs:

.. ipython:: python

    list(ds.groupby('letters'))

Just like in pandas, creating a GroupBy object is cheap: it does not actually
split the data until you access particular values.

Binning
~~~~~~~

Sometimes you don't want to use all the unique values to determine the groups
but instead want to "bin" the data into coarser groups. You could always create
a customized coordinate, but xarray facilitates this via the
:py:meth:`~xarray.Dataset.groupby_bins` method.

.. ipython:: python

    x_bins = [0,25,50]
    ds.groupby_bins('x', x_bins).groups

The binning is implemented via :func:`pandas.cut`, whose documentation details how
the bins are assigned. As seen in the example above, by default, the bins are
labeled with strings using set notation to precisely identify the bin limits. To
override this behavior, you can specify the bin labels explicitly. Here we
choose `float` labels which identify the bin centers:

.. ipython:: python

    x_bin_labels = [12.5,37.5]
    ds.groupby_bins('x', x_bins, labels=x_bin_labels).groups


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

    ds.groupby('x').std(xr.ALL_DIMS)

First and last
~~~~~~~~~~~~~~

There are two special aggregation operations that are currently only found on
groupby objects: first and last. These provide the first or last example of
values for group along the grouped dimension:

.. ipython:: python

    ds.groupby('letters').first(xr.ALL_DIMS)

By default, they skip missing values (control this with ``skipna``).

Grouped arithmetic
~~~~~~~~~~~~~~~~~~

GroupBy objects also support a limited set of binary arithmetic operations, as
a shortcut for mapping over all unique labels. Binary arithmetic is supported
for ``(GroupBy, Dataset)`` and ``(GroupBy, DataArray)`` pairs, as long as the
dataset or data array uses the unique grouped values as one of its index
coordinates. For example:

.. ipython:: python

    alt = arr.groupby('letters').mean(xr.ALL_DIMS)
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

.. _groupby.multidim:

Multidimensional Grouping
~~~~~~~~~~~~~~~~~~~~~~~~~

Many datasets have a multidimensional coordinate variable (e.g. longitude)
which is different from the logical grid dimensions (e.g. nx, ny). Such
variables are valid under the `CF conventions`__. Xarray supports groupby
operations over multidimensional coordinate variables:

__ http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#_two_dimensional_latitude_longitude_coordinate_variables

.. ipython:: python

    da = xr.DataArray([[0,1],[2,3]],
        coords={'lon': (['ny','nx'], [[30,40],[40,50]] ),
                'lat': (['ny','nx'], [[10,10],[20,20]] ),},
        dims=['ny','nx'])
    da
    da.groupby('lon').sum(xr.ALL_DIMS)
    da.groupby('lon').apply(lambda x: x - x.mean(), shortcut=False)

Because multidimensional groups have the ability to generate a very large
number of bins, coarse-binning via :py:meth:`~xarray.Dataset.groupby_bins`
may be desirable:

.. ipython:: python

    da.groupby_bins('lon', [0,45,50]).sum()

These methods group by `lon` values. It is also possible to groupby each
cell in a grid, regardless of value, by stacking multiple dimensions, 
applying your function, and then unstacking the result:

.. ipython:: python

   stacked = da.stack(gridcell=['ny', 'nx'])
   stacked.groupby('gridcell').sum(xr.ALL_DIMS).unstack('gridcell')
