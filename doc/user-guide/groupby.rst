.. currentmodule:: xarray

.. _groupby:

GroupBy: Group and Bin Data
---------------------------

Often we want to bin or group data, produce statistics (mean, variance) on
the groups, and then return a reduced data set. To do this, Xarray supports
`"group by"`__ operations with the same API as pandas to implement the
`split-apply-combine`__ strategy:

__ https://pandas.pydata.org/pandas-docs/stable/groupby.html
__ https://www.jstatsoft.org/v40/i01/paper

- Split your data into multiple independent groups.
- Apply some function to each group.
- Combine your groups back into a single data object.

Group by operations work on both :py:class:`Dataset` and
:py:class:`DataArray` objects. Most of the examples focus on grouping by
a single one-dimensional variable, although support for grouping
over a multi-dimensional variable has recently been implemented. Note that for
one-dimensional data, it is usually faster to rely on pandas' implementation of
the same pipeline.

.. tip::

   `Install the flox package <https://flox.readthedocs.io>`_ to substantially improve the performance
   of GroupBy operations, particularly with dask. flox
   `extends Xarray's in-built GroupBy capabilities <https://flox.readthedocs.io/en/latest/xarray.html>`_
   by allowing grouping by multiple variables, and lazy grouping by dask arrays.
   If installed, Xarray will automatically use flox by default.

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

    ds = xr.Dataset(
        {"foo": (("x", "y"), np.random.rand(4, 3))},
        coords={"x": [10, 20, 30, 40], "letters": ("x", list("abba"))},
    )
    arr = ds["foo"]
    ds

If we groupby the name of a variable or coordinate in a dataset (we can also
use a DataArray directly), we get back a ``GroupBy`` object:

.. ipython:: python

    ds.groupby("letters")

This object works very similarly to a pandas GroupBy object. You can view
the group indices with the ``groups`` attribute:

.. ipython:: python

    ds.groupby("letters").groups

You can also iterate over groups in ``(label, group)`` pairs:

.. ipython:: python

    list(ds.groupby("letters"))

You can index out a particular group:

.. ipython:: python

    ds.groupby("letters")["b"]

To group by multiple variables, see :ref:`this section <groupby.multiple>`.

Binning
~~~~~~~

Sometimes you don't want to use all the unique values to determine the groups
but instead want to "bin" the data into coarser groups. You could always create
a customized coordinate, but xarray facilitates this via the
:py:meth:`Dataset.groupby_bins` method.

.. ipython:: python

    x_bins = [0, 25, 50]
    ds.groupby_bins("x", x_bins).groups

The binning is implemented via :func:`pandas.cut`, whose documentation details how
the bins are assigned. As seen in the example above, by default, the bins are
labeled with strings using set notation to precisely identify the bin limits. To
override this behavior, you can specify the bin labels explicitly. Here we
choose ``float`` labels which identify the bin centers:

.. ipython:: python

    x_bin_labels = [12.5, 37.5]
    ds.groupby_bins("x", x_bins, labels=x_bin_labels).groups


Apply
~~~~~

To apply a function to each group, you can use the flexible
:py:meth:`core.groupby.DatasetGroupBy.map` method. The resulting objects are automatically
concatenated back together along the group axis:

.. ipython:: python

    def standardize(x):
        return (x - x.mean()) / x.std()


    arr.groupby("letters").map(standardize)

GroupBy objects also have a :py:meth:`core.groupby.DatasetGroupBy.reduce` method and
methods like :py:meth:`core.groupby.DatasetGroupBy.mean` as shortcuts for applying an
aggregation function:

.. ipython:: python

    arr.groupby("letters").mean(dim="x")

Using a groupby is thus also a convenient shortcut for aggregating over all
dimensions *other than* the provided one:

.. ipython:: python

    ds.groupby("x").std(...)

.. note::

    We use an ellipsis (`...`) here to indicate we want to reduce over all
    other dimensions


First and last
~~~~~~~~~~~~~~

There are two special aggregation operations that are currently only found on
groupby objects: first and last. These provide the first or last example of
values for group along the grouped dimension:

.. ipython:: python

    ds.groupby("letters").first(...)

By default, they skip missing values (control this with ``skipna``).

Grouped arithmetic
~~~~~~~~~~~~~~~~~~

GroupBy objects also support a limited set of binary arithmetic operations, as
a shortcut for mapping over all unique labels. Binary arithmetic is supported
for ``(GroupBy, Dataset)`` and ``(GroupBy, DataArray)`` pairs, as long as the
dataset or data array uses the unique grouped values as one of its index
coordinates. For example:

.. ipython:: python

    alt = arr.groupby("letters").mean(...)
    alt
    ds.groupby("letters") - alt

This last line is roughly equivalent to the following::

    results = []
    for label, group in ds.groupby('letters'):
        results.append(group - alt.sel(letters=label))
    xr.concat(results, dim='x')

.. _groupby.multidim:

Multidimensional Grouping
~~~~~~~~~~~~~~~~~~~~~~~~~

Many datasets have a multidimensional coordinate variable (e.g. longitude)
which is different from the logical grid dimensions (e.g. nx, ny). Such
variables are valid under the `CF conventions`__. Xarray supports groupby
operations over multidimensional coordinate variables:

__ https://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#_two_dimensional_latitude_longitude_coordinate_variables

.. ipython:: python

    da = xr.DataArray(
        [[0, 1], [2, 3]],
        coords={
            "lon": (["ny", "nx"], [[30, 40], [40, 50]]),
            "lat": (["ny", "nx"], [[10, 10], [20, 20]]),
        },
        dims=["ny", "nx"],
    )
    da
    da.groupby("lon").sum(...)
    da.groupby("lon").map(lambda x: x - x.mean(), shortcut=False)

Because multidimensional groups have the ability to generate a very large
number of bins, coarse-binning via :py:meth:`Dataset.groupby_bins`
may be desirable:

.. ipython:: python

    da.groupby_bins("lon", [0, 45, 50]).sum()

These methods group by ``lon`` values. It is also possible to groupby each
cell in a grid, regardless of value, by stacking multiple dimensions,
applying your function, and then unstacking the result:

.. ipython:: python

    stacked = da.stack(gridcell=["ny", "nx"])
    stacked.groupby("gridcell").sum(...).unstack("gridcell")

Alternatively, you can groupby both ``lat`` and ``lon`` at the :ref:`same time <groupby.multiple>`.

.. _groupby.groupers:

Grouper Objects
~~~~~~~~~~~~~~~

Both ``groupby_bins`` and ``resample`` are specializations of the core ``groupby`` operation for binning,
and time resampling. Many problems demand more complex GroupBy application: for example, grouping by multiple
variables with a combination of categorical grouping, binning, and resampling; or more specializations like
spatial resampling; or more complex time grouping like special handling of seasons, or the ability to specify
custom seasons. To handle these use-cases and more, Xarray is evolving to providing an
extension point using ``Grouper`` objects.

.. tip::

   See the `grouper design`_ doc for more detail on the motivation and design ideas behind
   Grouper objects.

.. _grouper design: https://github.com/pydata/xarray/blob/main/design_notes/grouper_objects.md

For now Xarray provides three specialized Grouper objects:

1. :py:class:`groupers.UniqueGrouper` for categorical grouping
2. :py:class:`groupers.BinGrouper` for binned grouping
3. :py:class:`groupers.TimeResampler` for resampling along a datetime coordinate

These provide functionality identical to the existing ``groupby``, ``groupby_bins``, and ``resample`` methods.
That is,

.. code-block:: python

    ds.groupby("x")

is identical to

.. code-block:: python

    from xarray.groupers import UniqueGrouper

    ds.groupby(x=UniqueGrouper())


Similarly,

.. code-block:: python

    ds.groupby_bins("x", bins=bins)

is identical to

.. code-block:: python

    from xarray.groupers import BinGrouper

    ds.groupby(x=BinGrouper(bins))

and

.. code-block:: python

    ds.resample(time="ME")

is identical to

.. code-block:: python

    from xarray.groupers import TimeResampler

    ds.resample(time=TimeResampler("ME"))


The :py:class:`groupers.UniqueGrouper` accepts an optional ``labels`` kwarg that is not present
in :py:meth:`DataArray.groupby` or :py:meth:`Dataset.groupby`.
Specifying ``labels`` is required when grouping by a lazy array type (e.g. dask or cubed).
The ``labels`` are used to construct the output coordinate (say for a reduction), and aggregations
will only be run over the specified labels.
You may use ``labels`` to also specify the ordering of groups to be used during iteration.
The order will be preserved in the output.


.. _groupby.multiple:

Grouping by multiple variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use grouper objects to group by multiple dimensions:

.. ipython:: python

    from xarray.groupers import UniqueGrouper

    da.groupby(["lat", "lon"]).sum()

The above is sugar for using ``UniqueGrouper`` objects directly:

.. ipython:: python

    da.groupby(lat=UniqueGrouper(), lon=UniqueGrouper()).sum()


Different groupers can be combined to construct sophisticated GroupBy operations.

.. ipython:: python

    from xarray.groupers import BinGrouper

    ds.groupby(x=BinGrouper(bins=[5, 15, 25]), letters=UniqueGrouper()).sum()


Shuffling
~~~~~~~~~

Shuffling is a generalization of sorting a DataArray or Dataset by another DataArray, named ``label`` for example, that follows from the idea of grouping by ``label``.
Shuffling reorders the DataArray or the DataArrays in a Dataset such that all members of a group occur sequentially. For example,
Shuffle the object using either :py:class:`DatasetGroupBy` or :py:class:`DataArrayGroupBy` as appropriate.

.. ipython:: python

    da = xr.DataArray(
        dims="x",
        data=[1, 2, 3, 4, 5, 6],
        coords={"label": ("x", "a b c a b c".split(" "))},
    )
    da.groupby("label").shuffle_to_chunks()


For chunked array types (e.g. dask or cubed), shuffle may result in a more optimized communication pattern when compared to direct indexing by the appropriate indexer.
Shuffling also makes GroupBy operations on chunked arrays an embarrassingly parallel problem, and may significantly improve workloads that use :py:meth:`DatasetGroupBy.map` or :py:meth:`DataArrayGroupBy.map`.
