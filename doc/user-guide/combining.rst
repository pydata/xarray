.. _combining data:

Combining data
--------------

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)

    %xmode minimal

* For combining datasets or data arrays along a single dimension, see concatenate_.
* For combining datasets with different variables, see merge_.
* For combining datasets or data arrays with different indexes or missing values, see combine_.
* For combining datasets or data arrays along multiple dimensions see combining.multi_.

.. _concatenate:

Concatenate
~~~~~~~~~~~

To combine :py:class:`~xarray.Dataset` / :py:class:`~xarray.DataArray` objects along an existing or new dimension
into a larger object, you can use :py:func:`~xarray.concat`. ``concat``
takes an iterable of ``DataArray`` or ``Dataset`` objects, as well as a
dimension name, and concatenates along that dimension:

.. jupyter-execute::

    da = xr.DataArray(
        np.arange(6).reshape(2, 3), [("x", ["a", "b"]), ("y", [10, 20, 30])]
    )
    da.isel(y=slice(0, 1))  # same as da[:, :1]

.. jupyter-execute::

    # This resembles how you would use np.concatenate:
    xr.concat([da[:, :1], da[:, 1:]], dim="y")

.. jupyter-execute::

    # For more friendly pandas-like indexing you can use:
    xr.concat([da.isel(y=slice(0, 1)), da.isel(y=slice(1, None))], dim="y")

In addition to combining along an existing dimension, ``concat`` can create a
new dimension by stacking lower dimensional arrays together:

.. jupyter-execute::

    da.sel(x="a")

.. jupyter-execute::

    xr.concat([da.isel(x=0), da.isel(x=1)], "x")

If the second argument to ``concat`` is a new dimension name, the arrays will
be concatenated along that new dimension, which is always inserted as the first
dimension:

.. jupyter-execute::

    da0 = da.isel(x=0, drop=True)
    da1 = da.isel(x=1, drop=True)

    xr.concat([da0, da1], "new_dim")

The second argument to ``concat`` can also be an :py:class:`~pandas.Index` or
:py:class:`~xarray.DataArray` object as well as a string, in which case it is
used to label the values along the new dimension:

.. jupyter-execute::

    xr.concat([da0, da1], pd.Index([-90, -100], name="new_dim"))

Of course, ``concat`` also works on ``Dataset`` objects:

.. jupyter-execute::

    ds = da.to_dataset(name="foo")
    xr.concat([ds.sel(x="a"), ds.sel(x="b")], "x")

:py:func:`~xarray.concat` has a number of options which provide deeper control
over which variables are concatenated and how it handles conflicting variables
between datasets. With the default parameters, xarray will load some coordinate
variables into memory to compare them between datasets. This may be prohibitively
expensive if you are manipulating your dataset lazily using :ref:`dask`.

.. note::

   In a future version of xarray the default values for many of these options
   will change. You can opt into the new default values early using
   ``xr.set_options(use_new_combine_kwarg_defaults=True)``.

.. _merge:

Merge
~~~~~

To combine variables and coordinates between multiple ``DataArray`` and/or
``Dataset`` objects, use :py:func:`~xarray.merge`. It can merge a list of
``Dataset``, ``DataArray`` or dictionaries of objects convertible to
``DataArray`` objects:

.. jupyter-execute::

    xr.merge([ds, ds.rename({"foo": "bar"})])

.. jupyter-execute::

    xr.merge([xr.DataArray(n, name="var%d" % n) for n in range(5)])

If you merge another dataset (or a dictionary including data array objects), by
default the resulting dataset will be aligned on the **union** of all index
coordinates:

.. note::

   In a future version of xarray the default value for ``join`` and ``compat``
   will change. This change will mean that xarray will no longer attempt
   to align the indices of the merged dataset. You can opt into the new default
   values early using ``xr.set_options(use_new_combine_kwarg_defaults=True)``.
   Or explicitly set ``join='outer'`` to preserve old behavior.

.. jupyter-execute::

    other = xr.Dataset({"bar": ("x", [1, 2, 3, 4]), "x": list("abcd")})
    xr.merge([ds, other], join="outer")

This ensures that ``merge`` is non-destructive. ``xarray.MergeError`` is raised
if you attempt to merge two variables with the same name but different values:

.. jupyter-execute::
    :raises:

    xr.merge([ds, ds + 1])


.. note::

    In a future version of xarray the default value for ``compat`` will change
    from ``compat='no_conflicts'`` to ``compat='override'``. In this scenario
    the values in the first object override all the values in other objects.

    .. jupyter-execute::

        xr.merge([ds, ds + 1], compat="override")

The same non-destructive merging between ``DataArray`` index coordinates is
used in the :py:class:`~xarray.Dataset` constructor:

.. jupyter-execute::

    xr.Dataset({"a": da.isel(x=slice(0, 1)), "b": da.isel(x=slice(1, 2))})

.. _combine:

Combine
~~~~~~~

The instance method :py:meth:`~xarray.DataArray.combine_first` combines two
datasets/data arrays and defaults to non-null values in the calling object,
using values from the called object to fill holes.  The resulting coordinates
are the union of coordinate labels. Vacant cells as a result of the outer-join
are filled with ``NaN``. For example:

.. jupyter-execute::

    ar0 = xr.DataArray([[0, 0], [0, 0]], [("x", ["a", "b"]), ("y", [-1, 0])])
    ar1 = xr.DataArray([[1, 1], [1, 1]], [("x", ["b", "c"]), ("y", [0, 1])])
    ar0.combine_first(ar1)

.. jupyter-execute::

    ar1.combine_first(ar0)

For datasets, ``ds0.combine_first(ds1)`` works similarly to
``xr.merge([ds0, ds1])``, except that ``xr.merge`` raises ``MergeError`` when
there are conflicting values in variables to be merged, whereas
``.combine_first`` defaults to the calling object's values.

.. note::

   In a future version of xarray the default options for ``xr.merge`` will change
   such that the behavior matches ``combine_first``.

.. _update:

Update
~~~~~~

In contrast to ``merge``, :py:meth:`~xarray.Dataset.update` modifies a dataset
in-place without checking for conflicts, and will overwrite any existing
variables with new values:

.. jupyter-execute::

    ds.update({"space": ("space", [10.2, 9.4, 3.9])})

However, dimensions are still required to be consistent between different
Dataset variables, so you cannot change the size of a dimension unless you
replace all dataset variables that use it.

``update`` also performs automatic alignment if necessary. Unlike ``merge``, it
maintains the alignment of the original array instead of merging indexes:

.. jupyter-execute::

    ds.update(other)

The exact same alignment logic when setting a variable with ``__setitem__``
syntax:

.. jupyter-execute::

    ds["baz"] = xr.DataArray([9, 9, 9, 9, 9], coords=[("x", list("abcde"))])
    ds.baz

Equals and identical
~~~~~~~~~~~~~~~~~~~~

Xarray objects can be compared by using the :py:meth:`~xarray.Dataset.equals`,
:py:meth:`~xarray.Dataset.identical` and
:py:meth:`~xarray.Dataset.broadcast_equals` methods. These methods are used by
the optional ``compat`` argument on ``concat`` and ``merge``.

:py:attr:`~xarray.Dataset.equals` checks dimension names, indexes and array
values:

.. jupyter-execute::

    da.equals(da.copy())

:py:attr:`~xarray.Dataset.identical` also checks attributes, and the name of each
object:

.. jupyter-execute::

    da.identical(da.rename("bar"))

:py:attr:`~xarray.Dataset.broadcast_equals` does a more relaxed form of equality
check that allows variables to have different dimensions, as long as values
are constant along those new dimensions:

.. jupyter-execute::

    left = xr.Dataset(coords={"x": 0})
    right = xr.Dataset({"x": [0, 0, 0]})
    left.broadcast_equals(right)

Like pandas objects, two xarray objects are still equal or identical if they have
missing values marked by ``NaN`` in the same locations.

In contrast, the ``==`` operation performs element-wise comparison (like
numpy):

.. jupyter-execute::

    da == da.copy()

Note that ``NaN`` does not compare equal to ``NaN`` in element-wise comparison;
you may need to deal with missing values explicitly.

.. _combining.no_conflicts:

Merging with 'no_conflicts'
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``compat`` argument ``'no_conflicts'`` is only available when
combining xarray objects with ``merge``. In addition to the above comparison
methods it allows the merging of xarray objects with locations where *either*
have ``NaN`` values. This can be used to combine data with overlapping
coordinates as long as any non-missing values agree or are disjoint:

.. jupyter-execute::

    ds1 = xr.Dataset({"a": ("x", [10, 20, 30, np.nan])}, {"x": [1, 2, 3, 4]})
    ds2 = xr.Dataset({"a": ("x", [np.nan, 30, 40, 50])}, {"x": [2, 3, 4, 5]})
    xr.merge([ds1, ds2], join="outer", compat="no_conflicts")

Note that due to the underlying representation of missing values as floating
point numbers (``NaN``), variable data type is not always preserved when merging
in this manner.

.. _combining.multi:

Combining along multiple dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For combining many objects along multiple dimensions xarray provides
:py:func:`~xarray.combine_nested` and :py:func:`~xarray.combine_by_coords`. These
functions use a combination of ``concat`` and ``merge`` across different
variables to combine many objects into one.

:py:func:`~xarray.combine_nested` requires specifying the order in which the
objects should be combined, while :py:func:`~xarray.combine_by_coords` attempts to
infer this ordering automatically from the coordinates in the data.

:py:func:`~xarray.combine_nested` is useful when you know the spatial
relationship between each object in advance. The datasets must be provided in
the form of a nested list, which specifies their relative position and
ordering. A common task is collecting data from a parallelized simulation where
each processor wrote out data to a separate file. A domain which was decomposed
into 4 parts, 2 each along both the x and y axes, requires organising the
datasets into a doubly-nested list, e.g:

.. jupyter-execute::

    arr = xr.DataArray(
        name="temperature", data=np.random.randint(5, size=(2, 2)), dims=["x", "y"]
    )
    arr

.. jupyter-execute::

    ds_grid = [[arr, arr], [arr, arr]]
    xr.combine_nested(ds_grid, concat_dim=["x", "y"])

:py:func:`~xarray.combine_nested` can also be used to explicitly merge datasets
with different variables. For example if we have 4 datasets, which are divided
along two times, and contain two different variables, we can pass ``None``
to ``'concat_dim'`` to specify the dimension of the nested list over which
we wish to use ``merge`` instead of ``concat``:

.. jupyter-execute::

    temp = xr.DataArray(name="temperature", data=np.random.randn(2), dims=["t"])
    precip = xr.DataArray(name="precipitation", data=np.random.randn(2), dims=["t"])
    ds_grid = [[temp, precip], [temp, precip]]
    xr.combine_nested(ds_grid, concat_dim=["t", None])

:py:func:`~xarray.combine_by_coords` is for combining objects which have dimension
coordinates which specify their relationship to and order relative to one
another, for example a linearly-increasing 'time' dimension coordinate.

Here we combine two datasets using their common dimension coordinates. Notice
they are concatenated in order based on the values in their dimension
coordinates, not on their position in the list passed to ``combine_by_coords``.

.. jupyter-execute::

    x1 = xr.DataArray(name="foo", data=np.random.randn(3), coords=[("x", [0, 1, 2])])
    x2 = xr.DataArray(name="foo", data=np.random.randn(3), coords=[("x", [3, 4, 5])])
    xr.combine_by_coords([x2, x1])

These functions are used by :py:func:`~xarray.open_mfdataset` to open many
files as one dataset. The particular function used is specified by setting the
argument ``'combine'`` to ``'by_coords'`` or ``'nested'``. This is useful for
situations where your data is split across many files in multiple locations,
which have some known relationship between one another.
