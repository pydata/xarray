.. _combining data:

Combining data
--------------

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr
    np.random.seed(123456)

* For combining datasets or data arrays along a dimension, see concatenate_.
* For combining datasets with different variables, see merge_.
* For combining datasets or data arrays with different indexes or missing values, see combine_.

.. _concatenate:

Concatenate
~~~~~~~~~~~

To combine arrays along existing or new dimension into a larger array, you
can use :py:func:`~xarray.concat`. ``concat`` takes an iterable of ``DataArray``
or ``Dataset`` objects, as well as a dimension name, and concatenates along
that dimension:

.. ipython:: python

    arr = xr.DataArray(np.random.randn(2, 3),
                       [('x', ['a', 'b']), ('y', [10, 20, 30])])
    arr[:, :1]
    # this resembles how you would use np.concatenate
    xr.concat([arr[:, :1], arr[:, 1:]], dim='y')

In addition to combining along an existing dimension, ``concat`` can create a
new dimension by stacking lower dimensional arrays together:

.. ipython:: python

    arr[0]
    # to combine these 1d arrays into a 2d array in numpy, you would use np.array
    xr.concat([arr[0], arr[1]], 'x')

If the second argument to ``concat`` is a new dimension name, the arrays will
be concatenated along that new dimension, which is always inserted as the first
dimension:

.. ipython:: python

    xr.concat([arr[0], arr[1]], 'new_dim')

The second argument to ``concat`` can also be an :py:class:`~pandas.Index` or
:py:class:`~xarray.DataArray` object as well as a string, in which case it is
used to label the values along the new dimension:

.. ipython:: python

    xr.concat([arr[0], arr[1]], pd.Index([-90, -100], name='new_dim'))

Of course, ``concat`` also works on ``Dataset`` objects:

.. ipython:: python

    ds = arr.to_dataset(name='foo')
    xr.concat([ds.sel(x='a'), ds.sel(x='b')], 'x')

:py:func:`~xarray.concat` has a number of options which provide deeper control
over which variables are concatenated and how it handles conflicting variables
between datasets. With the default parameters, xarray will load some coordinate
variables into memory to compare them between datasets. This may be prohibitively
expensive if you are manipulating your dataset lazily using :ref:`dask`.

.. _merge:

Merge
~~~~~

To combine variables and coordinates between multiple ``DataArray`` and/or
``Dataset`` object, use :py:func:`~xarray.merge`. It can merge a list of
``Dataset``, ``DataArray`` or dictionaries of objects convertible to
``DataArray`` objects:

.. ipython:: python

    xr.merge([ds, ds.rename({'foo': 'bar'})])
    xr.merge([xr.DataArray(n, name='var%d' % n) for n in range(5)])

If you merge another dataset (or a dictionary including data array objects), by
default the resulting dataset will be aligned on the **union** of all index
coordinates:

.. ipython:: python

    other = xr.Dataset({'bar': ('x', [1, 2, 3, 4]), 'x': list('abcd')})
    xr.merge([ds, other])

This ensures that ``merge`` is non-destructive. ``xarray.MergeError`` is raised
if you attempt to merge two variables with the same name but different values:

.. ipython::

    @verbatim
    In [1]: xr.merge([ds, ds + 1])
    MergeError: conflicting values for variable 'foo' on objects to be combined:
    first value: <xarray.Variable (x: 2, y: 3)>
    array([[ 0.4691123 , -0.28286334, -1.5090585 ],
           [-1.13563237,  1.21211203, -0.17321465]])
    second value: <xarray.Variable (x: 2, y: 3)>
    array([[ 1.4691123 ,  0.71713666, -0.5090585 ],
           [-0.13563237,  2.21211203,  0.82678535]])

The same non-destructive merging between ``DataArray`` index coordinates is
used in the :py:class:`~xarray.Dataset` constructor:

.. ipython:: python

    xr.Dataset({'a': arr[:-1], 'b': arr[1:]})

.. _combine:

Combine
~~~~~~~

The instance method :py:meth:`~xarray.DataArray.combine_first` combines two
datasets/data arrays and defaults to non-null values in the calling object,
using values from the called object to fill holes.  The resulting coordinates
are the union of coordinate labels. Vacant cells as a result of the outer-join
are filled with ``NaN``. For example:

.. ipython:: python

    ar0 = xr.DataArray([[0, 0], [0, 0]], [('x', ['a', 'b']), ('y', [-1, 0])])
    ar1 = xr.DataArray([[1, 1], [1, 1]], [('x', ['b', 'c']), ('y', [0, 1])])
    ar0.combine_first(ar1)
    ar1.combine_first(ar0)

For datasets, ``ds0.combine_first(ds1)`` works similarly to
``xr.merge([ds0, ds1])``, except that ``xr.merge`` raises ``MergeError`` when
there are conflicting values in variables to be merged, whereas
``.combine_first`` defaults to the calling object's values.

.. _update:

Update
~~~~~~

In contrast to ``merge``, :py:meth:`~xarray.Dataset.update` modifies a dataset
in-place without checking for conflicts, and will overwrite any existing
variables with new values:

.. ipython:: python

    ds.update({'space': ('space', [10.2, 9.4, 3.9])})

However, dimensions are still required to be consistent between different
Dataset variables, so you cannot change the size of a dimension unless you
replace all dataset variables that use it.

``update`` also performs automatic alignment if necessary. Unlike ``merge``, it
maintains the alignment of the original array instead of merging indexes:

.. ipython:: python

    ds.update(other)

The exact same alignment logic when setting a variable with ``__setitem__``
syntax:

.. ipython:: python

    ds['baz'] = xr.DataArray([9, 9, 9, 9, 9], coords=[('x', list('abcde'))])
    ds.baz

Equals and identical
~~~~~~~~~~~~~~~~~~~~

xarray objects can be compared by using the :py:meth:`~xarray.Dataset.equals`,
:py:meth:`~xarray.Dataset.identical` and
:py:meth:`~xarray.Dataset.broadcast_equals` methods. These methods are used by
the optional ``compat`` argument on ``concat`` and ``merge``.

:py:attr:`~xarray.Dataset.equals` checks dimension names, indexes and array
values:

.. ipython:: python

    arr.equals(arr.copy())

:py:attr:`~xarray.Dataset.identical` also checks attributes, and the name of each
object:

.. ipython:: python

    arr.identical(arr.rename('bar'))

:py:attr:`~xarray.Dataset.broadcast_equals` does a more relaxed form of equality
check that allows variables to have different dimensions, as long as values
are constant along those new dimensions:

.. ipython:: python

    left = xr.Dataset(coords={'x': 0})
    right = xr.Dataset({'x': [0, 0, 0]})
    left.broadcast_equals(right)

Like pandas objects, two xarray objects are still equal or identical if they have
missing values marked by ``NaN`` in the same locations.

In contrast, the ``==`` operation performs element-wise comparison (like
numpy):

.. ipython:: python

    arr == arr.copy()

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

.. ipython:: python

    ds1 = xr.Dataset({'a': ('x', [10, 20, 30, np.nan])}, {'x': [1, 2, 3, 4]})
    ds2 = xr.Dataset({'a': ('x', [np.nan, 30, 40, 50])}, {'x': [2, 3, 4, 5]})
    xr.merge([ds1, ds2], compat='no_conflicts')

Note that due to the underlying representation of missing values as floating
point numbers (``NaN``), variable data type is not always preserved when merging
in this manner.
