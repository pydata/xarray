.. _indexing:

Indexing and selecting data
===========================

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr
    np.random.seed(123456)

xarray offers extremely flexible indexing routines that combine the best
features of NumPy and pandas for data selection.

The most basic way to access elements of a :py:class:`~xarray.DataArray`
object is to use Python's ``[]`` syntax, such as ``array[i, j]``, where
``i`` and ``j`` are both integers.
As xarray objects can store coordinates corresponding to each dimension of an
array, label-based indexing similar to ``pandas.DataFrame.loc`` is also possible.
In label-based indexing, the element position ``i`` is automatically
looked-up from the coordinate values.

Dimensions of xarray objects have names, so you can also lookup the dimensions
by name, instead of remembering their positional order.

Thus in total, xarray supports four different kinds of indexing, as described
below and summarized in this table:

.. |br| raw:: html

   <br />

+------------------+--------------+---------------------------------+--------------------------------+
| Dimension lookup | Index lookup | ``DataArray`` syntax            | ``Dataset`` syntax             |
+==================+==============+=================================+================================+
| Positional       | By integer   | ``arr[:, 0]``                   | *not available*                |
+------------------+--------------+---------------------------------+--------------------------------+
| Positional       | By label     | ``arr.loc[:, 'IA']``            | *not available*                |
+------------------+--------------+---------------------------------+--------------------------------+
| By name          | By integer   | ``arr.isel(space=0)`` or |br|   | ``ds.isel(space=0)`` or |br|   |
|                  |              | ``arr[dict(space=0)]``          | ``ds[dict(space=0)]``          |
+------------------+--------------+---------------------------------+--------------------------------+
| By name          | By label     | ``arr.sel(space='IA')`` or |br| | ``ds.sel(space='IA')`` or |br| |
|                  |              | ``arr.loc[dict(space='IA')]``   | ``ds.loc[dict(space='IA')]``   |
+------------------+--------------+---------------------------------+--------------------------------+

More advanced indexing is also possible for all the methods by
supplying :py:class:`~xarray.DataArray` objects as indexer.
See :ref:`vectorized_indexing` for the details.


Positional indexing
-------------------

Indexing a :py:class:`~xarray.DataArray` directly works (mostly) just like it
does for numpy arrays, except that the returned object is always another
DataArray:

.. ipython:: python

    arr = xr.DataArray(np.random.rand(4, 3),
                       [('time', pd.date_range('2000-01-01', periods=4)),
                        ('space', ['IA', 'IL', 'IN'])])
    arr[:2]
    arr[0, 0]
    arr[:, [2, 1]]

Attributes are persisted in all indexing operations.

.. warning::

    Positional indexing deviates from the NumPy when indexing with multiple
    arrays like ``arr[[0, 1], [0, 1]]``, as described in
    :ref:`vectorized_indexing`.

xarray also supports label-based indexing, just like pandas. Because
we use a :py:class:`pandas.Index` under the hood, label based indexing is very
fast. To do label based indexing, use the :py:attr:`~xarray.DataArray.loc` attribute:

.. ipython:: python

    arr.loc['2000-01-01':'2000-01-02', 'IA']

In this example, the selected is a subpart of the array
in the range '2000-01-01':'2000-01-02' along the first coordinate `time`
and with 'IA' value from the second coordinate `space`.

You can perform any of the label indexing operations `supported by pandas`__,
including indexing with individual, slices and arrays of labels, as well as
indexing with boolean arrays. Like pandas, label based indexing in xarray is
*inclusive* of both the start and stop bounds.

__ http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-label

Setting values with label based indexing is also supported:

.. ipython:: python

    arr.loc['2000-01-01', ['IL', 'IN']] = -10
    arr


Indexing with dimension names
-----------------------------

With the dimension names, we do not have to rely on dimension order and can
use them explicitly to slice data. There are two ways to do this:

1. Use a dictionary as the argument for array positional or label based array
   indexing:

    .. ipython:: python

        # index by integer array indices
        arr[dict(space=0, time=slice(None, 2))]

        # index by dimension coordinate labels
        arr.loc[dict(time=slice('2000-01-01', '2000-01-02'))]

2. Use the :py:meth:`~xarray.DataArray.sel` and :py:meth:`~xarray.DataArray.isel`
   convenience methods:

    .. ipython:: python

        # index by integer array indices
        arr.isel(space=0, time=slice(None, 2))

        # index by dimension coordinate labels
        arr.sel(time=slice('2000-01-01', '2000-01-02'))

The arguments to these methods can be any objects that could index the array
along the dimension given by the keyword, e.g., labels for an individual value,
Python :py:func:`slice` objects or 1-dimensional arrays.

.. note::

    We would love to be able to do indexing with labeled dimension names inside
    brackets, but unfortunately, Python `does yet not support`__ indexing with
    keyword arguments like ``arr[space=0]``

__ http://legacy.python.org/dev/peps/pep-0472/


.. _nearest neighbor lookups:

Nearest neighbor lookups
------------------------

The label based selection methods :py:meth:`~xarray.Dataset.sel`,
:py:meth:`~xarray.Dataset.reindex` and :py:meth:`~xarray.Dataset.reindex_like` all
support ``method`` and ``tolerance`` keyword argument. The method parameter allows for
enabling nearest neighbor (inexact) lookups by use of the methods ``'pad'``,
``'backfill'`` or ``'nearest'``:

.. ipython:: python

   data = xr.DataArray([1, 2, 3], [('x', [0, 1, 2])])
   data.sel(x=[1.1, 1.9], method='nearest')
   data.sel(x=0.1, method='backfill')
   data.reindex(x=[0.5, 1, 1.5, 2, 2.5], method='pad')

Tolerance limits the maximum distance for valid matches with an inexact lookup:

.. ipython:: python

   data.reindex(x=[1.1, 1.5], method='nearest', tolerance=0.2)

The method parameter is not yet supported if any of the arguments
to ``.sel()`` is a ``slice`` object:

.. ipython::
   :verbatim:

   In [1]: data.sel(x=slice(1, 3), method='nearest')
   NotImplementedError

However, you don't need to use ``method`` to do inexact slicing. Slicing
already returns all values inside the range (inclusive), as long as the index
labels are monotonic increasing:

.. ipython:: python

   data.sel(x=slice(0.9, 3.1))

Indexing axes with monotonic decreasing labels also works, as long as the
``slice`` or ``.loc`` arguments are also decreasing:

.. ipython:: python

   reversed_data = data[::-1]
   reversed_data.loc[3.1:0.9]


Dataset indexing
----------------

We can also use these methods to index all variables in a dataset
simultaneously, returning a new dataset:

.. ipython:: python

    ds = arr.to_dataset(name='foo')
    ds.isel(space=[0], time=[0])
    ds.sel(time='2000-01-01')

Positional indexing on a dataset is not supported because the ordering of
dimensions in a dataset is somewhat ambiguous (it can vary between different
arrays). However, you can do normal indexing with dimension names:

.. ipython:: python


    ds[dict(space=[0], time=[0])]
    ds.loc[dict(time='2000-01-01')]

Using indexing to *assign* values to a subset of dataset (e.g.,
``ds[dict(space=0)] = 1``) is not yet supported.

Dropping labels
---------------

The :py:meth:`~xarray.Dataset.drop` method returns a new object with the listed
index labels along a dimension dropped:

.. ipython:: python

    ds.drop(['IN', 'IL'], dim='space')

``drop`` is both a ``Dataset`` and ``DataArray`` method.


.. _masking with where:

Masking with ``where``
----------------------

Indexing methods on xarray objects generally return a subset of the original data.
However, it is sometimes useful to select an object with the same shape as the
original data, but with some elements masked. To do this type of selection in
xarray, use :py:meth:`~xarray.DataArray.where`:

.. ipython:: python

    arr2 = xr.DataArray(np.arange(16).reshape(4, 4), dims=['x', 'y'])
    arr2.where(arr2.x + arr2.y < 4)

This is particularly useful for ragged indexing of multi-dimensional data,
e.g., to apply a 2D mask to an image. Note that ``where`` follows all the
usual xarray broadcasting and alignment rules for binary operations (e.g.,
``+``) between the object being indexed and the condition, as described in
:ref:`comput`:

.. ipython:: python

    arr2.where(arr2.y < 2)

By default ``where`` maintains the original size of the data.  For cases
where the selected data size is much smaller than the original data,
use of the option ``drop=True`` clips coordinate
elements that are fully masked:

.. ipython:: python

    arr2.where(arr2.y < 2, drop=True)


.. _vectorized_indexing:

Vectorized Indexing
-------------------

Like numpy and pandas, xarray supports indexing many array elements at once in a
`vectorized` manner.

If you only provide integers, slices, or unlabeled arrays (array without
dimension names, such as ``np.ndarray``, ``list``, but not
:py:meth:`~xarray.DataArray` or :py:meth:`~xarray.Variable`) indexing can be
understand as orthogonally. Each indexer component selects independently along
the corresponding dimension, similar to how vector indexing works in Fortran or
MATLAB, or after using the :py:func:`numpy.ix_` helper:

.. ipython:: python

    da = xr.DataArray(np.arange(12).reshape((3, 4)), dims=['x', 'y'],
                      coords={'x': [0, 1, 2], 'y': ['a', 'b', 'c', 'd']})
    da
    da[[0, 1], [1, 1]]

For more flexibility, you can supply :py:meth:`~xarray.DataArray` objects
as indexers.
Dimensions on resultant arrays are given by the ordered union of the indexers'
dimensions:

.. ipython:: python

    ind_x = xr.DataArray([0, 1], dims=['x'])
    ind_y = xr.DataArray([0, 1], dims=['y'])
    da[ind_x, ind_y]  # orthogonal indexing
    da[ind_x, ind_x]  # vectorized indexing

Slices or sequences/arrays without named-dimensions are treated as if they have
the same dimension which is indexed along:

.. ipython:: python

    # Because [0, 1] is used to index along dimension 'x',
    # it is assumed to have dimension 'x'
    da[[0, 1], ind_x]

Furthermore, you can use multi-dimensional :py:meth:`~xarray.DataArray`
as indexers, where the resultant array dimension is also determined by
indexers' dimension:

.. ipython:: python

    ind = xr.DataArray([[0, 1], [0, 1]], dims=['a', 'b'])
    da[ind]

Similar to how NumPy's `advanced indexing`_ works, vectorized
indexing for xarray is based on our
:ref:`broadcasting rules <compute.broadcasting>`.
See :ref:`indexing.rules` for the complete specification.

.. _advanced indexing: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html

Vectorized indexing also works with ``isel``, ``loc``, and ``sel``:

.. ipython:: python

    ind = xr.DataArray([[0, 1], [0, 1]], dims=['a', 'b'])
    da.isel(y=ind)  # same as da[:, ind]

    ind = xr.DataArray([['a', 'b'], ['b', 'a']], dims=['a', 'b'])
    da.loc[:, ind]  # same as da.sel(y=ind)

These methods may and also be applied to ``Dataset`` objects

.. ipython:: python

    ds2 = da.to_dataset(name='bar')
    ds2.isel(x=xr.DataArray([0, 1, 2], dims=['points']))

.. tip::

  If you are lazily loading your data from disk, not every form of vectorized
  indexing is supported (or if supported, may not be supported efficiently).
  You may find increased performance by loading your data into memory first,
  e.g., with :py:meth:`~xarray.Dataset.load`.

.. note::

  Vectorized indexing is a new feature in v0.10.
  In older versions of xarray, dimensions of indexers are ignored.
  Dedicated methods for some advanced indexing use cases,
  ``isel_points`` and ``sel_points`` are now deprecated.
  See :ref:`more_advanced_indexing` for their alternative.

.. _assigning_values:

Assigning values with indexing
------------------------------

Vectorized indexing can be used to assign values to xarray object.

.. ipython:: python

    da = xr.DataArray(np.arange(12).reshape((3, 4)), dims=['x', 'y'],
                      coords={'x': [0, 1, 2], 'y': ['a', 'b', 'c', 'd']})
    da
    da[0] = -1  # assignment with broadcasting
    da

    ind_x = xr.DataArray([0, 1], dims=['x'])
    ind_y = xr.DataArray([0, 1], dims=['y'])
    da[ind_x, ind_y] = -2  # assign -2 to (ix, iy) = (0, 0) and (1, 1)
    da

    da[ind_x, ind_y] += 100  # increment is also possible
    da

Like ``numpy.ndarray``, value assignment sometimes works differently from what one may expect.

.. ipython:: python

    da = xr.DataArray([0, 1, 2, 3], dims=['x'])
    ind = xr.DataArray([0, 0, 0], dims=['x'])
    da[ind] -= 1
    da

Where the 0th element will be subtracted 1 only once.
This is because ``v[0] = v[0] - 1`` is called three times, rather than
``v[0] = v[0] - 1 - 1 - 1``.
See `Assigning values to indexed arrays`__ for the details.

__ https://docs.scipy.org/doc/numpy/user/basics.indexing.html#assigning-values-to-indexed-arrays


.. note::
  Dask array does not support value assignment
  (see :ref:`dask` for the details).


.. warning::

  Do not try to assign values when using any of the indexing methods ``isel``
  or ``sel``::

    # DO NOT do this
    arr.isel(space=0) = 0

  Assigning values with the chained indexing using ``.sel`` or ``.isel`` fails silently.

  .. ipython:: python

      da = xr.DataArray([0, 1, 2, 3], dims=['x'])
      # DO NOT do this
      da.isel(x=[0, 1, 2])[1] = -1
      da


.. _more_advanced_indexing:

More advanced indexing
-----------------------

The use of :py:meth:`~xarray.DataArray` objects as indexers enables very
flexible indexing. The following is an example of the pointwise indexing:

.. ipython:: python

    da = xr.DataArray(np.arange(56).reshape((7, 8)), dims=['x', 'y'])
    da
    da.isel(x=xr.DataArray([0, 1, 6], dims='z'),
            y=xr.DataArray([0, 1, 0], dims='z'))

where three elements at ``(ix, iy) = ((0, 0), (1, 1), (6, 0))`` are selected
and mapped along a new dimension ``z``.

If you want to add a coordinate to the new dimension ``z``,
you can supply a :py:meth:`~xarray.DataArray` with a coordinate,

.. ipython:: python

    da.isel(x=xr.DataArray([0, 1, 6], dims='z',
                           coords={'z': ['a', 'b', 'c']}),
            y=xr.DataArray([0, 1, 0], dims='z'))

Analogously, label-based pointwise-indexing is also possible by the ``.sel``
method:

.. ipython:: python

    times = xr.DataArray(pd.to_datetime(['2000-01-03', '2000-01-02', '2000-01-01']),
                         dims='new_time')
    arr.sel(space=xr.DataArray(['IA', 'IL', 'IN'], dims=['new_time']),
            time=times)

.. _align and reindex:

Align and reindex
-----------------

xarray's ``reindex``, ``reindex_like`` and ``align`` impose a ``DataArray`` or
``Dataset`` onto a new set of coordinates corresponding to dimensions. The
original values are subset to the index labels still found in the new labels,
and values corresponding to new labels not found in the original object are
in-filled with `NaN`.

xarray operations that combine multiple objects generally automatically align
their arguments to share the same indexes. However, manual alignment can be
useful for greater control and for increased performance.

To reindex a particular dimension, use :py:meth:`~xarray.DataArray.reindex`:

.. ipython:: python

    arr.reindex(space=['IA', 'CA'])

The :py:meth:`~xarray.DataArray.reindex_like` method is a useful shortcut.
To demonstrate, we will make a subset DataArray with new values:

.. ipython:: python

    foo = arr.rename('foo')
    baz = (10 * arr[:2, :2]).rename('baz')
    baz

Reindexing ``foo`` with ``baz`` selects out the first two values along each
dimension:

.. ipython:: python

    foo.reindex_like(baz)

The opposite operation asks us to reindex to a larger shape, so we fill in
the missing values with `NaN`:

.. ipython:: python

    baz.reindex_like(foo)

The :py:func:`~xarray.align` function lets us perform more flexible database-like
``'inner'``, ``'outer'``, ``'left'`` and ``'right'`` joins:

.. ipython:: python

    xr.align(foo, baz, join='inner')
    xr.align(foo, baz, join='outer')

Both ``reindex_like`` and ``align`` work interchangeably between
:py:class:`~xarray.DataArray` and :py:class:`~xarray.Dataset` objects, and with any number of matching dimension names:

.. ipython:: python

    ds
    ds.reindex_like(baz)
    other = xr.DataArray(['a', 'b', 'c'], dims='other')
    # this is a no-op, because there are no shared dimension names
    ds.reindex_like(other)

.. _indexing.missing_coordinates:

Missing coordinate labels
-------------------------

Coordinate labels for each dimension are optional (as of xarray v0.9). Label
based indexing with ``.sel`` and ``.loc`` uses standard positional,
integer-based indexing as a fallback for dimensions without a coordinate label:

.. ipython:: python

    array = xr.DataArray([1, 2, 3], dims='x')
    array.sel(x=[0, -1])

Alignment between xarray objects where one or both do not have coordinate labels
succeeds only if all dimensions of the same name have the same length.
Otherwise, it raises an informative error:

.. ipython::
    :verbatim:

    In [62]: xr.align(array, array[:2])
    ValueError: arguments without labels along dimension 'x' cannot be aligned because they have different dimension sizes: {2, 3}

Underlying Indexes
------------------

xarray uses the :py:class:`pandas.Index` internally to perform indexing
operations.  If you need to access the underlying indexes, they are available
through the :py:attr:`~xarray.DataArray.indexes` attribute.

.. ipython:: python

   arr
   arr.indexes
   arr.indexes['time']

Use :py:meth:`~xarray.DataArray.get_index` to get an index for a dimension,
falling back to a default :py:class:`pandas.RangeIndex` if it has no coordinate
labels:

.. ipython:: python

    array
    array.get_index('x')


.. _copies_vs_views:

Copies vs. Views
----------------

Whether array indexing returns a view or a copy of the underlying
data depends on the nature of the labels.

For positional (integer)
indexing, xarray follows the same rules as NumPy:

* Positional indexing with only integers and slices returns a view.
* Positional indexing with arrays or lists returns a copy.

The rules for label based indexing are more complex:

* Label-based indexing with only slices returns a view.
* Label-based indexing with arrays returns a copy.
* Label-based indexing with scalars returns a view or a copy, depending
  upon if the corresponding positional indexer can be represented as an
  integer or a slice object. The exact rules are determined by pandas.

Whether data is a copy or a view is more predictable in xarray than in pandas, so
unlike pandas, xarray does not produce `SettingWithCopy warnings`_. However, you
should still avoid assignment with chained indexing.

.. _SettingWithCopy warnings: http://pandas.pydata.org/pandas-docs/stable/indexing.html#returning-a-view-versus-a-copy


.. _multi-level indexing:

Multi-level indexing
--------------------

Just like pandas, advanced indexing on multi-level indexes is possible with
``loc`` and ``sel``. You can slice a multi-index by providing multiple indexers,
i.e., a tuple of slices, labels, list of labels, or any selector allowed by
pandas:

.. ipython:: python

  midx = pd.MultiIndex.from_product([list('abc'), [0, 1]],
                                    names=('one', 'two'))
  mda = xr.DataArray(np.random.rand(6, 3),
                     [('x', midx), ('y', range(3))])
  mda
  mda.sel(x=(list('ab'), [0]))

You can also select multiple elements by providing a list of labels or tuples or
a slice of tuples:

.. ipython:: python

  mda.sel(x=[('a', 0), ('b', 1)])

Additionally, xarray supports dictionaries:

.. ipython:: python

  mda.sel(x={'one': 'a', 'two': 0})

For convenience, ``sel`` also accepts multi-index levels directly
as keyword arguments:

.. ipython:: python

  mda.sel(one='a', two=0)

Note that using ``sel`` it is not possible to mix a dimension
indexer with level indexers for that dimension
(e.g., ``mda.sel(x={'one': 'a'}, two=0)`` will raise a ``ValueError``).

Like pandas, xarray handles partial selection on multi-index (level drop).
As shown below, it also renames the dimension / coordinate when the
multi-index is reduced to a single index.

.. ipython:: python

  mda.loc[{'one': 'a'}, ...]

Unlike pandas, xarray does not guess whether you provide index levels or
dimensions when using ``loc`` in some ambiguous cases. For example, for
``mda.loc[{'one': 'a', 'two': 0}]`` and ``mda.loc['a', 0]`` xarray
always interprets ('one', 'two') and ('a', 0) as the names and
labels of the 1st and 2nd dimension, respectively. You must specify all
dimensions or use the ellipsis in the ``loc`` specifier, e.g. in the example
above, ``mda.loc[{'one': 'a', 'two': 0}, :]`` or ``mda.loc[('a', 0), ...]``.


.. _indexing.rules:

Indexing rules
--------------

Here we describe the full rules xarray uses for vectorized indexing. Note that
this is for the purposes of explanation: for the sake of efficiency and to
support various backends, the actual implementation is different.

0. (Only for label based indexing.) Look up positional indexes along each
   dimension from the corresponding :py:class:`pandas.Index`.

1. A full slice object ``:`` is inserted for each dimension without an indexer.

2. ``slice`` objects are converted into arrays, given by
   ``np.arange(*slice.indices(...))``.

3. Assume dimension names for array indexers without dimensions, such as
   ``np.ndarray`` and ``list``, from the dimensions to be indexed along.
   For example, ``v.isel(x=[0, 1])`` is understood as
   ``v.isel(x=xr.DataArray([0, 1], dims=['x']))``.

4. For each variable in a ``Dataset`` or  ``DataArray`` (the array and its
   coordinates):

   a. Broadcast all relevant indexers based on their dimension names
      (see :ref:`compute.broadcasting` for full details).

   b. Index the underling array by the broadcast indexers, using NumPy's
      advanced indexing rules.

5. If any indexer DataArray has coordinates and no coordinate with the
   same name exists, attach them to the indexed object.

.. note::

  Only 1-dimensional boolean arrays can be used as indexers.
