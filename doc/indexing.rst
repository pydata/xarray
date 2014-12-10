.. _indexing:

Indexing and selecting data
===========================

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xray
    np.random.seed(123456)

Similarly to pandas objects, xray objects support both integer and label
based lookups along each dimension. However, xray objects also have named
dimensions, so you can optionally use dimension names instead of relying on the
positional ordering of dimensions.

This in total, xray supports four different kinds of indexing, as described
below and summarized in this table:

================ ============ ======================= ======================
Dimension lookup Index lookup ``DataArray`` syntax    ``Dataset`` syntax
================ ============ ======================= ======================
Positional       By integer   ``arr[:, 0]``           *not available*
Positional       By label     ``arr.loc[:, 'IA']``    *not available*
By name          By integer   ``arr.isel(space=0)``   ``ds.isel(space=0)``
By name          By label     ``arr.sel(space='IA')`` ``ds.sel(space='IA')``
================ ============ ======================= ======================

Positional indexing
-------------------

Indexing a :py:class:`~xray.DataArray` directly works (mostly) just like it
does for numpy arrays, except that the returned object is always another
DataArray:

.. ipython:: python

    arr = xray.DataArray(np.random.rand(4, 3),
                         [('time', pd.date_range('2000-01-01', periods=4)),
                          ('space', ['IA', 'IL', 'IN'])])
    arr[:2]
    arr[0, 0]
    arr[:, [2, 1]]

xray also supports label-based indexing, just like pandas. Because
we use a :py:class:`pandas.Index` under the hood, label based indexing is very
fast. To do label based indexing, use the :py:attr:`~xray.DataArray.loc` attribute:

.. ipython:: python

    arr.loc['2000-01-01':'2000-01-02', 'IA']

You can perform any of the label indexing operations `supported by pandas`__,
including indexing with individual, slices and arrays of labels, as well as
indexing with boolean arrays. Like pandas, label based indexing in xray is
*inclusive* of both the start and stop bounds.

__ http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-label

Setting values with label based indexing is also supported:

.. ipython:: python

    arr.loc['2000-01-01', ['IL', 'IN']] = -10
    arr

Indexing with labeled dimensions
--------------------------------

With labeled dimensions, we do not have to rely on dimension order and can
use them explicitly to slice data. There are two ways to do this:

1. Use a dictionary as the argument for array positional or label based array
   indexing:

    .. ipython:: python

        # index by integer array indices
        arr[dict(space=0, time=slice(None, 2))]

        # index by dimension coordinate labels
        arr.loc[dict(time=slice('2000-01-01', '2000-01-02'))]

2. Use the :py:meth:`~xray.DataArray.sel` and :py:meth:`~xray.DataArray.isel`
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

.. warning::

    Do not try to assign values when using ``isel`` or ``sel``::

        # DO NOT do this
        arr.isel(space=0) = 0

    Depending on whether the underlying numpy indexing returns a copy or a
    view, the method will fail, and when it fails, **it will fail
    silently**. Instead, you should use normal index assignment::

        # this is safe
        arr[dict(space=0)] = 0

Dataset indexing
----------------

We can also use these methods to index all variables in a dataset
simultaneously, returning a new dataset:

.. ipython:: python

    ds = arr.to_dataset()
    ds.isel(space=[0], time=[0])
    ds.sel(time='2000-01-01')

Positional indexing on a dataset is not supported because the ordering of
dimensions in a dataset is somewhat ambiguous (it can vary between different
arrays). However, you can do normal indexing with labeled dimensions:

.. ipython:: python

    ds[dict(space=[0], time=[0])]
    ds.loc[dict(time='2000-01-01')]

Using indexing to *assign* values to a subset of dataset (e.g.,
``ds[dict(space=0)] = 1``) is not yet supported.

Indexing details
----------------

Like pandas, whether array indexing returns a view or a copy of the underlying
data depends entirely on numpy:

* Indexing with a single label or a slice returns a view.
* Indexing with a vector of array labels returns a copy.

Attributes are persisted in array indexing:

.. ipython:: python

    arr2 = arr.copy()
    arr2.attrs['units'] = 'meters'
    arr2[0, 0].attrs

Indexing with xray objects has one important difference from indexing numpy
arrays: you can only use one-dimensional arrays to index xray objects, and
each indexer is applied "orthogonally" along independent axes, instead of
using numpy's advanced broadcasting. This means you can do indexing like this,
which would require slightly more awkward syntax with numpy arrays:

.. ipython:: python

    arr[arr['time.day'] > 1, arr['space'] != 'IL']

This is a much simpler model than numpy's `advanced indexing`__,
and is basically the only model that works for labeled arrays. If you would
like to do array indexing, you can always index ``.values`` directly
instead:

__ http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

.. ipython:: python

    arr.values[arr.values > 0.5]

.. _align and reindex:

Align and reindex
-----------------

xray's ``reindex``, ``reindex_like`` and ``align`` impose a ``DataArray`` or
``Dataset`` onto a new set of coordinates corresponding to dimensions. The
original values are subset to the index labels still found in the new labels,
and values corresponding to new labels not found in the original object are
in-filled with `NaN`.

To reindex a particular dimension, use :py:meth:`~xray.DataArray.reindex`:

.. ipython:: python

    arr.reindex(space=['IA', 'CA'])

The :py:meth:`~xray.DataArray.reindex_like` method is a useful shortcut.
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

The :py:func:`~xray.align` function lets us perform more flexible database-like
``'inner'``, ``'outer'``, ``'left'`` and ``'right'`` joins:

.. ipython:: python

    xray.align(foo, baz, join='inner')
    xray.align(foo, baz, join='outer')

Both ``reindex_like`` and ``align`` work interchangeably between
:py:class:`~xray.DataArray` and :py:class:`~xray.Dataset` objects, and with any number of matching dimension names:

.. ipython:: python

    ds
    ds.reindex_like(baz)
    other = xray.DataArray(['a', 'b', 'c'], dims='other')
    # this is a no-op, because there are no shared dimension names
    ds.reindex_like(other)
