Combining data
--------------

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xray
    np.random.seed(123456)

Concatenate
~~~~~~~~~~~

To combine arrays along existing or new dimension into a larger array, you
can use :py:func:`~xray.concat`. ``concat`` takes an iterable of ``DataArray``
or ``Dataset`` objects, as well as a dimension name, and concatenates along
that dimension:

.. ipython:: python

    arr = xray.DataArray(np.random.randn(2, 3),
                         [('x', ['a', 'b']), ('y', [10, 20, 30])])
    arr[:, :1]
    # this resembles how you would use np.concatenate
    xray.concat([arr[:, :1], arr[:, 1:]], dim='y')

In addition to combining along an existing dimension, ``concat`` can create a
new dimension by stacking lower dimensional arrays together:

.. ipython:: python

    arr[0]
    # to combine these 1d arrays into a 2d array in numpy, you would use np.array
    xray.concat([arr[0], arr[1]], 'x')

If the second argument to ``concat`` is a new dimension name, the arrays will
be concatenated along that new dimension, which is always inserted as the first
dimension:

.. ipython:: python

    xray.concat([arr[0], arr[1]], 'new_dim')

This is actually the default behavior for ``concat``:

.. ipython:: python

    xray.concat([arr[0], arr[1]])

The second argument to ``concat`` can also be an :py:class:`~pandas.Index` or
:py:class:`~xray.DataArray` object as well as a string, in which case it is
used to label the values along the new dimension:

.. ipython:: python

    xray.concat([arr[0], arr[1]], pd.Index([-90, -100], name='new_dim'))

Of course, ``concat`` also works on ``Dataset`` objects:

.. ipython:: python

    ds = arr.to_dataset(name='foo')
    xray.concat([ds.sel(x='a'), ds.sel(x='b')], 'x')

:py:func:`~xray.concat` has a number of options which provide deeper control
over which variables and coordinates are concatenated and how it handles
conflicting variables between datasets. However, these should rarely be
necessary.

.. _merge:

Merge and ``Dataset.__init__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To combine variables and coordinates between multiple Datasets, you can use the
:py:meth:`~xray.Dataset.merge` and :py:meth:`~xray.Dataset.update` methods.
Merge checks for conflicting variables before merging and by default it returns
a new Dataset:

.. ipython:: python

    ds.merge({'hello': ('space', np.arange(3) + 10)})

If you merge another dataset (or a dictionary including data array objects), by
default the resulting dataset will be aligned on the **union** of all index
coordinates:

.. ipython:: python

    other = xray.Dataset({'bar': ('x', [1, 2, 3, 4]), 'x': list('abcd')})
    ds.merge(other)

This ensures that the ``merge`` is non-destructive.

The same non-destructive merging between ``DataArray`` index coordinates is
used in the :py:class:`~xray.Dataset` constructor:

.. ipython:: python

    xray.Dataset({'a': arr[:-1], 'b': arr[1:]})

.. _update:

Update and ``__setitem__``
~~~~~~~~~~~~~~~~~~~~~~~~~~

In contrast, update modifies a dataset in-place without checking for conflicts,
and will overwrite any existing variables with new values:

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

    ds['baz'] = xray.DataArray([9, 9, 9, 9, 9], coords=[('x', list('abcde'))])
    ds.baz

Equals and identical
~~~~~~~~~~~~~~~~~~~~

xray objects can be compared by using the :py:meth:`~xray.Dataset.equals`,
:py:meth:`~xray.Dataset.identical` and
:py:meth:`~xray.Dataset.broadcast_equals` methods. These methods are used by
the optional ``compat`` argument on ``concat`` and ``merge``.

:py:attr:`~xray.Dataset.equals` checks dimension names, indexes and array
values:

.. ipython:: python

    arr.equals(arr.copy())

:py:attr:`~xray.Dataset.identical` also checks attributes, and the name of each
object:

.. ipython:: python

    arr.identical(arr.rename('bar'))

:py:attr:`~xray.Dataset.broadcast_equals` does a more relaxed form of equality
check that allows variables to have different dimensions, as long as values
are constant along those new dimensions:

.. ipython:: python

    left = Dataset(coords={'x': 0})
    right = Dataset({'x': [0, 0, 0]})
    left.broadcast_equals(right)

Like pandas objects, two xray objects are still equal or identical if they have
missing values marked by ``NaN`` in the same locations.

In contrast, the ``==`` operation performs element-wise comparison (like
numpy):

.. ipython:: python

    arr == arr.copy()

Note that ``NaN`` does not compare equal to ``NaN`` in element-wise comparison;
you may need to deal with missing values explicitly.
