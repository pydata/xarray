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

Merge and update
~~~~~~~~~~~~~~~~

To combine variables and coordinates between multiple Datasets, you can use the
:py:meth:`~xray.Dataset.merge` and :py:meth:`~xray.Dataset.update` methods.
Merge checks for conflicting variables before merging and by
default it returns a new Dataset:

.. ipython:: python

    ds.merge({'hello': ('space', np.arange(3) + 10)})

In contrast, update modifies a dataset in-place without checking for conflicts,
and will overwrite any existing variables with new values:

.. ipython:: python

    ds.update({'space': ('space', [10.2, 9.4, 3.9])})

However, dimensions are still required to be consistent between different
Dataset variables, so you cannot change the size of a dimension unless you
replace all dataset variables that use it.

Equals and identical
~~~~~~~~~~~~~~~~~~~~

xray objects can be compared by using the :py:meth:`~xray.DataArray.equals`
and :py:meth:`~xray.DataArray.identical` methods. These methods are used by
the optional ``compat`` argument on ``concat`` and ``merge``.

``equals`` checks dimension names, indexes and array values:

.. ipython:: python

    arr.equals(arr.copy())

``identical`` also checks attributes, and the name of each object:

.. ipython:: python

    arr.identical(arr.rename('bar'))

In contrast, the ``==`` operation performs element-wise comparison (like
numpy):

.. ipython:: python

    arr == arr.copy()

Like pandas objects, two xray objects are still equal or identical if they have
missing values marked by `NaN` in the same locations.
