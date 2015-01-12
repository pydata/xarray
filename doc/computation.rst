.. _comput:

###########
Computation
###########

The labels associated with :py:class:`~xray.DataArray` and
:py:class:`~xray.Dataset` objects enables some powerful shortcuts for
computation, notably including aggregation and broadcasting by dimension
names.

Basic array math
================

Arithmetic operations with a single DataArray automatically vectorize (like
numpy) over all array values:

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xray
    np.random.seed(123456)

.. ipython:: python

    arr = xray.DataArray(np.random.randn(2, 3),
                         [('x', ['a', 'b']), ('y', [10, 20, 30])])
    arr - 3
    abs(arr)

You can also use any of numpy's or scipy's many `ufunc`__ functions directly on
a DataArray:

__ http://docs.scipy.org/doc/numpy/reference/ufuncs.html

.. ipython:: python

    np.sin(arr)

Data arrays also implement many :py:class:`numpy.ndarray` methods:

.. ipython:: python

    arr.round(2)
    arr.T

Missing values
==============

xray objects borrow the :py:meth:`~xray.DataArray.isnull`,
:py:meth:`~xray.DataArray.notnull`, :py:meth:`~xray.DataArray.count` and
:py:meth:`~xray.DataArray.dropna` methods for working with missing data from
pandas:

.. ipython:: python

    x = xray.DataArray([0, 1, np.nan, np.nan, 2], dims=['x'])
    x.isnull()
    x.notnull()
    x.count()
    x.dropna(dim='x')

Aggregation
===========

Aggregation methods from ndarray have been updated to take a `dim`
argument instead of `axis`. This allows for very intuitive syntax for
aggregation methods that are applied along particular dimension(s):

.. ipython:: python

    arr.sum(dim='x')
    arr.std(['x', 'y'])
    arr.min()

If you need to figure out the axis number for a dimension yourself (say,
for wrapping code designed to work with numpy arrays), you can use the
:py:meth:`~xray.DataArray.get_axis_num` method:

.. ipython:: python

    arr.get_axis_num('y')

To perform a NA skipping aggregations, pass the NA aware numpy functions
directly to :py:attr:`~xray.DataArray.reduce` method:

.. ipython:: python

    arr.reduce(np.nanmean, dim='y')

.. warning::

    Currently, xray uses the standard ndarray methods which do not
    automatically skip missing values, but we expect to switch the default
    to NA skipping versions (like pandas) in a future version (:issue:`130`).

Broadcasting by dimension name
==============================

``DataArray`` objects are automatically align themselves ("broadcasting" in
the numpy parlance) by dimension name instead of axis order. With xray, you
do not need to transpose arrays or insert dimensions of length 1 to get array
operations to work, as commonly done in numpy with :py:func:`np.reshape` or
:py:const:`np.newaxis`.

This is best illustrated by a few examples. Consider two one-dimensional
arrays with different sizes aligned along different dimensions:

.. ipython:: python

    a = xray.DataArray([1, 2], [('x', ['a', 'b'])])
    a
    b = xray.DataArray([-1, -2, -3], [('y', [10, 20, 30])])
    b

With xray, we can apply binary mathematical operations to these arrays, and
their dimensions are expanded automatically:

.. ipython:: python

    a * b

Moreover, dimensions are always reordered to the order in which they first
appeared:

.. ipython:: python

    c = xray.DataArray(np.arange(6).reshape(3, 2), [b['y'], a['x']])
    c
    a + c

This means, for example, that you always subtract an array from its transpose:

.. ipython:: python

    c - c.T

.. _alignment and coordinates:

Alignment and coordinates
=========================

For now, performing most binary operations on xray objects requires that the
all *index* :ref:`coordinates` (that is, coordinates with the same name as a
dimension, marked by ``*``) have the same values:

.. ipython::

    @verbatim
    In [1]: arr + arr[:1]
    ValueError: coordinate 'x' is not aligned

However, xray does have shortcuts (copied from pandas) that make aligning
``DataArray`` and ``Dataset`` objects easy and fast.

.. ipython:: python

    a, b = xray.align(arr, arr[:1])
    a + b

See :ref:`align and reindex` for more details.

.. warning::

    pandas does index based alignment automatically when doing math, using
    ``join='outer'``. xray doesn't have automatic alignment yet, but we do
    intend to enable it in a future version (:issue:`186`). Unlike pandas, we
    expect to default to ``join='inner'``.

Although index coordinates are required to match exactly, other coordinates are
not, and if their values conflict, they will be dropped. This is necessary,
for example, because indexing turns 1D coordinates into scalars:

.. ipython:: python

    arr[0]
    arr[1]
    # notice that the scalar coordinate 'x' is silently dropped
    arr[1] - arr[0]

Still, xray will persist other coordinates in arithmetic, as long as there
are no conflicting values:

.. ipython:: python

    # only one argument has the 'x' coordinate
    arr[0] + 1
    # both arguments have the same 'x' coordinate
    arr[0] - arr[0]

Math with Datasets
==================

Datasets support arithmetic operations by automatically looping over all
variables as well as dimensions:

.. ipython:: python

    ds = xray.Dataset({'x_and_y': (('x', 'y'), np.random.randn(2, 3)),
                       'x_only': ('x', np.random.randn(2))},
                       coords=arr.coords)
    ds > 0
    ds.mean(dim='x')

Datasets have most of the same ndarray methods found on data arrays. Again,
these operations loop over all dataset variables:

.. ipython:: python

    abs(ds)

:py:meth:`~xray.Dataset.transpose` can also be used to reorder dimensions on
all variables:

.. ipython:: python

    ds.transpose('y', 'x')

Unfortunately, a limitation of the current version of numpy means that we
cannot override ufuncs for datasets, because datasets cannot be written as
a single array [1]_. :py:meth:`~xray.Dataset.apply` works around this
limitation, by applying the given function to each variable in the dataset:

.. ipython:: python

    ds.apply(np.sin)

Datasets also use looping over variables for *broadcasting* in binary
arithmetic. You can do arithmetic between any ``DataArray`` and a dataset as
long as they have aligned indexes:

.. ipython:: python

    ds + arr

Arithmetic between two datasets requires that the datasets also have the same
variables:

.. ipython:: python

    ds2 = xray.Dataset({'x_and_y': 0, 'x_only': 100})
    ds - ds2

There is no shortcut similar to ``align`` for aligning variable names, but you
may find :py:meth:`~xray.Dataset.rename` and
:py:meth:`~xray.Dataset.drop_vars` useful.

.. note::

    When we enable automatic alignment over indexes, we will probably enable
    automatic alignment between dataset variables as well.

.. [1] When numpy 1.10 is released, we should be able to override ufuncs for
       datasets by making use of ``__numpy_ufunc__``.
