.. _comput:

###########
Computation
###########

The labels associated with :py:class:`~xarray.DataArray` and
:py:class:`~xarray.Dataset` objects enables some powerful shortcuts for
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
    import xarray as xr
    np.random.seed(123456)

.. ipython:: python

    arr = xr.DataArray(np.random.randn(2, 3),
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

.. _missing_values:

Missing values
==============

xarray objects borrow the :py:meth:`~xarray.DataArray.isnull`,
:py:meth:`~xarray.DataArray.notnull`, :py:meth:`~xarray.DataArray.count`,
:py:meth:`~xarray.DataArray.dropna` and :py:meth:`~xarray.DataArray.fillna` methods
for working with missing data from pandas:

.. ipython:: python

    x = xr.DataArray([0, 1, np.nan, np.nan, 2], dims=['x'])
    x.isnull()
    x.notnull()
    x.count()
    x.dropna(dim='x')
    x.fillna(-1)

Like pandas, xarray uses the float value ``np.nan`` (not-a-number) to represent
missing values.

Aggregation
===========

Aggregation methods have been updated to take a `dim` argument instead of
`axis`. This allows for very intuitive syntax for aggregation methods that are
applied along particular dimension(s):

.. ipython:: python

    arr.sum(dim='x')
    arr.std(['x', 'y'])
    arr.min()


If you need to figure out the axis number for a dimension yourself (say,
for wrapping code designed to work with numpy arrays), you can use the
:py:meth:`~xarray.DataArray.get_axis_num` method:

.. ipython:: python

    arr.get_axis_num('y')

These operations automatically skip missing values, like in pandas:

.. ipython:: python

    xr.DataArray([1, 2, np.nan, 3]).mean()

If desired, you can disable this behavior by invoking the aggregation method
with ``skipna=False``.

Broadcasting by dimension name
==============================

``DataArray`` objects are automatically align themselves ("broadcasting" in
the numpy parlance) by dimension name instead of axis order. With xarray, you
do not need to transpose arrays or insert dimensions of length 1 to get array
operations to work, as commonly done in numpy with :py:func:`np.reshape` or
:py:const:`np.newaxis`.

This is best illustrated by a few examples. Consider two one-dimensional
arrays with different sizes aligned along different dimensions:

.. ipython:: python

    a = xr.DataArray([1, 2], [('x', ['a', 'b'])])
    a
    b = xr.DataArray([-1, -2, -3], [('y', [10, 20, 30])])
    b

With xarray, we can apply binary mathematical operations to these arrays, and
their dimensions are expanded automatically:

.. ipython:: python

    a * b

Moreover, dimensions are always reordered to the order in which they first
appeared:

.. ipython:: python

    c = xr.DataArray(np.arange(6).reshape(3, 2), [b['y'], a['x']])
    c
    a + c

This means, for example, that you always subtract an array from its transpose:

.. ipython:: python

    c - c.T

You can explicitly broadcast xaray data structures by using the
:py:func:`~xarray.broadcast` function:

    a2, b2 = xr.broadcast(a, b2)
    a2
    b2

.. _math automatic alignment:

Automatic alignment
===================

xarray enforces alignment between *index* :ref:`coordinates` (that is,
coordinates with the same name as a dimension, marked by ``*``) on objects used
in binary operations.

Similarly to pandas, this alignment is automatic for arithmetic on binary
operations. Note that unlike pandas, this the result of a binary operation is
by the *intersection* (not the union) of coordinate labels:

.. ipython:: python

    arr + arr[:1]

If the result would be empty, an error is raised instead:

.. ipython::

    @verbatim
    In [1]: arr[:2] + arr[2:]
    ValueError: no overlapping labels for some dimensions: ['x']

Before loops or performance critical code, it's a good idea to align arrays
explicitly (e.g., by putting them in the same Dataset or using
:py:func:`~xarray.align`) to avoid the overhead of repeated alignment with each
operation. See :ref:`align and reindex` for more details.

.. note::

    There is no automatic alignment between arguments when performing in-place
    arithmetic operations such as ``+=``. You will need to use
    :ref:`manual alignment<align and reindex>`. This ensures in-place
    arithmetic never needs to modify data types.

.. _coordinates math:

Coordinates
===========

Although index coordinates are aligned, other coordinates are not, and if their
values conflict, they will be dropped. This is necessary, for example, because
indexing turns 1D coordinates into scalar coordinates:

.. ipython:: python

    arr[0]
    arr[1]
    # notice that the scalar coordinate 'x' is silently dropped
    arr[1] - arr[0]

Still, xarray will persist other coordinates in arithmetic, as long as there
are no conflicting values:

.. ipython:: python

    # only one argument has the 'x' coordinate
    arr[0] + 1
    # both arguments have the same 'x' coordinate
    arr[0] - arr[0]

Math with datasets
==================

Datasets support arithmetic operations by automatically looping over all data
variables:

.. ipython:: python

    ds = xr.Dataset({'x_and_y': (('x', 'y'), np.random.randn(2, 3)),
                     'x_only': ('x', np.random.randn(2))},
                     coords=arr.coords)
    ds > 0

Datasets support most of the same methods found on data arrays:

.. ipython:: python

    ds.mean(dim='x')
    abs(ds)

Unfortunately, a limitation of the current version of numpy means that we
cannot override ufuncs for datasets, because datasets cannot be written as
a single array [1]_. :py:meth:`~xarray.Dataset.apply` works around this
limitation, by applying the given function to each variable in the dataset:

.. ipython:: python

    ds.apply(np.sin)

Datasets also use looping over variables for *broadcasting* in binary
arithmetic. You can do arithmetic between any ``DataArray`` and a dataset:

.. ipython:: python

    ds + arr

Arithmetic between two datasets matches data variables of the same name:

.. ipython:: python

    ds2 = xr.Dataset({'x_and_y': 0, 'x_only': 100})
    ds - ds2

Similarly to index based alignment, the result has the intersection of all
matching variables, and ``ValueError`` is raised if the result would be empty.

.. [1] In some future version of NumPy, we should be able to override ufuncs for
       datasets by making use of ``__numpy_ufunc__``.
