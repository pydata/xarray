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

    arr = xr.DataArray(np.random.RandomState(0).randn(2, 3),
                       [('x', ['a', 'b']), ('y', [10, 20, 30])])
    arr - 3
    abs(arr)

You can also use any of numpy's or scipy's many `ufunc`__ functions directly on
a DataArray:

__ http://docs.scipy.org/doc/numpy/reference/ufuncs.html

.. ipython:: python

    np.sin(arr)

Use :py:func:`~xarray.where` to conditionally switch between values:

.. ipython:: python

    xr.where(arr > 0, 'positive', 'negative')

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

.. _comput.rolling:

Rolling window operations
=========================

``DataArray`` objects include a :py:meth:`~xarray.DataArray.rolling` method. This
method supports rolling window aggregation:

.. ipython:: python

    arr = xr.DataArray(np.arange(0, 7.5, 0.5).reshape(3, 5),
                       dims=('x', 'y'))
    arr

:py:meth:`~xarray.DataArray.rolling` is applied along one dimension using the
name of the dimension as a key (e.g. ``y``) and the window size as the value
(e.g. ``3``).  We get back a ``Rolling`` object:

.. ipython:: python

    arr.rolling(y=3)

The label position and minimum number of periods in the rolling window are
controlled by the ``center`` and ``min_periods`` arguments:

.. ipython:: python

    arr.rolling(y=3, min_periods=2, center=True)

Aggregation and summary methods can be applied directly to the ``Rolling`` object:

.. ipython:: python

    r = arr.rolling(y=3)
    r.mean()
    r.reduce(np.std)

Note that rolling window aggregations are much faster (both asymptotically and
because they avoid a loop in Python) when bottleneck_ is installed. Otherwise,
we fall back to a slower, pure Python implementation.

.. _bottleneck: https://github.com/kwgoodman/bottleneck/

Finally, we can manually iterate through ``Rolling`` objects:

.. ipython:: python

   @verbatim
   for label, arr_window in r:
      # arr_window is a view of x

.. _compute.broadcasting:

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

.. ipython:: python

    a2, b2 = xr.broadcast(a, b)
    a2
    b2

.. _math automatic alignment:

Automatic alignment
===================

xarray enforces alignment between *index* :ref:`coordinates` (that is,
coordinates with the same name as a dimension, marked by ``*``) on objects used
in binary operations.

Similarly to pandas, this alignment is automatic for arithmetic on binary
operations. The default result of a binary operation is by the *intersection*
(not the union) of coordinate labels:

.. ipython:: python

    arr = xr.DataArray(np.arange(3), [('x', range(3))])
    arr + arr[:-1]

If coordinate values for a dimension are missing on either argument, all
matching dimensions must have the same size:

.. ipython:: python

    @verbatim
    In [1]: arr + xr.DataArray([1, 2], dims='x')
    ValueError: arguments without labels along dimension 'x' cannot be aligned because they have different dimension size(s) {2} than the size of the aligned dimension labels: 3


However, one can explicitly change this default automatic alignment type ("inner")
via :py:func:`~xarray.set_options()` in context manager:

.. ipython:: python

    with xr.set_options(arithmetic_join="outer"):
        arr + arr[:1]
    arr + arr[:1]

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

    ds = xr.Dataset({'x_and_y': (('x', 'y'), np.random.randn(3, 5)),
                     'x_only': ('x', np.random.randn(3))},
                     coords=arr.coords)
    ds > 0

Datasets support most of the same methods found on data arrays:

.. ipython:: python

    ds.mean(dim='x')
    abs(ds)

Unfortunately, we currently do not support NumPy ufuncs for datasets [1]_.
:py:meth:`~xarray.Dataset.apply` works around this
limitation, by applying the given function to each variable in the dataset:

.. ipython:: python

    ds.apply(np.sin)

You can also use the wrapped functions in the ``xarray.ufuncs`` module:

.. ipython:: python

    import xarray.ufuncs as xu
    xu.sin(ds)

Datasets also use looping over variables for *broadcasting* in binary
arithmetic. You can do arithmetic between any ``DataArray`` and a dataset:

.. ipython:: python

    ds + arr

Arithmetic between two datasets matches data variables of the same name:

.. ipython:: python

    ds2 = xr.Dataset({'x_and_y': 0, 'x_only': 100})
    ds - ds2

Similarly to index based alignment, the result has the intersection of all
matching data variables.

.. [1] This was previously due to a limitation of NumPy, but with NumPy 1.13
       we should be able to support this by leveraging ``__array_ufunc__``
       (:issue:`1617`).

.. _comput.wrapping-custom:

Wrapping custom computation
===========================

It doesn't always make sense to do computation directly with xarray objects:

  - In the inner loop of performance limited code, using xarray can add
    considerable overhead compared to using NumPy or native Python types.
    This is particularly true when working with scalars or small arrays (less
    than ~1e6 elements). Keeping track of labels and ensuring their consistency
    adds overhead, and xarray's core itself is not especially fast, because it's
    written in Python rather than a compiled language like C. Also, xarray's
    high level label-based APIs removes low-level control over how operations
    are implemented.
  - Even if speed doesn't matter, it can be important to wrap existing code, or
    to support alternative interfaces that don't use xarray objects.

For these reasons, it is often well-advised to write low-level routines that
work with NumPy arrays, and to wrap these routines to work with xarray objects.
However, adding support for labels on both :py:class:`~xarray.Dataset` and
:py:class:`~xarray.DataArray` can be a bit of a chore.

To make this easier, xarray supplies the :py:func:`~xarray.apply_ufunc` helper
function, designed for wrapping functions that support broadcasting and
vectorization on unlabeled arrays in the style of a NumPy
`universal function <https://docs.scipy.org/doc/numpy-1.13.0/reference/ufuncs.html>`_ ("ufunc" for short).
``apply_ufunc`` takes care of everything needed for an idiomatic xarray wrapper,
including alignment, broadcasting, looping over ``Dataset`` variables (if
needed), and merging of coordinates. In fact, many internal xarray
functions/methods are written using ``apply_ufunc``.

Simple functions that act independently on each value should work without
any additional arguments:

.. ipython:: python

    squared_error = lambda x, y: (x - y) ** 2
    arr1 = xr.DataArray([0, 1, 2, 3], dims='x')
    xr.apply_ufunc(squared_error, arr1, 1)

For using more complex operations that consider some array values collectively,
it's important to understand the idea of "core dimensions" from NumPy's
`generalized ufuncs <http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html>`_. Core dimensions are defined as dimensions
that should *not* be broadcast over. Usually, they correspond to the fundamental
dimensions over which an operation is defined, e.g., the summed axis in
``np.sum``. A good clue that core dimensions are needed is the presence of an
``axis`` argument on the corresponding NumPy function.

With ``apply_ufunc``, core dimensions are recognized by name, and then moved to
the last dimension of any input arguments before applying the given function.
This means that for functions that accept an ``axis`` argument, you usually need
to set ``axis=-1``. As an example, here is how we would wrap
:py:func:`numpy.linalg.norm` to calculate the vector norm:

.. code-block:: python

    def vector_norm(x, dim, ord=None):
        return xr.apply_ufunc(np.linalg.norm, x,
                              input_core_dims=[[dim]],
                              kwargs={'ord': ord, 'axis': -1})

.. ipython:: python
   :suppress:

    def vector_norm(x, dim, ord=None):
        return xr.apply_ufunc(np.linalg.norm, x,
                              input_core_dims=[[dim]],
                              kwargs={'ord': ord, 'axis': -1})

.. ipython:: python

    vector_norm(arr1, dim='x')

Because ``apply_ufunc`` follows a standard convention for ufuncs, it plays
nicely with tools for building vectorized functions, like
:func:`numpy.broadcast_arrays` and :func:`numpy.vectorize`. For high performance
needs, consider using Numba's :doc:`vectorize and guvectorize <numba:user/vectorize>`.

In addition to wrapping functions, ``apply_ufunc`` can automatically parallelize
many functions when using dask by setting ``dask='parallelized'``. See
:ref:`dask.automatic-parallelization` for details.

:py:func:`~xarray.apply_ufunc` also supports some advanced options for
controlling alignment of variables and the form of the result. See the
docstring for full details and more examples.
