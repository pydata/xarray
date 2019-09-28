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

Use `@` to perform matrix multiplication:

.. ipython:: python

    arr @ arr

Data arrays also implement many :py:class:`numpy.ndarray` methods:

.. ipython:: python

    arr.round(2)
    arr.T

.. _missing_values:

Missing values
==============

xarray objects borrow the :py:meth:`~xarray.DataArray.isnull`,
:py:meth:`~xarray.DataArray.notnull`, :py:meth:`~xarray.DataArray.count`,
:py:meth:`~xarray.DataArray.dropna`, :py:meth:`~xarray.DataArray.fillna`,
:py:meth:`~xarray.DataArray.ffill`, and :py:meth:`~xarray.DataArray.bfill`
methods for working with missing data from pandas:

.. ipython:: python

    x = xr.DataArray([0, 1, np.nan, np.nan, 2], dims=['x'])
    x.isnull()
    x.notnull()
    x.count()
    x.dropna(dim='x')
    x.fillna(-1)
    x.ffill('x')
    x.bfill('x')

Like pandas, xarray uses the float value ``np.nan`` (not-a-number) to represent
missing values.

xarray objects also have an :py:meth:`~xarray.DataArray.interpolate_na` method
for filling missing values via 1D interpolation.

.. ipython:: python

    x = xr.DataArray([0, 1, np.nan, np.nan, 2], dims=['x'],
                     coords={'xx': xr.Variable('x', [0, 1, 1.1, 1.9, 3])})
    x.interpolate_na(dim='x', method='linear', use_coordinate='xx')

Note that xarray slightly diverges from the pandas ``interpolate`` syntax by
providing the ``use_coordinate`` keyword which facilitates a clear specification
of which values to use as the index in the interpolation.

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

Aggregation and summary methods can be applied directly to the ``Rolling``
object:

.. ipython:: python

    r = arr.rolling(y=3)
    r.reduce(np.std)
    r.mean()

Aggregation results are assigned the coordinate at the end of each window by
default, but can be centered by passing ``center=True`` when constructing the
``Rolling`` object:

.. ipython:: python

    r = arr.rolling(y=3, center=True)
    r.mean()

As can be seen above, aggregations of windows which overlap the border of the
array produce ``nan``\s.  Setting ``min_periods`` in the call to ``rolling``
changes the minimum number of observations within the window required to have
a value when aggregating:

.. ipython:: python

    r = arr.rolling(y=3, min_periods=2)
    r.mean()
    r = arr.rolling(y=3, center=True, min_periods=2)
    r.mean()

.. tip::

   Note that rolling window aggregations are faster and use less memory when bottleneck_ is installed. This only applies to numpy-backed xarray objects.

.. _bottleneck: https://github.com/kwgoodman/bottleneck/

We can also manually iterate through ``Rolling`` objects:

.. code:: python

   for label, arr_window in r:
      # arr_window is a view of x

.. _comput.rolling_exp:

While ``rolling`` provides a simple moving average, ``DataArray`` also supports
an exponential moving average with :py:meth:`~xarray.DataArray.rolling_exp`.
This is similiar to pandas' ``ewm`` method. numbagg_ is required.

.. _numbagg: https://github.com/shoyer/numbagg

.. code:: python

    arr.rolling_exp(y=3).mean()

The ``rolling_exp`` method takes a ``window_type`` kwarg, which can be ``'alpha'``,
``'com'`` (for ``center-of-mass``), ``'span'``, and ``'halflife'``. The default is
``span``.

Finally, the rolling object has a ``construct`` method which returns a
view of the original ``DataArray`` with the windowed dimension in
the last position.
You can use this for more advanced rolling operations such as strided rolling,
windowed rolling, convolution, short-time FFT etc.

.. ipython:: python

    # rolling with 2-point stride
    rolling_da = r.construct('window_dim', stride=2)
    rolling_da
    rolling_da.mean('window_dim', skipna=False)

Because the ``DataArray`` given by ``r.construct('window_dim')`` is a view
of the original array, it is memory efficient.
You can also use ``construct`` to compute a weighted rolling sum:

.. ipython:: python

   weight = xr.DataArray([0.25, 0.5, 0.25], dims=['window'])
   arr.rolling(y=3).construct('window').dot(weight)

.. note::
  numpy's Nan-aggregation functions such as ``nansum`` copy the original array.
  In xarray, we internally use these functions in our aggregation methods
  (such as ``.sum()``) if ``skipna`` argument is not specified or set to True.
  This means ``rolling_da.mean('window_dim')`` is memory inefficient.
  To avoid this, use ``skipna=False`` as the above example.


.. _comput.coarsen:

Coarsen large arrays
====================

``DataArray`` and ``Dataset`` objects include a
:py:meth:`~xarray.DataArray.coarsen` and :py:meth:`~xarray.Dataset.coarsen`
methods. This supports the block aggregation along multiple dimensions,

.. ipython:: python

  x = np.linspace(0, 10, 300)
  t = pd.date_range('15/12/1999', periods=364)
  da = xr.DataArray(np.sin(x) * np.cos(np.linspace(0, 1, 364)[:, np.newaxis]),
                    dims=['time', 'x'], coords={'time': t, 'x': x})
  da

In order to take a block mean for every 7 days along ``time`` dimension and
every 2 points along ``x`` dimension,

.. ipython:: python

  da.coarsen(time=7, x=2).mean()

:py:meth:`~xarray.DataArray.coarsen` raises an ``ValueError`` if the data
length is not a multiple of the corresponding window size.
You can choose ``boundary='trim'`` or ``boundary='pad'`` options for trimming
the excess entries or padding ``nan`` to insufficient entries,

.. ipython:: python

  da.coarsen(time=30, x=2, boundary='trim').mean()

If you want to apply a specific function to coordinate, you can pass the
function or method name to ``coord_func`` option,

.. ipython:: python

  da.coarsen(time=7, x=2, coord_func={'time': 'min'}).mean()


.. _compute.using_coordinates:

Computation using Coordinates
=============================

Xarray objects have some handy methods for the computation with their
coordinates. :py:meth:`~xarray.DataArray.differentiate` computes derivatives by
central finite differences using their coordinates,

.. ipython:: python

    a = xr.DataArray([0, 1, 2, 3], dims=['x'], coords=[[0.1, 0.11, 0.2, 0.3]])
    a
    a.differentiate('x')

This method can be used also for multidimensional arrays,

.. ipython:: python

    a = xr.DataArray(np.arange(8).reshape(4, 2), dims=['x', 'y'],
                     coords={'x': [0.1, 0.11, 0.2, 0.3]})
    a.differentiate('x')

:py:meth:`~xarray.DataArray.integrate` computes integration based on
trapezoidal rule using their coordinates,

.. ipython:: python

    a.integrate('x')

.. note::
    These methods are limited to simple cartesian geometry. Differentiation
    and integration along multidimensional coordinate are not supported.


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

You can explicitly broadcast xarray data structures by using the
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

.. ipython::
    :verbatim:

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

Datasets also support NumPy ufuncs (requires NumPy v1.13 or newer), or
alternatively you can use :py:meth:`~xarray.Dataset.apply` to apply a function
to each variable in a dataset:

.. ipython:: python

    np.sin(ds)
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
matching data variables.

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
