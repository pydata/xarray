.. currentmodule:: xarray

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

    arr = xr.DataArray(
        np.random.RandomState(0).randn(2, 3), [("x", ["a", "b"]), ("y", [10, 20, 30])]
    )
    arr - 3
    abs(arr)

You can also use any of numpy's or scipy's many `ufunc`__ functions directly on
a DataArray:

__ https://numpy.org/doc/stable/reference/ufuncs.html

.. ipython:: python

    np.sin(arr)

Use :py:func:`~xarray.where` to conditionally switch between values:

.. ipython:: python

    xr.where(arr > 0, "positive", "negative")

Use ``@`` to compute the :py:func:`~xarray.dot` product:

.. ipython:: python

    arr @ arr

Data arrays also implement many :py:class:`numpy.ndarray` methods:

.. ipython:: python

    arr.round(2)
    arr.T

    intarr = xr.DataArray([0, 1, 2, 3, 4, 5])
    intarr << 2  # only supported for int types
    intarr >> 1

.. _missing_values:

Missing values
==============

Xarray represents missing values using the "NaN" (Not a Number) value from NumPy, which is a
special floating-point value that indicates a value that is undefined or unrepresentable.
There are several methods for handling missing values in xarray:

Xarray objects borrow the :py:meth:`~xarray.DataArray.isnull`,
:py:meth:`~xarray.DataArray.notnull`, :py:meth:`~xarray.DataArray.count`,
:py:meth:`~xarray.DataArray.dropna`, :py:meth:`~xarray.DataArray.fillna`,
:py:meth:`~xarray.DataArray.ffill`, and :py:meth:`~xarray.DataArray.bfill`
methods for working with missing data from pandas:

:py:meth:`~xarray.DataArray.isnull` is a method in xarray that can be used to check for missing or null values in an xarray object.
It returns a new xarray object with the same dimensions as the original object, but with boolean values
indicating where **missing values** are present.

.. ipython:: python

    x = xr.DataArray([0, 1, np.nan, np.nan, 2], dims=["x"])
    x.isnull()

In this example, the third and fourth elements of 'x' are NaN, so the resulting :py:class:`~xarray.DataArray`
object has 'True' values in the third and fourth positions and 'False' values in the other positions.

:py:meth:`~xarray.DataArray.notnull` is a method in xarray that can be used to check for non-missing or non-null values in an xarray
object. It returns a new xarray object with the same dimensions as the original object, but with boolean
values indicating where **non-missing values** are present.

.. ipython:: python

    x = xr.DataArray([0, 1, np.nan, np.nan, 2], dims=["x"])
    x.notnull()

In this example, the first two and the last elements of x are not NaN, so the resulting
:py:class:`~xarray.DataArray` object has 'True' values in these positions, and 'False' values in the
third and fourth positions where NaN is located.

:py:meth:`~xarray.DataArray.count` is a method in xarray that can be used to count the number of
non-missing values along one or more dimensions of an xarray object. It returns a new xarray object with
the same dimensions as the original object, but with each element replaced by the count of non-missing
values along the specified dimensions.

.. ipython:: python

    x = xr.DataArray([0, 1, np.nan, np.nan, 2], dims=["x"])
    x.count()

In this example, 'x' has five elements, but two of them are NaN, so the resulting
:py:class:`~xarray.DataArray` object having a single element containing the value '3', which represents
the number of non-null elements in x.

:py:meth:`~xarray.DataArray.dropna` is a method in xarray that can be used to remove missing or null values from an xarray object.
It returns a new xarray object with the same dimensions as the original object, but with missing values
removed.

.. ipython:: python

    x = xr.DataArray([0, 1, np.nan, np.nan, 2], dims=["x"])
    x.dropna(dim="x")

In this example, on calling x.dropna(dim="x") removes any missing values and returns a new
:py:class:`~xarray.DataArray` object with only the non-null elements [0, 1, 2] of 'x', in the
original order.

:py:meth:`~xarray.DataArray.fillna` is a method in xarray that can be used to fill missing or null values in an xarray object with a
specified value or method. It returns a new xarray object with the same dimensions as the original object, but with missing values filled.

.. ipython:: python

    x = xr.DataArray([0, 1, np.nan, np.nan, 2], dims=["x"])
    x.fillna(-1)

In this example, there are two NaN values in 'x', so calling x.fillna(-1) replaces these values with -1 and
returns a new :py:class:`~xarray.DataArray` object with five elements, containing the values
[0, 1, -1, -1, 2] in the original order.

:py:meth:`~xarray.DataArray.ffill` is a method in xarray that can be used to forward fill (or fill forward) missing values in an
xarray object along one or more dimensions. It returns a new xarray object with the same dimensions as the
original object, but with missing values replaced by the last non-missing value along the specified dimensions.

.. ipython:: python

    x = xr.DataArray([0, 1, np.nan, np.nan, 2], dims=["x"])
    x.ffill("x")

In this example, there are two NaN values in 'x', so calling x.ffill("x") fills these values with the last
non-null value in the same dimension, which are 0 and 1, respectively. The resulting :py:class:`~xarray.DataArray` object has
five elements, containing the values [0, 1, 1, 1, 2] in the original order.

:py:meth:`~xarray.DataArray.bfill` is a method in xarray that can be used to backward fill (or fill backward) missing values in an
xarray object along one or more dimensions. It returns a new xarray object with the same dimensions as the original object, but
with missing values replaced by the next non-missing value along the specified dimensions.

.. ipython:: python

    x = xr.DataArray([0, 1, np.nan, np.nan, 2], dims=["x"])
    x.bfill("x")

In this example, there are two NaN values in 'x', so calling x.bfill("x") fills these values with the next
non-null value in the same dimension, which are 2 and 2, respectively. The resulting :py:class:`~xarray.DataArray` object has
five elements, containing the values [0, 1, 2, 2, 2] in the original order.

Like pandas, xarray uses the float value ``np.nan`` (not-a-number) to represent
missing values.

Xarray objects also have an :py:meth:`~xarray.DataArray.interpolate_na` method
for filling missing values via 1D interpolation. It returns a new xarray object with the same dimensions
as the original object, but with missing values interpolated.

.. ipython:: python

    x = xr.DataArray(
        [0, 1, np.nan, np.nan, 2],
        dims=["x"],
        coords={"xx": xr.Variable("x", [0, 1, 1.1, 1.9, 3])},
    )
    x.interpolate_na(dim="x", method="linear", use_coordinate="xx")

In this example, there are two NaN values in 'x', so calling x.interpolate_na(dim="x", method="linear",
use_coordinate="xx") fills these values with interpolated values along the "x" dimension using linear
interpolation based on the values of the xx coordinate. The resulting :py:class:`~xarray.DataArray` object has five elements,
containing the values [0., 1., 1.05, 1.45, 2.] in the original order. Note that the interpolated values
are calculated based on the values of the 'xx' coordinate, which has non-integer values, resulting in
non-integer interpolated values.

Note that xarray slightly diverges from the pandas ``interpolate`` syntax by
providing the ``use_coordinate`` keyword which facilitates a clear specification
of which values to use as the index in the interpolation.
Xarray also provides the ``max_gap`` keyword argument to limit the interpolation to
data gaps of length ``max_gap`` or smaller. See :py:meth:`~xarray.DataArray.interpolate_na`
for more.

.. _agg:

Aggregation
===========

Aggregation methods have been updated to take a ``dim`` argument instead of
``axis``. This allows for very intuitive syntax for aggregation methods that are
applied along particular dimension(s):

.. ipython:: python

    arr.sum(dim="x")
    arr.std(["x", "y"])
    arr.min()


If you need to figure out the axis number for a dimension yourself (say,
for wrapping code designed to work with numpy arrays), you can use the
:py:meth:`~xarray.DataArray.get_axis_num` method:

.. ipython:: python

    arr.get_axis_num("y")

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

    arr = xr.DataArray(np.arange(0, 7.5, 0.5).reshape(3, 5), dims=("x", "y"))
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

From version 0.17, xarray supports multidimensional rolling,

.. ipython:: python

    r = arr.rolling(x=2, y=3, min_periods=2)
    r.mean()

.. tip::

   Note that rolling window aggregations are faster and use less memory when bottleneck_ is installed. This only applies to numpy-backed xarray objects with 1d-rolling.

.. _bottleneck: https://github.com/pydata/bottleneck

We can also manually iterate through ``Rolling`` objects:

.. code:: python

    for label, arr_window in r:
        # arr_window is a view of x
        ...

.. _comput.rolling_exp:

While ``rolling`` provides a simple moving average, ``DataArray`` also supports
an exponential moving average with :py:meth:`~xarray.DataArray.rolling_exp`.
This is similar to pandas' ``ewm`` method. numbagg_ is required.

.. _numbagg: https://github.com/numbagg/numbagg

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
    rolling_da = r.construct(x="x_win", y="y_win", stride=2)
    rolling_da
    rolling_da.mean(["x_win", "y_win"], skipna=False)

Because the ``DataArray`` given by ``r.construct('window_dim')`` is a view
of the original array, it is memory efficient.
You can also use ``construct`` to compute a weighted rolling sum:

.. ipython:: python

    weight = xr.DataArray([0.25, 0.5, 0.25], dims=["window"])
    arr.rolling(y=3).construct(y="window").dot(weight)

.. note::
  numpy's Nan-aggregation functions such as ``nansum`` copy the original array.
  In xarray, we internally use these functions in our aggregation methods
  (such as ``.sum()``) if ``skipna`` argument is not specified or set to True.
  This means ``rolling_da.mean('window_dim')`` is memory inefficient.
  To avoid this, use ``skipna=False`` as the above example.


.. _comput.weighted:

Weighted array reductions
=========================

:py:class:`DataArray` and :py:class:`Dataset` objects include :py:meth:`DataArray.weighted`
and :py:meth:`Dataset.weighted` array reduction methods. They currently
support weighted ``sum``, ``mean``, ``std``, ``var`` and ``quantile``.

.. ipython:: python

    coords = dict(month=("month", [1, 2, 3]))

    prec = xr.DataArray([1.1, 1.0, 0.9], dims=("month",), coords=coords)
    weights = xr.DataArray([31, 28, 31], dims=("month",), coords=coords)

Create a weighted object:

.. ipython:: python

    weighted_prec = prec.weighted(weights)
    weighted_prec

Calculate the weighted sum:

.. ipython:: python

    weighted_prec.sum()

Calculate the weighted mean:

.. ipython:: python

    weighted_prec.mean(dim="month")

Calculate the weighted quantile:

.. ipython:: python

    weighted_prec.quantile(q=0.5, dim="month")

The weighted sum corresponds to:

.. ipython:: python

    weighted_sum = (prec * weights).sum()
    weighted_sum

the weighted mean to:

.. ipython:: python

    weighted_mean = weighted_sum / weights.sum()
    weighted_mean

the weighted variance to:

.. ipython:: python

    weighted_var = weighted_prec.sum_of_squares() / weights.sum()
    weighted_var

and the weighted standard deviation to:

.. ipython:: python

    weighted_std = np.sqrt(weighted_var)
    weighted_std

However, the functions also take missing values in the data into account:

.. ipython:: python

    data = xr.DataArray([np.nan, 2, 4])
    weights = xr.DataArray([8, 1, 1])

    data.weighted(weights).mean()

Using ``(data * weights).sum() / weights.sum()`` would (incorrectly) result
in 0.6.


If the weights add up to to 0, ``sum`` returns 0:

.. ipython:: python

    data = xr.DataArray([1.0, 1.0])
    weights = xr.DataArray([-1.0, 1.0])

    data.weighted(weights).sum()

and ``mean``, ``std`` and ``var`` return ``nan``:

.. ipython:: python

    data.weighted(weights).mean()


.. note::
  ``weights`` must be a :py:class:`DataArray` and cannot contain missing values.
  Missing values can be replaced manually by ``weights.fillna(0)``.

.. _compute.coarsen:

Coarsen large arrays
====================

:py:class:`DataArray` and :py:class:`Dataset` objects include a
:py:meth:`~xarray.DataArray.coarsen` and :py:meth:`~xarray.Dataset.coarsen`
methods. This supports block aggregation along multiple dimensions,

.. ipython:: python

    x = np.linspace(0, 10, 300)
    t = pd.date_range("1999-12-15", periods=364)
    da = xr.DataArray(
        np.sin(x) * np.cos(np.linspace(0, 1, 364)[:, np.newaxis]),
        dims=["time", "x"],
        coords={"time": t, "x": x},
    )
    da

In order to take a block mean for every 7 days along ``time`` dimension and
every 2 points along ``x`` dimension,

.. ipython:: python

    da.coarsen(time=7, x=2).mean()

:py:meth:`~xarray.DataArray.coarsen` raises a ``ValueError`` if the data
length is not a multiple of the corresponding window size.
You can choose ``boundary='trim'`` or ``boundary='pad'`` options for trimming
the excess entries or padding ``nan`` to insufficient entries,

.. ipython:: python

    da.coarsen(time=30, x=2, boundary="trim").mean()

If you want to apply a specific function to coordinate, you can pass the
function or method name to ``coord_func`` option,

.. ipython:: python

    da.coarsen(time=7, x=2, coord_func={"time": "min"}).mean()

You can also :ref:`use coarsen to reshape<reshape.coarsen>` without applying a computation.

.. _compute.using_coordinates:

Computation using Coordinates
=============================

Xarray objects have some handy methods for the computation with their
coordinates. :py:meth:`~xarray.DataArray.differentiate` computes derivatives by
central finite differences using their coordinates,

.. ipython:: python

    a = xr.DataArray([0, 1, 2, 3], dims=["x"], coords=[[0.1, 0.11, 0.2, 0.3]])
    a
    a.differentiate("x")

This method can be used also for multidimensional arrays,

.. ipython:: python

    a = xr.DataArray(
        np.arange(8).reshape(4, 2), dims=["x", "y"], coords={"x": [0.1, 0.11, 0.2, 0.3]}
    )
    a.differentiate("x")

:py:meth:`~xarray.DataArray.integrate` computes integration based on
trapezoidal rule using their coordinates,

.. ipython:: python

    a.integrate("x")

.. note::
    These methods are limited to simple cartesian geometry. Differentiation
    and integration along multidimensional coordinate are not supported.


.. _compute.polyfit:

Fitting polynomials
===================

Xarray objects provide an interface for performing linear or polynomial regressions
using the least-squares method. :py:meth:`~xarray.DataArray.polyfit` computes the
best fitting coefficients along a given dimension and for a given order,

.. ipython:: python

    x = xr.DataArray(np.arange(10), dims=["x"], name="x")
    a = xr.DataArray(3 + 4 * x, dims=["x"], coords={"x": x})
    out = a.polyfit(dim="x", deg=1, full=True)
    out

The method outputs a dataset containing the coefficients (and more if ``full=True``).
The inverse operation is done with :py:meth:`~xarray.polyval`,

.. ipython:: python

    xr.polyval(coord=x, coeffs=out.polyfit_coefficients)

.. note::
    These methods replicate the behaviour of :py:func:`numpy.polyfit` and :py:func:`numpy.polyval`.


.. _compute.curvefit:

Fitting arbitrary functions
===========================

Xarray objects also provide an interface for fitting more complex functions using
:py:func:`scipy.optimize.curve_fit`. :py:meth:`~xarray.DataArray.curvefit` accepts
user-defined functions and can fit along multiple coordinates.

For example, we can fit a relationship between two ``DataArray`` objects, maintaining
a unique fit at each spatial coordinate but aggregating over the time dimension:

.. ipython:: python

    def exponential(x, a, xc):
        return np.exp((x - xc) / a)


    x = np.arange(-5, 5, 0.1)
    t = np.arange(-5, 5, 0.1)
    X, T = np.meshgrid(x, t)
    Z1 = np.random.uniform(low=-5, high=5, size=X.shape)
    Z2 = exponential(Z1, 3, X)
    Z3 = exponential(Z1, 1, -X)

    ds = xr.Dataset(
        data_vars=dict(
            var1=(["t", "x"], Z1), var2=(["t", "x"], Z2), var3=(["t", "x"], Z3)
        ),
        coords={"t": t, "x": x},
    )
    ds[["var2", "var3"]].curvefit(
        coords=ds.var1,
        func=exponential,
        reduce_dims="t",
        bounds={"a": (0.5, 5), "xc": (-5, 5)},
    )

We can also fit multi-dimensional functions, and even use a wrapper function to
simultaneously fit a summation of several functions, such as this field containing
two gaussian peaks:

.. ipython:: python

    def gaussian_2d(coords, a, xc, yc, xalpha, yalpha):
        x, y = coords
        z = a * np.exp(
            -np.square(x - xc) / 2 / np.square(xalpha)
            - np.square(y - yc) / 2 / np.square(yalpha)
        )
        return z


    def multi_peak(coords, *args):
        z = np.zeros(coords[0].shape)
        for i in range(len(args) // 5):
            z += gaussian_2d(coords, *args[i * 5 : i * 5 + 5])
        return z


    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(x, y)

    n_peaks = 2
    names = ["a", "xc", "yc", "xalpha", "yalpha"]
    names = [f"{name}{i}" for i in range(n_peaks) for name in names]
    Z = gaussian_2d((X, Y), 3, 1, 1, 2, 1) + gaussian_2d((X, Y), 2, -1, -2, 1, 1)
    Z += np.random.normal(scale=0.1, size=Z.shape)

    da = xr.DataArray(Z, dims=["y", "x"], coords={"y": y, "x": x})
    da.curvefit(
        coords=["x", "y"],
        func=multi_peak,
        param_names=names,
        kwargs={"maxfev": 10000},
    )

.. note::
    This method replicates the behavior of :py:func:`scipy.optimize.curve_fit`.


.. _compute.broadcasting:

Broadcasting by dimension name
==============================

``DataArray`` objects automatically align themselves ("broadcasting" in
the numpy parlance) by dimension name instead of axis order. With xarray, you
do not need to transpose arrays or insert dimensions of length 1 to get array
operations to work, as commonly done in numpy with :py:func:`numpy.reshape` or
:py:data:`numpy.newaxis`.

This is best illustrated by a few examples. Consider two one-dimensional
arrays with different sizes aligned along different dimensions:

.. ipython:: python

    a = xr.DataArray([1, 2], [("x", ["a", "b"])])
    a
    b = xr.DataArray([-1, -2, -3], [("y", [10, 20, 30])])
    b

With xarray, we can apply binary mathematical operations to these arrays, and
their dimensions are expanded automatically:

.. ipython:: python

    a * b

Moreover, dimensions are always reordered to the order in which they first
appeared:

.. ipython:: python

    c = xr.DataArray(np.arange(6).reshape(3, 2), [b["y"], a["x"]])
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

Xarray enforces alignment between *index* :ref:`coordinates` (that is,
coordinates with the same name as a dimension, marked by ``*``) on objects used
in binary operations.

Similarly to pandas, this alignment is automatic for arithmetic on binary
operations. The default result of a binary operation is by the *intersection*
(not the union) of coordinate labels:

.. ipython:: python

    arr = xr.DataArray(np.arange(3), [("x", range(3))])
    arr + arr[:-1]

If coordinate values for a dimension are missing on either argument, all
matching dimensions must have the same size:

.. ipython::
    :verbatim:

    In [1]: arr + xr.DataArray([1, 2], dims="x")
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

    ds = xr.Dataset(
        {
            "x_and_y": (("x", "y"), np.random.randn(3, 5)),
            "x_only": ("x", np.random.randn(3)),
        },
        coords=arr.coords,
    )
    ds > 0

Datasets support most of the same methods found on data arrays:

.. ipython:: python

    ds.mean(dim="x")
    abs(ds)

Datasets also support NumPy ufuncs (requires NumPy v1.13 or newer), or
alternatively you can use :py:meth:`~xarray.Dataset.map` to map a function
to each variable in a dataset:

.. ipython:: python

    np.sin(ds)
    ds.map(np.sin)

Datasets also use looping over variables for *broadcasting* in binary
arithmetic. You can do arithmetic between any ``DataArray`` and a dataset:

.. ipython:: python

    ds + arr

Arithmetic between two datasets matches data variables of the same name:

.. ipython:: python

    ds2 = xr.Dataset({"x_and_y": 0, "x_only": 100})
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
`universal function <https://numpy.org/doc/stable/reference/ufuncs.html>`_ ("ufunc" for short).
``apply_ufunc`` takes care of everything needed for an idiomatic xarray wrapper,
including alignment, broadcasting, looping over ``Dataset`` variables (if
needed), and merging of coordinates. In fact, many internal xarray
functions/methods are written using ``apply_ufunc``.

Simple functions that act independently on each value should work without
any additional arguments:

.. ipython:: python

    squared_error = lambda x, y: (x - y) ** 2
    arr1 = xr.DataArray([0, 1, 2, 3], dims="x")
    xr.apply_ufunc(squared_error, arr1, 1)

For using more complex operations that consider some array values collectively,
it's important to understand the idea of "core dimensions" from NumPy's
`generalized ufuncs <https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html>`_. Core dimensions are defined as dimensions
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
        return xr.apply_ufunc(
            np.linalg.norm, x, input_core_dims=[[dim]], kwargs={"ord": ord, "axis": -1}
        )

.. ipython:: python
    :suppress:

    def vector_norm(x, dim, ord=None):
        return xr.apply_ufunc(
            np.linalg.norm, x, input_core_dims=[[dim]], kwargs={"ord": ord, "axis": -1}
        )

.. ipython:: python

    vector_norm(arr1, dim="x")

Because ``apply_ufunc`` follows a standard convention for ufuncs, it plays
nicely with tools for building vectorized functions, like
:py:func:`numpy.broadcast_arrays` and :py:class:`numpy.vectorize`. For high performance
needs, consider using :doc:`Numba's vectorize and guvectorize <numba:user/vectorize>`.

In addition to wrapping functions, ``apply_ufunc`` can automatically parallelize
many functions when using dask by setting ``dask='parallelized'``. See
:ref:`dask.automatic-parallelization` for details.

:py:func:`~xarray.apply_ufunc` also supports some advanced options for
controlling alignment of variables and the form of the result. See the
docstring for full details and more examples.
