.. _time-series:

================
Time series data
================

A major use case for xarray is multi-dimensional time-series data.
Accordingly, we've copied many of features that make working with time-series
data in pandas such a joy to xarray. In most cases, we rely on pandas for the
core functionality.

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr
    np.random.seed(123456)

Creating datetime64 data
------------------------

xarray uses the numpy dtypes ``datetime64[ns]`` and ``timedelta64[ns]`` to
represent datetime data, which offer vectorized (if sometimes buggy) operations
with numpy and smooth integration with pandas.

To convert to or create regular arrays of ``datetime64`` data, we recommend
using :py:func:`pandas.to_datetime` and :py:func:`pandas.date_range`:

.. ipython:: python

    pd.to_datetime(['2000-01-01', '2000-02-02'])
    pd.date_range('2000-01-01', periods=365)

Alternatively, you can supply arrays of Python ``datetime`` objects. These get
converted automatically when used as arguments in xarray objects:

.. ipython:: python

    import datetime
    xr.Dataset({'time': datetime.datetime(2000, 1, 1)})

When reading or writing netCDF files, xarray automatically decodes datetime and
timedelta arrays using `CF conventions`_ (that is, by using a ``units``
attribute like ``'days since 2000-01-01'``).

.. _CF conventions: http://cfconventions.org

.. note::

   When decoding/encoding datetimes for non-standard calendars or for dates
   before year 1678 or after year 2262, xarray uses the `cftime`_ library.
   It was previously packaged with the ``netcdf4-python`` package under the
   name ``netcdftime`` but is now distributed separately. ``cftime`` is an
   :ref:`optional dependency<installing>` of xarray.

.. _cftime: https://unidata.github.io/cftime


You can manual decode arrays in this form by passing a dataset to
:py:func:`~xarray.decode_cf`:

.. ipython:: python

    attrs = {'units': 'hours since 2000-01-01'}
    ds = xr.Dataset({'time': ('time', [0, 1, 2, 3], attrs)})
    xr.decode_cf(ds)

One unfortunate limitation of using ``datetime64[ns]`` is that it limits the
native representation of dates to those that fall between the years 1678 and
2262. When a netCDF file contains dates outside of these bounds, dates will be
returned as arrays of :py:class:`cftime.datetime` objects and a :py:class:`~xarray.CFTimeIndex`
will be used for indexing.  :py:class:`~xarray.CFTimeIndex` enables a subset of
the indexing functionality of a :py:class:`pandas.DatetimeIndex` and is only
fully compatible with the standalone version of ``cftime`` (not the version
packaged with earlier versions ``netCDF4``).  See :ref:`CFTimeIndex` for more
information.

Datetime indexing
-----------------

xarray borrows powerful indexing machinery from pandas (see :ref:`indexing`).

This allows for several useful and succinct forms of indexing, particularly for
`datetime64` data. For example, we support indexing with strings for single
items and with the `slice` object:

.. ipython:: python

    time = pd.date_range('2000-01-01', freq='H', periods=365 * 24)
    ds = xr.Dataset({'foo': ('time', np.arange(365 * 24)), 'time': time})
    ds.sel(time='2000-01')
    ds.sel(time=slice('2000-06-01', '2000-06-10'))

You can also select a particular time by indexing with a
:py:class:`datetime.time` object:

.. ipython:: python

    ds.sel(time=datetime.time(12))

For more details, read the pandas documentation.

Datetime components
-------------------

Similar `to pandas`_, the components of datetime objects contained in a
given ``DataArray`` can be quickly computed using a special ``.dt`` accessor.

.. _to pandas: http://pandas.pydata.org/pandas-docs/stable/basics.html#basics-dt-accessors

.. ipython:: python

    time = pd.date_range('2000-01-01', freq='6H', periods=365 * 4)
    ds = xr.Dataset({'foo': ('time', np.arange(365 * 4)), 'time': time})
    ds.time.dt.hour
    ds.time.dt.dayofweek

The ``.dt`` accessor works on both coordinate dimensions as well as
multi-dimensional data.

xarray also supports a notion of "virtual" or "derived" coordinates for
`datetime components`__ implemented by pandas, including "year", "month",
"day", "hour", "minute", "second", "dayofyear", "week", "dayofweek", "weekday"
and "quarter":

__ http://pandas.pydata.org/pandas-docs/stable/api.html#time-date-components

.. ipython:: python

    ds['time.month']
    ds['time.dayofyear']

For use as a derived coordinate, xarray adds ``'season'`` to the list of
datetime components supported by pandas:

.. ipython:: python

    ds['time.season']
    ds['time'].dt.season

The set of valid seasons consists of 'DJF', 'MAM', 'JJA' and 'SON', labeled by
the first letters of the corresponding months.

You can use these shortcuts with both Datasets and DataArray coordinates.

In addition, xarray supports rounding operations ``floor``, ``ceil``, and ``round``. These operations require that you supply a `rounding frequency as a string argument.`__

__ http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

.. ipython:: python

    ds['time'].dt.floor('D')

.. _resampling:

Resampling and grouped operations
---------------------------------

Datetime components couple particularly well with grouped operations (see
:ref:`groupby`) for analyzing features that repeat over time. Here's how to
calculate the mean by time of day:

.. ipython:: python
   :okwarning:

    ds.groupby('time.hour').mean()

For upsampling or downsampling temporal resolutions, xarray offers a
:py:meth:`~xarray.Dataset.resample` method building on the core functionality
offered by the pandas method of the same name. Resample uses essentially the
same api as ``resample`` `in pandas`_.

.. _in pandas: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#up-and-downsampling

For example, we can downsample our dataset from hourly to 6-hourly:

.. ipython:: python
   :okwarning:

    ds.resample(time='6H')

This will create a specialized ``Resample`` object which saves information
necessary for resampling. All of the reduction methods which work with
``Resample`` objects can also be used for resampling:

.. ipython:: python
   :okwarning:

   ds.resample(time='6H').mean()

You can also supply an arbitrary reduction function to aggregate over each
resampling group:

.. ipython:: python

   ds.resample(time='6H').reduce(np.mean)

For upsampling, xarray provides six methods: ``asfreq``, ``ffill``, ``bfill``, ``pad``,
``nearest`` and ``interpolate``. ``interpolate`` extends ``scipy.interpolate.interp1d``
and supports all of its schemes. All of these resampling operations work on both
Dataset and DataArray objects with an arbitrary number of dimensions.

In order to limit the scope of the methods ``ffill``, ``bfill``, ``pad`` and
``nearest`` the ``tolerance`` argument can be set in coordinate units.
Data that has indices outside of the given ``tolerance`` are set to ``NaN``.

.. ipython:: python

    ds.resample(time='1H').nearest(tolerance='1H')


For more examples of using grouped operations on a time dimension, see
:ref:`toy weather data`.


.. _CFTimeIndex:

Non-standard calendars and dates outside the Timestamp-valid range
------------------------------------------------------------------

Through the standalone ``cftime`` library and a custom subclass of
:py:class:`pandas.Index`, xarray supports a subset of the indexing
functionality enabled through the standard :py:class:`pandas.DatetimeIndex` for
dates from non-standard calendars commonly used in climate science or dates
using a standard calendar, but outside the `Timestamp-valid range`_
(approximately between years 1678 and 2262).

.. note::

   As of xarray version 0.11, by default, :py:class:`cftime.datetime` objects
   will be used to represent times (either in indexes, as a
   :py:class:`~xarray.CFTimeIndex`, or in data arrays with dtype object) if
   any of the following are true:

   - The dates are from a non-standard calendar
   - Any dates are outside the Timestamp-valid range.

   Otherwise pandas-compatible dates from a standard calendar will be
   represented with the ``np.datetime64[ns]`` data type, enabling the use of a
   :py:class:`pandas.DatetimeIndex` or arrays with dtype ``np.datetime64[ns]``
   and their full set of associated features.

For example, you can create a DataArray indexed by a time
coordinate with dates from a no-leap calendar and a
:py:class:`~xarray.CFTimeIndex` will automatically be used:

.. ipython:: python

   from itertools import product
   from cftime import DatetimeNoLeap
   dates = [DatetimeNoLeap(year, month, 1) for year, month in
            product(range(1, 3), range(1, 13))]
   da = xr.DataArray(np.arange(24), coords=[dates], dims=['time'], name='foo')

xarray also includes a :py:func:`~xarray.cftime_range` function, which enables
creating a :py:class:`~xarray.CFTimeIndex` with regularly-spaced dates.  For
instance, we can create the same dates and DataArray we created above using:

.. ipython:: python

   dates = xr.cftime_range(start='0001', periods=24, freq='MS', calendar='noleap')
   da = xr.DataArray(np.arange(24), coords=[dates], dims=['time'], name='foo')

For data indexed by a :py:class:`~xarray.CFTimeIndex` xarray currently supports:

- `Partial datetime string indexing`_ using strictly `ISO 8601-format`_ partial
  datetime strings:

.. ipython:: python

   da.sel(time='0001')
   da.sel(time=slice('0001-05', '0002-02'))

- Access of basic datetime components via the ``dt`` accessor (in this case
  just "year", "month", "day", "hour", "minute", "second", "microsecond",
  "season", "dayofyear", and "dayofweek"):

.. ipython:: python

   da.time.dt.year
   da.time.dt.month
   da.time.dt.season
   da.time.dt.dayofyear
   da.time.dt.dayofweek

- Group-by operations based on datetime accessor attributes (e.g. by month of
  the year):

.. ipython:: python

   da.groupby('time.month').sum()

- Interpolation using :py:class:`cftime.datetime` objects:

.. ipython:: python

   da.interp(time=[DatetimeNoLeap(1, 1, 15), DatetimeNoLeap(1, 2, 15)])

- Interpolation using datetime strings:

.. ipython:: python

   da.interp(time=['0001-01-15', '0001-02-15'])

- Differentiation:

.. ipython:: python

   da.differentiate('time')

- Serialization:

.. ipython:: python

   da.to_netcdf('example-no-leap.nc')
   xr.open_dataset('example-no-leap.nc')

- And resampling along the time dimension for data indexed by a :py:class:`~xarray.CFTimeIndex`:

.. ipython:: python

    da.resample(time='81T', closed='right', label='right', base=3).mean()

.. note::
    

   For some use-cases it may still be useful to convert from
   a :py:class:`~xarray.CFTimeIndex` to a :py:class:`pandas.DatetimeIndex`,
   despite the difference in calendar types. The recommended way of doing this
   is to use the built-in :py:meth:`~xarray.CFTimeIndex.to_datetimeindex`
   method:

   .. ipython:: python
      :okwarning:

       modern_times = xr.cftime_range('2000', periods=24, freq='MS', calendar='noleap')
       da = xr.DataArray(range(24), [('time', modern_times)])
       da
       datetimeindex = da.indexes['time'].to_datetimeindex()
       da['time'] = datetimeindex

   However in this case one should use caution to only perform operations which
   do not depend on differences between dates (e.g. differentiation,
   interpolation, or upsampling with resample), as these could introduce subtle
   and silent errors due to the difference in calendar types between the dates
   encoded in your data and the dates stored in memory.

.. _Timestamp-valid range: https://pandas.pydata.org/pandas-docs/stable/timeseries.html#timestamp-limitations
.. _ISO 8601-format: https://en.wikipedia.org/wiki/ISO_8601
.. _partial datetime string indexing: https://pandas.pydata.org/pandas-docs/stable/timeseries.html#partial-string-indexing
