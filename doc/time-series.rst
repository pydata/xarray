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

You can manual decode arrays in this form by passing a dataset to
:py:func:`~xarray.decode_cf`:

.. ipython:: python

    attrs = {'units': 'hours since 2000-01-01'}
    ds = xr.Dataset({'time': ('time', [0, 1, 2, 3], attrs)})
    xr.decode_cf(ds)

One unfortunate limitation of using ``datetime64[ns]`` is that it limits the
native representation of dates to those that fall between the years 1678 and
2262. When a netCDF file contains dates outside of these bounds, dates will be
returned as arrays of ``netcdftime.datetime`` objects.

Datetime indexing
-----------------

xarray borrows powerful indexing machinery from pandas (see :ref:`indexing`).

This allows for several useful and suscinct forms of indexing, particularly for
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

.. _resampling:

Resampling and grouped operations
---------------------------------

Datetime components couple particularly well with grouped operations (see
:ref:`groupby`) for analyzing features that repeat over time. Here's how to
calculate the mean by time of day:

.. ipython:: python

    ds.groupby('time.hour').mean()

For upsampling or downsampling temporal resolutions, xarray offers a
:py:meth:`~xarray.Dataset.resample` method building on the core functionality
offered by the pandas method of the same name. Resample uses essentially the
same api as ``resample`` `in pandas`_.

.. _in pandas: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#up-and-downsampling

For example, we can downsample our dataset from hourly to 6-hourly:

.. ipython:: python

    ds.resample(time='6H')

This will create a specialized ``Resample`` object which saves information
necessary for resampling. All of the reduction methods which work with
``Resample`` objects can also be used for resampling:

.. ipython:: python

   ds.resample(time='6H').mean()

You can also supply an arbitrary reduction function to aggregate over each
resampling group:

.. ipython:: python

   ds.resample(time='6H').reduce(np.mean)

For upsampling, xarray provides four methods: ``asfreq``, ``ffill``, ``bfill``,
and ``interpolate``. ``interpolate`` extends ``scipy.interpolate.interp1d`` and
supports all of its schemes. All of these resampling operations work on both
Dataset and DataArray objects with an arbitrary number of dimensions.

.. note::

   The ``resample`` api was updated in version 0.10.0 to reflect similar
   updates in pandas ``resample`` api to be more groupby-like. Older style
   calls to ``resample`` will still be supported for a short period:

   .. ipython:: python

    ds.resample('6H', dim='time', how='mean')


For more examples of using grouped operations on a time dimension, see
:ref:`toy weather data`.
