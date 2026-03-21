.. currentmodule:: xarray

.. _time-series:

================
Time series data
================

A major use case for xarray is multi-dimensional time-series data.
Accordingly, we've copied many of features that make working with time-series
data in pandas such a joy to xarray. In most cases, we rely on pandas for the
core functionality.

.. jupyter-execute::
    :hide-code:

    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)

Creating datetime64 data
------------------------

Xarray uses the numpy dtypes :py:class:`numpy.datetime64` and :py:class:`numpy.timedelta64`
with specified units (one of ``"s"``, ``"ms"``, ``"us"`` and ``"ns"``) to represent datetime
data, which offer vectorized operations with numpy and smooth integration with pandas.

To convert to or create regular arrays of :py:class:`numpy.datetime64` data, we recommend
using :py:func:`pandas.to_datetime`, :py:class:`pandas.DatetimeIndex`, or :py:func:`xarray.date_range`:

.. jupyter-execute::

    pd.to_datetime(["2000-01-01", "2000-02-02"])

.. jupyter-execute::

    pd.DatetimeIndex(
        ["2000-01-01 00:00:00", "2000-02-02 00:00:00"], dtype="datetime64[s]"
    )

.. jupyter-execute::

    xr.date_range("2000-01-01", periods=365)

.. jupyter-execute::

    xr.date_range("2000-01-01", periods=365, unit="s")


.. note::
    Care has to be taken to create the output with the wanted resolution.
    For :py:func:`pandas.date_range` the ``unit``-kwarg has to be specified
    and for :py:func:`pandas.to_datetime` the selection of the resolution
    isn't possible at all. For that :py:class:`pd.DatetimeIndex` can be used
    directly. There is more in-depth information in section
    :ref:`internals.timecoding`.

Alternatively, you can supply arrays of Python ``datetime`` objects. These get
converted automatically when used as arguments in xarray objects (with us-resolution):

.. jupyter-execute::

    import datetime

    xr.Dataset({"time": datetime.datetime(2000, 1, 1)})

When reading or writing netCDF files, xarray automatically decodes datetime and
timedelta arrays using `CF conventions`_ (that is, by using a ``units``
attribute like ``'days since 2000-01-01'``).

.. _CF conventions: https://cfconventions.org

.. note::

   When decoding/encoding datetimes for non-standard calendars or for dates
   before `1582-10-15`_, xarray uses the `cftime`_ library by default.
   It was previously packaged with the ``netcdf4-python`` package under the
   name ``netcdftime`` but is now distributed separately. ``cftime`` is an
   :ref:`optional dependency<installing>` of xarray.

.. _cftime: https://unidata.github.io/cftime
.. _1582-10-15: https://en.wikipedia.org/wiki/Gregorian_calendar


You can manual decode arrays in this form by passing a dataset to
:py:func:`decode_cf`:

.. jupyter-execute::

    attrs = {"units": "hours since 2000-01-01"}
    ds = xr.Dataset({"time": ("time", [0, 1, 2, 3], attrs)})
    # Default decoding to 'ns'-resolution
    xr.decode_cf(ds)

.. jupyter-execute::

    # Decoding to 's'-resolution
    coder = xr.coders.CFDatetimeCoder(time_unit="s")
    xr.decode_cf(ds, decode_times=coder)

From xarray 2025.01.2 the resolution of the dates can be one of ``"s"``, ``"ms"``, ``"us"`` or ``"ns"``. One limitation of using ``datetime64[ns]`` is that it limits the native representation of dates to those that fall between the years 1678 and 2262, which gets increased significantly with lower resolutions. When a store contains dates outside of these bounds (or dates < `1582-10-15`_ with a Gregorian, also known as standard, calendar), dates will be returned as arrays of :py:class:`cftime.datetime` objects and a :py:class:`CFTimeIndex` will be used for indexing.
:py:class:`CFTimeIndex` enables most of the indexing functionality of a :py:class:`pandas.DatetimeIndex`.
See :ref:`CFTimeIndex` for more information.

Datetime indexing
-----------------

Xarray borrows powerful indexing machinery from pandas (see :ref:`indexing`).

This allows for several useful and succinct forms of indexing, particularly for
``datetime64`` data. For example, we support indexing with strings for single
items and with the ``slice`` object:

.. jupyter-execute::

    time = pd.date_range("2000-01-01", freq="h", periods=365 * 24)
    ds = xr.Dataset({"foo": ("time", np.arange(365 * 24)), "time": time})
    ds.sel(time="2000-01")

.. jupyter-execute::

    ds.sel(time=slice("2000-06-01", "2000-06-10"))

You can also select a particular time by indexing with a
:py:class:`datetime.time` object:

.. jupyter-execute::

    ds.sel(time=datetime.time(12))

For more details, read the pandas documentation and the section on :ref:`datetime_component_indexing` (i.e. using the ``.dt`` accessor).

.. _dt_accessor:

Datetime components
-------------------

Similar to `pandas accessors`_, the components of datetime objects contained in a
given ``DataArray`` can be quickly computed using a special ``.dt`` accessor.

.. _pandas accessors: https://pandas.pydata.org/pandas-docs/stable/basics.html#basics-dt-accessors

.. jupyter-execute::

    time = pd.date_range("2000-01-01", freq="6h", periods=365 * 4)
    ds = xr.Dataset({"foo": ("time", np.arange(365 * 4)), "time": time})
    ds.time.dt.hour

.. jupyter-execute::

    ds.time.dt.dayofweek

The ``.dt`` accessor works on both coordinate dimensions as well as
multi-dimensional data.

Xarray also supports a notion of "virtual" or "derived" coordinates for
`datetime components`__ implemented by pandas, including "year", "month",
"day", "hour", "minute", "second", "dayofyear", "week", "dayofweek", "weekday"
and "quarter":

__ https://pandas.pydata.org/pandas-docs/stable/api.html#time-date-components

.. jupyter-execute::

    ds["time.month"]

.. jupyter-execute::

    ds["time.dayofyear"]

For use as a derived coordinate, xarray adds ``'season'`` to the list of
datetime components supported by pandas:

.. jupyter-execute::

    ds["time.season"]

.. jupyter-execute::

    ds["time"].dt.season

The set of valid seasons consists of 'DJF', 'MAM', 'JJA' and 'SON', labeled by
the first letters of the corresponding months.

You can use these shortcuts with both Datasets and DataArray coordinates.

In addition, xarray supports rounding operations ``floor``, ``ceil``, and ``round``. These operations require that you supply a `rounding frequency as a string argument.`__

__ https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

.. jupyter-execute::

    ds["time"].dt.floor("D")

The ``.dt`` accessor can also be used to generate formatted datetime strings
for arrays utilising the same formatting as the standard `datetime.strftime`_.

.. _datetime.strftime: https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

.. jupyter-execute::

    ds["time"].dt.strftime("%a, %b %d %H:%M")

.. _datetime_component_indexing:

Indexing Using Datetime Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can use use the ``.dt`` accessor when subsetting your data as well. For example, we can subset for the month of January using the following:

.. jupyter-execute::

    ds.isel(time=(ds.time.dt.month == 1))

You can also search for multiple months (in this case January through March), using ``isin``:

.. jupyter-execute::

    ds.isel(time=ds.time.dt.month.isin([1, 2, 3]))

.. _resampling:

Resampling and grouped operations
---------------------------------


.. seealso::

   For more generic documentation on grouping, see :ref:`groupby`.


Datetime components couple particularly well with grouped operations for analyzing features that repeat over time.
Here's how to calculate the mean by time of day:

.. jupyter-execute::

    ds.groupby("time.hour").mean()

For upsampling or downsampling temporal resolutions, xarray offers a
:py:meth:`Dataset.resample` method building on the core functionality
offered by the pandas method of the same name. Resample uses essentially the
same api as :py:meth:`pandas.DataFrame.resample` `in pandas`_.

.. _in pandas: https://pandas.pydata.org/pandas-docs/stable/timeseries.html#up-and-downsampling

For example, we can downsample our dataset from hourly to 6-hourly:

.. jupyter-execute::

    ds.resample(time="6h")

This will create a specialized :py:class:`~xarray.core.resample.DatasetResample` or :py:class:`~xarray.core.resample.DataArrayResample`
object which saves information necessary for resampling. All of the reduction methods which work with
:py:class:`Dataset` or :py:class:`DataArray` objects can also be used for resampling:

.. jupyter-execute::

    ds.resample(time="6h").mean()

You can also supply an arbitrary reduction function to aggregate over each
resampling group:

.. jupyter-execute::

    ds.resample(time="6h").reduce(np.mean)

You can also resample on the time dimension while applying reducing along other dimensions at the same time
by specifying the ``dim`` keyword argument

.. code-block:: python

    ds.resample(time="6h").mean(dim=["time", "latitude", "longitude"])

For upsampling, xarray provides six methods: ``asfreq``, ``ffill``, ``bfill``, ``pad``,
``nearest`` and ``interpolate``. ``interpolate`` extends :py:func:`scipy.interpolate.interp1d`
and supports all of its schemes. All of these resampling operations work on both
Dataset and DataArray objects with an arbitrary number of dimensions.

In order to limit the scope of the methods ``ffill``, ``bfill``, ``pad`` and
``nearest`` the ``tolerance`` argument can be set in coordinate units.
Data that has indices outside of the given ``tolerance`` are set to ``NaN``.

.. jupyter-execute::

    ds.resample(time="1h").nearest(tolerance="1h")

It is often desirable to center the time values after a resampling operation.
That can be accomplished by updating the resampled dataset time coordinate values
using time offset arithmetic via the :py:func:`pandas.tseries.frequencies.to_offset` function.

.. jupyter-execute::

    resampled_ds = ds.resample(time="6h").mean()
    offset = pd.tseries.frequencies.to_offset("6h") / 2
    resampled_ds["time"] = resampled_ds.get_index("time") + offset
    resampled_ds


.. seealso::

   For more examples of using grouped operations on a time dimension, see :doc:`../examples/weather-data`.


.. _seasonal_grouping:

Handling Seasons
~~~~~~~~~~~~~~~~

Two extremely common time series operations are to group by seasons, and resample to a seasonal frequency.
Xarray has historically supported some simple versions of these computations.
For example, ``.groupby("time.season")`` (where the seasons are DJF, MAM, JJA, SON)
and resampling to a seasonal frequency using Pandas syntax: ``.resample(time="QS-DEC")``.

Quite commonly one wants more flexibility in defining seasons. For these use-cases, Xarray provides
:py:class:`groupers.SeasonGrouper` and :py:class:`groupers.SeasonResampler`.


.. currentmodule:: xarray.groupers

.. jupyter-execute::

    from xarray.groupers import SeasonGrouper

    ds.groupby(time=SeasonGrouper(["DJF", "MAM", "JJA", "SON"])).mean()


Note how the seasons are in the specified order, unlike ``.groupby("time.season")`` where the
seasons are sorted alphabetically.

.. jupyter-execute::

    ds.groupby("time.season").mean()


:py:class:`SeasonGrouper` supports overlapping seasons:

.. jupyter-execute::

    ds.groupby(time=SeasonGrouper(["DJFM", "MAMJ", "JJAS", "SOND"])).mean()


Skipping months is allowed:

.. jupyter-execute::

    ds.groupby(time=SeasonGrouper(["JJAS"])).mean()


Use :py:class:`SeasonResampler` to specify custom seasons.

.. jupyter-execute::

    from xarray.groupers import SeasonResampler

    ds.resample(time=SeasonResampler(["DJF", "MAM", "JJA", "SON"])).mean()


:py:class:`SeasonResampler` is smart enough to correctly handle years for seasons that
span the end of the year (e.g. DJF). By default :py:class:`SeasonResampler` will skip any
season that is incomplete (e.g. the first DJF season for a time series that starts in Jan).
Pass the ``drop_incomplete=False`` kwarg to :py:class:`SeasonResampler` to disable this behaviour.

.. jupyter-execute::

    from xarray.groupers import SeasonResampler

    ds.resample(
        time=SeasonResampler(["DJF", "MAM", "JJA", "SON"], drop_incomplete=False)
    ).mean()


Seasons need not be of the same length:

.. jupyter-execute::

    ds.resample(time=SeasonResampler(["JF", "MAM", "JJAS", "OND"])).mean()
