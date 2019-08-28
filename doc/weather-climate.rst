.. _weather-climate:

Weather and climate data
========================

.. ipython:: python
   :suppress:

    import xarray as xr

``xarray`` can leverage metadata that follows the `Climate and Forecast (CF) conventions`_ if present. Examples include automatic labelling of plots with descriptive names and units if proper metadata is present (see :ref:`plotting`) and support for non-standard calendars used in climate science through the ``cftime`` module (see :ref:`CFTimeIndex`). There are also a number of geosciences-focused projects that build on xarray (see :ref:`related-projects`).

.. _Climate and Forecast (CF) conventions: http://cfconventions.org

.. _metpy_accessor:

CF-compliant coordinate variables
---------------------------------

`MetPy`_ adds a	``metpy`` accessor that allows accessing coordinates with appropriate CF metadata using generic names ``x``, ``y``, ``vertical`` and ``time``. There is also a `cartopy_crs` attribute that provides projection information, parsed from the appropriate CF metadata, as a `Cartopy`_ projection object. See `their documentation`_ for more information.

.. _`MetPy`: https://unidata.github.io/MetPy/dev/index.html
.. _`their documentation`:	https://unidata.github.io/MetPy/dev/tutorials/xarray_tutorial.html#coordinates
.. _`Cartopy`: https://scitools.org.uk/cartopy/docs/latest/crs/projections.html

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

With :py:meth:`~xarray.CFTimeIndex.strftime` we can also easily generate formatted strings from
the datetime values of a :py:class:`~xarray.CFTimeIndex` directly or through the
:py:meth:`~xarray.DataArray.dt` accessor for a :py:class:`~xarray.DataArray`
using the same formatting as the standard `datetime.strftime`_ convention .

.. _datetime.strftime: https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

.. ipython:: python

    dates.strftime('%c')
    da['time'].dt.strftime('%Y%m%d')

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

.. ipython:: python
    :suppress:

    import os
    os.remove('example-no-leap.nc')

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
