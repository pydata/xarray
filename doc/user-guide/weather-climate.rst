.. currentmodule:: xarray

.. _weather-climate:

Weather and climate data
========================

.. ipython:: python
    :suppress:

    import xarray as xr

Xarray can leverage metadata that follows the `Climate and Forecast (CF) conventions`_ if present. Examples include :ref:`automatic labelling of plots<plotting>` with descriptive names and units if proper metadata is present and support for non-standard calendars used in climate science through the ``cftime`` module (explained in the :ref:`CFTimeIndex` section). There are also a number of :ref:`geosciences-focused projects that build on xarray<ecosystem>`.

.. _Climate and Forecast (CF) conventions: https://cfconventions.org

.. _cf_variables:

Related Variables
-----------------

Several CF variable attributes contain lists of other variables
associated with the variable with the attribute.  A few of these are
now parsed by xarray, with the attribute value popped to encoding on
read and the variables in that value interpreted as non-dimension
coordinates:

- ``coordinates``
- ``bounds``
- ``grid_mapping``
- ``climatology``
- ``geometry``
- ``node_coordinates``
- ``node_count``
- ``part_node_count``
- ``interior_ring``
- ``cell_measures``
- ``formula_terms``

This decoding is controlled by the ``decode_coords`` kwarg to
:py:func:`open_dataset` and :py:func:`open_mfdataset`.

The CF attribute ``ancillary_variables`` was not included in the list
due to the variables listed there being associated primarily with the
variable with the attribute, rather than with the dimensions.

.. _metpy_accessor:

CF-compliant coordinate variables
---------------------------------

`MetPy`_ adds a ``metpy`` accessor that allows accessing coordinates with appropriate CF metadata using generic names ``x``, ``y``, ``vertical`` and ``time``. There is also a ``cartopy_crs`` attribute that provides projection information, parsed from the appropriate CF metadata, as a `Cartopy`_ projection object. See the `metpy documentation`_ for more information.

.. _`MetPy`: https://unidata.github.io/MetPy/dev/index.html
.. _`metpy documentation`:	https://unidata.github.io/MetPy/dev/tutorials/xarray_tutorial.html#coordinates
.. _`Cartopy`: https://scitools.org.uk/cartopy/docs/latest/crs/projections.html

.. _CFTimeIndex:

Non-standard calendars and dates outside the precision range
------------------------------------------------------------

Through the standalone ``cftime`` library and a custom subclass of
:py:class:`pandas.Index`, xarray supports a subset of the indexing
functionality enabled through the standard :py:class:`pandas.DatetimeIndex` for
dates from non-standard calendars commonly used in climate science or dates
using a standard calendar, but outside the `precision range`_ and dates prior to `1582-10-15`_.

.. note::

   As of xarray version 0.11, by default, :py:class:`cftime.datetime` objects
   will be used to represent times (either in indexes, as a
   :py:class:`~xarray.CFTimeIndex`, or in data arrays with dtype object) if
   any of the following are true:

   - The dates are from a non-standard calendar
   - Any dates are outside the nanosecond-precision range (prior xarray version 2025.01.2)
   - Any dates are outside the time span limited by the resolution (from xarray version 2025.01.2)

   Otherwise pandas-compatible dates from a standard calendar will be
   represented with the ``np.datetime64[unit]`` data type (where unit can be one of ``"s"``, ``"ms"``, ``"us"``, ``"ns"``), enabling the use of a :py:class:`pandas.DatetimeIndex` or arrays with dtype ``np.datetime64[unit]`` and their full set of associated features.

   As of pandas version 2.0.0, pandas supports non-nanosecond precision datetime
   values. From xarray version 2025.01.2 on, non-nanosecond precision datetime values are also supported in xarray (this can be parameterized via :py:class:`~xarray.coders.CFDatetimeCoder` and ``decode_times`` kwarg). See also :ref:`internals.timecoding`.

For example, you can create a DataArray indexed by a time
coordinate with dates from a no-leap calendar and a
:py:class:`~xarray.CFTimeIndex` will automatically be used:

.. ipython:: python

    from itertools import product
    from cftime import DatetimeNoLeap

    dates = [
        DatetimeNoLeap(year, month, 1)
        for year, month in product(range(1, 3), range(1, 13))
    ]
    da = xr.DataArray(np.arange(24), coords=[dates], dims=["time"], name="foo")

Xarray also includes a :py:func:`~xarray.date_range` function, which enables
creating a :py:class:`~xarray.CFTimeIndex` with regularly-spaced dates.  For
instance, we can create the same dates and DataArray we created above using
(note that ``use_cftime=True`` is not mandatory to return a
:py:class:`~xarray.CFTimeIndex` for non-standard calendars, but can be nice
to use to be explicit):

.. ipython:: python

    dates = xr.date_range(
        start="0001", periods=24, freq="MS", calendar="noleap", use_cftime=True
    )
    da = xr.DataArray(np.arange(24), coords=[dates], dims=["time"], name="foo")

Mirroring pandas' method with the same name, :py:meth:`~xarray.infer_freq` allows one to
infer the sampling frequency of a :py:class:`~xarray.CFTimeIndex` or a 1-D
:py:class:`~xarray.DataArray` containing cftime objects. It also works transparently with
``np.datetime64`` and ``np.timedelta64`` data (with "s", "ms", "us" or "ns" resolution).

.. ipython:: python

    xr.infer_freq(dates)

With :py:meth:`~xarray.CFTimeIndex.strftime` we can also easily generate formatted strings from
the datetime values of a :py:class:`~xarray.CFTimeIndex` directly or through the
``dt`` accessor for a :py:class:`~xarray.DataArray`
using the same formatting as the standard `datetime.strftime`_ convention .

.. _datetime.strftime: https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

.. ipython:: python

    dates.strftime("%c")
    da["time"].dt.strftime("%Y%m%d")

Conversion between non-standard calendar and to/from pandas DatetimeIndexes is
facilitated with the :py:meth:`xarray.Dataset.convert_calendar` method (also available as
:py:meth:`xarray.DataArray.convert_calendar`). Here, like elsewhere in xarray, the ``use_cftime``
argument controls which datetime backend is used in the output. The default (``None``) is to
use ``pandas`` when possible, i.e. when the calendar is ``standard``/``gregorian`` and dates starting with `1582-10-15`_. There is no such restriction when converting to a ``proleptic_gregorian`` calendar.

.. _1582-10-15: https://en.wikipedia.org/wiki/Gregorian_calendar

.. ipython:: python

    dates = xr.date_range(
        start="2001", periods=24, freq="MS", calendar="noleap", use_cftime=True
    )
    da_nl = xr.DataArray(np.arange(24), coords=[dates], dims=["time"], name="foo")
    da_std = da.convert_calendar("standard", use_cftime=True)

The data is unchanged, only the timestamps are modified. Further options are implemented
for the special ``"360_day"`` calendar and for handling missing dates. There is also
:py:meth:`xarray.Dataset.interp_calendar` (and :py:meth:`xarray.DataArray.interp_calendar`)
for interpolating data between calendars.

For data indexed by a :py:class:`~xarray.CFTimeIndex` xarray currently supports:

- `Partial datetime string indexing`_:

.. ipython:: python

    da.sel(time="0001")
    da.sel(time=slice("0001-05", "0002-02"))

.. note::


   For specifying full or partial datetime strings in cftime
   indexing, xarray supports two versions of the `ISO 8601 standard`_, the
   basic pattern (YYYYMMDDhhmmss) or the extended pattern
   (YYYY-MM-DDThh:mm:ss), as well as the default cftime string format
   (YYYY-MM-DD hh:mm:ss).  This is somewhat more restrictive than pandas;
   in other words, some datetime strings that would be valid for a
   :py:class:`pandas.DatetimeIndex` are not valid for an
   :py:class:`~xarray.CFTimeIndex`.

- Access of basic datetime components via the ``dt`` accessor (in this case
  just "year", "month", "day", "hour", "minute", "second", "microsecond",
  "season", "dayofyear", "dayofweek", and "days_in_month") with the addition
  of "calendar", absent from pandas:

.. ipython:: python

    da.time.dt.year
    da.time.dt.month
    da.time.dt.season
    da.time.dt.dayofyear
    da.time.dt.dayofweek
    da.time.dt.days_in_month
    da.time.dt.calendar

- Rounding of datetimes to fixed frequencies via the ``dt`` accessor:

.. ipython:: python

    da.time.dt.ceil("3D")
    da.time.dt.floor("5D")
    da.time.dt.round("2D")

- Group-by operations based on datetime accessor attributes (e.g. by month of
  the year):

.. ipython:: python

    da.groupby("time.month").sum()

- Interpolation using :py:class:`cftime.datetime` objects:

.. ipython:: python

    da.interp(time=[DatetimeNoLeap(1, 1, 15), DatetimeNoLeap(1, 2, 15)])

- Interpolation using datetime strings:

.. ipython:: python

    da.interp(time=["0001-01-15", "0001-02-15"])

- Differentiation:

.. ipython:: python

    da.differentiate("time")

- Serialization:

.. ipython:: python

    da.to_netcdf("example-no-leap.nc")
    reopened = xr.open_dataset("example-no-leap.nc")
    reopened

.. ipython:: python
    :suppress:

    import os

    reopened.close()
    os.remove("example-no-leap.nc")

- And resampling along the time dimension for data indexed by a :py:class:`~xarray.CFTimeIndex`:

.. ipython:: python

    da.resample(time="81min", closed="right", label="right", offset="3min").mean()

.. _precision range: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timestamp-limitations
.. _ISO 8601 standard: https://en.wikipedia.org/wiki/ISO_8601
.. _partial datetime string indexing: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#partial-string-indexing
