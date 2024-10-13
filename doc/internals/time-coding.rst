.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)
    np.set_printoptions(threshold=20)
    int64_max = np.iinfo("int64").max
    int64_min = np.iinfo("int64").min + 1
    uint64_max = np.iinfo("uint64").max

.. internals.timecoding:

Time Coding
===========

This page gives an overview how xarray encodes and decodes times and which conventions and functions are used.

Pandas functionality
--------------------

to_datetime
~~~~~~~~~~~

The function :py:func:`pandas.to_datetime` is used within xarray for inferring units and for testing purposes.

In normal operation :py:func:`pandas.to_datetime` returns :py:class:`pandas.Timestamp` (scalar input) or :py:class:`pandas.DatetimeIndex` (array-like input) which are datetime64 with inherited resolution (from the source). If no resolution can be inherited ``'ns'`` is assumed. That has the implication, that the maximum usable timerange for those cases is +-292 years centered around the epoch. To accommodate for that, we are carefully checking the units/resolution in the encoding and decoding step.

When args are numeric (no strings) "unit" can be anything from ``'Y'``, ``'W'``, ``'D'``, ``'h'``, ``'m'``, ``'s'``, ``'ms'``, ``'us'`` or ``'ns'``.

.. ipython:: python

    f"Maximum datetime range: ({pd.to_datetime(int64_min, unit="ns")}, {pd.to_datetime(int64_max, unit="ns")})"

For input values which can't be represented in nanosecond resolution :py:class:`pandas.OutOfBoundsDatetime` exception is raised:

.. ipython:: python

    try:
        dtime = pd.to_datetime(int64_max, unit="us")
    except Exception as err:
        print(err)
    try:
        dtime = pd.to_datetime(uint64_max, unit="ns")
        print("Wrong:", dtime)
        dtime = pd.to_datetime([uint64_max], unit="ns")
    except Exception as err:
        print(err)

Numpy datetime64 can be extracted with :py:meth:`pandas.Datetime.to_numpy` and :py:meth:`pandas.DatetimeIndex.to_numpy`. The returned resolution depends on the internal representation. This representation can be changed using :py:meth:`pandas.Datetime.as_unit`
and :py:meth:`pandas.DatetimeIndex.as_unit` respectively.

``as_unit`` takes one of ``'s'``, ``'ms'``, ``'us'``, ``'ns'`` as argument. That means we are able to represent datetimes with second, millisecond, microsecond or nanosecond resolution.

.. ipython:: python

    time = pd.to_datetime(np.datetime64(0, "D"))
    print("Datetime:", time, np.asarray([time.to_numpy()]).dtype)
    print("Datetime as_unit('s'):", time.as_unit("s"))
    print("Datetime to_numpy():", time.as_unit("s").to_numpy())
    time = pd.to_datetime(np.array([-1000, 1, 2], dtype="datetime64[Y]"))
    print("DatetimeIndex:", time)
    print("DatetimeIndex as_unit('s'):", time.as_unit("s"))
    print("DatetimeIndex to_numpy():", time.as_unit("s").to_numpy())

.. warning::
    Input data with resolution higher than ``'ns'`` (eg. ``'ps'``, ``'fs'``, ``'as'``) is truncated (not rounded) at the ``'ns'``-level. This is currently broken for the ``'ps'`` input, where it is interpreted as ``'ns'``.

    .. ipython:: python

        try:
            print("Good:", pd.to_datetime([np.datetime64(1901901901901, "as")]))
            print("Good:", pd.to_datetime([np.datetime64(1901901901901, "fs")]))
            print(" Bad:", pd.to_datetime([np.datetime64(1901901901901, "ps")]))
            print("Good:", pd.to_datetime([np.datetime64(1901901901901, "ns")]))
            print("Good:", pd.to_datetime([np.datetime64(1901901901901, "us")]))
            print("Good:", pd.to_datetime([np.datetime64(1901901901901, "ms")]))
            print(
                "Good:", pd.to_datetime(np.array([np.datetime64(1901901901901, "s")]))
            )
            print("Bad:", pd.to_datetime([np.datetime64(1901901901901, "s")]))
        except Exception as err:
            print("Raise:", err)

.. warning::
    Care has to be taken, as some configurations of input data will raise. The following shows, that we are safe to use :py:func:`pandas.to_datetime` when providing :py:class:`numpy.datetime64` as scalar or numpy array as input.

    .. ipython:: python

        print(
            "Works:",
            np.datetime64(1901901901901, "s"),
            pd.to_datetime(np.datetime64(1901901901901, "s")),
        )
        print(
            "Works:",
            np.array([np.datetime64(1901901901901, "s")]),
            pd.to_datetime(np.array([np.datetime64(1901901901901, "s")])),
        )
        try:
            pd.to_datetime([np.datetime64(1901901901901, "s")])
        except Exception as err:
            print("Raises:", err)
        try:
            pd.to_datetime(1901901901901, unit="s")
        except Exception as err:
            print("Raises:", err)
        try:
            pd.to_datetime([1901901901901], unit="s")
        except Exception as err:
            print("Raises:", err)
        try:
            pd.to_datetime(np.array([1901901901901]), unit="s")
        except Exception as err:
            print("Raises:", err)


to_timedelta
~~~~~~~~~~~~

The function :py:func:`pandas.to_timedelta` is used within xarray for inferring units and for testing purposes.

In normal operation :py:func:`pandas.to_timedelta` returns :py:class:`pandas.Timedelta` (scalar input) or :py:class:`pandas.TimedeltaIndex` (array-like input) which are timedelta64 with ``ns`` resolution internally. That has the implication, that the usable timedelta covers only roughly 585 years. To accommodate for that, we are working around that limitation in the encoding and decoding step.

.. ipython:: python

    f"Maximum timedelta range: ({pd.to_timedelta(int64_min, unit="ns")}, {pd.to_timedelta(int64_max, unit="ns")})"

For input values which can't be represented in nanosecond resolution :py:class:`pandas.OutOfBoundsTimedelta` exception is raised:

.. ipython:: python

    try:
        delta = pd.to_timedelta(int64_max, unit="us")
    except Exception as err:
        print("First:", err)
    try:
        delta = pd.to_timedelta(uint64_max, unit="ns")
    except Exception as err:
        print("Second:", err)

When args are numeric (no strings) "unit" can be anything from ``'W'``, ``'D'``, ``'h'``, ``'m'``, ``'s'``, ``'ms'``, ``'us'`` or ``'ns'``.

Numpy timedelta64 can be extracted with :py:meth:`pandas.Timedelta.to_numpy` and :py:meth:`pandas.TimedeltaIndex.to_numpy`. The returned resolution depends on the internal representation. This representation can be changed using :py:meth:`pandas.Timedelta.as_unit`
and :py:meth:`pandas.TimedeltaIndex.as_unit` respectively.

``as_unit`` takes one of ``'s'``, ``'ms'``, ``'us'``, ``'ns'`` as argument. That means we are able to represent timedeltas with second, millisecond, microsecond or nanosecond resolution.

.. ipython:: python

    delta = pd.to_timedelta(1, unit="D")
    print("Timedelta:", delta)
    print("Timedelta as_unit('s'):", delta.as_unit("s"))
    print("Timedelta to_numpy():", delta.as_unit("s").to_numpy())
    delta = pd.to_timedelta([0, 1, 2], unit="D")
    print("TimedeltaIndex:", delta)
    print("TimedeltaIndex as_unit('s'):", delta.as_unit("s"))
    print("TimedeltaIndex to_numpy():", delta.as_unit("s").to_numpy())

.. note::
    For the functionality in xarray the output resolution is converted from ``'ns'`` to the lowest needed resolution.

.. warning::
    Care has to be taken, as some configurations of input data will raise. The following shows, that we are safe to use :py:func:`pandas.to_datetime` when providing :py:class:`numpy.datetime64` as scalar or numpy array as input.

    .. ipython:: python

        print(
            "Works:",
            np.timedelta64(1901901901901, "s"),
            pd.to_timedelta(np.timedelta64(1901901901901, "s")),
        )
        print(
            "Works:",
            np.array([np.timedelta64(1901901901901, "s")]),
            pd.to_timedelta(np.array([np.timedelta64(1901901901901, "s")])),
        )
        try:
            pd.to_timedelta([np.timedelta64(1901901901901, "s")])
        except Exception as err:
            print("Raises:", err)
        try:
            pd.to_timedelta(1901901901901, unit="s")
        except Exception as err:
            print("Raises:", err)
        try:
            pd.to_timedelta([1901901901901], unit="s")
        except Exception as err:
            print("Raises:", err)
        try:
            pd.to_timedelta(np.array([1901901901901]), unit="s")
        except Exception as err:
            print("Raises:", err)

Timestamp
~~~~~~~~~

:py:class:`pandas.Timestamp` is used within xarray to wrap strings of CF reference times and datetime.datetime.

When args are numeric (no strings) "unit" can be anything from ``'Y'``, ``'W'``, ``'D'``, ``'h'``, ``'m'``, ``'s'``, ``'ms'``, ``'us'`` or ``'ns'``.

In normal operation :py:class:`pandas.Timestamp` holds the timestamp in the provided resolution, but only one of ``'s'``, ``'ms'``, ``'us'``, ``'ns'``. Lower resolution input is automatically converted to ``'s'``, higher resolution input is cutted to ``'ns'``.

Same conversion rules apply here as for :py:func:`pandas.to_timedelta` (see above).
Depending on the internal resolution Timestamps can be represented in the range:

.. ipython:: python

    for unit in ["s", "ms", "us", "ns"]:
        print(
            f"unit: {unit!r} time range ({pd.Timestamp(int64_min, unit=unit)}, {pd.Timestamp(int64_max, unit=unit)})"
        )

Since relaxing the resolution this enhances the range to several hundreds of thousands of centuries with microsecond representation. ``NaT`` will be at ``np.iinfo("int64").min`` for all of the different representations.

.. warning::
    When initialized with a datetime string this is only defined from ``-9999-01-01`` to ``9999-12-31``.

    .. ipython:: python

        try:
            print("Works:", pd.Timestamp("-9999-01-01 00:00:00"))
            print("Works, too:", pd.Timestamp("9999-12-31 23:59:59"))
            print(pd.Timestamp("10000-01-01 00:00:00"))
        except Exception as err:
            print("Errors:", err)

.. note::
    :py:class:`pandas.Timestamp` is the only current possibility to correctly import time reference strings. It handles non-ISO formatted strings, keeps the resolution of the strings (``'s'``, ``''ms''`` etc.) and imports time zones. When initialized with :py:class:`numpy.datetime64` instead of a string it even overcomes the above limitation of the possible time range.

    .. ipython:: python

        try:
            print("Handles non-ISO:", pd.Timestamp("92-1-8 151542"))
            print(
                "Keeps resolution 1:",
                pd.Timestamp("1992-10-08 15:15:42"),
                pd.Timestamp("1992-10-08 15:15:42").unit,
            )
            print(
                "Keeps resolution 2:",
                pd.Timestamp("1992-10-08 15:15:42.5"),
                pd.Timestamp("1992-10-08 15:15:42.5").unit,
            )
            print(
                "Keeps timezone:",
                pd.Timestamp("1992-10-08 15:15:42.5 -6:00"),
                pd.Timestamp("1992-10-08 15:15:42.5 -6:00").unit,
            )
            print(
                "Extends timerange :",
                pd.Timestamp(np.datetime64("-10000-10-08 15:15:42.5001")),
                pd.Timestamp(np.datetime64("-10000-10-08 15:15:42.5001")).unit,
            )
        except Exception as err:
            print("Errors:", err)

DatetimeIndex
~~~~~~~~~~~~~

:py:class:`pandas.DatetimeIndex` is used to wrap numpy datetime64 or other datetime-likes, when encoding. The resolution of the DatetimeIndex depends on the input, but can be only one of ``'s'``, ``'ms'``, ``'us'``, ``'ns'``. Lower resolution input is automatically converted to ``'s'``, higher resolution input is cutted to ``'ns'``.
:py:class:`pandas.DatetimeIndex` will raise :py:class:`pandas.OutOfBoundsDatetime` if the input can't be represented in the given resolution.

.. note::
    For xarray we assume that all :py:class:`numpy.datetime64` provided to :py:class:`pandas.DatetimeIndex` are up to the specs. This is especially true, when those values have been decoded upfront. If the data is provided by users, they should handle any issues before.

.. ipython:: python

    try:
        print(
            "Works:",
            pd.DatetimeIndex(
                np.array(["1992-01-08", "1992-01-09"], dtype="datetime64[D]")
            ),
        )
        print(
            "Works:",
            pd.DatetimeIndex(
                np.array(
                    ["1992-01-08 15:15:42", "1992-01-09 15:15:42"],
                    dtype="datetime64[s]",
                )
            ),
        )
        print(
            "Works:",
            pd.DatetimeIndex(
                np.array(
                    ["1992-01-08 15:15:42.5", "1992-01-09 15:15:42.0"],
                    dtype="datetime64[ms]",
                )
            ),
        )
        print(
            "Works:",
            pd.DatetimeIndex(
                np.array(
                    ["1970-01-01 00:00:00.401501601701801901", "1970-01-01 00:00:00"],
                    dtype="datetime64[as]",
                )
            ),
        )
        print(
            "Works:",
            pd.DatetimeIndex(
                np.array(
                    ["-10000-01-01 00:00:00.401501", "1970-01-01 00:00:00"],
                    dtype="datetime64[us]",
                )
            ),
        )
    except Exception as err:
        print("Errors:", err)

CF Conventions Time Handling
----------------------------

Xarray tries to adhere to the latest version of the `CF Conventions`_. Relevant is the section on `Time Coordinate`_ and the `Calendar`_ subsection.

.. _CF Conventions: https://cfconventions.org
.. _Time Coordinate: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#time-coordinate
.. _Calendar: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#calendar

CF time decoding
~~~~~~~~~~~~~~~~

Decoding of ``values`` with time unit specification like ``seconds since 1992-10-8 15:15:42.5 -6:00`` into datetimes (using CF convention) is a multistage process.

1. If we have a non-standard calendar (eg. ``noleap``) the decoding is done with ``cftime`` package (which is not covered in this section). For ``standard``/``gregorian`` calendar as well as ``proleptic_gregorian`` the above outlined pandas functionality is used.

2. ``standard``/``gregorian`` calendar and ``proleptic_gregorian`` are equivalent for any dates and reference times >= ``1582-10-15``. First the reference time is checked and any timezone information stripped off and in a second step, the minimum and maximum ``values`` are checked if they can be represented in the current reference time resolution. At the same time integer overflow would be caught. For ``standard``/``gregorian`` calendar the dates are checked to be >= ``1582-10-15``. If anything fails, the decoding is done with ``cftime``).

3. As the time unit (here ``seconds``) and the resolution of the reference time ``1992-10-8 15:15:42.5 -6:00`` (here ``milliseconds``) might be different, this has to be aligned to the higher resolution (retrieve new unit). This is done by multiplying the ``values`` by the ratio of nanoseconds per time unit and nanoseconds per reference time unit. To not break consistency for ``NaT`` a mask is kept and re-introduced after the multiplication.

4. Finally, the ``values`` (``int64``) are cast to ``datetime64[unit]`` (using the above retrieved unit) and added to the reference time :py:class:`pandas.Timestamp`.

.. ipython:: python

    calendar = "proleptic_gregorian"
    values = np.array([-1000 * 365, 0, 1000 * 365], dtype="int64")
    units = "days since 2000-01-01 00:00:00.000001"
    dt = xr.coding.times.decode_cf_datetime(values, units, calendar)
    print(dt)
    assert dt.dtype == "datetime64[us]"

    units = "microseconds since 2000-01-01 00:00:00"
    dt = xr.coding.times.decode_cf_datetime(values, units, calendar)
    print(dt)
    assert dt.dtype == "datetime64[us]"

    values = np.array([0, 0.25, 0.5, 0.75, 1.0], dtype="float64")
    units = "days since 2000-01-01 00:00:00.001"
    dt = xr.coding.times.decode_cf_datetime(values, units, calendar)
    print(dt)
    assert dt.dtype == "datetime64[ms]"

    values = np.array([0, 0.25, 0.5, 0.75, 1.0], dtype="float64")
    units = "hours since 2000-01-01"
    dt = xr.coding.times.decode_cf_datetime(values, units, calendar)
    print(dt)
    assert dt.dtype == "datetime64[s]"

    values = np.array([0, 0.25, 0.5, 0.75, 1.0], dtype="float64")
    units = "hours since 2000-01-01 00:00:00 03:30"
    dt = xr.coding.times.decode_cf_datetime(values, units, calendar)
    print(dt)
    assert dt.dtype == "datetime64[s]"

    values = np.array([-2002 * 365 - 121, -366, 365, 2000 * 365 + 119], dtype="int64")
    units = "days since 0001-01-01 00:00:00"
    dt = xr.coding.times.decode_cf_datetime(values, units, calendar)
    print(dt)
    assert dt.dtype == "datetime64[s]"

CF time encoding
~~~~~~~~~~~~~~~~

For encoding the process is more or less a reversal of the above, but we have to make some decisions on default values.

1. Infer ``data_units`` from the given ``dates``.
2. Infer ``units`` (either cleanup given ``units`` or use ``data_units``
3. Infer calendar name from given ``dates``.
4. If non standard calendar or object dates (CFTime) encode with ``cftime``
5. Retrieve ``time_units`` and ``ref_date`` from ``units``
6. Check ``ref_date`` >= ``1582-10-15``, otherwise -> ``cftime``
7. Wrap ``dates`` with pd.DatetimeIndex
8. Subtracting ``ref_date`` (:py:class:`pandas.Timestamp`) from above :py:class:`pandas.DatetimeIndex` will return :py:class:`pandas.TimedeltaIndex`
9. Align resolution of :py:class:`pandas.TimedeltaIndex` with resolution of ``time_units``
10. Retrieve needed ``units`` and ``delta`` to faithfully encode into int64
11. Divide ``time_deltas`` by ``delta``, use floor division (integer) or normal division (float)
12. Return result

.. ipython:: python
    :okwarning:

    calendar = "proleptic_gregorian"
    dates = np.array(
        [
            "-2000-01-01T00:00:00",
            "0000-01-01T00:00:00",
            "0002-01-01T00:00:00",
            "2000-01-01T00:00:00",
        ],
        dtype="datetime64[s]",
    )
    orig_values = np.array(
        [-2002 * 365 - 121, -366, 365, 2000 * 365 + 119], dtype="int64"
    )
    units = "days since 0001-01-01 00:00:00"
    values, _, _ = xr.coding.times.encode_cf_datetime(
        dates, units, calendar, dtype=np.dtype("int64")
    )
    print(values)
    np.testing.assert_array_equal(values, orig_values)

    dates = np.array(
        [
            "-2000-01-01T01:00:00",
            "0000-01-01T00:00:00",
            "0002-01-01T00:00:00",
            "2000-01-01T00:00:00",
        ],
        dtype="datetime64[s]",
    )
    orig_values = np.array(
        [-2002 * 365 - 121, -366, 365, 2000 * 365 + 119], dtype="int64"
    )
    units = "days since 0001-01-01 00:00:00"
    values, units, _ = xr.coding.times.encode_cf_datetime(
        dates, units, calendar, dtype=np.dtype("int64")
    )
    print(values, units)
