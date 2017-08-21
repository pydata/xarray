from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dtypes import is_datetime_like
from .pycompat import dask_array_type

import numpy as np
import pandas as pd


def _season_from_months(months):
    """Compute season (DJF, MAM, JJA, SON) from month ordinal
    """
    # TODO: Move "season" accessor upstream into pandas
    seasons = np.array(['DJF', 'MAM', 'JJA', 'SON'])
    months = np.asarray(months)
    return seasons[(months // 3) % 4]


def _access_through_series(values, name):
    """Coerce an array of datetime-like values to a pandas Series and
    access requested datetime component
    """
    values_as_series = pd.Series(values.ravel())
    if name == "season":
        months = values_as_series.dt.month.values
        field_values = _season_from_months(months)
    else:
        field_values = getattr(values_as_series.dt, name).values
    return field_values.reshape(values.shape)


def _get_date_field(values, name, dtype):
    """Indirectly access pandas' libts.get_date_field by wrapping data
    as a Series and calling through `.dt` attribute.

    Parameters
    ----------
    values : np.ndarray or dask.array-like
        Array-like container of datetime-like values
    name : str
        Name of datetime field to access
    dtype : dtype-like
        dtype for output date field values

    Returns
    -------
    datetime_fields : same type as values
        Array-like of datetime fields accessed for each element in values

    """
    if isinstance(values, dask_array_type):
        from dask.array import map_blocks
        return map_blocks(_access_through_series,
                          values, name, dtype=dtype)
    else:
        return _access_through_series(values, name)


class DatetimeAccessor(object):
    """Access datetime fields for DataArrays with datetime-like dtypes.

     Similar to pandas, fields can be accessed through the `.dt` attribute
     for applicable DataArrays:

        >>> ds = xarray.Dataset({'time': pd.date_range(start='2000/01/01',
        ...                                            freq='D', periods=100)})
        >>> ds.time.dt
        <xarray.core.accessors.DatetimeAccessor at 0x10c369f60>
        >>> ds.time.dt.dayofyear[:5]
        <xarray.DataArray 'dayofyear' (time: 5)>
        array([1, 2, 3, 4, 5], dtype=int32)
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03 ...

     All of the pandas fields are accessible here. Note that these fields are
     not calendar-aware; if your datetimes are encoded with a non-Gregorian
     calendar (e.g. a 360-day calendar) using netcdftime, then some fields like
     `dayofyear` may not be accurate.

     """
    def __init__(self, xarray_obj):
        if not is_datetime_like(xarray_obj.dtype):
            raise TypeError("'dt' accessor only available for "
                            "DataArray with datetime64 or timedelta64 dtype")
        self._obj = xarray_obj

    def _tslib_field_accessor(name, docstring=None, dtype=None):
        def f(self, dtype=dtype):
            if dtype is None:
                dtype = self._obj.dtype
            obj_type = type(self._obj)
            result = _get_date_field(self._obj.data, name, dtype)
            return obj_type(result, name=name,
                            coords=self._obj.coords, dims=self._obj.dims)

        f.__name__ = name
        f.__doc__ = docstring
        return property(f)

    year = _tslib_field_accessor('year', "The year of the datetime", np.int64)
    month = _tslib_field_accessor(
        'month', "The month as January=1, December=12", np.int64
    )
    day = _tslib_field_accessor('day', "The days of the datetime", np.int64)
    hour = _tslib_field_accessor('hour', "The hours of the datetime", np.int64)
    minute = _tslib_field_accessor(
        'minute', "The minutes of the datetime", np.int64
    )
    second = _tslib_field_accessor(
        'second', "The seconds of the datetime", np.int64
    )
    microsecond = _tslib_field_accessor(
        'microsecond', "The microseconds of the datetime", np.int64
    )
    nanosecond = _tslib_field_accessor(
        'nanosecond', "The nanoseconds of the datetime", np.int64
    )
    weekofyear = _tslib_field_accessor(
        'weekofyear', "The week ordinal of the year", np.int64
    )
    week = weekofyear
    dayofweek = _tslib_field_accessor(
        'dayofweek', "The day of the week with Monday=0, Sunday=6", np.int64
    )
    weekday = dayofweek

    weekday_name = _tslib_field_accessor(
        'weekday_name', "The name of day in a week (ex: Friday)", object
    )

    dayofyear = _tslib_field_accessor(
        'dayofyear', "The ordinal day of the year", np.int64
    )
    quarter = _tslib_field_accessor('quarter', "The quarter of the date")
    days_in_month = _tslib_field_accessor(
        'days_in_month', "The number of days in the month", np.int64
    )
    daysinmonth = days_in_month

    season = _tslib_field_accessor(
        "season", "Season of the year (ex: DJF)", object
    )

    time = _tslib_field_accessor(
        "time", "Timestamps corresponding to datetimes", object
    )
