from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .common import is_datetime_like
from .extensions import register_dataarray_accessor
from .pycompat import dask_array_type

import numpy as np
import pandas as pd

def _get_date_field(values, name):
    """Indirectly access pandas' libts.get_date_field by wrapping data
    as a Series and calling through `.dt` attribute.

    Parameters
    ----------
    values : np.ndarray or dask.array-like
        Array-like container of datetime-like values
    name : str
        Name of datetime field to access

    """
    def _access_through_series(values, name):
        values_as_series = pd.Series(values.ravel())
        field_values = getattr(values_as_series.dt, name).values
        return field_values.reshape(values.shape)

    if isinstance(values, dask_array_type):
        from dask.array import map_blocks
        out_dtype = object if name == "weekday_name" else np.int64
        return map_blocks(_access_through_series,
                          values, name, dtype=out_dtype)
    else:
        return _access_through_series(values, name)


@register_dataarray_accessor('dt')
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

     All of the pandas fields are accessible here. Note that these fields are not
     calendar-aware; if your datetimes are encoded with a non-Gregorian calendar
     (e.g. a 360-day calendar) using netcdftime, then some fields like
     `dayofyear` may not be accurate.

     """
    def __init__(self, xarray_obj):
        if not is_datetime_like(xarray_obj.dtype):
            raise TypeError("'dt' accessor only available for "
                            "DataArray with datetime64 or timedelta64 dtype")
        self._obj = xarray_obj
        self._dt = self._obj.data

    def _tslib_field_accessor(name, docstring=None):
        def f(self):
            from .dataarray import DataArray
            result = _get_date_field(self._dt, name)
            return DataArray(result, name=name,
                             coords=self._obj.coords, dims=self._obj.dims)

        f.__name__ = name
        f.__doc__ = docstring
        return property(f)

    year = _tslib_field_accessor('year', "The year of the datetime")
    month = _tslib_field_accessor(
        'month', "The month as January=1, December=12"
    )
    day = _tslib_field_accessor('day', "The days of the datetime")
    hour = _tslib_field_accessor('hour', "The hours of the datetime")
    minute = _tslib_field_accessor('minute', "The minutes of the datetime")
    second = _tslib_field_accessor('second', "The seconds of the datetime")
    microsecond = _tslib_field_accessor(
        'microsecond', "The microseconds of the datetime"
    )
    nanosecond = _tslib_field_accessor(
        'nanosecond', "The nanoseconds of the datetime"
    )
    weekofyear = _tslib_field_accessor(
        'weekofyear', "The week ordinal of the year"
    )
    week = weekofyear
    dayofweek = _tslib_field_accessor(
        'dayofweek', "The day of the week with Monday=0, Sunday=6"
    )
    weekday = dayofweek

    weekday_name = _tslib_field_accessor(
        'weekday_name', "The name of day in a week (ex: Friday)"
    )

    dayofyear = _tslib_field_accessor('dayofyear', "The ordinal day of the year")
    quarter = _tslib_field_accessor('quarter', "The quarter of the date")
    days_in_month = _tslib_field_accessor(
        'days_in_month', "The number of days in the month"
    )
    daysinmonth = days_in_month
