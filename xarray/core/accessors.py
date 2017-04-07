from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .common import is_datetime_like
from .extensions import register_dataarray_accessor

import pandas as pd

def _get_date_field(array, name):
    """Indirectly access pandas' libts.get_date_field by wrapping data
    as a Series and calling through `.dt` attribute.

    Parameters
    ----------
    array : np.ndarray or array-like
        Array-like container of datetime-like values
    name : str
        Name of datetime field to access

    """
    series = pd.Series(array.ravel())
    field_values = getattr(series.dt, name).values
    return field_values.reshape(array.shape)

_date_field_ops_and_descrs = [
    ('year', "The year of the datetime"),
    ('month', "The month as January=1, December=12"),
    ('day', "The days of the datetime"),
    ('hour', "The hours of the datetime"),
    ('minute', "The minutes of the datetime"),
    ('second', "The seconds of the datetime"),
    ('microsecond', "The microseconds of the datetime"),
    ('nanosecond', "The nanoseconds of the datetime"),
    ('weekofyear',  "The week ordinal of the year"),
    ('week', "The week ordinal of the year"),
    ('weekday', "The day of the week with Monday=0, Sunday=6"),
    ('weekday_name',
     "The name of day in a week (ex: Friday)"),
    ('dayofweek', "The day of the week with Monday=0, Sunday=6"),
    ('dayofyear',  "The ordinal day of the year"),
    ('quarter', "The quarter of the date"),
    ('days_in_month', "The number of days in the month"),
    ('daysinmonth', "The number of days in the month"),
]

def inject_date_field_accessors(cls):
    for op, descr in _date_field_ops_and_descrs:
        prop = cls._date_field_accessor(op, docstring=descr)
        setattr(cls, op, prop)


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
        self._dt = None

    @property
    def dt(self):
        """Attribute to cache a view of the underlying datetime-like
        array for passing to pandas.tslib for date_field operations
        """
        if self._dt is None:
            self._dt = self._obj.values
        return self._dt

    def _date_field_accessor(name, docstring=None):
        def f(self):
            from .dataarray import DataArray
            result = _get_date_field(self.dt, name)
            return DataArray(result, name=name,
                             coords=self._obj.coords, dims=self._obj.dims)

        f.__name__ = name
        f.__doc__ = docstring
        return property(f)

inject_date_field_accessors(DatetimeAccessor)
