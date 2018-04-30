from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

from .common import is_np_datetime_like, _contains_datetime_like_objects
from .pycompat import dask_array_type


def _season_from_months(months):
    """Compute season (DJF, MAM, JJA, SON) from month ordinal
    """
    # TODO: Move "season" accessor upstream into pandas
    seasons = np.array(['DJF', 'MAM', 'JJA', 'SON'])
    months = np.asarray(months)
    return seasons[(months // 3) % 4]


def _access_through_cftimeindex(values, name):
    """Coerce an array of datetime-like values to a CFTimeIndex
    and access requested datetime component
    """
    from ..coding.cftimeindex import CFTimeIndex
    values_as_cftimeindex = CFTimeIndex(values.ravel())
    if name == 'season':
        months = values_as_cftimeindex.month
        field_values = _season_from_months(months)
    else:
        field_values = getattr(values_as_cftimeindex, name)
    return field_values.reshape(values.shape)


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
    if is_np_datetime_like(values.dtype):
        access_method = _access_through_series
    else:
        access_method = _access_through_cftimeindex

    if isinstance(values, dask_array_type):
        from dask.array import map_blocks
        return map_blocks(access_method,
                          values, name, dtype=dtype)
    else:
        return access_method(values, name)


def _round_series(values, name, freq):
    """Coerce an array of datetime-like values to a pandas Series and
    apply requested rounding
    """
    values_as_series = pd.Series(values.ravel())
    method = getattr(values_as_series.dt, name)
    field_values = method(freq=freq).values

    return field_values.reshape(values.shape)


def _round_field(values, name, freq):
    """Indirectly access pandas rounding functions by wrapping data
    as a Series and calling through `.dt` attribute.

    Parameters
    ----------
    values : np.ndarray or dask.array-like
        Array-like container of datetime-like values
    name : str (ceil, floor, round)
        Name of rounding function
    freq : a freq string indicating the rounding resolution

    Returns
    -------
    rounded timestamps : same type as values
        Array-like of datetime fields accessed for each element in values

    """
    if isinstance(values, dask_array_type):
        from dask.array import map_blocks
        return map_blocks(_round_series,
                          values, name, freq=freq, dtype=np.datetime64)
    else:
        return _round_series(values, name, freq)


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
     calendar (e.g. a 360-day calendar) using cftime, then some fields like
     `dayofyear` may not be accurate.

     """

    def __init__(self, xarray_obj):
        if not _contains_datetime_like_objects(xarray_obj):
            raise TypeError("'dt' accessor only available for "
                            "DataArray with datetime64 timedelta64 dtype or "
                            "for arrays containing cftime datetime "
                            "objects.")
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

    def _tslib_round_accessor(self, name, freq):
        obj_type = type(self._obj)
        result = _round_field(self._obj.data, name, freq)
        return obj_type(result, name=name,
                        coords=self._obj.coords, dims=self._obj.dims)

    def floor(self, freq):
        '''
        Round timestamps downward to specified frequency resolution.

        Parameters
        ----------
        freq : a freq string indicating the rounding resolution
            e.g. 'D' for daily resolution

        Returns
        -------
        floor-ed timestamps : same type as values
            Array-like of datetime fields accessed for each element in values
        '''

        return self._tslib_round_accessor("floor", freq)

    def ceil(self, freq):
        '''
        Round timestamps upward to specified frequency resolution.

        Parameters
        ----------
        freq : a freq string indicating the rounding resolution
            e.g. 'D' for daily resolution

        Returns
        -------
        ceil-ed timestamps : same type as values
            Array-like of datetime fields accessed for each element in values
        '''
        return self._tslib_round_accessor("ceil", freq)

    def round(self, freq):
        '''
        Round timestamps to specified frequency resolution.

        Parameters
        ----------
        freq : a freq string indicating the rounding resolution
            e.g. 'D' for daily resolution

        Returns
        -------
        rounded timestamps : same type as values
            Array-like of datetime fields accessed for each element in values
        '''
        return self._tslib_round_accessor("round", freq)
