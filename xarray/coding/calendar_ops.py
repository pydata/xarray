from datetime import timedelta

import numpy as np

from ..core.common import is_np_datetime_like
from .cftime_offsets import date_range_like, get_date_type
from .times import _is_numpy_compatible_time_range, _is_standard_calendar, convert_times

try:
    import cftime
except ImportError:
    cftime = None


def _days_in_year(year, calendar, use_cftime=True):
    """Return the number of days in the input year according to the input calendar."""
    return (
        (
            get_date_type(calendar, use_cftime=use_cftime)(year + 1, 1, 1)
            - timedelta(days=1)
        )
        .timetuple()
        .tm_yday
    )


def convert_calendar(
    ds,
    calendar,
    dim="time",
    align_on=None,
    missing=None,
    use_cftime=None,
):
    """Convert the Dataset or DataArray to another calendar.

    Only converts the individual timestamps, does not modify any data except in dropping invalid/surplus dates or inserting missing dates.

    If the source and target calendars are either no_leap, all_leap or a standard type, only the type of the time array is modified.
    When converting to a leap year from a non-leap year, the 29th of February is removed from the array.
    In the other direction the 29th of February will be missing in the output, unless `missing` is specified, in which case that value is inserted.

    For conversions involving `360_day` calendars, see Notes.

    This method is safe to use with sub-daily data as it doesn't touch the time part of the timestamps.

    Parameters
    ----------
    ds : DataArray or Dataset
      Input array/dataset with a time coordinate of a valid dtype (datetime64 or a cftime.datetime).
    calendar : str
      The target calendar name.
    dim : str
      Name of the time coordinate.
    align_on : {None, 'date', 'year'}
      Must be specified when either source or target is a `360_day` calendar, ignored otherwise. See Notes.
    missing : Optional[any]
      A value to use for filling in dates in the target that were missing in the source.
      Default (None) is not to fill values, so the output time axis might be non-continuous.
    use_cftime : boolean, optional
      Whether to use cftime objects in the output, valid if `calendar` is one of {"proleptic_gregorian", "gregorian" or "standard"}.
      If True, the new time axis uses cftime objects. If None (default), it uses numpy objects if the date range permits it, and cftime ones if not.
      If False, it uses numpy objects or fails.

    Returns
    -------
      Copy of source with the time coordinate converted to the target calendar.
      If `missing` was None (default), invalid dates in the new calendar are dropped, but missing dates are not inserted.
      If `missing` was given, the new data is reindexed to have a continuous time axis, filling missing datas with `missing`.

    Notes
    -----
    If one of the source or target calendars is `360_day`, `align_on` must be specified and two options are offered.

    "year"
      The dates are translated according to their rank in the year (dayofyear), ignoring their original month and day information,
      meaning that the missing/surplus days are added/removed at regular intervals.

      From a `360_day` to a standard calendar, the output will be missing the following dates (day of year in parenthesis):
        To a leap year:
          January 31st (31), March 31st (91), June 1st (153), July 31st (213), September 31st (275) and November 30th (335).
        To a non-leap year:
          February 6th (36), April 19th (109), July 2nd (183), September 12th (255), November 25th (329).

      From standard calendar to a '360_day', the following dates in the source array will be dropped:
        From a leap year:
          January 31st (31), April 1st (92), June 1st (153), August 1st (214), September 31st (275), December 1st (336)
        From a non-leap year:
          February 6th (37), April 20th (110), July 2nd (183), September 13th (256), November 25th (329)

      This option is best used on daily and subdaily data.

    "date"
      The month/day information is conserved and invalid dates are dropped from the output. This means that when converting from
      a `360_day` to a standard calendar, all 31st (Jan, March, May, July, August, October and December) will be missing as there is no equivalent
      dates in the `360_day` and the 29th (on non-leap years) and 30th of February will be dropped as there are no equivalent dates in
      a standard calendar.

      This option is best used with data on a frequency coarser than daily.
    """
    # In the following the calendar name "default" is an
    # internal hack to mean pandas-backed standard calendar
    from ..core.dataarray import DataArray

    time = ds[dim]  # for convenience

    # Arguments Checks for target
    if use_cftime is not True:
        # Then we check is pandas is possible.
        if _is_standard_calendar(calendar):
            if _is_numpy_compatible_time_range(time):
                # Conversion is possible with pandas, force False if it was None.
                use_cftime = False
            elif use_cftime is False:
                raise ValueError(
                    "Source time range is not valid for numpy datetimes. Try using `use_cftime=True`."
                )
            # else : Default to cftime
        elif use_cftime is False:
            # target calendar is ctime-only.
            raise ValueError(
                f"Calendar '{calendar}' is only valid with cftime. Try using `use_cftime=True`."
            )
        else:
            use_cftime = True

    # Get source
    source = time.dt.calendar

    src_cal = "default" if is_np_datetime_like(time.dtype) else source
    tgt_cal = calendar if use_cftime else "default"
    if src_cal == tgt_cal:
        return ds

    if (source == "360_day" or calendar == "360_day") and align_on is None:
        raise ValueError(
            "Argument `align_on` must be specified with either 'date' or "
            "'year' when converting to or from a '360_day' calendar."
        )

    if source != "360_day" and calendar != "360_day":
        align_on = "date"

    out = ds.copy()

    if align_on == "year":
        # Special case for conversion involving 360_day calendar
        # Instead of translating dates directly, this tries to keep the position within a year similar.
        def _yearly_interp_doy(time):
            # Returns the nearest day in the target calendar of the corresponding "decimal year" in the source calendar
            yr = int(time.dt.year[0])
            return np.round(
                _days_in_year(yr, calendar, use_cftime)
                * time.dt.dayofyear
                / _days_in_year(yr, source, use_cftime)
            ).astype(int)

        def _convert_datetime(date, new_doy, calendar):
            """Convert a datetime object to another calendar.

            Redefining the day of year (thus ignoring month and day information from the source datetime).
            Nanosecond information are lost as cftime.datetime doesn't support them.
            """
            new_date = cftime.num2date(
                new_doy - 1,
                f"days since {date.year}-01-01",
                calendar=calendar if use_cftime else "standard",
            )
            try:
                return get_date_type(calendar, use_cftime)(
                    date.year,
                    new_date.month,
                    new_date.day,
                    date.hour,
                    date.minute,
                    date.second,
                    date.microsecond,
                )
            except ValueError:
                return np.nan

        new_doy = time.groupby(f"{dim}.year").map(_yearly_interp_doy)

        # Convert the source datetimes, but override the doy with our new doys
        out[dim] = DataArray(
            [
                _convert_datetime(date, newdoy, calendar)
                for date, newdoy in zip(time.variable._data.array, new_doy)
            ],
            dims=(dim,),
            name=dim,
        )
        # Remove duplicate timestamps, happens when reducing the number of days
        out = out.isel({dim: np.unique(out[dim], return_index=True)[1]})
    elif align_on == "date":
        new_times = convert_times(
            time.variable._data.array,
            get_date_type(calendar, use_cftime=use_cftime),
            raise_on_invalid=False,
        )
        out[dim] = new_times

        # Remove NaN that where put on invalid dates in target calendar
        out = out.where(out[dim].notnull(), drop=True)

    if missing is not None:
        time_target = date_range_like(time, calendar=calendar, use_cftime=use_cftime)
        out = out.reindex({dim: time_target}, fill_value=missing)

    # Copy attrs but remove `calendar` if still present.
    out[dim].attrs.update(time.attrs)
    out[dim].attrs.pop("calendar", None)
    return out


def _datetime_to_decimal_year(times, calendar=None):
    """Convert a datetime DataArray to decimal years according to its calendar or the given one.

    Decimal years are the number of years since 0001-01-01 00:00:00 AD.
    Ex: '2000-03-01 12:00' is 2000.1653 in a standard calendar, 2000.16301 in a "noleap" or 2000.16806 in a "360_day".
    """
    from ..core.dataarray import DataArray

    calendar = calendar or times.dt.calendar

    if is_np_datetime_like(times.dtype):
        times = times.copy(data=convert_times(times.values, get_date_type("standard")))

    def _make_index(time):
        year = int(time.dt.year[0])
        doys = cftime.date2num(times, f"days since {year:04d}-01-01", calendar=calendar)
        return DataArray(
            year + doys / _days_in_year(year, calendar),
            dims=time.dims,
            coords=time.coords,
            name="time",
        )

    return times.groupby("time.year").map(_make_index)


def interp_calendar(source, target, dim="time"):
    """Interpolates a DataArray/Dataset to another calendar based on decimal year measure.

    Each timestamp in source and target are first converted to their decimal year equivalent
    then source is interpolated on the target coordinate. The decimal year is the number of
    years since 0001-01-01 AD.
    Ex: '2000-03-01 12:00' is 2000.1653 in a standard calendar or 2000.16301 in a 'noleap' calendar.

    This method should be used with daily data or coarser. Sub-daily result will have a modified day cycle.

    Parameters
    ----------
    source: Union[DataArray, Dataset]
      The source data to interpolate, must have a time coordinate of a valid dtype (np.datetime64 or cftime objects)
    target: DataArray
      The target time coordinate of a valid dtype (np.datetime64 or cftime objects)
    dim : str
      The time coordinate name.

    Return
    ------
    Union[DataArray, Dataset]
      The source interpolated on the decimal years of target,
    """
    cal_src = source[dim].dt.calendar
    cal_tgt = target.dt.calendar

    out = source.copy()
    out[dim] = _datetime_to_decimal_year(source[dim], calendar=cal_src).drop_vars(dim)
    target_idx = _datetime_to_decimal_year(target, calendar=cal_tgt)
    out = out.interp(**{dim: target_idx})
    out[dim] = target
    return out
