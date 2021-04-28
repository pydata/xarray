import numpy as np
import pytest

from xarray import DataArray, infer_freq
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftime_offsets import date_range

# Maximum day of year in each calendar.
max_doy = {
    "default": 366,
    "standard": 366,
    "gregorian": 366,
    "proleptic_gregorian": 366,
    "julian": 366,
    "noleap": 365,
    "365_day": 365,
    "all_leap": 366,
    "366_day": 366,
    "360_day": 360,
}


@pytest.mark.parametrize(
    "source,target,use_cftime,freq",
    [
        ("standard", "noleap", None, "D"),
        ("noleap", "proleptic_gregorian", True, "D"),
        ("noleap", "all_leap", None, "D"),
        ("all_leap", "proleptic_gregorian", False, "4H"),
    ],
)
def test_convert_calendar(source, target, use_cftime, freq):
    src = DataArray(
        date_range("2004-01-01", "2004-12-31", freq=freq, calendar=source),
        dims=("time",),
        name="time",
    )
    da_src = DataArray(
        np.linspace(0, 1, src.size), dims=("time",), coords={"time": src}
    )

    conv = convert_calendar(da_src, target, use_cftime=use_cftime)

    assert conv.time.dt.calendar == target


@pytest.mark.parametrize(
    "source,target,freq",
    [
        ("standard", "360_day", "D"),
        ("360_day", "proleptic_gregorian", "D"),
        ("proleptic_gregorian", "360_day", "4H"),
    ],
)
@pytest.mark.parametrize("align_on", ["date", "year"])
def test_convert_calendar_360_days(source, target, freq, align_on):
    src = DataArray(
        date_range("2004-01-01", "2004-12-30", freq=freq, calendar=source),
        dims=("time",),
        name="time",
    )
    da_src = DataArray(
        np.linspace(0, 1, src.size), dims=("time",), coords={"time": src}
    )

    conv = convert_calendar(da_src, target, align_on=align_on)

    assert conv.time.dt.calendar == target

    if align_on == "date":
        np.testing.assert_array_equal(
            conv.time.resample(time="M").last().dt.day,
            [30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
        )
    elif target == "360_day":
        np.testing.assert_array_equal(
            conv.time.resample(time="M").last().dt.day,
            [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29],
        )
    else:
        np.testing.assert_array_equal(
            conv.time.resample(time="M").last().dt.day,
            [30, 29, 30, 30, 31, 30, 30, 31, 30, 31, 29, 31],
        )
    if source == "360_day" and align_on == "year":
        assert conv.size == 360 if freq == "D" else 360 * 4
    else:
        assert conv.size == 359 if freq == "D" else 359 * 4


@pytest.mark.parametrize(
    "source,target,freq",
    [
        ("standard", "noleap", "D"),
        ("noleap", "proleptic_gregorian", "4H"),
        ("noleap", "all_leap", "M"),
        ("360_day", "noleap", "D"),
        ("noleap", "360_day", "D"),
    ],
)
def test_convert_calendar_missing(source, target, freq):
    src = DataArray(
        date_range(
            "2004-01-01",
            "2004-12-31" if source != "360_day" else "2004-12-30",
            freq=freq,
            calendar=source,
        ),
        dims=("time",),
        name="time",
    )
    da_src = DataArray(
        np.linspace(0, 1, src.size), dims=("time",), coords={"time": src}
    )
    out = convert_calendar(da_src, target, missing=np.nan, align_on="date")
    assert infer_freq(out.time) == freq
    if source == "360_day":
        assert out.time[-1].dt.day == 31


@pytest.mark.parametrize(
    "source,target",
    [
        ("standard", "noleap"),
        ("noleap", "proleptic_gregorian"),
        ("standard", "360_day"),
        ("360_day", "proleptic_gregorian"),
        ("noleap", "all_leap"),
        ("360_day", "noleap"),
    ],
)
def test_interp_calendar(source, target):
    src = DataArray(
        date_range("2004-01-01", "2004-07-30", freq="D", calendar=source),
        dims=("time",),
        name="time",
    )
    tgt = DataArray(
        date_range("2004-01-01", "2004-07-30", freq="D", calendar=target),
        dims=("time",),
        name="time",
    )
    da_src = DataArray(
        np.linspace(0, 1, src.size), dims=("time",), coords={"time": src}
    )
    conv = interp_calendar(da_src, tgt)

    assert conv.size == tgt.size
    assert conv.time.dt.calendar == target

    np.testing.assert_almost_equal(conv.max(), 1, 2)
    assert conv.min() == 0
