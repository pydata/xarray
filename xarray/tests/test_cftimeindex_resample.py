import datetime

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core.resample_cftime import CFTimeGrouper

pytest.importorskip("cftime")


# Create a list of pairs of similar-length initial and resample frequencies
# that cover:
# - Resampling from shorter to longer frequencies
# - Resampling from longer to shorter frequencies
# - Resampling from one initial frequency to another.
# These are used to test the cftime version of resample against pandas
# with a standard calendar.
FREQS = [
    ("8003D", "4001D"),
    ("8003D", "16006D"),
    ("8003D", "21AS"),
    ("6H", "3H"),
    ("6H", "12H"),
    ("6H", "400T"),
    ("3D", "D"),
    ("3D", "6D"),
    ("11D", "MS"),
    ("3MS", "MS"),
    ("3MS", "6MS"),
    ("3MS", "85D"),
    ("7M", "3M"),
    ("7M", "14M"),
    ("7M", "2QS-APR"),
    ("43QS-AUG", "21QS-AUG"),
    ("43QS-AUG", "86QS-AUG"),
    ("43QS-AUG", "11A-JUN"),
    ("11Q-JUN", "5Q-JUN"),
    ("11Q-JUN", "22Q-JUN"),
    ("11Q-JUN", "51MS"),
    ("3AS-MAR", "AS-MAR"),
    ("3AS-MAR", "6AS-MAR"),
    ("3AS-MAR", "14Q-FEB"),
    ("7A-MAY", "3A-MAY"),
    ("7A-MAY", "14A-MAY"),
    ("7A-MAY", "85M"),
]


def da(index):
    return xr.DataArray(
        np.arange(100.0, 100.0 + index.size), coords=[index], dims=["time"]
    )


@pytest.mark.parametrize("freqs", FREQS, ids=lambda x: "{}->{}".format(*x))
@pytest.mark.parametrize("closed", [None, "left", "right"])
@pytest.mark.parametrize("label", [None, "left", "right"])
@pytest.mark.parametrize("base", [24, 31])
def test_resample(freqs, closed, label, base):
    initial_freq, resample_freq = freqs
    start = "2000-01-01T12:07:01"
    index_kwargs = dict(start=start, periods=5, freq=initial_freq)
    datetime_index = pd.date_range(**index_kwargs)
    cftime_index = xr.cftime_range(**index_kwargs)

    loffset = "12H"
    try:
        da_datetime = (
            da(datetime_index)
            .resample(
                time=resample_freq,
                closed=closed,
                label=label,
                base=base,
                loffset=loffset,
            )
            .mean()
        )
    except ValueError:
        with pytest.raises(ValueError):
            da(cftime_index).resample(
                time=resample_freq,
                closed=closed,
                label=label,
                base=base,
                loffset=loffset,
            ).mean()
    else:
        da_cftime = (
            da(cftime_index)
            .resample(
                time=resample_freq,
                closed=closed,
                label=label,
                base=base,
                loffset=loffset,
            )
            .mean()
        )
        # TODO (benbovy - flexible indexes): update when CFTimeIndex is a xarray Index subclass
        da_cftime["time"] = (
            da_cftime.xindexes["time"].to_pandas_index().to_datetimeindex()
        )
        xr.testing.assert_identical(da_cftime, da_datetime)


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("S", "left"),
        ("T", "left"),
        ("H", "left"),
        ("D", "left"),
        ("M", "right"),
        ("MS", "left"),
        ("Q", "right"),
        ("QS", "left"),
        ("A", "right"),
        ("AS", "left"),
    ],
)
def test_closed_label_defaults(freq, expected):
    assert CFTimeGrouper(freq=freq).closed == expected
    assert CFTimeGrouper(freq=freq).label == expected


@pytest.mark.filterwarnings("ignore:Converting a CFTimeIndex")
@pytest.mark.parametrize(
    "calendar", ["gregorian", "noleap", "all_leap", "360_day", "julian"]
)
def test_calendars(calendar):
    # Limited testing for non-standard calendars
    freq, closed, label, base = "8001T", None, None, 17
    loffset = datetime.timedelta(hours=12)
    xr_index = xr.cftime_range(
        start="2004-01-01T12:07:01", periods=7, freq="3D", calendar=calendar
    )
    pd_index = pd.date_range(start="2004-01-01T12:07:01", periods=7, freq="3D")
    da_cftime = (
        da(xr_index)
        .resample(time=freq, closed=closed, label=label, base=base, loffset=loffset)
        .mean()
    )
    da_datetime = (
        da(pd_index)
        .resample(time=freq, closed=closed, label=label, base=base, loffset=loffset)
        .mean()
    )
    # TODO (benbovy - flexible indexes): update when CFTimeIndex is a xarray Index subclass
    da_cftime["time"] = da_cftime.xindexes["time"].to_pandas_index().to_datetimeindex()
    xr.testing.assert_identical(da_cftime, da_datetime)
