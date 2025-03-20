import itertools

import pytest

pytest.importorskip("hypothesis")

import hypothesis.strategies as st
from hypothesis import given, note

import xarray as xr
import xarray.testing.strategies as xrst
from xarray.groupers import find_independent_seasons, season_to_month_tuple


@given(attrs=xrst.simple_attrs)
def test_assert_identical(attrs):
    v = xr.Variable(dims=(), data=0, attrs=attrs)
    xr.testing.assert_identical(v, v.copy(deep=True))

    ds = xr.Dataset(attrs=attrs)
    xr.testing.assert_identical(ds, ds.copy(deep=True))


@given(
    roll=st.integers(min_value=0, max_value=12),
    breaks=st.lists(
        st.integers(min_value=0, max_value=11), min_size=1, max_size=12, unique=True
    ),
)
def test_property_season_month_tuple(roll, breaks):
    chars = list("JFMAMJJASOND")
    months = tuple(range(1, 13))

    rolled_chars = chars[roll:] + chars[:roll]
    rolled_months = months[roll:] + months[:roll]
    breaks = sorted(breaks)
    if breaks[0] != 0:
        breaks = [0] + breaks
    if breaks[-1] != 12:
        breaks = breaks + [12]
    seasons = tuple(
        "".join(rolled_chars[start:stop]) for start, stop in itertools.pairwise(breaks)
    )
    actual = season_to_month_tuple(seasons)
    expected = tuple(
        rolled_months[start:stop] for start, stop in itertools.pairwise(breaks)
    )
    assert expected == actual


@given(data=st.data(), nmonths=st.integers(min_value=1, max_value=11))
def test_property_find_independent_seasons(data, nmonths):
    chars = "JFMAMJJASOND"
    # if stride > nmonths, then we can't infer season order
    stride = data.draw(st.integers(min_value=1, max_value=nmonths))
    chars = chars + chars[:nmonths]
    seasons = [list(chars[i : i + nmonths]) for i in range(0, 12, stride)]
    note(seasons)
    groups = find_independent_seasons(seasons)
    for group in groups:
        inds = tuple(itertools.chain(*group.inds))
        assert len(inds) == len(set(inds))
        assert len(group.codes) == len(set(group.codes))
