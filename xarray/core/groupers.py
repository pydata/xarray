import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from xarray.core.groupby import Grouper, Resampler
from xarray.core.variable import IndexVariable


## From toolz
## TODO: move to compat file, add license
def sliding_window(n, seq):
    """A sequence of overlapping subsequences

    >>> list(sliding_window(2, [1, 2, 3, 4]))
    [(1, 2), (2, 3), (3, 4)]

    This function creates a sliding window suitable for transformations like
    sliding means / smoothing

    >>> mean = lambda seq: float(sum(seq)) / len(seq)
    >>> list(map(mean, sliding_window(2, [1, 2, 3, 4])))
    [1.5, 2.5, 3.5]
    """
    import collections
    import itertools

    return zip(
        *(
            collections.deque(itertools.islice(it, i), 0) or it
            for i, it in enumerate(itertools.tee(seq, n))
        )
    )


def season_to_month_tuple(seasons: Sequence[str]) -> Sequence[Sequence[int]]:
    easy = {"D": 12, "F": 2, "S": 9, "O": 10, "N": 11}
    harder = {"DJF": 1, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8}

    if len("".join(seasons)) != 12:
        raise ValueError("SeasonGrouper requires exactly 12 months in total.")

    # Slide through with a window of 3.
    # A 3 letter string is enough to unambiguously
    # assign the right month number of the middle letter
    WINDOW = 3

    perseason = [seasons[-1], *seasons, seasons[0]]

    season_inds = []
    for sprev, sthis, snxt in sliding_window(WINDOW, perseason):
        inds = []
        permonth = "".join([sprev[-1], *sthis, snxt[0]])
        for mprev, mthis, mnxt in sliding_window(WINDOW, permonth):
            if mthis in easy:
                inds.append(easy[mthis])
            else:
                concatted = "".join([mprev, mthis, mnxt])
                # print(concatted)
                inds.append(harder[concatted])

        season_inds.append(inds)
    return season_inds


@dataclass
class SeasonGrouper(Grouper):
    """Allows grouping using a custom definition of seasons.

    Parameters
    ----------
    seasons: sequence of str
        List of strings representing seasons. E.g. ``"JF"`` or ``"JJA"`` etc.
    drop_incomplete: bool
        Whether to drop seasons that are not completely included in the data.
        For example, if a time series starts in Jan-2001, and seasons includes `"DJF"`
        then observations from Jan-2001, and Feb-2001 are ignored in the grouping
        since Dec-2000 isn't present.

    Examples
    --------
    >>> SeasonGrouper(["JF", "MAM", "JJAS", "OND"])
    >>> SeasonGrouper(["DJFM", "AM", "JJA", "SON"])
    """

    seasons: Sequence[str]
    season_inds: Sequence[Sequence[int]] = field(init=False)
    drop_incomplete: bool = field(default=True)

    def __post_init__(self):
        self.season_inds = season_to_month_tuple(self.seasons)

    def __repr__(self):
        return f"SeasonGrouper over {self.grouper.seasons!r}"

    def factorize(self, group):
        seasons = self.seasons
        season_inds = self.season_inds

        months = group.dt.month
        codes_ = np.full(group.shape, -1)
        group_indices = [[]] * len(seasons)

        index = np.arange(group.size)
        for idx, season in enumerate(season_inds):
            mask = months.isin(season)
            codes_[mask] = idx
            group_indices[idx] = index[mask]

        if np.all(codes_ == -1):
            raise ValueError(
                "Failed to group data. Are you grouping by a variable that is all NaN?"
            )
        codes = group.copy(data=codes_).rename("season")
        unique_coord = IndexVariable("season", seasons, attrs=group.attrs)
        full_index = unique_coord
        return codes, group_indices, unique_coord, full_index


@dataclass
class SeasonResampler(Resampler):
    """Allows grouping using a custom definition of seasons.

    Examples
    --------
    >>> SeasonResampler(["JF", "MAM", "JJAS", "OND"])
    >>> SeasonResampler(["DJFM", "AM", "JJA", "SON"])
    """

    seasons: Sequence[str]
    # drop_incomplete: bool = field(default=True)  # TODO:
    season_inds: Sequence[Sequence[int]] = field(init=False)
    season_tuples: Mapping[str, Sequence[int]] = field(init=False)

    def __post_init__(self):
        self.season_inds = season_to_month_tuple(self.seasons)
        self.season_tuples = dict(zip(self.seasons, self.season_inds))

    def factorize(self, group):
        assert group.ndim == 1

        seasons = self.seasons
        season_inds = self.season_inds
        season_tuples = self.season_tuples

        nstr = max(len(s) for s in seasons)
        year = group.dt.year.astype(int)
        month = group.dt.month.astype(int)
        season_label = np.full(group.shape, "", dtype=f"U{nstr}")

        # offset years for seasons with December and January
        for season_str, season_ind in zip(seasons, season_inds):
            season_label[month.isin(season_ind)] = season_str
            if "DJ" in season_str:
                after_dec = season_ind[season_str.index("D") + 1 :]
                year[month.isin(after_dec)] -= 1

        frame = pd.DataFrame(
            data={"index": np.arange(group.size), "month": month},
            index=pd.MultiIndex.from_arrays(
                [year.data, season_label], names=["year", "season"]
            ),
        )

        series = frame["index"]
        g = series.groupby(["year", "season"], sort=False)
        first_items = g.first()
        counts = g.count()

        # these are the seasons that are present
        unique_coord = pd.DatetimeIndex(
            [
                pd.Timestamp(year=year, month=season_tuples[season][0], day=1)
                for year, season in first_items.index
            ]
        )

        sbins = first_items.values.astype(int)
        group_indices = [slice(i, j) for i, j in zip(sbins[:-1], sbins[1:])]
        group_indices += [slice(sbins[-1], None)]

        # Make sure the first and last timestamps
        # are for the correct months,if not we have incomplete seasons
        unique_codes = np.arange(len(unique_coord))
        for idx, slicer in zip([0, -1], (slice(1, None), slice(-1))):
            stamp_year, stamp_season = frame.index[idx]
            code = seasons.index(stamp_season)
            stamp_month = season_inds[code][idx]
            if stamp_month != month[idx].item():
                # we have an incomplete season!
                group_indices = group_indices[slicer]
                unique_coord = unique_coord[slicer]
                if idx == 0:
                    unique_codes -= 1
                unique_codes[idx] = -1

        # all years and seasons
        complete_index = pd.DatetimeIndex(
            # This sorted call is a hack. It's hard to figure out how
            # to start the iteration
            sorted(
                [
                    pd.Timestamp(f"{y}-{m}-01")
                    for y, m in itertools.product(
                        range(year[0].item(), year[-1].item() + 1),
                        [s[0] for s in season_inds],
                    )
                ]
            )
        )
        # only keep that included in data
        range_ = complete_index.get_indexer(unique_coord[[0, -1]])
        full_index = complete_index[slice(range_[0], range_[-1] + 1)]
        # check that there are no "missing" seasons in the middle
        # print(full_index, unique_coord)
        if not full_index.equals(unique_coord):
            raise ValueError("Are there seasons missing in the middle of the dataset?")

        codes = group.copy(data=np.repeat(unique_codes, counts))
        unique_coord_var = IndexVariable(group.name, unique_coord, group.attrs)

        return codes, group_indices, unique_coord_var, full_index
