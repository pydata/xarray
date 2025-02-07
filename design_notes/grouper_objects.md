# Grouper Objects

**Author**: Deepak Cherian <deepak@cherian.net>
**Created**: Nov 21, 2023

## Abstract

I propose the addition of Grouper objects to Xarray's public API so that

```python
Dataset.groupby(x=BinGrouper(bins=np.arange(10, 2))))
```

is identical to today's syntax:

```python
Dataset.groupby_bins("x", bins=np.arange(10, 2))
```

## Motivation and scope

Xarray's GroupBy API implements the split-apply-combine pattern (Wickham, 2011)[^1], which applies to a very large number of problems: histogramming, compositing, climatological averaging, resampling to a different time frequency, etc.
The pattern abstracts the following pseudocode:

```python
results = []
for element in unique_labels:
    subset = ds.sel(x=(ds.x == element))  # split
    # subset = ds.where(ds.x == element, drop=True)  # alternative
    result = subset.mean() # apply
    results.append(result)

xr.concat(results)  # combine
```

to

```python
ds.groupby('x').mean()  # splits, applies, and combines
```

Efficient vectorized implementations of this pattern are implemented in numpy's [`ufunc.at`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html), [`ufunc.reduceat`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduceat.html), [`numbagg.grouped`](https://github.com/numbagg/numbagg/blob/main/numbagg/grouped.py), [`numpy_groupies`](https://github.com/ml31415/numpy-groupies), and probably more.
These vectorized implementations _all_ require, as input, an array of integer codes or labels that identify unique elements in the array being grouped over (`'x'` in the example above).

```python
import numpy as np

# array to reduce
a = np.array([1, 1, 1, 1, 2])

# initial value for result
out = np.zeros((3,), dtype=int)

# integer codes
labels = np.array([0, 0, 1, 2, 1])

# groupby-reduction
np.add.at(out, labels, a)
out  # array([2, 3, 1])
```

One can 'factorize' or construct such an array of integer codes using `pandas.factorize` or `numpy.unique(..., return_inverse=True)` for categorical arrays; `pandas.cut`, `pandas.qcut`, or `np.digitize` for discretizing continuous variables.
In practice, since `GroupBy` objects exist, much of complexity in applying the groupby paradigm stems from appropriately factorizing or generating labels for the operation.
Consider these two examples:

1. [Bins that vary in a dimension](https://flox.readthedocs.io/en/latest/user-stories/nD-bins.html)
2. [Overlapping groups](https://flox.readthedocs.io/en/latest/user-stories/overlaps.html)
3. [Rolling resampling](https://github.com/pydata/xarray/discussions/8361)

Anecdotally, less experienced users commonly resort to the for-loopy implementation illustrated by the pseudocode above when the analysis at hand is not easily expressed using the API presented by Xarray's GroupBy object.
Xarray's GroupBy API today abstracts away the split, apply, and combine stages but not the "factorize" stage.
Grouper objects will close the gap.

## Usage and impact

Grouper objects

1. Will abstract useful factorization algorithms, and
2. Present a natural way to extend GroupBy to grouping by multiple variables: `ds.groupby(x=BinGrouper(...), t=Resampler(freq="M", ...)).mean()`.

In addition, Grouper objects provide a nice interface to add often-requested grouping functionality

1. A new `SpaceResampler` would allow specifying resampling spatial dimensions. ([issue](https://github.com/pydata/xarray/issues/4008))
2. `RollingTimeResampler` would allow rolling-like functionality that understands timestamps ([issue](https://github.com/pydata/xarray/issues/3216))
3. A `QuantileBinGrouper` to abstract away `pd.cut` ([issue](https://github.com/pydata/xarray/discussions/7110))
4. A `SeasonGrouper` and `SeasonResampler` would abstract away common annoyances with such calculations today
   1. Support seasons that span a year-end.
   2. Only include seasons with complete data coverage.
   3. Allow grouping over seasons of unequal length
   4. See [this xcdat discussion](https://github.com/xCDAT/xcdat/issues/416) for a `SeasonGrouper` like functionality:
   5. Return results with seasons in a sensible order
5. Weighted grouping ([issue](https://github.com/pydata/xarray/issues/3937))
   1. Once `IntervalIndex` like objects are supported, `Resampler` groupers can account for interval lengths when resampling.

## Backward Compatibility

Xarray's existing grouping functionality will be exposed using two new Groupers:

1. `UniqueGrouper` which uses `pandas.factorize`
2. `BinGrouper` which uses `pandas.cut`
3. `TimeResampler` which mimics pandas' `.resample`

Grouping by single variables will be unaffected so that `ds.groupby('x')` will be identical to `ds.groupby(x=UniqueGrouper())`.
Similarly, `ds.groupby_bins('x', bins=np.arange(10, 2))` will be unchanged and identical to `ds.groupby(x=BinGrouper(bins=np.arange(10, 2)))`.

## Detailed description

All Grouper objects will subclass from a Grouper object

```python
import abc

class Grouper(abc.ABC):
    @abc.abstractmethod
    def factorize(self, by: DataArray):
        raise NotImplementedError

class CustomGrouper(Grouper):
    def factorize(self, by: DataArray):
        ...
        return codes, group_indices, unique_coord, full_index

    def weights(self, by: DataArray) -> DataArray:
        ...
        return weights
```

### The `factorize` method

Today, the `factorize` method takes as input the group variable and returns 4 variables (I propose to clean this up below):

1. `codes`: An array of same shape as the `group` with int dtype. NaNs in `group` are coded by `-1` and ignored later.
2. `group_indices` is a list of index location of `group` elements that belong to a single group.
3. `unique_coord` is (usually) a `pandas.Index` object of all unique `group` members present in `group`.
4. `full_index` is a `pandas.Index` of all `group` members. This is different from `unique_coord` for binning and resampling, where not all groups in the output may be represented in the input `group`. For grouping by a categorical variable e.g. `['a', 'b', 'a', 'c']`, `full_index` and `unique_coord` are identical.
   There is some redundancy here since `unique_coord` is always equal to or a subset of `full_index`.
   We can clean this up (see Implementation below).

### The `weights` method (?)

The proposed `weights` method is optional and unimplemented today.
Groupers with `weights` will allow composing `weighted` and `groupby` ([issue](https://github.com/pydata/xarray/issues/3937)).
The `weights` method should return an appropriate array of weights such that the following property is satisfied

```python
gb_sum = ds.groupby(by).sum()

weights = CustomGrouper.weights(by)
weighted_sum = xr.dot(ds, weights)

assert_identical(gb_sum, weighted_sum)
```

For example, the boolean weights for `group=np.array(['a', 'b', 'c', 'a', 'a'])` should be

```
[[1, 0, 0, 1, 1],
 [0, 1, 0, 0, 0],
 [0, 0, 1, 0, 0]]
```

This is the boolean "summarization matrix" referred to in the classic Iverson (1980, Section 4.3)[^2] and "nub sieve" in [various APLs](https://aplwiki.com/wiki/Nub_Sieve).

> [!NOTE]
> We can always construct `weights` automatically using `group_indices` from `factorize`, so this is not a required method.

For a rolling resampling, windowed weights are possible

```
[[0.5, 1,    0.5, 0, 0],
 [0,   0.25, 1,   1, 0],
 [0,   0,    0,   1, 1]]
```

### The `preferred_chunks` method (?)

Rechunking support is another optional extension point.
In `flox` I experimented some with automatically rechunking to make a groupby more parallel-friendly ([example 1](https://flox.readthedocs.io/en/latest/generated/flox.rechunk_for_blockwise.html), [example 2](https://flox.readthedocs.io/en/latest/generated/flox.rechunk_for_cohorts.html)).
A great example is for resampling-style groupby reductions, for which `codes` might look like

```
0001|11122|3333
```

where `|` represents chunk boundaries. A simple rechunking to

```
000|111122|3333
```

would make this resampling reduction an embarrassingly parallel blockwise problem.

Similarly consider monthly-mean climatologies for which the month numbers might be

```
1 2 3 4 5 | 6 7 8 9 10 | 11 12 1 2 3 | 4 5 6 7 8 | 9 10 11 12 |
```

A slight rechunking to

```
1 2 3 4 | 5 6 7 8 | 9 10 11 12 | 1 2 3 4 | 5 6 7 8 | 9 10 11 12 |
```

allows us to reduce `1, 2, 3, 4` separately from `5,6,7,8` and `9, 10, 11, 12` while still being parallel friendly (see the [flox documentation](https://flox.readthedocs.io/en/latest/implementation.html#method-cohorts) for more).

We could attempt to detect these patterns, or we could just have the Grouper take as input `chunks` and return a tuple of "nice" chunk sizes to rechunk to.

```python
def preferred_chunks(self, chunks: ChunksTuple) -> ChunksTuple:
    pass
```

For monthly means, since the period of repetition of labels is 12, the Grouper might choose possible chunk sizes of `((2,),(3,),(4,),(6,))`.
For resampling, the Grouper could choose to resample to a multiple or an even fraction of the resampling frequency.

## Related work

Pandas has [Grouper objects](https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html#pandas-grouper) that represent the GroupBy instruction.
However, these objects do not appear to be extension points, unlike the Grouper objects proposed here.
Instead, Pandas' `ExtensionArray` has a [`factorize`](https://pandas.pydata.org/docs/reference/api/pandas.api.extensions.ExtensionArray.factorize.html) method.

Composing rolling with time resampling is a common workload:

1. Polars has [`group_by_dynamic`](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.group_by_dynamic.html) which appears to be like the proposed `RollingResampler`.
2. scikit-downscale provides [`PaddedDOYGrouper`](https://github.com/pangeo-data/scikit-downscale/blob/e16944a32b44f774980fa953ea18e29a628c71b8/skdownscale/pointwise_models/groupers.py#L19)

## Implementation Proposal

1. Get rid of `squeeze` [issue](https://github.com/pydata/xarray/issues/2157): [PR](https://github.com/pydata/xarray/pull/8506)
2. Merge existing two class implementation to a single Grouper class
   1. This design was implemented in [this PR](https://github.com/pydata/xarray/pull/7206) to account for some annoying data dependencies.
   2. See [PR](https://github.com/pydata/xarray/pull/8509)
3. Clean up what's returned by `factorize` methods.
   1. A solution here might be to have `group_indices: Mapping[int, Sequence[int]]` be a mapping from group index in `full_index` to a sequence of integers.
   2. Return a `namedtuple` or `dataclass` from existing Grouper factorize methods to facilitate API changes in the future.
4. Figure out what to pass to `factorize`
   1. Xarray eagerly reshapes nD variables to 1D. This is an implementation detail we need not expose.
   2. When grouping by an unindexed variable Xarray passes a `_DummyGroup` object. This seems like something we don't want in the public interface. We could special case "internal" Groupers to preserve the optimizations in `UniqueGrouper`.
5. Grouper objects will exposed under the `xr.groupers` Namespace. At first these will include `UniqueGrouper`, `BinGrouper`, and `TimeResampler`.

## Alternatives

One major design choice made here was to adopt the syntax `ds.groupby(x=BinGrouper(...))` instead of `ds.groupby(BinGrouper('x', ...))`.
This allows reuse of Grouper objects, example

```python
grouper = BinGrouper(...)
ds.groupby(x=grouper, y=grouper)
```

but requires that all variables being grouped by (`x` and `y` above) are present in Dataset `ds`. This does not seem like a bad requirement.
Importantly `Grouper` instances will be copied internally so that they can safely cache state that might be shared between `factorize` and `weights`.

Today, it is possible to `ds.groupby(DataArray, ...)`. This syntax will still be supported.

## Discussion

This proposal builds on these discussions:

1. https://github.com/xarray-contrib/flox/issues/191#issuecomment-1328898836
2. https://github.com/pydata/xarray/issues/6610

## Copyright

This document has been placed in the public domain.

## References and footnotes

[^1]: Wickham, H. (2011). The split-apply-combine strategy for data analysis. https://vita.had.co.nz/papers/plyr.html

[^2]: Iverson, K.E. (1980). Notation as a tool of thought. Commun. ACM 23, 8 (Aug. 1980), 444–465. https://doi.org/10.1145/358896.358899
