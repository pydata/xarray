# Proposal: Xarray flexible indexes refactoring

Current status: https://github.com/pydata/xarray/projects/1

## 1. Data Model

Indexes are used in Xarray to extract data from Xarray objects using coordinate labels instead of using integer array indices. Although the indexes used in an Xarray object can be accessed (or built on-the-fly) via public methods like `to_index()` or properties like `indexes`, those are mainly used internally.

The goal of this project is to make those indexes 1st-class citizens of Xarray's data model. As such, indexes should clearly be separated from Xarray coordinates with the following relationships:

- Index -> Coordinate: one-to-many
- Coordinate -> Index: one-to-zero-or-one

An index may be built from one or more coordinates. However, each coordinate must relate to one index at most. Additionally, a coordinate may not be tied to any index.

The order in which multiple coordinates relate to an index should matter. For example, Scikit-Learn's `BallTree` index with the Haversine metric requires providing latitude and longitude values in that specific order. As another example, the order in which levels are defined in a `pandas.MultiIndex` may affect its lexsort depth.

Xarray's current data model is already based on the same index-coordinate relationships. The current data model also implies a one-to-one relationship between a dimension and an index. This one-to-one relationship works as currently it is not possible to perform label-based data selection in Xarray using multi-dimensional coordinates, but since we want to enable this feature in this proposal, we need the dimension-index relationship to evolve towards many-to-many.

In the example below, we'd like to select data points based on their x, y values and/or on their latitude/longitude positions:

```python
>>> da
<xarray.DataArray (x: 2, y: 2)>
array([[5.4, 7.8],
       [6.2, 4.7]])
Coordinates:
  * lon      (x, y) float64 10.2 15.2 12.6 17.6
  * lat      (x, y) float64 40.2 45.6 42.2 47.6
  * x        (x) float64 200.0 400.0
  * y        (y) float64 800.0 1e+03

>>> da.sel(lon=..., lat=...)
>>> da.sel(x=..., y=...)
```

We would need one geographic index for the `lat` and `lon` coordinates and two indexes for the `x` and `y` coordinates, respectively.

## 2. Proposed API changes

### 2.1 Index wrapper classes

Every index that is used to select data from Xarray objects should inherit from a base class, e.g., `XarrayIndex`, that provides some common API. `XarrayIndex` subclasses would generally consist of thin wrappers around existing index classes such as `pandas.Index`, `pandas.MultiIndex`, `scipy.spatial.KDTree`, etc.

There is a variety of features that an xarray index wrapper may or may not support:

- 1-dimensional vs. 2-dimensional vs. n-dimensional coordinate
- built from a single vs multiple coordinate(s)
- in-memory vs. out-of-core (dask) index data/coordinates (vs. other array backends)
- range-based vs. point-wise selection
- exact vs. inexact lookups

Whether or not a `XarrayIndex` subclass supports each of the features listed above should be either declared explicitly via a common API or left to the implementation. An `XarrayIndex` subclass may encapsulate more than one underlying object used to perform the actual indexing. Such "meta" index would typically support a range of features among those mentioned above and would automatically select the optimal index object for a given indexing operation.

Every `XarrayIndex` subclass must at least implement two methods:

- One `build` method that takes one or more Dataset/DataArray coordinates and that returns the object(s) that will be used for the actual indexing (e.g., `pandas.Index`, `scipy.spatial.KDTree`, etc.)
- One `query` method that takes label-based indexers as argument and that returns the corresponding position-based indexers.

These two methods may accept additional keyword arguments passed to the underlying index object constructor or query methods.

There are potentially other properties / methods that an `XarrayIndex` subclass must/should/may implement, e.g.,

- An `indexes` property to access the underlying index object(s) wrapped by the `XarrayIndex` subclass
- a `__getitem__()` implementation to propagate the index through DataArray/Dataset indexing operations
- `equals()`, `union()` and `intersection()` methods for data alignment (see [Section 2.6](#26-using-indexes-for-data-alignment))
- Xarray coordinate getters (see [Section 2.2.4](#224-implicit-coodinates))
- A method that may return a new index and that will be called when one of the corresponding coordinates is dropped from the Dataset/DataArray (multi-coordinate indexes)
- `encode()`/`decode()` methods that would allow storage-agnostic serialization and fast-path reconstruction of the underlying index object(s) (see [Section 2.8](#28-index-encoding))
- One or more "non-standard" methods or properties that could be leveraged in Xarray 3rd-party extensions like Dataset/DataArray accessors (see [Section 2.7](#27-using-indexes-for-other-purposes))

The `XarrayIndex` API has still to be defined in detail.

Xarray should provide a minimal set of built-in index wrappers (this could be reduced to the indexes currently supported in Xarray, i.e., `pandas.Index` and `pandas.MultiIndex`). Other index wrappers may be implemented in 3rd-party libraries (recommended). The `XarrayIndex` base class should be part of Xarray's public API.

#### 2.1.1 Index discoverability

For better discoverability of Xarray-compatible indexes, Xarray could provide some mechanism to register new index wrappers, e.g., something lire [xoak's `IndexRegistry`](https://xoak.readthedocs.io/en/latest/_api_generated/xoak.IndexRegistry.html#xoak.IndexRegistry).

Additionally (or alternatively), new index wrappers may be registered via entry points like it is already the case for storage backends and maybe other backends (plotting) in the future.

`XarrayIndex` subclasses may still be used directly when setting new indexes from DataArray/Dataset coordinates.

### 2.2 Explicit vs. implicit index creation

#### 2.2.1 Dataset/DataArray's `indexes` constructor argument

The new `indexes` argument of Dataset/DataArray constructors may be used to specify which kind of index to bind to which coordinate(s). It would consist of a mapping where, for each item, the key is one coordinate name (or a sequence of coordinate names) that must be given in `coords` and the value is the type of the index to build from this (these) coordinate(s):

```python
>>> da = xr.DataArray(
...     data=[[275.2, 273.5], [270.8, 278.6]],
...     dims=('x', 'y'),
...     coords={
...         'lat': (('x', 'y'), [[45.6, 46.5], [50.2, 51.6]]),
...         'lon': (('x', 'y'), [[5.7, 10.5], [6.2, 12.8]]),
...     },
...     indexes={('lat', 'lon'): SpatialIndex},
... )
<xarray.DataArray (x: 2, y: 2)>
array([[275.2, 273.5],
       [270.8, 278.6]])
Coordinates:
  * lat      (x, y) float64 45.6 46.5 50.2 51.6
  * lon      (x, y) float64 5.7 10.5 6.2 12.8
```

More formally, `indexes` would accept `Mapping[CoordinateNames, IndexSpec]` where:

- `CoordinateNames = Union[CoordinateName, Tuple[CoordinateName, ...]]` and `CoordinateName = Hashable`
- `IndexSpec = Union[Type[XarrayIndex], Tuple[XarrayIndex, Dict[str, Any]], XarrayIndex]`, so that index instances or index classes + build options could be also passed

Currently index objects like `pandas.MultiIndex` can be passed directly to `coords`, which in this specific case results in the implicit creation of virtual coordinates. With the new `indexes` argument this behavior may become even more confusing than it currently is. For the sake of clarity, it would be appropriate to eventually drop support for this specific behavior and treat any given mapping value given in `coords` as an array that can be wrapped into an Xarray variable, i.e., in the case of a multi-index:

```python
>>> xr.DataArray([1.0, 2.0], dims='x', coords={'x': midx})
<xarray.DataArray (x: 2)>
array([1., 2.])
Coordinates:
    x        (x) object ('a', 0) ('b', 1)
```

A possible, more explicit solution to reuse a `pandas.MultiIndex` in a DataArray/Dataset with levels exposed as coordinates is proposed in [Section 2.2.4](#224-implicit-coordinates).

#### 2.2.2 Dataset/DataArray's `set_index` method

New indexes may also be built from existing sets of coordinates or variables in a Dataset/DataArray using the `.set_index()` method.

The [current signature](http://xarray.pydata.org/en/stable/generated/xarray.DataArray.set_index.html#xarray.DataArray.set_index) of `.set_index()` is tailored to `pandas.MultiIndex` and tied to the concept of a dimension-index. It is therefore hardly reusable as-is in the context of flexible indexes proposed here.

The new signature may look like one of these:

- A. `.set_index(coords: CoordinateNames, index: Union[XarrayIndex, Type[XarrayIndex]], **index_kwargs)`: one index is set at a time, index construction options may be passed as keyword arguments
- B. `.set_index(indexes: Mapping[CoordinateNames, Union[Type[XarrayIndex], Tuple[Type[XarrayIndex], Dict[str, Any]]]])`: multiple indexes may be set at a time from a mapping of coordinate or variable name(s) as keys and `XarrayIndex` subclasses (maybe with a dict of build options) as values. If variable names are given as keys of they will be promoted as coordinates

Option A looks simple and elegant but significantly departs from the current signature. Option B is more consistent with the Dataset/DataArray constructor signature proposed in the previous section and would be easier to adopt in parallel with the current signature that we could still support through some depreciation cycle.

The `append` parameter of the current `.set_index()` is specific to `pandas.MultiIndex`. With option B we could still support it, although we might want to either drop it or move it to the index construction options in the future.

#### 2.2.3 Implicit default indexes

In general explicit index creation should be preferred over implicit index creation. However, there is a majority of cases where basic `pandas.Index` objects could be built and used as indexes for 1-dimensional coordinates. For convenience, Xarray should automatically build such indexes for the coordinates where no index has been explicitly assigned in the Dataset/DataArray constructor or when indexes have been reset / dropped.

For which coordinates?

- A. only 1D coordinates with a name matching their dimension name
- B. all 1D coordinates

When to create it?

- A. each time when a new Dataset/DataArray is created
- B. only when we need it (i.e., when calling `.sel()` or `indexes`)

Options A and A are what Xarray currently does and may be the best choice considering that indexes could possibly be invalidated by coordinate mutation.

Besides `pandas.Index`, other indexes currently supported in Xarray like `CFTimeIndex` could be built depending on the coordinate data type.

#### 2.2.4 Implicit coordinates

Like for the indexes, explicit coordinate creation should be preferred over implicit coordinate creation. However, there may be some situations where we would like to keep creating coordinates implicitly for backwards compatibility.

For example, it is currently possible to pass a `pandas.MulitIndex` object as a coordinate to the Dataset/DataArray constructor:

```python
>>> midx = pd.MultiIndex.from_arrays([['a', 'b'], [0, 1]], names=['lvl1', 'lvl2'])
>>> da = xr.DataArray([1.0, 2.0], dims='x', coords={'x': midx})
>>> da
<xarray.DataArray (x: 2)>
array([1., 2.])
Coordinates:
  * x        (x) MultiIndex
  - lvl1     (x) object 'a' 'b'
  - lvl2     (x) int64 0 1
```

In that case, virtual coordinates are created for each level of the multi-index. After the index refactoring, these coordinates would become real coordinates bound to the multi-index.

In the example above a coordinate is also created for the `x` dimension:

```python
>>> da.x
<xarray.DataArray 'x' (x: 2)>
array([('a', 0), ('b', 1)], dtype=object)
Coordinates:
  * x        (x) MultiIndex
  - lvl1     (x) object 'a' 'b'
  - lvl2     (x) int64 0 1
```

With the new proposed data model, this wouldn't be a requirement anymore: there is no concept of a dimension-index. However, some users might still rely on the `x` coordinate so we could still (temporarily) support it for backwards compatibility.

Besides `pandas.MultiIndex`, there may be other situations where we would like to reuse an existing index in a new Dataset/DataArray (e.g., when the index is very expensive to build), and which might require implicit creation of one or more coordinates.

The example given here is quite confusing, though: this is not an easily predictable behavior. We could entirely avoid the implicit creation of coordinates, e.g., using a helper function that generates coordinate + index dictionaries that we could then pass directly to the DataArray/Dataset constructor:

```python
>>> coords_dict, index_dict = create_coords_from_index(midx, dims='x', include_dim_coord=True)
>>> coords_dict
{'x': <xarray.Variable (x: 2)>
 array([('a', 0), ('b', 1)], dtype=object),
 'lvl1': <xarray.Variable (x: 2)>
 array(['a', 'b'], dtype=object),
 'lvl2': <xarray.Variable (x: 2)>
 array([0, 1])}
>>> index_dict
{('lvl1', 'lvl2'): midx}
>>> xr.DataArray([1.0, 2.0], dims='x', coords=coords_dict, indexes=index_dict)
<xarray.DataArray (x: 2)>
array([1., 2.])
Coordinates:
    x        (x) object ('a', 0) ('b', 1)
  * lvl1     (x) object 'a' 'b'
  * lvl2     (x) int64 0 1
```

### 2.3 Index access

#### 2.3.1 Dataset/DataArray's `indexes` property

The `indexes` property would allow easy access to all the indexes used in a Dataset/DataArray. There may be different options for the type returned:

A. `Dict[CoordinateName, XarrayIndex]`: keys are coordinate names and values may be duplicated
B. `Dict[CoordinateNames, XarrayIndex]`: keys may represent one or more coordinate names and values are unique

Option A allows easy index look-up by coordinate name, while option B allows easy iteration through all the indexes. Option A may be more useful than option B for many tasks and may also be less ambiguous if more complex hashable types than `str` are used for `CoordinateName`.

#### 2.3.2 Additional Dataset/DataArray's properties or methods

Both options A and B in the section above have pros and cons. For convenience, we could maybe add one more property / method to get the indexes in the desired format.

### 2.4 Propagate indexes through operations

#### 2.4.1 Mutable coordinates

Dataset/DataArray coordinates may be replaced (`__setitem__`) or dropped (`__delitem__`) in-place, which may invalidate some of the indexes. A drastic though probably reasonable solution in this case would be to simply drop all indexes bound to those replaced/dropped coordinates. For the case where a 1D basic coordinate that corresponds to a dimension is added/replaced, we could automatically generate a new index (see [Section 2.2.4](#224-implicit-indexes)).

We must also ensure that coordinates having a bound index are immutable, e.g., still wrap them into `IndexVariable` objects (even though the `IndexVariable` class might change substantially after this refactoring).

#### 2.4.2 New Dataset/DataArray with updated coordinates

Xarray provides a variety of Dataset/DataArray operations affecting the coordinates and where simply dropping the index(es) is not desirable. For example:

- Multi-coordinate indexes could be reduced to single coordinate indexes, like in `.reset_index()` or `.sel()` applied on a subset of the levels of a `pandas.MultiIndex` and that internally call `MultiIndex.droplevel` and `MultiIndex.get_loc_level`, respectively. There should be some API for wrapping this functionality in `XarrayIndex`.
- Indexes may be indexed themselves, like `pandas.Index` implements `__getitem__()`. When indexing their corresponding coordinate(s), e.g., via `.sel()` or `.isel()`, those indexes should be indexed too. This wouldn't be supported by all Xarray indexes, though. Some indexes that can't be indexed could still be automatically (re)built in the new Dataset/DataArray, like for example building a new `KDTree` index from the selection of a subset of an initial collection of data points. This is not always desirable, though, as indexes may be expensive to build. A more reasonable option would be to explicitly re-build the index, e.g., using `.set_index()`.
- Dataset/DataArray operations involving alignment (see [Section 2.6](#26-using-indexes-for-data-alignment))

### 2.5 Using indexes for data selection

One main use of indexes is label-based data selection using the DataArray/Dataset `.sel()` method. This refactoring would introduce a number of API changes that could go through some depreciation cycles:

- The keys of the mapping given to `indexers` (or the names of `indexer_kwargs`) would not correspond to only dimension names but could be the name of any coordinate that has an index
- For a `pandas.MultiIndex`, if no dimension-coordinate is created by default (see [Section 2.2.4](#224-implicit-coordinates)), providing dict-like objects as indexers should be depreciated
- There should be the possibility to provide additional options to the indexes that support specific selection features (e.g., Scikit-learn's `BallTree`'s `dualtree` query option to boost performance). The best API is not trivial here, since `.sel()` may accept indexers passed to several indexes (which should still be supported for convenience and compatibility), and indexes may have similar options with different semantics. We could introduce a new parameter like `index_options: Dict[XarrayIndex, Dict[str, Any]]` to pass options grouped by index.
- The `method` and `tolerance` parameters are specific to `pandas.Index` and would not be supported by all indexes. Probably best is to eventually pass those arguments as `index_options`.

With the new data model proposed here, once ambiguous situation may occur when indexers are given for several coordinates that share the same dimension but not the same index, e.g., from the example in [Section 1](#1-data-model):

```python
da.sel(x=..., y=..., lat=..., lon=...)
```

The easiest solution for this situation would be to raise an error. Alternatively, we could introduce a new parameter to specify how to combine the resulting integer indexers (i.e., union vs intersection), although this could already be achieved by chaining `.sel()` calls or combining `.sel()` with `.merge()` (it may or may not be straightforward).

### 2.6 Using indexes for data alignment

Another main use if indexes is data alignment in various operations. Some considerations regarding alignment and flexible indexes:

- support for alignment should probably be optional for an `XarrayIndex` subclass.
  - like `pandas.Index`, the index wrapper classes that support it should implement `.equals()`, `.union()` and/or `.intersection()`
  - support might be partial if that makes sense (outer, inner, left, right, exact...).
  - index equality might involve more than just the labels: for example a spatial index might be used to check if the coordinate system (CRS) is identical for two sets of coordinates
  - some indexes might implement inexact alignment, like in [#4489](https://github.com/pydata/xarray/pull/4489) or a `KDTree` index that selects nearest-neighbors within a given tolerance
  - alignment may be "multi-dimensional", i.e., the `KDTree` example above vs. dimensions aligned independently of each other
- we need to decide what to do when one dimension has more than one index that supports alignment
  - we should probably raise unless the user explicitly specify which index to use for the alignment
- we need to decide what to do when one dimension has one or more index(es) but none support alignment
  - either we raise or we fail back (silently) to alignment based on dimension size
- for inexact alignment, the tolerance threshold might be given when building the index and/or when performing the alignment
- are there cases where we want a specific index to perform alignment and another index to perform selection? It would be tricky to support that unless we allow multiple indexes per coordinate. Alternatively, underlying indexes could be picked internally in a "meta" index for one operation or another, although the risk is to eventually have to deal with an explosion of index wrapper classes with different meta indexes for each combination that we'd like to use.

### 2.7 Using indexes for other purposes

Xarray also provides a number of Dataset/DataArray methods where indexes are used in various ways, e.g.,

- `resample` (`CFTimeIndex` and a `DatetimeIntervalIndex`)
- `DatetimeAccessor` & `TimedeltaAccessor` properties (`CFTimeIndex` and a `DatetimeIntervalIndex`)
- `interp` & `interpolate_na`,
   - with `IntervalIndex`, these become regridding operations. Should we support hooks for these operations?
- `differentiate`, `integrate`, `polyfit`
   - raise an error if not a "simple" 1D index?
- `pad`
- `coarsen` has to make choices about output index labels.
- `sortby`
- `stack`/`unstack`
- plotting
    - `plot.pcolormesh` "infers" interval breaks along axes, which are really inferred `bounds` for the appropriate indexes.
    - `plot.step` again uses `bounds`. In fact, we may even want `step` to be the default 1D plotting function if the axis has `bounds` attached.

It would be reasonable to first restrict those methods to the indexes that are currently available in Xarray, and maybe extend the `XarrayIndex` API later upon request when the opportunity arises.

Conversely, nothing should prevent implementing "non-standard" API in 3rd-party `XarrayIndex` subclasses that could be used in DataArray/Dataset extensions (accessors). For example, we might want to reuse a `KDTree` index to compute k-nearest neighbors (returning a DataArray/Dataset with a new dimension) and/or the distances to the nearest neighbors (returning a DataArray/Dataset with a new data variable).

### 2.8 Index encoding

Indexes don't need to be directly serializable since we could (re)build them from their corresponding coordinate(s). However, we may take advantage that some indexes could be encoded/decoded to/from a set of arrays that would allow optimized reconstruction and/or storage, e.g.,

- `pandas.MultiIndex` -> `index.levels` and `index.codes`
- Scikit-learn's `KDTree` and `BallTree` that use an array-based representation of an immutable tree structure

## 3. Index representation in DataArray/Dataset's `repr`

Since indexes would become 1st class citizen of Xarray's data model, they deserve their own section in Dataset/DataArray `repr` that could look like:

```
<xarray.DataArray (x: 2, y: 2)>
array([[5.4, 7.8],
       [6.2, 4.7]])
Coordinates:
  * lon      (x, y) float64 10.2 15.2 12.6 17.6
  * lat      (x, y) float64 40.2 45.6 42.2 47.6
  * x        (x) float64 200.0 400.0
  * y        (y) float64 800.0 1e+03
Indexes:
  lat, lon     <SpatialIndex coords=(lat, lon) dims=(x, y)>
  x            <PandasIndexWrapper>
  y            <PandasIndexWrapper>
```

To keep the `repr` compact, we could:

- consolidate entries that map to the same index object, and have an short inline repr for `XarrayIndex` object
- collapse the index section by default in the HTML `repr`
- maybe omit all trivial indexes for 1D coordinates that match the dimension name

## 4. `IndexVariable`

TODO

## 5. Chunked coordinates and/or indexers

TODO

## 6. Coordinate duck arrays

TODO
