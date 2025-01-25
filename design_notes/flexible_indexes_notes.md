# Proposal: Xarray flexible indexes refactoring

Current status: https://github.com/pydata/xarray/projects/1

## 1. Data Model

Indexes are used in Xarray to extract data from Xarray objects using coordinate labels instead of using integer array indices. Although the indexes used in an Xarray object can be accessed (or built on-the-fly) via public methods like `to_index()` or properties like `indexes`, those are mainly used internally.

The goal of this project is to make those indexes 1st-class citizens of Xarray's data model. As such, indexes should clearly be separated from Xarray coordinates with the following relationships:

- Index -> Coordinate: one-to-many
- Coordinate -> Index: one-to-zero-or-one

An index may be built from one or more coordinates. However, each coordinate must relate to one index at most. Additionally, a coordinate may not be tied to any index.

The order in which multiple coordinates relate to an index may matter. For example, Scikit-Learn's [`BallTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree) index with the Haversine metric requires providing latitude and longitude values in that specific order. As another example, the order in which levels are defined in a `pandas.MultiIndex` may affect its lexsort depth (see [MultiIndex sorting](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#sorting-a-multiindex)).

Xarray's current data model has the same index-coordinate relationships than stated above, although this assumes that multi-index "virtual" coordinates are counted as coordinates (we can consider them as such, with some constraints). More importantly, This refactoring would turn the current one-to-one relationship between a dimension and an index into a many-to-many relationship, which would overcome some current limitations.

For example, we might want to select data along a dimension which has several coordinates:

```python
>>> da
<xarray.DataArray (river_profile: 100)>
array([...])
Coordinates:
  * drainage_area  (river_profile) float64 ...
  * chi            (river_profile) float64 ...
```

In this example, `chi` is a transformation of the `drainage_area` variable that is often used in geomorphology. We'd like to select data along the river profile using either `da.sel(drainage_area=...)` or `da.sel(chi=...)` but that's not currently possible. We could rename the `river_profile` dimension to one of the coordinates, then use `sel` with that coordinate, then call `swap_dims` if we want to use `sel` with the other coordinate, but that's not ideal. We could also build a `pandas.MultiIndex` from `drainage_area` and `chi`, but that's not optimal (there's no hierarchical relationship between these two coordinates).

Let's take another example:

```python
>>> da
<xarray.DataArray (x: 200, y: 100)>
array([[...], [...]])
Coordinates:
  * lon      (x, y) float64 ...
  * lat      (x, y) float64 ...
  * x        (x) float64 ...
  * y        (y) float64 ...
```

This refactoring would allow creating a geographic index for `lat` and `lon` and two simple indexes for `x` and `y` such that we could select data with either `da.sel(lon=..., lat=...)` or `da.sel(x=..., y=...)`.

Refactoring the dimension -> index one-to-one relationship into many-to-many would also introduce some issues that we'll need to address, e.g., ambiguous cases like `da.sel(chi=..., drainage_area=...)` where multiple indexes may potentially return inconsistent positional indexers along a dimension.

## 2. Proposed API changes

### 2.1 Index wrapper classes

Every index that is used to select data from Xarray objects should inherit from a base class, e.g., `XarrayIndex`, that provides some common API. `XarrayIndex` subclasses would generally consist of thin wrappers around existing index classes such as `pandas.Index`, `pandas.MultiIndex`, `scipy.spatial.KDTree`, etc.

There is a variety of features that an xarray index wrapper may or may not support:

- 1-dimensional vs. 2-dimensional vs. n-dimensional coordinate (e.g., `pandas.Index` only supports 1-dimensional coordinates while a geographic index could be built from n-dimensional coordinates)
- built from a single vs multiple coordinate(s) (e.g., `pandas.Index` is built from one coordinate, `pandas.MultiIndex` may be built from an arbitrary number of coordinates and a geographic index would typically require two latitude/longitude coordinates)
- in-memory vs. out-of-core (dask) index data/coordinates (vs. other array backends)
- range-based vs. point-wise selection
- exact vs. inexact lookups

Whether or not a `XarrayIndex` subclass supports each of the features listed above should be either declared explicitly via a common API or left to the implementation. An `XarrayIndex` subclass may encapsulate more than one underlying object used to perform the actual indexing. Such "meta" index would typically support a range of features among those mentioned above and would automatically select the optimal index object for a given indexing operation.

An `XarrayIndex` subclass must/should/may implement the following properties/methods:

- a `from_coords` class method that creates a new index wrapper instance from one or more Dataset/DataArray coordinates (+ some options)
- a `query` method that takes label-based indexers as argument (+ some options) and that returns the corresponding position-based indexers
- an `indexes` property to access the underlying index object(s) wrapped by the `XarrayIndex` subclass
- a `data` property to access index's data and map it to coordinate data (see [Section 4](#4-indexvariable))
- a `__getitem__()` implementation to propagate the index through DataArray/Dataset indexing operations
- `equals()`, `union()` and `intersection()` methods for data alignment (see [Section 2.6](#26-using-indexes-for-data-alignment))
- Xarray coordinate getters (see [Section 2.2.4](#224-implicit-coordinates))
- a method that may return a new index and that will be called when one of the corresponding coordinates is dropped from the Dataset/DataArray (multi-coordinate indexes)
- `encode()`/`decode()` methods that would allow storage-agnostic serialization and fast-path reconstruction of the underlying index object(s) (see [Section 2.8](#28-index-encoding))
- one or more "non-standard" methods or properties that could be leveraged in Xarray 3rd-party extensions like Dataset/DataArray accessors (see [Section 2.7](#27-using-indexes-for-other-purposes))

The `XarrayIndex` API has still to be defined in detail.

Xarray should provide a minimal set of built-in index wrappers (this could be reduced to the indexes currently supported in Xarray, i.e., `pandas.Index` and `pandas.MultiIndex`). Other index wrappers may be implemented in 3rd-party libraries (recommended). The `XarrayIndex` base class should be part of Xarray's public API.

#### 2.1.1 Index discoverability

For better discoverability of Xarray-compatible indexes, Xarray could provide some mechanism to register new index wrappers, e.g., something like [xoak's `IndexRegistry`](https://xoak.readthedocs.io/en/latest/_api_generated/xoak.IndexRegistry.html#xoak.IndexRegistry) or [numcodec's registry](https://numcodecs.readthedocs.io/en/stable/registry.html).

Additionally (or alternatively), new index wrappers may be registered via entry points as is already the case for storage backends and maybe other backends (plotting) in the future.

Registering new indexes either via a custom registry or via entry points should be optional. Xarray should also allow providing `XarrayIndex` subclasses in its API (Dataset/DataArray constructors, `set_index()`, etc.).

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
- `IndexSpec = Union[Type[XarrayIndex], Tuple[Type[XarrayIndex], Dict[str, Any]], XarrayIndex]`, so that index instances or index classes + build options could be also passed

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

The [current signature](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.set_index.html#xarray.DataArray.set_index) of `.set_index()` is tailored to `pandas.MultiIndex` and tied to the concept of a dimension-index. It is therefore hardly reusable as-is in the context of flexible indexes proposed here.

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

### 2.2.5 Immutable indexes

Some underlying indexes might be mutable (e.g., a tree-based index structure that allows dynamic addition of data points) while other indexes like `pandas.Index` aren't. To keep things simple, it is probably better to continue considering all indexes in Xarray as immutable (as well as their corresponding coordinates, see [Section 2.4.1](#241-mutable-coordinates)).

### 2.3 Index access

#### 2.3.1 Dataset/DataArray's `indexes` property

The `indexes` property would allow easy access to all the indexes used in a Dataset/DataArray. It would return a `Dict[CoordinateName, XarrayIndex]` for easy index lookup from coordinate name.

#### 2.3.2 Additional Dataset/DataArray properties or methods

In some cases the format returned by the `indexes` property would not be the best (e.g, it may return duplicate index instances as values). For convenience, we could add one more property / method to get the indexes in the desired format if needed.

### 2.4 Propagate indexes through operations

#### 2.4.1 Mutable coordinates

Dataset/DataArray coordinates may be replaced (`__setitem__`) or dropped (`__delitem__`) in-place, which may invalidate some of the indexes. A drastic though probably reasonable solution in this case would be to simply drop all indexes bound to those replaced/dropped coordinates. For the case where a 1D basic coordinate that corresponds to a dimension is added/replaced, we could automatically generate a new index (see [Section 2.2.4](#224-implicit-indexes)).

We must also ensure that coordinates having a bound index are immutable, e.g., still wrap them into `IndexVariable` objects (even though the `IndexVariable` class might change substantially after this refactoring).

#### 2.4.2 New Dataset/DataArray with updated coordinates

Xarray provides a variety of Dataset/DataArray operations affecting the coordinates and where simply dropping the index(es) is not desirable. For example:

- multi-coordinate indexes could be reduced to single coordinate indexes
  - like in `.reset_index()` or `.sel()` applied on a subset of the levels of a `pandas.MultiIndex` and that internally call `MultiIndex.droplevel` and `MultiIndex.get_loc_level`, respectively
- indexes may be indexed themselves
  - like `pandas.Index` implements `__getitem__()`
  - when indexing their corresponding coordinate(s), e.g., via `.sel()` or `.isel()`, those indexes should be indexed too
  - this might not be supported by all Xarray indexes, though
- some indexes that can't be indexed could still be automatically (re)built in the new Dataset/DataArray
  - like for example building a new `KDTree` index from the selection of a subset of an initial collection of data points
  - this is not always desirable, though, as indexes may be expensive to build
  - a more reasonable option would be to explicitly re-build the index, e.g., using `.set_index()`
- Dataset/DataArray operations involving alignment (see [Section 2.6](#26-using-indexes-for-data-alignment))

### 2.5 Using indexes for data selection

One main use of indexes is label-based data selection using the DataArray/Dataset `.sel()` method. This refactoring would introduce a number of API changes that could go through some depreciation cycles:

- the keys of the mapping given to `indexers` (or the names of `indexer_kwargs`) would not correspond to only dimension names but could be the name of any coordinate that has an index
- for a `pandas.MultiIndex`, if no dimension-coordinate is created by default (see [Section 2.2.4](#224-implicit-coordinates)), providing dict-like objects as indexers should be depreciated
- there should be the possibility to provide additional options to the indexes that support specific selection features (e.g., Scikit-learn's `BallTree`'s `dualtree` query option to boost performance).
  - the best API is not trivial here, since `.sel()` may accept indexers passed to several indexes (which should still be supported for convenience and compatibility), and indexes may have similar options with different semantics
  - we could introduce a new parameter like `index_options: Dict[XarrayIndex, Dict[str, Any]]` to pass options grouped by index
- the `method` and `tolerance` parameters are specific to `pandas.Index` and would not be supported by all indexes: probably best is to eventually pass those arguments as `index_options`
- the list valid indexer types might be extended in order to support new ways of indexing data, e.g., unordered selection of all points within a given range
  - alternatively, we could reuse existing indexer types with different semantics depending on the index, e.g., using `slice(min, max, None)` for unordered range selection

With the new data model proposed here, an ambiguous situation may occur when indexers are given for several coordinates that share the same dimension but not the same index, e.g., from the example in [Section 1](#1-data-model):

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
- are there cases where we want a specific index to perform alignment and another index to perform selection?
  - it would be tricky to support that unless we allow multiple indexes per coordinate
  - alternatively, underlying indexes could be picked internally in a "meta" index for one operation or another, although the risk is to eventually have to deal with an explosion of index wrapper classes with different meta indexes for each combination that we'd like to use.

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

Indexes don't need to be directly serializable since we could (re)build them from their corresponding coordinate(s). However, it would be useful if some indexes could be encoded/decoded to/from a set of arrays that would allow optimized reconstruction and/or storage, e.g.,

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

`IndexVariable` is currently used to wrap a `pandas.Index` as a variable, which would not be relevant after this refactoring since it is aimed at decoupling indexes and variables.

We'll probably need to move elsewhere some of the features implemented in `IndexVariable` to:

- ensure that all coordinates with an index are immutable (see [Section 2.4.1](#241-mutable-coordinates))
  - do not set values directly, do not (re)chunk (even though it may be already chunked), do not load, do not convert to sparse/dense, etc.
- directly reuse index's data when that's possible
  - in the case of a `pandas.Index`, it makes little sense to have duplicate data (e.g., as a NumPy array) for its corresponding coordinate
- convert a variable into a `pandas.Index` using `.to_index()` (for backwards compatibility).

Other `IndexVariable` API like `level_names` and `get_level_variable()` would not useful anymore: it is specific to how we currently deal with `pandas.MultiIndex` and virtual "level" coordinates in Xarray.

## 5. Chunked coordinates and/or indexers

We could take opportunity of this refactoring to better leverage chunked coordinates (and/or chunked indexers for data selection). There's two ways to enable it:

A. support for chunked coordinates is left to the index
B. support for chunked coordinates is index agnostic and is implemented in Xarray

As an example for B, [xoak](https://github.com/ESM-VFC/xoak) supports building an index for each chunk, which is coupled with a two-step data selection process (cross-index queries + brute force "reduction" look-up). There is an example [here](https://xoak.readthedocs.io/en/latest/examples/dask_support.html). This may be tedious to generalize this to other kinds of operations, though. Xoak's Dask support is rather experimental, not super stable (it's quite hard to control index replication and data transfer between Dask workers with the default settings), and depends on whether indexes are thread-safe and/or serializable.

Option A may be more reasonable for now.

## 6. Coordinate duck arrays

Another opportunity of this refactoring is support for duck arrays as index coordinates. Decoupling coordinates and indexes would _de-facto_ enable it.

However, support for duck arrays in index-based operations such as data selection or alignment would probably require some protocol extension, e.g.,

```python
class MyDuckArray:
    ...

    def _sel_(self, indexer):
        """Prepare the label-based indexer to conform to this coordinate array."""
        ...
        return new_indexer

    ...
```

For example, a `pint` array would implement `_sel_` to perform indexer unit conversion or raise, warn, or just pass the indexer through if it has no units.
