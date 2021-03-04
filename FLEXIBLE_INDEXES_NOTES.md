# Proposal: Xarray flexible indexes refactoring

Current status: https://github.com/pydata/xarray/projects/1

TODO: check if the project is up-to-date and update it if necessary.

## 1. Data Model

Indexes are used in Xarray to extract data from Xarray objects using coordinate labels instead of using integer array indices. Although the indexes used in an Xarray object can be accessed (or built on-the-fly) via public methods like `to_index()` or properties like `indexes`, those are mainly used internally.

The goal of this project is to make those indexes 1st-class citizens of Xarray's data model. As such, indexes should clearly be separated from Xarray coordinates with the following relationships:

- Index -> Coordinate: one-to-many
- Coordinate -> Index: one-to-zero-or-one

An index may be built from one or more coordinates. However, each coordinate must relate to one index at most. Additionally, a coordinate may not be tied to any index.

The order in which multiple coordinates relate to an index should matter. For example, Scikit-Learn's `BallTree` index with the Haversine metric requires providing latitude and longitude values in that specific order. As another example, the order in which levels are defined in a `pandas.MultiIndex` may affect its lexsort depth.

Xarray's current data model is already based on the same index-coordinate relationships. The current data model also implies a one-to-one relationship between a dimension and an index. This one-to-one relationship works as currently it is not possible to perform label-based data selection in Xarray using multi-dimensional coordinates, but since we want to enable this feature in this proposal, we need the dimension-index relationship to evolve towards many-to-many.

In the example below, we'd like to select data points based on their x, y values and/or on their latitude/longitude positions:

```
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
- orthogonal vs. vectorized indexing, range-based vs. point-wise selection
- exact vs. inexact lookups

Whether or not a `XarrayIndex` subclass supports each of the features listed above should be either declared explicitly via a common API or left to the implementation. An `XarrayIndex` subclass may encapsulate more than one underlying object used to perform the actual indexing. Such "meta" index would typically support a range of features among those mentioned above and would automatically select the optimal index object for a given indexing operation.

Every `XarrayIndex` subclass must at least implement two methods:

- One `build` method that takes one or more Dataset/DataArray coordinates and that returns the object(s) that will be used for the actual indexing (e.g., `pandas.Index`, `scipy.spatial.KDTree`, etc.)
- One `query` method that takes label-based indexers as argument and that returns the corresponding position-based indexers.

These two methods may accept additional keyword arguments passed to the underlying index object constructor or query methods.

There are potentially other properties / methods that an `XarrayIndex` subclass must/should/may implement, e.g.,

- An `indexes` property to access the underlying index object(s) wrapped by the `XarrayIndex` subclass.
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

#### 2.2.1 Dataset/DataArray's `set_index` method

New indexes can be built from an existing set of coordinates or variables in a Dataset/DataArray using the `.set_index()` method.



TODO: describe API updates needed to provide the kind of index that we want to build from a set of coordinates.

#### 2.2.2 Dataset/DataArray's `indexes` constructor argument

The new `indexes` argument of Dataset/DataArray constructors may be used to specify which kind of index to bind to which coordinate(s). It would consist of a mapping where, for each item, the key is one coordinate name (or a sequence of coordinate names) that must be given in `coords` and the value is the type of the index to build from this (these) coordinate(s) (it could also be an `XarrayIndex` instance, so the index does not need to be built).

Currently index objects like `pandas.MultiIndex` can be passed directly to `coords`, which in this specific case results in the implicit creation of virtual coordinates. With the new `indexes` argument this behavior may become even more confusing than it currently is. For the sake of clarity, it would be appropriate to drop support for this specific behavior and treat any given mapping value given in `coords` as an array that can be wrapped into an Xarray variable, i.e., in the case of a multi-index:

```python
>>> xr.DataArray([1.0, 2.0], dims='x', coords={'x': midx})
<xarray.DataArray (x: 2)>
array([1., 2.])
Coordinates:
    x        (x) object ('a', 0) ('b', 1)
```

A possible solution to reuse a `pandas.MultiIndex` in a DataArray/Dataset with levels exposed as coordinates is proposed in [Section 2.2.4](#224-implicit-coordinates).

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

Besides `pandas.MultiIndex`, there may be other situations where we would like to reuse an existing index in a new Dataset/DataArray (e.g., when the index is very expensive to build), and which would require the implicit creation of coordinates.

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

The `indexes` property would return a mapping, where keys are coordinate name(s) (i.e., either a string or a tuple) and values are index objects (i.e., instances of `XarrayIndex`). If a tuple is used as key, the order of its elements should be consistent with the order in which each coordinate participate to the index (see [Section 1](#1_Data_Model)).

#### 2.3.2 Additional Dataset/DataArray's convenient methods

Using tuples as keys in the `indexes` property has the advantage of providing clear information about the order that was used to build multi-coordinate indexes. For convenience, we could have an additional property `coord_indexes` or an additional method `get_index(coord_name: str)` to easily retrieve the index object that is tied to a specific coordinate.

### 2.4 Propagate indexes through operations

TODO

- Any operation affecting one coordinate of a multi-coordinate index -> simply drop the index or maybe allow `XarrayIndex` subclasses to optionally generate and return a new or updated index

### 2.5 Using indexes for data selection

TODO: `.sel()` would now accept coordinate names as indexer keys.

Indexes that are wrapped as `XarrayIndex` subclasses may provide more selection capabilities than what is currently possible using Dataset/DataArray `sel()`, e.g., radius or region selection, k-nearest neighbors, etc. There's no plan here to expand Xarray's selection API. This could be extended in 3rd-party libraries, e.g., using Dataset/DataArray accessors that would reuse the indexes attached to a Dataset/DataArray object.

### 2.6 Using indexes for data alignment

TODO

### 2.7 Using indexes for other purposes

TODO

### 2.8 Index encoding

TODO

## 3. Index representation in DataArray/Dataset's `repr`

TODO
