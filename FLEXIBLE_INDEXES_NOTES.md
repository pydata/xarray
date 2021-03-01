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

>>> da.sel(x=..., y=..., lon=..., lat=...)
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

Every `XarrayIndex` subclass must at least implement a method that takes label-based indexers as argument and that returns the corresponding position-based indexers. Whether or not a `XarrayIndex` subclass supports each of the features listed above should be either declared explicitly via a common API or left to the implementation.

There are potentially other methods that an `XarrayIndex` subclass must/should/may implement, like Xarray coordinate getters (see [Section 2.2.4](#224_Implicit_coodinates)) or `serialize()`/`deserialize()` methods. `XarrayIndex` API may be expanded in the future.

Xarray should provide a minimal set of built-in index wrappers (this could be reduced to the indexes currently supported in Xarray, i.e., `pandas.Index` and `pandas.MultiIndex`). Other index wrappers may be implemented in 3rd-party libraries (recommended). The `XarrayIndex` base class should be part of Xarray's public API.

#### 2.1.1 Index discoverability

For better discoverability of Xarray-compatible indexes, Xarray could provide some mechanism to register new index wrappers, e.g., something like [xoak's `IndexRegistry`](https://xoak.readthedocs.io/en/latest/_api_generated/xoak.IndexRegistry.html#xoak.IndexRegistry).

### 2.2 Explicit vs. implicit index creation

#### 2.2.1 Dataset/DataArray's `set_index` method

New indexes can be built from an existing set of coordinates or variables in a Dataset/DataArray using the `.set_index()` method.

TODO: describe API updates needed to provide the kind of index that we want to build from a set of coordinates.

#### 2.2.2 Dataset/DataArray's `indexes` constructor argument

TODO

#### 2.2.3 Implicit default indexes

In general explicit index creation should be preferred over implicit index creation. However, there is a majority of cases where basic `pandas.Index` objects could be built and used as indexes for 1-dimensional coordinates. For convenience, Xarray should automatically build such indexes for the coordinates where no index has been explicitly assigned in the Dataset/DataArray constructor or when indexes have been reset / dropped.

For which coordinates?

- A. only 1D coordinates with a name matching their dimension name
- B. all 1D coordinates

When to create it?

- A. each time when a new Dataset/DataArray is created
- B. only when we need it (i.e., when calling `.sel()` or `indexes`)

#### 2.2.4 Implicit coordinates

If we want to strictly follow the data model defined in [Section 1](#1_Data_Model), we need to ensure that at least one coordinate exists for each index (generally we need to ensure that the data indexed is properly exposed as one or more coordinates), especially when index objects are given directly to the Dataset/DataArray constructor ([Section 2.2.2](#222_DatasetDataArrays_indexes_constructor_argument)). This means that every `XarrayIndex` subclass must implement a method to get or generate one or more Xarray coordinates.

Additionally, we might want to generate a coordinate that is specific to the index, generally with a composite data type, like a coordinate with n-elements tuple labels for a n-levels `pandas.MultiIndex`. This should be optional, though. An `XarrayIndex` may or may not support this. This coordinate probably shouldn't be created implicitly or by default when setting the new index. Also, this might raise some issues, e.g., what would be the name of this coordinate? For a `pandas.MultiIndex` it would make sense in many (but not all) cases to choose the dimension name as the coordinate name.

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

### 2.6 Index serialization

TODO

## 3. Index representation in DataArray/Dataset's `repr`

TODO
