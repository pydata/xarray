# Migration guide for users of `xarray-contrib/datatree`

_15th October 2024_

This guide is for previous users of the prototype `datatree.DataTree` class in the `xarray-contrib/datatree repository`. That repository has now been archived, and will not be maintained. This guide is intended to help smooth your transition to using the new, updated `xarray.DataTree` class.

> [!IMPORTANT]
> There are breaking changes! You should not expect that code written with `xarray-contrib/datatree` will work without any modifications. At the absolute minimum you will need to change the top-level import statement, but there are other changes too.

We have made various changes compared to the prototype version. These can be split into three categories: data model changes, which affect the hierarchal structure itself; integration with xarray's IO backends; and minor API changes, which mostly consist of renaming methods to be more self-consistent.

### Data model changes

The most important changes made are to the data model of `DataTree`. Whilst previously data in different nodes was unrelated and therefore unconstrained, now trees have "internal alignment" - meaning that dimensions and indexes in child nodes must exactly align with those in their parents.

These alignment checks happen at tree construction time, meaning there are some netCDF4 files and zarr stores that could previously be opened as `datatree.DataTree` objects using `datatree.open_datatree`, but now cannot be opened as `xr.DataTree` objects using `xr.open_datatree`. For these cases we added a new opener function `xr.open_groups`, which returns a `dict[str, Dataset]`. This is intended as a fallback for tricky cases, where the idea is that you can still open the entire contents of the file using `open_groups`, edit the `Dataset` objects, then construct a valid tree from the edited dictionary using `DataTree.from_dict`.

The alignment checks allowed us to add "Coordinate Inheritance", a much-requested feature where indexed coordinate variables are now "inherited" down to child nodes. This allows you to define common coordinates in a parent group that are then automatically available on every child node. The distinction between a locally-defined coordinate variables and an inherited coordinate that was defined on a parent node is reflected in the `DataTree.__repr__`. Generally if you prefer not to have these variables be inherited you can get more similar behaviour to the old `datatree` package by removing indexes from coordinates, as this prevents inheritance.

Tree structure checks between multiple trees (i.e., `DataTree.isomorophic`) and pairing of nodes in arithmetic has also changed. Nodes are now matched (with `xarray.group_subtrees`) based on their relative paths, without regard to the order in which child nodes are defined.

For further documentation see the page in the user guide on Hierarchical Data.

### Integrated backends

Previously `datatree.open_datatree` used a different codepath from `xarray.open_dataset`, and was hard-coded to only support opening netCDF files and Zarr stores.
Now xarray's backend entrypoint system has been generalized to include `open_datatree` and the new `open_groups`.
This means we can now extend other xarray backends to support `open_datatree`! If you are the maintainer of an xarray backend we encourage you to add support for `open_datatree` and `open_groups`!

Additionally:

- A `group` kwarg has been added to `open_datatree` for choosing which group in the file should become the root group of the created tree.
- Various performance improvements have been made, which should help when opening netCDF files and Zarr stores with large numbers of groups.
- We anticipate further performance improvements being possible for datatree IO.

### API changes

A number of other API changes have been made, which should only require minor modifications to your code:

- The top-level import has changed, from `from datatree import DataTree, open_datatree` to `from xarray import DataTree, open_datatree`. Alternatively you can now just use the `import xarray as xr` namespace convention for everything datatree-related.
- The `DataTree.ds` property has been changed to `DataTree.dataset`, though `DataTree.ds` remains as an alias for `DataTree.dataset`.
- Similarly the `ds` kwarg in the `DataTree.__init__` constructor has been replaced by `dataset`, i.e. use `DataTree(dataset=)` instead of `DataTree(ds=...)`.
- The method `DataTree.to_dataset()` still exists but now has different options for controlling which variables are present on the resulting `Dataset`, e.g. `inherit=True/False`.
- `DataTree.copy()` also has a new `inherit` keyword argument for controlling whether or not coordinates defined on parents are copied (only relevant when copying a non-root node).
- The `DataTree.parent` property is now read-only. To assign a ancestral relationships directly you must instead use the `.children` property on the parent node, which remains settable.
- Similarly the `parent` kwarg has been removed from the `DataTree.__init__` constructor.
- DataTree objects passed to the `children` kwarg in `DataTree.__init__` are now shallow-copied.
- `DataTree.map_over_subtree` has been renamed to `DataTree.map_over_datasets`, and changed to no longer work like a decorator. Instead you use it to apply the function and arguments directly, more like how `xarray.apply_ufunc` works.
- `DataTree.as_array` has been replaced by `DataTree.to_dataarray`.
- A number of methods which were not well tested have been (temporarily) disabled. In general we have tried to only keep things that are known to work, with the plan to increase API surface incrementally after release.

## Thank you!

Thank you for trying out `xarray-contrib/datatree`!

We welcome contributions of any kind, including good ideas that never quite made it into the original datatree repository. Please also let us know if we have forgotten to mention a change that should have been listed in this guide.

Sincerely, the datatree team:

Tom Nicholas,
Owen Littlejohns,
Matt Savoie,
Eni Awowale,
Alfonso Ladino,
Justus Magin,
Stephan Hoyer
