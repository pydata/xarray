# Migration guide for users of `xarray-contrib/datatree`

_15th October 2024_

This guide is for previous users of the prototype `datatree.DataTree` class in the `xarray-contrib/datatree repository`. That repository has now been archived, and will not be maintained. This guide is intended to help smooth your transition to using the new, updated `xarray.DataTree` class.

.. important

   There are breaking changes! You should not expect that code written with `xarray-contrib/datatree` will work without any modifications.
   At the absolute minimum you will need to change the top-level import statement, but there are other changes too.

We have made various changes compared to the prototype version. These can be split into three categories: minor API changes, which mostly consist of renaming methods to be more self-consistent; and some deeper data model changes, which affect the hierarchal structure itself; and integration with xarray's IO backends.

### Data model changes

Internal alignment

Coordinate inheritance

Reflected in repr

Can no longer represent totally arbitrary datasets in each node - some on-disk structures that `xr.open_datatree` will now refuse to load.
For these cases we made `open_groups`.

Generally if you don't like this you can get more similar behaviour to the old package by removing indexes from coordinates.

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
- The method `DataTree.to_dataset()` still exists but now has different options for controlling which variables are present on the resulting `Dataset`, e.g. `inherited=True/False`.
- The `DataTree.parent` property is now read-only. To assign a node as the parent you should instead use the `.children` property on the other node, which remains settable.
- Similarly the `parent` kwarg has been removed from the `DataTree.__init__` constuctor. 
- DataTree objects passed to the `children` kwarg in `DataTree.__init__` are now shallow-copied.
- `DataTree.as_array` has been replaced by `DataTree.to_dataarray`.
- `map_over_subtree` -> ?
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
