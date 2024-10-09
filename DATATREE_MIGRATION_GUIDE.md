# Migration guide for users of `xarray-contrib/datatree`

This guide is for previous users of the prototype `datatree.DataTree` in the `xarray-contrib/datatree repository`. That repository has now been archived, and will not be maintained. This guide is intended to help smooth your transition to using the new, updated `xarray.DataTree`.

.. important
   
   There are breaking changes! You should not expect that code written with `xarray-contrib/datatree` will work without modifications.
   At the absolute minimum you will need to change the top-level import statement, but there are other changes too.

We have made various changes compared to the prototype version. These can be split into two main types: minor API changes, which mostly consist of renaming methods to be more self-consistent, and some deeper data model changes, which affect the hierarchal structure itself.

### Data model changes

Internal alignment

Coordinate inheritance

Reflected in repr

Can no longer represent totally arbitrary datasets in each node - some on-disk structures that `xr.open_datatree` will now refuse to load.
For these cases we made `open_groups`.

Generally if you don't like this you can get more similar behaviour to the old package by removing indexes from coordinates.

### Integrated backends

`open_datatree(group=...)`?

Performance improvements

Can now extend other xarray backends to support `open_datatree`!

### Other API changes

`from datatree import DataTree, open_datatree` -> `from xarray import DataTree, open_datatree`

`.ds` -> `.dataset`

`DataTree(ds=...)` to `DataTree(dataset=)`

`.to_dataset()` still exists but now has options (`inherited=...`)

`parent` kwarg removed from `DataTree.__init__`

`.parent` property is now read-only

`children` in `DataTree.__init__` are now shallow-copied

`map_over_subtree` -> ?

Arithmetic between `DataTree` and `Dataset`/scalars now raises

`.as_array` -> `.to_dataarray`

Disabled some methods which were not well tested. In general we have tried to only keep things that are known to work, with the plan to increase API surface incrementally after release.

## Thank you!

Thank you for trying out `xarray-contrib/datatree`!

We welcome contributions of any kind, including things that never quite made it into the original datatree repository. Please also let us know if we have forgotten to mention a change that should have been listed in this guide.

Sincerely, the datatree team

(Tom Nicholas, Owen Littlejohns, Matt Savoie, Eni Awowale, Alfonso Ladino, Justus Magin, Stephan Hoyer)