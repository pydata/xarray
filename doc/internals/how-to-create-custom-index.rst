.. currentmodule:: xarray

.. _internals.custom indexes:

How to create a custom index
============================

.. warning::

   This feature is highly experimental. Support for custom indexes has been
   introduced in v2022.06.0 and is still incomplete. API is subject to change
   without deprecation notice. However we encourage you to experiment and report issues that arise.

Xarray's built-in support for label-based indexing (e.g. ``ds.sel(latitude=40, method="nearest")``) and alignment operations
relies on :py:class:`pandas.Index` objects. Pandas Indexes are powerful and suitable for many
applications but also have some limitations:

- it only works with 1-dimensional coordinates where explicit labels
  are fully loaded in memory
- it is hard to reuse it with irregular data for which there exist more
  efficient, tree-based structures to perform data selection
- it doesn't support extra metadata that may be required for indexing and
  alignment (e.g., a coordinate reference system)

Fortunately, Xarray now allows extending this functionality with custom indexes,
which can be implemented in 3rd-party libraries.

The Index base class
--------------------

Every Xarray index must inherit from the :py:class:`Index` base class. It is for
example the case of Xarray built-in ``PandasIndex`` and ``PandasMultiIndex``
subclasses, which wrap :py:class:`pandas.Index` and
:py:class:`pandas.MultiIndex` respectively.

The ``Index`` API closely follows the :py:class:`Dataset` and
:py:class:`DataArray` API, e.g., for an index to support :py:meth:`DataArray.sel` it needs to
implement :py:meth:`Index.sel`, to support :py:meth:`DataArray.stack` and :py:meth:`DataArray.unstack` it
needs to implement :py:meth:`Index.stack` and :py:meth:`Index.unstack`, etc.

Some guidelines and examples are given below. More details can be found in the
documented :py:class:`Index` API.

Minimal requirements
--------------------

Every index must at least implement the :py:meth:`Index.from_variables` class
method, which is used by Xarray to build a new index instance from one or more
existing coordinates in a Dataset or DataArray.

Since any collection of coordinates can be passed to that method (i.e., the
number, order and dimensions of the coordinates are all arbitrary), it is the
responsibility of the index to check the consistency and validity of those input
coordinates.

For example, :py:class:`~xarray.core.indexes.PandasIndex` accepts only one coordinate and
:py:class:`~xarray.core.indexes.PandasMultiIndex` accepts one or more 1-dimensional coordinates that must all
share the same dimension. Other, custom indexes need not have the same
constraints, e.g.,

- a georeferenced raster index which only accepts two 1-d coordinates with
  distinct dimensions
- a staggered grid index which takes coordinates with different dimension name
  suffixes (e.g., "_c" and "_l" for center and left)

Optional requirements
---------------------

Pretty much everything else is optional. Depending on the method, in the absence
of a (re)implementation, an index will either raise a ``NotImplementedError``
or won't do anything specific (just drop, pass or copy itself
from/to the resulting Dataset or DataArray).

For example, you can just skip re-implementing :py:meth:`Index.rename` if there
is no internal attribute or object to rename according to the new desired
coordinate or dimension names. In the case of ``PandasIndex``, we rename the
underlying ``pandas.Index`` object and/or update the ``PandasIndex.dim``
attribute since the associated dimension name has been changed.

Wrap index data as coordinate data
----------------------------------

In some cases it is possible to reuse the index's underlying object or structure
as coordinate data and hence avoid data duplication.

For ``PandasIndex`` and ``PandasMultiIndex``, we
leverage the fact that ``pandas.Index`` objects expose some array-like API. In
Xarray we use some wrappers around those underlying objects as a thin
compatibility layer to preserve dtypes, handle explicit and n-dimensional
indexing, etc.

Other structures like tree-based indexes (e.g., kd-tree) may differ too much
from arrays to reuse it as coordinate data.

If the index data can be reused as coordinate data, the ``Index`` subclass
should implement :py:meth:`Index.create_variables`. This method accepts a
dictionary of variable names as keys and :py:class:`Variable` objects as values (used for propagating
variable metadata) and should return a dictionary of new :py:class:`Variable` or
:py:class:`IndexVariable` objects.

Data selection
--------------

For an index to support label-based selection, it needs to at least implement
:py:meth:`Index.sel`. This method accepts a dictionary of labels where the keys
are coordinate names (already filtered for the current index) and the values can
be pretty much anything (e.g., a slice, a tuple, a list, a numpy array, a
:py:class:`Variable` or a :py:class:`DataArray`). It is the responsibility of
the index to properly handle those input labels.

:py:meth:`Index.sel` must return an instance of :py:class:`IndexSelResult`. The
latter is a small data class that holds positional indexers (indices) and that
may also hold new variables, new indexes, names of variables or indexes to drop,
names of dimensions to rename, etc. For example, this is useful in the case of
``PandasMultiIndex`` as it allows Xarray to convert it into a single ``PandasIndex``
when only one level remains after the selection.

The :py:class:`IndexSelResult` class is also used to merge results from label-based
selection performed by different indexes. Note that it is now possible to have
two distinct indexes for two 1-d coordinates sharing the same dimension, but it
is not currently possible to use those two indexes in the same call to
:py:meth:`Dataset.sel`.

Optionally, the index may also implement :py:meth:`Index.isel`. In the case of
``PandasIndex`` we use it to create a new index object by just indexing the
underlying ``pandas.Index`` object. In other cases this may not be possible,
e.g., a kd-tree object may not be easily indexed. If ``Index.isel()`` is not
implemented, the index in just dropped in the DataArray or Dataset resulting
from the selection.

Alignment
---------

For an index to support alignment, it needs to implement:

- :py:meth:`Index.equals`, which compares the index with another index and
  returns either ``True`` or ``False``
- :py:meth:`Index.join`, which combines the index with another index and returns
  a new Index object
- :py:meth:`Index.reindex_like`, which queries the index with another index and
  returns positional indexers that are used to re-index Dataset or DataArray
  variables along one or more dimensions

Xarray ensures that those three methods are called with an index of the same
type as argument.

Meta-indexes
------------

Nothing prevents writing a custom Xarray index that itself encapsulates other
Xarray index(es). We call such index a "meta-index".

Here is a small example of a meta-index for geospatial, raster datasets (i.e.,
regularly spaced 2-dimensional data) that internally relies on two
``PandasIndex`` instances for the x and y dimensions respectively:

.. code-block:: python

    from xarray import Index
    from xarray.core.indexes import PandasIndex
    from xarray.core.indexing import merge_sel_results


    class RasterIndex(Index):
        def __init__(self, xy_indexes):
            assert len(xy_indexes) == 2

            # must have two distinct dimensions
            dim = [idx.dim for idx in xy_indexes.values()]
            assert dim[0] != dim[1]

            self._xy_indexes = xy_indexes

        @classmethod
        def from_variables(cls, variables):
            assert len(variables) == 2

            xy_indexes = {
                k: PandasIndex.from_variables({k: v}) for k, v in variables.items()
            }

            return cls(xy_indexes)

        def create_variables(self, variables):
            idx_variables = {}

            for index in self._xy_indexes.values():
                idx_variables.update(index.create_variables(variables))

            return idx_variables

        def sel(self, labels):
            results = []

            for k, index in self._xy_indexes.items():
                if k in labels:
                    results.append(index.sel({k: labels[k]}))

            return merge_sel_results(results)


This basic index only supports label-based selection. Providing a full-featured
index by implementing the other ``Index`` methods should be pretty
straightforward for this example, though.

This example is also not very useful unless we add some extra functionality on
top of the two encapsulated ``PandasIndex`` objects, such as a coordinate
reference system.

How to use a custom index
-------------------------

You can use :py:meth:`Dataset.set_xindex` or :py:meth:`DataArray.set_xindex` to assign a
custom index to a Dataset or DataArray, e.g., using the ``RasterIndex`` above:

.. code-block:: python

    import numpy as np
    import xarray as xr

    da = xr.DataArray(
        np.random.uniform(size=(100, 50)),
        coords={"x": ("x", np.arange(50)), "y": ("y", np.arange(100))},
        dims=("y", "x"),
    )

    # Xarray create default indexes for the 'x' and 'y' coordinates
    # we first need to explicitly drop it
    da = da.drop_indexes(["x", "y"])

    # Build a RasterIndex from the 'x' and 'y' coordinates
    da_raster = da.set_xindex(["x", "y"], RasterIndex)

    # RasterIndex now takes care of label-based selection
    selected = da_raster.sel(x=10, y=slice(20, 50))
