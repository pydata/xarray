.. currentmodule:: xarray

Custom Indexes
==============
.. currentmodule:: xarray

Building custom indexes
-----------------------

These classes are building blocks for more complex Indexes:

.. autosummary::
   :toctree: generated/

   indexes.CoordinateTransform
   indexes.CoordinateTransformIndex
   indexes.NDPointIndex
   indexes.TreeAdapter

The Index base class for building custom indexes:

.. autosummary::
   :toctree: generated/

   Index.from_variables
   Index.concat
   Index.stack
   Index.unstack
   Index.create_variables
   Index.should_add_coord_to_array
   Index.to_pandas_index
   Index.isel
   Index.sel
   Index.join
   Index.reindex_like
   Index.equals
   Index.roll
   Index.rename
   Index.copy
