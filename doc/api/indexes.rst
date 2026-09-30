.. currentmodule:: xarray

Indexes
=======


.. seealso::
    See the Xarray gallery on `custom indexes <https://xarray-indexes.readthedocs.io/>`_ for more examples.


Creating indexes
----------------
.. autosummary::
   :toctree: ../generated/

   cftime_range
   date_range
   date_range_like
   indexes.RangeIndex.arange
   indexes.RangeIndex.linspace


Built-in Indexes
----------------

Default, pandas-backed indexes built-in to Xarray:

.. autosummary::
   :toctree: ../generated/

   indexes.PandasIndex
   indexes.PandasMultiIndex


More complex indexes built-in to Xarray:

.. autosummary::
   :toctree: ../generated/

   CFTimeIndex
   indexes.RangeIndex
   indexes.NDPointIndex
   indexes.CoordinateTransformIndex


Building custom indexes
-----------------------

These classes are building blocks for more complex Indexes:

.. autosummary::
   :toctree: ../generated/

   indexes.CoordinateTransform
   indexes.CoordinateTransformIndex
   indexes.NDPointIndex
   indexes.TreeAdapter

The Index base class for building custom indexes:

.. autosummary::
   :toctree: ../generated/

   Index
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


The following are useful when building custom Indexes

.. autosummary::
   :toctree: ../generated/

   IndexSelResult
