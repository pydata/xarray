.. currentmodule:: xarray

Dataset
=======

.. autosummary::
   :toctree: ../generated/

   Dataset

Attributes
----------

.. autosummary::
   :toctree: ../generated/

   Dataset.dims
   Dataset.sizes
   Dataset.dtypes
   Dataset.data_vars
   Dataset.coords
   Dataset.attrs
   Dataset.encoding
   Dataset.indexes
   Dataset.xindexes
   Dataset.chunks
   Dataset.chunksizes
   Dataset.nbytes

Dictionary interface
--------------------

Datasets implement the mapping interface with keys given by variable names
and values given by ``DataArray`` objects.

.. autosummary::
   :toctree: ../generated/

   Dataset.__getitem__
   Dataset.__setitem__
   Dataset.__delitem__
   Dataset.update
   Dataset.get
   Dataset.items
   Dataset.keys
   Dataset.values

Dataset contents
----------------

.. autosummary::
   :toctree: ../generated/

   Dataset.copy
   Dataset.assign
   Dataset.assign_coords
   Dataset.assign_attrs
   Dataset.pipe
   Dataset.merge
   Dataset.rename
   Dataset.rename_vars
   Dataset.rename_dims
   Dataset.swap_dims
   Dataset.expand_dims
   Dataset.drop_vars
   Dataset.drop_indexes
   Dataset.drop_duplicates
   Dataset.drop_dims
   Dataset.drop_encoding
   Dataset.drop_attrs
   Dataset.set_coords
   Dataset.reset_coords
   Dataset.convert_calendar
   Dataset.interp_calendar
   Dataset.get_index

Comparisons
-----------

.. autosummary::
   :toctree: ../generated/

   Dataset.equals
   Dataset.identical
   Dataset.broadcast_equals

Indexing
--------

.. autosummary::
   :toctree: ../generated/

   Dataset.loc
   Dataset.isel
   Dataset.sel
   Dataset.drop_sel
   Dataset.drop_isel
   Dataset.head
   Dataset.tail
   Dataset.thin
   Dataset.squeeze
   Dataset.interp
   Dataset.interp_like
   Dataset.reindex
   Dataset.reindex_like
   Dataset.set_index
   Dataset.reset_index
   Dataset.set_xindex
   Dataset.reorder_levels
   Dataset.query

Missing value handling
----------------------

.. autosummary::
   :toctree: ../generated/

   Dataset.isnull
   Dataset.notnull
   Dataset.combine_first
   Dataset.count
   Dataset.dropna
   Dataset.fillna
   Dataset.ffill
   Dataset.bfill
   Dataset.interpolate_na
   Dataset.where
   Dataset.isin

Computation
-----------

.. autosummary::
   :toctree: ../generated/

   Dataset.map
   Dataset.reduce
   Dataset.groupby
   Dataset.groupby_bins
   Dataset.rolling
   Dataset.rolling_exp
   Dataset.cumulative
   Dataset.weighted
   Dataset.coarsen
   Dataset.resample
   Dataset.diff
   Dataset.quantile
   Dataset.differentiate
   Dataset.integrate
   Dataset.map_blocks
   Dataset.polyfit
   Dataset.curvefit
   Dataset.eval

Aggregation
-----------

.. autosummary::
   :toctree: ../generated/

   Dataset.all
   Dataset.any
   Dataset.argmax
   Dataset.argmin
   Dataset.count
   Dataset.idxmax
   Dataset.idxmin
   Dataset.max
   Dataset.min
   Dataset.mean
   Dataset.median
   Dataset.prod
   Dataset.sum
   Dataset.std
   Dataset.var
   Dataset.cumsum
   Dataset.cumprod

ndarray methods
---------------

.. autosummary::
   :toctree: ../generated/

   Dataset.argsort
   Dataset.astype
   Dataset.clip
   Dataset.conj
   Dataset.conjugate
   Dataset.imag
   Dataset.round
   Dataset.real
   Dataset.rank

Reshaping and reorganizing
--------------------------

.. autosummary::
   :toctree: ../generated/

   Dataset.transpose
   Dataset.stack
   Dataset.unstack
   Dataset.to_stacked_array
   Dataset.shift
   Dataset.roll
   Dataset.pad
   Dataset.sortby
   Dataset.broadcast_like
