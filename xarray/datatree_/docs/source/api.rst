.. currentmodule:: datatree

#############
API reference
#############

DataTree
========

.. autosummary::
   :toctree: generated/

   DataTree
   DataNode

Attributes
----------

.. autosummary::
   :toctree: generated/

   DataTree.dims
   DataTree.variables
   DataTree.encoding
   DataTree.sizes
   DataTree.attrs
   DataTree.nbytes
   DataTree.indexes
   DataTree.xindexes
   DataTree.coords
   DataTree.chunks
   DataTree.real
   DataTree.imag
   DataTree.ds
   DataTree.has_data
   DataTree.groups

Dictionary interface
--------------------

.. autosummary::
   :toctree: generated/

   DataTree.__getitem__
   DataTree.__setitem__
   DataTree.update

Methods
-------

.. autosummary::
   :toctree: generated/

   DataTree.load
   DataTree.compute
   DataTree.persist
   DataTree.unify_chunks
   DataTree.chunk
   DataTree.map_blocks
   DataTree.copy
   DataTree.as_numpy
   DataTree.__copy__
   DataTree.__deepcopy__
   DataTree.set_coords
   DataTree.reset_coords
   DataTree.info
   DataTree.isel
   DataTree.sel
   DataTree.head
   DataTree.tail
   DataTree.thin
   DataTree.broadcast_like
   DataTree.reindex_like
   DataTree.reindex
   DataTree.interp
   DataTree.interp_like
   DataTree.rename
   DataTree.rename_dims
   DataTree.rename_vars
   DataTree.swap_dims
   DataTree.expand_dims
   DataTree.set_index
   DataTree.reset_index
   DataTree.reorder_levels
   DataTree.stack
   DataTree.unstack
   DataTree.update
   DataTree.merge
   DataTree.drop_vars
   DataTree.drop_sel
   DataTree.drop_isel
   DataTree.drop_dims
   DataTree.transpose
   DataTree.dropna
   DataTree.fillna
   DataTree.interpolate_na
   DataTree.ffill
   DataTree.bfill
   DataTree.combine_first
   DataTree.reduce
   DataTree.map
   DataTree.assign
   DataTree.diff
   DataTree.shift
   DataTree.roll
   DataTree.sortby
   DataTree.quantile
   DataTree.rank
   DataTree.differentiate
   DataTree.integrate
   DataTree.cumulative_integrate
   DataTree.filter_by_attrs
   DataTree.polyfit
   DataTree.pad
   DataTree.idxmin
   DataTree.idxmax
   DataTree.argmin
   DataTree.argmax
   DataTree.query
   DataTree.curvefit
   DataTree.squeeze
   DataTree.clip
   DataTree.assign_coords
   DataTree.where
   DataTree.close
   DataTree.isnull
   DataTree.notnull
   DataTree.isin
   DataTree.astype

Utilities
=========

.. autosummary::
   :toctree: generated/

   map_over_subtree

I/O
===

.. autosummary::
   :toctree: generated/

   open_datatree
   DataTree.to_netcdf
   DataTree.to_zarr

..
   Missing
   DataTree.__delitem__
   DataTree.get
   DataTree.items
   DataTree.keys
   DataTree.values
