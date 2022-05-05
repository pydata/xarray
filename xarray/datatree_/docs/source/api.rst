.. currentmodule:: datatree

#############
API reference
#############

DataTree
========

Creating a DataTree
-------------------

.. autosummary::
   :toctree: generated/

   DataTree

Tree Attributes
---------------

.. autosummary::
   :toctree: generated/

   DataTree.parent
   DataTree.children
   DataTree.name
   DataTree.path
   DataTree.root
   DataTree.is_root
   DataTree.is_leaf
   DataTree.subtree
   DataTree.siblings
   DataTree.lineage
   DataTree.ancestors
   DataTree.groups

Data Attributes
---------------

.. autosummary::
   :toctree: generated/

   DataTree.dims
   DataTree.variables
   DataTree.encoding
   DataTree.sizes
   DataTree.attrs
   DataTree.indexes
   DataTree.xindexes
   DataTree.coords
   DataTree.chunks
   DataTree.ds
   DataTree.has_data
   DataTree.has_attrs
   DataTree.is_empty

..

   Missing
   DataTree.chunksizes

Dictionary interface
--------------------

.. autosummary::
   :toctree: generated/

   DataTree.__getitem__
   DataTree.__setitem__
   DataTree.update
   DataTree.get

..

   Missing
   DataTree.__delitem__
   DataTree.items
   DataTree.keys
   DataTree.values

Tree Manipulation Methods
-------------------------

.. autosummary::
   :toctree: generated/

   DataTree.orphan
   DataTree.same_tree
   DataTree.relative_to
   DataTree.iter_lineage
   DataTree.find_common_ancestor

Tree Manipulation Utilities
---------------------------

.. autosummary::
   :toctree: generated/

   map_over_subtree

Methods
-------
..

   TODO divide these up into "Dataset contents", "Indexing", "Computation" etc.

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
   DataTree.isomorphic
   DataTree.equals
   DataTree.identical
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

Comparisons
===========

.. autosummary::
   :toctree: generated/

    testing.assert_isomorphic
    testing.assert_equal
    testing.assert_identical

ndarray methods
---------------

.. autosummary::
   :toctree: generated/

   DataTree.nbytes
   DataTree.real
   DataTree.imag

I/O
===

.. autosummary::
   :toctree: generated/

   open_datatree
   DataTree.from_dict
   DataTree.to_dict
   DataTree.to_netcdf
   DataTree.to_zarr

..

   Missing
   open_mfdatatree

Exceptions
==========

.. autosummary::
   :toctree: generated/

    TreeError
    TreeIsomorphismError
