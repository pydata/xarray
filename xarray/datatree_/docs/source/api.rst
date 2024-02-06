.. currentmodule:: datatree

#############
API reference
#############

DataTree
========

Creating a DataTree
-------------------

Methods of creating a datatree.

.. autosummary::
   :toctree: generated/

   DataTree
   DataTree.from_dict

Tree Attributes
---------------

Attributes relating to the recursive tree-like structure of a ``DataTree``.

.. autosummary::
   :toctree: generated/

   DataTree.parent
   DataTree.children
   DataTree.name
   DataTree.path
   DataTree.root
   DataTree.is_root
   DataTree.is_leaf
   DataTree.leaves
   DataTree.level
   DataTree.depth
   DataTree.width
   DataTree.subtree
   DataTree.descendants
   DataTree.siblings
   DataTree.lineage
   DataTree.parents
   DataTree.ancestors
   DataTree.groups

Data Contents
-------------

Interface to the data objects (optionally) stored inside a single ``DataTree`` node.
This interface echoes that of ``xarray.Dataset``.

.. autosummary::
   :toctree: generated/

   DataTree.dims
   DataTree.sizes
   DataTree.data_vars
   DataTree.coords
   DataTree.attrs
   DataTree.encoding
   DataTree.indexes
   DataTree.nbytes
   DataTree.ds
   DataTree.to_dataset
   DataTree.has_data
   DataTree.has_attrs
   DataTree.is_empty
   DataTree.is_hollow

Dictionary Interface
--------------------

``DataTree`` objects also have a dict-like interface mapping keys to either ``xarray.DataArray``s or to child ``DataTree`` nodes.

.. autosummary::
   :toctree: generated/

   DataTree.__getitem__
   DataTree.__setitem__
   DataTree.__delitem__
   DataTree.update
   DataTree.get
   DataTree.items
   DataTree.keys
   DataTree.values

Tree Manipulation
-----------------

For manipulating, traversing, navigating, or mapping over the tree structure.

.. autosummary::
   :toctree: generated/

   DataTree.orphan
   DataTree.same_tree
   DataTree.relative_to
   DataTree.iter_lineage
   DataTree.find_common_ancestor
   DataTree.map_over_subtree
   map_over_subtree
   DataTree.pipe
   DataTree.match
   DataTree.filter

Pathlib-like Interface
----------------------

``DataTree`` objects deliberately echo some of the API of `pathlib.PurePath`.

.. autosummary::
   :toctree: generated/

   DataTree.name
   DataTree.parent
   DataTree.parents
   DataTree.relative_to

Missing:

..

   ``DataTree.glob``
   ``DataTree.joinpath``
   ``DataTree.with_name``
   ``DataTree.walk``
   ``DataTree.rename``
   ``DataTree.replace``

DataTree Contents
-----------------

Manipulate the contents of all nodes in a tree simultaneously.

.. autosummary::
   :toctree: generated/

   DataTree.copy
   DataTree.assign_coords
   DataTree.merge
   DataTree.rename
   DataTree.rename_vars
   DataTree.rename_dims
   DataTree.swap_dims
   DataTree.expand_dims
   DataTree.drop_vars
   DataTree.drop_dims
   DataTree.set_coords
   DataTree.reset_coords

DataTree Node Contents
----------------------

Manipulate the contents of a single DataTree node.

.. autosummary::
   :toctree: generated/

   DataTree.assign
   DataTree.drop_nodes

Comparisons
===========

Compare one ``DataTree`` object to another.

.. autosummary::
   :toctree: generated/

    DataTree.isomorphic
    DataTree.equals
    DataTree.identical

Indexing
========

Index into all nodes in the subtree simultaneously.

.. autosummary::
   :toctree: generated/

   DataTree.isel
   DataTree.sel
   DataTree.drop_sel
   DataTree.drop_isel
   DataTree.head
   DataTree.tail
   DataTree.thin
   DataTree.squeeze
   DataTree.interp
   DataTree.interp_like
   DataTree.reindex
   DataTree.reindex_like
   DataTree.set_index
   DataTree.reset_index
   DataTree.reorder_levels
   DataTree.query

..

   Missing:
   ``DataTree.loc``


Missing Value Handling
======================

.. autosummary::
   :toctree: generated/

   DataTree.isnull
   DataTree.notnull
   DataTree.combine_first
   DataTree.dropna
   DataTree.fillna
   DataTree.ffill
   DataTree.bfill
   DataTree.interpolate_na
   DataTree.where
   DataTree.isin

Computation
===========

Apply a computation to the data in all nodes in the subtree simultaneously.

.. autosummary::
   :toctree: generated/

   DataTree.map
   DataTree.reduce
   DataTree.diff
   DataTree.quantile
   DataTree.differentiate
   DataTree.integrate
   DataTree.map_blocks
   DataTree.polyfit
   DataTree.curvefit

Aggregation
===========

Aggregate data in all nodes in the subtree simultaneously.

.. autosummary::
   :toctree: generated/

   DataTree.all
   DataTree.any
   DataTree.argmax
   DataTree.argmin
   DataTree.idxmax
   DataTree.idxmin
   DataTree.max
   DataTree.min
   DataTree.mean
   DataTree.median
   DataTree.prod
   DataTree.sum
   DataTree.std
   DataTree.var
   DataTree.cumsum
   DataTree.cumprod

ndarray methods
===============

Methods copied from :py:class:`numpy.ndarray` objects, here applying to the data in all nodes in the subtree.

.. autosummary::
   :toctree: generated/

   DataTree.argsort
   DataTree.astype
   DataTree.clip
   DataTree.conj
   DataTree.conjugate
   DataTree.round
   DataTree.rank

Reshaping and reorganising
==========================

Reshape or reorganise the data in all nodes in the subtree.

.. autosummary::
   :toctree: generated/

   DataTree.transpose
   DataTree.stack
   DataTree.unstack
   DataTree.shift
   DataTree.roll
   DataTree.pad
   DataTree.sortby
   DataTree.broadcast_like

Plotting
========

I/O
===

Open a datatree from an on-disk store or serialize the tree.

.. autosummary::
   :toctree: generated/

   open_datatree
   DataTree.to_dict
   DataTree.to_netcdf
   DataTree.to_zarr

..

   Missing:
   ``open_mfdatatree``

Tutorial
========

Testing
=======

Test that two DataTree objects are similar.

.. autosummary::
   :toctree: generated/

   testing.assert_isomorphic
   testing.assert_equal
   testing.assert_identical

Exceptions
==========

Exceptions raised when manipulating trees.

.. autosummary::
   :toctree: generated/

   TreeIsomorphismError
   InvalidTreeError
   NotFoundInTreeError

Advanced API
============

Relatively advanced API for users or developers looking to understand the internals, or extend functionality.

.. autosummary::
   :toctree: generated/

   DataTree.variables
   register_datatree_accessor

..

   Missing:
   ``DataTree.set_close``
