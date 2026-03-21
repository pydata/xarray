.. currentmodule:: xarray

DataTree
========

Creating a DataTree
-------------------

Methods of creating a ``DataTree``.

.. autosummary::
   :toctree: ../generated/

   DataTree
   DataTree.from_dict

Tree Attributes
---------------

Attributes relating to the recursive tree-like structure of a ``DataTree``.

.. autosummary::
   :toctree: ../generated/

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
   DataTree.subtree_with_keys
   DataTree.descendants
   DataTree.siblings
   DataTree.lineage
   DataTree.parents
   DataTree.ancestors
   DataTree.groups
   DataTree.xindexes

Data Contents
-------------

Interface to the data objects (optionally) stored inside a single ``DataTree`` node.
This interface echoes that of ``xarray.Dataset``.

.. autosummary::
   :toctree: ../generated/

   DataTree.dims
   DataTree.sizes
   DataTree.data_vars
   DataTree.ds
   DataTree.coords
   DataTree.attrs
   DataTree.encoding
   DataTree.indexes
   DataTree.nbytes
   DataTree.dataset
   DataTree.to_dataset
   DataTree.has_data
   DataTree.has_attrs
   DataTree.is_empty
   DataTree.is_hollow
   DataTree.chunksizes

Dictionary Interface
--------------------

``DataTree`` objects also have a dict-like interface mapping keys to either ``xarray.DataArray``\s or to child ``DataTree`` nodes.

.. autosummary::
   :toctree: ../generated/

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
   :toctree: ../generated/

   DataTree.orphan
   DataTree.same_tree
   DataTree.relative_to
   DataTree.iter_lineage
   DataTree.find_common_ancestor
   DataTree.map_over_datasets
   DataTree.pipe
   DataTree.match
   DataTree.filter
   DataTree.filter_like

Pathlib-like Interface
----------------------

``DataTree`` objects deliberately echo some of the API of :py:class:`pathlib.PurePath`.

.. autosummary::
   :toctree: ../generated/

   DataTree.name
   DataTree.parent
   DataTree.parents
   DataTree.relative_to

.. Missing:

.. ..

..    ``DataTree.glob``
..    ``DataTree.joinpath``
..    ``DataTree.with_name``
..    ``DataTree.walk``
..    ``DataTree.rename``
..    ``DataTree.replace``

DataTree Contents
-----------------

Manipulate the contents of all nodes in a ``DataTree`` simultaneously.

.. autosummary::
   :toctree: ../generated/

   DataTree.copy

   .. DataTree.assign_coords
   .. DataTree.merge
   .. DataTree.rename
   .. DataTree.rename_vars
   .. DataTree.rename_dims
   .. DataTree.swap_dims
   .. DataTree.expand_dims
   .. DataTree.drop_vars
   .. DataTree.drop_dims
   .. DataTree.set_coords
   .. DataTree.reset_coords

DataTree Node Contents
----------------------

Manipulate the contents of a single ``DataTree`` node.

.. autosummary::
   :toctree: ../generated/

   DataTree.assign
   DataTree.drop_nodes

DataTree Operations
-------------------

Apply operations over multiple ``DataTree`` objects.

.. autosummary::
   :toctree: ../generated/

   map_over_datasets
   group_subtrees

Comparisons
-----------

Compare one ``DataTree`` object to another.

.. autosummary::
   :toctree: ../generated/

    DataTree.isomorphic
    DataTree.equals
    DataTree.identical

Indexing
--------

Index into all nodes in the subtree simultaneously.

.. autosummary::
   :toctree: ../generated/

   DataTree.isel
   DataTree.sel

..    DataTree.drop_sel
..    DataTree.drop_isel
..    DataTree.head
..    DataTree.tail
..    DataTree.thin
..    DataTree.squeeze
..    DataTree.interp
..    DataTree.interp_like
..    DataTree.reindex
..    DataTree.reindex_like
..    DataTree.set_index
..    DataTree.reset_index
..    DataTree.reorder_levels
..    DataTree.query

.. ..

..    Missing:
..    ``DataTree.loc``


.. Missing Value Handling
.. ----------------------

.. .. autosummary::
..    :toctree: ../generated/

..    DataTree.isnull
..    DataTree.notnull
..    DataTree.combine_first
..    DataTree.dropna
..    DataTree.fillna
..    DataTree.ffill
..    DataTree.bfill
..    DataTree.interpolate_na
..    DataTree.where
..    DataTree.isin

.. Computation
.. -----------

.. Apply a computation to the data in all nodes in the subtree simultaneously.

.. .. autosummary::
..    :toctree: ../generated/

..    DataTree.map
..    DataTree.reduce
..    DataTree.diff
..    DataTree.quantile
..    DataTree.differentiate
..    DataTree.integrate
..    DataTree.map_blocks
..    DataTree.polyfit
..    DataTree.curvefit

Aggregation
-----------

Aggregate data in all nodes in the subtree simultaneously.

.. autosummary::
   :toctree: ../generated/

   DataTree.all
   DataTree.any
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
---------------

Methods copied from :py:class:`numpy.ndarray` objects, here applying to the data in all nodes in the subtree.

.. autosummary::
   :toctree: ../generated/

   DataTree.argsort
   DataTree.conj
   DataTree.conjugate
   DataTree.round
..    DataTree.astype
..    DataTree.clip
..    DataTree.rank

.. Reshaping and reorganising
.. --------------------------

.. Reshape or reorganise the data in all nodes in the subtree.

.. .. autosummary::
..    :toctree: ../generated/

..    DataTree.transpose
..    DataTree.stack
..    DataTree.unstack
..    DataTree.shift
..    DataTree.roll
..    DataTree.pad
..    DataTree.sortby
..    DataTree.broadcast_like
