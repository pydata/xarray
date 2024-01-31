.. currentmodule:: datatree

What's New
==========

.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xray
    import xarray
    import xarray as xr
    import datatree

    np.random.seed(123456)

.. _whats-new.v0.0.14:

v0.0.14 (unreleased)
--------------------

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

- Renamed `DataTree.lineage` to `DataTree.parents` to match `pathlib` vocabulary
  (:issue:`283`, :pull:`286`)
- Minimum required version of xarray is now 2023.12.0, i.e. the latest version.
  This is required to prevent recent changes to xarray's internals from breaking datatree.
  (:issue:`293`, :pull:`294`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Change default write mode of :py:meth:`DataTree.to_zarr` to ``'w-'`` to match ``xarray``
  default and prevent accidental directory overwrites. (:issue:`274`, :pull:`275`)
  By `Sam Levang <https://github.com/slevang>`_.

Deprecations
~~~~~~~~~~~~

- Renamed `DataTree.lineage` to `DataTree.parents` to match `pathlib` vocabulary
  (:issue:`283`, :pull:`286`). `lineage` is now deprecated and use of `parents` is encouraged.
  By `Etienne Schalk <https://github.com/etienneschalk>`_.

Bug fixes
~~~~~~~~~
- Keep attributes on nodes containing no data in :py:func:`map_over_subtree`. (:issue:`278`, :pull:`279`)
  By `Sam Levang <https://github.com/slevang>`_.

Documentation
~~~~~~~~~~~~~
- Use ``napoleon`` instead of ``numpydoc`` to align with xarray documentation
  (:issue:`284`, :pull:`298`).
  By `Etienne Schalk <https://github.com/etienneschalk>`_.

Internal Changes
~~~~~~~~~~~~~~~~

.. _whats-new.v0.0.13:

v0.0.13 (27/10/2023)
--------------------

New Features
~~~~~~~~~~~~

- New :py:meth:`DataTree.match` method for glob-like pattern matching of node paths. (:pull:`267`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- New :py:meth:`DataTree.is_hollow` property for checking if data is only contained at the leaf nodes. (:pull:`272`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Indicate which node caused the problem if error encountered while applying user function using :py:func:`map_over_subtree`
  (:issue:`190`, :pull:`264`). Only works when using python 3.11 or later.
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- Nodes containing only attributes but no data are now ignored by :py:func:`map_over_subtree` (:issue:`262`, :pull:`263`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Disallow altering of given dataset inside function called by :py:func:`map_over_subtree` (:pull:`269`, reverts part of :pull:`194`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Bug fixes
~~~~~~~~~

- Fix unittests on i386. (:pull:`249`)
  By `Antonio Valentino <https://github.com/avalentino>`_.
- Ensure nodepath class is compatible with python 3.12 (:pull:`260`)
  By `Max Grover <https://github.com/mgrover1>`_.

Documentation
~~~~~~~~~~~~~

- Added new sections to page on ``Working with Hierarchical Data`` (:pull:`180`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~

* No longer use the deprecated `distutils` package.

.. _whats-new.v0.0.12:

v0.0.12 (03/07/2023)
--------------------

New Features
~~~~~~~~~~~~

- Added a :py:func:`DataTree.level`, :py:func:`DataTree.depth`, and :py:func:`DataTree.width` property (:pull:`208`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Allow dot-style (or "attribute-like") access to child nodes and variables, with ipython autocomplete. (:issue:`189`, :pull:`98`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

- Dropped support for python 3.8 (:issue:`212`, :pull:`214`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Bug fixes
~~~~~~~~~

- Allow for altering of given dataset inside function called by :py:func:`map_over_subtree` (:issue:`188`, :pull:`194`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- copy subtrees without creating ancestor nodes (:pull:`201`)
  By `Justus Magin <https://github.com/keewis>`_.

Documentation
~~~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

.. _whats-new.v0.0.11:

v0.0.11 (01/09/2023)
--------------------

Big update with entirely new pages in the docs,
new methods (``.drop_nodes``, ``.filter``, ``.leaves``, ``.descendants``), and bug fixes!

New Features
~~~~~~~~~~~~

- Added a :py:meth:`DataTree.drop_nodes` method (:issue:`161`, :pull:`175`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- New, more specific exception types for tree-related errors (:pull:`169`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added a new :py:meth:`DataTree.descendants` property (:pull:`170`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added a :py:meth:`DataTree.leaves` property (:pull:`177`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added a :py:meth:`DataTree.filter` method (:pull:`184`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- :py:meth:`DataTree.copy` copy method now only copies the subtree, not the parent nodes (:pull:`171`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Grafting a subtree onto another tree now leaves name of original subtree object unchanged (:issue:`116`, :pull:`172`, :pull:`178`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Changed the :py:meth:`DataTree.assign` method to just work on the local node (:pull:`181`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

- Fix bug with :py:meth:`DataTree.relative_to` method (:issue:`133`, :pull:`160`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fix links to API docs in all documentation (:pull:`183`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Documentation
~~~~~~~~~~~~~

- Changed docs theme to match xarray's main documentation. (:pull:`173`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added ``Terminology`` page. (:pull:`174`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added page on ``Working with Hierarchical Data`` (:pull:`179`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added context content to ``Index`` page (:pull:`182`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Updated the README (:pull:`187`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~


.. _whats-new.v0.0.10:

v0.0.10 (12/07/2022)
--------------------

Adds accessors and a `.pipe()` method.

New Features
~~~~~~~~~~~~

- Add the ability to register accessors on ``DataTree`` objects, by using ``register_datatree_accessor``. (:pull:`144`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Allow method chaining with a new :py:meth:`DataTree.pipe` method (:issue:`151`, :pull:`156`).
  By `Justus Magin <https://github.com/keewis>`_.

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

- Allow ``Datatree`` objects as values in :py:meth:`DataTree.from_dict` (:pull:`159`).
  By `Justus Magin <https://github.com/keewis>`_.

Documentation
~~~~~~~~~~~~~

- Added ``Reading and Writing Files`` page. (:pull:`158`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Avoid reading from same file twice with fsspec3 (:pull:`130`)
  By `William Roberts <https://github.com/wroberts4>`_.


.. _whats-new.v0.0.9:

v0.0.9 (07/14/2022)
-------------------

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

Documentation
~~~~~~~~~~~~~
- Switch docs theme (:pull:`123`).
  By `JuliusBusecke <https://github.com/jbusecke>`_.

Internal Changes
~~~~~~~~~~~~~~~~


.. _whats-new.v0.0.7:

v0.0.7 (07/11/2022)
-------------------

New Features
~~~~~~~~~~~~

- Improve the HTML repr by adding tree-style lines connecting groups and sub-groups (:pull:`109`).
  By `Benjamin Woods <https://github.com/benjaminwoods>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- The ``DataTree.ds`` attribute now returns a view onto an immutable Dataset-like object, instead of an actual instance
  of ``xarray.Dataset``. This make break existing ``isinstance`` checks or ``assert`` comparisons. (:pull:`99`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

- Modifying the contents of a ``DataTree`` object via the ``DataTree.ds`` attribute is now forbidden, which prevents
  any possibility of the contents of a ``DataTree`` object and its ``.ds`` attribute diverging. (:issue:`38`, :pull:`99`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fixed a bug so that names of children now always match keys under which parents store them (:pull:`99`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Documentation
~~~~~~~~~~~~~

- Added ``Data Structures`` page describing the internal structure of a ``DataTree`` object, and its relation to
  ``xarray.Dataset`` objects. (:pull:`103`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- API page updated with all the methods that are copied from ``xarray.Dataset``. (:pull:`41`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Refactored ``DataTree`` class to store a set of ``xarray.Variable`` objects instead of a single ``xarray.Dataset``.
  This approach means that the ``DataTree`` class now effectively copies and extends the internal structure of
  ``xarray.Dataset``. (:pull:`41`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Refactored to use intermediate ``NamedNode`` class, separating implementation of methods requiring a ``name``
  attribute from those not requiring it.
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Made ``testing.test_datatree.create_test_datatree`` into a pytest fixture (:pull:`107`).
  By `Benjamin Woods <https://github.com/benjaminwoods>`_.



.. _whats-new.v0.0.6:

v0.0.6 (06/03/2022)
-------------------

Various small bug fixes, in preparation for more significant changes in the next version.

Bug fixes
~~~~~~~~~

- Fixed bug with checking that assigning parent or new children did not create a loop in the tree (:pull:`105`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Do not call ``__exit__`` on Zarr store when opening (:pull:`90`)
  By `Matt McCormick <https://github.com/thewtex>`_.
- Fix netCDF encoding for compression (:pull:`95`)
  By `Joe Hamman <https://github.com/jhamman>`_.
- Added validity checking for node names (:pull:`106`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

.. _whats-new.v0.0.5:

v0.0.5 (05/05/2022)
-------------------

- Major refactor of internals, moving from the ``DataTree.children`` attribute being a ``Tuple[DataTree]`` to being a
  ``OrderedDict[str, DataTree]``. This was necessary in order to integrate better with xarray's dictionary-like API,
  solve several issues, simplify the code internally, remove dependencies, and enable new features. (:pull:`76`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

New Features
~~~~~~~~~~~~

- Syntax for accessing nodes now supports file-like paths, including parent nodes via ``"../"``, relative paths, the
  root node via ``"/"``, and the current node via ``"."``. (Internally it actually uses ``pathlib`` now.)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- New path-like API methods, such as ``.relative_to``, ``.find_common_ancestor``, and ``.same_tree``.
- Some new dictionary-like methods, such as ``DataTree.get`` and ``DataTree.update``. (:pull:`76`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- New HTML repr, which will automatically display in a jupyter notebook. (:pull:`78`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- New delitem method so you can delete nodes. (:pull:`88`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- New ``to_dict`` method. (:pull:`82`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- Node names are now optional, which means that the root of the tree can be unnamed. This has knock-on effects for
  a lot of the API.
- The ``__init__`` signature for ``DataTree`` has changed, so that ``name`` is now an optional kwarg.
- Files will now be loaded as a slightly different tree, because the root group no longer needs to be given a default
  name.
- Removed tag-like access to nodes.
- Removes the option to delete all data in a node by assigning None to the node (in favour of deleting data by replacing
  the node's ``.ds`` attribute with an empty Dataset), or to create a new empty node in the same way (in favour of
  assigning an empty DataTree object instead).
- Removes the ability to create a new node by assigning a ``Dataset`` object to ``DataTree.__setitem__``.
- Several other minor API changes such as ``.pathstr`` -> ``.path``, and ``from_dict``'s dictionary argument now being
  required. (:pull:`76`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Deprecations
~~~~~~~~~~~~

- No longer depends on the anytree library (:pull:`76`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Bug fixes
~~~~~~~~~

- Fixed indentation issue with the string repr (:pull:`86`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Documentation
~~~~~~~~~~~~~

- Quick-overview page updated to match change in path syntax (:pull:`76`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Basically every file was changed in some way to accommodate (:pull:`76`).
- No longer need the utility functions for string manipulation that were defined in ``utils.py``.
- A considerable amount of code copied over from the internals of anytree (e.g. in ``render.py`` and ``iterators.py``).
  The Apache license for anytree has now been bundled with datatree. (:pull:`76`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

.. _whats-new.v0.0.4:

v0.0.4 (31/03/2022)
-------------------

- Ensure you get the pretty tree-like string representation by default in ipython (:pull:`73`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Now available on conda-forge (as xarray-datatree)! (:pull:`71`)
  By `Anderson Banihirwe <https://github.com/andersy005>`_.
- Allow for python 3.8 (:pull:`70`).
  By `Don Setiawan <https://github.com/lsetiawan>`_.

.. _whats-new.v0.0.3:

v0.0.3 (30/03/2022)
-------------------

- First released version available on both pypi (as xarray-datatree)!
