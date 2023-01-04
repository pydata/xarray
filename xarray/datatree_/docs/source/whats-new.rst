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

.. _whats-new.v0.0.11:

v0.0.11 (unreleased)
--------------------

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

Breaking changes
~~~~~~~~~~~~~~~~

- :py:meth:`DataTree.copy` copy method now only copies the subtree, not the parent nodes (:pull:`171`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Grafting a subtree onto another tree now leaves name of original subtree object unchanged (:issue:`116`, :pull:`172`, :pull:`178`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

- Fix bug with :py:meth:`DataTree.relative_to` method (:issue:`133`, :pull:`160`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Documentation
~~~~~~~~~~~~~

- Changed docs theme to match xarray's main documentation. (:pull:`173`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added ``Terminology`` page. (:pull:`174`)
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
- Removes the ability to create a new node by assigning a ``Dataset`` object to ``DataTree.__setitem__`.
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
