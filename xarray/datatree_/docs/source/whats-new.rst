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

.. _whats-new.v0.0.6:

v0.0.6 (unreleased)
-------------------

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

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

Documentation
~~~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

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
