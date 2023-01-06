.. currentmodule:: datatree

Datatree
========

**Datatree is a prototype implementation of a tree-like hierarchical data structure for xarray.**

Why Datatree?
~~~~~~~~~~~~~

Datatree was born after the xarray team recognised a `need for a new hierarchical data structure <https://github.com/pydata/xarray/issues/4118>`_,
that was more flexible than a single :py:class:`xarray.Dataset` object.
The initial motivation was to represent netCDF files / Zarr stores with multiple nested groups in a single in-memory object,
but :py:class:`~datatree.DataTree` objects have many other uses.

You might want to use datatree for:

- Organising many related datasets, e.g. results of the same experiment with different parameters, or simulations of the same system using different models,
- Analysing similar data at multiple resolutions simultaneously, such as when doing a convergence study,
- Comparing heterogenous but related data, such as experimental and theoretical data,
- I/O with nested data formats such as netCDF / Zarr groups.

Development Roadmap
~~~~~~~~~~~~~~~~~~~

Datatree currently lives in a separate repository to the main xarray package.
This allows the datatree developers to make changes to it, experiment, and improve it faster.

Eventually we plan to fully integrate datatree upstream into xarray's main codebase, at which point the `github.com/xarray-contrib/datatree <https://github.com/xarray-contrib/datatree>`_ repository will be archived.
This should not cause much disruption to code that depends on datatree - you will likely only have to change the import line (i.e. from ``from datatree import DataTree`` to ``from xarray import DataTree``).

However, until this full integration occurs, datatree's API should not be considered to have the same `level of stability as xarray's <https://docs.xarray.dev/en/stable/contributing.html#backwards-compatibility>`_.

User Feedback
~~~~~~~~~~~~~

We really really really want to hear your opinions on datatree!
At this point in development, user feedback is critical to help us create something that will suit everyone's needs.
Please raise any thoughts, issues, suggestions or bugs, no matter how small or large, on the `github issue tracker <https://github.com/xarray-contrib/datatree/issues>`_.

.. toctree::
   :maxdepth: 2
   :caption: Documentation Contents

   Installation <installation>
   Quick Overview <quick-overview>
   Tutorial <tutorial>
   Data Model <data-structures>
   Hierarchical Data <hierarchical-data>
   Reading and Writing Files <io>
   API Reference <api>
   Terminology <terminology>
   How do I ... <howdoi>
   Contributing Guide <contributing>
   What's New <whats-new>
   GitHub repository <https://github.com/xarray-contrib/datatree>

Feedback
--------

If you encounter any errors, problems with **Datatree**, or have any suggestions, please open an issue
on `GitHub <http://github.com/xarray-contrib/datatree>`_.
