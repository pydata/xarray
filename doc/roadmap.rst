.. _roadmap:

Development roadmap
===================

Authors: Xarray developers

Date: September 7, 2021

Xarray is an open source Python library for labeled multidimensional
arrays and datasets.

Our philosophy
--------------

Why has xarray been successful? In our opinion:

-  Xarray does a great job of solving **specific use-cases** for
   multidimensional data analysis:

   -  The dominant use-case for xarray is for analysis of gridded
      dataset in the geosciences, e.g., as part of the
      `Pangeo <https://pangeo.io>`__ project.
   -  Xarray is also used more broadly in the physical sciences, where
      we've found the needs for analyzing multidimensional datasets are
      remarkably consistent (e.g., see
      `SunPy <https://github.com/sunpy/ndcube>`__ and
      `PlasmaPy <https://github.com/PlasmaPy/PlasmaPy/issues/59>`__).
   -  Finally, xarray is used in a variety of other domains, including
      finance, `probabilistic
      programming <https://arviz-devs.github.io/arviz/>`__ and
      genomics.

-  Xarray is also a **domain agnostic** solution:

   -  We focus on providing a flexible set of functionality related
      labeled multidimensional arrays, rather than solving particular
      problems.
   -  This facilitates collaboration between users with different needs,
      and helps us attract a broad community of contributors.
   -  Importantly, this retains flexibility, for use cases that don't
      fit particularly well into existing frameworks.

-  Xarray **integrates well** with other libraries in the scientific
   Python stack.

   -  We leverage first-class external libraries for core features of
      xarray (e.g., NumPy for ndarrays, pandas for indexing, dask for
      parallel computing)
   -  We expose our internal abstractions to users (e.g.,
      ``apply_ufunc()``), which facilitates extending xarray in various
      ways.

Together, these features have made xarray a first-class choice for
labeled multidimensional arrays in Python.

We want to double-down on xarray's strengths by making it an even more
flexible and powerful tool for multidimensional data analysis. We want
to continue to engage xarray's core geoscience users, and to also reach
out to new domains to learn from other successful data models like those
of `yt <https://yt-project.org>`__ or the `OLAP
cube <https://en.wikipedia.org/wiki/OLAP_cube>`__.

Specific needs
--------------

The user community has voiced a number specific needs related to how
xarray interfaces with domain specific problems. Xarray may not solve
all of these issues directly, but these areas provide opportunities for
xarray to provide better, more extensible, interfaces. Some examples of
these common needs are:

-  Non-regular grids (e.g., staggered and unstructured meshes).
-  Physical units.
-  Lazily computed arrays (e.g., for coordinate systems).
-  New file-formats.

Technical vision
----------------

We think the right approach to extending xarray's user community and the
usefulness of the project is to focus on improving key interfaces that
can be used externally to meet domain-specific needs.

We can generalize the community's needs into three main categories:

-  More flexible grids/indexing.
-  More flexible arrays/computing.
-  More flexible storage backends.
-  More flexible data structures.

Each of these are detailed further in the subsections below.

Flexible indexes
~~~~~~~~~~~~~~~~

.. note::
   Work on flexible grids and indexes is currently underway. See
   `GH Project #1 <https://github.com/pydata/xarray/projects/1>`__ for more detail.

Xarray currently keeps track of indexes associated with coordinates by
storing them in the form of a ``pandas.Index`` in special
``xarray.IndexVariable`` objects.

The limitations of this model became clear with the addition of
``pandas.MultiIndex`` support in xarray 0.9, where a single index
corresponds to multiple xarray variables. MultiIndex support is highly
useful, but xarray now has numerous special cases to check for
MultiIndex levels.

A cleaner model would be to elevate ``indexes`` to an explicit part of
xarray's data model, e.g., as attributes on the ``Dataset`` and
``DataArray`` classes. Indexes would need to be propagated along with
coordinates in xarray operations, but will no longer would need to have
a one-to-one correspondence with coordinate variables. Instead, an index
should be able to refer to multiple (possibly multidimensional)
coordinates that define it. See :issue:`1603` for full details.

Specific tasks:

-  Add an ``indexes`` attribute to ``xarray.Dataset`` and
   ``xarray.Dataset``, as dictionaries that map from coordinate names to
   xarray index objects.
-  Use the new index interface to write wrappers for ``pandas.Index``,
   ``pandas.MultiIndex`` and ``scipy.spatial.KDTree``.
-  Expose the interface externally to allow third-party libraries to
   implement custom indexing routines, e.g., for geospatial look-ups on
   the surface of the Earth.

In addition to the new features it directly enables, this clean up will
allow xarray to more easily implement some long-awaited features that
build upon indexing, such as groupby operations with multiple variables.

Flexible arrays
~~~~~~~~~~~~~~~

.. note::
   Work on flexible arrays is currently underway. See
   `GH Project #2 <https://github.com/pydata/xarray/projects/2>`__ for more detail.

Xarray currently supports wrapping multidimensional arrays defined by
NumPy, dask and to a limited-extent pandas. It would be nice to have
interfaces that allow xarray to wrap alternative N-D array
implementations, e.g.:

-  Arrays holding physical units.
-  Lazily computed arrays.
-  Other ndarray objects, e.g., sparse, xnd, xtensor.

Our strategy has been to pursue upstream improvements in NumPy (see
`NEP-22 <https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html>`__)
for supporting a complete duck-typing interface using with NumPy's
higher level array API. Improvements in NumPy's support for custom data
types would also be highly useful for xarray users.

By pursuing these improvements in NumPy we hope to extend the benefits
to the full scientific Python community, and avoid tight coupling
between xarray and specific third-party libraries (e.g., for
implementing units). This will allow xarray to maintain its domain
agnostic strengths.

We expect that we may eventually add some minimal interfaces in xarray
for features that we delegate to external array libraries (e.g., for
getting units and changing units). If we do add these features, we
expect them to be thin wrappers, with core functionality implemented by
third-party libraries.

Flexible storage
~~~~~~~~~~~~~~~~

.. note::
   Work on flexible storage backends is currently underway. See
   `GH Project #3 <https://github.com/pydata/xarray/projects/3>`__ for more detail.

The xarray backends module has grown in size and complexity. Much of
this growth has been "organic" and mostly to support incremental
additions to the supported backends. This has left us with a fragile
internal API that is difficult for even experienced xarray developers to
use. Moreover, the lack of a public facing API for building xarray
backends means that users can not easily build backend interface for
xarray in third-party libraries.

The idea of refactoring the backends API and exposing it to users was
originally proposed in :issue:`1970`. The idea would be to develop a
well tested and generic backend base class and associated utilities
for external use. Specific tasks for this development would include:

-  Exposing an abstract backend for writing new storage systems.
-  Exposing utilities for features like automatic closing of files,
   LRU-caching and explicit/lazy indexing.
-  Possibly moving some infrequently used backends to third-party
   packages.

Flexible data structures
~~~~~~~~~~~~~~~~~~~~~~~~

Xarray provides two primary data structures, the ``xarray.DataArray`` and
the ``xarray.Dataset``. This section describes two possible data model
extensions.

Tree-like data structure
++++++++++++++++++++++++

.. note::

   After some time, the community DataTree project has now been updated and
   merged into xarray exposing :py:class:`xarray.DataTree`. This is just
   released and a bit experimental, but please try it out and let us know what
   you think. Take a look at our :ref:`quick-overview-datatrees` quickstart.

Xarray’s highest-level object was previously an ``xarray.Dataset``, whose data
model echoes that of a single netCDF group. However real-world datasets are
often better represented by a collection of related Datasets. Particular common
examples include:

-  Multi-resolution datasets,
-  Collections of time series datasets with differing lengths,
-  Heterogeneous datasets comprising multiple different types of related
   observational or simulation data,
-  Bayesian workflows involving various statistical distributions over multiple
   variables,
-  Whole netCDF files containing multiple groups.
-  Comparison of output from many similar models (such as in the IPCC's Coupled Model Intercomparison Projects)

A new tree-like data structure, ``xarray.DataTree``, which is essentially a
structured hierarchical collection of Datasets, represents these cases and
instead maps to multiple netCDF groups (see :issue:`4118`).

Currently there are several libraries which have wrapped xarray in order to
build domain-specific data structures (e.g. `xarray-multiscale
<https://github.com/JaneliaSciComp/xarray-multiscale>`__.), but the general
``xarray.DataTree`` object obviates the need for these and consolidates effort
in a single domain-agnostic tool, much as xarray has already achieved.


Labeled array without coordinates
+++++++++++++++++++++++++++++++++

There is a need for a lightweight array structure with named dimensions for
convenient indexing and broadcasting. Xarray includes such a structure internally
(``xarray.Variable``). We want to factor out xarray's “Variable”  object into a
standalone package with minimal dependencies for integration with libraries that
don't want to inherit xarray's dependency on pandas (e.g. scikit-learn).
The new “Variable” class will follow established array protocols and the new
data-apis standard. It will be capable of wrapping multiple array-like objects
(e.g. NumPy, Dask, Sparse, Pint, CuPy, Pytorch). While “DataArray” fits some of
these requirements, it offers a more complex data model than is desired for
many applications and depends on pandas.

Engaging more users
-------------------

.. note::
   Work on improving xarray’s documentation and user engagement is
   currently underway. See `GH Project #4 <https://github.com/pydata/xarray/projects/4>`__
   for more detail.

Like many open-source projects, the documentation of xarray has grown
together with the library's features. While we think that the xarray
documentation is comprehensive already, we acknowledge that the adoption
of xarray might be slowed down because of the substantial time
investment required to learn its working principles. In particular,
non-computer scientists or users less familiar with the pydata ecosystem
might find it difficult to learn xarray and realize how xarray can help
them in their daily work.

In order to lower this adoption barrier, we propose to:

-  Develop entry-level tutorials for users with different backgrounds. For
   example, we would like to develop tutorials for users with or without
   previous knowledge of pandas, NumPy, netCDF, etc. These tutorials may be
   built as part of xarray's documentation or included in a separate repository
   to enable interactive use (e.g. mybinder.org).
-  Document typical user workflows in a dedicated website, following the example
   of `dask-stories
   <https://matthewrocklin.com/blog/work/2018/07/16/dask-stories>`__.
-  Write a basic glossary that defines terms that might not be familiar to all
   (e.g. "lazy", "labeled", "serialization", "indexing", "backend").


Administrative
--------------

NumFOCUS
~~~~~~~~

On July 16, 2018, Joe and Stephan submitted xarray's fiscal sponsorship
application to NumFOCUS.
