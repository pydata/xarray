.. currentmodule:: xarray

What's New
==========

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xray
    import xarray
    import xarray as xr
    np.random.seed(123456)

.. _whats-new.0.15.1:


v0.15.1 (23 Mar 2020)
---------------------

This release brings many new features such as :py:meth:`Dataset.weighted` methods for weighted array
reductions, a new jupyter repr by default, and the start of units integration with pint. There's also
the usual batch of usability improvements, documentation additions, and bug fixes.

Breaking changes
~~~~~~~~~~~~~~~~

- Raise an error when assigning to the ``.values`` or ``.data`` attribute of
  dimension coordinates i.e. ``IndexVariable`` objects. This has been broken since
  v0.12.0. Please use :py:meth:`DataArray.assign_coords` or :py:meth:`Dataset.assign_coords`
  instead. (:issue:`3470`, :pull:`3862`)
  By `Deepak Cherian <https://github.com/dcherian>`_

New Features
~~~~~~~~~~~~

- Weighted array reductions are now supported via the new :py:meth:`DataArray.weighted`
  and :py:meth:`Dataset.weighted` methods. See :ref:`comput.weighted`. (:issue:`422`, :pull:`2922`).
  By `Mathias Hauser <https://github.com/mathause>`_
- The new jupyter notebook repr (``Dataset._repr_html_`` and
  ``DataArray._repr_html_``) (introduced in 0.14.1) is now on by default. To
  disable, use ``xarray.set_options(display_style="text")``.
  By `Julia Signell <https://github.com/jsignell>`_.
- Added support for :py:class:`pandas.DatetimeIndex`-style rounding of
  ``cftime.datetime`` objects directly via a :py:class:`CFTimeIndex` or via the
  :py:class:`~core.accessor_dt.DatetimeAccessor`.
  By `Spencer Clark <https://github.com/spencerkclark>`_ 
- Support new h5netcdf backend keyword `phony_dims` (available from h5netcdf
  v0.8.0 for :py:class:`~xarray.backends.H5NetCDFStore`.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Add partial support for unit aware arrays with pint. (:pull:`3706`, :pull:`3611`)
  By `Justus Magin <https://github.com/keewis>`_.
- :py:meth:`Dataset.groupby` and :py:meth:`DataArray.groupby` now raise a 
  `TypeError` on multiple string arguments. Receiving multiple string arguments
  often means a user is attempting to pass multiple dimensions as separate
  arguments and should instead pass a single list of dimensions.
  (:pull:`3802`)
  By `Maximilian Roos <https://github.com/max-sixty>`_
- :py:func:`map_blocks` can now apply functions that add new unindexed dimensions.
  By `Deepak Cherian <https://github.com/dcherian>`_
- An ellipsis (``...``) is now supported in the ``dims`` argument of
  :py:meth:`Dataset.stack` and :py:meth:`DataArray.stack`, meaning all
  unlisted dimensions, similar to its meaning in :py:meth:`DataArray.transpose`.
  (:pull:`3826`)
  By `Maximilian Roos <https://github.com/max-sixty>`_
- :py:meth:`Dataset.where` and :py:meth:`DataArray.where` accept a lambda as a
  first argument, which is then called on the input; replicating pandas' behavior.
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- ``skipna`` is available in :py:meth:`Dataset.quantile`, :py:meth:`DataArray.quantile`,
  :py:meth:`core.groupby.DatasetGroupBy.quantile`, :py:meth:`core.groupby.DataArrayGroupBy.quantile`
  (:issue:`3843`, :pull:`3844`) 
  By `Aaron Spring <https://github.com/aaronspring>`_.

Bug fixes
~~~~~~~~~

- Fix :py:meth:`Dataset.interp` when indexing array shares coordinates with the
  indexed variable (:issue:`3252`).
  By `David Huard <https://github.com/huard>`_.
- Fix recombination of groups in :py:meth:`Dataset.groupby` and
  :py:meth:`DataArray.groupby` when performing an operation that changes the
  size of the groups along the grouped dimension. By `Eric Jansen
  <https://github.com/ej81>`_.
- Fix use of multi-index with categorical values (:issue:`3674`).
  By `Matthieu Ancellin <https://github.com/mancellin>`_.
- Fix alignment with ``join="override"`` when some dimensions are unindexed. (:issue:`3681`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix :py:meth:`Dataset.swap_dims` and :py:meth:`DataArray.swap_dims` producing
  index with name reflecting the previous dimension name instead of the new one
  (:issue:`3748`, :pull:`3752`). By `Joseph K Aicher
  <https://github.com/jaicher>`_.
- Use ``dask_array_type`` instead of ``dask_array.Array`` for type
  checking. (:issue:`3779`, :pull:`3787`)
  By `Justus Magin <https://github.com/keewis>`_.
- :py:func:`concat` can now handle coordinate variables only present in one of
  the objects to be concatenated when ``coords="different"``.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- xarray now respects the over, under and bad colors if set on a provided colormap.
  (:issue:`3590`, :pull:`3601`)
  By `johnomotani <https://github.com/johnomotani>`_.
- :py:func:`coarsen` now respects ``xr.set_options(keep_attrs=True)``
  to preserve attributes. :py:meth:`Dataset.coarsen` accepts a keyword
  argument ``keep_attrs`` to change this setting. (:issue:`3376`,
  :pull:`3801`) By `Andrew Thomas <https://github.com/amcnicho>`_.
- Delete associated indexes when deleting coordinate variables. (:issue:`3746`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix :py:meth:`xarray.core.dataset.Dataset.to_zarr` when using `append_dim` and `group`
  simultaneously. (:issue:`3170`). By `Matthias Meyer <https://github.com/niowniow>`_.
- Fix html repr on :py:class:`Dataset` with non-string keys (:pull:`3807`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.

Documentation
~~~~~~~~~~~~~

- Fix documentation of :py:class:`DataArray` removing the deprecated mention
  that when omitted, `dims` are inferred from a `coords`-dict. (:pull:`3821`)
  By `Sander van Rijn <https://github.com/sjvrijn>`_.
- Improve the :py:func:`where` docstring.
  By `Maximilian Roos <https://github.com/max-sixty>`_
- Update the installation instructions: only explicitly list recommended dependencies
  (:issue:`3756`).
  By `Mathias Hauser <https://github.com/mathause>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Remove the internal ``import_seaborn`` function which handled the deprecation of
  the ``seaborn.apionly`` entry point (:issue:`3747`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Don't test pint integration in combination with datetime objects. (:issue:`3778`, :pull:`3788`)
  By `Justus Magin <https://github.com/keewis>`_.
- Change test_open_mfdataset_list_attr to only run with dask installed
  (:issue:`3777`, :pull:`3780`).
  By `Bruno Pagani <https://github.com/ArchangeGabriel>`_.
- Preserve the ability to index with ``method="nearest"`` with a
  :py:class:`CFTimeIndex` with pandas versions greater than 1.0.1
  (:issue:`3751`). By `Spencer Clark <https://github.com/spencerkclark>`_.
- Greater flexibility and improved test coverage of subtracting various types
  of objects from a :py:class:`CFTimeIndex`. By `Spencer Clark
  <https://github.com/spencerkclark>`_.
- Update Azure CI MacOS image, given pending removal.
  By `Maximilian Roos <https://github.com/max-sixty>`_
- Remove xfails for scipy 1.0.1 for tests that append to netCDF files (:pull:`3805`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Remove conversion to :py:class:`pandas.Panel`, given its removal in pandas
  in favor of xarray's objects.
  By `Maximilian Roos <https://github.com/max-sixty>`_

.. _whats-new.0.15.0:


v0.15.0 (30 Jan 2020)
---------------------

This release brings many improvements to xarray's documentation: our examples are now binderized notebooks (`click here <https://mybinder.org/v2/gh/pydata/xarray/master?urlpath=lab/tree/doc/examples/weather-data.ipynb>`_)
and we have new example notebooks from our SciPy 2019 sprint (many thanks to our contributors!).

This release also features many API improvements such as a new
:py:class:`~core.accessor_dt.TimedeltaAccessor` and support for :py:class:`CFTimeIndex` in
:py:meth:`~DataArray.interpolate_na`); as well as many bug fixes.

Breaking changes
~~~~~~~~~~~~~~~~
- Bumped minimum tested versions for dependencies:

  - numpy 1.15
  - pandas 0.25
  - dask 2.2
  - distributed 2.2
  - scipy 1.3

- Remove ``compat`` and ``encoding`` kwargs from ``DataArray``, which
  have been deprecated since 0.12. (:pull:`3650`).
  Instead, specify the ``encoding`` kwarg when writing to disk or set
  the :py:attr:`DataArray.encoding` attribute directly.
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- :py:func:`xarray.dot`, :py:meth:`DataArray.dot`, and the ``@`` operator now
  use ``align="inner"`` (except when ``xarray.set_options(arithmetic_join="exact")``;
  :issue:`3694`) by `Mathias Hauser <https://github.com/mathause>`_.

New Features
~~~~~~~~~~~~
- Implement :py:meth:`DataArray.pad` and :py:meth:`Dataset.pad`. (:issue:`2605`, :pull:`3596`).
  By `Mark Boer <https://github.com/mark-boer>`_.
- :py:meth:`DataArray.sel` and :py:meth:`Dataset.sel` now support :py:class:`pandas.CategoricalIndex`. (:issue:`3669`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Support using an existing, opened h5netcdf ``File`` with
  :py:class:`~xarray.backends.H5NetCDFStore`. This permits creating an
  :py:class:`~xarray.Dataset` from a h5netcdf ``File`` that has been opened
  using other means (:issue:`3618`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Implement ``median`` and ``nanmedian`` for dask arrays. This works by rechunking
  to a single chunk along all reduction axes. (:issue:`2999`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- :py:func:`~xarray.concat` now preserves attributes from the first Variable.
  (:issue:`2575`, :issue:`2060`, :issue:`1614`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- :py:meth:`Dataset.quantile`, :py:meth:`DataArray.quantile` and ``GroupBy.quantile``
  now work with dask Variables.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Added the ``count`` reduction method to both :py:class:`~core.rolling.DatasetCoarsen`
  and :py:class:`~core.rolling.DataArrayCoarsen` objects. (:pull:`3500`)
  By `Deepak Cherian <https://github.com/dcherian>`_
- Add ``meta`` kwarg to :py:func:`~xarray.apply_ufunc`;
  this is passed on to :py:func:`dask.array.blockwise`. (:pull:`3660`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Add ``attrs_file`` option in :py:func:`~xarray.open_mfdataset` to choose the
  source file for global attributes in a multi-file dataset (:issue:`2382`,
  :pull:`3498`). By `Julien Seguinot <https://github.com/juseg>`_.
- :py:meth:`Dataset.swap_dims` and :py:meth:`DataArray.swap_dims`
  now allow swapping to dimension names that don't exist yet. (:pull:`3636`)
  By `Justus Magin <https://github.com/keewis>`_.
- Extend :py:class:`~core.accessor_dt.DatetimeAccessor` properties
  and support ``.dt`` accessor for timedeltas
  via :py:class:`~core.accessor_dt.TimedeltaAccessor` (:pull:`3612`)
  By `Anderson Banihirwe <https://github.com/andersy005>`_.
- Improvements to interpolating along time axes (:issue:`3641`, :pull:`3631`).
  By `David Huard <https://github.com/huard>`_.

  - Support :py:class:`CFTimeIndex` in :py:meth:`DataArray.interpolate_na`
  - define 1970-01-01 as the default offset for the interpolation index for both
    :py:class:`pandas.DatetimeIndex` and :py:class:`CFTimeIndex`,
  - use microseconds in the conversion from timedelta objects to floats to avoid
    overflow errors.

Bug fixes
~~~~~~~~~
- Applying a user-defined function that adds new dimensions using :py:func:`apply_ufunc`
  and ``vectorize=True`` now works with ``dask > 2.0``. (:issue:`3574`, :pull:`3660`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix :py:meth:`~xarray.combine_by_coords` to allow for combining incomplete
  hypercubes of Datasets (:issue:`3648`).  By `Ian Bolliger
  <https://github.com/bolliger32>`_.
- Fix :py:func:`~xarray.combine_by_coords` when combining cftime coordinates
  which span long time intervals (:issue:`3535`).  By `Spencer Clark
  <https://github.com/spencerkclark>`_.
- Fix plotting with transposed 2D non-dimensional coordinates. (:issue:`3138`, :pull:`3441`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- :py:meth:`plot.FacetGrid.set_titles` can now replace existing row titles of a
  :py:class:`~xarray.plot.FacetGrid` plot. In addition :py:class:`~xarray.plot.FacetGrid` gained
  two new attributes: :py:attr:`~xarray.plot.FacetGrid.col_labels` and
  :py:attr:`~xarray.plot.FacetGrid.row_labels` contain :py:class:`matplotlib.text.Text` handles for both column and
  row labels. These can be used to manually change the labels.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix issue with Dask-backed datasets raising a ``KeyError`` on some computations involving :py:func:`map_blocks` (:pull:`3598`).
  By `Tom Augspurger <https://github.com/TomAugspurger>`_.
- Ensure :py:meth:`Dataset.quantile`, :py:meth:`DataArray.quantile` issue the correct error
  when ``q`` is out of bounds (:issue:`3634`) by `Mathias Hauser <https://github.com/mathause>`_.
- Fix regression in xarray 0.14.1 that prevented encoding times with certain
  ``dtype``, ``_FillValue``, and ``missing_value`` encodings (:issue:`3624`).
  By `Spencer Clark <https://github.com/spencerkclark>`_
- Raise an error when trying to use :py:meth:`Dataset.rename_dims` to
  rename to an existing name (:issue:`3438`, :pull:`3645`)
  By `Justus Magin <https://github.com/keewis>`_.
- :py:meth:`Dataset.rename`, :py:meth:`DataArray.rename` now check for conflicts with
  MultiIndex level names.
- :py:meth:`Dataset.merge` no longer fails when passed a :py:class:`DataArray` instead of a :py:class:`Dataset`.
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fix a regression in :py:meth:`Dataset.drop`: allow passing any
  iterable when dropping variables (:issue:`3552`, :pull:`3693`)
  By `Justus Magin <https://github.com/keewis>`_.
- Fixed errors emitted by ``mypy --strict`` in modules that import xarray.
  (:issue:`3695`) by `Guido Imperiale <https://github.com/crusaderky>`_.
- Allow plotting of binned coordinates on the y axis in :py:meth:`plot.line`
  and :py:meth:`plot.step` plots (:issue:`3571`,
  :pull:`3685`) by `Julien Seguinot <https://github.com/juseg>`_.
- setuptools is now marked as a dependency of xarray
  (:pull:`3628`) by `Richard Höchenberger <https://github.com/hoechenberger>`_.

Documentation
~~~~~~~~~~~~~
- Switch doc examples to use `nbsphinx <https://nbsphinx.readthedocs.io>`_ and replace
  ``sphinx_gallery`` scripts with Jupyter notebooks. (:pull:`3105`, :pull:`3106`, :pull:`3121`)
  By `Ryan Abernathey <https://github.com/rabernat>`_.
- Added :doc:`example notebook <examples/ROMS_ocean_model>` demonstrating use of xarray with
  Regional Ocean Modeling System (ROMS) ocean hydrodynamic model output. (:pull:`3116`)
  By `Robert Hetland <https://github.com/hetland>`_.
- Added :doc:`example notebook <examples/ERA5-GRIB-example>` demonstrating the visualization of
  ERA5 GRIB data. (:pull:`3199`)
  By `Zach Bruick <https://github.com/zbruick>`_ and
  `Stephan Siemen <https://github.com/StephanSiemen>`_.
- Added examples for :py:meth:`DataArray.quantile`, :py:meth:`Dataset.quantile` and
  ``GroupBy.quantile``. (:pull:`3576`)
  By `Justus Magin <https://github.com/keewis>`_.
- Add new :doc:`example notebook <examples/apply_ufunc_vectorize_1d>` example notebook demonstrating
  vectorization of a 1D function using :py:func:`apply_ufunc` , dask and numba.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Added example for :py:func:`~xarray.map_blocks`. (:pull:`3667`)
  By `Riley X. Brady <https://github.com/bradyrx>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Make sure dask names change when rechunking by different chunk sizes. Conversely, make sure they
  stay the same when rechunking by the same chunk size. (:issue:`3350`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- 2x to 5x speed boost (on small arrays) for :py:meth:`Dataset.isel`,
  :py:meth:`DataArray.isel`, and :py:meth:`DataArray.__getitem__` when indexing by int,
  slice, list of int, scalar ndarray, or 1-dimensional ndarray.
  (:pull:`3533`) by `Guido Imperiale <https://github.com/crusaderky>`_.
- Removed internal method ``Dataset._from_vars_and_coord_names``,
  which was dominated by ``Dataset._construct_direct``. (:pull:`3565`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Replaced versioneer with setuptools-scm. Moved contents of setup.py to setup.cfg.
  Removed pytest-runner from setup.py, as per deprecation notice on the pytest-runner
  project. (:pull:`3714`) by `Guido Imperiale <https://github.com/crusaderky>`_.
- Use of isort is now enforced by CI.
  (:pull:`3721`) by `Guido Imperiale <https://github.com/crusaderky>`_


.. _whats-new.0.14.1:

v0.14.1 (19 Nov 2019)
---------------------

Breaking changes
~~~~~~~~~~~~~~~~

- Broken compatibility with ``cftime < 1.0.3`` . By `Deepak Cherian <https://github.com/dcherian>`_.

  .. warning::

    cftime version 1.0.4 is broken
    (`cftime/126 <https://github.com/Unidata/cftime/issues/126>`_);
    please use version 1.0.4.2 instead.

- All leftover support for dates from non-standard calendars through ``netcdftime``, the
  module included in versions of netCDF4 prior to 1.4 that eventually became the
  `cftime <https://github.com/Unidata/cftime/>`_ package, has been removed in favor of relying solely on
  the standalone ``cftime`` package (:pull:`3450`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.

New Features
~~~~~~~~~~~~
- Added the ``sparse`` option to :py:meth:`~xarray.DataArray.unstack`,
  :py:meth:`~xarray.Dataset.unstack`, :py:meth:`~xarray.DataArray.reindex`,
  :py:meth:`~xarray.Dataset.reindex` (:issue:`3518`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Added the ``fill_value`` option to :py:meth:`DataArray.unstack` and
  :py:meth:`Dataset.unstack` (:issue:`3518`, :pull:`3541`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Added the ``max_gap`` kwarg to :py:meth:`~xarray.DataArray.interpolate_na` and
  :py:meth:`~xarray.Dataset.interpolate_na`. This controls the maximum size of the data
  gap that will be filled by interpolation. By `Deepak Cherian <https://github.com/dcherian>`_.
- Added :py:meth:`Dataset.drop_sel` & :py:meth:`DataArray.drop_sel` for dropping labels.
  :py:meth:`Dataset.drop_vars` & :py:meth:`DataArray.drop_vars` have been added for
  dropping variables (including coordinates). The existing :py:meth:`Dataset.drop` &
  :py:meth:`DataArray.drop` methods remain as a backward compatible
  option for dropping either labels or variables, but using the more specific methods is encouraged.
  (:pull:`3475`)
  By `Maximilian Roos <https://github.com/max-sixty>`_
- Added :py:meth:`Dataset.map` & ``GroupBy.map`` & ``Resample.map`` for
  mapping / applying a function over each item in the collection, reflecting the widely used
  and least surprising name for this operation.
  The existing ``apply`` methods remain for backward compatibility, though using the ``map``
  methods is encouraged.
  (:pull:`3459`)
  By `Maximilian Roos <https://github.com/max-sixty>`_
- :py:meth:`Dataset.transpose` and :py:meth:`DataArray.transpose` now support an ellipsis (``...``)
  to represent all 'other' dimensions. For example, to move one dimension to the front,
  use ``.transpose('x', ...)``. (:pull:`3421`)
  By `Maximilian Roos <https://github.com/max-sixty>`_
- Changed ``xr.ALL_DIMS`` to equal python's ``Ellipsis`` (``...``), and changed internal usages to use
  ``...`` directly. As before, you can use this to instruct a ``groupby`` operation
  to reduce over all dimensions. While we have no plans to remove ``xr.ALL_DIMS``, we suggest
  using ``...``. (:pull:`3418`)
  By `Maximilian Roos <https://github.com/max-sixty>`_
- :py:func:`xarray.dot`, and :py:meth:`DataArray.dot` now support the
  ``dims=...`` option to sum over the union of dimensions of all input arrays
  (:issue:`3423`) by `Mathias Hauser <https://github.com/mathause>`_.
- Added new ``Dataset._repr_html_`` and ``DataArray._repr_html_`` to improve
  representation of objects in Jupyter. By default this feature is turned off
  for now. Enable it with ``xarray.set_options(display_style="html")``.
  (:pull:`3425`) by `Benoit Bovy <https://github.com/benbovy>`_ and
  `Julia Signell <https://github.com/jsignell>`_.
- Implement `dask deterministic hashing
  <https://docs.dask.org/en/latest/custom-collections.html#deterministic-hashing>`_
  for xarray objects. Note that xarray objects with a dask.array backend already used
  deterministic hashing in previous releases; this change implements it when whole
  xarray objects are embedded in a dask graph, e.g. when :py:meth:`DataArray.map_blocks` is
  invoked. (:issue:`3378`, :pull:`3446`, :pull:`3515`)
  By `Deepak Cherian <https://github.com/dcherian>`_ and
  `Guido Imperiale <https://github.com/crusaderky>`_.
- Add the documented-but-missing :py:meth:`~core.groupby.DatasetGroupBy.quantile`.
- xarray now respects the ``DataArray.encoding["coordinates"]`` attribute when writing to disk.
  See :ref:`io.coordinates` for more. (:issue:`3351`, :pull:`3487`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Add the documented-but-missing :py:meth:`~core.groupby.DatasetGroupBy.quantile`.
  (:issue:`3525`, :pull:`3527`). By `Justus Magin <https://github.com/keewis>`_.

Bug fixes
~~~~~~~~~
- Ensure an index of type ``CFTimeIndex`` is not converted to a ``DatetimeIndex`` when
  calling :py:meth:`Dataset.rename`, :py:meth:`Dataset.rename_dims` and :py:meth:`Dataset.rename_vars`.
  By `Mathias Hauser <https://github.com/mathause>`_. (:issue:`3522`).
- Fix a bug in :py:meth:`DataArray.set_index` in case that an existing dimension becomes a level
  variable of MultiIndex. (:pull:`3520`). By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Harmonize ``_FillValue``, ``missing_value`` during encoding and decoding steps. (:pull:`3502`)
  By `Anderson Banihirwe <https://github.com/andersy005>`_.
- Fix regression introduced in v0.14.0 that would cause a crash if dask is installed
  but cloudpickle isn't (:issue:`3401`) by `Rhys Doyle <https://github.com/rdoyle45>`_
- Fix grouping over variables with NaNs. (:issue:`2383`, :pull:`3406`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Make alignment and concatenation significantly more efficient by using dask names to compare dask
  objects prior to comparing values after computation. This change makes it more convenient to carry
  around large non-dimensional coordinate variables backed by dask arrays. Existing workarounds involving
  ``reset_coords(drop=True)`` should now be unnecessary in most cases.
  (:issue:`3068`, :issue:`3311`, :issue:`3454`, :pull:`3453`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Add support for cftime>=1.0.4. By `Anderson Banihirwe <https://github.com/andersy005>`_.
- Rolling reduction operations no longer compute dask arrays by default. (:issue:`3161`).
  In addition, the ``allow_lazy`` kwarg to ``reduce`` is deprecated.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix ``GroupBy.reduce`` when reducing over multiple dimensions.
  (:issue:`3402`). By `Deepak Cherian <https://github.com/dcherian>`_
- Allow appending datetime and bool data variables to zarr stores.
  (:issue:`3480`). By `Akihiro Matsukawa <https://github.com/amatsukawa>`_.
- Add support for numpy >=1.18 (); bugfix mean() on datetime64 arrays on dask backend
  (:issue:`3409`, :pull:`3537`). By `Guido Imperiale <https://github.com/crusaderky>`_.
- Add support for pandas >=0.26 (:issue:`3440`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Add support for pseudonetcdf >=3.1 (:pull:`3485`).
  By `Barron Henderson <https://github.com/barronh>`_.

Documentation
~~~~~~~~~~~~~
- Fix leap year condition in `monthly means example <http://xarray.pydata.org/en/stable/examples/monthly-means.html>`_.
  By `Mickaël Lalande <https://github.com/mickaellalande>`_.
- Fix the documentation of :py:meth:`DataArray.resample` and
  :py:meth:`Dataset.resample`,  explicitly stating that a
  datetime-like dimension is required. (:pull:`3400`)
  By `Justus Magin <https://github.com/keewis>`_.
- Update the :ref:`terminology` page to address multidimensional coordinates. (:pull:`3410`)
  By `Jon Thielen <https://github.com/jthielen>`_.
- Fix the documentation of :py:meth:`Dataset.integrate` and
  :py:meth:`DataArray.integrate` and add an example to
  :py:meth:`Dataset.integrate`. (:pull:`3469`)
  By `Justus Magin <https://github.com/keewis>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Added integration tests against `pint <https://pint.readthedocs.io/>`_.
  (:pull:`3238`, :pull:`3447`, :pull:`3493`, :pull:`3508`)
  by `Justus Magin <https://github.com/keewis>`_.

  .. note::

    At the moment of writing, these tests *as well as the ability to use pint in general*
    require `a highly experimental version of pint
    <https://github.com/andrewgsavage/pint/pull/6>`_ (install with
    ``pip install git+https://github.com/andrewgsavage/pint.git@refs/pull/6/head)``.
    Even with it, interaction with non-numpy array libraries, e.g. dask or sparse, is broken.

- Use Python 3.6 idioms throughout the codebase. (:pull:`3419`)
  By `Maximilian Roos <https://github.com/max-sixty>`_

- Run basic CI tests on Python 3.8. (:pull:`3477`)
  By `Maximilian Roos <https://github.com/max-sixty>`_

- Enable type checking on default sentinel values (:pull:`3472`)
  By `Maximilian Roos <https://github.com/max-sixty>`_

- Add ``Variable._replace`` for simpler replacing of a subset of attributes (:pull:`3472`)
  By `Maximilian Roos <https://github.com/max-sixty>`_

.. _whats-new.0.14.0:

v0.14.0 (14 Oct 2019)
---------------------

Breaking changes
~~~~~~~~~~~~~~~~
- This release introduces a rolling policy for minimum dependency versions:
  :ref:`mindeps_policy`.

  Several minimum versions have been increased:

  ============ ================== ====
  Package      Old                New
  ============ ================== ====
  Python       3.5.3              3.6
  numpy        1.12               1.14
  pandas       0.19.2             0.24
  dask         0.16 (tested: 2.4) 1.2
  bottleneck   1.1 (tested: 1.2)  1.2
  matplotlib   1.5 (tested: 3.1)  3.1
  ============ ================== ====

  Obsolete patch versions (x.y.Z) are not tested anymore.
  The oldest supported versions of all optional dependencies are now covered by
  automated tests (before, only the very latest versions were tested).

  (:issue:`3222`, :issue:`3293`, :issue:`3340`, :issue:`3346`, :issue:`3358`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.

- Dropped the ``drop=False`` optional parameter from :py:meth:`Variable.isel`.
  It was unused and doesn't make sense for a Variable. (:pull:`3375`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.

- Remove internal usage of :py:class:`collections.OrderedDict`. After dropping support for
  Python <=3.5, most uses of ``OrderedDict`` in Xarray were no longer necessary. We
  have removed the internal use of the ``OrderedDict`` in favor of Python's builtin
  ``dict`` object which is now ordered itself. This change will be most obvious when
  interacting with the ``attrs`` property on Dataset and DataArray objects.
  (:issue:`3380`, :pull:`3389`). By `Joe Hamman <https://github.com/jhamman>`_.

New functions/methods
~~~~~~~~~~~~~~~~~~~~~

- Added :py:func:`~xarray.map_blocks`, modeled after :py:func:`dask.array.map_blocks`.
  Also added :py:meth:`Dataset.unify_chunks`, :py:meth:`DataArray.unify_chunks` and
  :py:meth:`testing.assert_chunks_equal`. (:pull:`3276`).
  By `Deepak Cherian <https://github.com/dcherian>`_ and
  `Guido Imperiale <https://github.com/crusaderky>`_.

Enhancements
~~~~~~~~~~~~

- ``core.groupby.GroupBy`` enhancements. By `Deepak Cherian <https://github.com/dcherian>`_.

  - Added a repr (:pull:`3344`). Example::

      >>> da.groupby("time.season")
      DataArrayGroupBy, grouped over 'season'
      4 groups with labels 'DJF', 'JJA', 'MAM', 'SON'

  - Added a ``GroupBy.dims`` property that mirrors the dimensions
    of each group (:issue:`3344`).

- Speed up :py:meth:`Dataset.isel` up to 33% and :py:meth:`DataArray.isel` up to 25% for small
  arrays (:issue:`2799`, :pull:`3375`). By
  `Guido Imperiale <https://github.com/crusaderky>`_.

Bug fixes
~~~~~~~~~
- Reintroduce support for :mod:`weakref` (broken in v0.13.0). Support has been
  reinstated for :py:class:`~xarray.DataArray` and :py:class:`~xarray.Dataset` objects only.
  Internal xarray objects remain unaddressable by weakref in order to save memory
  (:issue:`3317`). By `Guido Imperiale <https://github.com/crusaderky>`_.
- Line plots with the ``x`` or ``y`` argument set to a 1D non-dimensional coord
  now plot the correct data for 2D DataArrays
  (:issue:`3334`). By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Make :py:func:`~xarray.concat` more robust when merging variables present in some datasets but
  not others (:issue:`508`). By `Deepak Cherian <https://github.com/dcherian>`_.
- The default behaviour of reducing across all dimensions for
  :py:class:`~xarray.core.groupby.DataArrayGroupBy` objects has now been properly removed
  as was done for :py:class:`~xarray.core.groupby.DatasetGroupBy` in 0.13.0 (:issue:`3337`).
  Use ``xarray.ALL_DIMS`` if you need to replicate previous behaviour.
  Also raise nicer error message when no groups are created (:issue:`1764`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix error in concatenating unlabeled dimensions (:pull:`3362`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Warn if the ``dim`` kwarg is passed to rolling operations. This is redundant since a dimension is
  specified when the :py:class:`~core.rolling.DatasetRolling` or :py:class:`~core.rolling.DataArrayRolling` object is created.
  (:pull:`3362`). By `Deepak Cherian <https://github.com/dcherian>`_.

Documentation
~~~~~~~~~~~~~

- Created a glossary of important xarray terms (:issue:`2410`, :pull:`3352`).
  By `Gregory Gundersen <https://github.com/gwgundersen>`_.
- Created a "How do I..." section (:ref:`howdoi`) for solutions to common questions. (:pull:`3357`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Add examples for :py:meth:`Dataset.swap_dims` and :py:meth:`DataArray.swap_dims`
  (pull:`3331`, pull:`3331`). By `Justus Magin <https://github.com/keewis>`_.
- Add examples for :py:meth:`align`, :py:meth:`merge`, :py:meth:`combine_by_coords`,
  :py:meth:`full_like`, :py:meth:`zeros_like`, :py:meth:`ones_like`, :py:meth:`Dataset.pipe`,
  :py:meth:`Dataset.assign`, :py:meth:`Dataset.reindex`, :py:meth:`Dataset.fillna` (:pull:`3328`).
  By `Anderson Banihirwe <https://github.com/andersy005>`_.
- Fixed documentation to clean up an unwanted file created in ``ipython`` example
  (:pull:`3353`). By `Gregory Gundersen <https://github.com/gwgundersen>`_.

.. _whats-new.0.13.0:

v0.13.0 (17 Sep 2019)
---------------------

This release includes many exciting changes: wrapping of
`NEP18 <https://www.numpy.org/neps/nep-0018-array-function-protocol.html>`_ compliant
numpy-like arrays; new :py:meth:`~Dataset.plot.scatter` plotting method that can scatter
two ``DataArrays`` in a ``Dataset`` against each other; support for converting pandas
DataFrames to xarray objects that wrap ``pydata/sparse``; and more!

Breaking changes
~~~~~~~~~~~~~~~~

- This release increases the minimum required Python version from 3.5.0 to 3.5.3
  (:issue:`3089`). By `Guido Imperiale <https://github.com/crusaderky>`_.
- The ``isel_points`` and ``sel_points`` methods are removed, having been deprecated
  since v0.10.0. These are redundant with the ``isel`` / ``sel`` methods.
  See :ref:`vectorized_indexing` for the details
  By `Maximilian Roos <https://github.com/max-sixty>`_
- The ``inplace`` kwarg for public methods now raises an error, having been deprecated
  since v0.11.0.
  By `Maximilian Roos <https://github.com/max-sixty>`_
- :py:func:`~xarray.concat` now requires the ``dim`` argument. Its ``indexers``, ``mode``
  and ``concat_over`` kwargs have now been removed.
  By `Deepak Cherian <https://github.com/dcherian>`_
- Passing a list of colors in ``cmap`` will now raise an error, having been deprecated since
  v0.6.1.
- Most xarray objects now define ``__slots__``. This reduces overall RAM usage by ~22%
  (not counting the underlying numpy buffers); on CPython 3.7/x64, a trivial DataArray
  has gone down from 1.9kB to 1.5kB.

  Caveats:

  - Pickle streams produced by older versions of xarray can't be loaded using this
    release, and vice versa.
  - Any user code that was accessing the ``__dict__`` attribute of
    xarray objects will break. The best practice to attach custom metadata to xarray
    objects is to use the ``attrs`` dictionary.
  - Any user code that defines custom subclasses of xarray classes must now explicitly
    define ``__slots__`` itself. Subclasses that don't add any attributes must state so
    by defining ``__slots__ = ()`` right after the class header.
    Omitting ``__slots__`` will now cause a ``FutureWarning`` to be logged, and will raise an
    error in a later release.

  (:issue:`3250`) by `Guido Imperiale <https://github.com/crusaderky>`_.
- The default dimension for :py:meth:`Dataset.groupby`, :py:meth:`Dataset.resample`,
  :py:meth:`DataArray.groupby` and :py:meth:`DataArray.resample` reductions is now the
  grouping or resampling dimension.
- :py:meth:`DataArray.to_dataset` requires ``name`` to be passed as a kwarg (previously ambiguous
  positional arguments were deprecated)
- Reindexing with variables of a different dimension now raise an error (previously deprecated)
- ``xarray.broadcast_array`` is removed (previously deprecated in favor of
  :py:func:`~xarray.broadcast`)
- ``Variable.expand_dims`` is removed (previously deprecated in favor of
  :py:meth:`Variable.set_dims`)

New functions/methods
~~~~~~~~~~~~~~~~~~~~~

- xarray can now wrap around any
  `NEP18 <https://www.numpy.org/neps/nep-0018-array-function-protocol.html>`_ compliant
  numpy-like library (important: read notes about ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION`` in
  the above link). Added explicit test coverage for
  `sparse <https://github.com/pydata/sparse>`_. (:issue:`3117`, :issue:`3202`).
  This requires `sparse>=0.8.0`. By `Nezar Abdennur <https://github.com/nvictus>`_
  and `Guido Imperiale <https://github.com/crusaderky>`_.

- :py:meth:`~Dataset.from_dataframe` and :py:meth:`~DataArray.from_series` now
  support ``sparse=True`` for converting pandas objects into xarray objects
  wrapping sparse arrays. This is particularly useful with sparsely populated
  hierarchical indexes. (:issue:`3206`)
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- The xarray package is now discoverable by mypy (although typing hints coverage is not
  complete yet). mypy type checking is now enforced by CI. Libraries that depend on
  xarray and use mypy can now remove from their setup.cfg the lines::

    [mypy-xarray]
    ignore_missing_imports = True

  (:issue:`2877`, :issue:`3088`, :issue:`3090`, :issue:`3112`, :issue:`3117`,
  :issue:`3207`)
  By `Guido Imperiale <https://github.com/crusaderky>`_
  and `Maximilian Roos <https://github.com/max-sixty>`_.

- Added :py:meth:`DataArray.broadcast_like` and :py:meth:`Dataset.broadcast_like`.
  By `Deepak Cherian <https://github.com/dcherian>`_ and `David Mertz
  <https://github.com/DavidMertz>`_.

- Dataset plotting API for visualizing dependencies between two DataArrays!
  Currently only :py:meth:`Dataset.plot.scatter` is implemented.
  By `Yohai Bar Sinai <https://github.com/yohai>`_ and `Deepak Cherian <https://github.com/dcherian>`_

- Added :py:meth:`DataArray.head`, :py:meth:`DataArray.tail` and :py:meth:`DataArray.thin`;
  as well as :py:meth:`Dataset.head`, :py:meth:`Dataset.tail` and :py:meth:`Dataset.thin` methods.
  (:issue:`319`) By `Gerardo Rivera <https://github.com/dangomelon>`_.

Enhancements
~~~~~~~~~~~~

- Multiple enhancements to :py:func:`~xarray.concat` and :py:func:`~xarray.open_mfdataset`.
  By `Deepak Cherian <https://github.com/dcherian>`_

  - Added ``compat='override'``. When merging, this option picks the variable from the first dataset
    and skips all comparisons.

  - Added ``join='override'``. When aligning, this only checks that index sizes are equal among objects
    and skips checking indexes for equality.

  - :py:func:`~xarray.concat` and :py:func:`~xarray.open_mfdataset` now support the ``join`` kwarg.
    It is passed down to :py:func:`~xarray.align`.

  - :py:func:`~xarray.concat` now calls :py:func:`~xarray.merge` on variables that are not concatenated
    (i.e. variables without ``concat_dim`` when ``data_vars`` or ``coords`` are ``"minimal"``).
    :py:func:`~xarray.concat` passes its new ``compat`` kwarg down to :py:func:`~xarray.merge`.
    (:issue:`2064`)

  Users can avoid a common bottleneck when using :py:func:`~xarray.open_mfdataset` on a large number of
  files with variables that are known to be aligned and some of which need not be concatenated.
  Slow equality comparisons can now be avoided, for e.g.::

    data = xr.open_mfdataset(files, concat_dim='time', data_vars='minimal',
                             coords='minimal', compat='override', join='override')

- In :py:meth:`~xarray.Dataset.to_zarr`, passing ``mode`` is not mandatory if
  ``append_dim`` is set, as it will automatically be set to ``'a'`` internally.
  By `David Brochart <https://github.com/davidbrochart>`_.

- Added the ability to initialize an empty or full DataArray
  with a single value. (:issue:`277`)
  By `Gerardo Rivera <https://github.com/dangomelon>`_.

- :py:func:`~xarray.Dataset.to_netcdf()` now supports the ``invalid_netcdf`` kwarg when used
  with ``engine="h5netcdf"``. It is passed to ``h5netcdf.File``.
  By `Ulrich Herter <https://github.com/ulijh>`_.

- ``xarray.Dataset.drop`` now supports keyword arguments; dropping index
  labels by using both ``dim`` and ``labels`` or using a
  :py:class:`~core.coordinates.DataArrayCoordinates` object are deprecated (:issue:`2910`).
  By `Gregory Gundersen <https://github.com/gwgundersen>`_.

- Added examples of :py:meth:`Dataset.set_index` and
  :py:meth:`DataArray.set_index`, as well are more specific error messages
  when the user passes invalid arguments (:issue:`3176`).
  By `Gregory Gundersen <https://github.com/gwgundersen>`_.

- :py:meth:`Dataset.filter_by_attrs` now filters the coordinates as well as the variables.
  By `Spencer Jones <https://github.com/cspencerjones>`_.

Bug fixes
~~~~~~~~~

- Improve "missing dimensions" error message for :py:func:`~xarray.apply_ufunc`
  (:issue:`2078`).
  By `Rick Russotto <https://github.com/rdrussotto>`_.
- :py:meth:`~xarray.DataArray.assign_coords` now supports dictionary arguments
  (:issue:`3231`).
  By `Gregory Gundersen <https://github.com/gwgundersen>`_.
- Fix regression introduced in v0.12.2 where ``copy(deep=True)`` would convert
  unicode indices to dtype=object (:issue:`3094`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.
- Improved error handling and documentation for `.expand_dims()`
  read-only view.
- Fix tests for big-endian systems (:issue:`3125`).
  By `Graham Inggs <https://github.com/ginggs>`_.
- XFAIL several tests which are expected to fail on ARM systems
  due to a ``datetime`` issue in NumPy (:issue:`2334`).
  By `Graham Inggs <https://github.com/ginggs>`_.
- Fix KeyError that arises when using .sel method with float values
  different from coords float type (:issue:`3137`).
  By `Hasan Ahmad <https://github.com/HasanAhmadQ7>`_.
- Fixed bug in ``combine_by_coords()`` causing a `ValueError` if the input had
  an unused dimension with coordinates which were not monotonic (:issue:`3150`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fixed crash when applying ``distributed.Client.compute()`` to a DataArray
  (:issue:`3171`). By `Guido Imperiale <https://github.com/crusaderky>`_.
- Better error message when using groupby on an empty DataArray (:issue:`3037`).
  By `Hasan Ahmad <https://github.com/HasanAhmadQ7>`_.
- Fix error that arises when using open_mfdataset on a series of netcdf files
  having differing values for a variable attribute of type list. (:issue:`3034`)
  By `Hasan Ahmad <https://github.com/HasanAhmadQ7>`_.
- Prevent :py:meth:`~xarray.DataArray.argmax` and :py:meth:`~xarray.DataArray.argmin` from calling
  dask compute (:issue:`3237`). By `Ulrich Herter <https://github.com/ulijh>`_.
- Plots in 2 dimensions (pcolormesh, contour) now allow to specify levels as numpy
  array (:issue:`3284`). By `Mathias Hauser <https://github.com/mathause>`_.
- Fixed bug in :meth:`DataArray.quantile` failing to keep attributes when
  `keep_attrs` was True (:issue:`3304`). By David Huard `<https://github.com/huard>`_.

Documentation
~~~~~~~~~~~~~

- Created a `PR checklist <https://xarray.pydata.org/en/stable/contributing.html/contributing.html#pr-checklist>`_ as a quick reference for tasks before creating a new PR
  or pushing new commits.
  By `Gregory Gundersen <https://github.com/gwgundersen>`_.

- Fixed documentation to clean up unwanted files created in ``ipython`` examples
  (:issue:`3227`).
  By `Gregory Gundersen <https://github.com/gwgundersen>`_.

.. _whats-new.0.12.3:

v0.12.3 (10 July 2019)
----------------------

New functions/methods
~~~~~~~~~~~~~~~~~~~~~

- New methods :py:meth:`Dataset.to_stacked_array` and
  :py:meth:`DataArray.to_unstacked_dataset` for reshaping Datasets of variables
  with different dimensions
  (:issue:`1317`).
  This is useful for feeding data from xarray into machine learning models,
  as described in :ref:`reshape.stacking_different`.
  By `Noah Brenowitz <https://github.com/nbren12>`_.

Enhancements
~~~~~~~~~~~~

- Support for renaming ``Dataset`` variables and dimensions independently
  with :py:meth:`~Dataset.rename_vars` and :py:meth:`~Dataset.rename_dims`
  (:issue:`3026`).
  By `Julia Kent <https://github.com/jukent>`_.

- Add ``scales``, ``offsets``, ``units`` and ``descriptions``
  attributes to :py:class:`~xarray.DataArray` returned by
  :py:func:`~xarray.open_rasterio`. (:issue:`3013`)
  By `Erle Carrara <https://github.com/ecarrara>`_.

Bug fixes
~~~~~~~~~

- Resolved deprecation warnings from newer versions of matplotlib and dask.
- Compatibility fixes for the upcoming pandas 0.25 and NumPy 1.17 releases.
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Fix summaries for multiindex coordinates (:issue:`3079`).
  By `Jonas Hörsch <https://github.com/coroa>`_.
- Fix HDF5 error that could arise when reading multiple groups from a file at
  once (:issue:`2954`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

.. _whats-new.0.12.2:

v0.12.2 (29 June 2019)
----------------------

New functions/methods
~~~~~~~~~~~~~~~~~~~~~

- Two new functions, :py:func:`~xarray.combine_nested` and
  :py:func:`~xarray.combine_by_coords`, allow for combining datasets along any
  number of dimensions, instead of the one-dimensional list of datasets
  supported by :py:func:`~xarray.concat`.

  The new ``combine_nested`` will accept the datasets as a nested
  list-of-lists, and combine by applying a series of concat and merge
  operations. The new ``combine_by_coords`` instead uses the dimension
  coordinates of datasets to order them.

  :py:func:`~xarray.open_mfdataset` can use either ``combine_nested`` or
  ``combine_by_coords`` to combine datasets along multiple dimensions, by
  specifying the argument ``combine='nested'`` or ``combine='by_coords'``.

  The older function :py:func:`~xarray.auto_combine` has been deprecated,
  because its functionality has been subsumed by the new functions.
  To avoid FutureWarnings switch to using ``combine_nested`` or
  ``combine_by_coords``, (or set the ``combine`` argument in
  ``open_mfdataset``). (:issue:`2159`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

- :py:meth:`~xarray.DataArray.rolling_exp` and
  :py:meth:`~xarray.Dataset.rolling_exp` added, similar to pandas'
  ``pd.DataFrame.ewm`` method. Calling ``.mean`` on the resulting object
  will return an exponentially weighted moving average.
  By `Maximilian Roos <https://github.com/max-sixty>`_.

- New :py:func:`DataArray.str <core.accessor_str.StringAccessor>` for string
  related manipulations, based on ``pandas.Series.str``.
  By `0x0L <https://github.com/0x0L>`_.

- Added ``strftime`` method to ``.dt`` accessor, making it simpler to hand a
  datetime ``DataArray`` to other code expecting formatted dates and times.
  (:issue:`2090`). :py:meth:`~xarray.CFTimeIndex.strftime` is also now
  available on :py:class:`CFTimeIndex`.
  By `Alan Brammer <https://github.com/abrammer>`_ and
  `Ryan May <https://github.com/dopplershift>`_.

- ``GroupBy.quantile`` is now a method of ``GroupBy``
  objects  (:issue:`3018`).
  By `David Huard <https://github.com/huard>`_.

- Argument and return types are added to most methods on ``DataArray`` and
  ``Dataset``, allowing static type checking both within xarray and external
  libraries. Type checking with `mypy <http://mypy-lang.org/>`_ is enabled in
  CI (though not required yet).
  By `Guido Imperiale <https://github.com/crusaderky>`_
  and `Maximilian Roos <https://github.com/max-sixty>`_.

Enhancements to existing functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Add ``keepdims`` argument for reduce operations (:issue:`2170`)
  By `Scott Wales <https://github.com/ScottWales>`_.
- Enable ``@`` operator for DataArray. This is equivalent to :py:meth:`DataArray.dot`
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Add ``fill_value`` argument for reindex, align, and merge operations
  to enable custom fill values. (:issue:`2876`)
  By `Zach Griffith <https://github.com/zdgriffith>`_.
- :py:meth:`DataArray.transpose` now accepts a keyword argument
  ``transpose_coords`` which enables transposition of coordinates in the
  same way as :py:meth:`Dataset.transpose`. :py:meth:`DataArray.groupby`
  :py:meth:`DataArray.groupby_bins`, and :py:meth:`DataArray.resample` now
  accept a keyword argument ``restore_coord_dims`` which keeps the order
  of the dimensions of multi-dimensional coordinates intact (:issue:`1856`).
  By `Peter Hausamann <https://github.com/phausamann>`_.
- Clean up Python 2 compatibility in code (:issue:`2950`)
  By `Guido Imperiale <https://github.com/crusaderky>`_.
- Better warning message when supplying invalid objects to ``xr.merge``
  (:issue:`2948`).  By `Mathias Hauser <https://github.com/mathause>`_.
- Add ``errors`` keyword argument to ``Dataset.drop`` and :py:meth:`Dataset.drop_dims`
  that allows ignoring errors if a passed label or dimension is not in the dataset
  (:issue:`2994`).
  By `Andrew Ross <https://github.com/andrew-c-ross>`_.

IO related enhancements
~~~~~~~~~~~~~~~~~~~~~~~

- Implement :py:func:`~xarray.load_dataset` and
  :py:func:`~xarray.load_dataarray` as alternatives to
  :py:func:`~xarray.open_dataset` and :py:func:`~xarray.open_dataarray` to
  open, load into memory, and close files, returning the Dataset or DataArray.
  These functions are helpful for avoiding file-lock errors when trying to
  write to files opened using ``open_dataset()`` or ``open_dataarray()``.
  (:issue:`2887`)
  By `Dan Nowacki <https://github.com/dnowacki-usgs>`_.
- It is now possible to extend existing :ref:`io.zarr` datasets, by using
  ``mode='a'`` and the new ``append_dim`` argument in
  :py:meth:`~xarray.Dataset.to_zarr`.
  By `Jendrik Jördening <https://github.com/jendrikjoe>`_,
  `David Brochart <https://github.com/davidbrochart>`_,
  `Ryan Abernathey <https://github.com/rabernat>`_ and
  `Shikhar Goenka <https://github.com/shikharsg>`_.
- ``xr.open_zarr`` now accepts manually specified chunks with the ``chunks=``
  parameter. ``auto_chunk=True`` is equivalent to ``chunks='auto'`` for
  backwards compatibility. The ``overwrite_encoded_chunks`` parameter is
  added to remove the original zarr chunk encoding.
  By `Lily Wang <https://github.com/lilyminium>`_.
- netCDF chunksizes are now only dropped when original_shape is different,
  not when it isn't found. (:issue:`2207`)
  By `Karel van de Plassche <https://github.com/Karel-van-de-Plassche>`_.
- Character arrays' character dimension name decoding and encoding handled by
  ``var.encoding['char_dim_name']`` (:issue:`2895`)
  By `James McCreight <https://github.com/jmccreight>`_.
- open_rasterio() now supports rasterio.vrt.WarpedVRT with custom transform,
  width and height (:issue:`2864`).
  By `Julien Michel <https://github.com/jmichel-otb>`_.

Bug fixes
~~~~~~~~~

- Rolling operations on xarray objects containing dask arrays could silently
  compute the incorrect result or use large amounts of memory (:issue:`2940`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Don't set encoding attributes on bounds variables when writing to netCDF.
  (:issue:`2921`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- NetCDF4 output: variables with unlimited dimensions must be chunked (not
  contiguous) on output. (:issue:`1849`)
  By `James McCreight <https://github.com/jmccreight>`_.
- indexing with an empty list creates an object with zero-length axis (:issue:`2882`)
  By `Mayeul d'Avezac <https://github.com/mdavezac>`_.
- Return correct count for scalar datetime64 arrays (:issue:`2770`)
  By `Dan Nowacki <https://github.com/dnowacki-usgs>`_.
- Fixed max, min exception when applied to a multiIndex (:issue:`2923`)
  By `Ian Castleden <https://github.com/arabidopsis>`_
- A deep copy deep-copies the coords (:issue:`1463`)
  By `Martin Pletcher <https://github.com/pletchm>`_.
- Increased support for `missing_value` (:issue:`2871`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Removed usages of `pytest.config`, which is deprecated (:issue:`2988`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Fixed performance issues with cftime installed (:issue:`3000`)
  By `0x0L <https://github.com/0x0L>`_.
- Replace incorrect usages of `message` in pytest assertions
  with `match` (:issue:`3011`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Add explicit pytest markers, now required by pytest
  (:issue:`3032`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Test suite fixes for newer versions of pytest (:issue:`3011`, :issue:`3032`).
  By `Maximilian Roos <https://github.com/max-sixty>`_
  and `Stephan Hoyer <https://github.com/shoyer>`_.

.. _whats-new.0.12.1:

v0.12.1 (4 April 2019)
----------------------

Enhancements
~~~~~~~~~~~~

- Allow ``expand_dims`` method to support inserting/broadcasting dimensions
  with size > 1. (:issue:`2710`)
  By `Martin Pletcher <https://github.com/pletchm>`_.

Bug fixes
~~~~~~~~~

- Dataset.copy(deep=True) now creates a deep copy of the attrs (:issue:`2835`).
  By `Andras Gefferth <https://github.com/kefirbandi>`_.
- Fix incorrect ``indexes`` resulting from various ``Dataset`` operations
  (e.g., ``swap_dims``, ``isel``, ``reindex``, ``[]``) (:issue:`2842`,
  :issue:`2856`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

.. _whats-new.0.12.0:

v0.12.0 (15 March 2019)
-----------------------

Highlights include:

- Removed support for Python 2. This is the first version of xarray that is
  Python 3 only!
- New :py:meth:`~xarray.DataArray.coarsen` and
  :py:meth:`~xarray.DataArray.integrate` methods. See :ref:`comput.coarsen`
  and :ref:`compute.using_coordinates` for details.
- Many improvements to cftime support. See below for details.

Deprecations
~~~~~~~~~~~~

- The ``compat`` argument to ``Dataset`` and the ``encoding`` argument to
  ``DataArray`` are deprecated and will be removed in a future release.
  (:issue:`1188`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.

cftime related enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Resampling of standard and non-standard calendars indexed by
  :py:class:`~xarray.CFTimeIndex` is now possible. (:issue:`2191`).
  By `Jwen Fai Low <https://github.com/jwenfai>`_ and
  `Spencer Clark <https://github.com/spencerkclark>`_.

- Taking the mean of arrays of :py:class:`cftime.datetime` objects, and
  by extension, use of :py:meth:`~xarray.DataArray.coarsen` with
  :py:class:`cftime.datetime` coordinates is now possible. By `Spencer Clark
  <https://github.com/spencerkclark>`_.

- Internal plotting now supports ``cftime.datetime`` objects as time series.
  (:issue:`2164`)
  By `Julius Busecke <https://github.com/jbusecke>`_ and
  `Spencer Clark <https://github.com/spencerkclark>`_.

- :py:meth:`~xarray.cftime_range` now supports QuarterBegin and QuarterEnd offsets (:issue:`2663`).
  By `Jwen Fai Low <https://github.com/jwenfai>`_

- :py:meth:`~xarray.open_dataset` now accepts a ``use_cftime`` argument, which
  can be used to require that ``cftime.datetime`` objects are always used, or
  never used when decoding dates encoded with a standard calendar.  This can be
  used to ensure consistent date types are returned when using
  :py:meth:`~xarray.open_mfdataset` (:issue:`1263`) and/or to silence
  serialization warnings raised if dates from a standard calendar are found to
  be outside the :py:class:`pandas.Timestamp`-valid range (:issue:`2754`).  By
  `Spencer Clark <https://github.com/spencerkclark>`_.

- :py:meth:`pandas.Series.dropna` is now supported for a
  :py:class:`pandas.Series` indexed by a :py:class:`~xarray.CFTimeIndex`
  (:issue:`2688`). By `Spencer Clark <https://github.com/spencerkclark>`_.

Other enhancements
~~~~~~~~~~~~~~~~~~

- Added ability to open netcdf4/hdf5 file-like objects with ``open_dataset``.
  Requires (h5netcdf>0.7 and h5py>2.9.0). (:issue:`2781`)
  By `Scott Henderson <https://github.com/scottyhq>`_
- Add ``data=False`` option to ``to_dict()`` methods. (:issue:`2656`)
  By `Ryan Abernathey <https://github.com/rabernat>`_
- :py:meth:`DataArray.coarsen` and
  :py:meth:`Dataset.coarsen` are newly added.
  See :ref:`comput.coarsen` for details.
  (:issue:`2525`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Upsampling an array via interpolation with resample is now dask-compatible,
  as long as the array is not chunked along the resampling dimension.
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- :py:func:`xarray.testing.assert_equal` and
  :py:func:`xarray.testing.assert_identical` now provide a more detailed
  report showing what exactly differs between the two objects (dimensions /
  coordinates / variables / attributes)  (:issue:`1507`).
  By `Benoit Bovy <https://github.com/benbovy>`_.
- Add ``tolerance`` option to ``resample()`` methods ``bfill``, ``pad``,
  ``nearest``. (:issue:`2695`)
  By `Hauke Schulz <https://github.com/observingClouds>`_.
- :py:meth:`DataArray.integrate` and
  :py:meth:`Dataset.integrate` are newly added.
  See :ref:`compute.using_coordinates` for the detail.
  (:issue:`1332`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Added :py:meth:`~xarray.Dataset.drop_dims` (:issue:`1949`).
  By `Kevin Squire <https://github.com/kmsquire>`_.

Bug fixes
~~~~~~~~~

- Silenced warnings that appear when using pandas 0.24.
  By `Stephan Hoyer <https://github.com/shoyer>`_
- Interpolating via resample now internally specifies ``bounds_error=False``
  as an argument to ``scipy.interpolate.interp1d``, allowing for interpolation
  from higher frequencies to lower frequencies.  Datapoints outside the bounds
  of the original time coordinate are now filled with NaN (:issue:`2197`). By
  `Spencer Clark <https://github.com/spencerkclark>`_.
- Line plots with the ``x`` argument set to a non-dimensional coord now plot
  the correct data for 1D DataArrays.
  (:issue:`2725`). By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Subtracting a scalar ``cftime.datetime`` object from a
  :py:class:`CFTimeIndex` now results in a :py:class:`pandas.TimedeltaIndex`
  instead of raising a ``TypeError`` (:issue:`2671`).  By `Spencer Clark
  <https://github.com/spencerkclark>`_.
- backend_kwargs are no longer ignored when using open_dataset with pynio engine
  (:issue:'2380')
  By `Jonathan Joyce <https://github.com/jonmjoyce>`_.
- Fix ``open_rasterio`` creating a WKT CRS instead of PROJ.4 with
  ``rasterio`` 1.0.14+ (:issue:`2715`).
  By `David Hoese <https://github.com/djhoese>`_.
- Masking data arrays with :py:meth:`xarray.DataArray.where` now returns an
  array with the name of the original masked array (:issue:`2748` and :issue:`2457`).
  By `Yohai Bar-Sinai <https://github.com/yohai>`_.
- Fixed error when trying to reduce a DataArray using a function which does not
  require an axis argument. (:issue:`2768`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Concatenating a sequence of :py:class:`~xarray.DataArray` with varying names
  sets the name of the output array to ``None``, instead of the name of the
  first input array. If the names are the same it sets the name to that,
  instead to the name of the first DataArray in the list as it did before.
  (:issue:`2775`). By `Tom Nicholas <https://github.com/TomNicholas>`_.

- Per the `CF conventions section on calendars
  <http://cfconventions.org/cf-conventions/cf-conventions.html#calendar>`_,
  specifying ``'standard'`` as the calendar type in
  :py:meth:`~xarray.cftime_range` now correctly refers to the ``'gregorian'``
  calendar instead of the ``'proleptic_gregorian'`` calendar (:issue:`2761`).

.. _whats-new.0.11.3:

v0.11.3 (26 January 2019)
-------------------------

Bug fixes
~~~~~~~~~

- Saving files with times encoded with reference dates with timezones
  (e.g. '2000-01-01T00:00:00-05:00') no longer raises an error
  (:issue:`2649`).  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Fixed performance regression with ``open_mfdataset`` (:issue:`2662`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fixed supplying an explicit dimension in the ``concat_dim`` argument to
  to ``open_mfdataset`` (:issue:`2647`).
  By `Ben Root <https://github.com/WeatherGod>`_.

.. _whats-new.0.11.2:

v0.11.2 (2 January 2019)
------------------------

Removes inadvertently introduced setup dependency on pytest-runner
(:issue:`2641`). Otherwise, this release is exactly equivalent to 0.11.1.

.. warning::

  This is the last xarray release that will support Python 2.7. Future releases
  will be Python 3 only, but older versions of xarray will always be available
  for Python 2.7 users. For the more details, see:

  - `Xarray Github issue discussing dropping Python 2 <https://github.com/pydata/xarray/issues/1829>`__
  - `Python 3 Statement <http://www.python3statement.org/>`__
  - `Tips on porting to Python 3 <https://docs.python.org/3/howto/pyporting.html>`__

.. _whats-new.0.11.1:

v0.11.1 (29 December 2018)
--------------------------

This minor release includes a number of enhancements and bug fixes, and two
(slightly) breaking changes.

Breaking changes
~~~~~~~~~~~~~~~~

- Minimum rasterio version increased from 0.36 to 1.0 (for ``open_rasterio``)
- Time bounds variables are now also decoded according to CF conventions
  (:issue:`2565`). The previous behavior was to decode them only if they
  had specific time attributes, now these attributes are copied
  automatically from the corresponding time coordinate. This might
  break downstream code that was relying on these variables to be
  brake downstream code that was relying on these variables to be
  not decoded.
  By `Fabien Maussion <https://github.com/fmaussion>`_.

Enhancements
~~~~~~~~~~~~

- Ability to read and write consolidated metadata in zarr stores (:issue:`2558`).
  By `Ryan Abernathey <https://github.com/rabernat>`_.
- :py:class:`CFTimeIndex` uses slicing for string indexing when possible (like
  :py:class:`pandas.DatetimeIndex`), which avoids unnecessary copies.
  By `Stephan Hoyer <https://github.com/shoyer>`_
- Enable passing ``rasterio.io.DatasetReader`` or ``rasterio.vrt.WarpedVRT`` to
  ``open_rasterio`` instead of file path string. Allows for in-memory
  reprojection, see  (:issue:`2588`).
  By `Scott Henderson <https://github.com/scottyhq>`_.
- Like :py:class:`pandas.DatetimeIndex`, :py:class:`CFTimeIndex` now supports
  "dayofyear" and "dayofweek" accessors (:issue:`2597`).  Note this requires a
  version of cftime greater than 1.0.2.  By `Spencer Clark
  <https://github.com/spencerkclark>`_.
- The option ``'warn_for_unclosed_files'`` (False by default) has been added to
  allow users to enable a warning when files opened by xarray are deallocated
  but were not explicitly closed. This is mostly useful for debugging; we
  recommend enabling it in your test suites if you use xarray for IO.
  By `Stephan Hoyer <https://github.com/shoyer>`_
- Support Dask ``HighLevelGraphs`` by `Matthew Rocklin <https://github.com/mrocklin>`_.
- :py:meth:`DataArray.resample` and :py:meth:`Dataset.resample` now supports the
  ``loffset`` kwarg just like Pandas.
  By `Deepak Cherian <https://github.com/dcherian>`_
- Datasets are now guaranteed to have a ``'source'`` encoding, so the source
  file name is always stored (:issue:`2550`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- The ``apply`` methods for ``DatasetGroupBy``, ``DataArrayGroupBy``,
  ``DatasetResample`` and ``DataArrayResample`` now support passing positional
  arguments to the applied function as a tuple to the ``args`` argument.
  By `Matti Eskelinen <https://github.com/maaleske>`_.
- 0d slices of ndarrays are now obtained directly through indexing, rather than
  extracting and wrapping a scalar, avoiding unnecessary copying. By `Daniel
  Wennberg <https://github.com/danielwe>`_.
- Added support for ``fill_value`` with
  :py:meth:`~xarray.DataArray.shift` and :py:meth:`~xarray.Dataset.shift`
  By `Maximilian Roos <https://github.com/max-sixty>`_

Bug fixes
~~~~~~~~~

- Ensure files are automatically closed, if possible, when no longer referenced
  by a Python variable (:issue:`2560`).
  By `Stephan Hoyer <https://github.com/shoyer>`_
- Fixed possible race conditions when reading/writing to disk in parallel
  (:issue:`2595`).
  By `Stephan Hoyer <https://github.com/shoyer>`_
- Fix h5netcdf saving scalars with filters or chunks (:issue:`2563`).
  By `Martin Raspaud <https://github.com/mraspaud>`_.
- Fix parsing of ``_Unsigned`` attribute set by OPENDAP servers. (:issue:`2583`).
  By `Deepak Cherian <https://github.com/dcherian>`_
- Fix failure in time encoding when exporting to netCDF with versions of pandas
  less than 0.21.1 (:issue:`2623`).  By `Spencer Clark
  <https://github.com/spencerkclark>`_.
- Fix MultiIndex selection to update label and level (:issue:`2619`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

.. _whats-new.0.11.0:

v0.11.0 (7 November 2018)
-------------------------

Breaking changes
~~~~~~~~~~~~~~~~

- Finished deprecations (changed behavior with this release):

  - ``Dataset.T`` has been removed as a shortcut for :py:meth:`Dataset.transpose`.
    Call :py:meth:`Dataset.transpose` directly instead.
  - Iterating over a ``Dataset`` now includes only data variables, not coordinates.
    Similarily, calling ``len`` and ``bool`` on a ``Dataset`` now
    includes only data variables.
  - ``DataArray.__contains__`` (used by Python's ``in`` operator) now checks
    array data, not coordinates.
  - The old resample syntax from before xarray 0.10, e.g.,
    ``data.resample('1D', dim='time', how='mean')``, is no longer supported will
    raise an error in most cases. You need to use the new resample syntax
    instead, e.g., ``data.resample(time='1D').mean()`` or
    ``data.resample({'time': '1D'}).mean()``.


- New deprecations (behavior will be changed in xarray 0.12):

  - Reduction of :py:meth:`DataArray.groupby` and :py:meth:`DataArray.resample`
    without dimension argument will change in the next release.
    Now we warn a FutureWarning.
    By `Keisuke Fujii <https://github.com/fujiisoup>`_.
  - The ``inplace`` kwarg of a number of `DataArray` and `Dataset` methods is being
    deprecated and will be removed in the next release.
    By `Deepak Cherian <https://github.com/dcherian>`_.


- Refactored storage backends:

  - Xarray's storage backends now automatically open and close files when
    necessary, rather than requiring opening a file with ``autoclose=True``. A
    global least-recently-used cache is used to store open files; the default
    limit of 128 open files should suffice in most cases, but can be adjusted if
    necessary with
    ``xarray.set_options(file_cache_maxsize=...)``. The ``autoclose`` argument
    to ``open_dataset`` and related functions has been deprecated and is now a
    no-op.

    This change, along with an internal refactor of xarray's storage backends,
    should significantly improve performance when reading and writing
    netCDF files with Dask, especially when working with many files or using
    Dask Distributed. By `Stephan Hoyer <https://github.com/shoyer>`_


- Support for non-standard calendars used in climate science:

  - Xarray will now always use :py:class:`cftime.datetime` objects, rather
    than by default trying to coerce them into ``np.datetime64[ns]`` objects.
    A :py:class:`~xarray.CFTimeIndex` will be used for indexing along time
    coordinates in these cases.
  - A new method :py:meth:`~xarray.CFTimeIndex.to_datetimeindex` has been added
    to aid in converting from a  :py:class:`~xarray.CFTimeIndex` to a
    :py:class:`pandas.DatetimeIndex` for the remaining use-cases where
    using a :py:class:`~xarray.CFTimeIndex` is still a limitation (e.g. for
    resample or plotting).
  - Setting the ``enable_cftimeindex`` option is now a no-op and emits a
    ``FutureWarning``.

Enhancements
~~~~~~~~~~~~

- :py:meth:`xarray.DataArray.plot.line` can now accept multidimensional
  coordinate variables as input. `hue` must be a dimension name in this case.
  (:issue:`2407`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Added support for Python 3.7. (:issue:`2271`).
  By `Joe Hamman <https://github.com/jhamman>`_.
- Added support for plotting data with `pandas.Interval` coordinates, such as those
  created by :py:meth:`~xarray.DataArray.groupby_bins`
  By `Maximilian Maahn <https://github.com/maahn>`_.
- Added :py:meth:`~xarray.CFTimeIndex.shift` for shifting the values of a
  CFTimeIndex by a specified frequency. (:issue:`2244`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Added support for using ``cftime.datetime`` coordinates with
  :py:meth:`~xarray.DataArray.differentiate`,
  :py:meth:`~xarray.Dataset.differentiate`,
  :py:meth:`~xarray.DataArray.interp`, and
  :py:meth:`~xarray.Dataset.interp`.
  By `Spencer Clark <https://github.com/spencerkclark>`_
- There is now a global option to either always keep or always discard
  dataset and dataarray attrs upon operations. The option is set with
  ``xarray.set_options(keep_attrs=True)``, and the default is to use the old
  behaviour.
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added a new backend for the GRIB file format based on ECMWF *cfgrib*
  python driver and *ecCodes* C-library. (:issue:`2475`)
  By `Alessandro Amici <https://github.com/alexamici>`_,
  sponsored by `ECMWF <https://github.com/ecmwf>`_.
- Resample now supports a dictionary mapping from dimension to frequency as
  its first argument, e.g., ``data.resample({'time': '1D'}).mean()``. This is
  consistent with other xarray functions that accept either dictionaries or
  keyword arguments. By `Stephan Hoyer <https://github.com/shoyer>`_.

- The preferred way to access tutorial data is now to load it lazily with
  :py:meth:`xarray.tutorial.open_dataset`.
  :py:meth:`xarray.tutorial.load_dataset` calls `Dataset.load()` prior
  to returning (and is now deprecated). This was changed in order to facilitate
  using tutorial datasets with dask.
  By `Joe Hamman <https://github.com/jhamman>`_.
- ``DataArray`` can now use ``xr.set_option(keep_attrs=True)`` and retain attributes in binary operations,
  such as (``+, -, * ,/``). Default behaviour is unchanged (*Attributes will be dismissed*). By `Michael Blaschek <https://github.com/MBlaschek>`_

Bug fixes
~~~~~~~~~

- ``FacetGrid`` now properly uses the ``cbar_kwargs`` keyword argument.
  (:issue:`1504`, :issue:`1717`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Addition and subtraction operators used with a CFTimeIndex now preserve the
  index's type. (:issue:`2244`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- We now properly handle arrays of ``datetime.datetime`` and ``datetime.timedelta``
  provided as coordinates. (:issue:`2512`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- ``xarray.DataArray.roll`` correctly handles multidimensional arrays.
  (:issue:`2445`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- ``xarray.plot()`` now properly accepts a ``norm`` argument and does not override
  the norm's ``vmin`` and ``vmax``. (:issue:`2381`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- ``xarray.DataArray.std()`` now correctly accepts ``ddof`` keyword argument.
  (:issue:`2240`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Restore matplotlib's default of plotting dashed negative contours when
  a single color is passed to ``DataArray.contour()`` e.g. ``colors='k'``.
  By `Deepak Cherian <https://github.com/dcherian>`_.


- Fix a bug that caused some indexing operations on arrays opened with
  ``open_rasterio`` to error (:issue:`2454`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Subtracting one CFTimeIndex from another now returns a
  ``pandas.TimedeltaIndex``, analogous to the behavior for DatetimeIndexes
  (:issue:`2484`).  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Adding a TimedeltaIndex to, or subtracting a TimedeltaIndex from a
  CFTimeIndex is now allowed (:issue:`2484`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Avoid use of Dask's deprecated ``get=`` parameter in tests
  by `Matthew Rocklin <https://github.com/mrocklin>`_.
- An ``OverflowError`` is now accurately raised and caught during the
  encoding process if a reference date is used that is so distant that
  the dates must be encoded using cftime rather than NumPy (:issue:`2272`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.

- Chunked datasets can now roundtrip to Zarr storage continually
  with `to_zarr` and ``open_zarr`` (:issue:`2300`).
  By `Lily Wang <https://github.com/lilyminium>`_.

.. _whats-new.0.10.9:

v0.10.9 (21 September 2018)
---------------------------

This minor release contains a number of backwards compatible enhancements.

Announcements of note:

- Xarray is now a NumFOCUS fiscally sponsored project! Read
  `the anouncement <https://numfocus.org/blog/xarray-joins-numfocus-sponsored-projects>`_
  for more details.
- We have a new :doc:`roadmap` that outlines our future development plans.

- ``Dataset.apply`` now properly documents the way `func` is called.
  By `Matti Eskelinen <https://github.com/maaleske>`_.

Enhancements
~~~~~~~~~~~~

- :py:meth:`~xarray.DataArray.differentiate` and
  :py:meth:`~xarray.Dataset.differentiate` are newly added.
  (:issue:`1332`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Default colormap for sequential and divergent data can now be set via
  :py:func:`~xarray.set_options()`
  (:issue:`2394`)
  By `Julius Busecke <https://github.com/jbusecke>`_.

- min_count option is newly supported in :py:meth:`~xarray.DataArray.sum`,
  :py:meth:`~xarray.DataArray.prod` and :py:meth:`~xarray.Dataset.sum`, and
  :py:meth:`~xarray.Dataset.prod`.
  (:issue:`2230`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- :py:func:`~plot.plot()` now accepts the kwargs
  ``xscale, yscale, xlim, ylim, xticks, yticks`` just like Pandas. Also ``xincrease=False, yincrease=False`` now use matplotlib's axis inverting methods instead of setting limits.
  By `Deepak Cherian <https://github.com/dcherian>`_. (:issue:`2224`)

- DataArray coordinates and Dataset coordinates and data variables are
  now displayed as `a b ... y z` rather than `a b c d ...`.
  (:issue:`1186`)
  By `Seth P <https://github.com/seth-p>`_.
- A new CFTimeIndex-enabled :py:func:`cftime_range` function for use in
  generating dates from standard or non-standard calendars.  By `Spencer Clark
  <https://github.com/spencerkclark>`_.

- When interpolating over a ``datetime64`` axis, you can now provide a datetime string instead of a ``datetime64`` object. E.g. ``da.interp(time='1991-02-01')``
  (:issue:`2284`)
  By `Deepak Cherian <https://github.com/dcherian>`_.

- A clear error message is now displayed if a ``set`` or ``dict`` is passed in place of an array
  (:issue:`2331`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.

- Applying ``unstack`` to a large DataArray or Dataset is now much faster if the MultiIndex has not been modified after stacking the indices.
  (:issue:`1560`)
  By `Maximilian Maahn <https://github.com/maahn>`_.

- You can now control whether or not to offset the coordinates when using
  the ``roll`` method and the current behavior, coordinates rolled by default,
  raises a deprecation warning unless explicitly setting the keyword argument.
  (:issue:`1875`)
  By `Andrew Huang <https://github.com/ahuang11>`_.

- You can now call ``unstack`` without arguments to unstack every MultiIndex in a DataArray or Dataset.
  By `Julia Signell <https://github.com/jsignell>`_.

- Added the ability to pass a data kwarg to ``copy`` to create a new object with the
  same metadata as the original object but using new values.
  By `Julia Signell <https://github.com/jsignell>`_.

Bug fixes
~~~~~~~~~

- ``xarray.plot.imshow()`` correctly uses the ``origin`` argument.
  (:issue:`2379`)
  By `Deepak Cherian <https://github.com/dcherian>`_.

- Fixed ``DataArray.to_iris()`` failure while creating ``DimCoord`` by
  falling back to creating ``AuxCoord``. Fixed dependency on ``var_name``
  attribute being set.
  (:issue:`2201`)
  By `Thomas Voigt <https://github.com/tv3141>`_.
- Fixed a bug in ``zarr`` backend which prevented use with datasets with
  invalid chunk size encoding after reading from an existing store
  (:issue:`2278`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Tests can be run in parallel with pytest-xdist
  By `Tony Tung <https://github.com/ttung>`_.

- Follow up the renamings in dask; from dask.ghost to dask.overlap
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Now raises a ValueError when there is a conflict between dimension names and
  level names of MultiIndex. (:issue:`2299`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Follow up the renamings in dask; from dask.ghost to dask.overlap
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Now :py:func:`~xarray.apply_ufunc` raises a ValueError when the size of
  ``input_core_dims`` is inconsistent with the number of arguments.
  (:issue:`2341`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Fixed ``Dataset.filter_by_attrs()`` behavior not matching ``netCDF4.Dataset.get_variables_by_attributes()``.
  When more than one ``key=value`` is passed into ``Dataset.filter_by_attrs()`` it will now return a Dataset with variables which pass
  all the filters.
  (:issue:`2315`)
  By `Andrew Barna <https://github.com/docotak>`_.

.. _whats-new.0.10.8:

v0.10.8 (18 July 2018)
----------------------

Breaking changes
~~~~~~~~~~~~~~~~

- Xarray no longer supports python 3.4. Additionally, the minimum supported
  versions of the following dependencies has been updated and/or clarified:

  - Pandas: 0.18 -> 0.19
  - NumPy: 1.11 -> 1.12
  - Dask: 0.9 -> 0.16
  - Matplotlib: unspecified -> 1.5

  (:issue:`2204`). By `Joe Hamman <https://github.com/jhamman>`_.

Enhancements
~~~~~~~~~~~~

- :py:meth:`~xarray.DataArray.interp_like` and
  :py:meth:`~xarray.Dataset.interp_like` methods are newly added.
  (:issue:`2218`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Added support for curvilinear and unstructured generic grids
  to :py:meth:`~xarray.DataArray.to_cdms2` and
  :py:meth:`~xarray.DataArray.from_cdms2` (:issue:`2262`).
  By `Stephane Raynaud <https://github.com/stefraynaud>`_.

Bug fixes
~~~~~~~~~

- Fixed a bug in ``zarr`` backend which prevented use with datasets with
  incomplete chunks in multiple dimensions (:issue:`2225`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Fixed a bug in :py:meth:`~Dataset.to_netcdf` which prevented writing
  datasets when the arrays had different chunk sizes (:issue:`2254`).
  By `Mike Neish <https://github.com/neishm>`_.

- Fixed masking during the conversion to cdms2 objects by
  :py:meth:`~xarray.DataArray.to_cdms2` (:issue:`2262`).
  By `Stephane Raynaud <https://github.com/stefraynaud>`_.

- Fixed a bug in 2D plots which incorrectly raised an error when 2D coordinates
  weren't monotonic (:issue:`2250`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- Fixed warning raised in :py:meth:`~Dataset.to_netcdf` due to deprecation of
  `effective_get` in dask (:issue:`2238`).
  By `Joe Hamman <https://github.com/jhamman>`_.

.. _whats-new.0.10.7:

v0.10.7 (7 June 2018)
---------------------

Enhancements
~~~~~~~~~~~~

- Plot labels now make use of metadata that follow CF conventions
  (:issue:`2135`).
  By `Deepak Cherian <https://github.com/dcherian>`_ and `Ryan Abernathey <https://github.com/rabernat>`_.

- Line plots now support facetting with ``row`` and ``col`` arguments
  (:issue:`2107`).
  By `Yohai Bar Sinai <https://github.com/yohai>`_.

- :py:meth:`~xarray.DataArray.interp` and :py:meth:`~xarray.Dataset.interp`
  methods are newly added.
  See :ref:`interp` for the detail.
  (:issue:`2079`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

Bug fixes
~~~~~~~~~

- Fixed a bug in ``rasterio`` backend which prevented use with ``distributed``.
  The ``rasterio`` backend now returns pickleable objects (:issue:`2021`).
  By `Joe Hamman <https://github.com/jhamman>`_.

.. _whats-new.0.10.6:

v0.10.6 (31 May 2018)
---------------------

The minor release includes a number of bug-fixes and backwards compatible
enhancements.

Enhancements
~~~~~~~~~~~~

- New PseudoNetCDF backend for many Atmospheric data formats including
  GEOS-Chem, CAMx, NOAA arlpacked bit and many others. See
  :ref:`io.PseudoNetCDF` for more details.
  By `Barron Henderson <https://github.com/barronh>`_.

- The :py:class:`Dataset` constructor now aligns :py:class:`DataArray`
  arguments in ``data_vars`` to indexes set explicitly in ``coords``,
  where previously an error would be raised.
  (:issue:`674`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.

- :py:meth:`~DataArray.sel`, :py:meth:`~DataArray.isel` & :py:meth:`~DataArray.reindex`,
  (and their :py:class:`Dataset` counterparts) now support supplying a ``dict``
  as a first argument, as an alternative to the existing approach
  of supplying `kwargs`. This allows for more robust behavior
  of dimension names which conflict with other keyword names, or are
  not strings.
  By `Maximilian Roos <https://github.com/max-sixty>`_.

- :py:meth:`~DataArray.rename` now supports supplying ``**kwargs``, as an
  alternative to the existing approach of supplying a ``dict`` as the
  first argument.
  By `Maximilian Roos <https://github.com/max-sixty>`_.

- :py:meth:`~DataArray.cumsum` and :py:meth:`~DataArray.cumprod` now support
  aggregation over multiple dimensions at the same time. This is the default
  behavior when dimensions are not specified (previously this raised an error).
  By `Stephan Hoyer <https://github.com/shoyer>`_

- :py:meth:`DataArray.dot` and :py:func:`dot` are partly supported with older
  dask<0.17.4. (related to :issue:`2203`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Xarray now uses `Versioneer <https://github.com/warner/python-versioneer>`__
  to manage its version strings. (:issue:`1300`).
  By `Joe Hamman <https://github.com/jhamman>`_.

Bug fixes
~~~~~~~~~

- Fixed a regression in 0.10.4, where explicitly specifying ``dtype='S1'`` or
  ``dtype=str`` in ``encoding`` with ``to_netcdf()`` raised an error
  (:issue:`2149`).
  `Stephan Hoyer <https://github.com/shoyer>`_

- :py:func:`apply_ufunc` now directly validates output variables
  (:issue:`1931`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed a bug where ``to_netcdf(..., unlimited_dims='bar')`` yielded NetCDF
  files with spurious 0-length dimensions (i.e. ``b``, ``a``, and ``r``)
  (:issue:`2134`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Removed spurious warnings with ``Dataset.update(Dataset)`` (:issue:`2161`)
  and ``array.equals(array)`` when ``array`` contains ``NaT`` (:issue:`2162`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Aggregations with :py:meth:`Dataset.reduce` (including ``mean``, ``sum``,
  etc) no longer drop unrelated coordinates (:issue:`1470`). Also fixed a
  bug where non-scalar data-variables that did not include the aggregation
  dimension were improperly skipped.
  By `Stephan Hoyer <https://github.com/shoyer>`_

- Fix :meth:`~DataArray.stack` with non-unique coordinates on pandas 0.23
  (:issue:`2160`).
  By `Stephan Hoyer <https://github.com/shoyer>`_

- Selecting data indexed by a length-1 ``CFTimeIndex`` with a slice of strings
  now behaves as it does when using a length-1 ``DatetimeIndex`` (i.e. it no
  longer falsely returns an empty array when the slice includes the value in
  the index) (:issue:`2165`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.

- Fix ``DataArray.groupby().reduce()`` mutating coordinates on the input array
  when grouping over dimension coordinates with duplicated entries
  (:issue:`2153`).
  By `Stephan Hoyer <https://github.com/shoyer>`_

- Fix ``Dataset.to_netcdf()`` cannot create group with ``engine="h5netcdf"``
  (:issue:`2177`).
  By `Stephan Hoyer <https://github.com/shoyer>`_

.. _whats-new.0.10.4:

v0.10.4 (16 May 2018)
----------------------

The minor release includes a number of bug-fixes and backwards compatible
enhancements. A highlight is ``CFTimeIndex``, which offers support for
non-standard calendars used in climate modeling.

Documentation
~~~~~~~~~~~~~

- New FAQ entry, :ref:`related-projects`.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- :ref:`assigning_values` now includes examples on how to select and assign
  values to a :py:class:`~xarray.DataArray` with ``.loc``.
  By `Chiara Lepore <https://github.com/chiaral>`_.

Enhancements
~~~~~~~~~~~~

- Add an option for using a ``CFTimeIndex`` for indexing times with
  non-standard calendars and/or outside the Timestamp-valid range; this index
  enables a subset of the functionality of a standard
  ``pandas.DatetimeIndex``.
  See :ref:`CFTimeIndex` for full details.
  (:issue:`789`, :issue:`1084`, :issue:`1252`)
  By `Spencer Clark <https://github.com/spencerkclark>`_ with help from
  `Stephan Hoyer <https://github.com/shoyer>`_.
- Allow for serialization of ``cftime.datetime`` objects (:issue:`789`,
  :issue:`1084`, :issue:`2008`, :issue:`1252`) using the standalone ``cftime``
  library.
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Support writing lists of strings as netCDF attributes (:issue:`2044`).
  By `Dan Nowacki <https://github.com/dnowacki-usgs>`_.
- :py:meth:`~xarray.Dataset.to_netcdf` with ``engine='h5netcdf'`` now accepts h5py
  encoding settings ``compression`` and ``compression_opts``, along with the
  NetCDF4-Python style settings ``gzip=True`` and ``complevel``.
  This allows using any compression plugin installed in hdf5, e.g. LZF
  (:issue:`1536`). By `Guido Imperiale <https://github.com/crusaderky>`_.
- :py:meth:`~xarray.dot` on dask-backed data will now call :func:`dask.array.einsum`.
  This greatly boosts speed and allows chunking on the core dims.
  The function now requires dask >= 0.17.3 to work on dask-backed data
  (:issue:`2074`). By `Guido Imperiale <https://github.com/crusaderky>`_.
- ``plot.line()`` learned new kwargs: ``xincrease``, ``yincrease`` that change
  the direction of the respective axes.
  By `Deepak Cherian <https://github.com/dcherian>`_.

- Added the ``parallel`` option to :py:func:`open_mfdataset`. This option uses
  ``dask.delayed`` to parallelize the open and preprocessing steps within
  ``open_mfdataset``. This is expected to provide performance improvements when
  opening many files, particularly when used in conjunction with dask's
  multiprocessing or distributed schedulers (:issue:`1981`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- New ``compute`` option in :py:meth:`~xarray.Dataset.to_netcdf`,
  :py:meth:`~xarray.Dataset.to_zarr`, and :py:func:`~xarray.save_mfdataset` to
  allow for the lazy computation of netCDF and zarr stores. This feature is
  currently only supported by the netCDF4 and zarr backends. (:issue:`1784`).
  By `Joe Hamman <https://github.com/jhamman>`_.


Bug fixes
~~~~~~~~~

- ``ValueError`` is raised when coordinates with the wrong size are assigned to
  a :py:class:`DataArray`. (:issue:`2112`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Fixed a bug in :py:meth:`~xarray.DataArray.rolling` with bottleneck. Also,
  fixed a bug in rolling an integer dask array. (:issue:`2113`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Fixed a bug where `keep_attrs=True` flag was neglected if
  :py:func:`apply_ufunc` was used with :py:class:`Variable`. (:issue:`2114`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- When assigning a :py:class:`DataArray` to :py:class:`Dataset`, any conflicted
  non-dimensional coordinates of the DataArray are now dropped.
  (:issue:`2068`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Better error handling in ``open_mfdataset`` (:issue:`2077`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- ``plot.line()`` does not call ``autofmt_xdate()`` anymore. Instead it changes
  the rotation and horizontal alignment of labels without removing the x-axes of
  any other subplots in the figure (if any).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Colorbar limits are now determined by excluding ±Infs too.
  By `Deepak Cherian <https://github.com/dcherian>`_.
  By `Joe Hamman <https://github.com/jhamman>`_.
- Fixed ``to_iris`` to maintain lazy dask array after conversion (:issue:`2046`).
  By `Alex Hilson <https://github.com/AlexHilson>`_ and `Stephan Hoyer <https://github.com/shoyer>`_.

.. _whats-new.0.10.3:

v0.10.3 (13 April 2018)
------------------------

The minor release includes a number of bug-fixes and backwards compatible enhancements.

Enhancements
~~~~~~~~~~~~

- :py:meth:`~xarray.DataArray.isin` and :py:meth:`~xarray.Dataset.isin` methods,
  which test each value in the array for whether it is contained in the
  supplied list, returning a bool array. See :ref:`selecting values with isin`
  for full details. Similar to the ``np.isin`` function.
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Some speed improvement to construct :py:class:`~xarray.core.rolling.DataArrayRolling`
  object (:issue:`1993`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Handle variables with different values for ``missing_value`` and
  ``_FillValue`` by masking values for both attributes; previously this
  resulted in a ``ValueError``. (:issue:`2016`)
  By `Ryan May <https://github.com/dopplershift>`_.

Bug fixes
~~~~~~~~~

- Fixed ``decode_cf`` function to operate lazily on dask arrays
  (:issue:`1372`). By `Ryan Abernathey <https://github.com/rabernat>`_.
- Fixed labeled indexing with slice bounds given by xarray objects with
  datetime64 or timedelta64 dtypes (:issue:`1240`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Attempting to convert an xarray.Dataset into a numpy array now raises an
  informative error message.
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Fixed a bug in decode_cf_datetime where ``int32`` arrays weren't parsed
  correctly (:issue:`2002`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- When calling `xr.auto_combine()` or `xr.open_mfdataset()` with a `concat_dim`,
  the resulting dataset will have that one-element dimension (it was
  silently dropped, previously) (:issue:`1988`).
  By `Ben Root <https://github.com/WeatherGod>`_.

.. _whats-new.0.10.2:

v0.10.2 (13 March 2018)
-----------------------

The minor release includes a number of bug-fixes and enhancements, along with
one possibly **backwards incompatible change**.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The addition of ``__array_ufunc__`` for xarray objects (see below) means that
  NumPy `ufunc methods`_ (e.g., ``np.add.reduce``) that previously worked on
  ``xarray.DataArray`` objects by converting them into NumPy arrays will now
  raise ``NotImplementedError`` instead. In all cases, the work-around is
  simple: convert your objects explicitly into NumPy arrays before calling the
  ufunc (e.g., with ``.values``).

.. _ufunc methods: https://docs.scipy.org/doc/numpy/reference/ufuncs.html#methods

Enhancements
~~~~~~~~~~~~

- Added :py:func:`~xarray.dot`, equivalent to :py:func:`numpy.einsum`.
  Also, :py:func:`~xarray.DataArray.dot` now supports ``dims`` option,
  which specifies the dimensions to sum over.
  (:issue:`1951`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Support for writing xarray datasets to netCDF files (netcdf4 backend only)
  when using the `dask.distributed <https://distributed.readthedocs.io>`_
  scheduler (:issue:`1464`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Support lazy vectorized-indexing. After this change, flexible indexing such
  as orthogonal/vectorized indexing, becomes possible for all the backend
  arrays. Also, lazy ``transpose`` is now also supported. (:issue:`1897`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Implemented NumPy's ``__array_ufunc__`` protocol for all xarray objects
  (:issue:`1617`). This enables using NumPy ufuncs directly on
  ``xarray.Dataset`` objects with recent versions of NumPy (v1.13 and newer):

  .. ipython:: python

    ds = xr.Dataset({'a': 1})
    np.sin(ds)

  This obliviates the need for the ``xarray.ufuncs`` module, which will be
  deprecated in the future when xarray drops support for older versions of
  NumPy. By `Stephan Hoyer <https://github.com/shoyer>`_.

- Improve :py:func:`~xarray.DataArray.rolling` logic.
  :py:func:`~xarray.core.rolling.DataArrayRolling` object now supports
  :py:func:`~xarray.core.rolling.DataArrayRolling.construct` method that returns a view
  of the DataArray / Dataset object with the rolling-window dimension added
  to the last axis. This enables more flexible operation, such as strided
  rolling, windowed rolling, ND-rolling, short-time FFT and convolution.
  (:issue:`1831`, :issue:`1142`, :issue:`819`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- :py:func:`~plot.line()` learned to make plots with data on x-axis if so specified. (:issue:`575`)
  By `Deepak Cherian <https://github.com/dcherian>`_.

Bug fixes
~~~~~~~~~

- Raise an informative error message when using ``apply_ufunc`` with numpy
  v1.11 (:issue:`1956`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Fix the precision drop after indexing datetime64 arrays (:issue:`1932`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Silenced irrelevant warnings issued by ``open_rasterio`` (:issue:`1964`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Fix kwarg `colors` clashing with auto-inferred `cmap` (:issue:`1461`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix :py:func:`~xarray.plot.imshow` error when passed an RGB array with
  size one in a spatial dimension.
  By `Zac Hatfield-Dodds <https://github.com/Zac-HD>`_.

.. _whats-new.0.10.1:

v0.10.1 (25 February 2018)
--------------------------

The minor release includes a number of bug-fixes and backwards compatible enhancements.

Documentation
~~~~~~~~~~~~~

- Added a new guide on :ref:`contributing` (:issue:`640`)
  By `Joe Hamman <https://github.com/jhamman>`_.
- Added apply_ufunc example to :ref:`/examples/weather-data.ipynb#Toy-weather-data` (:issue:`1844`).
  By `Liam Brannigan <https://github.com/braaannigan>`_.
- New entry `Why don’t aggregations return Python scalars?` in the
  :doc:`faq` (:issue:`1726`).
  By `0x0L <https://github.com/0x0L>`_.

Enhancements
~~~~~~~~~~~~
**New functions and methods**:

- Added :py:meth:`DataArray.to_iris` and
  :py:meth:`DataArray.from_iris` for
  converting data arrays to and from Iris_ Cubes with the same data and coordinates
  (:issue:`621` and :issue:`37`).
  By `Neil Parley <https://github.com/nparley>`_ and `Duncan Watson-Parris <https://github.com/duncanwp>`_.
- Experimental support for using `Zarr`_ as storage layer for xarray
  (:issue:`1223`).
  By `Ryan Abernathey <https://github.com/rabernat>`_ and
  `Joe Hamman <https://github.com/jhamman>`_.
- New :py:meth:`~xarray.DataArray.rank` on arrays and datasets. Requires
  bottleneck (:issue:`1731`).
  By `0x0L <https://github.com/0x0L>`_.
- ``.dt`` accessor can now ceil, floor and round timestamps to specified frequency.
  By `Deepak Cherian <https://github.com/dcherian>`_.

**Plotting enhancements**:

- :func:`xarray.plot.imshow` now handles RGB and RGBA images.
  Saturation can be adjusted with ``vmin`` and ``vmax``, or with ``robust=True``.
  By `Zac Hatfield-Dodds <https://github.com/Zac-HD>`_.
- :py:func:`~plot.contourf()` learned to contour 2D variables that have both a
  1D coordinate (e.g. time) and a 2D coordinate (e.g. depth as a function of
  time) (:issue:`1737`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- :py:func:`~plot.plot()` rotates x-axis ticks if x-axis is time.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- :py:func:`~plot.line()` can draw multiple lines if provided with a
  2D variable.
  By `Deepak Cherian <https://github.com/dcherian>`_.

**Other enhancements**:

- Reduce methods such as :py:func:`DataArray.sum()` now handles object-type array.

  .. ipython:: python

    da = xr.DataArray(np.array([True, False, np.nan], dtype=object), dims='x')
    da.sum()

  (:issue:`1866`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Reduce methods such as :py:func:`DataArray.sum()` now accepts ``dtype``
  arguments. (:issue:`1838`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Added nodatavals attribute to DataArray when using :py:func:`~xarray.open_rasterio`. (:issue:`1736`).
  By `Alan Snow <https://github.com/snowman2>`_.
- Use ``pandas.Grouper`` class in xarray resample methods rather than the
  deprecated ``pandas.TimeGrouper`` class (:issue:`1766`).
  By `Joe Hamman <https://github.com/jhamman>`_.
- Experimental support for parsing ENVI metadata to coordinates and attributes
  in :py:func:`xarray.open_rasterio`.
  By `Matti Eskelinen <https://github.com/maaleske>`_.
- Reduce memory usage when decoding a variable with a scale_factor, by
  converting 8-bit and 16-bit integers to float32 instead of float64
  (:pull:`1840`), and keeping float16 and float32 as float32 (:issue:`1842`).
  Correspondingly, encoded variables may also be saved with a smaller dtype.
  By `Zac Hatfield-Dodds <https://github.com/Zac-HD>`_.
- Speed of reindexing/alignment with dask array is orders of magnitude faster
  when inserting missing values  (:issue:`1847`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Fix ``axis`` keyword ignored when applying ``np.squeeze`` to ``DataArray`` (:issue:`1487`).
  By `Florian Pinault <https://github.com/floriankrb>`_.
- ``netcdf4-python`` has moved the its time handling in the ``netcdftime`` module to
  a standalone package (`netcdftime`_). As such, xarray now considers `netcdftime`_
  an optional dependency. One benefit of this change is that it allows for
  encoding/decoding of datetimes with non-standard calendars without the
  ``netcdf4-python`` dependency (:issue:`1084`).
  By `Joe Hamman <https://github.com/jhamman>`_.

.. _Zarr: http://zarr.readthedocs.io/

.. _Iris: http://scitools.org.uk/iris

.. _netcdftime: https://unidata.github.io/netcdftime

**New functions/methods**

- New :py:meth:`~xarray.DataArray.rank` on arrays and datasets. Requires
  bottleneck (:issue:`1731`).
  By `0x0L <https://github.com/0x0L>`_.

Bug fixes
~~~~~~~~~
- Rolling aggregation with ``center=True`` option now gives the same result
  with pandas including the last element (:issue:`1046`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Support indexing with a 0d-np.ndarray (:issue:`1921`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Added warning in api.py of a netCDF4 bug that occurs when
  the filepath has 88 characters (:issue:`1745`).
  By `Liam Brannigan <https://github.com/braaannigan>`_.
- Fixed encoding of multi-dimensional coordinates in
  :py:meth:`~Dataset.to_netcdf` (:issue:`1763`).
  By `Mike Neish <https://github.com/neishm>`_.
- Fixed chunking with non-file-based rasterio datasets (:issue:`1816`) and
  refactored rasterio test suite.
  By `Ryan Abernathey <https://github.com/rabernat>`_
- Bug fix in open_dataset(engine='pydap') (:issue:`1775`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Bug fix in vectorized assignment  (:issue:`1743`, :issue:`1744`).
  Now item assignment to :py:meth:`~DataArray.__setitem__` checks
- Bug fix in vectorized assignment  (:issue:`1743`, :issue:`1744`).
  Now item assignment to :py:meth:`DataArray.__setitem__` checks
  coordinates of target, destination and keys. If there are any conflict among
  these coordinates, ``IndexError`` will be raised.
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Properly point ``DataArray.__dask_scheduler__`` to
  ``dask.threaded.get``.  By `Matthew Rocklin <https://github.com/mrocklin>`_.
- Bug fixes in :py:meth:`DataArray.plot.imshow`: all-NaN arrays and arrays
  with size one in some dimension can now be plotted, which is good for
  exploring satellite imagery (:issue:`1780`).
  By `Zac Hatfield-Dodds <https://github.com/Zac-HD>`_.
- Fixed ``UnboundLocalError`` when opening netCDF file (:issue:`1781`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- The ``variables``, ``attrs``, and ``dimensions`` properties have been
  deprecated as part of a bug fix addressing an issue where backends were
  unintentionally loading the datastores data and attributes repeatedly during
  writes (:issue:`1798`).
  By `Joe Hamman <https://github.com/jhamman>`_.
- Compatibility fixes to plotting module for Numpy 1.14 and Pandas 0.22
  (:issue:`1813`).
  By `Joe Hamman <https://github.com/jhamman>`_.
- Bug fix in encoding coordinates with ``{'_FillValue': None}`` in netCDF
  metadata (:issue:`1865`).
  By `Chris Roth <https://github.com/czr137>`_.
- Fix indexing with lists for arrays loaded from netCDF files with
  ``engine='h5netcdf`` (:issue:`1864`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Corrected a bug with incorrect coordinates for non-georeferenced geotiff
  files (:issue:`1686`). Internally, we now use the rasterio coordinate
  transform tool instead of doing the computations ourselves. A
  ``parse_coordinates`` kwarg has beed added to :py:func:`~open_rasterio`
  (set to ``True`` per default).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The colors of discrete colormaps are now the same regardless if `seaborn`
  is installed or not (:issue:`1896`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Fixed dtype promotion rules in :py:func:`where` and :py:func:`concat` to
  match pandas (:issue:`1847`). A combination of strings/numbers or
  unicode/bytes now promote to object dtype, instead of strings or unicode.
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Fixed bug where :py:meth:`~xarray.DataArray.isnull` was loading data
  stored as dask arrays (:issue:`1937`).
  By `Joe Hamman <https://github.com/jhamman>`_.

.. _whats-new.0.10.0:

v0.10.0 (20 November 2017)
--------------------------

This is a major release that includes bug fixes, new features and a few
backwards incompatible changes. Highlights include:

- Indexing now supports broadcasting over dimensions, similar to NumPy's
  vectorized indexing (but better!).
- :py:meth:`~DataArray.resample` has a new groupby-like API like pandas.
- :py:func:`~xarray.apply_ufunc` facilitates wrapping and parallelizing
  functions written for NumPy arrays.
- Performance improvements, particularly for dask and :py:func:`open_mfdataset`.

Breaking changes
~~~~~~~~~~~~~~~~

- xarray now supports a form of vectorized indexing with broadcasting, where
  the result of indexing depends on dimensions of indexers,
  e.g., ``array.sel(x=ind)`` with ``ind.dims == ('y',)``. Alignment between
  coordinates on indexed and indexing objects is also now enforced.
  Due to these changes, existing uses of xarray objects to index other xarray
  objects will break in some cases.

  The new indexing API is much more powerful, supporting outer, diagonal and
  vectorized indexing in a single interface.
  The ``isel_points`` and ``sel_points`` methods are deprecated, since they are
  now redundant with the ``isel`` / ``sel`` methods.
  See :ref:`vectorized_indexing` for the details (:issue:`1444`,
  :issue:`1436`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_ and
  `Stephan Hoyer <https://github.com/shoyer>`_.

- A new resampling interface to match pandas' groupby-like API was added to
  :py:meth:`Dataset.resample` and :py:meth:`DataArray.resample`
  (:issue:`1272`). :ref:`Timeseries resampling <resampling>` is
  fully supported for data  with arbitrary dimensions as is both downsampling
  and upsampling (including linear, quadratic, cubic, and spline interpolation).

  Old syntax:

  .. ipython::
    :verbatim:

    In [1]: ds.resample('24H', dim='time', how='max')
    Out[1]:
    <xarray.Dataset>
    [...]

  New syntax:

  .. ipython::
    :verbatim:

    In [1]: ds.resample(time='24H').max()
    Out[1]:
    <xarray.Dataset>
    [...]

  Note that both versions are currently supported, but using the old syntax will
  produce a warning encouraging users to adopt the new syntax.
  By `Daniel Rothenberg <https://github.com/darothen>`_.

- Calling ``repr()`` or printing xarray objects at the command line or in a
  Jupyter Notebook will not longer automatically compute dask variables or
  load data on arrays lazily loaded from disk (:issue:`1522`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.

- Supplying ``coords`` as a dictionary to the ``DataArray`` constructor without
  also supplying an explicit ``dims`` argument is no longer supported. This
  behavior was deprecated in version 0.9 but will now raise an error
  (:issue:`727`).

- Several existing features have been deprecated and will change to new
  behavior in xarray v0.11. If you use any of them with xarray v0.10, you
  should see a ``FutureWarning`` that describes how to update your code:

  - ``Dataset.T`` has been deprecated an alias for ``Dataset.transpose()``
    (:issue:`1232`). In the next major version of xarray, it will provide short-
    cut lookup for variables or attributes with name ``'T'``.
  - ``DataArray.__contains__`` (e.g., ``key in data_array``) currently checks
    for membership in ``DataArray.coords``. In the next major version of
    xarray, it will check membership in the array data found in
    ``DataArray.values`` instead (:issue:`1267`).
  - Direct iteration over and counting a ``Dataset`` (e.g., ``[k for k in ds]``,
    ``ds.keys()``, ``ds.values()``, ``len(ds)`` and ``if ds``) currently
    includes all variables, both data and coordinates. For improved usability
    and consistency with pandas, in the next major version of xarray these will
    change to only include data variables (:issue:`884`). Use ``ds.variables``,
    ``ds.data_vars`` or ``ds.coords`` as alternatives.

- Changes to minimum versions of dependencies:

  - Old numpy < 1.11 and pandas < 0.18 are no longer supported (:issue:`1512`).
    By `Keisuke Fujii <https://github.com/fujiisoup>`_.
  - The minimum supported version bottleneck has increased to 1.1
    (:issue:`1279`).
    By `Joe Hamman <https://github.com/jhamman>`_.

Enhancements
~~~~~~~~~~~~

**New functions/methods**

- New helper function :py:func:`~xarray.apply_ufunc` for wrapping functions
  written to work on NumPy arrays to support labels on xarray objects
  (:issue:`770`). ``apply_ufunc`` also support automatic parallelization for
  many functions with dask. See :ref:`comput.wrapping-custom` and
  :ref:`dask.automatic-parallelization` for details.
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Added new method :py:meth:`Dataset.to_dask_dataframe`, convert a dataset into
  a dask dataframe.
  This allows lazy loading of data from a dataset containing dask arrays (:issue:`1462`).
  By `James Munroe <https://github.com/jmunroe>`_.

- New function :py:func:`~xarray.where` for conditionally switching between
  values in xarray objects, like :py:func:`numpy.where`:

  .. ipython::
    :verbatim:

    In [1]: import xarray as xr

    In [2]: arr = xr.DataArray([[1, 2, 3], [4, 5, 6]], dims=('x', 'y'))

    In [3]: xr.where(arr % 2, 'even', 'odd')
    Out[3]:
    <xarray.DataArray (x: 2, y: 3)>
    array([['even', 'odd', 'even'],
           ['odd', 'even', 'odd']],
          dtype='<U4')
    Dimensions without coordinates: x, y

  Equivalently, the :py:meth:`~xarray.Dataset.where` method also now supports
  the ``other`` argument, for filling with a value other than ``NaN``
  (:issue:`576`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Added :py:func:`~xarray.show_versions` function to aid in debugging
  (:issue:`1485`).
  By `Joe Hamman <https://github.com/jhamman>`_.

**Performance improvements**

- :py:func:`~xarray.concat` was computing variables that aren't in memory
  (e.g. dask-based) multiple times; :py:func:`~xarray.open_mfdataset`
  was loading them multiple times from disk. Now, both functions will instead
  load them at most once and, if they do, store them in memory in the
  concatenated array/dataset (:issue:`1521`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.

- Speed-up (x 100) of ``xarray.conventions.decode_cf_datetime``.
  By `Christian Chwala <https://github.com/cchwala>`_.

**IO related improvements**

- Unicode strings (``str`` on Python 3) are now round-tripped successfully even
  when written as character arrays (e.g., as netCDF3 files or when using
  ``engine='scipy'``) (:issue:`1638`). This is controlled by the ``_Encoding``
  attribute convention, which is also understood directly by the netCDF4-Python
  interface. See :ref:`io.string-encoding` for full details.
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Support for ``data_vars`` and ``coords`` keywords from
  :py:func:`~xarray.concat` added to :py:func:`~xarray.open_mfdataset`
  (:issue:`438`). Using these keyword arguments can significantly reduce
  memory usage and increase speed.
  By `Oleksandr Huziy <https://github.com/guziy>`_.

- Support for :py:class:`pathlib.Path` objects added to
  :py:func:`~xarray.open_dataset`, :py:func:`~xarray.open_mfdataset`,
  ``xarray.to_netcdf``, and :py:func:`~xarray.save_mfdataset`
  (:issue:`799`):

  .. ipython::
    :verbatim:

    In [2]: from pathlib import Path  # In Python 2, use pathlib2!

    In [3]: data_dir = Path("data/")

    In [4]: one_file = data_dir / "dta_for_month_01.nc"

    In [5]: xr.open_dataset(one_file)
    Out[5]:
    <xarray.Dataset>
    [...]

  By `Willi Rath <https://github.com/willirath>`_.

- You can now explicitly disable any default ``_FillValue`` (``NaN`` for
  floating point values) by passing the enconding ``{'_FillValue': None}``
  (:issue:`1598`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- More attributes available in :py:attr:`~xarray.Dataset.attrs` dictionary when
  raster files are opened with :py:func:`~xarray.open_rasterio`.
  By `Greg Brener <https://github.com/gbrener>`_.

- Support for NetCDF files using an ``_Unsigned`` attribute to indicate that a
  a signed integer data type should be interpreted as unsigned bytes
  (:issue:`1444`).
  By `Eric Bruning <https://github.com/deeplycloudy>`_.

- Support using an existing, opened netCDF4 ``Dataset`` with
  :py:class:`~xarray.backends.NetCDF4DataStore`. This permits creating an
  :py:class:`~xarray.Dataset` from a netCDF4 ``Dataset`` that has been opened using
  other means (:issue:`1459`).
  By `Ryan May <https://github.com/dopplershift>`_.

- Changed :py:class:`~xarray.backends.PydapDataStore` to take a Pydap dataset.
  This permits opening Opendap datasets that require authentication, by
  instantiating a Pydap dataset with a session object. Also added
  :py:meth:`xarray.backends.PydapDataStore.open` which takes a url and session
  object (:issue:`1068`).
  By `Philip Graae <https://github.com/mrpgraae>`_.

- Support reading and writing unlimited dimensions with h5netcdf (:issue:`1636`).
  By `Joe Hamman <https://github.com/jhamman>`_.

**Other improvements**

- Added ``_ipython_key_completions_`` to xarray objects, to enable
  autocompletion for dictionary-like access in IPython, e.g.,
  ``ds['tem`` + tab -> ``ds['temperature']`` (:issue:`1628`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Support passing keyword arguments to ``load``, ``compute``, and ``persist``
  methods. Any keyword arguments supplied to these methods are passed on to
  the corresponding dask function (:issue:`1523`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Encoding attributes are now preserved when xarray objects are concatenated.
  The encoding is copied from the first object  (:issue:`1297`).
  By `Joe Hamman <https://github.com/jhamman>`_ and
  `Gerrit Holl <https://github.com/gerritholl>`_.

- Support applying rolling window operations using bottleneck's moving window
  functions on data stored as dask arrays (:issue:`1279`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Experimental support for the Dask collection interface (:issue:`1674`).
  By `Matthew Rocklin <https://github.com/mrocklin>`_.

Bug fixes
~~~~~~~~~

- Suppress ``RuntimeWarning`` issued by ``numpy`` for "invalid value comparisons"
  (e.g. ``NaN``). Xarray now behaves similarly to Pandas in its treatment of
  binary and unary operations on objects with NaNs (:issue:`1657`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Unsigned int support for reduce methods with ``skipna=True``
  (:issue:`1562`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Fixes to ensure xarray works properly with pandas 0.21:

  - Fix :py:meth:`~xarray.DataArray.isnull` method (:issue:`1549`).
  - :py:meth:`~xarray.DataArray.to_series` and
    :py:meth:`~xarray.Dataset.to_dataframe` should not return a ``pandas.MultiIndex``
    for 1D data (:issue:`1548`).
  - Fix plotting with datetime64 axis labels (:issue:`1661`).

  By `Stephan Hoyer <https://github.com/shoyer>`_.

- :py:func:`~xarray.open_rasterio` method now shifts the rasterio
  coordinates so that they are centered in each pixel (:issue:`1468`).
  By `Greg Brener <https://github.com/gbrener>`_.

- :py:meth:`~xarray.Dataset.rename` method now doesn't throw errors
  if some ``Variable`` is renamed to the same name as another ``Variable``
  as long as that other ``Variable`` is also renamed (:issue:`1477`). This
  method now does throw when two ``Variables`` would end up with the same name
  after the rename (since one of them would get overwritten in this case).
  By `Prakhar Goel <https://github.com/newt0311>`_.

- Fix :py:func:`xarray.testing.assert_allclose` to actually use ``atol`` and
  ``rtol`` arguments when called on ``DataArray`` objects (:issue:`1488`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- xarray ``quantile`` methods now properly raise a ``TypeError`` when applied to
  objects with data stored as ``dask`` arrays (:issue:`1529`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Fix positional indexing to allow the use of unsigned integers (:issue:`1405`).
  By `Joe Hamman <https://github.com/jhamman>`_ and
  `Gerrit Holl <https://github.com/gerritholl>`_.

- Creating a :py:class:`Dataset` now raises ``MergeError`` if a coordinate
  shares a name with a dimension but is comprised of arbitrary dimensions
  (:issue:`1120`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- :py:func:`~xarray.open_rasterio` method now skips rasterio's ``crs``
  attribute if its value is ``None`` (:issue:`1520`).
  By `Leevi Annala <https://github.com/leevei>`_.

- Fix :py:func:`xarray.DataArray.to_netcdf` to return bytes when no path is
  provided (:issue:`1410`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Fix :py:func:`xarray.save_mfdataset` to properly raise an informative error
  when objects other than  ``Dataset`` are provided (:issue:`1555`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- :py:func:`xarray.Dataset.copy` would not preserve the encoding property
  (:issue:`1586`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.

- :py:func:`xarray.concat` would eagerly load dask variables into memory if
  the first argument was a numpy variable (:issue:`1588`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.

- Fix bug in :py:meth:`~xarray.Dataset.to_netcdf` when writing in append mode
  (:issue:`1215`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Fix ``netCDF4`` backend to properly roundtrip the ``shuffle`` encoding option
  (:issue:`1606`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Fix bug when using ``pytest`` class decorators to skiping certain unittests.
  The previous behavior unintentionally causing additional tests to be skipped
  (:issue:`1531`). By `Joe Hamman <https://github.com/jhamman>`_.

- Fix pynio backend for upcoming release of pynio with Python 3 support
  (:issue:`1611`). By `Ben Hillman <https://github/brhillman>`_.

- Fix ``seaborn`` import warning for Seaborn versions 0.8 and newer when the
  ``apionly`` module was deprecated.
  (:issue:`1633`). By `Joe Hamman <https://github.com/jhamman>`_.

- Fix COMPAT: MultiIndex checking is fragile
  (:issue:`1833`). By `Florian Pinault <https://github.com/floriankrb>`_.

- Fix ``rasterio`` backend for Rasterio versions 1.0alpha10 and newer.
  (:issue:`1641`). By `Chris Holden <https://github.com/ceholden>`_.

Bug fixes after rc1
~~~~~~~~~~~~~~~~~~~

- Suppress warning in IPython autocompletion, related to the deprecation
  of ``.T`` attributes (:issue:`1675`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Fix a bug in lazily-indexing netCDF array. (:issue:`1688`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- (Internal bug) MemoryCachedArray now supports the orthogonal indexing.
  Also made some internal cleanups around array wrappers (:issue:`1429`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- (Internal bug) MemoryCachedArray now always wraps ``np.ndarray`` by
  ``NumpyIndexingAdapter``. (:issue:`1694`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Fix importing xarray when running Python with ``-OO`` (:issue:`1706`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Saving a netCDF file with a coordinates with a spaces in its names now raises
  an appropriate warning (:issue:`1689`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix two bugs that were preventing dask arrays from being specified as
  coordinates in the DataArray constructor (:issue:`1684`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Fixed ``apply_ufunc`` with ``dask='parallelized'`` for scalar arguments
  (:issue:`1697`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix "Chunksize cannot exceed dimension size" error when writing netCDF4 files
  loaded from disk (:issue:`1225`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Validate the shape of coordinates with names matching dimensions in the
  DataArray constructor (:issue:`1709`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Raise ``NotImplementedError`` when attempting to save a MultiIndex to a
  netCDF file (:issue:`1547`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Remove netCDF dependency from rasterio backend tests.
  By `Matti Eskelinen <https://github.com/maaleske>`_

Bug fixes after rc2
~~~~~~~~~~~~~~~~~~~

- Fixed unexpected behavior in ``Dataset.set_index()`` and
  ``DataArray.set_index()`` introduced by Pandas 0.21.0. Setting a new
  index with a single variable resulted in 1-level
  ``pandas.MultiIndex`` instead of a simple ``pandas.Index``
  (:issue:`1722`).  By `Benoit Bovy <https://github.com/benbovy>`_.

- Fixed unexpected memory loading of backend arrays after ``print``.
  (:issue:`1720`).  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

.. _whats-new.0.9.6:

v0.9.6 (8 June 2017)
--------------------

This release includes a number of backwards compatible enhancements and bug
fixes.

Enhancements
~~~~~~~~~~~~

- New :py:meth:`~xarray.Dataset.sortby` method to ``Dataset`` and ``DataArray``
  that enable sorting along dimensions (:issue:`967`).
  See :ref:`the docs <reshape.sort>` for examples.
  By `Chun-Wei Yuan <https://github.com/chunweiyuan>`_ and
  `Kyle Heuton <https://github.com/kheuton>`_.

- Add ``.dt`` accessor to DataArrays for computing datetime-like properties
  for the values they contain, similar to ``pandas.Series`` (:issue:`358`).
  By `Daniel Rothenberg <https://github.com/darothen>`_.

- Renamed internal dask arrays created by ``open_dataset`` to match new dask
  conventions (:issue:`1343`).
  By `Ryan Abernathey <https://github.com/rabernat>`_.

- :py:meth:`~xarray.as_variable` is now part of the public API (:issue:`1303`).
  By `Benoit Bovy <https://github.com/benbovy>`_.

- :py:func:`~xarray.align` now supports ``join='exact'``, which raises
  an error instead of aligning when indexes to be aligned are not equal.
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- New function :py:func:`~xarray.open_rasterio` for opening raster files with
  the `rasterio <https://rasterio.readthedocs.io/en/latest/>`_ library.
  See :ref:`the docs <io.rasterio>` for details.
  By `Joe Hamman <https://github.com/jhamman>`_,
  `Nic Wayand <https://github.com/NicWayand>`_ and
  `Fabien Maussion <https://github.com/fmaussion>`_

Bug fixes
~~~~~~~~~

- Fix error from repeated indexing of datasets loaded from disk (:issue:`1374`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix a bug where ``.isel_points`` wrongly assigns unselected coordinate to
  ``data_vars``.
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Tutorial datasets are now checked against a reference MD5 sum to confirm
  successful download (:issue:`1392`). By `Matthew Gidden
  <https://github.com/gidden>`_.

- ``DataArray.chunk()`` now accepts dask specific kwargs like
  ``Dataset.chunk()`` does. By `Fabien Maussion <https://github.com/fmaussion>`_.

- Support for ``engine='pydap'`` with recent releases of Pydap (3.2.2+),
  including on Python 3 (:issue:`1174`).

Documentation
~~~~~~~~~~~~~

- A new `gallery <http://xarray.pydata.org/en/latest/auto_gallery/index.html>`_
  allows to add interactive examples to the documentation.
  By `Fabien Maussion <https://github.com/fmaussion>`_.

Testing
~~~~~~~

- Fix test suite failure caused by changes to ``pandas.cut`` function
  (:issue:`1386`).
  By `Ryan Abernathey <https://github.com/rabernat>`_.

- Enhanced tests suite by use of ``@network`` decorator, which is
  controlled via ``--run-network-tests`` command line argument
  to ``py.test`` (:issue:`1393`).
  By `Matthew Gidden <https://github.com/gidden>`_.

.. _whats-new.0.9.5:

v0.9.5 (17 April, 2017)
-----------------------

Remove an inadvertently introduced print statement.

.. _whats-new.0.9.3:

v0.9.3 (16 April, 2017)
-----------------------

This minor release includes bug-fixes and backwards compatible enhancements.

Enhancements
~~~~~~~~~~~~

- New :py:meth:`~xarray.DataArray.persist` method to Datasets and DataArrays to
  enable persisting data in distributed memory when using Dask (:issue:`1344`).
  By `Matthew Rocklin <https://github.com/mrocklin>`_.

- New :py:meth:`~xarray.DataArray.expand_dims` method for ``DataArray`` and
  ``Dataset`` (:issue:`1326`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

Bug fixes
~~~~~~~~~

- Fix ``.where()`` with ``drop=True`` when arguments do not have indexes
  (:issue:`1350`). This bug, introduced in v0.9, resulted in xarray producing
  incorrect results in some cases.
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed writing to file-like objects with :py:meth:`~xarray.Dataset.to_netcdf`
  (:issue:`1320`).
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed explicitly setting ``engine='scipy'`` with ``to_netcdf`` when not
  providing a path (:issue:`1321`).
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed open_dataarray does not pass properly its parameters to open_dataset
  (:issue:`1359`).
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Ensure test suite works when runs from an installed version of xarray
  (:issue:`1336`). Use ``@pytest.mark.slow`` instead of a custom flag to mark
  slow tests.
  By `Stephan Hoyer <https://github.com/shoyer>`_

.. _whats-new.0.9.2:

v0.9.2 (2 April 2017)
---------------------

The minor release includes bug-fixes and backwards compatible enhancements.

Enhancements
~~~~~~~~~~~~

- ``rolling`` on Dataset is now supported (:issue:`859`).

- ``.rolling()`` on Dataset is now supported (:issue:`859`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- When bottleneck version 1.1 or later is installed, use bottleneck for rolling
  ``var``, ``argmin``, ``argmax``, and ``rank`` computations. Also, rolling
  median now accepts a ``min_periods`` argument (:issue:`1276`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- When ``.plot()`` is called on a 2D DataArray and only one dimension is
  specified with ``x=`` or ``y=``, the other dimension is now guessed
  (:issue:`1291`).
  By `Vincent Noel <https://github.com/vnoel>`_.

- Added new method :py:meth:`~Dataset.assign_attrs` to ``DataArray`` and
  ``Dataset``, a chained-method compatible implementation of the
  ``dict.update`` method on attrs (:issue:`1281`).
  By `Henry S. Harrison <https://hsharrison.github.io>`_.

- Added new ``autoclose=True`` argument to
  :py:func:`~xarray.open_mfdataset` to explicitly close opened files when not in
  use to prevent occurrence of an OS Error related to too many open files
  (:issue:`1198`).
  Note, the default is ``autoclose=False``, which is consistent with
  previous xarray behavior.
  By `Phillip J. Wolfram <https://github.com/pwolfram>`_.

- The ``repr()`` of ``Dataset`` and ``DataArray`` attributes uses a similar
  format to coordinates and variables, with vertically aligned entries
  truncated to fit on a single line (:issue:`1319`).  Hopefully this will stop
  people writing ``data.attrs = {}`` and discarding metadata in notebooks for
  the sake of cleaner output.  The full metadata is still available as
  ``data.attrs``.
  By `Zac Hatfield-Dodds <https://github.com/Zac-HD>`_.

- Enhanced tests suite by use of ``@slow`` and ``@flaky`` decorators, which are
  controlled via ``--run-flaky`` and ``--skip-slow`` command line arguments
  to ``py.test`` (:issue:`1336`).
  By `Stephan Hoyer <https://github.com/shoyer>`_ and
  `Phillip J. Wolfram <https://github.com/pwolfram>`_.

- New aggregation on rolling objects :py:meth:`~core.rolling.DataArrayRolling.count`
  which providing a rolling count of valid values (:issue:`1138`).

Bug fixes
~~~~~~~~~
- Rolling operations now keep preserve original dimension order (:issue:`1125`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Fixed ``sel`` with ``method='nearest'`` on Python 2.7 and 64-bit Windows
  (:issue:`1140`).
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed ``where`` with ``drop='True'`` for empty masks (:issue:`1341`).
  By `Stephan Hoyer <https://github.com/shoyer>`_ and
  `Phillip J. Wolfram <https://github.com/pwolfram>`_.

.. _whats-new.0.9.1:

v0.9.1 (30 January 2017)
------------------------

Renamed the "Unindexed dimensions" section in the ``Dataset`` and
``DataArray`` repr (added in v0.9.0) to "Dimensions without coordinates"
(:issue:`1199`).

.. _whats-new.0.9.0:

v0.9.0 (25 January 2017)
------------------------

This major release includes five months worth of enhancements and bug fixes from
24 contributors, including some significant changes that are not fully backwards
compatible. Highlights include:

- Coordinates are now *optional* in the xarray data model, even for dimensions.
- Changes to caching, lazy loading and pickling to improve xarray's experience
  for parallel computing.
- Improvements for accessing and manipulating ``pandas.MultiIndex`` levels.
- Many new methods and functions, including
  :py:meth:`~DataArray.quantile`,
  :py:meth:`~DataArray.cumsum`,
  :py:meth:`~DataArray.cumprod`
  :py:attr:`~DataArray.combine_first`
  :py:meth:`~DataArray.set_index`,
  :py:meth:`~DataArray.reset_index`,
  :py:meth:`~DataArray.reorder_levels`,
  :py:func:`~xarray.full_like`,
  :py:func:`~xarray.zeros_like`,
  :py:func:`~xarray.ones_like`
  :py:func:`~xarray.open_dataarray`,
  :py:meth:`~DataArray.compute`,
  :py:meth:`Dataset.info`,
  :py:func:`testing.assert_equal`,
  :py:func:`testing.assert_identical`, and
  :py:func:`testing.assert_allclose`.

Breaking changes
~~~~~~~~~~~~~~~~

- Index coordinates for each dimensions are now optional, and no longer created
  by default :issue:`1017`. You can identify such dimensions without coordinates
  by their appearance in list of "Dimensions without coordinates" in the
  ``Dataset`` or ``DataArray`` repr:

  .. ipython::
    :verbatim:

    In [1]: xr.Dataset({'foo': (('x', 'y'), [[1, 2]])})
    Out[1]:
    <xarray.Dataset>
    Dimensions:  (x: 1, y: 2)
    Dimensions without coordinates: x, y
    Data variables:
        foo      (x, y) int64 1 2

  This has a number of implications:

  - :py:func:`~align` and :py:meth:`~Dataset.reindex` can now error, if
    dimensions labels are missing and dimensions have different sizes.
  - Because pandas does not support missing indexes, methods such as
    ``to_dataframe``/``from_dataframe`` and ``stack``/``unstack`` no longer
    roundtrip faithfully on all inputs. Use :py:meth:`~Dataset.reset_index` to
    remove undesired indexes.
  - ``Dataset.__delitem__`` and :py:meth:`~Dataset.drop` no longer delete/drop
    variables that have dimensions matching a deleted/dropped variable.
  - ``DataArray.coords.__delitem__`` is now allowed on variables matching
    dimension names.
  - ``.sel`` and ``.loc`` now handle indexing along a dimension without
    coordinate labels by doing integer based indexing. See
    :ref:`indexing.missing_coordinates` for an example.
  - :py:attr:`~Dataset.indexes` is no longer guaranteed to include all
    dimensions names as keys. The new method :py:meth:`~Dataset.get_index` has
    been added to get an index for a dimension guaranteed, falling back to
    produce a default ``RangeIndex`` if necessary.

- The default behavior of ``merge`` is now ``compat='no_conflicts'``, so some
  merges will now succeed in cases that previously raised
  ``xarray.MergeError``. Set ``compat='broadcast_equals'`` to restore the
  previous default. See :ref:`combining.no_conflicts` for more details.

- Reading :py:attr:`~DataArray.values` no longer always caches values in a NumPy
  array :issue:`1128`. Caching of ``.values`` on variables read from netCDF
  files on disk is still the default when :py:func:`open_dataset` is called with
  ``cache=True``.
  By `Guido Imperiale <https://github.com/crusaderky>`_ and
  `Stephan Hoyer <https://github.com/shoyer>`_.
- Pickling a ``Dataset`` or ``DataArray`` linked to a file on disk no longer
  caches its values into memory before pickling (:issue:`1128`). Instead, pickle
  stores file paths and restores objects by reopening file references. This
  enables preliminary, experimental use of xarray for opening files with
  `dask.distributed <https://distributed.readthedocs.io>`_.
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Coordinates used to index a dimension are now loaded eagerly into
  :py:class:`pandas.Index` objects, instead of loading the values lazily.
  By `Guido Imperiale <https://github.com/crusaderky>`_.
- Automatic levels for 2d plots are now guaranteed to land on ``vmin`` and
  ``vmax`` when these kwargs are explicitly provided (:issue:`1191`). The
  automated level selection logic also slightly changed.
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- ``DataArray.rename()`` behavior changed to strictly change the ``DataArray.name``
  if called with string argument, or strictly change coordinate names if called with
  dict-like argument.
  By `Markus Gonser <https://github.com/magonser>`_.

- By default ``to_netcdf()`` add a ``_FillValue = NaN`` attributes to float types.
  By `Frederic Laliberte <https://github.com/laliberte>`_.

- ``repr`` on ``DataArray`` objects uses an shortened display for NumPy array
  data that is less likely to overflow onto multiple pages (:issue:`1207`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- xarray no longer supports python 3.3, versions of dask prior to v0.9.0,
  or versions of bottleneck prior to v1.0.

Deprecations
~~~~~~~~~~~~

- Renamed the ``Coordinate`` class from xarray's low level API to
  :py:class:`~xarray.IndexVariable`. ``Variable.to_variable`` and
  ``Variable.to_coord`` have been renamed to
  :py:meth:`~xarray.Variable.to_base_variable` and
  :py:meth:`~xarray.Variable.to_index_variable`.
- Deprecated supplying ``coords`` as a dictionary to the ``DataArray``
  constructor without also supplying an explicit ``dims`` argument. The old
  behavior encouraged relying on the iteration order of dictionaries, which is
  a bad practice (:issue:`727`).
- Removed a number of methods deprecated since v0.7.0 or earlier:
  ``load_data``, ``vars``, ``drop_vars``, ``dump``, ``dumps`` and the
  ``variables`` keyword argument to ``Dataset``.
- Removed the dummy module that enabled ``import xray``.

Enhancements
~~~~~~~~~~~~

- Added new method :py:meth:`~DataArray.combine_first` to ``DataArray`` and
  ``Dataset``, based on the pandas method of the same name (see :ref:`combine`).
  By `Chun-Wei Yuan <https://github.com/chunweiyuan>`_.

- Added the ability to change default automatic alignment (arithmetic_join="inner")
  for binary operations via :py:func:`~xarray.set_options()`
  (see :ref:`math automatic alignment`).
  By `Chun-Wei Yuan <https://github.com/chunweiyuan>`_.

- Add checking of ``attr`` names and values when saving to netCDF, raising useful
  error messages if they are invalid. (:issue:`911`).
  By `Robin Wilson <https://github.com/robintw>`_.
- Added ability to save ``DataArray`` objects directly to netCDF files using
  :py:meth:`~xarray.DataArray.to_netcdf`, and to load directly from netCDF files
  using :py:func:`~xarray.open_dataarray` (:issue:`915`). These remove the need
  to convert a ``DataArray`` to a ``Dataset`` before saving as a netCDF file,
  and deals with names to ensure a perfect 'roundtrip' capability.
  By `Robin Wilson <https://github.com/robintw>`_.
- Multi-index levels are now accessible as "virtual" coordinate variables,
  e.g., ``ds['time']`` can pull out the ``'time'`` level of a multi-index
  (see :ref:`coordinates`). ``sel`` also accepts providing multi-index levels
  as keyword arguments, e.g., ``ds.sel(time='2000-01')``
  (see :ref:`multi-level indexing`).
  By `Benoit Bovy <https://github.com/benbovy>`_.
- Added ``set_index``, ``reset_index`` and ``reorder_levels`` methods to
  easily create and manipulate (multi-)indexes (see :ref:`reshape.set_index`).
  By `Benoit Bovy <https://github.com/benbovy>`_.
- Added the ``compat`` option ``'no_conflicts'`` to ``merge``, allowing the
  combination of xarray objects with disjoint (:issue:`742`) or
  overlapping (:issue:`835`) coordinates as long as all present data agrees.
  By `Johnnie Gray <https://github.com/jcmgray>`_. See
  :ref:`combining.no_conflicts` for more details.
- It is now possible to set ``concat_dim=None`` explicitly in
  :py:func:`~xarray.open_mfdataset` to disable inferring a dimension along
  which to concatenate.
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Added methods :py:meth:`DataArray.compute`, :py:meth:`Dataset.compute`, and
  :py:meth:`Variable.compute` as a non-mutating alternative to
  :py:meth:`~DataArray.load`.
  By `Guido Imperiale <https://github.com/crusaderky>`_.
- Adds DataArray and Dataset methods :py:meth:`~xarray.DataArray.cumsum` and
  :py:meth:`~xarray.DataArray.cumprod`.  By `Phillip J. Wolfram
  <https://github.com/pwolfram>`_.

- New properties :py:attr:`Dataset.sizes` and :py:attr:`DataArray.sizes` for
  providing consistent access to dimension length on both ``Dataset`` and
  ``DataArray`` (:issue:`921`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- New keyword argument ``drop=True`` for :py:meth:`~DataArray.sel`,
  :py:meth:`~DataArray.isel` and :py:meth:`~DataArray.squeeze` for dropping
  scalar coordinates that arise from indexing.
  ``DataArray`` (:issue:`242`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- New top-level functions :py:func:`~xarray.full_like`,
  :py:func:`~xarray.zeros_like`, and :py:func:`~xarray.ones_like`
  By `Guido Imperiale <https://github.com/crusaderky>`_.
- Overriding a preexisting attribute with
  :py:func:`~xarray.register_dataset_accessor` or
  :py:func:`~xarray.register_dataarray_accessor` now issues a warning instead of
  raising an error (:issue:`1082`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Options for axes sharing between subplots are exposed to
  :py:class:`~xarray.plot.FacetGrid` and :py:func:`~xarray.plot.plot`, so axes
  sharing can be disabled for polar plots.
  By `Bas Hoonhout <https://github.com/hoonhout>`_.
- New utility functions :py:func:`~xarray.testing.assert_equal`,
  :py:func:`~xarray.testing.assert_identical`, and
  :py:func:`~xarray.testing.assert_allclose` for asserting relationships
  between xarray objects, designed for use in a pytest test suite.
- ``figsize``, ``size`` and ``aspect`` plot arguments are now supported for all
  plots (:issue:`897`). See :ref:`plotting.figsize` for more details.
  By `Stephan Hoyer <https://github.com/shoyer>`_ and
  `Fabien Maussion <https://github.com/fmaussion>`_.
- New :py:meth:`~Dataset.info` method to summarize ``Dataset`` variables
  and attributes. The method prints to a buffer (e.g. ``stdout``) with output
  similar to what the command line utility ``ncdump -h`` produces (:issue:`1150`).
  By `Joe Hamman <https://github.com/jhamman>`_.
- Added the ability write unlimited netCDF dimensions with the ``scipy`` and
  ``netcdf4`` backends via the new ``xray.Dataset.encoding`` attribute
  or via the ``unlimited_dims`` argument to ``xray.Dataset.to_netcdf``.
  By `Joe Hamman <https://github.com/jhamman>`_.
- New :py:meth:`~DataArray.quantile` method to calculate quantiles from
  DataArray objects (:issue:`1187`).
  By `Joe Hamman <https://github.com/jhamman>`_.


Bug fixes
~~~~~~~~~
- ``groupby_bins`` now restores empty bins by default (:issue:`1019`).
  By `Ryan Abernathey <https://github.com/rabernat>`_.

- Fix issues for dates outside the valid range of pandas timestamps
  (:issue:`975`). By `Mathias Hauser <https://github.com/mathause>`_.

- Unstacking produced flipped array after stacking decreasing coordinate values
  (:issue:`980`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Setting ``dtype`` via the ``encoding`` parameter of ``to_netcdf`` failed if
  the encoded dtype was the same as the dtype of the original array
  (:issue:`873`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix issues with variables where both attributes ``_FillValue`` and
  ``missing_value`` are set to ``NaN`` (:issue:`997`).
  By `Marco Zühlke <https://github.com/mzuehlke>`_.

- ``.where()`` and ``.fillna()`` now preserve attributes (:issue:`1009`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- Applying :py:func:`broadcast()` to an xarray object based on the dask backend
  won't accidentally convert the array from dask to numpy anymore (:issue:`978`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.

- ``Dataset.concat()`` now preserves variables order (:issue:`1027`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- Fixed an issue with pcolormesh (:issue:`781`). A new
  ``infer_intervals`` keyword gives control on whether the cell intervals
  should be computed or not.
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- Grouping over an dimension with non-unique values with ``groupby`` gives
  correct groups.
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed accessing coordinate variables with non-string names from ``.coords``.
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- :py:meth:`~xarray.DataArray.rename` now simultaneously renames the array and
  any coordinate with the same name, when supplied via a :py:class:`dict`
  (:issue:`1116`).
  By `Yves Delley <https://github.com/burnpanck>`_.

- Fixed sub-optimal performance in certain operations with object arrays (:issue:`1121`).
  By `Yves Delley <https://github.com/burnpanck>`_.

- Fix ``.groupby(group)`` when ``group`` has datetime dtype (:issue:`1132`).
  By `Jonas Sølvsteen <https://github.com/j08lue>`_.

- Fixed a bug with facetgrid (the ``norm`` keyword was ignored, :issue:`1159`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- Resolved a concurrency bug that could cause Python to crash when
  simultaneously reading and writing netCDF4 files with dask (:issue:`1172`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix to make ``.copy()`` actually copy dask arrays, which will be relevant for
  future releases of dask in which dask arrays will be mutable (:issue:`1180`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix opening NetCDF files with multi-dimensional time variables
  (:issue:`1229`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

Performance improvements
~~~~~~~~~~~~~~~~~~~~~~~~

- ``xarray.Dataset.isel_points`` and ``xarray.Dataset.sel_points`` now
  use vectorised indexing in numpy and dask (:issue:`1161`), which can
  result in several orders of magnitude speedup.
  By `Jonathan Chambers <https://github.com/mangecoeur>`_.

.. _whats-new.0.8.2:

v0.8.2 (18 August 2016)
-----------------------

This release includes a number of bug fixes and minor enhancements.

Breaking changes
~~~~~~~~~~~~~~~~

- :py:func:`~xarray.broadcast` and :py:func:`~xarray.concat` now auto-align
  inputs, using ``join=outer``. Previously, these functions raised
  ``ValueError`` for non-aligned inputs.
  By `Guido Imperiale <https://github.com/crusaderky>`_.

Enhancements
~~~~~~~~~~~~

- New documentation on :ref:`panel transition`. By
  `Maximilian Roos <https://github.com/max-sixty>`_.
- New ``Dataset`` and ``DataArray`` methods :py:meth:`~xarray.Dataset.to_dict`
  and :py:meth:`~xarray.Dataset.from_dict` to allow easy conversion between
  dictionaries and xarray objects (:issue:`432`). See
  :ref:`dictionary IO<dictionary io>` for more details.
  By `Julia Signell <https://github.com/jsignell>`_.
- Added ``exclude`` and ``indexes`` optional parameters to :py:func:`~xarray.align`,
  and ``exclude`` optional parameter to :py:func:`~xarray.broadcast`.
  By `Guido Imperiale <https://github.com/crusaderky>`_.
- Better error message when assigning variables without dimensions
  (:issue:`971`). By `Stephan Hoyer <https://github.com/shoyer>`_.
- Better error message when reindex/align fails due to duplicate index values
  (:issue:`956`). By `Stephan Hoyer <https://github.com/shoyer>`_.

Bug fixes
~~~~~~~~~

- Ensure xarray works with h5netcdf v0.3.0 for arrays with ``dtype=str``
  (:issue:`953`). By `Stephan Hoyer <https://github.com/shoyer>`_.
- ``Dataset.__dir__()`` (i.e. the method python calls to get autocomplete
  options) failed if one of the dataset's keys was not a string (:issue:`852`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- ``Dataset`` constructor can now take arbitrary objects as values
  (:issue:`647`). By `Maximilian Roos <https://github.com/max-sixty>`_.
- Clarified ``copy`` argument for :py:meth:`~xarray.DataArray.reindex` and
  :py:func:`~xarray.align`, which now consistently always return new xarray
  objects (:issue:`927`).
- Fix ``open_mfdataset`` with ``engine='pynio'`` (:issue:`936`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- ``groupby_bins`` sorted bin labels as strings (:issue:`952`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Fix bug introduced by v0.8.0 that broke assignment to datasets when both the
  left and right side have the same non-unique index values (:issue:`956`).

.. _whats-new.0.8.1:

v0.8.1 (5 August 2016)
----------------------

Bug fixes
~~~~~~~~~

- Fix bug in v0.8.0 that broke assignment to Datasets with non-unique
  indexes (:issue:`943`). By `Stephan Hoyer <https://github.com/shoyer>`_.

.. _whats-new.0.8.0:

v0.8.0 (2 August 2016)
----------------------

This release includes four months of new features and bug fixes, including
several breaking changes.

.. _v0.8.0.breaking:

Breaking changes
~~~~~~~~~~~~~~~~

- Dropped support for Python 2.6 (:issue:`855`).
- Indexing on multi-index now drop levels, which is consistent with pandas.
  It also changes the name of the dimension / coordinate when the multi-index is
  reduced to a single index (:issue:`802`).
- Contour plots no longer add a colorbar per default (:issue:`866`). Filled
  contour plots are unchanged.
- ``DataArray.values`` and ``.data`` now always returns an NumPy array-like
  object, even for 0-dimensional arrays with object dtype (:issue:`867`).
  Previously, ``.values`` returned native Python objects in such cases. To
  convert the values of scalar arrays to Python objects, use the ``.item()``
  method.

Enhancements
~~~~~~~~~~~~

- Groupby operations now support grouping over multidimensional variables. A new
  method called :py:meth:`~xarray.Dataset.groupby_bins` has also been added to
  allow users to specify bins for grouping. The new features are described in
  :ref:`groupby.multidim` and :ref:`/examples/multidimensional-coords.ipynb`.
  By `Ryan Abernathey <https://github.com/rabernat>`_.

- DataArray and Dataset method :py:meth:`where` now supports a ``drop=True``
  option that clips coordinate elements that are fully masked.  By
  `Phillip J. Wolfram <https://github.com/pwolfram>`_.

- New top level :py:func:`merge` function allows for combining variables from
  any number of ``Dataset`` and/or ``DataArray`` variables. See :ref:`merge`
  for more details. By `Stephan Hoyer <https://github.com/shoyer>`_.

- :py:meth:`DataArray.resample` and :py:meth:`Dataset.resample` now support the
  ``keep_attrs=False`` option that determines whether variable and dataset
  attributes are retained in the resampled object. By
  `Jeremy McGibbon <https://github.com/mcgibbon>`_.

- Better multi-index support in :py:meth:`DataArray.sel`,
  :py:meth:`DataArray.loc`, :py:meth:`Dataset.sel` and
  :py:meth:`Dataset.loc`, which now behave more closely to pandas and
  which also accept dictionaries for indexing based on given level names
  and labels (see :ref:`multi-level indexing`).
  By `Benoit Bovy <https://github.com/benbovy>`_.

- New (experimental) decorators :py:func:`~xarray.register_dataset_accessor` and
  :py:func:`~xarray.register_dataarray_accessor` for registering custom xarray
  extensions without subclassing. They are described in the new documentation
  page on :ref:`internals`. By `Stephan Hoyer <https://github.com/shoyer>`_.

- Round trip boolean datatypes. Previously, writing boolean datatypes to netCDF
  formats would raise an error since netCDF does not have a `bool` datatype.
  This feature reads/writes a `dtype` attribute to boolean variables in netCDF
  files. By `Joe Hamman <https://github.com/jhamman>`_.

- 2D plotting methods now have two new keywords (`cbar_ax` and `cbar_kwargs`),
  allowing more control on the colorbar (:issue:`872`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- New Dataset method :py:meth:`Dataset.filter_by_attrs`, akin to
  ``netCDF4.Dataset.get_variables_by_attributes``, to easily filter
  data variables using its attributes.
  `Filipe Fernandes <https://github.com/ocefpaf>`_.

Bug fixes
~~~~~~~~~

- Attributes were being retained by default for some resampling
  operations when they should not. With the ``keep_attrs=False`` option, they
  will no longer be retained by default. This may be backwards-incompatible
  with some scripts, but the attributes may be kept by adding the
  ``keep_attrs=True`` option. By
  `Jeremy McGibbon <https://github.com/mcgibbon>`_.

- Concatenating xarray objects along an axis with a MultiIndex or PeriodIndex
  preserves the nature of the index (:issue:`875`). By
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed bug in arithmetic operations on DataArray objects whose dimensions
  are numpy structured arrays or recarrays :issue:`861`, :issue:`837`. By
  `Maciek Swat <https://github.com/maciekswat>`_.

- ``decode_cf_timedelta`` now accepts arrays with ``ndim`` >1 (:issue:`842`).
   This fixes issue :issue:`665`.
   `Filipe Fernandes <https://github.com/ocefpaf>`_.

- Fix a bug where `xarray.ufuncs` that take two arguments would incorrectly
  use to numpy functions instead of dask.array functions (:issue:`876`). By
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Support for pickling functions from  ``xarray.ufuncs`` (:issue:`901`). By
  `Stephan Hoyer <https://github.com/shoyer>`_.

- ``Variable.copy(deep=True)`` no longer converts MultiIndex into a base Index
  (:issue:`769`). By `Benoit Bovy <https://github.com/benbovy>`_.

- Fixes for groupby on dimensions with a multi-index (:issue:`867`). By
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix printing datasets with unicode attributes on Python 2 (:issue:`892`). By
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed incorrect test for dask version (:issue:`891`). By
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed `dim` argument for `isel_points`/`sel_points` when a `pandas.Index` is
  passed. By `Stephan Hoyer <https://github.com/shoyer>`_.

- :py:func:`~xarray.plot.contour` now plots the correct number of contours
  (:issue:`866`). By `Fabien Maussion <https://github.com/fmaussion>`_.

.. _whats-new.0.7.2:

v0.7.2 (13 March 2016)
----------------------

This release includes two new, entirely backwards compatible features and
several bug fixes.

Enhancements
~~~~~~~~~~~~

- New DataArray method :py:meth:`DataArray.dot` for calculating the dot
  product of two DataArrays along shared dimensions. By
  `Dean Pospisil <https://github.com/deanpospisil>`_.

- Rolling window operations on DataArray objects are now supported via a new
  :py:meth:`DataArray.rolling` method. For example:

  .. ipython::
    :verbatim:

    In [1]: import xarray as xr; import numpy as np

    In [2]: arr = xr.DataArray(np.arange(0, 7.5, 0.5).reshape(3, 5),
                               dims=('x', 'y'))

    In [3]: arr
    Out[3]:
    <xarray.DataArray (x: 3, y: 5)>
    array([[ 0. ,  0.5,  1. ,  1.5,  2. ],
           [ 2.5,  3. ,  3.5,  4. ,  4.5],
           [ 5. ,  5.5,  6. ,  6.5,  7. ]])
    Coordinates:
      * x        (x) int64 0 1 2
      * y        (y) int64 0 1 2 3 4

    In [4]: arr.rolling(y=3, min_periods=2).mean()
    Out[4]:
    <xarray.DataArray (x: 3, y: 5)>
    array([[  nan,  0.25,  0.5 ,  1.  ,  1.5 ],
           [  nan,  2.75,  3.  ,  3.5 ,  4.  ],
           [  nan,  5.25,  5.5 ,  6.  ,  6.5 ]])
    Coordinates:
      * x        (x) int64 0 1 2
      * y        (y) int64 0 1 2 3 4

  See :ref:`comput.rolling` for more details. By
  `Joe Hamman <https://github.com/jhamman>`_.

Bug fixes
~~~~~~~~~

- Fixed an issue where plots using pcolormesh and Cartopy axes were being distorted
  by the inference of the axis interval breaks. This change chooses not to modify
  the coordinate variables when the axes have the attribute ``projection``, allowing
  Cartopy to handle the extent of pcolormesh plots (:issue:`781`). By
  `Joe Hamman <https://github.com/jhamman>`_.

- 2D plots now better handle additional coordinates which are not ``DataArray``
  dimensions (:issue:`788`). By `Fabien Maussion <https://github.com/fmaussion>`_.


.. _whats-new.0.7.1:

v0.7.1 (16 February 2016)
-------------------------

This is a bug fix release that includes two small, backwards compatible enhancements.
We recommend that all users upgrade.

Enhancements
~~~~~~~~~~~~

- Numerical operations now return empty objects on no overlapping labels rather
  than raising ``ValueError`` (:issue:`739`).
- :py:class:`~pandas.Series` is now supported as valid input to the ``Dataset``
  constructor (:issue:`740`).

Bug fixes
~~~~~~~~~

- Restore checks for shape consistency between data and coordinates in the
  DataArray constructor (:issue:`758`).
- Single dimension variables no longer transpose as part of a broader
  ``.transpose``. This  behavior was causing ``pandas.PeriodIndex`` dimensions
  to lose their type (:issue:`749`)
- :py:class:`~xarray.Dataset` labels remain as their native type on ``.to_dataset``.
  Previously they were coerced to strings (:issue:`745`)
- Fixed a bug where replacing a ``DataArray`` index coordinate would improperly
  align the coordinate (:issue:`725`).
- ``DataArray.reindex_like`` now maintains the dtype of complex numbers when
  reindexing leads to NaN values (:issue:`738`).
- ``Dataset.rename`` and ``DataArray.rename`` support the old and new names
  being the same (:issue:`724`).
- Fix :py:meth:`~xarray.Dataset.from_dataframe` for DataFrames with Categorical
  column and a MultiIndex index (:issue:`737`).
- Fixes to ensure xarray works properly after the upcoming pandas v0.18 and
  NumPy v1.11 releases.

Acknowledgments
~~~~~~~~~~~~~~~

The following individuals contributed to this release:

- Edward Richards
- Maximilian Roos
- Rafael Guedes
- Spencer Hill
- Stephan Hoyer

.. _whats-new.0.7.0:

v0.7.0 (21 January 2016)
------------------------

This major release includes redesign of :py:class:`~xarray.DataArray`
internals, as well as new methods for reshaping, rolling and shifting
data. It includes preliminary support for :py:class:`pandas.MultiIndex`,
as well as a number of other features and bug fixes, several of which
offer improved compatibility with pandas.

New name
~~~~~~~~

The project formerly known as "xray" is now "xarray", pronounced "x-array"!
This avoids a namespace conflict with the entire field of x-ray science. Renaming
our project seemed like the right thing to do, especially because some
scientists who work with actual x-rays are interested in using this project in
their work. Thanks for your understanding and patience in this transition. You
can now find our documentation and code repository at new URLs:

- http://xarray.pydata.org
- http://github.com/pydata/xarray/

To ease the transition, we have simultaneously released v0.7.0 of both
``xray`` and ``xarray`` on the Python Package Index. These packages are
identical. For now, ``import xray`` still works, except it issues a
deprecation warning. This will be the last xray release. Going forward, we
recommend switching your import statements to ``import xarray as xr``.

.. _v0.7.0.breaking:

Breaking changes
~~~~~~~~~~~~~~~~

- The internal data model used by ``xray.DataArray`` has been
  rewritten to fix several outstanding issues (:issue:`367`, :issue:`634`,
  `this stackoverflow report`_). Internally, ``DataArray`` is now implemented
  in terms of ``._variable`` and ``._coords`` attributes instead of holding
  variables in a ``Dataset`` object.

  This refactor ensures that if a DataArray has the
  same name as one of its coordinates, the array and the coordinate no longer
  share the same data.

  In practice, this means that creating a DataArray with the same ``name`` as
  one of its dimensions no longer automatically uses that array to label the
  corresponding coordinate. You will now need to provide coordinate labels
  explicitly. Here's the old behavior:

  .. ipython::
    :verbatim:

    In [2]: xray.DataArray([4, 5, 6], dims='x', name='x')
    Out[2]:
    <xray.DataArray 'x' (x: 3)>
    array([4, 5, 6])
    Coordinates:
      * x        (x) int64 4 5 6

  and the new behavior (compare the values of the ``x`` coordinate):

  .. ipython::
    :verbatim:

    In [2]: xray.DataArray([4, 5, 6], dims='x', name='x')
    Out[2]:
    <xray.DataArray 'x' (x: 3)>
    array([4, 5, 6])
    Coordinates:
      * x        (x) int64 0 1 2

- It is no longer possible to convert a DataArray to a Dataset with
  ``xray.DataArray.to_dataset`` if it is unnamed. This will now
  raise ``ValueError``. If the array is unnamed, you need to supply the
  ``name`` argument.

.. _this stackoverflow report: http://stackoverflow.com/questions/33158558/python-xray-extract-first-and-last-time-value-within-each-month-of-a-timeseries

Enhancements
~~~~~~~~~~~~

- Basic support for :py:class:`~pandas.MultiIndex` coordinates on xray objects, including
  indexing, :py:meth:`~DataArray.stack` and :py:meth:`~DataArray.unstack`:

  .. ipython::
    :verbatim:

    In [7]: df = pd.DataFrame({'foo': range(3),
       ...:                    'x': ['a', 'b', 'b'],
       ...:                    'y': [0, 0, 1]})

    In [8]: s = df.set_index(['x', 'y'])['foo']

    In [12]: arr = xray.DataArray(s, dims='z')

    In [13]: arr
    Out[13]:
    <xray.DataArray 'foo' (z: 3)>
    array([0, 1, 2])
    Coordinates:
      * z        (z) object ('a', 0) ('b', 0) ('b', 1)

    In [19]: arr.indexes['z']
    Out[19]:
    MultiIndex(levels=[[u'a', u'b'], [0, 1]],
               labels=[[0, 1, 1], [0, 0, 1]],
               names=[u'x', u'y'])

    In [14]: arr.unstack('z')
    Out[14]:
    <xray.DataArray 'foo' (x: 2, y: 2)>
    array([[  0.,  nan],
           [  1.,   2.]])
    Coordinates:
      * x        (x) object 'a' 'b'
      * y        (y) int64 0 1

    In [26]: arr.unstack('z').stack(z=('x', 'y'))
    Out[26]:
    <xray.DataArray 'foo' (z: 4)>
    array([  0.,  nan,   1.,   2.])
    Coordinates:
      * z        (z) object ('a', 0) ('a', 1) ('b', 0) ('b', 1)

  See :ref:`reshape.stack` for more details.

  .. warning::

      xray's MultiIndex support is still experimental, and we have a long to-
      do list of desired additions (:issue:`719`), including better display of
      multi-index levels when printing a ``Dataset``, and support for saving
      datasets with a MultiIndex to a netCDF file. User contributions in this
      area would be greatly appreciated.

- Support for reading GRIB, HDF4 and other file formats via PyNIO_. See
  :ref:`io.pynio` for more details.
- Better error message when a variable is supplied with the same name as
  one of its dimensions.
- Plotting: more control on colormap parameters (:issue:`642`). ``vmin`` and
  ``vmax`` will not be silently ignored anymore. Setting ``center=False``
  prevents automatic selection of a divergent colormap.
- New ``xray.Dataset.shift`` and ``xray.Dataset.roll`` methods
  for shifting/rotating datasets or arrays along a dimension:

  .. ipython:: python
     :okwarning:

      array = xray.DataArray([5, 6, 7, 8], dims='x')
      array.shift(x=2)
      array.roll(x=2)

  Notice that ``shift`` moves data independently of coordinates, but ``roll``
  moves both data and coordinates.
- Assigning a ``pandas`` object directly as a ``Dataset`` variable is now permitted. Its
  index names correspond to the ``dims`` of the ``Dataset``, and its data is aligned.
- Passing a :py:class:`pandas.DataFrame` or ``pandas.Panel`` to a Dataset constructor
  is now permitted.
- New function ``xray.broadcast`` for explicitly broadcasting
  ``DataArray`` and ``Dataset`` objects against each other. For example:

  .. ipython:: python

      a = xray.DataArray([1, 2, 3], dims='x')
      b = xray.DataArray([5, 6], dims='y')
      a
      b
      a2, b2 = xray.broadcast(a, b)
      a2
      b2

.. _PyNIO: https://www.pyngl.ucar.edu/Nio.shtml

Bug fixes
~~~~~~~~~

- Fixes for several issues found on ``DataArray`` objects with the same name
  as one of their coordinates (see :ref:`v0.7.0.breaking` for more details).
- ``DataArray.to_masked_array`` always returns masked array with mask being an
  array (not a scalar value) (:issue:`684`)
- Allows for (imperfect) repr of Coords when underlying index is PeriodIndex (:issue:`645`).
- Fixes for several issues found on ``DataArray`` objects with the same name
  as one of their coordinates (see :ref:`v0.7.0.breaking` for more details).
- Attempting to assign a ``Dataset`` or ``DataArray`` variable/attribute using
  attribute-style syntax (e.g., ``ds.foo = 42``) now raises an error rather
  than silently failing (:issue:`656`, :issue:`714`).
- You can now pass pandas objects with non-numpy dtypes (e.g., ``categorical``
  or ``datetime64`` with a timezone) into xray without an error
  (:issue:`716`).

Acknowledgments
~~~~~~~~~~~~~~~

The following individuals contributed to this release:

- Antony Lee
- Fabien Maussion
- Joe Hamman
- Maximilian Roos
- Stephan Hoyer
- Takeshi Kanmae
- femtotrader

v0.6.1 (21 October 2015)
------------------------

This release contains a number of bug and compatibility fixes, as well
as enhancements to plotting, indexing and writing files to disk.

Note that the minimum required version of dask for use with xray is now
version 0.6.

API Changes
~~~~~~~~~~~

- The handling of colormaps and discrete color lists for 2D plots in
  ``xray.DataArray.plot`` was changed to provide more compatibility
  with matplotlib's ``contour`` and ``contourf`` functions (:issue:`538`).
  Now discrete lists of colors should be specified using ``colors`` keyword,
  rather than ``cmap``.

Enhancements
~~~~~~~~~~~~

- Faceted plotting through ``xray.plot.FacetGrid`` and the
  ``xray.plot.plot`` method. See :ref:`plotting.faceting` for more details
  and examples.
- ``xray.Dataset.sel`` and ``xray.Dataset.reindex`` now support
  the ``tolerance`` argument for controlling nearest-neighbor selection
  (:issue:`629`):

  .. ipython::
    :verbatim:

    In [5]: array = xray.DataArray([1, 2, 3], dims='x')

    In [6]: array.reindex(x=[0.9, 1.5], method='nearest', tolerance=0.2)
    Out[6]:
    <xray.DataArray (x: 2)>
    array([  2.,  nan])
    Coordinates:
      * x        (x) float64 0.9 1.5

  This feature requires pandas v0.17 or newer.
- New ``encoding`` argument in ``xray.Dataset.to_netcdf`` for writing
  netCDF files with compression, as described in the new documentation
  section on :ref:`io.netcdf.writing_encoded`.
- Add ``xray.Dataset.real`` and ``xray.Dataset.imag``
  attributes to Dataset and DataArray (:issue:`553`).
- More informative error message with ``xray.Dataset.from_dataframe``
  if the frame has duplicate columns.
- xray now uses deterministic names for dask arrays it creates or opens from
  disk. This allows xray users to take advantage of dask's nascent support for
  caching intermediate computation results. See :issue:`555` for an example.

Bug fixes
~~~~~~~~~

- Forwards compatibility with the latest pandas release (v0.17.0). We were
  using some internal pandas routines for datetime conversion, which
  unfortunately have now changed upstream (:issue:`569`).
- Aggregation functions now correctly skip ``NaN`` for data for ``complex128``
  dtype (:issue:`554`).
- Fixed indexing 0d arrays with unicode dtype (:issue:`568`).
- ``xray.DataArray.name`` and Dataset keys must be a string or None to
  be written to netCDF (:issue:`533`).
- ``xray.DataArray.where`` now uses dask instead of numpy if either the
  array or ``other`` is a dask array. Previously, if ``other`` was a numpy array
  the method was evaluated eagerly.
- Global attributes are now handled more consistently when loading remote
  datasets using ``engine='pydap'`` (:issue:`574`).
- It is now possible to assign to the ``.data`` attribute of DataArray objects.
- ``coordinates`` attribute is now kept in the encoding dictionary after
  decoding (:issue:`610`).
- Compatibility with numpy 1.10 (:issue:`617`).

Acknowledgments
~~~~~~~~~~~~~~~

The following individuals contributed to this release:

- Ryan Abernathey
- Pete Cable
- Clark Fitzgerald
- Joe Hamman
- Stephan Hoyer
- Scott Sinclair

v0.6.0 (21 August 2015)
-----------------------

This release includes numerous bug fixes and enhancements. Highlights
include the introduction of a plotting module and the new Dataset and DataArray
methods ``xray.Dataset.isel_points``, ``xray.Dataset.sel_points``,
``xray.Dataset.where`` and ``xray.Dataset.diff``. There are no
breaking changes from v0.5.2.

Enhancements
~~~~~~~~~~~~

- Plotting methods have been implemented on DataArray objects
  ``xray.DataArray.plot`` through integration with matplotlib
  (:issue:`185`). For an introduction, see :ref:`plotting`.
- Variables in netCDF files with multiple missing values are now decoded as NaN
  after issuing a warning if open_dataset is called with mask_and_scale=True.
- We clarified our rules for when the result from an xray operation is a copy
  vs. a view (see :ref:`copies_vs_views` for more details).
- Dataset variables are now written to netCDF files in order of appearance
  when using the netcdf4 backend (:issue:`479`).

- Added ``xray.Dataset.isel_points`` and ``xray.Dataset.sel_points``
  to support pointwise indexing of Datasets and DataArrays (:issue:`475`).

  .. ipython::
    :verbatim:

    In [1]: da = xray.DataArray(np.arange(56).reshape((7, 8)),
       ...:                     coords={'x': list('abcdefg'),
       ...:                             'y': 10 * np.arange(8)},
       ...:                     dims=['x', 'y'])

    In [2]: da
    Out[2]:
    <xray.DataArray (x: 7, y: 8)>
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29, 30, 31],
           [32, 33, 34, 35, 36, 37, 38, 39],
           [40, 41, 42, 43, 44, 45, 46, 47],
           [48, 49, 50, 51, 52, 53, 54, 55]])
    Coordinates:
    * y        (y) int64 0 10 20 30 40 50 60 70
    * x        (x) |S1 'a' 'b' 'c' 'd' 'e' 'f' 'g'

    # we can index by position along each dimension
    In [3]: da.isel_points(x=[0, 1, 6], y=[0, 1, 0], dim='points')
    Out[3]:
    <xray.DataArray (points: 3)>
    array([ 0,  9, 48])
    Coordinates:
        y        (points) int64 0 10 0
        x        (points) |S1 'a' 'b' 'g'
      * points   (points) int64 0 1 2

    # or equivalently by label
    In [9]: da.sel_points(x=['a', 'b', 'g'], y=[0, 10, 0], dim='points')
    Out[9]:
    <xray.DataArray (points: 3)>
    array([ 0,  9, 48])
    Coordinates:
        y        (points) int64 0 10 0
        x        (points) |S1 'a' 'b' 'g'
      * points   (points) int64 0 1 2

- New ``xray.Dataset.where`` method for masking xray objects according
  to some criteria. This works particularly well with multi-dimensional data:

  .. ipython:: python

    ds = xray.Dataset(coords={'x': range(100), 'y': range(100)})
    ds['distance'] = np.sqrt(ds.x ** 2 + ds.y ** 2)

    @savefig where_example.png width=4in height=4in
    ds.distance.where(ds.distance < 100).plot()

- Added new methods ``xray.DataArray.diff`` and ``xray.Dataset.diff``
  for finite difference calculations along a given axis.

- New ``xray.DataArray.to_masked_array`` convenience method for
  returning a numpy.ma.MaskedArray.

  .. ipython:: python

    da = xray.DataArray(np.random.random_sample(size=(5, 4)))
    da.where(da < 0.5)
    da.where(da < 0.5).to_masked_array(copy=True)

- Added new flag "drop_variables" to ``xray.open_dataset`` for
  excluding variables from being parsed. This may be useful to drop
  variables with problems or inconsistent values.

Bug fixes
~~~~~~~~~

- Fixed aggregation functions (e.g., sum and mean) on big-endian arrays when
  bottleneck is installed (:issue:`489`).
- Dataset aggregation functions dropped variables with unsigned integer dtype
  (:issue:`505`).
- ``.any()`` and ``.all()`` were not lazy when used on xray objects containing
  dask arrays.
- Fixed an error when attempting to saving datetime64 variables to netCDF
  files when the first element is ``NaT`` (:issue:`528`).
- Fix pickle on DataArray objects (:issue:`515`).
- Fixed unnecessary coercion of float64 to float32 when using netcdf3 and
  netcdf4_classic formats (:issue:`526`).

v0.5.2 (16 July 2015)
---------------------

This release contains bug fixes, several additional options for opening and
saving netCDF files, and a backwards incompatible rewrite of the advanced
options for ``xray.concat``.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The optional arguments ``concat_over`` and ``mode`` in ``xray.concat`` have
  been removed and replaced by ``data_vars`` and ``coords``. The new arguments are both
  more easily understood and more robustly implemented, and allowed us to fix a bug
  where ``concat`` accidentally loaded data into memory. If you set values for
  these optional arguments manually, you will need to update your code. The default
  behavior should be unchanged.

Enhancements
~~~~~~~~~~~~

- ``xray.open_mfdataset`` now supports a ``preprocess`` argument for
  preprocessing datasets prior to concatenaton. This is useful if datasets
  cannot be otherwise merged automatically, e.g., if the original datasets
  have conflicting index coordinates (:issue:`443`).
- ``xray.open_dataset`` and ``xray.open_mfdataset`` now use a
  global thread lock by default for reading from netCDF files with dask. This
  avoids possible segmentation faults for reading from netCDF4 files when HDF5
  is not configured properly for concurrent access (:issue:`444`).
- Added support for serializing arrays of complex numbers with `engine='h5netcdf'`.
- The new ``xray.save_mfdataset`` function allows for saving multiple
  datasets to disk simultaneously. This is useful when processing large datasets
  with dask.array. For example, to save a dataset too big to fit into memory
  to one file per year, we could write:

  .. ipython::
    :verbatim:

    In [1]: years, datasets = zip(*ds.groupby('time.year'))

    In [2]: paths = ['%s.nc' % y for y in years]

    In [3]: xray.save_mfdataset(datasets, paths)

Bug fixes
~~~~~~~~~

- Fixed ``min``, ``max``, ``argmin`` and ``argmax`` for arrays with string or
  unicode types (:issue:`453`).
- ``xray.open_dataset`` and ``xray.open_mfdataset`` support
  supplying chunks as a single integer.
- Fixed a bug in serializing scalar datetime variable to netCDF.
- Fixed a bug that could occur in serialization of 0-dimensional integer arrays.
- Fixed a bug where concatenating DataArrays was not always lazy (:issue:`464`).
- When reading datasets with h5netcdf, bytes attributes are decoded to strings.
  This allows conventions decoding to work properly on Python 3 (:issue:`451`).

v0.5.1 (15 June 2015)
---------------------

This minor release fixes a few bugs and an inconsistency with pandas. It also
adds the ``pipe`` method, copied from pandas.

Enhancements
~~~~~~~~~~~~

- Added ``xray.Dataset.pipe``, replicating the `new pandas method`_ in version
  0.16.2. See :ref:`transforming datasets` for more details.
- ``xray.Dataset.assign`` and ``xray.Dataset.assign_coords``
  now assign new variables in sorted (alphabetical) order, mirroring the
  behavior in pandas. Previously, the order was arbitrary.

.. _new pandas method: http://pandas.pydata.org/pandas-docs/version/0.16.2/whatsnew.html#pipe

Bug fixes
~~~~~~~~~

- ``xray.concat`` fails in an edge case involving identical coordinate variables (:issue:`425`)
- We now decode variables loaded from netCDF3 files with the scipy engine using native
  endianness (:issue:`416`). This resolves an issue when aggregating these arrays with
  bottleneck installed.

v0.5 (1 June 2015)
------------------

Highlights
~~~~~~~~~~

The headline feature in this release is experimental support for out-of-core
computing (data that doesn't fit into memory) with dask_. This includes a new
top-level function ``xray.open_mfdataset`` that makes it easy to open
a collection of netCDF (using dask) as a single ``xray.Dataset`` object. For
more on dask, read the `blog post introducing xray + dask`_ and the new
documentation section :doc:`dask`.

.. _blog post introducing xray + dask: https://www.anaconda.com/blog/developer-blog/xray-dask-out-core-labeled-arrays-python/

Dask makes it possible to harness parallelism and manipulate gigantic datasets
with xray. It is currently an optional dependency, but it may become required
in the future.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The logic used for choosing which variables are concatenated with
  ``xray.concat`` has changed. Previously, by default any variables
  which were equal across a dimension were not concatenated. This lead to some
  surprising behavior, where the behavior of groupby and concat operations
  could depend on runtime values (:issue:`268`). For example:

  .. ipython::
    :verbatim:

    In [1]: ds = xray.Dataset({'x': 0})

    In [2]: xray.concat([ds, ds], dim='y')
    Out[2]:
    <xray.Dataset>
    Dimensions:  ()
    Coordinates:
        *empty*
    Data variables:
        x        int64 0

  Now, the default always concatenates data variables:

  .. ipython:: python
    :suppress:

    ds = xray.Dataset({'x': 0})

  .. ipython:: python

    xray.concat([ds, ds], dim='y')

  To obtain the old behavior, supply the argument ``concat_over=[]``.

Enhancements
~~~~~~~~~~~~

- New ``xray.Dataset.to_array`` and enhanced
  ``xray.DataArray.to_dataset`` methods make it easy to switch back
  and forth between arrays and datasets:

  .. ipython:: python

      ds = xray.Dataset({'a': 1, 'b': ('x', [1, 2, 3])},
                        coords={'c': 42}, attrs={'Conventions': 'None'})
      ds.to_array()
      ds.to_array().to_dataset(dim='variable')

- New ``xray.Dataset.fillna`` method to fill missing values, modeled
  off the pandas method of the same name:

  .. ipython:: python

      array = xray.DataArray([np.nan, 1, np.nan, 3], dims='x')
      array.fillna(0)

  ``fillna`` works on both ``Dataset`` and ``DataArray`` objects, and uses
  index based alignment and broadcasting like standard binary operations. It
  also can be applied by group, as illustrated in
  :ref:`/examples/weather-data.ipynb#Fill-missing-values-with-climatology`.
- New ``xray.Dataset.assign`` and ``xray.Dataset.assign_coords``
  methods patterned off the new :py:meth:`DataFrame.assign <pandas.DataFrame.assign>`
  method in pandas:

  .. ipython:: python

      ds = xray.Dataset({'y': ('x', [1, 2, 3])})
      ds.assign(z = lambda ds: ds.y ** 2)
      ds.assign_coords(z = ('x', ['a', 'b', 'c']))

  These methods return a new Dataset (or DataArray) with updated data or
  coordinate variables.
- ``xray.Dataset.sel`` now supports the ``method`` parameter, which works
  like the paramter of the same name on ``xray.Dataset.reindex``. It
  provides a simple interface for doing nearest-neighbor interpolation:

  .. use verbatim because I can't seem to install pandas 0.16.1 on RTD :(

  .. ipython::
      :verbatim:

      In [12]: ds.sel(x=1.1, method='nearest')
      Out[12]:
      <xray.Dataset>
      Dimensions:  ()
      Coordinates:
          x        int64 1
      Data variables:
          y        int64 2

      In [13]: ds.sel(x=[1.1, 2.1], method='pad')
      Out[13]:
      <xray.Dataset>
      Dimensions:  (x: 2)
      Coordinates:
        * x        (x) int64 1 2
      Data variables:
          y        (x) int64 2 3

  See :ref:`nearest neighbor lookups` for more details.
- You can now control the underlying backend used for accessing remote
  datasets (via OPeNDAP) by specifying ``engine='netcdf4'`` or
  ``engine='pydap'``.
- xray now provides experimental support for reading and writing netCDF4 files directly
  via `h5py`_ with the `h5netcdf`_ package, avoiding the netCDF4-Python package. You
  will need to install h5netcdf and specify ``engine='h5netcdf'`` to try this
  feature.
- Accessing data from remote datasets now has retrying logic (with exponential
  backoff) that should make it robust to occasional bad responses from DAP
  servers.
- You can control the width of the Dataset repr with ``xray.set_options``.
  It can be used either as a context manager, in which case the default is restored
  outside the context:

  .. ipython:: python

      ds = xray.Dataset({'x': np.arange(1000)})
      with xray.set_options(display_width=40):
          print(ds)

  Or to set a global option:

  .. ipython::
      :verbatim:

      In [1]: xray.set_options(display_width=80)

  The default value for the ``display_width`` option is 80.

.. _h5py: http://www.h5py.org/
.. _h5netcdf: https://github.com/shoyer/h5netcdf

Deprecations
~~~~~~~~~~~~

- The method ``load_data()`` has been renamed to the more succinct
  ``xray.Dataset.load``.

v0.4.1 (18 March 2015)
----------------------

The release contains bug fixes and several new features. All changes should be
fully backwards compatible.

Enhancements
~~~~~~~~~~~~

- New documentation sections on :ref:`time-series` and
  :ref:`combining multiple files`.
- ``xray.Dataset.resample`` lets you resample a dataset or data array to
  a new temporal resolution. The syntax is the `same as pandas`_, except you
  need to supply the time dimension explicitly:

  .. ipython:: python
     :verbatim:

      time = pd.date_range('2000-01-01', freq='6H', periods=10)
      array = xray.DataArray(np.arange(10), [('time', time)])
      array.resample('1D', dim='time')

  You can specify how to do the resampling with the ``how`` argument and other
  options such as ``closed`` and ``label`` let you control labeling:

  .. ipython:: python
     :verbatim:

      array.resample('1D', dim='time', how='sum', label='right')

  If the desired temporal resolution is higher than the original data
  (upsampling), xray will insert missing values:

  .. ipython:: python
     :verbatim:

      array.resample('3H', 'time')

- ``first`` and ``last`` methods on groupby objects let you take the first or
  last examples from each group along the grouped axis:

  .. ipython:: python
     :verbatim:

      array.groupby('time.day').first()

  These methods combine well with ``resample``:

  .. ipython:: python
     :verbatim:

      array.resample('1D', dim='time', how='first')


- ``xray.Dataset.swap_dims`` allows for easily swapping one dimension
  out for another:

  .. ipython:: python

       ds = xray.Dataset({'x': range(3), 'y': ('x', list('abc'))})
       ds
       ds.swap_dims({'x': 'y'})

  This was possible in earlier versions of xray, but required some contortions.
- ``xray.open_dataset`` and ``xray.Dataset.to_netcdf`` now
  accept an ``engine`` argument to explicitly select which underlying library
  (netcdf4 or scipy) is used for reading/writing a netCDF file.

.. _same as pandas: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#up-and-downsampling

Bug fixes
~~~~~~~~~

- Fixed a bug where data netCDF variables read from disk with
  ``engine='scipy'`` could still be associated with the file on disk, even
  after closing the file (:issue:`341`). This manifested itself in warnings
  about mmapped arrays and segmentation faults (if the data was accessed).
- Silenced spurious warnings about all-NaN slices when using nan-aware
  aggregation methods (:issue:`344`).
- Dataset aggregations with ``keep_attrs=True`` now preserve attributes on
  data variables, not just the dataset itself.
- Tests for xray now pass when run on Windows (:issue:`360`).
- Fixed a regression in v0.4 where saving to netCDF could fail with the error
  ``ValueError: could not automatically determine time units``.

v0.4 (2 March, 2015)
--------------------

This is one of the biggest releases yet for xray: it includes some major
changes that may break existing code, along with the usual collection of minor
enhancements and bug fixes. On the plus side, this release includes all
hitherto planned breaking changes, so the upgrade path for xray should be
smoother going forward.

Breaking changes
~~~~~~~~~~~~~~~~

- We now automatically align index labels in arithmetic, dataset construction,
  merging and updating. This means the need for manually invoking methods like
  ``xray.align`` and ``xray.Dataset.reindex_like`` should be
  vastly reduced.

  :ref:`For arithmetic<math automatic alignment>`, we align
  based on the **intersection** of labels:

  .. ipython:: python

      lhs = xray.DataArray([1, 2, 3], [('x', [0, 1, 2])])
      rhs = xray.DataArray([2, 3, 4], [('x', [1, 2, 3])])
      lhs + rhs

  :ref:`For dataset construction and merging<merge>`, we align based on the
  **union** of labels:

  .. ipython:: python

      xray.Dataset({'foo': lhs, 'bar': rhs})

  :ref:`For update and __setitem__<update>`, we align based on the **original**
  object:

  .. ipython:: python

      lhs.coords['rhs'] = rhs
      lhs

- Aggregations like ``mean`` or ``median`` now skip missing values by default:

  .. ipython:: python

      xray.DataArray([1, 2, np.nan, 3]).mean()

  You can turn this behavior off by supplying the keyword arugment
  ``skipna=False``.

  These operations are lightning fast thanks to integration with bottleneck_,
  which is a new optional dependency for xray (numpy is used if bottleneck is
  not installed).
- Scalar coordinates no longer conflict with constant arrays with the same
  value (e.g., in arithmetic, merging datasets and concat), even if they have
  different shape (:issue:`243`). For example, the coordinate ``c`` here
  persists through arithmetic, even though it has different shapes on each
  DataArray:

  .. ipython:: python

      a = xray.DataArray([1, 2], coords={'c': 0}, dims='x')
      b = xray.DataArray([1, 2], coords={'c': ('x', [0, 0])}, dims='x')
      (a + b).coords

  This functionality can be controlled through the ``compat`` option, which
  has also been added to the ``xray.Dataset`` constructor.
- Datetime shortcuts such as ``'time.month'`` now return a ``DataArray`` with
  the name ``'month'``, not ``'time.month'`` (:issue:`345`). This makes it
  easier to index the resulting arrays when they are used with ``groupby``:

  .. ipython:: python

      time = xray.DataArray(pd.date_range('2000-01-01', periods=365),
                            dims='time', name='time')
      counts = time.groupby('time.month').count()
      counts.sel(month=2)

  Previously, you would need to use something like
  ``counts.sel(**{'time.month': 2}})``, which is much more awkward.
- The ``season`` datetime shortcut now returns an array of string labels
  such `'DJF'`:

  .. ipython:: python

      ds = xray.Dataset({'t': pd.date_range('2000-01-01', periods=12, freq='M')})
      ds['t.season']

  Previously, it returned numbered seasons 1 through 4.
- We have updated our use of the terms of "coordinates" and "variables". What
  were known in previous versions of xray as "coordinates" and "variables" are
  now referred to throughout the documentation as "coordinate variables" and
  "data variables". This brings xray in closer alignment to `CF Conventions`_.
  The only visible change besides the documentation is that ``Dataset.vars``
  has been renamed ``Dataset.data_vars``.
- You will need to update your code if you have been ignoring deprecation
  warnings: methods and attributes that were deprecated in xray v0.3 or earlier
  (e.g., ``dimensions``, ``attributes```) have gone away.

.. _bottleneck: https://github.com/pydata/bottleneck

Enhancements
~~~~~~~~~~~~

- Support for ``xray.Dataset.reindex`` with a fill method. This
  provides a useful shortcut for upsampling:

  .. ipython:: python

      data = xray.DataArray([1, 2, 3], [('x', range(3))])
      data.reindex(x=[0.5, 1, 1.5, 2, 2.5], method='pad')

  This will be especially useful once pandas 0.16 is released, at which point
  xray will immediately support reindexing with
  `method='nearest' <https://github.com/pydata/pandas/pull/9258>`_.
- Use functions that return generic ndarrays with DataArray.groupby.apply and
  Dataset.apply (:issue:`327` and :issue:`329`). Thanks Jeff Gerard!
- Consolidated the functionality of ``dumps`` (writing a dataset to a netCDF3
  bytestring) into ``xray.Dataset.to_netcdf`` (:issue:`333`).
- ``xray.Dataset.to_netcdf`` now supports writing to groups in netCDF4
  files (:issue:`333`). It also finally has a full docstring -- you should read
  it!
- ``xray.open_dataset`` and ``xray.Dataset.to_netcdf`` now
  work on netCDF3 files when netcdf4-python is not installed as long as scipy
  is available (:issue:`333`).
- The new ``xray.Dataset.drop`` and ``xray.DataArray.drop`` methods
  makes it easy to drop explicitly listed variables or index labels:

  .. ipython:: python
     :okwarning:

      # drop variables
      ds = xray.Dataset({'x': 0, 'y': 1})
      ds.drop('x')

      # drop index labels
      arr = xray.DataArray([1, 2, 3], coords=[('x', list('abc'))])
      arr.drop(['a', 'c'], dim='x')

- ``xray.Dataset.broadcast_equals`` has been added to correspond to
  the new ``compat`` option.
- Long attributes are now truncated at 500 characters when printing a dataset
  (:issue:`338`). This should make things more convenient for working with
  datasets interactively.
- Added a new documentation example, :ref:`/examples/monthly-means.ipynb`. Thanks Joe
  Hamman!

Bug fixes
~~~~~~~~~

- Several bug fixes related to decoding time units from netCDF files
  (:issue:`316`, :issue:`330`). Thanks Stefan Pfenninger!
- xray no longer requires ``decode_coords=False`` when reading datasets with
  unparseable coordinate attributes (:issue:`308`).
- Fixed ``DataArray.loc`` indexing with ``...`` (:issue:`318`).
- Fixed an edge case that resulting in an error when reindexing
  multi-dimensional variables (:issue:`315`).
- Slicing with negative step sizes (:issue:`312`).
- Invalid conversion of string arrays to numeric dtype (:issue:`305`).
- Fixed``repr()`` on dataset objects with non-standard dates (:issue:`347`).

Deprecations
~~~~~~~~~~~~

- ``dump`` and ``dumps`` have been deprecated in favor of
  ``xray.Dataset.to_netcdf``.
- ``drop_vars`` has been deprecated in favor of ``xray.Dataset.drop``.

Future plans
~~~~~~~~~~~~

The biggest feature I'm excited about working toward in the immediate future
is supporting out-of-core operations in xray using Dask_, a part of the Blaze_
project. For a preview of using Dask with weather data, read
`this blog post`_ by Matthew Rocklin. See :issue:`328` for more details.

.. _Dask: http://dask.pydata.org
.. _Blaze: http://blaze.pydata.org
.. _this blog post: http://matthewrocklin.com/blog/work/2015/02/13/Towards-OOC-Slicing-and-Stacking/

v0.3.2 (23 December, 2014)
--------------------------

This release focused on bug-fixes, speedups and resolving some niggling
inconsistencies.

There are a few cases where the behavior of xray differs from the previous
version. However, I expect that in almost all cases your code will continue to
run unmodified.

.. warning::

    xray now requires pandas v0.15.0 or later. This was necessary for
    supporting TimedeltaIndex without too many painful hacks.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Arrays of :py:class:`datetime.datetime` objects are now automatically cast to
  ``datetime64[ns]`` arrays when stored in an xray object, using machinery
  borrowed from pandas:

  .. ipython:: python

      from datetime import datetime
      xray.Dataset({'t': [datetime(2000, 1, 1)]})

- xray now has support (including serialization to netCDF) for
  :py:class:`~pandas.TimedeltaIndex`. :py:class:`datetime.timedelta` objects
  are thus accordingly cast to ``timedelta64[ns]`` objects when appropriate.
- Masked arrays are now properly coerced to use ``NaN`` as a sentinel value
  (:issue:`259`).

Enhancements
~~~~~~~~~~~~

- Due to popular demand, we have added experimental attribute style access as
  a shortcut for dataset variables, coordinates and attributes:

  .. ipython:: python

     ds = xray.Dataset({'tmin': ([], 25, {'units': 'celsius'})})
     ds.tmin.units

  Tab-completion for these variables should work in editors such as IPython.
  However, setting variables or attributes in this fashion is not yet
  supported because there are some unresolved ambiguities (:issue:`300`).
- You can now use a dictionary for indexing with labeled dimensions. This
  provides a safe way to do assignment with labeled dimensions:

  .. ipython:: python

      array = xray.DataArray(np.zeros(5), dims=['x'])
      array[dict(x=slice(3))] = 1
      array

- Non-index coordinates can now be faithfully written to and restored from
  netCDF files. This is done according to CF conventions when possible by
  using the ``coordinates`` attribute on a data variable. When not possible,
  xray defines a global ``coordinates`` attribute.
- Preliminary support for converting ``xray.DataArray`` objects to and from
  CDAT_ ``cdms2`` variables.
- We sped up any operation that involves creating a new Dataset or DataArray
  (e.g., indexing, aggregation, arithmetic) by a factor of 30 to 50%. The full
  speed up requires cyordereddict_ to be installed.

.. _CDAT: http://uvcdat.llnl.gov/
.. _cyordereddict: https://github.com/shoyer/cyordereddict

Bug fixes
~~~~~~~~~

- Fix for ``to_dataframe()`` with 0d string/object coordinates (:issue:`287`)
- Fix for ``to_netcdf`` with 0d string variable (:issue:`284`)
- Fix writing datetime64 arrays to netcdf if NaT is present (:issue:`270`)
- Fix align silently upcasts data arrays when NaNs are inserted (:issue:`264`)

Future plans
~~~~~~~~~~~~

- I am contemplating switching to the terms "coordinate variables" and "data
  variables" instead of the (currently used) "coordinates" and "variables",
  following their use in `CF Conventions`_ (:issue:`293`). This would mostly
  have implications for the documentation, but I would also change the
  ``Dataset`` attribute ``vars`` to ``data``.
- I no longer certain that automatic label alignment for arithmetic would be a
  good idea for xray -- it is a feature from pandas that I have not missed
  (:issue:`186`).
- The main API breakage that I *do* anticipate in the next release is finally
  making all aggregation operations skip missing values by default
  (:issue:`130`). I'm pretty sick of writing ``ds.reduce(np.nanmean, 'time')``.
- The next version of xray (0.4) will remove deprecated features and aliases
  whose use currently raises a warning.

If you have opinions about any of these anticipated changes, I would love to
hear them -- please add a note to any of the referenced GitHub issues.

.. _CF Conventions: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.6/build/cf-conventions.html

v0.3.1 (22 October, 2014)
-------------------------

This is mostly a bug-fix release to make xray compatible with the latest
release of pandas (v0.15).

We added several features to better support working with missing values and
exporting xray objects to pandas. We also reorganized the internal API for
serializing and deserializing datasets, but this change should be almost
entirely transparent to users.

Other than breaking the experimental DataStore API, there should be no
backwards incompatible changes.

New features
~~~~~~~~~~~~

- Added ``xray.Dataset.count`` and ``xray.Dataset.dropna``
  methods, copied from pandas, for working with missing values (:issue:`247`,
  :issue:`58`).
- Added ``xray.DataArray.to_pandas`` for
  converting a data array into the pandas object with the same dimensionality
  (1D to Series, 2D to DataFrame, etc.) (:issue:`255`).
- Support for reading gzipped netCDF3 files (:issue:`239`).
- Reduced memory usage when writing netCDF files (:issue:`251`).
- 'missing_value' is now supported as an alias for the '_FillValue' attribute
  on netCDF variables (:issue:`245`).
- Trivial indexes, equivalent to ``range(n)`` where ``n`` is the length of the
  dimension, are no longer written to disk (:issue:`245`).

Bug fixes
~~~~~~~~~

- Compatibility fixes for pandas v0.15 (:issue:`262`).
- Fixes for display and indexing of ``NaT`` (not-a-time) (:issue:`238`,
  :issue:`240`)
- Fix slicing by label was an argument is a data array (:issue:`250`).
- Test data is now shipped with the source distribution (:issue:`253`).
- Ensure order does not matter when doing arithmetic with scalar data arrays
  (:issue:`254`).
- Order of dimensions preserved with ``DataArray.to_dataframe`` (:issue:`260`).

v0.3 (21 September 2014)
------------------------

New features
~~~~~~~~~~~~

- **Revamped coordinates**: "coordinates" now refer to all arrays that are not
  used to index a dimension. Coordinates are intended to allow for keeping track
  of arrays of metadata that describe the grid on which the points in "variable"
  arrays lie. They are preserved (when unambiguous) even though mathematical
  operations.
- **Dataset math** ``xray.Dataset`` objects now support all arithmetic
  operations directly. Dataset-array operations map across all dataset
  variables; dataset-dataset operations act on each pair of variables with the
  same name.
- **GroupBy math**: This provides a convenient shortcut for normalizing by the
  average value of a group.
- The dataset ``__repr__`` method has been entirely overhauled; dataset
  objects now show their values when printed.
- You can now index a dataset with a list of variables to return a new dataset:
  ``ds[['foo', 'bar']]``.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``Dataset.__eq__`` and ``Dataset.__ne__`` are now element-wise operations
  instead of comparing all values to obtain a single boolean. Use the method
  ``xray.Dataset.equals`` instead.

Deprecations
~~~~~~~~~~~~

- ``Dataset.noncoords`` is deprecated: use ``Dataset.vars`` instead.
- ``Dataset.select_vars`` deprecated: index a ``Dataset`` with a list of
  variable names instead.
- ``DataArray.select_vars`` and ``DataArray.drop_vars`` deprecated: use
  ``xray.DataArray.reset_coords`` instead.

v0.2 (14 August 2014)
---------------------

This is major release that includes some new features and quite a few bug
fixes. Here are the highlights:

- There is now a direct constructor for ``DataArray`` objects, which makes it
  possible to create a DataArray without using a Dataset. This is highlighted
  in the refreshed ``tutorial``.
- You can perform aggregation operations like ``mean`` directly on
  ``xray.Dataset`` objects, thanks to Joe Hamman. These aggregation
  methods also worked on grouped datasets.
- xray now works on Python 2.6, thanks to Anna Kuznetsova.
- A number of methods and attributes were given more sensible (usually shorter)
  names: ``labeled`` -> ``sel``,  ``indexed`` -> ``isel``, ``select`` ->
  ``select_vars``, ``unselect`` -> ``drop_vars``, ``dimensions`` -> ``dims``,
  ``coordinates`` -> ``coords``, ``attributes`` -> ``attrs``.
- New ``xray.Dataset.load_data`` and ``xray.Dataset.close``
  methods for datasets facilitate lower level of control of data loaded from
  disk.

v0.1.1 (20 May 2014)
--------------------

xray 0.1.1 is a bug-fix release that includes changes that should be almost
entirely backwards compatible with v0.1:

- Python 3 support (:issue:`53`)
- Required numpy version relaxed to 1.7 (:issue:`129`)
- Return numpy.datetime64 arrays for non-standard calendars (:issue:`126`)
- Support for opening datasets associated with NetCDF4 groups (:issue:`127`)
- Bug-fixes for concatenating datetime arrays (:issue:`134`)

Special thanks to new contributors Thomas Kluyver, Joe Hamman and Alistair
Miles.

v0.1 (2 May 2014)
-----------------

Initial release.
