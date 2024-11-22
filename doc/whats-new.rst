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

.. _whats-new.2024.10.1:

v.2024.11.0 (Nov 22, 2024)
--------------------------

This release brings better support for wrapping JAX arrays and Astropy Quantity objects, :py:meth:`DataTree.persist`, algorithmic improvements
to many methods with dask (:py:meth:`Dataset.polyfit`, :py:meth:`Dataset.ffill`, :py:meth:`Dataset.bfill`, rolling reductions), and bug fixes.
Thanks to the 22 contributors to this release:
Benoit Bovy, Deepak Cherian, Dimitri Papadopoulos Orfanos, Holly Mandel, James Bourbeau, Joe Hamman, Justus Magin, Kai Mühlbauer, Lukas Trippe, Mathias Hauser, Maximilian Roos, Michael Niklas, Pascal Bourgault, Patrick Hoefler, Sam Levang, Sarah Charlotte Johnson, Scott Huberty, Stephan Hoyer, Tom Nicholas, Virgile Andreani, joseph nowak and tvo

New Features
~~~~~~~~~~~~
- Added :py:meth:`DataTree.persist` method (:issue:`9675`, :pull:`9682`).
  By `Sam Levang <https://github.com/slevang>`_.
- Added ``write_inherited_coords`` option to :py:meth:`DataTree.to_netcdf`
  and :py:meth:`DataTree.to_zarr` (:pull:`9677`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Support lazy grouping by dask arrays, and allow specifying ordered groups with ``UniqueGrouper(labels=["a", "b", "c"])``
  (:issue:`2852`, :issue:`757`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Add new ``automatic_rechunk`` kwarg to :py:meth:`DataArrayRolling.construct` and
  :py:meth:`DatasetRolling.construct`. This is only useful on ``dask>=2024.11.0``
  (:issue:`9550`). By `Deepak Cherian <https://github.com/dcherian>`_.
- Optimize ffill, bfill with dask when limit is specified
  (:pull:`9771`).
  By `Joseph Nowak <https://github.com/josephnowak>`_, and
  `Patrick Hoefler <https://github.com/phofl>`_.
- Allow wrapping ``np.ndarray`` subclasses, e.g. ``astropy.units.Quantity`` (:issue:`9704`, :pull:`9760`).
  By `Sam Levang <https://github.com/slevang>`_ and `Tien Vo <https://github.com/tien-vo>`_.
- Optimize :py:meth:`DataArray.polyfit` and :py:meth:`Dataset.polyfit` with dask, when used with
  arrays with more than two dimensions.
  (:issue:`5629`). By `Deepak Cherian <https://github.com/dcherian>`_.
- Support for directly opening remote files as string paths (for example, ``s3://bucket/data.nc``)
  with ``fsspec`` when using the ``h5netcdf`` engine (:issue:`9723`, :pull:`9797`).
  By `James Bourbeau <https://github.com/jrbourbeau>`_.
- Re-implement the :py:mod:`ufuncs` module, which now dynamically dispatches to the
  underlying array's backend. Provides better support for certain wrapped array types
  like ``jax.numpy.ndarray``. (:issue:`7848`, :pull:`9776`).
  By `Sam Levang <https://github.com/slevang>`_.
- Speed up loading of large zarr stores using dask arrays. (:issue:`8902`)
  By `Deepak Cherian <https://github.com/dcherian>`_.

Breaking Changes
~~~~~~~~~~~~~~~~
- The minimum versions of some dependencies were changed

  ===================== =========  =======
   Package                    Old      New
  ===================== =========  =======
    boto3                    1.28     1.29
    dask-core             2023.9   2023.11
    distributed           2023.9   2023.11
    h5netcdf                 1.2      1.3
    numbagg                0.2.1      0.6
    typing_extensions       4.7       4.8
  ===================== =========  =======

Deprecations
~~~~~~~~~~~~
- Grouping by a chunked array (e.g. dask or cubed) currently eagerly loads that variable in to
  memory. This behaviour is deprecated. If eager loading was intended, please load such arrays
  manually using ``.load()`` or ``.compute()``. Else pass ``eagerly_compute_group=False``, and
  provide expected group labels using the ``labels`` kwarg to a grouper object such as
  :py:class:`grouper.UniqueGrouper` or :py:class:`grouper.BinGrouper`.

Bug fixes
~~~~~~~~~

- Fix inadvertent deep-copying of child data in DataTree (:issue:`9683`,
  :pull:`9684`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Avoid including parent groups when writing DataTree subgroups to Zarr or
  netCDF (:pull:`9682`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Fix regression in the interoperability of :py:meth:`DataArray.polyfit` and :py:meth:`xr.polyval` for date-time coordinates. (:pull:`9691`).
  By `Pascal Bourgault <https://github.com/aulemahal>`_.
- Fix CF decoding of ``grid_mapping`` to allow all possible formats, add tests (:issue:`9761`, :pull:`9765`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Add ``User-Agent`` to request-headers when retrieving tutorial data (:issue:`9774`, :pull:`9782`)
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Documentation
~~~~~~~~~~~~~

- Mention attribute peculiarities in docs/docstrings (:issue:`4798`, :pull:`9700`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.


Internal Changes
~~~~~~~~~~~~~~~~
- ``persist`` methods now route through the :py:class:`xr.core.parallelcompat.ChunkManagerEntrypoint` (:pull:`9682`).
  By `Sam Levang <https://github.com/slevang>`_.

.. _whats-new.2024.10.0:

v2024.10.0 (Oct 24th, 2024)
---------------------------

This release brings official support for ``xarray.DataTree``, and compatibility with zarr-python v3!

Aside from these two huge features, it also improves support for vectorised interpolation and fixes various bugs.

Thanks to the 31 contributors to this release:
Alfonso Ladino, DWesl, Deepak Cherian, Eni, Etienne Schalk, Holly Mandel, Ilan Gold, Illviljan, Joe Hamman, Justus Magin, Kai Mühlbauer, Karl Krauth, Mark Harfouche, Martey Dodoo, Matt Savoie, Maximilian Roos, Patrick Hoefler, Peter Hill, Renat Sibgatulin, Ryan Abernathey, Spencer Clark, Stephan Hoyer, Tom Augspurger, Tom Nicholas, Vecko, Virgile Andreani, Yvonne Fröhlich, carschandler, joseph nowak, mgunyho and owenlittlejohns

New Features
~~~~~~~~~~~~
- ``DataTree`` related functionality is now exposed in the main ``xarray`` public
  API. This includes: ``xarray.DataTree``, ``xarray.open_datatree``, ``xarray.open_groups``,
  ``xarray.map_over_datasets``, ``xarray.group_subtrees``,
  ``xarray.register_datatree_accessor`` and ``xarray.testing.assert_isomorphic``.
  By `Owen Littlejohns <https://github.com/owenlittlejohns>`_,
  `Eni Awowale <https://github.com/eni-awowale>`_,
  `Matt Savoie <https://github.com/flamingbear>`_,
  `Stephan Hoyer <https://github.com/shoyer>`_,
  `Tom Nicholas <https://github.com/TomNicholas>`_,
  `Justus Magin <https://github.com/keewis>`_, and
  `Alfonso Ladino <https://github.com/aladinor>`_.
- A migration guide for users of the prototype `xarray-contrib/datatree repository <https://github.com/xarray-contrib/datatree>`_ has been added, and can be found in the ``DATATREE_MIGRATION_GUIDE.md`` file in the repository root.
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Support for Zarr-Python 3 (:issue:`95515`, :pull:`9552`).
  By `Tom Augspurger <https://github.com/TomAugspurger>`_,
  `Ryan Abernathey <https://github.com/rabernat>`_ and
  `Joe Hamman <https://github.com/jhamman>`_.
- Added zarr backends for :py:func:`open_groups` (:issue:`9430`, :pull:`9469`).
  By `Eni Awowale <https://github.com/eni-awowale>`_.
- Added support for vectorized interpolation using additional interpolators
  from the ``scipy.interpolate`` module (:issue:`9049`, :pull:`9526`).
  By `Holly Mandel <https://github.com/hollymandel>`_.
- Implement handling of complex numbers (netcdf4/h5netcdf) and enums (h5netcdf) (:issue:`9246`, :issue:`3297`, :pull:`9509`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Fix passing missing arguments to when opening hdf5 and netCDF4 datatrees
  (:issue:`9427`, :pull:`9428`).
  By `Alfonso Ladino <https://github.com/aladinor>`_.

Bug fixes
~~~~~~~~~

- Make illegal path-like variable names when constructing a DataTree from a Dataset
  (:issue:`9339`, :pull:`9378`)
  By `Etienne Schalk <https://github.com/etienneschalk>`_.
- Work around `upstream pandas issue
  <https://github.com/pandas-dev/pandas/issues/56996>`_ to ensure that we can
  decode times encoded with small integer dtype values (e.g. ``np.int32``) in
  environments with NumPy 2.0 or greater without needing to fall back to cftime
  (:pull:`9518`). By `Spencer Clark <https://github.com/spencerkclark>`_.
- Fix bug when encoding times with missing values as floats in the case when
  the non-missing times could in theory be encoded with integers
  (:issue:`9488`, :pull:`9497`). By `Spencer Clark
  <https://github.com/spencerkclark>`_.
- Fix a few bugs affecting groupby reductions with ``flox``. (:issue:`8090`, :issue:`9398`, :issue:`9648`).
- Fix a few bugs affecting groupby reductions with ``flox``. (:issue:`8090`, :issue:`9398`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix the safe_chunks validation option on the to_zarr method
  (:issue:`5511`, :pull:`9559`). By `Joseph Nowak
  <https://github.com/josephnowak>`_.
- Fix binning by multiple variables where some bins have no observations. (:issue:`9630`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix issue where polyfit wouldn't handle non-dimension coordinates. (:issue:`4375`, :pull:`9369`)
  By `Karl Krauth <https://github.com/Karl-Krauth>`_.

Documentation
~~~~~~~~~~~~~

- Migrate documentation for ``datatree`` into main ``xarray`` documentation (:pull:`9033`).
  For information on previous ``datatree`` releases, please see:
  `datatree's historical release notes <https://xarray-datatree.readthedocs.io/en/latest/>`_.
  By `Owen Littlejohns <https://github.com/owenlittlejohns>`_, `Matt Savoie <https://github.com/flamingbear>`_, and
  `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~

.. _whats-new.2024.09.0:

v2024.09.0 (Sept 11, 2024)
--------------------------
This release drops support for Python 3.9, and adds support for grouping by :ref:`multiple arrays <groupby.multiple>`, while providing numerous performance improvements and bug fixes.

Thanks to the 33 contributors to this release:
Alfonso Ladino, Andrew Scherer, Anurag Nayak, David Hoese, Deepak Cherian, Diogo Teles Sant'Anna, Dom, Elliott Sales de Andrade, Eni, Holly Mandel, Illviljan, Jack Kelly, Julius Busecke, Justus Magin, Kai Mühlbauer, Manish Kumar Gupta, Matt Savoie, Maximilian Roos, Michele Claus, Miguel Jimenez, Niclas Rieger, Pascal Bourgault, Philip Chmielowiec, Spencer Clark, Stephan Hoyer, Tao Xin, Tiago Sanona, TimothyCera-NOAA, Tom Nicholas, Tom White, Virgile Andreani, oliverhiggs and tiago

New Features
~~~~~~~~~~~~

- Add :py:attr:`~core.accessor_dt.DatetimeAccessor.days_in_year` and
  :py:attr:`~core.accessor_dt.DatetimeAccessor.decimal_year` to the
  ``DatetimeAccessor`` on ``xr.DataArray``. (:pull:`9105`).
  By `Pascal Bourgault <https://github.com/aulemahal>`_.

Performance
~~~~~~~~~~~

- Make chunk manager an option in ``set_options`` (:pull:`9362`).
  By `Tom White <https://github.com/tomwhite>`_.
- Support for :ref:`grouping by multiple variables <groupby.multiple>`.
  This is quite new, so please check your results and report bugs.
  Binary operations after grouping by multiple arrays are not supported yet.
  (:issue:`1056`, :issue:`9332`, :issue:`324`, :pull:`9372`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Allow data variable specific ``constant_values`` in the dataset ``pad`` function (:pull:`9353`).
  By `Tiago Sanona <https://github.com/tsanona>`_.
- Speed up grouping by avoiding deep-copy of non-dimension coordinates (:issue:`9426`, :pull:`9393`)
  By `Deepak Cherian <https://github.com/dcherian>`_.

Breaking changes
~~~~~~~~~~~~~~~~
- Support for ``python 3.9`` has been dropped (:pull:`8937`)
- The minimum versions of some dependencies were changed

  ===================== =========  =======
   Package                    Old      New
  ===================== =========  =======
    boto3                   1.26      1.28
    cartopy                 0.21      0.22
    dask-core             2023.4    2023.9
    distributed           2023.4    2023.9
    h5netcdf                1.1        1.2
    iris                    3.4        3.7
    numba                   0.56      0.57
    numpy                   1.23      1.24
    pandas                  2.0        2.1
    scipy                   1.10      1.11
    typing_extensions       4.5        4.7
    zarr                    2.14      2.16
  ===================== =========  =======

Bug fixes
~~~~~~~~~

- Fix bug with rechunking to a frequency when some periods contain no data (:issue:`9360`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix bug causing ``DataTree.from_dict`` to be sensitive to insertion order (:issue:`9276`, :pull:`9292`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fix resampling error with monthly, quarterly, or yearly frequencies with
  cftime when the time bins straddle the date "0001-01-01". For example, this
  can happen in certain circumstances when the time coordinate contains the
  date "0001-01-01". (:issue:`9108`, :pull:`9116`) By `Spencer Clark
  <https://github.com/spencerkclark>`_ and `Deepak Cherian
  <https://github.com/dcherian>`_.
- Fix issue with passing parameters to ZarrStore.open_store when opening
  datatree in zarr format (:issue:`9376`, :pull:`9377`).
  By `Alfonso Ladino <https://github.com/aladinor>`_
- Fix deprecation warning that was raised when calling ``np.array`` on an ``xr.DataArray``
  in NumPy 2.0 (:issue:`9312`, :pull:`9393`)
  By `Andrew Scherer <https://github.com/andrew-s28>`_.
- Fix passing missing arguments to when opening hdf5 and netCDF4 datatrees
  (:issue:`9427`, :pull:`9428`).
  By `Alfonso Ladino <https://github.com/aladinor>`_.
- Fix support for using ``pandas.DateOffset``, ``pandas.Timedelta``, and
  ``datetime.timedelta`` objects as ``resample`` frequencies
  (:issue:`9408`, :pull:`9413`).
  By `Oliver Higgs <https://github.com/oliverhiggs>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Re-enable testing ``pydap`` backend with ``numpy>=2`` (:pull:`9391`).
  By `Miguel Jimenez <https://github.com/Mikejmnez>`_ .

.. _whats-new.2024.07.0:

v2024.07.0 (Jul 30, 2024)
-------------------------
This release extends the API for groupby operations with various `grouper objects <groupby.groupers>`_, and includes improvements to the documentation and numerous bugfixes.

Thanks to the 22 contributors to this release:
Alfonso Ladino, ChrisCleaner, David Hoese, Deepak Cherian, Dieter Werthmüller, Illviljan, Jessica Scheick, Joel Jaeschke, Justus Magin, K. Arthur Endsley, Kai Mühlbauer, Mark Harfouche, Martin Raspaud, Mathijs Verhaegh, Maximilian Roos, Michael Niklas, Michał Górny, Moritz Schreiber, Pontus Lurcock, Spencer Clark, Stephan Hoyer and Tom Nicholas

New Features
~~~~~~~~~~~~
- Use fastpath when grouping both montonically increasing and decreasing variable
  in :py:class:`GroupBy` (:issue:`6220`, :pull:`7427`).
  By `Joel Jaeschke <https://github.com/joeljaeschke>`_.
- Introduce new :py:class:`groupers.UniqueGrouper`, :py:class:`groupers.BinGrouper`, and
  :py:class:`groupers.TimeResampler` objects as a step towards supporting grouping by
  multiple variables. See the `docs <groupby.groupers>`_ and the `grouper design doc
  <https://github.com/pydata/xarray/blob/main/design_notes/grouper_objects.md>`_ for more.
  (:issue:`6610`, :pull:`8840`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Allow rechunking to a frequency using ``Dataset.chunk(time=TimeResampler("YE"))`` syntax. (:issue:`7559`, :pull:`9109`)
  Such rechunking allows many time domain analyses to be executed in an embarrassingly parallel fashion.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Allow per-variable specification of ```mask_and_scale``, ``decode_times``, ``decode_timedelta``
  ``use_cftime`` and ``concat_characters`` params in :py:func:`~xarray.open_dataset`  (:pull:`9218`).
  By `Mathijs Verhaegh <https://github.com/Ostheer>`_.
- Allow chunking for arrays with duplicated dimension names (:issue:`8759`, :pull:`9099`).
  By `Martin Raspaud <https://github.com/mraspaud>`_.
- Extract the source url from fsspec objects (:issue:`9142`, :pull:`8923`).
  By `Justus Magin <https://github.com/keewis>`_.
- Add :py:meth:`DataArray.drop_attrs` & :py:meth:`Dataset.drop_attrs` methods,
  to return an object without ``attrs``. A ``deep`` parameter controls whether
  variables' ``attrs`` are also dropped.
  By `Maximilian Roos <https://github.com/max-sixty>`_. (:pull:`8288`)
- Added :py:func:`open_groups` for h5netcdf and netCDF4 backends (:issue:`9137`, :pull:`9243`).
  By `Eni Awowale <https://github.com/eni-awowale>`_.

Breaking changes
~~~~~~~~~~~~~~~~
- The ``base`` and ``loffset`` parameters to :py:meth:`Dataset.resample` and
  :py:meth:`DataArray.resample` are now removed. These parameters have been deprecated since
  v2023.03.0. Using the ``origin`` or ``offset`` parameters is recommended as a replacement for
  using the ``base`` parameter and using time offset arithmetic is recommended as a replacement for
  using the ``loffset`` parameter. (:pull:`9233`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- The ``squeeze`` kwarg to ``groupby`` is now ignored. This has been the source of some
  quite confusing behaviour and has been deprecated since v2024.01.0. ``groupby`` behavior is now
  always consistent with the existing ``.groupby(..., squeeze=False)`` behavior. No errors will
  be raised if ``squeeze=False``. (:pull:`9280`)
  By `Deepak Cherian <https://github.com/dcherian>`_.


Bug fixes
~~~~~~~~~
- Fix scatter plot broadcasting unnecessarily. (:issue:`9129`, :pull:`9206`)
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Don't convert custom indexes to ``pandas`` indexes when computing a diff (:pull:`9157`)
  By `Justus Magin <https://github.com/keewis>`_.
- Make :py:func:`testing.assert_allclose` work with numpy 2.0 (:issue:`9165`, :pull:`9166`).
  By `Pontus Lurcock <https://github.com/pont-us>`_.
- Allow diffing objects with array attributes on variables (:issue:`9153`, :pull:`9169`).
  By `Justus Magin <https://github.com/keewis>`_.
- ``numpy>=2`` compatibility in the ``netcdf4`` backend (:pull:`9136`).
  By `Justus Magin <https://github.com/keewis>`_ and `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Promote floating-point numeric datetimes before decoding (:issue:`9179`, :pull:`9182`).
  By `Justus Magin <https://github.com/keewis>`_.
- Address regression introduced in :pull:`9002` that prevented objects returned
  by py:meth:`DataArray.convert_calendar` to be indexed by a time index in
  certain circumstances (:issue:`9138`, :pull:`9192`).
  By `Mark Harfouche <https://github.com/hmaarrfk>`_ and `Spencer Clark <https://github.com/spencerkclark>`_.
- Fix static typing of tolerance arguments by allowing ``str`` type (:issue:`8892`, :pull:`9194`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Dark themes are now properly detected for ``html[data-theme=dark]``-tags (:pull:`9200`).
  By `Dieter Werthmüller <https://github.com/prisae>`_.
- Reductions no longer fail for ``np.complex_`` dtype arrays when numbagg is
  installed. (:pull:`9210`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.

Documentation
~~~~~~~~~~~~~

- Adds intro to backend section of docs, including a flow-chart to navigate types of backends (:pull:`9175`).
  By `Jessica Scheick <https://github.com/jessicas11>`_.
- Adds a flow-chart diagram to help users navigate help resources (:discussion:`8990`, :pull:`9147`).
  By `Jessica Scheick <https://github.com/jessicas11>`_.
- Improvements to Zarr & chunking docs (:pull:`9139`, :pull:`9140`, :pull:`9132`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Fix copybutton for multi line examples and double digit ipython cell numbers (:pull:`9264`).
  By `Moritz Schreiber <https://github.com/mosc9575>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Enable typing checks of pandas (:pull:`9213`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.

.. _whats-new.2024.06.0:

v2024.06.0 (Jun 13, 2024)
-------------------------
This release brings various performance optimizations and compatibility with the upcoming numpy 2.0 release.

Thanks to the 22 contributors to this release:
Alfonso Ladino, David Hoese, Deepak Cherian, Eni Awowale, Ilan Gold, Jessica Scheick, Joe Hamman, Justus Magin, Kai Mühlbauer, Mark Harfouche, Mathias Hauser, Matt Savoie, Maximilian Roos, Mike Thramann, Nicolas Karasiak, Owen Littlejohns, Paul Ockenfuß, Philippe THOMY, Scott Henderson, Spencer Clark, Stephan Hoyer and Tom Nicholas

Performance
~~~~~~~~~~~

- Small optimization to the netCDF4 and h5netcdf backends (:issue:`9058`, :pull:`9067`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Small optimizations to help reduce indexing speed of datasets (:pull:`9002`).
  By `Mark Harfouche <https://github.com/hmaarrfk>`_.
- Performance improvement in ``open_datatree`` method for Zarr, netCDF4 and h5netcdf backends (:issue:`8994`, :pull:`9014`).
  By `Alfonso Ladino <https://github.com/aladinor>`_.


Bug fixes
~~~~~~~~~
- Preserve conversion of timezone-aware pandas Datetime arrays to numpy object arrays
  (:issue:`9026`, :pull:`9042`).
  By `Ilan Gold <https://github.com/ilan-gold>`_.
- :py:meth:`DataArrayResample.interpolate` and :py:meth:`DatasetResample.interpolate` method now
  support arbitrary kwargs such as ``order`` for polynomial interpolation (:issue:`8762`).
  By `Nicolas Karasiak <https://github.com/nkarasiak>`_.

Documentation
~~~~~~~~~~~~~
- Add link to CF Conventions on packed data and sentence on type determination in the I/O user guide (:issue:`9041`, :pull:`9045`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.


Internal Changes
~~~~~~~~~~~~~~~~
- Migrates remainder of ``io.py`` to ``xarray/core/datatree_io.py`` and
  ``TreeAttrAccessMixin`` into ``xarray/core/common.py`` (:pull:`9011`).
  By `Owen Littlejohns <https://github.com/owenlittlejohns>`_ and
  `Tom Nicholas <https://github.com/TomNicholas>`_.
- Compatibility with numpy 2 (:issue:`8844`, :pull:`8854`, :pull:`8946`).
  By `Justus Magin <https://github.com/keewis>`_ and `Stephan Hoyer <https://github.com/shoyer>`_.


.. _whats-new.2024.05.0:

v2024.05.0 (May 12, 2024)
-------------------------

This release brings support for pandas ExtensionArray objects, optimizations when reading Zarr, the ability to concatenate datasets without pandas indexes,
more compatibility fixes for the upcoming numpy 2.0, and the migration of most of the xarray-datatree project code into xarray ``main``!

Thanks to the 18 contributors to this release:
Aimilios Tsouvelekakis, Andrey Akinshin, Deepak Cherian, Eni Awowale, Ilan Gold, Illviljan, Justus Magin, Mark Harfouche, Matt Savoie, Maximilian Roos, Noah C. Benson, Pascal Bourgault, Ray Bell, Spencer Clark, Tom Nicholas, ignamv, owenlittlejohns, and saschahofmann.

New Features
~~~~~~~~~~~~
- New "random" method for converting to and from 360_day calendars (:pull:`8603`).
  By `Pascal Bourgault <https://github.com/aulemahal>`_.
- Xarray now makes a best attempt not to coerce :py:class:`pandas.api.extensions.ExtensionArray` to a numpy array
  by supporting 1D ``ExtensionArray`` objects internally where possible.  Thus, :py:class:`Dataset` objects initialized with a ``pd.Categorical``,
  for example, will retain the object.  However, one cannot do operations that are not possible on the ``ExtensionArray``
  then, such as broadcasting. (:issue:`5287`, :issue:`8463`, :pull:`8723`)
  By `Ilan Gold <https://github.com/ilan-gold>`_.
- :py:func:`testing.assert_allclose`/:py:func:`testing.assert_equal` now accept a new argument ``check_dims="transpose"``, controlling whether a transposed array is considered equal. (:issue:`5733`, :pull:`8991`)
  By `Ignacio Martinez Vazquez <https://github.com/ignamv>`_.
- Added the option to avoid automatically creating 1D pandas indexes in :py:meth:`Dataset.expand_dims()`, by passing the new kwarg
  ``create_index_for_new_dim=False``. (:pull:`8960`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Avoid automatically re-creating 1D pandas indexes in :py:func:`concat()`. Also added option to avoid creating 1D indexes for
  new dimension coordinates by passing the new kwarg ``create_index_for_new_dim=False``. (:issue:`8871`, :pull:`8872`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Breaking changes
~~~~~~~~~~~~~~~~
- The PyNIO backend has been deleted (:issue:`4491`, :pull:`7301`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- The minimum versions of some dependencies were changed, in particular our minimum supported pandas version is now Pandas 2.

  ===================== =========  =======
   Package                    Old      New
  ===================== =========  =======
   dask-core              2022.12   2023.4
   distributed            2022.12   2023.4
   h5py                       3.7      3.8
   matplotlib-base            3.6      3.7
   packaging                 22.0     23.1
   pandas                     1.5      2.0
   pydap                      3.3      3.4
   sparse                    0.13     0.14
   typing_extensions          4.4      4.5
   zarr                      2.13     2.14
  ===================== =========  =======

Bug fixes
~~~~~~~~~
- Following `an upstream bug fix
  <https://github.com/pandas-dev/pandas/issues/56147>`_ to
  :py:func:`pandas.date_range`, date ranges produced by
  :py:func:`xarray.cftime_range` with negative frequencies will now fall fully
  within the bounds of the provided start and end dates (:pull:`8999`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Enforces failures on CI when tests raise warnings from within xarray (:pull:`8974`)
  By `Maximilian Roos <https://github.com/max-sixty>`_
- Migrates ``formatting_html`` functionality for ``DataTree`` into ``xarray/core`` (:pull:`8930`)
  By `Eni Awowale <https://github.com/eni-awowale>`_, `Julia Signell <https://github.com/jsignell>`_
  and `Tom Nicholas <https://github.com/TomNicholas>`_.
- Migrates ``datatree_mapping`` functionality into ``xarray/core`` (:pull:`8948`)
  By `Matt Savoie <https://github.com/flamingbear>`_ `Owen Littlejohns
  <https://github.com/owenlittlejohns>`_ and `Tom Nicholas <https://github.com/TomNicholas>`_.
- Migrates ``extensions``, ``formatting`` and ``datatree_render`` functionality for
  ``DataTree`` into ``xarray/core``. Also migrates ``testing`` functionality into
  ``xarray/testing/assertions`` for ``DataTree``. (:pull:`8967`)
  By `Owen Littlejohns <https://github.com/owenlittlejohns>`_ and
  `Tom Nicholas <https://github.com/TomNicholas>`_.
- Migrates ``ops.py`` functionality into ``xarray/core/datatree_ops.py`` (:pull:`8976`)
  By `Matt Savoie <https://github.com/flamingbear>`_ and `Tom Nicholas <https://github.com/TomNicholas>`_.
- Migrates ``iterator`` functionality into ``xarray/core`` (:pull:`8879`)
  By `Owen Littlejohns <https://github.com/owenlittlejohns>`_, `Matt Savoie
  <https://github.com/flamingbear>`_ and `Tom Nicholas <https://github.com/TomNicholas>`_.
- ``transpose``, ``set_dims``, ``stack`` & ``unstack`` now use a ``dim`` kwarg
  rather than ``dims`` or ``dimensions``. This is the final change to make xarray methods
  consistent with their use of ``dim``. Using the existing kwarg will raise a
  warning.
  By `Maximilian Roos <https://github.com/max-sixty>`_

.. _whats-new.2024.03.0:

v2024.03.0 (Mar 29, 2024)
-------------------------

This release brings performance improvements for grouped and resampled quantile calculations, CF decoding improvements,
minor optimizations to distributed Zarr writes, and compatibility fixes for Numpy 2.0 and Pandas 3.0.

Thanks to the 18 contributors to this release:
Anderson Banihirwe, Christoph Hasse, Deepak Cherian, Etienne Schalk, Justus Magin, Kai Mühlbauer, Kevin Schwarzwald, Mark Harfouche, Martin, Matt Savoie, Maximilian Roos, Ray Bell, Roberto Chang, Spencer Clark, Tom Nicholas, crusaderky, owenlittlejohns, saschahofmann

New Features
~~~~~~~~~~~~
- Partial writes to existing chunks with ``region`` or ``append_dim`` will now raise an error
  (unless ``safe_chunks=False``); previously an error would only be raised on
  new variables. (:pull:`8459`, :issue:`8371`, :issue:`8882`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Grouped and resampling quantile calculations now use the vectorized algorithm in ``flox>=0.9.4`` if present.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Do not broadcast in arithmetic operations when global option ``arithmetic_broadcast=False``
  (:issue:`6806`, :pull:`8784`).
  By `Etienne Schalk <https://github.com/etienneschalk>`_ and `Deepak Cherian <https://github.com/dcherian>`_.
- Add the ``.oindex`` property to Explicitly Indexed Arrays for orthogonal indexing functionality. (:issue:`8238`, :pull:`8750`)
  By `Anderson Banihirwe <https://github.com/andersy005>`_.
- Add the ``.vindex`` property to Explicitly Indexed Arrays for vectorized indexing functionality. (:issue:`8238`, :pull:`8780`)
  By `Anderson Banihirwe <https://github.com/andersy005>`_.
- Expand use of ``.oindex`` and ``.vindex`` properties. (:pull:`8790`)
  By `Anderson Banihirwe <https://github.com/andersy005>`_ and `Deepak Cherian <https://github.com/dcherian>`_.
- Allow creating :py:class:`xr.Coordinates` objects with no indexes (:pull:`8711`)
  By `Benoit Bovy <https://github.com/benbovy>`_ and `Tom Nicholas
  <https://github.com/TomNicholas>`_.
- Enable plotting of ``datetime.dates``. (:issue:`8866`, :pull:`8873`)
  By `Sascha Hofmann <https://github.com/saschahofmann>`_.

Breaking changes
~~~~~~~~~~~~~~~~
- Don't allow overwriting index variables with ``to_zarr`` region writes. (:issue:`8589`, :pull:`8876`).
  By `Deepak Cherian <https://github.com/dcherian>`_.


Bug fixes
~~~~~~~~~
- The default ``freq`` parameter in :py:meth:`xr.date_range` and :py:meth:`xr.cftime_range` is
  set to ``'D'`` only if ``periods``, ``start``, or ``end`` are ``None`` (:issue:`8770`, :pull:`8774`).
  By `Roberto Chang <https://github.com/rjavierch>`_.
- Ensure that non-nanosecond precision :py:class:`numpy.datetime64` and
  :py:class:`numpy.timedelta64` values are cast to nanosecond precision values
  when used in :py:meth:`DataArray.expand_dims` and
  ::py:meth:`Dataset.expand_dims` (:pull:`8781`).  By `Spencer
  Clark <https://github.com/spencerkclark>`_.
- CF conform handling of ``_FillValue``/``missing_value`` and ``dtype`` in
  ``CFMaskCoder``/``CFScaleOffsetCoder`` (:issue:`2304`, :issue:`5597`,
  :issue:`7691`, :pull:`8713`, see also discussion in :pull:`7654`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Do not cast ``_FillValue``/``missing_value`` in ``CFMaskCoder`` if ``_Unsigned`` is provided
  (:issue:`8844`, :pull:`8852`).
- Adapt handling of copy keyword argument for numpy >= 2.0dev
  (:issue:`8844`, :pull:`8851`, :pull:`8865`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Import trapz/trapezoid depending on numpy version
  (:issue:`8844`, :pull:`8865`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Warn and return bytes undecoded in case of UnicodeDecodeError in h5netcdf-backend
  (:issue:`5563`, :pull:`8874`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Fix bug incorrectly disallowing creation of a dataset with a multidimensional coordinate variable with the same name as one of its dims.
  (:issue:`8884`, :pull:`8886`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Migrates ``treenode`` functionality into ``xarray/core`` (:pull:`8757`)
  By `Matt Savoie <https://github.com/flamingbear>`_ and `Tom Nicholas
  <https://github.com/TomNicholas>`_.
- Migrates ``datatree`` functionality into ``xarray/core``. (:pull:`8789`)
  By `Owen Littlejohns <https://github.com/owenlittlejohns>`_, `Matt Savoie
  <https://github.com/flamingbear>`_ and `Tom Nicholas <https://github.com/TomNicholas>`_.


.. _whats-new.2024.02.0:

v2024.02.0 (Feb 19, 2024)
-------------------------

This release brings size information to the text ``repr``, changes to the accepted frequency
strings, and various bug fixes.

Thanks to our 12 contributors:

Anderson Banihirwe, Deepak Cherian, Eivind Jahren, Etienne Schalk, Justus Magin, Marco Wolsza,
Mathias Hauser, Matt Savoie, Maximilian Roos, Rambaud Pierrick, Tom Nicholas

New Features
~~~~~~~~~~~~

- Added a simple ``nbytes`` representation in DataArrays and Dataset ``repr``.
  (:issue:`8690`, :pull:`8702`).
  By `Etienne Schalk <https://github.com/etienneschalk>`_.
- Allow negative frequency strings (e.g. ``"-1YE"``). These strings are for example used in
  :py:func:`date_range`, and :py:func:`cftime_range` (:pull:`8651`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Add :py:meth:`NamedArray.expand_dims`, :py:meth:`NamedArray.permute_dims` and
  :py:meth:`NamedArray.broadcast_to` (:pull:`8380`)
  By `Anderson Banihirwe <https://github.com/andersy005>`_.
- Xarray now defers to `flox's heuristics <https://flox.readthedocs.io/en/latest/implementation.html#heuristics>`_
  to set the default ``method`` for groupby problems. This only applies to ``flox>=0.9``.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- All ``quantile`` methods (e.g. :py:meth:`DataArray.quantile`) now use ``numbagg``
  for the calculation of nanquantiles (i.e., ``skipna=True``) if it is installed.
  This is currently limited to the linear interpolation method (`method='linear'`).
  (:issue:`7377`, :pull:`8684`)
  By `Marco Wolsza <https://github.com/maawoo>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- :py:func:`infer_freq` always returns the frequency strings as defined in pandas 2.2
  (:issue:`8612`, :pull:`8627`).
  By `Mathias Hauser <https://github.com/mathause>`_.

Deprecations
~~~~~~~~~~~~
- The ``dt.weekday_name`` parameter wasn't functional on modern pandas versions and has been
  removed. (:issue:`8610`, :pull:`8664`)
  By `Sam Coleman <https://github.com/nameloCmaS>`_.


Bug fixes
~~~~~~~~~

- Fixed a regression that prevented multi-index level coordinates being serialized after resetting
  or dropping the multi-index (:issue:`8628`, :pull:`8672`).
  By `Benoit Bovy <https://github.com/benbovy>`_.
- Fix bug with broadcasting when wrapping array API-compliant classes. (:issue:`8665`, :pull:`8669`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Ensure :py:meth:`DataArray.unstack` works when wrapping array API-compliant
  classes. (:issue:`8666`, :pull:`8668`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fix negative slicing of Zarr arrays without dask installed. (:issue:`8252`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Preserve chunks when writing time-like variables to zarr by enabling lazy CF encoding of time-like
  variables (:issue:`7132`, :issue:`8230`, :issue:`8432`, :pull:`8575`).
  By `Spencer Clark <https://github.com/spencerkclark>`_ and `Mattia Almansi <https://github.com/malmans2>`_.
- Preserve chunks when writing time-like variables to zarr by enabling their lazy encoding
  (:issue:`7132`, :issue:`8230`, :issue:`8432`, :pull:`8253`, :pull:`8575`; see also discussion in
  :pull:`8253`).
  By `Spencer Clark <https://github.com/spencerkclark>`_ and `Mattia Almansi <https://github.com/malmans2>`_.
- Raise an informative error if dtype encoding of time-like variables would lead to integer overflow
  or unsafe conversion from floating point to integer values (:issue:`8542`, :pull:`8575`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Raise an error when unstacking a MultiIndex that has duplicates as this would lead to silent data
  loss (:issue:`7104`, :pull:`8737`).
  By `Mathias Hauser <https://github.com/mathause>`_.

Documentation
~~~~~~~~~~~~~
- Fix ``variables`` arg typo in ``Dataset.sortby()`` docstring (:issue:`8663`, :pull:`8670`)
  By `Tom Vo <https://github.com/tomvothecoder>`_.
- Fixed documentation where the use of the depreciated pandas frequency string prevented the
  documentation from being built. (:pull:`8638`)
  By `Sam Coleman <https://github.com/nameloCmaS>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- ``DataArray.dt`` now raises an ``AttributeError`` rather than a ``TypeError`` when the data isn't
  datetime-like. (:issue:`8718`, :pull:`8724`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Move ``parallelcompat`` and ``chunk managers`` modules from ``xarray/core`` to
  ``xarray/namedarray``. (:pull:`8319`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_ and `Anderson Banihirwe <https://github.com/andersy005>`_.
- Imports ``datatree`` repository and history into internal location. (:pull:`8688`)
  By `Matt Savoie <https://github.com/flamingbear>`_, `Justus Magin <https://github.com/keewis>`_
  and `Tom Nicholas <https://github.com/TomNicholas>`_.
- Adds :py:func:`open_datatree` into ``xarray/backends`` (:pull:`8697`)
  By `Matt Savoie <https://github.com/flamingbear>`_ and `Tom Nicholas
  <https://github.com/TomNicholas>`_.
- Refactor :py:meth:`xarray.core.indexing.DaskIndexingAdapter.__getitem__` to remove an unnecessary
  rewrite of the indexer key (:issue:`8377`, :pull:`8758`)
  By `Anderson Banihirwe <https://github.com/andersy005>`_.

.. _whats-new.2024.01.1:

v2024.01.1 (23 Jan, 2024)
-------------------------

This release is to fix a bug with the rendering of the documentation, but it also includes changes to the handling of pandas frequency strings.

Breaking changes
~~~~~~~~~~~~~~~~

- Following pandas, :py:meth:`infer_freq` will return ``"YE"``, instead of ``"Y"`` (formerly ``"A"``).
  This is to be consistent with the deprecation of the latter frequency string in pandas 2.2.
  This is a follow up to :pull:`8415` (:issue:`8612`, :pull:`8642`).
  By `Mathias Hauser <https://github.com/mathause>`_.

Deprecations
~~~~~~~~~~~~

- Following pandas, the frequency string ``"Y"`` (formerly ``"A"``) is deprecated in
  favor of ``"YE"``. These strings are used, for example, in :py:func:`date_range`,
  :py:func:`cftime_range`, :py:meth:`DataArray.resample`, and :py:meth:`Dataset.resample`
  among others (:issue:`8612`, :pull:`8629`).
  By `Mathias Hauser <https://github.com/mathause>`_.

Documentation
~~~~~~~~~~~~~

- Pin ``sphinx-book-theme`` to ``1.0.1`` to fix a rendering issue with the sidebar in the docs. (:issue:`8619`, :pull:`8632`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

.. _whats-new.2024.01.0:

v2024.01.0 (17 Jan, 2024)
-------------------------

This release brings support for weights in correlation and covariance functions,
a new ``DataArray.cumulative`` aggregation, improvements to ``xr.map_blocks``,
an update to our minimum dependencies, and various bugfixes.

Thanks to our 17 contributors to this release:

Abel Aoun, Deepak Cherian, Illviljan, Johan Mathe, Justus Magin, Kai Mühlbauer,
Llorenç Lledó, Mark Harfouche, Markel, Mathias Hauser, Maximilian Roos, Michael Niklas,
Niclas Rieger, Sébastien Celles, Tom Nicholas, Trinh Quoc Anh, and crusaderky.

New Features
~~~~~~~~~~~~

- :py:meth:`xr.cov` and :py:meth:`xr.corr` now support using weights (:issue:`8527`, :pull:`7392`).
  By `Llorenç Lledó <https://github.com/lluritu>`_.
- Accept the compression arguments new in netCDF 1.6.0 in the netCDF4 backend.
  See `netCDF4 documentation <https://unidata.github.io/netcdf4-python/#efficient-compression-of-netcdf-variables>`_ for details.
  Note that some new compression filters needs plugins to be installed which may not be available in all netCDF distributions.
  By `Markel García-Díez <https://github.com/markelg>`_. (:issue:`6929`, :pull:`7551`)
- Add :py:meth:`DataArray.cumulative` & :py:meth:`Dataset.cumulative` to compute
  cumulative aggregations, such as ``sum``, along a dimension — for example
  ``da.cumulative('time').sum()``. This is similar to pandas' ``.expanding``,
  and mostly equivalent to ``.cumsum`` methods, or to
  :py:meth:`DataArray.rolling` with a window length equal to the dimension size.
  By `Maximilian Roos <https://github.com/max-sixty>`_. (:pull:`8512`)
- Decode/Encode netCDF4 enums and store the enum definition in dataarrays' dtype metadata.
  If multiple variables share the same enum in netCDF4, each dataarray will have its own
  enum definition in their respective dtype metadata.
  By `Abel Aoun <https://github.com/bzah>`_. (:issue:`8144`, :pull:`8147`)

Breaking changes
~~~~~~~~~~~~~~~~

- The minimum versions of some dependencies were changed (:pull:`8586`):

  ===================== =========  ========
   Package                    Old      New
  ===================== =========  ========
   cartopy                   0.20      0.21
   dask-core               2022.7   2022.12
   distributed             2022.7   2022.12
   flox                       0.5      0.7
   iris                       3.2      3.4
   matplotlib-base            3.5      3.6
   numpy                     1.22     1.23
   numba                     0.55     0.56
   packaging                 21.3     22.0
   seaborn                   0.11     0.12
   scipy                      1.8     1.10
   typing_extensions          4.3      4.4
   zarr                      2.12     2.13
  ===================== =========  ========

Deprecations
~~~~~~~~~~~~

- The ``squeeze`` kwarg to GroupBy is now deprecated. (:issue:`2157`, :pull:`8507`)
  By `Deepak Cherian <https://github.com/dcherian>`_.

Bug fixes
~~~~~~~~~

- Support non-string hashable dimensions in :py:class:`xarray.DataArray` (:issue:`8546`, :pull:`8559`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Reverse index output of bottleneck's rolling move_argmax/move_argmin functions (:issue:`8541`, :pull:`8552`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Vendor ``SerializableLock`` from dask and use as default lock for netcdf4 backends (:issue:`8442`, :pull:`8571`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Add tests and fixes for empty :py:class:`CFTimeIndex`, including broken html repr (:issue:`7298`, :pull:`8600`).
  By `Mathias Hauser <https://github.com/mathause>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- The implementation of :py:func:`map_blocks` has changed to minimize graph size and duplication of data.
  This should be a strict improvement even though the graphs are not always embarrassingly parallel any more.
  Please open an issue if you spot a regression. (:pull:`8412`, :issue:`8409`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Remove null values before plotting. (:pull:`8535`).
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Redirect cumulative reduction functions internally through the :py:class:`ChunkManagerEntryPoint`,
  potentially allowing :py:meth:`~xarray.DataArray.ffill` and :py:meth:`~xarray.DataArray.bfill` to
  use non-dask chunked array types.
  (:pull:`8019`) By `Tom Nicholas <https://github.com/TomNicholas>`_.

.. _whats-new.2023.12.0:

v2023.12.0 (2023 Dec 08)
------------------------

This release brings new `hypothesis <https://hypothesis.works/>`_ strategies for testing, significantly faster rolling aggregations as well as
``ffill`` and ``bfill`` with ``numbagg``, a new :py:meth:`Dataset.eval` method, and improvements to
reading and writing Zarr arrays (including a new ``"a-"`` mode).

Thanks to our 16 contributors:

Anderson Banihirwe, Ben Mares, Carl Andersson, Deepak Cherian, Doug Latornell, Gregorio L. Trevisan, Illviljan, Jens Hedegaard Nielsen, Justus Magin, Mathias Hauser, Max Jones, Maximilian Roos, Michael Niklas, Patrick Hoefler, Ryan Abernathey, Tom Nicholas

New Features
~~~~~~~~~~~~

- Added hypothesis strategies for generating :py:class:`xarray.Variable` objects containing arbitrary data, useful for parametrizing downstream tests.
  Accessible under :py:mod:`testing.strategies`, and documented in a new page on testing in the User Guide.
  (:issue:`6911`, :pull:`8404`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- :py:meth:`rolling` uses `numbagg <https://github.com/numbagg/numbagg>`_ for
  most of its computations by default. Numbagg is up to 5x faster than bottleneck
  where parallelization is possible. Where parallelization isn't possible — for
  example a 1D array — it's about the same speed as bottleneck, and 2-5x faster
  than pandas' default functions. (:pull:`8493`). numbagg is an optional
  dependency, so requires installing separately.
- Use a concise format when plotting datetime arrays. (:pull:`8449`).
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Avoid overwriting unchanged existing coordinate variables when appending with :py:meth:`Dataset.to_zarr` by setting ``mode='a-'``.
  By `Ryan Abernathey <https://github.com/rabernat>`_ and `Deepak Cherian <https://github.com/dcherian>`_.
- :py:meth:`~xarray.DataArray.rank` now operates on dask-backed arrays, assuming
  the core dim has exactly one chunk. (:pull:`8475`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Add a :py:meth:`Dataset.eval` method, similar to the pandas' method of the
  same name. (:pull:`7163`). This is currently marked as experimental and
  doesn't yet support the ``numexpr`` engine.
- :py:meth:`Dataset.drop_vars` & :py:meth:`DataArray.drop_vars` allow passing a
  callable, similar to :py:meth:`Dataset.where` & :py:meth:`Dataset.sortby` & others.
  (:pull:`8511`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- Explicitly warn when creating xarray objects with repeated dimension names.
  Such objects will also now raise when :py:meth:`DataArray.get_axis_num` is called,
  which means many functions will raise.
  This latter change is technically a breaking change, but whilst allowed,
  this behaviour was never actually supported! (:issue:`3731`, :pull:`8491`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Deprecations
~~~~~~~~~~~~
- As part of an effort to standardize the API, we're renaming the ``dims``
  keyword arg to ``dim`` for the minority of functions which current use
  ``dims``. This started with :py:func:`xarray.dot` & :py:meth:`DataArray.dot`
  and we'll gradually roll this out across all functions. The warnings are
  currently ``PendingDeprecationWarning``, which are silenced by default. We'll
  convert these to ``DeprecationWarning`` in a future release.
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Raise a ``FutureWarning`` warning that the type of :py:meth:`Dataset.dims` will be changed
  from a mapping of dimension names to lengths to a set of dimension names.
  This is to increase consistency with :py:meth:`DataArray.dims`.
  To access a mapping of dimension names to lengths please use :py:meth:`Dataset.sizes`.
  The same change also applies to ``DatasetGroupBy.dims``.
  (:issue:`8496`, :pull:`8500`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- :py:meth:`Dataset.drop` & :py:meth:`DataArray.drop` are now deprecated, since pending deprecation for
  several years. :py:meth:`DataArray.drop_sel` & :py:meth:`DataArray.drop_var`
  replace them for labels & variables respectively. (:pull:`8497`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.

Bug fixes
~~~~~~~~~

- Fix dtype inference for ``pd.CategoricalIndex`` when categories are backed by a ``pd.ExtensionDtype`` (:pull:`8481`)
- Fix writing a variable that requires transposing when not writing to a region (:pull:`8484`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Static typing of ``p0`` and ``bounds`` arguments of :py:func:`xarray.DataArray.curvefit` and :py:func:`xarray.Dataset.curvefit`
  was changed to ``Mapping`` (:pull:`8502`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Fix typing of :py:func:`xarray.DataArray.to_netcdf` and :py:func:`xarray.Dataset.to_netcdf`
  when ``compute`` is evaluated to bool instead of a Literal (:pull:`8268`).
  By `Jens Hedegaard Nielsen <https://github.com/jenshnielsen>`_.

Documentation
~~~~~~~~~~~~~

- Added illustration of updating the time coordinate values of a resampled dataset using
  time offset arithmetic.
  This is the recommended technique to replace the use of the deprecated ``loffset`` parameter
  in ``resample`` (:pull:`8479`).
  By `Doug Latornell <https://github.com/douglatornell>`_.
- Improved error message when attempting to get a variable which doesn't exist from a Dataset.
  (:pull:`8474`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Fix default value of ``combine_attrs`` in :py:func:`xarray.combine_by_coords` (:pull:`8471`)
  By `Gregorio L. Trevisan <https://github.com/gtrevisan>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- :py:meth:`DataArray.bfill` & :py:meth:`DataArray.ffill` now use numbagg <https://github.com/numbagg/numbagg>`_ by
  default, which is up to 5x faster where parallelization is possible. (:pull:`8339`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Update mypy version to 1.7 (:issue:`8448`, :pull:`8501`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.

.. _whats-new.2023.11.0:

v2023.11.0 (Nov 16, 2023)
-------------------------


.. tip::

     `This is our 10th year anniversary release! <https://github.com/pydata/xarray/discussions/8462>`_ Thank you for your love and support.


This release brings the ability to use ``opt_einsum`` for :py:func:`xarray.dot` by default,
support for auto-detecting ``region`` when writing partial datasets to Zarr, and the use of h5py
drivers with ``h5netcdf``.

Thanks to the 19 contributors to this release:
Aman Bagrecha, Anderson Banihirwe, Ben Mares, Deepak Cherian, Dimitri Papadopoulos Orfanos, Ezequiel Cimadevilla Alvarez,
Illviljan, Justus Magin, Katelyn FitzGerald, Kai Muehlbauer, Martin Durant, Maximilian Roos, Metamess, Sam Levang, Spencer Clark, Tom Nicholas, mgunyho, templiert

New Features
~~~~~~~~~~~~

- Use `opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/>`_ for :py:func:`xarray.dot` by default if installed.
  By `Deepak Cherian <https://github.com/dcherian>`_. (:issue:`7764`, :pull:`8373`).
- Add ``DataArray.dt.total_seconds()`` method to match the Pandas API. (:pull:`8435`).
  By `Ben Mares <https://github.com/maresb>`_.
- Allow passing ``region="auto"`` in  :py:meth:`Dataset.to_zarr` to automatically infer the
  region to write in the original store. Also implement automatic transpose when dimension
  order does not match the original store. (:issue:`7702`, :issue:`8421`, :pull:`8434`).
  By `Sam Levang <https://github.com/slevang>`_.
- Allow the usage of h5py drivers (eg: ros3) via h5netcdf (:pull:`8360`).
  By `Ezequiel Cimadevilla <https://github.com/zequihg50>`_.
- Enable VLEN string fill_values, preserve VLEN string dtypes (:issue:`1647`, :issue:`7652`, :issue:`7868`, :pull:`7869`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Breaking changes
~~~~~~~~~~~~~~~~
- drop support for `cdms2 <https://github.com/CDAT/cdms>`_. Please use
  `xcdat <https://github.com/xCDAT/xcdat>`_ instead (:pull:`8441`).
  By `Justus Magin <https://github.com/keewis>`_.
- Following pandas, :py:meth:`infer_freq` will return ``"Y"``, ``"YS"``,
  ``"QE"``, ``"ME"``, ``"h"``, ``"min"``, ``"s"``, ``"ms"``, ``"us"``, or
  ``"ns"`` instead of ``"A"``, ``"AS"``, ``"Q"``, ``"M"``, ``"H"``, ``"T"``,
  ``"S"``, ``"L"``, ``"U"``, or ``"N"``.  This is to be consistent with the
  deprecation of the latter frequency strings (:issue:`8394`, :pull:`8415`). By
  `Spencer Clark <https://github.com/spencerkclark>`_.
- Bump minimum tested pint version to ``>=0.22``. By `Deepak Cherian <https://github.com/dcherian>`_.
- Minimum supported versions for the following packages have changed: ``h5py >=3.7``, ``h5netcdf>=1.1``.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Deprecations
~~~~~~~~~~~~
- The PseudoNetCDF backend has been removed. By `Deepak Cherian <https://github.com/dcherian>`_.
- Supplying dimension-ordered sequences to :py:meth:`DataArray.chunk` &
  :py:meth:`Dataset.chunk` is deprecated in favor of supplying a dictionary of
  dimensions, or a single ``int`` or ``"auto"`` argument covering all
  dimensions. Xarray favors using dimensions names rather than positions, and
  this was one place in the API where dimension positions were used.
  (:pull:`8341`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Following pandas, the frequency strings ``"A"``, ``"AS"``, ``"Q"``, ``"M"``,
  ``"H"``, ``"T"``, ``"S"``, ``"L"``, ``"U"``, and ``"N"`` are deprecated in
  favor of ``"Y"``, ``"YS"``, ``"QE"``, ``"ME"``, ``"h"``, ``"min"``, ``"s"``,
  ``"ms"``, ``"us"``, and ``"ns"``, respectively.  These strings are used, for
  example, in :py:func:`date_range`, :py:func:`cftime_range`,
  :py:meth:`DataArray.resample`, and :py:meth:`Dataset.resample` among others
  (:issue:`8394`, :pull:`8415`).  By `Spencer Clark
  <https://github.com/spencerkclark>`_.
- Rename :py:meth:`Dataset.to_array` to  :py:meth:`Dataset.to_dataarray` for
  consistency with :py:meth:`DataArray.to_dataset` &
  :py:func:`open_dataarray` functions. This is a "soft" deprecation — the
  existing methods work and don't raise any warnings, given the relatively small
  benefits of the change.
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Finally remove ``keep_attrs`` kwarg from :py:meth:`DataArray.resample` and
  :py:meth:`Dataset.resample`. These were deprecated a long time ago.
  By `Deepak Cherian <https://github.com/dcherian>`_.

Bug fixes
~~~~~~~~~

- Port `bug fix from pandas <https://github.com/pandas-dev/pandas/pull/55283>`_
  to eliminate the adjustment of resample bin edges in the case that the
  resampling frequency has units of days and is greater than one day
  (e.g. ``"2D"``, ``"3D"`` etc.) and the ``closed`` argument is set to
  ``"right"`` to xarray's implementation of resample for data indexed by a
  :py:class:`CFTimeIndex` (:pull:`8393`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Fix to once again support date offset strings as input to the loffset
  parameter of resample and test this functionality (:pull:`8422`, :issue:`8399`).
  By `Katelyn FitzGerald <https://github.com/kafitzgerald>`_.
- Fix a bug where :py:meth:`DataArray.to_dataset` silently drops a variable
  if a coordinate with the same name already exists (:pull:`8433`, :issue:`7823`).
  By `András Gunyhó <https://github.com/mgunyho>`_.
- Fix for :py:meth:`DataArray.to_zarr` & :py:meth:`Dataset.to_zarr` to close
  the created zarr store when passing a path with ``.zip`` extension (:pull:`8425`).
  By `Carl Andersson <https://github.com/CarlAndersson>`_.

Documentation
~~~~~~~~~~~~~
- Small updates to documentation on distributed writes: See :ref:`io.zarr.appending` to Zarr.
  By `Deepak Cherian <https://github.com/dcherian>`_.

.. _whats-new.2023.10.1:

v2023.10.1 (19 Oct, 2023)
-------------------------

This release updates our minimum numpy version in ``pyproject.toml`` to 1.22,
consistent with our documentation below.

.. _whats-new.2023.10.0:

v2023.10.0 (19 Oct, 2023)
-------------------------

This release brings performance enhancements to reading Zarr datasets, the ability to use `numbagg <https://github.com/numbagg/numbagg>`_ for reductions,
an expansion in API for ``rolling_exp``, fixes two regressions with datetime decoding,
and many other bugfixes and improvements. Groupby reductions will also use ``numbagg`` if ``flox>=0.8.1`` and ``numbagg`` are both installed.

Thanks to our 13 contributors:
Anderson Banihirwe, Bart Schilperoort, Deepak Cherian, Illviljan, Kai Mühlbauer, Mathias Hauser, Maximilian Roos, Michael Niklas, Pieter Eendebak, Simon Høxbro Hansen, Spencer Clark, Tom White, olimcc

New Features
~~~~~~~~~~~~
- Support high-performance reductions with `numbagg <https://github.com/numbagg/numbagg>`_.
  This is enabled by default if ``numbagg`` is installed.
  By `Deepak Cherian <https://github.com/dcherian>`_. (:pull:`8316`)
- Add ``corr``, ``cov``, ``std`` & ``var`` to ``.rolling_exp``.
  By `Maximilian Roos <https://github.com/max-sixty>`_. (:pull:`8307`)
- :py:meth:`DataArray.where` & :py:meth:`Dataset.where` accept a callable for
  the ``other`` parameter, passing the object as the only argument. Previously,
  this was only valid for the ``cond`` parameter. (:issue:`8255`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- ``.rolling_exp`` functions can now take a ``min_weight`` parameter, to only
  output values when there are sufficient recent non-nan values.
  ``numbagg>=0.3.1`` is required. (:pull:`8285`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- :py:meth:`DataArray.sortby` & :py:meth:`Dataset.sortby` accept a callable for
  the ``variables`` parameter, passing the object as the only argument.
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- ``.rolling_exp`` functions can now operate on dask-backed arrays, assuming the
  core dim has exactly one chunk. (:pull:`8284`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- Made more arguments keyword-only (e.g. ``keep_attrs``, ``skipna``) for many :py:class:`xarray.DataArray` and
  :py:class:`xarray.Dataset` methods (:pull:`6403`). By `Mathias Hauser <https://github.com/mathause>`_.
- :py:meth:`Dataset.to_zarr` & :py:meth:`DataArray.to_zarr` require keyword
  arguments after the initial 7 positional arguments.
  By `Maximilian Roos <https://github.com/max-sixty>`_.


Deprecations
~~~~~~~~~~~~
- Rename :py:meth:`Dataset.reset_encoding` & :py:meth:`DataArray.reset_encoding`
  to :py:meth:`Dataset.drop_encoding` & :py:meth:`DataArray.drop_encoding` for
  consistency with other ``drop`` & ``reset`` methods — ``drop`` generally
  removes something, while ``reset`` generally resets to some default or
  standard value. (:pull:`8287`, :issue:`8259`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.

Bug fixes
~~~~~~~~~

- :py:meth:`DataArray.rename` & :py:meth:`Dataset.rename` would emit a warning
  when the operation was a no-op. (:issue:`8266`)
  By `Simon Hansen <https://github.com/hoxbro>`_.
- Fixed a regression introduced in the previous release checking time-like units
  when encoding/decoding masked data (:issue:`8269`, :pull:`8277`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

- Fix datetime encoding precision loss regression introduced in the previous
  release for datetimes encoded with units requiring floating point values, and
  a reference date not equal to the first value of the datetime array
  (:issue:`8271`, :pull:`8272`). By `Spencer Clark
  <https://github.com/spencerkclark>`_.

- Fix excess metadata requests when using a Zarr store. Prior to this, metadata
  was re-read every time data was retrieved from the array, now metadata is retrieved only once
  when they array is initialized.
  (:issue:`8290`, :pull:`8297`).
  By `Oliver McCormack <https://github.com/olimcc>`_.

- Fix to_zarr ending in a ReadOnlyError when consolidated metadata was used and the
  write_empty_chunks was provided.
  (:issue:`8323`, :pull:`8326`)
  By `Matthijs Amesz <https://github.com/Metamess>`_.


Documentation
~~~~~~~~~~~~~

- Added page on the interoperability of xarray objects.
  (:pull:`7992`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added xarray-regrid to the list of xarray related projects (:pull:`8272`).
  By `Bart Schilperoort <https://github.com/BSchilperoort>`_.


Internal Changes
~~~~~~~~~~~~~~~~

- More improvements to support the Python `array API standard <https://data-apis.org/array-api/latest/>`_
  by using duck array ops in more places in the codebase. (:pull:`8267`)
  By `Tom White <https://github.com/tomwhite>`_.


.. _whats-new.2023.09.0:

v2023.09.0 (Sep 26, 2023)
-------------------------

This release continues work on the new :py:class:`xarray.Coordinates` object, allows to provide ``preferred_chunks`` when
reading from netcdf files, enables :py:func:`xarray.apply_ufunc` to handle missing core dimensions and fixes several bugs.

Thanks to the 24 contributors to this release: Alexander Fischer, Amrest Chinkamol, Benoit Bovy, Darsh Ranjan, Deepak Cherian,
Gianfranco Costamagna, Gregorio L. Trevisan, Illviljan, Joe Hamman, JR, Justus Magin, Kai Mühlbauer, Kian-Meng Ang, Kyle Sunden,
Martin Raspaud, Mathias Hauser, Mattia Almansi, Maximilian Roos, András Gunyhó, Michael Niklas, Richard Kleijn, Riulinchen,
Tom Nicholas and Wiktor Kraśnicki.

We welcome the following new contributors to Xarray!: Alexander Fischer, Amrest Chinkamol, Darsh Ranjan, Gianfranco Costamagna, Gregorio L. Trevisan,
Kian-Meng Ang, Riulinchen and Wiktor Kraśnicki.

New Features
~~~~~~~~~~~~

- Added the :py:meth:`Coordinates.assign` method that can be used to combine
  different collections of coordinates prior to assign them to a Dataset or
  DataArray (:pull:`8102`) at once.
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Provide ``preferred_chunks`` for data read from netcdf files (:issue:`1440`, :pull:`7948`).
  By `Martin Raspaud <https://github.com/mraspaud>`_.
- Added ``on_missing_core_dims`` to :py:meth:`apply_ufunc` to allow for copying or
  dropping a :py:class:`Dataset`'s variables with missing core dimensions (:pull:`8138`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- The :py:class:`Coordinates` constructor now creates a (pandas) index by
  default for each dimension coordinate. To keep the previous behavior (no index
  created), pass an empty dictionary to ``indexes``. The constructor now also
  extracts and add the indexes from another :py:class:`Coordinates` object
  passed via ``coords`` (:pull:`8107`).
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Static typing of ``xlim`` and ``ylim`` arguments in plotting functions now must
  be ``tuple[float, float]`` to align with matplotlib requirements. (:issue:`7802`, :pull:`8030`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.

Deprecations
~~~~~~~~~~~~

- Deprecate passing a :py:class:`pandas.MultiIndex` object directly to the
  :py:class:`Dataset` and :py:class:`DataArray` constructors as well as to
  :py:meth:`Dataset.assign` and :py:meth:`Dataset.assign_coords`.
  A new Xarray :py:class:`Coordinates` object has to be created first using
  :py:meth:`Coordinates.from_pandas_multiindex` (:pull:`8094`).
  By `Benoît Bovy <https://github.com/benbovy>`_.

Bug fixes
~~~~~~~~~

- Improved static typing of reduction methods (:pull:`6746`).
  By `Richard Kleijn <https://github.com/rhkleijn>`_.
- Fix bug where empty attrs would generate inconsistent tokens (:issue:`6970`, :pull:`8101`).
  By `Mattia Almansi <https://github.com/malmans2>`_.
- Improved handling of multi-coordinate indexes when updating coordinates, including bug fixes
  (and improved warnings for deprecated features) for pandas multi-indexes (:pull:`8094`).
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Fixed a bug in :py:func:`merge` with ``compat='minimal'`` where the coordinate
  names were not updated properly internally (:issue:`7405`, :issue:`7588`,
  :pull:`8104`).
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Fix bug where :py:class:`DataArray` instances on the right-hand side
  of :py:meth:`DataArray.__setitem__` lose dimension names (:issue:`7030`, :pull:`8067`).
  By `Darsh Ranjan <https://github.com/dranjan>`_.
- Return ``float64`` in presence of ``NaT`` in :py:class:`~core.accessor_dt.DatetimeAccessor` and
  special case ``NaT`` handling in :py:meth:`~core.accessor_dt.DatetimeAccessor.isocalendar`
  (:issue:`7928`, :pull:`8084`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Fix :py:meth:`~core.rolling.DatasetRolling.construct` with stride on Datasets without indexes.
  (:issue:`7021`, :pull:`7578`).
  By `Amrest Chinkamol <https://github.com/p4perf4ce>`_ and `Michael Niklas <https://github.com/headtr1ck>`_.
- Calling plot with kwargs ``col``, ``row`` or ``hue`` no longer squeezes dimensions passed via these arguments
  (:issue:`7552`, :pull:`8174`).
  By `Wiktor Kraśnicki <https://github.com/wkrasnicki>`_.
- Fixed a bug where casting from ``float`` to ``int64`` (undefined for ``NaN``) led to varying issues (:issue:`7817`, :issue:`7942`, :issue:`7790`, :issue:`6191`, :issue:`7096`,
  :issue:`1064`, :pull:`7827`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Fixed a bug where inaccurate ``coordinates`` silently failed to decode variable (:issue:`1809`, :pull:`8195`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_
- ``.rolling_exp`` functions no longer mistakenly lose non-dimensioned coords
  (:issue:`6528`, :pull:`8114`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- In the event that user-provided datetime64/timedelta64 units and integer dtype encoding parameters conflict with each other, override the units to preserve an integer dtype for most faithful serialization to disk (:issue:`1064`, :pull:`8201`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Static typing of dunder ops methods (like :py:meth:`DataArray.__eq__`) has been fixed.
  Remaining issues are upstream problems (:issue:`7780`, :pull:`8204`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Fix type annotation for ``center`` argument of plotting methods (like :py:meth:`xarray.plot.dataarray_plot.pcolormesh`) (:pull:`8261`).
  By `Pieter Eendebak <https://github.com/eendebakpt>`_.

Documentation
~~~~~~~~~~~~~

- Make documentation of :py:meth:`DataArray.where` clearer (:issue:`7767`, :pull:`7955`).
  By `Riulinchen <https://github.com/Riulinchen>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Many error messages related to invalid dimensions or coordinates now always show the list of valid dims/coords (:pull:`8079`).
  By `András Gunyhó <https://github.com/mgunyho>`_.
- Refactor of encoding and decoding times/timedeltas to preserve nanosecond resolution in arrays that contain missing values (:pull:`7827`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Transition ``.rolling_exp`` functions to use ``.apply_ufunc`` internally rather
  than ``.reduce``, as the start of a broader effort to move non-reducing
  functions away from ```.reduce``, (:pull:`8114`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Test range of fill_value's in test_interpolate_pd_compat (:issue:`8146`, :pull:`8189`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

.. _whats-new.2023.08.0:

v2023.08.0 (Aug 18, 2023)
-------------------------

This release brings changes to minimum dependencies, allows reading of datasets where a dimension name is
associated with a multidimensional variable (e.g. finite volume ocean model output), and introduces
a new :py:class:`xarray.Coordinates` object.

Thanks to the 16 contributors to this release: Anderson Banihirwe, Articoking, Benoit Bovy, Deepak Cherian, Harshitha, Ian Carroll,
Joe Hamman, Justus Magin, Peter Hill, Rachel Wegener, Riley Kuttruff, Thomas Nicholas, Tom Nicholas, ilgast, quantsnus, vallirep

Announcements
~~~~~~~~~~~~~

The :py:class:`xarray.Variable` class is being refactored out to a new project title 'namedarray'.
See the `design doc <https://github.com/pydata/xarray/blob/main/design_notes/named_array_design_doc.md>`_ for more
details. Reach out to us on this [discussion topic](https://github.com/pydata/xarray/discussions/8080) if you have any thoughts.

New Features
~~~~~~~~~~~~

- :py:class:`Coordinates` can now be constructed independently of any Dataset or
  DataArray (it is also returned by the :py:attr:`Dataset.coords` and
  :py:attr:`DataArray.coords` properties). ``Coordinates`` objects are useful for
  passing both coordinate variables and indexes to new Dataset / DataArray objects,
  e.g., via their constructor or via :py:meth:`Dataset.assign_coords`. We may also
  wrap coordinate variables in a ``Coordinates`` object in order to skip
  the automatic creation of (pandas) indexes for dimension coordinates.
  The :py:class:`Coordinates.from_pandas_multiindex` constructor may be used to
  create coordinates directly from a :py:class:`pandas.MultiIndex` object (it is
  preferred over passing it directly as coordinate data, which may be deprecated soon).
  Like Dataset and DataArray objects, ``Coordinates`` objects may now be used in
  :py:func:`align` and :py:func:`merge`.
  (:issue:`6392`, :pull:`7368`).
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Visually group together coordinates with the same indexes in the index section of the text repr (:pull:`7225`).
  By `Justus Magin <https://github.com/keewis>`_.
- Allow creating Xarray objects where a multidimensional variable shares its name
  with a dimension. Examples include output from finite volume models like FVCOM.
  (:issue:`2233`, :pull:`7989`)
  By `Deepak Cherian <https://github.com/dcherian>`_ and `Benoit Bovy <https://github.com/benbovy>`_.
- When outputting :py:class:`Dataset` objects as Zarr via :py:meth:`Dataset.to_zarr`,
  user can now specify that chunks that will contain no valid data will not be written.
  Originally, this could be done by specifying ``"write_empty_chunks": True`` in the
  ``encoding`` parameter; however, this setting would not carry over when appending new
  data to an existing dataset. (:issue:`8009`) Requires ``zarr>=2.11``.


Breaking changes
~~~~~~~~~~~~~~~~

- The minimum versions of some dependencies were changed (:pull:`8022`):

  ===================== =========  ========
   Package                    Old      New
  ===================== =========  ========
   boto3                     1.20     1.24
   cftime                     1.5      1.6
   dask-core               2022.1   2022.7
   distributed             2022.1   2022.7
   hfnetcdf                  0.13      1.0
   iris                       3.1      3.2
   lxml                       4.7      4.9
   netcdf4                  1.5.7    1.6.0
   numpy                     1.21     1.22
   pint                      0.18     0.19
   pydap                      3.2      3.3
   rasterio                   1.2      1.3
   scipy                      1.7      1.8
   toolz                     0.11     0.12
   typing_extensions          4.0      4.3
   zarr                      2.10     2.12
   numbagg                    0.1    0.2.1
  ===================== =========  ========

Documentation
~~~~~~~~~~~~~

- Added page on the internal design of xarray objects.
  (:pull:`7991`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added examples to docstrings of :py:meth:`Dataset.assign_attrs`, :py:meth:`Dataset.broadcast_equals`,
  :py:meth:`Dataset.equals`, :py:meth:`Dataset.identical`, :py:meth:`Dataset.expand_dims`,:py:meth:`Dataset.drop_vars`
  (:issue:`6793`, :pull:`7937`) By `Harshitha <https://github.com/harshitha1201>`_.
- Add docstrings for the :py:class:`Index` base class and add some documentation on how to
  create custom, Xarray-compatible indexes (:pull:`6975`)
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Added a page clarifying the role of Xarray core team members.
  (:pull:`7999`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fixed broken links in "See also" section of :py:meth:`Dataset.count` (:issue:`8055`, :pull:`8057`)
  By `Articoking <https://github.com/Articoking>`_.
- Extended the glossary by adding terms Aligning, Broadcasting, Merging, Concatenating, Combining, lazy,
  labeled, serialization, indexing (:issue:`3355`, :pull:`7732`)
  By `Harshitha <https://github.com/harshitha1201>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- :py:func:`as_variable` now consistently includes the variable name in any exceptions
  raised. (:pull:`7995`). By `Peter Hill <https://github.com/ZedThree>`_
- :py:func:`encode_dataset_coordinates` now sorts coordinates automatically assigned to
  ``coordinates`` attributes during serialization (:issue:`8026`, :pull:`8034`).
  `By Ian Carroll <https://github.com/itcarroll>`_.

.. _whats-new.2023.07.0:

v2023.07.0 (July 17, 2023)
--------------------------

This release brings improvements to the documentation on wrapping numpy-like arrays, improved docstrings, and bug fixes.

Deprecations
~~~~~~~~~~~~

- ``hue_style`` is being deprecated for scatter plots. (:issue:`7907`, :pull:`7925`).
  By `Jimmy Westling <https://github.com/illviljan>`_.

Bug fixes
~~~~~~~~~

- Ensure no forward slashes in variable and dimension names for HDF5-based engines.
  (:issue:`7943`, :pull:`7953`) By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Documentation
~~~~~~~~~~~~~

- Added examples to docstrings of :py:meth:`Dataset.assign_attrs`, :py:meth:`Dataset.broadcast_equals`,
  :py:meth:`Dataset.equals`, :py:meth:`Dataset.identical`, :py:meth:`Dataset.expand_dims`,:py:meth:`Dataset.drop_vars`
  (:issue:`6793`, :pull:`7937`) By `Harshitha <https://github.com/harshitha1201>`_.
- Added page on wrapping chunked numpy-like arrays as alternatives to dask arrays.
  (:pull:`7951`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Expanded the page on wrapping numpy-like "duck" arrays.
  (:pull:`7911`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added examples to docstrings of :py:meth:`Dataset.isel`, :py:meth:`Dataset.reduce`, :py:meth:`Dataset.argmin`,
  :py:meth:`Dataset.argmax` (:issue:`6793`, :pull:`7881`)
  By `Harshitha <https://github.com/harshitha1201>`_ .

Internal Changes
~~~~~~~~~~~~~~~~

- Allow chunked non-dask arrays (i.e. Cubed arrays) in groupby operations. (:pull:`7941`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.


.. _whats-new.2023.06.0:

v2023.06.0 (June 21, 2023)
--------------------------

This release adds features to ``curvefit``, improves the performance of concatenation, and fixes various bugs.

Thank to our 13 contributors to this release:
Anderson Banihirwe, Deepak Cherian, dependabot[bot], Illviljan, Juniper Tyree, Justus Magin, Martin Fleischmann,
Mattia Almansi, mgunyho, Rutger van Haasteren, Thomas Nicholas, Tom Nicholas, Tom White.


New Features
~~~~~~~~~~~~

- Added support for multidimensional initial guess and bounds in :py:meth:`DataArray.curvefit` (:issue:`7768`, :pull:`7821`).
  By `András Gunyhó <https://github.com/mgunyho>`_.
- Add an ``errors`` option to :py:meth:`Dataset.curve_fit` that allows
  returning NaN for the parameters and covariances of failed fits, rather than
  failing the whole series of fits (:issue:`6317`, :pull:`7891`).
  By `Dominik Stańczak <https://github.com/StanczakDominik>`_ and `András Gunyhó <https://github.com/mgunyho>`_.

Breaking changes
~~~~~~~~~~~~~~~~


Deprecations
~~~~~~~~~~~~
- Deprecate the `cdms2 <https://github.com/CDAT/cdms>`_ conversion methods (:pull:`7876`)
  By `Justus Magin <https://github.com/keewis>`_.

Performance
~~~~~~~~~~~
- Improve concatenation performance (:issue:`7833`, :pull:`7824`).
  By `Jimmy Westling <https://github.com/illviljan>`_.

Bug fixes
~~~~~~~~~
- Fix bug where weighted ``polyfit`` were changing the original object (:issue:`5644`, :pull:`7900`).
  By `Mattia Almansi <https://github.com/malmans2>`_.
- Don't call ``CachingFileManager.__del__`` on interpreter shutdown (:issue:`7814`, :pull:`7880`).
  By `Justus Magin <https://github.com/keewis>`_.
- Preserve vlen dtype for empty string arrays (:issue:`7328`, :pull:`7862`).
  By `Tom White <https://github.com/tomwhite>`_ and `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Ensure dtype of reindex result matches dtype of the original DataArray (:issue:`7299`, :pull:`7917`)
  By `Anderson Banihirwe <https://github.com/andersy005>`_.
- Fix bug where a zero-length zarr ``chunk_store`` was ignored as if it was ``None`` (:pull:`7923`)
  By `Juniper Tyree <https://github.com/juntyr>`_.

Documentation
~~~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

- Minor improvements to support of the python `array api standard <https://data-apis.org/array-api/latest/>`_,
  internally using the function ``xp.astype()`` instead of the method ``arr.astype()``, as the latter is not in the standard.
  (:pull:`7847`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Xarray now uploads nightly wheels to https://pypi.anaconda.org/scientific-python-nightly-wheels/simple/ (:issue:`7863`, :pull:`7865`).
  By `Martin Fleischmann <https://github.com/martinfleis>`_.
- Stop uploading development wheels to TestPyPI (:pull:`7889`)
  By `Justus Magin <https://github.com/keewis>`_.
- Added an exception catch for ``AttributeError`` along with ``ImportError`` when duck typing the dynamic imports in pycompat.py. This catches some name collisions between packages. (:issue:`7870`, :pull:`7874`)

.. _whats-new.2023.05.0:

v2023.05.0 (May 18, 2023)
-------------------------

This release adds some new methods and operators, updates our deprecation policy for python versions, fixes some bugs with groupby,
and introduces experimental support for alternative chunked parallel array computation backends via a new plugin system!

**Note:** If you are using a locally-installed development version of xarray then pulling the changes from this release may require you to re-install.
This avoids an error where xarray cannot detect dask via the new entrypoints system introduced in :pull:`7019`. See :issue:`7856` for details.

Thanks to our 14 contributors:
Alan Brammer, crusaderky, David Stansby, dcherian, Deeksha, Deepak Cherian, Illviljan, James McCreight,
Joe Hamman, Justus Magin, Kyle Sunden, Max Hollmann, mgunyho, and Tom Nicholas


New Features
~~~~~~~~~~~~
- Added new method :py:meth:`DataArray.to_dask_dataframe`, convert a dataarray into a dask dataframe (:issue:`7409`).
  By `Deeksha <https://github.com/dsgreen2>`_.
- Add support for lshift and rshift binary operators (``<<``, ``>>``) on
  :py:class:`xr.DataArray` of type :py:class:`int` (:issue:`7727` , :pull:`7741`).
  By `Alan Brammer <https://github.com/abrammer>`_.
- Keyword argument ``data='array'`` to both :py:meth:`xarray.Dataset.to_dict` and
  :py:meth:`xarray.DataArray.to_dict` will now return data as the underlying array type.
  Python lists are returned for ``data='list'`` or ``data=True``. Supplying ``data=False`` only returns the schema without data.
  ``encoding=True`` returns the encoding dictionary for the underlying variable also. (:issue:`1599`, :pull:`7739`) .
  By `James McCreight <https://github.com/jmccreight>`_.

Breaking changes
~~~~~~~~~~~~~~~~
- adjust the deprecation policy for python to once again align with NEP-29 (:issue:`7765`, :pull:`7793`)
  By `Justus Magin <https://github.com/keewis>`_.

Performance
~~~~~~~~~~~
- Optimize ``.dt `` accessor performance with ``CFTimeIndex``. (:pull:`7796`)
  By `Deepak Cherian <https://github.com/dcherian>`_.

Bug fixes
~~~~~~~~~
- Fix ``as_compatible_data`` for masked float arrays, now always creates a copy when mask is present (:issue:`2377`, :pull:`7788`).
  By `Max Hollmann <https://github.com/maxhollmann>`_.
- Fix groupby binary ops when grouped array is subset relative to other. (:issue:`7797`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix groupby sum, prod for all-NaN groups with ``flox``. (:issue:`7808`).
  By `Deepak Cherian <https://github.com/dcherian>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Experimental support for wrapping chunked array libraries other than dask.
  A new ABC is defined - :py:class:`xr.core.parallelcompat.ChunkManagerEntrypoint` - which can be subclassed and then
  registered by alternative chunked array implementations. (:issue:`6807`, :pull:`7019`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.


.. _whats-new.2023.04.2:

v2023.04.2 (April 20, 2023)
---------------------------

This is a patch release to fix a bug with binning (:issue:`7766`)

Bug fixes
~~~~~~~~~

- Fix binning when ``labels`` is specified. (:issue:`7766`).
  By `Deepak Cherian <https://github.com/dcherian>`_.


Documentation
~~~~~~~~~~~~~
- Added examples to docstrings for :py:meth:`xarray.core.accessor_str.StringAccessor` methods.
  (:pull:`7669`) .
  By `Mary Gathoni <https://github.com/remigathoni>`_.


.. _whats-new.2023.04.1:

v2023.04.1 (April 18, 2023)
---------------------------

This is a patch release to fix a bug with binning (:issue:`7759`)

Bug fixes
~~~~~~~~~

- Fix binning by unsorted arrays. (:issue:`7759`)


.. _whats-new.2023.04.0:

v2023.04.0 (April 14, 2023)
---------------------------

This release includes support for pandas v2, allows refreshing of backend engines in a session, and removes deprecated backends
for ``rasterio`` and ``cfgrib``.

Thanks to our 19 contributors:
Chinemere, Tom Coleman, Deepak Cherian, Harshitha, Illviljan, Jessica Scheick, Joe Hamman, Justus Magin, Kai Mühlbauer, Kwonil-Kim, Mary Gathoni, Michael Niklas, Pierre, Scott Henderson, Shreyal Gupta, Spencer Clark,  mccloskey, nishtha981, veenstrajelmer

We welcome the following new contributors to Xarray!:
Mary Gathoni, Harshitha, veenstrajelmer, Chinemere, nishtha981, Shreyal Gupta, Kwonil-Kim, mccloskey.

New Features
~~~~~~~~~~~~
- New methods to reset an objects encoding (:py:meth:`Dataset.reset_encoding`, :py:meth:`DataArray.reset_encoding`).
  (:issue:`7686`, :pull:`7689`).
  By `Joe Hamman <https://github.com/jhamman>`_.
- Allow refreshing backend engines with :py:meth:`xarray.backends.refresh_engines` (:issue:`7478`, :pull:`7523`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Added ability to save ``DataArray`` objects directly to Zarr using :py:meth:`~xarray.DataArray.to_zarr`.
  (:issue:`7692`, :pull:`7693`) .
  By `Joe Hamman <https://github.com/jhamman>`_.

Breaking changes
~~~~~~~~~~~~~~~~
- Remove deprecated rasterio backend in favor of rioxarray  (:pull:`7392`).
  By `Scott Henderson <https://github.com/scottyhq>`_.

Deprecations
~~~~~~~~~~~~

Performance
~~~~~~~~~~~
- Optimize alignment with ``join="exact", copy=False`` by avoiding copies. (:pull:`7736`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Avoid unnecessary copies of ``CFTimeIndex``. (:pull:`7735`)
  By `Deepak Cherian <https://github.com/dcherian>`_.

Bug fixes
~~~~~~~~~

- Fix :py:meth:`xr.polyval` with non-system standard integer coeffs (:pull:`7619`).
  By `Shreyal Gupta <https://github.com/Ravenin7>`_ and `Michael Niklas <https://github.com/headtr1ck>`_.
- Improve error message when trying to open a file which you do not have permission to read (:issue:`6523`, :pull:`7629`).
  By `Thomas Coleman <https://github.com/ColemanTom>`_.
- Proper plotting when passing :py:class:`~matplotlib.colors.BoundaryNorm` type argument in :py:meth:`DataArray.plot`. (:issue:`4061`, :issue:`7014`,:pull:`7553`)
  By `Jelmer Veenstra <https://github.com/veenstrajelmer>`_.
- Ensure the formatting of time encoding reference dates outside the range of
  nanosecond-precision datetimes remains the same under pandas version 2.0.0
  (:issue:`7420`, :pull:`7441`).
  By `Justus Magin <https://github.com/keewis>`_ and
  `Spencer Clark  <https://github.com/spencerkclark>`_.
- Various ``dtype`` related fixes needed to support ``pandas>=2.0`` (:pull:`7724`)
  By `Justus Magin <https://github.com/keewis>`_.
- Preserve boolean dtype within encoding (:issue:`7652`, :pull:`7720`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_

Documentation
~~~~~~~~~~~~~

- Update FAQ page on how do I open format X file as an xarray dataset? (:issue:`1285`, :pull:`7638`) using :py:func:`~xarray.open_dataset`
  By `Harshitha <https://github.com/harshitha1201>`_ , `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Don't assume that arrays read from disk will be Numpy arrays. This is a step toward
  enabling reads from a Zarr store using the `Kvikio <https://docs.rapids.ai/api/kvikio/stable/api.html#zarr>`_
  or `TensorStore <https://google.github.io/tensorstore/>`_ libraries.
  (:pull:`6874`). By `Deepak Cherian <https://github.com/dcherian>`_.

- Remove internal support for reading GRIB files through the ``cfgrib`` backend. ``cfgrib`` now uses the external
  backend interface, so no existing code should break.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Implement CF coding functions in ``VariableCoders`` (:pull:`7719`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_

- Added a config.yml file with messages for the welcome bot when a Github user creates their first ever issue or pull request or has their first PR merged. (:issue:`7685`, :pull:`7685`)
  By `Nishtha P <https://github.com/nishthap981>`_.

- Ensure that only nanosecond-precision :py:class:`pd.Timestamp` objects
  continue to be used internally under pandas version 2.0.0.  This is mainly to
  ease the transition to this latest version of pandas.  It should be relaxed
  when addressing :issue:`7493`.  By `Spencer Clark
  <https://github.com/spencerkclark>`_ (:issue:`7707`, :pull:`7731`).

.. _whats-new.2023.03.0:

v2023.03.0 (March 22, 2023)
---------------------------

This release brings many bug fixes, and some new features. The maximum pandas version is pinned to ``<2`` until we can support the new pandas datetime types.
Thanks to our 19 contributors:
Abel Aoun, Alex Goodman, Deepak Cherian, Illviljan, Jody Klymak, Joe Hamman, Justus Magin, Mary Gathoni, Mathias Hauser, Mattia Almansi, Mick, Oriol Abril-Pla, Patrick Hoefler, Paul Ockenfuß, Pierre, Shreyal Gupta, Spencer Clark, Tom Nicholas, Tom Vo

New Features
~~~~~~~~~~~~

- Fix :py:meth:`xr.cov` and :py:meth:`xr.corr` now support complex valued arrays  (:issue:`7340`, :pull:`7392`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Allow indexing along unindexed dimensions with dask arrays
  (:issue:`2511`, :issue:`4276`, :issue:`4663`, :pull:`5873`).
  By `Abel Aoun <https://github.com/bzah>`_ and `Deepak Cherian <https://github.com/dcherian>`_.
- Support dask arrays in ``first`` and ``last`` reductions.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Improved performance in ``open_dataset`` for datasets with large object arrays (:issue:`7484`, :pull:`7494`).
  By `Alex Goodman <https://github.com/agoodm>`_ and `Deepak Cherian <https://github.com/dcherian>`_.

Breaking changes
~~~~~~~~~~~~~~~~


Deprecations
~~~~~~~~~~~~
- Following pandas, the ``base`` and ``loffset`` parameters of
  :py:meth:`xr.DataArray.resample` and :py:meth:`xr.Dataset.resample` have been
  deprecated and will be removed in a future version of xarray.  Using the
  ``origin`` or ``offset`` parameters is recommended as a replacement for using
  the ``base`` parameter and using time offset arithmetic is recommended as a
  replacement for using the ``loffset`` parameter (:pull:`8459`).  By `Spencer
  Clark <https://github.com/spencerkclark>`_.


Bug fixes
~~~~~~~~~

- Improve error message when using in :py:meth:`Dataset.drop_vars` to state which variables can't be dropped. (:pull:`7518`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Require to explicitly defining optional dimensions such as hue
  and markersize for scatter plots. (:issue:`7314`, :pull:`7277`).
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Fix matplotlib raising a UserWarning when plotting a scatter plot
  with an unfilled marker (:issue:`7313`, :pull:`7318`).
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Fix issue with ``max_gap`` in ``interpolate_na``, when applied to
  multidimensional arrays. (:issue:`7597`, :pull:`7598`).
  By `Paul Ockenfuß <https://github.com/Ockenfuss>`_.
- Fix :py:meth:`DataArray.plot.pcolormesh` which now works if one of the coordinates has str dtype  (:issue:`6775`, :pull:`7612`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.

Documentation
~~~~~~~~~~~~~

- Clarify language in contributor's guide (:issue:`7495`, :pull:`7595`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Pin pandas to ``<2``. By `Deepak Cherian <https://github.com/dcherian>`_.

.. _whats-new.2023.02.0:

v2023.02.0 (Feb 7, 2023)
------------------------

This release brings a major upgrade to :py:func:`xarray.concat`, many bug fixes,
and a bump in supported dependency versions. Thanks to our 11 contributors:
Aron Gergely, Deepak Cherian, Illviljan, James Bourbeau, Joe Hamman,
Justus Magin, Hauke Schulz, Kai Mühlbauer, Ken Mankoff, Spencer Clark, Tom Nicholas.

Breaking changes
~~~~~~~~~~~~~~~~

- Support for ``python 3.8`` has been dropped and the minimum versions of some
  dependencies were changed (:pull:`7461`):

  ===================== =========  ========
   Package                    Old      New
  ===================== =========  ========
   python                     3.8      3.9
   numpy                     1.20     1.21
   pandas                     1.3      1.4
   dask                   2021.11   2022.1
   distributed            2021.11   2022.1
   h5netcdf                  0.11     0.13
   lxml                       4.6      4.7
   numba                      5.4      5.5
  ===================== =========  ========

Deprecations
~~~~~~~~~~~~
- Following pandas, the ``closed`` parameters of :py:func:`cftime_range` and
  :py:func:`date_range` are deprecated in favor of the ``inclusive`` parameters,
  and will be removed in a future version of xarray (:issue:`6985`:,
  :pull:`7373`).  By `Spencer Clark <https://github.com/spencerkclark>`_.

Bug fixes
~~~~~~~~~
- :py:func:`xarray.concat` can now concatenate variables present in some datasets but
  not others (:issue:`508`, :pull:`7400`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ and `Scott Chamberlin <https://github.com/scottcha>`_.
- Handle ``keep_attrs`` option in binary operators of :py:meth:`Dataset` (:issue:`7390`, :pull:`7391`).
  By `Aron Gergely <https://github.com/arongergely>`_.
- Improve error message when using dask in :py:func:`apply_ufunc` with ``output_sizes`` not supplied. (:pull:`7509`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- :py:func:`xarray.Dataset.to_zarr` now drops variable encodings that have been added by xarray during reading
  a dataset. (:issue:`7129`, :pull:`7500`).
  By `Hauke Schulz <https://github.com/observingClouds>`_.

Documentation
~~~~~~~~~~~~~
- Mention the `flox package <https://flox.readthedocs.io>`_ in GroupBy documentation and docstrings.
  By `Deepak Cherian <https://github.com/dcherian>`_.


.. _whats-new.2023.01.0:

v2023.01.0 (Jan 17, 2023)
-------------------------

This release includes a number of bug fixes. Thanks to the 14 contributors to this release:
Aron Gergely, Benoit Bovy, Deepak Cherian, Ian Carroll, Illviljan, Joe Hamman, Justus Magin, Mark Harfouche,
Matthew Roeschke, Paige Martin, Pierre, Sam Levang, Tom White,  stefank0.

Breaking changes
~~~~~~~~~~~~~~~~

- :py:meth:`CFTimeIndex.get_loc` has removed the ``method`` and ``tolerance`` keyword arguments.
  Use ``.get_indexer([key], method=..., tolerance=...)`` instead (:pull:`7361`).
  By `Matthew Roeschke <https://github.com/mroeschke>`_.

Bug fixes
~~~~~~~~~

- Avoid in-memory broadcasting when converting to a dask dataframe
  using ``.to_dask_dataframe.`` (:issue:`6811`, :pull:`7472`).
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Accessing the property ``.nbytes`` of a DataArray, or Variable no longer
  accidentally triggers loading the variable into memory.
- Allow numpy-only objects in :py:func:`where` when ``keep_attrs=True`` (:issue:`7362`, :pull:`7364`).
  By `Sam Levang <https://github.com/slevang>`_.
- add a ``keep_attrs`` parameter to :py:meth:`Dataset.pad`, :py:meth:`DataArray.pad`,
  and :py:meth:`Variable.pad` (:pull:`7267`).
  By `Justus Magin <https://github.com/keewis>`_.
- Fixed performance regression in alignment between indexed and non-indexed objects
  of the same shape (:pull:`7382`).
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Preserve original dtype on accessing MultiIndex levels (:issue:`7250`,
  :pull:`7393`). By `Ian Carroll <https://github.com/itcarroll>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Add the pre-commit hook ``absolufy-imports`` to convert relative xarray imports to
  absolute imports (:pull:`7204`, :pull:`7370`).
  By `Jimmy Westling <https://github.com/illviljan>`_.

.. _whats-new.2022.12.0:

v2022.12.0 (2022 Dec 2)
-----------------------

This release includes a number of bug fixes and experimental support for Zarr V3.
Thanks to the 16 contributors to this release:
Deepak Cherian, Francesco Zanetta, Gregory Lee, Illviljan, Joe Hamman, Justus Magin, Luke Conibear, Mark Harfouche, Mathias Hauser,
Mick, Mike Taves, Sam Levang, Spencer Clark, Tom Nicholas, Wei Ji, templiert

New Features
~~~~~~~~~~~~
- Enable using ``offset`` and ``origin`` arguments in :py:meth:`DataArray.resample`
  and :py:meth:`Dataset.resample` (:issue:`7266`, :pull:`7284`).  By `Spencer
  Clark <https://github.com/spencerkclark>`_.
- Add experimental support for Zarr's in-progress V3 specification. (:pull:`6475`).
  By `Gregory Lee  <https://github.com/grlee77>`_ and `Joe Hamman <https://github.com/jhamman>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- The minimum versions of some dependencies were changed (:pull:`7300`):

  ========================== =========  ========
   Package                         Old      New
  ========================== =========  ========
   boto                           1.18     1.20
   cartopy                        0.19     0.20
   distributed                 2021.09  2021.11
   dask                        2021.09  2021.11
   h5py                            3.1      3.6
   hdf5                           1.10     1.12
   matplotlib-base                 3.4      3.5
   nc-time-axis                    1.3      1.4
   netcdf4                       1.5.3    1.5.7
   packaging                      20.3     21.3
   pint                           0.17     0.18
   pseudonetcdf                    3.1      3.2
   typing_extensions              3.10      4.0
  ========================== =========  ========

Deprecations
~~~~~~~~~~~~
- The PyNIO backend has been deprecated (:issue:`4491`, :pull:`7301`).
  By `Joe Hamman <https://github.com/jhamman>`_.

Bug fixes
~~~~~~~~~
- Fix handling of coordinate attributes in :py:func:`where`. (:issue:`7220`, :pull:`7229`)
  By `Sam Levang <https://github.com/slevang>`_.
- Import ``nc_time_axis`` when needed (:issue:`7275`, :pull:`7276`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Fix static typing of :py:meth:`xr.polyval` (:issue:`7312`, :pull:`7315`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Fix multiple reads on fsspec S3 files by resetting file pointer to 0 when reading file streams (:issue:`6813`, :pull:`7304`).
  By `David Hoese <https://github.com/djhoese>`_ and `Wei Ji Leong <https://github.com/weiji14>`_.
- Fix :py:meth:`Dataset.assign_coords` resetting all dimension coordinates to default (pandas) index (:issue:`7346`, :pull:`7347`).
  By `Benoît Bovy <https://github.com/benbovy>`_.

Documentation
~~~~~~~~~~~~~

- Add example of reading and writing individual groups to a single netCDF file to I/O docs page. (:pull:`7338`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~


.. _whats-new.2022.11.0:

v2022.11.0 (Nov 4, 2022)
------------------------

This release brings a number of bugfixes and documentation improvements. Both text and HTML
reprs now have a new "Indexes" section, which we expect will help with development of new
Index objects. This release also features more support for the Python Array API.

Many thanks to the 16 contributors to this release: Daniel Goman, Deepak Cherian, Illviljan, Jessica Scheick, Justus Magin, Mark Harfouche, Maximilian Roos, Mick, Patrick Naylor, Pierre, Spencer Clark, Stephan Hoyer, Tom Nicholas, Tom White

New Features
~~~~~~~~~~~~

- Add static typing to plot accessors (:issue:`6949`, :pull:`7052`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Display the indexes in a new section of the text and HTML reprs
  (:pull:`6795`, :pull:`7183`, :pull:`7185`)
  By `Justus Magin <https://github.com/keewis>`_ and `Benoît Bovy <https://github.com/benbovy>`_.
- Added methods :py:meth:`DataArrayGroupBy.cumprod` and :py:meth:`DatasetGroupBy.cumprod`.
  (:pull:`5816`)
  By `Patrick Naylor <https://github.com/patrick-naylor>`_

Breaking changes
~~~~~~~~~~~~~~~~

- ``repr(ds)`` may not show the same result because it doesn't load small,
  lazy data anymore. Use ``ds.head().load()`` when wanting to see just a sample
  of the data. (:issue:`6722`, :pull:`7203`).
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Many arguments of plotmethods have been made keyword-only.
- ``xarray.plot.plot`` module renamed to ``xarray.plot.dataarray_plot`` to prevent
  shadowing of the ``plot`` method. (:issue:`6949`, :pull:`7052`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.

Deprecations
~~~~~~~~~~~~

- Positional arguments for all plot methods have been deprecated (:issue:`6949`, :pull:`7052`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- ``xarray.plot.FacetGrid.axes`` has been renamed to ``xarray.plot.FacetGrid.axs``
  because it's not clear if ``axes`` refers to single or multiple ``Axes`` instances.
  This aligns with ``matplotlib.pyplot.subplots``. (:pull:`7194`)
  By `Jimmy Westling <https://github.com/illviljan>`_.

Bug fixes
~~~~~~~~~

- Explicitly opening a file multiple times (e.g., after modifying it on disk)
  now reopens the file from scratch for h5netcdf and scipy netCDF backends,
  rather than reusing a cached version (:issue:`4240`, :issue:`4862`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Fixed bug where :py:meth:`Dataset.coarsen.construct` would demote non-dimension coordinates to variables. (:pull:`7233`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Raise a TypeError when trying to plot empty data (:issue:`7156`, :pull:`7228`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.

Documentation
~~~~~~~~~~~~~

- Improves overall documentation around available backends, including adding docstrings for :py:func:`xarray.backends.list_engines`
  Add :py:meth:`__str__` to surface the new :py:class:`BackendEntrypoint` ``description``
  and ``url`` attributes. (:issue:`6577`, :pull:`7000`)
  By `Jessica Scheick <https://github.com/jessicas11>`_.
- Created docstring examples for :py:meth:`DataArray.cumsum`, :py:meth:`DataArray.cumprod`, :py:meth:`Dataset.cumsum`, :py:meth:`Dataset.cumprod`, :py:meth:`DatasetGroupBy.cumsum`, :py:meth:`DataArrayGroupBy.cumsum`. (:issue:`5816`, :pull:`7152`)
  By `Patrick Naylor <https://github.com/patrick-naylor>`_
- Add example of using :py:meth:`DataArray.coarsen.construct` to User Guide. (:pull:`7192`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Rename ``axes`` to ``axs`` in plotting to align with ``matplotlib.pyplot.subplots``. (:pull:`7194`)
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Add documentation of specific BackendEntrypoints (:pull:`7200`).
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Add examples to docstring for :py:meth:`DataArray.drop_vars`, :py:meth:`DataArray.reindex_like`, :py:meth:`DataArray.interp_like`. (:issue:`6793`, :pull:`7123`)
  By `Daniel Goman <https://github.com/DanielGoman>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Doctests fail on any warnings (:pull:`7166`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Improve import time by lazy loading ``dask.distributed`` (:pull:`7172`).
- Explicitly specify ``longdouble=False`` in :py:func:`cftime.date2num` when
  encoding times to preserve existing behavior and prevent future errors when it
  is eventually set to ``True`` by default in cftime (:pull:`7171`).  By
  `Spencer Clark <https://github.com/spencerkclark>`_.
- Improved import time by lazily importing backend modules, matplotlib, dask.array and flox. (:issue:`6726`, :pull:`7179`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Emit a warning under the development version of pandas when we convert
  non-nanosecond precision datetime or timedelta values to nanosecond precision.
  This was required in the past, because pandas previously was not compatible
  with non-nanosecond precision values.  However pandas is currently working
  towards removing this restriction.  When things stabilize in pandas we will
  likely consider relaxing this behavior in xarray as well (:issue:`7175`,
  :pull:`7201`).  By `Spencer Clark <https://github.com/spencerkclark>`_.

.. _whats-new.2022.10.0:

v2022.10.0 (Oct 14 2022)
------------------------

This release brings numerous bugfixes, a change in minimum supported versions,
and a new scatter plot method for DataArrays.

Many thanks to 11 contributors to this release: Anderson Banihirwe, Benoit Bovy,
Dan Adriaansen, Illviljan, Justus Magin, Lukas Bindreiter, Mick, Patrick Naylor,
Spencer Clark, Thomas Nicholas


New Features
~~~~~~~~~~~~

- Add scatter plot for datarrays. Scatter plots now also supports 3d plots with
  the z argument. (:pull:`6778`)
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Include the variable name in the error message when CF decoding fails to allow
  for easier identification of problematic variables (:issue:`7145`, :pull:`7147`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- The minimum versions of some dependencies were changed:

  ========================== =========  ========
   Package                         Old      New
  ========================== =========  ========
   cftime                          1.4      1.5
   distributed                 2021.08  2021.09
   dask                        2021.08  2021.09
   iris                            2.4      3.1
   nc-time-axis                    1.2      1.3
   numba                          0.53     0.54
   numpy                          1.19     1.20
   pandas                          1.2      1.3
   packaging                      20.0     21.0
   scipy                           1.6      1.7
   sparse                         0.12     0.13
   typing_extensions               3.7     3.10
   zarr                            2.8     2.10
  ========================== =========  ========


Bug fixes
~~~~~~~~~

- Remove nested function from :py:func:`open_mfdataset` to allow Dataset objects to be pickled. (:issue:`7109`, :pull:`7116`)
  By `Daniel Adriaansen <https://github.com/DanielAdriaansen>`_.
- Support for recursively defined Arrays. Fixes repr and deepcopy. (:issue:`7111`, :pull:`7112`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Fixed :py:meth:`Dataset.transpose` to raise a more informative error. (:issue:`6502`, :pull:`7120`)
  By `Patrick Naylor <https://github.com/patrick-naylor>`_
- Fix groupby on a multi-index level coordinate and fix
  :py:meth:`DataArray.to_index` for multi-index levels (convert to single index).
  (:issue:`6836`, :pull:`7105`)
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Support for open_dataset backends that return datasets containing multi-indexes (:issue:`7139`, :pull:`7150`)
  By `Lukas Bindreiter <https://github.com/lukasbindreiter>`_.


.. _whats-new.2022.09.0:

v2022.09.0 (September 30, 2022)
-------------------------------

This release brings a large number of bugfixes and documentation improvements, as well as an external interface for
setting custom indexes!

Many thanks to our 40 contributors:

Anderson Banihirwe, Andrew Ronald Friedman, Bane Sullivan, Benoit Bovy, ColemanTom, Deepak Cherian,
Dimitri Papadopoulos Orfanos, Emma Marshall, Fabian Hofmann, Francesco Nattino, ghislainp, Graham Inggs, Hauke Schulz,
Illviljan, James Bourbeau, Jody Klymak, Julia Signell, Justus Magin, Keewis, Ken Mankoff, Luke Conibear, Mathias Hauser,
Max Jones, mgunyho, Michael Delgado, Mick, Mike Taves, Oliver Lopez, Patrick Naylor, Paul Hockett, Pierre Manchon,
Ray Bell, Riley Brady, Sam Levang, Spencer Clark, Stefaan Lippens, Tom Nicholas, Tom White, Travis A. O'Brien,
and Zachary Moon.

New Features
~~~~~~~~~~~~

- Add :py:meth:`Dataset.set_xindex` and :py:meth:`Dataset.drop_indexes` and
  their DataArray counterpart for setting and dropping pandas or custom indexes
  given a set of arbitrary coordinates. (:pull:`6971`)
  By `Benoît Bovy <https://github.com/benbovy>`_ and `Justus Magin <https://github.com/keewis>`_.
- Enable taking the mean of dask-backed :py:class:`cftime.datetime` arrays
  (:pull:`6556`, :pull:`6940`).
  By `Deepak Cherian <https://github.com/dcherian>`_ and `Spencer Clark <https://github.com/spencerkclark>`_.

Bug fixes
~~~~~~~~~

- Allow reading netcdf files where the 'units' attribute is a number. (:pull:`7085`)
  By `Ghislain Picard <https://github.com/ghislainp>`_.
- Allow decoding of 0 sized datetimes. (:issue:`1329`, :pull:`6882`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Make sure DataArray.name is always a string when used as label for plotting. (:issue:`6826`, :pull:`6832`)
  By `Jimmy Westling <https://github.com/illviljan>`_.
- :py:attr:`DataArray.nbytes` now uses the ``nbytes`` property of the underlying array if available. (:pull:`6797`)
  By `Max Jones <https://github.com/maxrjones>`_.
- Rely on the array backend for string formatting. (:pull:`6823`).
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Fix incompatibility with numpy 1.20. (:issue:`6818`, :pull:`6821`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Fix side effects on index coordinate metadata after aligning objects. (:issue:`6852`, :pull:`6857`)
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Make FacetGrid.set_titles send kwargs correctly using ``handle.update(kwargs)``. (:issue:`6839`, :pull:`6843`)
  By `Oliver Lopez <https://github.com/lopezvoliver>`_.
- Fix bug where index variables would be changed inplace. (:issue:`6931`, :pull:`6938`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Allow taking the mean over non-time dimensions of datasets containing
  dask-backed cftime arrays. (:issue:`5897`, :pull:`6950`)
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Harmonize returned multi-indexed indexes when applying ``concat`` along new dimension. (:issue:`6881`, :pull:`6889`)
  By `Fabian Hofmann <https://github.com/FabianHofmann>`_.
- Fix step plots with ``hue`` arg. (:pull:`6944`)
  By `András Gunyhó <https://github.com/mgunyho>`_.
- Avoid use of random numbers in ``test_weighted.test_weighted_operations_nonequal_coords``. (:issue:`6504`, :pull:`6961`)
  By `Luke Conibear <https://github.com/lukeconibear>`_.
- Fix multiple regression issues with :py:meth:`Dataset.set_index` and
  :py:meth:`Dataset.reset_index`. (:pull:`6992`)
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Raise a ``UserWarning`` when renaming a coordinate or a dimension creates a
  non-indexed dimension coordinate, and suggest the user creating an index
  either with ``swap_dims`` or ``set_index``. (:issue:`6607`, :pull:`6999`)
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Use ``keep_attrs=True`` in grouping and resampling operations by default. (:issue:`7012`)
  This means :py:attr:`Dataset.attrs` and :py:attr:`DataArray.attrs` are now preserved by default.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- ``Dataset.encoding['source']`` now exists when reading from a Path object. (:issue:`5888`, :pull:`6974`)
  By `Thomas Coleman <https://github.com/ColemanTom>`_.
- Better dtype consistency for ``rolling.mean()``. (:issue:`7062`, :pull:`7063`)
  By `Sam Levang <https://github.com/slevang>`_.
- Allow writing NetCDF files including only dimensionless variables using the distributed or multiprocessing scheduler. (:issue:`7013`, :pull:`7040`)
  By `Francesco Nattino <https://github.com/fnattino>`_.
- Fix deepcopy of attrs and encoding of DataArrays and Variables. (:issue:`2835`, :pull:`7089`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Fix bug where subplot_kwargs were not working when plotting with figsize, size or aspect. (:issue:`7078`, :pull:`7080`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.

Documentation
~~~~~~~~~~~~~

- Update merge docstrings. (:issue:`6935`, :pull:`7033`)
  By `Zach Moon <https://github.com/zmoon>`_.
- Raise a more informative error when trying to open a non-existent zarr store. (:issue:`6484`, :pull:`7060`)
  By `Sam Levang <https://github.com/slevang>`_.
- Added examples to docstrings for :py:meth:`DataArray.expand_dims`, :py:meth:`DataArray.drop_duplicates`, :py:meth:`DataArray.reset_coords`, :py:meth:`DataArray.equals`, :py:meth:`DataArray.identical`, :py:meth:`DataArray.broadcast_equals`, :py:meth:`DataArray.bfill`, :py:meth:`DataArray.ffill`, :py:meth:`DataArray.fillna`, :py:meth:`DataArray.dropna`, :py:meth:`DataArray.drop_isel`, :py:meth:`DataArray.drop_sel`, :py:meth:`DataArray.head`, :py:meth:`DataArray.tail`. (:issue:`5816`, :pull:`7088`)
  By `Patrick Naylor <https://github.com/patrick-naylor>`_.
- Add missing docstrings to various array properties. (:pull:`7090`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Added test for DataArray attrs deepcopy recursion/nested attrs. (:issue:`2835`, :pull:`7086`)
  By `Paul hockett <https://github.com/phockett>`_.

.. _whats-new.2022.06.0:

v2022.06.0 (July 21, 2022)
--------------------------

This release brings a number of bug fixes and improvements, most notably a major internal
refactor of the indexing functionality, the use of `flox`_ in ``groupby`` operations,
and experimental support for the new Python `Array API standard <https://data-apis.org/array-api/latest/>`_.
It also stops testing support for the abandoned PyNIO.

Much effort has been made to preserve backwards compatibility as part of the indexing refactor.
We are aware of one `unfixed issue <https://github.com/pydata/xarray/issues/6607>`_.

Please also see the `whats-new.2022.06.0rc0`_ for a full list of changes.

Many thanks to our 18 contributors:
Bane Sullivan, Deepak Cherian, Dimitri Papadopoulos Orfanos, Emma Marshall, Hauke Schulz, Illviljan,
Julia Signell, Justus Magin, Keewis, Mathias Hauser, Michael Delgado, Mick, Pierre Manchon, Ray Bell,
Spencer Clark, Stefaan Lippens, Tom White, Travis A. O'Brien,

New Features
~~~~~~~~~~~~

- Add :py:attr:`Dataset.dtypes`, :py:attr:`core.coordinates.DatasetCoordinates.dtypes`,
  :py:attr:`core.coordinates.DataArrayCoordinates.dtypes` properties: Mapping from variable names to dtypes.
  (:pull:`6706`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Initial typing support for :py:meth:`groupby`, :py:meth:`rolling`, :py:meth:`rolling_exp`,
  :py:meth:`coarsen`, :py:meth:`weighted`, :py:meth:`resample`,
  (:pull:`6702`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Experimental support for wrapping any array type that conforms to the python
  `array api standard <https://data-apis.org/array-api/latest/>`_. (:pull:`6804`)
  By `Tom White <https://github.com/tomwhite>`_.
- Allow string formatting of scalar DataArrays. (:pull:`5981`)
  By `fmaussion <https://github.com/fmaussion>`_.

Bug fixes
~~~~~~~~~

- :py:meth:`save_mfdataset` now passes ``**kwargs`` on to :py:meth:`Dataset.to_netcdf`,
  allowing the ``encoding`` and ``unlimited_dims`` options with :py:meth:`save_mfdataset`.
  (:issue:`6684`)
  By `Travis A. O'Brien <https://github.com/taobrienlbl>`_.
- Fix backend support of pydap versions <3.3.0  (:issue:`6648`, :pull:`6656`).
  By `Hauke Schulz <https://github.com/observingClouds>`_.
- :py:meth:`Dataset.where` with ``drop=True`` now behaves correctly with mixed dimensions.
  (:issue:`6227`, :pull:`6690`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Accommodate newly raised ``OutOfBoundsTimedelta`` error in the development version of
  pandas when decoding times outside the range that can be represented with
  nanosecond-precision values (:issue:`6716`, :pull:`6717`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- :py:meth:`open_dataset` with dask and ``~`` in the path now resolves the home directory
  instead of raising an error. (:issue:`6707`, :pull:`6710`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- :py:meth:`DataArrayRolling.__iter__` with ``center=True`` now works correctly.
  (:issue:`6739`, :pull:`6744`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- ``xarray.core.groupby``, ``xarray.core.rolling``,
  ``xarray.core.rolling_exp``, ``xarray.core.weighted``
  and ``xarray.core.resample`` modules are no longer imported by default.
  (:pull:`6702`)

.. _whats-new.2022.06.0rc0:

v2022.06.0rc0 (9 June 2022)
---------------------------

This pre-release brings a number of bug fixes and improvements, most notably a major internal
refactor of the indexing functionality and the use of `flox`_ in ``groupby`` operations. It also stops
testing support for the abandoned PyNIO.

Install it using

::

    mamba create -n <name> python=3.10 xarray
    python -m pip install --pre --upgrade --no-deps xarray


Many thanks to the 39 contributors:

Abel Soares Siqueira, Alex Santana, Anderson Banihirwe, Benoit Bovy, Blair Bonnett, Brewster
Malevich, brynjarmorka, Charles Stern, Christian Jauvin, Deepak Cherian, Emma Marshall, Fabien
Maussion, Greg Behm, Guelate Seyo, Illviljan, Joe Hamman, Joseph K Aicher, Justus Magin, Kevin Paul,
Louis Stenger, Mathias Hauser, Mattia Almansi, Maximilian Roos, Michael Bauer, Michael Delgado,
Mick, ngam, Oleh Khoma, Oriol Abril-Pla, Philippe Blain, PLSeuJ, Sam Levang, Spencer Clark, Stan
West, Thomas Nicholas, Thomas Vogt, Tom White, Xianxiang Li

Known Regressions
~~~~~~~~~~~~~~~~~

- ``reset_coords(drop=True)`` does not create indexes (:issue:`6607`)

New Features
~~~~~~~~~~~~

- The ``zarr`` backend is now able to read NCZarr.
  By `Mattia Almansi <https://github.com/malmans2>`_.
- Add a weighted ``quantile`` method to :py:class:`~core.weighted.DatasetWeighted` and
  :py:class:`~core.weighted.DataArrayWeighted` (:pull:`6059`).
  By `Christian Jauvin <https://github.com/cjauvin>`_ and `David Huard <https://github.com/huard>`_.
- Add a ``create_index=True`` parameter to :py:meth:`Dataset.stack` and
  :py:meth:`DataArray.stack` so that the creation of multi-indexes is optional
  (:pull:`5692`).
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Multi-index levels are now accessible through their own, regular coordinates
  instead of virtual coordinates (:pull:`5692`).
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Add a ``display_values_threshold`` option to control the total number of array
  elements which trigger summarization rather than full repr in (numpy) array
  detailed views of the html repr (:pull:`6400`).
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Allow passing chunks in ``kwargs`` form to :py:meth:`Dataset.chunk`, :py:meth:`DataArray.chunk`, and
  :py:meth:`Variable.chunk`. (:pull:`6471`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Add :py:meth:`core.groupby.DatasetGroupBy.cumsum` and :py:meth:`core.groupby.DataArrayGroupBy.cumsum`.
  By `Vladislav Skripniuk <https://github.com/VladSkripniuk>`_ and `Deepak Cherian <https://github.com/dcherian>`_. (:pull:`3147`, :pull:`6525`, :issue:`3141`)
- Expose ``inline_array`` kwarg from ``dask.array.from_array`` in :py:func:`open_dataset`, :py:meth:`Dataset.chunk`,
  :py:meth:`DataArray.chunk`, and :py:meth:`Variable.chunk`. (:pull:`6471`)
- Expose the ``inline_array`` kwarg from :py:func:`dask.array.from_array` in :py:func:`open_dataset`,
  :py:meth:`Dataset.chunk`, :py:meth:`DataArray.chunk`, and :py:meth:`Variable.chunk`. (:pull:`6471`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- :py:func:`polyval` now supports :py:class:`Dataset` and :py:class:`DataArray` args of any shape,
  is faster and requires less memory. (:pull:`6548`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Improved overall typing.
- :py:meth:`Dataset.to_dict` and :py:meth:`DataArray.to_dict` may now optionally include encoding
  attributes. (:pull:`6635`)
  By `Joe Hamman <https://github.com/jhamman>`_.
- Upload development versions to `TestPyPI <https://test.pypi.org>`_.
  By `Justus Magin <https://github.com/keewis>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- PyNIO support is now untested. The minimum versions of some dependencies were changed:

  =============== ===== ====
  Package         Old   New
  =============== ===== ====
  cftime          1.2   1.4
  dask            2.30  2021.4
  distributed     2.30  2021.4
  h5netcdf        0.8   0.11
  matplotlib-base 3.3   3.4
  numba           0.51  0.53
  numpy           1.18  1.19
  pandas          1.1   1.2
  pint            0.16  0.17
  rasterio        1.1   1.2
  scipy           1.5   1.6
  sparse          0.11  0.12
  zarr            2.5   2.8
  =============== ===== ====

- The Dataset and DataArray ``rename```` methods do not implicitly add or drop
  indexes. (:pull:`5692`).
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Many arguments like ``keep_attrs``, ``axis``, and ``skipna`` are now keyword
  only for all reduction operations like ``.mean``.
  By `Deepak Cherian <https://github.com/dcherian>`_, `Jimmy Westling <https://github.com/illviljan>`_.
- Xarray's ufuncs have been removed, now that they can be replaced by numpy's ufuncs in all
  supported versions of numpy.
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- :py:meth:`xr.polyval` now uses the ``coord`` argument directly instead of its index coordinate.
  (:pull:`6548`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.

Bug fixes
~~~~~~~~~

- :py:meth:`Dataset.to_zarr` now allows to write all attribute types supported by ``zarr-python``.
  By `Mattia Almansi <https://github.com/malmans2>`_.
- Set ``skipna=None`` for all ``quantile`` methods (e.g. :py:meth:`Dataset.quantile`) and
  ensure it skips missing values for float dtypes (consistent with other methods). This should
  not change the behavior (:pull:`6303`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Many bugs fixed by the explicit indexes refactor, mainly related to multi-index (virtual)
  coordinates. See the corresponding pull-request on GitHub for more details. (:pull:`5692`).
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Fixed "unhashable type" error trying to read NetCDF file with variable having its 'units'
  attribute not ``str`` (e.g. ``numpy.ndarray``) (:issue:`6368`).
  By `Oleh Khoma <https://github.com/okhoma>`_.
- Omit warning about specified dask chunks separating chunks on disk when the
  underlying array is empty (e.g., because of an empty dimension) (:issue:`6401`).
  By `Joseph K Aicher <https://github.com/jaicher>`_.
- Fixed the poor html repr performance on large multi-indexes (:pull:`6400`).
  By `Benoît Bovy <https://github.com/benbovy>`_.
- Allow fancy indexing of duck dask arrays along multiple dimensions. (:pull:`6414`)
  By `Justus Magin <https://github.com/keewis>`_.
- In the API for backends, support dimensions that express their preferred chunk sizes
  as a tuple of integers. (:issue:`6333`, :pull:`6334`)
  By `Stan West <https://github.com/stanwest>`_.
- Fix bug in :py:func:`where` when passing non-xarray objects with ``keep_attrs=True``. (:issue:`6444`, :pull:`6461`)
  By `Sam Levang <https://github.com/slevang>`_.
- Allow passing both ``other`` and ``drop=True`` arguments to :py:meth:`DataArray.where`
  and :py:meth:`Dataset.where` (:pull:`6466`, :pull:`6467`).
  By `Michael Delgado <https://github.com/delgadom>`_.
- Ensure dtype encoding attributes are not added or modified on variables that contain datetime-like
  values prior to being passed to :py:func:`xarray.conventions.decode_cf_variable` (:issue:`6453`,
  :pull:`6489`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Dark themes are now properly detected in Furo-themed Sphinx documents (:issue:`6500`, :pull:`6501`).
  By `Kevin Paul <https://github.com/kmpaul>`_.
- :py:meth:`Dataset.isel`, :py:meth:`DataArray.isel` with ``drop=True`` works as intended with scalar :py:class:`DataArray` indexers.
  (:issue:`6554`, :pull:`6579`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Fixed silent overflow issue when decoding times encoded with 32-bit and below
  unsigned integer data types (:issue:`6589`, :pull:`6598`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Fixed ``.chunks`` loading lazy data (:issue:`6538`).
  By `Deepak Cherian <https://github.com/dcherian>`_.

Documentation
~~~~~~~~~~~~~

- Revise the documentation for developers on specifying a backend's preferred chunk
  sizes. In particular, correct the syntax and replace lists with tuples in the
  examples. (:issue:`6333`, :pull:`6334`)
  By `Stan West <https://github.com/stanwest>`_.
- Mention that :py:meth:`DataArray.rename` can rename coordinates.
  (:issue:`5458`, :pull:`6665`)
  By `Michael Niklas <https://github.com/headtr1ck>`_.
- Added examples to :py:meth:`Dataset.thin` and :py:meth:`DataArray.thin`
  By `Emma Marshall <https://github.com/e-marshall>`_.

Performance
~~~~~~~~~~~

- GroupBy binary operations are now vectorized.
  Previously this involved looping over all groups. (:issue:`5804`, :pull:`6160`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Substantially improved GroupBy operations using `flox <https://flox.readthedocs.io/en/latest/>`_.
  This is auto-enabled when ``flox`` is installed. Use ``xr.set_options(use_flox=False)`` to use
  the old algorithm. (:issue:`4473`, :issue:`4498`, :issue:`659`, :issue:`2237`, :pull:`271`).
  By `Deepak Cherian <https://github.com/dcherian>`_, `Anderson Banihirwe <https://github.com/andersy005>`_, `Jimmy Westling <https://github.com/illviljan>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Many internal changes due to the explicit indexes refactor. See the
  corresponding pull-request on GitHub for more details. (:pull:`5692`).
  By `Benoît Bovy <https://github.com/benbovy>`_.

.. _whats-new.2022.03.0:

v2022.03.0 (2 March 2022)
-------------------------

This release brings a number of small improvements, as well as a move to `calendar versioning <https://calver.org/>`_ (:issue:`6176`).

Many thanks to the 16 contributors to the v2022.02.0 release!

Aaron Spring, Alan D. Snow, Anderson Banihirwe, crusaderky, Illviljan, Joe Hamman, Jonas Gliß,
Lukas Pilz, Martin Bergemann, Mathias Hauser, Maximilian Roos, Romain Caneill, Stan West, Stijn Van Hoey,
Tobias Kölling, and Tom Nicholas.


New Features
~~~~~~~~~~~~

- Enabled multiplying tick offsets by floats. Allows ``float`` ``n`` in
  :py:meth:`CFTimeIndex.shift` if ``shift_freq`` is between ``Day``
  and ``Microsecond``. (:issue:`6134`, :pull:`6135`).
  By `Aaron Spring <https://github.com/aaronspring>`_.
- Enable providing more keyword arguments to the ``pydap`` backend when reading
  OpenDAP datasets (:issue:`6274`).
  By `Jonas Gliß <https://github.com/jgliss>`_.
- Allow :py:meth:`DataArray.drop_duplicates` to drop duplicates along multiple dimensions at once,
  and add :py:meth:`Dataset.drop_duplicates`. (:pull:`6307`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- Renamed the ``interpolation`` keyword of all ``quantile`` methods (e.g. :py:meth:`DataArray.quantile`)
  to ``method`` for consistency with numpy v1.22.0 (:pull:`6108`).
  By `Mathias Hauser <https://github.com/mathause>`_.

Deprecations
~~~~~~~~~~~~


Bug fixes
~~~~~~~~~

- Variables which are chunked using dask in larger (but aligned) chunks than the target zarr chunk size
  can now be stored using ``to_zarr()`` (:pull:`6258`) By `Tobias Kölling <https://github.com/d70-t>`_.
- Multi-file datasets containing encoded :py:class:`cftime.datetime` objects can be read in parallel again (:issue:`6226`, :pull:`6249`, :pull:`6305`).  By `Martin Bergemann <https://github.com/antarcticrainforest>`_ and `Stan West <https://github.com/stanwest>`_.

Documentation
~~~~~~~~~~~~~

- Delete files of datasets saved to disk while building the documentation and enable
  building on Windows via ``sphinx-build`` (:pull:`6237`).
  By `Stan West <https://github.com/stanwest>`_.


Internal Changes
~~~~~~~~~~~~~~~~


.. _whats-new.0.21.1:

v0.21.1 (31 January 2022)
-------------------------

This is a bugfix release to resolve (:issue:`6216`, :pull:`6207`).

Bug fixes
~~~~~~~~~
- Add ``packaging`` as a dependency to Xarray (:issue:`6216`, :pull:`6207`).
  By `Sebastian Weigand <https://github.com/s-weigand>`_ and `Joe Hamman <https://github.com/jhamman>`_.


.. _whats-new.0.21.0:

v0.21.0 (27 January 2022)
-------------------------

Many thanks to the 20 contributors to the v0.21.0 release!

Abel Aoun, Anderson Banihirwe, Ant Gib, Chris Roat, Cindy Chiao,
Deepak Cherian, Dominik Stańczak, Fabian Hofmann, Illviljan, Jody Klymak, Joseph
K Aicher, Mark Harfouche, Mathias Hauser, Matthew Roeschke, Maximilian Roos,
Michael Delgado, Pascal Bourgault, Pierre, Ray Bell, Romain Caneill, Tim Heap,
Tom Nicholas, Zeb Nicholls, joseph nowak, keewis.


New Features
~~~~~~~~~~~~
- New top-level function :py:func:`cross`. (:issue:`3279`, :pull:`5365`).
  By `Jimmy Westling <https://github.com/illviljan>`_.
- ``keep_attrs`` support for :py:func:`where` (:issue:`4141`, :issue:`4682`, :pull:`4687`).
  By `Justus Magin <https://github.com/keewis>`_.
- Enable the limit option for dask array in the following methods :py:meth:`DataArray.ffill`, :py:meth:`DataArray.bfill`, :py:meth:`Dataset.ffill` and :py:meth:`Dataset.bfill` (:issue:`6112`)
  By `Joseph Nowak <https://github.com/josephnowak>`_.


Breaking changes
~~~~~~~~~~~~~~~~
- Rely on matplotlib's default datetime converters instead of pandas' (:issue:`6102`, :pull:`6109`).
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Improve repr readability when there are a large number of dimensions in datasets or dataarrays by
  wrapping the text once the maximum display width has been exceeded. (:issue:`5546`, :pull:`5662`)
  By `Jimmy Westling <https://github.com/illviljan>`_.


Deprecations
~~~~~~~~~~~~
- Removed the lock kwarg from the zarr and pydap backends, completing the deprecation cycle started in :issue:`5256`.
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Support for ``python 3.7`` has been dropped. (:pull:`5892`)
  By `Jimmy Westling <https://github.com/illviljan>`_.


Bug fixes
~~~~~~~~~
- Preserve chunks when creating a :py:class:`DataArray` from another :py:class:`DataArray`
  (:pull:`5984`). By `Fabian Hofmann <https://github.com/FabianHofmann>`_.
- Properly support :py:meth:`DataArray.ffill`, :py:meth:`DataArray.bfill`, :py:meth:`Dataset.ffill` and :py:meth:`Dataset.bfill` along chunked dimensions (:issue:`6112`).
  By `Joseph Nowak <https://github.com/josephnowak>`_.

- Subclasses of ``byte`` and ``str`` (e.g. ``np.str_`` and ``np.bytes_``) will now serialise to disk rather than raising a ``ValueError: unsupported dtype for netCDF4 variable: object`` as they did previously (:pull:`5264`).
  By `Zeb Nicholls <https://github.com/znicholls>`_.

- Fix applying function with non-xarray arguments using :py:func:`xr.map_blocks`.
  By `Cindy Chiao <https://github.com/tcchiao>`_.

- No longer raise an error for an all-nan-but-one argument to
  :py:meth:`DataArray.interpolate_na` when using ``method='nearest'`` (:issue:`5994`, :pull:`6144`).
  By `Michael Delgado <https://github.com/delgadom>`_.
- `dt.season <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.dt.season.html>`_  can now handle NaN and NaT.  (:pull:`5876`).
  By `Pierre Loicq <https://github.com/pierreloicq>`_.
- Determination of zarr chunks handles empty lists for encoding chunks or variable chunks that occurs in certain circumstances (:pull:`5526`). By `Chris Roat <https://github.com/chrisroat>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Replace ``distutils.version`` with ``packaging.version``  (:issue:`6092`).
  By `Mathias Hauser <https://github.com/mathause>`_.

- Removed internal checks for ``pd.Panel`` (:issue:`6145`).
  By `Matthew Roeschke <https://github.com/mroeschke>`_.

- Add ``pyupgrade`` pre-commit hook (:pull:`6152`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.

.. _whats-new.0.20.2:

v0.20.2 (9 December 2021)
-------------------------

This is a bugfix release to resolve (:issue:`3391`, :issue:`5715`). It also
includes performance improvements in unstacking to a ``sparse`` array and a
number of documentation improvements.

Many thanks to the 20 contributors:

Aaron Spring, Alexandre Poux, Deepak Cherian, Enrico Minack, Fabien Maussion,
Giacomo Caria, Gijom, Guillaume Maze, Illviljan, Joe Hamman, Joseph Hardin, Kai
Mühlbauer, Matt Henderson, Maximilian Roos, Michael Delgado, Robert Gieseke,
Sebastian Weigand and Stephan Hoyer.


Breaking changes
~~~~~~~~~~~~~~~~
- Use complex nan when interpolating complex values out of bounds by default (instead of real nan) (:pull:`6019`).
  By `Alexandre Poux <https://github.com/pums974>`_.

Performance
~~~~~~~~~~~

- Significantly faster unstacking to a ``sparse`` array. :pull:`5577`
  By `Deepak Cherian <https://github.com/dcherian>`_.

Bug fixes
~~~~~~~~~
- :py:func:`xr.map_blocks` and :py:func:`xr.corr` now work when dask is not installed (:issue:`3391`, :issue:`5715`, :pull:`5731`).
  By `Gijom <https://github.com/Gijom>`_.
- Fix plot.line crash for data of shape ``(1, N)`` in _title_for_slice on format_item (:pull:`5948`).
  By `Sebastian Weigand <https://github.com/s-weigand>`_.
- Fix a regression in the removal of duplicate backend entrypoints (:issue:`5944`, :pull:`5959`)
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Fix an issue that datasets from being saved when time variables with units that ``cftime`` can parse but pandas can not were present (:pull:`6049`).
  By `Tim Heap <https://github.com/mx-moth>`_.

Documentation
~~~~~~~~~~~~~

- Better examples in docstrings for groupby and resampling reductions (:pull:`5871`).
  By `Deepak Cherian <https://github.com/dcherian>`_,
  `Maximilian Roos <https://github.com/max-sixty>`_,
  `Jimmy Westling <https://github.com/illviljan>`_ .
- Add list-like possibility for tolerance parameter in the reindex functions.
  By `Antoine Gibek <https://github.com/antscloud>`_,

Internal Changes
~~~~~~~~~~~~~~~~

- Use ``importlib`` to replace functionality of ``pkg_resources`` in
  backend plugins tests. (:pull:`5959`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.


.. _whats-new.0.20.1:

v0.20.1 (5 November 2021)
-------------------------

This is a bugfix release to fix :issue:`5930`.

Bug fixes
~~~~~~~~~
- Fix a regression in the detection of the backend entrypoints (:issue:`5930`, :pull:`5931`)
  By `Justus Magin <https://github.com/keewis>`_.

Documentation
~~~~~~~~~~~~~

- Significant improvements to  :ref:`api`. By `Deepak Cherian <https://github.com/dcherian>`_.

.. _whats-new.0.20.0:

v0.20.0 (1 November 2021)
-------------------------

This release brings improved support for pint arrays, methods for weighted standard deviation, variance,
and sum of squares, the option to disable the use of the bottleneck library, significantly improved performance of
unstack, as well as many bugfixes and internal changes.

Many thanks to the 40 contributors to this release!:

Aaron Spring, Akio Taniguchi, Alan D. Snow, arfy slowy, Benoit Bovy, Christian Jauvin, crusaderky, Deepak Cherian,
Giacomo Caria, Illviljan, James Bourbeau, Joe Hamman, Joseph K Aicher, Julien Herzen, Kai Mühlbauer,
keewis, lusewell, Martin K. Scherer, Mathias Hauser, Max Grover, Maxime Liquet, Maximilian Roos, Mike Taves, Nathan Lis,
pmav99, Pushkar Kopparla, Ray Bell, Rio McMahon, Scott Staniewicz, Spencer Clark, Stefan Bender, Taher Chegini,
Thomas Nicholas, Tomas Chor, Tom Augspurger, Victor Negîrneac, Zachary Blackwood, Zachary Moon, and Zeb Nicholls.

New Features
~~~~~~~~~~~~
- Add ``std``, ``var``,  ``sum_of_squares`` to :py:class:`~core.weighted.DatasetWeighted` and :py:class:`~core.weighted.DataArrayWeighted`.
  By `Christian Jauvin <https://github.com/cjauvin>`_.
- Added a :py:func:`get_options` method to xarray's root namespace (:issue:`5698`, :pull:`5716`)
  By `Pushkar Kopparla <https://github.com/pkopparla>`_.
- Xarray now does a better job rendering variable names that are long LaTeX sequences when plotting (:issue:`5681`, :pull:`5682`).
  By `Tomas Chor <https://github.com/tomchor>`_.
- Add an option (``"use_bottleneck"``) to disable the use of ``bottleneck`` using :py:func:`set_options` (:pull:`5560`)
  By `Justus Magin <https://github.com/keewis>`_.
- Added ``**kwargs`` argument to :py:meth:`open_rasterio` to access overviews (:issue:`3269`).
  By `Pushkar Kopparla <https://github.com/pkopparla>`_.
- Added ``storage_options`` argument to :py:meth:`to_zarr` (:issue:`5601`, :pull:`5615`).
  By `Ray Bell <https://github.com/raybellwaves>`_, `Zachary Blackwood <https://github.com/blackary>`_ and
  `Nathan Lis <https://github.com/wxman22>`_.
- Added calendar utilities :py:func:`DataArray.convert_calendar`, :py:func:`DataArray.interp_calendar`, :py:func:`date_range`, :py:func:`date_range_like` and :py:attr:`DataArray.dt.calendar` (:issue:`5155`, :pull:`5233`).
  By `Pascal Bourgault <https://github.com/aulemahal>`_.
- Histogram plots are set with a title displaying the scalar coords if any, similarly to the other plots (:issue:`5791`, :pull:`5792`).
  By `Maxime Liquet <https://github.com/maximlt>`_.
- Slice plots display the coords units in the same way as x/y/colorbar labels (:pull:`5847`).
  By `Victor Negîrneac <https://github.com/caenrigen>`_.
- Added a new :py:attr:`Dataset.chunksizes`, :py:attr:`DataArray.chunksizes`, and :py:attr:`Variable.chunksizes`
  property, which will always return a mapping from dimension names to chunking pattern along that dimension,
  regardless of whether the object is a Dataset, DataArray, or Variable. (:issue:`5846`, :pull:`5900`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Breaking changes
~~~~~~~~~~~~~~~~
- The minimum versions of some dependencies were changed:

  =============== ====== ====
  Package         Old    New
  =============== ====== ====
  cftime          1.1    1.2
  dask            2.15   2.30
  distributed     2.15   2.30
  lxml            4.5    4.6
  matplotlib-base 3.2    3.3
  numba           0.49   0.51
  numpy           1.17   1.18
  pandas          1.0    1.1
  pint            0.15   0.16
  scipy           1.4    1.5
  seaborn         0.10   0.11
  sparse          0.8    0.11
  toolz           0.10   0.11
  zarr            2.4    2.5
  =============== ====== ====

- The ``__repr__`` of a :py:class:`xarray.Dataset`'s ``coords`` and ``data_vars``
  ignore ``xarray.set_option(display_max_rows=...)`` and show the full output
  when called directly as, e.g., ``ds.data_vars`` or ``print(ds.data_vars)``
  (:issue:`5545`, :pull:`5580`).
  By `Stefan Bender <https://github.com/st-bender>`_.

Deprecations
~~~~~~~~~~~~

- Deprecate :py:func:`open_rasterio` (:issue:`4697`, :pull:`5808`).
  By `Alan Snow <https://github.com/snowman2>`_.
- Set the default argument for ``roll_coords`` to ``False`` for :py:meth:`DataArray.roll`
  and :py:meth:`Dataset.roll`. (:pull:`5653`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- :py:meth:`xarray.open_mfdataset` will now error instead of warn when a value for ``concat_dim`` is
  passed alongside ``combine='by_coords'``.
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Bug fixes
~~~~~~~~~

- Fix ZeroDivisionError from saving dask array with empty dimension (:issue:`5741`).
  By `Joseph K Aicher <https://github.com/jaicher>`_.
- Fixed performance bug where ``cftime`` import attempted within various core operations if ``cftime`` not
  installed (:pull:`5640`).
  By `Luke Sewell <https://github.com/lusewell>`_
- Fixed bug when combining named DataArrays using :py:func:`combine_by_coords`. (:pull:`5834`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- When a custom engine was used in :py:func:`~xarray.open_dataset` the engine
  wasn't initialized properly, causing missing argument errors or inconsistent
  method signatures. (:pull:`5684`)
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Numbers are properly formatted in a plot's title (:issue:`5788`, :pull:`5789`).
  By `Maxime Liquet <https://github.com/maximlt>`_.
- Faceted plots will no longer raise a ``pint.UnitStrippedWarning`` when a ``pint.Quantity`` array is plotted,
  and will correctly display the units of the data in the colorbar (if there is one) (:pull:`5886`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- With backends, check for path-like objects rather than ``pathlib.Path``
  type, use ``os.fspath`` (:pull:`5879`).
  By `Mike Taves <https://github.com/mwtoews>`_.
- ``open_mfdataset()`` now accepts a single ``pathlib.Path`` object (:issue:`5881`).
  By `Panos Mavrogiorgos <https://github.com/pmav99>`_.
- Improved performance of :py:meth:`Dataset.unstack` (:pull:`5906`). By `Tom Augspurger <https://github.com/TomAugspurger>`_.

Documentation
~~~~~~~~~~~~~

- Users are instructed to try ``use_cftime=True`` if a ``TypeError`` occurs when combining datasets and one of the types involved is a subclass of ``cftime.datetime`` (:pull:`5776`).
  By `Zeb Nicholls <https://github.com/znicholls>`_.
- A clearer error is now raised if a user attempts to assign a Dataset to a single key of
  another Dataset. (:pull:`5839`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Explicit indexes refactor: avoid ``len(index)`` in ``map_blocks`` (:pull:`5670`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Explicit indexes refactor: decouple ``xarray.Index``` from ``xarray.Variable`` (:pull:`5636`).
  By `Benoit Bovy <https://github.com/benbovy>`_.
- Fix ``Mapping`` argument typing to allow mypy to pass on ``str`` keys (:pull:`5690`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Annotate many of our tests, and fix some of the resulting typing errors. This will
  also mean our typing annotations are tested as part of CI. (:pull:`5728`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Improve the performance of reprs for large datasets or dataarrays. (:pull:`5661`)
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Use isort's ``float_to_top`` config. (:pull:`5695`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Remove use of the deprecated ``kind`` argument in
  :py:meth:`pandas.Index.get_slice_bound` inside :py:class:`xarray.CFTimeIndex`
  tests (:pull:`5723`).  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Refactor ``xarray.core.duck_array_ops`` to no longer special-case dispatching to
  dask versions of functions when acting on dask arrays, instead relying numpy
  and dask's adherence to NEP-18 to dispatch automatically. (:pull:`5571`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Add an ASV benchmark CI and improve performance of the benchmarks (:pull:`5796`)
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Use ``importlib`` to replace functionality of ``pkg_resources`` such
  as version setting and loading of resources. (:pull:`5845`).
  By `Martin K. Scherer <https://github.com/marscher>`_.


.. _whats-new.0.19.0:

v0.19.0 (23 July 2021)
----------------------

This release brings improvements to plotting of categorical data, the ability to specify how attributes
are combined in xarray operations, a new high-level :py:func:`unify_chunks` function, as well as various
deprecations, bug fixes, and minor improvements.


Many thanks to the 29 contributors to this release!:

Andrew Williams, Augustus, Aureliana Barghini, Benoit Bovy, crusaderky, Deepak Cherian, ellesmith88,
Elliott Sales de Andrade, Giacomo Caria, github-actions[bot], Illviljan, Joeperdefloep, joooeey, Julia Kent,
Julius Busecke, keewis, Mathias Hauser, Matthias Göbel, Mattia Almansi, Maximilian Roos, Peter Andreas Entschev,
Ray Bell, Sander, Santiago Soler, Sebastian, Spencer Clark, Stephan Hoyer, Thomas Hirtz, Thomas Nicholas.

New Features
~~~~~~~~~~~~
- Allow passing argument ``missing_dims`` to :py:meth:`Variable.transpose` and :py:meth:`Dataset.transpose`
  (:issue:`5550`, :pull:`5586`)
  By `Giacomo Caria <https://github.com/gcaria>`_.
- Allow passing a dictionary as coords to a :py:class:`DataArray` (:issue:`5527`,
  reverts :pull:`1539`, which had deprecated this due to python's inconsistent ordering in earlier versions).
  By `Sander van Rijn <https://github.com/sjvrijn>`_.
- Added :py:meth:`Dataset.coarsen.construct`, :py:meth:`DataArray.coarsen.construct` (:issue:`5454`, :pull:`5475`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Xarray now uses consolidated metadata by default when writing and reading Zarr
  stores (:issue:`5251`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- New top-level function :py:func:`unify_chunks`.
  By `Mattia Almansi <https://github.com/malmans2>`_.
- Allow assigning values to a subset of a dataset using positional or label-based
  indexing (:issue:`3015`, :pull:`5362`).
  By `Matthias Göbel <https://github.com/matzegoebel>`_.
- Attempting to reduce a weighted object over missing dimensions now raises an error (:pull:`5362`).
  By `Mattia Almansi <https://github.com/malmans2>`_.
- Add ``.sum`` to :py:meth:`~xarray.DataArray.rolling_exp` and
  :py:meth:`~xarray.Dataset.rolling_exp` for exponentially weighted rolling
  sums. These require numbagg 0.2.1;
  (:pull:`5178`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- :py:func:`xarray.cov` and :py:func:`xarray.corr` now lazily check for missing
  values if inputs are dask arrays (:issue:`4804`, :pull:`5284`).
  By `Andrew Williams <https://github.com/AndrewWilliams3142>`_.
- Attempting to ``concat`` list of elements that are not all ``Dataset`` or all ``DataArray`` now raises an error (:issue:`5051`, :pull:`5425`).
  By `Thomas Hirtz <https://github.com/thomashirtz>`_.
- allow passing a function to ``combine_attrs`` (:pull:`4896`).
  By `Justus Magin <https://github.com/keewis>`_.
- Allow plotting categorical data (:pull:`5464`).
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Allow removal of the coordinate attribute ``coordinates`` on variables by setting ``.attrs['coordinates']= None``
  (:issue:`5510`).
  By `Elle Smith <https://github.com/ellesmith88>`_.
- Added :py:meth:`DataArray.to_numpy`, :py:meth:`DataArray.as_numpy`, and :py:meth:`Dataset.as_numpy`. (:pull:`5568`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Units in plot labels are now automatically inferred from wrapped :py:meth:`pint.Quantity` arrays. (:pull:`5561`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- The default ``mode`` for :py:meth:`Dataset.to_zarr` when ``region`` is set
  has changed to the new ``mode="r+"``, which only allows for overriding
  pre-existing array values. This is a safer default than the prior ``mode="a"``,
  and allows for higher performance writes (:pull:`5252`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- The main parameter to :py:func:`combine_by_coords` is renamed to ``data_objects`` instead
  of ``datasets`` so anyone calling this method using a named parameter will need to update
  the name accordingly (:issue:`3248`, :pull:`4696`).
  By `Augustus Ijams <https://github.com/aijams>`_.

Deprecations
~~~~~~~~~~~~

- Removed the deprecated ``dim`` kwarg to :py:func:`DataArray.integrate` (:pull:`5630`)
- Removed the deprecated ``keep_attrs`` kwarg to :py:func:`DataArray.rolling` (:pull:`5630`)
- Removed the deprecated ``keep_attrs`` kwarg to :py:func:`DataArray.coarsen` (:pull:`5630`)
- Completed deprecation of passing an ``xarray.DataArray`` to :py:func:`Variable` - will now raise a ``TypeError`` (:pull:`5630`)

Bug fixes
~~~~~~~~~
- Fix a minor incompatibility between partial datetime string indexing with a
  :py:class:`CFTimeIndex` and upcoming pandas version 1.3.0 (:issue:`5356`,
  :pull:`5359`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Fix 1-level multi-index incorrectly converted to single index (:issue:`5384`,
  :pull:`5385`).
  By `Benoit Bovy <https://github.com/benbovy>`_.
- Don't cast a duck array in a coordinate to :py:class:`numpy.ndarray` in
  :py:meth:`DataArray.differentiate` (:pull:`5408`)
  By `Justus Magin <https://github.com/keewis>`_.
- Fix the ``repr`` of :py:class:`Variable` objects with ``display_expand_data=True``
  (:pull:`5406`)
  By `Justus Magin <https://github.com/keewis>`_.
- Plotting a pcolormesh with ``xscale="log"`` and/or ``yscale="log"`` works as
  expected after improving the way the interval breaks are generated (:issue:`5333`).
  By `Santiago Soler <https://github.com/santisoler>`_
- :py:func:`combine_by_coords` can now handle combining a list of unnamed
  ``DataArray`` as input (:issue:`3248`, :pull:`4696`).
  By `Augustus Ijams <https://github.com/aijams>`_.


Internal Changes
~~~~~~~~~~~~~~~~
- Run CI on the first & last python versions supported only; currently 3.7 & 3.9.
  (:pull:`5433`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Publish test results & timings on each PR.
  (:pull:`5537`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Explicit indexes refactor: add a ``xarray.Index.query()`` method in which
  one may eventually provide a custom implementation of label-based data
  selection (not ready yet for public use). Also refactor the internal,
  pandas-specific implementation into ``PandasIndex.query()`` and
  ``PandasMultiIndex.query()`` (:pull:`5322`).
  By `Benoit Bovy <https://github.com/benbovy>`_.

.. _whats-new.0.18.2:

v0.18.2 (19 May 2021)
---------------------

This release reverts a regression in xarray's unstacking of dask-backed arrays.

.. _whats-new.0.18.1:

v0.18.1 (18 May 2021)
---------------------

This release is intended as a small patch release to be compatible with the new
2021.5.0 ``dask.distributed`` release. It also includes a new
``drop_duplicates`` method, some documentation improvements, the beginnings of
our internal Index refactoring, and some bug fixes.

Thank you to all 16 contributors!

Anderson Banihirwe, Andrew, Benoit Bovy, Brewster Malevich, Giacomo Caria,
Illviljan, James Bourbeau, Keewis, Maximilian Roos, Ravin Kumar, Stephan Hoyer,
Thomas Nicholas, Tom Nicholas, Zachary Moon.

New Features
~~~~~~~~~~~~
- Implement :py:meth:`DataArray.drop_duplicates`
  to remove duplicate dimension values (:pull:`5239`).
  By `Andrew Huang <https://github.com/ahuang11>`_.
- Allow passing ``combine_attrs`` strategy names to the ``keep_attrs`` parameter of
  :py:func:`apply_ufunc` (:pull:`5041`)
  By `Justus Magin <https://github.com/keewis>`_.
- :py:meth:`Dataset.interp` now allows interpolation with non-numerical datatypes,
  such as booleans, instead of dropping them. (:issue:`4761` :pull:`5008`).
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Raise more informative error when decoding time variables with invalid reference dates.
  (:issue:`5199`, :pull:`5288`). By `Giacomo Caria <https://github.com/gcaria>`_.


Bug fixes
~~~~~~~~~
- Opening netCDF files from a path that doesn't end in ``.nc`` without supplying
  an explicit ``engine`` works again (:issue:`5295`), fixing a bug introduced in
  0.18.0.
  By `Stephan Hoyer <https://github.com/shoyer>`_

Documentation
~~~~~~~~~~~~~
- Clean up and enhance docstrings for the :py:class:`DataArray.plot` and ``Dataset.plot.*``
  families of methods (:pull:`5285`).
  By `Zach Moon <https://github.com/zmoon>`_.

- Explanation of deprecation cycles and how to implement them added to contributors
  guide. (:pull:`5289`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.


Internal Changes
~~~~~~~~~~~~~~~~

- Explicit indexes refactor: add an ``xarray.Index`` base class and
  ``Dataset.xindexes`` / ``DataArray.xindexes`` properties. Also rename
  ``PandasIndexAdapter`` to ``PandasIndex``, which now inherits from
  ``xarray.Index`` (:pull:`5102`).
  By `Benoit Bovy <https://github.com/benbovy>`_.
- Replace ``SortedKeysDict`` with python's ``dict``, given dicts are now ordered.
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Updated the release guide for developers. Now accounts for actions that are automated via github
  actions. (:pull:`5274`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

.. _whats-new.0.18.0:

v0.18.0 (6 May 2021)
--------------------

This release brings a few important performance improvements, a wide range of
usability upgrades, lots of bug fixes, and some new features. These include
a plugin API to add backend engines, a new theme for the documentation,
curve fitting methods, and several new plotting functions.

Many thanks to the 38 contributors to this release: Aaron Spring, Alessandro Amici,
Alex Marandon, Alistair Miles, Ana Paula Krelling, Anderson Banihirwe, Aureliana Barghini,
Baudouin Raoult, Benoit Bovy, Blair Bonnett, David Trémouilles, Deepak Cherian,
Gabriel Medeiros Abrahão, Giacomo Caria, Hauke Schulz, Illviljan, Mathias Hauser, Matthias Bussonnier,
Mattia Almansi, Maximilian Roos, Ray Bell, Richard Kleijn, Ryan Abernathey, Sam Levang, Spencer Clark,
Spencer Jones, Tammas Loughran, Tobias Kölling, Todd, Tom Nicholas, Tom White, Victor Negîrneac,
Xianxiang Li, Zeb Nicholls, crusaderky, dschwoerer, johnomotani, keewis


New Features
~~~~~~~~~~~~

- apply ``combine_attrs`` on data variables and coordinate variables when concatenating
  and merging datasets and dataarrays (:pull:`4902`).
  By `Justus Magin <https://github.com/keewis>`_.
- Add :py:meth:`Dataset.to_pandas` (:pull:`5247`)
  By `Giacomo Caria <https://github.com/gcaria>`_.
- Add :py:meth:`DataArray.plot.surface` which wraps matplotlib's ``plot_surface`` to make
  surface plots (:issue:`2235` :issue:`5084` :pull:`5101`).
  By `John Omotani <https://github.com/johnomotani>`_.
- Allow passing multiple arrays to :py:meth:`Dataset.__setitem__` (:pull:`5216`).
  By `Giacomo Caria <https://github.com/gcaria>`_.
- Add 'cumulative' option to :py:meth:`Dataset.integrate` and
  :py:meth:`DataArray.integrate` so that result is a cumulative integral, like
  :py:func:`scipy.integrate.cumulative_trapezoidal` (:pull:`5153`).
  By `John Omotani <https://github.com/johnomotani>`_.
- Add ``safe_chunks`` option to :py:meth:`Dataset.to_zarr` which allows overriding
  checks made to ensure Dask and Zarr chunk compatibility (:issue:`5056`).
  By `Ryan Abernathey <https://github.com/rabernat>`_
- Add :py:meth:`Dataset.query` and :py:meth:`DataArray.query` which enable indexing
  of datasets and data arrays by evaluating query expressions against the values of the
  data variables (:pull:`4984`).
  By `Alistair Miles <https://github.com/alimanfoo>`_.
- Allow passing ``combine_attrs`` to :py:meth:`Dataset.merge` (:pull:`4895`).
  By `Justus Magin <https://github.com/keewis>`_.
- Support for `dask.graph_manipulation
  <https://docs.dask.org/en/latest/graph_manipulation.html>`_ (requires dask >=2021.3)
  By `Guido Imperiale <https://github.com/crusaderky>`_
- Add :py:meth:`Dataset.plot.streamplot` for streamplot plots with :py:class:`Dataset`
  variables (:pull:`5003`).
  By `John Omotani <https://github.com/johnomotani>`_.
- Many of the arguments for the :py:attr:`DataArray.str` methods now support
  providing an array-like input. In this case, the array provided to the
  arguments is broadcast against the original array and applied elementwise.
- :py:attr:`DataArray.str` now supports ``+``, ``*``, and ``%`` operators. These
  behave the same as they do for :py:class:`str`, except that they follow
  array broadcasting rules.
- A large number of new :py:attr:`DataArray.str` methods were implemented,
  :py:meth:`DataArray.str.casefold`, :py:meth:`DataArray.str.cat`,
  :py:meth:`DataArray.str.extract`, :py:meth:`DataArray.str.extractall`,
  :py:meth:`DataArray.str.findall`, :py:meth:`DataArray.str.format`,
  :py:meth:`DataArray.str.get_dummies`, :py:meth:`DataArray.str.islower`,
  :py:meth:`DataArray.str.join`, :py:meth:`DataArray.str.normalize`,
  :py:meth:`DataArray.str.partition`, :py:meth:`DataArray.str.rpartition`,
  :py:meth:`DataArray.str.rsplit`, and  :py:meth:`DataArray.str.split`.
  A number of these methods allow for splitting or joining the strings in an
  array. (:issue:`4622`)
  By `Todd Jennings <https://github.com/toddrjen>`_
- Thanks to the new pluggable backend infrastructure external packages may now
  use the ``xarray.backends`` entry point to register additional engines to be used in
  :py:func:`open_dataset`, see the documentation in :ref:`add_a_backend`
  (:issue:`4309`, :issue:`4803`, :pull:`4989`, :pull:`4810` and many others).
  The backend refactor has been sponsored with the "Essential Open Source Software for Science"
  grant from the `Chan Zuckerberg Initiative <https://chanzuckerberg.com>`_ and
  developed by `B-Open <https://www.bopen.eu>`_.
  By `Aureliana Barghini <https://github.com/aurghs>`_ and `Alessandro Amici <https://github.com/alexamici>`_.
- :py:attr:`~core.accessor_dt.DatetimeAccessor.date` added (:issue:`4983`, :pull:`4994`).
  By `Hauke Schulz <https://github.com/observingClouds>`_.
- Implement ``__getitem__`` for both :py:class:`~core.groupby.DatasetGroupBy` and
  :py:class:`~core.groupby.DataArrayGroupBy`, inspired by pandas'
  :py:meth:`~pandas.core.groupby.GroupBy.get_group`.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Switch the tutorial functions to use `pooch <https://github.com/fatiando/pooch>`_
  (which is now a optional dependency) and add :py:func:`tutorial.open_rasterio` as a
  way to open example rasterio files (:issue:`3986`, :pull:`4102`, :pull:`5074`).
  By `Justus Magin <https://github.com/keewis>`_.
- Add typing information to unary and binary arithmetic operators operating on
  :py:class:`Dataset`, :py:class:`DataArray`, :py:class:`Variable`,
  :py:class:`~core.groupby.DatasetGroupBy` or
  :py:class:`~core.groupby.DataArrayGroupBy` (:pull:`4904`).
  By `Richard Kleijn <https://github.com/rhkleijn>`_.
- Add a ``combine_attrs`` parameter to :py:func:`open_mfdataset` (:pull:`4971`).
  By `Justus Magin <https://github.com/keewis>`_.
- Enable passing arrays with a subset of dimensions to
  :py:meth:`DataArray.clip` & :py:meth:`Dataset.clip`; these methods now use
  :py:func:`xarray.apply_ufunc`; (:pull:`5184`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Disable the ``cfgrib`` backend if the ``eccodes`` library is not installed (:pull:`5083`).
  By `Baudouin Raoult <https://github.com/b8raoult>`_.
- Added :py:meth:`DataArray.curvefit` and :py:meth:`Dataset.curvefit` for general curve fitting applications. (:issue:`4300`, :pull:`4849`)
  By `Sam Levang <https://github.com/slevang>`_.
- Add options to control expand/collapse of sections in display of Dataset and
  DataArray. The function :py:func:`set_options` now takes keyword arguments
  ``display_expand_attrs``, ``display_expand_coords``, ``display_expand_data``,
  ``display_expand_data_vars``, all of which can be one of ``True`` to always
  expand, ``False`` to always collapse, or ``default`` to expand unless over a
  pre-defined limit (:pull:`5126`).
  By `Tom White <https://github.com/tomwhite>`_.
- Significant speedups in :py:meth:`Dataset.interp` and :py:meth:`DataArray.interp`.
  (:issue:`4739`, :pull:`4740`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Prevent passing ``concat_dim`` to :py:func:`xarray.open_mfdataset` when
  ``combine='by_coords'`` is specified, which should never have been possible (as
  :py:func:`xarray.combine_by_coords` has no ``concat_dim`` argument to pass to).
  Also removes unneeded internal reordering of datasets in
  :py:func:`xarray.open_mfdataset` when ``combine='by_coords'`` is specified.
  Fixes (:issue:`5230`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Implement ``__setitem__`` for ``xarray.core.indexing.DaskIndexingAdapter`` if
  dask version supports item assignment. (:issue:`5171`, :pull:`5174`)
  By `Tammas Loughran <https://github.com/tammasloughran>`_.

Breaking changes
~~~~~~~~~~~~~~~~
- The minimum versions of some dependencies were changed:

  ============ ====== ====
  Package      Old    New
  ============ ====== ====
  boto3        1.12   1.13
  cftime       1.0    1.1
  dask         2.11   2.15
  distributed  2.11   2.15
  matplotlib   3.1    3.2
  numba        0.48   0.49
  ============ ====== ====

- :py:func:`open_dataset` and :py:func:`open_dataarray` now accept only the first argument
  as positional, all others need to be passed are keyword arguments. This is part of the
  refactor to support external backends (:issue:`4309`, :pull:`4989`).
  By `Alessandro Amici <https://github.com/alexamici>`_.
- Functions that are identities for 0d data return the unchanged data
  if axis is empty. This ensures that Datasets where some variables do
  not have the averaged dimensions are not accidentally changed
  (:issue:`4885`, :pull:`5207`).
  By `David Schwörer <https://github.com/dschwoerer>`_.
- :py:attr:`DataArray.coarsen` and :py:attr:`Dataset.coarsen` no longer support passing ``keep_attrs``
  via its constructor. Pass ``keep_attrs`` via the applied function, i.e. use
  ``ds.coarsen(...).mean(keep_attrs=False)`` instead of ``ds.coarsen(..., keep_attrs=False).mean()``.
  Further, coarsen now keeps attributes per default (:pull:`5227`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- switch the default of the :py:func:`merge` ``combine_attrs`` parameter to
  ``"override"``. This will keep the current behavior for merging the ``attrs`` of
  variables but stop dropping the ``attrs`` of the main objects (:pull:`4902`).
  By `Justus Magin <https://github.com/keewis>`_.

Deprecations
~~~~~~~~~~~~

- Warn when passing ``concat_dim`` to :py:func:`xarray.open_mfdataset` when
  ``combine='by_coords'`` is specified, which should never have been possible (as
  :py:func:`xarray.combine_by_coords` has no ``concat_dim`` argument to pass to).
  Also removes unneeded internal reordering of datasets in
  :py:func:`xarray.open_mfdataset` when ``combine='by_coords'`` is specified.
  Fixes (:issue:`5230`), via (:pull:`5231`, :pull:`5255`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- The ``lock`` keyword argument to :py:func:`open_dataset` and :py:func:`open_dataarray` is now
  a backend specific option. It will give a warning if passed to a backend that doesn't support it
  instead of being silently ignored. From the next version it will raise an error.
  This is part of the refactor to support external backends (:issue:`5073`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_ and `Alessandro Amici <https://github.com/alexamici>`_.


Bug fixes
~~~~~~~~~
- Properly support :py:meth:`DataArray.ffill`, :py:meth:`DataArray.bfill`, :py:meth:`Dataset.ffill`, :py:meth:`Dataset.bfill` along chunked dimensions.
  (:issue:`2699`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix 2d plot failure for certain combinations of dimensions when ``x`` is 1d and ``y`` is
  2d (:issue:`5097`, :pull:`5099`).
  By `John Omotani <https://github.com/johnomotani>`_.
- Ensure standard calendar times encoded with large values (i.e. greater than
  approximately 292 years), can be decoded correctly without silently overflowing
  (:pull:`5050`).  This was a regression in xarray 0.17.0.
  By `Zeb Nicholls <https://github.com/znicholls>`_.
- Added support for ``numpy.bool_`` attributes in roundtrips using ``h5netcdf`` engine with ``invalid_netcdf=True`` [which casts ``bool`` s to ``numpy.bool_``] (:issue:`4981`, :pull:`4986`).
  By `Victor Negîrneac <https://github.com/caenrigen>`_.
- Don't allow passing ``axis`` to :py:meth:`Dataset.reduce` methods (:issue:`3510`, :pull:`4940`).
  By `Justus Magin <https://github.com/keewis>`_.
- Decode values as signed if attribute ``_Unsigned = "false"`` (:issue:`4954`)
  By `Tobias Kölling <https://github.com/d70-t>`_.
- Keep coords attributes when interpolating when the indexer is not a Variable. (:issue:`4239`, :issue:`4839` :pull:`5031`)
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Ensure standard calendar dates encoded with a calendar attribute with some or
  all uppercase letters can be decoded or encoded to or from
  ``np.datetime64[ns]`` dates with or without ``cftime`` installed
  (:issue:`5093`, :pull:`5180`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Warn on passing ``keep_attrs`` to ``resample`` and ``rolling_exp`` as they are ignored, pass ``keep_attrs``
  to the applied function instead (:pull:`5265`).
  By `Mathias Hauser <https://github.com/mathause>`_.

Documentation
~~~~~~~~~~~~~
- New section on :ref:`add_a_backend` in the "Internals" chapter aimed to backend developers
  (:issue:`4803`, :pull:`4810`).
  By `Aureliana Barghini <https://github.com/aurghs>`_.
- Add :py:meth:`Dataset.polyfit` and :py:meth:`DataArray.polyfit` under "See also" in
  the docstrings of :py:meth:`Dataset.polyfit` and :py:meth:`DataArray.polyfit`
  (:issue:`5016`, :pull:`5020`).
  By `Aaron Spring <https://github.com/aaronspring>`_.
- New sphinx theme & rearrangement of the docs (:pull:`4835`).
  By `Anderson Banihirwe <https://github.com/andersy005>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Enable displaying mypy error codes and ignore only specific error codes using
  ``# type: ignore[error-code]`` (:pull:`5096`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Replace uses of ``raises_regex`` with the more standard
  ``pytest.raises(Exception, match="foo")``;
  (:pull:`5188`), (:pull:`5191`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.

.. _whats-new.0.17.0:

v0.17.0 (24 Feb 2021)
---------------------

This release brings a few important performance improvements, a wide range of
usability upgrades, lots of bug fixes, and some new features. These include
better ``cftime`` support, a new quiver plot, better ``unstack`` performance,
more efficient memory use in rolling operations, and some python packaging
improvements. We also have a few documentation improvements (and more planned!).

Many thanks to the 36 contributors to this release: Alessandro Amici, Anderson
Banihirwe, Aureliana Barghini, Ayrton Bourn, Benjamin Bean, Blair Bonnett, Chun
Ho Chow, DWesl, Daniel Mesejo-León, Deepak Cherian, Eric Keenan, Illviljan, Jens
Hedegaard Nielsen, Jody Klymak, Julien Seguinot, Julius Busecke, Kai Mühlbauer,
Leif Denby, Martin Durant, Mathias Hauser, Maximilian Roos, Michael Mann, Ray
Bell, RichardScottOZ, Spencer Clark, Tim Gates, Tom Nicholas, Yunus Sevinchan,
alexamici, aurghs, crusaderky, dcherian, ghislainp, keewis, rhkleijn

Breaking changes
~~~~~~~~~~~~~~~~
- xarray no longer supports python 3.6

  The minimum version policy was changed to also apply to projects with irregular
  releases. As a result, the minimum versions of some dependencies have changed:

  ============ ====== ====
  Package      Old    New
  ============ ====== ====
  Python       3.6    3.7
  setuptools   38.4   40.4
  numpy        1.15   1.17
  pandas       0.25   1.0
  dask         2.9    2.11
  distributed  2.9    2.11
  bottleneck   1.2    1.3
  h5netcdf     0.7    0.8
  iris         2.2    2.4
  netcdf4      1.4    1.5
  pseudonetcdf 3.0    3.1
  rasterio     1.0    1.1
  scipy        1.3    1.4
  seaborn      0.9    0.10
  zarr         2.3    2.4
  ============ ====== ====

  (:issue:`4688`, :pull:`4720`, :pull:`4907`, :pull:`4942`)
- As a result of :pull:`4684` the default units encoding for
  datetime-like values (``np.datetime64[ns]`` or ``cftime.datetime``) will now
  always be set such that ``int64`` values can be used.  In the past, no units
  finer than "seconds" were chosen, which would sometimes mean that ``float64``
  values were required, which would lead to inaccurate I/O round-trips.
- Variables referred to in attributes like ``bounds`` and ``grid_mapping``
  can be set as coordinate variables. These attributes are moved to
  :py:attr:`DataArray.encoding` from :py:attr:`DataArray.attrs`. This behaviour
  is controlled by the ``decode_coords`` kwarg to :py:func:`open_dataset` and
  :py:func:`open_mfdataset`.  The full list of decoded attributes is in
  :ref:`weather-climate` (:pull:`2844`, :issue:`3689`)
- As a result of :pull:`4911` the output from calling :py:meth:`DataArray.sum`
  or :py:meth:`DataArray.prod` on an integer array with ``skipna=True`` and a
  non-None value for ``min_count`` will now be a float array rather than an
  integer array.

Deprecations
~~~~~~~~~~~~

- ``dim`` argument to :py:meth:`DataArray.integrate` is being deprecated in
  favour of a ``coord`` argument, for consistency with :py:meth:`Dataset.integrate`.
  For now using ``dim`` issues a ``FutureWarning``. It will be removed in
  version 0.19.0 (:pull:`3993`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Deprecated ``autoclose`` kwargs from :py:func:`open_dataset` are removed (:pull:`4725`).
  By `Aureliana Barghini <https://github.com/aurghs>`_.
- the return value of :py:meth:`Dataset.update` is being deprecated to make it work more
  like :py:meth:`dict.update`. It will be removed in version 0.19.0 (:pull:`4932`).
  By `Justus Magin <https://github.com/keewis>`_.

New Features
~~~~~~~~~~~~
- :py:meth:`~xarray.cftime_range` and :py:meth:`DataArray.resample` now support
  millisecond (``"L"`` or ``"ms"``) and microsecond (``"U"`` or ``"us"``) frequencies
  for ``cftime.datetime`` coordinates (:issue:`4097`, :pull:`4758`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Significantly higher ``unstack`` performance on numpy-backed arrays which
  contain missing values; 8x faster than previous versions in our benchmark, and
  now 2x faster than pandas (:pull:`4746`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Add :py:meth:`Dataset.plot.quiver` for quiver plots with :py:class:`Dataset` variables.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Add ``"drop_conflicts"`` to the strategies supported by the ``combine_attrs`` kwarg
  (:issue:`4749`, :pull:`4827`).
  By `Justus Magin <https://github.com/keewis>`_.
- Allow installing from git archives (:pull:`4897`).
  By `Justus Magin <https://github.com/keewis>`_.
- :py:class:`~core.rolling.DataArrayCoarsen` and :py:class:`~core.rolling.DatasetCoarsen`
  now implement a ``reduce`` method, enabling coarsening operations with custom
  reduction functions (:issue:`3741`, :pull:`4939`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Most rolling operations use significantly less memory. (:issue:`4325`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Add :py:meth:`Dataset.drop_isel` and :py:meth:`DataArray.drop_isel`
  (:issue:`4658`, :pull:`4819`).
  By `Daniel Mesejo <https://github.com/mesejo>`_.
- Xarray now leverages updates as of cftime version 1.4.1, which enable exact I/O
  roundtripping of ``cftime.datetime`` objects (:pull:`4758`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- :py:func:`open_dataset` and :py:func:`open_mfdataset` now accept ``fsspec`` URLs
  (including globs for the latter) for ``engine="zarr"``, and so allow reading from
  many remote and other file systems (:pull:`4461`)
  By `Martin Durant <https://github.com/martindurant>`_
- :py:meth:`DataArray.swap_dims` & :py:meth:`Dataset.swap_dims` now accept dims
  in the form of kwargs as well as a dict, like most similar methods.
  By `Maximilian Roos <https://github.com/max-sixty>`_.

Bug fixes
~~~~~~~~~
- Use specific type checks in ``xarray.core.variable.as_compatible_data`` instead of
  blanket access to ``values`` attribute (:issue:`2097`)
  By `Yunus Sevinchan <https://github.com/blsqr>`_.
- :py:meth:`DataArray.resample` and :py:meth:`Dataset.resample` do not trigger
  computations anymore if :py:meth:`Dataset.weighted` or
  :py:meth:`DataArray.weighted` are applied (:issue:`4625`, :pull:`4668`). By
  `Julius Busecke <https://github.com/jbusecke>`_.
- :py:func:`merge` with ``combine_attrs='override'`` makes a copy of the attrs
  (:issue:`4627`).
- By default, when possible, xarray will now always use values of
  type ``int64`` when encoding and decoding ``numpy.datetime64[ns]`` datetimes.  This
  ensures that maximum precision and accuracy are maintained in the round-tripping
  process (:issue:`4045`, :pull:`4684`). It also enables encoding and decoding standard
  calendar dates with time units of nanoseconds (:pull:`4400`).
  By `Spencer Clark <https://github.com/spencerkclark>`_ and `Mark Harfouche
  <https://github.com/hmaarrfk>`_.
- :py:meth:`DataArray.astype`, :py:meth:`Dataset.astype` and :py:meth:`Variable.astype` support
  the ``order`` and ``subok`` parameters again. This fixes a regression introduced in version 0.16.1
  (:issue:`4644`, :pull:`4683`).
  By `Richard Kleijn <https://github.com/rhkleijn>`_ .
- Remove dictionary unpacking when using ``.loc`` to avoid collision with ``.sel`` parameters (:pull:`4695`).
  By `Anderson Banihirwe <https://github.com/andersy005>`_.
- Fix the legend created by :py:meth:`Dataset.plot.scatter` (:issue:`4641`, :pull:`4723`).
  By `Justus Magin <https://github.com/keewis>`_.
- Fix a crash in orthogonal indexing on geographic coordinates with ``engine='cfgrib'``
  (:issue:`4733` :pull:`4737`).
  By `Alessandro Amici <https://github.com/alexamici>`_.
- Coordinates with dtype ``str`` or ``bytes`` now retain their dtype on many operations,
  e.g. ``reindex``, ``align``, ``concat``, ``assign``, previously they were cast to an object dtype
  (:issue:`2658` and :issue:`4543`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Limit number of data rows when printing large datasets. (:issue:`4736`, :pull:`4750`).
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Add ``missing_dims`` parameter to transpose (:issue:`4647`, :pull:`4767`).
  By `Daniel Mesejo <https://github.com/mesejo>`_.
- Resolve intervals before appending other metadata to labels when plotting (:issue:`4322`, :pull:`4794`).
  By `Justus Magin <https://github.com/keewis>`_.
- Fix regression when decoding a variable with a ``scale_factor`` and ``add_offset`` given
  as a list of length one (:issue:`4631`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Expand user directory paths (e.g. ``~/``) in :py:func:`open_mfdataset` and
  :py:meth:`Dataset.to_zarr` (:issue:`4783`, :pull:`4795`).
  By `Julien Seguinot <https://github.com/juseg>`_.
- Raise DeprecationWarning when trying to typecast a tuple containing a :py:class:`DataArray`.
  User now prompted to first call ``.data`` on it (:issue:`4483`).
  By `Chun Ho Chow <https://github.com/chunhochow>`_.
- Ensure that :py:meth:`Dataset.interp` raises ``ValueError`` when interpolating
  outside coordinate range and ``bounds_error=True`` (:issue:`4854`,
  :pull:`4855`).
  By `Leif Denby <https://github.com/leifdenby>`_.
- Fix time encoding bug associated with using cftime versions greater than
  1.4.0 with xarray (:issue:`4870`, :pull:`4871`).
  By `Spencer Clark <https://github.com/spencerkclark>`_.
- Stop :py:meth:`DataArray.sum` and :py:meth:`DataArray.prod` computing lazy
  arrays when called with a ``min_count`` parameter (:issue:`4898`, :pull:`4911`).
  By `Blair Bonnett <https://github.com/bcbnz>`_.
- Fix bug preventing the ``min_count`` parameter to :py:meth:`DataArray.sum` and
  :py:meth:`DataArray.prod` working correctly when calculating over all axes of
  a float64 array (:issue:`4898`, :pull:`4911`).
  By `Blair Bonnett <https://github.com/bcbnz>`_.
- Fix decoding of vlen strings using h5py versions greater than 3.0.0 with h5netcdf backend (:issue:`4570`, :pull:`4893`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Allow converting :py:class:`Dataset` or :py:class:`DataArray` objects with a ``MultiIndex``
  and at least one other dimension to a ``pandas`` object (:issue:`3008`, :pull:`4442`).
  By `ghislainp <https://github.com/ghislainp>`_.

Documentation
~~~~~~~~~~~~~
- Add information about requirements for accessor classes (:issue:`2788`, :pull:`4657`).
  By `Justus Magin <https://github.com/keewis>`_.
- Start a list of external I/O integrating with ``xarray`` (:issue:`683`, :pull:`4566`).
  By `Justus Magin <https://github.com/keewis>`_.
- Add concat examples and improve combining documentation (:issue:`4620`, :pull:`4645`).
  By `Ray Bell <https://github.com/raybellwaves>`_ and
  `Justus Magin <https://github.com/keewis>`_.
- explicitly mention that :py:meth:`Dataset.update` updates inplace (:issue:`2951`, :pull:`4932`).
  By `Justus Magin <https://github.com/keewis>`_.
- Added docs on vectorized indexing (:pull:`4711`).
  By `Eric Keenan <https://github.com/EricKeenan>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Speed up of the continuous integration tests on azure.

  - Switched to mamba and use matplotlib-base for a faster installation of all dependencies (:pull:`4672`).
  - Use ``pytest.mark.skip`` instead of ``pytest.mark.xfail`` for some tests that can currently not
    succeed (:pull:`4685`).
  - Run the tests in parallel using pytest-xdist (:pull:`4694`).

  By `Justus Magin <https://github.com/keewis>`_ and `Mathias Hauser <https://github.com/mathause>`_.
- Use ``pyproject.toml`` instead of the ``setup_requires`` option for
  ``setuptools`` (:pull:`4897`).
  By `Justus Magin <https://github.com/keewis>`_.
- Replace all usages of ``assert x.identical(y)`` with ``assert_identical(x,  y)``
  for clearer error messages (:pull:`4752`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Speed up attribute style access (e.g. ``ds.somevar`` instead of ``ds["somevar"]``) and
  tab completion in IPython (:issue:`4741`, :pull:`4742`).
  By `Richard Kleijn <https://github.com/rhkleijn>`_.
- Added the ``set_close`` method to ``Dataset`` and ``DataArray`` for backends
  to specify how to voluntary release all resources. (:pull:`#4809`)
  By `Alessandro Amici <https://github.com/alexamici>`_.
- Update type hints to work with numpy v1.20 (:pull:`4878`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Ensure warnings cannot be turned into exceptions in :py:func:`testing.assert_equal` and
  the other ``assert_*`` functions (:pull:`4864`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Performance improvement when constructing DataArrays. Significantly speeds up
  repr for Datasets with large number of variables.
  By `Deepak Cherian <https://github.com/dcherian>`_.

.. _whats-new.0.16.2:

v0.16.2 (30 Nov 2020)
---------------------

This release brings the ability to write to limited regions of ``zarr`` files,
open zarr files with :py:func:`open_dataset` and :py:func:`open_mfdataset`,
increased support for propagating ``attrs`` using the ``keep_attrs`` flag, as
well as numerous bugfixes and documentation improvements.

Many thanks to the 31 contributors who contributed to this release: Aaron
Spring, Akio Taniguchi, Aleksandar Jelenak, alexamici, Alexandre Poux, Anderson
Banihirwe, Andrew Pauling, Ashwin Vishnu, aurghs, Brian Ward, Caleb, crusaderky,
Dan Nowacki, darikg, David Brochart, David Huard, Deepak Cherian, Dion Häfner,
Gerardo Rivera, Gerrit Holl, Illviljan, inakleinbottle, Jacob Tomlinson, James
A. Bednar, jenssss, Joe Hamman, johnomotani, Joris Van den Bossche, Julia Kent,
Julius Busecke, Kai Mühlbauer, keewis, Keisuke Fujii, Kyle Cranmer, Luke
Volpatti, Mathias Hauser, Maximilian Roos, Michaël Defferrard, Michal
Baumgartner, Nick R. Papior, Pascal Bourgault, Peter Hausamann, PGijsbers, Ray
Bell, Romain Martinez, rpgoldman, Russell Manser, Sahid Velji, Samnan Rahee,
Sander, Spencer Clark, Stephan Hoyer, Thomas Zilio, Tobias Kölling, Tom
Augspurger, Wei Ji, Yash Saboo, Zeb Nicholls,

Deprecations
~~~~~~~~~~~~

- :py:attr:`~core.accessor_dt.DatetimeAccessor.weekofyear` and :py:attr:`~core.accessor_dt.DatetimeAccessor.week`
  have been deprecated. Use ``DataArray.dt.isocalendar().week``
  instead (:pull:`4534`). By `Mathias Hauser <https://github.com/mathause>`_.
  `Maximilian Roos <https://github.com/max-sixty>`_, and `Spencer Clark <https://github.com/spencerkclark>`_.
- :py:attr:`DataArray.rolling` and :py:attr:`Dataset.rolling` no longer support passing ``keep_attrs``
  via its constructor. Pass ``keep_attrs`` via the applied function, i.e. use
  ``ds.rolling(...).mean(keep_attrs=False)`` instead of ``ds.rolling(..., keep_attrs=False).mean()``
  Rolling operations now keep their attributes per default (:pull:`4510`).
  By `Mathias Hauser <https://github.com/mathause>`_.

New Features
~~~~~~~~~~~~

- :py:func:`open_dataset` and :py:func:`open_mfdataset`
  now works with ``engine="zarr"`` (:issue:`3668`, :pull:`4003`, :pull:`4187`).
  By `Miguel Jimenez <https://github.com/Mikejmnez>`_ and `Wei Ji Leong <https://github.com/weiji14>`_.
- Unary & binary operations follow the ``keep_attrs`` flag (:issue:`3490`, :issue:`4065`, :issue:`3433`, :issue:`3595`, :pull:`4195`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Added :py:meth:`~core.accessor_dt.DatetimeAccessor.isocalendar()` that returns a Dataset
  with year, week, and weekday calculated according to the ISO 8601 calendar. Requires
  pandas version 1.1.0 or greater (:pull:`4534`). By `Mathias Hauser <https://github.com/mathause>`_,
  `Maximilian Roos <https://github.com/max-sixty>`_, and `Spencer Clark <https://github.com/spencerkclark>`_.
- :py:meth:`Dataset.to_zarr` now supports a ``region`` keyword for writing to
  limited regions of existing Zarr stores (:pull:`4035`).
  See :ref:`io.zarr.appending` for full details.
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Added typehints in :py:func:`align` to reflect that the same type received in ``objects`` arg will be returned (:pull:`4522`).
  By `Michal Baumgartner <https://github.com/m1so>`_.
- :py:meth:`Dataset.weighted` and :py:meth:`DataArray.weighted` are now executing value checks lazily if weights are provided as dask arrays (:issue:`4541`, :pull:`4559`).
  By `Julius Busecke <https://github.com/jbusecke>`_.
- Added the ``keep_attrs`` keyword to ``rolling_exp.mean()``; it now keeps attributes
  per default. By `Mathias Hauser <https://github.com/mathause>`_ (:pull:`4592`).
- Added ``freq`` as property to :py:class:`CFTimeIndex` and into the
  ``CFTimeIndex.repr``. (:issue:`2416`, :pull:`4597`)
  By `Aaron Spring <https://github.com/aaronspring>`_.

Bug fixes
~~~~~~~~~

- Fix bug where reference times without padded years (e.g. ``since 1-1-1``) would lose their units when
  being passed by ``encode_cf_datetime`` (:issue:`4422`, :pull:`4506`). Such units are ambiguous
  about which digit represents the years (is it YMD or DMY?). Now, if such formatting is encountered,
  it is assumed that the first digit is the years, they are padded appropriately (to e.g. ``since 0001-1-1``)
  and a warning that this assumption is being made is issued. Previously, without ``cftime``, such times
  would be silently parsed incorrectly (at least based on the CF conventions) e.g. "since 1-1-1" would
  be parsed (via ``pandas`` and ``dateutil``) to ``since 2001-1-1``.
  By `Zeb Nicholls <https://github.com/znicholls>`_.
- Fix :py:meth:`DataArray.plot.step`. By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix bug where reading a scalar value from a NetCDF file opened with the ``h5netcdf`` backend would raise a ``ValueError`` when ``decode_cf=True`` (:issue:`4471`, :pull:`4485`).
  By `Gerrit Holl <https://github.com/gerritholl>`_.
- Fix bug where datetime64 times are silently changed to incorrect values if they are outside the valid date range for ns precision when provided in some other units (:issue:`4427`, :pull:`4454`).
  By `Andrew Pauling <https://github.com/andrewpauling>`_
- Fix silently overwriting the ``engine`` key when passing :py:func:`open_dataset` a file object
  to an incompatible netCDF (:issue:`4457`). Now incompatible combinations of files and engines raise
  an exception instead. By `Alessandro Amici <https://github.com/alexamici>`_.
- The ``min_count`` argument to :py:meth:`DataArray.sum()` and :py:meth:`DataArray.prod()`
  is now ignored when not applicable, i.e. when ``skipna=False`` or when ``skipna=None``
  and the dtype does not have a missing value (:issue:`4352`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- :py:func:`combine_by_coords` now raises an informative error when passing coordinates
  with differing calendars (:issue:`4495`). By `Mathias Hauser <https://github.com/mathause>`_.
- :py:attr:`DataArray.rolling` and :py:attr:`Dataset.rolling` now also keep the attributes and names of of (wrapped)
  ``DataArray`` objects, previously only the global attributes were retained (:issue:`4497`, :pull:`4510`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Improve performance where reading small slices from huge dimensions was slower than necessary (:pull:`4560`). By `Dion Häfner <https://github.com/dionhaefner>`_.
- Fix bug where ``dask_gufunc_kwargs`` was silently changed in :py:func:`apply_ufunc` (:pull:`4576`). By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.

Documentation
~~~~~~~~~~~~~
- document the API not supported with duck arrays (:pull:`4530`).
  By `Justus Magin <https://github.com/keewis>`_.
- Mention the possibility to pass functions to :py:meth:`Dataset.where` or
  :py:meth:`DataArray.where` in the parameter documentation (:issue:`4223`, :pull:`4613`).
  By `Justus Magin <https://github.com/keewis>`_.
- Update the docstring of :py:class:`DataArray` and :py:class:`Dataset`.
  (:pull:`4532`);
  By `Jimmy Westling <https://github.com/illviljan>`_.
- Raise a more informative error when :py:meth:`DataArray.to_dataframe` is
  is called on a scalar, (:issue:`4228`);
  By `Pieter Gijsbers <https://github.com/pgijsbers>`_.
- Fix grammar and typos in the :doc:`contributing` guide (:pull:`4545`).
  By `Sahid Velji <https://github.com/sahidvelji>`_.
- Fix grammar and typos in the :doc:`user-guide/io` guide (:pull:`4553`).
  By `Sahid Velji <https://github.com/sahidvelji>`_.
- Update link to NumPy docstring standard in the :doc:`contributing` guide (:pull:`4558`).
  By `Sahid Velji <https://github.com/sahidvelji>`_.
- Add docstrings to ``isnull`` and ``notnull``, and fix the displayed signature
  (:issue:`2760`, :pull:`4618`).
  By `Justus Magin <https://github.com/keewis>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Optional dependencies can be installed along with xarray by specifying
  extras as ``pip install "xarray[extra]"`` where ``extra`` can be one of ``io``,
  ``accel``, ``parallel``, ``viz`` and ``complete``. See docs for updated
  :ref:`installation instructions <installation-instructions>`.
  (:issue:`2888`, :pull:`4480`).
  By `Ashwin Vishnu <https://github.com/ashwinvis>`_, `Justus Magin
  <https://github.com/keewis>`_ and `Mathias Hauser
  <https://github.com/mathause>`_.
- Removed stray spaces that stem from black removing new lines (:pull:`4504`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Ensure tests are not skipped in the ``py38-all-but-dask`` test environment
  (:issue:`4509`). By `Mathias Hauser <https://github.com/mathause>`_.
- Ignore select numpy warnings around missing values, where xarray handles
  the values appropriately, (:pull:`4536`);
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Replace the internal use of ``pd.Index.__or__`` and ``pd.Index.__and__`` with ``pd.Index.union``
  and ``pd.Index.intersection`` as they will stop working as set operations in the future
  (:issue:`4565`). By `Mathias Hauser <https://github.com/mathause>`_.
- Add GitHub action for running nightly tests against upstream dependencies (:pull:`4583`).
  By `Anderson Banihirwe <https://github.com/andersy005>`_.
- Ensure all figures are closed properly in plot tests (:pull:`4600`).
  By `Yash Saboo <https://github.com/yashsaboo>`_, `Nirupam K N
  <https://github.com/Nirupamkn>`_ and `Mathias Hauser
  <https://github.com/mathause>`_.

.. _whats-new.0.16.1:

v0.16.1 (2020-09-20)
---------------------

This patch release fixes an incompatibility with a recent pandas change, which
was causing an issue indexing with a ``datetime64``. It also includes
improvements to ``rolling``, ``to_dataframe``, ``cov`` & ``corr`` methods and
bug fixes. Our documentation has a number of improvements, including fixing all
doctests and confirming their accuracy on every commit.

Many thanks to the 36 contributors who contributed to this release:

Aaron Spring, Akio Taniguchi, Aleksandar Jelenak, Alexandre Poux,
Caleb, Dan Nowacki, Deepak Cherian, Gerardo Rivera, Jacob Tomlinson, James A.
Bednar, Joe Hamman, Julia Kent, Kai Mühlbauer, Keisuke Fujii, Mathias Hauser,
Maximilian Roos, Nick R. Papior, Pascal Bourgault, Peter Hausamann, Romain
Martinez, Russell Manser, Samnan Rahee, Sander, Spencer Clark, Stephan Hoyer,
Thomas Zilio, Tobias Kölling, Tom Augspurger, alexamici, crusaderky, darikg,
inakleinbottle, jenssss, johnomotani, keewis, and rpgoldman.

Breaking changes
~~~~~~~~~~~~~~~~

- :py:meth:`DataArray.astype` and :py:meth:`Dataset.astype` now preserve attributes. Keep the
  old behavior by passing ``keep_attrs=False`` (:issue:`2049`, :pull:`4314`).
  By `Dan Nowacki <https://github.com/dnowacki-usgs>`_ and `Gabriel Joel Mitchell <https://github.com/gajomi>`_.

New Features
~~~~~~~~~~~~

- :py:meth:`~xarray.DataArray.rolling` and :py:meth:`~xarray.Dataset.rolling`
  now accept more than 1 dimension. (:pull:`4219`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- :py:meth:`~xarray.DataArray.to_dataframe` and :py:meth:`~xarray.Dataset.to_dataframe`
  now accept a ``dim_order`` parameter allowing to specify the resulting dataframe's
  dimensions order (:issue:`4331`, :pull:`4333`).
  By `Thomas Zilio <https://github.com/thomas-z>`_.
- Support multiple outputs in :py:func:`xarray.apply_ufunc` when using
  ``dask='parallelized'``. (:issue:`1815`, :pull:`4060`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- ``min_count`` can be supplied to reductions such as ``.sum`` when specifying
  multiple dimension to reduce over; (:pull:`4356`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- :py:func:`xarray.cov` and :py:func:`xarray.corr` now handle missing values; (:pull:`4351`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Add support for parsing datetime strings formatted following the default
  string representation of cftime objects, i.e. YYYY-MM-DD hh:mm:ss, in
  partial datetime string indexing, as well as :py:meth:`~xarray.cftime_range`
  (:issue:`4337`). By `Spencer Clark <https://github.com/spencerkclark>`_.
- Build ``CFTimeIndex.__repr__`` explicitly as :py:class:`pandas.Index`. Add ``calendar`` as a new
  property for :py:class:`CFTimeIndex` and show ``calendar`` and ``length`` in
  ``CFTimeIndex.__repr__`` (:issue:`2416`, :pull:`4092`)
  By `Aaron Spring <https://github.com/aaronspring>`_.
- Use a wrapped array's ``_repr_inline_`` method to construct the collapsed ``repr``
  of :py:class:`DataArray` and :py:class:`Dataset` objects and
  document the new method in :doc:`internals/index`. (:pull:`4248`).
  By `Justus Magin <https://github.com/keewis>`_.
- Allow per-variable fill values in most functions. (:pull:`4237`).
  By `Justus Magin <https://github.com/keewis>`_.
- Expose ``use_cftime`` option in :py:func:`~xarray.open_zarr` (:issue:`2886`, :pull:`3229`)
  By `Samnan Rahee <https://github.com/Geektrovert>`_ and `Anderson Banihirwe <https://github.com/andersy005>`_.

Bug fixes
~~~~~~~~~

- Fix indexing with datetime64 scalars with pandas 1.1 (:issue:`4283`).
  By `Stephan Hoyer <https://github.com/shoyer>`_ and
  `Justus Magin <https://github.com/keewis>`_.
- Variables which are chunked using dask only along some dimensions can be chunked while storing with zarr along previously
  unchunked dimensions (:pull:`4312`) By `Tobias Kölling <https://github.com/d70-t>`_.
- Fixed a bug in backend caused by basic installation of Dask (:issue:`4164`, :pull:`4318`)
  `Sam Morley <https://github.com/inakleinbottle>`_.
- Fixed a few bugs with :py:meth:`Dataset.polyfit` when encountering deficient matrix ranks (:issue:`4190`, :pull:`4193`). By `Pascal Bourgault <https://github.com/aulemahal>`_.
- Fixed inconsistencies between docstring and functionality for :py:meth:`DataArray.str.get`
  and :py:meth:`DataArray.str.wrap` (:issue:`4334`). By `Mathias Hauser <https://github.com/mathause>`_.
- Fixed overflow issue causing incorrect results in computing means of :py:class:`cftime.datetime`
  arrays (:issue:`4341`). By `Spencer Clark <https://github.com/spencerkclark>`_.
- Fixed :py:meth:`Dataset.coarsen`, :py:meth:`DataArray.coarsen` dropping attributes on original object (:issue:`4120`, :pull:`4360`). By `Julia Kent <https://github.com/jukent>`_.
- fix the signature of the plot methods. (:pull:`4359`) By `Justus Magin <https://github.com/keewis>`_.
- Fix :py:func:`xarray.apply_ufunc` with ``vectorize=True`` and ``exclude_dims`` (:issue:`3890`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Fix ``KeyError`` when doing linear interpolation to an nd ``DataArray``
  that contains NaNs (:pull:`4233`).
  By `Jens Svensmark <https://github.com/jenssss>`_
- Fix incorrect legend labels for :py:meth:`Dataset.plot.scatter` (:issue:`4126`).
  By `Peter Hausamann <https://github.com/phausamann>`_.
- Fix ``dask.optimize`` on ``DataArray`` producing an invalid Dask task graph (:issue:`3698`)
  By `Tom Augspurger <https://github.com/TomAugspurger>`_
- Fix ``pip install .`` when no ``.git`` directory exists; namely when the xarray source
  directory has been rsync'ed by PyCharm Professional for a remote deployment over SSH.
  By `Guido Imperiale <https://github.com/crusaderky>`_
- Preserve dimension and coordinate order during :py:func:`xarray.concat` (:issue:`2811`, :issue:`4072`, :pull:`4419`).
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Avoid relying on :py:class:`set` objects for the ordering of the coordinates (:pull:`4409`)
  By `Justus Magin <https://github.com/keewis>`_.

Documentation
~~~~~~~~~~~~~

- Update the docstring of :py:meth:`DataArray.copy` to remove incorrect mention of 'dataset' (:issue:`3606`)
  By `Sander van Rijn <https://github.com/sjvrijn>`_.
- Removed skipna argument from :py:meth:`DataArray.count`, :py:meth:`DataArray.any`, :py:meth:`DataArray.all`. (:issue:`755`)
  By `Sander van Rijn <https://github.com/sjvrijn>`_
- Update the contributing guide to use merges instead of rebasing and state
  that we squash-merge. (:pull:`4355`). By `Justus Magin <https://github.com/keewis>`_.
- Make sure the examples from the docstrings actually work (:pull:`4408`).
  By `Justus Magin <https://github.com/keewis>`_.
- Updated Vectorized Indexing to a clearer example.
  By `Maximilian Roos <https://github.com/max-sixty>`_

Internal Changes
~~~~~~~~~~~~~~~~

- Fixed all doctests and enabled their running in CI.
  By `Justus Magin <https://github.com/keewis>`_.
- Relaxed the :ref:`mindeps_policy` to support:

  - all versions of setuptools released in the last 42 months (but no older than 38.4)
  - all versions of dask and dask.distributed released in the last 12 months (but no
    older than 2.9)
  - all versions of other packages released in the last 12 months

  All are up from 6 months (:issue:`4295`)
  `Guido Imperiale <https://github.com/crusaderky>`_.
- Use :py:func:`dask.array.apply_gufunc <dask.array.gufunc.apply_gufunc>` instead of
  :py:func:`dask.array.blockwise` in :py:func:`xarray.apply_ufunc` when using
  ``dask='parallelized'``. (:pull:`4060`, :pull:`4391`, :pull:`4392`)
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Align ``mypy`` versions to ``0.782`` across ``requirements`` and
  ``.pre-commit-config.yml`` files. (:pull:`4390`)
  By `Maximilian Roos <https://github.com/max-sixty>`_
- Only load resource files when running inside a Jupyter Notebook
  (:issue:`4294`) By `Guido Imperiale <https://github.com/crusaderky>`_
- Silenced most ``numpy`` warnings such as ``Mean of empty slice``. (:pull:`4369`)
  By `Maximilian Roos <https://github.com/max-sixty>`_
- Enable type checking for :py:func:`concat` (:issue:`4238`)
  By `Mathias Hauser <https://github.com/mathause>`_.
- Updated plot functions for matplotlib version 3.3 and silenced warnings in the
  plot tests (:pull:`4365`). By `Mathias Hauser <https://github.com/mathause>`_.
- Versions in ``pre-commit.yaml`` are now pinned, to reduce the chances of
  conflicting versions. (:pull:`4388`)
  By `Maximilian Roos <https://github.com/max-sixty>`_



.. _whats-new.0.16.0:

v0.16.0 (2020-07-11)
---------------------

This release adds ``xarray.cov`` & ``xarray.corr`` for covariance & correlation
respectively; the ``idxmax`` & ``idxmin`` methods, the ``polyfit`` method &
``xarray.polyval`` for fitting polynomials, as well as a number of documentation
improvements, other features, and bug fixes. Many thanks to all 44 contributors
who contributed to this release:

Akio Taniguchi, Andrew Williams, Aurélien Ponte, Benoit Bovy, Dave Cole, David
Brochart, Deepak Cherian, Elliott Sales de Andrade, Etienne Combrisson, Hossein
Madadi, Huite, Joe Hamman, Kai Mühlbauer, Keisuke Fujii, Maik Riechert, Marek
Jacob, Mathias Hauser, Matthieu Ancellin, Maximilian Roos, Noah D Brenowitz,
Oriol Abril, Pascal Bourgault, Phillip Butcher, Prajjwal Nijhara, Ray Bell, Ryan
Abernathey, Ryan May, Spencer Clark, Spencer Hill, Srijan Saurav, Stephan Hoyer,
Taher Chegini, Todd, Tom Nicholas, Yohai Bar Sinai, Yunus Sevinchan,
arabidopsis, aurghs, clausmichele, dmey, johnomotani, keewis, raphael dussin,
risebell

Breaking changes
~~~~~~~~~~~~~~~~

- Minimum supported versions for the following packages have changed: ``dask >=2.9``,
  ``distributed>=2.9``.
  By `Deepak Cherian <https://github.com/dcherian>`_
- ``groupby`` operations will restore coord dimension order. Pass ``restore_coord_dims=False``
  to revert to previous behavior.
- :meth:`DataArray.transpose` will now transpose coordinates by default.
  Pass ``transpose_coords=False`` to revert to previous behaviour.
  By `Maximilian Roos <https://github.com/max-sixty>`_
- Alternate draw styles for :py:meth:`plot.step` must be passed using the
  ``drawstyle`` (or ``ds``) keyword argument, instead of the ``linestyle`` (or
  ``ls``) keyword argument, in line with the `upstream change in Matplotlib
  <https://matplotlib.org/api/prev_api_changes/api_changes_3.1.0.html#passing-a-line2d-s-drawstyle-together-with-the-linestyle-is-deprecated>`_.
  (:pull:`3274`)
  By `Elliott Sales de Andrade <https://github.com/QuLogic>`_
- The old ``auto_combine`` function has now been removed in
  favour of the :py:func:`combine_by_coords` and
  :py:func:`combine_nested` functions. This also means that
  the default behaviour of :py:func:`open_mfdataset` has changed to use
  ``combine='by_coords'`` as the default argument value. (:issue:`2616`, :pull:`3926`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- The ``DataArray`` and ``Variable`` HTML reprs now expand the data section by
  default (:issue:`4176`)
  By `Stephan Hoyer <https://github.com/shoyer>`_.

New Features
~~~~~~~~~~~~
- :py:meth:`DataArray.argmin` and :py:meth:`DataArray.argmax` now support
  sequences of 'dim' arguments, and if a sequence is passed return a dict
  (which can be passed to :py:meth:`DataArray.isel` to get the value of the minimum) of
  the indices for each dimension of the minimum or maximum of a DataArray.
  (:pull:`3936`)
  By `John Omotani <https://github.com/johnomotani>`_, thanks to `Keisuke Fujii
  <https://github.com/fujiisoup>`_ for work in :pull:`1469`.
- Added :py:func:`xarray.cov` and :py:func:`xarray.corr` (:issue:`3784`, :pull:`3550`, :pull:`4089`).
  By `Andrew Williams <https://github.com/AndrewWilliams3142>`_ and `Robin Beer <https://github.com/r-beer>`_.
- Implement :py:meth:`DataArray.idxmax`, :py:meth:`DataArray.idxmin`,
  :py:meth:`Dataset.idxmax`, :py:meth:`Dataset.idxmin`.  (:issue:`60`, :pull:`3871`)
  By `Todd Jennings <https://github.com/toddrjen>`_
- Added :py:meth:`DataArray.polyfit` and :py:func:`xarray.polyval` for fitting
  polynomials. (:issue:`3349`, :pull:`3733`, :pull:`4099`)
  By `Pascal Bourgault <https://github.com/aulemahal>`_.
- Added :py:meth:`xarray.infer_freq` for extending frequency inferring to CFTime indexes and data (:pull:`4033`).
  By `Pascal Bourgault <https://github.com/aulemahal>`_.
- ``chunks='auto'`` is now supported in the ``chunks`` argument of
  :py:meth:`Dataset.chunk`. (:issue:`4055`)
  By `Andrew Williams <https://github.com/AndrewWilliams3142>`_
- Control over attributes of result in :py:func:`merge`, :py:func:`concat`,
  :py:func:`combine_by_coords` and :py:func:`combine_nested` using
  combine_attrs keyword argument. (:issue:`3865`, :pull:`3877`)
  By `John Omotani <https://github.com/johnomotani>`_
- ``missing_dims`` argument to :py:meth:`Dataset.isel`,
  :py:meth:`DataArray.isel` and :py:meth:`Variable.isel` to allow replacing
  the exception when a dimension passed to ``isel`` is not present with a
  warning, or just ignore the dimension. (:issue:`3866`, :pull:`3923`)
  By `John Omotani <https://github.com/johnomotani>`_
- Support dask handling for :py:meth:`DataArray.idxmax`, :py:meth:`DataArray.idxmin`,
  :py:meth:`Dataset.idxmax`, :py:meth:`Dataset.idxmin`.  (:pull:`3922`, :pull:`4135`)
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_ and `Pascal Bourgault <https://github.com/aulemahal>`_.
- More support for unit aware arrays with pint (:pull:`3643`, :pull:`3975`, :pull:`4163`)
  By `Justus Magin <https://github.com/keewis>`_.
- Support overriding existing variables in ``to_zarr()`` with ``mode='a'`` even
  without ``append_dim``, as long as dimension sizes do not change.
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Allow plotting of boolean arrays. (:pull:`3766`)
  By `Marek Jacob <https://github.com/MeraX>`_
- Enable using MultiIndex levels as coordinates in 1D and 2D plots (:issue:`3927`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- A ``days_in_month`` accessor for :py:class:`xarray.CFTimeIndex`, analogous to
  the ``days_in_month`` accessor for a :py:class:`pandas.DatetimeIndex`, which
  returns the days in the month each datetime in the index.  Now days in month
  weights for both standard and non-standard calendars can be obtained using
  the :py:class:`~core.accessor_dt.DatetimeAccessor` (:pull:`3935`).  This
  feature requires cftime version 1.1.0 or greater.  By
  `Spencer Clark <https://github.com/spencerkclark>`_.
- For the netCDF3 backend, added dtype coercions for unsigned integer types.
  (:issue:`4014`, :pull:`4018`)
  By `Yunus Sevinchan <https://github.com/blsqr>`_
- :py:meth:`map_blocks` now accepts a ``template`` kwarg. This allows use cases
  where the result of a computation could not be inferred automatically.
  By `Deepak Cherian <https://github.com/dcherian>`_
- :py:meth:`map_blocks` can now handle dask-backed xarray objects in ``args``. (:pull:`3818`)
  By `Deepak Cherian <https://github.com/dcherian>`_
- Add keyword ``decode_timedelta`` to :py:func:`xarray.open_dataset`,
  (:py:func:`xarray.open_dataarray`, :py:func:`xarray.open_dataarray`,
  :py:func:`xarray.decode_cf`) that allows to disable/enable the decoding of timedeltas
  independently of time decoding (:issue:`1621`)
  `Aureliana Barghini <https://github.com/aurghs>`_

Enhancements
~~~~~~~~~~~~
- Performance improvement of :py:meth:`DataArray.interp` and :py:func:`Dataset.interp`
  We performs independent interpolation sequentially rather than interpolating in
  one large multidimensional space. (:issue:`2223`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- :py:meth:`DataArray.interp` now support interpolations over chunked dimensions (:pull:`4155`). By `Alexandre Poux <https://github.com/pums974>`_.
- Major performance improvement for :py:meth:`Dataset.from_dataframe` when the
  dataframe has a MultiIndex (:pull:`4184`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
  - :py:meth:`DataArray.reset_index` and :py:meth:`Dataset.reset_index` now keep
  coordinate attributes (:pull:`4103`). By `Oriol Abril <https://github.com/OriolAbril>`_.
- Axes kwargs such as ``facecolor`` can now be passed to :py:meth:`DataArray.plot` in ``subplot_kws``.
  This works for both single axes plots and FacetGrid plots.
  By `Raphael Dussin <https://github.com/raphaeldussin>`_.
- Array items with long string reprs are now limited to a
  reasonable width (:pull:`3900`)
  By `Maximilian Roos <https://github.com/max-sixty>`_
- Large arrays whose numpy reprs would have greater than 40 lines are now
  limited to a reasonable length.
  (:pull:`3905`)
  By `Maximilian Roos <https://github.com/max-sixty>`_

Bug fixes
~~~~~~~~~
- Fix errors combining attrs in :py:func:`open_mfdataset` (:issue:`4009`, :pull:`4173`)
  By `John Omotani <https://github.com/johnomotani>`_
- If groupby receives a ``DataArray`` with name=None, assign a default name (:issue:`158`)
  By `Phil Butcher <https://github.com/pjbutcher>`_.
- Support dark mode in VS code (:issue:`4024`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.
- Fix bug when converting multiindexed pandas objects to sparse xarray objects. (:issue:`4019`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- ``ValueError`` is raised when ``fill_value`` is not a scalar in :py:meth:`full_like`. (:issue:`3977`)
  By `Huite Bootsma <https://github.com/huite>`_.
- Fix wrong order in converting a ``pd.Series`` with a MultiIndex to ``DataArray``.
  (:issue:`3951`, :issue:`4186`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_ and `Stephan Hoyer <https://github.com/shoyer>`_.
- Fix renaming of coords when one or more stacked coords is not in
  sorted order during stack+groupby+apply operations. (:issue:`3287`,
  :pull:`3906`) By `Spencer Hill <https://github.com/spencerahill>`_
- Fix a regression where deleting a coordinate from a copied :py:class:`DataArray`
  can affect the original :py:class:`DataArray`.  (:issue:`3899`, :pull:`3871`)
  By `Todd Jennings <https://github.com/toddrjen>`_
- Fix :py:class:`~xarray.plot.FacetGrid` plots with a single contour. (:issue:`3569`, :pull:`3915`).
  By `Deepak Cherian <https://github.com/dcherian>`_
- Use divergent colormap if ``levels`` spans 0. (:issue:`3524`)
  By `Deepak Cherian <https://github.com/dcherian>`_
- Fix :py:class:`~xarray.plot.FacetGrid` when ``vmin == vmax``. (:issue:`3734`)
  By `Deepak Cherian <https://github.com/dcherian>`_
- Fix plotting when ``levels`` is a scalar and ``norm`` is provided. (:issue:`3735`)
  By `Deepak Cherian <https://github.com/dcherian>`_
- Fix bug where plotting line plots with 2D coordinates depended on dimension
  order. (:issue:`3933`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fix ``RasterioDeprecationWarning`` when using a ``vrt`` in ``open_rasterio``. (:issue:`3964`)
  By `Taher Chegini <https://github.com/cheginit>`_.
- Fix ``AttributeError`` on displaying a :py:class:`Variable`
  in a notebook context. (:issue:`3972`, :pull:`3973`)
  By `Ian Castleden <https://github.com/arabidopsis>`_.
- Fix bug causing :py:meth:`DataArray.interpolate_na` to always drop attributes,
  and added ``keep_attrs`` argument. (:issue:`3968`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fix bug in time parsing failing to fall back to cftime. This was causing time
  variables with a time unit of ``'msecs'`` to fail to parse. (:pull:`3998`)
  By `Ryan May <https://github.com/dopplershift>`_.
- Fix weighted mean when passing boolean weights (:issue:`4074`).
  By `Mathias Hauser <https://github.com/mathause>`_.
- Fix html repr in untrusted notebooks: fallback to plain text repr. (:pull:`4053`)
  By `Benoit Bovy <https://github.com/benbovy>`_.
- Fix :py:meth:`DataArray.to_unstacked_dataset` for single-dimension variables. (:issue:`4049`)
  By `Deepak Cherian <https://github.com/dcherian>`_
- Fix :py:func:`open_rasterio` for ``WarpedVRT`` with specified ``src_crs``. (:pull:`4104`)
  By `Dave Cole <https://github.com/dtpc>`_.

Documentation
~~~~~~~~~~~~~
- update the docstring of :py:meth:`DataArray.assign_coords` : clarify how to
  add a new coordinate to an existing dimension and illustrative example
  (:issue:`3952`, :pull:`3958`) By
  `Etienne Combrisson <https://github.com/EtienneCmb>`_.
- update the docstring of :py:meth:`Dataset.diff` and
  :py:meth:`DataArray.diff` so it does document the ``dim``
  parameter as required. (:issue:`1040`, :pull:`3909`)
  By `Justus Magin <https://github.com/keewis>`_.
- Updated :doc:`Calculating Seasonal Averages from Timeseries of Monthly Means
  <examples/monthly-means>` example notebook to take advantage of the new
  ``days_in_month`` accessor for :py:class:`xarray.CFTimeIndex`
  (:pull:`3935`). By `Spencer Clark <https://github.com/spencerkclark>`_.
- Updated the list of current core developers. (:issue:`3892`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Add example for multi-dimensional extrapolation and note different behavior
  of ``kwargs`` in :py:meth:`Dataset.interp` and :py:meth:`DataArray.interp`
  for 1-d and n-d interpolation (:pull:`3956`).
  By `Matthias Riße <https://github.com/risebell>`_.
- Apply ``black`` to all the code in the documentation (:pull:`4012`)
  By `Justus Magin <https://github.com/keewis>`_.
- Narrative documentation now describes :py:meth:`map_blocks`: :ref:`dask.automatic-parallelization`.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Document ``.plot``, ``.dt``, ``.str`` accessors the way they are called. (:issue:`3625`, :pull:`3988`)
  By `Justus Magin <https://github.com/keewis>`_.
- Add documentation for the parameters and return values of :py:meth:`DataArray.sel`.
  By `Justus Magin <https://github.com/keewis>`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Raise more informative error messages for chunk size conflicts when writing to zarr files.
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Run the ``isort`` pre-commit hook only on python source files
  and update the ``flake8`` version. (:issue:`3750`, :pull:`3711`)
  By `Justus Magin <https://github.com/keewis>`_.
- Add `blackdoc <https://blackdoc.readthedocs.io>`_ to the list of
  checkers for development. (:pull:`4177`)
  By `Justus Magin <https://github.com/keewis>`_.
- Add a CI job that runs the tests with every optional dependency
  except ``dask``. (:issue:`3794`, :pull:`3919`)
  By `Justus Magin <https://github.com/keewis>`_.
- Use ``async`` / ``await`` for the asynchronous distributed
  tests. (:issue:`3987`, :pull:`3989`)
  By `Justus Magin <https://github.com/keewis>`_.
- Various internal code clean-ups (:pull:`4026`,  :pull:`4038`).
  By `Prajjwal Nijhara <https://github.com/pnijhara>`_.

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
  By `Mathias Hauser <https://github.com/mathause>`_.
- The new jupyter notebook repr (``Dataset._repr_html_`` and
  ``DataArray._repr_html_``) (introduced in 0.14.1) is now on by default. To
  disable, use ``xarray.set_options(display_style="text")``.
  By `Julia Signell <https://github.com/jsignell>`_.
- Added support for :py:class:`pandas.DatetimeIndex`-style rounding of
  ``cftime.datetime`` objects directly via a :py:class:`CFTimeIndex` or via the
  :py:class:`~core.accessor_dt.DatetimeAccessor`.
  By `Spencer Clark <https://github.com/spencerkclark>`_
- Support new h5netcdf backend keyword ``phony_dims`` (available from h5netcdf
  v0.8.0 for :py:class:`~xarray.backends.H5NetCDFStore`.
  By `Kai Mühlbauer <https://github.com/kmuehlbauer>`_.
- Add partial support for unit aware arrays with pint. (:pull:`3706`, :pull:`3611`)
  By `Justus Magin <https://github.com/keewis>`_.
- :py:meth:`Dataset.groupby` and :py:meth:`DataArray.groupby` now raise a
  ``TypeError`` on multiple string arguments. Receiving multiple string arguments
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
- Add a diff summary for ``testing.assert_allclose``. (:issue:`3617`, :pull:`3847`)
  By `Justus Magin <https://github.com/keewis>`_.

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
- ``coarsen`` and ``rolling`` now respect ``xr.set_options(keep_attrs=True)``
  to preserve attributes. :py:meth:`Dataset.coarsen` accepts a keyword
  argument ``keep_attrs`` to change this setting. (:issue:`3376`,
  :pull:`3801`) By `Andrew Thomas <https://github.com/amcnicho>`_.
- Delete associated indexes when deleting coordinate variables. (:issue:`3746`).
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Fix :py:meth:`Dataset.to_zarr` when using ``append_dim`` and ``group``
  simultaneously. (:issue:`3170`). By `Matthias Meyer <https://github.com/niowniow>`_.
- Fix html repr on :py:class:`Dataset` with non-string keys (:pull:`3807`).
  By `Maximilian Roos <https://github.com/max-sixty>`_.

Documentation
~~~~~~~~~~~~~

- Fix documentation of :py:class:`DataArray` removing the deprecated mention
  that when omitted, ``dims`` are inferred from a ``coords``-dict. (:pull:`3821`)
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
- Remove conversion to ``pandas.Panel``, given its removal in pandas
  in favor of xarray's objects.
  By `Maximilian Roos <https://github.com/max-sixty>`_

.. _whats-new.0.15.0:


v0.15.0 (30 Jan 2020)
---------------------

This release brings many improvements to xarray's documentation: our examples are now binderized notebooks (`click here <https://mybinder.org/v2/gh/pydata/xarray/main?urlpath=lab/tree/doc/examples/weather-data.ipynb>`_)
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
- Fix leap year condition in `monthly means example <https://docs.xarray.dev/en/stable/examples/monthly-means.html>`_.
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
  Python <=3.5, most uses of ``OrderedDict`` in xarray were no longer necessary. We
  have removed the internal use of the ``OrderedDict`` in favor of Python's builtin
  ``dict`` object which is now ordered itself. This change will be most obvious when
  interacting with the ``attrs`` property on Dataset and DataArray objects.
  (:issue:`3380`, :pull:`3389`).  By `Joe Hamman <https://github.com/jhamman>`_.

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
  (:pull:`3331`, :pull:`3331`). By `Justus Magin <https://github.com/keewis>`_.
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
  This requires ``sparse>=0.8.0``. By `Nezar Abdennur <https://github.com/nvictus>`_
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
- Improved error handling and documentation for ``.expand_dims()``
  read-only view.
- Fix tests for big-endian systems (:issue:`3125`).
  By `Graham Inggs <https://github.com/ginggs>`_.
- XFAIL several tests which are expected to fail on ARM systems
  due to a ``datetime`` issue in NumPy (:issue:`2334`).
  By `Graham Inggs <https://github.com/ginggs>`_.
- Fix KeyError that arises when using .sel method with float values
  different from coords float type (:issue:`3137`).
  By `Hasan Ahmad <https://github.com/HasanAhmadQ7>`_.
- Fixed bug in ``combine_by_coords()`` causing a ``ValueError`` if the input had
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
  ``keep_attrs`` was True (:issue:`3304`). By `David Huard <https://github.com/huard>`_.

Documentation
~~~~~~~~~~~~~

- Created a `PR checklist <https://docs.xarray.dev/en/stable/contributing.html#pr-checklist>`_
  as a quick reference for tasks before creating a new PR
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

  The older function ``auto_combine`` has been deprecated,
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
- Increased support for ``missing_value`` (:issue:`2871`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Removed usages of ``pytest.config``, which is deprecated (:issue:`2988`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.
- Fixed performance issues with cftime installed (:issue:`3000`)
  By `0x0L <https://github.com/0x0L>`_.
- Replace incorrect usages of ``message`` in pytest assertions
  with ``match`` (:issue:`3011`)
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
  :py:meth:`~xarray.DataArray.integrate` methods. See :ref:`compute.coarsen`
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
  See :ref:`compute.coarsen` for details.
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

  - :issue:`Xarray Github issue discussing dropping Python 2 <1829>`
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
  ``loffset`` kwarg just like pandas.
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
    Similarly, calling ``len`` and ``bool`` on a ``Dataset`` now
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
  - The ``inplace`` kwarg of a number of ``DataArray`` and ``Dataset`` methods is being
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
  coordinate variables as input. ``hue`` must be a dimension name in this case.
  (:issue:`2407`)
  By `Deepak Cherian <https://github.com/dcherian>`_.
- Added support for Python 3.7. (:issue:`2271`).
  By `Joe Hamman <https://github.com/jhamman>`_.
- Added support for plotting data with ``pandas.Interval`` coordinates, such as those
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
  :py:meth:`xarray.tutorial.load_dataset` calls ``Dataset.load()`` prior
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
  with ``to_zarr`` and ``open_zarr`` (:issue:`2300`).
  By `Lily Wang <https://github.com/lilyminium>`_.

.. _whats-new.0.10.9:

v0.10.9 (21 September 2018)
---------------------------

This minor release contains a number of backwards compatible enhancements.

Announcements of note:

- Xarray is now a NumFOCUS fiscally sponsored project! Read
  `the announcement <https://numfocus.org/blog/xarray-joins-numfocus-sponsored-projects>`_
  for more details.
- We have a new :doc:`roadmap` that outlines our future development plans.

- ``Dataset.apply`` now properly documents the way ``func`` is called.
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
  ``xscale, yscale, xlim, ylim, xticks, yticks`` just like pandas. Also ``xincrease=False, yincrease=False`` now use matplotlib's axis inverting methods instead of setting limits.
  By `Deepak Cherian <https://github.com/dcherian>`_. (:issue:`2224`)

- DataArray coordinates and Dataset coordinates and data variables are
  now displayed as ``a b ... y z`` rather than ``a b c d ...``.
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

  - pandas: 0.18 -> 0.19
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
  ``effective_get`` in dask (:issue:`2238`).
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
  ``io.PseudoNetCDF`` for more details.
  By `Barron Henderson <https://github.com/barronh>`_.

- The :py:class:`Dataset` constructor now aligns :py:class:`DataArray`
  arguments in ``data_vars`` to indexes set explicitly in ``coords``,
  where previously an error would be raised.
  (:issue:`674`)
  By `Maximilian Roos <https://github.com/max-sixty>`_.

- :py:meth:`~DataArray.sel`, :py:meth:`~DataArray.isel` & :py:meth:`~DataArray.reindex`,
  (and their :py:class:`Dataset` counterparts) now support supplying a ``dict``
  as a first argument, as an alternative to the existing approach
  of supplying ``kwargs``. This allows for more robust behavior
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

- New FAQ entry, :ref:`ecosystem`.
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
- Fixed a bug where ``keep_attrs=True`` flag was neglected if
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
- When calling ``xr.auto_combine()`` or ``xr.open_mfdataset()`` with a ``concat_dim``,
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

.. _ufunc methods: https://numpy.org/doc/stable/reference/ufuncs.html#methods

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

      ds = xr.Dataset({"a": 1})
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
- Fix kwarg ``colors`` clashing with auto-inferred ``cmap`` (:issue:`1461`)
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
- New entry ``Why don’t aggregations return Python scalars?`` in the
  :doc:`getting-started-guide/faq` (:issue:`1726`).
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

      da = xr.DataArray(np.array([True, False, np.nan], dtype=object), dims="x")
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
- Compatibility fixes to plotting module for NumPy 1.14 and pandas 0.22
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
  ``parse_coordinates`` kwarg has been added to :py:func:`~open_rasterio`
  (set to ``True`` per default).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The colors of discrete colormaps are now the same regardless if ``seaborn``
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
  fully supported for data with arbitrary dimensions as is both downsampling
  and upsampling (including linear, quadratic, cubic, and spline interpolation).

  Old syntax:

  .. ipython::
    :verbatim:

    In [1]: ds.resample("24H", dim="time", how="max")
    Out[1]:
    <xarray.Dataset>
    [...]

  New syntax:

  .. ipython::
    :verbatim:

    In [1]: ds.resample(time="24H").max()
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

    In [2]: arr = xr.DataArray([[1, 2, 3], [4, 5, 6]], dims=("x", "y"))

    In [3]: xr.where(arr % 2, "even", "odd")
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
  floating point values) by passing the encoding ``{'_FillValue': None}``
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
  (e.g. ``NaN``). Xarray now behaves similarly to pandas in its treatment of
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

- Fix bug when using ``pytest`` class decorators to skipping certain unittests.
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
  ``DataArray.set_index()`` introduced by pandas 0.21.0. Setting a new
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

- A new `gallery <https://docs.xarray.dev/en/latest/auto_gallery/index.html>`_
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

    In [1]: xr.Dataset({"foo": (("x", "y"), [[1, 2]])})
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
  formats would raise an error since netCDF does not have a ``bool`` datatype.
  This feature reads/writes a ``dtype`` attribute to boolean variables in netCDF
  files. By `Joe Hamman <https://github.com/jhamman>`_.

- 2D plotting methods now have two new keywords (``cbar_ax`` and ``cbar_kwargs``),
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

- Fix a bug where ``xarray.ufuncs`` that take two arguments would incorrectly
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

- Fixed ``dim`` argument for ``isel_points``/``sel_points`` when a ``pandas.Index`` is
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

    In [1]: import xarray as xr
       ...: import numpy as np

    In [2]: arr = xr.DataArray(np.arange(0, 7.5, 0.5).reshape(3, 5), dims=("x", "y"))

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
  ``.transpose``. This behavior was causing ``pandas.PeriodIndex`` dimensions
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

- https://docs.xarray.dev
- https://github.com/pydata/xarray/

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

    In [2]: xray.DataArray([4, 5, 6], dims="x", name="x")
    Out[2]:
    <xray.DataArray 'x' (x: 3)>
    array([4, 5, 6])
    Coordinates:
      * x        (x) int64 4 5 6

  and the new behavior (compare the values of the ``x`` coordinate):

  .. ipython::
    :verbatim:

    In [2]: xray.DataArray([4, 5, 6], dims="x", name="x")
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

    In [7]: df = pd.DataFrame({"foo": range(3), "x": ["a", "b", "b"], "y": [0, 0, 1]})

    In [8]: s = df.set_index(["x", "y"])["foo"]

    In [12]: arr = xray.DataArray(s, dims="z")

    In [13]: arr
    Out[13]:
    <xray.DataArray 'foo' (z: 3)>
    array([0, 1, 2])
    Coordinates:
      * z        (z) object ('a', 0) ('b', 0) ('b', 1)

    In [19]: arr.indexes["z"]
    Out[19]:
    MultiIndex(levels=[[u'a', u'b'], [0, 1]],
               labels=[[0, 1, 1], [0, 0, 1]],
               names=[u'x', u'y'])

    In [14]: arr.unstack("z")
    Out[14]:
    <xray.DataArray 'foo' (x: 2, y: 2)>
    array([[  0.,  nan],
           [  1.,   2.]])
    Coordinates:
      * x        (x) object 'a' 'b'
      * y        (y) int64 0 1

    In [26]: arr.unstack("z").stack(z=("x", "y"))
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

- Support for reading GRIB, HDF4 and other file formats via PyNIO_.
- Better error message when a variable is supplied with the same name as
  one of its dimensions.
- Plotting: more control on colormap parameters (:issue:`642`). ``vmin`` and
  ``vmax`` will not be silently ignored anymore. Setting ``center=False``
  prevents automatic selection of a divergent colormap.
- New ``xray.Dataset.shift`` and ``xray.Dataset.roll`` methods
  for shifting/rotating datasets or arrays along a dimension:

  .. ipython:: python
      :okwarning:

      array = xray.DataArray([5, 6, 7, 8], dims="x")
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

      a = xray.DataArray([1, 2, 3], dims="x")
      b = xray.DataArray([5, 6], dims="y")
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

    In [5]: array = xray.DataArray([1, 2, 3], dims="x")

    In [6]: array.reindex(x=[0.9, 1.5], method="nearest", tolerance=0.2)
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

    In [1]: da = xray.DataArray(
       ...:     np.arange(56).reshape((7, 8)),
       ...:     coords={"x": list("abcdefg"), "y": 10 * np.arange(8)},
       ...:     dims=["x", "y"],
       ...: )

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
    In [3]: da.isel_points(x=[0, 1, 6], y=[0, 1, 0], dim="points")
    Out[3]:
    <xray.DataArray (points: 3)>
    array([ 0,  9, 48])
    Coordinates:
        y        (points) int64 0 10 0
        x        (points) |S1 'a' 'b' 'g'
      * points   (points) int64 0 1 2

    # or equivalently by label
    In [9]: da.sel_points(x=["a", "b", "g"], y=[0, 10, 0], dim="points")
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

      ds = xray.Dataset(coords={"x": range(100), "y": range(100)})
      ds["distance"] = np.sqrt(ds.x**2 + ds.y**2)

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
- Added support for serializing arrays of complex numbers with ``engine='h5netcdf'``.
- The new ``xray.save_mfdataset`` function allows for saving multiple
  datasets to disk simultaneously. This is useful when processing large datasets
  with dask.array. For example, to save a dataset too big to fit into memory
  to one file per year, we could write:

  .. ipython::
    :verbatim:

    In [1]: years, datasets = zip(*ds.groupby("time.year"))

    In [2]: paths = ["%s.nc" % y for y in years]

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
computing (data that doesn't fit into memory) with :doc:`user-guide/dask`. This includes a new
top-level function ``xray.open_mfdataset`` that makes it easy to open
a collection of netCDF (using dask) as a single ``xray.Dataset`` object. For
more on dask, read the `blog post introducing xray + dask`_ and the new
documentation section :doc:`user-guide/dask`.

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

    In [1]: ds = xray.Dataset({"x": 0})

    In [2]: xray.concat([ds, ds], dim="y")
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

      ds = xray.Dataset({"x": 0})

  .. ipython:: python

      xray.concat([ds, ds], dim="y")

  To obtain the old behavior, supply the argument ``concat_over=[]``.

Enhancements
~~~~~~~~~~~~

- New ``xray.Dataset.to_dataarray`` and enhanced
  ``xray.DataArray.to_dataset`` methods make it easy to switch back
  and forth between arrays and datasets:

  .. ipython:: python

      ds = xray.Dataset(
          {"a": 1, "b": ("x", [1, 2, 3])},
          coords={"c": 42},
          attrs={"Conventions": "None"},
      )
      ds.to_dataarray()
      ds.to_dataarray().to_dataset(dim="variable")

- New ``xray.Dataset.fillna`` method to fill missing values, modeled
  off the pandas method of the same name:

  .. ipython:: python

      array = xray.DataArray([np.nan, 1, np.nan, 3], dims="x")
      array.fillna(0)

  ``fillna`` works on both ``Dataset`` and ``DataArray`` objects, and uses
  index based alignment and broadcasting like standard binary operations. It
  also can be applied by group, as illustrated in
  :ref:`/examples/weather-data.ipynb#Fill-missing-values-with-climatology`.
- New ``xray.Dataset.assign`` and ``xray.Dataset.assign_coords``
  methods patterned off the new :py:meth:`DataFrame.assign <pandas.DataFrame.assign>`
  method in pandas:

  .. ipython:: python

      ds = xray.Dataset({"y": ("x", [1, 2, 3])})
      ds.assign(z=lambda ds: ds.y**2)
      ds.assign_coords(z=("x", ["a", "b", "c"]))

  These methods return a new Dataset (or DataArray) with updated data or
  coordinate variables.
- ``xray.Dataset.sel`` now supports the ``method`` parameter, which works
  like the parameter of the same name on ``xray.Dataset.reindex``. It
  provides a simple interface for doing nearest-neighbor interpolation:

  .. use verbatim because I can't seem to install pandas 0.16.1 on RTD :(

  .. ipython::
      :verbatim:

      In [12]: ds.sel(x=1.1, method="nearest")
      Out[12]:
      <xray.Dataset>
      Dimensions:  ()
      Coordinates:
          x        int64 1
      Data variables:
          y        int64 2

      In [13]: ds.sel(x=[1.1, 2.1], method="pad")
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

      ds = xray.Dataset({"x": np.arange(1000)})
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

      time = pd.date_range("2000-01-01", freq="6H", periods=10)
      array = xray.DataArray(np.arange(10), [("time", time)])
      array.resample("1D", dim="time")

  You can specify how to do the resampling with the ``how`` argument and other
  options such as ``closed`` and ``label`` let you control labeling:

  .. ipython:: python
      :verbatim:

      array.resample("1D", dim="time", how="sum", label="right")

  If the desired temporal resolution is higher than the original data
  (upsampling), xray will insert missing values:

  .. ipython:: python
      :verbatim:

      array.resample("3H", "time")

- ``first`` and ``last`` methods on groupby objects let you take the first or
  last examples from each group along the grouped axis:

  .. ipython:: python
      :verbatim:

      array.groupby("time.day").first()

  These methods combine well with ``resample``:

  .. ipython:: python
      :verbatim:

      array.resample("1D", dim="time", how="first")


- ``xray.Dataset.swap_dims`` allows for easily swapping one dimension
  out for another:

  .. ipython:: python

      ds = xray.Dataset({"x": range(3), "y": ("x", list("abc"))})
      ds
      ds.swap_dims({"x": "y"})

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

      lhs = xray.DataArray([1, 2, 3], [("x", [0, 1, 2])])
      rhs = xray.DataArray([2, 3, 4], [("x", [1, 2, 3])])
      lhs + rhs

  :ref:`For dataset construction and merging<merge>`, we align based on the
  **union** of labels:

  .. ipython:: python

      xray.Dataset({"foo": lhs, "bar": rhs})

  :ref:`For update and __setitem__<update>`, we align based on the **original**
  object:

  .. ipython:: python

      lhs.coords["rhs"] = rhs
      lhs

- Aggregations like ``mean`` or ``median`` now skip missing values by default:

  .. ipython:: python

      xray.DataArray([1, 2, np.nan, 3]).mean()

  You can turn this behavior off by supplying the keyword argument
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

      a = xray.DataArray([1, 2], coords={"c": 0}, dims="x")
      b = xray.DataArray([1, 2], coords={"c": ("x", [0, 0])}, dims="x")
      (a + b).coords

  This functionality can be controlled through the ``compat`` option, which
  has also been added to the ``xray.Dataset`` constructor.
- Datetime shortcuts such as ``'time.month'`` now return a ``DataArray`` with
  the name ``'month'``, not ``'time.month'`` (:issue:`345`). This makes it
  easier to index the resulting arrays when they are used with ``groupby``:

  .. ipython:: python

      time = xray.DataArray(
          pd.date_range("2000-01-01", periods=365), dims="time", name="time"
      )
      counts = time.groupby("time.month").count()
      counts.sel(month=2)

  Previously, you would need to use something like
  ``counts.sel(**{'time.month': 2}})``, which is much more awkward.
- The ``season`` datetime shortcut now returns an array of string labels
  such ``'DJF'``:

  .. code-block:: ipython

      In[92]: ds = xray.Dataset({"t": pd.date_range("2000-01-01", periods=12, freq="M")})

      In[93]: ds["t.season"]
      Out[93]:
      <xarray.DataArray 'season' (t: 12)>
      array(['DJF', 'DJF', 'MAM', ..., 'SON', 'SON', 'DJF'], dtype='<U3')
      Coordinates:
        * t        (t) datetime64[ns] 2000-01-31 2000-02-29 ... 2000-11-30 2000-12-31

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

      data = xray.DataArray([1, 2, 3], [("x", range(3))])
      data.reindex(x=[0.5, 1, 1.5, 2, 2.5], method="pad")

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
      ds = xray.Dataset({"x": 0, "y": 1})
      ds.drop("x")

      # drop index labels
      arr = xray.DataArray([1, 2, 3], coords=[("x", list("abc"))])
      arr.drop(["a", "c"], dim="x")

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
  unparsable coordinate attributes (:issue:`308`).
- Fixed ``DataArray.loc`` indexing with ``...`` (:issue:`318`).
- Fixed an edge case that resulting in an error when reindexing
  multi-dimensional variables (:issue:`315`).
- Slicing with negative step sizes (:issue:`312`).
- Invalid conversion of string arrays to numeric dtype (:issue:`305`).
- Fixed ``repr()`` on dataset objects with non-standard dates (:issue:`347`).

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

.. _Dask: https://dask.org
.. _Blaze: https://blaze.pydata.org
.. _this blog post: https://matthewrocklin.com/blog/work/2015/02/13/Towards-OOC-Slicing-and-Stacking

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

      xray.Dataset({"t": [datetime(2000, 1, 1)]})

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

      ds = xray.Dataset({"tmin": ([], 25, {"units": "celsius"})})
      ds.tmin.units

  Tab-completion for these variables should work in editors such as IPython.
  However, setting variables or attributes in this fashion is not yet
  supported because there are some unresolved ambiguities (:issue:`300`).
- You can now use a dictionary for indexing with labeled dimensions. This
  provides a safe way to do assignment with labeled dimensions:

  .. ipython:: python

      array = xray.DataArray(np.zeros(5), dims=["x"])
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
