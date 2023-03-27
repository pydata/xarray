.. currentmodule:: xarray

.. _options:

Configuration
=============

Xarray offers a small number of configuration options through :py:func:`set_options`. With these, you can

1. Control the ``repr``:

   - ``display_expand_attrs``
   - ``display_expand_coords``
   - ``display_expand_data``
   - ``display_expand_data_vars``
   - ``display_max_rows``
   - ``display_style``

2. Control behaviour during operations: ``arithmetic_join``, ``keep_attrs``, ``use_bottleneck``.
3. Control colormaps for plots:``cmap_divergent``, ``cmap_sequential``.
4. Aspects of file reading: ``file_cache_maxsize``, ``warn_on_unclosed_files``.


You can set these options either globally

::

  xr.set_options(arithmetic_join="exact")

or locally as a context manager:

::

   with xr.set_options(arithmetic_join="exact"):
       # do operation here
       pass
