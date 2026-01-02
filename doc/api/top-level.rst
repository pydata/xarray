.. currentmodule:: xarray

Top-level functions
===================

Computation
-----------

.. note::

   For worked examples and advanced usage of ``apply_ufunc``, see the
   :doc:`User Guide on Computation </user-guide/computation>`, and the
   `apply_ufunc tutorial <https://tutorial.xarray.dev/advanced/apply_ufunc/apply_ufunc.html>`_.
 
.. autosummary::
   :toctree: ../generated/

   apply_ufunc
   cov
   corr
   cross
   dot
   map_blocks
   polyval
   unify_chunks
   where

Combining Data
--------------

.. autosummary::
   :toctree: ../generated/

   align
   broadcast
   concat
   merge
   combine_by_coords
   combine_nested

Creation
--------
.. autosummary::
   :toctree: ../generated/

   DataArray
   Dataset
   DataTree
   full_like
   zeros_like
   ones_like

Miscellaneous
-------------

.. autosummary::
   :toctree: ../generated/

   decode_cf
   infer_freq
   show_versions
   set_options
   get_options
