.. currentmodule:: xarray

Top-level functions
===================

Computation
-----------

.. note::

   For worked examples and advanced usage of ``apply_ufunc``, see the
   :doc:`User Guide on Computation </user-guide/computation>`, 
   
   the tutorial examples at https://tutorial.xarray.dev/ 
   (see the Fundamentals → Computation section), 
   
   and the Gallery example
   :doc:`apply_ufunc_vectorize_1d </gallery/examples/apply_ufunc_vectorize_1d>`.



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
