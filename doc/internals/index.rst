.. _internals:

xarray Internals
================

Xarray builds upon two of the foundational libraries of the scientific Python
stack, NumPy and pandas. It is written in pure Python (no C or Cython
extensions), which makes it easy to develop and extend. Instead, we push
compiled code to :ref:`optional dependencies<installing>`.


.. toctree::
   :maxdepth: 2
   :hidden:

   variable-objects
   duck-arrays-integration
   extending-xarray
   zarr-encoding-spec
   how-to-add-new-backend
