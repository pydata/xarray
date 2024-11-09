.. _internals:

Xarray Internals
================

Xarray builds upon two of the foundational libraries of the scientific Python
stack, NumPy and pandas. It is written in pure Python (no C or Cython
extensions), which makes it easy to develop and extend. Instead, we push
compiled code to :ref:`optional dependencies<installing>`.

The pages in this section are intended for:

* Contributors to xarray who wish to better understand some of the internals,
* Developers from other fields who wish to extend xarray with domain-specific logic, perhaps to support a new scientific community of users,
* Developers of other packages who wish to interface xarray with their existing tools, e.g. by creating a backend for reading a new file format, or wrapping a custom array type.

.. toctree::
   :maxdepth: 2
   :hidden:

   internal-design
   interoperability
   duck-arrays-integration
   chunked-arrays
   extending-xarray
   how-to-add-new-backend
   how-to-create-custom-index
   zarr-encoding-spec
   time-coding
