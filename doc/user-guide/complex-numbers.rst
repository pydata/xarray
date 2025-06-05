.. currentmodule:: xarray

.. _complex:

Complex Numbers
===============

.. jupyter-execute::
    :hide-code:

    import numpy as np
    import xarray as xr

Xarray leverages NumPy to seamlessly handle complex numbers in :py:class:`~xarray.DataArray` and :py:class:`~xarray.Dataset` objects.

In the examples below, we are using a DataArray named ``da`` with complex elements (of :math:`\mathbb{C}`):

.. jupyter-execute::

    data = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    da = xr.DataArray(
        data,
        dims=["x", "y"],
        coords={"x": ["a", "b"], "y": [1, 2]},
        name="complex_nums",
    )


Operations on Complex Data
--------------------------
You can access real and imaginary components using the ``.real`` and ``.imag`` attributes. Most NumPy universal functions (ufuncs) like :py:doc:`numpy.abs <numpy:reference/generated/numpy.absolute>` or :py:doc:`numpy.angle <numpy:reference/generated/numpy.angle>` work directly.

.. jupyter-execute::

    da.real

.. jupyter-execute::

    np.abs(da)

.. note::
    Like NumPy, ``.real`` and ``.imag`` typically return *views*, not copies, of the original data.


Reading and Writing Complex Data
--------------------------------

Writing complex data to NetCDF files (see :ref:`io.netcdf`) is supported via :py:meth:`~xarray.DataArray.to_netcdf` using specific backend engines that handle complex types:


.. tab:: h5netcdf

   This requires the `h5netcdf <https://h5netcdf.org>`_ library to be installed.

   .. jupyter-execute::

       # write the data to disk
       da.to_netcdf("complex_nums_h5.nc", engine="h5netcdf")
       # read the file back into memory
       ds_h5 = xr.open_dataset("complex_nums_h5.nc", engine="h5netcdf")
       # check the dtype
       ds_h5[da.name].dtype


.. tab:: netcdf4

   Requires the `netcdf4-python (>= 1.7.1) <https://github.com/Unidata/netcdf4-python>`_ library and you have to enable ``auto_complex=True``.

   .. jupyter-execute::

       # write the data to disk
       da.to_netcdf("complex_nums_nc4.nc", engine="netcdf4", auto_complex=True)
       # read the file back into memory
       ds_nc4 = xr.open_dataset(
           "complex_nums_nc4.nc", engine="netcdf4", auto_complex=True
       )
       # check the dtype
       ds_nc4[da.name].dtype


.. warning::
   The ``scipy`` engine only supports NetCDF V3 and does *not* support complex arrays; writing with ``engine="scipy"`` raises a ``TypeError``.


Alternative: Manual Handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If direct writing is not supported (e.g., targeting NetCDF3), you can manually
split the complex array into separate real and imaginary variables before saving:

.. jupyter-execute::

    # Write data to file
    ds_manual = xr.Dataset(
        {
            f"{da.name}_real": da.real,
            f"{da.name}_imag": da.imag,
        }
    )
    ds_manual.to_netcdf("complex_manual.nc", engine="scipy")  # Example

    # Read data from file
    ds = xr.open_dataset("complex_manual.nc", engine="scipy")
    reconstructed = ds[f"{da.name}_real"] + 1j * ds[f"{da.name}_imag"]

Recommendations
^^^^^^^^^^^^^^^

- Use ``engine="netcdf4"`` with ``auto_complex=True`` for full compliance and ease.
- Use ``h5netcdf`` for HDF5-based storage when interoperability with HDF5 is desired.
- For maximum legacy support (NetCDF3), manually handle real/imaginary components.

.. jupyter-execute::
    :hide-code:

    # Cleanup
    import os

    for f in ["complex_nums_nc4.nc", "complex_nums_h5.nc", "complex_manual.nc"]:
        if os.path.exists(f):
            os.remove(f)



See also
--------
- :ref:`io.netcdf` — full NetCDF I/O guide
- `NumPy complex numbers <https://numpy.org/doc/stable/user/basics.types.html#complex>`__
