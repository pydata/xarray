.. currentmodule:: xarray

.. _howdoi:

How do I ...
============

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - add variables from other datasets to my dataset
     - :py:meth:`Dataset.merge`
   * - add a new dimension and/or coordinate
     - :py:meth:`DataArray.expand_dims`, :py:meth:`Dataset.expand_dims`
   * - add a new coordinate variable
     - :py:meth:`DataArray.assign_coords`
   * - change a data variable to a coordinate variable
     - :py:meth:`Dataset.set_coords`
   * - change the order of dimensions
     - :py:meth:`DataArray.transpose`, :py:meth:`Dataset.transpose`
   * - remove a variable from my object
     - :py:meth:`Dataset.drop`, :py:meth:`DataArray.drop`
   * - remove dimensions of length 1 or 0
     - :py:meth:`DataArray.squeeze`, :py:meth:`Dataset.squeeze`
   * - remove all variables with a particular dimension
     - :py:meth:`Dataset.drop_dims`
   * - convert non-dimension coordinates to data variables or remove them
     - :py:meth:`DataArray.reset_coords`, :py:meth:`Dataset.reset_coords`
   * - rename a variable, dimension or coordinate
     - :py:meth:`Dataset.rename`, :py:meth:`DataArray.rename`, :py:meth:`Dataset.rename_vars`, :py:meth:`Dataset.rename_dims`,
   * - convert a DataArray to Dataset or vice versa
     - :py:meth:`DataArray.to_dataset`, :py:meth:`Dataset.to_array`
   * - extract the underlying array (e.g. numpy or Dask arrays)
     - :py:attr:`DataArray.data`
   * - convert to and extract the underlying numpy array
     - :py:attr:`DataArray.values`
   * - find out if my xarray object is wrapping a Dask Array
     - :py:func:`dask.is_dask_collection`
   * - know how much memory my object requires
     - :py:attr:`DataArray.nbytes`, :py:attr:`Dataset.nbytes`
   * - convert a possibly irregularly sampled timeseries to a regularly sampled timeseries
     - :py:meth:`DataArray.resample`, :py:meth:`Dataset.resample` (see :ref:`resampling` for more)
   * - apply a function on all data variables in a Dataset
     - :py:meth:`Dataset.apply`
   * - write xarray objects with complex values to a netCDF file
     - :py:func:`Dataset.to_netcdf`, :py:func:`DataArray.to_netcdf` specifying ``engine="h5netcdf", invalid_netcdf=True``
   * - make xarray objects look like other xarray objects
     - :py:func:`~xarray.ones_like`, :py:func:`~xarray.zeros_like`, :py:func:`~xarray.full_like`, :py:meth:`Dataset.reindex_like`, :py:meth:`Dataset.interpolate_like`, :py:meth:`Dataset.broadcast_like`, :py:meth:`DataArray.reindex_like`, :py:meth:`DataArray.interpolate_like`, :py:meth:`DataArray.broadcast_like`
   * - replace NaNs with other values
     - :py:meth:`Dataset.fillna`, :py:meth:`Dataset.ffill`, :py:meth:`Dataset.bfill`, :py:meth:`Dataset.interpolate_na`, :py:meth:`DataArray.fillna`, :py:meth:`DataArray.ffill`, :py:meth:`DataArray.bfill`, :py:meth:`DataArray.interpolate_na`
   * - extract the year, month, day or similar from a DataArray of time values
     - ``obj.dt.month`` for example where ``obj`` is a :py:class:`~xarray.DataArray` containing ``datetime64`` or ``cftime`` values. See :ref:`dt_accessor` for more.
   * - round off time values to a specified frequency
     - ``obj.dt.ceil``, ``obj.dt.floor``, ``obj.dt.round``. See :ref:`dt_accessor` for more.
   * - make a mask that is ``True`` where an object contains any of the values in a array
     - :py:meth:`Dataset.isin`, :py:meth:`DataArray.isin`
