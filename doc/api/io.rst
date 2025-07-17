.. currentmodule:: xarray

IO / Conversion
===============

Dataset methods
---------------

.. autosummary::
   :toctree: ../generated/

   load_dataset
   open_dataset
   open_mfdataset
   open_zarr
   save_mfdataset
   Dataset.as_numpy
   Dataset.from_dataframe
   Dataset.from_dict
   Dataset.to_dataarray
   Dataset.to_dataframe
   Dataset.to_dask_dataframe
   Dataset.to_dict
   Dataset.to_netcdf
   Dataset.to_pandas
   Dataset.to_zarr
   Dataset.chunk
   Dataset.close
   Dataset.compute
   Dataset.filter_by_attrs
   Dataset.info
   Dataset.load
   Dataset.persist
   Dataset.unify_chunks

DataArray methods
-----------------

.. autosummary::
   :toctree: ../generated/

   load_dataarray
   open_dataarray
   DataArray.as_numpy
   DataArray.from_dict
   DataArray.from_iris
   DataArray.from_series
   DataArray.to_dask_dataframe
   DataArray.to_dataframe
   DataArray.to_dataset
   DataArray.to_dict
   DataArray.to_index
   DataArray.to_iris
   DataArray.to_masked_array
   DataArray.to_netcdf
   DataArray.to_numpy
   DataArray.to_pandas
   DataArray.to_series
   DataArray.to_zarr
   DataArray.chunk
   DataArray.close
   DataArray.compute
   DataArray.persist
   DataArray.load
   DataArray.unify_chunks

DataTree methods
----------------

.. autosummary::
   :toctree: ../generated/

   open_datatree
   open_groups
   DataTree.to_dict
   DataTree.to_netcdf
   DataTree.to_zarr
   DataTree.chunk
   DataTree.load
   DataTree.compute
   DataTree.persist

.. ..

..    Missing:
..    ``open_mfdatatree``
