.. currentmodule:: xarray

Advanced API
============

.. autosummary::
   :toctree: generated/

   Coordinates
   Dataset.variables
   DataArray.variable
   DataTree.variables
   Variable
   IndexVariable
   as_variable
   Index
   IndexSelResult
   Context
   register_dataset_accessor
   register_dataarray_accessor
   register_datatree_accessor
   Dataset.set_close
   backends.BackendArray
   backends.BackendEntrypoint
   backends.list_engines
   backends.refresh_engines

.. ..

..    Missing:
..    ``DataTree.set_close``

These backends provide a low-level interface for lazily loading data from
external file-formats or protocols, and can be manually invoked to create
arguments for the ``load_store`` and ``dump_to_store`` Dataset methods:

.. autosummary::
   :toctree: generated/

   backends.NetCDF4DataStore
   backends.H5NetCDFStore
   backends.PydapDataStore
   backends.ScipyDataStore
   backends.ZarrStore
   backends.FileManager
   backends.CachingFileManager
   backends.DummyFileManager

These BackendEntrypoints provide a basic interface to the most commonly
used filetypes in the xarray universe.

.. autosummary::
   :toctree: generated/

   backends.NetCDF4BackendEntrypoint
   backends.H5netcdfBackendEntrypoint
   backends.PydapBackendEntrypoint
   backends.ScipyBackendEntrypoint
   backends.StoreBackendEntrypoint
   backends.ZarrBackendEntrypoint
