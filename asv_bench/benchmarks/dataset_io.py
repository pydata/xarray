import os

import numpy as np
import pandas as pd

import xarray as xr

from . import _skip_slow, randint, randn, requires_dask

try:
    import dask
    import dask.multiprocessing
except ImportError:
    pass


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class IOSingleNetCDF:
    """
    A few examples that benchmark reading/writing a single netCDF file with
    xarray
    """

    timeout = 300.0
    repeat = 1
    number = 5

    def make_ds(self):
        # TODO: Lazily skipped in CI as it is very demanding and slow.
        # Improve times and remove errors.
        _skip_slow()

        # single Dataset
        self.ds = xr.Dataset()
        self.nt = 1000
        self.nx = 90
        self.ny = 45

        self.block_chunks = {
            "time": self.nt / 4,
            "lon": self.nx / 3,
            "lat": self.ny / 3,
        }

        self.time_chunks = {"time": int(self.nt / 36)}

        times = pd.date_range("1970-01-01", periods=self.nt, freq="D")
        lons = xr.DataArray(
            np.linspace(0, 360, self.nx),
            dims=("lon",),
            attrs={"units": "degrees east", "long_name": "longitude"},
        )
        lats = xr.DataArray(
            np.linspace(-90, 90, self.ny),
            dims=("lat",),
            attrs={"units": "degrees north", "long_name": "latitude"},
        )
        self.ds["foo"] = xr.DataArray(
            randn((self.nt, self.nx, self.ny), frac_nan=0.2),
            coords={"lon": lons, "lat": lats, "time": times},
            dims=("time", "lon", "lat"),
            name="foo",
            attrs={"units": "foo units", "description": "a description"},
        )
        self.ds["bar"] = xr.DataArray(
            randn((self.nt, self.nx, self.ny), frac_nan=0.2),
            coords={"lon": lons, "lat": lats, "time": times},
            dims=("time", "lon", "lat"),
            name="bar",
            attrs={"units": "bar units", "description": "a description"},
        )
        self.ds["baz"] = xr.DataArray(
            randn((self.nx, self.ny), frac_nan=0.2).astype(np.float32),
            coords={"lon": lons, "lat": lats},
            dims=("lon", "lat"),
            name="baz",
            attrs={"units": "baz units", "description": "a description"},
        )

        self.ds.attrs = {"history": "created for xarray benchmarking"}

        self.oinds = {
            "time": randint(0, self.nt, 120),
            "lon": randint(0, self.nx, 20),
            "lat": randint(0, self.ny, 10),
        }
        self.vinds = {
            "time": xr.DataArray(randint(0, self.nt, 120), dims="x"),
            "lon": xr.DataArray(randint(0, self.nx, 120), dims="x"),
            "lat": slice(3, 20),
        }


class IOWriteSingleNetCDF3(IOSingleNetCDF):
    def setup(self):
        self.format = "NETCDF3_64BIT"
        self.make_ds()

    def time_write_dataset_netcdf4(self):
        self.ds.to_netcdf("test_netcdf4_write.nc", engine="netcdf4", format=self.format)

    def time_write_dataset_scipy(self):
        self.ds.to_netcdf("test_scipy_write.nc", engine="scipy", format=self.format)


class IOReadSingleNetCDF4(IOSingleNetCDF):
    def setup(self):

        self.make_ds()

        self.filepath = "test_single_file.nc4.nc"
        self.format = "NETCDF4"
        self.ds.to_netcdf(self.filepath, format=self.format)

    def time_load_dataset_netcdf4(self):
        xr.open_dataset(self.filepath, engine="netcdf4").load()

    def time_orthogonal_indexing(self):
        ds = xr.open_dataset(self.filepath, engine="netcdf4")
        ds = ds.isel(**self.oinds).load()

    def time_vectorized_indexing(self):
        ds = xr.open_dataset(self.filepath, engine="netcdf4")
        ds = ds.isel(**self.vinds).load()


class IOReadSingleNetCDF3(IOReadSingleNetCDF4):
    def setup(self):

        self.make_ds()

        self.filepath = "test_single_file.nc3.nc"
        self.format = "NETCDF3_64BIT"
        self.ds.to_netcdf(self.filepath, format=self.format)

    def time_load_dataset_scipy(self):
        xr.open_dataset(self.filepath, engine="scipy").load()

    def time_orthogonal_indexing(self):
        ds = xr.open_dataset(self.filepath, engine="scipy")
        ds = ds.isel(**self.oinds).load()

    def time_vectorized_indexing(self):
        ds = xr.open_dataset(self.filepath, engine="scipy")
        ds = ds.isel(**self.vinds).load()


class IOReadSingleNetCDF4Dask(IOSingleNetCDF):
    def setup(self):

        requires_dask()

        self.make_ds()

        self.filepath = "test_single_file.nc4.nc"
        self.format = "NETCDF4"
        self.ds.to_netcdf(self.filepath, format=self.format)

    def time_load_dataset_netcdf4_with_block_chunks(self):
        xr.open_dataset(
            self.filepath, engine="netcdf4", chunks=self.block_chunks
        ).load()

    def time_load_dataset_netcdf4_with_block_chunks_oindexing(self):
        ds = xr.open_dataset(self.filepath, engine="netcdf4", chunks=self.block_chunks)
        ds = ds.isel(**self.oinds).load()

    def time_load_dataset_netcdf4_with_block_chunks_vindexing(self):
        ds = xr.open_dataset(self.filepath, engine="netcdf4", chunks=self.block_chunks)
        ds = ds.isel(**self.vinds).load()

    def time_load_dataset_netcdf4_with_block_chunks_multiprocessing(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_dataset(
                self.filepath, engine="netcdf4", chunks=self.block_chunks
            ).load()

    def time_load_dataset_netcdf4_with_time_chunks(self):
        xr.open_dataset(self.filepath, engine="netcdf4", chunks=self.time_chunks).load()

    def time_load_dataset_netcdf4_with_time_chunks_multiprocessing(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_dataset(
                self.filepath, engine="netcdf4", chunks=self.time_chunks
            ).load()


class IOReadSingleNetCDF3Dask(IOReadSingleNetCDF4Dask):
    def setup(self):

        requires_dask()

        self.make_ds()

        self.filepath = "test_single_file.nc3.nc"
        self.format = "NETCDF3_64BIT"
        self.ds.to_netcdf(self.filepath, format=self.format)

    def time_load_dataset_scipy_with_block_chunks(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_dataset(
                self.filepath, engine="scipy", chunks=self.block_chunks
            ).load()

    def time_load_dataset_scipy_with_block_chunks_oindexing(self):
        ds = xr.open_dataset(self.filepath, engine="scipy", chunks=self.block_chunks)
        ds = ds.isel(**self.oinds).load()

    def time_load_dataset_scipy_with_block_chunks_vindexing(self):
        ds = xr.open_dataset(self.filepath, engine="scipy", chunks=self.block_chunks)
        ds = ds.isel(**self.vinds).load()

    def time_load_dataset_scipy_with_time_chunks(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_dataset(
                self.filepath, engine="scipy", chunks=self.time_chunks
            ).load()


class IOMultipleNetCDF:
    """
    A few examples that benchmark reading/writing multiple netCDF files with
    xarray
    """

    timeout = 300.0
    repeat = 1
    number = 5

    def make_ds(self, nfiles=10):
        # TODO: Lazily skipped in CI as it is very demanding and slow.
        # Improve times and remove errors.
        _skip_slow()

        # multiple Dataset
        self.ds = xr.Dataset()
        self.nt = 1000
        self.nx = 90
        self.ny = 45
        self.nfiles = nfiles

        self.block_chunks = {
            "time": self.nt / 4,
            "lon": self.nx / 3,
            "lat": self.ny / 3,
        }

        self.time_chunks = {"time": int(self.nt / 36)}

        self.time_vars = np.split(
            pd.date_range("1970-01-01", periods=self.nt, freq="D"), self.nfiles
        )

        self.ds_list = []
        self.filenames_list = []
        for i, times in enumerate(self.time_vars):
            ds = xr.Dataset()
            nt = len(times)
            lons = xr.DataArray(
                np.linspace(0, 360, self.nx),
                dims=("lon",),
                attrs={"units": "degrees east", "long_name": "longitude"},
            )
            lats = xr.DataArray(
                np.linspace(-90, 90, self.ny),
                dims=("lat",),
                attrs={"units": "degrees north", "long_name": "latitude"},
            )
            ds["foo"] = xr.DataArray(
                randn((nt, self.nx, self.ny), frac_nan=0.2),
                coords={"lon": lons, "lat": lats, "time": times},
                dims=("time", "lon", "lat"),
                name="foo",
                attrs={"units": "foo units", "description": "a description"},
            )
            ds["bar"] = xr.DataArray(
                randn((nt, self.nx, self.ny), frac_nan=0.2),
                coords={"lon": lons, "lat": lats, "time": times},
                dims=("time", "lon", "lat"),
                name="bar",
                attrs={"units": "bar units", "description": "a description"},
            )
            ds["baz"] = xr.DataArray(
                randn((self.nx, self.ny), frac_nan=0.2).astype(np.float32),
                coords={"lon": lons, "lat": lats},
                dims=("lon", "lat"),
                name="baz",
                attrs={"units": "baz units", "description": "a description"},
            )

            ds.attrs = {"history": "created for xarray benchmarking"}

            self.ds_list.append(ds)
            self.filenames_list.append("test_netcdf_%i.nc" % i)


class IOWriteMultipleNetCDF3(IOMultipleNetCDF):
    def setup(self):
        self.make_ds()
        self.format = "NETCDF3_64BIT"

    def time_write_dataset_netcdf4(self):
        xr.save_mfdataset(
            self.ds_list, self.filenames_list, engine="netcdf4", format=self.format
        )

    def time_write_dataset_scipy(self):
        xr.save_mfdataset(
            self.ds_list, self.filenames_list, engine="scipy", format=self.format
        )


class IOReadMultipleNetCDF4(IOMultipleNetCDF):
    def setup(self):

        requires_dask()

        self.make_ds()
        self.format = "NETCDF4"
        xr.save_mfdataset(self.ds_list, self.filenames_list, format=self.format)

    def time_load_dataset_netcdf4(self):
        xr.open_mfdataset(self.filenames_list, engine="netcdf4").load()

    def time_open_dataset_netcdf4(self):
        xr.open_mfdataset(self.filenames_list, engine="netcdf4")


class IOReadMultipleNetCDF3(IOReadMultipleNetCDF4):
    def setup(self):

        requires_dask()

        self.make_ds()
        self.format = "NETCDF3_64BIT"
        xr.save_mfdataset(self.ds_list, self.filenames_list, format=self.format)

    def time_load_dataset_scipy(self):
        xr.open_mfdataset(self.filenames_list, engine="scipy").load()

    def time_open_dataset_scipy(self):
        xr.open_mfdataset(self.filenames_list, engine="scipy")


class IOReadMultipleNetCDF4Dask(IOMultipleNetCDF):
    def setup(self):

        requires_dask()

        self.make_ds()
        self.format = "NETCDF4"
        xr.save_mfdataset(self.ds_list, self.filenames_list, format=self.format)

    def time_load_dataset_netcdf4_with_block_chunks(self):
        xr.open_mfdataset(
            self.filenames_list, engine="netcdf4", chunks=self.block_chunks
        ).load()

    def time_load_dataset_netcdf4_with_block_chunks_multiprocessing(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="netcdf4", chunks=self.block_chunks
            ).load()

    def time_load_dataset_netcdf4_with_time_chunks(self):
        xr.open_mfdataset(
            self.filenames_list, engine="netcdf4", chunks=self.time_chunks
        ).load()

    def time_load_dataset_netcdf4_with_time_chunks_multiprocessing(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="netcdf4", chunks=self.time_chunks
            ).load()

    def time_open_dataset_netcdf4_with_block_chunks(self):
        xr.open_mfdataset(
            self.filenames_list, engine="netcdf4", chunks=self.block_chunks
        )

    def time_open_dataset_netcdf4_with_block_chunks_multiprocessing(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="netcdf4", chunks=self.block_chunks
            )

    def time_open_dataset_netcdf4_with_time_chunks(self):
        xr.open_mfdataset(
            self.filenames_list, engine="netcdf4", chunks=self.time_chunks
        )

    def time_open_dataset_netcdf4_with_time_chunks_multiprocessing(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="netcdf4", chunks=self.time_chunks
            )


class IOReadMultipleNetCDF3Dask(IOReadMultipleNetCDF4Dask):
    def setup(self):

        requires_dask()

        self.make_ds()
        self.format = "NETCDF3_64BIT"
        xr.save_mfdataset(self.ds_list, self.filenames_list, format=self.format)

    def time_load_dataset_scipy_with_block_chunks(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="scipy", chunks=self.block_chunks
            ).load()

    def time_load_dataset_scipy_with_time_chunks(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="scipy", chunks=self.time_chunks
            ).load()

    def time_open_dataset_scipy_with_block_chunks(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="scipy", chunks=self.block_chunks
            )

    def time_open_dataset_scipy_with_time_chunks(self):
        with dask.config.set(scheduler="multiprocessing"):
            xr.open_mfdataset(
                self.filenames_list, engine="scipy", chunks=self.time_chunks
            )


def create_delayed_write():
    import dask.array as da

    # TODO: Lazily skipped in CI as it is very demanding and slow.
    # Improve times and remove errors.
    _skip_slow()

    vals = da.random.random(300, chunks=(1,))
    ds = xr.Dataset({"vals": (["a"], vals)})
    return ds.to_netcdf("file.nc", engine="netcdf4", compute=False)


class IOWriteNetCDFDask:
    timeout = 60
    repeat = 1
    number = 5

    def setup(self):
        requires_dask()
        self.write = create_delayed_write()

    def time_write(self):
        self.write.compute()


class IOWriteNetCDFDaskDistributed:
    def setup(self):
        try:
            import distributed
        except ImportError:
            raise NotImplementedError()

        # TODO: Lazily skipped in CI as it is very demanding and slow.
        # Improve times and remove errors.
        _skip_slow()

        self.client = distributed.Client()
        self.write = create_delayed_write()

    def cleanup(self):
        self.client.shutdown()

    def time_write(self):
        self.write.compute()
