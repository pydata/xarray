import pytest

import xarray as xr


@pytest.mark.parametrize("test_type", ["baseline", "encoding_and_unlimited_dims"])
def test_save_mfdataset_pass_kwargs(test_type):
    # create a timeseries to store in a netCDF file
    times = list(range(0, 1000))
    time = xr.DataArray(times, dims=("time",))

    # create a simple dataset to write using save_mfdataset
    test_ds = xr.Dataset()
    test_ds["time"] = time

    # make sure the times are written as double and
    # turn off fill values
    encoding = dict(time=dict(dtype="double"))
    unlimited_dims = ["time"]

    # set the output file name
    output_path = "test.nc"

    # the test fails if save_mfdataset doesn't accept kwargs
    # but it works if instead the dataset is saved using
    # test_ds.to_netcdf(output_path, encoding = encoding)
    if test_type == "baseline":
        test_ds.to_netcdf(output_path, encoding=encoding, unlimited_dims=unlimited_dims)
    elif test_type == "encoding_and_unlimited_dims":
        xr.save_mfdataset(
            [test_ds], [output_path], encoding=encoding, unlimited_dims=unlimited_dims
        )
    else:
        raise RuntimeError("Unexpected value for argument `test_type`")
