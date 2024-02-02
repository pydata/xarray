import netCDF4 as nc
import numpy as np

import xarray as xr

xr.get_options()

ds = nc.Dataset("mre.nc", "w", format="NETCDF4")
cloud_type_enum = ds.createEnumType(int, "cloud_type", {"clear": 0, "cloudy": 1})  #
ds.createDimension("time", size=(10))
x = np.arange(10)
ds.createVariable("x", np.int32, dimensions=("time",))
ds.variables["x"][:] = x
# {'cloud_type': <class 'netCDF4._netCDF4.EnumType'>: name = 'cloud_type', numpy dtype = int64, fields/values ={'clear': 0, 'cloudy': 1}}
ds.createVariable("cloud", cloud_type_enum, dimensions=("time",))
ds["cloud"][:] = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
ds.close()

# -- Open dataset with xarray
xr_ds = xr.open_dataset("./mre.nc")
xr_ds.to_netcdf("./mre_new.nc")
xr_ds = xr.open_dataset("./mre_new.nc")
xr_ds
ds_re_read = nc.Dataset("./mre_new.nc", "r", format="NETCDF4")
ds_re_read

# import numpy as np
# import xarray as xr

# codes = np.array([0, 1, 2, 1, 0])
# categories = {0: 'foo', 1: 'jazz', 2: 'bar'}
# cat_arr = xr.coding.variables.CategoricalArray(codes=codes, categories=categories)
# v = xr.Variable(("time,"), cat_arr, fastpath=True)
# ds = xr.Dataset({'cloud': v})
# ds.to_zarr('test.zarr')
