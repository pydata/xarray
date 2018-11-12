import xarray as xr
import numpy as np


# Equal sampling comparisons:
times = xr.cftime_range('2000-01-01', periods=9, freq='T', tz='UTC')
da = xr.DataArray(np.arange(100, 100+times.size), [('time', times)])
print(da.resample(time='T', label='left', closed='left').mean())
print(da.resample(time='T', label='right', closed='left').mean())
print(da.resample(time='T', label='left', closed='right').mean())
print(da.resample(time='T', label='right', closed='right').mean())
print(da.resample(time='60S', label='left', closed='left').mean())
print(da.resample(time='60S', label='right', closed='left').mean())
print(da.resample(time='60S', label='left', closed='right').mean())
print(da.resample(time='60S', label='right', closed='right').mean())
times = xr.cftime_range('2000', periods=30, freq='MS')
da = xr.DataArray(np.arange(100, 100+times.size), [('time', times)])
print(da.resample(time='M', label='left', closed='left').max())
print(da.resample(time='M', label='right', closed='left').max())
print(da.resample(time='M', label='left', closed='right').max())
print(da.resample(time='M', label='right', closed='right').max())
print(da.resample(time='MS', label='left', closed='left').max())
print(da.resample(time='MS', label='right', closed='left').max())
print(da.resample(time='MS', label='left', closed='right').max())
print(da.resample(time='MS', label='right', closed='right').max())


# Downsampling comparisons:
times = xr.cftime_range('2000-01-01', periods=9, freq='MS', tz='UTC')
da = xr.DataArray(np.arange(100, 100+times.size), [('time', times)])
print(da.resample(time='3M', label='left', closed='left').max())
print(da.resample(time='3M', label='right', closed='left').max())
print(da.resample(time='3M', label='left', closed='right').max())
print(da.resample(time='3M', label='right', closed='right').max())
print(da.resample(time='3MS', label='left', closed='left').max())
print(da.resample(time='3MS', label='right', closed='left').max())
print(da.resample(time='3MS', label='left', closed='right').max())
print(da.resample(time='3MS', label='right', closed='right').max())
print(da.resample(time='3M', label='left', closed='left').mean())
print(da.resample(time='3M', label='right', closed='left').mean())
print(da.resample(time='3M', label='left', closed='right').mean())
print(da.resample(time='3M', label='right', closed='right').mean())
print(da.resample(time='3MS', label='left', closed='left').mean())
print(da.resample(time='3MS', label='right', closed='left').mean())
print(da.resample(time='3MS', label='left', closed='right').mean())
print(da.resample(time='3MS', label='right', closed='right').mean())
print(da.resample(time='2M', label='left', closed='left').mean())
print(da.resample(time='2M', label='right', closed='left').mean())
print(da.resample(time='2M', label='left', closed='right').mean())
print(da.resample(time='2M', label='right', closed='right').mean())
print(da.resample(time='2MS', label='left', closed='left').mean())
print(da.resample(time='2MS', label='right', closed='left').mean())
print(da.resample(time='2MS', label='left', closed='right').mean())
print(da.resample(time='2MS', label='right', closed='right').mean())
# Checking how label and closed args affect outputs
times = xr.cftime_range('2000-01-01', periods=9, freq='T', tz='UTC')
da = xr.DataArray(np.arange(100, 100+times.size), [('time', times)])
print(da.resample(time='3T', label='left', closed='left').mean())
print(da.resample(time='3T', label='right', closed='left').mean())
print(da.resample(time='3T', label='left', closed='right').mean())
print(da.resample(time='3T', label='right', closed='right').mean())
times = xr.cftime_range('2000', periods=30, freq='MS')
da = xr.DataArray(np.arange(100, 100+times.size), [('time', times)])
print(da.resample(time='6MS', label='left', closed='left').max())
print(da.resample(time='6MS', label='right', closed='left').max())
print(da.resample(time='6MS', label='left', closed='right').max())
print(da.resample(time='6MS', label='right', closed='right').max())
# Checking different aggregation funcs, also checking cases when label and closed == None
times = xr.cftime_range('2000', periods=30, freq='MS')
da = xr.DataArray(np.arange(100, 100+times.size), [('time', times)])
print(da.resample(time='MS').mean())  # essentially doing no resampling, should return original data
print(da.resample(time='6MS').mean())
print(da.resample(time='6MS').asfreq())  # results do not match since xarray makes asfreq = mean (see resample.py)
print(da.resample(time='6MS').sum())
print(da.resample(time='6MS').min())
print(da.resample(time='6MS').max())


# Upsampling comparisons:
# At seconds-resolution, xr.cftime_range is 1 second off from pd.date_range
times = xr.cftime_range('2011-01-01T13:02:03', '2012-01-01T00:00:00', freq='D')
da = xr.DataArray(np.arange(100, 100+times.size), [('time', times)])
print(da.resample(time='12T', base=0).interpolate().indexes)  # testing 'T' vs 'min'
print(da.resample(time='12min', base=0).interpolate().indexes)
print(da.resample(time='12min', base=1).interpolate().indexes)
print(da.resample(time='12min', base=5).mean().indexes)
print(da.resample(time='12min', base=17).mean().indexes)
print(da.resample(time='12S', base=17).interpolate().indexes)
print(da.resample(time='1D', base=0).interpolate().values)  # essentially doing no resampling, should return original data
print(da.resample(time='1D', base=0).mean().values)  # essentially doing no resampling, should return original data
# Upsampling with non 00:00:00 dates. Sum and mean matches pandas behavior but interpolate doesn't.
times = xr.cftime_range('2000-01-01T13:02:03', '2000-02-01T00:00:00', freq='D')
da = xr.DataArray(np.arange(100, 100+times.size), [('time', times)])
print(da.resample(time='8H', base=0, closed='left').interpolate().values)
print(da.resample(time='8H', base=0, closed='left').sum().values)
print(da.resample(time='8H', base=0, closed='left').mean().values)
print(da.resample(time='8H', base=0, closed='right').interpolate().values)
print(da.resample(time='8H', base=0, closed='right').sum().values)
print(da.resample(time='8H', base=0, closed='right').mean().values)
# Neat start times (00:00:00) produces behavior matching pandas'
times = xr.cftime_range('2000-01-01T00:00:00', '2000-02-01T00:00:00', freq='D')
da = xr.DataArray(np.arange(100, 100+times.size), [('time', times)])
print(da.resample(time='12T', base=0).interpolate())
print(da.resample(time='12T', base=24, closed='left').interpolate())
print(da.resample(time='12T', base=24, closed='right', label='left').interpolate())
times = xr.cftime_range('2000', periods=30, freq='MS')
da = xr.DataArray(np.arange(100, 100+times.size), [('time', times)])
print(da.resample(time='D').interpolate())


# Check that Dataset and DataArray returns the same resampling results
times = xr.cftime_range('2000-01-01', periods=9, freq='T', tz='UTC')
ds = xr.Dataset(data_vars={'data1': ('time', np.arange(100, 100+times.size)),
                           'data2': ('time', np.arange(500, 500+times.size))},
                coords={'time': times})
print(ds.resample(time='3T', label='left', closed='left').mean())
print(ds.resample(time='3T', label='right', closed='left').mean())
print(ds.resample(time='3T', label='left', closed='right').mean())
print(ds.resample(time='3T', label='right', closed='right').mean())
times = xr.cftime_range('2000', periods=30, freq='MS')
ds = xr.Dataset(data_vars={'data1': ('time', np.arange(100, 100+times.size)),
                           'data2': ('time', np.arange(500, 500+times.size))},
                coords={'time': times})
print(ds.resample(time='6MS', label='left', closed='left').max())
print(ds.resample(time='6MS', label='right', closed='left').max())
print(ds.resample(time='6MS', label='left', closed='right').max())
print(ds.resample(time='6MS', label='right', closed='right').max())


# Check that nc files read as dask arrays can be resampled
#
import os
testfilepath = os.path.join(os.path.expanduser('~'), 'Dropbox', 'code', 'Ouranos', 'testdata', 'NRCANdaily', 'nrcan_canada_daily_tasmax_1990.nc')
xr.set_options(enable_cftimeindex=True)
test_ds = xr.open_dataset(testfilepath, chunks={'time': 10})
test_ds['time'] = xr.cftime_range('1999-01-01', '1999-12-31', freq='D')  # regular calendars are still read as pandas date_range even though enable_cftimeindex=True
# test_ds.fillna(0).resample(time='3MS')  # NaN in results still present
print(test_ds.resample(time='MS').mean())
monthly_avg = test_ds.resample(time='MS').mean()
monthly_avg.sel(lat=49.95833206176758, lon=-79.95833587646484).to_dataframe()
chunked_cftime_val = monthly_avg.sel(lat=49.95833206176758, lon=-79.95833587646484).to_dataframe().values

xr.set_options(enable_cftimeindex=False)
test_ds = xr.open_dataset(testfilepath, chunks={'time': 10})
# print(test_ds.resample(time='MS').interpolate())  # xarray does not allow dask upsampling
print(test_ds.resample(time='MS').mean())
monthly_avg = test_ds.resample(time='MS').mean()
monthly_avg.sel(lat=49.95833206176758, lon=-79.95833587646484).to_dataframe()
chunked_pandas_val = monthly_avg.sel(lat=49.95833206176758, lon=-79.95833587646484).to_dataframe().values

# assert(np.all(chunked_cftime_val == chunked_pandas_val))
print(np.all(chunked_cftime_val == chunked_pandas_val))
