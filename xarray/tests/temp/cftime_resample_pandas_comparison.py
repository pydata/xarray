import xarray as xr
import pandas as pd
import numpy as np


# Equal sampling comparisons:
ti = pd.date_range('2000-01-01', periods=9, freq='T', tz='UTC')
ps = pd.Series(np.arange(100, 100+ti.size), index=ti)
print(ps.resample('T', label='left', closed='left').mean())
print(ps.resample('T', label='right', closed='left').mean())
print(ps.resample('T', label='left', closed='right').mean())
print(ps.resample('T', label='right', closed='right').mean())
print(ps.resample('60S', label='left', closed='left').mean())
print(ps.resample('60S', label='right', closed='left').mean())
print(ps.resample('60S', label='left', closed='right').mean())
print(ps.resample('60S', label='right', closed='right').mean())
ti = pd.date_range('2000', periods=30, freq='MS', tz='UTC')
ps = pd.Series(np.arange(100, 100+ti.size), index=ti)
print(ps.resample('M', label='left', closed='left').max())
print(ps.resample('M', label='right', closed='left').max())
print(ps.resample('M', label='left', closed='right').max())
print(ps.resample('M', label='right', closed='right').max())
print(ps.resample('MS', label='left', closed='left').max())
print(ps.resample('MS', label='right', closed='left').max())
print(ps.resample('MS', label='left', closed='right').max())
print(ps.resample('MS', label='right', closed='right').max())


# Downsampling comparisons:
ti = pd.date_range('2000-01-01', periods=9, freq='MS', tz='UTC')
ps = pd.Series(np.arange(100, 100+ti.size), index=ti)
print(ps.resample('3M', label='left', closed='left').max())
print(ps.resample('3M', label='right', closed='left').max())
print(ps.resample('3M', label='left', closed='right').max())
print(ps.resample('3M', label='right', closed='right').max())
print(ps.resample('3MS', label='left', closed='left').max())
print(ps.resample('3MS', label='right', closed='left').max())
print(ps.resample('3MS', label='left', closed='right').max())
print(ps.resample('3MS', label='right', closed='right').max())
print(ps.resample('3M', label='left', closed='left').mean())
print(ps.resample('3M', label='right', closed='left').mean())
print(ps.resample('3M', label='left', closed='right').mean())
print(ps.resample('3M', label='right', closed='right').mean())
print(ps.resample('3MS', label='left', closed='left').mean())
print(ps.resample('3MS', label='right', closed='left').mean())
print(ps.resample('3MS', label='left', closed='right').mean())
print(ps.resample('3MS', label='right', closed='right').mean())
print(ps.resample('2M', label='left', closed='left').mean())
print(ps.resample('2M', label='right', closed='left').mean())
print(ps.resample('2M', label='left', closed='right').mean())
print(ps.resample('2M', label='right', closed='right').mean())
print(ps.resample('2MS', label='left', closed='left').mean())
print(ps.resample('2MS', label='right', closed='left').mean())
print(ps.resample('2MS', label='left', closed='right').mean())
print(ps.resample('2MS', label='right', closed='right').mean())
# Checking how label and closed args affect outputs
ti = pd.date_range('2000-01-01', periods=9, freq='T', tz='UTC')
ps = pd.Series(np.arange(100, 100+ti.size), index=ti)
print(ps.resample('3T', label='left', closed='left').mean())
print(ps.resample('3T', label='right', closed='left').mean())
print(ps.resample('3T', label='left', closed='right').mean())
print(ps.resample('3T', label='right', closed='right').mean())
ti = pd.date_range('2000', periods=30, freq='MS')
ps = pd.Series(np.arange(100, 100+ti.size), index=ti)
print(ps.resample('6MS', label='left', closed='left').max())
print(ps.resample('6MS', label='right', closed='left').max())
print(ps.resample('6MS', label='left', closed='right').max())
print(ps.resample('6MS', label='right', closed='right').max())
# Checking different aggregation funcs, also checking cases when label and closed == None
ti = pd.date_range('2000', periods=30, freq='MS')
ps = pd.Series(np.arange(100, 100+ti.size), index=ti)
print(ps.resample('MS').mean())  # essentially doing no resampling, should return original data
print(ps.resample('6MS').mean())
print(ps.resample('6MS').asfreq())  # results do not match since xarray makes asfreq = mean (see resample.py)
print(ps.resample('6MS').sum())
print(ps.resample('6MS').min())
print(ps.resample('6MS').max())


# Upsampling comparisons:
# At seconds-resolution, xr.cftime_range is 1 second off from pd.date_range
ti = pd.date_range('2011-01-01T13:02:03', '2012-01-01T00:00:00', freq='D', tz='UTC')
ps = pd.Series(np.arange(100, 100+ti.size), index=ti)
print(ps.resample('12T', base=0).interpolate().index) # testing T vs min
print(ps.resample('12min', base=0).interpolate().index)
print(ps.resample('12min', base=1).interpolate().index)
print(ps.resample('12min', base=5).interpolate().index)
print(ps.resample('12min', base=17).interpolate().index)
print(ps.resample('12S', base=17).interpolate().index)
print(ps.resample('1D', base=0).interpolate().values)  # essentially doing no resampling, should return original data
print(ps.resample('1D', base=0).mean().values)  # essentially doing no resampling, should return original data
# Pandas' upsampling behave aberrantly if start times for dates are not neat, should we replicate?
ti = pd.date_range('2000-01-01T13:02:03', '2000-02-01T00:00:00', freq='D', tz='UTC')
ps = pd.Series(np.arange(100, 100+ti.size), index=ti)  # results unchanged if array of floats used
print(ps.resample('8H', base=0, closed='left').interpolate().values)
print(ps.resample('8H', base=0, closed='left').sum().values)
print(ps.resample('8H', base=0, closed='left').mean().values)
print(ps.resample('8H', base=0, closed='right').interpolate().values)
print(ps.resample('8H', base=0, closed='right').sum().values)
print(ps.resample('8H', base=0, closed='right').mean().values)
# Neat start times (00:00:00) produces expected behavior when upsampling with pandas
ti = pd.date_range('2000-01-01T00:00:00', '2000-02-01T00:00:00', freq='D', tz='UTC')
ps = pd.Series(np.arange(100, 100+ti.size), index=ti)
print(ps.resample('12T', base=0).interpolate())
print(ps.resample('12T', base=24, closed='left').interpolate())
print(ps.resample('12T', base=24, closed='right', label='left').interpolate())
ti = pd.date_range('2000', periods=30, freq='MS')
ps = pd.Series(np.arange(100, 100+ti.size), index=ti)
print(ps.resample('D').interpolate())


# Shows how resample-apply produces different results with Series and DataArray
ti = pd.date_range('2000', periods=30, freq='MS')
da = xr.DataArray(np.arange(100, 100+ti.size), [('time', ti)])
print(da.resample(time='6MS').sum())
ti = pd.date_range('2000', periods=30, freq='MS')
ps = pd.Series(np.arange(100, 100+ti.size), index=ti)
print(ps.resample('6MS').sum())
