.. _toy weather data:

Toy weather data
================

Here is an example of how to easily manipulate a toy weather dataset using
xarray and other recommended Python libraries:

.. contents::
   :local:
   :depth: 1

Shared setup:

.. literalinclude:: _code/weather_data_setup.py

.. ipython:: python
   :suppress:

    fpath = "examples/_code/weather_data_setup.py"
    with open(fpath) as f:
        code = compile(f.read(), fpath, 'exec')
        exec(code)


Examine a dataset with pandas_ and seaborn_
-------------------------------------------

.. _pandas: http://pandas.pydata.org
.. _seaborn: http://stanford.edu/~mwaskom/software/seaborn

.. ipython:: python

    ds

    df = ds.to_dataframe()

    df.head()

    df.describe()

    @savefig examples_tmin_tmax_plot.png
    ds.mean(dim='location').to_dataframe().plot()


.. ipython:: python

    @savefig examples_pairplot.png
    sns.pairplot(df.reset_index(), vars=ds.data_vars)

.. _average by month:

Probability of freeze by calendar month
---------------------------------------

.. ipython:: python

    freeze = (ds['tmin'] <= 0).groupby('time.month').mean('time')
    freeze

    @savefig examples_freeze_prob.png
    freeze.to_pandas().plot()

.. _monthly average:

Monthly averaging
-----------------

.. ipython:: python

    monthly_avg = ds.resample(time='1MS').mean()

    @savefig examples_tmin_tmax_plot_mean.png
    monthly_avg.sel(location='IA').to_dataframe().plot(style='s-')

Note that ``MS`` here refers to Month-Start; ``M`` labels Month-End (the last
day of the month).

.. _monthly anomalies:

Calculate monthly anomalies
---------------------------

In climatology, "anomalies" refer to the difference between observations and
typical weather for a particular season. Unlike observations, anomalies should
not show any seasonal cycle.

.. ipython:: python

    climatology = ds.groupby('time.month').mean('time')
    anomalies = ds.groupby('time.month') - climatology

    @savefig examples_anomalies_plot.png
    anomalies.mean('location').to_dataframe()[['tmin', 'tmax']].plot()

.. _standardized monthly anomalies:

Calculate standardized monthly anomalies
----------------------------------------

You can create standardized anomalies where the difference between the
observations and the climatological monthly mean is
divided by the climatological standard deviation.

.. ipython:: python

    climatology_mean = ds.groupby('time.month').mean('time')
    climatology_std = ds.groupby('time.month').std('time')
    stand_anomalies = xr.apply_ufunc(
                                     lambda x, m, s: (x - m) / s,
                                     ds.groupby('time.month'),
                                     climatology_mean, climatology_std)

    @savefig examples_standardized_anomalies_plot.png
    stand_anomalies.mean('location').to_dataframe()[['tmin', 'tmax']].plot()

.. _fill with climatology:

Fill missing values with climatology
------------------------------------

The :py:func:`~xarray.Dataset.fillna` method on grouped objects lets you easily
fill missing values by group:

.. ipython:: python
   :okwarning:

    # throw away the first half of every month
    some_missing = ds.tmin.sel(time=ds['time.day'] > 15).reindex_like(ds)
    filled = some_missing.groupby('time.month').fillna(climatology.tmin)

    both = xr.Dataset({'some_missing': some_missing, 'filled': filled})
    both

    df = both.sel(time='2000').mean('location').reset_coords(drop=True).to_dataframe()

    @savefig examples_filled.png
    df[['filled', 'some_missing']].plot()
