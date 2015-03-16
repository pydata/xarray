.. _toy weather data:

Toy weather data
================

Here is an example of how to easily manipulate a toy weather dataset using
xray and other recommended Python libraries:

.. contents::
   :local:
   :depth: 1

Shared setup:

.. literalinclude:: _code/weather_data_setup.py

.. ipython:: python
   :suppress:

   execfile("examples/_code/weather_data_setup.py")

Examine a dataset with pandas_ and seaborn_
-------------------------------------------

.. _pandas: http://pandas.pydata.org
.. _seaborn: http://stanford.edu/~mwaskom/software/seaborn

.. ipython:: python

    ds

    ds.to_dataframe().head()

    ds.to_dataframe().describe()

    @savefig examples_tmin_tmax_plot.png
    ds.mean(dim='location').to_dataframe().plot()

.. ipython:: python

    @savefig examples_pairplot.png
    for var in ['tmin', 'tmax']:
        sns.kdeplot(ds[var].to_series())

.. _average by month:

Probability of freeze by calendar month
---------------------------------------

.. ipython:: python

    freeze = (ds['tmin'] <= 0).groupby('time.month').mean('time')
    freeze

    @savefig examples_freeze_prob.png
    freeze.to_pandas().T.plot()

.. _monthly average:

Monthly averaging
-----------------

.. ipython:: python

    monthly_avg = ds.resample('1MS', dim='time', how='mean')

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

