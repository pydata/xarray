Working with weather data
=========================

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

    @savefig examples_pairplot.png
    sns.pairplot(ds[['tmin', 'tmax', 'time.month']].to_dataframe(),
                 vars=ds.vars, hue='time.month')


Probability of freeze by calendar month
---------------------------------------

.. ipython:: python

    freeze = (ds['tmin'] <= 0).groupby('time.month').mean('time')
    freeze

    @savefig examples_freeze_prob.png
    freeze.to_series().unstack('location').plot()


Monthly averaging
-----------------

.. literalinclude:: _code/monthly_averaging.py

.. ipython:: python
   :suppress:

   execfile("examples/_code/monthly_averaging.py")

.. ipython:: python

    monthly_avg = ds.groupby(year_month(ds)).mean()

    @savefig examples_tmin_tmax_plot_mean.png
    monthly_avg.to_dataframe().plot(style='s-')


Calculate monthly anomalies
---------------------------

In climatology, "anomalies" refer to the difference between observations and
typical weather for a particular season. Unlike observations, anomalies should
not show any seasonal cycle.

.. ipython:: python

    climatology = ds.groupby('time.month').mean('time')
    anomalies = ds.groupby('time.month') - climatology

    @savefig examples_anomalies_plot.png
    anomalies.mean('location').reset_coords(drop=True).to_dataframe().plot()

