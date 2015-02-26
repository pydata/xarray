Calculating Seasonal Averages from Timeseries of Monthly Means
==============================================================

Author: `Joe Hamman <http://www.hydro.washington.edu/~jhamman/>`_

Suppose we have a netCDF or xray Dataset of monthly mean data and we
want to calculate the seasonal average. To do this properly, we need to
calculate the weighted average considering that each month has a
different number of days.

.. code:: python

    import numpy as np
    import pandas as pd
    import xray
    from netCDF4 import num2date
    import matplotlib.pyplot as plt

    print("numpy version  : ", np.__version__)
    print("pandas version : ", pd.version.version)
    print("xray version   : ", xray.version.version)

.. parsed-literal::

    numpy version  :  1.9.1
    pandas version :  0.15.2
    xray version   :  0.3.2


Some calendar information so we can support any netCDF calendar.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           'all_leap': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           '366_day': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]}
A few calendar functions to determine the number of days in each month
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you were just using the standard calendar, it would be easy to use
the ``calendar.month_range`` function.

.. code:: python

    def leap_year(year, calendar='standard'):
        """Determine if year is a leap year"""
        leap = False
        if ((calendar in ['standard', 'gregorian',
            'proleptic_gregorian', 'julian']) and
            (year % 4 == 0)):
            leap = True
            if ((calendar == 'proleptic_gregorian') and
                (year % 100 == 0) and
                (year % 400 != 0)):
                leap = False
            elif ((calendar in ['standard', 'gregorian']) and
                     (year % 100 == 0) and (year % 400 != 0) and
                     (year < 1583)):
                leap = False
        return leap

    def get_dpm(time, calendar='standard'):
        """
        return a array of days per month corresponding to the months provided in `months`
        """
        month_length = np.zeros(len(time), dtype=np.int)

        cal_days = dpm[calendar]

        for i, (month, year) in enumerate(zip(time.month, time.year)):
            month_length[i] = cal_days[month]
            if leap_year(year, calendar=calendar):
                month_length[i] += 1
        return month_length
Open the ``Dataset``
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    monthly_mean_file = '/raid2/jhamman/projects/RASM/data/processed/R1002RBRxaaa01a/lnd/monthly_mean_timeseries/R1002RBRxaaa01a.vic.hmm.197909-201212.nc'
    ds = xray.open_dataset(monthly_mean_file, decode_coords=False)
    ds.attrs['history'] = ''  # get rid of the history attribute because its obnoxiously long
    print(ds)

.. parsed-literal::

    <xray.Dataset>
    Dimensions:        (depth: 3, time: 400, x: 275, y: 205)
    Coordinates:
      * time           (time) datetime64[ns] 1979-09-16T12:00:00 1979-10-17 ...
      * depth          (depth) int64 0 1 2
      * x              (x) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ...
      * y              (y) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ...
    Variables:
        Precipitation  (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Evap           (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Runoff         (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Baseflow       (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Soilw          (time, depth, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Swq            (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Swd            (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Swnet          (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Lwnet          (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Lwin           (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Netrad         (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Swin           (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Latht          (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Senht          (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Grdht          (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Albedo         (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Radt           (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Surft          (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Relhum         (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Tair           (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
        Tsoil          (time, depth, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Wind           (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan nan nan ...
    Attributes:
        title: /workspace/jhamman/processed/R1002RBRxaaa01a/lnd/temp/R1002RBRxaaa01a.vic.ha.1979-09-01.nc
        institution: U.W.
        source: RACM R1002RBRxaaa01a
        output_frequency: daily
        output_mode: averaged
        convention: CF-1.4
        history:
        references: Based on the initial model of Liang et al., 1994, JGR, 99, 14,415- 14,429.
        comment: Output from the Variable Infiltration Capacity (VIC) model.
        nco_openmp_thread_number: 1


Now for the heavy lifting:
^^^^^^^^^^^^^^^^^^^^^^^^^^

We first have to come up with the weights, - calculate the month lengths
for each monthly data record - calculate weights using
``groupby('time.season')``

Finally, we just need to multiply our weights by the ``Dataset`` and sum
allong the time dimension.

.. code:: python

    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = xray.DataArray(get_dpm(ds.time.to_index(), calendar='noleap'),
                                  coords=[ds.time], name='month_length')
    # Calculate the weights by grouping by 'time.season'
    weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))

    # Calculate the weighted average
    ds_weighted = (ds * weights).groupby('time.season').sum(dim='time')
.. code:: python

    print(ds_weighted)

.. parsed-literal::

    <xray.Dataset>
    Dimensions:        (depth: 3, time.season: 4, x: 275, y: 205)
    Coordinates:
      * depth          (depth) int64 0 1 2
      * x              (x) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ...
      * y              (y) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ...
      * time.season    (time.season) int32 1 2 3 4
    Variables:
        Lwnet          (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Tair           (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Surft          (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Senht          (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Tsoil          (time.season, depth, y, x) float64 nan nan nan nan nan nan nan nan nan ...
        Netrad         (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Evap           (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Latht          (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Wind           (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Precipitation  (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Soilw          (time.season, depth, y, x) float64 nan nan nan nan nan nan nan nan nan ...
        Relhum         (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Swd            (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Swnet          (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Swq            (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Swin           (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Albedo         (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Lwin           (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Radt           (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Runoff         (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Grdht          (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...
        Baseflow       (time.season, y, x) float64 nan nan nan nan nan nan nan nan nan nan nan ...


.. code:: python

    # only used for comparisons
    ds_unweighted = ds.groupby('time.season').mean('time')
    ds_diff = ds_weighted - ds_unweighted
.. code:: python

    # Quick plot to show the results
    is_null = np.isnan(ds_weighted['Tair'][0].values)

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14,12))
    for i, season in enumerate(('DJF', 'MAM', 'JJA', 'SON')):
        plt.sca(axes[i, 0])
        plt.pcolormesh(np.ma.masked_where(is_null, ds_weighted['Tair'][i].values),
                              vmin=-30, vmax=30, cmap='Spectral_r')
        plt.colorbar(extend='both')

        plt.sca(axes[i, 1])
        plt.pcolormesh(np.ma.masked_where(is_null, ds_unweighted['Tair'][i].values),
                              vmin=-30, vmax=30, cmap='Spectral_r')
        plt.colorbar(extend='both')

        plt.sca(axes[i, 2])
        plt.pcolormesh(np.ma.masked_where(is_null, ds_diff['Tair'][i].values),
                              vmin=-0.1, vmax=.1, cmap='RdBu_r')
        plt.colorbar(extend='both')
        for j in range(3):
            axes[i, j].axes.get_xaxis().set_ticklabels([])
            axes[i, j].axes.get_yaxis().set_ticklabels([])
            axes[i, j].axes.axis('tight')

        axes[i, 0].set_ylabel(season)

    axes[0, 0].set_title('Weighted by DPM')
    axes[0, 1].set_title('No Weighting')
    axes[0, 2].set_title('Difference')

    plt.tight_layout()

    fig.suptitle('Seasonal Surface Air Temperature', fontsize=16, y=1.02)


.. image:: monthly_means_output.png


.. code:: python

    # Wrap it into a simple function
    def season_mean(ds, calendar='standard'):
        # Make a DataArray of season/year groups
        year_season = xray.DataArray(ds.time.to_index().to_period(freq='Q-NOV').to_timestamp(how='E'),
                                     coords=[ds.time], name='year_season')

        # Make a DataArray with the number of days in each month, size = len(time)
        month_length = xray.DataArray(get_dpm(ds.time.to_index(), calendar=calendar),
                                      coords=[ds.time], name='month_length')
        # Calculate the weights by grouping by 'time.season'
        weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()

        # Test that the sum of the weights for each season is 1.0
        np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))

        # Calculate the weighted average
        return (ds * weights).groupby('time.season').sum(dim='time')