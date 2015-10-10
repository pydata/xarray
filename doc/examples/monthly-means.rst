.. _monthly means example:

Calculating Seasonal Averages from Timeseries of Monthly Means
==============================================================

Author: `Joe Hamman <http://uw-hydro.github.io/current_member/joe_hamman/>`_

The data for this example can be found in the `xray-data <https://github.com/xray/xray-data>`_ repository. This example is also available in an IPython Notebook that is available `here <https://github.com/xray/xray/tree/master/examples/xray_seasonal_means.ipynb>`_.

Suppose we have a netCDF or xray Dataset of monthly mean data and we
want to calculate the seasonal average. To do this properly, we need to
calculate the weighted average considering that each month has a
different number of days.

.. code:: python

    %matplotlib inline
    import numpy as np
    import pandas as pd
    import xray
    from netCDF4 import num2date
    import matplotlib.pyplot as plt

    print("numpy version  : ", np.__version__)
    print("pandas version : ", pd.version.version)
    print("xray version   : ", xray.version.version)

.. parsed-literal::

    numpy version  :  1.9.2
    pandas version :  0.16.2
    xray version   :  0.5.1


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

    monthly_mean_file = 'RASM_example_data.nc'
    ds = xray.open_dataset(monthly_mean_file, decode_coords=False)
    print(ds)


.. parsed-literal::

    <xray.Dataset>
    Dimensions:  (time: 36, x: 275, y: 205)
    Coordinates:
      * time     (time) datetime64[ns] 1980-09-16T12:00:00 1980-10-17 ...
      * x        (x) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
      * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
    Data variables:
        Tair     (time, y, x) float64 nan nan nan nan nan nan nan nan nan nan ...
    Attributes:
        title: /workspace/jhamman/processed/R1002RBRxaaa01a/lnd/temp/R1002RBRxaaa01a.vic.ha.1979-09-01.nc
        institution: U.W.
        source: RACM R1002RBRxaaa01a
        output_frequency: daily
        output_mode: averaged
        convention: CF-1.4
        references: Based on the initial model of Liang et al., 1994, JGR, 99, 14,415- 14,429.
        comment: Output from the Variable Infiltration Capacity (VIC) model.
        nco_openmp_thread_number: 1
        NCO: 4.3.7
        history: history deleted for brevity


Now for the heavy lifting:
^^^^^^^^^^^^^^^^^^^^^^^^^^

We first have to come up with the weights, - calculate the month lengths
for each monthly data record - calculate weights using
``groupby('time.season')``

Finally, we just need to multiply our weights by the ``Dataset`` and sum
allong the time dimension.

.. code:: python

    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = xray.DataArray(get_dpm(ds.time.to_index(),
                                          calendar='noleap'),
                                  coords=[ds.time], name='month_length')

    # Calculate the weights by grouping by 'time.season'.
    # Conversion to float type ('astype(float)') only necessary for Python 2.x
    weights = month_length.groupby('time.season') / month_length.astype(float).groupby('time.season').sum()

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))

    # Calculate the weighted average
    ds_weighted = (ds * weights).groupby('time.season').sum(dim='time')

.. code:: python

    print(ds_weighted)


.. parsed-literal::

    <xray.Dataset>
    Dimensions:  (season: 4, x: 275, y: 205)
    Coordinates:
      * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
      * x        (x) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
      * season   (season) object 'DJF' 'JJA' 'MAM' 'SON'
    Data variables:
        Tair     (season, y, x) float64 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...


.. code:: python

    # only used for comparisons
    ds_unweighted = ds.groupby('time.season').mean('time')
    ds_diff = ds_weighted - ds_unweighted

.. code:: python

    # Quick plot to show the results
    is_null = np.isnan(ds_unweighted['Tair'][0].values)

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14,12))
    for i, season in enumerate(('DJF', 'MAM', 'JJA', 'SON')):
        plt.sca(axes[i, 0])
        plt.pcolormesh(np.ma.masked_where(is_null, ds_weighted['Tair'].sel(season=season).values),
                       vmin=-30, vmax=30, cmap='Spectral_r')
        plt.colorbar(extend='both')

        plt.sca(axes[i, 1])
        plt.pcolormesh(np.ma.masked_where(is_null, ds_unweighted['Tair'].sel(season=season).values),
                       vmin=-30, vmax=30, cmap='Spectral_r')
        plt.colorbar(extend='both')

        plt.sca(axes[i, 2])
        plt.pcolormesh(np.ma.masked_where(is_null, ds_diff['Tair'].sel(season=season).values),
                       vmin=-0.1, vmax=.1, cmap='RdBu_r')
        plt.colorbar(extend='both')
        for j in range(3):
            axes[i, j].axes.get_xaxis().set_ticklabels([])
            axes[i, j].axes.get_yaxis().set_ticklabels([])
            axes[i, j].axes.axis('tight')

        axes[i, 0].set_ylabel(season)

    axes[0, 0].set_title('Weighted by DPM')
    axes[0, 1].set_title('Equal Weighting')
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
