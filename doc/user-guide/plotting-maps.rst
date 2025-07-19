.. currentmodule:: xarray
.. _plot-maps:

Maps
====

To follow this section you'll need to have Cartopy installed and working.

This script will plot the air temperature on a map.

.. jupyter-execute::

    import cartopy.crs as ccrs
    import xarray as xr

.. jupyter-execute::
    :stderr:

    air = xr.tutorial.open_dataset("air_temperature").air

    p = air.isel(time=0).plot(
        subplot_kws=dict(projection=ccrs.Orthographic(-80, 35), facecolor="gray"),
        transform=ccrs.PlateCarree(),
    )
    p.axes.set_global()

    p.axes.coastlines();

When faceting on maps, the projection can be transferred to the ``plot``
function using the ``subplot_kws`` keyword. The axes for the subplots created
by faceting are accessible in the object returned by ``plot``:

.. jupyter-execute::

    p = air.isel(time=[0, 4]).plot(
        transform=ccrs.PlateCarree(),
        col="time",
        subplot_kws={"projection": ccrs.Orthographic(-80, 35)},
    )
    for ax in p.axs.flat:
        ax.coastlines()
        ax.gridlines()
