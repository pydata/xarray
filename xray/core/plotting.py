"""
Plotting functions are implemented here and also monkeypatched in to
DataArray and DataSet classes
"""

import numpy as np


# TODO - Is there a better way to import matplotlib in the function?
# Other piece of duplicated logic is the checking for axes.
# Decorators don't preserve the argument names
# But if all the plotting methods have same signature...


class FacetGrid():
    pass


def plot(darray, *args, **kwargs):
    """
    Default plot of DataArray using matplotlib / pylab.

    Calls a plotting function based on the dimensions of
    the array:

    =============== ======================================
    Dimensions      Plotting function
    --------------- --------------------------------------
    1               :py:meth:`xray.DataArray.plot_line` 
    2               :py:meth:`xray.DataArray.plot_imshow` 
    Anything else   :py:meth:`xray.DataArray.plot_hist` 
    =============== ======================================

    Parameters
    ----------
    darray : DataArray
    ax : matplotlib axes object
        If not passed, uses the current axis
    args, kwargs
        Additional arguments to matplotlib
    """
    defaults = {1: plot_line, 2: plot_imshow}
    ndims = len(darray.dims)

    if ndims in defaults:
        plotfunc = defaults[ndims]
    else:
        plotfunc = plot_hist

    return plotfunc(darray, *args, **kwargs)


def plot_line(darray, *args, **kwargs):
    """
    Line plot of 1 dimensional darray index against values

    Wraps matplotlib.pyplot.plot

    Parameters
    ----------
    darray : DataArray
        Must be 1 dimensional
    ax : matplotlib axes object
        If not passed, uses the current axis
    args, kwargs
        Additional arguments to matplotlib.pyplot.plot

    Examples
    --------

    """
    import matplotlib.pyplot as plt

    ndims = len(darray.dims)
    if ndims != 1:
        raise ValueError('Line plots are for 1 dimensional DataArrays. '
        'Passed DataArray has {} dimensions'.format(ndims))

    # Was an axis passed in?
    try:
        ax = kwargs.pop('ax')
    except KeyError:
        ax = plt.gca()

    xlabel, x = list(darray.indexes.items())[0]

    ax.plot(x, darray.values, *args, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(darray.name)

    return ax


def plot_imshow(darray, add_colorbar=True, *args, **kwargs):
    """
    Image plot of 2d DataArray using matplotlib / pylab.

    Wraps matplotlib.pyplot.imshow

    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional
    ax : matplotlib axes object
        If not passed, uses the current axis
    args, kwargs
        Additional arguments to matplotlib.pyplot.imshow
    add_colorbar : Boolean
        Adds colorbar to axis

    Examples
    --------

    """
    import matplotlib.pyplot as plt

    # Was an axis passed in?
    try:
        ax = kwargs.pop('ax')
    except KeyError:
        ax = plt.gca()

    # Seems strange that ylab comes first
    try:
        ylab, xlab = darray.dims
    except ValueError:
        raise ValueError('Line plots are for 2 dimensional DataArrays. '
        'Passed DataArray has {} dimensions'.format(len(darray.dims)))

    # Need these as Numpy arrays for colormesh
    x = darray[xlab].values
    y = darray[ylab].values
    z = darray.values

    ax.imshow(z, extent=[x.min(), x.max(), y.min(), y.max()],
            *args, **kwargs)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if add_colorbar:
        # mesh contains color mapping
        mesh = ax.pcolormesh(x, y, z)
        plt.colorbar(mesh, ax=ax)

    return ax


# TODO - Could refactor this to avoid duplicating plot_image logic above
def plot_contourf(darray, add_colorbar=True, *args, **kwargs):
    """
    Contour plot
    """
    import matplotlib.pyplot as plt

    # Was an axis passed in?
    try:
        ax = kwargs.pop('ax')
    except KeyError:
        ax = plt.gca()

    # Seems strange that ylab comes first
    try:
        ylab, xlab = darray.dims
    except ValueError:
        raise ValueError('Contour plots are for 2 dimensional DataArrays. '
        'Passed DataArray has {} dimensions'.format(len(darray.dims)))

    # Need these as Numpy arrays for colormesh
    x = darray[xlab].values
    y = darray[ylab].values
    z = darray.values

    ax.contourf(x, y, z, *args, **kwargs)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if add_colorbar:
        # Contains color mapping
        mesh = ax.pcolormesh(x, y, z)
        plt.colorbar(mesh, ax=ax)

    return ax


def plot_hist(darray, *args, **kwargs):
    """
    Histogram of DataArray using matplotlib / pylab.
    Plots N dimensional arrays by first flattening the array.

    Wraps matplotlib.pyplot.hist

    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional
    ax : matplotlib axes object
        If not passed, uses the current axis
    args, kwargs
        Additional arguments to matplotlib.pyplot.imshow

    Examples
    --------

    """
    import matplotlib.pyplot as plt

    # Was an axis passed in?
    try:
        ax = kwargs.pop('ax')
    except KeyError:
        ax = plt.gca()

    ax.hist(np.ravel(darray), *args, **kwargs)

    ax.set_ylabel('Count')

    if darray.name is not None:
        ax.set_title('Histogram of {}'.format(darray.name))

    return ax
