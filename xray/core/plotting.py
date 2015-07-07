"""
Plotting functions are implemented here and also monkeypatched in to
DataArray and DataSet classes
"""

import numpy as np


# TODO - Is there a better way to import matplotlib in the function?
# Decorators don't preserve the argument names
# But if all the plotting methods have same signature...


class FacetGrid():
    pass


def plot(darray, ax=None, *args, **kwargs):
    """
    Default plot of DataArray using matplotlib / pylab.

    Parameters
    ----------
    darray : DataArray
        Must be 1 dimensional
    ax : matplotlib axes object
        If not passed, uses plt.gca()
    args, kwargs
        Additional arguments to matplotlib
    """
    defaults = {1: plot_line, 2: plot_image}
    ndims = len(darray.dims)

    if ndims in defaults:
        plotfunc = defaults[ndims]
    else:
        plotfunc = plot_hist

    return plotfunc(darray, ax, *args, **kwargs)


def plot_line(darray, ax=None, *args, **kwargs):
    """
    Line plot of DataArray using matplotlib / pylab.

    Parameters
    ----------
    darray : DataArray
        Must be 1 dimensional
    ax : matplotlib axes object
        If not passed, uses plt.gca()
    """
    import matplotlib.pyplot as plt

    ndims = len(darray.dims)
    if ndims != 1:
        raise ValueError('Line plots are for 1 dimensional DataArrays. '
        'Passed DataArray has {} dimensions'.format(ndims))

    if ax is None:
        ax = plt.gca()

    xlabel, x = list(darray.indexes.items())[0]

    ax.plot(x, darray.values, *args, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(darray.name)

    return ax


def plot_imshow(darray, ax=None, add_colorbar=True, *args, **kwargs):
    """
    Image plot of 2d DataArray using matplotlib / pylab.

    Parameters
    ----------
    darray : DataArray
        Must be 1 dimensional
    ax : matplotlib axes object
        If not passed, uses plt.gca()
    add_colorbar : Boolean
        Adds colorbar to axis
    args, kwargs
        Additional arguments to matplotlib

    """
    import matplotlib.pyplot as plt

    if ax is None:
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
def plot_contourf(darray, ax=None, add_colorbar=True, *args, **kwargs):
    """
    Contour plot
    """
    import matplotlib.pyplot as plt

    if ax is None:
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


def plot_hist(darray, ax=None, *args, **kwargs):
    """
    Histogram of DataArray using matplotlib / pylab.
    
    Uses numpy.ravel to first flatten the array.

    Parameters
    ----------
    darray : DataArray
        Can be any dimensions
    ax : matplotlib axes object
        If not passed, uses plt.gca()
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    ax.hist(np.ravel(darray))

    ax.set_ylabel('Count')
    ax.set_title('Histogram of {}'.format(darray.name))

    return ax
