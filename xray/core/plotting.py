"""
Plotting functions are implemented here and also monkeypatched in to
DataArray and DataSet classes
"""

import functools
import numpy as np

from .utils import is_uniform_spaced

# TODO - Is there a better way to import matplotlib in the function?
# Other piece of duplicated logic is the checking for axes.
# Decorators don't preserve the argument names
# But if all the plotting methods have same signature...


class FacetGrid():
    pass


def plot(darray, ax=None, rtol=0.01, **kwargs):
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
        If None, uses the current axis
    rtol : relative tolerance
        Relative tolerance used to determine if the indexes
        are uniformly spaced
    kwargs
        Additional keyword arguments to matplotlib
    """
    ndims = len(darray.dims)

    if ndims == 1:
        plotfunc = plot_line
    elif ndims == 2:
        if all(is_uniform_spaced(i, rtol=rtol) for i in darray.indexes.values()):
            plotfunc = plot_imshow
        else:
            plotfunc = plot_contourf
    else:
        plotfunc = plot_hist

    kwargs['ax'] = ax
    return plotfunc(darray, **kwargs)


# This function signature should not change so that it can pass format
# strings
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

    # Ensures consistency with .plot method
    try:
        ax = kwargs.pop('ax')
    except KeyError:
        ax = None

    if ax is None:
        ax = plt.gca()

    xlabel, x = list(darray.indexes.items())[0]

    ax.plot(x, darray.values, *args, **kwargs)

    ax.set_xlabel(xlabel)

    if darray.name is not None:
        ax.set_ylabel(darray.name)

    return ax


def plot_imshow(darray, ax=None, add_colorbar=True, *args, **kwargs):
    """
    Image plot of 2d DataArray using matplotlib / pylab.

    Warning: This function needs sorted, uniformly spaced coordinates to
    properly label the axes.

    Wraps matplotlib.pyplot.imshow

    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional
    ax : matplotlib axes object
        If None, uses the current axis
    args, kwargs
        Additional arguments to matplotlib.pyplot.imshow
    add_colorbar : Boolean
        Adds colorbar to axis

    Details
    -------
    The pixels are centered on the coordinates values. Ie, if the coordinate
    value is 3.2 then the pixel for that data point will be centered on 3.2.

    Examples
    --------

    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    # Seems strange that ylab comes first
    try:
        ylab, xlab = darray.dims
    except ValueError:
        raise ValueError('Image plots are for 2 dimensional DataArrays. '
        'Passed DataArray has {} dimensions'.format(len(darray.dims)))

    x = darray[xlab]
    y = darray[ylab]

    # Use to center the pixels- Assumes uniform spacing
    xstep = (x[1] - x[0]) / 2.0
    ystep = (y[1] - y[0]) / 2.0
    left, right = x[0] - xstep, x[-1] + xstep
    bottom, top = y[-1] + ystep, y[0] - ystep

    ax.imshow(darray, extent=[left, right, bottom, top],
            interpolation='nearest', *args, **kwargs)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    #plt.colorbar(image, ax=ax)

    return ax


# TODO - Could refactor this to avoid duplicating plot_image logic above
def plot_contourf(darray, ax=None, add_colorbar=True, **kwargs):
    """
    Contour plot
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    try:
        ylab, xlab = darray.dims
    except ValueError:
        raise ValueError('Contour plots are for 2 dimensional DataArrays. '
        'Passed DataArray has {} dimensions'.format(len(darray.dims)))

    # Need arrays here?
    #x = darray[xlab].values
    #y = darray[ylab].values
    #z = darray.values

    #ax.contourf(x, y, z, *args, **kwargs)

    x = darray[xlab]
    y = darray[ylab]

    contours = ax.contourf(x, y, darray, **kwargs)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if add_colorbar:
        plt.colorbar(contours, ax=ax)

    return ax


def plot_hist(darray, ax=None, **kwargs):
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
    kwargs
        Additional arguments to matplotlib.pyplot.hist

    Examples
    --------

    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    ax.hist(np.ravel(darray), **kwargs)

    ax.set_ylabel('Count')

    if darray.name is not None:
        ax.set_title('Histogram of {}'.format(darray.name))

    return ax
