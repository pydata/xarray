"""
Plotting functions are implemented here and also monkeypatched into
the DataArray class
"""

import pkg_resources
import functools

import numpy as np
import pandas as pd

from ..core.utils import is_uniform_spaced


# TODO - implement this
class FacetGrid():
    pass


# Maybe more appropriate to keep this in .utils
def _right_dtype(arr, types):
    """
    Is the numpy array a sub dtype of anything in types?
    """
    return any(np.issubdtype(arr.dtype, t) for t in types)


def _ensure_plottable(*args):
    """
    Raise exception if there is anything in args that can't be plotted on
    an axis.
    """
    plottypes = [np.floating, np.integer, np.timedelta64, np.datetime64]

    # Lists need to be converted to np.arrays here.
    if not any(_right_dtype(np.array(x), plottypes) for x in args):
        raise TypeError('Plotting requires coordinates to be numeric '
                        'or dates.')


def _load_default_cmap(fname='default_colormap.csv'):
    """
    Returns viridis color map
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Not sure what the first arg here should be
    f = pkg_resources.resource_stream(__name__, fname)
    cm_data = pd.read_csv(f, header=None).values

    return LinearSegmentedColormap.from_list('viridis', cm_data)


def plot(darray, ax=None, rtol=0.01, **kwargs):
    """
    Default plot of DataArray using matplotlib / pylab.

    Calls xray plotting function based on the dimensions of
    the array:

    =============== =========== ===========================
    Dimensions      Coordinates Plotting function
    --------------- ----------- ---------------------------
    1                           :py:meth:`xray.DataArray.line`
    2               Uniform     :py:meth:`xray.DataArray.imshow`
    2               Irregular   :py:meth:`xray.DataArray.contourf`
    Anything else               :py:meth:`xray.DataArray.hist`
    =============== =========== ===========================

    Parameters
    ----------
    darray : DataArray
    ax : matplotlib axes, optional
        If None, uses the current axis
    rtol : number, optional
        Relative tolerance used to determine if the indexes
        are uniformly spaced. Usually a small positive number.
    **kwargs : optional
        Additional keyword arguments to matplotlib

    """
    ndims = len(darray.dims)

    if ndims == 1:
        plotfunc = line
    elif ndims == 2:
        indexes = darray.indexes.values()
        if all(is_uniform_spaced(i, rtol=rtol) for i in indexes):
            plotfunc = imshow
        else:
            plotfunc = contourf
    else:
        plotfunc = hist

    kwargs['ax'] = ax
    return plotfunc(darray, **kwargs)


# This function signature should not change so that it can use
# matplotlib format strings
def line(darray, *args, **kwargs):
    """
    Line plot of 1 dimensional DataArray index against values

    Wraps matplotlib.pyplot.plot

    Parameters
    ----------
    darray : DataArray
        Must be 1 dimensional
    ax : matplotlib axes, optional
        If not passed, uses the current axis
    *args, **kwargs : optional
        Additional arguments to matplotlib.pyplot.plot

    """
    import matplotlib.pyplot as plt

    ndims = len(darray.dims)
    if ndims != 1:
        raise ValueError('Line plots are for 1 dimensional DataArrays. '
                         'Passed DataArray has {} dimensions'.format(ndims))

    # Ensures consistency with .plot method
    ax = kwargs.pop('ax', None)

    if ax is None:
        ax = plt.gca()

    xlabel, x = list(darray.indexes.items())[0]

    _ensure_plottable([x])

    primitive = ax.plot(x, darray, *args, **kwargs)

    ax.set_xlabel(xlabel)

    if darray.name is not None:
        ax.set_ylabel(darray.name)

    # Rotate dates on xlabels
    if np.issubdtype(x.dtype, np.datetime64):
        for label in ax.get_xticklabels():
            label.set_rotation('vertical')

    return primitive


def hist(darray, ax=None, **kwargs):
    """
    Histogram of DataArray

    Wraps matplotlib.pyplot.hist

    Plots N dimensional arrays by first flattening the array.

    Parameters
    ----------
    darray : DataArray
        Can be any dimension
    ax : matplotlib axes, optional
        If not passed, uses the current axis
    **kwargs : optional
        Additional keyword arguments to matplotlib.pyplot.hist

    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    no_nan = np.ravel(darray.values)
    no_nan = no_nan[pd.notnull(no_nan)]

    primitive = ax.hist(no_nan, **kwargs)

    ax.set_ylabel('Count')

    if darray.name is not None:
        ax.set_title('Histogram of {}'.format(darray.name))

    return primitive


def _update_axes_limits(ax, xincrease, yincrease):
    """
    Update axes in place to increase or decrease
    For use in _plot2d
    """
    if xincrease is None:
        pass
    elif xincrease:
        ax.set_xlim(sorted(ax.get_xlim()))
    elif not xincrease:
        ax.set_xlim(sorted(ax.get_xlim(), reverse=True))

    if yincrease is None:
        pass
    elif yincrease:
        ax.set_ylim(sorted(ax.get_ylim()))
    elif not yincrease:
        ax.set_ylim(sorted(ax.get_ylim(), reverse=True))


# Decorator will fill this list
_plot2dlist = []

def _plot2d(plotfunc):
    """
    Decorator for common 2d plotting logic.
    """
    commondoc = '''
    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional
    ax : matplotlib axes object, optional
        If None, uses the current axis
    xincrease : None, True, or False, optional
        Should the values on the x axes be increasing from left to right?
        if None, use the default for the matplotlib function
    yincrease : None, True, or False, optional
        Should the values on the y axes be increasing from top to bottom?
        if None, use the default for the matplotlib function
    add_colorbar : Boolean, optional
        Adds colorbar to axis
    **kwargs : optional
        Additional arguments to wrapped matplotlib function

    Returns
    -------
    artist :
        The same type of primitive artist that the wrapped matplotlib
        function returns
    '''

    # Build on the original docstring
    plotfunc.__doc__ = '\n'.join((plotfunc.__doc__, commondoc))

    # global list of 2d plotting functions
    _plot2dlist.append(plotfunc)

    @functools.wraps(plotfunc)
    def wrapper(darray, ax=None, xincrease=None, yincrease=None,
                add_colorbar=True, **kwargs):
        # All 2d plots in xray share this function signature

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        try:
            ylab, xlab = darray.dims
        except ValueError:
            raise ValueError('{} plots are for 2 dimensional DataArrays. '
                             'Passed DataArray has {} dimensions'
                             .format(plotfunc.__name__, len(darray.dims)))

        # some plotting functions only know how to handle ndarrays
        x = darray[xlab].values
        y = darray[ylab].values
        z = np.ma.MaskedArray(darray.values, pd.isnull(darray.values))

        _ensure_plottable(x, y)

        cmap = kwargs.pop('cmap', _load_default_cmap())

        ax, primitive = plotfunc(x, y, z, ax=ax, cmap=cmap, **kwargs)

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        if add_colorbar:
            plt.colorbar(primitive, ax=ax)

        _update_axes_limits(ax, xincrease, yincrease)

        return primitive
    return wrapper


@_plot2d
def imshow(x, y, z, ax, **kwargs):
    """
    Image plot of 2d DataArray using matplotlib / pylab

    Wraps matplotlib.pyplot.imshow

    ..warning::

        This function needs uniformly spaced coordinates to
        properly label the axes. Call DataArray.plot() to check.

    The pixels are centered on the coordinates values. Ie, if the coordinate
    value is 3.2 then the pixels for those coordinates will be centered on 3.2.
    """
    # Centering the pixels- Assumes uniform spacing
    xstep = (x[1] - x[0]) / 2.0
    ystep = (y[1] - y[0]) / 2.0
    left, right = x[0] - xstep, x[-1] + xstep
    bottom, top = y[-1] + ystep, y[0] - ystep

    defaults = {'extent': [left, right, bottom, top],
                'aspect': 'auto',
                'interpolation': 'nearest',
                }

    # Allow user to override these defaults
    defaults.update(kwargs)

    primitive = ax.imshow(z, **defaults)

    return ax, primitive


@_plot2d
def contour(x, y, z, ax, **kwargs):
    """
    Contour plot of 2d DataArray

    Wraps matplotlib.pyplot.contour
    """
    primitive = ax.contour(x, y, z, **kwargs)
    return ax, primitive


@_plot2d
def contourf(x, y, z, ax, **kwargs):
    """
    Filled contour plot of 2d DataArray

    Wraps matplotlib.pyplot.contourf
    """
    primitive = ax.contourf(x, y, z, **kwargs)
    return ax, primitive


def _infer_interval_breaks(coord):
    """
    >>> _infer_interval_breaks(np.arange(5))
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
    """
    coord = np.asarray(coord)
    deltas = 0.5 * (coord[1:] - coord[:-1])
    first = coord[0] - deltas[0]
    last = coord[-1] + deltas[-1]
    return np.r_[[first], coord[:-1] + deltas, [last]]


@_plot2d
def pcolormesh(x, y, z, ax, **kwargs):
    """
    Pseudocolor plot of 2d DataArray

    Wraps matplotlib.pyplot.pcolormesh
    """
    x = _infer_interval_breaks(x)
    y = _infer_interval_breaks(y)

    primitive = ax.pcolormesh(x, y, z, **kwargs)

    # by default, pcolormesh picks "round" values for bounds
    # this results in ugly looking plots with lots of surrounding whitespace
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])

    return ax, primitive


class _PlotMethods(object):
    '''
    Plotmethods class
    '''

    # Basically this is a bunch of hacking to put in the dataarray as the
    # first arg.

    def __init__(self, DataArray_instance):
        self._da = DataArray_instance
        #for f in _plot2dlist:
        #    setattr(self, f.__name__, self._2dplot(f))

    def __call__(self, *args, **kwargs):
        return plot(self._da, *args, **kwargs)

    def _2dplot(self, plotfunc):
        @functools.wraps(plotfunc)
        def inner(self, ax=None, xincrease=None, yincrease=None,
                  add_colorbar=True, **kwargs):
            return plotfunc(self._da, ax=ax, xincrease=xincrease,
                    yincrease=yincrease, add_colorbar=add_colorbar,
                    **kwargs)
        return inner

    @functools.wraps(hist)
    def hist(self, ax=None, **kwargs):
        return hist(self._da, ax=None, **kwargs)

    @functools.wraps(line)
    def line(self, *args, **kwargs):
        return line(self._da, *args, **kwargs)

    @functools.wraps(plot)
    def plot(darray, ax=None, rtol=0.01, **kwargs):
        return plot(self._da, ax=ax, rtol=rtol, **kwargs)

    @functools.wraps(contour)
    def contour(self, ax=None, xincrease=None, yincrease=None,
              add_colorbar=True, **kwargs):
        return contour(self._da, ax=ax, xincrease=xincrease,
                    yincrease=yincrease, add_colorbar=add_colorbar,
                    **kwargs)

    @functools.wraps(contourf)
    def contourf(self, ax=None, xincrease=None, yincrease=None,
              add_colorbar=True, **kwargs):
        return contourf(self._da, ax=ax, xincrease=xincrease,
                    yincrease=yincrease, add_colorbar=add_colorbar,
                    **kwargs)

    @functools.wraps(imshow)
    def imshow(self, ax=None, xincrease=None, yincrease=None,
              add_colorbar=True, **kwargs):
        return imshow(self._da, ax=ax, xincrease=xincrease,
                    yincrease=yincrease, add_colorbar=add_colorbar,
                    **kwargs)

    @functools.wraps(pcolormesh)
    def pcolormesh(self, ax=None, xincrease=None, yincrease=None,
              add_colorbar=True, **kwargs):
        return pcolormesh(self._da, ax=ax, xincrease=xincrease,
                    yincrease=yincrease, add_colorbar=add_colorbar,
                    **kwargs)
