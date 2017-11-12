"""
Use this module directly:
    import xarray.plot as xplt

Or use the methods on a DataArray:
    DataArray.plot._____
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import warnings

import numpy as np
import pandas as pd
from datetime import datetime

from .utils import (_determine_cmap_params, _infer_xy_labels, get_axis,
                    import_matplotlib_pyplot)
from .facetgrid import FacetGrid
from xarray.core.pycompat import basestring


def _valid_numpy_subdtype(x, numpy_types):
    """
    Is any dtype from numpy_types superior to the dtype of x?
    """
    # If any of the types given in numpy_types is understood as numpy.generic,
    # all possible x will be considered valid.  This is probably unwanted.
    for t in numpy_types:
        assert not np.issubdtype(np.generic, t)

    return any(np.issubdtype(x.dtype, t) for t in numpy_types)


def _valid_other_type(x, types):
    """
    Do all elements of x have a type from types?
    """
    if not np.issubdtype(np.generic, x.dtype):
        return False
    else:
        return all(any(isinstance(el, t) for t in types) for el in np.ravel(x))


def _ensure_plottable(*args):
    """
    Raise exception if there is anything in args that can't be plotted on an
    axis.
    """
    numpy_types = [np.floating, np.integer, np.timedelta64, np.datetime64]
    other_types = [datetime]

    for x in args:
        if not (_valid_numpy_subdtype(np.array(x), numpy_types)
                or _valid_other_type(np.array(x), other_types)):
            raise TypeError('Plotting requires coordinates to be numeric '
                            'or dates.')

def _easy_facetgrid(darray, plotfunc, x, y, row=None, col=None,
                    col_wrap=None, sharex=True, sharey=True, aspect=None,
                    size=None, subplot_kws=None, **kwargs):
    """
    Convenience method to call xarray.plot.FacetGrid from 2d plotting methods

    kwargs are the arguments to 2d plotting method
    """
    ax = kwargs.pop('ax', None)
    figsize = kwargs.pop('figsize', None)
    if ax is not None:
        raise ValueError("Can't use axes when making faceted plots.")
    if aspect is None:
        aspect = 1
    if size is None:
        size = 3
    elif figsize is not None:
        raise ValueError('cannot provide both `figsize` and `size` arguments')

    g = FacetGrid(data=darray, col=col, row=row, col_wrap=col_wrap,
                  sharex=sharex, sharey=sharey, figsize=figsize,
                  aspect=aspect, size=size, subplot_kws=subplot_kws)
    return g.map_dataarray(plotfunc, x, y, **kwargs)


def plot(darray, row=None, col=None, col_wrap=None, ax=None, rtol=0.01,
         subplot_kws=None, **kwargs):
    """
    Default plot of DataArray using matplotlib.pyplot.

    Calls xarray plotting function based on the dimensions of
    darray.squeeze()

    =============== ===========================
    Dimensions      Plotting function
    --------------- ---------------------------
    1               :py:func:`xarray.plot.line`
    2               :py:func:`xarray.plot.pcolormesh`
    Anything else   :py:func:`xarray.plot.hist`
    =============== ===========================

    Parameters
    ----------
    darray : DataArray
    row : string, optional
        If passed, make row faceted plots on this dimension name
    col : string, optional
        If passed, make column faceted plots on this dimension name
    col_wrap : integer, optional
        Use together with ``col`` to wrap faceted plots
    ax : matplotlib axes, optional
        If None, uses the current axis. Not applicable when using facets.
    rtol : number, optional
        Relative tolerance used to determine if the indexes
        are uniformly spaced. Usually a small positive number.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for matplotlib subplots. Only applies
        to FacetGrid plotting.
    **kwargs : optional
        Additional keyword arguments to matplotlib

    """
    darray = darray.squeeze()

    plot_dims = set(darray.dims)
    plot_dims.discard(row)
    plot_dims.discard(col)

    ndims = len(plot_dims)

    error_msg = ('Only 2d plots are supported for facets in xarray. '
                 'See the package `Seaborn` for more options.')

    if ndims == 1:
        if row or col:
            raise ValueError(error_msg)
        plotfunc = line
    elif ndims == 2:
        # Only 2d can FacetGrid
        kwargs['row'] = row
        kwargs['col'] = col
        kwargs['col_wrap'] = col_wrap
        kwargs['subplot_kws'] = subplot_kws

        plotfunc = pcolormesh
    else:
        if row or col:
            raise ValueError(error_msg)
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
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches.
        Mutually exclusive with ``size`` and ``ax``.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the width in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size.
        Height (in inches) of each plot. See also: ``aspect``.
    ax : matplotlib axes object, optional
        Axis on which to plot this figure. By default, use the current axis.
        Mutually exclusive with ``size`` and ``figsize``.
    *args, **kwargs : optional
        Additional arguments to matplotlib.pyplot.plot

    """
    plt = import_matplotlib_pyplot()

    ndims = len(darray.dims)
    if ndims != 1:
        raise ValueError('Line plots are for 1 dimensional DataArrays. '
                         'Passed DataArray has {ndims} '
                         'dimensions'.format(ndims=ndims))

    # Ensures consistency with .plot method
    figsize = kwargs.pop('figsize', None)
    aspect = kwargs.pop('aspect', None)
    size = kwargs.pop('size', None)
    ax = kwargs.pop('ax', None)

    ax = get_axis(figsize, size, aspect, ax)

    xlabel, = darray.dims
    x = darray.coords[xlabel]

    _ensure_plottable(x)

    primitive = ax.plot(x, darray, *args, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_title(darray._title_for_slice())

    if darray.name is not None:
        ax.set_ylabel(darray.name)

    # Rotate dates on xlabels
    if np.issubdtype(x.dtype, np.datetime64):
        plt.gcf().autofmt_xdate()

    return primitive


def hist(darray, figsize=None, size=None, aspect=None, ax=None, **kwargs):
    """
    Histogram of DataArray

    Wraps matplotlib.pyplot.hist

    Plots N dimensional arrays by first flattening the array.

    Parameters
    ----------
    darray : DataArray
        Can be any dimension
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches.
        Mutually exclusive with ``size`` and ``ax``.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the width in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size.
        Height (in inches) of each plot. See also: ``aspect``.
    ax : matplotlib axes object, optional
        Axis on which to plot this figure. By default, use the current axis.
        Mutually exclusive with ``size`` and ``figsize``.
    **kwargs : optional
        Additional keyword arguments to matplotlib.pyplot.hist

    """
    ax = get_axis(figsize, size, aspect, ax)

    no_nan = np.ravel(darray.values)
    no_nan = no_nan[pd.notnull(no_nan)]

    primitive = ax.hist(no_nan, **kwargs)

    ax.set_ylabel('Count')

    if darray.name is not None:
        ax.set_title('Histogram of {0}'.format(darray.name))

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


# MUST run before any 2d plotting functions are defined since
# _plot2d decorator adds them as methods here.
class _PlotMethods(object):
    """
    Enables use of xarray.plot functions as attributes on a DataArray.
    For example, DataArray.plot.imshow
    """

    def __init__(self, darray):
        self._da = darray

    def __call__(self, **kwargs):
        return plot(self._da, **kwargs)

    @functools.wraps(hist)
    def hist(self, ax=None, **kwargs):
        return hist(self._da, ax=ax, **kwargs)

    @functools.wraps(line)
    def line(self, *args, **kwargs):
        return line(self._da, *args, **kwargs)


def _plot2d(plotfunc):
    """
    Decorator for common 2d plotting logic

    Also adds the 2d plot method to class _PlotMethods
    """
    commondoc = """
    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional, unless creating faceted plots
    x : string, optional
        Coordinate for x axis. If None use darray.dims[1]
    y : string, optional
        Coordinate for y axis. If None use darray.dims[0]
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches.
        Mutually exclusive with ``size`` and ``ax``.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the width in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size.
        Height (in inches) of each plot. See also: ``aspect``.
    ax : matplotlib axes object, optional
        Axis on which to plot this figure. By default, use the current axis.
        Mutually exclusive with ``size`` and ``figsize``.
    row : string, optional
        If passed, make row faceted plots on this dimension name
    col : string, optional
        If passed, make column faceted plots on this dimension name
    col_wrap : integer, optional
        Use together with ``col`` to wrap faceted plots
    xincrease : None, True, or False, optional
        Should the values on the x axes be increasing from left to right?
        if None, use the default for the matplotlib function
    yincrease : None, True, or False, optional
        Should the values on the y axes be increasing from top to bottom?
        if None, use the default for the matplotlib function
    add_colorbar : Boolean, optional
        Adds colorbar to axis
    add_labels : Boolean, optional
        Use xarray metadata to label axes
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting one of these values will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    cmap : matplotlib colormap name or object, optional
        The mapping from data values to color space. If not provided, this
        will be either be ``viridis`` (if the function infers a sequential
        dataset) or ``RdBu_r`` (if the function infers a diverging dataset).
        When `Seaborn` is installed, ``cmap`` may also be a `seaborn`
        color palette. If ``cmap`` is seaborn color palette and the plot type
        is not ``contour`` or ``contourf``, ``levels`` must also be specified.
    colors : discrete colors to plot, optional
        A single color or a list of colors. If the plot type is not ``contour``
        or ``contourf``, the ``levels`` argument is required.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap. Setting it to ``False`` prevents use of a
        diverging colormap.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    extend : {'neither', 'both', 'min', 'max'}, optional
        How to draw arrows extending the colorbar beyond its limits. If not
        provided, extend is inferred from vmin, vmax and the data limits.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals. If an integer
        is provided, "nice" levels are chosen based on the data range: this can
        imply that the final number of levels is not exactly the expected one.
        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
        setting ``levels=np.linspace(vmin, vmax, N)``.
    infer_intervals : bool, optional
        Only applies to pcolormesh. If True, the coordinate intervals are
        passed to pcolormesh. If False, the original coordinates are used
        (this can be useful for certain map projections). The default is to
        always infer intervals, unless the mesh is irregular and plotted on
        a map projection.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for matplotlib subplots. Only applies
        to FacetGrid plotting.
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar.
    cbar_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the colorbar.
    **kwargs : optional
        Additional arguments to wrapped matplotlib function

    Returns
    -------
    artist :
        The same type of primitive artist that the wrapped matplotlib
        function returns
    """

    # Build on the original docstring
    plotfunc.__doc__ = '%s\n%s' % (plotfunc.__doc__, commondoc)

    @functools.wraps(plotfunc)
    def newplotfunc(darray, x=None, y=None, figsize=None, size=None,
                    aspect=None, ax=None, row=None, col=None,
                    col_wrap=None, xincrease=True, yincrease=True,
                    add_colorbar=None, add_labels=True, vmin=None, vmax=None,
                    cmap=None, center=None, robust=False, extend=None,
                    levels=None, infer_intervals=None, colors=None,
                    subplot_kws=None, cbar_ax=None, cbar_kwargs=None,
                    **kwargs):
        # All 2d plots in xarray share this function signature.
        # Method signature below should be consistent.

        # Decide on a default for the colorbar before facetgrids
        if add_colorbar is None:
            add_colorbar = plotfunc.__name__ != 'contour'

        # Handle facetgrids first
        if row or col:
            allargs = locals().copy()
            allargs.update(allargs.pop('kwargs'))

            # Need the decorated plotting function
            allargs['plotfunc'] = globals()[plotfunc.__name__]

            return _easy_facetgrid(**allargs)

        plt = import_matplotlib_pyplot()

        # colors is mutually exclusive with cmap
        if cmap and colors:
            raise ValueError("Can't specify both cmap and colors.")
        # colors is only valid when levels is supplied or the plot is of type
        # contour or contourf
        if colors and (('contour' not in plotfunc.__name__) and (not levels)):
            raise ValueError("Can only specify colors with contour or levels")
        # we should not be getting a list of colors in cmap anymore
        # is there a better way to do this test?
        if isinstance(cmap, (list, tuple)):
            warnings.warn("Specifying a list of colors in cmap is deprecated. "
                          "Use colors keyword instead.",
                          DeprecationWarning, stacklevel=3)

        xlab, ylab = _infer_xy_labels(darray=darray, x=x, y=y)

        # better to pass the ndarrays directly to plotting functions
        xval = darray[xlab].values
        yval = darray[ylab].values
        zval = darray.to_masked_array(copy=False)

        # May need to transpose for correct x, y labels
        # xlab may be the name of a coord, we have to check for dim names
        if darray[xlab].dims[-1] == darray.dims[0]:
            zval = zval.T

        _ensure_plottable(xval, yval)

        if 'contour' in plotfunc.__name__ and levels is None:
            levels = 7  # this is the matplotlib default

        cmap_kwargs = {'plot_data': zval.data,
                       'vmin': vmin,
                       'vmax': vmax,
                       'cmap': colors if colors else cmap,
                       'center': center,
                       'robust': robust,
                       'extend': extend,
                       'levels': levels,
                       'filled': plotfunc.__name__ != 'contour',
                       }

        cmap_params = _determine_cmap_params(**cmap_kwargs)

        if 'contour' in plotfunc.__name__:
            # extend is a keyword argument only for contour and contourf, but
            # passing it to the colorbar is sufficient for imshow and
            # pcolormesh
            kwargs['extend'] = cmap_params['extend']
            kwargs['levels'] = cmap_params['levels']

        if 'pcolormesh' == plotfunc.__name__:
            kwargs['infer_intervals'] = infer_intervals

        # This allows the user to pass in a custom norm coming via kwargs
        kwargs.setdefault('norm', cmap_params['norm'])

        if 'imshow' == plotfunc.__name__ and isinstance(aspect, basestring):
            # forbid usage of mpl strings
            raise ValueError("plt.imshow's `aspect` kwarg is not available "
                             "in xarray")

        ax = get_axis(figsize, size, aspect, ax)
        primitive = plotfunc(xval, yval, zval, ax=ax, cmap=cmap_params['cmap'],
                             vmin=cmap_params['vmin'],
                             vmax=cmap_params['vmax'],
                             **kwargs)

        # Label the plot with metadata
        if add_labels:
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_title(darray._title_for_slice())

        if add_colorbar:
            cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)
            cbar_kwargs.setdefault('extend', cmap_params['extend'])
            if cbar_ax is None:
                cbar_kwargs.setdefault('ax', ax)
            else:
                cbar_kwargs.setdefault('cax', cbar_ax)
            cbar = plt.colorbar(primitive, **cbar_kwargs)
            if darray.name and add_labels and 'label' not in cbar_kwargs:
                cbar.set_label(darray.name, rotation=90)
        elif cbar_ax is not None or cbar_kwargs is not None:
            # inform the user about keywords which aren't used
            raise ValueError("cbar_ax and cbar_kwargs can't be used with "
                             "add_colorbar=False.")

        _update_axes_limits(ax, xincrease, yincrease)

        return primitive

    # For use as DataArray.plot.plotmethod
    @functools.wraps(newplotfunc)
    def plotmethod(_PlotMethods_obj, x=None, y=None, figsize=None, size=None,
                   aspect=None, ax=None, row=None, col=None, col_wrap=None,
                   xincrease=True, yincrease=True, add_colorbar=None,
                   add_labels=True, vmin=None, vmax=None, cmap=None,
                   colors=None, center=None, robust=False, extend=None,
                   levels=None, infer_intervals=None, subplot_kws=None,
                   cbar_ax=None, cbar_kwargs=None, **kwargs):
        """
        The method should have the same signature as the function.

        This just makes the method work on Plotmethods objects,
        and passes all the other arguments straight through.
        """
        allargs = locals()
        allargs['darray'] = _PlotMethods_obj._da
        allargs.update(kwargs)
        for arg in ['_PlotMethods_obj', 'newplotfunc', 'kwargs']:
            del allargs[arg]
        return newplotfunc(**allargs)

    # Add to class _PlotMethods
    setattr(_PlotMethods, plotmethod.__name__, plotmethod)

    return newplotfunc


@_plot2d
def imshow(x, y, z, ax, **kwargs):
    """
    Image plot of 2d DataArray using matplotlib.pyplot

    Wraps matplotlib.pyplot.imshow

    ..note::

        This function needs uniformly spaced coordinates to
        properly label the axes. Call DataArray.plot() to check.

    The pixels are centered on the coordinates values. Ie, if the coordinate
    value is 3.2 then the pixels for those coordinates will be centered on 3.2.
    """

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('imshow requires 1D coordinates, try using '
                         'pcolormesh or contour(f)')

    # Centering the pixels- Assumes uniform spacing
    xstep = (x[1] - x[0]) / 2.0
    ystep = (y[1] - y[0]) / 2.0
    left, right = x[0] - xstep, x[-1] + xstep
    bottom, top = y[-1] + ystep, y[0] - ystep

    defaults = {'extent': [left, right, bottom, top],
                'origin': 'upper',
                'interpolation': 'nearest',
                }

    if not hasattr(ax, 'projection'):
        # not for cartopy geoaxes
        defaults['aspect'] = 'auto'

    # Allow user to override these defaults
    defaults.update(kwargs)

    primitive = ax.imshow(z, **defaults)

    return primitive


@_plot2d
def contour(x, y, z, ax, **kwargs):
    """
    Contour plot of 2d DataArray

    Wraps matplotlib.pyplot.contour
    """
    primitive = ax.contour(x, y, z, **kwargs)
    return primitive


@_plot2d
def contourf(x, y, z, ax, **kwargs):
    """
    Filled contour plot of 2d DataArray

    Wraps matplotlib.pyplot.contourf
    """
    primitive = ax.contourf(x, y, z, **kwargs)
    return primitive


def _infer_interval_breaks(coord, axis=0):
    """
    >>> _infer_interval_breaks(np.arange(5))
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
    >>> _infer_interval_breaks([[0, 1], [3, 4]], axis=1)
    array([[-0.5,  0.5,  1.5],
           [ 2.5,  3.5,  4.5]])
    """
    coord = np.asarray(coord)
    deltas = 0.5 * np.diff(coord, axis=axis)
    first = np.take(coord, [0], axis=axis) - np.take(deltas, [0], axis=axis)
    last = np.take(coord, [-1], axis=axis) + np.take(deltas, [-1], axis=axis)
    trim_last = tuple(slice(None, -1) if n == axis else slice(None)
                      for n in range(coord.ndim))
    return np.concatenate([first, coord[trim_last] + deltas, last], axis=axis)


@_plot2d
def pcolormesh(x, y, z, ax, infer_intervals=None, **kwargs):
    """
    Pseudocolor plot of 2d DataArray

    Wraps matplotlib.pyplot.pcolormesh
    """

    # decide on a default for infer_intervals (GH781)
    x = np.asarray(x)
    if infer_intervals is None:
        if hasattr(ax, 'projection'):
            if len(x.shape) == 1:
                infer_intervals = True
            else:
                infer_intervals = False
        else:
            infer_intervals = True

    if infer_intervals:
        if len(x.shape) == 1:
            x = _infer_interval_breaks(x)
            y = _infer_interval_breaks(y)
        else:
            # we have to infer the intervals on both axes
            x = _infer_interval_breaks(x, axis=1)
            x = _infer_interval_breaks(x, axis=0)
            y = _infer_interval_breaks(y, axis=1)
            y = _infer_interval_breaks(y, axis=0)

    primitive = ax.pcolormesh(x, y, z, **kwargs)

    # by default, pcolormesh picks "round" values for bounds
    # this results in ugly looking plots with lots of surrounding whitespace
    if not hasattr(ax, 'projection') and x.ndim == 1 and y.ndim == 1:
        # not a cartopy geoaxis
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])

    return primitive
