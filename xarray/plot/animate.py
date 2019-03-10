"""
Use this module directly:
    import xarray.animate as xanim

Or use the methods on a DataArray:
    DataArray.plot.animate._____

Or supply an ``animate`` keyword
argument to a normal plotting function:
    DataArray.plot._____(animate='__')
"""

import datetime
import functools

import numpy as np
import pandas as pd

from .utils import _infer_line_data, _infer_plot_type
from .utils import (_ensure_plottable, _interval_to_mid_points, _update_axes,
                    _valid_other_type, get_axis, _rotate_date_xlabels,
                    _check_animate, _transpose_before_animation)


def animate(darray, animate=None, **kwargs):
    """
    Default plot of DataArray using animatplot.

    Calls xarray animated plotting function based on the dimensions of
    darray.squeeze()

    =============== ===========================
    Dimensions      Plotting function
    --------------- ---------------------------
    2               :py:func:`xarray.plot.animate.line`
    Anything else   Not yet implemented
    =============== ===========================

    Parameters
    ----------
    darray : DataArray
    row : string, optional
        If passed, make row faceted plots on this dimension name
    col : string, optional
        If passed, make column faceted plots on this dimension name
    hue : string, optional
        If passed, make faceted line plots with hue on this dimension name
    col_wrap : integer, optional
        Use together with ``col`` to wrap faceted plots
    animate: str
        Dimension or coord in the DataArray over which to animate.
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
    return _AnimateMethods(darray, animate=animate, **kwargs)


def line(darray, animate=None, **kwargs):
    """
    Line plot of DataArray index against values

    Wraps :func:`animatplot:animatplot.blocks.Line`

    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional.
    animate: str
        Dimension or coord in the DataArray over which to animate.
        ``animatplot.blocks.Line`` will be used to animate the plot over this
        dimension.
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
    x, y : string, optional
        Dimensions or coordinates for x, y axis.
        Only one of these may be specified.
        The other coordinate plots values from the DataArray on which this
        plot method is called.
    xscale, yscale : 'linear', 'symlog', 'log', 'logit', optional
        Specifies scaling for the x- and y-axes respectively
    xticks, yticks : Specify tick locations for x- and y-axes
    xlim, ylim : optional
        Specify x- and y-axes limits.
    xincrease : None, True, or False, optional
        Should the values on the x axes be increasing from left to right?
        if None, use the default for the matplotlib function.
    yincrease : None, True, or False, optional
        Should the values on the y axes be increasing from top to bottom?
        if None, use the default for the matplotlib function.
    **kwargs : optional
        Additional arguments to animatplot.blocks.Line

    """

    from animatplot.blocks import Line, Title
    from animatplot.animation import Animation

    row = kwargs.pop('row', None)
    col = kwargs.pop('col', None)
    if row or col:
        raise NotImplementedError

    hue = kwargs.pop('hue', None)
    if hue:
        raise NotImplementedError

    _check_animate(darray, animate)
    darray = _transpose_before_animation(darray, animate)

    ndims = len(darray[animate].dims)
    if ndims > 1:
        raise NotImplementedError

    # Ensures consistency with .plot method
    figsize = kwargs.pop('figsize', None)
    aspect = kwargs.pop('aspect', None)
    size = kwargs.pop('size', None)
    ax = kwargs.pop('ax', None)
    hue = kwargs.pop('hue', None)
    x = kwargs.pop('x', None)
    y = kwargs.pop('y', None)
    xincrease = kwargs.pop('xincrease', None)  # default needs to be None
    yincrease = kwargs.pop('yincrease', None)
    xscale = kwargs.pop('xscale', None)  # default needs to be None
    yscale = kwargs.pop('yscale', None)
    xticks = kwargs.pop('xticks', None)
    yticks = kwargs.pop('yticks', None)
    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)
    _labels = kwargs.pop('_labels', True)

    ax = get_axis(figsize, size, aspect, ax)
    xplt, yplt, hueplt, xlabel, ylabel, huelabel = \
        _infer_line_data(darray, x, y, hue, animate)

    # Remove pd.Intervals if contained in xplt.values.
    if _valid_other_type(xplt.values, [pd.Interval]):
        # Is it a step plot? (see matplotlib.Axes.step)
        if kwargs.get('linestyle', '').startswith('steps-'):
            raise NotImplementedError
        else:
            xplt_val = _interval_to_mid_points(xplt.values)
            yplt_val = yplt.values
            xlabel += '_center'
    else:
        xplt_val = xplt.values
        yplt_val = yplt.values

    _ensure_plottable(xplt_val, yplt_val)

    fps = kwargs.pop('fps', 10)
    timeline = _create_timeline(darray, animate, fps)

    if ylim is None:
        ylim = [np.min(yplt_val), np.max(yplt_val)]

    # animatplot assumes that the x positions might vary over time too
    line_block = Line(xplt_val, yplt_val, ax=ax, t_axis=-1, **kwargs)

    if _labels:
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Would be nicer if we had something like in GH issue #266
        frame_titles = [darray[{animate: i}]._title_for_slice()
                        for i in range(len(timeline))]
        title_block = Title(frame_titles, ax=ax)

    _rotate_date_xlabels(xplt, ax)

    _update_axes(ax, xincrease, yincrease, xscale, yscale,
                 xticks, yticks, xlim, ylim)

    anim = Animation([line_block, title_block], timeline=timeline)
    # TODO I think ax should be passed to timeline_slider args
    # but that just plots a single huge timeline and no line plot?!
    anim.controls(timeline_slider_args={'text': animate, 'valfmt': '%s'})
    return anim


def _create_timeline(darray, animate, fps):

    from animatplot.animation import Timeline

    if animate in darray.coords:
        t_array = darray.coords[animate].values

        # Format datetimes in a nicer way
        if isinstance(t_array[0], datetime.date) \
                or np.issubdtype(t_array.dtype, np.datetime64):
            t_array = [pd.to_datetime(date) for date in t_array]

    else:  # animating over a dimension without coords
        t_array = np.arange(darray.sizes[animate])

    if darray.coords[animate].attrs.get('units'):
        units = ' [{}]'.format(darray.coords[animate].attrs['units'])
    else:
        units = ''
    return Timeline(t_array, units=units, fps=fps)


class _AnimateMethods:
    """
    Enables use of xarray.plot.animate functions as attributes on a DataArray.
    For example, DataArray.plot.animate.line
    """

    def __init__(self, darray, animate, **kwargs):
        self._da = darray
        self._animate = animate

    def __call__(self, **kwargs):
        return animate(self._da, self._animate, **kwargs)

    @functools.wraps(line)
    def line(self, animate, **kwargs):
        return line(self._da, animate, **kwargs)
