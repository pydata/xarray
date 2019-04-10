"""
Use this module directly:
    import xarray.animate as xanim

Or supply an ``animate`` keyword
argument to a normal plotting function:
    DataArray.plot._____(animate='__')
"""

import datetime

import numpy as np
import pandas as pd

from .utils import (_infer_line_data, _ensure_plottable, _update_axes,
                    get_axis, _rotate_date_xlabels, _check_animate,
                    _transpose_before_animation, import_matplotlib_pyplot)


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
        raise NotImplementedError("Animated FacetGrids not yet implemented")

    hue = kwargs.pop('hue', None)
    if hue:
        raise NotImplementedError("Animating multiple lines at once is not yet"
                                  "implemented")

    _check_animate(darray, animate)
    darray = _transpose_before_animation(darray, animate)

    ndims = len(darray[animate].dims)
    if ndims > 2:
        raise ValueError('Animated line plots are for 2- or 3-dimensional '
                         'DataArrays. Passed DataArray has {ndims} '
                         'dimensions'.format(ndims=ndims+1))

    # Ensures consistency with .plot method
    figsize = kwargs.pop('figsize', None)
    aspect = kwargs.pop('aspect', None)
    size = kwargs.pop('size', None)
    ax = kwargs.pop('ax', None)
    hue = kwargs.pop('hue', None)
    x = kwargs.pop('x', None)
    y = kwargs.pop('y', None)
    linestyle = kwargs.get('linestyle', '')
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
    xplt_val, yplt_val, hueplt, xlabel, ylabel, huelabel = \
        _infer_line_data(darray, x, y, hue, animate, linestyle)

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

    _rotate_date_xlabels(xplt_val, ax)

    _update_axes(ax, xincrease, yincrease, xscale, yscale,
                 xticks, yticks, xlim, ylim)

    anim = Animation([line_block, title_block], timeline=timeline)
    anim.controls(timeline_slider_args={'text': animate, 'valfmt': '%s'})

    # Stop subsequent matplotlib plotting calls plotting onto the pause button!
    plt = import_matplotlib_pyplot()
    plt.sca(ax)

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
