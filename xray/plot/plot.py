"""
Use this module directly:
    import xray.plot as xplt

Or use the methods on a DataArray:
    DataArray.plot._____
"""

from __future__ import division
import pkg_resources
import functools
from textwrap import dedent
from itertools import cycle
from distutils.version import LooseVersion
import warnings

import numpy as np
import pandas as pd

from ..core.utils import is_uniform_spaced
from ..core.pycompat import basestring


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


def _infer_xy_labels(plotfunc, darray, x, y):
    """
    Determine x and y labels when some are missing. For use in _plot2d

    darray is a 2 dimensional data array.
    """
    dims = list(darray.dims)

    if len(dims) != 2:
        raise ValueError('{type} plots are for 2 dimensional DataArrays. '
                         'Passed DataArray has {ndim} dimensions'
                         .format(type=plotfunc.__name__, ndim=len(dims)))

    if x and x not in dims:
        raise KeyError('{0} is not a dimension of this DataArray. Use '
                       '{1} or {2} for x'
                       .format(x, *dims))

    if y and y not in dims:
        raise KeyError('{0} is not a dimension of this DataArray. Use '
                       '{1} or {2} for y'
                       .format(y, *dims))

    # Get label names
    if x and y:
        xlab = x
        ylab = y
    elif x and not y:
        xlab = x
        del dims[dims.index(x)]
        ylab = dims.pop()
    elif y and not x:
        ylab = y
        del dims[dims.index(y)]
        xlab = dims.pop()
    else:
        ylab, xlab = dims

    return xlab, ylab


def plot(darray, ax=None, rtol=0.01, **kwargs):
    """
    Default plot of DataArray using matplotlib / pylab.

    Calls xray plotting function based on the dimensions of
    darray.squeeze()

    =============== =========== ===========================
    Dimensions      Coordinates Plotting function
    --------------- ----------- ---------------------------
    1                           :py:func:`xray.plot.line`
    2               Uniform     :py:func:`xray.plot.imshow`
    2               Irregular   :py:func:`xray.plot.contourf`
    Anything else               :py:func:`xray.plot.hist`
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
    darray = darray.squeeze()

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
                         'Passed DataArray has {ndims} '
                         'dimensions'.format(ndims=ndims))

    # Ensures consistency with .plot method
    ax = kwargs.pop('ax', None)

    if ax is None:
        ax = plt.gca()

    xlabel, x = list(darray.indexes.items())[0]

    _ensure_plottable([x])

    primitive = ax.plot(x, darray, *args, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_title(darray._title_for_slice())

    if darray.name is not None:
        ax.set_ylabel(darray.name)

    # Rotate dates on xlabels
    if np.issubdtype(x.dtype, np.datetime64):
        plt.gcf().autofmt_xdate()

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
        ax.set_title('Histogram of {0}'.format(darray.name))

    return primitive


def plotter_1d(darray, plot_method='plot', *args, **kwargs):
    """
    Bar plot of 1 dimensional DataArray index against values

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
                         'Passed DataArray has {ndims} '
                         'dimensions'.format(ndims=ndims))

    # Ensures consistency with .plot method
    ax = kwargs.pop('ax', plt.gca())

    xlabel, x = list(darray.indexes.items())[0]

    _ensure_plottable([x])

    try:
        plotter = ax.__getattribute__(plot_method)
    except AttributeError:
        msg = 'Axes does not have plotting method {}'.format(plot_method)
        raise AttributeError(msg)

    primitive = plotter(x, darray, *args, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_title(darray._title_for_slice())

    if darray.name is not None:
        ax.set_ylabel(darray.name)

    # Rotate dates on xlabels
    if np.issubdtype(x.dtype, np.datetime64):
        plt.gcf().autofmt_xdate()

    return primitive


def plotter_data(darray, plot_method='hist', **kwargs):
    """
    Bar plot of 1 dimensional DataArray index against values

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
                         'Passed DataArray has {ndims} '
                         'dimensions'.format(ndims=ndims))

    # Ensures consistency with .plot method
    ax = kwargs.pop('ax', plt.gca())

    try:
        plotter = ax.__getattribute__(plot_method)
    except AttributeError:
        msg = 'Axes does not have plotting method {}'.format(plot_method)
        raise AttributeError(msg)

    no_nan = np.ravel(darray.values)
    no_nan = no_nan[pd.notnull(no_nan)]

    primitive = plotter(no_nan, **kwargs)

    # ax.set_xlabel(xlabel)
    ax.set_title(darray._title_for_slice())

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


def _determine_cmap_params(plot_data, vmin=None, vmax=None, cmap=None,
                           center=None, robust=False, extend=None,
                           levels=None, filled=True, cnorm=None):
    """
    Use some heuristics to set good defaults for colorbar and range.

    Adapted from Seaborn:
    https://github.com/mwaskom/seaborn/blob/v0.6/seaborn/matrix.py#L158

    Parameters
    ==========
    plot_data: Numpy array
        Doesn't handle xray objects

    Returns
    =======
    cmap_params : dict
        Use depends on the type of the plotting function
    """
    ROBUST_PERCENTILE = 2.0
    import matplotlib as mpl

    calc_data = np.ravel(plot_data[~pd.isnull(plot_data)])

    if vmin is None:
        vmin = np.percentile(calc_data, ROBUST_PERCENTILE) if robust else calc_data.min()
    if vmax is None:
        vmax = np.percentile(calc_data, 100 - ROBUST_PERCENTILE) if robust else calc_data.max()

    # Simple heuristics for whether these data should  have a divergent map
    divergent = ((vmin < 0) and (vmax > 0)) or center is not None

    # Now set center to 0 so math below makes sense
    if center is None:
        center = 0

    # A divergent map should be symmetric around the center value
    if divergent:
        vlim = max(abs(vmin - center), abs(vmax - center))
        vmin, vmax = -vlim, vlim

    # Now add in the centering value and set the limits
    vmin += center
    vmax += center

    # Choose default colormaps if not provided
    if cmap is None:
        if divergent:
            cmap = "RdBu_r"
        else:
            cmap = "viridis"

    # Allow viridis before matplotlib 1.5
    if cmap == "viridis":
        cmap = _load_default_cmap()

    # Handle discrete levels
    if levels is not None:
        if isinstance(levels, int):
            ticker = mpl.ticker.MaxNLocator(levels)
            levels = ticker.tick_values(vmin, vmax)
        vmin, vmax = levels[0], levels[-1]

    if extend is None:
        extend = _determine_extend(calc_data, vmin, vmax)

    if levels is not None:
        cmap, cnorm = _build_discrete_cmap(cmap, levels, extend, filled)

    return dict(vmin=vmin, vmax=vmax, cmap=cmap, extend=extend,
                levels=levels, cnorm=cnorm)


def _determine_extend(calc_data, vmin, vmax):
    extend_min = calc_data.min() < vmin
    extend_max = calc_data.max() > vmax
    if extend_min and extend_max:
        extend = 'both'
    elif extend_min:
        extend = 'min'
    elif extend_max:
        extend = 'max'
    else:
        extend = 'neither'
    return extend


def _color_palette(cmap, n_colors):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors_i = np.linspace(0, 1., n_colors)
    if isinstance(cmap, (list, tuple)):
        # we have a list of colors
        try:
            # first try to turn it into a palette with seaborn
            from seaborn.apionly import color_palette
            pal = color_palette(cmap, n_colors=n_colors)
        except ImportError:
            # if that fails, use matplotlib
            # in this case, is there any difference between mpl and seaborn?
            cmap = ListedColormap(cmap, N=n_colors)
            pal = cmap(colors_i)
    elif isinstance(cmap, basestring):
        # we have some sort of named palette
        try:
            # first try to turn it into a palette with seaborn
            from seaborn.apionly import color_palette
            pal = color_palette(cmap, n_colors=n_colors)
        except (ImportError, ValueError):
            # ValueError is raised when seaborn doesn't like a colormap
            # (e.g. jet). if that fails, use matplotlib
            try:
                # is this a matplotlib cmap?
                cmap = plt.get_cmap(cmap)
            except ValueError:
                # or maybe we just got a single color as a string
                cmap = ListedColormap([cmap], N=n_colors)
            pal = cmap(colors_i)
    else:
        # cmap better be a LinearSegmentedColormap (e.g. viridis)
        pal = cmap(colors_i)

    return pal

def _build_discrete_cmap(cmap, levels, extend, filled):
    """
    Build a discrete colormap and normalization of the data.
    """
    import matplotlib as mpl

    if not filled:
        # non-filled contour plots
        extend = 'neither'

    if extend == 'both':
        ext_n = 2
    elif extend in ['min', 'max']:
        ext_n = 1
    else:
        ext_n = 0

    n_colors = len(levels) + ext_n - 1
    pal = _color_palette(cmap, n_colors)

    new_cmap, cnorm = mpl.colors.from_levels_and_colors(
        levels, pal, extend=extend)
    # copy the old cmap name, for easier testing
    new_cmap.name = getattr(cmap, 'name', cmap)

    return new_cmap, cnorm


# MUST run before any 2d plotting functions are defined since
# _plot2d decorator adds them as methods here.
class _PlotMethods(object):
    """
    Enables use of xray.plot functions as attributes on a DataArray.
    For example, DataArray.plot.imshow
    """

    def __init__(self, DataArray_instance):
        self._da = DataArray_instance

    def __call__(self, ax=None, rtol=0.01, **kwargs):
        return plot(self._da, ax=ax, rtol=rtol, **kwargs)

    @functools.wraps(hist)
    def hist(self, ax=None, **kwargs):
        return hist(self._da, ax=ax, **kwargs)

    @functools.wraps(line)
    def line(self, *args, **kwargs):
        return line(self._da, *args, **kwargs)

    @functools.wraps(plotter_1d)
    def plotter_1d(self, plot_method='plot', *args, **kwargs):
        return plotter_1d(self._da, plot_method=plot_method, *args, **kwargs)

    @functools.wraps(plotter_data)
    def plotter_data(self, plot_method='hist', **kwargs):
        return plotter_data(self._da, plot_method=plot_method, **kwargs)


def _plot2d(plotfunc):
    """
    Decorator for common 2d plotting logic

    Also adds the 2d plot method to class _PlotMethods
    """
    commondoc = """
    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional
    x : string, optional
        Coordinate for x axis. If None use darray.dims[1]
    y : string, optional
        Coordinate for y axis. If None use darray.dims[0]
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
    add_labels : Boolean, optional
        Use xray metadata to label axes
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        one of these values may be ignored. If discrete levels are provided as
        an explicit list, both of these values are ignored.
    cmap : matplotlib colormap name or object, optional
        The mapping from data values to color space. If not provided, this
        will be either be ``viridis`` (if the function infers a sequential
        dataset) or ``RdBu_r`` (if the function infers a diverging dataset).
        When when `Seaborn` is installed, ``cmap`` may also be a `seaborn`
        color palette. If ``cmap`` is seaborn color palette and the plot type
        is not ``contour`` or ``contourf``, ``levels`` must also be specified.
    colors : discrete colors to plot, optional
        A single color or a list of colors. If the plot type is not ``contour``
        or ``contourf``, the ``levels`` argument is required.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    extend : {'neither', 'both', 'min', 'max'}, optional
        How to draw arrows extending the colorbar beyond its limits. If not
        provided, extend is inferred from vmin, vmax and the data limits.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals.
    **kwargs : optional
        Additional arguments to wrapped matplotlib function

    Returns
    -------
    artist :
        The same type of primitive artist that the wrapped matplotlib
        function returns
    """

    # Build on the original docstring
    plotfunc.__doc__ = '\n'.join((plotfunc.__doc__, commondoc))

    @functools.wraps(plotfunc)
    def newplotfunc(darray, x=None, y=None, ax=None, xincrease=None,
                    yincrease=None, add_colorbar=True, add_labels=True,
                    vmin=None, vmax=None, cmap=None, center=None, robust=False,
                    extend=None, levels=None, colors=None,
                    **kwargs):
        # All 2d plots in xray share this function signature.
        # Method signature below should be consistent.

        import matplotlib.pyplot as plt

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

        if ax is None:
            ax = plt.gca()

        xlab, ylab = _infer_xy_labels(plotfunc=plotfunc, darray=darray, x=x, y=y)

        # better to pass the ndarrays directly to plotting functions
        xval = darray[xlab].values
        yval = darray[ylab].values
        zval = darray.to_masked_array(copy=False)

        # May need to transpose for correct x, y labels
        if xlab == darray.dims[0]:
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

        # This allows the user to pass in a custom norm coming via kwargs
        kwargs.setdefault('norm', cmap_params['cnorm'])

        ax, primitive = plotfunc(xval, yval, zval, ax=ax,
                                 cmap=cmap_params['cmap'],
                                 vmin=cmap_params['vmin'],
                                 vmax=cmap_params['vmax'],
                                 **kwargs)

        # Label the plot with metadata
        if add_labels:
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_title(darray._title_for_slice())

        if add_colorbar:
            cbar = plt.colorbar(primitive, ax=ax, extend=cmap_params['extend'])
            if darray.name and add_labels:
                cbar.set_label(darray.name)

        _update_axes_limits(ax, xincrease, yincrease)

        return primitive

    # For use as DataArray.plot.plotmethod
    @functools.wraps(newplotfunc)
    def plotmethod(_PlotMethods_obj, x=None, y=None, ax=None, xincrease=None, yincrease=None,
                   add_colorbar=True, add_labels=True, vmin=None, vmax=None, cmap=None,
                   colors=None, center=None, robust=False, extend=None, levels=None,
                   **kwargs):
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
    Image plot of 2d DataArray using matplotlib / pylab

    Wraps matplotlib.pyplot.imshow

    ..note::

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
