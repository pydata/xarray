from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pkg_resources
import warnings

import numpy as np
import pandas as pd

from ..core.pycompat import basestring
from ..core.utils import is_scalar


def _load_default_cmap(fname='default_colormap.csv'):
    """
    Returns viridis color map
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Not sure what the first arg here should be
    f = pkg_resources.resource_stream(__name__, fname)
    cm_data = pd.read_csv(f, header=None).values

    return LinearSegmentedColormap.from_list('viridis', cm_data)


def import_seaborn():
    '''import seaborn and handle deprecation of apionly module'''
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            import seaborn.apionly as sns
            if (w and issubclass(w[-1].category, UserWarning) and
                    ("seaborn.apionly module" in str(w[-1].message))):
                raise ImportError
        except ImportError:
            import seaborn as sns
        finally:
            warnings.resetwarnings()
    return sns


_registered = False


def register_pandas_datetime_converter_if_needed():
    # based on https://github.com/pandas-dev/pandas/pull/17710
    global _registered
    if not _registered:
        from pandas.tseries import converter
        converter.register()
        _registered = True


def import_matplotlib_pyplot():
    """Import pyplot as register appropriate converters."""
    register_pandas_datetime_converter_if_needed()
    import matplotlib.pyplot as plt
    return plt


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


def _build_discrete_cmap(cmap, levels, extend, filled):
    """
    Build a discrete colormap and normalization of the data.
    """
    import matplotlib as mpl

    if not filled:
        # non-filled contour plots
        extend = 'max'

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


def _color_palette(cmap, n_colors):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors_i = np.linspace(0, 1., n_colors)
    if isinstance(cmap, (list, tuple)):
        # we have a list of colors
        try:
            sns = import_seaborn()
        except ImportError:
            # if that fails, use matplotlib
            # in this case, is there any difference between mpl and seaborn?
            cmap = ListedColormap(cmap, N=n_colors)
            pal = cmap(colors_i)
        else:
            # first try to turn it into a palette with seaborn
            pal = sns.color_palette(cmap, n_colors=n_colors)
    elif isinstance(cmap, basestring):
        # we have some sort of named palette
        try:
            # first try to turn it into a palette with seaborn
            from seaborn.apionly import color_palette
            pal = color_palette(cmap, n_colors=n_colors)
        except (ImportError, ValueError):
            # ValueError is raised when seaborn doesn't like a colormap
            # (e.g. jet). If that fails, use matplotlib
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


# _determine_cmap_params is adapted from Seaborn:
# https://github.com/mwaskom/seaborn/blob/v0.6/seaborn/matrix.py#L158
# Used under the terms of Seaborn's license, see licenses/SEABORN_LICENSE.

def _determine_cmap_params(plot_data, vmin=None, vmax=None, cmap=None,
                           center=None, robust=False, extend=None,
                           levels=None, filled=True, norm=None):
    """
    Use some heuristics to set good defaults for colorbar and range.

    Parameters
    ==========
    plot_data: Numpy array
        Doesn't handle xarray objects

    Returns
    =======
    cmap_params : dict
        Use depends on the type of the plotting function
    """
    ROBUST_PERCENTILE = 2.0
    import matplotlib as mpl

    calc_data = np.ravel(plot_data[~pd.isnull(plot_data)])

    # Setting center=False prevents a divergent cmap
    possibly_divergent = center is not False

    # Set center to 0 so math below makes sense but remember its state
    center_is_none = False
    if center is None:
        center = 0
        center_is_none = True

    # Setting both vmin and vmax prevents a divergent cmap
    if (vmin is not None) and (vmax is not None):
        possibly_divergent = False

    # Setting vmin or vmax implies linspaced levels
    user_minmax = (vmin is not None) or (vmax is not None)

    # vlim might be computed below
    vlim = None

    if vmin is None:
        if robust:
            vmin = np.percentile(calc_data, ROBUST_PERCENTILE)
        else:
            vmin = calc_data.min()
    elif possibly_divergent:
        vlim = abs(vmin - center)

    if vmax is None:
        if robust:
            vmax = np.percentile(calc_data, 100 - ROBUST_PERCENTILE)
        else:
            vmax = calc_data.max()
    elif possibly_divergent:
        vlim = abs(vmax - center)

    if possibly_divergent:
        # kwargs not specific about divergent or not: infer defaults from data
        divergent = ((vmin < 0) and (vmax > 0)) or not center_is_none
    else:
        divergent = False

    # A divergent map should be symmetric around the center value
    if divergent:
        if vlim is None:
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
        if is_scalar(levels):
            if user_minmax or levels == 1:
                levels = np.linspace(vmin, vmax, levels)
            else:
                # N in MaxNLocator refers to bins, not ticks
                ticker = mpl.ticker.MaxNLocator(levels-1)
                levels = ticker.tick_values(vmin, vmax)
        vmin, vmax = levels[0], levels[-1]

    if extend is None:
        extend = _determine_extend(calc_data, vmin, vmax)

    if levels is not None:
        cmap, norm = _build_discrete_cmap(cmap, levels, extend, filled)

    return dict(vmin=vmin, vmax=vmax, cmap=cmap, extend=extend,
                levels=levels, norm=norm)


def _infer_xy_labels(darray, x, y):
    """
    Determine x and y labels. For use in _plot2d

    darray must be a 2 dimensional data array.
    """

    if x is None and y is None:
        if darray.ndim != 2:
            raise ValueError('DataArray must be 2d')
        y, x = darray.dims
    elif x is None:
        if y not in darray.dims:
            raise ValueError('y must be a dimension name if x is not supplied')
        x = darray.dims[0] if y == darray.dims[1] else darray.dims[1]
    elif y is None:
        if x not in darray.dims:
            raise ValueError('x must be a dimension name if y is not supplied')
        y = darray.dims[0] if x == darray.dims[1] else darray.dims[1]
    elif any(k not in darray.coords and k not in darray.dims for k in (x, y)):
        raise ValueError('x and y must be coordinate variables')
    return x, y


def get_axis(figsize, size, aspect, ax):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if figsize is not None:
        if ax is not None:
            raise ValueError('cannot provide both `figsize` and '
                             '`ax` arguments')
        if size is not None:
            raise ValueError('cannot provide both `figsize` and '
                             '`size` arguments')
        _, ax = plt.subplots(figsize=figsize)
    elif size is not None:
        if ax is not None:
            raise ValueError('cannot provide both `size` and `ax` arguments')
        if aspect is None:
            width, height = mpl.rcParams['figure.figsize']
            aspect = width / height
        figsize = (size * aspect, size)
        _, ax = plt.subplots(figsize=figsize)
    elif aspect is not None:
        raise ValueError('cannot provide `aspect` argument without `size`')

    if ax is None:
        ax = plt.gca()

    return ax
