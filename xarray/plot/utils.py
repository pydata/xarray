from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
import textwrap

from ..core.pycompat import basestring
from ..core.utils import is_scalar
from ..core.options import OPTIONS

ROBUST_PERCENTILE = 2.0


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
        try:
            from pandas.plotting import register_matplotlib_converters
            register_matplotlib_converters()
        except ImportError:
            # register_matplotlib_converters new in pandas 0.22
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
        cmap = ListedColormap(cmap, N=n_colors)
        pal = cmap(colors_i)
    elif isinstance(cmap, basestring):
        # we have some sort of named palette
        try:
            # is this a matplotlib cmap?
            cmap = plt.get_cmap(cmap)
            pal = cmap(colors_i)
        except ValueError:
            # ValueError happens when mpl doesn't like a colormap, try seaborn
            try:
                from seaborn.apionly import color_palette
                pal = color_palette(cmap, n_colors=n_colors)
            except (ValueError, ImportError):
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
    import matplotlib as mpl

    calc_data = np.ravel(plot_data[np.isfinite(plot_data)])

    # Handle all-NaN input data gracefully
    if calc_data.size == 0:
        # Arbitrary default for when all values are NaN
        calc_data = np.array(0.0)

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
            cmap = OPTIONS['cmap_divergent']
        else:
            cmap = OPTIONS['cmap_sequential']

    # Handle discrete levels
    if levels is not None:
        if is_scalar(levels):
            if user_minmax:
                levels = np.linspace(vmin, vmax, levels)
            elif levels == 1:
                levels = np.asarray([(vmin + vmax) / 2])
            else:
                # N in MaxNLocator refers to bins, not ticks
                ticker = mpl.ticker.MaxNLocator(levels - 1)
                levels = ticker.tick_values(vmin, vmax)
        vmin, vmax = levels[0], levels[-1]

    if extend is None:
        extend = _determine_extend(calc_data, vmin, vmax)

    if levels is not None:
        cmap, norm = _build_discrete_cmap(cmap, levels, extend, filled)

    return dict(vmin=vmin, vmax=vmax, cmap=cmap, extend=extend,
                levels=levels, norm=norm)


def _infer_xy_labels_3d(darray, x, y, rgb):
    """
    Determine x and y labels for showing RGB images.

    Attempts to infer which dimension is RGB/RGBA by size and order of dims.

    """
    assert rgb is None or rgb != x
    assert rgb is None or rgb != y
    # Start by detecting and reporting invalid combinations of arguments
    assert darray.ndim == 3
    not_none = [a for a in (x, y, rgb) if a is not None]
    if len(set(not_none)) < len(not_none):
        raise ValueError(
            'Dimension names must be None or unique strings, but imshow was '
            'passed x=%r, y=%r, and rgb=%r.' % (x, y, rgb))
    for label in not_none:
        if label not in darray.dims:
            raise ValueError('%r is not a dimension' % (label,))

    # Then calculate rgb dimension if certain and check validity
    could_be_color = [label for label in darray.dims
                      if darray[label].size in (3, 4) and label not in (x, y)]
    if rgb is None and not could_be_color:
        raise ValueError(
            'A 3-dimensional array was passed to imshow(), but there is no '
            'dimension that could be color.  At least one dimension must be '
            'of size 3 (RGB) or 4 (RGBA), and not given as x or y.')
    if rgb is None and len(could_be_color) == 1:
        rgb = could_be_color[0]
    if rgb is not None and darray[rgb].size not in (3, 4):
        raise ValueError('Cannot interpret dim %r of size %s as RGB or RGBA.'
                         % (rgb, darray[rgb].size))

    # If rgb dimension is still unknown, there must be two or three dimensions
    # in could_be_color.  We therefore warn, and use a heuristic to break ties.
    if rgb is None:
        assert len(could_be_color) in (2, 3)
        rgb = could_be_color[-1]
        warnings.warn(
            'Several dimensions of this array could be colors.  Xarray '
            'will use the last possible dimension (%r) to match '
            'matplotlib.pyplot.imshow.  You can pass names of x, y, '
            'and/or rgb dimensions to override this guess.' % rgb)
    assert rgb is not None

    # Finally, we pick out the red slice and delegate to the 2D version:
    return _infer_xy_labels(darray.isel(**{rgb: 0}), x, y)


def _infer_xy_labels(darray, x, y, imshow=False, rgb=None):
    """
    Determine x and y labels. For use in _plot2d

    darray must be a 2 dimensional data array, or 3d for imshow only.
    """
    assert x is None or x != y
    if imshow and darray.ndim == 3:
        return _infer_xy_labels_3d(darray, x, y, rgb)

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


def label_from_attrs(da):
    ''' Makes informative labels if variable metadata (attrs) follows
        CF conventions. '''

    if da.attrs.get('long_name'):
        name = da.attrs['long_name']
    elif da.attrs.get('standard_name'):
        name = da.attrs['standard_name']
    elif da.name is not None:
        name = da.name
    else:
        name = ''

    if da.attrs.get('units'):
        units = ' [{}]'.format(da.attrs['units'])
    else:
        units = ''

    return '\n'.join(textwrap.wrap(name + units, 30))
