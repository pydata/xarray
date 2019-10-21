import itertools
import textwrap
import warnings
from datetime import datetime
from inspect import getfullargspec
from typing import Any, Iterable, Mapping, Tuple, Union

import numpy as np
import pandas as pd

from ..core.options import OPTIONS
from ..core.utils import is_scalar

try:
    import nc_time_axis  # noqa: F401

    nc_time_axis_available = True
except ImportError:
    nc_time_axis_available = False

ROBUST_PERCENTILE = 2.0


def import_seaborn():
    """import seaborn and handle deprecation of apionly module"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            import seaborn.apionly as sns

            if (
                w
                and issubclass(w[-1].category, UserWarning)
                and ("seaborn.apionly module" in str(w[-1].message))
            ):
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
        pd.plotting.register_matplotlib_converters()
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
        extend = "both"
    elif extend_min:
        extend = "min"
    elif extend_max:
        extend = "max"
    else:
        extend = "neither"
    return extend


def _build_discrete_cmap(cmap, levels, extend, filled):
    """
    Build a discrete colormap and normalization of the data.
    """
    import matplotlib as mpl

    if not filled:
        # non-filled contour plots
        extend = "max"

    if extend == "both":
        ext_n = 2
    elif extend in ["min", "max"]:
        ext_n = 1
    else:
        ext_n = 0

    n_colors = len(levels) + ext_n - 1
    pal = _color_palette(cmap, n_colors)

    new_cmap, cnorm = mpl.colors.from_levels_and_colors(levels, pal, extend=extend)
    # copy the old cmap name, for easier testing
    new_cmap.name = getattr(cmap, "name", cmap)

    return new_cmap, cnorm


def _color_palette(cmap, n_colors):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors_i = np.linspace(0, 1.0, n_colors)
    if isinstance(cmap, (list, tuple)):
        # we have a list of colors
        cmap = ListedColormap(cmap, N=n_colors)
        pal = cmap(colors_i)
    elif isinstance(cmap, str):
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


def _determine_cmap_params(
    plot_data,
    vmin=None,
    vmax=None,
    cmap=None,
    center=None,
    robust=False,
    extend=None,
    levels=None,
    filled=True,
    norm=None,
):
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

    # save state; needed later
    vmin_was_none = vmin is None
    vmax_was_none = vmax is None

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

    # now check norm and harmonize with vmin, vmax
    if norm is not None:
        if norm.vmin is None:
            norm.vmin = vmin
        else:
            if not vmin_was_none and vmin != norm.vmin:
                raise ValueError(
                    "Cannot supply vmin and a norm" + " with a different vmin."
                )
            vmin = norm.vmin

        if norm.vmax is None:
            norm.vmax = vmax
        else:
            if not vmax_was_none and vmax != norm.vmax:
                raise ValueError(
                    "Cannot supply vmax and a norm" + " with a different vmax."
                )
            vmax = norm.vmax

    # if BoundaryNorm, then set levels
    if isinstance(norm, mpl.colors.BoundaryNorm):
        levels = norm.boundaries

    # Choose default colormaps if not provided
    if cmap is None:
        if divergent:
            cmap = OPTIONS["cmap_divergent"]
        else:
            cmap = OPTIONS["cmap_sequential"]

    # Handle discrete levels
    if levels is not None and norm is None:
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

    if levels is not None or isinstance(norm, mpl.colors.BoundaryNorm):
        cmap, newnorm = _build_discrete_cmap(cmap, levels, extend, filled)
        norm = newnorm if norm is None else norm

    return dict(
        vmin=vmin, vmax=vmax, cmap=cmap, extend=extend, levels=levels, norm=norm
    )


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
            "Dimension names must be None or unique strings, but imshow was "
            "passed x=%r, y=%r, and rgb=%r." % (x, y, rgb)
        )
    for label in not_none:
        if label not in darray.dims:
            raise ValueError(f"{label!r} is not a dimension")

    # Then calculate rgb dimension if certain and check validity
    could_be_color = [
        label
        for label in darray.dims
        if darray[label].size in (3, 4) and label not in (x, y)
    ]
    if rgb is None and not could_be_color:
        raise ValueError(
            "A 3-dimensional array was passed to imshow(), but there is no "
            "dimension that could be color.  At least one dimension must be "
            "of size 3 (RGB) or 4 (RGBA), and not given as x or y."
        )
    if rgb is None and len(could_be_color) == 1:
        rgb = could_be_color[0]
    if rgb is not None and darray[rgb].size not in (3, 4):
        raise ValueError(
            "Cannot interpret dim %r of size %s as RGB or RGBA."
            % (rgb, darray[rgb].size)
        )

    # If rgb dimension is still unknown, there must be two or three dimensions
    # in could_be_color.  We therefore warn, and use a heuristic to break ties.
    if rgb is None:
        assert len(could_be_color) in (2, 3)
        rgb = could_be_color[-1]
        warnings.warn(
            "Several dimensions of this array could be colors.  Xarray "
            "will use the last possible dimension (%r) to match "
            "matplotlib.pyplot.imshow.  You can pass names of x, y, "
            "and/or rgb dimensions to override this guess." % rgb
        )
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
            raise ValueError("DataArray must be 2d")
        y, x = darray.dims
    elif x is None:
        if y not in darray.dims and y not in darray.coords:
            raise ValueError("y must be a dimension name if x is not supplied")
        x = darray.dims[0] if y == darray.dims[1] else darray.dims[1]
    elif y is None:
        if x not in darray.dims and x not in darray.coords:
            raise ValueError("x must be a dimension name if y is not supplied")
        y = darray.dims[0] if x == darray.dims[1] else darray.dims[1]
    elif any(k not in darray.coords and k not in darray.dims for k in (x, y)):
        raise ValueError("x and y must be coordinate variables")
    return x, y


def get_axis(figsize, size, aspect, ax):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if figsize is not None:
        if ax is not None:
            raise ValueError("cannot provide both `figsize` and " "`ax` arguments")
        if size is not None:
            raise ValueError("cannot provide both `figsize` and " "`size` arguments")
        _, ax = plt.subplots(figsize=figsize)
    elif size is not None:
        if ax is not None:
            raise ValueError("cannot provide both `size` and `ax` arguments")
        if aspect is None:
            width, height = mpl.rcParams["figure.figsize"]
            aspect = width / height
        figsize = (size * aspect, size)
        _, ax = plt.subplots(figsize=figsize)
    elif aspect is not None:
        raise ValueError("cannot provide `aspect` argument without `size`")

    if ax is None:
        ax = plt.gca()

    return ax


def label_from_attrs(da, extra=""):
    """ Makes informative labels if variable metadata (attrs) follows
        CF conventions. """

    if da.attrs.get("long_name"):
        name = da.attrs["long_name"]
    elif da.attrs.get("standard_name"):
        name = da.attrs["standard_name"]
    elif da.name is not None:
        name = da.name
    else:
        name = ""

    if da.attrs.get("units"):
        units = " [{}]".format(da.attrs["units"])
    else:
        units = ""

    return "\n".join(textwrap.wrap(name + extra + units, 30))


def _interval_to_mid_points(array):
    """
    Helper function which returns an array
    with the Intervals' mid points.
    """

    return np.array([x.mid for x in array])


def _interval_to_bound_points(array):
    """
    Helper function which returns an array
    with the Intervals' boundaries.
    """

    array_boundaries = np.array([x.left for x in array])
    array_boundaries = np.concatenate((array_boundaries, np.array([array[-1].right])))

    return array_boundaries


def _interval_to_double_bound_points(xarray, yarray):
    """
    Helper function to deal with a xarray consisting of pd.Intervals. Each
    interval is replaced with both boundaries. I.e. the length of xarray
    doubles. yarray is modified so it matches the new shape of xarray.
    """

    xarray1 = np.array([x.left for x in xarray])
    xarray2 = np.array([x.right for x in xarray])

    xarray = list(itertools.chain.from_iterable(zip(xarray1, xarray2)))
    yarray = list(itertools.chain.from_iterable(zip(yarray, yarray)))

    return xarray, yarray


def _resolve_intervals_2dplot(val, func_name):
    """
    Helper function to replace the values of a coordinate array containing
    pd.Interval with their mid-points or - for pcolormesh - boundaries which
    increases length by 1.
    """
    label_extra = ""
    if _valid_other_type(val, [pd.Interval]):
        if func_name == "pcolormesh":
            val = _interval_to_bound_points(val)
        else:
            val = _interval_to_mid_points(val)
            label_extra = "_center"

    return val, label_extra


def _valid_other_type(x, types):
    """
    Do all elements of x have a type from types?
    """
    return all(any(isinstance(el, t) for t in types) for el in np.ravel(x))


def _valid_numpy_subdtype(x, numpy_types):
    """
    Is any dtype from numpy_types superior to the dtype of x?
    """
    # If any of the types given in numpy_types is understood as numpy.generic,
    # all possible x will be considered valid.  This is probably unwanted.
    for t in numpy_types:
        assert not np.issubdtype(np.generic, t)

    return any(np.issubdtype(x.dtype, t) for t in numpy_types)


def _ensure_plottable(*args):
    """
    Raise exception if there is anything in args that can't be plotted on an
    axis by matplotlib.
    """
    numpy_types = [np.floating, np.integer, np.timedelta64, np.datetime64]
    other_types = [datetime]
    try:
        import cftime

        cftime_datetime = [cftime.datetime]
    except ImportError:
        cftime_datetime = []
    other_types = other_types + cftime_datetime
    for x in args:
        if not (
            _valid_numpy_subdtype(np.array(x), numpy_types)
            or _valid_other_type(np.array(x), other_types)
        ):
            raise TypeError(
                "Plotting requires coordinates to be numeric "
                "or dates of type np.datetime64, "
                "datetime.datetime, cftime.datetime or "
                "pd.Interval."
            )
        if (
            _valid_other_type(np.array(x), cftime_datetime)
            and not nc_time_axis_available
        ):
            raise ImportError(
                "Plotting of arrays of cftime.datetime "
                "objects or arrays indexed by "
                "cftime.datetime objects requires the "
                "optional `nc-time-axis` (v1.2.0 or later) "
                "package."
            )


def _is_numeric(arr):
    numpy_types = [np.floating, np.integer]
    return _valid_numpy_subdtype(arr, numpy_types)


def _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params):
    plt = import_matplotlib_pyplot()
    cbar_kwargs.setdefault("extend", cmap_params["extend"])
    if cbar_ax is None:
        cbar_kwargs.setdefault("ax", ax)
    else:
        cbar_kwargs.setdefault("cax", cbar_ax)

    cbar = plt.colorbar(primitive, **cbar_kwargs)

    return cbar


def _rescale_imshow_rgb(darray, vmin, vmax, robust):
    assert robust or vmin is not None or vmax is not None
    # TODO: remove when min numpy version is bumped to 1.13
    # There's a cyclic dependency via DataArray, so we can't import from
    # xarray.ufuncs in global scope.
    from xarray.ufuncs import maximum, minimum

    # Calculate vmin and vmax automatically for `robust=True`
    if robust:
        if vmax is None:
            vmax = np.nanpercentile(darray, 100 - ROBUST_PERCENTILE)
        if vmin is None:
            vmin = np.nanpercentile(darray, ROBUST_PERCENTILE)
    # If not robust and one bound is None, calculate the default other bound
    # and check that an interval between them exists.
    elif vmax is None:
        vmax = 255 if np.issubdtype(darray.dtype, np.integer) else 1
        if vmax < vmin:
            raise ValueError(
                "vmin=%r is less than the default vmax (%r) - you must supply "
                "a vmax > vmin in this case." % (vmin, vmax)
            )
    elif vmin is None:
        vmin = 0
        if vmin > vmax:
            raise ValueError(
                "vmax=%r is less than the default vmin (0) - you must supply "
                "a vmin < vmax in this case." % vmax
            )
    # Scale interval [vmin .. vmax] to [0 .. 1], with darray as 64-bit float
    # to avoid precision loss, integer over/underflow, etc with extreme inputs.
    # After scaling, downcast to 32-bit float.  This substantially reduces
    # memory usage after we hand `darray` off to matplotlib.
    darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "xarray.ufuncs", PendingDeprecationWarning)
        return minimum(maximum(darray, 0), 1)


def _update_axes(
    ax,
    xincrease,
    yincrease,
    xscale=None,
    yscale=None,
    xticks=None,
    yticks=None,
    xlim=None,
    ylim=None,
):
    """
    Update axes with provided parameters
    """
    if xincrease is None:
        pass
    elif xincrease and ax.xaxis_inverted():
        ax.invert_xaxis()
    elif not xincrease and not ax.xaxis_inverted():
        ax.invert_xaxis()

    if yincrease is None:
        pass
    elif yincrease and ax.yaxis_inverted():
        ax.invert_yaxis()
    elif not yincrease and not ax.yaxis_inverted():
        ax.invert_yaxis()

    # The default xscale, yscale needs to be None.
    # If we set a scale it resets the axes formatters,
    # This means that set_xscale('linear') on a datetime axis
    # will remove the date labels. So only set the scale when explicitly
    # asked to. https://github.com/matplotlib/matplotlib/issues/8740
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def _is_monotonic(coord, axis=0):
    """
    >>> _is_monotonic(np.array([0, 1, 2]))
    True
    >>> _is_monotonic(np.array([2, 1, 0]))
    True
    >>> _is_monotonic(np.array([0, 2, 1]))
    False
    """
    if coord.shape[axis] < 3:
        return True
    else:
        n = coord.shape[axis]
        delta_pos = coord.take(np.arange(1, n), axis=axis) >= coord.take(
            np.arange(0, n - 1), axis=axis
        )
        delta_neg = coord.take(np.arange(1, n), axis=axis) <= coord.take(
            np.arange(0, n - 1), axis=axis
        )
        return np.all(delta_pos) or np.all(delta_neg)


def _infer_interval_breaks(coord, axis=0, check_monotonic=False):
    """
    >>> _infer_interval_breaks(np.arange(5))
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
    >>> _infer_interval_breaks([[0, 1], [3, 4]], axis=1)
    array([[-0.5,  0.5,  1.5],
           [ 2.5,  3.5,  4.5]])
    """
    coord = np.asarray(coord)

    if check_monotonic and not _is_monotonic(coord, axis=axis):
        raise ValueError(
            "The input coordinate is not sorted in increasing "
            "order along axis %d. This can lead to unexpected "
            "results. Consider calling the `sortby` method on "
            "the input DataArray. To plot data with categorical "
            "axes, consider using the `heatmap` function from "
            "the `seaborn` statistical plotting library." % axis
        )

    deltas = 0.5 * np.diff(coord, axis=axis)
    if deltas.size == 0:
        deltas = np.array(0.0)
    first = np.take(coord, [0], axis=axis) - np.take(deltas, [0], axis=axis)
    last = np.take(coord, [-1], axis=axis) + np.take(deltas, [-1], axis=axis)
    trim_last = tuple(
        slice(None, -1) if n == axis else slice(None) for n in range(coord.ndim)
    )
    return np.concatenate([first, coord[trim_last] + deltas, last], axis=axis)


def _process_cmap_cbar_kwargs(
    func,
    data,
    cmap=None,
    colors=None,
    cbar_kwargs: Union[Iterable[Tuple[str, Any]], Mapping[str, Any]] = None,
    levels=None,
    **kwargs,
):
    """
    Parameters
    ==========
    func : plotting function
    data : ndarray,
        Data values

    Returns
    =======
    cmap_params

    cbar_kwargs
    """
    cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)

    if "contour" in func.__name__ and levels is None:
        levels = 7  # this is the matplotlib default

    # colors is mutually exclusive with cmap
    if cmap and colors:
        raise ValueError("Can't specify both cmap and colors.")

    # colors is only valid when levels is supplied or the plot is of type
    # contour or contourf
    if colors and (("contour" not in func.__name__) and (levels is None)):
        raise ValueError("Can only specify colors with contour or levels")

    # we should not be getting a list of colors in cmap anymore
    # is there a better way to do this test?
    if isinstance(cmap, (list, tuple)):
        raise ValueError(
            "Specifying a list of colors in cmap is deprecated. "
            "Use colors keyword instead."
        )

    cmap_kwargs = {
        "plot_data": data,
        "levels": levels,
        "cmap": colors if colors else cmap,
        "filled": func.__name__ != "contour",
    }

    cmap_args = getfullargspec(_determine_cmap_params).args
    cmap_kwargs.update((a, kwargs[a]) for a in cmap_args if a in kwargs)
    cmap_params = _determine_cmap_params(**cmap_kwargs)

    return cmap_params, cbar_kwargs
