"""
Use this module directly:
    import xarray.plot as xplt

Or use the methods on a DataArray or Dataset:
    DataArray.plot._____
    Dataset.plot._____
"""
import functools

import numpy as np
import pandas as pd

from ..core.alignment import broadcast
from .facetgrid import _easy_facetgrid
from .utils import (
    _add_colorbar,
    _assert_valid_xy,
    _ensure_plottable,
    _infer_interval_breaks,
    _infer_xy_labels,
    _is_numeric,
    _process_cmap_cbar_kwargs,
    _rescale_imshow_rgb,
    _resolve_intervals_1dplot,
    _resolve_intervals_2dplot,
    _update_axes,
    get_axis,
    import_matplotlib_pyplot,
    label_from_attrs,
)

# copied from seaborn
_MARKERSIZE_RANGE = np.array([18.0, 72.0])


def _infer_meta_data(darray, x, y, hue, hue_style, size):
    def _determine_hue(darray, hue, hue_style):
        """Find and determine what type of hue it is."""
        hue_is_numeric = _is_numeric(darray[hue].values)

        if hue_style is None:
            hue_style = "continuous" if hue_is_numeric else "discrete"
        elif hue_style not in ["discrete", "continuous"]:
            raise ValueError(
                "hue_style must be either None, 'discrete' or 'continuous'."
            )

        if not hue_is_numeric and (hue_style == "continuous"):
            raise ValueError(
                f"Cannot create a colorbar for a non numeric coordinate: {hue}"
            )

        hue_label = label_from_attrs(darray[hue])
        hue = darray[hue]

        return hue, hue_style, hue_label

    if x is not None and y is not None:
        raise ValueError("Cannot specify both x and y kwargs for line plots.")

    if x is not None:
        xplt = darray[x]
        yplt = darray
    elif y is not None:
        xplt = darray
        yplt = darray[y]
    xlabel = label_from_attrs(xplt)
    ylabel = label_from_attrs(yplt)

    if hue:
        hue, hue_style, hue_label = _determine_hue(darray, hue, hue_style)
    else:
        # Try finding a hue:
        _, hue = _infer_xy_labels(darray=yplt, x=xplt.name, y=hue)

        if hue:
            hue, hue_style, hue_label = _determine_hue(darray, hue, hue_style)
        else:
            hue_label = None
            hue = None

    if size:
        size_label = label_from_attrs(darray[size])
        size = darray[size]
    else:
        size_label = None
        size = None

    return dict(
        hue_label=hue_label,
        hue_style=hue_style,
        xlabel=xlabel,
        ylabel=ylabel,
        hue=hue,
        size=size,
        size_label=size_label,
    )


# copied from seaborn
def _parse_size(data, norm, width):
    """Parse sizes."""
    plt = import_matplotlib_pyplot

    if data is None:
        return None

    data = data.values.flatten()

    if not _is_numeric(data):
        levels = np.unique(data)
        numbers = np.arange(1, 1 + len(levels))[::-1]
    else:
        levels = numbers = np.sort(np.unique(data))

    min_width, max_width = width
    # width_range = min_width, max_width

    if norm is None:
        norm = plt.colors.Normalize()
    elif isinstance(norm, tuple):
        norm = plt.colors.Normalize(*norm)
    elif not isinstance(norm, plt.colors.Normalize):
        err = "``size_norm`` must be None, tuple, or Normalize object."
        raise ValueError(err)

    norm.clip = True
    if not norm.scaled():
        norm(np.asarray(numbers))
    # limits = norm.vmin, norm.vmax

    scl = norm(numbers)
    widths = np.asarray(min_width + scl * (max_width - min_width))
    if scl.mask.any():
        widths[scl.mask] = 0
    sizes = dict(zip(levels, widths))

    return pd.Series(sizes)


def _infer_scatter_data(
    darray, x, y, hue, size, size_norm, size_mapping=None, size_range=(1, 10)
):
    if x is not None and y is not None:
        raise ValueError("Cannot specify both x and y kwargs for scatter plots.")

    # Broadcast together all the chosen variables:
    if x is not None:
        to_broadcast = dict(x=darray[x], y=darray)
    elif y is not None:
        to_broadcast = dict(x=darray, y=darray[y])
    to_broadcast.update(
        {
            key: darray[value]
            for key, value in dict(hue=hue, size=size).items()
            if value in darray.dims
        }
    )
    broadcasted = dict(zip(to_broadcast.keys(), broadcast(*(to_broadcast.values()))))

    data = dict(x=broadcasted["x"], y=broadcasted["y"], sizes=None)
    data.update(hue=broadcasted.get("hue", None))

    if size:
        size = broadcasted["size"]

        if size_mapping is None:
            size_mapping = _parse_size(size, size_norm, size_range)

        data["sizes"] = size.copy(
            data=np.reshape(size_mapping.loc[size.values.ravel()].values, size.shape)
        )
        data["sizes_to_labels"] = pd.Series(size_mapping.index, index=size_mapping)

    return data


def _infer_line_data(darray, x, y, hue):

    ndims = len(darray.dims)

    if x is not None and y is not None:
        raise ValueError("Cannot specify both x and y kwargs for line plots.")

    if x is not None:
        _assert_valid_xy(darray, x, "x")

    if y is not None:
        _assert_valid_xy(darray, y, "y")

    if ndims == 1:
        huename = None
        hueplt = None
        huelabel = ""

        if x is not None:
            xplt = darray[x]
            yplt = darray

        elif y is not None:
            xplt = darray
            yplt = darray[y]

        else:  # Both x & y are None
            dim = darray.dims[0]
            xplt = darray[dim]
            yplt = darray

    else:
        if x is None and y is None and hue is None:
            raise ValueError("For 2D inputs, please specify either hue, x or y.")

        if y is None:
            xname, huename = _infer_xy_labels(darray=darray, x=x, y=hue)
            xplt = darray[xname]
            if xplt.ndim > 1:
                if huename in darray.dims:
                    otherindex = 1 if darray.dims.index(huename) == 0 else 0
                    otherdim = darray.dims[otherindex]
                    yplt = darray.transpose(otherdim, huename, transpose_coords=False)
                    xplt = xplt.transpose(otherdim, huename, transpose_coords=False)
                else:
                    raise ValueError(
                        "For 2D inputs, hue must be a dimension"
                        " i.e. one of " + repr(darray.dims)
                    )

            else:
                (xdim,) = darray[xname].dims
                (huedim,) = darray[huename].dims
                yplt = darray.transpose(xdim, huedim)

        else:
            yname, huename = _infer_xy_labels(darray=darray, x=y, y=hue)
            yplt = darray[yname]
            if yplt.ndim > 1:
                if huename in darray.dims:
                    otherindex = 1 if darray.dims.index(huename) == 0 else 0
                    otherdim = darray.dims[otherindex]
                    xplt = darray.transpose(otherdim, huename, transpose_coords=False)
                    yplt = yplt.transpose(otherdim, huename, transpose_coords=False)
                else:
                    raise ValueError(
                        "For 2D inputs, hue must be a dimension"
                        " i.e. one of " + repr(darray.dims)
                    )

            else:
                (ydim,) = darray[yname].dims
                (huedim,) = darray[huename].dims
                xplt = darray.transpose(ydim, huedim)

        huelabel = label_from_attrs(darray[huename])
        hueplt = darray[huename]

    return xplt, yplt, hueplt, huelabel


def plot(
    darray,
    row=None,
    col=None,
    col_wrap=None,
    ax=None,
    hue=None,
    rtol=0.01,
    subplot_kws=None,
    **kwargs,
):
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
    row : str, optional
        If passed, make row faceted plots on this dimension name
    col : str, optional
        If passed, make column faceted plots on this dimension name
    hue : str, optional
        If passed, make faceted line plots with hue on this dimension name
    col_wrap : int, optional
        Use together with ``col`` to wrap faceted plots
    ax : matplotlib.axes.Axes, optional
        If None, uses the current axis. Not applicable when using facets.
    rtol : float, optional
        Relative tolerance used to determine if the indexes
        are uniformly spaced. Usually a small positive number.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for matplotlib subplots.
    **kwargs : optional
        Additional keyword arguments to matplotlib

    """
    darray = darray.squeeze().compute()

    plot_dims = set(darray.dims)
    plot_dims.discard(row)
    plot_dims.discard(col)
    plot_dims.discard(hue)

    ndims = len(plot_dims)

    error_msg = (
        "Only 1d and 2d plots are supported for facets in xarray. "
        "See the package `Seaborn` for more options."
    )

    if ndims in [1, 2]:
        if row or col:
            kwargs["subplot_kws"] = subplot_kws
            kwargs["row"] = row
            kwargs["col"] = col
            kwargs["col_wrap"] = col_wrap
        if ndims == 1:
            plotfunc = line
            kwargs["hue"] = hue
        elif ndims == 2:
            if hue:
                plotfunc = line
                kwargs["hue"] = hue
            else:
                plotfunc = pcolormesh
                kwargs["subplot_kws"] = subplot_kws
    else:
        if row or col or hue:
            raise ValueError(error_msg)
        plotfunc = hist

    kwargs["ax"] = ax

    return plotfunc(darray, **kwargs)


# This function signature should not change so that it can use
# matplotlib format strings
def line(
    darray,
    *args,
    row=None,
    col=None,
    figsize=None,
    aspect=None,
    size=None,
    ax=None,
    hue=None,
    x=None,
    y=None,
    xincrease=None,
    yincrease=None,
    xscale=None,
    yscale=None,
    xticks=None,
    yticks=None,
    xlim=None,
    ylim=None,
    add_legend=True,
    _labels=True,
    **kwargs,
):
    """
    Line plot of DataArray index against values

    Wraps :func:`matplotlib:matplotlib.pyplot.plot`

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
    hue : string, optional
        Dimension or coordinate for which you want multiple lines plotted.
        If plotting against a 2D coordinate, ``hue`` must be a dimension.
    x, y : string, optional
        Dimension, coordinate or MultiIndex level for x, y axis.
        Only one of these may be specified.
        The other coordinate plots values from the DataArray on which this
        plot method is called.
    xscale, yscale : 'linear', 'symlog', 'log', 'logit', optional
        Specifies scaling for the x- and y-axes respectively
    xticks, yticks : Specify tick locations for x- and y-axes
    xlim, ylim : Specify x- and y-axes limits
    xincrease : None, True, or False, optional
        Should the values on the x axes be increasing from left to right?
        if None, use the default for the matplotlib function.
    yincrease : None, True, or False, optional
        Should the values on the y axes be increasing from top to bottom?
        if None, use the default for the matplotlib function.
    add_legend : bool, optional
        Add legend with y axis coordinates (2D inputs only).
    *args, **kwargs : optional
        Additional arguments to matplotlib.pyplot.plot
    """
    # Handle facetgrids first
    if row or col:
        allargs = locals().copy()
        allargs.update(allargs.pop("kwargs"))
        allargs.pop("darray")
        return _easy_facetgrid(darray, line, kind="line", **allargs)

    ndims = len(darray.dims)
    if ndims > 2:
        raise ValueError(
            "Line plots are for 1- or 2-dimensional DataArrays. "
            "Passed DataArray has {ndims} "
            "dimensions".format(ndims=ndims)
        )

    # The allargs dict passed to _easy_facetgrid above contains args
    if args == ():
        args = kwargs.pop("args", ())
    else:
        assert "args" not in kwargs

    ax = get_axis(figsize, size, aspect, ax)
    xplt, yplt, hueplt, hue_label = _infer_line_data(darray, x, y, hue)

    # Remove pd.Intervals if contained in xplt.values and/or yplt.values.
    xplt_val, yplt_val, x_suffix, y_suffix, kwargs = _resolve_intervals_1dplot(
        xplt.values, yplt.values, kwargs
    )
    xlabel = label_from_attrs(xplt, extra=x_suffix)
    ylabel = label_from_attrs(yplt, extra=y_suffix)

    _ensure_plottable(xplt_val, yplt_val)

    primitive = ax.plot(xplt_val, yplt_val, *args, **kwargs)

    if _labels:
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        ax.set_title(darray._title_for_slice())

    if darray.ndim == 2 and add_legend:
        ax.legend(handles=primitive, labels=list(hueplt.values), title=hue_label)

    # Rotate dates on xlabels
    # Do this without calling autofmt_xdate so that x-axes ticks
    # on other subplots (if any) are not deleted.
    # https://stackoverflow.com/questions/17430105/autofmt-xdate-deletes-x-axis-labels-of-all-subplots
    if np.issubdtype(xplt.dtype, np.datetime64):
        for xlabels in ax.get_xticklabels():
            xlabels.set_rotation(30)
            xlabels.set_ha("right")

    _update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)

    return primitive


def step(darray, *args, where="pre", drawstyle=None, ds=None, **kwargs):
    """
    Step plot of DataArray index against values

    Similar to :func:`matplotlib:matplotlib.pyplot.step`

    Parameters
    ----------
    where : {"pre", "post", "mid"}, default: "pre"
        Define where the steps should be placed:

        - "pre": The y value is continued constantly to the left from
          every *x* position, i.e. the interval ``(x[i-1], x[i]]`` has the
          value ``y[i]``.
        - "post": The y value is continued constantly to the right from
          every *x* position, i.e. the interval ``[x[i], x[i+1])`` has the
          value ``y[i]``.
        - "mid": Steps occur half-way between the *x* positions.

        Note that this parameter is ignored if one coordinate consists of
        :py:func:`pandas.Interval` values, e.g. as a result of
        :py:func:`xarray.Dataset.groupby_bins`. In this case, the actual
        boundaries of the interval are used.
    *args, **kwargs : optional
        Additional arguments following :py:func:`xarray.plot.line`
    """
    if where not in {"pre", "post", "mid"}:
        raise ValueError("'where' argument to step must be 'pre', 'post' or 'mid'")

    if ds is not None:
        if drawstyle is None:
            drawstyle = ds
        else:
            raise TypeError("ds and drawstyle are mutually exclusive")
    if drawstyle is None:
        drawstyle = ""
    drawstyle = "steps-" + where + drawstyle

    return line(darray, *args, drawstyle=drawstyle, **kwargs)


def hist(
    darray,
    figsize=None,
    size=None,
    aspect=None,
    ax=None,
    xincrease=None,
    yincrease=None,
    xscale=None,
    yscale=None,
    xticks=None,
    yticks=None,
    xlim=None,
    ylim=None,
    **kwargs,
):
    """
    Histogram of DataArray

    Wraps :func:`matplotlib:matplotlib.pyplot.hist`

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
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot this figure. By default, use the current axis.
        Mutually exclusive with ``size`` and ``figsize``.
    **kwargs : optional
        Additional keyword arguments to matplotlib.pyplot.hist

    """
    ax = get_axis(figsize, size, aspect, ax)

    no_nan = np.ravel(darray.values)
    no_nan = no_nan[pd.notnull(no_nan)]

    primitive = ax.hist(no_nan, **kwargs)

    ax.set_title("Histogram")
    ax.set_xlabel(label_from_attrs(darray))

    _update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)

    return primitive


def scatter(
    darray,
    *args,
    row=None,
    col=None,
    figsize=None,
    aspect=None,
    size=None,
    ax=None,
    hue=None,
    hue_style=None,
    x=None,
    y=None,
    xincrease=None,
    yincrease=None,
    xscale=None,
    yscale=None,
    xticks=None,
    yticks=None,
    xlim=None,
    ylim=None,
    add_legend=None,
    add_colorbar=None,
    _labels=True,
    **kwargs,
):
    """
    Scatter plot a DataArray along some coordinates.

    Parameters
    ----------
    darray : DataArray
        Dataarray to plot.
    x, y : str
        Variable names for x, y axis.
    hue: str, optional
        Variable by which to color scattered points
    hue_style: str, optional
        Can be either 'discrete' (legend) or 'continuous' (color bar).
    markersize: str, optional
        scatter only. Variable by which to vary size of scattered points.
    size_norm: optional
        Either None or 'Norm' instance to normalize the 'markersize' variable.
    add_guide: bool, optional
        Add a guide that depends on hue_style
            - for "discrete", build a legend.
              This is the default for non-numeric `hue` variables.
            - for "continuous",  build a colorbar
    row : str, optional
        If passed, make row faceted plots on this dimension name
    col : str, optional
        If passed, make column faceted plots on this dimension name
    col_wrap : int, optional
        Use together with ``col`` to wrap faceted plots
    ax : matplotlib axes object, optional
        If None, uses the current axis. Not applicable when using facets.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for matplotlib subplots. Only applies
        to FacetGrid plotting.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the width in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size.
        Height (in inches) of each plot. See also: ``aspect``.
    norm : ``matplotlib.colors.Normalize`` instance, optional
        If the ``norm`` has vmin or vmax specified, the corresponding kwarg
        must be None.
    vmin, vmax : float, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting one of these values will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    cmap : str or colormap, optional
        The mapping from data values to color space. Either a
        matplotlib colormap name or object. If not provided, this will
        be either ``viridis`` (if the function infers a sequential
        dataset) or ``RdBu_r`` (if the function infers a diverging
        dataset).  When `Seaborn` is installed, ``cmap`` may also be a
        `seaborn` color palette. If ``cmap`` is seaborn color palette
        and the plot type is not ``contour`` or ``contourf``, ``levels``
        must also be specified.
    colors : color-like or list of color-like, optional
        A single color or a list of colors. If the plot type is not ``contour``
        or ``contourf``, the ``levels`` argument is required.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap. Setting it to ``False`` prevents use of a
        diverging colormap.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    extend : {"neither", "both", "min", "max"}, optional
        How to draw arrows extending the colorbar beyond its limits. If not
        provided, extend is inferred from vmin, vmax and the data limits.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals. If an integer
        is provided, "nice" levels are chosen based on the data range: this can
        imply that the final number of levels is not exactly the expected one.
        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
        setting ``levels=np.linspace(vmin, vmax, N)``.
    **kwargs : optional
        Additional keyword arguments to matplotlib
    """
    # Handle facetgrids first
    if row or col:
        allargs = locals().copy()
        allargs.update(allargs.pop("kwargs"))
        allargs.pop("darray")
        return _easy_facetgrid(darray, scatter, kind="dataarray", **allargs)

    # _is_facetgrid = kwargs.pop("_is_facetgrid", False)
    _sizes = kwargs.pop("markersize", kwargs.pop("linewidth", None))
    size_norm = kwargs.pop("size_norm", None)
    size_mapping = kwargs.pop("size_mapping", None)  # set by facetgrid
    cbar_ax = kwargs.pop("cbar_ax", None)
    cbar_kwargs = kwargs.pop("cbar_kwargs", None)
    cmap_params = kwargs.pop("cmap_params", None)

    figsize = kwargs.pop("figsize", None)
    ax = get_axis(figsize, size, aspect, ax)

    _data = _infer_meta_data(darray, x, y, hue, hue_style, _sizes)

    add_guide = kwargs.pop("add_guide", None)
    if add_legend:
        pass
    elif add_guide is None or add_guide is True:
        add_legend = True if _data["hue_style"] == "discrete" else False
    elif add_legend is None:
        add_legend = False

    if add_colorbar:
        pass
    elif add_guide is None or add_guide is True:
        add_colorbar = True if _data["hue_style"] == "continuous" else False
    else:
        add_colorbar = False

    # need to infer size_mapping with full dataset
    _data.update(
        _infer_scatter_data(
            darray,
            x,
            y,
            _data["hue_label"],
            _sizes,
            size_norm,
            size_mapping,
            _MARKERSIZE_RANGE,
        )
    )

    # Plot the data:
    if _data["hue_style"] == "discrete":
        # Plot discrete data. ax.scatter only supports numerical values
        # in colors and sizes. Use a for loop to work around this issue.

        primitive = []
        # use pd.unique instead of np.unique because that keeps the order of the labels,
        # which is important to keep them in sync with the ones used in
        # FacetGrid.add_legend
        for label in pd.unique(_data["hue"].values.ravel()):
            mask = _data["hue"] == label
            if _data["sizes"] is not None:
                kwargs.update(s=_data["sizes"].where(mask, drop=True).values.ravel())

            primitive.append(
                ax.scatter(
                    _data["x"].where(mask, drop=True).values.ravel(),
                    _data["y"].where(mask, drop=True).values.ravel(),
                    label=label,
                    **kwargs,
                )
            )
        primitives = primitive

    elif _data["hue_label"] is None or _data["hue_style"] == "continuous":
        # ax.scatter suppoerts numerical values in colors and sizes.
        # So no need for for loops.

        cmap_params_subset = {}
        if _data["hue"] is not None:
            kwargs.update(c=_data["hue"].values.ravel())

            cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(
                scatter, darray[_data["hue_label"]].values, **locals()
            )

            # subset that can be passed to scatter, hist2d
            cmap_params_subset = {
                vv: cmap_params[vv] for vv in ["vmin", "vmax", "norm", "cmap"]
            }

        if _data["sizes"] is not None:
            kwargs.update(s=_data["sizes"].values.ravel())

        primitive = ax.scatter(
            _data["x"].values.ravel(),
            _data["y"].values.ravel(),
            **cmap_params_subset,
            **kwargs,
        )

        primitives = [primitive]

    # Set x and y labels:
    if _data["xlabel"]:
        ax.set_xlabel(_data["xlabel"])
    if _data["ylabel"]:
        ax.set_ylabel(_data["ylabel"])

    def _legend_elements_from_list(primitives, prop, **kwargs):
        """
        Get unique legend elements from a list of pathcollections.

        Getting multiple pathcollections happens when adding multiple
        scatters to the same plot.
        """
        import warnings

        handles = np.array([], dtype=object)
        labels = np.array([], dtype=str)

        for i, pc in enumerate(primitives):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                # Get legend elements, suppress empty data warnings
                # because it will be handled later:
                hdl, lbl = pc.legend_elements(prop=prop, **kwargs)
            handles = np.append(handles, hdl)
            labels = np.append(labels, lbl)

        # Duplicate legend entries is not necessary, therefore return
        # unique labels:
        unique_indices = np.sort(np.unique(labels, return_index=True)[1])
        handles = handles[unique_indices]
        labels = labels[unique_indices]

        return [handles, labels]

    def _legend_add_subtitle(handles, labels, text, func):
        """Add a subtitle to legend handles."""
        if text and len(handles) > 1:
            # Create a blank handle that's not visible, the
            # invisibillity will be used to discern which are subtitles
            # or not:
            blank_handle = func([], [], label=text)
            blank_handle.set_visible(False)

            # Subtitles are shown first:
            handles = np.insert(handles, 0, blank_handle)
            labels = np.insert(labels, 0, text)

        return handles, labels

    def _adjust_legend_subtitles(legend):
        """Make invisible-handle "subtitles" entries look more like titles."""
        plt = import_matplotlib_pyplot()

        # Legend title not in rcParams until 3.0
        font_size = plt.rcParams.get("legend.title_fontsize", None)
        hpackers = legend.findobj(plt.matplotlib.offsetbox.VPacker)[0].get_children()
        for hpack in hpackers:
            draw_area, text_area = hpack.get_children()
            handles = draw_area.get_children()

            # Assume that all artists that are not visible are
            # subtitles:
            if not all(artist.get_visible() for artist in handles):
                # Remove the dummy marker which will bring the text
                # more to the center:
                draw_area.set_width(0)
                for text in text_area.get_children():
                    if font_size is not None:
                        # The sutbtitles should have the same font size
                        # as normal legend titles:
                        text.set_size(font_size)

    def _add_legend(primitives, handles, labels, ax, **kwargs):
        if len(handles) > 1:
            # The normal case where a prop has been defined and
            # legend_elements finds results:
            legend = ax.legend(handles, labels, framealpha=0.5, **kwargs)
        elif len(primitives) > 1:
            # When no handles have been found use the primitives instead.
            # Example: hue and sizes have the same string:
            legend = ax.legend(handles=primitives, framealpha=0.5, **kwargs)

        return legend

    if add_legend:
        handles, labels = np.array([]), np.array([])
        if _data["hue_label"] and (_data["hue_label"] == _data["size_label"]):
            _add_legend(primitives, handles, labels, ax, title=_data["hue_label"])
        else:
            for hue_lbl, prop, func in [
                (_data["hue_label"], "colors", lambda x: x),
                (_data["size_label"], "sizes", lambda x: _data["sizes_to_labels"][x]),
            ]:
                if hue_lbl:
                    hdl, lbl = _legend_elements_from_list(
                        primitives, prop, num="auto", func=func,
                    )
                    hdl, lbl = _legend_add_subtitle(hdl, lbl, hue_lbl, ax.scatter)
                    handles, labels = np.append(handles, hdl), np.append(labels, lbl)

            legend = _add_legend(primitives, handles, labels, ax)
            _adjust_legend_subtitles(legend)

    if add_colorbar and _data["hue_label"]:
        if _data["hue_style"] == "discrete":
            raise NotImplementedError("Cannot create a colorbar for non numerics.")
        cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
        if "label" not in cbar_kwargs:
            cbar_kwargs["label"] = _data["hue_label"]
        _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params)

    return primitive


# MUST run before any 2d plotting functions are defined since
# _plot2d decorator adds them as methods here.
class _PlotMethods:
    """
    Enables use of xarray.plot functions as attributes on a DataArray.
    For example, DataArray.plot.imshow
    """

    __slots__ = ("_da",)

    def __init__(self, darray):
        self._da = darray

    def __call__(self, **kwargs):
        return plot(self._da, **kwargs)

    # we can't use functools.wraps here since that also modifies the name / qualname
    __doc__ = __call__.__doc__ = plot.__doc__
    __call__.__wrapped__ = plot  # type: ignore
    __call__.__annotations__ = plot.__annotations__

    @functools.wraps(hist)
    def hist(self, ax=None, **kwargs):
        return hist(self._da, ax=ax, **kwargs)

    @functools.wraps(line)
    def line(self, *args, **kwargs):
        return line(self._da, *args, **kwargs)

    @functools.wraps(step)
    def step(self, *args, **kwargs):
        return step(self._da, *args, **kwargs)

    @functools.wraps(scatter)
    def scatter(self, *args, **kwargs):
        return scatter(self._da, *args, **kwargs)


def override_signature(f):
    def wrapper(func):
        func.__wrapped__ = f

        return func

    return wrapper


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
    col_wrap : int, optional
        Use together with ``col`` to wrap faceted plots
    xscale, yscale : 'linear', 'symlog', 'log', 'logit', optional
        Specifies scaling for the x- and y-axes respectively
    xticks, yticks : Specify tick locations for x- and y-axes
    xlim, ylim : Specify x- and y-axes limits
    xincrease : None, True, or False, optional
        Should the values on the x axes be increasing from left to right?
        if None, use the default for the matplotlib function.
    yincrease : None, True, or False, optional
        Should the values on the y axes be increasing from top to bottom?
        if None, use the default for the matplotlib function.
    add_colorbar : bool, optional
        Adds colorbar to axis
    add_labels : bool, optional
        Use xarray metadata to label axes
    norm : ``matplotlib.colors.Normalize`` instance, optional
        If the ``norm`` has vmin or vmax specified, the corresponding kwarg
        must be None.
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
    extend : {"neither", "both", "min", "max"}, optional
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
        Dictionary of keyword arguments for matplotlib subplots. Only used
        for 2D and FacetGrid plots.
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
    plotfunc.__doc__ = f"{plotfunc.__doc__}\n{commondoc}"

    # plotfunc and newplotfunc have different signatures:
    # - plotfunc: (x, y, z, ax, **kwargs)
    # - newplotfunc: (darray, x, y, **kwargs)
    # where plotfunc accepts numpy arrays, while newplotfunc accepts a DataArray
    # and variable names. newplotfunc also explicitly lists most kwargs, so we
    # need to shorten it
    def signature(darray, x, y, **kwargs):
        pass

    @override_signature(signature)
    @functools.wraps(plotfunc)
    def newplotfunc(
        darray,
        x=None,
        y=None,
        figsize=None,
        size=None,
        aspect=None,
        ax=None,
        row=None,
        col=None,
        col_wrap=None,
        xincrease=True,
        yincrease=True,
        add_colorbar=None,
        add_labels=True,
        vmin=None,
        vmax=None,
        cmap=None,
        center=None,
        robust=False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws=None,
        cbar_ax=None,
        cbar_kwargs=None,
        xscale=None,
        yscale=None,
        xticks=None,
        yticks=None,
        xlim=None,
        ylim=None,
        norm=None,
        **kwargs,
    ):
        # All 2d plots in xarray share this function signature.
        # Method signature below should be consistent.

        # Decide on a default for the colorbar before facetgrids
        if add_colorbar is None:
            add_colorbar = plotfunc.__name__ != "contour"
        imshow_rgb = plotfunc.__name__ == "imshow" and darray.ndim == (
            3 + (row is not None) + (col is not None)
        )
        if imshow_rgb:
            # Don't add a colorbar when showing an image with explicit colors
            add_colorbar = False
            # Matplotlib does not support normalising RGB data, so do it here.
            # See eg. https://github.com/matplotlib/matplotlib/pull/10220
            if robust or vmax is not None or vmin is not None:
                darray = _rescale_imshow_rgb(darray, vmin, vmax, robust)
                vmin, vmax, robust = None, None, False

        # Handle facetgrids first
        if row or col:
            allargs = locals().copy()
            del allargs["darray"]
            del allargs["imshow_rgb"]
            allargs.update(allargs.pop("kwargs"))
            # Need the decorated plotting function
            allargs["plotfunc"] = globals()[plotfunc.__name__]
            return _easy_facetgrid(darray, kind="dataarray", **allargs)

        plt = import_matplotlib_pyplot()

        rgb = kwargs.pop("rgb", None)
        if rgb is not None and plotfunc.__name__ != "imshow":
            raise ValueError('The "rgb" keyword is only valid for imshow()')
        elif rgb is not None and not imshow_rgb:
            raise ValueError(
                'The "rgb" keyword is only valid for imshow()'
                "with a three-dimensional array (per facet)"
            )

        xlab, ylab = _infer_xy_labels(
            darray=darray, x=x, y=y, imshow=imshow_rgb, rgb=rgb
        )

        # better to pass the ndarrays directly to plotting functions
        xval = darray[xlab].values
        yval = darray[ylab].values

        # check if we need to broadcast one dimension
        if xval.ndim < yval.ndim:
            dims = darray[ylab].dims
            if xval.shape[0] == yval.shape[0]:
                xval = np.broadcast_to(xval[:, np.newaxis], yval.shape)
            else:
                xval = np.broadcast_to(xval[np.newaxis, :], yval.shape)

        elif yval.ndim < xval.ndim:
            dims = darray[xlab].dims
            if yval.shape[0] == xval.shape[0]:
                yval = np.broadcast_to(yval[:, np.newaxis], xval.shape)
            else:
                yval = np.broadcast_to(yval[np.newaxis, :], xval.shape)
        elif xval.ndim == 2:
            dims = darray[xlab].dims
        else:
            dims = (darray[ylab].dims[0], darray[xlab].dims[0])

        # May need to transpose for correct x, y labels
        # xlab may be the name of a coord, we have to check for dim names
        if imshow_rgb:
            # For RGB[A] images, matplotlib requires the color dimension
            # to be last.  In Xarray the order should be unimportant, so
            # we transpose to (y, x, color) to make this work.
            yx_dims = (ylab, xlab)
            dims = yx_dims + tuple(d for d in darray.dims if d not in yx_dims)

        if dims != darray.dims:
            darray = darray.transpose(*dims, transpose_coords=True)

        # Pass the data as a masked ndarray too
        zval = darray.to_masked_array(copy=False)

        # Replace pd.Intervals if contained in xval or yval.
        xplt, xlab_extra = _resolve_intervals_2dplot(xval, plotfunc.__name__)
        yplt, ylab_extra = _resolve_intervals_2dplot(yval, plotfunc.__name__)

        _ensure_plottable(xplt, yplt, zval)

        cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(
            plotfunc,
            zval.data,
            **locals(),
            _is_facetgrid=kwargs.pop("_is_facetgrid", False),
        )

        if "contour" in plotfunc.__name__:
            # extend is a keyword argument only for contour and contourf, but
            # passing it to the colorbar is sufficient for imshow and
            # pcolormesh
            kwargs["extend"] = cmap_params["extend"]
            kwargs["levels"] = cmap_params["levels"]
            # if colors == a single color, matplotlib draws dashed negative
            # contours. we lose this feature if we pass cmap and not colors
            if isinstance(colors, str):
                cmap_params["cmap"] = None
                kwargs["colors"] = colors

        if "pcolormesh" == plotfunc.__name__:
            kwargs["infer_intervals"] = infer_intervals

        if "imshow" == plotfunc.__name__ and isinstance(aspect, str):
            # forbid usage of mpl strings
            raise ValueError("plt.imshow's `aspect` kwarg is not available in xarray")

        if subplot_kws is None:
            subplot_kws = dict()
        ax = get_axis(figsize, size, aspect, ax, **subplot_kws)

        primitive = plotfunc(
            xplt,
            yplt,
            zval,
            ax=ax,
            cmap=cmap_params["cmap"],
            vmin=cmap_params["vmin"],
            vmax=cmap_params["vmax"],
            norm=cmap_params["norm"],
            **kwargs,
        )

        # Label the plot with metadata
        if add_labels:
            ax.set_xlabel(label_from_attrs(darray[xlab], xlab_extra))
            ax.set_ylabel(label_from_attrs(darray[ylab], ylab_extra))
            ax.set_title(darray._title_for_slice())

        if add_colorbar:
            if add_labels and "label" not in cbar_kwargs:
                cbar_kwargs["label"] = label_from_attrs(darray)
            cbar = _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params)
        elif cbar_ax is not None or cbar_kwargs:
            # inform the user about keywords which aren't used
            raise ValueError(
                "cbar_ax and cbar_kwargs can't be used with add_colorbar=False."
            )

        # origin kwarg overrides yincrease
        if "origin" in kwargs:
            yincrease = None

        _update_axes(
            ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim
        )

        # Rotate dates on xlabels
        # Do this without calling autofmt_xdate so that x-axes ticks
        # on other subplots (if any) are not deleted.
        # https://stackoverflow.com/questions/17430105/autofmt-xdate-deletes-x-axis-labels-of-all-subplots
        if np.issubdtype(xplt.dtype, np.datetime64):
            for xlabels in ax.get_xticklabels():
                xlabels.set_rotation(30)
                xlabels.set_ha("right")

        return primitive

    # For use as DataArray.plot.plotmethod
    @functools.wraps(newplotfunc)
    def plotmethod(
        _PlotMethods_obj,
        x=None,
        y=None,
        figsize=None,
        size=None,
        aspect=None,
        ax=None,
        row=None,
        col=None,
        col_wrap=None,
        xincrease=True,
        yincrease=True,
        add_colorbar=None,
        add_labels=True,
        vmin=None,
        vmax=None,
        cmap=None,
        colors=None,
        center=None,
        robust=False,
        extend=None,
        levels=None,
        infer_intervals=None,
        subplot_kws=None,
        cbar_ax=None,
        cbar_kwargs=None,
        xscale=None,
        yscale=None,
        xticks=None,
        yticks=None,
        xlim=None,
        ylim=None,
        norm=None,
        **kwargs,
    ):
        """
        The method should have the same signature as the function.

        This just makes the method work on Plotmethods objects,
        and passes all the other arguments straight through.
        """
        allargs = locals()
        allargs["darray"] = _PlotMethods_obj._da
        allargs.update(kwargs)
        for arg in ["_PlotMethods_obj", "newplotfunc", "kwargs"]:
            del allargs[arg]
        return newplotfunc(**allargs)

    # Add to class _PlotMethods
    setattr(_PlotMethods, plotmethod.__name__, plotmethod)

    return newplotfunc


@_plot2d
def imshow(x, y, z, ax, **kwargs):
    """
    Image plot of 2d DataArray using matplotlib.pyplot

    Wraps :func:`matplotlib:matplotlib.pyplot.imshow`

    While other plot methods require the DataArray to be strictly
    two-dimensional, ``imshow`` also accepts a 3D array where some
    dimension can be interpreted as RGB or RGBA color channels and
    allows this dimension to be specified via the kwarg ``rgb=``.

    Unlike matplotlib, Xarray can apply ``vmin`` and ``vmax`` to RGB or RGBA
    data, by applying a single scaling factor and offset to all bands.
    Passing  ``robust=True`` infers ``vmin`` and ``vmax``
    :ref:`in the usual way <robust-plotting>`.

    .. note::
        This function needs uniformly spaced coordinates to
        properly label the axes. Call DataArray.plot() to check.

    The pixels are centered on the coordinates values. Ie, if the coordinate
    value is 3.2 then the pixels for those coordinates will be centered on 3.2.
    """

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            "imshow requires 1D coordinates, try using pcolormesh or contour(f)"
        )

    # Centering the pixels- Assumes uniform spacing
    try:
        xstep = (x[1] - x[0]) / 2.0
    except IndexError:
        # Arbitrary default value, similar to matplotlib behaviour
        xstep = 0.1
    try:
        ystep = (y[1] - y[0]) / 2.0
    except IndexError:
        ystep = 0.1
    left, right = x[0] - xstep, x[-1] + xstep
    bottom, top = y[-1] + ystep, y[0] - ystep

    defaults = {"origin": "upper", "interpolation": "nearest"}

    if not hasattr(ax, "projection"):
        # not for cartopy geoaxes
        defaults["aspect"] = "auto"

    # Allow user to override these defaults
    defaults.update(kwargs)

    if defaults["origin"] == "upper":
        defaults["extent"] = [left, right, bottom, top]
    else:
        defaults["extent"] = [left, right, top, bottom]

    if z.ndim == 3:
        # matplotlib imshow uses black for missing data, but Xarray makes
        # missing data transparent.  We therefore add an alpha channel if
        # there isn't one, and set it to transparent where data is masked.
        if z.shape[-1] == 3:
            alpha = np.ma.ones(z.shape[:2] + (1,), dtype=z.dtype)
            if np.issubdtype(z.dtype, np.integer):
                alpha *= 255
            z = np.ma.concatenate((z, alpha), axis=2)
        else:
            z = z.copy()
        z[np.any(z.mask, axis=-1), -1] = 0

    primitive = ax.imshow(z, **defaults)

    return primitive


@_plot2d
def contour(x, y, z, ax, **kwargs):
    """
    Contour plot of 2d DataArray

    Wraps :func:`matplotlib:matplotlib.pyplot.contour`
    """
    primitive = ax.contour(x, y, z, **kwargs)
    return primitive


@_plot2d
def contourf(x, y, z, ax, **kwargs):
    """
    Filled contour plot of 2d DataArray

    Wraps :func:`matplotlib:matplotlib.pyplot.contourf`
    """
    primitive = ax.contourf(x, y, z, **kwargs)
    return primitive


@_plot2d
def pcolormesh(x, y, z, ax, infer_intervals=None, **kwargs):
    """
    Pseudocolor plot of 2d DataArray

    Wraps :func:`matplotlib:matplotlib.pyplot.pcolormesh`
    """

    # decide on a default for infer_intervals (GH781)
    x = np.asarray(x)
    if infer_intervals is None:
        if hasattr(ax, "projection"):
            if len(x.shape) == 1:
                infer_intervals = True
            else:
                infer_intervals = False
        else:
            infer_intervals = True

    if infer_intervals and (
        (np.shape(x)[0] == np.shape(z)[1])
        or ((x.ndim > 1) and (np.shape(x)[1] == np.shape(z)[1]))
    ):
        if len(x.shape) == 1:
            x = _infer_interval_breaks(x, check_monotonic=True)
        else:
            # we have to infer the intervals on both axes
            x = _infer_interval_breaks(x, axis=1)
            x = _infer_interval_breaks(x, axis=0)

    if infer_intervals and (np.shape(y)[0] == np.shape(z)[0]):
        if len(y.shape) == 1:
            y = _infer_interval_breaks(y, check_monotonic=True)
        else:
            # we have to infer the intervals on both axes
            y = _infer_interval_breaks(y, axis=1)
            y = _infer_interval_breaks(y, axis=0)

    primitive = ax.pcolormesh(x, y, z, **kwargs)

    # by default, pcolormesh picks "round" values for bounds
    # this results in ugly looking plots with lots of surrounding whitespace
    if not hasattr(ax, "projection") and x.ndim == 1 and y.ndim == 1:
        # not a cartopy geoaxis
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])

    return primitive
