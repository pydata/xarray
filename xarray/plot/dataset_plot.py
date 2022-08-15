from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Hashable, Mapping

from ..core.alignment import broadcast
from .facetgrid import _easy_facetgrid
from .plot import _PlotMethods
from .utils import (
    _add_colorbar,
    _get_nice_quiver_magnitude,
    _infer_meta_data,
    _process_cmap_cbar_kwargs,
    get_axis,
)

if TYPE_CHECKING:
    from ..core.dataarray import DataArray
    from ..core.types import T_Dataset


class _Dataset_PlotMethods:
    """
    Enables use of xarray.plot functions as attributes on a Dataset.
    For example, Dataset.plot.scatter
    """

    def __init__(self, dataset):
        self._ds = dataset

    def __call__(self, *args, **kwargs):
        raise ValueError(
            "Dataset.plot cannot be called directly. Use "
            "an explicit plot method, e.g. ds.plot.scatter(...)"
        )


def _dsplot(plotfunc):
    commondoc = """
    Parameters
    ----------

    ds : Dataset
    x, y : str
        Variable names for the *x* and *y* grid positions.
    u, v : str, optional
        Variable names for the *u* and *v* velocities
        (in *x* and *y* direction, respectively; quiver/streamplot plots only).
    hue: str, optional
        Variable by which to color scatter points or arrows.
    hue_style: {'continuous', 'discrete'}, optional
        How to use the ``hue`` variable:

        - ``'continuous'`` -- continuous color scale
          (default for numeric ``hue`` variables)
        - ``'discrete'`` -- a color for each unique value, using the default color cycle
          (default for non-numeric ``hue`` variables)
    markersize: str, optional
        Variable by which to vary the size of scattered points (scatter plot only).
    size_norm: matplotlib.colors.Normalize or tuple, optional
        Used to normalize the ``markersize`` variable.
        If a tuple is passed, the values will be passed to
        :py:class:`matplotlib:matplotlib.colors.Normalize` as arguments.
        Default: no normalization (``vmin=None``, ``vmax=None``, ``clip=False``).
    scale: scalar, optional
        Quiver only. Number of data units per arrow length unit.
        Use this to control the length of the arrows: larger values lead to
        smaller arrows.
    add_guide: bool, optional, default: True
        Add a guide that depends on ``hue_style``:

        - ``'continuous'`` -- build a colorbar
        - ``'discrete'`` -- build a legend
    row : str, optional
        If passed, make row faceted plots on this dimension name.
    col : str, optional
        If passed, make column faceted plots on this dimension name.
    col_wrap : int, optional
        Use together with ``col`` to wrap faceted plots.
    ax : matplotlib axes object, optional
        If ``None``, use the current axes. Not applicable when using facets.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for Matplotlib subplots
        (see :py:meth:`matplotlib:matplotlib.figure.Figure.add_subplot`).
        Only applies to FacetGrid plotting.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the *width* in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size:
        *height* (in inches) of each plot. See also: ``aspect``.
    norm : matplotlib.colors.Normalize, optional
        If ``norm`` has ``vmin`` or ``vmax`` specified, the corresponding
        kwarg must be ``None``.
    vmin, vmax : float, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting one of these values will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    cmap : matplotlib colormap name or colormap, optional
        The mapping from data values to color space. Either a
        Matplotlib colormap name or object. If not provided, this will
        be either ``'viridis'`` (if the function infers a sequential
        dataset) or ``'RdBu_r'`` (if the function infers a diverging
        dataset).
        See :doc:`Choosing Colormaps in Matplotlib <matplotlib:tutorials/colors/colormaps>`
        for more information.

        If *seaborn* is installed, ``cmap`` may also be a
        `seaborn color palette <https://seaborn.pydata.org/tutorial/color_palettes.html>`_.
        Note: if ``cmap`` is a seaborn color palette,
        ``levels`` must also be specified.
    colors : str or array-like of color-like, optional
        A single color or a list of colors. The ``levels`` argument
        is required.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap. Setting it to ``False`` prevents use of a
        diverging colormap.
    robust : bool, optional
        If ``True`` and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    extend : {'neither', 'both', 'min', 'max'}, optional
        How to draw arrows extending the colorbar beyond its limits. If not
        provided, ``extend`` is inferred from ``vmin``, ``vmax`` and the data limits.
    levels : int or array-like, optional
        Split the colormap (``cmap``) into discrete color intervals. If an integer
        is provided, "nice" levels are chosen based on the data range: this can
        imply that the final number of levels is not exactly the expected one.
        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
        setting ``levels=np.linspace(vmin, vmax, N)``.
    **kwargs : optional
        Additional keyword arguments to wrapped Matplotlib function.
    """

    # Build on the original docstring
    plotfunc.__doc__ = f"{plotfunc.__doc__}\n{commondoc}"

    @functools.wraps(plotfunc)
    def newplotfunc(
        ds,
        x=None,
        y=None,
        u=None,
        v=None,
        hue=None,
        hue_style=None,
        col=None,
        row=None,
        ax=None,
        figsize=None,
        size=None,
        col_wrap=None,
        sharex=True,
        sharey=True,
        aspect=None,
        subplot_kws=None,
        add_guide=None,
        cbar_kwargs=None,
        cbar_ax=None,
        vmin=None,
        vmax=None,
        norm=None,
        infer_intervals=None,
        center=None,
        levels=None,
        robust=None,
        colors=None,
        extend=None,
        cmap=None,
        **kwargs,
    ):

        _is_facetgrid = kwargs.pop("_is_facetgrid", False)
        if _is_facetgrid:  # facetgrid call
            meta_data = kwargs.pop("meta_data")
        else:
            meta_data = _infer_meta_data(
                ds, x, y, hue, hue_style, add_guide, funcname=plotfunc.__name__
            )

        hue_style = meta_data["hue_style"]

        # handle facetgrids first
        if col or row:
            allargs = locals().copy()
            allargs["plotfunc"] = globals()[plotfunc.__name__]
            allargs["data"] = ds
            # remove kwargs to avoid passing the information twice
            for arg in ["meta_data", "kwargs", "ds"]:
                del allargs[arg]

            return _easy_facetgrid(kind="dataset", **allargs, **kwargs)

        figsize = kwargs.pop("figsize", None)
        ax = get_axis(figsize, size, aspect, ax)

        if hue_style == "continuous" and hue is not None:
            if _is_facetgrid:
                cbar_kwargs = meta_data["cbar_kwargs"]
                cmap_params = meta_data["cmap_params"]
            else:
                cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(
                    plotfunc, ds[hue].values, **locals()
                )

            # subset that can be passed to scatter, hist2d
            cmap_params_subset = {
                vv: cmap_params[vv] for vv in ["vmin", "vmax", "norm", "cmap"]
            }

        else:
            cmap_params_subset = {}

        if (u is not None or v is not None) and plotfunc.__name__ not in (
            "quiver",
            "streamplot",
        ):
            raise ValueError("u, v are only allowed for quiver or streamplot plots.")

        primitive = plotfunc(
            ds=ds,
            x=x,
            y=y,
            ax=ax,
            u=u,
            v=v,
            hue=hue,
            hue_style=hue_style,
            cmap_params=cmap_params_subset,
            **kwargs,
        )

        if _is_facetgrid:  # if this was called from Facetgrid.map_dataset,
            return primitive  # finish here. Else, make labels

        if meta_data.get("xlabel", None):
            ax.set_xlabel(meta_data.get("xlabel"))
        if meta_data.get("ylabel", None):
            ax.set_ylabel(meta_data.get("ylabel"))

        if meta_data["add_legend"]:
            ax.legend(handles=primitive, title=meta_data.get("hue_label", None))
        if meta_data["add_colorbar"]:
            cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
            if "label" not in cbar_kwargs:
                cbar_kwargs["label"] = meta_data.get("hue_label", None)
            _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params)

        if meta_data["add_quiverkey"]:
            magnitude = _get_nice_quiver_magnitude(ds[u], ds[v])
            units = ds[u].attrs.get("units", "")
            ax.quiverkey(
                primitive,
                X=0.85,
                Y=0.9,
                U=magnitude,
                label=f"{magnitude}\n{units}",
                labelpos="E",
                coordinates="figure",
            )

        if plotfunc.__name__ in ("quiver", "streamplot"):
            title = ds[u]._title_for_slice()
        else:
            title = ds[x]._title_for_slice()
        ax.set_title(title)

        return primitive

    @functools.wraps(newplotfunc)
    def plotmethod(
        _PlotMethods_obj,
        x=None,
        y=None,
        u=None,
        v=None,
        hue=None,
        hue_style=None,
        col=None,
        row=None,
        ax=None,
        figsize=None,
        col_wrap=None,
        sharex=True,
        sharey=True,
        aspect=None,
        size=None,
        subplot_kws=None,
        add_guide=None,
        cbar_kwargs=None,
        cbar_ax=None,
        vmin=None,
        vmax=None,
        norm=None,
        infer_intervals=None,
        center=None,
        levels=None,
        robust=None,
        colors=None,
        extend=None,
        cmap=None,
        **kwargs,
    ):
        """
        The method should have the same signature as the function.

        This just makes the method work on Plotmethods objects,
        and passes all the other arguments straight through.
        """
        allargs = locals()
        allargs["ds"] = _PlotMethods_obj._ds
        allargs.update(kwargs)
        for arg in ["_PlotMethods_obj", "newplotfunc", "kwargs"]:
            del allargs[arg]
        return newplotfunc(**allargs)

    # Add to class _PlotMethods
    setattr(_Dataset_PlotMethods, plotmethod.__name__, plotmethod)

    return newplotfunc


@_dsplot
def quiver(ds, x, y, ax, u, v, **kwargs):
    """Quiver plot of Dataset variables.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.quiver`.
    """
    import matplotlib as mpl

    if x is None or y is None or u is None or v is None:
        raise ValueError("Must specify x, y, u, v for quiver plots.")

    x, y, u, v = broadcast(ds[x], ds[y], ds[u], ds[v])

    args = [x.values, y.values, u.values, v.values]
    hue = kwargs.pop("hue")
    cmap_params = kwargs.pop("cmap_params")

    if hue:
        args.append(ds[hue].values)

        # TODO: Fix this by always returning a norm with vmin, vmax in cmap_params
        if not cmap_params["norm"]:
            cmap_params["norm"] = mpl.colors.Normalize(
                cmap_params.pop("vmin"), cmap_params.pop("vmax")
            )

    kwargs.pop("hue_style")
    kwargs.setdefault("pivot", "middle")
    hdl = ax.quiver(*args, **kwargs, **cmap_params)
    return hdl


@_dsplot
def streamplot(ds, x, y, ax, u, v, **kwargs):
    """Plot streamlines of Dataset variables.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.streamplot`.
    """
    import matplotlib as mpl

    if x is None or y is None or u is None or v is None:
        raise ValueError("Must specify x, y, u, v for streamplot plots.")

    # Matplotlib's streamplot has strong restrictions on what x and y can be, so need to
    # get arrays transposed the 'right' way around. 'x' cannot vary within 'rows', so
    # the dimension of x must be the second dimension. 'y' cannot vary with 'columns' so
    # the dimension of y must be the first dimension. If x and y are both 2d, assume the
    # user has got them right already.
    if len(ds[x].dims) == 1:
        xdim = ds[x].dims[0]
    if len(ds[y].dims) == 1:
        ydim = ds[y].dims[0]
    if xdim is not None and ydim is None:
        ydim = set(ds[y].dims) - {xdim}
    if ydim is not None and xdim is None:
        xdim = set(ds[x].dims) - {ydim}

    x, y, u, v = broadcast(ds[x], ds[y], ds[u], ds[v])

    if xdim is not None and ydim is not None:
        # Need to ensure the arrays are transposed correctly
        x = x.transpose(ydim, xdim)
        y = y.transpose(ydim, xdim)
        u = u.transpose(ydim, xdim)
        v = v.transpose(ydim, xdim)

    args = [x.values, y.values, u.values, v.values]
    hue = kwargs.pop("hue")
    cmap_params = kwargs.pop("cmap_params")

    if hue:
        kwargs["color"] = ds[hue].values

        # TODO: Fix this by always returning a norm with vmin, vmax in cmap_params
        if not cmap_params["norm"]:
            cmap_params["norm"] = mpl.colors.Normalize(
                cmap_params.pop("vmin"), cmap_params.pop("vmax")
            )

    kwargs.pop("hue_style")
    hdl = ax.streamplot(*args, **kwargs, **cmap_params)

    # Return .lines so colorbar creation works properly
    return hdl.lines


def _attach_to_plot_class(plotfunc: Callable) -> None:
    """
    Set the function to the plot class and add a common docstring.

    Use this decorator when relying on DataArray.plot methods for
    creating the Dataset plot.

    TODO: Reduce code duplication.

    * The goal is to reduce code duplication by moving all Dataset
      specific plots to the DataArray side and use this thin wrapper to
      handle the conversion between Dataset and DataArray.
    * Improve docstring handling, maybe reword the DataArray versions to
      explain Datasets better.
    * Consider automatically adding all _PlotMethods to
      _Dataset_PlotMethods.

    Parameters
    ----------
    plotfunc : function
        Function that returns a finished plot primitive.
    """
    # Build on the original docstring:
    original_doc = getattr(_PlotMethods, plotfunc.__name__, object)
    commondoc = original_doc.__doc__
    if commondoc is not None:
        doc_warning = (
            f"This docstring was copied from xr.DataArray.plot.{original_doc.__name__}."
            " Some inconsistencies may exist."
        )
        # Add indentation so it matches the original doc:
        commondoc = f"\n\n    {doc_warning}\n\n    {commondoc}"
    else:
        commondoc = ""
    plotfunc.__doc__ = (
        f"    {plotfunc.__doc__}\n\n"
        "    The `y` DataArray will be used as base,"
        "    any other variables are added as coords.\n\n"
        f"{commondoc}"
    )

    @functools.wraps(plotfunc)
    def plotmethod(self, *args, **kwargs):
        return plotfunc(self._ds, *args, **kwargs)

    # Add to class _PlotMethods
    setattr(_Dataset_PlotMethods, plotmethod.__name__, plotmethod)


def _normalize_args(plotmethod: str, args, kwargs) -> dict[str, Any]:
    from ..core.dataarray import DataArray

    # Determine positional arguments keyword by inspecting the
    # signature of the plotmethod:
    locals_ = dict(
        inspect.signature(getattr(DataArray().plot, plotmethod))
        .bind(*args, **kwargs)
        .arguments.items()
    )
    locals_.update(locals_.pop("kwargs", {}))

    return locals_


def _temp_dataarray(ds: T_Dataset, y: Hashable, locals_: Mapping) -> DataArray:
    """Create a temporary datarray with extra coords."""
    from ..core.dataarray import DataArray

    # Base coords:
    coords = dict(ds.coords)

    # Add extra coords to the DataArray from valid kwargs, if using all
    # kwargs there is a risk that we add unneccessary dataarrays as
    # coords straining RAM further for example:
    # ds.both and extend="both" would add ds.both to the coords:
    valid_coord_kwargs = {"x", "z", "markersize", "hue", "row", "col", "u", "v"}
    coord_kwargs = locals_.keys() & valid_coord_kwargs
    for k in coord_kwargs:
        key = locals_[k]
        if ds.data_vars.get(key) is not None:
            coords[key] = ds[key]

    # The dataarray has to include all the dims. Broadcast to that shape
    # and add the additional coords:
    _y = ds[y].broadcast_like(ds)

    return DataArray(_y, coords=coords)


@_attach_to_plot_class
def scatter(ds: T_Dataset, x: Hashable, y: Hashable, *args, **kwargs):
    """Scatter plot Dataset data variables against each other."""
    plotmethod = "scatter"
    kwargs.update(x=x)
    locals_ = _normalize_args(plotmethod, args, kwargs)
    da = _temp_dataarray(ds, y, locals_)

    return getattr(da.plot, plotmethod)(*locals_.pop("args", ()), **locals_)
