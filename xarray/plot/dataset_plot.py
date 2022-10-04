from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Literal, overload

import numpy as np
import pandas as pd

from ..core.alignment import broadcast
from .facetgrid import _easy_facetgrid
from .utils import (
    _add_colorbar,
    _get_nice_quiver_magnitude,
    _infer_meta_data,
    _parse_size,
    _process_cmap_cbar_kwargs,
    get_axis,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.colors import Normalize
    from matplotlib.quiver import Quiver

    from ..core.dataarray import DataArray
    from ..core.dataset import Dataset
    from ..core.types import AspectOptions, HueStyleOptions
    from .facetgrid import FacetGrid


def _infer_scatter_data(
    ds: Dataset,
    x: Hashable,
    y: Hashable,
    hue: Hashable | None,
    markersize: Hashable | None,
    size_norm,
    size_mapping=None,
) -> dict[str, DataArray | None]:

    broadcast_keys = ["x", "y"]
    to_broadcast = [ds[x], ds[y]]
    if hue:
        to_broadcast.append(ds[hue])
        broadcast_keys.append("hue")
    if markersize:
        to_broadcast.append(ds[markersize])
        broadcast_keys.append("size")

    broadcasted = dict(zip(broadcast_keys, broadcast(*to_broadcast)))

    data: dict[str, DataArray | None] = {
        "x": broadcasted["x"],
        "y": broadcasted["y"],
        "hue": None,
        "sizes": None,
    }

    if hue:
        data["hue"] = broadcasted["hue"]

    if markersize:
        size = broadcasted["size"]

        if size_mapping is None:
            size_mapping = _parse_size(size, size_norm)

        data["sizes"] = size.copy(
            data=np.reshape(size_mapping.loc[size.values.ravel()].values, size.shape)
        )

    return data


def _dsplot(plotfunc):
    commondoc = """
    Parameters
    ----------

    ds : Dataset
    x, y : Hashable
        Variable names for the *x* and *y* grid positions.
    u, v : Hashable or None, optional
        Variable names for the *u* and *v* velocities
        (in *x* and *y* direction, respectively; quiver/streamplot plots only).
    hue: Hashable or None, optional
        Variable by which to color scatter points or arrows.
    hue_style: {'continuous', 'discrete'}, optional
        How to use the ``hue`` variable:

        - ``'continuous'`` -- continuous color scale
          (default for numeric ``hue`` variables)
        - ``'discrete'`` -- a color for each unique value, using the default color cycle
          (default for non-numeric ``hue`` variables)
    markersize: Hashable or None, optional
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
    row : Hashable or None, optional
        If passed, make row faceted plots on this dimension name.
    col : Hashable or None, optional
        If passed, make column faceted plots on this dimension name.
    col_wrap : int, optional
        Use together with ``col`` to wrap faceted plots.
    ax : matplotlib axes object, optional
        If ``None``, use the current axes. Not applicable when using facets.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for Matplotlib subplots
        (see :py:meth:`matplotlib:matplotlib.figure.Figure.add_subplot`).
        Only applies to FacetGrid plotting.
    aspect : "auto", "equal", scalar or None, optional
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
        ds: Dataset,
        *args: Any,
        x: Hashable | None = None,
        y: Hashable | None = None,
        u: Hashable | None = None,
        v: Hashable | None = None,
        hue: Hashable | None = None,
        hue_style: HueStyleOptions = None,
        col: Hashable | None = None,
        row: Hashable | None = None,
        ax: Axes | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        col_wrap: int | None = None,
        sharex: bool = True,
        sharey: bool = True,
        aspect: AspectOptions = None,
        subplot_kws: dict[str, Any] | None = None,
        add_guide: bool | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        norm: Normalize | None = None,
        infer_intervals=None,
        center=None,
        levels=None,
        robust: bool | None = None,
        colors=None,
        extend=None,
        cmap=None,
        **kwargs: Any,
    ) -> Any:

        if args:
            warnings.warn(
                "Using positional arguments is deprecated for all plot methods, use keyword arguments instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            assert x is None
            x = args[0]
            if len(args) > 0:
                assert y is None
                y = args[1]
            if len(args) > 1:
                assert u is None
                u = args[2]
            if len(args) > 2:
                assert v is None
                v = args[3]
            if len(args) > 3:
                assert hue is None
                hue = args[4]
            if len(args) > 4:
                raise ValueError(
                    "Using positional arguments is deprecated for all plot methods, use keyword arguments instead."
                )
        del args

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

    # we want to actually expose the signature of newplotfunc
    # and not the copied **kwargs from the plotfunc which
    # functools.wraps adds, so delete the wrapped attr
    del newplotfunc.__wrapped__

    return newplotfunc


@overload
def scatter(
    ds: Dataset,
    *args: Any,
    x: Hashable | None = None,
    y: Hashable | None = None,
    u: Hashable | None = None,
    v: Hashable | None = None,
    hue: Hashable | None = None,
    hue_style: HueStyleOptions = None,
    col: Hashable,  # wrap -> FacetGrid
    row: Hashable | None = None,
    ax: Axes | None = None,
    figsize: Iterable[float] | None = None,
    size: float | None = None,
    col_wrap: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    aspect: AspectOptions = None,
    subplot_kws: dict[str, Any] | None = None,
    add_guide: bool | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    infer_intervals=None,
    center=None,
    levels=None,
    robust: bool | None = None,
    colors=None,
    extend=None,
    cmap=None,
    **kwargs: Any,
) -> FacetGrid[Dataset]:
    ...


@overload
def scatter(
    ds: Dataset,
    *args: Any,
    x: Hashable | None = None,
    y: Hashable | None = None,
    u: Hashable | None = None,
    v: Hashable | None = None,
    hue: Hashable | None = None,
    hue_style: HueStyleOptions = None,
    col: Hashable | None = None,
    row: Hashable,  # wrap -> FacetGrid
    ax: Axes | None = None,
    figsize: Iterable[float] | None = None,
    size: float | None = None,
    col_wrap: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    aspect: AspectOptions = None,
    subplot_kws: dict[str, Any] | None = None,
    add_guide: bool | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    infer_intervals=None,
    center=None,
    levels=None,
    robust: bool | None = None,
    colors=None,
    extend=None,
    cmap=None,
    **kwargs: Any,
) -> FacetGrid[Dataset]:
    ...


@overload
def scatter(
    ds: Dataset,
    *args: Any,
    x: Hashable | None = None,
    y: Hashable | None = None,
    u: Hashable | None = None,
    v: Hashable | None = None,
    hue: Hashable | None = None,
    hue_style: Literal["discrete"],  # list of primitives
    col: None = None,  # no wrap
    row: None = None,  # no wrap
    ax: Axes | None = None,
    figsize: Iterable[float] | None = None,
    size: float | None = None,
    col_wrap: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    aspect: AspectOptions = None,
    subplot_kws: dict[str, Any] | None = None,
    add_guide: bool | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    infer_intervals=None,
    center=None,
    levels=None,
    robust: bool | None = None,
    colors=None,
    extend=None,
    cmap=None,
    **kwargs: Any,
) -> list[PathCollection]:
    ...


@overload
def scatter(
    ds: Dataset,
    *args: Any,
    x: Hashable | None = None,
    y: Hashable | None = None,
    u: Hashable | None = None,
    v: Hashable | None = None,
    hue: Hashable | None = None,
    hue_style: Literal["continuous"] | None = None,  # primitive
    col: None = None,  # no wrap
    row: None = None,  # no wrap
    ax: Axes | None = None,
    figsize: Iterable[float] | None = None,
    size: float | None = None,
    col_wrap: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    aspect: AspectOptions = None,
    subplot_kws: dict[str, Any] | None = None,
    add_guide: bool | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    infer_intervals=None,
    center=None,
    levels=None,
    robust: bool | None = None,
    colors=None,
    extend=None,
    cmap=None,
    **kwargs: Any,
) -> PathCollection:
    ...


@_dsplot
def scatter(
    ds: Dataset,
    x: Hashable,
    y: Hashable,
    ax: Axes,
    hue: Hashable | None,
    hue_style: HueStyleOptions,
    **kwargs: Any,
) -> PathCollection | list[PathCollection]:
    """
    Scatter Dataset data variables against each other.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.scatter`.
    """

    if "add_colorbar" in kwargs or "add_legend" in kwargs:
        raise ValueError(
            "Dataset.plot.scatter does not accept "
            "'add_colorbar' or 'add_legend'. "
            "Use 'add_guide' instead."
        )

    cmap_params = kwargs.pop("cmap_params")
    markersize = kwargs.pop("markersize", None)
    size_norm = kwargs.pop("size_norm", None)
    size_mapping = kwargs.pop("size_mapping", None)  # set by facetgrid

    # Remove `u` and `v` so they don't get passed to `ax.scatter`
    kwargs.pop("u", None)
    kwargs.pop("v", None)

    # need to infer size_mapping with full dataset
    data = _infer_scatter_data(ds, x, y, hue, markersize, size_norm, size_mapping)

    dhue = data["hue"]
    dx = data["x"]
    dy = data["y"]
    assert dx is not None
    assert dy is not None

    if hue_style == "discrete":
        primitive = []
        # use pd.unique instead of np.unique because that keeps the order of the labels,
        # which is important to keep them in sync with the ones used in
        # FacetGrid.add_legend
        assert dhue is not None
        for label in pd.unique(dhue.values.ravel()):
            mask = dhue == label
            if data["sizes"] is not None:
                kwargs.update(s=data["sizes"].where(mask, drop=True).values.flatten())

            primitive.append(
                ax.scatter(
                    dx.where(mask, drop=True).values.flatten(),
                    dy.where(mask, drop=True).values.flatten(),
                    label=label,
                    **kwargs,
                )
            )

    elif hue is None or hue_style == "continuous":
        if data["sizes"] is not None:
            kwargs.update(s=data["sizes"].values.ravel())
        if dhue is not None:
            kwargs.update(c=dhue.values.ravel())

        dx = data["x"]
        assert dx is not None
        primitive = ax.scatter(
            dx.values.ravel(), dy.values.ravel(), **cmap_params, **kwargs
        )

    return primitive


@overload
def quiver(
    ds: Dataset,
    *args: Any,
    x: Hashable | None = None,
    y: Hashable | None = None,
    u: Hashable | None = None,
    v: Hashable | None = None,
    hue: Hashable | None = None,
    hue_style: HueStyleOptions = None,
    col: None = None,  # no wrap -> primitive
    row: None = None,  # no wrap -> primitive
    ax: Axes | None = None,
    figsize: Iterable[float] | None = None,
    size: float | None = None,
    col_wrap: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    aspect: AspectOptions = None,
    subplot_kws: dict[str, Any] | None = None,
    add_guide: bool | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    infer_intervals=None,
    center=None,
    levels=None,
    robust: bool | None = None,
    colors=None,
    extend=None,
    cmap=None,
    **kwargs: Any,
) -> Quiver:
    ...


@overload
def quiver(
    ds: Dataset,
    *args: Any,
    x: Hashable | None = None,
    y: Hashable | None = None,
    u: Hashable | None = None,
    v: Hashable | None = None,
    hue: Hashable | None = None,
    hue_style: HueStyleOptions = None,
    col: Hashable,  # wrap -> FacetGrid
    row: Hashable | None = None,
    ax: Axes | None = None,
    figsize: Iterable[float] | None = None,
    size: float | None = None,
    col_wrap: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    aspect: AspectOptions = None,
    subplot_kws: dict[str, Any] | None = None,
    add_guide: bool | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    infer_intervals=None,
    center=None,
    levels=None,
    robust: bool | None = None,
    colors=None,
    extend=None,
    cmap=None,
    **kwargs: Any,
) -> FacetGrid[Dataset]:
    ...


@overload
def quiver(
    ds: Dataset,
    *args: Any,
    x: Hashable | None = None,
    y: Hashable | None = None,
    u: Hashable | None = None,
    v: Hashable | None = None,
    hue: Hashable | None = None,
    hue_style: HueStyleOptions = None,
    col: Hashable | None = None,
    row: Hashable,  # wrap -> FacetGrid
    ax: Axes | None = None,
    figsize: Iterable[float] | None = None,
    size: float | None = None,
    col_wrap: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    aspect: AspectOptions = None,
    subplot_kws: dict[str, Any] | None = None,
    add_guide: bool | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    infer_intervals=None,
    center=None,
    levels=None,
    robust: bool | None = None,
    colors=None,
    extend=None,
    cmap=None,
    **kwargs: Any,
) -> FacetGrid[Dataset]:
    ...


@_dsplot
def quiver(
    ds: Dataset,
    x: Hashable,
    y: Hashable,
    ax: Axes,
    u: Hashable,
    v: Hashable,
    **kwargs: Any,
) -> Quiver:
    """Quiver plot of Dataset variables.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.quiver`.
    """
    import matplotlib as mpl

    if x is None or y is None or u is None or v is None:
        raise ValueError("Must specify x, y, u, v for quiver plots.")

    dx, dy, du, dv = broadcast(ds[x], ds[y], ds[u], ds[v])

    args = [dx.values, dy.values, du.values, dv.values]
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


@overload
def streamplot(
    ds: Dataset,
    *args: Any,
    x: Hashable | None = None,
    y: Hashable | None = None,
    u: Hashable | None = None,
    v: Hashable | None = None,
    hue: Hashable | None = None,
    hue_style: HueStyleOptions = None,
    col: None = None,  # no wrap -> primitive
    row: None = None,  # no wrap -> primitive
    ax: Axes | None = None,
    figsize: Iterable[float] | None = None,
    size: float | None = None,
    col_wrap: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    aspect: AspectOptions = None,
    subplot_kws: dict[str, Any] | None = None,
    add_guide: bool | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    infer_intervals=None,
    center=None,
    levels=None,
    robust: bool | None = None,
    colors=None,
    extend=None,
    cmap=None,
    **kwargs: Any,
) -> LineCollection:
    ...


@overload
def streamplot(
    ds: Dataset,
    *args: Any,
    x: Hashable | None = None,
    y: Hashable | None = None,
    u: Hashable | None = None,
    v: Hashable | None = None,
    hue: Hashable | None = None,
    hue_style: HueStyleOptions = None,
    col: Hashable,  # wrap -> FacetGrid
    row: Hashable | None = None,
    ax: Axes | None = None,
    figsize: Iterable[float] | None = None,
    size: float | None = None,
    col_wrap: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    aspect: AspectOptions = None,
    subplot_kws: dict[str, Any] | None = None,
    add_guide: bool | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    infer_intervals=None,
    center=None,
    levels=None,
    robust: bool | None = None,
    colors=None,
    extend=None,
    cmap=None,
    **kwargs: Any,
) -> FacetGrid[Dataset]:
    ...


@overload
def streamplot(
    ds: Dataset,
    *args: Any,
    x: Hashable | None = None,
    y: Hashable | None = None,
    u: Hashable | None = None,
    v: Hashable | None = None,
    hue: Hashable | None = None,
    hue_style: HueStyleOptions = None,
    col: Hashable | None = None,
    row: Hashable,  # wrap -> FacetGrid
    ax: Axes | None = None,
    figsize: Iterable[float] | None = None,
    size: float | None = None,
    col_wrap: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    aspect: AspectOptions = None,
    subplot_kws: dict[str, Any] | None = None,
    add_guide: bool | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    infer_intervals=None,
    center=None,
    levels=None,
    robust: bool | None = None,
    colors=None,
    extend=None,
    cmap=None,
    **kwargs: Any,
) -> FacetGrid[Dataset]:
    ...


@_dsplot
def streamplot(
    ds: Dataset,
    x: Hashable,
    y: Hashable,
    ax: Axes,
    u: Hashable,
    v: Hashable,
    **kwargs: Any,
) -> LineCollection:
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

    dx, dy, du, dv = broadcast(ds[x], ds[y], ds[u], ds[v])

    if xdim is not None and ydim is not None:
        # Need to ensure the arrays are transposed correctly
        dx = dx.transpose(ydim, xdim)
        dy = dy.transpose(ydim, xdim)
        du = du.transpose(ydim, xdim)
        dv = dv.transpose(ydim, xdim)

    args = [dx.values, dy.values, du.values, dv.values]
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
