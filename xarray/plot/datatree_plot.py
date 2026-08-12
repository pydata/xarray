from __future__ import annotations

import functools
from collections.abc import Callable, Hashable, Iterable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from xarray.plot.facetgrid import _easy_facetgrid
from xarray.plot.utils import (
    _add_colorbar,
    _infer_meta_data,
    _process_cmap_cbar_kwargs,
    get_axis,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.colors import Colormap, Normalize
    from numpy.typing import ArrayLike

    from xarray.core.dataarray import DataArray
    from xarray.core.datatree import DataTree
    from xarray.core.types import (
        ExtendOptions,
        HueStyleOptions,
        ScaleOptions,
    )
    from xarray.plot.facetgrid import FacetGrid


def _dtplot(plotfunc):
    commondoc = """
Parameters
----------
dt : DataTree
variable : str
    name of the variable in multiple nodes,
x : Hashable or None, optional
    Variable name for x-axis.
y : Hashable or None, optional
    Variable name for y-axis.
z : Hashable or None, optional
    if specified plot 3D and use this coordinate for z axis.
hue: Hashable or None, optional
    Variable by which to color scatter points or arrows.
hue_style: {'continuous', 'discrete'} or None, optional
    How to use the ``hue`` variable:

    - ``'continuous'`` -- continuous color scale
        (default for numeric ``hue`` variables)
    - ``'discrete'`` -- a color for each unique value, using the default color cycle
        (default for non-numeric ``hue`` variables)

row : Hashable or None, optional
    If passed, make row faceted plots on this dimension name.
col : Hashable or None, optional
    If passed, make column faceted plots on this dimension name.
col_wrap : int, None or "auto", optional
    "Wrap" the grid for the column variable after this number of columns,
    adding rows if ``col_wrap`` is less than the number of facets.
    If "auto" align the grid to the figsize or keep it as square as possible.
subplot_kws : dict or None, optional
    Dictionary of keyword arguments for Matplotlib subplots
    (see :py:meth:`matplotlib:matplotlib.figure.Figure.add_subplot`).
    Only applies to FacetGrid plotting.
cmap : matplotlib colormap name or colormap, optional
    The mapping from data values to color space. Either a
    Matplotlib colormap name or object. If not provided, this will
    be either ``'viridis'`` (if the function infers a sequential
    dataset) or ``'RdBu_r'`` (if the function infers a diverging
    dataset).
    See :doc:`Choosing Colormaps in Matplotlib <matplotlib:users/explain/colors/colormaps>`
    for more information.

    If *seaborn* is installed, ``cmap`` may also be a
    `seaborn color palette <https://seaborn.pydata.org/tutorial/color_palettes.html>`_.
    Note: if ``cmap`` is a seaborn color palette,
    ``levels`` must also be specified.
vmin : float or None, optional
    Lower value to anchor the colormap, otherwise it is inferred from the
    data and other keyword arguments. When a diverging dataset is inferred,
    setting `vmin` or `vmax` will fix the other by symmetry around
    ``center``. Setting both values prevents use of a diverging colormap.
    If discrete levels are provided as an explicit list, both of these
    values are ignored.
vmax : float or None, optional
    Upper value to anchor the colormap, otherwise it is inferred from the
    data and other keyword arguments. When a diverging dataset is inferred,
    setting `vmin` or `vmax` will fix the other by symmetry around
    ``center``. Setting both values prevents use of a diverging colormap.
    If discrete levels are provided as an explicit list, both of these
    values are ignored.
norm : matplotlib.colors.Normalize, optional
    If ``norm`` has ``vmin`` or ``vmax`` specified, the corresponding
    kwarg must be ``None``.
extend : {'neither', 'both', 'min', 'max'}, optional
    How to draw arrows extending the colorbar beyond its limits. If not
    provided, ``extend`` is inferred from ``vmin``, ``vmax`` and the data limits.
levels : int or array-like, optional
    Split the colormap (``cmap``) into discrete color intervals. If an integer
    is provided, "nice" levels are chosen based on the data range: this can
    imply that the final number of levels is not exactly the expected one.
    Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
    setting ``levels=np.linspace(vmin, vmax, N)``.
fig_kw : Hashable or None, optional
    Matplotlib kwargs that get passed to pyplot.figure
**kwargs : optional
    Additional keyword arguments to wrapped Matplotlib function.
    """

    # Build on the original docstring
    plotfunc.__doc__ = f"{plotfunc.__doc__}\n{commondoc}"

    @functools.wraps(
        plotfunc, assigned=("__module__", "__name__", "__qualname__", "__doc__")
    )
    def newplotfunc(
        dt: DataTree,
        *args: Any,
        x: Hashable | None = None,
        y: Hashable | None = None,
        z: Hashable | None = None,
        hue: Hashable | None = None,
        hue_style: HueStyleOptions = None,
        row: Hashable | None = None,
        col: Hashable | None = None,
        col_wrap: int | Literal["auto"] | None = None,
        subplot_kws: dict[str, Any] | None = None,
        cmap: str | Colormap | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        norm: Normalize | None = None,
        extend: ExtendOptions = None,
        levels: ArrayLike | None = None,
        **kwargs: Any,
    ) -> Any:

        _is_facetgrid = kwargs.pop("_is_facetgrid", False)
        if _is_facetgrid:  # facetgrid call
            meta_data = kwargs.pop("meta_data")
        else:
            meta_data = _infer_meta_data(
                dt, x, y, z, hue, hue_style, funcname=plotfunc.__name__
            )

        hue_style = meta_data["hue_style"]

        # handle facetgridt first
        if col or row:
            allargs = locals().copy()
            allargs["plotfunc"] = globals()[plotfunc.__name__]
            allargs["data"] = dt
            # remove kwargs to avoid passing the information twice
            for arg in ["meta_data", "kwargs", "dt"]:
                del allargs[arg]

            return _easy_facetgrid(kind="dataarray", **allargs, **kwargs)

        figsize = kwargs.pop("figsize", None)
        ax = get_axis(figsize)

        if hue_style == "continuous" and hue is not None:
            if _is_facetgrid:
                cbar_kwargs = meta_data["cbar_kwargs"]
                cmap_params = meta_data["cmap_params"]
            else:
                cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(
                    plotfunc, dt[hue].values, **locals()
                )

            # subset that can be passed to scatter, hist2d
            cmap_params_subset = {
                vv: cmap_params[vv] for vv in ["vmin", "vmax", "norm", "cmap"]
            }

        else:
            cmap_params_subset = {}

        primitive = plotfunc(
            dt=dt,
            x=x,
            y=y,
            ax=ax,
            hue=hue,
            hue_style=hue_style,
            cmap_params=cmap_params_subset,
            **kwargs,
        )

        if _is_facetgrid:  # if this was called from Facetgrid.map_datatree,
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
            _add_colorbar(primitive, ax, cbar_kwargs, cmap_params)

        title = dt[x]._title_for_slice()
        ax.set_title(title)

        return primitive

    # we want to actually expose the signature of newplotfunc
    # and not the copied **kwargs from the plotfunc which
    # functools.wraps addt, so delete the wrapped attr
    del newplotfunc.__wrapped__

    return newplotfunc


F = TypeVar("F", bound=Callable)


def _update_doc_to_datatree(dataarray_plotfunc: Callable) -> Callable[[F], F]:
    """
    Add a common docstring by reusing the DataArray one.

    TODO: Reduce code duplication.

    * The goal is to reduce code duplication by mov all DataTree
      specific plots to the DataArray side and use this thin wrapper to
      handle the converts between DataTree and DataArray.
    * Improve docstring handling, maybe reword the DataArray versions to explain DataTrees better.

    Parameters
    ----------
    dataarray_plotfunc : Callable
        Function that returns a finished plot primitive.
    """

    # Build on the original docstring
    da_doc = dataarray_plotfunc.__doc__
    if da_doc is None:
        raise NotImplementedError("DataArray plot method requires a docstring")

    da_str = """
    Parameters
    ----------
    darray : DataArray
    """
    dt_str = """

    The `y` DataArray will be used as base, any other variables are added as coordt.

    Parameters
    ----------
 : DataTree
    """
    # TODO: improve this?
    if da_str in da_doc:
        dt_doc = da_doc.replace(da_str, dt_str).replace("darray", "dt")
    else:
        dt_doc = da_doc

    @functools.wraps(dataarray_plotfunc)
    def wrapper(datatree_plotfunc: F) -> F:
        datatree_plotfunc.__doc__ = dt_doc
        return datatree_plotfunc

    return wrapper  # type: ignore[return-value]


@overload
def scatter(  # type: ignore[misc,unused-ignore]  # None is hashable :(s: DataTree,
    dt: DataTree,
    variable: str,
    ax: Axes | None = None,
    *,
    x: Hashable | None = None,
    y: Hashable | None = None,
    z: Hashable | None = None,
    hue: Hashable | None = None,
    markersize: Hashable | None = None,
    row: Hashable | None = None,
    col: Hashable | None = None,
    col_wrap: int | Literal["auto"] | None = None,
    xincrease: bool | None = True,
    yincrease: bool | None = True,
    add_legend: bool | None = None,
    add_colorbar: bool | None = None,
    add_labels: bool | Iterable[bool] = True,
    add_title: bool = True,
    subplot_kws: dict[str, Any] | None = None,
    xscale: ScaleOptions = None,
    yscale: ScaleOptions = None,
    xticks: ArrayLike | None = None,
    yticks: ArrayLike | None = None,
    xlim: ArrayLike | None = None,
    ylim: ArrayLike | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    extend: ExtendOptions = None,
    levels: ArrayLike | None = None,
    fig_kw: Hashable | None = None,
    **kwargs: Any,
) -> PathCollection: ...


@overload
def scatter(
    dt: DataTree,
    variable: str,
    ax: Axes | None = None,
    *,
    x: Hashable | None = None,
    y: Hashable | None = None,
    z: Hashable | None = None,
    hue: Hashable | None = None,
    markersize: Hashable | None = None,
    row: Hashable | None = None,
    col: Hashable | None = None,
    col_wrap: int | Literal["auto"] | None = None,
    xincrease: bool | None = True,
    yincrease: bool | None = True,
    add_legend: bool | None = None,
    add_colorbar: bool | None = None,
    add_labels: bool | Iterable[bool] = True,
    add_title: bool = True,
    subplot_kws: dict[str, Any] | None = None,
    xscale: ScaleOptions = None,
    yscale: ScaleOptions = None,
    xticks: ArrayLike | None = None,
    yticks: ArrayLike | None = None,
    xlim: ArrayLike | None = None,
    ylim: ArrayLike | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    extend: ExtendOptions = None,
    levels: ArrayLike | None = None,
    fig_kw: Hashable | None = None,
    **kwargs: Any,
) -> FacetGrid[DataArray]: ...


@overload
def scatter(
    dt: DataTree,
    variable: str,
    ax: Axes | None = None,
    *,
    x: Hashable | None = None,
    y: Hashable | None = None,
    z: Hashable | None = None,
    hue: Hashable | None = None,
    markersize: Hashable | None = None,
    row: Hashable | None = None,
    col: Hashable | None = None,
    col_wrap: int | Literal["auto"] | None = None,
    xincrease: bool | None = True,
    yincrease: bool | None = True,
    add_legend: bool | None = None,
    add_colorbar: bool | None = None,
    add_labels: bool | Iterable[bool] = True,
    add_title: bool = True,
    subplot_kws: dict[str, Any] | None = None,
    xscale: ScaleOptions = None,
    yscale: ScaleOptions = None,
    xticks: ArrayLike | None = None,
    yticks: ArrayLike | None = None,
    xlim: ArrayLike | None = None,
    ylim: ArrayLike | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    extend: ExtendOptions = None,
    levels: ArrayLike | None = None,
    fig_kw: Hashable | None = None,
    **kwargs: Any,
) -> FacetGrid[DataArray]: ...


# @_update_doc_to_datatree(dataarray_plot.scatter)
@_dtplot
def scatter(
    dt: DataTree,
    variable: str,
    ax: Axes | None = None,
    *,
    x: Hashable | None = None,
    y: Hashable | None = None,
    z: Hashable | None = None,
    hue: Hashable | None = None,
    markersize: Hashable | None = None,
    row: Hashable | None = None,
    col: Hashable | None = None,
    col_wrap: int | Literal["auto"] | None = None,
    xincrease: bool | None = True,
    yincrease: bool | None = True,
    add_legend: bool | None = None,
    add_colorbar: bool | None = None,
    add_labels: bool | Iterable[bool] = True,
    add_title: bool = True,
    subplot_kws: dict[str, Any] | None = None,
    xscale: ScaleOptions = None,
    yscale: ScaleOptions = None,
    xticks: ArrayLike | None = None,
    yticks: ArrayLike | None = None,
    xlim: ArrayLike | None = None,
    ylim: ArrayLike | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    extend: ExtendOptions = None,
    levels: ArrayLike | None = None,
    fig_kw: Hashable | None = None,
    **kwargs: Any,
) -> PathCollection | FacetGrid[DataArray]:
    """Scat plot DataTree data variables against each other."""

    if fig_kw is None:
        fig_kw = {}

    locals_ = locals()
    del locals_["dt"]
    locals_.update(locals_.pop("kwargs", {}))
    locals_.pop("variable")
    (locals_.pop("ax"),)
    locals_.pop("fig_kw")
    locals_.pop("add_legend")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(**fig_kw)
    for node in dt.descendants:
        try:
            da = node[variable]
        except KeyError as err:
            raise KeyError(f"{variable} not found at node: {node.name}") from err

        da.plot.scatter(
            ax=ax,
            label=node.name,
            **locals_,
        )
    if add_legend:
        ax.legend()
