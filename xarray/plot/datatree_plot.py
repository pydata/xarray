from __future__ import annotations

import functools
from collections.abc import Callable, Hashable, Iterable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from xarray.plot import dataarray_plot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.colors import Colormap, Normalize
    from numpy.typing import ArrayLike

    from xarray.core.dataarray import DataArray
    from xarray.core.datatree import DataTree
    from xarray.core.types import (
        ExtendOptions,
        ScaleOptions,
    )
    from xarray.plot.facetgrid import FacetGrid


F = TypeVar("F", bound=Callable)


def _update_doc_to_datatree(dataarray_plotfunc: Callable) -> Callable[[F], F]:
    """
    Add a common docstring by reusing the DataArray one.

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


@_update_doc_to_datatree(dataarray_plot.scatter)
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
