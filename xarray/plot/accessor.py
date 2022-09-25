from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Literal, overload

import numpy as np

from . import dataarray_plot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import Normalize
    from matplotlib.container import BarContainer
    from matplotlib.contour import QuadContourSet
    from matplotlib.image import AxesImage
    from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection

    from ..core.dataarray import DataArray
    from ..core.npcompat import ArrayLike
    from ..core.types import MPLAspectOptions, MPLScaleOptions
    from .facetgrid import FacetGrid


class DataArrayPlotAccessor:
    """
    Enables use of xarray.plot functions as attributes on a DataArray.
    For example, DataArray.plot.imshow
    """

    _da: DataArray

    __slots__ = ("_da",)
    __doc__ = dataarray_plot.plot.__doc__

    def __init__(self, darray: DataArray) -> None:
        self._da = darray

    # Should return Any such that the user does not run into problems
    # with the many possible return values
    @functools.wraps(dataarray_plot.plot, assigned=("__doc__", "__annotations__"))
    def __call__(self, **kwargs) -> Any:
        return dataarray_plot.plot(self._da, **kwargs)

    @functools.wraps(dataarray_plot.hist)
    def hist(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray, BarContainer]:
        return dataarray_plot.hist(self._da, *args, **kwargs)

    @overload
    def line(
        self,
        *args: Any,
        row: None = None,  # no wrap -> primitive
        col: None = None,  # no wrap -> primitive
        figsize: Iterable[float] | None = None,
        aspect: MPLAspectOptions = None,
        size: float | None = None,
        ax: Axes | None = None,
        hue: Hashable | None = None,
        x: Hashable | None = None,
        y: Hashable | None = None,
        xincrease: bool | None = None,
        yincrease: bool | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        add_legend: bool = True,
        _labels: bool = True,
        **kwargs: Any,
    ) -> list[Line3D]:
        ...

    @overload
    def line(
        self,
        *args: Any,
        row: Hashable,  # wrap -> FacetGrid
        col: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        aspect: MPLAspectOptions = None,
        size: float | None = None,
        ax: Axes | None = None,
        hue: Hashable | None = None,
        x: Hashable | None = None,
        y: Hashable | None = None,
        xincrease: bool | None = None,
        yincrease: bool | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        add_legend: bool = True,
        _labels: bool = True,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @overload
    def line(
        self,
        *args: Any,
        row: Hashable | None = None,
        col: Hashable,  # wrap -> FacetGrid
        figsize: Iterable[float] | None = None,
        aspect: MPLAspectOptions = None,
        size: float | None = None,
        ax: Axes | None = None,
        hue: Hashable | None = None,
        x: Hashable | None = None,
        y: Hashable | None = None,
        xincrease: bool | None = None,
        yincrease: bool | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        add_legend: bool = True,
        _labels: bool = True,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @functools.wraps(dataarray_plot.line)
    def line(self, *args, **kwargs) -> list[Line3D] | FacetGrid:
        return dataarray_plot.line(self._da, *args, **kwargs)

    @overload
    def step(
        self,
        *args: Any,
        where: Literal["pre", "post", "mid"] = "pre",
        drawstyle: str | None = None,
        ds: str | None = None,
        row: None = None,  # no wrap -> primitive
        col: None = None,  # no wrap -> primitive
        **kwargs: Any,
    ) -> list[Line3D]:
        ...

    @overload
    def step(
        self,
        *args: Any,
        where: Literal["pre", "post", "mid"] = "pre",
        drawstyle: str | None = None,
        ds: str | None = None,
        row: Hashable,  # wrap -> FacetGrid
        col: Hashable | None = None,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @overload
    def step(
        self,
        *args: Any,
        where: Literal["pre", "post", "mid"] = "pre",
        drawstyle: str | None = None,
        ds: str | None = None,
        row: Hashable | None = None,
        col: Hashable,  # wrap -> FacetGrid
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @functools.wraps(dataarray_plot.step)
    def step(self, *args, **kwargs) -> list[Line3D] | FacetGrid:
        return dataarray_plot.step(self._da, *args, **kwargs)

    @functools.wraps(dataarray_plot.scatter)
    def _scatter(self, *args, **kwargs):
        return dataarray_plot.scatter(self._da, *args, **kwargs)

    @overload
    def imshow(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        row: None = None,  # no wrap -> primitive
        col: None = None,  # no wrap -> primitive
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> AxesImage:
        ...

    @overload
    def imshow(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        row: Hashable | None = None,
        *,
        col: Hashable,  # wrap -> FacetGrid
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @overload
    def imshow(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        *,
        row: Hashable,  # wrap -> FacetGrid
        col: Hashable | None = None,
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @functools.wraps(dataarray_plot.imshow)
    def imshow(self, *args, **kwargs) -> AxesImage:
        return dataarray_plot.imshow(self._da, *args, **kwargs)

    @overload
    def contour(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        row: None = None,  # no wrap -> primitive
        col: None = None,  # no wrap -> primitive
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> QuadContourSet:
        ...

    @overload
    def contour(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        row: Hashable | None = None,
        *,
        col: Hashable,  # wrap -> FacetGrid
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @overload
    def contour(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        *,
        row: Hashable,  # wrap -> FacetGrid
        col: Hashable | None = None,
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @functools.wraps(dataarray_plot.contour)
    def contour(self, *args, **kwargs) -> QuadContourSet:
        return dataarray_plot.contour(self._da, *args, **kwargs)

    @overload
    def contourf(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        row: None = None,  # no wrap -> primitive
        col: None = None,  # no wrap -> primitive
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> QuadContourSet:
        ...

    @overload
    def contourf(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        row: Hashable | None = None,
        *,
        col: Hashable,  # wrap -> FacetGrid
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @overload
    def contourf(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        *,
        row: Hashable,  # wrap -> FacetGrid
        col: Hashable | None = None,
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @functools.wraps(dataarray_plot.contourf)
    def contourf(self, *args, **kwargs) -> QuadContourSet:
        return dataarray_plot.contourf(self._da, *args, **kwargs)

    @overload
    def pcolormesh(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        row: None = None,  # no wrap -> primitive
        col: None = None,  # no wrap -> primitive
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> QuadMesh:
        ...

    @overload
    def pcolormesh(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        row: Hashable | None = None,
        *,
        col: Hashable,  # wrap -> FacetGrid
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @overload
    def pcolormesh(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        *,
        row: Hashable,  # wrap -> FacetGrid
        col: Hashable | None = None,
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @functools.wraps(dataarray_plot.pcolormesh)
    def pcolormesh(self, *args, **kwargs) -> QuadMesh:
        return dataarray_plot.pcolormesh(self._da, *args, **kwargs)

    @overload
    def surface(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        row: None = None,  # no wrap -> primitive
        col: None = None,  # no wrap -> primitive
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> Poly3DCollection:
        ...

    @overload
    def surface(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        row: Hashable | None = None,
        *,
        col: Hashable,  # wrap -> FacetGrid
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @overload
    def surface(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        figsize: Iterable[float] | None = None,
        size: float | None = None,
        aspect: MPLAspectOptions = None,
        ax: Axes | None = None,
        *,
        row: Hashable,  # wrap -> FacetGrid
        col: Hashable | None = None,
        col_wrap: int | None = None,
        xincrease: bool | None = True,
        yincrease: bool | None = True,
        add_colorbar: bool | None = None,
        add_labels: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        center=None,
        robust: bool = False,
        extend=None,
        levels=None,
        infer_intervals=None,
        colors=None,
        subplot_kws: dict[str, Any] | None = None,
        cbar_ax: Axes | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        xscale: MPLScaleOptions = None,
        yscale: MPLScaleOptions = None,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
        xlim: ArrayLike | None = None,
        ylim: ArrayLike | None = None,
        norm: Normalize | None = None,
        **kwargs: Any,
    ) -> FacetGrid:
        ...

    @functools.wraps(dataarray_plot.surface)
    def surface(self, *args, **kwargs) -> Poly3DCollection:
        return dataarray_plot.surface(self._da, *args, **kwargs)
