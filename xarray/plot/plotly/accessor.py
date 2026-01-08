"""
Accessor classes for Plotly Express plotting on DataArray.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xarray.plot.plotly import dataarray_plot
from xarray.plot.plotly.common import SlotValue, auto

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from xarray.core.dataarray import DataArray


class DataArrayPlotlyAccessor:
    """
    Enables use of Plotly Express plotting functions on a DataArray.

    Methods return Plotly Figure objects for interactive visualization.

    Examples
    --------
    >>> da = xr.DataArray(np.random.rand(10, 3), dims=["time", "city"])
    >>> fig = da.plotly.line()  # Auto-assign dims: time→x, city→color
    >>> fig.show()

    >>> fig = da.plotly.line(x="time", color=None)  # Explicit assignment
    >>> fig.update_layout(title="My Plot")
    """

    _da: DataArray

    __slots__ = ("_da",)

    def __init__(self, darray: DataArray) -> None:
        self._da = darray

    def line(
        self,
        *,
        x: SlotValue = auto,
        color: SlotValue = auto,
        line_dash: SlotValue = auto,
        symbol: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive line plot using Plotly Express.

        The y-axis always shows the DataArray values. Dimensions are assigned
        to other slots by their order:
        x → color → line_dash → symbol → facet_col → facet_row → animation_frame

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        color : str, auto, or None
            Dimension for color grouping. Default: second dimension.
        line_dash : str, auto, or None
            Dimension for line dash style. Default: third dimension.
        symbol : str, auto, or None
            Dimension for marker symbol. Default: fourth dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Default: fifth dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Default: sixth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Default: seventh dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.line()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return dataarray_plot.line(
            self._da,
            x=x,
            color=color,
            line_dash=line_dash,
            symbol=symbol,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def bar(
        self,
        *,
        x: SlotValue = auto,
        color: SlotValue = auto,
        pattern_shape: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive bar chart using Plotly Express.

        The y-axis always shows the DataArray values. Dimensions are assigned
        to other slots by their order:
        x → color → pattern_shape → facet_col → facet_row → animation_frame

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        color : str, auto, or None
            Dimension for color grouping. Default: second dimension.
        pattern_shape : str, auto, or None
            Dimension for bar fill pattern. Default: third dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Default: fourth dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Default: fifth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Default: sixth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.bar()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return dataarray_plot.bar(
            self._da,
            x=x,
            color=color,
            pattern_shape=pattern_shape,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def area(
        self,
        *,
        x: SlotValue = auto,
        color: SlotValue = auto,
        pattern_shape: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive stacked area chart using Plotly Express.

        The y-axis always shows the DataArray values. Dimensions are assigned
        to other slots by their order:
        x → color → pattern_shape → facet_col → facet_row → animation_frame

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        color : str, auto, or None
            Dimension for color/stacking. Default: second dimension.
        pattern_shape : str, auto, or None
            Dimension for fill pattern. Default: third dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Default: fourth dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Default: fifth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Default: sixth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.area()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return dataarray_plot.area(
            self._da,
            x=x,
            color=color,
            pattern_shape=pattern_shape,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def scatter(
        self,
        *,
        x: SlotValue = auto,
        y: SlotValue | str = "value",
        color: SlotValue = auto,
        size: SlotValue = auto,
        symbol: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive scatter plot using Plotly Express.

        By default, y-axis shows the DataArray values. Set y to a dimension
        name to create dimension-vs-dimension plots (e.g., lat vs lon colored
        by value).

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        y : str
            What to plot on y-axis. Default "value" uses DataArray values.
            Can be a dimension name for dimension vs dimension plots.
        color : str, auto, None, or "value"
            Dimension for color grouping. Default: second dimension.
            Use "value" to color by DataArray values (useful with y=dimension).
        size : str, auto, or None
            Dimension for marker size. Default: third dimension.
        symbol : str, auto, or None
            Dimension for marker symbol. Default: fourth dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Default: fifth dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Default: sixth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Default: seventh dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.scatter()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return dataarray_plot.scatter(
            self._da,
            x=x,
            y=y,
            color=color,
            size=size,
            symbol=symbol,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def box(
        self,
        *,
        x: SlotValue = auto,
        color: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive box plot using Plotly Express.

        The y-axis always shows the DataArray values. Dimensions are assigned
        to other slots by their order:
        x → color → facet_col → facet_row → animation_frame

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis categories. Default: first dimension.
        color : str, auto, or None
            Dimension for color grouping. Default: second dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Default: third dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Default: fourth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Default: fifth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.box()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return dataarray_plot.box(
            self._da,
            x=x,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def imshow(
        self,
        *,
        x: SlotValue = auto,
        y: SlotValue = auto,
        facet_col: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive heatmap image using Plotly Express.

        Dimensions are assigned to plot slots by their order:
        x → y → facet_col → animation_frame

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        y : str, auto, or None
            Dimension for y-axis. Default: second dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Default: third dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Default: fourth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.imshow()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return dataarray_plot.imshow(
            self._da,
            x=x,
            y=y,
            facet_col=facet_col,
            animation_frame=animation_frame,
            **px_kwargs,
        )
