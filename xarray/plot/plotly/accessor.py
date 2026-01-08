"""
Accessor classes for Plotly Express plotting on DataArray and Dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xarray.plot.plotly import dataarray_plot, dataset_plot
from xarray.plot.plotly.common import SlotValue, auto

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset


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
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive line plot using Plotly Express.

        Dimensions are assigned to plot slots by their order:
        x → color → facet_col → facet_row → animation_frame

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        color : str, auto, or None
            Dimension for color grouping. Default: second dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Default: third dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Default: fourth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Default: fifth dimension.
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
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive bar chart using Plotly Express.

        Dimensions are assigned to plot slots by their order:
        x → color → facet_col → facet_row → animation_frame

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        color : str, auto, or None
            Dimension for color grouping. Default: second dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Default: third dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Default: fourth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Default: fifth dimension.
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
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive stacked area chart using Plotly Express.

        Dimensions are assigned to plot slots by their order:
        x → color → facet_col → facet_row → animation_frame

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        color : str, auto, or None
            Dimension for color/stacking. Default: second dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Default: third dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Default: fourth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Default: fifth dimension.
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
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def scatter(
        self,
        *,
        x: SlotValue = auto,
        y: SlotValue = auto,
        color: SlotValue = auto,
        size: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive scatter plot using Plotly Express.

        Dimensions are assigned to plot slots by their order:
        x → y → color → size → facet_col → facet_row → animation_frame

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        y : str, auto, or None
            Dimension for y-axis. Default: second dimension.
        color : str, auto, or None
            Dimension for color grouping. Default: third dimension.
        size : str, auto, or None
            Dimension for marker size. Default: fourth dimension.
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

        Dimensions are assigned to plot slots by their order:
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
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive heatmap image using Plotly Express.

        Dimensions are assigned to plot slots by their order:
        x → y → facet_col → facet_row → animation_frame

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        y : str, auto, or None
            Dimension for y-axis. Default: second dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Default: third dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Default: fourth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Default: fifth dimension.
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
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )


class DatasetPlotlyAccessor:
    """
    Enables use of Plotly Express plotting functions on a Dataset.

    For Datasets with multiple data variables, 'variable' is treated as a
    pseudo-dimension that can be assigned to any slot.

    Methods return Plotly Figure objects for interactive visualization.

    Examples
    --------
    >>> ds = xr.Dataset(
    ...     {
    ...         "temp": (["time", "city"], np.random.rand(10, 3)),
    ...         "humidity": (["time", "city"], np.random.rand(10, 3)),
    ...     }
    ... )
    >>> fig = ds.plotly.line()  # time→x, city→color, variable→facet_col
    >>> fig = ds.plotly.line(color="variable")  # Compare temp vs humidity
    """

    _ds: Dataset

    __slots__ = ("_ds",)

    def __init__(self, dataset: Dataset) -> None:
        self._ds = dataset

    def line(
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
        Create an interactive line plot using Plotly Express.

        Dimensions are assigned to plot slots by their order. For multi-variable
        Datasets, 'variable' is an additional pseudo-dimension.

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        color : str, auto, or None
            Dimension for color grouping. Can be 'variable'. Default: second dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Can be 'variable'. Default: third dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Can be 'variable'. Default: fourth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Can be 'variable'. Default: fifth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.line()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return dataset_plot.line(
            self._ds,
            x=x,
            color=color,
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
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive bar chart using Plotly Express.

        Dimensions are assigned to plot slots by their order. For multi-variable
        Datasets, 'variable' is an additional pseudo-dimension.

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        color : str, auto, or None
            Dimension for color grouping. Can be 'variable'. Default: second dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Can be 'variable'. Default: third dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Can be 'variable'. Default: fourth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Can be 'variable'. Default: fifth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.bar()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return dataset_plot.bar(
            self._ds,
            x=x,
            color=color,
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
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive stacked area chart using Plotly Express.

        Dimensions are assigned to plot slots by their order. For multi-variable
        Datasets, 'variable' is an additional pseudo-dimension.

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        color : str, auto, or None
            Dimension for color/stacking. Can be 'variable'. Default: second dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Can be 'variable'. Default: third dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Can be 'variable'. Default: fourth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Can be 'variable'. Default: fifth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.area()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return dataset_plot.area(
            self._ds,
            x=x,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def scatter(
        self,
        *,
        x: SlotValue = auto,
        y: SlotValue = auto,
        color: SlotValue = auto,
        size: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive scatter plot using Plotly Express.

        Dimensions are assigned to plot slots by their order. For multi-variable
        Datasets, 'variable' is an additional pseudo-dimension.

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        y : str, auto, or None
            Dimension for y-axis. Default: second dimension.
        color : str, auto, or None
            Dimension for color grouping. Can be 'variable'. Default: third dimension.
        size : str, auto, or None
            Dimension for marker size. Can be 'variable'. Default: fourth dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Can be 'variable'. Default: fifth dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Can be 'variable'. Default: sixth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Can be 'variable'. Default: seventh dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.scatter()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return dataset_plot.scatter(
            self._ds,
            x=x,
            y=y,
            color=color,
            size=size,
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

        Dimensions are assigned to plot slots by their order. For multi-variable
        Datasets, 'variable' is an additional pseudo-dimension.

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis categories. Default: first dimension.
        color : str, auto, or None
            Dimension for color grouping. Can be 'variable'. Default: second dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Can be 'variable'. Default: third dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Can be 'variable'. Default: fourth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Can be 'variable'. Default: fifth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.box()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return dataset_plot.box(
            self._ds,
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
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive heatmap image using Plotly Express.

        Dimensions are assigned to plot slots by their order. For multi-variable
        Datasets, 'variable' is an additional pseudo-dimension.

        Parameters
        ----------
        x : str, auto, or None
            Dimension for x-axis. Default: first dimension.
        y : str, auto, or None
            Dimension for y-axis. Default: second dimension.
        facet_col : str, auto, or None
            Dimension for subplot columns. Can be 'variable'. Default: third dimension.
        facet_row : str, auto, or None
            Dimension for subplot rows. Can be 'variable'. Default: fourth dimension.
        animation_frame : str, auto, or None
            Dimension for animation. Can be 'variable'. Default: fifth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.imshow()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return dataset_plot.imshow(
            self._ds,
            x=x,
            y=y,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )
