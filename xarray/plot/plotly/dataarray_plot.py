"""
Plotly Express plotting functions for DataArray objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xarray.core.utils import attempt_import
from xarray.plot.plotly.common import (
    SlotValue,
    assign_slots,
    auto,
    dataarray_to_dataframe,
    get_axis_label,
    get_value_label,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from xarray.core.dataarray import DataArray


def line(
    darray: DataArray,
    *,
    x: SlotValue = auto,
    color: SlotValue = auto,
    facet_col: SlotValue = auto,
    facet_row: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive line plot from a DataArray using Plotly Express.

    Parameters
    ----------
    darray : DataArray
        The xarray DataArray to plot.
    x : str, auto, or None
        Dimension for the x-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    color : str, auto, or None
        Dimension for color grouping. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    facet_col : str, auto, or None
        Dimension for subplot columns. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    facet_row : str, auto, or None
        Dimension for subplot rows. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    animation_frame : str, auto, or None
        Dimension for animation frames. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    **px_kwargs
        Additional keyword arguments passed to `plotly.express.line()`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.

    Examples
    --------
    >>> da = xr.DataArray(
    ...     np.random.rand(10, 3),
    ...     dims=["time", "city"],
    ...     coords={"time": pd.date_range("2020", periods=10), "city": ["NYC", "LA", "Chicago"]},
    ...     name="temperature",
    ... )
    >>> fig = da.pxplot.line()  # time→x, city→color
    >>> fig = da.pxplot.line(color=None)  # time→x, city→facet_col
    >>> fig = da.pxplot.line(x="city", color="time")  # explicit assignment
    """
    px = attempt_import("plotly.express")

    slots = assign_slots(
        list(darray.dims),
        "line",
        x=x,
        color=color,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = dataarray_to_dataframe(darray)
    value_col = darray.name if darray.name is not None else "value"

    # Build labels for axes
    labels = px_kwargs.pop("labels", {})
    if "x" in slots and slots["x"] not in labels:
        labels[str(slots["x"])] = get_axis_label(darray, slots["x"])
    if value_col not in labels:
        labels[value_col] = get_value_label(darray)

    fig = px.line(
        df,
        x=slots.get("x"),
        y=value_col,
        color=slots.get("color"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        labels=labels,
        **px_kwargs,
    )

    return fig


def bar(
    darray: DataArray,
    *,
    x: SlotValue = auto,
    color: SlotValue = auto,
    facet_col: SlotValue = auto,
    facet_row: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive bar chart from a DataArray using Plotly Express.

    Parameters
    ----------
    darray : DataArray
        The xarray DataArray to plot.
    x : str, auto, or None
        Dimension for the x-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    color : str, auto, or None
        Dimension for color grouping. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    facet_col : str, auto, or None
        Dimension for subplot columns. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    facet_row : str, auto, or None
        Dimension for subplot rows. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    animation_frame : str, auto, or None
        Dimension for animation frames. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    **px_kwargs
        Additional keyword arguments passed to `plotly.express.bar()`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.

    Examples
    --------
    >>> da = xr.DataArray(
    ...     np.random.rand(3, 2),
    ...     dims=["city", "scenario"],
    ...     coords={"city": ["NYC", "LA", "Chicago"], "scenario": ["baseline", "warming"]},
    ...     name="temperature",
    ... )
    >>> fig = da.pxplot.bar()  # city→x, scenario→color
    """
    px = attempt_import("plotly.express")

    slots = assign_slots(
        list(darray.dims),
        "bar",
        x=x,
        color=color,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = dataarray_to_dataframe(darray)
    value_col = darray.name if darray.name is not None else "value"

    # Build labels for axes
    labels = px_kwargs.pop("labels", {})
    if "x" in slots and slots["x"] not in labels:
        labels[str(slots["x"])] = get_axis_label(darray, slots["x"])
    if value_col not in labels:
        labels[value_col] = get_value_label(darray)

    fig = px.bar(
        df,
        x=slots.get("x"),
        y=value_col,
        color=slots.get("color"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        labels=labels,
        **px_kwargs,
    )

    return fig


def area(
    darray: DataArray,
    *,
    x: SlotValue = auto,
    color: SlotValue = auto,
    facet_col: SlotValue = auto,
    facet_row: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive stacked area chart from a DataArray using Plotly Express.

    Parameters
    ----------
    darray : DataArray
        The xarray DataArray to plot.
    x : str, auto, or None
        Dimension for the x-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    color : str, auto, or None
        Dimension for color/stacking grouping. Use `auto` for positional
        assignment, a dimension name for explicit assignment, or `None` to skip.
    facet_col : str, auto, or None
        Dimension for subplot columns. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    facet_row : str, auto, or None
        Dimension for subplot rows. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    animation_frame : str, auto, or None
        Dimension for animation frames. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    **px_kwargs
        Additional keyword arguments passed to `plotly.express.area()`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.

    Examples
    --------
    >>> da = xr.DataArray(
    ...     np.random.rand(10, 3),
    ...     dims=["time", "category"],
    ...     name="sales",
    ... )
    >>> fig = da.pxplot.area()  # time→x, category→color (stacked)
    """
    px = attempt_import("plotly.express")

    slots = assign_slots(
        list(darray.dims),
        "area",
        x=x,
        color=color,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = dataarray_to_dataframe(darray)
    value_col = darray.name if darray.name is not None else "value"

    # Build labels for axes
    labels = px_kwargs.pop("labels", {})
    if "x" in slots and slots["x"] not in labels:
        labels[str(slots["x"])] = get_axis_label(darray, slots["x"])
    if value_col not in labels:
        labels[value_col] = get_value_label(darray)

    fig = px.area(
        df,
        x=slots.get("x"),
        y=value_col,
        color=slots.get("color"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        labels=labels,
        **px_kwargs,
    )

    return fig


def heatmap(
    darray: DataArray,
    *,
    x: SlotValue = auto,
    y: SlotValue = auto,
    facet_col: SlotValue = auto,
    facet_row: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive heatmap from a DataArray using Plotly Express.

    Parameters
    ----------
    darray : DataArray
        The xarray DataArray to plot.
    x : str, auto, or None
        Dimension for the x-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    y : str, auto, or None
        Dimension for the y-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    facet_col : str, auto, or None
        Dimension for subplot columns. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    facet_row : str, auto, or None
        Dimension for subplot rows. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    animation_frame : str, auto, or None
        Dimension for animation frames. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    **px_kwargs
        Additional keyword arguments passed to `plotly.express.imshow()`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.

    Examples
    --------
    >>> da = xr.DataArray(
    ...     np.random.rand(10, 20),
    ...     dims=["y", "x"],
    ...     name="temperature",
    ... )
    >>> fig = da.pxplot.heatmap()  # y→y, x→x
    """
    px = attempt_import("plotly.express")

    slots = assign_slots(
        list(darray.dims),
        "heatmap",
        x=x,
        y=y,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    # For heatmap, we need x and y to use imshow properly
    x_dim = slots.get("x")
    y_dim = slots.get("y")
    facet_col_dim = slots.get("facet_col")
    facet_row_dim = slots.get("facet_row")
    animation_dim = slots.get("animation_frame")

    # Build labels
    labels = px_kwargs.pop("labels", {})
    if x_dim and str(x_dim) not in labels:
        labels[str(x_dim)] = get_axis_label(darray, x_dim)
    if y_dim and str(y_dim) not in labels:
        labels[str(y_dim)] = get_axis_label(darray, y_dim)

    # Get color label
    color_label = get_value_label(darray)

    # Handle faceting and animation for heatmap
    if facet_col_dim or facet_row_dim or animation_dim:
        # Use density_heatmap with DataFrame for faceting
        df = dataarray_to_dataframe(darray)
        value_col = darray.name if darray.name is not None else "value"

        if value_col not in labels:
            labels[value_col] = color_label

        # px.density_heatmap doesn't support facets directly for pre-aggregated data
        # Use imshow with make_subplots for complex cases
        # For now, fall back to a simple implementation

        fig = px.density_heatmap(
            df,
            x=x_dim,
            y=y_dim,
            z=value_col,
            facet_col=facet_col_dim,
            facet_row=facet_row_dim,
            animation_frame=animation_dim,
            labels=labels,
            histfunc="avg",
            **px_kwargs,
        )
    else:
        # Simple 2D case - use imshow directly on the array
        # Transpose to get correct orientation (y on rows, x on columns)
        if x_dim and y_dim:
            plot_data = darray.transpose(y_dim, x_dim)
        else:
            plot_data = darray

        # Get coordinate values for axis labels
        x_coords = None
        y_coords = None
        if x_dim and x_dim in darray.coords:
            x_coords = darray.coords[x_dim].values
        if y_dim and y_dim in darray.coords:
            y_coords = darray.coords[y_dim].values

        fig = px.imshow(
            plot_data.values,
            x=x_coords,
            y=y_coords,
            labels={"x": labels.get(str(x_dim), str(x_dim)) if x_dim else "x",
                    "y": labels.get(str(y_dim), str(y_dim)) if y_dim else "y",
                    "color": color_label},
            **px_kwargs,
        )

    return fig
