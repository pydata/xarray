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
    ...     coords={
    ...         "time": pd.date_range("2020", periods=10),
    ...         "city": ["NYC", "LA", "Chicago"],
    ...     },
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
    ...     coords={
    ...         "city": ["NYC", "LA", "Chicago"],
    ...         "scenario": ["baseline", "warming"],
    ...     },
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


def scatter(
    darray: DataArray,
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
    Create an interactive scatter plot from a DataArray using Plotly Express.

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
    color : str, auto, or None
        Dimension for color grouping. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    size : str, auto, or None
        Dimension for marker size. Use `auto` for positional assignment,
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
        Additional keyword arguments passed to `plotly.express.scatter()`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.
    """
    px = attempt_import("plotly.express")

    slots = assign_slots(
        list(darray.dims),
        "scatter",
        x=x,
        y=y,
        color=color,
        size=size,
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
    if "y" in slots and slots["y"] not in labels:
        labels[str(slots["y"])] = get_axis_label(darray, slots["y"])

    fig = px.scatter(
        df,
        x=slots.get("x"),
        y=slots.get("y", value_col),
        color=slots.get("color"),
        size=slots.get("size"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        labels=labels,
        **px_kwargs,
    )

    return fig


def box(
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
    Create an interactive box plot from a DataArray using Plotly Express.

    Parameters
    ----------
    darray : DataArray
        The xarray DataArray to plot.
    x : str, auto, or None
        Dimension for the x-axis categories. Use `auto` for positional assignment,
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
        Additional keyword arguments passed to `plotly.express.box()`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.
    """
    px = attempt_import("plotly.express")

    slots = assign_slots(
        list(darray.dims),
        "box",
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

    fig = px.box(
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


def imshow(
    darray: DataArray,
    *,
    x: SlotValue = auto,
    y: SlotValue = auto,
    facet_col: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive heatmap/image from a DataArray using Plotly Express.

    The x and y parameters control which dimensions appear on each axis by
    transposing the DataArray appropriately. Plotly Express handles coordinate
    labels automatically from the xarray metadata.

    Parameters
    ----------
    darray : DataArray
        The xarray DataArray to plot (2D, or higher with faceting/animation).
    x : str, auto, or None
        Dimension for the x-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    y : str, auto, or None
        Dimension for the y-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    facet_col : str, auto, or None
        Dimension for subplot columns. Use `auto` for positional assignment,
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
    ...     dims=["lat", "lon"],
    ...     coords={"lat": np.arange(10), "lon": np.arange(20)},
    ...     name="temperature",
    ... )
    >>> fig = da.plotly.imshow()  # lat→x, lon→y (default order)
    >>> fig = da.plotly.imshow(x="lon", y="lat")  # lon on x-axis, lat on y-axis
    """
    px = attempt_import("plotly.express")

    slots = assign_slots(
        list(darray.dims),
        "imshow",
        x=x,
        y=y,
        facet_col=facet_col,
        animation_frame=animation_frame,
    )

    x_dim = slots.get("x")
    y_dim = slots.get("y")
    facet_col_dim = slots.get("facet_col")
    animation_dim = slots.get("animation_frame")

    # Build the transpose order: y first (rows), x second (columns),
    # then facet_col, animation_frame
    transpose_order = []
    if y_dim:
        transpose_order.append(y_dim)
    if x_dim:
        transpose_order.append(x_dim)
    if facet_col_dim:
        transpose_order.append(facet_col_dim)
    if animation_dim:
        transpose_order.append(animation_dim)

    if transpose_order:
        plot_data = darray.transpose(*transpose_order)
    else:
        plot_data = darray

    # px.imshow accepts xarray DataArray directly and uses coordinates
    fig = px.imshow(
        plot_data,
        facet_col=facet_col_dim,
        animation_frame=animation_dim,
        **px_kwargs,
    )

    return fig
