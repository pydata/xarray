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
    line_dash: SlotValue = auto,
    symbol: SlotValue = auto,
    facet_col: SlotValue = auto,
    facet_row: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive line plot from a DataArray using Plotly Express.

    The y-axis always shows the DataArray values. Dimensions are assigned
    to other slots: x → color → line_dash → symbol → facet_col → facet_row → animation_frame

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
    line_dash : str, auto, or None
        Dimension for line dash style. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    symbol : str, auto, or None
        Dimension for marker symbol. Use `auto` for positional assignment,
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
    >>> fig = da.plotly.line()  # time→x, city→color, values→y
    >>> fig = da.plotly.line(x="city", color="time")  # explicit assignment
    """
    px = attempt_import("plotly.express")

    slots = assign_slots(
        list(darray.dims),
        "line",
        x=x,
        color=color,
        line_dash=line_dash,
        symbol=symbol,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = dataarray_to_dataframe(darray)
    value_col = darray.name if darray.name is not None else "value"

    # Build labels for axes
    labels = px_kwargs.pop("labels", {})
    x_col = slots.get("x")
    if x_col and str(x_col) not in labels:
        labels[str(x_col)] = get_axis_label(darray, x_col)
    if value_col not in labels:
        labels[value_col] = get_value_label(darray)

    fig = px.line(
        df,
        x=x_col,
        y=value_col,
        color=slots.get("color"),
        line_dash=slots.get("line_dash"),
        symbol=slots.get("symbol"),
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
    pattern_shape: SlotValue = auto,
    facet_col: SlotValue = auto,
    facet_row: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive bar chart from a DataArray using Plotly Express.

    The y-axis always shows the DataArray values. Dimensions are assigned
    to other slots: x → color → pattern_shape → facet_col → facet_row → animation_frame

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
    pattern_shape : str, auto, or None
        Dimension for bar fill pattern. Use `auto` for positional assignment,
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
    >>> fig = da.plotly.bar()  # city→x, scenario→color
    """
    px = attempt_import("plotly.express")

    slots = assign_slots(
        list(darray.dims),
        "bar",
        x=x,
        color=color,
        pattern_shape=pattern_shape,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = dataarray_to_dataframe(darray)
    value_col = darray.name if darray.name is not None else "value"

    # Build labels for axes
    labels = px_kwargs.pop("labels", {})
    x_col = slots.get("x")
    if x_col and str(x_col) not in labels:
        labels[str(x_col)] = get_axis_label(darray, x_col)
    if value_col not in labels:
        labels[value_col] = get_value_label(darray)

    fig = px.bar(
        df,
        x=x_col,
        y=value_col,
        color=slots.get("color"),
        pattern_shape=slots.get("pattern_shape"),
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
    pattern_shape: SlotValue = auto,
    facet_col: SlotValue = auto,
    facet_row: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive stacked area chart from a DataArray using Plotly Express.

    The y-axis always shows the DataArray values. Dimensions are assigned
    to other slots: x → color → pattern_shape → facet_col → facet_row → animation_frame

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
    pattern_shape : str, auto, or None
        Dimension for fill pattern. Use `auto` for positional assignment,
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
    >>> fig = da.plotly.area()  # time→x, category→color (stacked)
    """
    px = attempt_import("plotly.express")

    slots = assign_slots(
        list(darray.dims),
        "area",
        x=x,
        color=color,
        pattern_shape=pattern_shape,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = dataarray_to_dataframe(darray)
    value_col = darray.name if darray.name is not None else "value"

    # Build labels for axes
    labels = px_kwargs.pop("labels", {})
    x_col = slots.get("x")
    if x_col and str(x_col) not in labels:
        labels[str(x_col)] = get_axis_label(darray, x_col)
    if value_col not in labels:
        labels[value_col] = get_value_label(darray)

    fig = px.area(
        df,
        x=x_col,
        y=value_col,
        color=slots.get("color"),
        pattern_shape=slots.get("pattern_shape"),
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
    Create an interactive scatter plot from a DataArray using Plotly Express.

    By default, y-axis shows the DataArray values. Set y to a dimension name
    to create dimension-vs-dimension plots (e.g., lat vs lon colored by value).

    Parameters
    ----------
    darray : DataArray
        The xarray DataArray to plot.
    x : str, auto, or None
        Dimension for the x-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    y : str
        What to plot on y-axis. Default "value" uses DataArray values.
        Can be a dimension name for dimension vs dimension plots.
    color : str, auto, None, or "value"
        Dimension for color grouping. Use `auto` for positional assignment,
        a dimension name for explicit assignment, `None` to skip,
        or "value" to color by DataArray values.
    size : str, auto, or None
        Dimension for marker size. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    symbol : str, auto, or None
        Dimension for marker symbol. Use `auto` for positional assignment,
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

    Examples
    --------
    >>> # Default: x-axis is dimension, y-axis is values
    >>> fig = da.plotly.scatter()

    >>> # Dimension vs dimension, colored by values
    >>> fig = da.plotly.scatter(x="lon", y="lat", color="value")
    """
    px = attempt_import("plotly.express")

    # Handle y parameter: if it's a dimension, exclude it from slot assignment
    y_is_dim = y != "value" and y in darray.dims
    dims_for_slots = [d for d in darray.dims if not (y_is_dim and d == y)]

    slots = assign_slots(
        dims_for_slots,
        "scatter",
        x=x,
        color=color,
        size=size,
        symbol=symbol,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = dataarray_to_dataframe(darray)
    value_col = darray.name if darray.name is not None else "value"

    # Convert "value" references to actual column name
    def slot_to_col(slot_value):
        return value_col if slot_value == "value" else slot_value

    y_col = value_col if y == "value" else y
    x_col = slots.get("x")
    color_col = slot_to_col(slots.get("color"))

    # Build labels for axes
    labels = px_kwargs.pop("labels", {})
    if x_col and str(x_col) not in labels:
        labels[str(x_col)] = get_axis_label(darray, x_col)
    if y == "value" and value_col not in labels:
        labels[value_col] = get_value_label(darray)
    elif y_is_dim and str(y) not in labels:
        labels[str(y)] = get_axis_label(darray, y)
    # Add label for value column if used as color
    if slots.get("color") == "value" and value_col not in labels:
        labels[value_col] = get_value_label(darray)

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=slots.get("size"),
        symbol=slots.get("symbol"),
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

    The y-axis always shows the DataArray values. Dimensions are assigned
    to other slots: x → color → facet_col → facet_row → animation_frame

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
    x_col = slots.get("x")
    if x_col and str(x_col) not in labels:
        labels[str(x_col)] = get_axis_label(darray, x_col)
    if value_col not in labels:
        labels[value_col] = get_value_label(darray)

    fig = px.box(
        df,
        x=x_col,
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
