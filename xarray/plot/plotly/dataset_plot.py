"""
Plotly Express plotting functions for Dataset objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xarray.core.utils import attempt_import
from xarray.plot.plotly.common import (
    SlotValue,
    assign_slots,
    auto,
    dataset_to_dataframe,
    get_dims_with_variable,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from xarray.core.dataset import Dataset


def line(
    dataset: Dataset,
    *,
    x: SlotValue = auto,
    color: SlotValue = auto,
    facet_col: SlotValue = auto,
    facet_row: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive line plot from a Dataset using Plotly Express.

    For Datasets with multiple data variables, 'variable' is treated as a
    pseudo-dimension that can be assigned to color, facet_col, etc.

    Parameters
    ----------
    dataset : Dataset
        The xarray Dataset to plot.
    x : str, auto, or None
        Dimension for the x-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    color : str, auto, or None
        Dimension for color grouping. Can be 'variable' to color by data
        variable. Use `auto` for positional assignment, a dimension name
        for explicit assignment, or `None` to skip.
    facet_col : str, auto, or None
        Dimension for subplot columns. Can be 'variable'. Use `auto` for
        positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    facet_row : str, auto, or None
        Dimension for subplot rows. Can be 'variable'. Use `auto` for
        positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    animation_frame : str, auto, or None
        Dimension for animation frames. Can be 'variable'. Use `auto` for
        positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    **px_kwargs
        Additional keyword arguments passed to `plotly.express.line()`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.

    Examples
    --------
    >>> ds = xr.Dataset({
    ...     "temperature": (["time", "city"], np.random.rand(10, 3)),
    ...     "humidity": (["time", "city"], np.random.rand(10, 3)),
    ... })
    >>> fig = ds.pxplot.line()  # time→x, city→color, variable→facet_col
    >>> fig = ds.pxplot.line(color="variable")  # time→x, variable→color, city→facet_col
    """
    px = attempt_import("plotly.express")

    dims = get_dims_with_variable(dataset)

    slots = assign_slots(
        dims,
        "line",
        x=x,
        color=color,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    # Check if 'variable' is used in any slot
    variable_in_use = "variable" in slots.values()

    df = dataset_to_dataframe(
        dataset,
        variable_slot="variable" if variable_in_use else None,
    )

    # Determine the y column name
    y_col: str
    if variable_in_use:
        y_col = "value"
    else:
        # If single variable, use its name
        var_names = list(dataset.data_vars)
        if len(var_names) == 1:
            y_col = str(var_names[0])
        else:
            # Multiple variables but not using 'variable' dimension
            # This shouldn't happen if dims includes 'variable', but handle gracefully
            y_col = str(var_names[0])

    fig = px.line(
        df,
        x=slots.get("x"),
        y=y_col,
        color=slots.get("color"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        **px_kwargs,
    )

    return fig


def bar(
    dataset: Dataset,
    *,
    x: SlotValue = auto,
    color: SlotValue = auto,
    facet_col: SlotValue = auto,
    facet_row: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive bar chart from a Dataset using Plotly Express.

    For Datasets with multiple data variables, 'variable' is treated as a
    pseudo-dimension that can be assigned to color, facet_col, etc.

    Parameters
    ----------
    dataset : Dataset
        The xarray Dataset to plot.
    x : str, auto, or None
        Dimension for the x-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    color : str, auto, or None
        Dimension for color grouping. Can be 'variable' to color by data
        variable. Use `auto` for positional assignment, a dimension name
        for explicit assignment, or `None` to skip.
    facet_col : str, auto, or None
        Dimension for subplot columns. Can be 'variable'. Use `auto` for
        positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    facet_row : str, auto, or None
        Dimension for subplot rows. Can be 'variable'. Use `auto` for
        positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    animation_frame : str, auto, or None
        Dimension for animation frames. Can be 'variable'. Use `auto` for
        positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    **px_kwargs
        Additional keyword arguments passed to `plotly.express.bar()`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.

    Examples
    --------
    >>> ds = xr.Dataset({
    ...     "sales": (["region", "product"], np.random.rand(3, 4)),
    ...     "profit": (["region", "product"], np.random.rand(3, 4)),
    ... })
    >>> fig = ds.pxplot.bar()  # region→x, product→color, variable→facet_col
    """
    px = attempt_import("plotly.express")

    dims = get_dims_with_variable(dataset)

    slots = assign_slots(
        dims,
        "bar",
        x=x,
        color=color,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    variable_in_use = "variable" in slots.values()

    df = dataset_to_dataframe(
        dataset,
        variable_slot="variable" if variable_in_use else None,
    )

    y_col: str
    if variable_in_use:
        y_col = "value"
    else:
        var_names = list(dataset.data_vars)
        y_col = str(var_names[0]) if var_names else "value"

    fig = px.bar(
        df,
        x=slots.get("x"),
        y=y_col,
        color=slots.get("color"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        **px_kwargs,
    )

    return fig


def area(
    dataset: Dataset,
    *,
    x: SlotValue = auto,
    color: SlotValue = auto,
    facet_col: SlotValue = auto,
    facet_row: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive stacked area chart from a Dataset using Plotly Express.

    For Datasets with multiple data variables, 'variable' is treated as a
    pseudo-dimension that can be assigned to color, facet_col, etc.

    Parameters
    ----------
    dataset : Dataset
        The xarray Dataset to plot.
    x : str, auto, or None
        Dimension for the x-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    color : str, auto, or None
        Dimension for color/stacking grouping. Can be 'variable'. Use `auto`
        for positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    facet_col : str, auto, or None
        Dimension for subplot columns. Can be 'variable'. Use `auto` for
        positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    facet_row : str, auto, or None
        Dimension for subplot rows. Can be 'variable'. Use `auto` for
        positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    animation_frame : str, auto, or None
        Dimension for animation frames. Can be 'variable'. Use `auto` for
        positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    **px_kwargs
        Additional keyword arguments passed to `plotly.express.area()`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.

    Examples
    --------
    >>> ds = xr.Dataset({
    ...     "sales": (["time", "product"], np.random.rand(10, 3)),
    ...     "returns": (["time", "product"], np.random.rand(10, 3)),
    ... })
    >>> fig = ds.pxplot.area()  # time→x, product→color, variable→facet_col
    """
    px = attempt_import("plotly.express")

    dims = get_dims_with_variable(dataset)

    slots = assign_slots(
        dims,
        "area",
        x=x,
        color=color,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    variable_in_use = "variable" in slots.values()

    df = dataset_to_dataframe(
        dataset,
        variable_slot="variable" if variable_in_use else None,
    )

    y_col: str
    if variable_in_use:
        y_col = "value"
    else:
        var_names = list(dataset.data_vars)
        y_col = str(var_names[0]) if var_names else "value"

    fig = px.area(
        df,
        x=slots.get("x"),
        y=y_col,
        color=slots.get("color"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        **px_kwargs,
    )

    return fig


def heatmap(
    dataset: Dataset,
    *,
    x: SlotValue = auto,
    y: SlotValue = auto,
    facet_col: SlotValue = auto,
    facet_row: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive heatmap from a Dataset using Plotly Express.

    For Datasets with multiple data variables, 'variable' is treated as a
    pseudo-dimension that can be assigned to facet_col, facet_row, or
    animation_frame.

    Parameters
    ----------
    dataset : Dataset
        The xarray Dataset to plot.
    x : str, auto, or None
        Dimension for the x-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    y : str, auto, or None
        Dimension for the y-axis. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip.
    facet_col : str, auto, or None
        Dimension for subplot columns. Can be 'variable'. Use `auto` for
        positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    facet_row : str, auto, or None
        Dimension for subplot rows. Can be 'variable'. Use `auto` for
        positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    animation_frame : str, auto, or None
        Dimension for animation frames. Can be 'variable'. Use `auto` for
        positional assignment, a dimension name for explicit assignment,
        or `None` to skip.
    **px_kwargs
        Additional keyword arguments passed to `plotly.express.imshow()` or
        `plotly.express.density_heatmap()`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.

    Examples
    --------
    >>> ds = xr.Dataset({
    ...     "temperature": (["lat", "lon"], np.random.rand(10, 20)),
    ...     "pressure": (["lat", "lon"], np.random.rand(10, 20)),
    ... })
    >>> fig = ds.pxplot.heatmap()  # lat→y, lon→x, variable→facet_col
    """
    px = attempt_import("plotly.express")

    dims = get_dims_with_variable(dataset)

    slots = assign_slots(
        dims,
        "heatmap",
        x=x,
        y=y,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    variable_in_use = "variable" in slots.values()

    df = dataset_to_dataframe(
        dataset,
        variable_slot="variable" if variable_in_use else None,
    )

    x_dim = slots.get("x")
    y_dim = slots.get("y")
    facet_col_dim = slots.get("facet_col")
    facet_row_dim = slots.get("facet_row")
    animation_dim = slots.get("animation_frame")

    z_col: str
    if variable_in_use:
        z_col = "value"
    else:
        var_names = list(dataset.data_vars)
        z_col = str(var_names[0]) if var_names else "value"

    # Use density_heatmap for faceted or animated heatmaps
    fig = px.density_heatmap(
        df,
        x=x_dim,
        y=y_dim,
        z=z_col,
        facet_col=facet_col_dim,
        facet_row=facet_row_dim,
        animation_frame=animation_dim,
        histfunc="avg",
        **px_kwargs,
    )

    return fig
