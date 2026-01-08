"""
Plotly Express plotting functions for DataArray objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xarray.core.utils import attempt_import
from xarray.plot.plotly.common import (
    SlotValue,
    _build_labels,
    _get_value_col,
    _to_dataframe,
    assign_slots,
    auto,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from xarray.core.dataarray import DataArray


# =============================================================================
# Standard plots (y-axis = values)
# =============================================================================


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
    Create an interactive line plot from a DataArray.

    The y-axis shows DataArray values. Dimensions fill slots in order:
    x → color → line_dash → symbol → facet_col → facet_row → animation_frame
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

    df = _to_dataframe(darray)
    value_col = _get_value_col(darray)
    labels = {**_build_labels(darray, slots, value_col), **px_kwargs.pop("labels", {})}

    return px.line(
        df,
        x=slots.get("x"),
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
    Create an interactive bar chart from a DataArray.

    The y-axis shows DataArray values. Dimensions fill slots in order:
    x → color → pattern_shape → facet_col → facet_row → animation_frame
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

    df = _to_dataframe(darray)
    value_col = _get_value_col(darray)
    labels = {**_build_labels(darray, slots, value_col), **px_kwargs.pop("labels", {})}

    return px.bar(
        df,
        x=slots.get("x"),
        y=value_col,
        color=slots.get("color"),
        pattern_shape=slots.get("pattern_shape"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        labels=labels,
        **px_kwargs,
    )


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
    Create an interactive stacked area chart from a DataArray.

    The y-axis shows DataArray values. Dimensions fill slots in order:
    x → color → pattern_shape → facet_col → facet_row → animation_frame
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

    df = _to_dataframe(darray)
    value_col = _get_value_col(darray)
    labels = {**_build_labels(darray, slots, value_col), **px_kwargs.pop("labels", {})}

    return px.area(
        df,
        x=slots.get("x"),
        y=value_col,
        color=slots.get("color"),
        pattern_shape=slots.get("pattern_shape"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        labels=labels,
        **px_kwargs,
    )


def box(
    darray: DataArray,
    *,
    x: SlotValue = auto,
    color: SlotValue = None,
    facet_col: SlotValue = None,
    facet_row: SlotValue = None,
    animation_frame: SlotValue = None,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive box plot from a DataArray.

    The y-axis shows DataArray values. By default, only x is auto-assigned;
    other dimensions are aggregated into the box statistics.

    Dimensions fill slots in order: x → color → facet_col → facet_row → animation_frame
    """
    px = attempt_import("plotly.express")

    slots = assign_slots(
        list(darray.dims),
        "box",
        allow_unassigned=True,
        x=x,
        color=color,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = _to_dataframe(darray)
    value_col = _get_value_col(darray)
    labels = {**_build_labels(darray, slots, value_col), **px_kwargs.pop("labels", {})}

    return px.box(
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


# =============================================================================
# Scatter (y can be value or dimension)
# =============================================================================


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
    Create an interactive scatter plot from a DataArray.

    By default, y-axis shows DataArray values. Set y to a dimension name
    for dimension-vs-dimension plots (e.g., lat vs lon colored by value).

    Dimensions fill slots in order:
    x → color → size → symbol → facet_col → facet_row → animation_frame
    """
    px = attempt_import("plotly.express")

    # If y is a dimension, exclude it from slot assignment
    y_is_dim = y != "value" and y in darray.dims
    dims_for_slots = (
        [d for d in darray.dims if d != y] if y_is_dim else list(darray.dims)
    )

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

    df = _to_dataframe(darray)
    value_col = _get_value_col(darray)

    # Resolve y and color columns (may be "value" -> actual column name)
    y_col = value_col if y == "value" else y
    color_col = value_col if slots.get("color") == "value" else slots.get("color")

    # Build labels
    labels = {**_build_labels(darray, slots, value_col), **px_kwargs.pop("labels", {})}
    if y_is_dim and str(y) not in labels:
        from xarray.plot.plotly.common import _get_label

        labels[str(y)] = _get_label(darray, y)

    return px.scatter(
        df,
        x=slots.get("x"),
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


# =============================================================================
# Imshow (y and x are both dimensions)
# =============================================================================


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
    Create an interactive heatmap from a DataArray.

    Both x and y are dimensions. Dimensions fill slots in order:
    y (rows) → x (columns) → facet_col → animation_frame
    """
    px = attempt_import("plotly.express")

    slots = assign_slots(
        list(darray.dims),
        "imshow",
        y=y,
        x=x,
        facet_col=facet_col,
        animation_frame=animation_frame,
    )

    # Transpose to: y (rows), x (cols), facet_col, animation_frame
    transpose_order = [
        slots[k]
        for k in ("y", "x", "facet_col", "animation_frame")
        if slots.get(k) is not None
    ]
    plot_data = darray.transpose(*transpose_order) if transpose_order else darray

    return px.imshow(
        plot_data,
        facet_col=slots.get("facet_col"),
        animation_frame=slots.get("animation_frame"),
        **px_kwargs,
    )
