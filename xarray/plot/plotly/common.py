"""
Common utilities for Plotly Express plotting.

This module provides the dimension-to-slot assignment algorithm and
shared utilities for converting xarray data to Plotly-compatible formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import pandas as pd

    from xarray.core.dataarray import DataArray


class _AUTO:
    """Sentinel value for automatic slot assignment."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "auto"


auto = _AUTO()

# Type alias for slot values: auto, explicit dimension name, or None (skip)
SlotValue = _AUTO | str | None

# Slot orders per plot type (consistent with Plotly Express naming)
# Note: For most plots, y-axis shows DataArray values (not a dimension slot).
# For imshow, both y and x are dimensions (rows and columns of the heatmap).
SLOT_ORDERS: dict[str, tuple[str, ...]] = {
    "line": (
        "x",
        "color",
        "line_dash",
        "symbol",
        "facet_col",
        "facet_row",
        "animation_frame",
    ),
    "bar": ("x", "color", "pattern_shape", "facet_col", "facet_row", "animation_frame"),
    "area": (
        "x",
        "color",
        "pattern_shape",
        "facet_col",
        "facet_row",
        "animation_frame",
    ),
    "scatter": (
        "x",
        "color",
        "size",
        "symbol",
        "facet_col",
        "facet_row",
        "animation_frame",
    ),
    "imshow": ("y", "x", "facet_col", "animation_frame"),
    "box": ("x", "color", "facet_col", "facet_row", "animation_frame"),
}


def assign_slots(
    dims: Sequence[Hashable],
    plot_type: str,
    *,
    allow_unassigned: bool = False,
    **slot_kwargs: SlotValue,
) -> dict[str, Hashable]:
    """
    Assign dimensions to plot slots based on position.

    Positional assignment: dimensions fill slots in order.
    - Explicit assignments lock a dimension to a slot
    - None skips a slot
    - Remaining dims fill remaining slots by position
    - Error if dims left over after all slots filled (unless allow_unassigned=True)

    Parameters
    ----------
    dims : Sequence of Hashable
        Dimension names.
    plot_type : str
        Type of plot (line, bar, area, scatter, box, imshow).
    allow_unassigned : bool, optional
        If True, allow dimensions to remain unassigned. Default False.
    **slot_kwargs : auto, str, or None
        Explicit slot assignments. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip the slot.

    Returns
    -------
    dict
        Mapping of slot names to dimension names.
    """
    if plot_type not in SLOT_ORDERS:
        raise ValueError(
            f"Unknown plot type: {plot_type!r}. "
            f"Available types: {list(SLOT_ORDERS.keys())}"
        )

    slot_order = SLOT_ORDERS[plot_type]
    dims_list = list(dims)

    slots: dict[str, Hashable] = {}
    used_dims: set[Hashable] = set()
    available_slots = list(slot_order)

    # Pass 1: Process explicit assignments (non-auto, non-None)
    for slot in slot_order:
        value = slot_kwargs.get(slot, auto)

        if value is None:
            # Skip this slot
            if slot in available_slots:
                available_slots.remove(slot)
        elif not isinstance(value, _AUTO):
            # Explicit assignment - can be a dimension name or "value" (DataArray values)
            if value == "value":
                slots[slot] = "value"
            elif value not in dims_list:
                raise ValueError(
                    f"Dimension {value!r} assigned to slot {slot!r} "
                    f"is not in the data dimensions: {dims_list}"
                )
            else:
                slots[slot] = value
                used_dims.add(value)
            if slot in available_slots:
                available_slots.remove(slot)

    # Pass 2: Fill remaining slots with remaining dims (by position)
    remaining_dims = [d for d in dims_list if d not in used_dims]
    for slot, dim in zip(available_slots, remaining_dims, strict=False):
        slots[slot] = dim
        used_dims.add(dim)

    # Check for unassigned dimensions
    unassigned = [d for d in dims_list if d not in used_dims]
    if unassigned and not allow_unassigned:
        raise ValueError(
            f"Unassigned dimension(s): {unassigned}. "
            "Reduce with .sel(), .isel(), or .mean() before plotting."
        )

    return slots


def _get_value_col(darray: DataArray) -> str:
    """Get the column name for DataArray values."""
    return str(darray.name) if darray.name is not None else "value"


def _to_dataframe(darray: DataArray) -> pd.DataFrame:
    """Convert a DataArray to a long-form DataFrame for Plotly Express."""
    if darray.name is None:
        darray = darray.rename("value")
    return darray.to_dataframe().reset_index()


def _get_label(darray: DataArray, name: Hashable) -> str:
    """
    Get a human-readable label for a dimension or the value column.

    Uses long_name/standard_name and units from attributes if available.
    """
    # Check if it's asking for the value column label
    value_col = _get_value_col(darray)
    if str(name) == value_col or name == "value":
        label = darray.attrs.get("long_name") or darray.attrs.get("standard_name")
        if label:
            units = darray.attrs.get("units")
            return f"{label} [{units}]" if units else str(label)
        return value_col

    # It's a dimension/coordinate
    if name in darray.coords:
        coord = darray.coords[name]
        label = coord.attrs.get("long_name") or coord.attrs.get("standard_name")
        if label:
            units = coord.attrs.get("units")
            return f"{label} [{units}]" if units else str(label)
    return str(name)


def _build_labels(
    darray: DataArray,
    slots: dict[str, Hashable],
    value_col: str,
    include_value: bool = True,
) -> dict[str, str]:
    """
    Build a labels dict for Plotly Express from slot assignments.

    Parameters
    ----------
    darray : DataArray
        The source DataArray.
    slots : dict
        Slot assignments from assign_slots().
    value_col : str
        The name of the value column in the DataFrame.
    include_value : bool
        Whether to include a label for the value column.

    Returns
    -------
    dict
        Mapping of column names to human-readable labels.
    """
    labels: dict[str, str] = {}

    # Add labels for assigned dimensions
    for slot_value in slots.values():
        if slot_value and slot_value != "value":
            key = str(slot_value)
            if key not in labels:
                labels[key] = _get_label(darray, slot_value)

    # Add label for value column
    if include_value and value_col not in labels:
        labels[value_col] = _get_label(darray, "value")

    return labels
