"""
Common utilities for Plotly Express plotting.

This module provides the dimension-to-slot assignment algorithm and
shared utilities for converting xarray data to Plotly-compatible formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

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
# Faceting and animation are always last
# Note: "y" slot is first and defaults to "value" (DataArray values), can also be a dimension
SLOT_ORDERS: dict[str, tuple[str, ...]] = {
    "line": ("y", "x", "color", "line_dash", "symbol", "facet_col", "facet_row", "animation_frame"),
    "bar": ("y", "x", "color", "pattern_shape", "facet_col", "facet_row", "animation_frame"),
    "area": ("y", "x", "color", "pattern_shape", "facet_col", "facet_row", "animation_frame"),
    "scatter": ("y", "x", "color", "size", "symbol", "facet_col", "facet_row", "animation_frame"),
    "imshow": ("y", "x", "facet_col", "animation_frame"),
    "box": ("y", "x", "color", "facet_col", "facet_row", "animation_frame"),
}

# Slots that default to "value" (DataArray values) instead of auto-assigning a dimension
# Note: imshow is excluded because y is a real dimension (rows) for heatmaps
VALUE_DEFAULT_SLOTS: dict[str, set[str]] = {
    "line": {"y"},
    "bar": {"y"},
    "area": {"y"},
    "scatter": {"y"},
    "box": {"y"},
    # imshow: y is a dimension, not "value"
}


def assign_slots(
    dims: Sequence[Hashable],
    plot_type: str,
    **slot_kwargs: SlotValue,
) -> dict[str, Hashable]:
    """
    Assign dimensions to plot slots based on position.

    Positional assignment: dimensions fill slots in order.
    - Explicit assignments lock a dimension to a slot
    - None skips a slot
    - Slots in VALUE_DEFAULT_SLOTS (like "y") default to "value" instead of a dimension
    - Remaining dims fill remaining slots by position
    - Error if dims left over after all slots filled

    Parameters
    ----------
    dims : Sequence of Hashable
        Dimension names.
    plot_type : str
        Type of plot (line, bar, area, scatter, box, imshow).
    **slot_kwargs : auto, str, or None
        Explicit slot assignments. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip the slot.
        For slots in VALUE_DEFAULT_SLOTS, `auto` assigns "value" (DataArray values).

    Returns
    -------
    dict
        Mapping of slot names to dimension names (or "value" for y-axis).

    Raises
    ------
    ValueError
        If there are unassigned dimensions after all slots are filled.
    ValueError
        If an explicitly assigned dimension is not in the data.

    Examples
    --------
    >>> assign_slots(["time", "city", "scenario"], "line")
    {'x': 'time', 'y': 'value', 'color': 'city', 'facet_col': 'scenario'}

    >>> assign_slots(["time", "city"], "line", y="city")
    {'x': 'time', 'y': 'city'}

    >>> assign_slots(["time", "city"], "line", color=None)
    {'x': 'time', 'y': 'value', 'facet_col': 'city'}

    >>> assign_slots(["time", "city"], "line", x="city", color="time")
    {'x': 'city', 'y': 'value', 'color': 'time'}
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

    # Get the value-default slots for this plot type (if any)
    value_default_slots = VALUE_DEFAULT_SLOTS.get(plot_type, set())

    # Pass 1: Process value-default slots (like "y") - these default to "value"
    for slot in slot_order:
        if slot in value_default_slots:
            value = slot_kwargs.get(slot, auto)
            if value is None:
                # Skip this slot
                available_slots.remove(slot)
            elif isinstance(value, _AUTO) or value == "value":
                # Auto or "value" -> assign "value" (DataArray values)
                slots[slot] = "value"
                available_slots.remove(slot)
            else:
                # Explicit dimension assignment
                if value not in dims_list:
                    raise ValueError(
                        f"Dimension {value!r} assigned to slot {slot!r} "
                        f"is not in the data dimensions: {dims_list}"
                    )
                slots[slot] = value
                used_dims.add(value)
                available_slots.remove(slot)

    # Pass 2: Process other explicit assignments (non-auto, non-None)
    for slot in slot_order:
        if slot in value_default_slots:
            continue  # Already handled
        value = slot_kwargs.get(slot, auto)

        if value is None:
            # Skip this slot
            if slot in available_slots:
                available_slots.remove(slot)
        elif not isinstance(value, _AUTO):
            # Explicit assignment - allow "value" as special assignment for any slot
            if value == "value":
                slots[slot] = "value"
                if slot in available_slots:
                    available_slots.remove(slot)
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

    # Pass 3: Fill remaining slots with remaining dims (by position)
    remaining_dims = [d for d in dims_list if d not in used_dims]
    for slot, dim in zip(available_slots, remaining_dims, strict=False):
        slots[slot] = dim
        used_dims.add(dim)

    # Check for unassigned dimensions
    unassigned = [d for d in dims_list if d not in used_dims]
    if unassigned:
        raise ValueError(
            f"Unassigned dimension(s): {unassigned}. "
            "Reduce with .sel(), .isel(), or .mean() before plotting."
        )

    return slots


def dataarray_to_dataframe(darray: DataArray) -> Any:
    """
    Convert a DataArray to a pandas DataFrame suitable for Plotly Express.

    Parameters
    ----------
    darray : DataArray
        The xarray DataArray to convert.

    Returns
    -------
    pandas.DataFrame
        Long-form DataFrame with dimension coordinates as columns.
    """
    # Create a copy with a name if it doesn't have one
    if darray.name is None:
        darray = darray.rename("value")

    df = darray.to_dataframe().reset_index()
    return df


def get_axis_label(darray: DataArray, dim: Hashable) -> str:
    """
    Get a label for an axis from DataArray attributes.

    Uses the coordinate's long_name or standard_name if available,
    falls back to the dimension name.

    Parameters
    ----------
    darray : DataArray
        The xarray DataArray.
    dim : Hashable
        The dimension name.

    Returns
    -------
    str
        A human-readable label for the axis.
    """
    if dim in darray.coords:
        coord = darray.coords[dim]
        # Try long_name, then standard_name, then dimension name
        label = coord.attrs.get("long_name") or coord.attrs.get("standard_name")
        if label:
            units = coord.attrs.get("units")
            if units:
                return f"{label} [{units}]"
            return str(label)
    return str(dim)


def get_value_label(darray: DataArray) -> str:
    """
    Get a label for the data values from DataArray attributes.

    Uses the DataArray's long_name or standard_name if available,
    falls back to the DataArray name or 'value'.

    Parameters
    ----------
    darray : DataArray
        The xarray DataArray.

    Returns
    -------
    str
        A human-readable label for the values.
    """
    label = darray.attrs.get("long_name") or darray.attrs.get("standard_name")
    if label:
        units = darray.attrs.get("units")
        if units:
            return f"{label} [{units}]"
        return str(label)
    return str(darray.name) if darray.name is not None else "value"
