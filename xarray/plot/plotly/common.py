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
    from xarray.core.dataset import Dataset


class _AUTO:
    """Sentinel value for automatic slot assignment."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "auto"


auto = _AUTO()

# Type alias for slot values: auto, explicit dimension name, or None (skip)
SlotValue = _AUTO | str | None

# Slot orders per plot type (consistent with Plotly Express naming)
SLOT_ORDERS: dict[str, tuple[str, ...]] = {
    "line": ("x", "color", "facet_col", "facet_row", "animation_frame"),
    "bar": ("x", "color", "facet_col", "facet_row", "animation_frame"),
    "area": ("x", "color", "facet_col", "facet_row", "animation_frame"),
    "heatmap": ("x", "y", "facet_col", "facet_row", "animation_frame"),
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
    - Remaining dims fill remaining slots by position
    - Error if dims left over after all slots filled

    Parameters
    ----------
    dims : Sequence of Hashable
        Dimension names (including 'variable' pseudo-dim for multi-var Datasets).
    plot_type : str
        Type of plot (line, bar, area, heatmap).
    **slot_kwargs : auto, str, or None
        Explicit slot assignments. Use `auto` for positional assignment,
        a dimension name for explicit assignment, or `None` to skip the slot.

    Returns
    -------
    dict
        Mapping of slot names to dimension names.

    Raises
    ------
    ValueError
        If there are unassigned dimensions after all slots are filled.
    ValueError
        If an explicitly assigned dimension is not in the data.

    Examples
    --------
    >>> assign_slots(["time", "city", "scenario"], "line")
    {'x': 'time', 'color': 'city', 'facet_col': 'scenario'}

    >>> assign_slots(["time", "city"], "line", color=None)
    {'x': 'time', 'facet_col': 'city'}

    >>> assign_slots(["time", "city"], "line", x="city", color="time")
    {'x': 'city', 'color': 'time'}
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
            # Explicit assignment
            if value not in dims_list:
                raise ValueError(
                    f"Dimension {value!r} assigned to slot {slot!r} "
                    f"is not in the data dimensions: {dims_list}"
                )
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
    if unassigned:
        raise ValueError(
            f"Unassigned dimension(s): {unassigned}. "
            "Reduce with .sel(), .isel(), or .mean() before plotting."
        )

    return slots


def get_dims_with_variable(dataset: Dataset) -> list[Hashable]:
    """
    Get dimension names, adding 'variable' pseudo-dimension if multiple data vars.

    For Datasets with multiple data variables, 'variable' is treated as a
    pseudo-dimension that can be assigned to color, facet_col, etc.

    Parameters
    ----------
    dataset : Dataset
        The xarray Dataset.

    Returns
    -------
    list of Hashable
        Dimension names, with 'variable' appended if there are multiple data vars.
    """
    dims: list[Hashable] = list(dataset.dims)
    if len(dataset.data_vars) > 1:
        dims.append("variable")
    return dims


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


def dataset_to_dataframe(
    dataset: Dataset,
    variable_slot: str | None = None,
) -> Any:
    """
    Convert a Dataset to a pandas DataFrame suitable for Plotly Express.

    If 'variable' is assigned to a slot, the DataFrame is melted to long form
    with a 'variable' column containing the data variable names.

    Parameters
    ----------
    dataset : Dataset
        The xarray Dataset to convert.
    variable_slot : str or None
        If 'variable' is assigned to a slot, this is the slot name.
        If None, each data variable becomes a separate column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with dimension coordinates as columns.
    """
    df = dataset.to_dataframe().reset_index()

    # If 'variable' is used as a dimension, melt the dataframe
    if variable_slot is not None:
        # Get the coordinate columns (dimensions)
        coord_cols = [col for col in df.columns if col in dataset.dims]
        # Get the data variable columns
        var_cols = [col for col in df.columns if col in dataset.data_vars]

        if var_cols:
            df = df.melt(
                id_vars=coord_cols,
                value_vars=var_cols,
                var_name="variable",
                value_name="value",
            )

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
