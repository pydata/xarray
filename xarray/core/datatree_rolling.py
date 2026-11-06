from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from typing import TYPE_CHECKING, Any

from xarray.core import dtypes
from xarray.core.datatree_mapping import map_over_datasets
from xarray.core.utils import either_dict_or_kwargs

if TYPE_CHECKING:
    from xarray.core.dataset import Dataset
    from xarray.core.datatree import DataTree


class DataTreeRolling:
    """
    Rolling window object for hierarchical DataTree structures.

    Created by calling :meth:`DataTree.rolling`. Rolling window operations
    are applied to all nodes in the tree that contain the specified
    dimension(s). Nodes lacking the specified dimension(s) remain untouched.

    Parameters
    ----------
    datatree : DataTree
        The DataTree object to roll over.
    dim : mapping of hashable to int, optional
        A mapping from the dimension name to integer window size.
    min_periods : int, default: None
        Minimum number of observations in window required to have a value
        (otherwise result is NA). The default, None, defaults to window size.
    center : bool or mapping of hashable to bool, default: False
        Set the labels at the center of the window.
    **dim_kwargs : int
        The keyword arguments form of ``dim``.

    See Also
    --------
    xarray.DataTree.rolling
    xarray.core.rolling.DatasetRolling
    """

    datatree: DataTree
    dim: dict[Hashable, int]
    min_periods: int | None
    center: bool | Mapping[Any, bool]

    def __init__(
        self,
        datatree: DataTree,
        dim: Mapping[Any, int] | None = None,
        min_periods: int | None = None,
        center: bool | Mapping[Any, bool] = False,
        **dim_kwargs: int,
    ) -> None:
        self.datatree = datatree
        self.dim = either_dict_or_kwargs(dim, dim_kwargs, "DataTree.rolling")
        self.min_periods = min_periods
        self.center = center

    def _apply_reduction(
        self,
        method_name: str,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> DataTree:
        """Apply a named reduction method across all nodes containing rolling dimensions."""

        def _node_reduction(ds: Dataset) -> Dataset:
            if len(ds) == 0:
                return ds.copy()
            matching_dims = {d: w for d, w in self.dim.items() if d in ds.dims}
            if matching_dims:
                r = ds.rolling(
                    matching_dims,
                    min_periods=self.min_periods,
                    center=self.center,
                )
                method = getattr(r, method_name)
                return method(keep_attrs=keep_attrs, **kwargs)
            return ds.copy()

        return map_over_datasets(_node_reduction, self.datatree)

    def reduce(
        self,
        func: Callable[..., Any],
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> DataTree:
        """
        Reduce the rolling windows by applying ``func`` across all eligible nodes.

        Parameters
        ----------
        func : callable
            Function to apply for reduction across the rolling windows.
        keep_attrs : bool, optional
            Whether to copy attributes to the new object.
        **kwargs : dict
            Additional arguments passed to ``func``.

        Returns
        -------
        reduced : DataTree
            New DataTree with ``func`` applied to rolling windows on all eligible nodes.
        """

        def _node_reduce(ds: Dataset) -> Dataset:
            if len(ds) == 0:
                return ds.copy()
            matching_dims = {d: w for d, w in self.dim.items() if d in ds.dims}
            if matching_dims:
                return ds.rolling(
                    matching_dims,
                    min_periods=self.min_periods,
                    center=self.center,
                ).reduce(func, keep_attrs=keep_attrs, **kwargs)
            return ds.copy()

        return map_over_datasets(_node_reduce, self.datatree)

    def mean(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Rolling mean across all nodes."""
        return self._apply_reduction("mean", keep_attrs=keep_attrs, **kwargs)

    def sum(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Rolling sum across all nodes."""
        return self._apply_reduction("sum", keep_attrs=keep_attrs, **kwargs)

    def std(
        self, ddof: int = 0, keep_attrs: bool | None = None, **kwargs: Any
    ) -> DataTree:
        """Rolling standard deviation across all nodes."""
        return self._apply_reduction("std", keep_attrs=keep_attrs, ddof=ddof, **kwargs)

    def var(
        self, ddof: int = 0, keep_attrs: bool | None = None, **kwargs: Any
    ) -> DataTree:
        """Rolling variance across all nodes."""
        return self._apply_reduction("var", keep_attrs=keep_attrs, ddof=ddof, **kwargs)

    def min(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Rolling minimum across all nodes."""
        return self._apply_reduction("min", keep_attrs=keep_attrs, **kwargs)

    def max(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Rolling maximum across all nodes."""
        return self._apply_reduction("max", keep_attrs=keep_attrs, **kwargs)

    def median(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Rolling median across all nodes."""
        return self._apply_reduction("median", keep_attrs=keep_attrs, **kwargs)

    def prod(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Rolling product across all nodes."""
        return self._apply_reduction("prod", keep_attrs=keep_attrs, **kwargs)

    def count(self, keep_attrs: bool | None = None) -> DataTree:
        """Rolling count of non-NaN observations across all nodes."""
        return self._apply_reduction("count", keep_attrs=keep_attrs)

    def argmax(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Indices of rolling maximum values across all nodes."""
        return self._apply_reduction("argmax", keep_attrs=keep_attrs, **kwargs)

    def argmin(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Indices of rolling minimum values across all nodes."""
        return self._apply_reduction("argmin", keep_attrs=keep_attrs, **kwargs)

    def construct(
        self,
        window_dim: Hashable | Mapping[Any, Hashable] | None = None,
        *,
        stride: int | Mapping[Any, int] = 1,
        fill_value: Any = dtypes.NA,
        keep_attrs: bool | None = None,
        **window_dim_kwargs: Hashable,
    ) -> DataTree:
        """
        Convert this rolling object to a DataTree, where the window dimension
        is stacked as a new dimension on all eligible nodes.

        Parameters
        ----------
        window_dim : str or mapping, optional
            A mapping from dimension name to the new window dimension names.
        stride : int or mapping, default: 1
            Size of stride for the rolling window.
        fill_value : Any, default: dtypes.NA
            Filling value to match the dimension size.
        keep_attrs : bool, optional
            Whether to preserve attributes.
        **window_dim_kwargs : Hashable
            The keyword arguments form of ``window_dim``.

        Returns
        -------
        DataTree
            New DataTree with constructed rolling views on each node.
        """

        def _node_construct(ds: Dataset) -> Dataset:
            if len(ds) == 0:
                return ds.copy()
            matching_dims = {d: w for d, w in self.dim.items() if d in ds.dims}
            if matching_dims:
                r = ds.rolling(
                    matching_dims,
                    min_periods=self.min_periods,
                    center=self.center,
                )
                return r.construct(
                    window_dim=window_dim,
                    stride=stride,
                    fill_value=fill_value,
                    keep_attrs=keep_attrs,
                    **window_dim_kwargs,
                )
            return ds.copy()

        return map_over_datasets(_node_construct, self.datatree)

    def __repr__(self) -> str:
        dim_summary = ", ".join(f"{d}: {w}" for d, w in self.dim.items())
        return f"DataTreeRolling [dim: {{{dim_summary}}}, min_periods: {self.min_periods}, center: {self.center}]"
