from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from typing import TYPE_CHECKING, Any

from xarray.core.datatree_mapping import map_over_datasets
from xarray.core.types import CoarsenBoundaryOptions, SideOptions
from xarray.core.utils import either_dict_or_kwargs

if TYPE_CHECKING:
    from xarray.core.dataset import Dataset
    from xarray.core.datatree import DataTree


class DataTreeCoarsen:
    """
    Coarsen object for hierarchical DataTree structures.

    Created by calling :meth:`DataTree.coarsen`. Coarsening reductions
    are applied to all nodes in the tree that contain the specified
    dimension(s). Nodes lacking the specified dimension(s) remain untouched.

    Parameters
    ----------
    datatree : DataTree
        The DataTree object to coarsen.
    dim : mapping of hashable to int, optional
        A mapping from the dimension name to integer block size.
    boundary : {"exact", "trim", "pad"}, default: "exact"
        How to handle the boundary when dimension length is not evenly divisible by block size.
    side : {"left", "right"} or mapping of hashable to {"left", "right"}, default: "left"
        Which side to pad or trim.
    coord_func : str or callable or mapping, default: "mean"
        Function to apply to the coordinates.
    **dim_kwargs : int
        The keyword arguments form of ``dim``.

    See Also
    --------
    xarray.DataTree.coarsen
    xarray.core.rolling.DatasetCoarsen
    """

    datatree: DataTree
    dim: dict[Hashable, int]
    boundary: CoarsenBoundaryOptions
    side: SideOptions | Mapping[Any, SideOptions]
    coord_func: str | Callable[..., Any] | Mapping[Any, str | Callable[..., Any]]

    def __init__(
        self,
        datatree: DataTree,
        dim: Mapping[Any, int] | None = None,
        boundary: CoarsenBoundaryOptions = "exact",
        side: SideOptions | Mapping[Any, SideOptions] = "left",
        coord_func: (
            str | Callable[..., Any] | Mapping[Any, str | Callable[..., Any]]
        ) = "mean",
        **dim_kwargs: int,
    ) -> None:
        self.datatree = datatree
        self.dim = either_dict_or_kwargs(dim, dim_kwargs, "DataTree.coarsen")
        self.boundary = boundary
        self.side = side
        self.coord_func = coord_func

    def _apply_reduction(
        self,
        method_name: str,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> DataTree:
        """Apply a named coarsening reduction across all eligible nodes."""

        def _node_coarsen(ds: Dataset) -> Dataset:
            if len(ds) == 0:
                return ds.copy()
            matching_dims = {d: w for d, w in self.dim.items() if d in ds.dims}
            if matching_dims:
                c = ds.coarsen(
                    matching_dims,
                    boundary=self.boundary,
                    side=self.side,
                    coord_func=self.coord_func,
                )
                method = getattr(c, method_name)
                return method(keep_attrs=keep_attrs, **kwargs)
            return ds.copy()

        return map_over_datasets(_node_coarsen, self.datatree)

    def reduce(
        self,
        func: Callable[..., Any],
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> DataTree:
        """
        Reduce the coarsened blocks by applying ``func`` across all eligible nodes.

        Parameters
        ----------
        func : callable
            Function to apply for block reduction.
        keep_attrs : bool, optional
            Whether to preserve attributes.
        **kwargs : dict
            Additional arguments passed to ``func``.

        Returns
        -------
        reduced : DataTree
            New DataTree with ``func`` applied to coarsened blocks.
        """

        def _node_reduce(ds: Dataset) -> Dataset:
            if len(ds) == 0:
                return ds.copy()
            matching_dims = {d: w for d, w in self.dim.items() if d in ds.dims}
            if matching_dims:
                return ds.coarsen(
                    matching_dims,
                    boundary=self.boundary,
                    side=self.side,
                    coord_func=self.coord_func,
                ).reduce(func, keep_attrs=keep_attrs, **kwargs)
            return ds.copy()

        return map_over_datasets(_node_reduce, self.datatree)

    def mean(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Coarsened mean across all eligible nodes."""
        return self._apply_reduction("mean", keep_attrs=keep_attrs, **kwargs)

    def sum(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Coarsened sum across all eligible nodes."""
        return self._apply_reduction("sum", keep_attrs=keep_attrs, **kwargs)

    def std(
        self, ddof: int = 0, keep_attrs: bool | None = None, **kwargs: Any
    ) -> DataTree:
        """Coarsened standard deviation across all eligible nodes."""
        return self._apply_reduction("std", keep_attrs=keep_attrs, ddof=ddof, **kwargs)

    def var(
        self, ddof: int = 0, keep_attrs: bool | None = None, **kwargs: Any
    ) -> DataTree:
        """Coarsened variance across all eligible nodes."""
        return self._apply_reduction("var", keep_attrs=keep_attrs, ddof=ddof, **kwargs)

    def min(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Coarsened minimum across all eligible nodes."""
        return self._apply_reduction("min", keep_attrs=keep_attrs, **kwargs)

    def max(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Coarsened maximum across all eligible nodes."""
        return self._apply_reduction("max", keep_attrs=keep_attrs, **kwargs)

    def median(self, keep_attrs: bool | None = None, **kwargs: Any) -> DataTree:
        """Coarsened median across all eligible nodes."""
        return self._apply_reduction("median", keep_attrs=keep_attrs, **kwargs)

    def construct(
        self,
        window_dim: Hashable | Mapping[Any, Hashable] | None = None,
        keep_attrs: bool | None = None,
        **window_dim_kwargs: Hashable,
    ) -> DataTree:
        """
        Convert this coarsen object to a DataTree with block dimensions.
        """

        def _node_construct(ds: Dataset) -> Dataset:
            if len(ds) == 0:
                return ds.copy()
            matching_dims = {d: w for d, w in self.dim.items() if d in ds.dims}
            if matching_dims:
                c = ds.coarsen(
                    matching_dims,
                    boundary=self.boundary,
                    side=self.side,
                    coord_func=self.coord_func,
                )
                return c.construct(
                    window_dim=window_dim,
                    keep_attrs=keep_attrs,
                    **window_dim_kwargs,
                )
            return ds.copy()

        return map_over_datasets(_node_construct, self.datatree)

    def __repr__(self) -> str:
        dim_summary = ", ".join(f"{d}: {w}" for d, w in self.dim.items())
        return (
            f"DataTreeCoarsen [dim: {{{dim_summary}}}, boundary: {self.boundary}, "
            f"side: {self.side}, coord_func: {self.coord_func}]"
        )
