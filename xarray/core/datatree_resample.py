from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from typing import TYPE_CHECKING, Any

from xarray.core.datatree_mapping import map_over_datasets
from xarray.core.types import Dims, InterpOptions, SideOptions
from xarray.core.utils import either_dict_or_kwargs

if TYPE_CHECKING:
    from xarray.core.dataset import Dataset
    from xarray.core.datatree import DataTree


class DataTreeResample:
    """
    Resample object for hierarchical DataTree structures.

    Created by calling :meth:`DataTree.resample`. Resample operations
    are applied to all nodes in the tree that contain the resampled
    dimension or coordinate. Nodes lacking the dimension remain untouched.

    Parameters
    ----------
    datatree : DataTree
        The DataTree object to resample.
    indexer : mapping of hashable to str, optional
        A mapping from the dimension name to frequency string.
    skipna : bool, optional
        Whether to skip missing values when aggregating.
    closed : {"left", "right"}, optional
        Which side of bin interval is closed.
    label : {"left", "right"}, optional
        Which bin edge label to label bucket with.
    base : int, default: 0
        For frequencies that evenly subdivide 1 day of elapsed time,
        the "origin" of the aggregated intervals.
    keep_attrs : bool, optional
        Whether to copy attributes from the original object to the new one.
    loffset : timedelta or str, optional
        Adjust the resampled time labels.
    restore_coord_dims : bool, optional
        Whether to restore original coordinate dimensions.
    **indexer_kwargs : str
        The keyword arguments form of ``indexer``.

    See Also
    --------
    xarray.DataTree.resample
    xarray.core.resample.DatasetResample
    """

    datatree: DataTree
    indexer: dict[Hashable, Any]
    resample_kwargs: dict[str, Any]

    def __init__(
        self,
        datatree: DataTree,
        indexer: Mapping[Any, Any] | None = None,
        skipna: bool | None = None,
        closed: SideOptions | None = None,
        label: SideOptions | None = None,
        base: int | None = None,
        keep_attrs: bool | None = None,
        loffset: Any | None = None,
        restore_coord_dims: bool | None = None,
        **indexer_kwargs: Any,
    ) -> None:
        self.datatree = datatree
        self.indexer = either_dict_or_kwargs(
            indexer, indexer_kwargs, "DataTree.resample"
        )
        self.resample_kwargs = {
            "skipna": skipna,
            "closed": closed,
            "label": label,
            "base": base,
            "keep_attrs": keep_attrs,
            "loffset": loffset,
            "restore_coord_dims": restore_coord_dims,
        }

    def _apply_op(
        self,
        op_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> DataTree:
        target_dims = set(self.indexer.keys())

        def _node_resample(ds: Dataset) -> Dataset:
            # Check if this node has any data variable depending on target_dims
            has_var_dim = any(
                d in var.dims for var in ds.data_vars.values() for d in target_dims
            )
            if not has_var_dim:
                to_drop = [d for d in target_dims if d in ds.coords]
                if to_drop:
                    return ds.drop_vars(to_drop)
                return ds.copy()

            # Apply resample on this node's dataset
            clean_kwargs = {
                k: v for k, v in self.resample_kwargs.items() if v is not None
            }
            resampled = ds.resample(self.indexer, **clean_kwargs)
            res_func = getattr(resampled, op_name)
            return res_func(*args, **kwargs)

        return map_over_datasets(_node_resample, self.datatree)

    def count(self, dim: Dims = None, keep_attrs: bool | None = None) -> DataTree:
        """Compute count along resampled dimension across all nodes."""
        return self._apply_op("count", dim=dim, keep_attrs=keep_attrs)

    def first(
        self,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> DataTree:
        """Return first value along resampled dimension across all nodes."""
        return self._apply_op("first", skipna=skipna, keep_attrs=keep_attrs)

    def last(
        self,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> DataTree:
        """Return last value along resampled dimension across all nodes."""
        return self._apply_op("last", skipna=skipna, keep_attrs=keep_attrs)

    def mean(
        self,
        dim: Dims = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> DataTree:
        """Compute mean along resampled dimension across all nodes."""
        return self._apply_op("mean", dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    def median(
        self,
        dim: Dims = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> DataTree:
        """Compute median along resampled dimension across all nodes."""
        return self._apply_op("median", dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    def min(
        self,
        dim: Dims = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> DataTree:
        """Compute min along resampled dimension across all nodes."""
        return self._apply_op("min", dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    def max(
        self,
        dim: Dims = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> DataTree:
        """Compute max along resampled dimension across all nodes."""
        return self._apply_op("max", dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    def std(
        self,
        dim: Dims = None,
        skipna: bool | None = None,
        ddof: int = 0,
        keep_attrs: bool | None = None,
    ) -> DataTree:
        """Compute standard deviation along resampled dimension across all nodes."""
        return self._apply_op(
            "std", dim=dim, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs
        )

    def var(
        self,
        dim: Dims = None,
        skipna: bool | None = None,
        ddof: int = 0,
        keep_attrs: bool | None = None,
    ) -> DataTree:
        """Compute variance along resampled dimension across all nodes."""
        return self._apply_op(
            "var", dim=dim, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs
        )

    def sum(
        self,
        dim: Dims = None,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
    ) -> DataTree:
        """Compute sum along resampled dimension across all nodes."""
        return self._apply_op(
            "sum", dim=dim, skipna=skipna, min_count=min_count, keep_attrs=keep_attrs
        )

    def prod(
        self,
        dim: Dims = None,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
    ) -> DataTree:
        """Compute product along resampled dimension across all nodes."""
        return self._apply_op(
            "prod", dim=dim, skipna=skipna, min_count=min_count, keep_attrs=keep_attrs
        )

    def nearest(self, tolerance: float | str | None = None) -> DataTree:
        """Take nearest value at new resampled coordinates across all nodes."""
        return self._apply_op("nearest", tolerance=tolerance)

    def asfreq(self) -> DataTree:
        """Return values at new resampled frequency across all nodes."""
        return self._apply_op("asfreq")

    def interpolate(self, kind: InterpOptions = "linear") -> DataTree:
        """Interpolate values at new resampled frequency across all nodes."""
        return self._apply_op("interpolate", kind=kind)

    def pad(self, tolerance: float | str | None = None) -> DataTree:
        """Forward fill values at new resampled frequency across all nodes."""
        return self._apply_op("pad", tolerance=tolerance)

    def ffill(self, tolerance: float | str | None = None) -> DataTree:
        """Forward fill values at new resampled frequency across all nodes."""
        return self._apply_op("ffill", tolerance=tolerance)

    def bfill(self, tolerance: float | str | None = None) -> DataTree:
        """Backward fill values at new resampled frequency across all nodes."""
        return self._apply_op("bfill", tolerance=tolerance)

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Dims = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> DataTree:
        """Apply a reduction function across all nodes along the resampled dimension."""
        return self._apply_op("reduce", func, dim=dim, keep_attrs=keep_attrs, **kwargs)

    def __repr__(self) -> str:
        indexer_str = ", ".join(f"{k}: {v}" for k, v in self.indexer.items())
        return f"DataTreeResample, indexer: {{{indexer_str}}}"
