from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np

from xarray.core.coordinate_transform import CoordinateTransform
from xarray.core.dataarray import DataArray
from xarray.core.indexes import CoordinateTransformIndex, Index
from xarray.core.indexing import IndexSelResult
from xarray.core.variable import Variable


class RangeCoordinateTransform(CoordinateTransform):
    """Simple bounded interval 1-d coordinate transform."""

    left: float
    right: float
    dim: str
    size: int

    def __init__(
        self,
        left: float,
        right: float,
        coord_name: Hashable,
        dim: str,
        size: int,
        dtype: Any = None,
    ):
        if dtype is None:
            dtype = np.dtype(np.float64)

        super().__init__([coord_name], {dim: size}, dtype=dtype)

        self.left = left
        self.right = right
        self.dim = dim
        self.size = size

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        positions = dim_positions[self.dim]
        labels = self.left + positions * (self.right - self.left) / self.size
        return {self.dim: labels}

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        labels = coord_labels[self.coord_names[0]]
        positions = (labels - self.left) * self.size / (self.right - self.left)
        return {self.dim: positions}

    def equals(self, other: CoordinateTransform) -> bool:
        if not isinstance(other, RangeCoordinateTransform):
            return False

        return (
            self.left == other.left
            and self.right == other.right
            and self.size == other.size
        )


class RangeIndex(CoordinateTransformIndex):
    transform: RangeCoordinateTransform
    dim: str
    coord_name: Hashable
    size: int

    def __init__(
        self,
        left: float,
        right: float,
        coord_name: Hashable,
        dim: str,
        size: int,
        dtype: Any = None,
    ):
        self.transform = RangeCoordinateTransform(
            left, right, coord_name, dim, size, dtype
        )
        self.dim = dim
        self.coord_name = coord_name
        self.size = size

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> Index | None:
        idxer = indexers[self.dim]

        # straightforward to generate a new index if a slice is given with step 1
        if isinstance(idxer, slice) and (idxer.step == 1 or idxer.step is None):
            start = max(idxer.start, 0)
            stop = min(idxer.stop, self.size)

            new_left = self.transform.forward({self.dim: start})[self.coord_name]
            new_right = self.transform.forward({self.dim: stop})[self.coord_name]
            new_size = stop - start

            return RangeIndex(new_left, new_right, self.coord_name, self.dim, new_size)

        return None

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        label = labels[self.dim]

        if isinstance(label, slice):
            if label.step is None:
                # slice indexing (preserve the index)
                pos = self.transform.reverse(
                    {self.dim: np.array([label.start, label.stop])}
                )
                pos = np.round(pos[str(self.coord_name)]).astype("int")
                new_start = max(pos[0], 0)
                new_stop = min(pos[1], self.size)
                return IndexSelResult({self.dim: slice(new_start, new_stop)})
            else:
                # otherwise convert to basic (array) indexing
                label = np.arange(label.start, label.stop, label.step)

        # support basic indexing (in the 1D case basic vs. vectorized indexing
        # are pretty much similar)
        unwrap_xr = False
        if not isinstance(label, Variable | DataArray):
            # basic indexing -> either scalar or 1-d array
            try:
                var = Variable("_", label)
            except ValueError:
                var = Variable((), label)
            labels = {self.dim: var}
            unwrap_xr = True

        result = super().sel(labels, method=method, tolerance=tolerance)

        if unwrap_xr:
            dim_indexers = {self.dim: result.dim_indexers[self.dim].values}
            result = IndexSelResult(dim_indexers)

        return result
