import math
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from xarray.core.coordinate_transform import CoordinateTransform
from xarray.core.dataarray import DataArray
from xarray.core.indexes import CoordinateTransformIndex, Index, PandasIndex
from xarray.core.indexing import IndexSelResult, normalize_slice
from xarray.core.variable import Variable


class RangeCoordinateTransform(CoordinateTransform):
    """1-dimensional coordinate transform representing a simple bounded interval
    with evenly spaced, floating-point values.
    """

    start: float
    stop: float
    size: int
    coord_name: Hashable
    dim: str

    def __init__(
        self,
        start: float,
        stop: float,
        size: int,
        coord_name: Hashable,
        dim: str,
        dtype: Any = None,
    ):
        if dtype is None:
            dtype = np.dtype(np.float64)

        super().__init__([coord_name], {dim: size}, dtype=dtype)

        self.start = start
        self.stop = stop
        self.step = (stop - start) / size
        self.coord_name = coord_name
        self.dim = dim
        self.size = size

    def _replace(
        self, start: float, stop: float, size: int
    ) -> "RangeCoordinateTransform":
        return type(self)(
            start, stop, size, self.coord_name, self.dim, dtype=self.dtype
        )

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        positions = dim_positions[self.dim]
        labels = self.start + positions * self.step
        return {self.dim: labels}

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        labels = coord_labels[self.coord_names[0]]
        positions = (labels - self.start) - self.step
        return {self.dim: positions}

    def equals(self, other: CoordinateTransform) -> bool:
        if not isinstance(other, RangeCoordinateTransform):
            return False

        return (
            self.start == other.start
            and self.stop == other.stop
            and self.size == other.size
        )

    def slice(self, sl: slice) -> "RangeCoordinateTransform":
        sl = normalize_slice(sl, self.size)

        # TODO: support reverse transform (i.e., start > stop)?
        assert sl.start < sl.stop

        new_size = (sl.stop - sl.start) / sl.step
        new_start = self.start + sl.start * self.step
        new_stop = new_start + new_size * sl.step * self.step

        return self._replace(new_start, new_stop, new_size)


class RangeIndex(CoordinateTransformIndex):
    transform: RangeCoordinateTransform
    dim: str
    coord_name: Hashable
    size: int

    def __init__(self, transform: RangeCoordinateTransform):
        super().__init__(transform)
        self.dim = self.transform.dim
        self.size = self.transform.size
        self.coord_name = self.transform.coord_names[0]

    @classmethod
    def arange(
        cls,
        coord_name: Hashable,
        dim: str,
        start: float = 0.0,
        stop: float = 0.0,
        step: float = 1.0,
        dtype: Any = None,
    ) -> "RangeIndex":
        size = math.ceil((stop - start) / step)

        transform = RangeCoordinateTransform(
            start, stop, size, coord_name, dim, dtype=dtype
        )

        return cls(transform)

    @classmethod
    def linspace(
        cls,
        coord_name: Hashable,
        dim: str,
        start: float,
        stop: float,
        num: int = 50,
        endpoint: bool = True,
        dtype: Any = None,
    ) -> "RangeIndex":
        if endpoint:
            stop += (stop - start) / (num - 1)

        transform = RangeCoordinateTransform(
            start, stop, num, coord_name, dim, dtype=dtype
        )

        return cls(transform)

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> Index | None:
        idxer = indexers[self.dim]

        if isinstance(idxer, slice):
            return RangeIndex(self.transform.slice(idxer))
        elif isinstance(idxer, Variable) and idxer.ndim > 1:
            # vectorized (fancy) indexing with n-dimensional Variable: drop the index
            return None
        elif np.ndim(idxer) == 0:
            # scalar value
            return None
        else:
            # otherwise convert to a PandasIndex
            values = self.transform.forward({self.dim: idxer})[self.coord_name]
            if isinstance(idxer, Variable):
                new_dim = idxer.dims[0]
            else:
                new_dim = self.dim
            return PandasIndex(values, new_dim, coord_dtype=values.dtype)

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        label = labels[self.dim]

        if isinstance(label, slice):
            if label.step is None:
                # continuous interval slice indexing (preserves the index)
                positions = self.transform.reverse(
                    {self.coord_name: np.array([label.start, label.stop])}
                )
                pos = np.round(positions[self.dim]).astype("int")
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

    def to_pandas_index(self) -> pd.Index:
        values = self.transform.generate_coords()
        return pd.Index(values[self.dim])
