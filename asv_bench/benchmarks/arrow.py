from __future__ import annotations

from typing import Any

import numpy as np

import xarray as xr

from . import parameterized, requires_pyarrow


def _dims_coords(shape: tuple[int, ...]):
    dims = [f"dim_{i}" for i in range(len(shape))]
    coords = {dim: np.arange(size) for dim, size in zip(dims, shape, strict=True)}
    return dims, coords


def _curvilinear_dims_coords(shape: tuple[int, int]):
    dims = ("y", "x")
    coords = {
        "lat": (dims, np.random.uniform(-90, 90, size=shape)),
        "lon": (dims, np.random.uniform(-180, 180, size=shape)),
    }
    return dims, coords


SHAPES = [
    (1_000_000,),  # 1D
    (1_000, 1_000),  # 2D
    (100, 100, 100),  # 3D
    (10_000, 10, 10),  # Unbalanced
]


class ToArrowDataset:
    params = [SHAPES]
    param_names = ["shape"]

    def setup(self, shape: tuple[int, ...]) -> None:
        requires_pyarrow()

        dims, coords = _dims_coords(shape)
        self.ds = xr.Dataset(
            {"temperature": (dims, np.random.random(shape))}, coords=coords
        )

    def time_to_arrow(self, shape: tuple[int, ...]) -> None:
        self.ds.to_arrow()

    def peakmem_to_arrow(self, shape: tuple[int, ...]) -> None:
        self.ds.to_arrow()


class ToArrowDataArray:
    params = [SHAPES]
    param_names = ["shape"]

    def setup(self, shape: tuple[int, ...]) -> None:
        requires_pyarrow()

        dims, coords = _dims_coords(shape)
        self.da = xr.DataArray(
            np.random.random(shape), dims=dims, coords=coords, name="temperature"
        )

    def time_to_arrow(self, shape: tuple[int, ...]) -> None:
        self.da.to_arrow()

    def peakmem_to_arrow(self, shape: tuple[int, ...]) -> None:
        self.da.to_arrow()


CURVILINEAR_SHAPES = [(1_000, 1_000)]


class ToArrowCurvilinear:
    """Benchmark ``to_arrow`` on a curvilinear grid with 2-D lat/lon coords."""

    params = [CURVILINEAR_SHAPES]
    param_names = ["shape"]

    def setup(self, shape: tuple[int, int]) -> None:
        requires_pyarrow()

        dims, coords = _curvilinear_dims_coords(shape)
        self.ds = xr.Dataset(
            {"temperature": (dims, np.random.random(shape))}, coords=coords
        )

    def time_to_arrow(self, shape: tuple[int, int]) -> None:
        self.ds.to_arrow()

    def peakmem_to_arrow(self, shape: tuple[int, int]) -> None:
        self.ds.to_arrow()


class ToArrowManyVars:
    """Benchmark Dataset with many data variables."""

    def setup(self, *args: Any, **kwargs: Any) -> None:
        requires_pyarrow()

        nvars = kwargs.get("nvars", 1)
        dim1 = 10_000
        dim2 = 10_000

        var = xr.Variable(dims=("dim1", "dim2"), data=np.random.random((dim1, dim2)))
        data_vars = {f"long_name_{v}": (("dim1", "dim2"), var) for v in range(nvars)}

        self.ds = xr.Dataset(
            data_vars, coords={"dim1": np.arange(dim1), "dim2": np.arange(dim2)}
        )

    @parameterized(["nvars"], ([1, 100]))
    def time_to_arrow(self, nvars: int) -> None:
        self.setup(nvars=nvars)
        self.ds.to_arrow()
