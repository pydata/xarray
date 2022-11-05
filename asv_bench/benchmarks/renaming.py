import numpy as np

import xarray as xr

from . import parameterized

nx = {"s": 1000, "m": int(1e5), "l": int(1e7)}


class SwapDims:
    def setup(self) -> None:
        self.ds = {
            size: xr.Dataset(
                {"a": (("x", "t"), np.ones((n, 2)))},
                coords={
                    "x": np.arange(n),
                    "y": np.arange(n),
                    "z": np.arange(n),
                    "x2": ("x", np.arange(n)),
                    "y2": ("y", np.arange(n)),
                    "z2": ("z", np.arange(n)),
                },
            )
            for size, n in nx.items()
        }

    @parameterized(["size"], [list(nx.keys())])
    def time_swap_dims(self, size: str) -> None:
        self.ds[size].swap_dims({"x": "xn", "y": "yn", "z": "zn"})

    @parameterized(["size"], [list(nx.keys())])
    def time_swap_dims_newindex(self, size: str) -> None:
        self.ds[size].swap_dims({"x": "x2", "y": "y2", "z": "z2"})
