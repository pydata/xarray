import numpy as np

import xarray as xr


class SwapDims:
    param_names = ["size"]
    params = [[int(1e3), int(1e5), int(1e7)]]

    def setup(self, size: int) -> None:
        self.ds = xr.Dataset(
            {"a": (("x", "t"), np.ones((size, 2)))},
            coords={
                "x": np.arange(size),
                "y": np.arange(size),
                "z": np.arange(size),
                "x2": ("x", np.arange(size)),
                "y2": ("y", np.arange(size)),
                "z2": ("z", np.arange(size)),
            },
        )

    def time_swap_dims(self, size: int) -> None:
        self.ds.swap_dims({"x": "xn", "y": "yn", "z": "zn"})

    def time_swap_dims_newindex(self, size: int) -> None:
        self.ds.swap_dims({"x": "x2", "y": "y2", "z": "z2"})
