import numpy as np

import xarray as xr
from xarray.coding import cf


class TestCFCoordinateCoder:
    def test_init(self):
        coder = cf.CFCoordinateCoder()

        assert coder.kind == "dataset"

    def test_decode(self):
        encoded = xr.Dataset(
            {
                "var1": ("x", [1, 2, 3], {"coordinates": "coord1 coord2"}),
                "var2": (
                    ("x", "y"),
                    [[9, 4], [3, 6], [87, 7]],
                    {"coordinates": "coord1 coord2 coord3 coord4"},
                ),
                "x": ("x", [0, 1, 2]),
                "y": ("y", [-1, 1]),
                "coord1": ("x", [0, 2, 4]),
                "coord2": ("x", [1, 3, 5]),
                "coord3": ("y", [7, 6]),
                "coord4": (["x", "y"], np.ones((3, 2))),
                "coord5": ("z", [0]),
            },
            coords=xr.Coordinates(),
            attrs={"coordinates": "coord5"},
        )

        coder = cf.CFCoordinateCoder()
        decoded = coder.decode(encoded)

        expected = {"x", "y", "coord1", "coord2", "coord3", "coord4", "coord5"}

        assert set(decoded.coords) == expected

    def test_encode(self):
        decoded = xr.Dataset(
            {"var1": ("x", [1, 2, 3]), "var2": (["x", "y"], [[9, 4], [3, 6], [87, 7]])},
            coords={
                "x": [0, 1, 2],
                "y": [-1, 1],
                "coord1": ("x", [0, 2, 4]),
                "coord2": ("x", [1, 3, 5]),
                "coord3": ("y", [7, 6]),
                "coord4": (["x", "y"], np.ones((3, 2))),
                "coord5": ("z", [0]),
            },
        )

        coder = cf.CFCoordinateCoder()
        encoded = coder.encode(decoded)

        assert encoded.attrs == {"coordinates": "coord5"}
        assert encoded["var1"].attrs == {"coordinates": "coord1 coord2"}
        assert encoded["var2"].attrs == {"coordinates": "coord1 coord2 coord3 coord4"}
