# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Trajectory benchmark test."""

# import iris tests first so that some things can be initialised before
# importing anything else
from iris import tests  # isort:skip

import numpy as np

import iris
from iris.analysis.trajectory import interpolate


class TrajectoryInterpolation:
    def setup(self) -> None:
        # Prepare a cube and a template

        cube_file_path = tests.get_data_path(["NetCDF", "regrid", "regrid_xyt.nc"])
        self.cube = iris.load_cube(cube_file_path)

        trajectory = np.array([np.array((-50 + i, -50 + i)) for i in range(100)])
        self.sample_points = [
            ("longitude", trajectory[:, 0]),
            ("latitude", trajectory[:, 1]),
        ]

    def time_trajectory_linear(self) -> None:
        # Regrid the cube onto the template.
        out_cube = interpolate(self.cube, self.sample_points, method="linear")
        # Realise the data
        out_cube.data

    def tracemalloc_trajectory_linear(self) -> None:
        # Regrid the cube onto the template.
        out_cube = interpolate(self.cube, self.sample_points, method="linear")
        # Realise the data
        out_cube.data

    tracemalloc_trajectory_linear.number = 3  # type: ignore[attr-defined]

    def time_trajectory_nearest(self) -> None:
        # Regrid the cube onto the template.
        out_cube = interpolate(self.cube, self.sample_points, method="nearest")
        # Realise the data
        out_cube.data

    def tracemalloc_trajectory_nearest(self) -> None:
        # Regrid the cube onto the template.
        out_cube = interpolate(self.cube, self.sample_points, method="nearest")
        # Realise the data
        out_cube.data

    tracemalloc_trajectory_nearest.number = 3  # type: ignore[attr-defined]
