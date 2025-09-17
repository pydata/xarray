# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Stats benchmark tests."""

import iris
from iris.analysis.stats import pearsonr
import iris.tests


class PearsonR:
    def setup(self):
        cube_temp = iris.load_cube(
            iris.tests.get_data_path(
                ("NetCDF", "global", "xyt", "SMALL_total_column_co2.nc")
            )
        )

        # Make data non-lazy.
        cube_temp.data

        self.cube_a = cube_temp[:6]
        self.cube_b = cube_temp[20:26]
        self.cube_b.replace_coord(self.cube_a.coord("time"))
        for name in ["latitude", "longitude"]:
            self.cube_b.coord(name).guess_bounds()
        self.weights = iris.analysis.cartography.area_weights(self.cube_b)

    def time_real(self):
        pearsonr(self.cube_a, self.cube_b, weights=self.weights)

    def tracemalloc_real(self):
        pearsonr(self.cube_a, self.cube_b, weights=self.weights)

    tracemalloc_real.number = 3  # type: ignore[attr-defined]

    def time_lazy(self):
        for cube in self.cube_a, self.cube_b:
            cube.data = cube.lazy_data()

        result = pearsonr(self.cube_a, self.cube_b, weights=self.weights)
        result.data

    def tracemalloc_lazy(self):
        for cube in self.cube_a, self.cube_b:
            cube.data = cube.lazy_data()

        result = pearsonr(self.cube_a, self.cube_b, weights=self.weights)
        result.data

    tracemalloc_lazy.number = 3  # type: ignore[attr-defined]
