# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Plot benchmark tests."""

import matplotlib as mpl
import numpy as np

from iris import coords, cube, plot

mpl.use("agg")


class AuxSort:
    def setup(self):
        # Manufacture data from which contours can be derived.
        # Should generate 10 distinct contours, regardless of dim size.
        dim_size = 200
        repeat_number = int(dim_size / 10)
        repeat_range = range(int((dim_size**2) / repeat_number))
        data = np.repeat(repeat_range, repeat_number)
        data = data.reshape((dim_size,) * 2)

        # These benchmarks are from a user perspective, so setting up a
        # user-level case that will prompt the calling of aux_coords.sort in plot.py.
        dim_coord = coords.DimCoord(np.arange(dim_size))
        local_cube = cube.Cube(data)
        local_cube.add_aux_coord(dim_coord, 0)
        self.cube = local_cube

    def time_aux_sort(self):
        # Contour plot arbitrarily picked. Known to prompt aux_coords.sort.
        plot.contour(self.cube)
