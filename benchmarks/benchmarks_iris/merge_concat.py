# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Benchmarks relating to :meth:`iris.cube.CubeList.merge` and ``concatenate``."""

import warnings

import numpy as np

from iris.cube import CubeList
from iris.warnings import IrisVagueMetadataWarning

from .generate_data.stock import realistic_4d_w_everything


class Merge:
    # TODO: Improve coverage.

    cube_list: CubeList

    def setup(self):
        source_cube = realistic_4d_w_everything()

        # Merge does not yet fully support cell measures and ancillary variables.
        for cm in source_cube.cell_measures():
            source_cube.remove_cell_measure(cm)
        for av in source_cube.ancillary_variables():
            source_cube.remove_ancillary_variable(av)

        second_cube = source_cube.copy()
        scalar_coord = second_cube.coords(dimensions=[])[0]
        scalar_coord.points = scalar_coord.points + 1
        self.cube_list = CubeList([source_cube, second_cube])

    def time_merge(self):
        _ = self.cube_list.merge_cube()

    def tracemalloc_merge(self):
        _ = self.cube_list.merge_cube()

    tracemalloc_merge.number = 3  # type: ignore[attr-defined]


class Concatenate:
    # TODO: Improve coverage.

    cube_list: CubeList

    params = [[False, True]]
    param_names = ["Lazy operations"]

    def setup(self, lazy_run: bool):
        warnings.filterwarnings("ignore", message="Ignoring a datum")
        warnings.filterwarnings("ignore", category=IrisVagueMetadataWarning)
        source_cube = realistic_4d_w_everything(lazy=lazy_run)
        self.cube_list = CubeList([source_cube])
        for _ in range(24):
            next_cube = self.cube_list[-1].copy()
            first_dim_coord = next_cube.coord(dimensions=0, dim_coords=True)
            first_dim_coord.points = (
                first_dim_coord.points + np.ptp(first_dim_coord.points) + 1
            )
            self.cube_list.append(next_cube)

    def time_concatenate(self, _):
        _ = self.cube_list.concatenate_cube()

    def tracemalloc_concatenate(self, _):
        _ = self.cube_list.concatenate_cube()

    tracemalloc_concatenate.number = 3  # type: ignore[attr-defined]
