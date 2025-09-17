# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""File saving benchmarks for the SPerf scheme of the UK Met Office's NG-VAT project."""

import os.path

from iris import save
from iris.mesh import save_mesh

from .. import on_demand_benchmark
from ..generate_data.ugrid import make_cube_like_2d_cubesphere


@on_demand_benchmark
class NetcdfSave:
    """Benchmark time and memory costs of saving ~large-ish data cubes to netcdf."""

    params = [[1, 100, 200, 300, 500, 1000, 1668], [False, True]]
    param_names = ["cubesphere_C<N>", "is_unstructured"]
    # Fix result units for the tracking benchmarks.
    unit = "Mb"

    def setup(self, n_cubesphere, is_unstructured):
        self.cube = make_cube_like_2d_cubesphere(
            n_cube=n_cubesphere, with_mesh=is_unstructured
        )

    def _save_cube(self, cube):
        save(cube, "tmp.nc")

    def _save_mesh(self, cube):
        save_mesh(cube.mesh, "mesh.nc")

    def time_save_cube(self, n_cubesphere, is_unstructured):
        self._save_cube(self.cube)

    def tracemalloc_save_cube(self, n_cubesphere, is_unstructured):
        self._save_cube(self.cube)

    def time_save_mesh(self, n_cubesphere, is_unstructured):
        if is_unstructured:
            self._save_mesh(self.cube)

    # The filesizes make a good reference point for the 'addedmem' memory
    #  usage results.
    def track_filesize_save_cube(self, n_cubesphere, is_unstructured):
        self._save_cube(self.cube)
        return os.path.getsize("tmp.nc") * 1.0e-6
