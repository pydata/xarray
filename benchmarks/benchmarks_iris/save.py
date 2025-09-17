# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""File saving benchmarks."""

from iris import save
from iris.mesh import save_mesh

from .generate_data.ugrid import make_cube_like_2d_cubesphere


class NetcdfSave:
    params = [[50, 600], [False, True]]
    param_names = ["cubesphere-N", "is_unstructured"]

    def setup(self, n_cubesphere, is_unstructured):
        self.cube = make_cube_like_2d_cubesphere(
            n_cube=n_cubesphere, with_mesh=is_unstructured
        )

    def _save_data(self, cube, do_copy=True):
        if do_copy:
            # Copy the cube, to avoid distorting the results by changing it
            # Because we known that older Iris code realises lazy coords
            cube = cube.copy()
        save(cube, "tmp.nc")

    def _save_mesh(self, cube):
        # In this case, we are happy that the mesh is *not* modified
        save_mesh(cube.mesh, "mesh.nc")

    def time_netcdf_save_cube(self, n_cubesphere, is_unstructured):
        self._save_data(self.cube)

    def time_netcdf_save_mesh(self, n_cubesphere, is_unstructured):
        if is_unstructured:
            self._save_mesh(self.cube)

    def tracemalloc_netcdf_save(self, n_cubesphere, is_unstructured):
        # Don't need to copy the cube here since track_ benchmarks don't
        #  do repeats between self.setup() calls.
        self._save_data(self.cube, do_copy=False)
