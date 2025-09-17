# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""File saving benchmarks for the CPerf scheme of the UK Met Office's NG-VAT project."""

from iris import save

from .. import on_demand_benchmark
from ..generate_data.ugrid import make_cube_like_2d_cubesphere, make_cube_like_umfield
from . import _N_CUBESPHERE_UM_EQUIVALENT, _UM_DIMS_YX


@on_demand_benchmark
class NetcdfSave:
    """Benchmark time and memory costs of saving ~large-ish data cubes to netcdf.

    Parametrised by file type.

    """

    params = ["LFRic", "UM"]
    param_names = ["data type"]

    def setup(self, data_type):
        if data_type == "LFRic":
            self.cube = make_cube_like_2d_cubesphere(
                n_cube=_N_CUBESPHERE_UM_EQUIVALENT, with_mesh=True
            )
        else:
            self.cube = make_cube_like_umfield(_UM_DIMS_YX)

    def _save_data(self, cube):
        save(cube, "tmp.nc")

    def time_save_data_netcdf(self, data_type):
        self._save_data(self.cube)

    def tracemalloc_save_data_netcdf(self, data_type):
        self._save_data(self.cube)
