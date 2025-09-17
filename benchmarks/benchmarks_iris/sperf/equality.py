# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Equality benchmarks for the SPerf scheme of the UK Met Office's NG-VAT project."""

from .. import on_demand_benchmark
from . import FileMixin


@on_demand_benchmark
class CubeEquality(FileMixin):
    r"""Benchmark time and memory costs.

    Benchmark time and memory costs of comparing :class:`~iris.cube.Cube`\\ s
    with attached :class:`~iris.mesh.MeshXY`\\ es.

    Uses :class:`FileMixin` as the realistic case will be comparing
    :class:`~iris.cube.Cube`\\ s that have been loaded from file.

    """

    # Cut down paremt parameters.
    params = [FileMixin.params[0]]

    def setup(self, c_size, n_levels=1, n_times=1):
        super().setup(c_size, n_levels, n_times)
        self.cube = self.load_cube()
        self.other_cube = self.load_cube()

    def peakmem_eq(self, n_cube):
        _ = self.cube == self.other_cube

    def time_eq(self, n_cube):
        _ = self.cube == self.other_cube
