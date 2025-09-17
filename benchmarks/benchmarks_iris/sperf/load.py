# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""File loading benchmarks for the SPerf scheme of the UK Met Office's NG-VAT project."""

from .. import on_demand_benchmark
from . import FileMixin


@on_demand_benchmark
class Load(FileMixin):
    def time_load_cube(self, _, __, ___):
        _ = self.load_cube()


@on_demand_benchmark
class Realise(FileMixin):
    def setup(self, c_size, n_levels, n_times):
        super().setup(c_size, n_levels, n_times)
        self.loaded_cube = self.load_cube()

    def time_realise_cube(self, _, __, ___):
        # Don't touch loaded_cube.data - permanent realisation plays badly with
        #  ASV's re-run strategy.
        assert self.loaded_cube.has_lazy_data()
        self.loaded_cube.core_data().compute()
