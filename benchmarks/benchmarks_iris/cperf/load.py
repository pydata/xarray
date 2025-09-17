# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""File loading benchmarks for the CPerf scheme of the UK Met Office's NG-VAT project."""

from .. import on_demand_benchmark
from . import SingleDiagnosticMixin


@on_demand_benchmark
class SingleDiagnosticLoad(SingleDiagnosticMixin):
    def time_load(self, _, __, ___):
        """Perform a 'real world comparison'.

        * UM coords are always realised (DimCoords).
        * LFRic coords are not realised by default (MeshCoords).

        """
        cube = self.load()
        assert cube.has_lazy_data()
        # UM files load lon/lat as DimCoords, which are always realised.
        expecting_lazy_coords = self.file_type == "LFRic"
        for coord_name in "longitude", "latitude":
            coord = cube.coord(coord_name)
            assert coord.has_lazy_points() == expecting_lazy_coords
            assert coord.has_lazy_bounds() == expecting_lazy_coords

    def time_load_w_realised_coords(self, _, __, ___):
        """Valuable extra comparison where both UM and LFRic coords are realised."""
        cube = self.load()
        for coord_name in "longitude", "latitude":
            coord = cube.coord(coord_name)
            # Don't touch actual points/bounds objects - permanent
            #  realisation plays badly with ASV's re-run strategy.
            if coord.has_lazy_points():
                coord.core_points().compute()
            if coord.has_lazy_bounds():
                coord.core_bounds().compute()


@on_demand_benchmark
class SingleDiagnosticRealise(SingleDiagnosticMixin):
    # The larger files take a long time to realise.
    timeout = 600.0

    def setup(self, file_type, three_d, three_times):
        super().setup(file_type, three_d, three_times)
        self.loaded_cube = self.load()

    def time_realise(self, _, __, ___):
        # Don't touch loaded_cube.data - permanent realisation plays badly with
        #  ASV's re-run strategy.
        assert self.loaded_cube.has_lazy_data()
        self.loaded_cube.core_data().compute()
