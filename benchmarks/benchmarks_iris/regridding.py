# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Regridding benchmark test."""

# import iris tests first so that some things can be initialised before
# importing anything else
from iris import tests  # isort:skip

import numpy as np

import iris
from iris.analysis import AreaWeighted, PointInCell
from iris.coords import AuxCoord


class HorizontalChunkedRegridding:
    def setup(self) -> None:
        # Prepare a cube and a template

        cube_file_path = tests.get_data_path(["NetCDF", "regrid", "regrid_xyt.nc"])
        self.cube = iris.load_cube(cube_file_path)

        # Prepare a tougher cube and chunk it
        chunked_cube_file_path = tests.get_data_path(
            ["NetCDF", "regrid", "regrid_xyt.nc"]
        )
        self.chunked_cube = iris.load_cube(chunked_cube_file_path)

        # Chunked data makes the regridder run repeatedly
        self.cube.data = self.cube.lazy_data().rechunk((1, -1, -1))

        template_file_path = tests.get_data_path(
            ["NetCDF", "regrid", "regrid_template_global_latlon.nc"]
        )
        self.template_cube = iris.load_cube(template_file_path)

        # Prepare a regridding scheme
        self.scheme_area_w = AreaWeighted()

    def time_regrid_area_w(self) -> None:
        # Regrid the cube onto the template.
        out = self.cube.regrid(self.template_cube, self.scheme_area_w)
        # Realise the data
        out.data

    def time_regrid_area_w_new_grid(self) -> None:
        # Regrid the chunked cube
        out = self.chunked_cube.regrid(self.template_cube, self.scheme_area_w)
        # Realise data
        out.data

    def tracemalloc_regrid_area_w(self) -> None:
        # Regrid the chunked cube
        out = self.cube.regrid(self.template_cube, self.scheme_area_w)
        # Realise data
        out.data

    tracemalloc_regrid_area_w.number = 3  # type: ignore[attr-defined]

    def tracemalloc_regrid_area_w_new_grid(self) -> None:
        # Regrid the chunked cube
        out = self.chunked_cube.regrid(self.template_cube, self.scheme_area_w)
        # Realise data
        out.data

    tracemalloc_regrid_area_w_new_grid.number = 3  # type: ignore[attr-defined]


class CurvilinearRegridding:
    def setup(self) -> None:
        # Prepare a cube and a template

        cube_file_path = tests.get_data_path(["NetCDF", "regrid", "regrid_xyt.nc"])
        self.cube = iris.load_cube(cube_file_path)

        # Make the source cube curvilinear
        x_coord = self.cube.coord("longitude")
        y_coord = self.cube.coord("latitude")
        xx, yy = np.meshgrid(x_coord.points, y_coord.points)
        self.cube.remove_coord(x_coord)
        self.cube.remove_coord(y_coord)
        x_coord_2d = AuxCoord(
            xx,
            standard_name=x_coord.standard_name,
            units=x_coord.units,
            coord_system=x_coord.coord_system,
        )
        y_coord_2d = AuxCoord(
            yy,
            standard_name=y_coord.standard_name,
            units=y_coord.units,
            coord_system=y_coord.coord_system,
        )
        self.cube.add_aux_coord(x_coord_2d, (1, 2))
        self.cube.add_aux_coord(y_coord_2d, (1, 2))

        template_file_path = tests.get_data_path(
            ["NetCDF", "regrid", "regrid_template_global_latlon.nc"]
        )
        self.template_cube = iris.load_cube(template_file_path)

        # Prepare a regridding scheme
        self.scheme_pic = PointInCell()

    def time_regrid_pic(self) -> None:
        # Regrid the cube onto the template.
        out = self.cube.regrid(self.template_cube, self.scheme_pic)
        # Realise the data
        out.data

    def tracemalloc_regrid_pic(self) -> None:
        # Regrid the cube onto the template.
        out = self.cube.regrid(self.template_cube, self.scheme_pic)
        # Realise the data
        out.data

    tracemalloc_regrid_pic.number = 3  # type: ignore[attr-defined]
