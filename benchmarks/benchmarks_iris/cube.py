# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Cube benchmark tests."""

from collections.abc import Iterable

from iris import coords
from iris.cube import Cube

from .generate_data.stock import realistic_4d_w_everything


class CubeCreation:
    params = [[False, True], ["instantiate", "construct"]]
    param_names = ["Cube has mesh", "Cube creation strategy"]

    cube_kwargs: dict

    def setup(self, w_mesh: bool, _) -> None:
        # Loaded as two cubes due to the hybrid height.
        source_cube = realistic_4d_w_everything(w_mesh=w_mesh)

        def get_coords_and_dims(
            coords_iter: Iterable[coords._DimensionalMetadata],
        ) -> list[tuple[coords._DimensionalMetadata, tuple[int, ...]]]:
            return [(c, c.cube_dims(source_cube)) for c in coords_iter]

        self.cube_kwargs = dict(
            data=source_cube.data,
            standard_name=source_cube.standard_name,
            long_name=source_cube.long_name,
            var_name=source_cube.var_name,
            units=source_cube.units,
            attributes=source_cube.attributes,
            cell_methods=source_cube.cell_methods,
            dim_coords_and_dims=get_coords_and_dims(source_cube.dim_coords),
            aux_coords_and_dims=get_coords_and_dims(source_cube.aux_coords),
            aux_factories=source_cube.aux_factories,
            cell_measures_and_dims=get_coords_and_dims(source_cube.cell_measures()),
            ancillary_variables_and_dims=get_coords_and_dims(
                source_cube.ancillary_variables()
            ),
        )

    def time_create(self, _, cube_creation_strategy: str) -> None:
        if cube_creation_strategy == "instantiate":
            _ = Cube(**self.cube_kwargs)

        elif cube_creation_strategy == "construct":
            new_cube = Cube(data=self.cube_kwargs["data"])
            new_cube.standard_name = self.cube_kwargs["standard_name"]
            new_cube.long_name = self.cube_kwargs["long_name"]
            new_cube.var_name = self.cube_kwargs["var_name"]
            new_cube.units = self.cube_kwargs["units"]
            new_cube.attributes = self.cube_kwargs["attributes"]
            new_cube.cell_methods = self.cube_kwargs["cell_methods"]
            for coord, dims in self.cube_kwargs["dim_coords_and_dims"]:
                assert isinstance(coord, coords.DimCoord)  # Type hint to help linters.
                new_cube.add_dim_coord(coord, dims)
            for coord, dims in self.cube_kwargs["aux_coords_and_dims"]:
                new_cube.add_aux_coord(coord, dims)
            for aux_factory in self.cube_kwargs["aux_factories"]:
                new_cube.add_aux_factory(aux_factory)
            for cell_measure, dims in self.cube_kwargs["cell_measures_and_dims"]:
                new_cube.add_cell_measure(cell_measure, dims)
            for ancillary_variable, dims in self.cube_kwargs[
                "ancillary_variables_and_dims"
            ]:
                new_cube.add_ancillary_variable(ancillary_variable, dims)

        else:
            message = f"Unknown cube creation strategy: {cube_creation_strategy}"
            raise NotImplementedError(message)


class CubeEquality:
    params = [
        [False, True],
        [False, True],
        ["metadata_inequality", "coord_inequality", "data_inequality", "all_equal"],
    ]
    param_names = ["Cubes are lazy", "Cubes have meshes", "Scenario"]

    cube_1: Cube
    cube_2: Cube
    coord_name = "surface_altitude"

    def setup(self, lazy: bool, w_mesh: bool, scenario: str) -> None:
        self.cube_1 = realistic_4d_w_everything(w_mesh=w_mesh, lazy=lazy)
        # Using Cube.copy() produces different results due to sharing of the
        #  Mesh instance.
        self.cube_2 = realistic_4d_w_everything(w_mesh=w_mesh, lazy=lazy)

        match scenario:
            case "metadata_inequality":
                self.cube_2.long_name = "different"
            case "coord_inequality":
                coord = self.cube_2.coord(self.coord_name)
                coord.points = coord.core_points() * 2
            case "data_inequality":
                self.cube_2.data = self.cube_2.core_data() * 2
            case "all_equal":
                pass
            case _:
                message = f"Unknown scenario: {scenario}"
                raise NotImplementedError(message)

    def time_equality(self, lazy: bool, __, ___) -> None:
        _ = self.cube_1 == self.cube_2
        if lazy:
            for cube in (self.cube_1, self.cube_2):
                # Confirm that this benchmark is safe for repetition.
                assert cube.coord(self.coord_name).has_lazy_points()
                assert cube.has_lazy_data()
