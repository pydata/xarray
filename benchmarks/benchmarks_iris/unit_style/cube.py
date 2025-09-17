# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Small-scope Cube benchmark tests."""

import numpy as np

from iris import analysis, aux_factory, coords, cube

from .. import disable_repeat_between_setup
from ..generate_data.stock import sample_meshcoord


def setup(*params):
    """General variables needed by multiple benchmark classes."""
    global data_1d
    global data_2d
    global general_cube

    data_2d = np.zeros((1000,) * 2)
    data_1d = data_2d[0]
    general_cube = cube.Cube(data_2d)


class ComponentCommon:
    # TODO: once https://github.com/airspeed-velocity/asv/pull/828 is released:
    #       * make class an ABC
    #       * remove NotImplementedError
    #       * combine setup_common into setup
    """Run a generalised suite of benchmarks for cubes.

    A base class running a generalised suite of benchmarks for cubes that
    include a specified component (e.g. Coord, CellMeasure etc.). Component to
    be specified in a subclass.

    ASV will run the benchmarks within this class for any subclasses.

    Should only be instantiated within subclasses, but cannot enforce this
    since ASV cannot handle classes that include abstract methods.
    """

    def setup(self):
        """Prevent ASV instantiating (must therefore override setup() in any subclasses.)."""
        raise NotImplementedError

    def create(self):
        """Create a cube (generic).

        cube_kwargs allow dynamic inclusion of different components;
        specified in subclasses.
        """
        return cube.Cube(data=data_2d, **self.cube_kwargs)

    def setup_common(self):
        """Shared setup code that can be called by subclasses."""
        self.cube = self.create()

    def time_create(self):
        """Create a cube that includes an instance of the benchmarked component."""
        self.create()

    def time_add(self):
        """Add an instance of the benchmarked component to an existing cube."""
        # Unable to create the copy during setup since this needs to be re-done
        # for every repeat of the test (some components disallow duplicates).
        general_cube_copy = general_cube.copy(data=data_2d)
        self.add_method(general_cube_copy, *self.add_args)


class Cube:
    def time_basic(self):
        cube.Cube(data_2d)

    def time_rename(self):
        general_cube.name = "air_temperature"


class AuxCoord(ComponentCommon):
    def setup(self):
        self.coord_name = "test"
        coord_bounds = np.array([data_1d - 1, data_1d + 1]).transpose()
        aux_coord = coords.AuxCoord(
            long_name=self.coord_name,
            points=data_1d,
            bounds=coord_bounds,
            units="days since 1970-01-01",
            climatological=True,
        )

        # Variables needed by the ComponentCommon base class.
        self.cube_kwargs = {"aux_coords_and_dims": [(aux_coord, 0)]}
        self.add_method = cube.Cube.add_aux_coord
        self.add_args = (aux_coord, (0))

        self.setup_common()

    def time_return_coords(self):
        self.cube.coords()

    def time_return_coord_dims(self):
        self.cube.coord_dims(self.coord_name)


class AuxFactory(ComponentCommon):
    def setup(self):
        coord = coords.AuxCoord(points=data_1d, units="m")
        self.hybrid_factory = aux_factory.HybridHeightFactory(delta=coord)

        # Variables needed by the ComponentCommon base class.
        self.cube_kwargs = {
            "aux_coords_and_dims": [(coord, 0)],
            "aux_factories": [self.hybrid_factory],
        }

        self.setup_common()

        # Variables needed by the overridden time_add benchmark in this subclass.
        cube_w_coord = self.cube.copy()
        [cube_w_coord.remove_aux_factory(i) for i in cube_w_coord.aux_factories]
        self.cube_w_coord = cube_w_coord

    def time_add(self):
        # Requires override from super().time_add because the cube needs an
        # additional coord.
        self.cube_w_coord.add_aux_factory(self.hybrid_factory)


class CellMeasure(ComponentCommon):
    def setup(self):
        cell_measure = coords.CellMeasure(data_1d)

        # Variables needed by the ComponentCommon base class.
        self.cube_kwargs = {"cell_measures_and_dims": [(cell_measure, 0)]}
        self.add_method = cube.Cube.add_cell_measure
        self.add_args = (cell_measure, 0)

        self.setup_common()


class CellMethod(ComponentCommon):
    def setup(self):
        cell_method = coords.CellMethod("test")

        # Variables needed by the ComponentCommon base class.
        self.cube_kwargs = {"cell_methods": [cell_method]}
        self.add_method = cube.Cube.add_cell_method
        self.add_args = [cell_method]

        self.setup_common()


class AncillaryVariable(ComponentCommon):
    def setup(self):
        ancillary_variable = coords.AncillaryVariable(data_1d)

        # Variables needed by the ComponentCommon base class.
        self.cube_kwargs = {"ancillary_variables_and_dims": [(ancillary_variable, 0)]}
        self.add_method = cube.Cube.add_ancillary_variable
        self.add_args = (ancillary_variable, 0)

        self.setup_common()


class MeshCoord:
    params = [
        6,  # minimal cube-sphere
        int(1e6),  # realistic cube-sphere size
        1000,  # To match size in :class:`AuxCoord`
    ]
    param_names = ["number of faces"]

    def setup(self, n_faces):
        mesh_kwargs = dict(n_nodes=n_faces + 2, n_edges=n_faces * 2, n_faces=n_faces)

        self.mesh_coord = sample_meshcoord(sample_mesh_kwargs=mesh_kwargs)
        self.data = np.zeros(n_faces)
        self.cube_blank = cube.Cube(data=self.data)
        self.cube = self.create()

    def create(self):
        return cube.Cube(data=self.data, aux_coords_and_dims=[(self.mesh_coord, 0)])

    def time_create(self, n_faces):
        _ = self.create()

    @disable_repeat_between_setup
    def time_add(self, n_faces):
        self.cube_blank.add_aux_coord(self.mesh_coord, 0)

    @disable_repeat_between_setup
    def time_remove(self, n_faces):
        self.cube.remove_coord(self.mesh_coord)


class Merge:
    def setup(self):
        self.cube_list = cube.CubeList()
        for i in np.arange(2):
            i_cube = general_cube.copy()
            i_coord = coords.AuxCoord([i])
            i_cube.add_aux_coord(i_coord)
            self.cube_list.append(i_cube)

    def time_merge(self):
        self.cube_list.merge()


class Concatenate:
    def setup(self):
        dim_size = 1000
        self.cube_list = cube.CubeList()
        for i in np.arange(dim_size * 2, step=dim_size):
            i_cube = general_cube.copy()
            i_coord = coords.DimCoord(np.arange(dim_size) + (i * dim_size))
            i_cube.add_dim_coord(i_coord, 0)
            self.cube_list.append(i_cube)

    def time_concatenate(self):
        self.cube_list.concatenate()


class Equality:
    def setup(self):
        self.cube_a = general_cube.copy()
        self.cube_b = general_cube.copy()

        aux_coord = coords.AuxCoord(data_1d)
        self.cube_a.add_aux_coord(aux_coord, 0)
        self.cube_b.add_aux_coord(aux_coord, 1)

    def time_equality(self):
        self.cube_a == self.cube_b


class Aggregation:
    def setup(self):
        repeat_number = 10
        repeat_range = range(int(1000 / repeat_number))
        array_repeat = np.repeat(repeat_range, repeat_number)
        array_unique = np.arange(len(array_repeat))

        coord_repeat = coords.AuxCoord(points=array_repeat, long_name="repeat")
        coord_unique = coords.DimCoord(points=array_unique, long_name="unique")

        local_cube = general_cube.copy()
        local_cube.add_aux_coord(coord_repeat, 0)
        local_cube.add_dim_coord(coord_unique, 0)
        self.cube = local_cube

    def time_aggregated_by(self):
        self.cube.aggregated_by("repeat", analysis.MEAN)
