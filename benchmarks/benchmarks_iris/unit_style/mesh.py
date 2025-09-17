# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Benchmark tests for the iris.mesh module."""

from copy import deepcopy

import numpy as np

from iris import mesh

from .. import disable_repeat_between_setup
from ..generate_data.stock import sample_mesh


class UGridCommon:
    """Run a generalised suite of benchmarks for any mesh object.

    A base class running a generalised suite of benchmarks for any mesh object.
    Object to be specified in a subclass.

    ASV will run the benchmarks within this class for any subclasses.

    ASV will not benchmark this class as setup() triggers a NotImplementedError.
    (ASV has not yet released ABC/abstractmethod support - asv#838).

    """

    params = [
        6,  # minimal cube-sphere
        int(1e6),  # realistic cube-sphere size
    ]
    param_names = ["number of faces"]

    def setup(self, *params):
        self.object = self.create()

    def create(self):
        raise NotImplementedError

    def time_create(self, *params):
        """Create an instance of the benchmarked object.

        create() method is specified in the subclass.
        """
        self.create()


class Connectivity(UGridCommon):
    def setup(self, n_faces):
        self.array = np.zeros([n_faces, 3], dtype=int)
        super().setup(n_faces)

    def create(self):
        return mesh.Connectivity(indices=self.array, cf_role="face_node_connectivity")

    def time_indices(self, n_faces):
        _ = self.object.indices

    def time_location_lengths(self, n_faces):
        # Proofed against the Connectivity name change (633ed17).
        if getattr(self.object, "src_lengths", False):
            meth = self.object.src_lengths
        else:
            meth = self.object.location_lengths
        _ = meth()

    def time_validate_indices(self, n_faces):
        self.object.validate_indices()


@disable_repeat_between_setup
class ConnectivityLazy(Connectivity):
    """Lazy equivalent of :class:`Connectivity`."""

    def setup(self, n_faces):
        super().setup(n_faces)
        self.array = self.object.lazy_indices()
        self.object = self.create()


class MeshXY(UGridCommon):
    def setup(self, n_faces, lazy=False):
        ####
        # Steal everything from the sample mesh for benchmarking creation of a
        #  brand new mesh.
        source_mesh = sample_mesh(
            n_nodes=n_faces + 2,
            n_edges=n_faces * 2,
            n_faces=n_faces,
            lazy_values=lazy,
        )

        def get_coords_and_axes(location):
            return [
                (source_mesh.coord(axis=axis, location=location), axis)
                for axis in ("x", "y")
            ]

        self.mesh_kwargs = dict(
            topology_dimension=source_mesh.topology_dimension,
            node_coords_and_axes=get_coords_and_axes("node"),
            connectivities=source_mesh.connectivities(),
            edge_coords_and_axes=get_coords_and_axes("edge"),
            face_coords_and_axes=get_coords_and_axes("face"),
        )
        ####

        super().setup(n_faces)

        self.face_node = self.object.face_node_connectivity
        self.node_x = self.object.node_coords.node_x
        # Kwargs for reuse in search and remove methods.
        self.connectivities_kwarg = dict(cf_role="edge_node_connectivity")
        self.coords_kwarg = dict(location="face")

        # TODO: an opportunity for speeding up runtime if needed, since
        #  eq_object is not needed for all benchmarks. Just don't generate it
        #  within a benchmark - the execution time is large enough that it
        #  could be a significant portion of the benchmark - makes regressions
        #  smaller and could even pick up regressions in copying instead!
        self.eq_object = deepcopy(self.object)

    def create(self):
        return mesh.MeshXY(**self.mesh_kwargs)

    def time_add_connectivities(self, n_faces):
        self.object.add_connectivities(self.face_node)

    def time_add_coords(self, n_faces):
        self.object.add_coords(node_x=self.node_x)

    def time_connectivities(self, n_faces):
        _ = self.object.connectivities(**self.connectivities_kwarg)

    def time_coords(self, n_faces):
        _ = self.object.coords(**self.coords_kwarg)

    def time_eq(self, n_faces):
        _ = self.object == self.eq_object

    def time_remove_connectivities(self, n_faces):
        self.object.remove_connectivities(**self.connectivities_kwarg)

    def time_remove_coords(self, n_faces):
        self.object.remove_coords(**self.coords_kwarg)


@disable_repeat_between_setup
class MeshXYLazy(MeshXY):
    """Lazy equivalent of :class:`MeshXY`."""

    def setup(self, n_faces, lazy=True):
        super().setup(n_faces, lazy=lazy)


class MeshCoord(UGridCommon):
    # Add extra parameter value to match AuxCoord benchmarking.
    params = UGridCommon.params + [1000]

    def setup(self, n_faces, lazy=False):
        self.mesh = sample_mesh(
            n_nodes=n_faces + 2,
            n_edges=n_faces * 2,
            n_faces=n_faces,
            lazy_values=lazy,
        )

        super().setup(n_faces)

    def create(self):
        return mesh.MeshCoord(mesh=self.mesh, location="face", axis="x")

    def time_points(self, n_faces):
        _ = self.object.points

    def time_bounds(self, n_faces):
        _ = self.object.bounds


@disable_repeat_between_setup
class MeshCoordLazy(MeshCoord):
    """Lazy equivalent of :class:`MeshCoord`."""

    def setup(self, n_faces, lazy=True):
        super().setup(n_faces, lazy=lazy)
