# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Small-scope metadata manager factory benchmark tests."""

from iris.common import (
    AncillaryVariableMetadata,
    BaseMetadata,
    CellMeasureMetadata,
    CoordMetadata,
    CubeMetadata,
    DimCoordMetadata,
    metadata_manager_factory,
)


class MetadataManagerFactory__create:
    params = [1, 10, 100]

    def time_AncillaryVariableMetadata(self, n):
        [metadata_manager_factory(AncillaryVariableMetadata) for _ in range(n)]

    def time_BaseMetadata(self, n):
        [metadata_manager_factory(BaseMetadata) for _ in range(n)]

    def time_CellMeasureMetadata(self, n):
        [metadata_manager_factory(CellMeasureMetadata) for _ in range(n)]

    def time_CoordMetadata(self, n):
        [metadata_manager_factory(CoordMetadata) for _ in range(n)]

    def time_CubeMetadata(self, n):
        [metadata_manager_factory(CubeMetadata) for _ in range(n)]

    def time_DimCoordMetadata(self, n):
        [metadata_manager_factory(DimCoordMetadata) for _ in range(n)]


class MetadataManagerFactory:
    def setup(self):
        self.ancillary = metadata_manager_factory(AncillaryVariableMetadata)
        self.base = metadata_manager_factory(BaseMetadata)
        self.cell = metadata_manager_factory(CellMeasureMetadata)
        self.coord = metadata_manager_factory(CoordMetadata)
        self.cube = metadata_manager_factory(CubeMetadata)
        self.dim = metadata_manager_factory(DimCoordMetadata)

    def time_AncillaryVariableMetadata_fields(self):
        self.ancillary.fields

    def time_AncillaryVariableMetadata_values(self):
        self.ancillary.values

    def time_BaseMetadata_fields(self):
        self.base.fields

    def time_BaseMetadata_values(self):
        self.base.values

    def time_CellMeasuresMetadata_fields(self):
        self.cell.fields

    def time_CellMeasuresMetadata_values(self):
        self.cell.values

    def time_CoordMetadata_fields(self):
        self.coord.fields

    def time_CoordMetadata_values(self):
        self.coord.values

    def time_CubeMetadata_fields(self):
        self.cube.fields

    def time_CubeMetadata_values(self):
        self.cube.values

    def time_DimCoordMetadata_fields(self):
        self.dim.fields

    def time_DimCoordMetadata_values(self):
        self.dim.values
