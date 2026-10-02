import itertools
from typing import ClassVar

from xarray.coding.core import CoderKind
from xarray.coding.variables import SerializationWarning
from xarray.core.dataset import Dataset
from xarray.core.utils import emit_user_level_warning


class CFCoordinateCoder:
    """CF coordinate coder

    Allows for roundtripping variables as coordinates. Note that `xarray`
    associates coordinates based on dimensions, so the association of
    coordinates with specific coordinates is lost.
    """

    kind: ClassVar[CoderKind] = "dataset"

    def decode(self, obj: Dataset) -> Dataset:
        unparsed = [
            var.attrs.pop("coordinates", None) for var in obj.variables.values()
        ] + [obj.attrs.get("coordinates", None)]

        dim_coords = [name for name in obj.variables if name in obj.dims]
        non_dim_coords = list(
            itertools.chain.from_iterable(
                coordinates.split(" ")
                for coordinates in unparsed
                if isinstance(coordinates, str)
            )
        )

        return obj.set_coords(dim_coords + non_dim_coords)

    def encode(self, obj: Dataset) -> Dataset:
        encoded = obj.copy(deep=False)
        coords = dict(encoded.coords)
        for name in list(coords):
            if isinstance(name, str) and " " in name:
                emit_user_level_warning(
                    f"coordinate {name!r} has a space in its name, which means it "
                    "cannot be marked as a coordinate on disk and will be "
                    "saved as a data variable instead",
                    category=SerializationWarning,
                )
                del coords[name]

        covered = set()
        for variable in encoded.values():
            dims = set(variable.dims)
            coordinates = [
                name
                for name, coord in coords.items()
                if name not in encoded.dims and set(coord.dims).issubset(dims)
            ]
            covered.update(coordinates)
            variable.attrs["coordinates"] = " ".join(map(str, coordinates))

        uncovered = [
            name for name in coords if name not in covered and name not in encoded.dims
        ]
        encoded.attrs["coordinates"] = " ".join(uncovered)

        # TODO: compare with the algorithm in xarray.conventions._encode_coordinates
        return encoded
