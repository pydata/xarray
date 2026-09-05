import operator
from functools import reduce
from typing import ClassVar

import xarray as xr
from xarray.coding.core import CoderKind


class RangeIndexCoder:
    kind: ClassVar[CoderKind] = "dataset"

    def decode(self, obj: xr.Dataset) -> xr.Dataset:
        def _decode_index(coord, params):
            return xr.indexes.RangeIndex.arange(**params, coord_name=coord)

        encoded_ranges = obj.attrs.get("ranges")
        if encoded_ranges is None:
            return obj

        indexes = [
            _decode_index(name, params) for name, params in encoded_ranges.items()
        ]
        coords = reduce(
            operator.or_, (xr.Coordinates.from_xindex(index) for index in indexes)
        )

        decoded = obj.assign_coords(coords)
        del decoded.attrs["ranges"]

        return decoded

    def encode(self, obj: xr.Dataset) -> xr.Dataset:
        def _encode_index(index):
            return {
                "start": index.start,
                "stop": index.stop,
                "step": index.step,
                "dim": index.dim,
            }

        encoded_ranges = {
            name: _encode_index(index)
            for name, index in obj.xindexes.items()
            if isinstance(index, xr.indexes.RangeIndex)
        }
        encoded = obj.drop_indexes(list(encoded_ranges)).drop_vars(list(encoded_ranges))
        encoded.attrs["ranges"] = encoded_ranges
        return encoded
