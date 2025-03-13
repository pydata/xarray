from collections.abc import Hashable, Iterable, Mapping
from functools import lru_cache
from typing import Any

import affine
import numpy as np
import pyproj
from affine import Affine

import xarray as xr
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import CoordinateTransform, CoordinateTransformIndex
from xarray.core.types import Self
from xarray.core.variable import Variable

# https://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objects
transformer_from_crs = lru_cache(pyproj.Transformer.from_crs)


def with_geoindex(ds: Dataset, dims=("x", "y")) -> Dataset:
    """simple helper function."""

    index = GeoTransformIndex.from_grid_mapping(
        ds.cf["grid_mapping"], dim_sizes={k: ds.sizes[k] for k in dims}
    )
    newcoords = xr.Coordinates.from_xindex(index)
    # TODO: assign coordinates to (xc, yc)
    newds = ds.assign_coords(newcoords)  # TODO: got confused with set_xindex
    return newds


class Affine2DCoordinateTransform(CoordinateTransform):
    """Affine 2D coordinate transform."""

    # Copied from benbovy's https://github.com/pydata/xarray/pull/9543

    affine: affine.Affine
    xy_dims = tuple[str]

    def __init__(
        self,
        affine: affine.Affine,
        coord_names: Iterable[Hashable],
        dim_size: Mapping[str, int],
        dtype: Any = np.dtype(np.float64),
    ):
        # two dimensions
        assert len(coord_names) == 2
        assert len(dim_size) == 2

        super().__init__(coord_names, dim_size, dtype=dtype)
        self.affine = affine

        # array dimensions in reverse order (y = rows, x = cols)
        self.xy_dims = tuple(self.dims)
        self.dims = (self.dims[1], self.dims[0])

    def forward(self, dim_positions):
        positions = [dim_positions[dim] for dim in self.xy_dims]
        x_labels, y_labels = self.affine * tuple(positions)

        results = {}
        for name, labels in zip(self.coord_names, [x_labels, y_labels], strict=False):
            results[name] = labels

        return results

    def reverse(self, coord_labels):
        labels = [coord_labels[name] for name in self.coord_names]
        x_positions, y_positions = ~self.affine * tuple(labels)

        results = {}
        for dim, positions in zip(
            self.xy_dims, [x_positions, y_positions], strict=False
        ):
            results[dim] = positions

        return results

    def equals(self, other):
        return self.affine == other.affine and self.dim_size == other.dim_size


class GeoTransformIndex(CoordinateTransformIndex):
    def __init__(self, *, transform, grid_mapping: DataArray, coord_names) -> Self:
        self.gm_name = grid_mapping.name
        self.crs = pyproj.CRS.from_cf(grid_mapping.attrs)
        self.xcoord, self.ycoord = coord_names
        return super().__init__(transform=transform)

    def create_variables(self, variables=None):
        res = super().create_variables(variables)
        # assign the spatial_ref since the GeoTransform is updated when isel is called.
        res[self.gm_name] = Variable(dims=(), data=0, attrs=self.crs.to_cf())
        res[self.gm_name].attrs["GeoTransform"] = " ".join(
            map(str, self.transform.affine.to_gdal())
        )
        return res

    @classmethod
    def from_grid_mapping(
        cls,
        grid_mapping: Variable,
        *,
        dim_sizes: dict[str, int],
        coord_names=("xc", "yc"),
    ) -> Self:
        geotransform = np.fromstring(
            grid_mapping.attrs["GeoTransform"], sep=" "
        ).tolist()
        fwd = Affine.from_gdal(*geotransform[:6])
        centers = fwd * fwd.translation(0.5, 0.5)

        xtransform = Affine2DCoordinateTransform(
            affine=centers,
            coord_names=coord_names,
            dim_size=dim_sizes,
        )
        return cls(
            transform=xtransform, grid_mapping=grid_mapping, coord_names=coord_names
        )

    def isel(self, indexers) -> Self:
        # FIXME: now this gets called after sel and breaks.
        indexers.setdefault("x", slice(None))
        indexers.setdefault("y", slice(None))

        assert isinstance(indexers["x"], slice)
        assert isinstance(indexers["y"], slice)

        offsets = tuple((indexers[dim].start or 0) for dim in ("x", "y"))
        scales = tuple(indexers[dim].step or 1 for dim in ("x", "y"))
        fwd = self.transform.affine
        new_transform = fwd * fwd.scale(*scales) * fwd.translation(*offsets)
        xtransform = Affine2DCoordinateTransform(
            affine=new_transform,
            coord_names=self.transform.coord_names,
            dim_size=self.transform.dim_size,
        )
        new_spatial_ref = DataArray(0, attrs=self.crs.to_cf())
        new_spatial_ref.attrs["GeoTransform"] = " ".join(
            map(str, new_transform.to_gdal())
        )
        new_index = type(self)(
            transform=xtransform,
            grid_mapping=new_spatial_ref,
            coord_names=self.transform.coord_names,
        )
        # TODO: we need to return the new spatial_ref, since the GeoTransform may have changed.
        return new_index

    def sel(self, labels: dict[Any, Any], method="nearest", tolerance=None, crs=None):
        """
        This implements a selection API like OGC EDR where queries are always X,Y
        interpreted using the provided CRS.
        """
        from xarray.core.variable import broadcast_variables

        # FIXME: Assume default CRS is self.crs, but we could alternatively assume EPSG:4326
        qcrs = self.crs if crs is None else pyproj.CRS.from_user_input(crs)
        transformer = transformer_from_crs(
            crs_from=qcrs, crs_to=self.crs, always_xy=True
        )

        X = labels.get(self.xcoord, None)
        Y = labels.get(self.ycoord, None)

        # TODO: handle, X, Y being None.
        assert X is not None
        assert Y is not None

        transformed_labels = dict(
            zip((self.xcoord, self.ycoord), transformer.transform(X, Y), strict=False)
        )
        # Transformer always returns numpy?
        for coord, raw, trans in zip(
            labels.keys(), labels.values(), transformed_labels.values(), strict=False
        ):
            if isinstance(raw, Variable):
                transformed_labels[coord] = raw.copy(data=trans)
            elif isinstance(raw, DataArray):
                transformed_labels[coord] = raw.variable.copy(data=trans)
            elif isinstance(raw, (np.ndarray | list | tuple)):
                transformed_labels[coord] = Variable(dims=(coord,), data=trans)
            else:
                assert np.isscalar(trans)
                transformed_labels[coord] = Variable(dims=(), data=trans)
        transformed_labels = dict(
            zip(
                labels.keys(),
                broadcast_variables(*tuple(transformed_labels.values())),
                strict=False,
            )
        )
        res = super().sel(labels=transformed_labels, method=method, tolerance=tolerance)
        # TODO: create a new spatial_ref here without GeoTransform and return that.
        return res
