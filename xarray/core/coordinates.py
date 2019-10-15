from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterator,
    Mapping,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import pandas as pd

from . import formatting, indexing
from .indexes import Indexes
from .merge import merge_coordinates_without_align, merge_coords
from .utils import Frozen, ReprObject, either_dict_or_kwargs
from .variable import Variable

if TYPE_CHECKING:
    from .dataarray import DataArray
    from .dataset import Dataset

# Used as the key corresponding to a DataArray's variable when converting
# arbitrary DataArray objects to datasets
_THIS_ARRAY = ReprObject("<this-array>")


class Coordinates(Mapping[Hashable, "DataArray"]):
    __slots__ = ()

    def __getitem__(self, key: Hashable) -> "DataArray":
        raise NotImplementedError()

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.update({key: value})

    @property
    def _names(self) -> Set[Hashable]:
        raise NotImplementedError()

    @property
    def dims(self) -> Union[Mapping[Hashable, int], Tuple[Hashable, ...]]:
        raise NotImplementedError()

    @property
    def indexes(self) -> Indexes:
        return self._data.indexes  # type: ignore

    @property
    def variables(self):
        raise NotImplementedError()

    def _update_coords(self, coords, indexes):
        raise NotImplementedError()

    def __iter__(self) -> Iterator["Hashable"]:
        # needs to be in the same order as the dataset variables
        for k in self.variables:
            if k in self._names:
                yield k

    def __len__(self) -> int:
        return len(self._names)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._names

    def __repr__(self) -> str:
        return formatting.coords_repr(self)

    def to_dataset(self) -> "Dataset":
        raise NotImplementedError()

    def to_index(self, ordered_dims: Sequence[Hashable] = None) -> pd.Index:
        """Convert all index coordinates into a :py:class:`pandas.Index`.

        Parameters
        ----------
        ordered_dims : sequence of hashable, optional
            Possibly reordered version of this object's dimensions indicating
            the order in which dimensions should appear on the result.

        Returns
        -------
        pandas.Index
            Index subclass corresponding to the outer-product of all dimension
            coordinates. This will be a MultiIndex if this object is has more
            than more dimension.
        """
        if ordered_dims is None:
            ordered_dims = list(self.dims)
        elif set(ordered_dims) != set(self.dims):
            raise ValueError(
                "ordered_dims must match dims, but does not: "
                "{} vs {}".format(ordered_dims, self.dims)
            )

        if len(ordered_dims) == 0:
            raise ValueError("no valid index for a 0-dimensional object")
        elif len(ordered_dims) == 1:
            (dim,) = ordered_dims
            return self._data.get_index(dim)  # type: ignore
        else:
            indexes = [self._data.get_index(k) for k in ordered_dims]  # type: ignore
            names = list(ordered_dims)
            return pd.MultiIndex.from_product(indexes, names=names)

    def update(self, other: Mapping[Hashable, Any]) -> None:
        other_vars = getattr(other, "variables", other)
        coords, indexes = merge_coords(
            [self.variables, other_vars], priority_arg=1, indexes=self.indexes
        )
        self._update_coords(coords, indexes)

    def _merge_raw(self, other):
        """For use with binary arithmetic."""
        if other is None:
            variables = dict(self.variables)
            indexes = dict(self.indexes)
        else:
            variables, indexes = merge_coordinates_without_align([self, other])
        return variables, indexes

    @contextmanager
    def _merge_inplace(self, other):
        """For use with in-place binary arithmetic."""
        if other is None:
            yield
        else:
            # don't include indexes in prioritized, because we didn't align
            # first and we want indexes to be checked
            prioritized = {
                k: (v, None) for k, v in self.variables.items() if k not in self.indexes
            }
            variables, indexes = merge_coordinates_without_align(
                [self, other], prioritized
            )
            yield
            self._update_coords(variables, indexes)

    def merge(self, other: "Coordinates") -> "Dataset":
        """Merge two sets of coordinates to create a new Dataset

        The method implements the logic used for joining coordinates in the
        result of a binary operation performed on xarray objects:

        - If two index coordinates conflict (are not equal), an exception is
          raised. You must align your data before passing it to this method.
        - If an index coordinate and a non-index coordinate conflict, the non-
          index coordinate is dropped.
        - If two non-index coordinates conflict, both are dropped.

        Parameters
        ----------
        other : DatasetCoordinates or DataArrayCoordinates
            The coordinates from another dataset or data array.

        Returns
        -------
        merged : Dataset
            A new Dataset with merged coordinates.
        """
        from .dataset import Dataset

        if other is None:
            return self.to_dataset()

        if not isinstance(other, Coordinates):
            other = Dataset(coords=other).coords

        coords, indexes = merge_coordinates_without_align([self, other])
        coord_names = set(coords)
        merged = Dataset._construct_direct(
            variables=coords, coord_names=coord_names, indexes=indexes
        )
        return merged


class DatasetCoordinates(Coordinates):
    """Dictionary like container for Dataset coordinates.

    Essentially an immutable dictionary with keys given by the array's
    dimensions and the values given by the corresponding xarray.Coordinate
    objects.
    """

    __slots__ = ("_data",)

    def __init__(self, dataset: "Dataset"):
        self._data = dataset

    @property
    def _names(self) -> Set[Hashable]:
        return self._data._coord_names

    @property
    def dims(self) -> Mapping[Hashable, int]:
        return self._data.dims

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        return Frozen(
            {k: v for k, v in self._data.variables.items() if k in self._names}
        )

    def __getitem__(self, key: Hashable) -> "DataArray":
        if key in self._data.data_vars:
            raise KeyError(key)
        return cast("DataArray", self._data[key])

    def to_dataset(self) -> "Dataset":
        """Convert these coordinates into a new Dataset
        """
        return self._data._copy_listed(self._names)

    def _update_coords(
        self, coords: Dict[Hashable, Variable], indexes: Mapping[Hashable, pd.Index]
    ) -> None:
        from .dataset import calculate_dimensions

        variables = self._data._variables.copy()
        variables.update(coords)

        # check for inconsistent state *before* modifying anything in-place
        dims = calculate_dimensions(variables)
        new_coord_names = set(coords)
        for dim, size in dims.items():
            if dim in variables:
                new_coord_names.add(dim)

        self._data._variables = variables
        self._data._coord_names.update(new_coord_names)
        self._data._dims = dims

        # TODO(shoyer): once ._indexes is always populated by a dict, modify
        # it to update inplace instead.
        original_indexes = dict(self._data.indexes)
        original_indexes.update(indexes)
        self._data._indexes = original_indexes

    def __delitem__(self, key: Hashable) -> None:
        if key in self:
            del self._data[key]
        else:
            raise KeyError(key)

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython. """
        return [
            key
            for key in self._data._ipython_key_completions_()
            if key not in self._data.data_vars
        ]


class DataArrayCoordinates(Coordinates):
    """Dictionary like container for DataArray coordinates.

    Essentially a dict with keys given by the array's
    dimensions and the values given by corresponding DataArray objects.
    """

    __slots__ = ("_data",)

    def __init__(self, dataarray: "DataArray"):
        self._data = dataarray

    @property
    def dims(self) -> Tuple[Hashable, ...]:
        return self._data.dims

    @property
    def _names(self) -> Set[Hashable]:
        return set(self._data._coords)

    def __getitem__(self, key: Hashable) -> "DataArray":
        return self._data._getitem_coord(key)

    def _update_coords(
        self, coords: Dict[Hashable, Variable], indexes: Mapping[Hashable, pd.Index]
    ) -> None:
        from .dataset import calculate_dimensions

        coords_plus_data = coords.copy()
        coords_plus_data[_THIS_ARRAY] = self._data.variable
        dims = calculate_dimensions(coords_plus_data)
        if not set(dims) <= set(self.dims):
            raise ValueError(
                "cannot add coordinates with new dimensions to " "a DataArray"
            )
        self._data._coords = coords

        # TODO(shoyer): once ._indexes is always populated by a dict, modify
        # it to update inplace instead.
        original_indexes = dict(self._data.indexes)
        original_indexes.update(indexes)
        self._data._indexes = original_indexes

    @property
    def variables(self):
        return Frozen(self._data._coords)

    def to_dataset(self) -> "Dataset":
        from .dataset import Dataset

        coords = {k: v.copy(deep=False) for k, v in self._data._coords.items()}
        return Dataset._from_vars_and_coord_names(coords, set(coords))

    def __delitem__(self, key: Hashable) -> None:
        del self._data._coords[key]

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython. """
        return self._data._ipython_key_completions_()


class LevelCoordinatesSource(Mapping[Hashable, Any]):
    """Iterator for MultiIndex level coordinates.

    Used for attribute style lookup with AttrAccessMixin. Not returned directly
    by any public methods.
    """

    __slots__ = ("_data",)

    def __init__(self, data_object: "Union[DataArray, Dataset]"):
        self._data = data_object

    def __getitem__(self, key):
        # not necessary -- everything here can already be found in coords.
        raise KeyError()

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._data._level_coords)

    def __len__(self) -> int:
        return len(self._data._level_coords)


def assert_coordinate_consistent(
    obj: Union["DataArray", "Dataset"], coords: Mapping[Hashable, Variable]
) -> None:
    """Make sure the dimension coordinate of obj is consistent with coords.

    obj: DataArray or Dataset
    coords: Dict-like of variables
    """
    for k in obj.dims:
        # make sure there are no conflict in dimension coordinates
        if k in coords and k in obj.coords:
            if not coords[k].equals(obj[k].variable):
                raise IndexError(
                    "dimension coordinate {!r} conflicts between "
                    "indexed and indexing objects:\n{}\nvs.\n{}".format(
                        k, obj[k], coords[k]
                    )
                )


def remap_label_indexers(
    obj: Union["DataArray", "Dataset"],
    indexers: Mapping[Hashable, Any] = None,
    method: str = None,
    tolerance=None,
    **indexers_kwargs: Any
) -> Tuple[dict, dict]:  # TODO more precise return type after annotations in indexing
    """Remap indexers from obj.coords.
    If indexer is an instance of DataArray and it has coordinate, then this coordinate
    will be attached to pos_indexers.

    Returns
    -------
    pos_indexers: Same type of indexers.
        np.ndarray or Variable or DataArray
    new_indexes: mapping of new dimensional-coordinate.
    """
    from .dataarray import DataArray

    indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "remap_label_indexers")

    v_indexers = {
        k: v.variable.data if isinstance(v, DataArray) else v
        for k, v in indexers.items()
    }

    pos_indexers, new_indexes = indexing.remap_label_indexers(
        obj, v_indexers, method=method, tolerance=tolerance
    )
    # attach indexer's coordinate to pos_indexers
    for k, v in indexers.items():
        if isinstance(v, Variable):
            pos_indexers[k] = Variable(v.dims, pos_indexers[k])
        elif isinstance(v, DataArray):
            # drop coordinates found in indexers since .sel() already
            # ensures alignments
            coords = {k: var for k, var in v._coords.items() if k not in indexers}
            pos_indexers[k] = DataArray(pos_indexers[k], coords=coords, dims=v.dims)
    return pos_indexers, new_indexes
