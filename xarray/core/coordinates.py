from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Hashable,
    Iterator,
    List,
    Mapping,
    Sequence,
)

import numpy as np
import pandas as pd

from xarray.core import formatting
from xarray.core.indexes import (
    Index,
    Indexes,
    PandasMultiIndex,
    assert_no_index_corrupted,
)
from xarray.core.merge import merge_coordinates_without_align, merge_coords
from xarray.core.types import T_DataArray
from xarray.core.utils import Frozen, ReprObject
from xarray.core.variable import Variable, as_variable, calculate_dimensions

if TYPE_CHECKING:
    from xarray.core.common import DataWithCoords
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset

# Used as the key corresponding to a DataArray's variable when converting
# arbitrary DataArray objects to datasets
_THIS_ARRAY = ReprObject("<this-array>")

# TODO: Remove when min python version >= 3.9:
GenericAlias = type(List[int])


class AbstractCoordinates(Mapping[Hashable, "T_DataArray"]):
    _data: DataWithCoords
    __slots__ = ("_data",)

    # TODO: Remove when min python version >= 3.9:
    __class_getitem__ = classmethod(GenericAlias)

    def __getitem__(self, key: Hashable) -> T_DataArray:
        raise NotImplementedError()

    @property
    def _names(self) -> set[Hashable]:
        raise NotImplementedError()

    @property
    def dims(self) -> Mapping[Hashable, int] | tuple[Hashable, ...]:
        raise NotImplementedError()

    @property
    def dtypes(self) -> Frozen[Hashable, np.dtype]:
        raise NotImplementedError()

    @property
    def indexes(self) -> Indexes[pd.Index]:
        return self._data.indexes

    @property
    def xindexes(self) -> Indexes[Index]:
        return self._data.xindexes

    @property
    def variables(self):
        raise NotImplementedError()

    def _update_coords(self, coords, indexes):
        raise NotImplementedError()

    def _maybe_drop_multiindex_coords(self, coords):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Hashable]:
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

    def to_dataset(self) -> Dataset:
        raise NotImplementedError()

    def to_index(self, ordered_dims: Sequence[Hashable] | None = None) -> pd.Index:
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
            return self._data.get_index(dim)
        else:
            indexes = [self._data.get_index(k) for k in ordered_dims]

            # compute the sizes of the repeat and tile for the cartesian product
            # (taken from pandas.core.reshape.util)
            index_lengths = np.fromiter(
                (len(index) for index in indexes), dtype=np.intp
            )
            cumprod_lengths = np.cumproduct(index_lengths)

            if cumprod_lengths[-1] == 0:
                # if any factor is empty, the cartesian product is empty
                repeat_counts = np.zeros_like(cumprod_lengths)

            else:
                # sizes of the repeats
                repeat_counts = cumprod_lengths[-1] / cumprod_lengths
            # sizes of the tiles
            tile_counts = np.roll(cumprod_lengths, 1)
            tile_counts[0] = 1

            # loop over the indexes
            # for each MultiIndex or Index compute the cartesian product of the codes

            code_list = []
            level_list = []
            names = []

            for i, index in enumerate(indexes):
                if isinstance(index, pd.MultiIndex):
                    codes, levels = index.codes, index.levels
                else:
                    code, level = pd.factorize(index)
                    codes = [code]
                    levels = [level]

                # compute the cartesian product
                code_list += [
                    np.tile(np.repeat(code, repeat_counts[i]), tile_counts[i])
                    for code in codes
                ]
                level_list += levels
                names += index.names

        return pd.MultiIndex(level_list, code_list, names=names)


class Coordinates(AbstractCoordinates):
    """Dictionary like container for Xarray coordinates (variables + indexes).

    This collection is a mapping of coordinate names to
    :py:class:`~xarray.DataArray` objects.

    It can be passed directly to the :py:class:`~xarray.Dataset` and
    :py:class:`~xarray.DataArray` constructors via their `coords` argument. This
    will add both the coordinates variables and their index.

    Coordinates are either:

    - returned via the :py:attr:`Dataset.coords` and :py:attr:`DataArray.coords`
      properties.
    - built from index objects (e.g., :py:meth:`Coordinates.from_pandas_multiindex`).
    - built directly from coordinate data and index objects (beware that no consistency
      check is done on those inputs).

    In the latter case, no default (pandas) index is created.

    Parameters
    ----------
    coords: dict-like
         Mapping of coordinate names to any objects that can be converted
         into a :py:class:`Variable`.
    indexes: dict-like
         Mapping of coordinate names to :py:class:`~indexes.Index` objects.

    """

    _data: DataWithCoords

    __slots__ = ("_data",)

    def __init__(
        self,
        coords: Mapping[Any, Any] | None = None,
        indexes: Mapping[Any, Index] | None = None,
    ):
        # When coordinates are constructed directly, an internal Dataset is
        # created so that it is compatible with the DatasetCoordinates and
        # DataArrayCoordinates classes serving as a proxy for the data.
        # TODO: refactor DataArray / Dataset so that Coordinates store the data.
        from xarray.core.dataset import Dataset

        if coords is None:
            variables = {}
        elif isinstance(coords, Coordinates):
            variables = dict(coords.variables)
        else:
            variables = {k: as_variable(v) for k, v in coords.items()}

        if indexes is None:
            indexes = {}
        else:
            indexes = dict(indexes)

        no_coord_index = set(indexes) - set(variables)
        if no_coord_index:
            raise ValueError(
                f"no coordinate variables found for these indexes: {no_coord_index}"
            )

        for k, idx in indexes.items():
            if not isinstance(idx, Index):
                raise TypeError(f"'{k}' is not an Xarray Index")

        # maybe convert to base variable
        for k, v in variables.items():
            if k not in indexes:
                variables[k] = v.to_base_variable()

        self._data = Dataset._construct_direct(
            coord_names=set(variables), variables=variables, indexes=indexes
        )

    @classmethod
    def from_pandas_multiindex(cls, midx: pd.MultiIndex, dim: str) -> Coordinates:
        """Wrap a pandas multi-index as Xarray coordinates (dimension + levels).

        The returned coordinates can be directly assigned to a
        :py:class:`~xarray.Dataset` or :py:class:`~xarray.DataArray` via the
        ``coords`` argument of their constructor.

        Parameters
        ----------
        midx : :py:class:`pandas.MultiIndex`
            Pandas multi-index object.
        dim : str
            Dimension name.

        Returns
        -------
        coords : Coordinates
            A collection of Xarray indexed coordinates created from the multi-index.

        """
        xr_idx = PandasMultiIndex(midx, dim)

        variables = xr_idx.create_variables()
        indexes = {k: xr_idx for k in variables}

        return cls(coords=variables, indexes=indexes)

    @property
    def _names(self) -> set[Hashable]:
        return self._data._coord_names

    @property
    def dims(self) -> Mapping[Hashable, int] | tuple[Hashable, ...]:
        return self._data.dims

    @property
    def dtypes(self) -> Frozen[Hashable, np.dtype]:
        """Mapping from coordinate names to dtypes.

        Cannot be modified directly.

        See Also
        --------
        Dataset.dtypes
        """
        return Frozen({n: v.dtype for n, v in self._data.variables.items()})

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        return self._data.variables

    def to_dataset(self) -> Dataset:
        """Convert these coordinates into a new Dataset"""
        return self._data.copy()

    def __getitem__(self, key: Hashable) -> DataArray:
        return self._data[key]

    def __delitem__(self, key: Hashable) -> None:
        # redirect to DatasetCoordinates.__delitem__
        del self._data.coords[key]

    def _update_coords(
        self, coords: dict[Hashable, Variable], indexes: Mapping[Any, Index]
    ) -> None:
        # redirect to DatasetCoordinates._update_coords
        self._data.coords._update_coords(coords, indexes)

    def _maybe_drop_multiindex_coords(self, coords: set[Hashable]) -> None:
        # redirect to DatasetCoordinates._maybe_drop_multiindex_coords
        self._data.coords._maybe_drop_multiindex_coords(coords)

    def _merge_raw(self, other, reflexive):
        """For use with binary arithmetic."""
        if other is None:
            variables = dict(self.variables)
            indexes = dict(self.xindexes)
        else:
            coord_list = [self, other] if not reflexive else [other, self]
            variables, indexes = merge_coordinates_without_align(coord_list)
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
                k: (v, None)
                for k, v in self.variables.items()
                if k not in self.xindexes
            }
            variables, indexes = merge_coordinates_without_align(
                [self, other], prioritized
            )
            yield
            self._update_coords(variables, indexes)

    def merge(self, other: Mapping[Any, Any] | None) -> Dataset:
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
        other : dict-like, optional
            A :py:class:`Coordinates` object or any mapping that can be turned
            into coordinates.

        Returns
        -------
        merged : Dataset
            A new Dataset with merged coordinates.
        """
        from xarray.core.dataset import Dataset

        if other is None:
            return self.to_dataset()

        if not isinstance(other, Coordinates):
            other = Dataset(coords=other).coords

        coords, indexes = merge_coordinates_without_align([self, other])
        coord_names = set(coords)
        return Dataset._construct_direct(
            variables=coords, coord_names=coord_names, indexes=indexes
        )

    def merge_coords(self, other: Mapping[Any, Any] | None = None) -> Coordinates:
        """Merge two sets of coordinates to create a new :py:class:`Coordinates`
        object.

        The method implements the logic used for joining coordinates in the
        result of a binary operation performed on xarray objects:

        - If two index coordinates conflict (are not equal), an exception is
          raised. You must align your data before passing it to this method.
        - If an index coordinate and a non-index coordinate conflict, the non-
          index coordinate is dropped.
        - If two non-index coordinates conflict, both are dropped.

        Parameters
        ----------
        other : dict-like, optional
            A :py:class:`Coordinates` object or any mapping that can be turned
            into coordinates.

        Returns
        -------
        merged : Coordinates
            A new Coordinates object with merged coordinates.
        """
        from xarray.core.dataset import Dataset

        if not isinstance(other, Coordinates):
            other = Dataset(coords=other).coords

        return self.merge(other).coords

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.update({key: value})

    def update(self, other: Mapping[Any, Any]) -> None:
        other_obj: Dataset | Mapping[Hashable, Variable]

        if isinstance(other, Coordinates):
            # special case: do not create default indexes
            # converting to Dataset will allow reusing existing indexes
            # when merging coordinates below
            other_obj = other.to_dataset()
            create_default_indexes = False
        else:
            other_obj = getattr(other, "variables", other)
            create_default_indexes = True

        self._maybe_drop_multiindex_coords(set(other_obj))

        coords, indexes = merge_coords(
            [self.variables, other_obj],
            priority_arg=1,
            indexes=self.xindexes,
            create_default_indexes=create_default_indexes,
        )

        self._update_coords(coords, indexes)

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython."""
        return self._data._ipython_key_completions_()


class DatasetCoordinates(Coordinates):
    """Dictionary like container for Dataset coordinates (variables + indexes).

    This collection can be passed directly to the :py:class:`~xarray.Dataset`
    and :py:class:`~xarray.DataArray` constructors via their `coords` argument.
    This will add both the coordinates variables and their index.
    """

    _data: Dataset

    __slots__ = ("_data",)

    def __init__(self, dataset: Dataset):
        self._data = dataset

    @property
    def _names(self) -> set[Hashable]:
        return self._data._coord_names

    @property
    def dims(self) -> Mapping[Hashable, int]:
        return self._data.dims

    @property
    def dtypes(self) -> Frozen[Hashable, np.dtype]:
        """Mapping from coordinate names to dtypes.

        Cannot be modified directly, but is updated when adding new variables.

        See Also
        --------
        Dataset.dtypes
        """
        return Frozen(
            {
                n: v.dtype
                for n, v in self._data._variables.items()
                if n in self._data._coord_names
            }
        )

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        return Frozen(
            {k: v for k, v in self._data.variables.items() if k in self._names}
        )

    def __getitem__(self, key: Hashable) -> DataArray:
        if key in self._data.data_vars:
            raise KeyError(key)
        return self._data[key]

    def to_dataset(self) -> Dataset:
        """Convert these coordinates into a new Dataset"""

        names = [name for name in self._data._variables if name in self._names]
        return self._data._copy_listed(names)

    def _update_coords(
        self, coords: dict[Hashable, Variable], indexes: Mapping[Any, Index]
    ) -> None:
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
        original_indexes = dict(self._data.xindexes)
        original_indexes.update(indexes)
        self._data._indexes = original_indexes

    def _maybe_drop_multiindex_coords(self, coords: set[Hashable]) -> None:
        """Drops variables in coords, and any associated variables as well."""
        assert self._data.xindexes is not None
        variables, indexes = drop_coords(
            coords, self._data._variables, self._data.xindexes
        )
        self._data._coord_names.intersection_update(variables)
        self._data._variables = variables
        self._data._indexes = indexes

    def __delitem__(self, key: Hashable) -> None:
        if key in self:
            del self._data[key]
        else:
            raise KeyError(f"{key!r} is not a coordinate variable.")

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython."""
        return [
            key
            for key in self._data._ipython_key_completions_()
            if key not in self._data.data_vars
        ]


class DataArrayCoordinates(Coordinates, Generic[T_DataArray]):
    """Dictionary like container for DataArray coordinates (variables + indexes).

    This collection can be passed directly to the :py:class:`~xarray.Dataset`
    and :py:class:`~xarray.DataArray` constructors via their `coords` argument.
    This will add both the coordinates variables and their index.
    """

    _data: T_DataArray

    __slots__ = ("_data",)

    def __init__(self, dataarray: T_DataArray) -> None:
        self._data = dataarray

    @property
    def dims(self) -> tuple[Hashable, ...]:
        return self._data.dims

    @property
    def dtypes(self) -> Frozen[Hashable, np.dtype]:
        """Mapping from coordinate names to dtypes.

        Cannot be modified directly, but is updated when adding new variables.

        See Also
        --------
        DataArray.dtype
        """
        return Frozen({n: v.dtype for n, v in self._data._coords.items()})

    @property
    def _names(self) -> set[Hashable]:
        return set(self._data._coords)

    def __getitem__(self, key: Hashable) -> T_DataArray:
        return self._data._getitem_coord(key)

    def _update_coords(
        self, coords: dict[Hashable, Variable], indexes: Mapping[Any, Index]
    ) -> None:
        coords_plus_data = coords.copy()
        coords_plus_data[_THIS_ARRAY] = self._data.variable
        dims = calculate_dimensions(coords_plus_data)
        if not set(dims) <= set(self.dims):
            raise ValueError(
                "cannot add coordinates with new dimensions to a DataArray"
            )
        self._data._coords = coords

        # TODO(shoyer): once ._indexes is always populated by a dict, modify
        # it to update inplace instead.
        original_indexes = dict(self._data.xindexes)
        original_indexes.update(indexes)
        self._data._indexes = original_indexes

    def _maybe_drop_multiindex_coords(self, coords: set[Hashable]) -> None:
        """Drops variables in coords, and any associated variables as well."""
        variables, indexes = drop_coords(
            coords, self._data._coords, self._data.xindexes
        )
        self._data._coords = variables
        self._data._indexes = indexes

    @property
    def variables(self):
        return Frozen(self._data._coords)

    def to_dataset(self) -> Dataset:
        from xarray.core.dataset import Dataset

        coords = {k: v.copy(deep=False) for k, v in self._data._coords.items()}
        indexes = dict(self._data.xindexes)
        return Dataset._construct_direct(coords, set(coords), indexes=indexes)

    def __delitem__(self, key: Hashable) -> None:
        if key not in self:
            raise KeyError(f"{key!r} is not a coordinate variable.")
        assert_no_index_corrupted(self._data.xindexes, {key})

        del self._data._coords[key]
        if self._data._indexes is not None and key in self._data._indexes:
            del self._data._indexes[key]

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython."""
        return self._data._ipython_key_completions_()


def drop_coords(
    coords_to_drop: set[Hashable], variables, indexes: Indexes
) -> tuple[dict, dict]:
    """Drop index variables associated with variables in coords_to_drop."""
    # Only warn when we're dropping the dimension with the multi-indexed coordinate
    # If asked to drop a subset of the levels in a multi-index, we raise an error
    # later but skip the warning here.
    new_variables = dict(variables.copy())
    new_indexes = dict(indexes.copy())
    for key in coords_to_drop & set(indexes):
        maybe_midx = indexes[key]
        idx_coord_names = set(indexes.get_all_coords(key))
        if (
            isinstance(maybe_midx, PandasMultiIndex)
            and key == maybe_midx.dim
            and (idx_coord_names - coords_to_drop)
        ):
            warnings.warn(
                f"Updating MultiIndexed coordinate {key!r} would corrupt indices for "
                f"other variables: {list(maybe_midx.index.names)!r}. "
                f"This will raise an error in the future. Use `.drop_vars({idx_coord_names!r})` before "
                "assigning new coordinate values.",
                FutureWarning,
                stacklevel=4,
            )
            for k in idx_coord_names:
                del new_variables[k]
                del new_indexes[k]
    return new_variables, new_indexes


def assert_coordinate_consistent(
    obj: T_DataArray | Dataset, coords: Mapping[Any, Variable]
) -> None:
    """Make sure the dimension coordinate of obj is consistent with coords.

    obj: DataArray or Dataset
    coords: Dict-like of variables
    """
    for k in obj.dims:
        # make sure there are no conflict in dimension coordinates
        if k in coords and k in obj.coords and not coords[k].equals(obj[k].variable):
            raise IndexError(
                f"dimension coordinate {k!r} conflicts between "
                f"indexed and indexing objects:\n{obj[k]}\nvs.\n{coords[k]}"
            )
