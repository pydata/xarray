from __future__ import annotations

from typing import Any, Hashable, Mapping, TypeVar

import numpy as np

from ..core.indexes import Index, IndexVars, PandasIndex
from ..core.indexing import IndexSelResult, merge_sel_results
from ..core.utils import Frozen
from ..core.variable import Variable

T_MultiPandasIndex = TypeVar("T_MultiPandasIndex", bound="MultiPandasIndex")


class MultiPandasIndex(Index):
    """Helper class to implement meta-indexes encapsulating
    one or more (single) pandas indexes.

    Each pandas index must relate to a separate dimension.

    This class shoudn't be instantiated directly.

    """

    indexes: Frozen[Hashable, PandasIndex]
    dims: Frozen[Hashable, int]

    __slots__ = ("indexes", "dims")

    def __init__(self, indexes: Mapping[Hashable, PandasIndex]):
        dims = {idx.dim: idx.index.size for idx in indexes.values()}

        seen = set()
        dup_dims = []
        for d in dims:
            if d in seen:
                dup_dims.append(d)
            else:
                seen.add(d)

        if dup_dims:
            raise ValueError(
                f"cannot create a {self.__class__.__name__} from coordinates "
                f"sharing common dimension(s): {dup_dims}"
            )

        self.indexes = Frozen(indexes)
        self.dims = Frozen(dims)

    @classmethod
    def from_variables(
        cls: type[T_MultiPandasIndex], variables: Mapping[Any, Variable], options
    ) -> T_MultiPandasIndex:
        indexes = {
            k: PandasIndex.from_variables({k: v}, options={})
            for k, v in variables.items()
        }

        return cls(indexes)

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> IndexVars:

        idx_variables = {}

        for idx in self.indexes.values():
            idx_variables.update(idx.create_variables(variables))

        return idx_variables

    def isel(
        self: T_MultiPandasIndex,
        indexers: Mapping[Any, int | slice | np.ndarray | Variable],
    ) -> T_MultiPandasIndex | PandasIndex | None:
        new_indexes = {}

        for k, idx in self.indexes.items():
            if k in indexers:
                new_idx = idx.isel({k: indexers[k]})
                if new_idx is not None:
                    new_indexes[k] = new_idx
            else:
                new_indexes[k] = idx

        #
        # How should we deal with dropped index(es) (scalar selection)?
        # - drop the whole index?
        # - always return a MultiPandasIndex with remaining index(es)?
        # - return either a MultiPandasIndex or a PandasIndex?
        #

        if not len(new_indexes):
            return None
        elif len(new_indexes) == 1:
            return next(iter(new_indexes.values()))
        else:
            return type(self)(new_indexes)

    def sel(self, labels: dict[Any, Any], **kwargs) -> IndexSelResult:
        results: list[IndexSelResult] = []

        for k, idx in self.indexes.items():
            if k in labels:
                results.append(idx.sel({k: labels[k]}, **kwargs))

        return merge_sel_results(results)

    def _get_unmatched_names(
        self: T_MultiPandasIndex, other: T_MultiPandasIndex
    ) -> set:
        return set(self.indexes).symmetric_difference(other.indexes)

    def equals(self: T_MultiPandasIndex, other: T_MultiPandasIndex) -> bool:
        # We probably don't need to check for matching coordinate names
        # as this is already done during alignment when finding matching indexes.
        # This may change in the future, though.
        # see https://github.com/pydata/xarray/issues/7002
        if self._get_unmatched_names(other):
            return False
        else:
            return all(
                [idx.equals(other.indexes[k]) for k, idx in self.indexes.items()]
            )

    def join(
        self: T_MultiPandasIndex, other: T_MultiPandasIndex, how: str = "inner"
    ) -> T_MultiPandasIndex:
        new_indexes = {}

        for k, idx in self.indexes.items():
            new_indexes[k] = idx.join(other.indexes[k], how=how)

        return type(self)(new_indexes)

    def reindex_like(
        self: T_MultiPandasIndex, other: T_MultiPandasIndex
    ) -> dict[Hashable, Any]:
        dim_indexers = {}

        for k, idx in self.indexes.items():
            dim_indexers.update(idx.reindex_like(other.indexes[k]))

        return dim_indexers

    def roll(self: T_MultiPandasIndex, shifts: Mapping[Any, int]) -> T_MultiPandasIndex:
        new_indexes = {}

        for k, idx in self.indexes.items():
            if k in shifts:
                new_indexes[k] = idx.roll({k: shifts[k]})
            else:
                new_indexes[k] = idx

        return type(self)(new_indexes)

    def rename(
        self: T_MultiPandasIndex,
        name_dict: Mapping[Any, Hashable],
        dims_dict: Mapping[Any, Hashable],
    ) -> T_MultiPandasIndex:
        new_indexes = {}

        for k, idx in self.indexes.items():
            new_indexes[k] = idx.rename(name_dict, dims_dict)

        return type(self)(new_indexes)

    def copy(self: T_MultiPandasIndex, deep: bool = True) -> T_MultiPandasIndex:
        new_indexes = {}

        for k, idx in self.indexes.items():
            new_indexes[k] = idx.copy(deep=deep)

        return type(self)(new_indexes)
