import random
from collections.abc import Hashable

import pytest

from xarray import Dataset
from xarray.indexes import PandasMultiIndex
from xarray.testing import _assert_internal_invariants

pytest.importorskip("hypothesis")

import hypothesis.strategies as st
from hypothesis import assume, note, settings
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    consumes,
    invariant,
    multiple,
    precondition,
    rule,
)

import xarray.testing.strategies as xrst


def get_not_multiindex_dims(ds: Dataset) -> tuple[Hashable]:
    dims = ds.dims
    mindexes = [
        name
        for name, index in ds.xindexes.items()
        if isinstance(index, PandasMultiIndex)
    ]
    return tuple(set(dims) - set(mindexes))


def get_multiindex_dims(ds: Dataset) -> list[Hashable]:
    mindexes = [
        name
        for name, index in ds.xindexes.items()
        if isinstance(index, PandasMultiIndex)
    ]
    return mindexes


def get_dimension_coordinates(ds: Dataset) -> tuple[Hashable]:
    return tuple(set(ds.dims) & set(ds._variables))


@st.composite
def unique(draw, strategy):
    # https://stackoverflow.com/questions/73737073/create-hypothesis-strategy-that-returns-unique-values
    seen = draw(st.shared(st.builds(set), key="key-for-unique-elems"))
    return draw(
        strategy.filter(lambda x: x not in seen).map(lambda x: seen.add(x) or x)
    )


random.seed(123456)

# Share to ensure we get unique names on each draw?
UNIQUE_NAME = unique(strategy=xrst.names())
DIM_NAME = xrst.dimension_names(name_strategy=UNIQUE_NAME, min_dims=1, max_dims=1)


class DatasetStateMachine(RuleBasedStateMachine):
    indexed_dims = Bundle("indexed_dims")
    multi_indexed_dims = Bundle("multi_indexed_dims")

    def __init__(self):
        super().__init__()
        self.dataset = Dataset()
        self.check_default_indexes = True

    @rule(var=xrst.index_variables(dims=DIM_NAME), target=indexed_dims)
    def add_dim_coord(self, var):
        (name,) = var.dims
        # dim coord
        self.dataset[name] = var
        # non-dim coord of same size; this allows renaming
        self.dataset[name + "_"] = var
        return name

    @rule(dim=st.one_of(consumes(indexed_dims), consumes(multi_indexed_dims)))
    def reset_index(self, dim):
        self.check_default_indexes = False
        note(f"> resetting {dim}")
        self.dataset = self.dataset.reset_index(dim)

    @rule(
        newname=UNIQUE_NAME,
        oldnames=st.lists(consumes(indexed_dims), min_size=1),
        target=multi_indexed_dims,
    )
    def stack(self, newname, oldnames):
        note(f"> stacking {oldnames} as {newname}")
        self.dataset = self.dataset.stack({newname: oldnames})
        return newname

    # TODO: add st.none() to dim
    @rule(dim=consumes(multi_indexed_dims), target=indexed_dims)
    def unstack(self, dim):
        if dim is not None:
            pd_index = self.dataset.xindexes[dim].index
            assume(pd_index.is_unique)
        note(f"> unstacking {dim}")
        self.dataset = self.dataset.unstack(dim)
        if dim is not None:
            return multiple(pd_index.names)
        else:
            # TODO Fix this when adding st.none()
            return multiple()

    @rule(newname=UNIQUE_NAME, oldname=consumes(indexed_dims))
    def rename_vars(self, newname, oldname):
        # benbovy: "skip the default indexes invariant test when the name of an
        # existing dimension coordinate is passed as input kwarg or dict key
        # to .rename_vars()."
        self.check_default_indexes = False
        note(f"> renaming {oldname} to {newname}")
        self.dataset = self.dataset.rename_vars({oldname: newname})

    @rule(data=st.data(), dim=consumes(indexed_dims), target=indexed_dims)
    @precondition(lambda self: len(self.dataset._variables) >= 2)
    def swap_dims(self, data, dim):
        ds = self.dataset
        choices = [name for name, var in ds._variables.items() if var.dims == (dim,)]
        # TODO: is there a better way to skip if choices == []
        # note(choices)
        # if not choices:
        #     return dim
        # Can only swap to a variable with the same dim
        to = data.draw(st.sampled_from(choices))
        # TODO: swapping a dimension to itself
        # TODO: swapping from Index to a MultiIndex level
        # TODO: swapping from MultiIndex to a level of the same MultiIndex
        note(f"> swapping {dim} to {to}")
        self.dataset = ds.swap_dims({dim: to})
        return to

    # TODO: enable when we have serializable attrs only
    # @rule()
    # def roundtrip_zarr(self):
    #     if not has_zarr:
    #         return
    #     expected = self.dataset
    #     with create_tmp_file(allow_cleanup_failure=ON_WINDOWS) as path:
    #         self.dataset.to_zarr(path + ".zarr")
    #         with xr.open_dataset(path + ".zarr", engine="zarr") as ds:
    #             assert_identical(expected, ds)

    @invariant()
    def assert_invariants(self):
        # note(f"> ===\n\n {self.dataset!r} \n===\n\n")
        _assert_internal_invariants(self.dataset, self.check_default_indexes)


DatasetStateMachine.TestCase.settings = settings(max_examples=1000, deadline=None)
DatasetTest = DatasetStateMachine.TestCase
