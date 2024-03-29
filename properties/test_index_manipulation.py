import itertools

import pytest

from xarray import Dataset
from xarray.testing import _assert_internal_invariants

pytest.importorskip("hypothesis")

import hypothesis.strategies as st
from hypothesis import note, settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)

import xarray.testing.strategies as xrst


@st.composite
def unique(draw, strategy):
    # https://stackoverflow.com/questions/73737073/create-hypothesis-strategy-that-returns-unique-values
    seen = draw(st.shared(st.builds(set), key="key-for-unique-elems"))
    return draw(
        strategy.filter(lambda x: x not in seen).map(lambda x: seen.add(x) or x)
    )


# Share to ensure we get unique names on each draw?
UNIQUE_NAME = unique(strategy=xrst.names())
DIM_NAME = xrst.dimension_names(name_strategy=UNIQUE_NAME, min_dims=1, max_dims=1)


class DatasetStateMachine(RuleBasedStateMachine):
    # Can't use bundles because we'd need pre-conditions on consumes(bundle)
    # indexed_dims = Bundle("indexed_dims")
    # multi_indexed_dims = Bundle("multi_indexed_dims")

    def __init__(self):
        super().__init__()
        self.dataset = Dataset()
        self.check_default_indexes = True

        # We track these separately as lists so we can guarantee order of iteration over them.
        # Order of iteration over Dataset.dims is not guaranteed
        self.indexed_dims = []
        self.multi_indexed_dims = []

    @rule(var=xrst.index_variables(dims=DIM_NAME))
    def add_dim_coord(self, var):
        (name,) = var.dims
        # dim coord
        self.dataset[name] = var
        # non-dim coord of same size; this allows renaming
        self.dataset[name + "_"] = var

        self.indexed_dims.append(name)

    @property
    def has_dims(self) -> bool:
        return bool(self.indexed_dims + self.multi_indexed_dims)

    @rule(data=st.data())
    @precondition(lambda self: self.has_dims)
    def reset_index(self, data):
        dim = data.draw(st.sampled_from(self.indexed_dims + self.multi_indexed_dims))
        self.check_default_indexes = False
        note(f"> resetting {dim}")
        self.dataset = self.dataset.reset_index(dim)

        if dim in self.indexed_dims:
            del self.indexed_dims[self.indexed_dims.index(dim)]
        elif dim in self.multi_indexed_dims:
            del self.multi_indexed_dims[self.multi_indexed_dims.index(dim)]

    @rule(newname=UNIQUE_NAME, data=st.data())
    @precondition(lambda self: bool(self.indexed_dims))
    def stack(self, newname, data):
        oldnames = data.draw(
            st.lists(st.sampled_from(self.indexed_dims), min_size=1, unique=True)
        )
        note(f"> stacking {oldnames} as {newname}")
        self.dataset = self.dataset.stack({newname: oldnames})

        self.multi_indexed_dims += [newname]
        for dim in oldnames:
            del self.indexed_dims[self.indexed_dims.index(dim)]

    @rule(data=st.data())
    @precondition(lambda self: bool(self.multi_indexed_dims))
    def unstack(self, data):
        # TODO: add None
        dim = data.draw(st.sampled_from(self.multi_indexed_dims))
        note(f"> unstacking {dim}")
        if dim is not None:
            pd_index = self.dataset.xindexes[dim].index
        self.dataset = self.dataset.unstack(dim)

        del self.multi_indexed_dims[self.multi_indexed_dims.index(dim)]

        if dim is not None:
            pd_index = self.dataset.xindexes[dim].index
            self.indexed_dims.extend(pd_index.names)
        else:
            # TODO: fix this
            pass

    @rule(newname=UNIQUE_NAME, data=st.data())
    @precondition(lambda self: self.has_dims)
    def rename_vars(self, newname, data):
        oldname = data.draw(
            st.sampled_from(self.indexed_dims + self.multi_indexed_dims)
        )
        # benbovy: "skip the default indexes invariant test when the name of an
        # existing dimension coordinate is passed as input kwarg or dict key
        # to .rename_vars()."
        self.check_default_indexes = False
        note(f"> renaming {oldname} to {newname}")
        self.dataset = self.dataset.rename_vars({oldname: newname})

        dim = oldname
        if dim in self.indexed_dims:
            del self.indexed_dims[self.indexed_dims.index(dim)]
        elif dim in self.multi_indexed_dims:
            del self.multi_indexed_dims[self.multi_indexed_dims.index(dim)]

    @property
    def swappable_dims(self):
        options = []
        for dim in self.indexed_dims:
            choices = [
                name
                for name, var in self.dataset._variables.items()
                if var.dims == (dim,)
            ]
            options.extend(
                (a, b) for a, b in itertools.zip_longest((dim,), choices, fillvalue=dim)
            )
        note(f"found swappable dims: {options}, all_dims: {tuple(self.dataset.dims)}")
        return options

    @rule(data=st.data())
    @precondition(lambda self: bool(self.swappable_dims))
    def swap_dims(self, data):
        ds = self.dataset
        dim, to = data.draw(st.sampled_from(self.swappable_dims))
        # TODO: swapping a dimension to itself
        # TODO: swapping from Index to a MultiIndex level
        # TODO: swapping from MultiIndex to a level of the same MultiIndex
        note(f"> swapping {dim} to {to}")
        self.dataset = ds.swap_dims({dim: to})

        del self.indexed_dims[self.indexed_dims.index(dim)]
        self.indexed_dims += [to]

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


DatasetStateMachine.TestCase.settings = settings(max_examples=200, deadline=None)
DatasetTest = DatasetStateMachine.TestCase
