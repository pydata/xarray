import random

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import note, settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

import xarray.testing.strategies as xrst
from xarray import Dataset
from xarray.testing import _assert_internal_invariants

random.seed(123456)


def pandas_index_dtypes() -> st.SearchStrategy[np.dtype]:
    return (
        npst.integer_dtypes(sizes=(32, 64))
        | npst.unsigned_integer_dtypes(sizes=(32, 64))
        | npst.floating_dtypes(sizes=(32, 64))
    )


# https://stackoverflow.com/a/73810689
# TODO: Consider building this into `dimension_names` and `names` strategies.
@st.composite
def unique(draw: st.DrawFn, strategy):
    seen = draw(st.shared(st.builds(set), key="key-for-unique-elems"))
    return draw(
        strategy.map(lambda x: tuple(x))
        .filter(lambda x: x not in seen)
        .map(lambda x: seen.add(x) or x)
    )


class DatasetStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.dataset = Dataset()
        self.check_default_indexes = True

    @rule(
        var=xrst.variables(
            dims=unique(xrst.dimension_names(min_dims=1, max_dims=1)),
            dtype=pandas_index_dtypes(),
        )
    )
    def add_dim_coord(self, var):
        (name,) = var.dims
        # dim coord
        self.dataset[name] = var
        # non-dim coord of same size; this allows renaming
        self.dataset[name + "_"] = var

    @precondition(lambda self: len(self.dataset.dims) >= 1)
    def reset_index(self):
        dim = random.choice(tuple(self.dataset.dims))
        self.dataset = self.dataset.reset_index(dim)

    @rule(newname=unique(xrst.names()))
    @precondition(lambda self: len(self.dataset.dims) >= 1)
    def stack(self, newname):
        # benbovy: "skip the default indexes invariant test when the name of an
        # existing dimension coordinate is passed as input kwarg or dict key
        # to .rename_vars()."
        while newname in self.dataset._variables:
            newname += "_foo"
        oldnames = random.choices(tuple(self.dataset.dims), k=2)
        self.dataset = self.dataset.stack({newname: oldnames})

    @rule()
    def unstack(self):
        self.dataset = self.dataset.unstack()

    @rule(newname=unique(xrst.names()))
    @precondition(lambda self: len(self.dataset.dims) >= 1)
    def rename_vars(self, newname):
        # benbovy: "skip the default indexes invariant test when the name of an
        # existing dimension coordinate is passed as input kwarg or dict key
        # to .rename_vars()."
        while newname in self.dataset._variables:
            newname += "_foo"
        oldname = random.choice(tuple(self.dataset.dims))
        self.check_default_indexes = False
        self.dataset = self.dataset.rename_vars({oldname: newname})
        note(f"> renaming {oldname} to {newname}")

    @rule()
    @precondition(lambda self: len(self.dataset._variables) >= 2)
    def swap_dims(self):
        ds = self.dataset
        dim = random.choice(tuple(ds.dims))

        to = dim + "_" if "_" not in dim else dim[:-1]
        assert to in ds._variables
        self.dataset = ds.swap_dims({dim: to})
        note(f"> swapping {dim} to {to}")

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
        note(f"> ===\n\n {self.dataset!r} \n===\n\n")
        _assert_internal_invariants(self.dataset, self.check_default_indexes)


DatasetStateMachine.TestCase.settings = settings(max_examples=1000)
DatasetTest = DatasetStateMachine.TestCase
