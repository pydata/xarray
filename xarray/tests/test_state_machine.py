import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import note, settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

import xarray.testing.strategies as xrst
from xarray import Dataset
from xarray.testing import _assert_internal_invariants


def pandas_index_dtypes() -> st.SearchStrategy[np.dtype]:
    return (
        npst.integer_dtypes(sizes=(32, 64))
        | npst.unsigned_integer_dtypes(sizes=(32, 64))
        | npst.floating_dtypes(sizes=(32, 64))
    )


class DatasetStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.dataset = Dataset()
        self.check_default_indexes = True

    @rule(
        var=xrst.variables(
            dims=xrst.dimension_names(min_dims=1, max_dims=1),
            dtype=pandas_index_dtypes(),
        )
    )
    def add_dim_coord(self, var):
        (name,) = var.dims
        # dim coord
        self.dataset[name] = var
        # non-dim coord of same size
        self.dataset[name + "_"] = var
        note(f"> vars: {tuple(self.dataset._variables)}")

    @rule(newname=xrst.names())
    @precondition(lambda self: len(self.dataset.dims) >= 1)
    def rename_vars(self, newname):
        # TODO: randomize this
        # benbovy: "skip the default indexes invariant test when the name of an
        # existing dimension coordinate is passed as input kwarg or dict key
        # to .rename_vars()."
        oldname = tuple(self.dataset.dims)[0]
        self.check_default_indexes = False
        self.dataset = self.dataset.rename_vars({oldname: newname})
        note(f"> renaming {oldname} to {newname}")

    @rule()
    @precondition(lambda self: len(self.dataset._variables) >= 2)
    def swap_dims(self):
        ds = self.dataset
        # TODO: randomize?
        dim = tuple(ds.dims)[0]

        to = dim + "_" if "_" not in dim else dim[:-1]
        assert to in ds._variables
        self.dataset = ds.swap_dims({dim: to})
        note(f"> swapping {dim} to {to}")

    @invariant()
    def assert_invariants(self):
        # ndims = len(self.dataset.dims)

        note(f"> ===\n\n {self.dataset!r} \n===\n\n")
        _assert_internal_invariants(self.dataset, self.check_default_indexes)


DatasetStateMachine.TestCase.settings = settings(max_examples=1000)
DatasetTest = DatasetStateMachine.TestCase
