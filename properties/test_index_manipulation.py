import itertools
import warnings

import numpy as np
import pytest

import xarray as xr
from xarray import Dataset
from xarray.testing import _assert_internal_invariants

pytest.importorskip("hypothesis")
pytestmark = pytest.mark.slow_hypothesis

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import note, settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
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


# Share to ensure we get unique names on each draw,
# so we don't try to add two variables with the same name
# or stack to a dimension with a name that already exists in the Dataset.
UNIQUE_NAME = unique(strategy=xrst.names())
DIM_NAME = xrst.dimension_names(name_strategy=UNIQUE_NAME, min_dims=1, max_dims=1)
index_variables = st.builds(
    xr.Variable,
    data=npst.arrays(
        dtype=xrst.pandas_index_dtypes(),
        shape=npst.array_shapes(min_dims=1, max_dims=1),
        elements=dict(allow_nan=False, allow_infinity=False, allow_subnormal=False),
        unique=True,
    ),
    dims=DIM_NAME,
    attrs=xrst.attrs(),
)


def add_dim_coord_and_data_var(ds, var):
    (name,) = var.dims
    # dim coord
    ds[name] = var
    # non-dim coord of same size; this allows renaming
    ds[name + "_"] = var


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

    @initialize(var=index_variables)
    def init_ds(self, var):
        """Initialize the Dataset so that at least one rule will always fire."""
        (name,) = var.dims
        add_dim_coord_and_data_var(self.dataset, var)

        self.indexed_dims.append(name)

    # TODO: stacking with a timedelta64 index and unstacking converts it to object
    @rule(var=index_variables)
    def add_dim_coord(self, var):
        (name,) = var.dims
        note(f"adding dimension coordinate {name}")
        add_dim_coord_and_data_var(self.dataset, var)

        self.indexed_dims.append(name)

    @rule(var=index_variables)
    def assign_coords(self, var):
        (name,) = var.dims
        note(f"assign_coords: {name}")
        self.dataset = self.dataset.assign_coords({name: var})

        self.indexed_dims.append(name)

    @property
    def has_indexed_dims(self) -> bool:
        return bool(self.indexed_dims + self.multi_indexed_dims)

    @rule(data=st.data())
    @precondition(lambda self: self.has_indexed_dims)
    def reset_index(self, data):
        dim = data.draw(st.sampled_from(self.indexed_dims + self.multi_indexed_dims))
        self.check_default_indexes = False
        note(f"> resetting {dim}")
        self.dataset = self.dataset.reset_index(dim)

        if dim in self.indexed_dims:
            del self.indexed_dims[self.indexed_dims.index(dim)]
        elif dim in self.multi_indexed_dims:
            del self.multi_indexed_dims[self.multi_indexed_dims.index(dim)]

    @rule(newname=UNIQUE_NAME, data=st.data(), create_index=st.booleans())
    @precondition(lambda self: bool(self.indexed_dims))
    def stack(self, newname, data, create_index):
        oldnames = data.draw(
            st.lists(
                st.sampled_from(self.indexed_dims),
                min_size=1,
                max_size=3 if create_index else None,
                unique=True,
            )
        )
        note(f"> stacking {oldnames} as {newname}")
        self.dataset = self.dataset.stack(
            {newname: oldnames}, create_index=create_index
        )

        if create_index:
            self.multi_indexed_dims += [newname]

        # if create_index is False, then we just drop these
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
            self.indexed_dims.extend(pd_index.names)
        else:
            # TODO: fix this
            pass

    @rule(newname=UNIQUE_NAME, data=st.data())
    @precondition(lambda self: bool(self.dataset.variables))
    def rename_vars(self, newname, data):
        dim = data.draw(st.sampled_from(sorted(self.dataset.variables)))
        # benbovy: "skip the default indexes invariant test when the name of an
        # existing dimension coordinate is passed as input kwarg or dict key
        # to .rename_vars()."
        self.check_default_indexes = False
        note(f"> renaming {dim} to {newname}")
        self.dataset = self.dataset.rename_vars({dim: newname})

        if dim in self.indexed_dims:
            del self.indexed_dims[self.indexed_dims.index(dim)]
        elif dim in self.multi_indexed_dims:
            del self.multi_indexed_dims[self.multi_indexed_dims.index(dim)]

    @precondition(lambda self: bool(self.dataset.dims))
    @rule(data=st.data())
    def drop_dims(self, data):
        dims = data.draw(
            st.lists(
                st.sampled_from(sorted(self.dataset.dims)),
                min_size=1,
                unique=True,
            )
        )
        note(f"> drop_dims: {dims}")
        # TODO: dropping a multi-index dimension raises a DeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self.dataset = self.dataset.drop_dims(dims)

        for dim in dims:
            if dim in self.indexed_dims:
                del self.indexed_dims[self.indexed_dims.index(dim)]
            elif dim in self.multi_indexed_dims:
                del self.multi_indexed_dims[self.multi_indexed_dims.index(dim)]

    @precondition(lambda self: bool(self.indexed_dims))
    @rule(data=st.data())
    def drop_indexes(self, data):
        self.check_default_indexes = False

        dims = data.draw(
            st.lists(st.sampled_from(self.indexed_dims), min_size=1, unique=True)
        )
        note(f"> drop_indexes: {dims}")
        self.dataset = self.dataset.drop_indexes(dims)

        for dim in dims:
            if dim in self.indexed_dims:
                del self.indexed_dims[self.indexed_dims.index(dim)]
            elif dim in self.multi_indexed_dims:
                del self.multi_indexed_dims[self.multi_indexed_dims.index(dim)]

    @property
    def swappable_dims(self):
        ds = self.dataset
        options = []
        for dim in self.indexed_dims:
            choices = [
                name
                for name, var in ds._variables.items()
                if var.dims == (dim,)
                # TODO: Avoid swapping a dimension to itself
                and name != dim
            ]
            options.extend(
                (a, b) for a, b in itertools.zip_longest((dim,), choices, fillvalue=dim)
            )
        return options

    @rule(data=st.data())
    # TODO: swap_dims is basically all broken if a multiindex is present
    # TODO: Avoid swapping from Index to a MultiIndex level
    # TODO: Avoid swapping from MultiIndex to a level of the same MultiIndex
    # TODO: Avoid swapping when a MultiIndex is present
    @precondition(lambda self: not bool(self.multi_indexed_dims))
    @precondition(lambda self: bool(self.swappable_dims))
    def swap_dims(self, data):
        ds = self.dataset
        options = self.swappable_dims
        dim, to = data.draw(st.sampled_from(options))
        note(
            f"> swapping {dim} to {to}, found swappable dims: {options}, all_dims: {tuple(self.dataset.dims)}"
        )
        self.dataset = ds.swap_dims({dim: to})

        del self.indexed_dims[self.indexed_dims.index(dim)]
        self.indexed_dims += [to]

    @invariant()
    def assert_invariants(self):
        # note(f"> ===\n\n {self.dataset!r} \n===\n\n")
        _assert_internal_invariants(self.dataset, self.check_default_indexes)


DatasetStateMachine.TestCase.settings = settings(max_examples=300, deadline=None)
DatasetTest = DatasetStateMachine.TestCase


@pytest.mark.skip(reason="failure detected by hypothesis")
def test_unstack_object():
    import xarray as xr

    ds = xr.Dataset()
    ds["0"] = np.array(["", "\x000"], dtype=object)
    ds.stack({"1": ["0"]}).unstack()


@pytest.mark.skip(reason="failure detected by hypothesis")
def test_unstack_timedelta_index():
    import xarray as xr

    ds = xr.Dataset()
    ds["0"] = np.array([0, 1, 2, 3], dtype="timedelta64[ns]")
    ds.stack({"1": ["0"]}).unstack()
