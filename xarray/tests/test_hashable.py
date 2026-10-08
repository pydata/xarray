from __future__ import annotations

from enum import Enum

import numpy as np
import pytest

from xarray import DataArray, Dataset, Variable
from xarray.core.variable import IndexVariable, as_variable
from xarray.tests import assert_identical, requires_bottleneck


class DEnum(Enum):
    dim = "dim"


class CustomHashable:
    def __init__(self, a: int) -> None:
        self.a = a

    def __hash__(self) -> int:
        return self.a


type DimT = int | tuple | DEnum | CustomHashable


parametrize_dim = pytest.mark.parametrize(
    "dim",
    [
        pytest.param(5, id="int"),
        pytest.param(("a", "b"), id="tuple"),
        pytest.param(DEnum.dim, id="enum"),
        pytest.param(CustomHashable(3), id="HashableObject"),
    ],
)


@parametrize_dim
def test_hashable_dims(dim: DimT) -> None:
    v = Variable([dim], [1, 2, 3])
    da = DataArray([1, 2, 3], dims=[dim])
    Dataset({"a": ([dim], [1, 2, 3])})

    # alternative constructors
    DataArray(v)
    Dataset({"a": v})
    Dataset({"a": da})


@parametrize_dim
def test_dataset_variable_hashable_names(dim: DimT) -> None:
    Dataset({dim: ("x", [1, 2, 3])})


OTHER = 7


@parametrize_dim
class TestVariableHashable:
    @pytest.fixture
    def var(self, dim: DimT) -> Variable:
        return Variable([dim, OTHER], np.arange(6).reshape(2, 3), attrs={"a": 1})

    def test_dims(self, dim: DimT, var: Variable) -> None:
        assert var.dims == (dim, OTHER)
        assert var.sizes == {dim: 2, OTHER: 3}
        assert var.get_axis_num([dim]) == (0,)
        assert var.get_axis_num([OTHER, dim]) == (1, 0)

    def test_isel(self, dim: DimT, var: Variable) -> None:
        actual = var.isel({dim: 0})
        assert_identical(actual, Variable([OTHER], [0, 1, 2], attrs={"a": 1}))
        actual = var.isel({dim: Variable([dim], [False, True])})
        assert_identical(actual, Variable([dim, OTHER], [[3, 4, 5]], attrs={"a": 1}))

    def test_transpose(self, dim: DimT, var: Variable) -> None:
        assert var.transpose(OTHER, dim).dims == (OTHER, dim)
        assert var.transpose(..., dim).dims == (OTHER, dim)
        assert var.T.dims == (OTHER, dim)

    def test_squeeze(self, dim: DimT) -> None:
        var = Variable([dim, OTHER], np.zeros((1, 3)))
        assert var.squeeze([dim]).dims == (OTHER,)
        assert var.squeeze().dims == (OTHER,)

    def test_reduce(self, dim: DimT, var: Variable) -> None:
        expected = Variable([OTHER], [3, 5, 7], attrs={"a": 1})
        assert_identical(var.sum([dim]), expected)
        assert_identical(var.reduce(np.sum, dim=[dim]), expected)
        assert var.mean(...).dims == ()

    def test_argmin(self, dim: DimT, var: Variable) -> None:
        actual = var.argmin(dim=[dim, OTHER])
        assert isinstance(actual, dict)
        assert set(actual) == {dim, OTHER}

    def test_quantile(self, dim: DimT, var: Variable) -> None:
        assert var.quantile(0.5, dim=[dim]).dims == (OTHER,)

    def test_shift_roll_pad(self, dim: DimT, var: Variable) -> None:
        assert var.shift({dim: 1}).dims == (dim, OTHER)
        assert var.roll({dim: 1}).dims == (dim, OTHER)
        assert var.pad({dim: (1, 1)}).sizes == {dim: 4, OTHER: 3}

    def test_stack_unstack(self, dim: DimT, var: Variable) -> None:
        stacked = var.stack({"z": [dim, OTHER]})
        assert stacked.sizes == {"z": 6}
        unstacked = stacked.unstack({"z": {dim: 2, OTHER: 3}})
        assert_identical(unstacked, var)

    def test_set_dims(self, dim: DimT) -> None:
        var = Variable([OTHER], [1, 2, 3])
        assert var.set_dims({dim: 2, OTHER: 3}).sizes == {dim: 2, OTHER: 3}

    def test_concat(self, dim: DimT, var: Variable) -> None:
        actual = Variable.concat([var, var], dim=dim)
        assert actual.sizes == {dim: 4, OTHER: 3}

    @requires_bottleneck
    def test_rank(self, dim: DimT, var: Variable) -> None:
        assert var.rank(dim).dims == (dim, OTHER)

    def test_index_variable(self, dim: DimT) -> None:
        var = IndexVariable([dim], [1, 2, 3])
        assert var.name == dim
        assert var.to_index().name == dim
        assert Variable([dim], [1, 2, 3]).to_index_variable().name == dim
        actual = IndexVariable.concat([var, var], dim=dim)
        assert actual.sizes == {dim: 6}

    def test_as_variable(self, dim: DimT) -> None:
        expected = Variable([dim], [1, 2, 3])
        assert_identical(as_variable(([dim], [1, 2, 3])), expected)
        assert_identical(as_variable([1, 2, 3], name=dim, auto_convert=False), expected)
