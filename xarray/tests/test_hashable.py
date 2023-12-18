from __future__ import annotations

from enum import Enum
from typing import Union

import pytest

from xarray import DataArray, Dataset, Variable
from xarray.core.types import TypeAlias


class DEnum(Enum):
    dim = "dim"


class CustmomHashable:
    def __init__(self, a: int) -> None:
        self.a = a

    def __hash__(self) -> int:
        return self.a


parametrize_dim = pytest.mark.parametrize(
    "dim",
    [
        pytest.param(5, id="int"),
        pytest.param(("a", "b"), id="tuple"),
        pytest.param(DEnum.dim, id="enum"),
        pytest.param(CustmomHashable(3), id="HashableObject"),
    ],
)
DimT: TypeAlias = Union[int, tuple, DEnum, CustmomHashable]


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
