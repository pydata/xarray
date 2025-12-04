from __future__ import annotations

import uuid
from enum import Enum
from typing import TYPE_CHECKING, Union

import pytest

from xarray import DataArray, Dataset, Variable

if TYPE_CHECKING:
    from xarray.core.types import TypeAlias

    DimT: TypeAlias = Union[int, tuple, "DEnum", "CustomHashable", uuid.UUID]


class DEnum(Enum):
    dim = "dim"


class CustomHashable:
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
        pytest.param(CustomHashable(3), id="HashableObject"),
        pytest.param(uuid.UUID("12345678-1234-5678-1234-567812345678"), id="uuid"),
    ],
)

parametrize_wrapped = pytest.mark.parametrize(
    "wrapped",
    [
        pytest.param(True, id="wrapped"),
        pytest.param(False, id="bare"),
    ],
)


@parametrize_dim
@parametrize_wrapped
def test_hashable_dims(dim: DimT, wrapped: bool) -> None:
    # Pass dims either wrapped in a list or bare
    dims_arg = [dim] if wrapped else dim

    # Bare tuple case should error with helpful message for 1D data
    if not wrapped and isinstance(dim, tuple):
        with pytest.raises(ValueError, match="This is ambiguous"):
            Variable(dims_arg, [1, 2, 3])
        with pytest.raises(ValueError, match="This is ambiguous"):
            DataArray([1, 2, 3], dims=dims_arg)
        with pytest.raises(ValueError):
            Dataset({"a": (dims_arg, [1, 2, 3])})
        return  # Don't run the other tests for this case

    v = Variable(dims_arg, [1, 2, 3])
    da = DataArray([1, 2, 3], dims=dims_arg)
    Dataset({"a": (dims_arg, [1, 2, 3])})

    # alternative constructors
    DataArray(v)
    Dataset({"a": v})
    Dataset({"a": da})


@parametrize_dim
def test_dataset_variable_hashable_names(dim: DimT) -> None:
    Dataset({dim: ("x", [1, 2, 3])})
