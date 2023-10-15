from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, cast

import numpy as np

import xarray as xr
from xarray.namedarray.core import from_array
from xarray.namedarray.utils import T_DuckArray, _array

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from xarray.namedarray.utils import (
        Self,  # type: ignore[attr-defined]
        _Shape,
    )


# TODO: Make xr.core.indexing.ExplicitlyIndexed pass is_duck_array and remove this test.
class CustomArrayBase(xr.core.indexing.NDArrayMixin, Generic[T_DuckArray]):
    def __init__(self, array: T_DuckArray) -> None:
        self.array: T_DuckArray = array

    @property
    def dtype(self) -> np.dtype[np.generic]:
        return self.array.dtype

    @property
    def shape(self) -> _Shape:
        return self.array.shape

    @property
    def real(self) -> Self:
        raise NotImplementedError

    @property
    def imag(self) -> Self:
        raise NotImplementedError

    def astype(self, dtype: np.typing.DTypeLike) -> Self:
        raise NotImplementedError


class CustomArray(CustomArrayBase[T_DuckArray], Generic[T_DuckArray]):
    def __array__(self) -> np.ndarray[Any, np.dtype[np.generic]]:
        return np.array(self.array)


class CustomArrayIndexable(
    CustomArrayBase[T_DuckArray],
    xr.core.indexing.ExplicitlyIndexed,
    Generic[T_DuckArray],
):
    pass


def test_duck_array_class() -> None:
    def test_duck_array_typevar(a: T_DuckArray) -> T_DuckArray:
        # Mypy checks a calid:
        b: T_DuckArray = a

        # Runtime check if valid:
        if isinstance(b, _array):
            # TODO: cast is a mypy workaround for https://github.com/python/mypy/issues/10817
            # pyright doesn't need it.
            return cast(T_DuckArray, b)
        else:
            raise TypeError(f"a ({type(a)}) is not a valid _array")

    numpy_a: NDArray[np.int64] = np.array([2.1, 4], dtype=np.dtype(np.int64))
    custom_a: CustomArrayBase[NDArray[np.int64]] = CustomArrayBase(numpy_a)

    test_duck_array_typevar(numpy_a)
    test_duck_array_typevar(custom_a)


def test_typing() -> None:
    a = [1, 2, 3]
    reveal_type(from_array("x", a))
    reveal_type(from_array([None], a))

    # b = np.array([1, 2, 3])
    # reveal_type(b)
    # reveal_type(b.shape)
    # reveal_type(from_array("a", b))
    # reveal_type(from_array([None], b))

    b = np.array([1 + 1.2j, 2, 3], dtype=np.complexfloating)
    reveal_type(b)
    reveal_type(b.real)
    reveal_type(b.imag)

    from xarray.namedarray.core import _new

    bb = from_array("a", b)
    reveal_type(bb)
    reveal_type(_new(bb))
    reveal_type(_new(bb, np.array([1, 2, 3, 4], dtype=int)))
    reveal_type(bb.copy())

    reveal_type(bb.real)
    reveal_type(bb.imag)

    from xarray.namedarray.core import _replace_with_new_data_type

    bb = from_array("a", b)
    reveal_type(_replace_with_new_data_type(type(bb), bb._dims, bb.data.real, bb.attrs))
    reveal_type(_replace_with_new_data_type(type(bb), bb._dims, bb.data.imag, bb.attrs))

    # c: DaskArray = DaskArray([1, 2, 3], "c", {})
    # reveal_type(c)
    # reveal_type(c.shape)
    # reveal_type(from_array("a", c))
    # reveal_type(from_array([None], c))

    # custom_a = CustomArrayBase(np.array([2], dtype=np.dtype(int)))
    # reveal_type(custom_a)
    # reveal_type(custom_a.shape)
    # dims: tuple[str, ...] = ("x",)
    # reveal_type(from_array(dims, custom_a))
