from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, cast

import numpy as np
import pytest

import xarray as xr
from xarray.namedarray.core import NamedArray, from_array
from xarray.namedarray.utils import T_DuckArray, _array

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from xarray.namedarray.utils import (
        Self,  # type: ignore[attr-defined]
        _Shape,
    )


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


@pytest.fixture
def random_inputs() -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))


@pytest.mark.parametrize(
    "input_data, expected_output, raise_error",
    [
        ([1, 2, 3], np.array([1, 2, 3]), False),
        (np.array([4, 5, 6]), np.array([4, 5, 6]), False),
        (NamedArray("time", np.array([1, 2, 3])), np.array([1, 2, 3]), True),
        (2, np.array(2), False),
    ],
)
def test_from_array(
    input_data: np.typing.ArrayLike,
    expected_output: np.ndarray[Any, Any],
    raise_error: bool,
) -> None:
    output: NamedArray[np.ndarray[Any, Any]]
    if raise_error:
        with pytest.raises(NotImplementedError):
            output = from_array(input_data)  # type: ignore
    else:
        output = from_array(input_data)

    assert np.array_equal(output.data, expected_output)


def test_from_array_with_masked_array() -> None:
    masked_array: np.ndarray[Any, np.dtype[np.generic]] = np.ma.array([1, 2, 3], mask=[False, True, False])  # type: ignore[no-untyped-call]
    with pytest.raises(NotImplementedError):
        from_array(masked_array)


def test_from_array_with_0d_object() -> None:
    data = np.empty((), dtype=object)
    data[()] = (10, 12, 12)
    np.array_equal(from_array(data).data, data)


# TODO: Make xr.core.indexing.ExplicitlyIndexed pass as a subclass of _array
# and remove this test.
def test_from_array_with_explicitly_indexed(
    random_inputs: np.ndarray[Any, Any]
) -> None:
    array = CustomArray(random_inputs)
    output: NamedArray[CustomArray[np.ndarray[Any, Any]]] = from_array(
        ("x", "y", "z"), array
    )
    assert isinstance(output.data, np.ndarray)

    array2 = CustomArrayIndexable(random_inputs)
    output2: NamedArray[CustomArrayIndexable[np.ndarray[Any, Any]]] = from_array(
        ("x", "y", "z"), array2
    )
    assert isinstance(output2.data, CustomArrayIndexable)


def test_properties() -> None:
    data = 0.5 * np.arange(10).reshape(2, 5)
    named_array: NamedArray[np.ndarray[Any, Any]]
    named_array = NamedArray(["x", "y"], data, {"key": "value"})
    assert named_array.dims == ("x", "y")
    assert np.array_equal(named_array.data, data)
    assert named_array.attrs == {"key": "value"}
    assert named_array.ndim == 2
    assert named_array.sizes == {"x": 2, "y": 5}
    assert named_array.size == 10
    assert named_array.nbytes == 80
    assert len(named_array) == 2


def test_attrs() -> None:
    named_array: NamedArray[np.ndarray[Any, Any]]
    named_array = NamedArray(["x", "y"], np.arange(10).reshape(2, 5))
    assert named_array.attrs == {}
    named_array.attrs["key"] = "value"
    assert named_array.attrs == {"key": "value"}
    named_array.attrs = {"key": "value2"}
    assert named_array.attrs == {"key": "value2"}


def test_data(random_inputs: np.ndarray[Any, Any]) -> None:
    named_array: NamedArray[np.ndarray[Any, Any]]
    named_array = NamedArray(["x", "y", "z"], random_inputs)
    assert np.array_equal(named_array.data, random_inputs)
    with pytest.raises(ValueError):
        named_array.data = np.random.random((3, 4)).astype(np.float64)


# Additional tests as per your original class-based code
@pytest.mark.parametrize(
    "data, dtype",
    [
        ("foo", np.dtype("U3")),
        (b"foo", np.dtype("S3")),
    ],
)
def test_0d_string(data: Any, dtype: np.typing.DTypeLike) -> None:
    named_array: NamedArray[np.ndarray[Any, Any]]
    named_array = from_array([], data)
    assert named_array.data == data
    assert named_array.dims == ()
    assert named_array.sizes == {}
    assert named_array.attrs == {}
    assert named_array.ndim == 0
    assert named_array.size == 1
    assert named_array.dtype == dtype


def test_0d_object() -> None:
    named_array: NamedArray[np.ndarray[Any, Any]]
    named_array = from_array([], (10, 12, 12))
    expected_data = np.empty((), dtype=object)
    expected_data[()] = (10, 12, 12)
    assert np.array_equal(named_array.data, expected_data)

    assert named_array.dims == ()
    assert named_array.sizes == {}
    assert named_array.attrs == {}
    assert named_array.ndim == 0
    assert named_array.size == 1
    assert named_array.dtype == np.dtype("O")


def test_0d_datetime() -> None:
    named_array: NamedArray[np.ndarray[Any, Any]]
    named_array = from_array([], np.datetime64("2000-01-01"))
    assert named_array.dtype == np.dtype("datetime64[D]")


@pytest.mark.parametrize(
    "timedelta, expected_dtype",
    [
        (np.timedelta64(1, "D"), np.dtype("timedelta64[D]")),
        (np.timedelta64(1, "s"), np.dtype("timedelta64[s]")),
        (np.timedelta64(1, "m"), np.dtype("timedelta64[m]")),
        (np.timedelta64(1, "h"), np.dtype("timedelta64[h]")),
        (np.timedelta64(1, "us"), np.dtype("timedelta64[us]")),
        (np.timedelta64(1, "ns"), np.dtype("timedelta64[ns]")),
        (np.timedelta64(1, "ps"), np.dtype("timedelta64[ps]")),
        (np.timedelta64(1, "fs"), np.dtype("timedelta64[fs]")),
        (np.timedelta64(1, "as"), np.dtype("timedelta64[as]")),
    ],
)
def test_0d_timedelta(
    timedelta: np.timedelta64, expected_dtype: np.dtype[np.timedelta64]
) -> None:
    named_array: NamedArray[np.ndarray[Any, Any]]
    named_array = from_array([], timedelta)
    assert named_array.dtype == expected_dtype
    assert named_array.data == timedelta


@pytest.mark.parametrize(
    "dims, data_shape, new_dims, raises",
    [
        (["x", "y", "z"], (2, 3, 4), ["a", "b", "c"], False),
        (["x", "y", "z"], (2, 3, 4), ["a", "b"], True),
        (["x", "y", "z"], (2, 4, 5), ["a", "b", "c", "d"], True),
        ([], [], (), False),
        ([], [], ("x",), True),
    ],
)
def test_dims_setter(dims: Any, data_shape: Any, new_dims: Any, raises: bool) -> None:
    named_array: NamedArray[np.ndarray[Any, Any]]
    named_array = NamedArray(dims, np.asarray(np.random.random(data_shape)))
    assert named_array.dims == tuple(dims)
    if raises:
        with pytest.raises(ValueError):
            named_array.dims = new_dims
    else:
        named_array.dims = new_dims
        assert named_array.dims == tuple(new_dims)


def test_duck_array_class() -> None:
    def test_duck_array_typevar(a: T_DuckArray) -> T_DuckArray:
        # Mypy checks a is valid:
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
