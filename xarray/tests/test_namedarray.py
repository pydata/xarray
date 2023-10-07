from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# import xarray as xr
from xarray.namedarray.core import NamedArray

# from xarray.namedarray.utils import T_DuckArray

# if TYPE_CHECKING:
#     from xarray.namedarray.utils import Self  # type: ignore[attr-defined]


@pytest.fixture
def random_inputs() -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))


# @pytest.mark.parametrize(
#     "input_data, expected_output",
#     [
#         ([1, 2, 3], np.array([1, 2, 3])),
#         (np.array([4, 5, 6]), np.array([4, 5, 6])),
#         (NamedArray("time", np.array([1, 2, 3])), np.array([1, 2, 3])),
#         (2, np.array(2)),
#     ],
# )
# def test_as_compatible_data(
#     input_data: T_DuckArray, expected_output: T_DuckArray
# ) -> None:
#     output: T_DuckArray = as_compatible_data(input_data)
#     assert np.array_equal(output, expected_output)


# def test_as_compatible_data_with_masked_array() -> None:
#     masked_array = np.ma.array([1, 2, 3], mask=[False, True, False])  # type: ignore[no-untyped-call]
#     with pytest.raises(NotImplementedError):
#         as_compatible_data(masked_array)


# def test_as_compatible_data_with_0d_object() -> None:
#     data = np.empty((), dtype=object)
#     data[()] = (10, 12, 12)
#     np.array_equal(as_compatible_data(data), data)


# def test_as_compatible_data_with_explicitly_indexed(
#     random_inputs: np.ndarray[Any, Any]
# ) -> None:
#     # TODO: Make xr.core.indexing.ExplicitlyIndexed pass is_duck_array and remove this test.
#     class CustomArrayBase(xr.core.indexing.NDArrayMixin):
#         def __init__(self, array: T_DuckArray) -> None:
#             self.array = array

#         @property
#         def dtype(self) -> np.dtype[np.generic]:
#             return self.array.dtype

#         @property
#         def shape(self) -> tuple[int, ...]:
#             return self.array.shape

#         @property
#         def real(self) -> Self:
#             raise NotImplementedError

#         @property
#         def imag(self) -> Self:
#             raise NotImplementedError

#         def astype(self, dtype: np.typing.DTypeLike) -> Self:
#             raise NotImplementedError

#     class CustomArray(CustomArrayBase):
#         def __array__(self) -> np.ndarray[Any, np.dtype[np.generic]]:
#             return np.array(self.array)

#     class CustomArrayIndexable(CustomArrayBase, xr.core.indexing.ExplicitlyIndexed):
#         pass

#     array = CustomArray(random_inputs)
#     output: CustomArray = as_compatible_data(array)
#     assert isinstance(output, np.ndarray)

#     array2 = CustomArrayIndexable(random_inputs)
#     output2: CustomArrayIndexable = as_compatible_data(array2)
#     assert isinstance(output2, CustomArrayIndexable)


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
        (np.bytes_("foo"), np.dtype("S3")),
    ],
)
def test_0d_string(data: Any, dtype: np.typing.DTypeLike) -> None:
    named_array: NamedArray[np.ndarray[Any, Any]]
    named_array = NamedArray([], data)
    assert named_array.data == data
    assert named_array.dims == ()
    assert named_array.sizes == {}
    assert named_array.attrs == {}
    assert named_array.ndim == 0
    assert named_array.size == 1
    assert named_array.dtype == dtype


def test_0d_object() -> None:
    named_array: NamedArray[np.ndarray[Any, Any]]
    named_array = NamedArray([], (10, 12, 12))
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
    named_array = NamedArray([], np.datetime64("2000-01-01"))
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
    named_array: NamedArray[np.ndarray[Any, np.dtype[np.timedelta64]]]
    named_array = NamedArray([], timedelta)
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
    named_array = NamedArray(dims, np.random.random(data_shape))
    assert named_array.dims == tuple(dims)
    if raises:
        with pytest.raises(ValueError):
            named_array.dims = new_dims
    else:
        named_array.dims = new_dims
        assert named_array.dims == tuple(new_dims)


def test_typing() -> None:
    from typing import Generic

    from numpy.typing import DTypeLike
    from dask.array.core import Array as DaskArray

    from xarray.namedarray.core import from_array
    from xarray.namedarray.utils import T_DType_co

    a = [1, 2, 3]
    reveal_type(from_array("x", a))
    reveal_type(from_array([None], a))

    b = np.array([1, 2, 3])
    reveal_type(b)
    reveal_type(from_array("a", b))
    reveal_type(from_array([None], b))

    c: DaskArray = DaskArray([1, 2, 3], "c", {})
    reveal_type(c.shape)
    reveal_type(c)
    reveal_type(from_array("a", c))
    reveal_type(from_array([None], c))

    class CustomArray(Generic[T_DType_co]):
        def __init__(self, x: object, dtype: T_DType_co | None = None) -> None:
            ...

        @property
        def dtype(self) -> T_DType_co:
            ...

        @property
        def shape(self) -> tuple[int, ...]:
            ...

        @property
        def real(self) -> Self:
            ...

        @property
        def imag(self) -> Self:
            ...

        def astype(self, dtype: DTypeLike) -> Self:
            ...

        # # TODO: numpy doesn't use any inputs:
        # # https://github.com/numpy/numpy/blob/v1.24.3/numpy/_typing/_array_like.py#L38
        # def __array__(self) -> np.ndarray[Any, T_DType_co]:
        #     ...

    custom_a = CustomArray(2, dtype=np.dtype(int))
    reveal_type(custom_a)
    dims: tuple[str, ...] = ("x",)
    reveal_type(from_array(dims, custom_a))
