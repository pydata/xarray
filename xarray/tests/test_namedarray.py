import numpy as np
import pytest

from xarray.namedarray.core import NamedArray, as_compatible_data
from xarray.namedarray.utils import T_DuckArray


@pytest.mark.parametrize(
    "dims, data, attrs", [("x", [1, 2, 3], {"key": "value"}), ("y", [4, 5], None)]
)
def test_named_array_initialization(dims: str, data: T_DuckArray, attrs: dict) -> None:
    named_array = NamedArray(dims, data, attrs)
    assert named_array.dims == (dims,)
    assert np.array_equal(named_array.data, data)
    assert named_array.attrs == (attrs or {})


@pytest.mark.parametrize(
    "dims, data, expected_ndim, expected_size, expected_dtype, expected_shape, expected_len",
    [
        ("x", [1, 2, 3], 1, 3, np.dtype(int), (3,), 3),
        (["x", "y"], [[1, 2], [3, 4]], 2, 4, np.dtype(int), (2, 2), 2),
    ],
)
def test_named_array_properties(
    dims: str,
    data: T_DuckArray,
    expected_ndim: int,
    expected_size: int,
    expected_dtype: np.dtype,
    expected_shape: tuple,
    expected_len: int,
) -> None:
    named_array = NamedArray(dims, data)
    expected_nbytes = expected_size * np.array(data).dtype.itemsize
    assert named_array.ndim == expected_ndim
    assert named_array.size == expected_size
    assert named_array.dtype == expected_dtype
    assert named_array.shape == expected_shape
    assert named_array.nbytes == expected_nbytes
    assert len(named_array) == expected_len


@pytest.mark.parametrize(
    "initial_dims, initial_data, new_dims",
    [
        ("x", [1, 2, 3], "y"),
        (["x", "y"], [[1, 2], [3, 4]], ["a", "b"]),
    ],
)
def test_named_array_dims_setter(
    initial_dims: str, initial_data: T_DuckArray, new_dims: str
) -> None:
    named_array = NamedArray(initial_dims, initial_data)
    named_array.dims = new_dims
    assert named_array.dims == tuple(new_dims)


@pytest.mark.parametrize(
    "initial_dims, initial_data, new_attrs",
    [
        ("x", [1, 2, 3], {"new_key": "new_value"}),
        (["x", "y"], [[1, 2], [3, 4]], {"a": 1, "b": 2}),
        ("x", [1, 2, 3], {}),
    ],
)
def test_named_array_attrs_setter(
    initial_dims: str, initial_data: T_DuckArray, new_attrs: dict
) -> None:
    named_array = NamedArray(initial_dims, initial_data)
    named_array.attrs = new_attrs
    assert named_array.attrs == new_attrs


@pytest.mark.parametrize(
    "initial_dims, initial_data, new_data",
    [
        ("x", [1, 2, 3], [4, 5, 6]),
        (["x", "y"], [[1, 2], [3, 4]], [[4, 5], [6, 7]]),
    ],
)
def test_named_array_data_setter(
    initial_dims: str, initial_data: T_DuckArray, new_data: T_DuckArray
) -> None:
    named_array = NamedArray(initial_dims, initial_data)
    named_array.data = new_data
    assert np.array_equal(named_array.data, new_data)


@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        ([1, 2, 3], np.array([1, 2, 3])),
        (np.array([4, 5, 6]), np.array([4, 5, 6])),
    ],
)
def test_as_compatible_data(
    input_data: T_DuckArray, expected_output: T_DuckArray
) -> None:
    output = as_compatible_data(input_data)
    assert np.array_equal(output, expected_output)
