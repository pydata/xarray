from abc import ABC, abstractclassmethod

import numpy as np
import pytest

from xarray.namedarray.core import NamedArray, as_compatible_data
from xarray.namedarray.utils import T_DuckArray


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


class NamedArraySubclassObjects(ABC):
    @abstractclassmethod
    def cls(self, *args, **kwargs) -> NamedArray:
        raise NotImplementedError

    def test_properties(self):
        data = 0.5 * np.arange(10).reshape(2, 5)
        named_array = self.cls(["x", "y"], data, {"key": "value"})
        assert named_array.dims == ("x", "y")
        assert np.array_equal(named_array.data, data)
        assert named_array.attrs == {"key": "value"}
        assert named_array.ndim == 2
        assert named_array.sizes == {"x": 2, "y": 5}
        assert named_array.size == 10
        assert named_array.nbytes == 80
        assert len(named_array) == 2

    def test_attrs(self):
        named_array = self.cls(["x", "y"], np.arange(10).reshape(2, 5))
        assert named_array.attrs == {}
        named_array.attrs["key"] = "value"
        assert named_array.attrs == {"key": "value"}
        named_array.attrs = {"key": "value2"}
        assert named_array.attrs == {"key": "value2"}


class TestNamedArray(NamedArraySubclassObjects):
    def cls(self, *args, **kwargs) -> NamedArray:
        return NamedArray(*args, **kwargs)

    @pytest.fixture(autouse=True)
    def setup(self):
        self.inputs = np.random.random((3, 4, 5)).astype(np.float32)

    def test_data(self):
        named_array = NamedArray(["x", "y", "z"], self.inputs)
        assert np.array_equal(named_array.data, self.inputs)
        with pytest.raises(ValueError):
            named_array.data = np.random.random((3, 4)).astype(np.float64)

    @pytest.mark.parametrize(
        "data, dtype",
        [
            ("foo", np.dtype("U3")),
            (np.bytes_("foo"), np.dtype("S3")),
        ],
    )
    def test_0d_string(self, data, dtype):
        named_array = NamedArray([], data)
        assert named_array.data == data
        assert named_array.dims == ()
        assert named_array.sizes == {}
        assert named_array.attrs == {}
        assert named_array.ndim == 0
        assert named_array.size == 1
        assert named_array.dtype == dtype

    def test_0d_datetime(self):
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
    def test_0d_timedelta(self, timedelta, expected_dtype):
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
    def test_dims_setter(self, dims, data_shape, new_dims, raises):
        named_array = NamedArray(dims, np.random.random(data_shape))
        assert named_array.dims == tuple(dims)
        if raises:
            with pytest.raises(ValueError):
                named_array.dims = new_dims
        else:
            named_array.dims = new_dims
            assert named_array.dims == tuple(new_dims)
