import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, note, settings

from ... import assert_identical
from . import strategies


class VariableReduceTests:
    def check_reduce(self, obj, op, *args, **kwargs):
        actual = getattr(obj, op)(*args, **kwargs)

        data = np.asarray(obj.data)
        expected = getattr(obj.copy(data=data), op)(*args, **kwargs)

        note(f"actual:\n{actual}")
        note(f"expected:\n{expected}")

        assert_identical(actual, expected)

    @staticmethod
    def create(shape, dtypes):
        return strategies.numpy_array(shape)

    @pytest.mark.parametrize(
        "method",
        (
            "all",
            "any",
            "cumprod",
            "cumsum",
            "max",
            "mean",
            "median",
            "min",
            "prod",
            "std",
            "sum",
            "var",
        ),
    )
    @given(st.data())
    @settings(deadline=None)
    def test_reduce(self, method, data):
        var = data.draw(
            strategies.variable(lambda shape, dtypes: self.create(shape, dtypes))
        )

        reduce_dims = data.draw(strategies.valid_dims(var.dims))

        self.check_reduce(var, method, dim=reduce_dims)


class DataArrayReduceTests:
    def check_reduce(self, obj, op, *args, **kwargs):
        actual = getattr(obj, op)(*args, **kwargs)

        data = np.asarray(obj.data)
        expected = getattr(obj.copy(data=data), op)(*args, **kwargs)

        note(f"actual:\n{actual}")
        note(f"expected:\n{expected}")

        assert_identical(actual, expected)

    @staticmethod
    def create(shape, dtypes, op):
        return strategies.numpy_array(shape, dtypes)

    @pytest.mark.parametrize(
        "method",
        (
            "all",
            "any",
            "cumprod",
            "cumsum",
            "max",
            "mean",
            "median",
            "min",
            "prod",
            "std",
            "sum",
            "var",
        ),
    )
    @given(st.data())
    @settings(deadline=None)
    def test_reduce(self, method, data):
        arr = data.draw(
            strategies.data_array(lambda shape, dtypes: self.create(shape, dtypes))
        )

        reduce_dims = data.draw(strategies.valid_dims(arr.dims))

        self.check_reduce(arr, method, dim=reduce_dims)


class DatasetReduceTests:
    def check_reduce(self, obj, op, *args, **kwargs):
        actual = getattr(obj, op)(*args, **kwargs)

        data = {name: np.asarray(obj.data) for name, obj in obj.variables.items()}
        expected = getattr(obj.copy(data=data), op)(*args, **kwargs)

        note(f"actual:\n{actual}")
        note(f"expected:\n{expected}")

        assert_identical(actual, expected)

    @staticmethod
    def create(shape, dtypes):
        return strategies.numpy_array(shape, dtypes)

    @pytest.mark.parametrize(
        "method",
        (
            "all",
            "any",
            "cumprod",
            "cumsum",
            "max",
            "mean",
            "median",
            "min",
            "prod",
            "std",
            "sum",
            "var",
        ),
    )
    @given(st.data())
    @settings(deadline=None)
    def test_reduce(self, method, data):
        ds = data.draw(
            strategies.dataset(
                lambda shape, dtypes: self.create(shape, dtypes), max_size=5
            )
        )

        reduce_dims = data.draw(strategies.valid_dims(ds.dims))

        self.check_reduce(ds, method, dim=reduce_dims)
