import numpy as np

import xarray as xr

from .duckarray_testing_utils import apply_marks, preprocess_marks


def duckarray_module(name, create, global_marks=None, marks=None):
    import pytest

    class TestModule:
        class TestVariable:
            @pytest.mark.parametrize(
                "method",
                (
                    "all",
                    "any",
                    "argmax",
                    "argmin",
                    "argsort",
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
            def test_reduce(self, method):
                data = create(np.linspace(0, 1, 10), method)

                reduced = getattr(np, method)(data, axis=0)

                var = xr.Variable("x", data)

                expected_dims = (
                    () if method not in ("argsort", "cumsum", "cumprod") else "x"
                )
                expected = xr.Variable(expected_dims, reduced)

                actual = getattr(var, method)(dim="x")
                xr.testing.assert_allclose(actual, expected)

    if global_marks is not None:
        TestModule.pytestmark = global_marks

    if marks is not None:
        processed = preprocess_marks(marks)
        for components, marks_ in processed:
            apply_marks(TestModule, components, marks_)

    TestModule.__name__ = f"Test{name.title()}"
    TestModule.__qualname__ = f"Test{name.title()}"

    return TestModule
