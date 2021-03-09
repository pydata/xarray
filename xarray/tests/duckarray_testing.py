import numpy as np

import xarray as xr

from .duckarray_testing_utils import apply_marks, preprocess_marks


def duckarray_module(name, create, global_marks=None, marks=None):
    import pytest

    class TestModule:
        class TestDataset:
            @pytest.mark.parametrize(
                "method",
                (
                    "mean",
                    "median",
                    "prod",
                    "sum",
                    "std",
                    "var",
                ),
            )
            def test_reduce(self, method):
                a = create(np.linspace(0, 1, 10), method)
                b = create(np.arange(10), method)

                reduced_a = getattr(np, method)(a)
                reduced_b = getattr(np, method)(b)

                ds = xr.Dataset({"a": ("x", a), "b": ("x", b)})
                expected = xr.Dataset({"a": reduced_a, "b": reduced_b})
                actual = getattr(ds, method)()
                xr.testing.assert_identical(actual, expected)

    if global_marks is not None:
        TestModule.pytestmark = global_marks

    if marks is not None:
        processed = preprocess_marks(marks)
        for components, marks_ in processed:
            apply_marks(TestModule, components, marks_)

    TestModule.__name__ = f"Test{name.title()}"
    TestModule.__qualname__ = f"Test{name.title()}"

    return TestModule
