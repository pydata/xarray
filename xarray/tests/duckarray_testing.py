import numpy as np

import xarray as xr

from .duckarray_testing_utils import apply_marks, preprocess_marks


def is_iterable(x):
    try:
        iter(x)
    except TypeError:
        return False

    return True


def always_iterable(x):
    if is_iterable(x) and not isinstance(x, (str, bytes)):
        return x
    else:
        return [x]


def duckarray_module(name, create, extra_asserts=None, global_marks=None, marks=None):
    import pytest

    def extra_assert(a, b):
        if extra_asserts is None:
            return

        for func in always_iterable(extra_asserts):
            func(a, b)

    # TODO:
    # - find a way to add create args as parametrizations
    # - add a optional type parameter to the create func spec
    # - how do we construct the expected values?
    # - should we check multiple dtypes?
    # - should we check multiple fill values?
    # - should we allow duckarray libraries to expect errors (pytest.raises / pytest.warns)?
    # - low priority: how do we redistribute the apply_marks mechanism?

    # convention: method specs for parametrize: one of
    # - method name
    # - tuple of method name, args, kwargs
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
                expected_dims = (
                    () if method not in ("argsort", "cumsum", "cumprod") else "x"
                )
                expected = xr.Variable(expected_dims, reduced)

                var = xr.Variable("x", data)
                actual = getattr(var, method)(dim="x")

                extra_assert(actual, expected)
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
