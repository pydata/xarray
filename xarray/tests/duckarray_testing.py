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


class Label:
    """mark a parameter as in coordinate space"""

    def __init__(self, value):
        self.value = value


def convert_labels(draw, create, args, kwargs):
    def convert(value):
        if not isinstance(value, Label):
            return value
        else:
            return draw(create(value))

    args = [convert(value) for value in args]
    kwargs = {key: convert(value) for key, value in kwargs.items()}
    return args, kwargs


def default_expect_error(method, data, *args, **kwargs):
    return None, None


def duckarray_module(
    name,
    create,
    *,
    expect_error=None,
    extra_asserts=None,
    global_marks=None,
    marks=None,
):
    import hypothesis.extra.numpy as npst
    import hypothesis.strategies as st
    import pytest
    from hypothesis import given

    if expect_error is None:
        expect_error = default_expect_error
    dtypes = (
        npst.floating_dtypes() | npst.integer_dtypes() | npst.complex_number_dtypes()
    )

    def extra_assert(a, b):
        __tracebackhide__ = True
        if extra_asserts is None:
            return

        for func in always_iterable(extra_asserts):
            func(a, b)

    def numpy_data(shape):
        return npst.arrays(shape=shape, dtype=dtypes)

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
            @given(st.data())
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
            def test_reduce(self, method, data):
                shape = (10,)
                x = data.draw(create(numpy_data(shape), method))

                var = xr.Variable("x", x)
                actual = getattr(var, method)(dim="x")

                func = getattr(np, method)
                if x.dtype.kind in "cf":
                    # nan values possible
                    func = getattr(np, f"nan{method}", func)
                reduced = func(x, axis=0)
                expected_dims = (
                    () if method not in ("argsort", "cumsum", "cumprod") else "x"
                )
                expected = xr.Variable(expected_dims, reduced)

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
