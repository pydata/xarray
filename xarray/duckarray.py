import re

import numpy as np

import xarray as xr

identifier_re = r"[a-zA-Z_][a-zA-Z0-9_]*"
variant_re = re.compile(
    rf"^(?P<name>{identifier_re}(?:\.{identifier_re})*)(?:\[(?P<variant>[^]]+)\])?$"
)


def apply_marks(module, name, marks):
    def get_test(module, components):
        *parent_names, name = components

        parent = module
        for parent_name in parent_names:
            parent = getattr(parent, parent_name)

        test = getattr(parent, name)

        return parent, test, name

    match = variant_re.match(name)
    if match is not None:
        groups = match.groupdict()
        variant = groups["variant"]
        name = groups["name"]
    else:
        raise ValueError(f"invalid test name: {name!r}")

    components = name.split(".")
    if variant is not None:
        raise ValueError("variants are not supported, yet")
    else:
        parent, test, test_name = get_test(module, components)
        for mark in marks:
            test = mark(test)
        setattr(parent, test_name, test)


def duckarray_module(name, create, global_marks=None, marks=None):
    import pytest

    class TestModule:
        pytestmarks = global_marks

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

    for name, marks_ in marks.items():
        apply_marks(TestModule, name, marks_)

    TestModule.__name__ = f"Test{name.title()}"
    TestModule.__qualname__ = f"Test{name.title()}"

    return TestModule
