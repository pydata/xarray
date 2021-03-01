import re

import numpy as np

import xarray as xr

identifier_re = r"[a-zA-Z_][a-zA-Z0-9_]*"
variant_re = re.compile(
    rf"^(?P<name>{identifier_re}(?:\.{identifier_re})*)(?:\[(?P<variant>[^]]+)\])?$"
)


def get_test(module, components):
    *parent_names, name = components

    parent = module
    for parent_name in parent_names:
        parent = getattr(parent, parent_name)

    test = getattr(parent, name)

    return parent, test, name


def parse_selector(selector):
    match = variant_re.match(selector)
    if match is not None:
        groups = match.groupdict()
        variant = groups["variant"]
        name = groups["name"]
    else:
        raise ValueError(f"invalid test name: {name!r}")

    components = name.split(".")
    return components, variant


def apply_marks_normal(test, marks):
    for mark in marks:
        test = mark(test)
    return test


def apply_marks_variant(test, variant, marks):
    raise NotImplementedError("variants are not supported, yet")


def apply_marks(module, name, marks):
    components, variant = parse_selector(name)
    parent, test, test_name = get_test(module, components)
    if variant is not None:
        marked_test = apply_marks_variant(test, variant, marks)
    else:
        marked_test = apply_marks_normal(test, marks)

    setattr(parent, test_name, marked_test)


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
        for name, marks_ in marks.items():
            apply_marks(TestModule, name, marks_)

    TestModule.__name__ = f"Test{name.title()}"
    TestModule.__qualname__ = f"Test{name.title()}"

    return TestModule
