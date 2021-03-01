import pytest

from xarray import duckarray


class Module:
    def module_test1(self):
        pass

    def module_test2(self):
        pass

    @pytest.mark.parametrize("param1", ("a", "b", "c"))
    def parametrized_test(self, param1):
        pass

    class Submodule:
        def submodule_test(self):
            pass


@pytest.mark.parametrize(
    ["selector", "expected"],
    (
        ("test_function", (["test_function"], None)),
        (
            "TestGroup.TestSubgroup.test_function",
            (["TestGroup", "TestSubgroup", "test_function"], None),
        ),
        ("test_function[variant]", (["test_function"], "variant")),
        (
            "TestGroup.test_function[variant]",
            (["TestGroup", "test_function"], "variant"),
        ),
    ),
)
def test_parse_selector(selector, expected):
    actual = duckarray.parse_selector(selector)
    assert actual == expected


@pytest.mark.parametrize(
    ["components", "expected"],
    (
        (["module_test1"], (Module, Module.module_test1, "module_test1")),
        (
            ["Submodule", "submodule_test"],
            (Module.Submodule, Module.Submodule.submodule_test, "submodule_test"),
        ),
    ),
)
def test_get_test(components, expected):
    module = Module
    actual = duckarray.get_test(module, components)
    assert actual == expected
