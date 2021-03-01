import pytest

from xarray import duckarray
from xarray.duckarray import duckarray_module

pint = pytest.importorskip("pint")
sparse = pytest.importorskip("sparse")
ureg = pint.UnitRegistry(force_ndarray_like=True)


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


class TestUtils:
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
    def test_parse_selector(self, selector, expected):
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
    def test_get_test(self, components, expected):
        module = Module
        actual = duckarray.get_test(module, components)
        print(actual)
        print(expected)
        assert actual == expected


def create_pint(data, method):
    if method in ("prod",):
        units = "dimensionless"
    else:
        units = "m"
    return ureg.Quantity(data, units)


def create_sparse(data, method):
    return sparse.COO.from_numpy(data)


TestPint = duckarray_module(
    "pint",
    create_pint,
    marks={
        "TestDataset.test_reduce": [pytest.mark.skip(reason="not implemented yet")],
    },
    global_marks=[
        pytest.mark.filterwarnings("error::pint.UnitStrippedWarning"),
    ],
)

TestSparse = duckarray_module(
    "sparse",
    create_sparse,
    marks={
        "TestDataset.test_reduce": [pytest.mark.skip(reason="not implemented, yet")],
    },
)
