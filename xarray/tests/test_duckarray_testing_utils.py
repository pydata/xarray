import pytest

from . import duckarray_testing_utils


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
    actual = duckarray_testing_utils.parse_selector(selector)
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
    actual = duckarray_testing_utils.get_test(module, components)
    assert actual == expected


@pytest.mark.parametrize(
    "marks",
    (
        pytest.param([pytest.mark.skip(reason="arbitrary")], id="single mark"),
        pytest.param(
            [
                pytest.mark.filterwarnings("error"),
                pytest.mark.parametrize("a", (0, 1, 2)),
            ],
            id="multiple marks",
        ),
    ),
)
def test_apply_marks_normal(marks):
    class Module:
        def module_test(self):
            pass

    module = Module
    components = ["module_test"]

    duckarray_testing_utils.apply_marks(module, components, marks)
    marked = Module.module_test
    actual = marked.pytestmark
    expected = [m.mark for m in marks]

    assert actual == expected


@pytest.mark.parametrize(
    ["mappings", "use_others", "duplicates", "expected", "error", "message"],
    (
        pytest.param(
            [{"a": 1}, {"b": 2}],
            False,
            False,
            {"a": 1, "b": 2},
            False,
            None,
            id="iterable",
        ),
        pytest.param(
            [{"a": 1}, {"b": 2}],
            True,
            False,
            {"a": 1, "b": 2},
            False,
            None,
            id="use others",
        ),
        pytest.param(
            [[{"a": 1}], {"b": 2}],
            True,
            False,
            None,
            True,
            "cannot pass both a iterable and multiple values",
            id="iterable and args",
        ),
        pytest.param(
            [{"a": 1}, {"a": 2}],
            False,
            "error",
            None,
            True,
            "duplicate keys found: ['a']",
            id="raise on duplicates",
        ),
    ),
)
def test_concat_mappings(mappings, use_others, duplicates, expected, error, message):
    func = duckarray_testing_utils.concat_mappings
    call = (
        lambda m: func(*m, duplicates=duplicates)
        if use_others
        else func(m, duplicates=duplicates)
    )
    if error:
        with pytest.raises(ValueError):
            call(mappings)
    else:
        actual = call(mappings)

        assert actual == expected


@pytest.mark.parametrize(
    "marks",
    (
        pytest.param([pytest.mark.skip(reason="arbitrary")], id="single mark"),
        pytest.param(
            [
                pytest.mark.filterwarnings("error"),
                pytest.mark.parametrize("a", (0, 1, 2)),
            ],
            id="multiple marks",
        ),
    ),
)
@pytest.mark.parametrize("variant", ("[a]", "[b]", "[c]"))
def test_apply_marks_variant(marks, variant):
    class Module:
        @pytest.mark.parametrize("param1", ("a", "b", "c"))
        def func(param1):
            pass

    module = Module
    components = ["func"]

    duckarray_testing_utils.apply_marks(module, components, {variant: marks})
    marked = Module.func
    actual = marked.pytestmark

    assert len(actual) > 1 and any(mark.name == "attach_marks" for mark in actual)
