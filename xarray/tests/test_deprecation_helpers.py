import warnings

import pytest

from xarray.util.deprecation_helpers import _deprecate_positional_args, deprecate_dims


def test_deprecate_positional_args_warns_for_function():
    @_deprecate_positional_args("v0.1")
    def f1(a, b, *, c="c", d="d"):
        return a, b, c, d

    result = f1(1, 2)
    assert result == (1, 2, "c", "d")

    result = f1(1, 2, c=3, d=4)
    assert result == (1, 2, 3, 4)

    with pytest.warns(FutureWarning, match=r".*v0.1"):
        result = f1(1, 2, 3)  # type: ignore[call-arg]
    assert result == (1, 2, 3, "d")

    with pytest.warns(FutureWarning, match=r"Passing 'c' as positional"):
        result = f1(1, 2, 3)  # type: ignore[call-arg]
    assert result == (1, 2, 3, "d")

    with pytest.warns(FutureWarning, match=r"Passing 'c, d' as positional"):
        result = f1(1, 2, 3, 4)  # type: ignore[call-arg]
    assert result == (1, 2, 3, 4)

    @_deprecate_positional_args("v0.1")
    def f2(a="a", *, b="b", c="c", d="d"):
        return a, b, c, d

    with pytest.warns(FutureWarning, match=r"Passing 'b' as positional"):
        result = f2(1, 2)  # type: ignore[call-arg]
    assert result == (1, 2, "c", "d")

    @_deprecate_positional_args("v0.1")
    def f3(a, *, b="b", **kwargs):
        return a, b, kwargs

    with pytest.warns(FutureWarning, match=r"Passing 'b' as positional"):
        result = f3(1, 2)  # type: ignore[call-arg]
    assert result == (1, 2, {})

    with pytest.warns(FutureWarning, match=r"Passing 'b' as positional"):
        result = f3(1, 2, f="f")  # type: ignore[call-arg]
    assert result == (1, 2, {"f": "f"})

    @_deprecate_positional_args("v0.1")
    def f4(a, /, *, b="b", **kwargs):
        return a, b, kwargs

    result = f4(1)
    assert result == (1, "b", {})

    result = f4(1, b=2, f="f")
    assert result == (1, 2, {"f": "f"})

    with pytest.warns(FutureWarning, match=r"Passing 'b' as positional"):
        result = f4(1, 2, f="f")  # type: ignore[call-arg]
    assert result == (1, 2, {"f": "f"})

    with pytest.raises(TypeError, match=r"Keyword-only param without default"):

        @_deprecate_positional_args("v0.1")
        def f5(a, *, b, c=3, **kwargs):
            pass


def test_deprecate_positional_args_warns_for_class():
    class A1:
        @_deprecate_positional_args("v0.1")
        def method(self, a, b, *, c="c", d="d"):
            return a, b, c, d

    result = A1().method(1, 2)
    assert result == (1, 2, "c", "d")

    result = A1().method(1, 2, c=3, d=4)
    assert result == (1, 2, 3, 4)

    with pytest.warns(FutureWarning, match=r".*v0.1"):
        result = A1().method(1, 2, 3)  # type: ignore[call-arg]
    assert result == (1, 2, 3, "d")

    with pytest.warns(FutureWarning, match=r"Passing 'c' as positional"):
        result = A1().method(1, 2, 3)  # type: ignore[call-arg]
    assert result == (1, 2, 3, "d")

    with pytest.warns(FutureWarning, match=r"Passing 'c, d' as positional"):
        result = A1().method(1, 2, 3, 4)  # type: ignore[call-arg]
    assert result == (1, 2, 3, 4)

    class A2:
        @_deprecate_positional_args("v0.1")
        def method(self, a=1, b=1, *, c="c", d="d"):
            return a, b, c, d

    with pytest.warns(FutureWarning, match=r"Passing 'c' as positional"):
        result = A2().method(1, 2, 3)  # type: ignore[call-arg]
    assert result == (1, 2, 3, "d")

    with pytest.warns(FutureWarning, match=r"Passing 'c, d' as positional"):
        result = A2().method(1, 2, 3, 4)  # type: ignore[call-arg]
    assert result == (1, 2, 3, 4)

    class A3:
        @_deprecate_positional_args("v0.1")
        def method(self, a, *, b="b", **kwargs):
            return a, b, kwargs

    with pytest.warns(FutureWarning, match=r"Passing 'b' as positional"):
        result = A3().method(1, 2)  # type: ignore[call-arg]
    assert result == (1, 2, {})

    with pytest.warns(FutureWarning, match=r"Passing 'b' as positional"):
        result = A3().method(1, 2, f="f")  # type: ignore[call-arg]
    assert result == (1, 2, {"f": "f"})

    class A4:
        @_deprecate_positional_args("v0.1")
        def method(self, a, /, *, b="b", **kwargs):
            return a, b, kwargs

    result = A4().method(1)
    assert result == (1, "b", {})

    result = A4().method(1, b=2, f="f")
    assert result == (1, 2, {"f": "f"})

    with pytest.warns(FutureWarning, match=r"Passing 'b' as positional"):
        result = A4().method(1, 2, f="f")  # type: ignore[call-arg]
    assert result == (1, 2, {"f": "f"})

    with pytest.raises(TypeError, match=r"Keyword-only param without default"):

        class A5:
            @_deprecate_positional_args("v0.1")
            def __init__(self, a, *, b, c=3, **kwargs):
                pass


@pytest.mark.parametrize("old_name", ["dims", "dimensions"])
def test_deprecate_dims_warns_and_forwards(old_name: str) -> None:
    def func(*, dim: str, value: int) -> tuple[str, int]:
        return dim, value

    wrapped = (
        deprecate_dims(func)
        if old_name == "dims"
        else deprecate_dims(func, old_name=old_name)
    )
    with pytest.warns(
        FutureWarning, match=f"The `{old_name}` argument has been renamed to `dim`"
    ):
        result = wrapped(**{old_name: "x"}, value=42)
    assert result == ("x", 42)


@pytest.mark.parametrize("old_name", ["dims", "dimensions"])
def test_deprecate_dims_canonical_dim_is_quiet(old_name: str) -> None:
    def func(*, dim: str, value: int) -> tuple[str, int]:
        return dim, value

    wrapped = (
        deprecate_dims(func)
        if old_name == "dims"
        else deprecate_dims(func, old_name=old_name)
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = wrapped(dim="x", value=42)
    assert result == ("x", 42)
    assert not recorded
