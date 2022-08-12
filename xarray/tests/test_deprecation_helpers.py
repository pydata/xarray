import pytest

from xarray.util.deprecation_helpers import _deprecate_positional_args


def test_deprecate_positional_args_warns_for_function():
    @_deprecate_positional_args("v0.1")
    def f1(a, b, *, c=1, d=1):
        pass

    with pytest.warns(FutureWarning, match=r".*v0.1"):
        f1(1, 2, 3)

    with pytest.warns(FutureWarning, match=r"Passing 'c' as positional"):
        f1(1, 2, 3)

    with pytest.warns(FutureWarning, match=r"Passing 'c, d' as positional"):
        f1(1, 2, 3, 4)

    @_deprecate_positional_args("v0.1")
    def f2(a=1, *, b=1, c=1, d=1):
        pass

    with pytest.warns(FutureWarning, match=r"Passing 'b' as positional"):
        f2(1, 2)

    @_deprecate_positional_args("v0.1")
    def f3(a, *, b=1, **kwargs):
        pass

    with pytest.warns(FutureWarning, match=r"Passing 'b' as positional"):
        f3(1, 2)

    with pytest.raises(TypeError, match=r"Cannot handle positional-only params"):

        @_deprecate_positional_args("v0.1")
        def f4(a, /, *, b=2, **kwargs):
            pass


def test_deprecate_positional_args_warns_for_class():
    class A1:
        @_deprecate_positional_args("v0.1")
        def __init__(self, a, b, *, c=1, d=1):
            pass

    with pytest.warns(FutureWarning, match=r".*v0.1"):
        A1(1, 2, 3)

    with pytest.warns(FutureWarning, match=r"Passing 'c' as positional"):
        A1(1, 2, 3)

    with pytest.warns(FutureWarning, match=r"Passing 'c, d' as positional"):
        A1(1, 2, 3, 4)

    class A2:
        @_deprecate_positional_args("v0.1")
        def __init__(self, a=1, b=1, *, c=1, d=1):
            pass

    with pytest.warns(FutureWarning, match=r"Passing 'c' as positional"):
        A2(1, 2, 3)

    with pytest.warns(FutureWarning, match=r"Passing 'c, d' as positional"):
        A2(1, 2, 3, 4)

    class A3:
        @_deprecate_positional_args("v0.1")
        def __init__(self, a, *, b=1, **kwargs):
            pass

    with pytest.warns(FutureWarning, match=r"Passing 'b' as positional"):
        A3(1, 2)

    with pytest.raises(TypeError, match=r"Cannot handle positional-only params"):

        class A3:
            @_deprecate_positional_args("v0.1")
            def __init__(self, a, /, *, b=1, **kwargs):
                pass
