import pytest

from datatree import DataTree, register_datatree_accessor


class TestAccessor:
    def test_register(self) -> None:
        @register_datatree_accessor("demo")
        class DemoAccessor:
            """Demo accessor."""

            def __init__(self, xarray_obj):
                self._obj = xarray_obj

            @property
            def foo(self):
                return "bar"

        dt: DataTree = DataTree()
        assert dt.demo.foo == "bar"  # type: ignore

        # accessor is cached
        assert dt.demo is dt.demo  # type: ignore

        # check descriptor
        assert dt.demo.__doc__ == "Demo accessor."  # type: ignore
        # TODO: typing doesn't seem to work with accessors
        assert DataTree.demo.__doc__ == "Demo accessor."  # type: ignore
        assert isinstance(dt.demo, DemoAccessor)  # type: ignore
        assert DataTree.demo is DemoAccessor  # type: ignore

        with pytest.warns(Warning, match="overriding a preexisting attribute"):

            @register_datatree_accessor("demo")
            class Foo:
                pass

        # ensure we can remove it
        del DataTree.demo  # type: ignore
        assert not hasattr(DataTree, "demo")
