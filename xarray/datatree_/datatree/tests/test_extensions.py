import pytest

from xarray.core.datatree import DataTree
from xarray.datatree_.datatree.extensions import register_datatree_accessor


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
        assert dt.demo.foo == "bar"

        # accessor is cached
        assert dt.demo is dt.demo

        # check descriptor
        assert dt.demo.__doc__ == "Demo accessor."
        assert DataTree.demo.__doc__ == "Demo accessor." # type: ignore
        assert isinstance(dt.demo, DemoAccessor)
        assert DataTree.demo is DemoAccessor  # type: ignore

        with pytest.warns(Warning, match="overriding a preexisting attribute"):

            @register_datatree_accessor("demo")
            class Foo:
                pass

        # ensure we can remove it
        del DataTree.demo  # type: ignore
        assert not hasattr(DataTree, "demo")
