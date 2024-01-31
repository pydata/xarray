import pytest
import xarray as xr

from datatree import DataTree, formatting_html


@pytest.fixture(scope="module", params=["some html", "some other html"])
def repr(request):
    return request.param


class Test_summarize_children:
    """
    Unit tests for summarize_children.
    """

    func = staticmethod(formatting_html.summarize_children)

    @pytest.fixture(scope="class")
    def childfree_tree_factory(self):
        """
        Fixture for a child-free DataTree factory.
        """
        from random import randint

        def _childfree_tree_factory():
            return DataTree(
                data=xr.Dataset({"z": ("y", [randint(1, 100) for _ in range(3)])})
            )

        return _childfree_tree_factory

    @pytest.fixture(scope="class")
    def childfree_tree(self, childfree_tree_factory):
        """
        Fixture for a child-free DataTree.
        """
        return childfree_tree_factory()

    @pytest.fixture(scope="function")
    def mock_node_repr(self, monkeypatch):
        """
        Apply mocking for node_repr.
        """

        def mock(group_title, dt):
            """
            Mock with a simple result
            """
            return group_title + " " + str(id(dt))

        monkeypatch.setattr(formatting_html, "node_repr", mock)

    @pytest.fixture(scope="function")
    def mock_wrap_repr(self, monkeypatch):
        """
        Apply mocking for _wrap_repr.
        """

        def mock(r, *, end, **kwargs):
            """
            Mock by appending "end" or "not end".
            """
            return r + " " + ("end" if end else "not end") + "//"

        monkeypatch.setattr(formatting_html, "_wrap_repr", mock)

    def test_empty_mapping(self):
        """
        Test with an empty mapping of children.
        """
        children = {}
        assert self.func(children) == (
            "<div style='display: inline-grid; grid-template-columns: 100%'>" "</div>"
        )

    def test_one_child(self, childfree_tree, mock_wrap_repr, mock_node_repr):
        """
        Test with one child.

        Uses a mock of _wrap_repr and node_repr to essentially mock
        the inline lambda function "lines_callback".
        """
        # Create mapping of children
        children = {"a": childfree_tree}

        # Expect first line to be produced from the first child, and
        # wrapped as the last child
        first_line = f"a {id(children['a'])} end//"

        assert self.func(children) == (
            "<div style='display: inline-grid; grid-template-columns: 100%'>"
            f"{first_line}"
            "</div>"
        )

    def test_two_children(self, childfree_tree_factory, mock_wrap_repr, mock_node_repr):
        """
        Test with two level deep children.

        Uses a mock of _wrap_repr and node_repr to essentially mock
        the inline lambda function "lines_callback".
        """

        # Create mapping of children
        children = {"a": childfree_tree_factory(), "b": childfree_tree_factory()}

        # Expect first line to be produced from the first child, and
        # wrapped as _not_ the last child
        first_line = f"a {id(children['a'])} not end//"

        # Expect second line to be produced from the second child, and
        # wrapped as the last child
        second_line = f"b {id(children['b'])} end//"

        assert self.func(children) == (
            "<div style='display: inline-grid; grid-template-columns: 100%'>"
            f"{first_line}"
            f"{second_line}"
            "</div>"
        )


class Test__wrap_repr:
    """
    Unit tests for _wrap_repr.
    """

    func = staticmethod(formatting_html._wrap_repr)

    def test_end(self, repr):
        """
        Test with end=True.
        """
        r = self.func(repr, end=True)
        assert r == (
            "<div style='display: inline-grid;'>"
            "<div style='"
            "grid-column-start: 1;"
            "border-right: 0.2em solid;"
            "border-color: var(--xr-border-color);"
            "height: 1.2em;"
            "width: 0px;"
            "'>"
            "</div>"
            "<div style='"
            "grid-column-start: 2;"
            "grid-row-start: 1;"
            "height: 1em;"
            "width: 20px;"
            "border-bottom: 0.2em solid;"
            "border-color: var(--xr-border-color);"
            "'>"
            "</div>"
            "<div style='"
            "grid-column-start: 3;"
            "'>"
            "<ul class='xr-sections'>"
            f"{repr}"
            "</ul>"
            "</div>"
            "</div>"
        )

    def test_not_end(self, repr):
        """
        Test with end=False.
        """
        r = self.func(repr, end=False)
        assert r == (
            "<div style='display: inline-grid;'>"
            "<div style='"
            "grid-column-start: 1;"
            "border-right: 0.2em solid;"
            "border-color: var(--xr-border-color);"
            "height: 100%;"
            "width: 0px;"
            "'>"
            "</div>"
            "<div style='"
            "grid-column-start: 2;"
            "grid-row-start: 1;"
            "height: 1em;"
            "width: 20px;"
            "border-bottom: 0.2em solid;"
            "border-color: var(--xr-border-color);"
            "'>"
            "</div>"
            "<div style='"
            "grid-column-start: 3;"
            "'>"
            "<ul class='xr-sections'>"
            f"{repr}"
            "</ul>"
            "</div>"
            "</div>"
        )
