import pytest
from anytree.node.exceptions import TreeError
from anytree.resolver import ChildResolverError

from datatree.treenode import TreeNode


class TestFamilyTree:
    def test_lonely(self):
        root = TreeNode("root")
        assert root.name == "root"
        assert root.parent is None
        assert root.children == ()

    def test_parenting(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)

        assert mary.parent == john
        assert mary in john.children

        with pytest.raises(KeyError, match="already has a child named"):
            TreeNode("mary", parent=john)

        with pytest.raises(TreeError, match="not of type 'NodeMixin'"):
            mary.parent = "apple"

    def test_parent_swap(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)

        steve = TreeNode("steve")
        mary.parent = steve
        assert mary in steve.children
        assert mary not in john.children

    def test_multi_child_family(self):
        mary = TreeNode("mary")
        kate = TreeNode("kate")
        john = TreeNode("john", children=[mary, kate])
        assert mary in john.children
        assert kate in john.children
        assert mary.parent is john
        assert kate.parent is john

    def test_disown_child(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)
        mary.parent = None
        assert mary not in john.children

    def test_add_child(self):
        john = TreeNode("john")
        kate = TreeNode("kate")
        john.add_child(kate)
        assert kate in john.children
        assert kate.parent is john
        with pytest.raises(KeyError, match="already has a child named"):
            john.add_child(TreeNode("kate"))

    def test_assign_children(self):
        john = TreeNode("john")
        jack = TreeNode("jack")
        jill = TreeNode("jill")

        john.children = (jack, jill)
        assert jack in john.children
        assert jack.parent is john
        assert jill in john.children
        assert jill.parent is john

        evil_twin_jill = TreeNode("jill")
        with pytest.raises(KeyError, match="already has a child named"):
            john.children = (jack, jill, evil_twin_jill)

    def test_sibling_relationships(self):
        mary = TreeNode("mary")
        kate = TreeNode("kate")
        ashley = TreeNode("ashley")
        john = TreeNode("john", children=[mary, kate, ashley])
        assert mary in kate.siblings
        assert ashley in kate.siblings
        assert kate not in kate.siblings
        with pytest.raises(AttributeError):
            kate.siblings = john

    @pytest.mark.xfail
    def test_adoption(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_root(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_ancestors(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_descendants(self):
        raise NotImplementedError


class TestGetNodes:
    def test_get_child(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)
        assert john.get_node("mary") is mary
        assert john.get_node(("mary",)) is mary

    def test_get_nonexistent_child(self):
        john = TreeNode("john")
        TreeNode("jill", parent=john)
        with pytest.raises(ChildResolverError):
            john.get_node("mary")

    def test_get_grandchild(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)
        sue = TreeNode("sue", parent=mary)
        assert john.get_node("mary/sue") is sue
        assert john.get_node(("mary", "sue")) is sue

    def test_get_great_grandchild(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)
        sue = TreeNode("sue", parent=mary)
        steven = TreeNode("steven", parent=sue)
        assert john.get_node("mary/sue/steven") is steven
        assert john.get_node(("mary", "sue", "steven")) is steven

    def test_get_from_middle_of_tree(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)
        sue = TreeNode("sue", parent=mary)
        steven = TreeNode("steven", parent=sue)
        assert mary.get_node("sue/steven") is steven
        assert mary.get_node(("sue", "steven")) is steven


class TestSetNodes:
    def test_set_child_node(self):
        john = TreeNode("john")
        mary = TreeNode("mary")
        john.set_node("/", mary)

        mary = john.children[0]
        assert mary.name == "mary"
        assert isinstance(mary, TreeNode)
        assert mary.children == ()

    def test_child_already_exists(self):
        john = TreeNode("john")
        TreeNode("mary", parent=john)
        marys_replacement = TreeNode("mary")

        with pytest.raises(KeyError):
            john.set_node("/", marys_replacement, allow_overwrite=False)

    def test_set_grandchild(self):
        john = TreeNode("john")
        mary = TreeNode("mary")
        rose = TreeNode("rose")
        john.set_node("/", mary)
        john.set_node("/mary/", rose)

        mary = john.children[0]
        assert mary.name == "mary"
        assert isinstance(mary, TreeNode)
        assert rose in mary.children

        rose = mary.children[0]
        assert rose.name == "rose"
        assert isinstance(rose, TreeNode)
        assert rose.children == ()

    def test_set_grandchild_and_create_intermediate_child(self):
        john = TreeNode("john")
        rose = TreeNode("rose")
        john.set_node("/mary/", rose)

        mary = john.children[0]
        assert mary.name == "mary"
        assert isinstance(mary, TreeNode)
        assert mary.children[0] is rose

        rose = mary.children[0]
        assert rose.name == "rose"
        assert isinstance(rose, TreeNode)
        assert rose.children == ()

    def test_no_intermediate_children_allowed(self):
        john = TreeNode("john")
        rose = TreeNode("rose")
        with pytest.raises(KeyError, match="Cannot reach"):
            john.set_node(
                path="mary", node=rose, new_nodes_along_path=False, allow_overwrite=True
            )

    def test_set_great_grandchild(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)
        rose = TreeNode("rose", parent=mary)
        sue = TreeNode("sue")
        john.set_node("mary/rose", sue)
        assert sue.parent is rose

    def test_overwrite_child(self):
        john = TreeNode("john")
        mary = TreeNode("mary")
        john.set_node("/", mary)
        assert mary in john.children

        marys_evil_twin = TreeNode("mary")
        john.set_node("/", marys_evil_twin)
        assert marys_evil_twin in john.children
        assert mary not in john.children

    def test_dont_overwrite_child(self):
        john = TreeNode("john")
        mary = TreeNode("mary")
        john.set_node("/", mary)
        assert mary in john.children

        marys_evil_twin = TreeNode("mary")
        with pytest.raises(KeyError, match="path already points"):
            john.set_node(
                "", marys_evil_twin, new_nodes_along_path=True, allow_overwrite=False
            )
        assert mary in john.children
        assert marys_evil_twin not in john.children


class TestPruning:
    ...


class TestPaths:
    def test_pathstr(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)
        rose = TreeNode("rose", parent=mary)
        sue = TreeNode("sue", parent=rose)
        assert sue.pathstr == "john/mary/rose/sue"

    def test_relative_path(self):
        ...


class TestTags:
    ...


class TestRenderTree:
    def test_render_nodetree(self):
        mary = TreeNode("mary")
        kate = TreeNode("kate")
        john = TreeNode("john", children=[mary, kate])
        TreeNode("Sam", parent=mary)
        TreeNode("Ben", parent=mary)

        printout = john.__str__()
        expected_nodes = [
            "TreeNode('john')",
            "TreeNode('mary')",
            "TreeNode('Sam')",
            "TreeNode('Ben')",
            "TreeNode('kate')",
        ]
        for expected_node, printed_node in zip(expected_nodes, printout.splitlines()):
            assert expected_node in printed_node
