import pytest

from datatree.iterators import LevelOrderIter, PreOrderIter
from datatree.treenode import NamedNode, TreeError, TreeNode


class TestFamilyTree:
    def test_lonely(self):
        root = TreeNode()
        assert root.parent is None
        assert root.children == {}

    def test_parenting(self):
        john = TreeNode()
        mary = TreeNode()
        mary._set_parent(john, "Mary")

        assert mary.parent == john
        assert john.children["Mary"] is mary

    def test_no_time_traveller_loops(self):
        john = TreeNode()

        with pytest.raises(TreeError, match="cannot be a parent of itself"):
            john._set_parent(john, "John")

        with pytest.raises(TreeError, match="cannot be a parent of itself"):
            john.children = {"John": john}

        mary = TreeNode()
        rose = TreeNode()
        mary._set_parent(john, "Mary")
        rose._set_parent(mary, "Rose")

        with pytest.raises(TreeError, match="is already a descendant"):
            john._set_parent(rose, "John")

        with pytest.raises(TreeError, match="is already a descendant"):
            rose.children = {"John": john}

    def test_parent_swap(self):
        john = TreeNode()
        mary = TreeNode()
        mary._set_parent(john, "Mary")

        steve = TreeNode()
        mary._set_parent(steve, "Mary")

        assert mary.parent == steve
        assert steve.children["Mary"] is mary
        assert "Mary" not in john.children

    def test_multi_child_family(self):
        mary = TreeNode()
        kate = TreeNode()
        john = TreeNode(children={"Mary": mary, "Kate": kate})
        assert john.children["Mary"] is mary
        assert john.children["Kate"] is kate
        assert mary.parent is john
        assert kate.parent is john

    def test_disown_child(self):
        mary = TreeNode()
        john = TreeNode(children={"Mary": mary})
        mary.orphan()
        assert mary.parent is None
        assert "Mary" not in john.children

    def test_doppelganger_child(self):
        kate = TreeNode()
        john = TreeNode()

        with pytest.raises(TypeError):
            john.children = {"Kate": 666}

        with pytest.raises(TreeError, match="Cannot add same node"):
            john.children = {"Kate": kate, "Evil_Kate": kate}

        john = TreeNode(children={"Kate": kate})
        evil_kate = TreeNode()
        evil_kate._set_parent(john, "Kate")
        assert john.children["Kate"] is evil_kate

    def test_sibling_relationships(self):
        mary = TreeNode()
        kate = TreeNode()
        ashley = TreeNode()
        TreeNode(children={"Mary": mary, "Kate": kate, "Ashley": ashley})
        assert kate.siblings["Mary"] is mary
        assert kate.siblings["Ashley"] is ashley
        assert "Kate" not in kate.siblings

    def test_ancestors(self):
        tony = TreeNode()
        michael = TreeNode(children={"Tony": tony})
        vito = TreeNode(children={"Michael": michael})
        assert tony.root is vito
        assert tony.lineage == (tony, michael, vito)
        assert tony.ancestors == (vito, michael, tony)


class TestGetNodes:
    def test_get_child(self):
        steven = TreeNode()
        sue = TreeNode(children={"Steven": steven})
        mary = TreeNode(children={"Sue": sue})
        john = TreeNode(children={"Mary": mary})

        # get child
        assert john._get_item("Mary") is mary
        assert mary._get_item("Sue") is sue

        # no child exists
        with pytest.raises(KeyError):
            john._get_item("Kate")

        # get grandchild
        assert john._get_item("Mary/Sue") is sue

        # get great-grandchild
        assert john._get_item("Mary/Sue/Steven") is steven

        # get from middle of tree
        assert mary._get_item("Sue/Steven") is steven

    def test_get_upwards(self):
        sue = TreeNode()
        kate = TreeNode()
        mary = TreeNode(children={"Sue": sue, "Kate": kate})
        john = TreeNode(children={"Mary": mary})

        assert sue._get_item("../") is mary
        assert sue._get_item("../../") is john

        # relative path
        assert sue._get_item("../Kate") is kate

    def test_get_from_root(self):
        sue = TreeNode()
        mary = TreeNode(children={"Sue": sue})
        john = TreeNode(children={"Mary": mary})  # noqa

        assert sue._get_item("/Mary") is mary


class TestSetNodes:
    def test_set_child_node(self):
        john = TreeNode()
        mary = TreeNode()
        john._set_item("Mary", mary)

        assert john.children["Mary"] is mary
        assert isinstance(mary, TreeNode)
        assert mary.children == {}
        assert mary.parent is john

    def test_child_already_exists(self):
        mary = TreeNode()
        john = TreeNode(children={"Mary": mary})
        mary_2 = TreeNode()
        with pytest.raises(KeyError):
            john._set_item("Mary", mary_2, allow_overwrite=False)

    def test_set_grandchild(self):
        rose = TreeNode()
        mary = TreeNode()
        john = TreeNode()

        john._set_item("Mary", mary)
        john._set_item("Mary/Rose", rose)

        assert john.children["Mary"] is mary
        assert isinstance(mary, TreeNode)
        assert "Rose" in mary.children
        assert rose.parent is mary

    def test_create_intermediate_child(self):
        john = TreeNode()
        rose = TreeNode()

        # test intermediate children not allowed
        with pytest.raises(KeyError, match="Could not reach"):
            john._set_item(path="Mary/Rose", item=rose, new_nodes_along_path=False)

        # test intermediate children allowed
        john._set_item("Mary/Rose", rose, new_nodes_along_path=True)
        assert "Mary" in john.children
        mary = john.children["Mary"]
        assert isinstance(mary, TreeNode)
        assert mary.children == {"Rose": rose}
        assert rose.parent == mary
        assert rose.parent == mary

    def test_overwrite_child(self):
        john = TreeNode()
        mary = TreeNode()
        john._set_item("Mary", mary)

        # test overwriting not allowed
        marys_evil_twin = TreeNode()
        with pytest.raises(KeyError, match="Already a node object"):
            john._set_item("Mary", marys_evil_twin, allow_overwrite=False)
        assert john.children["Mary"] is mary
        assert marys_evil_twin.parent is None

        # test overwriting allowed
        marys_evil_twin = TreeNode()
        john._set_item("Mary", marys_evil_twin, allow_overwrite=True)
        assert john.children["Mary"] is marys_evil_twin
        assert marys_evil_twin.parent is john


class TestPruning:
    def test_del_child(self):
        john = TreeNode()
        mary = TreeNode()
        john._set_item("Mary", mary)

        del john["Mary"]
        assert "Mary" not in john.children
        assert mary.parent is None

        with pytest.raises(KeyError):
            del john["Mary"]


class TestNames:
    def test_child_gets_named_on_attach(self):
        sue = NamedNode()
        mary = NamedNode(children={"Sue": sue})  # noqa
        assert sue.name == "Sue"

    @pytest.mark.xfail(reason="requires refactoring to retain name")
    def test_grafted_subtree_retains_name(self):
        subtree = NamedNode("original")
        root = NamedNode(children={"new_name": subtree})  # noqa
        assert subtree.name == "original"


class TestPaths:
    def test_path_property(self):
        sue = NamedNode()
        mary = NamedNode(children={"Sue": sue})
        john = NamedNode(children={"Mary": mary})  # noqa
        assert sue.path == "/Mary/Sue"
        assert john.path == "/"

    def test_path_roundtrip(self):
        sue = NamedNode()
        mary = NamedNode(children={"Sue": sue})
        john = NamedNode(children={"Mary": mary})  # noqa
        assert john._get_item(sue.path) == sue

    def test_same_tree(self):
        mary = NamedNode()
        kate = NamedNode()
        john = NamedNode(children={"Mary": mary, "Kate": kate})  # noqa
        assert mary.same_tree(kate)

    def test_relative_paths(self):
        sue = NamedNode()
        mary = NamedNode(children={"Sue": sue})
        annie = NamedNode()
        john = NamedNode(children={"Mary": mary, "Annie": annie})

        assert sue.relative_to(john) == "Mary/Sue"
        assert john.relative_to(sue) == "../.."
        assert annie.relative_to(sue) == "../../Annie"
        assert sue.relative_to(annie) == "../Mary/Sue"
        assert sue.relative_to(sue) == "."

        evil_kate = NamedNode()
        with pytest.raises(ValueError, match="nodes do not lie within the same tree"):
            sue.relative_to(evil_kate)


def create_test_tree():
    f = NamedNode()
    b = NamedNode()
    a = NamedNode()
    d = NamedNode()
    c = NamedNode()
    e = NamedNode()
    g = NamedNode()
    i = NamedNode()
    h = NamedNode()

    f.children = {"b": b, "g": g}
    b.children = {"a": a, "d": d}
    d.children = {"c": c, "e": e}
    g.children = {"i": i}
    i.children = {"h": h}

    return f


class TestIterators:
    def test_preorderiter(self):
        tree = create_test_tree()
        result = [node.name for node in PreOrderIter(tree)]
        expected = [
            None,  # root Node is unnamed
            "b",
            "a",
            "d",
            "c",
            "e",
            "g",
            "i",
            "h",
        ]
        assert result == expected

    def test_levelorderiter(self):
        tree = create_test_tree()
        result = [node.name for node in LevelOrderIter(tree)]
        expected = [
            None,  # root Node is unnamed
            "b",
            "g",
            "a",
            "d",
            "i",
            "c",
            "e",
            "h",
        ]
        assert result == expected


class TestRenderTree:
    def test_render_nodetree(self):
        sam = NamedNode()
        ben = NamedNode()
        mary = NamedNode(children={"Sam": sam, "Ben": ben})
        kate = NamedNode()
        john = NamedNode(children={"Mary": mary, "Kate": kate})

        printout = john.__str__()
        expected_nodes = [
            "NamedNode()",
            "NamedNode('Mary')",
            "NamedNode('Sam')",
            "NamedNode('Ben')",
            "NamedNode('Kate')",
        ]
        for expected_node, printed_node in zip(expected_nodes, printout.splitlines()):
            assert expected_node in printed_node
