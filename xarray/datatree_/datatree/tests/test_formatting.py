from textwrap import dedent

from xarray import Dataset

from datatree import DataTree
from datatree.formatting import diff_tree_repr


class TestRepr:
    def test_print_empty_node(self):
        dt = DataTree(name="root")
        printout = dt.__str__()
        assert printout == "DataTree('root', parent=None)"

    def test_print_empty_node_with_attrs(self):
        dat = Dataset(attrs={"note": "has attrs"})
        dt = DataTree(name="root", data=dat)
        printout = dt.__str__()
        assert printout == dedent(
            """\
            DataTree('root', parent=None)
                Dimensions:  ()
                Data variables:
                    *empty*
                Attributes:
                    note:     has attrs"""
        )

    def test_print_node_with_data(self):
        dat = Dataset({"a": [0, 2]})
        dt = DataTree(name="root", data=dat)
        printout = dt.__str__()
        expected = [
            "DataTree('root', parent=None)",
            "Dimensions",
            "Coordinates",
            "a",
            "Data variables",
            "*empty*",
        ]
        for expected_line, printed_line in zip(expected, printout.splitlines()):
            assert expected_line in printed_line

    def test_nested_node(self):
        dat = Dataset({"a": [0, 2]})
        root = DataTree(name="root")
        DataTree(name="results", data=dat, parent=root)
        printout = root.__str__()
        assert printout.splitlines()[2].startswith("    ")

    def test_print_datatree(self, simple_datatree):
        dt = simple_datatree
        print(dt)

        # TODO work out how to test something complex like this

    def test_repr_of_node_with_data(self):
        dat = Dataset({"a": [0, 2]})
        dt = DataTree(name="root", data=dat)
        assert "Coordinates" in repr(dt)


class TestDiffFormatting:
    def test_diff_structure(self):
        dt_1 = DataTree.from_dict({"a": None, "a/b": None, "a/c": None})
        dt_2 = DataTree.from_dict({"d": None, "d/e": None})

        expected = dedent(
            """\
        Left and right DataTree objects are not isomorphic

        Number of children on node '/a' of the left object: 2
        Number of children on node '/d' of the right object: 1"""
        )
        actual = diff_tree_repr(dt_1, dt_2, "isomorphic")
        assert actual == expected

    def test_diff_node_names(self):
        dt_1 = DataTree.from_dict({"a": None})
        dt_2 = DataTree.from_dict({"b": None})

        expected = dedent(
            """\
        Left and right DataTree objects are not identical

        Node '/a' in the left object has name 'a'
        Node '/b' in the right object has name 'b'"""
        )
        actual = diff_tree_repr(dt_1, dt_2, "identical")
        assert actual == expected

    def test_diff_node_data(self):
        ds1 = Dataset({"u": 0, "v": 1})
        ds3 = Dataset({"w": 5})
        dt_1 = DataTree.from_dict({"a": ds1, "a/b": ds3})
        ds2 = Dataset({"u": 0})
        ds4 = Dataset({"w": 6})
        dt_2 = DataTree.from_dict({"a": ds2, "a/b": ds4})

        expected = dedent(
            """\
        Left and right DataTree objects are not equal


        Data in nodes at position '/a' do not match:

        Data variables only on the left object:
            v        int64 1

        Data in nodes at position '/a/b' do not match:

        Differing data variables:
        L   w        int64 5
        R   w        int64 6"""
        )
        actual = diff_tree_repr(dt_1, dt_2, "equals")
        assert actual == expected
