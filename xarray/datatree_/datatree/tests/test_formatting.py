from textwrap import dedent

from xarray import Dataset

from datatree import DataTree
from datatree.formatting import diff_tree_repr


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
