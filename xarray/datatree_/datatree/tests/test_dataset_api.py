import pytest

import xarray as xr
from xarray.testing import assert_equal

from datatree import DataTree, map_over_subtree
from datatree.datatree import DatasetNode

from test_datatree import create_test_datatree


class TestMapOverSubTree:
    def test_map_over_subtree(self):
        dt = create_test_datatree()

        @map_over_subtree
        def times_ten(ds):
            return 10.0 * ds

        result_tree = times_ten(dt)

        # TODO write an assert_tree_equal function
        for result_node, original_node, in zip(result_tree.subtree_nodes, dt.subtree_nodes):
            assert isinstance(result_node, DatasetNode)

            if original_node.has_data:
                assert_equal(result_node.ds, original_node.ds * 10.0)
            else:
                assert not result_node.has_data

    def test_map_over_subtree_with_args_and_kwargs(self):
        dt = create_test_datatree()

        @map_over_subtree
        def multiply_then_add(ds, times, add=0.0):
            return times * ds + add

        result_tree = multiply_then_add(dt, 10.0, add=2.0)

        for result_node, original_node, in zip(result_tree.subtree_nodes, dt.subtree_nodes):
            assert isinstance(result_node, DatasetNode)

            if original_node.has_data:
                assert_equal(result_node.ds, (original_node.ds * 10.0) + 2.0)
            else:
                assert not result_node.has_data

    def test_map_over_subtree_method(self):
        dt = create_test_datatree()

        def multiply_then_add(ds, times, add=0.0):
            return times * ds + add

        result_tree = dt.map_over_subtree(multiply_then_add, 10.0, add=2.0)

        for result_node, original_node, in zip(result_tree.subtree_nodes, dt.subtree_nodes):
            assert isinstance(result_node, DatasetNode)

            if original_node.has_data:
                assert_equal(result_node.ds, (original_node.ds * 10.0) + 2.0)
            else:
                assert not result_node.has_data

    @pytest.mark.xfail
    def test_map_over_subtree_inplace(self):
        raise NotImplementedError


class TestDSPropertyInheritance:
    ...


class TestDSMethodInheritance:
    ...


class TestBinaryOps:
    ...


class TestUFuncs:
    ...
