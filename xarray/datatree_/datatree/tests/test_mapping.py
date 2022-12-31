import pytest
import xarray as xr

from datatree.datatree import DataTree
from datatree.mapping import TreeIsomorphismError, check_isomorphic, map_over_subtree
from datatree.testing import assert_equal

empty = xr.Dataset()


class TestCheckTreesIsomorphic:
    def test_not_a_tree(self):
        with pytest.raises(TypeError, match="not a tree"):
            check_isomorphic("s", 1)

    def test_different_widths(self):
        dt1 = DataTree.from_dict(d={"a": empty})
        dt2 = DataTree.from_dict(d={"b": empty, "c": empty})
        expected_err_str = (
            "Number of children on node '/' of the left object: 1\n"
            "Number of children on node '/' of the right object: 2"
        )
        with pytest.raises(TreeIsomorphismError, match=expected_err_str):
            check_isomorphic(dt1, dt2)

    def test_different_heights(self):
        dt1 = DataTree.from_dict({"a": empty})
        dt2 = DataTree.from_dict({"b": empty, "b/c": empty})
        expected_err_str = (
            "Number of children on node '/a' of the left object: 0\n"
            "Number of children on node '/b' of the right object: 1"
        )
        with pytest.raises(TreeIsomorphismError, match=expected_err_str):
            check_isomorphic(dt1, dt2)

    def test_names_different(self):
        dt1 = DataTree.from_dict({"a": xr.Dataset()})
        dt2 = DataTree.from_dict({"b": empty})
        expected_err_str = (
            "Node '/a' in the left object has name 'a'\n"
            "Node '/b' in the right object has name 'b'"
        )
        with pytest.raises(TreeIsomorphismError, match=expected_err_str):
            check_isomorphic(dt1, dt2, require_names_equal=True)

    def test_isomorphic_names_equal(self):
        dt1 = DataTree.from_dict({"a": empty, "b": empty, "b/c": empty, "b/d": empty})
        dt2 = DataTree.from_dict({"a": empty, "b": empty, "b/c": empty, "b/d": empty})
        check_isomorphic(dt1, dt2, require_names_equal=True)

    def test_isomorphic_ordering(self):
        dt1 = DataTree.from_dict({"a": empty, "b": empty, "b/d": empty, "b/c": empty})
        dt2 = DataTree.from_dict({"a": empty, "b": empty, "b/c": empty, "b/d": empty})
        check_isomorphic(dt1, dt2, require_names_equal=False)

    def test_isomorphic_names_not_equal(self):
        dt1 = DataTree.from_dict({"a": empty, "b": empty, "b/c": empty, "b/d": empty})
        dt2 = DataTree.from_dict({"A": empty, "B": empty, "B/C": empty, "B/D": empty})
        check_isomorphic(dt1, dt2)

    def test_not_isomorphic_complex_tree(self, create_test_datatree):
        dt1 = create_test_datatree()
        dt2 = create_test_datatree()
        dt2["set1/set2/extra"] = DataTree(name="extra")
        with pytest.raises(TreeIsomorphismError, match="/set1/set2"):
            check_isomorphic(dt1, dt2)

    def test_checking_from_root(self, create_test_datatree):
        dt1 = create_test_datatree()
        dt2 = create_test_datatree()
        real_root = DataTree(name="real root")
        dt2.name = "not_real_root"
        dt2.parent = real_root
        with pytest.raises(TreeIsomorphismError):
            check_isomorphic(dt1, dt2, check_from_root=True)


class TestMapOverSubTree:
    def test_no_trees_passed(self):
        @map_over_subtree
        def times_ten(ds):
            return 10.0 * ds

        with pytest.raises(TypeError, match="Must pass at least one tree"):
            times_ten("dt")

    def test_not_isomorphic(self, create_test_datatree):
        dt1 = create_test_datatree()
        dt2 = create_test_datatree()
        dt2["set1/set2/extra"] = DataTree(name="extra")

        @map_over_subtree
        def times_ten(ds1, ds2):
            return ds1 * ds2

        with pytest.raises(TreeIsomorphismError):
            times_ten(dt1, dt2)

    def test_no_trees_returned(self, create_test_datatree):
        dt1 = create_test_datatree()
        dt2 = create_test_datatree()

        @map_over_subtree
        def bad_func(ds1, ds2):
            return None

        with pytest.raises(TypeError, match="return value of None"):
            bad_func(dt1, dt2)

    def test_single_dt_arg(self, create_test_datatree):
        dt = create_test_datatree()

        @map_over_subtree
        def times_ten(ds):
            return 10.0 * ds

        expected = create_test_datatree(modify=lambda ds: 10.0 * ds)
        result_tree = times_ten(dt)
        assert_equal(result_tree, expected)

    def test_single_dt_arg_plus_args_and_kwargs(self, create_test_datatree):
        dt = create_test_datatree()

        @map_over_subtree
        def multiply_then_add(ds, times, add=0.0):
            return (times * ds) + add

        expected = create_test_datatree(modify=lambda ds: (10.0 * ds) + 2.0)
        result_tree = multiply_then_add(dt, 10.0, add=2.0)
        assert_equal(result_tree, expected)

    def test_multiple_dt_args(self, create_test_datatree):
        dt1 = create_test_datatree()
        dt2 = create_test_datatree()

        @map_over_subtree
        def add(ds1, ds2):
            return ds1 + ds2

        expected = create_test_datatree(modify=lambda ds: 2.0 * ds)
        result = add(dt1, dt2)
        assert_equal(result, expected)

    def test_dt_as_kwarg(self, create_test_datatree):
        dt1 = create_test_datatree()
        dt2 = create_test_datatree()

        @map_over_subtree
        def add(ds1, value=0.0):
            return ds1 + value

        expected = create_test_datatree(modify=lambda ds: 2.0 * ds)
        result = add(dt1, value=dt2)
        assert_equal(result, expected)

    def test_return_multiple_dts(self, create_test_datatree):
        dt = create_test_datatree()

        @map_over_subtree
        def minmax(ds):
            return ds.min(), ds.max()

        dt_min, dt_max = minmax(dt)
        expected_min = create_test_datatree(modify=lambda ds: ds.min())
        assert_equal(dt_min, expected_min)
        expected_max = create_test_datatree(modify=lambda ds: ds.max())
        assert_equal(dt_max, expected_max)

    def test_return_wrong_type(self, simple_datatree):
        dt1 = simple_datatree

        @map_over_subtree
        def bad_func(ds1):
            return "string"

        with pytest.raises(TypeError, match="not Dataset or DataArray"):
            bad_func(dt1)

    def test_return_tuple_of_wrong_types(self, simple_datatree):
        dt1 = simple_datatree

        @map_over_subtree
        def bad_func(ds1):
            return xr.Dataset(), "string"

        with pytest.raises(TypeError, match="not Dataset or DataArray"):
            bad_func(dt1)

    @pytest.mark.xfail
    def test_return_inconsistent_number_of_results(self, simple_datatree):
        dt1 = simple_datatree

        @map_over_subtree
        def bad_func(ds):
            # Datasets in simple_datatree have different numbers of dims
            # TODO need to instead return different numbers of Dataset objects for this test to catch the intended error
            return tuple(ds.dims)

        with pytest.raises(TypeError, match="instead returns"):
            bad_func(dt1)

    def test_wrong_number_of_arguments_for_func(self, simple_datatree):
        dt = simple_datatree

        @map_over_subtree
        def times_ten(ds):
            return 10.0 * ds

        with pytest.raises(
            TypeError, match="takes 1 positional argument but 2 were given"
        ):
            times_ten(dt, dt)

    def test_map_single_dataset_against_whole_tree(self, create_test_datatree):
        dt = create_test_datatree()

        @map_over_subtree
        def nodewise_merge(node_ds, fixed_ds):
            return xr.merge([node_ds, fixed_ds])

        other_ds = xr.Dataset({"z": ("z", [0])})
        expected = create_test_datatree(modify=lambda ds: xr.merge([ds, other_ds]))
        result_tree = nodewise_merge(dt, other_ds)
        assert_equal(result_tree, expected)

    @pytest.mark.xfail
    def test_trees_with_different_node_names(self):
        # TODO test this after I've got good tests for renaming nodes
        raise NotImplementedError

    def test_dt_method(self, create_test_datatree):
        dt = create_test_datatree()

        def multiply_then_add(ds, times, add=0.0):
            return times * ds + add

        expected = create_test_datatree(modify=lambda ds: (10.0 * ds) + 2.0)
        result_tree = dt.map_over_subtree(multiply_then_add, 10.0, add=2.0)
        assert_equal(result_tree, expected)

    def test_discard_ancestry(self, create_test_datatree):
        # Check for datatree GH issue https://github.com/xarray-contrib/datatree/issues/48
        dt = create_test_datatree()
        subtree = dt["set1"]

        @map_over_subtree
        def times_ten(ds):
            return 10.0 * ds

        expected = create_test_datatree(modify=lambda ds: 10.0 * ds)["set1"]
        result_tree = times_ten(subtree)
        assert_equal(result_tree, expected, from_root=False)


@pytest.mark.xfail
class TestMapOverSubTreeInplace:
    def test_map_over_subtree_inplace(self):
        raise NotImplementedError
