from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast, overload

from xarray.core.dataset import Dataset
from xarray.core.treenode import group_subtrees
from xarray.core.utils import result_name

if TYPE_CHECKING:
    from xarray.core.datatree import DataTree


@overload
def map_over_datasets(
    func: Callable[..., Dataset | None],
    *args: Any,
    kwargs: Mapping[str, Any] | None = None,
) -> DataTree: ...


# add an explicit overload for the most common case of two return values
# (python typing does not have a way to match tuple lengths in general)
@overload
def map_over_datasets(
    func: Callable[..., tuple[Dataset | None, Dataset | None]],
    *args: Any,
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[DataTree, DataTree]: ...


@overload
def map_over_datasets(
    func: Callable[..., tuple[Dataset | None, ...]],
    *args: Any,
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[DataTree, ...]: ...


def map_over_datasets(
    func: Callable[..., Dataset | tuple[Dataset | None, ...] | None],
    *args: Any,
    kwargs: Mapping[str, Any] | None = None,
) -> DataTree | tuple[DataTree, ...]:
    """Apply a function to every dataset in one or more DataTree objects.

    The DataTree objects must have the same structure (ie.., be isomorphic).
    Returns new trees which store the results.

    The function will be applied to any dataset stored in any of the nodes in
    the trees. The returned trees will have the same structure as the supplied
    trees.

    ``map_over_datasets`` is essentially syntactic sugar for the combination of
    ``group_subtrees`` and ``DataTree.from_dict``. For example, in the case of
    a two argument function that return one result, it is equivalent to::

        results = {}
        for path, (left, right) in group_subtrees(left_tree, right_tree):
            results[path] = func(left.dataset, right.dataset)
        return DataTree.from_dict(results)

    Parameters
    ----------
    func : callable
        Function called with the datasets at each matching node. Return a Dataset
        to store it at that path, or None to leave the node without local data.
        For multiple outputs, return a tuple of the same length at every node.
        Each element may be a Dataset or None.
    *args : tuple, optional
        Positional arguments passed on to `func`. Any DataTree arguments will be
        converted to Dataset objects via `.dataset`.
    kwargs : dict, optional
        Optional keyword arguments passed directly to ``func``.

    Returns
    -------
    DataTree or tuple of DataTree
        One tree if ``func`` returns a Dataset or None. If it returns a tuple,
        each position in the tuple produces a separate tree.

    See Also
    --------
    DataTree.map_over_datasets
    group_subtrees
    DataTree.from_dict

    Examples
    --------
    Keep only datasets containing temperature readings. Returning None for the
    root preserves its path and its child while dropping the root dataset.

    >>> tree = xr.DataTree.from_dict(
    ...     {
    ...         "/": xr.Dataset({"elevation": 350}),
    ...         "/station": xr.Dataset({"temperature": ("time", [8, 12])}),
    ...     }
    ... )
    >>> readings = xr.map_over_datasets(
    ...     lambda ds: ds if "temperature" in ds else None, tree
    ... )
    >>> readings.groups
    ('/', '/station')
    >>> list(readings.data_vars)
    []

    Return two datasets per node to collect minima and maxima in separate trees.

    >>> lows, highs = xr.map_over_datasets(lambda ds: (ds.min(), ds.max()), readings)
    >>> lows["station/temperature"].item(), highs["station/temperature"].item()
    (8, 12)
    """
    # TODO inspect function to work out immediately if the wrong number of arguments were passed for it?

    from xarray.core.datatree import DataTree

    if kwargs is None:
        kwargs = {}

    # Walk all trees simultaneously, applying func to all nodes that lie in same position in different trees
    # We don't know which arguments are DataTrees so we zip all arguments together as iterables
    # Store tuples of results in a dict because we don't yet know how many trees we need to rebuild to return
    out_data_objects: dict[str, Dataset | tuple[Dataset | None, ...] | None] = {}

    tree_args = [arg for arg in args if isinstance(arg, DataTree)]
    name = result_name(tree_args)

    for path, node_tree_args in group_subtrees(*tree_args):
        node_dataset_args = [arg.dataset for arg in node_tree_args]
        for i, arg in enumerate(args):
            if not isinstance(arg, DataTree):
                node_dataset_args.insert(i, arg)
        with add_path_context_to_errors(path):
            results = func(*node_dataset_args, **kwargs)
        out_data_objects[path] = results

    num_return_values = _check_all_return_values(out_data_objects)

    if num_return_values is None:
        # one return value
        out_data = cast(Mapping[str, Dataset | None], out_data_objects)
        return DataTree.from_dict(out_data, name=name)

    # multiple return values
    out_data_tuples = cast(Mapping[str, tuple[Dataset | None, ...]], out_data_objects)
    output_dicts: list[dict[str, Dataset | None]] = [
        {} for _ in range(num_return_values)
    ]
    for path, outputs in out_data_tuples.items():
        for output_dict, output in zip(output_dicts, outputs, strict=False):
            output_dict[path] = output

    return tuple(
        DataTree.from_dict(output_dict, name=name) for output_dict in output_dicts
    )


@contextmanager
def add_path_context_to_errors(path: str):
    """Add path context to any errors."""
    try:
        yield
    except Exception as e:
        e.add_note(f"Raised whilst mapping function over node(s) with path {path!r}")
        raise


def _check_single_set_return_values(path_to_node: str, obj: Any) -> int | None:
    """Check types returned from single evaluation of func, and return number of return values received from func."""
    if isinstance(obj, Dataset | None):
        return None  # no need to pack results

    if not isinstance(obj, tuple) or not all(
        isinstance(r, Dataset | None) for r in obj
    ):
        raise TypeError(
            f"the result of calling func on the node at position '{path_to_node}' is"
            f" not a Dataset or None or a tuple of such types:\n{obj!r}"
        )

    return len(obj)


def _check_all_return_values(returned_objects) -> int | None:
    """Walk through all values returned by mapping func over subtrees, raising on any invalid or inconsistent types."""
    result_data_objects = list(returned_objects.items())

    first_path, result = result_data_objects[0]
    return_values = _check_single_set_return_values(first_path, result)

    for path_to_node, obj in result_data_objects[1:]:
        cur_return_values = _check_single_set_return_values(path_to_node, obj)

        if return_values != cur_return_values:
            if return_values is None:
                raise TypeError(
                    f"Calling func on the nodes at position {path_to_node} returns "
                    f"a tuple of {cur_return_values} datasets, whereas calling func on the "
                    f"nodes at position {first_path} instead returns a single dataset."
                )
            elif cur_return_values is None:
                raise TypeError(
                    f"Calling func on the nodes at position {path_to_node} returns "
                    f"a single dataset, whereas calling func on the nodes at position "
                    f"{first_path} instead returns a tuple of {return_values} datasets."
                )
            else:
                raise TypeError(
                    f"Calling func on the nodes at position {path_to_node} returns "
                    f"a tuple of {cur_return_values} datasets, whereas calling func on "
                    f"the nodes at position {first_path} instead returns a tuple of "
                    f"{return_values} datasets."
                )

    return return_values
