from __future__ import annotations

from collections.abc import Callable, Mapping
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
    func: Callable[..., Dataset | None | tuple[Dataset | None, ...]],
    *args: Any,
    kwargs: Mapping[str, Any] | None = None,
) -> DataTree | tuple[DataTree, ...]:
    """
    Applies a function to every non-empty dataset in one or more DataTree objects
    with the same structure (i.e., that are isomorphic), returning new trees which
    store the results.

    The function will be applied to every node containing data (i.e., which has
    ``data_vars`` and/ or ``coordinates``) in the trees. The returned tree(s) will
    have the same structure as the supplied trees.

    ``func`` needs to return a Dataset, tuple of Dataset objects or None
    to be able to rebuild the subtrees after mapping, as each result will be
    assigned to its respective node of a new tree via `DataTree.from_dict`. Any
    returned value that is one of these types will be stacked into a separate
    tree before returning all of them.

    ``map_over_datasets`` is essentially syntactic sugar for the combination of
    ``group_subtrees`` and ``DataTree.from_dict``. For example, in the case of
    a two argument function that returns one result, it is equivalent to::

        results = {}
        for path, (left, right) in group_subtrees(left_tree, right_tree):
            results[path] = func(left.dataset, right.dataset)
        return DataTree.from_dict(results)

    Parameters
    ----------
    func : callable
        Function to apply to datasets with signature:

        `func(*args: Dataset, **kwargs) -> Union[Dataset, tuple[Dataset, ...]]`.

        (i.e. func must accept at least one Dataset and return at least one Dataset.)
    *args : tuple, optional
        Positional arguments passed on to `func`. Any DataTree arguments will be
        converted to Dataset objects via `.dataset`.
    kwargs : dict, optional
        Optional keyword arguments passed directly to ``func``.

    Returns
    -------
    Result of applying `func` to each node in the provided trees, packed back
    into DataTree objects via `DataTree.from_dict`.

    See also
    --------
    DataTree.map_over_datasets
    group_subtrees
    DataTree.from_dict
    """
    # TODO examples in the docstring
    # TODO inspect function to work out immediately if the wrong number of arguments were passed for it?

    from xarray.core.datatree import DataTree

    if kwargs is None:
        kwargs = {}

    # Walk all trees simultaneously, applying func to all nodes that lie in same position in different trees
    # We don't know which arguments are DataTrees so we zip all arguments together as iterables
    # Store tuples of results in a dict because we don't yet know how many trees we need to rebuild to return
    out_data_objects: dict[str, Dataset | tuple[Dataset | None, ...] | None] = {}
    func_called: dict[str, bool] = {}  # empty nodes don't call `func`

    tree_args = [arg for arg in args if isinstance(arg, DataTree)]
    name = result_name(tree_args)

    for path, node_tree_args in group_subtrees(*tree_args):
        if node_tree_args[0].has_data:
            node_dataset_args = [arg.dataset for arg in node_tree_args]
            for i, arg in enumerate(args):
                if not isinstance(arg, DataTree):
                    node_dataset_args.insert(i, arg)

            func_with_error_context = _handle_errors_with_path_context(path)(func)
            results = func_with_error_context(*node_dataset_args, **kwargs)
            func_called[path] = True

        elif node_tree_args[0].has_attrs:
            # propagate attrs
            results = node_tree_args[0].dataset
            func_called[path] = False

        else:
            # use Dataset instead of None to ensure it has copy method
            results = Dataset()
            func_called[path] = False

        out_data_objects[path] = results

    num_return_values = _check_all_return_values(out_data_objects, func_called)

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
        # duplicate outputs when func was not called (empty nodes)
        if not func_called[path]:
            out = cast(Dataset, outputs)
            outputs = tuple(out.copy() for _ in range(num_return_values))

        for output_dict, output in zip(output_dicts, outputs, strict=False):
            output_dict[path] = output

    return tuple(
        DataTree.from_dict(output_dict, name=name) for output_dict in output_dicts
    )


def _handle_errors_with_path_context(path: str):
    """Wraps given function so that if it fails it also raises path to node on which it failed."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add the context information to the error message
                add_note(
                    e, f"Raised whilst mapping function over node with path {path!r}"
                )
                raise

        return wrapper

    return decorator


def add_note(err: BaseException, msg: str) -> None:
    err.add_note(msg)


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


def _check_all_return_values(returned_objects, func_called) -> int | None:
    """Walk through all values returned by mapping func over subtrees, raising on any invalid or inconsistent types."""

    result_data_objects = list(returned_objects.items())

    func_called_before = False

    # initialize to None if all nodes are empty
    return_values = None

    for path_to_node, obj in result_data_objects:
        cur_return_values = _check_single_set_return_values(path_to_node, obj)

        cur_func_called = func_called[path_to_node]

        # the first node where func was actually called - needed to find the number of
        # return values
        if cur_func_called and not func_called_before:
            return_values = cur_return_values
            func_called_before = True
            first_path = path_to_node

        # no need to check if the function was not called
        if not cur_func_called:
            continue

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
