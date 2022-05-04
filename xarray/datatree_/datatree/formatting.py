from typing import TYPE_CHECKING

from xarray.core.formatting import _compat_to_str, diff_dataset_repr

from .mapping import diff_treestructure
from .render import RenderTree

if TYPE_CHECKING:
    from .datatree import DataTree


def diff_nodewise_summary(a, b, compat):
    """Iterates over all corresponding nodes, recording differences between data at each location."""

    compat_str = _compat_to_str(compat)

    summary = []
    for node_a, node_b in zip(a.subtree, b.subtree):
        a_ds, b_ds = node_a.ds, node_b.ds

        if not a_ds._all_compat(b_ds, compat):
            dataset_diff = diff_dataset_repr(a_ds, b_ds, compat_str)
            data_diff = "\n".join(dataset_diff.split("\n", 1)[1:])

            nodediff = (
                f"\nData in nodes at position '{node_a.path}' do not match:"
                f"{data_diff}"
            )
            summary.append(nodediff)

    return "\n".join(summary)


def diff_tree_repr(a, b, compat):
    summary = [
        f"Left and right {type(a).__name__} objects are not {_compat_to_str(compat)}"
    ]

    # TODO check root parents?

    strict_names = True if compat in ["equals", "identical"] else False
    treestructure_diff = diff_treestructure(a, b, strict_names)

    # If the trees structures are different there is no point comparing each node
    # TODO we could show any differences in nodes up to the first place that structure differs?
    if treestructure_diff or compat == "isomorphic":
        summary.append("\n" + treestructure_diff)
    else:
        nodewise_diff = diff_nodewise_summary(a, b, compat)
        summary.append("\n" + nodewise_diff)

    return "\n".join(summary)


def datatree_repr(dt):
    """A printable representation of the structure of this entire tree."""
    renderer = RenderTree(dt)

    lines = []
    for pre, fill, node in renderer:
        node_repr = _single_node_repr(node)

        node_line = f"{pre}{node_repr.splitlines()[0]}"
        lines.append(node_line)

        if node.has_data or node.has_attrs:
            ds_repr = node_repr.splitlines()[2:]
            for line in ds_repr:
                if len(node.children) > 0:
                    lines.append(f"{fill}{renderer.style.vertical}{line}")
                else:
                    lines.append(f"{fill}{' ' * len(renderer.style.vertical)}{line}")

    # Tack on info about whether or not root node has a parent at the start
    first_line = lines[0]
    parent = f'"{dt.parent.name}"' if dt.parent is not None else "None"
    first_line_with_parent = first_line[:-1] + f", parent={parent})"
    lines[0] = first_line_with_parent

    return "\n".join(lines)


def _single_node_repr(node: "DataTree") -> str:
    """Information about this node, not including its relationships to other nodes."""
    node_info = f"DataTree('{node.name}')"

    if node.has_data or node.has_attrs:
        ds_info = "\n" + repr(node.ds)
    else:
        ds_info = ""
    return node_info + ds_info
