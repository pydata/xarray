from xarray.core.formatting import _compat_to_str, diff_dataset_repr

from .mapping import diff_treestructure


def diff_nodewise_summary(a, b, compat):
    """Iterates over all corresponding nodes, recording differences between data at each location."""

    compat_str = _compat_to_str(compat)

    summary = []
    for node_a, node_b in zip(a.subtree, b.subtree):
        a_ds, b_ds = node_a.ds, node_b.ds

        if not a_ds._all_compat(b_ds, compat):
            path = node_a.pathstr
            dataset_diff = diff_dataset_repr(a_ds, b_ds, compat_str)
            data_diff = "\n".join(dataset_diff.split("\n", 1)[1:])

            nodediff = (
                f"\nData in nodes at position '{path}' do not match:" f"{data_diff}"
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
