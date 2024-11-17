"""xarray specific universal functions."""

import textwrap

import numpy as np

import xarray as xr
from xarray.core.groupby import GroupBy


def _walk_array_namespaces(obj, namespaces):
    if isinstance(obj, xr.DataTree):
        # TODO: DataTree doesn't actually support ufuncs yet
        for node in obj.subtree:
            _walk_array_namespaces(node.dataset, namespaces)
    elif isinstance(obj, xr.Dataset):
        for name in obj.data_vars:
            _walk_array_namespaces(obj[name], namespaces)
    elif isinstance(obj, GroupBy):
        _walk_array_namespaces(next(iter(obj))[1], namespaces)
    elif isinstance(obj, xr.DataArray | xr.Variable):
        _walk_array_namespaces(obj.data, namespaces)
    else:
        namespace = getattr(obj, "__array_namespace__", None)
        if namespace is not None:
            namespaces.add(namespace())

    return namespaces


def get_array_namespace(*args):
    xps = set()
    for arg in args:
        _walk_array_namespaces(arg, xps)

    xps.discard(np)
    if len(xps) > 1:
        names = [module.__name__ for module in xps]
        raise ValueError(f"Mixed array types {names} are not supported.")

    return next(iter(xps)) if len(xps) else np


class _UFuncDispatcher:
    """Wrapper for dispatching ufuncs."""

    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        xp = get_array_namespace(*args)
        func = getattr(xp, self._name)
        return xr.apply_ufunc(func, *args, dask="allowed", **kwargs)


def _skip_signature(doc, name):
    if not isinstance(doc, str):
        return doc

    if doc.startswith(name):
        signature_end = doc.find("\n\n")
        doc = doc[signature_end + 2 :]

    return doc


def _remove_unused_reference_labels(doc):
    if not isinstance(doc, str):
        return doc

    max_references = 5
    for num in range(max_references):
        label = f".. [{num}]"
        reference = f"[{num}]_"
        index = f"{num}.    "

        if label not in doc or reference in doc:
            continue

        doc = doc.replace(label, index)

    return doc


def _dedent(doc):
    if not isinstance(doc, str):
        return doc

    return textwrap.dedent(doc)


def _create_op(name):
    func = _UFuncDispatcher(name)
    func.__name__ = name
    doc = getattr(np, name).__doc__

    doc = _remove_unused_reference_labels(_skip_signature(_dedent(doc), name))

    func.__doc__ = (
        f"xarray specific variant of numpy.{name}. Handles "
        "xarray objects by dispatching to the appropriate "
        "function for the underlying array type.\n\n"
        f"Documentation from numpy:\n\n{doc}"
    )
    return func


# Auto generate from the public numpy ufuncs
np_ufuncs = {name for name in dir(np) if isinstance(getattr(np, name), np.ufunc)}
excluded_ufuncs = {"divmod", "frexp", "isnat", "matmul", "modf", "vecdot"}
additional_ufuncs = {"isreal"}  # "angle", "iscomplex"
__all__ = sorted(np_ufuncs - excluded_ufuncs | additional_ufuncs)


for name in __all__:
    globals()[name] = _create_op(name)
