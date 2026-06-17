from __future__ import annotations

from importlib import import_module
from typing import Any


def is_dask_array_expr_array(data: Any) -> bool:
    try:
        dask_array = import_module("dask_array")
    except ImportError:
        return False

    array_type = getattr(dask_array, "Array", None)
    return array_type is not None and isinstance(data, array_type)
