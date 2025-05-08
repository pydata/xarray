"""Generate module and stub file for arithmetic operators of various xarray classes.

For internal xarray development use only.

Usage:
    ipython xarray/util/generate_datatree_methods.py > xarray/core/_datatree_methods.py
    ruff check --fix xarray/core/_datatree_methods.py
    ruff format xarray/core/_datatree_methods.py


This requires ruff to be installed locally.
"""

import inspect
import textwrap
from collections.abc import Callable

MODULE_PREAMBLE = '''\
"""Mixin class to add Dataset methods to DataTree"""

# This file was generated using xarray.util.generate_datatree_methods. Do not edit manually.

from __future__ import annotations

from collections.abc import Hashable, Iterable
from functools import wraps
from typing import Literal

from xarray.core.dataset import Dataset
from xarray.core.datatree_mapping import map_over_datasets
from xarray.core.types import ErrorOptionsWithWarn, Self
'''


CLASS_PREAMBLE = """\
class TreeMethodsMixin:
    __slots__ = ()

"""

WRAPPER = """\
def _wrap_dataset_method(to_apply):
    def wrap_method(func):

        @wraps(func)
        def inner(self, *args, **kwargs):
            return map_over_datasets(to_apply, self, *args, kwargs=kwargs)

        return inner

    return wrap_method
"""

METHODS = (
    "argmax",
    "dropna",
    "transpose",
)


METHOD_TEMPLATE = '''\n
@_wrap_dataset_method(Dataset.{funcname})
def {funcname}{signature}:
    """{doc}"""
    # NOTE: the method is executed in the wrapper
    return self'''


def generate_method(method: Callable):
    kwargs = {
        "funcname": method.__name__,
        "doc": method.__doc__,
        "signature": inspect.signature(method),
    }

    m = METHOD_TEMPLATE.format(**kwargs)
    return textwrap.indent(m, "    ")


def write():
    from xarray.core.dataset import Dataset

    print(MODULE_PREAMBLE)
    print(WRAPPER)
    print(CLASS_PREAMBLE)

    for method in METHODS:
        print(generate_method(getattr(Dataset, method)))


if __name__ == "__main__":
    # fix hen and egg problem (file needs to exist so we can import Dataset)
    file = "xarray/core/_datatree_methods.py"
    with open(file, "w") as f:
        f.write("class TreeMethodsMixin: pass")

    write()
