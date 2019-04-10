from collections import abc
import sys

import numpy as np

integer_types = (int, np.integer, )

try:
    # solely for isinstance checks
    import dask.array
    dask_array_type = (dask.array.Array,)
except ImportError:  # pragma: no cover
    dask_array_type = ()


if sys.version < '3.5.3':
    TYPE_CHECKING = False

    class _ABCDummyBrackets(type(abc.Mapping)):  # abc.ABCMeta
        def __getitem__(cls, name):
            return cls

    class Mapping(abc.Mapping, metaclass=_ABCDummyBrackets):
        pass

    class MutableMapping(abc.MutableMapping, metaclass=_ABCDummyBrackets):
        pass

    class MutableSet(abc.MutableSet, metaclass=_ABCDummyBrackets):
        pass

else:
    from typing import (  # noqa: F401
        TYPE_CHECKING, Mapping, MutableMapping, MutableSet)
