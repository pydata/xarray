import importlib
import math
import typing
from collections.abc import Iterator, Mapping

import numpy as np

K = typing.TypeVar("K")
V = typing.TypeVar("V")
T = typing.TypeVar("T")


class NdimSizeLenMixin:
    """Mixin class that extends a class that defines a ``shape`` property to
    one that also defines ``ndim``, ``size`` and ``__len__``.
    """

    __slots__ = ()

    @property
    def ndim(self: typing.Any) -> int:
        """
        Number of array dimensions.

        See Also
        --------
        numpy.ndarray.ndim
        """
        return len(self.shape)

    @property
    def size(self: typing.Any) -> int:
        """
        Number of elements in the array.

        Equal to ``np.prod(a.shape)``, i.e., the product of the array’s dimensions.

        See Also
        --------
        numpy.ndarray.size
        """
        return math.prod(self.shape)

    def __len__(self: typing.Any) -> int:
        try:
            return self.shape[0]
        except IndexError:
            raise TypeError("len() of unsized object")


class NDArrayMixin(NdimSizeLenMixin):
    """Mixin class for making wrappers of N-dimensional arrays that conform to
    the ndarray interface required for the data argument to Variable objects.

    A subclass should set the `array` property and override one or more of
    `dtype`, `shape` and `__getitem__`.
    """

    __slots__ = ()

    @property
    def dtype(self: typing.Any) -> np.dtype:
        return self.array.dtype

    @property
    def shape(self: typing.Any) -> tuple[int, ...]:
        return self.array.shape

    def __getitem__(self: typing.Any, key):
        return self.array[key]

    def __repr__(self: typing.Any) -> str:
        return f"{type(self).__name__}(array={self.array!r})"


class Frozen(Mapping[K, V]):
    """Wrapper around an object implementing the mapping interface to make it
    immutable. If you really want to modify the mapping, the mutable version is
    saved under the `mapping` attribute.
    """

    __slots__ = ("mapping",)

    def __init__(self, mapping: Mapping[K, V]):
        self.mapping = mapping

    def __getitem__(self, key: K) -> V:
        return self.mapping[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __contains__(self, key: object) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.mapping!r})"


def module_available(module: str) -> bool:
    """Checks whether a module is installed without importing it.

    Use this for a lightweight check and lazy imports.

    Parameters
    ----------
    module : str
        Name of the module.

    Returns
    -------
    available : bool
        Whether the module is installed.
    """
    return importlib.util.find_spec(module) is not None


def is_dask_collection(x):
    if module_available("dask"):
        from dask.base import is_dask_collection

        return is_dask_collection(x)
    return False


def is_duck_array(value: typing.Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    return (
        hasattr(value, "ndim")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and (
            (hasattr(value, "__array_function__") and hasattr(value, "__array_ufunc__"))
            or hasattr(value, "__array_namespace__")
        )
    )


def is_duck_dask_array(x):
    return is_duck_array(x) and is_dask_collection(x)
