import math
import typing

import numpy as np


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
