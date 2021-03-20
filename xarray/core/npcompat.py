# Copyright (c) 2005-2011, NumPy Developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import builtins
import operator
import sys
from typing import TYPE_CHECKING, Any, Sequence, TypeVar, Union

import numpy as np


# Vendored from NumPy 1.12; we need a version that support duck typing, even
# on dask arrays with __array_function__ enabled.
def _validate_axis(axis, ndim, argname):
    try:
        axis = [operator.index(axis)]
    except TypeError:
        axis = list(axis)
    axis = [a + ndim if a < 0 else a for a in axis]
    if not builtins.all(0 <= a < ndim for a in axis):
        raise ValueError("invalid axis for this array in `%s` argument" % argname)
    if len(set(axis)) != len(axis):
        raise ValueError("repeated axis in `%s` argument" % argname)
    return axis


def moveaxis(a, source, destination):
    try:
        # allow duck-array types if they define transpose
        transpose = a.transpose
    except AttributeError:
        a = np.asarray(a)
        transpose = a.transpose

    source = _validate_axis(source, a.ndim, "source")
    destination = _validate_axis(destination, a.ndim, "destination")
    if len(source) != len(destination):
        raise ValueError(
            "`source` and `destination` arguments must have "
            "the same number of elements"
        )

    order = [n for n in range(a.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    result = transpose(order)
    return result


# Type annotations stubs
try:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    # fall back for numpy < 1.20, ArrayLike adapted from numpy.typing._array_like
    if sys.version_info >= (3, 8):
        from typing import Protocol

        HAVE_PROTOCOL = True
    else:
        try:
            from typing_extensions import Protocol
        except ImportError:
            HAVE_PROTOCOL = False
        else:
            HAVE_PROTOCOL = True

    if TYPE_CHECKING or HAVE_PROTOCOL:

        class _SupportsArray(Protocol):
            def __array__(self) -> np.ndarray:
                ...

    else:
        _SupportsArray = Any

    _T = TypeVar("_T")
    _NestedSequence = Union[
        _T,
        Sequence[_T],
        Sequence[Sequence[_T]],
        Sequence[Sequence[Sequence[_T]]],
        Sequence[Sequence[Sequence[Sequence[_T]]]],
    ]
    _RecursiveSequence = Sequence[Sequence[Sequence[Sequence[Sequence[Any]]]]]
    _ArrayLike = Union[
        _NestedSequence[_SupportsArray],
        _NestedSequence[_T],
    ]
    _ArrayLikeFallback = Union[
        _ArrayLike[Union[bool, int, float, complex, str, bytes]],
        _RecursiveSequence,
    ]
    # The extra step defining _ArrayLikeFallback and using ArrayLike as a type
    # alias for it works around an issue with mypy.
    # The `# type: ignore` below silences the warning of having multiple types
    # with the same name (ArrayLike and DTypeLike from the try block)
    ArrayLike = _ArrayLikeFallback  # type: ignore
    DTypeLike = Union[np.dtype, str]  # type: ignore


# from dask/array/utils.py
def _is_nep18_active():
    class A:
        def __array_function__(self, *args, **kwargs):
            return True

    try:
        return np.concatenate([A()])
    except ValueError:
        return False


IS_NEP18_ACTIVE = _is_nep18_active()
