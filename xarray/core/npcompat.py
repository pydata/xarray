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
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeVar, Union

import numpy as np
from packaging.version import Version

# Type annotations stubs
try:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    # fall back for numpy < 1.20, ArrayLike adapted from numpy.typing._array_like
    from typing import Protocol

    if TYPE_CHECKING:

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
    # fall back for numpy < 1.20
    DTypeLike = Union[np.dtype, str]  # type: ignore[misc]


if Version(np.__version__) >= Version("1.20.0"):
    sliding_window_view = np.lib.stride_tricks.sliding_window_view
else:
    from numpy.core.numeric import normalize_axis_tuple  # type: ignore[attr-defined]
    from numpy.lib.stride_tricks import as_strided

    # copied from numpy.lib.stride_tricks
    def sliding_window_view(
        x, window_shape, axis=None, *, subok=False, writeable=False
    ):
        """
        Create a sliding window view into the array with the given window shape.

        Also known as rolling or moving window, the window slides across all
        dimensions of the array and extracts subsets of the array at all window
        positions.

        .. versionadded:: 1.20.0

        Parameters
        ----------
        x : array_like
            Array to create the sliding window view from.
        window_shape : int or tuple of int
            Size of window over each axis that takes part in the sliding window.
            If `axis` is not present, must have same length as the number of input
            array dimensions. Single integers `i` are treated as if they were the
            tuple `(i,)`.
        axis : int or tuple of int, optional
            Axis or axes along which the sliding window is applied.
            By default, the sliding window is applied to all axes and
            `window_shape[i]` will refer to axis `i` of `x`.
            If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to
            the axis `axis[i]` of `x`.
            Single integers `i` are treated as if they were the tuple `(i,)`.
        subok : bool, optional
            If True, sub-classes will be passed-through, otherwise the returned
            array will be forced to be a base-class array (default).
        writeable : bool, optional
            When true, allow writing to the returned view. The default is false,
            as this should be used with caution: the returned view contains the
            same memory location multiple times, so writing to one location will
            cause others to change.

        Returns
        -------
        view : ndarray
            Sliding window view of the array. The sliding window dimensions are
            inserted at the end, and the original dimensions are trimmed as
            required by the size of the sliding window.
            That is, ``view.shape = x_shape_trimmed + window_shape``, where
            ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less
            than the corresponding window size.
        """
        window_shape = (
            tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
        )
        # first convert input to array, possibly keeping subclass
        x = np.array(x, copy=False, subok=subok)

        window_shape_array = np.array(window_shape)
        if np.any(window_shape_array < 0):
            raise ValueError("`window_shape` cannot contain negative values")

        if axis is None:
            axis = tuple(range(x.ndim))
            if len(window_shape) != len(axis):
                raise ValueError(
                    f"Since axis is `None`, must provide "
                    f"window_shape for all dimensions of `x`; "
                    f"got {len(window_shape)} window_shape elements "
                    f"and `x.ndim` is {x.ndim}."
                )
        else:
            axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
            if len(window_shape) != len(axis):
                raise ValueError(
                    f"Must provide matching length window_shape and "
                    f"axis; got {len(window_shape)} window_shape "
                    f"elements and {len(axis)} axes elements."
                )

        out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

        # note: same axis can be windowed repeatedly
        x_shape_trimmed = list(x.shape)
        for ax, dim in zip(axis, window_shape):
            if x_shape_trimmed[ax] < dim:
                raise ValueError("window shape cannot be larger than input array shape")
            x_shape_trimmed[ax] -= dim - 1
        out_shape = tuple(x_shape_trimmed) + window_shape
        return as_strided(
            x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
        )


if Version(np.__version__) >= Version("1.22.0"):
    QUANTILE_METHODS = Literal[
        "inverted_cdf",
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "linear",
        "median_unbiased",
        "normal_unbiased",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ]
else:
    QUANTILE_METHODS = Literal[  # type: ignore[misc]
        "linear",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ]
