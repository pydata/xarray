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
from distutils.version import LooseVersion
from typing import Union

import numpy as np

try:
    from numpy import isin
except ImportError:

    def isin(element, test_elements, assume_unique=False, invert=False):
        """
        Calculates `element in test_elements`, broadcasting over `element`
        only. Returns a boolean array of the same shape as `element` that is
        True where an element of `element` is in `test_elements` and False
        otherwise.

        Parameters
        ----------
        element : array_like
            Input array.
        test_elements : array_like
            The values against which to test each value of `element`.
            This argument is flattened if it is an array or array_like.
            See notes for behavior with non-array-like parameters.
        assume_unique : bool, optional
            If True, the input arrays are both assumed to be unique, which
            can speed up the calculation.  Default is False.
        invert : bool, optional
            If True, the values in the returned array are inverted, as if
            calculating `element not in test_elements`. Default is False.
            ``np.isin(a, b, invert=True)`` is equivalent to (but faster
            than) ``np.invert(np.isin(a, b))``.

        Returns
        -------
        isin : ndarray, bool
            Has the same shape as `element`. The values `element[isin]`
            are in `test_elements`.

        See Also
        --------
        in1d                  : Flattened version of this function.
        numpy.lib.arraysetops : Module with a number of other functions for
                                performing set operations on arrays.

        Notes
        -----

        `isin` is an element-wise function version of the python keyword `in`.
        ``isin(a, b)`` is roughly equivalent to
        ``np.array([item in b for item in a])`` if `a` and `b` are 1-D
        sequences.

        `element` and `test_elements` are converted to arrays if they are not
        already. If `test_elements` is a set (or other non-sequence collection)
        it will be converted to an object array with one element, rather than
        an array of the values contained in `test_elements`. This is a
        consequence of the `array` constructor's way of handling non-sequence
        collections. Converting the set to a list usually gives the desired
        behavior.

        .. versionadded:: 1.13.0

        Examples
        --------
        >>> element = 2*np.arange(4).reshape((2, 2))
        >>> element
        array([[0, 2],
               [4, 6]])
        >>> test_elements = [1, 2, 4, 8]
        >>> mask = np.isin(element, test_elements)
        >>> mask
        array([[ False,  True],
               [ True,  False]])
        >>> element[mask]
        array([2, 4])
        >>> mask = np.isin(element, test_elements, invert=True)
        >>> mask
        array([[ True, False],
               [ False, True]])
        >>> element[mask]
        array([0, 6])

        Because of how `array` handles sets, the following does not
        work as expected:

        >>> test_set = {1, 2, 4, 8}
        >>> np.isin(element, test_set)
        array([[ False, False],
               [ False, False]])

        Casting the set to a list gives the expected result:

        >>> np.isin(element, list(test_set))
        array([[ False,  True],
               [ True,  False]])
        """
        element = np.asarray(element)
        return np.in1d(
            element, test_elements, assume_unique=assume_unique, invert=invert
        ).reshape(element.shape)


if LooseVersion(np.__version__) >= LooseVersion("1.13"):
    gradient = np.gradient
else:

    def normalize_axis_tuple(axes, N):
        if isinstance(axes, int):
            axes = (axes,)
        return tuple([N + a if a < 0 else a for a in axes])

    def gradient(f, *varargs, axis=None, edge_order=1):
        f = np.asanyarray(f)
        N = f.ndim  # number of dimensions

        axes = axis
        del axis

        if axes is None:
            axes = tuple(range(N))
        else:
            axes = normalize_axis_tuple(axes, N)

        len_axes = len(axes)
        n = len(varargs)
        if n == 0:
            # no spacing argument - use 1 in all axes
            dx = [1.0] * len_axes
        elif n == 1 and np.ndim(varargs[0]) == 0:
            # single scalar for all axes
            dx = varargs * len_axes
        elif n == len_axes:
            # scalar or 1d array for each axis
            dx = list(varargs)
            for i, distances in enumerate(dx):
                if np.ndim(distances) == 0:
                    continue
                elif np.ndim(distances) != 1:
                    raise ValueError("distances must be either scalars or 1d")
                if len(distances) != f.shape[axes[i]]:
                    raise ValueError(
                        "when 1d, distances must match the "
                        "length of the corresponding dimension"
                    )
                diffx = np.diff(distances)
                # if distances are constant reduce to the scalar case
                # since it brings a consistent speedup
                if (diffx == diffx[0]).all():
                    diffx = diffx[0]
                dx[i] = diffx
        else:
            raise TypeError("invalid number of arguments")

        if edge_order > 2:
            raise ValueError("'edge_order' greater than 2 not supported")

        # use central differences on interior and one-sided differences on the
        # endpoints. This preserves second order-accuracy over the full domain.

        outvals = []

        # create slice objects --- initially all are [:, :, ..., :]
        slice1 = [slice(None)] * N
        slice2 = [slice(None)] * N
        slice3 = [slice(None)] * N
        slice4 = [slice(None)] * N

        otype = f.dtype.char
        if otype not in ["f", "d", "F", "D", "m", "M"]:
            otype = "d"

        # Difference of datetime64 elements results in timedelta64
        if otype == "M":
            # Need to use the full dtype name because it contains unit
            # information
            otype = f.dtype.name.replace("datetime", "timedelta")
        elif otype == "m":
            # Needs to keep the specific units, can't be a general unit
            otype = f.dtype

        # Convert datetime64 data into ints. Make dummy variable `y`
        # that is a view of ints if the data is datetime64, otherwise
        # just set y equal to the array `f`.
        if f.dtype.char in ["M", "m"]:
            y = f.view("int64")
        else:
            y = f

        for i, axis in enumerate(axes):
            if y.shape[axis] < edge_order + 1:
                raise ValueError(
                    "Shape of array too small to calculate a numerical "
                    "gradient, at least (edge_order + 1) elements are "
                    "required."
                )
            # result allocation
            out = np.empty_like(y, dtype=otype)

            uniform_spacing = np.ndim(dx[i]) == 0

            # Numerical differentiation: 2nd order interior
            slice1[axis] = slice(1, -1)
            slice2[axis] = slice(None, -2)
            slice3[axis] = slice(1, -1)
            slice4[axis] = slice(2, None)

            if uniform_spacing:
                out[slice1] = (f[slice4] - f[slice2]) / (2.0 * dx[i])
            else:
                dx1 = dx[i][0:-1]
                dx2 = dx[i][1:]
                a = -(dx2) / (dx1 * (dx1 + dx2))
                b = (dx2 - dx1) / (dx1 * dx2)
                c = dx1 / (dx2 * (dx1 + dx2))
                # fix the shape for broadcasting
                shape = np.ones(N, dtype=int)
                shape[axis] = -1
                a.shape = b.shape = c.shape = shape
                # 1D equivalent --
                # out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
                out[slice1] = a * f[slice2] + b * f[slice3] + c * f[slice4]

            # Numerical differentiation: 1st order edges
            if edge_order == 1:
                slice1[axis] = 0
                slice2[axis] = 1
                slice3[axis] = 0
                dx_0 = dx[i] if uniform_spacing else dx[i][0]
                # 1D equivalent -- out[0] = (y[1] - y[0]) / (x[1] - x[0])
                out[slice1] = (y[slice2] - y[slice3]) / dx_0

                slice1[axis] = -1
                slice2[axis] = -1
                slice3[axis] = -2
                dx_n = dx[i] if uniform_spacing else dx[i][-1]
                # 1D equivalent -- out[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
                out[slice1] = (y[slice2] - y[slice3]) / dx_n

            # Numerical differentiation: 2nd order edges
            else:
                slice1[axis] = 0
                slice2[axis] = 0
                slice3[axis] = 1
                slice4[axis] = 2
                if uniform_spacing:
                    a = -1.5 / dx[i]
                    b = 2.0 / dx[i]
                    c = -0.5 / dx[i]
                else:
                    dx1 = dx[i][0]
                    dx2 = dx[i][1]
                    a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                    b = (dx1 + dx2) / (dx1 * dx2)
                    c = -dx1 / (dx2 * (dx1 + dx2))
                # 1D equivalent -- out[0] = a * y[0] + b * y[1] + c * y[2]
                out[slice1] = a * y[slice2] + b * y[slice3] + c * y[slice4]

                slice1[axis] = -1
                slice2[axis] = -3
                slice3[axis] = -2
                slice4[axis] = -1
                if uniform_spacing:
                    a = 0.5 / dx[i]
                    b = -2.0 / dx[i]
                    c = 1.5 / dx[i]
                else:
                    dx1 = dx[i][-2]
                    dx2 = dx[i][-1]
                    a = (dx2) / (dx1 * (dx1 + dx2))
                    b = -(dx2 + dx1) / (dx1 * dx2)
                    c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
                # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
                out[slice1] = a * y[slice2] + b * y[slice3] + c * y[slice4]

            outvals.append(out)

            # reset the slice object in this dimension to ":"
            slice1[axis] = slice(None)
            slice2[axis] = slice(None)
            slice3[axis] = slice(None)
            slice4[axis] = slice(None)

        if len_axes == 1:
            return outvals[0]
        else:
            return outvals


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


# Type annotations stubs. See also / to be replaced by:
# https://github.com/numpy/numpy/issues/7370
# https://github.com/numpy/numpy-stubs/
DTypeLike = Union[np.dtype, str]


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
