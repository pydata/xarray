from __future__ import absolute_import, division, print_function

from distutils.version import LooseVersion

import numpy as np

if LooseVersion(np.__version__) >= LooseVersion('1.12'):
    as_strided = np.lib.stride_tricks.as_strided
else:
    def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
        array = np.lib.stride_tricks.as_strided(x, shape, strides, subok)
        array.setflags(write=writeable)
        return array


try:
    from numpy import nancumsum, nancumprod, flip
except ImportError:  # pragma: no cover
    # Code copied from newer versions of NumPy (v1.12).
    # Used under the terms of NumPy's license, see licenses/NUMPY_LICENSE.

    def _replace_nan(a, val):
        """
        If `a` is of inexact type, make a copy of `a`, replace NaNs with
        the `val` value, and return the copy together with a boolean mask
        marking the locations where NaNs were present. If `a` is not of
        inexact type, do nothing and return `a` together with a mask of None.

        Note that scalars will end up as array scalars, which is important
        for using the result as the value of the out argument in some
        operations.

        Parameters
        ----------
        a : array-like
            Input array.
        val : float
            NaN values are set to val before doing the operation.

        Returns
        -------
        y : ndarray
            If `a` is of inexact type, return a copy of `a` with the NaNs
            replaced by the fill value, otherwise return `a`.
        mask: {bool, None}
            If `a` is of inexact type, return a boolean mask marking locations
            of NaNs, otherwise return None.

        """
        is_new = not isinstance(a, np.ndarray)
        if is_new:
            a = np.array(a)
        if not issubclass(a.dtype.type, np.inexact):
            return a, None
        if not is_new:
            # need copy
            a = np.array(a, subok=True)

        mask = np.isnan(a)
        np.copyto(a, val, where=mask)
        return a, mask

    def nancumsum(a, axis=None, dtype=None, out=None):
        """
        Return the cumulative sum of array elements over a given axis treating
        Not a Numbers (NaNs) as zero.  The cumulative sum does not change when
        NaNs are encountered and leading NaNs are replaced by zeros.

        Zeros are returned for slices that are all-NaN or empty.

        .. versionadded:: 1.12.0

        Parameters
        ----------
        a : array_like
            Input array.
        axis : int, optional
            Axis along which the cumulative sum is computed. The default
            (None) is to compute the cumsum over the flattened array.
        dtype : dtype, optional
            Type of the returned array and of the accumulator in which the
            elements are summed.  If `dtype` is not specified, it defaults
            to the dtype of `a`, unless `a` has an integer dtype with a
            precision less than that of the default platform integer.  In
            that case, the default platform integer is used.
        out : ndarray, optional
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output
            but the type will be cast if necessary. See `doc.ufuncs`
            (Section "Output arguments") for more details.

        Returns
        -------
        nancumsum : ndarray.
            A new array holding the result is returned unless `out` is
            specified, in which it is returned. The result has the same
            size as `a`, and the same shape as `a` if `axis` is not None
            or `a` is a 1-d array.

        See Also
        --------
        numpy.cumsum : Cumulative sum across array propagating NaNs.
        isnan : Show which elements are NaN.

        Examples
        --------
        >>> np.nancumsum(1)
        array([1])
        >>> np.nancumsum([1])
        array([1])
        >>> np.nancumsum([1, np.nan])
        array([ 1.,  1.])
        >>> a = np.array([[1, 2], [3, np.nan]])
        >>> np.nancumsum(a)
        array([ 1.,  3.,  6.,  6.])
        >>> np.nancumsum(a, axis=0)
        array([[ 1.,  2.],
               [ 4.,  2.]])
        >>> np.nancumsum(a, axis=1)
        array([[ 1.,  3.],
               [ 3.,  3.]])

        """
        a, mask = _replace_nan(a, 0)
        return np.cumsum(a, axis=axis, dtype=dtype, out=out)

    def nancumprod(a, axis=None, dtype=None, out=None):
        """
        Return the cumulative product of array elements over a given axis
        treating Not a Numbers (NaNs) as one.  The cumulative product does not
        change when NaNs are encountered and leading NaNs are replaced by ones.

        Ones are returned for slices that are all-NaN or empty.

        .. versionadded:: 1.12.0

        Parameters
        ----------
        a : array_like
            Input array.
        axis : int, optional
            Axis along which the cumulative product is computed.  By default
            the input is flattened.
        dtype : dtype, optional
            Type of the returned array, as well as of the accumulator in which
            the elements are multiplied.  If *dtype* is not specified, it
            defaults to the dtype of `a`, unless `a` has an integer dtype with
            a precision less than that of the default platform integer.  In
            that case, the default platform integer is used instead.
        out : ndarray, optional
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output
            but the type of the resulting values will be cast if necessary.

        Returns
        -------
        nancumprod : ndarray
            A new array holding the result is returned unless `out` is
            specified, in which case it is returned.

        See Also
        --------
        numpy.cumprod : Cumulative product across array propagating NaNs.
        isnan : Show which elements are NaN.

        Examples
        --------
        >>> np.nancumprod(1)
        array([1])
        >>> np.nancumprod([1])
        array([1])
        >>> np.nancumprod([1, np.nan])
        array([ 1.,  1.])
        >>> a = np.array([[1, 2], [3, np.nan]])
        >>> np.nancumprod(a)
        array([ 1.,  2.,  6.,  6.])
        >>> np.nancumprod(a, axis=0)
        array([[ 1.,  2.],
               [ 3.,  2.]])
        >>> np.nancumprod(a, axis=1)
        array([[ 1.,  2.],
               [ 3.,  3.]])

        """
        a, mask = _replace_nan(a, 1)
        return np.cumprod(a, axis=axis, dtype=dtype, out=out)

    def flip(m, axis):
        """
        Reverse the order of elements in an array along the given axis.

        The shape of the array is preserved, but the elements are reordered.

        .. versionadded:: 1.12.0

        Parameters
        ----------
        m : array_like
            Input array.
        axis : integer
            Axis in array, which entries are reversed.


        Returns
        -------
        out : array_like
            A view of `m` with the entries of axis reversed.  Since a view is
            returned, this operation is done in constant time.

        See Also
        --------
        flipud : Flip an array vertically (axis=0).
        fliplr : Flip an array horizontally (axis=1).

        Notes
        -----
        flip(m, 0) is equivalent to flipud(m).
        flip(m, 1) is equivalent to fliplr(m).
        flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at index n.

        Examples
        --------
        >>> A = np.arange(8).reshape((2,2,2))
        >>> A
        array([[[0, 1],
                [2, 3]],

               [[4, 5],
                [6, 7]]])

        >>> flip(A, 0)
        array([[[4, 5],
                [6, 7]],

               [[0, 1],
                [2, 3]]])

        >>> flip(A, 1)
        array([[[2, 3],
                [0, 1]],

               [[6, 7],
                [4, 5]]])

        >>> A = np.random.randn(3,4,5)
        >>> np.all(flip(A,2) == A[:,:,::-1,...])
        True
        """
        if not hasattr(m, 'ndim'):
            m = np.asarray(m)
        indexer = [slice(None)] * m.ndim
        try:
            indexer[axis] = slice(None, None, -1)
        except IndexError:
            raise ValueError("axis=%i is invalid for the %i-dimensional "
                             "input array" % (axis, m.ndim))
        return m[tuple(indexer)]

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
        return np.in1d(element, test_elements, assume_unique=assume_unique,
                       invert=invert).reshape(element.shape)
