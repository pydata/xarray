from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

try:
    from numpy import nancumsum, nancumprod, flip, isclose
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
        
    def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
        """
        Returns a boolean array where two arrays are element-wise equal within a
        tolerance.
        The tolerance values are positive, typically very small numbers.  The
        relative difference (`rtol` * abs(`b`)) and the absolute difference
        `atol` are added together to compare against the absolute difference
        between `a` and `b`.
        Parameters
        ----------
        a, b : array_like
            Input arrays to compare.
        rtol : float
            The relative tolerance parameter (see Notes).
        atol : float
            The absolute tolerance parameter (see Notes).
        equal_nan : bool
            Whether to compare NaN's as equal.  If True, NaN's in `a` will be
            considered equal to NaN's in `b` in the output array.
        Returns
        -------
        y : array_like
            Returns a boolean array of where `a` and `b` are equal within the
            given tolerance. If both `a` and `b` are scalars, returns a single
            boolean value.
        See Also
        --------
        allclose
        Notes
        -----
        .. versionadded:: 1.7.0
        For finite values, isclose uses the following equation to test whether
        two floating point values are equivalent.
         absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
        The above equation is not symmetric in `a` and `b`, so that
        `isclose(a, b)` might be different from `isclose(b, a)` in
        some rare cases.
        Examples
        --------
        >>> np.isclose([1e10,1e-7], [1.00001e10,1e-8])
        array([True, False])
        >>> np.isclose([1e10,1e-8], [1.00001e10,1e-9])
        array([True, True])
        >>> np.isclose([1e10,1e-8], [1.0001e10,1e-9])
        array([False, True])
        >>> np.isclose([1.0, np.nan], [1.0, np.nan])
        array([True, False])
        >>> np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
        array([True, True])
        """
        def within_tol(x, y, atol, rtol):
            with np.errstate(invalid='ignore'):
                result = np.less_equal(abs(x-y), atol + rtol * abs(y))
            if np.isscalar(a) and np.isscalar(b):
                result = bool(result)
            return result

        x = np.array(a, copy=False, subok=True, ndmin=1)
        y = np.array(b, copy=False, subok=True, ndmin=1)

        # Make sure y is an inexact type to avoid bad behavior on abs(MIN_INT).
        # This will cause casting of x later. Also, make sure to allow subclasses
        # (e.g., for numpy.ma).
        dt = np.multiarray.result_type(y, 1.)
        y = np.array(y, dtype=dt, copy=False, subok=True)

        xfin = np.isfinite(x)
        yfin = np.isfinite(y)
        if all(xfin) and all(yfin):
            return within_tol(x, y, atol, rtol)
        else:
            finite = xfin & yfin
            cond = np.zeros_like(finite, subok=True)
            # Because we're using boolean indexing, x & y must be the same shape.
            # Ideally, we'd just do x, y = broadcast_arrays(x, y). It's in
            # lib.stride_tricks, though, so we can't import it here.
            x = x * np.ones_like(cond)
            y = y * np.ones_like(cond)
            # Avoid subtraction with infinite/nan values...
            cond[finite] = within_tol(x[finite], y[finite], atol, rtol)
            # Check for equality of infinite values...
            cond[~finite] = (x[~finite] == y[~finite])
            if equal_nan:
                # Make NaN == NaN
                both_nan = np.isnan(x) & np.isnan(y)
                cond[both_nan] = both_nan[both_nan]

            if np.isscalar(a) and np.isscalar(b):
                return bool(cond)
            else:
                return cond
