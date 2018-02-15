from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from distutils.version import LooseVersion


if LooseVersion(np.__version__) >= LooseVersion('1.12'):
    as_strided = np.lib.stride_tricks.as_strided
else:
    def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
        array = np.lib.stride_tricks.as_strided(x, shape, strides, subok)
        array.setflags(write=writeable)
        return array


if LooseVersion(np.__version__) >= LooseVersion('1.13'):
    nanmin = np.nanmin
    nanmax = np.nanmax
    nansum = np.nansum

else:
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
            If `a` is of inexact type, return a boolean mask marking locations of
            NaNs, otherwise return None.
        """
        a = np.array(a, subok=True, copy=True)

        if a.dtype == np.object_:
            # object arrays do not support `isnan` (gh-9009), so make a guess
            mask = a != a
        elif issubclass(a.dtype.type, np.inexact):
            mask = np.isnan(a)
        else:
            mask = None

        if mask is not None:
            np.copyto(a, val, where=mask)

        return a, mask

    def nanmin(a, axis=None, out=None, keepdims=np._NoValue):
        """
        Return minimum of an array or minimum along an axis, ignoring any NaNs.
        When all-NaN slices are encountered a ``RuntimeWarning`` is raised and
        Nan is returned for that slice.
        Parameters
        ----------
        a : array_like
            Array containing numbers whose minimum is desired. If `a` is not an
            array, a conversion is attempted.
        axis : int, optional
            Axis along which the minimum is computed. The default is to compute
            the minimum of the flattened array.
        out : ndarray, optional
            Alternate output array in which to place the result.  The default
            is ``None``; if provided, it must have the same shape as the
            expected output, but the type will be cast if necessary.  See
            `doc.ufuncs` for details.
            .. versionadded:: 1.8.0
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.
            If the value is anything but the default, then
            `keepdims` will be passed through to the `min` method
            of sub-classes of `ndarray`.  If the sub-classes methods
            does not implement `keepdims` any exceptions will be raised.
            .. versionadded:: 1.8.0
        Returns
        -------
        nanmin : ndarray
            An array with the same shape as `a`, with the specified axis
            removed.  If `a` is a 0-d array, or if axis is None, an ndarray
            scalar is returned.  The same dtype as `a` is returned.
        See Also
        --------
        nanmax :
            The maximum value of an array along a given axis, ignoring any NaNs.
        amin :
            The minimum value of an array along a given axis, propagating any NaNs.
        fmin :
            Element-wise minimum of two arrays, ignoring any NaNs.
        minimum :
            Element-wise minimum of two arrays, propagating any NaNs.
        isnan :
            Shows which elements are Not a Number (NaN).
        isfinite:
            Shows which elements are neither NaN nor infinity.
        amax, fmax, maximum
        Notes
        -----
        NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
        (IEEE 754). This means that Not a Number is not equivalent to infinity.
        Positive infinity is treated as a very large number and negative
        infinity is treated as a very small (i.e. negative) number.
        If the input has a integer type the function is equivalent to np.min.
        Examples
        --------
        >>> a = np.array([[1, 2], [3, np.nan]])
        >>> np.nanmin(a)
        1.0
        >>> np.nanmin(a, axis=0)
        array([ 1.,  2.])
        >>> np.nanmin(a, axis=1)
        array([ 1.,  3.])
        When positive infinity and negative infinity are present:
        >>> np.nanmin([1, 2, np.nan, np.inf])
        1.0
        >>> np.nanmin([1, 2, np.nan, np.NINF])
        -inf
        """
        kwargs = {}
        if keepdims is not np._NoValue:
            kwargs['keepdims'] = keepdims
        if type(a) is np.ndarray and a.dtype != np.object_:
            # Fast, but not safe for subclasses of ndarray, or object arrays,
            # which do not implement isnan (gh-9009), or fmin correctly (gh-8975)
            res = np.fmin.reduce(a, axis=axis, out=out, **kwargs)
            if np.isnan(res).any():
                warnings.warn("All-NaN slice encountered",
                              RuntimeWarning, stacklevel=2)
        else:
            # Slow, but safe for subclasses of ndarray
            a, mask = _replace_nan(a, +np.inf)
            res = np.amin(a, axis=axis, out=out, **kwargs)
            if mask is None:
                return res

            # Check for all-NaN axis
            mask = np.all(mask, axis=axis, **kwargs)
            if np.any(mask):
                res = _copyto(res, np.nan, mask)
                warnings.warn("All-NaN axis encountered",
                              RuntimeWarning, stacklevel=2)
        return res

    def nanmax(a, axis=None, out=None, keepdims=np._NoValue):
        """
        Return the maximum of an array or maximum along an axis, ignoring any
        NaNs.  When all-NaN slices are encountered a ``RuntimeWarning`` is
        raised and NaN is returned for that slice.
        Parameters
        ----------
        a : array_like
            Array containing numbers whose maximum is desired. If `a` is not an
            array, a conversion is attempted.
        axis : int, optional
            Axis along which the maximum is computed. The default is to compute
            the maximum of the flattened array.
        out : ndarray, optional
            Alternate output array in which to place the result.  The default
            is ``None``; if provided, it must have the same shape as the
            expected output, but the type will be cast if necessary.  See
            `doc.ufuncs` for details.
            .. versionadded:: 1.8.0
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.
            If the value is anything but the default, then
            `keepdims` will be passed through to the `max` method
            of sub-classes of `ndarray`.  If the sub-classes methods
            does not implement `keepdims` any exceptions will be raised.
            .. versionadded:: 1.8.0
        Returns
        -------
        nanmax : ndarray
            An array with the same shape as `a`, with the specified axis removed.
            If `a` is a 0-d array, or if axis is None, an ndarray scalar is
            returned.  The same dtype as `a` is returned.
        See Also
        --------
        nanmin :
            The minimum value of an array along a given axis, ignoring any NaNs.
        amax :
            The maximum value of an array along a given axis, propagating any NaNs.
        fmax :
            Element-wise maximum of two arrays, ignoring any NaNs.
        maximum :
            Element-wise maximum of two arrays, propagating any NaNs.
        isnan :
            Shows which elements are Not a Number (NaN).
        isfinite:
            Shows which elements are neither NaN nor infinity.
        amin, fmin, minimum
        Notes
        -----
        NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
        (IEEE 754). This means that Not a Number is not equivalent to infinity.
        Positive infinity is treated as a very large number and negative
        infinity is treated as a very small (i.e. negative) number.
        If the input has a integer type the function is equivalent to np.max.
        Examples
        --------
        >>> a = np.array([[1, 2], [3, np.nan]])
        >>> np.nanmax(a)
        3.0
        >>> np.nanmax(a, axis=0)
        array([ 3.,  2.])
        >>> np.nanmax(a, axis=1)
        array([ 2.,  3.])
        When positive infinity and negative infinity are present:
        >>> np.nanmax([1, 2, np.nan, np.NINF])
        2.0
        >>> np.nanmax([1, 2, np.nan, np.inf])
        inf
        """
        kwargs = {}
        if keepdims is not np._NoValue:
            kwargs['keepdims'] = keepdims
        if type(a) is np.ndarray and a.dtype != np.object_:
            # Fast, but not safe for subclasses of ndarray, or object arrays,
            # which do not implement isnan (gh-9009), or fmax correctly (gh-8975)
            res = np.fmax.reduce(a, axis=axis, out=out, **kwargs)
            if np.isnan(res).any():
                warnings.warn("All-NaN slice encountered",
                              RuntimeWarning, stacklevel=2)
        else:
            # Slow, but safe for subclasses of ndarray
            a, mask = _replace_nan(a, -np.inf)
            res = np.amax(a, axis=axis, out=out, **kwargs)
            if mask is None:
                return res

            # Check for all-NaN axis
            mask = np.all(mask, axis=axis, **kwargs)
            if np.any(mask):
                res = _copyto(res, np.nan, mask)
                warnings.warn("All-NaN axis encountered",
                              RuntimeWarning, stacklevel=2)
        return res

    def nansum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        """
        Return the sum of array elements over a given axis treating Not a
        Numbers (NaNs) as zero.
        In NumPy versions <= 1.8.0 Nan is returned for slices that are all-NaN or
        empty. In later versions zero is returned.
        Parameters
        ----------
        a : array_like
            Array containing numbers whose sum is desired. If `a` is not an
            array, a conversion is attempted.
        axis : int, optional
            Axis along which the sum is computed. The default is to compute the
            sum of the flattened array.
        dtype : data-type, optional
            The type of the returned array and of the accumulator in which the
            elements are summed.  By default, the dtype of `a` is used.  An
            exception is when `a` has an integer type with less precision than
            the platform (u)intp. In that case, the default will be either
            (u)int32 or (u)int64 depending on whether the platform is 32 or 64
            bits. For inexact inputs, dtype must be inexact.
            .. versionadded:: 1.8.0
        out : ndarray, optional
            Alternate output array in which to place the result.  The default
            is ``None``. If provided, it must have the same shape as the
            expected output, but the type will be cast if necessary.  See
            `doc.ufuncs` for details. The casting of NaN to integer can yield
            unexpected results.
            .. versionadded:: 1.8.0
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.
            If the value is anything but the default, then
            `keepdims` will be passed through to the `mean` or `sum` methods
            of sub-classes of `ndarray`.  If the sub-classes methods
            does not implement `keepdims` any exceptions will be raised.
            .. versionadded:: 1.8.0
        Returns
        -------
        nansum : ndarray.
            A new array holding the result is returned unless `out` is
            specified, in which it is returned. The result has the same
            size as `a`, and the same shape as `a` if `axis` is not None
            or `a` is a 1-d array.
        See Also
        --------
        numpy.sum : Sum across array propagating NaNs.
        isnan : Show which elements are NaN.
        isfinite: Show which elements are not NaN or +/-inf.
        Notes
        -----
        If both positive and negative infinity are present, the sum will be Not
        A Number (NaN).
        Examples
        --------
        >>> np.nansum(1)
        1
        >>> np.nansum([1])
        1
        >>> np.nansum([1, np.nan])
        1.0
        >>> a = np.array([[1, 1], [1, np.nan]])
        >>> np.nansum(a)
        3.0
        >>> np.nansum(a, axis=0)
        array([ 2.,  1.])
        >>> np.nansum([1, np.nan, np.inf])
        inf
        >>> np.nansum([1, np.nan, np.NINF])
        -inf
        >>> np.nansum([1, np.nan, np.inf, -np.inf]) # both +/- infinity present
        nan
        """
        a, mask = _replace_nan(a, 0)
        return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


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
