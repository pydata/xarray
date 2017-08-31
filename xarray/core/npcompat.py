from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

import numpy as np

try:
    from numpy import (broadcast_to, stack, nanprod, nancumsum, nancumprod,
                       moveaxis)
except ImportError:  # pragma: no cover
    # Code copied from newer versions of NumPy (v1.10 to v1.12).
    # Used under the terms of NumPy's license, see licenses/NUMPY_LICENSE.

    try:
        from numpy.core.multiarray import normalize_axis_index
    except ImportError:
        def normalize_axis_index(axis, ndim, msg_prefix=None):
            """ In house version of normalize_axis_index."""
            if axis < -ndim and ndim <= axis:
                msg = 'axis {0:d} is out of bounds for array of dimension {1:d}'.format(axis, ndim)
                if msg_prefix:
                    msg = msg_prefix + msg
                # Note: original normalize_axis_index raises AxisError
                raise IndexError(msg)

            if axis < 0:
                return axis + ndim
            return axis

    def _maybe_view_as_subclass(original_array, new_array):
        if type(original_array) is not type(new_array):
            # if input was an ndarray subclass and subclasses were OK,
            # then view the result as that subclass.
            new_array = new_array.view(type=type(original_array))
            # Since we have done something akin to a view from original_array, we
            # should let the subclass finalize (if it has it implemented, i.e., is
            # not None).
            if new_array.__array_finalize__:
                new_array.__array_finalize__(original_array)
        return new_array

    def _broadcast_to(array, shape, subok, readonly):
        shape = tuple(shape) if np.iterable(shape) else (shape,)
        array = np.array(array, copy=False, subok=subok)
        if not shape and array.shape:
            raise ValueError('cannot broadcast a non-scalar to a scalar array')
        if any(size < 0 for size in shape):
            raise ValueError('all elements of broadcast shape must be non-'
                             'negative')
        broadcast = np.nditer(
            (array,), flags=['multi_index', 'zerosize_ok', 'refs_ok'],
            op_flags=['readonly'], itershape=shape, order='C').itviews[0]
        result = _maybe_view_as_subclass(array, broadcast)
        if not readonly and array.flags.writeable:
            result.flags.writeable = True
        return result

    def broadcast_to(array, shape, subok=False):
        """Broadcast an array to a new shape.

        Parameters
        ----------
        array : array_like
            The array to broadcast.
        shape : tuple
            The shape of the desired array.
        subok : bool, optional
            If True, then sub-classes will be passed-through, otherwise
            the returned array will be forced to be a base-class array (default).

        Returns
        -------
        broadcast : array
            A readonly view on the original array with the given shape. It is
            typically not contiguous. Furthermore, more than one element of a
            broadcasted array may refer to a single memory location.

        Raises
        ------
        ValueError
            If the array is not compatible with the new shape according to NumPy's
            broadcasting rules.

        Examples
        --------
        >>> x = np.array([1, 2, 3])
        >>> np.broadcast_to(x, (3, 3))
        array([[1, 2, 3],
               [1, 2, 3],
               [1, 2, 3]])
        """
        return _broadcast_to(array, shape, subok=subok, readonly=True)

    def stack(arrays, axis=0):
        """
        Join a sequence of arrays along a new axis.

        .. versionadded:: 1.10.0

        Parameters
        ----------
        arrays : sequence of ndarrays
            Each array must have the same shape.
        axis : int, optional
            The axis along which the arrays will be stacked.

        Returns
        -------
        stacked : ndarray
            The stacked array has one more dimension than the input arrays.
        See Also
        --------
        concatenate : Join a sequence of arrays along an existing axis.
        split : Split array into a list of multiple sub-arrays of equal size.

        Examples
        --------
        >>> arrays = [np.random.randn(3, 4) for _ in range(10)]
        >>> np.stack(arrays, axis=0).shape
        (10, 3, 4)

        >>> np.stack(arrays, axis=1).shape
        (3, 10, 4)

        >>> np.stack(arrays, axis=2).shape
        (3, 4, 10)

        >>> a = np.array([1, 2, 3])
        >>> b = np.array([2, 3, 4])
        >>> np.stack((a, b))
        array([[1, 2, 3],
               [2, 3, 4]])

        >>> np.stack((a, b), axis=-1)
        array([[1, 2],
               [2, 3],
               [3, 4]])
        """
        arrays = [np.asanyarray(arr) for arr in arrays]
        if not arrays:
            raise ValueError('need at least one array to stack')

        shapes = set(arr.shape for arr in arrays)
        if len(shapes) != 1:
            raise ValueError('all input arrays must have the same shape')

        result_ndim = arrays[0].ndim + 1
        if not -result_ndim <= axis < result_ndim:
            msg = 'axis {0} out of bounds [-{1}, {1})'.format(axis, result_ndim)
            raise IndexError(msg)
        if axis < 0:
            axis += result_ndim

        sl = (slice(None),) * axis + (np.newaxis,)
        expanded_arrays = [arr[sl] for arr in arrays]
        return np.concatenate(expanded_arrays, axis=axis)

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

    def nanprod(a, axis=None, dtype=None, out=None, keepdims=0):
        """
        Return the product of array elements over a given axis treating Not a
        Numbers (NaNs) as zero.

        One is returned for slices that are all-NaN or empty.

        .. versionadded:: 1.10.0

        Parameters
        ----------
        a : array_like
            Array containing numbers whose sum is desired. If `a` is not an
            array, a conversion is attempted.
        axis : int, optional
            Axis along which the product is computed. The default is to compute
            the product of the flattened array.
        dtype : data-type, optional
            The type of the returned array and of the accumulator in which the
            elements are summed.  By default, the dtype of `a` is used.  An
            exception is when `a` has an integer type with less precision than
            the platform (u)intp. In that case, the default will be either
            (u)int32 or (u)int64 depending on whether the platform is 32 or 64
            bits. For inexact inputs, dtype must be inexact.
        out : ndarray, optional
            Alternate output array in which to place the result.  The default
            is ``None``. If provided, it must have the same shape as the
            expected output, but the type will be cast if necessary.  See
            `doc.ufuncs` for details. The casting of NaN to integer can yield
            unexpected results.
        keepdims : bool, optional
            If True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will
            broadcast correctly against the original `arr`.

        Returns
        -------
        y : ndarray or numpy scalar

        See Also
        --------
        numpy.prod : Product across array propagating NaNs.
        isnan : Show which elements are NaN.

        Notes
        -----
        Numpy integer arithmetic is modular. If the size of a product exceeds
        the size of an integer accumulator, its value will wrap around and the
        result will be incorrect. Specifying ``dtype=double`` can alleviate
        that problem.

        Examples
        --------
        >>> np.nanprod(1)
        1
        >>> np.nanprod([1])
        1
        >>> np.nanprod([1, np.nan])
        1.0
        >>> a = np.array([[1, 2], [3, np.nan]])
        >>> np.nanprod(a)
        6.0
        >>> np.nanprod(a, axis=0)
        array([ 3.,  2.])

        """
        a, mask = _replace_nan(a, 1)
        return np.prod(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

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

    def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
        """
        Normalizes an axis argument into a tuple of non-negative integer axes.

        This handles shorthands such as ``1`` and converts them to ``(1,)``,
        as well as performing the handling of negative indices covered by
        `normalize_axis_index`.

        By default, this forbids axes from being specified multiple times.

        Used internally by multi-axis-checking logic.

        .. versionadded:: 1.13.0

        Parameters
        ----------
        axis : int, iterable of int
            The un-normalized index or indices of the axis.
        ndim : int
            The number of dimensions of the array that `axis` should be
            normalized against.
        argname : str, optional
            A prefix to put before the error message, typically the name of the
            argument.
        allow_duplicate : bool, optional
            If False, the default, disallow an axis from being specified twice.

        Returns
        -------
        normalized_axes : tuple of int
            The normalized axis index, such that `0 <= normalized_axis < ndim`

        Raises
        ------
        AxisError
            If any axis provided is out of range
        ValueError
            If an axis is repeated

        See also
        --------
        normalize_axis_index : normalizing a single scalar axis
        """
        try:
            axis = [operator.index(axis)]
        except TypeError:
            axis = tuple(axis)
        axis = tuple(normalize_axis_index(ax, ndim, argname) for ax in axis)
        if not allow_duplicate and len(set(axis)) != len(axis):
            if argname:
                raise ValueError('repeated axis in `{}` argument'.format(
                                                                    argname))
            else:
                raise ValueError('repeated axis')
        return axis

    def moveaxis(a, source, destination):
        """
        Move axes of an array to new positions.

        Other axes remain in their original order.

        .. versionadded::1.11.0

        Parameters
        ----------
        a : np.ndarray
            The array whose axes should be reordered.
        source : int or sequence of int
            Original positions of the axes to move. These must be unique.
        destination : int or sequence of int
            Destination positions for each of the original axes. These must
            also be unique.

        Returns
        -------
        result : np.ndarray
            Array with moved axes. This array is a view of the input array.

        See Also
        --------
        transpose: Permute the dimensions of an array.
        swapaxes: Interchange two axes of an array.

        Examples
        --------

        >>> x = np.zeros((3, 4, 5))
        >>> np.moveaxis(x, 0, -1).shape
        (4, 5, 3)
        >>> np.moveaxis(x, -1, 0).shape
        (5, 3, 4)

        These all achieve the same result:

        >>> np.transpose(x).shape
        (5, 4, 3)
        >>> np.swapaxes(x, 0, -1).shape
        (5, 4, 3)
        >>> np.moveaxis(x, [0, 1], [-1, -2]).shape
        (5, 4, 3)
        >>> np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
        (5, 4, 3)

        """
        try:
            # allow duck-array types if they define transpose
            transpose = a.transpose
        except AttributeError:
            a = np.asarray(a)
            transpose = a.transpose

        source = normalize_axis_tuple(source, a.ndim, 'source')
        destination = normalize_axis_tuple(destination, a.ndim, 'destination')
        if len(source) != len(destination):
            raise ValueError('`source` and `destination` arguments must have '
                             'the same number of elements')

        order = [n for n in range(a.ndim) if n not in source]

        for dest, src in sorted(zip(destination, source)):
            order.insert(dest, src)

        result = transpose(order)
        return result
