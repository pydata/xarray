"""Functions copied from unreleased versions of numpy.

See the NumPy license in the licenses directory.
"""
import numpy as np


try:
    from numpy import broadcast_to, stack
except ImportError: # pragma: no cover
    # these functions should arrive in numpy v1.10

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
        result_ndim = arrays[0].ndim + 1
        if not -result_ndim <= axis < result_ndim:
            raise IndexError('axis %r out of bounds [-%r, %r)'
                             % (axis, result_ndim, result_ndim))
        if axis < 0:
            axis += result_ndim
        sl = (slice(None),) * axis + (np.newaxis,)
        try:
            expanded_arrays = [arr[sl] for arr in arrays]
        except IndexError:
            msg = 'all the input arrays must have same number of dimensions'
            raise ValueError(msg)
        return np.concatenate(expanded_arrays, axis=axis)
