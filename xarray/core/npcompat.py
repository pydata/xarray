from __future__ import absolute_import, division, print_function

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
        return np.in1d(element, test_elements, assume_unique=assume_unique,
                       invert=invert).reshape(element.shape)
