import operator
from collections import OrderedDict

import numpy as np


def expanded_indexer(key, ndim):
    """Given a key for indexing an ndarray, return an equivalent key which is a
    tuple with length equal to the number of dimensions

    The expansion is done by replacing all `Ellipsis` items with the right
    number of full slices and then padding the key with full slices so that it
    reaches the appropriate dimensionality.
    """
    if not isinstance(key, tuple):
        # numpy treats non-tuple keys equivalent to tuples of length 1
        key = (key,)
    new_key = []
    # handling Ellipsis right is a little tricky, see:
    # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
    found_ellipsis = False
    for k in key:
        if k is Ellipsis:
            if not found_ellipsis:
                new_key.extend((ndim + 1 - len(key)) * [slice(None)])
                found_ellipsis = True
            else:
                new_key.append(slice(None))
        else:
            new_key.append(k)
    new_key.extend((ndim - len(new_key)) * [slice(None)])
    return tuple(new_key)


def orthogonal_indexer(key, shape):
    """Given a key for orthogonal array indexing, returns an equivalent key
    suitable for indexing a numpy.ndarray with fancy indexing
    """
    def expand_array(k, length):
        if isinstance(k, slice):
            return np.arange(k.start or 0, k.stop or length, k.step or 1)
        else:
            k = np.asarray(k)
            if k.ndim != 1:
                raise ValueError('orthogonal array indexing only supports '
                                 '1d arrays')
            return k
    # replace Ellipsis objects with slices
    key = list(expanded_indexer(key, len(shape)))
    # replace 1d arrays and slices with broadcast compatible arrays
    # note: we treat integers separately (instead of turning them into 1d
    # arrays) because integers (and only integers) collapse axes when used with
    # __getitem__
    non_int_keys = [n for n, k in enumerate(key) if not isinstance(k, int)]
    array_indexers = np.ix_(*(expand_array(key[n], shape[n])
                              for n in non_int_keys))
    for i, n in enumerate(non_int_keys):
        key[n] = array_indexers[i]
    return tuple(key)


def update_safety_check(first_dict, second_dict, compat=operator.eq):
    """Check the safety of updating one dictionary with another

    Raises ValueError if dictionaries have non-compatible values for any key,
    where compatibility is determined by the `compat` function.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        All items in the second dictionary are checked against for conflicts
        against items in the first dictionary.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equality.
    """
    for k, v in second_dict.iteritems():
        if k in first_dict and not compat(v, first_dict[k]):
            raise ValueError('unsafe to merge dictionaries without '
                             'overriding values')


def safe_update(first_dict, second_dict, compat=operator.eq):
    """Safely update a dictionary with another dictionary

    Raises ValueError if dictionaries have non-compatible values for any key,
    where compatibility is determined by the `compat` function.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge. The first dictionary is modified in place.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equality.
    """
    update_safety_check(first_dict, second_dict, compat=compat)
    first_dict.update(second_dict)


def safe_merge(first_dict, second_dict, compat=operator.eq):
    """Safely merge two dictionaries into a new OrderedDict

    Raises ValueError if dictionaries have non-compatible values for any key,
    where compatibility is determined by the `compat` function.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equality.

    Returns
    -------
    merged : OrderedDict
        Merged contents.
    """
    update_safety_check(first_dict, second_dict, compat=compat)
    new_dict = OrderedDict(first_dict)
    new_dict.update(second_dict)
    return new_dict


def variable_equal(v1, v2):
    """True if two objects have the same dimensions, attributes and data;
    otherwise False

    This function is necessary because `v1 == v2` does element-wise comparison
    (like numpy.ndarrays).
    """
    if (v1.dimensions == v2.dimensions
            and v1.attributes == v2.attributes):
        try:
            # if _data is identical, skip checking arrays by value
            if v1._data is v2._data:
                return True
        except AttributeError:
            # _data is not part of the public interface, so it's okay if its
            # missing
            pass
        return np.array_equal(v1.data, v2.data)
    else:
        return False


# class DisabledMixin(object):


class FrozenOrderedDict(OrderedDict):
    """A subclass of OrderedDict whose contents are frozen after initialization
    to prevent tampering
    """
    def __init__(self, *args, **kwds):
        # bypass the disabled __setitem__ method
        # initialize as an empty OrderedDict
        super(FrozenOrderedDict, self).__init__()
        # Capture arguments in an OrderedDict
        args_dict = OrderedDict(*args, **kwds)
        # Call __setitem__ of the superclass
        for (key, value) in args_dict.iteritems():
            super(FrozenOrderedDict, self).__setitem__(key, value)

    def _not_implemented(self, *args, **kwargs):
        raise TypeError('%s is immutable' % type(self).__name__)

    __setitem__ = __delitem__ = setdefault = update = pop = popitem = clear = \
        _not_implemented
