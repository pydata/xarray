import operator
from collections import OrderedDict

import numpy as np


def expanded_indexer(key, ndim):
    """Given a key for indexing an ndarray, return an equivalent
    key which is a tuple with length equal to the number of dimensions
    """
    if not isinstance(key, tuple):
        key = (key,)
    new_key = [slice(None)] * ndim
    new_key[:len(key)] = key
    return tuple(new_key)


def safe_merge(*dicts, **kwargs):
    """Merge any number of dictionaries into a new OrderedDict

    Raises ValueError if dictionaries have non-compatible values for any key,
    where compatibility is determined by the `compat` function.

    Parameters
    ----------
    *dicts : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equality.

    Returns
    -------
    merged : OrderedDict
        Merged contents.
    """
    compat = kwargs.pop('compat', operator.eq)
    merged = OrderedDict()
    for d in dicts:
        for k, v in d.iteritems():
            if k in merged and not compat(v, merged[k]):
                raise ValueError('cannot override values with safe_merge')
            merged[k] = v
    return merged


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
