import numpy as np
import sys

PY3 = sys.version_info[0] >= 3

if PY3: # pragma: no cover
    basestring = str
    unicode_type = str
    bytes_type = bytes
    def iteritems(d):
        return iter(d.items())
    def itervalues(d):
        return iter(d.values())
    range = range
    zip = zip
    from functools import reduce
    import builtins
else: # pragma: no cover
    # Python 2
    basestring = basestring
    unicode_type = unicode
    bytes_type = str
    def iteritems(d):
        return d.iteritems()
    def itervalues(d):
        return d.itervalues()
    range = xrange
    from itertools import izip as zip, imap as map
    reduce = reduce
    import __builtin__ as builtins

try:
    from cyordereddict import OrderedDict
except ImportError: # pragma: no cover
    try:
        from collections import OrderedDict
    except ImportError:
        from ordereddict import OrderedDict

try:
    # solely for isinstance checks
    import dask.array
    dask_array_type = (dask.array.Array,)
except ImportError: # pragma: no cover
    dask_array_type = ()
