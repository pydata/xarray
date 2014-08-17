import sys

PY3 = sys.version_info[0] >= 3

if PY3:
    basestring = str
    unicode_type = str
    bytes_type = bytes
    def iteritems(d):
        return iter(d.items())
    def itervalues(d):
        return iter(d.values())
    range = range
    zip = zip
    from collections import OrderedDict
else:
    # Python 2
    basestring = basestring
    unicode_type = unicode
    bytes_type = str
    def iteritems(d):
        return d.iteritems()
    def itervalues(d):
        return d.itervalues()
    range = xrange
    from itertools import izip as zip
    try:
        from collections import OrderedDict
    except ImportError:
        from ordereddict import OrderedDict
