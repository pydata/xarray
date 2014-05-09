import sys

PY3 = sys.version_info[0] >= 3

if PY3:
    basestring = str
    unicode_type = str
    def iteritems(d):
        return iter(d.items())
    def itervalues(d):
        return iter(d.values())
    xrange = range
else:
    # Python 2
    basestring = basestring
    unicode_type = unicode
    def iteritems(d):
        return d.iteritems()
    def itervalues(d):
        return d.itervalues()
    xrange = xrange