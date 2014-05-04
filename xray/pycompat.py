import sys

PY3 = sys.version_info[0] >= 3

if PY3:
    basestring = str
    def iteritems(d):
        return iter(d.items())
else:
    # Python 2
    basestring = basestring
    def iteritems(d):
        return d.iteritems()
