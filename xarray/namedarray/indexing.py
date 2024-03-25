from xarray.namedarray.pycompat import integer_types

# https://github.com/python/mypy/issues/224
BASIC_INDEXING_TYPES = integer_types + (slice,)
