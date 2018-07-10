import contextlib
import threading

from .lru_cache import LRUCache


# Global cache for storing open files.
FILE_CACHE = LRUCache(512, on_evict=lambda k, v: v.close())

# TODO(shoyer): add an option (xarray.set_options) for resizing the cache.
# Note: the cache has a minimum size of one.


class FileManager(object):
    """Wrapper for automatically opening and closing file objects.

    Unlike files, FileManager objects can be safely pickled and passed between
    processes. They should be explicitly closed to release resources, but
    a per-process least-recently-used cache for open files ensures that you can
    safely create arbitrarily large numbers of FileManager objects.

    Example usage:

        manager = FileManager(open, 'example.txt', mode='w')
        with manager.acquire() as f:
            f.write(...)
        manager.close()
    """

    def __init__(self, opener, *args, **kwargs):
        """Initialize a FileManager.

        Parameters
        ----------
        opener : callable
            Function that when called like ``opener(*args, **kwargs)`` returns
            an open file object. The file object must implement a ``close()``
            method.
        *args
            Positional arguments for opener. A ``mode`` argument should be
            provided as a keyword argument (see below).
        **kwargs
            Keyword arguments for opener. The keyword argument ``mode`` has
            special handling if it is provided with a value of 'w': on all
            calls after the first, it is changed to 'a' instead to avoid
            overriding the newly created file. All argument values must be
            hashable.
        """
        self._opener = opener
        self._args = args
        self._kwargs = kwargs
        self._key = self._make_key()
        self._lock = threading.RLock()

    def _make_key(self):
        return _make_key(self._opener, self._args, self._kwargs)

    @contextlib.contextmanager
    def acquire(self):
        """Context manager for acquiring a file object.

        A new file is only opened if it has expired from the
        least-recently-used cache.

        This method uses a reentrant lock, which ensures that it is
        thread-safe. You can safely acquire a file in multiple threads at the
        same time, as long as the underlying file object is thread-safe.

        Yields
        ------
        Open file object, as returned by ``opener(*args, **kwargs)``.
        """
        with self._lock:
            try:
                file = FILE_CACHE[self._key]
            except KeyError:
                file = self._opener(*self._args, **self._kwargs)
                if self._kwargs.get('mode') == 'w':
                    # ensure file doesn't get overriden when opened again
                    self._kwargs['mode'] = 'a'
                    self._key = self._make_key()
                FILE_CACHE[self._key] = file
        yield file

    def close(self):
        """Explicitly close any associated file object (if necessary)."""
        file = FILE_CACHE.pop(self._key, default=None)
        if file is not None:
            file.close()

    def __getstate__(self):
        """State for pickling."""
        return (self._opener, self._args, self._kwargs)

    def __setstate__(self, state):
        """Restore from a pickle."""
        opener, args, kwargs = state
        self.__init__(opener, *args, **kwargs)


class _HashedSequence(list):
    """Speedup repeated look-ups by caching hash values.

    Based on what Python uses internally in functools.lru_cache.

    Python doesn't perform this optimization automatically:
    https://bugs.python.org/issue1462796
    """

    def __init__(self, tuple_value):
        self[:] = tuple_value
        self.hashvalue = hash(tuple_value)

    def __hash__(self):
        return self.hashvalue


def _make_key(opener, args, kwargs):
    """Make a key for caching files in the LRU cache."""
    value = (opener, args, tuple(sorted(kwargs.items())))
    return _HashedSequence(value)
