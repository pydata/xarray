import contextlib
import threading

from ..core import utils
from ..core.options import OPTIONS
from .lru_cache import LRUCache


# Global cache for storing open files.
FILE_CACHE = LRUCache(
    OPTIONS['file_cache_maxsize'], on_evict=lambda k, v: v.close())
assert FILE_CACHE.maxsize, 'file cache must be at least size one'


_DEFAULT_MODE = utils.ReprObject('<unused>')


class FileManager(object):
    """Manager for acquiring and closing a file object.

    Use FileManager subclasses (CachingFileManager in particular) on backend
    storage classes to automatically handle issues related to keeping track of
    many open files and transferring them between multiple processes.
    """

    def acquire(self):
        """Acquire the file object from this manager."""
        raise NotImplementedError

    def close(self, needs_lock=True):
        """Close the file object associated with this manager, if needed."""
        raise NotImplementedError


@contextlib.contextmanager
def optional_locking(lock, needs_lock):
    """Context manager for optionally acquiring a lock."""
    try:
        if needs_lock:
            lock.acquire()
        yield
    finally:
        if needs_lock:
            lock.release()


class CachingFileManager(FileManager):
    """Wrapper for automatically opening and closing file objects.

    Unlike files, CachingFileManager objects can be safely pickled and passed
    between processes. They should be explicitly closed to release resources,
    but a per-process least-recently-used cache for open files ensures that you
    can safely create arbitrarily large numbers of FileManager objects.

    Don't directly close files acquired from a FileManager. Instead, call
    FileManager.close(), which ensures that closed files are removed from the
    cache as well.

    Example usage:

        manager = FileManager(open, 'example.txt', mode='w')
        f = manager.acquire()
        f.write(...)
        manager.close()  # ensures file is closed

    Note that as long as previous files are still cached, acquiring a file
    multiple times from the same FileManager is essentially free:

        f1 = manager.acquire()
        f2 = manager.acquire()
        assert f1 is f2

    """

    def __init__(self, opener, *args, **keywords):
        """Initialize a FileManager.

        Parameters
        ----------
        opener : callable
            Function that when called like ``opener(*args, **kwargs)`` returns
            an open file object. The file object must implement a ``close()``
            method.
        *args
            Positional arguments for opener. A ``mode`` argument should be
            provided as a keyword argument (see below). All arguments must be
            hashable.
        mode : optional
            If provided, passed as a keyword argument to ``opener`` along with
            ``**kwargs``. ``mode='w' `` has special treatment: after the first
            call it is replaced by ``mode='a'`` in all subsequent function to
            avoid overriding the newly created file.
        kwargs : dict, optional
            Keyword arguments for opener, excluding ``mode``. All values must
            be hashable.
        lock : duck-compatible threading.Lock, optional
            Lock to use when modifying the cache inside acquire() and close().
            By default, uses a new threading.Lock() object. If set, this object
            should be pickleable.
        cache : MutableMapping, optional
            Mapping to use as a cache for open files. By default, uses xarray's
            global LRU file cache. Because ``cache`` typically points to a
            global variable and contains non-picklable file objects, an
            unpickled FileManager objects will be restored with the default
            cache.
        """
        # TODO: replace with real keyword arguments when we drop Python 2
        # support
        mode = keywords.pop('mode', _DEFAULT_MODE)
        kwargs = keywords.pop('kwargs', None)
        lock = keywords.pop('lock', None)
        cache = keywords.pop('cache', FILE_CACHE)
        if keywords:
            raise TypeError('FileManager() got unexpected keyword arguments: '
                            '%s' % list(keywords))

        self._opener = opener
        self._args = args
        self._mode = mode
        self._kwargs = {} if kwargs is None else dict(kwargs)
        self._default_lock = lock is None or lock is False
        self._lock = threading.Lock() if self._default_lock else lock
        self._cache = cache
        self._key = self._make_key()

    def _make_key(self):
        """Make a key for caching files in the LRU cache."""
        value = (self._opener,
                 self._args,
                 self._mode,
                 tuple(sorted(self._kwargs.items())))
        return _HashedSequence(value)

    def acquire(self, needs_lock=True):
        """Acquiring a file object from the manager.

        A new file is only opened if it has expired from the
        least-recently-used cache.

        This method uses a reentrant lock, which ensures that it is
        thread-safe. You can safely acquire a file in multiple threads at the
        same time, as long as the underlying file object is thread-safe.

        Returns
        -------
        An open file object, as returned by ``opener(*args, **kwargs)``.
        """
        with optional_locking(self._lock, needs_lock):
            try:
                file = self._cache[self._key]
            except KeyError:
                kwargs = self._kwargs
                if self._mode is not _DEFAULT_MODE:
                    kwargs = kwargs.copy()
                    kwargs['mode'] = self._mode
                file = self._opener(*self._args, **kwargs)
                if self._mode == 'w':
                    # ensure file doesn't get overriden when opened again
                    self._mode = 'a'
                    self._key = self._make_key()
                self._cache[self._key] = file
        return file

    def close(self, needs_lock=True):
        """Explicitly close any associated file object (if necessary)."""
        # TODO: remove needs_lock if/when we have a reentrant lock in
        # dask.distributed: https://github.com/dask/dask/issues/3832
        with optional_locking(self._lock, needs_lock):
            default = None
            file = self._cache.pop(self._key, default)
            if file is not None:
                file.close()

    def __del__(self):
        # remove files from the cache when garbage collection happens
        if self.lock.locked:
            raise RuntimeError('closing a file with an unreleased lock')
        self.close(needs_lock=False)

    def __getstate__(self):
        """State for pickling."""
        lock = None if self._default_lock else self._lock
        return (self._opener, self._args, self._mode, self._kwargs, lock)

    def __setstate__(self, state):
        """Restore from a pickle."""
        opener, args, mode, kwargs, lock = state
        self.__init__(opener, *args, mode=mode, kwargs=kwargs, lock=lock)


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


class DummyFileManager(FileManager):
    """FileManager that simply wraps an open file in the FileManager interface.
    """
    def __init__(self, value):
        self._value = value

    def acquire(self):
        return self._value

    def close(self, needs_lock=True):
        del needs_lock  # ignored
        self._value.close()
