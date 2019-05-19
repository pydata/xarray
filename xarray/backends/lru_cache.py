import collections
import collections.abc
import threading


class LRUCache(collections.abc.MutableMapping):
    """Thread-safe LRUCache based on an OrderedDict.

    All dict operations (__getitem__, __setitem__, __contains__) update the
    priority of the relevant key and take O(1) time. The dict is iterated over
    in order from the oldest to newest key, which means that a complete pass
    over the dict should not affect the order of any entries.

    When a new item is set and the maximum size of the cache is exceeded, the
    oldest item is dropped and called with ``on_evict(key, value)``.

    The ``maxsize`` property can be used to view or adjust the capacity of
    the cache, e.g., ``cache.maxsize = new_size``.
    """
    def __init__(self, maxsize, on_evict=None):
        """
        Parameters
        ----------
        maxsize : int
            Integer maximum number of items to hold in the cache.
        on_evict: callable, optional
            Function to call like ``on_evict(key, value)`` when items are
            evicted.
        """
        if not isinstance(maxsize, int):
            raise TypeError('maxsize must be an integer')
        if maxsize < 0:
            raise ValueError('maxsize must be non-negative')
        self._maxsize = maxsize
        self._on_evict = on_evict
        self._cache = collections.OrderedDict()
        self._lock = threading.RLock()

    def __getitem__(self, key):
        # record recent use of the key by moving it to the front of the list
        with self._lock:
            value = self._cache[key]
            self._cache.move_to_end(key)
            return value

    def _enforce_size_limit(self, capacity):
        """Shrink the cache if necessary, evicting the oldest items."""
        while len(self._cache) > capacity:
            key, value = self._cache.popitem(last=False)
            if self._on_evict is not None:
                self._on_evict(key, value)

    def __setitem__(self, key, value):
        with self._lock:
            if key in self._cache:
                # insert the new value at the end
                del self._cache[key]
                self._cache[key] = value
            elif self._maxsize:
                # make room if necessary
                self._enforce_size_limit(self._maxsize - 1)
                self._cache[key] = value
            elif self._on_evict is not None:
                # not saving, immediately evict
                self._on_evict(key, value)

    def __delitem__(self, key):
        del self._cache[key]

    def __iter__(self):
        # create a list, so accessing the cache during iteration cannot change
        # the iteration order
        return iter(list(self._cache))

    def __len__(self):
        return len(self._cache)

    @property
    def maxsize(self):
        """Maximum number of items can be held in the cache."""
        return self._maxsize

    @maxsize.setter
    def maxsize(self, size):
        """Resize the cache, evicting the oldest items if necessary."""
        if size < 0:
            raise ValueError('maxsize must be non-negative')
        with self._lock:
            self._enforce_size_limit(size)
            self._maxsize = size
