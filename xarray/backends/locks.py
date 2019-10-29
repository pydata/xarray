import multiprocessing
import threading
import weakref
from typing import Any, MutableMapping, Optional

try:
    from dask.utils import SerializableLock
except ImportError:
    # no need to worry about serializing the lock
    SerializableLock = threading.Lock

try:
    from dask.distributed import Lock as DistributedLock
except ImportError:
    DistributedLock = None


# Locks used by multiple backends.
# Neither HDF5 nor the netCDF-C library are thread-safe.
HDF5_LOCK = SerializableLock()
NETCDFC_LOCK = SerializableLock()


_FILE_LOCKS: MutableMapping[Any, threading.Lock] = weakref.WeakValueDictionary()


def _get_threaded_lock(key):
    try:
        lock = _FILE_LOCKS[key]
    except KeyError:
        lock = _FILE_LOCKS[key] = threading.Lock()
    return lock


def _get_multiprocessing_lock(key):
    # TODO: make use of the key -- maybe use locket.py?
    # https://github.com/mwilliamson/locket.py
    del key  # unused
    return multiprocessing.Lock()


_LOCK_MAKERS = {
    None: _get_threaded_lock,
    "threaded": _get_threaded_lock,
    "multiprocessing": _get_multiprocessing_lock,
    "distributed": DistributedLock,
}


def _get_lock_maker(scheduler=None):
    """Returns an appropriate function for creating resource locks.

    Parameters
    ----------
    scheduler : str or None
        Dask scheduler being used.

    See Also
    --------
    dask.utils.get_scheduler_lock
    """
    return _LOCK_MAKERS[scheduler]


def _get_scheduler(get=None, collection=None) -> Optional[str]:
    """Determine the dask scheduler that is being used.

    None is returned if no dask scheduler is active.

    See also
    --------
    dask.base.get_scheduler
    """
    try:
        import dask  # noqa: F401
    except ImportError:
        return None

    actual_get = dask.base.get_scheduler(get, collection)

    try:
        from dask.distributed import Client

        if isinstance(actual_get.__self__, Client):
            return "distributed"
    except (ImportError, AttributeError):
        pass

    try:
        # As of dask=2.6, dask.multiprocessing requires cloudpickle to be installed
        # Dependency removed in https://github.com/dask/dask/pull/5511
        if actual_get is dask.multiprocessing.get:
            return "multiprocessing"
    except AttributeError:
        pass

    return "threaded"


def get_write_lock(key):
    """Get a scheduler appropriate lock for writing to the given resource.

    Parameters
    ----------
    key : str
        Name of the resource for which to acquire a lock. Typically a filename.

    Returns
    -------
    Lock object that can be used like a threading.Lock object.
    """
    scheduler = _get_scheduler()
    lock_maker = _get_lock_maker(scheduler)
    return lock_maker(key)


def acquire(lock, blocking=True):
    """Acquire a lock, possibly in a non-blocking fashion.

    Includes backwards compatibility hacks for old versions of Python, dask
    and dask-distributed.
    """
    if blocking:
        # no arguments needed
        return lock.acquire()
    elif DistributedLock is not None and isinstance(lock, DistributedLock):
        # distributed.Lock doesn't support the blocking argument yet:
        # https://github.com/dask/distributed/pull/2412
        return lock.acquire(timeout=0)
    else:
        # "blocking" keyword argument not supported for:
        # - threading.Lock on Python 2.
        # - dask.SerializableLock with dask v1.0.0 or earlier.
        # - multiprocessing.Lock calls the argument "block" instead.
        return lock.acquire(blocking)


class CombinedLock:
    """A combination of multiple locks.

    Like a locked door, a CombinedLock is locked if any of its constituent
    locks are locked.
    """

    def __init__(self, locks):
        self.locks = tuple(set(locks))  # remove duplicates

    def acquire(self, blocking=True):
        return all(acquire(lock, blocking=blocking) for lock in self.locks)

    def release(self):
        for lock in self.locks:
            lock.release()

    def __enter__(self):
        for lock in self.locks:
            lock.__enter__()

    def __exit__(self, *args):
        for lock in self.locks:
            lock.__exit__(*args)

    def locked(self):
        return any(lock.locked for lock in self.locks)

    def __repr__(self):
        return "CombinedLock(%r)" % list(self.locks)


class DummyLock:
    """DummyLock provides the lock API without any actual locking."""

    def acquire(self, blocking=True):
        pass

    def release(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def locked(self):
        return False


def combine_locks(locks):
    """Combine a sequence of locks into a single lock."""
    all_locks = []
    for lock in locks:
        if isinstance(lock, CombinedLock):
            all_locks.extend(lock.locks)
        elif lock is not None:
            all_locks.append(lock)

    num_locks = len(all_locks)
    if num_locks > 1:
        return CombinedLock(all_locks)
    elif num_locks == 1:
        return all_locks[0]
    else:
        return DummyLock()


def ensure_lock(lock):
    """Ensure that the given object is a lock."""
    if lock is None or lock is False:
        return DummyLock()
    return lock
