from __future__ import annotations

import multiprocessing
import sys
import threading
import uuid
import weakref
from collections.abc import Callable, Hashable, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeGuard
from weakref import WeakValueDictionary

from xarray.core.types import Lock

if TYPE_CHECKING:
    from distributed import Lock as DistributedLock


# SerializableLock is adapted from Dask:
# https://github.com/dask/dask/blob/74e898f0ec712e8317ba86cc3b9d18b6b9922be0/dask/utils.py#L1160-L1224
# Used under the terms of Dask's license, see licenses/DASK_LICENSE.
class SerializableLock(Lock):
    """A Serializable per-process Lock

    This wraps a normal ``threading.Lock`` object and satisfies the same
    interface.  However, this lock can also be serialized and sent to different
    processes.  It will not block concurrent operations between processes (for
    this you should look at ``dask.multiprocessing.Lock`` or ``locket.lock_file``
    but will consistently deserialize into the same lock.

    So if we make a lock in one process::

        lock = SerializableLock()

    And then send it over to another process multiple times::

        bytes = pickle.dumps(lock)
        a = pickle.loads(bytes)
        b = pickle.loads(bytes)

    Then the deserialized objects will operate as though they were the same
    lock, and collide as appropriate.

    This is useful for consistently protecting resources on a per-process
    level.

    The creation of locks is itself not threadsafe.
    """

    _locks: ClassVar[WeakValueDictionary[Hashable, threading.Lock]] = (
        WeakValueDictionary()
    )
    token: Hashable
    lock: threading.Lock

    def __init__(self, token: Hashable | None = None):
        self.token = token or str(uuid.uuid4())
        if self.token in SerializableLock._locks:
            self.lock = SerializableLock._locks[self.token]
        else:
            self.lock = threading.Lock()
            SerializableLock._locks[self.token] = self.lock

    def acquire(self, *args, **kwargs):
        return self.lock.acquire(*args, **kwargs)

    def release(self, *args, **kwargs):
        return self.lock.release(*args, **kwargs)

    def __enter__(self):
        self.lock.__enter__()

    def __exit__(self, *args):
        self.lock.__exit__(*args)

    def locked(self):
        return self.lock.locked()

    def __getstate__(self):
        return self.token

    def __setstate__(self, token):
        self.__init__(token)

    def __str__(self):
        return f"<{self.__class__.__name__}: {self.token}>"

    __repr__ = __str__


# Locks used by multiple backends.
# Neither HDF5 nor the netCDF-C library are thread-safe.
HDF5_LOCK = SerializableLock()
NETCDFC_LOCK = SerializableLock()


_FILE_LOCKS: MutableMapping[Any, threading.Lock] = weakref.WeakValueDictionary()


def _get_threaded_lock(key: str) -> threading.Lock:
    try:
        lock = _FILE_LOCKS[key]
    except KeyError:
        lock = _FILE_LOCKS[key] = threading.Lock()
    return lock


def _get_multiprocessing_lock(key: str) -> Lock:
    # TODO: make use of the key -- maybe use locket.py?
    # https://github.com/mwilliamson/locket.py
    del key  # unused
    return multiprocessing.Lock()


def _get_lock_maker(scheduler: str | None = None) -> Callable[..., Lock]:
    """Returns an appropriate function for creating resource locks.

    Parameters
    ----------
    scheduler : str or None
        Dask scheduler being used.

    See Also
    --------
    dask.utils.get_scheduler_lock
    """

    if scheduler is None or scheduler == "threaded":
        return _get_threaded_lock
    elif scheduler == "multiprocessing":
        return _get_multiprocessing_lock
    elif scheduler == "distributed":
        # Lazy import distributed since it is can add a significant
        # amount of time to import
        from dask.distributed import Lock as DistributedLock

        return DistributedLock
    else:
        raise KeyError(scheduler)


def get_dask_scheduler(get=None, collection=None) -> str | None:
    """Determine the dask scheduler that is being used.

    None is returned if no dask scheduler is active.

    See Also
    --------
    dask.base.get_scheduler
    """
    try:
        # Fix for bug caused by dask installation that doesn't involve the toolz library
        # Issue: 4164
        import dask
        from dask.base import get_scheduler

        actual_get = get_scheduler(get, collection)
    except ImportError:
        return None

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


def get_write_lock(key: str) -> Lock:
    """Get a scheduler appropriate lock for writing to the given resource.

    Parameters
    ----------
    key : str
        Name of the resource for which to acquire a lock. Typically a filename.

    Returns
    -------
    Lock object that can be used like a threading.Lock object.
    """
    scheduler = get_dask_scheduler()
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
    else:
        # "blocking" keyword argument not supported for:
        # - threading.Lock on Python 2.
        # - dask.SerializableLock with dask v1.0.0 or earlier.
        # - multiprocessing.Lock calls the argument "block" instead.
        # - dask.distributed.Lock uses the blocking argument as the first one
        return lock.acquire(blocking)


# Lock ordering
# -------------
# A thread holding lock A while waiting for lock B deadlocks with a thread
# holding B while waiting for A. The standard way to rule this out is a lock
# hierarchy: all locks are acquired in one global order, so nobody ever waits
# for a lock that comes before one it already holds. CombinedLock enforces this
# by sorting its locks with _lock_order_key.
#
# The key must identify the lock that is actually acquired, not the Python
# object wrapping it. Locks are pickled all the time: every dask task gets its
# own unpickled copy of the locks it uses, so two copies of the same lock are
# different objects with different ids. Ordering by id() made the order differ
# between tasks: two threads of a distributed worker could take HDF5_LOCK and
# the per-file write lock in opposite orders and deadlock.
#
# So the key has two parts:
#
# 1. A level, depending on the kind of lock. xarray combines the process-wide
#    library locks (HDF5_LOCK and NETCDFC_LOCK, both SerializableLocks) with a
#    per-file write lock from get_write_lock. The write lock is acquired first:
#    it may be a distributed or multiprocessing lock that is slow to acquire, and
#    waiting for it while holding a library lock would block all HDF5 and
#    netCDF-C calls of the process in the meantime. Deadlock freedom only needs
#    the order to be the same everywhere, not this particular one.
# 2. Within a level, a name that stays the same when the lock is pickled: the
#    token of a SerializableLock and the name of a dask.distributed.Lock. Other
#    locks, like threading and multiprocessing locks, have no such name, so they
#    fall back to id(). That is fine as long as no two of them are combined,
#    which xarray never does: there is at most one write lock per file.
_RESOURCE_LOCK_LEVEL = 0  # per-file write locks and user supplied locks
_LIBRARY_LOCK_LEVEL = 1  # SerializableLocks, like HDF5_LOCK and NETCDFC_LOCK


def _is_distributed_lock(lock: Lock) -> TypeGuard[DistributedLock]:
    # A dask.distributed.Lock can only exist if distributed was imported, so
    # don't import it just to check.
    distributed = sys.modules.get("distributed")
    return distributed is not None and isinstance(lock, distributed.Lock)


def _lock_order_key(lock: Lock) -> tuple[int, str, str | int]:
    """Position of ``lock`` in the global lock order, see "Lock ordering" above.

    Copies of the same lock get the same key, also across processes, so the key
    is used to remove duplicates too.

    The second element names the kind of the last one, so that only values of
    the same type are ever compared and sorting never fails.
    """
    if isinstance(lock, SerializableLock):
        # All copies with the same token wrap the same threading.Lock. Tokens
        # can be any hashable, so compare their reprs, which keeps 1 and "1"
        # apart.
        return (_LIBRARY_LOCK_LEVEL, "token", repr(lock.token))
    if _is_distributed_lock(lock):
        # all copies with the same name are the same lock on the scheduler
        return (_RESOURCE_LOCK_LEVEL, "name", lock.name)
    return (_RESOURCE_LOCK_LEVEL, "id", id(lock))


class CombinedLock(Lock):
    """A combination of multiple locks.

    Like a locked door, a CombinedLock is locked if any of its constituent
    locks are locked.

    The locks are always acquired in one global order, independent of the order
    they are passed in, see "Lock ordering" above.
    """

    def __init__(self, locks: Sequence[Lock]):
        # Remove duplicates, as acquiring a lock that is not reentrant twice
        # deadlocks, and sort into the global lock order. If not careful,
        # CombinedLocks sharing locks could acquire them in opposite orders and
        # deadlock each other.
        unique = {_lock_order_key(lock): lock for lock in locks}
        self.locks = tuple(lock for _, lock in sorted(unique.items()))

    def __reduce__(self):
        # Sort again when unpickling. The keys of SerializableLocks and
        # distributed locks are the same in every process, so their order does
        # not change. But other locks are ordered by id(), which differs between
        # processes.
        return (type(self), (list(self.locks),))

    def acquire(self, blocking=True):
        acquired = []
        for lock in self.locks:
            if not acquire(lock, blocking=blocking):
                # Release the locks we already hold, otherwise a failed
                # non-blocking acquire leaves them locked forever.
                for held in reversed(acquired):
                    held.release()
                return False
            acquired.append(lock)
        return True

    def release(self):
        for lock in reversed(self.locks):
            lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *args):
        self.release()

    def locked(self):
        return any(lock.locked() for lock in self.locks)

    def __repr__(self):
        return f"CombinedLock({list(self.locks)!r})"


class DummyLock(Lock):
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


def combine_locks(locks: Sequence[Lock]) -> Lock:
    """Combine a sequence of locks into a single lock."""
    all_locks: list[Lock] = []
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


def ensure_lock(lock: Lock | Literal[False] | None) -> Lock:
    """Ensure that the given object is a lock."""
    if lock is None or lock is False:
        return DummyLock()
    return lock
