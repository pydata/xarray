from __future__ import annotations

import multiprocessing
import sys
import threading
import uuid
import weakref
from collections.abc import Callable, Hashable, Iterable, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeGuard, override
from weakref import WeakValueDictionary

from xarray.core.types import Lock

if TYPE_CHECKING:
    from types import TracebackType

    from distributed import Lock as DistributedLock


class _ReentrantLock:
    """Like ``threading.RLock``, but with ``locked()`` on all Python versions.

    The netCDF4 backend holds its lock while reading and writing metadata, and
    the calls it makes meanwhile acquire the same lock again, so the default
    locks must be reentrant. ``threading.RLock`` only has ``locked()`` since
    Python 3.14, but ``SerializableLock.locked()`` and ``CombinedLock.locked()``
    rely on it.
    """

    # TODO: replace with threading.RLock once we require Python >= 3.14

    __slots__ = ("__weakref__", "_count", "_lock", "_owner")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None
        self._count = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        me = threading.get_ident()
        if self._owner == me:
            self._count += 1
            return True
        if not self._lock.acquire(blocking, timeout):
            return False
        self._owner = me
        self._count = 1
        return True

    def release(self) -> None:
        if self._owner != threading.get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._count -= 1
        if not self._count:
            self._owner = None
            self._lock.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def locked(self) -> bool:
        return self._lock.locked()


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

    With ``reentrant=True`` the lock can be acquired again by the thread that
    already holds it, like ``threading.RLock``.

    The creation of locks is itself not threadsafe.
    """

    _locks: ClassVar[WeakValueDictionary[Hashable, threading.Lock | _ReentrantLock]] = (
        WeakValueDictionary()
    )
    token: Hashable
    reentrant: bool
    lock: threading.Lock | _ReentrantLock

    def __init__(self, token: Hashable | None = None, reentrant: bool = False):
        self.token = token or str(uuid.uuid4())
        self.reentrant = reentrant
        if self.token in SerializableLock._locks:
            self.lock = SerializableLock._locks[self.token]
        else:
            self.lock = _ReentrantLock() if reentrant else threading.Lock()
            SerializableLock._locks[self.token] = self.lock

    @override
    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        return self.lock.acquire(*args, **kwargs)

    @override
    def release(self) -> None:
        self.lock.release()

    @override
    def __enter__(self) -> bool:
        return self.lock.__enter__()

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.lock.__exit__(exc_type, exc_value, traceback)

    def locked(self):
        return self.lock.locked()

    def __getstate__(self):
        # include reentrant, so that a process that does not know the token yet
        # creates the right kind of lock
        return (self.token, self.reentrant)

    def __setstate__(self, state):
        self.__init__(*state)

    def __str__(self):
        return f"<{self.__class__.__name__}: {self.token}>"

    __repr__ = __str__


# Locks used by multiple backends.
# Neither HDF5 nor the netCDF-C library are thread-safe. The locks are reentrant
# so that backends can hold them across calls that acquire them again. They
# have fixed tokens, so that an unpickled lock, e.g. in a dask worker, is the
# global lock of that process and not a separate lock. The tokens also give them
# the same place in the lock order in every process, see CombinedLock._sort_locks.
HDF5_LOCK = SerializableLock("xarray-hdf5-lock", reentrant=True)
NETCDFC_LOCK = SerializableLock("xarray-netcdfc-lock", reentrant=True)


_FILE_LOCKS: MutableMapping[Any, _ReentrantLock] = weakref.WeakValueDictionary()


def _get_threaded_lock(key: str) -> _ReentrantLock:
    # reentrant, as it is combined with the global locks into the lock that
    # netCDF4 holds while writing metadata (see is_reentrant_lock)
    try:
        lock = _FILE_LOCKS[key]
    except KeyError:
        lock = _FILE_LOCKS[key] = _ReentrantLock()
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


def acquire(lock: Lock, blocking: bool = True) -> object:
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


def _is_distributed_lock(lock: Lock) -> TypeGuard[DistributedLock]:
    # A dask.distributed.Lock can only exist if distributed was imported, so
    # don't import it just to check.
    distributed = sys.modules.get("distributed")
    return distributed is not None and isinstance(lock, distributed.Lock)


def _lock_key(lock: Lock) -> tuple[str, str | int]:
    """Key that is the same for all copies of a lock, e.g. after pickling."""
    if isinstance(lock, SerializableLock):
        # All copies with the same token wrap the same threading.Lock. Tokens
        # can be any hashable, so use their reprs, which keeps 1 and "1" apart.
        return ("token", repr(lock.token))
    if _is_distributed_lock(lock):
        # all copies with the same name are the same lock on the scheduler
        return ("distributed", lock.name)
    return ("object", id(lock))


class CombinedLock(Lock):
    """A combination of multiple locks.

    Like a locked door, a CombinedLock is locked if any of its constituent
    locks are locked.

    The locks are always acquired in the same order, independent of the order
    they are passed in, see ``CombinedLock._sort_locks``.
    """

    locks: tuple[Lock, ...]

    def __init__(self, locks: Sequence[Lock]) -> None:
        self.locks = self._sort_locks(locks)

    @staticmethod
    def _sort_locks(locks: Iterable[Lock]) -> tuple[Lock, ...]:
        """Remove duplicate locks and sort them into the order they are acquired in.

        A thread holding lock A while waiting for lock B deadlocks with a thread
        holding B while waiting for A. The standard way to rule this out is to
        acquire all locks in the same order everywhere, so nobody ever waits for
        a lock that comes before one it already holds.

        xarray combines the process-wide library locks (HDF5_LOCK and
        NETCDFC_LOCK, both SerializableLocks) with at most one other lock,
        usually the per-file write lock from get_write_lock. The order is:

        1. All locks that are not SerializableLocks, in the order they are
           passed in. This is the per-file lock: it may be a distributed or
           multiprocessing lock that is slow to acquire, and waiting for it
           while holding a library lock would block all HDF5 and netCDF-C calls
           of the process in the meantime. Should two such locks ever be
           combined, they must be passed in the same order everywhere.
        2. The SerializableLocks, sorted by their token.

        The order must not depend on the lock objects themselves, e.g. their
        id(): locks are pickled all the time, every dask task gets its own copy
        of the locks it uses, and these copies are new objects. When the order
        depended on id(), two threads of a distributed worker could take
        HDF5_LOCK and the per-file write lock in opposite orders and deadlock.
        Tokens and the order of the arguments stay the same when pickled.

        Copies of the same lock are removed, as acquiring a lock that is not
        reentrant twice deadlocks.
        """
        # Use a dict to keep ordering
        unique = list({_lock_key(lock): lock for lock in locks}.values())
        file_locks = [lock for lock in unique if not isinstance(lock, SerializableLock)]
        library_locks = sorted(
            (lock for lock in unique if isinstance(lock, SerializableLock)),
            key=lambda lock: repr(lock.token),
        )
        return (*file_locks, *library_locks)

    @override
    def acquire(self, blocking: bool = True) -> bool:
        acquired: list[Lock] = []
        for lock in self.locks:
            if not acquire(lock, blocking=blocking):
                # Release the locks we already hold, otherwise a failed
                # non-blocking acquire leaves them locked forever.
                for held in reversed(acquired):
                    held.release()
                return False
            acquired.append(lock)
        return True

    @override
    def release(self) -> None:
        for lock in reversed(self.locks):
            lock.release()

    @override
    def __enter__(self) -> bool:
        return self.acquire()

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def locked(self) -> bool:
        """Whether any of the locks is locked.

        ``locked()`` is not part of the Lock protocol, so this raises an
        ``AttributeError`` if one of the locks doesn't support it, e.g.
        ``multiprocessing.Lock`` before Python 3.14.
        """
        return any(lock.locked() for lock in self.locks)  # type: ignore[attr-defined]

    @property
    def reentrant(self) -> bool:
        """True if all locks are reentrant, otherwise False."""
        return all(is_reentrant_lock(lock) for lock in self.locks)

    def __repr__(self) -> str:
        return f"CombinedLock({list(self.locks)!r})"


class DummyLock(Lock):
    """DummyLock provides the lock API without any actual locking."""

    @override
    def acquire(self, blocking: bool = True) -> Literal[True]:
        return True

    @override
    def release(self) -> None:
        pass

    @override
    def __enter__(self) -> Literal[True]:
        return True

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def locked(self) -> Literal[False]:
        return False


def is_reentrant_lock(lock: Lock) -> bool:
    """Whether the thread holding ``lock`` can safely acquire it again."""
    if isinstance(lock, (SerializableLock, CombinedLock)):
        return lock.reentrant
    return isinstance(lock, _ReentrantLock)


def combine_locks(locks: Iterable[Lock]) -> Lock:
    """Combine multiple locks into a single lock."""
    all_locks: list[Lock] = []
    for lock in locks:
        if isinstance(lock, CombinedLock):
            all_locks.extend(lock.locks)
        elif lock is not None:
            all_locks.append(lock)

    num_locks = len(all_locks)
    if num_locks > 1:
        return CombinedLock(all_locks)
    if num_locks == 1:
        return all_locks[0]
    return DummyLock()


def ensure_lock(lock: Lock | Literal[False] | None) -> Lock:
    """Ensure that the given object is a lock."""
    if lock is None or lock is False:
        return DummyLock()
    return lock
