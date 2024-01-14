from __future__ import annotations

import multiprocessing
import threading
import uuid
import weakref
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Callable, ClassVar, Literal, overload

if TYPE_CHECKING:
    from xarray.core.types import LockLike, T_LockLike, TypeAlias

    SchedulerOptions: TypeAlias = Literal[
        "threaded", "multiprocessing", "distributed", None
    ]


# SerializableLock is adapted from Dask:
# https://github.com/dask/dask/blob/74e898f0ec712e8317ba86cc3b9d18b6b9922be0/dask/utils.py#L1160-L1224
# Used under the terms of Dask's license, see licenses/DASK_LICENSE.
class SerializableLock:
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

    _locks: ClassVar[
        weakref.WeakValueDictionary[Hashable, threading.Lock]
    ] = weakref.WeakValueDictionary()
    token: Hashable
    lock: threading.Lock

    def __init__(self, token: Hashable = None):
        self._set_token_and_lock(token)

    def _set_token_and_lock(self, token: Hashable) -> None:
        self.token = token or str(uuid.uuid4())
        if self.token in SerializableLock._locks:
            self.lock = SerializableLock._locks[self.token]
        else:
            self.lock = threading.Lock()
            SerializableLock._locks[self.token] = self.lock

    def acquire(self, *args, **kwargs) -> bool:
        return self.lock.acquire(*args, **kwargs)

    def release(self, *args, **kwargs) -> None:
        self.lock.release(*args, **kwargs)

    def __enter__(self) -> bool:
        return self.lock.__enter__()

    def __exit__(self, *args) -> None:
        self.lock.__exit__(*args)

    def locked(self) -> bool:
        return self.lock.locked()

    def __getstate__(self) -> Hashable:
        return self.token

    def __setstate__(self, token: Hashable) -> None:
        self._set_token_and_lock(token)

    def __str__(self) -> str:
        return f"<{type(self).__name__}: {self.token}>"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.token!r})"


# Locks used by multiple backends.
# Neither HDF5 nor the netCDF-C library are thread-safe.
HDF5_LOCK = SerializableLock()
NETCDFC_LOCK = SerializableLock()


_FILE_LOCKS: weakref.WeakValueDictionary[
    Hashable, threading.Lock
] = weakref.WeakValueDictionary()


def _get_threaded_lock(key: Hashable) -> threading.Lock:
    try:
        lock = _FILE_LOCKS[key]
    except KeyError:
        lock = _FILE_LOCKS[key] = threading.Lock()
    return lock


def _get_multiprocessing_lock(key: Hashable) -> LockLike:
    # TODO: make use of the key -- maybe use locket.py?
    # https://github.com/mwilliamson/locket.py
    del key  # unused
    # multiprocessing.Lock is missing the "locked" method???
    return multiprocessing.Lock()  # type: ignore[return-value]


def _get_lock_maker(
    scheduler: SchedulerOptions = None,
) -> Callable[[Hashable], LockLike] | None:
    """Returns an appropriate function for creating resource locks.

    Parameters
    ----------
    scheduler : str or None
        Dask scheduler being used.

    See Also
    --------
    dask.utils.get_scheduler_lock
    """

    if scheduler is None:
        return _get_threaded_lock
    if scheduler == "threaded":
        return _get_threaded_lock
    if scheduler == "multiprocessing":
        return _get_multiprocessing_lock
    if scheduler == "distributed":
        # Lazy import distributed since it is can add a significant
        # amount of time to import
        try:
            from dask.distributed import Lock as DistributedLock

            return DistributedLock
        except ImportError:
            return None
    raise KeyError(scheduler)


def _get_scheduler(get=None, collection=None) -> SchedulerOptions:
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
        from dask.base import get_scheduler  # noqa: F401

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


def get_write_lock(key: Hashable) -> LockLike:
    """Get a scheduler appropriate lock for writing to the given resource.

    Parameters
    ----------
    key : hashable
        Name of the resource for which to acquire a lock. Typically a filename.

    Returns
    -------
    Lock object that can be used like a threading.Lock object.
    """
    scheduler = _get_scheduler()
    lock_maker = _get_lock_maker(scheduler)
    assert lock_maker is not None
    return lock_maker(key)


def acquire(lock: LockLike, blocking: bool = True) -> bool:
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


class CombinedLock:
    """A combination of multiple locks.

    Like a locked door, a CombinedLock is locked if any of its constituent
    locks are locked.
    """

    locks: tuple[LockLike, ...]

    def __init__(self, locks: Iterable[LockLike]):
        self.locks = tuple(set(locks))  # remove duplicates

    def acquire(self, blocking: bool = True) -> bool:
        return all(acquire(lock, blocking=blocking) for lock in self.locks)

    def release(self) -> None:
        for lock in self.locks:
            lock.release()

    def __enter__(self) -> bool:
        return all(lock.__enter__() for lock in self.locks)

    def __exit__(self, *args) -> None:
        for lock in self.locks:
            lock.__exit__(*args)

    def locked(self) -> bool:
        return any(lock.locked for lock in self.locks)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self.locks)!r})"


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


def combine_locks(locks: Iterable[LockLike]) -> LockLike:
    """Combine a sequence of locks into a single lock."""
    all_locks: list[LockLike] = []
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


@overload
def ensure_lock(lock: Literal[False] | None) -> DummyLock:
    ...


@overload
def ensure_lock(lock: T_LockLike) -> T_LockLike:
    ...


def ensure_lock(lock: Literal[False] | T_LockLike | None) -> T_LockLike | DummyLock:
    """Ensure that the given object is a lock."""
    if lock is None or lock is False:
        return DummyLock()
    return lock
