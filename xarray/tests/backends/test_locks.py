from __future__ import annotations

import pickle
import threading

from xarray.backends import locks
from xarray.backends.locks import CombinedLock, SerializableLock, combine_locks


def test_threaded_lock() -> None:
    lock1 = locks._get_threaded_lock("foo")
    assert isinstance(lock1, type(threading.Lock()))
    lock2 = locks._get_threaded_lock("foo")
    assert lock1 is lock2

    lock3 = locks._get_threaded_lock("bar")
    assert lock1 is not lock3


def test_combined_lock_locked_returns_false_when_no_locks_acquired() -> None:
    """CombinedLock.locked() should return False when no locks are held."""
    lock1 = threading.Lock()
    lock2 = threading.Lock()
    combined = CombinedLock([lock1, lock2])

    assert combined.locked() is False
    assert lock1.locked() is False
    assert lock2.locked() is False


def test_combined_lock_locked_returns_true_when_one_lock_acquired() -> None:
    """CombinedLock.locked() should return True when any lock is held."""
    lock1 = threading.Lock()
    lock2 = threading.Lock()
    combined = CombinedLock([lock1, lock2])

    lock1.acquire()
    try:
        assert combined.locked() is True
    finally:
        lock1.release()

    assert combined.locked() is False


def test_combined_lock_locked_returns_true_when_all_locks_acquired() -> None:
    """CombinedLock.locked() should return True when all locks are held."""
    lock1 = threading.Lock()
    lock2 = threading.Lock()
    combined = CombinedLock([lock1, lock2])

    lock1.acquire()
    lock2.acquire()
    try:
        assert combined.locked() is True
    finally:
        lock1.release()
        lock2.release()

    assert combined.locked() is False


def test_combined_lock_locked_with_serializable_locks() -> None:
    """CombinedLock.locked() should work with SerializableLock instances."""
    lock1 = SerializableLock()
    lock2 = SerializableLock()
    combined = CombinedLock([lock1, lock2])

    assert combined.locked() is False

    lock1.acquire()
    try:
        assert combined.locked() is True
    finally:
        lock1.release()

    assert combined.locked() is False


def test_combined_lock_failed_nonblocking_acquire_releases_held_locks() -> None:
    """A failed non-blocking acquire must not leave any constituent lock held."""
    lock1 = SerializableLock()
    lock2 = SerializableLock()
    combined = CombinedLock([lock1, lock2])

    # The acquisition order is an implementation detail, so try both.
    for busy, other in [(lock1, lock2), (lock2, lock1)]:
        busy.acquire()
        try:
            assert combined.acquire(blocking=False) is False
            assert other.locked() is False
        finally:
            busy.release()

    assert combined.acquire(blocking=False) is True
    combined.release()
    assert combined.locked() is False


def test_combined_lock_locked_with_context_manager() -> None:
    """CombinedLock.locked() should reflect state when using context manager."""
    lock1 = threading.Lock()
    lock2 = threading.Lock()
    combined = CombinedLock([lock1, lock2])

    assert combined.locked() is False

    with combined:
        assert combined.locked() is True

    assert combined.locked() is False


def test_combined_lock_order_independent_of_construction_order() -> None:
    """Combined locks sharing locks must acquire them in the same order."""
    lock_a = SerializableLock()
    lock_b = SerializableLock()
    lock_c = threading.Lock()

    assert CombinedLock([lock_a, lock_b]).locks == CombinedLock([lock_b, lock_a]).locks
    forward = CombinedLock([lock_a, lock_b, lock_c]).locks
    backward = CombinedLock([lock_c, lock_b, lock_a]).locks
    assert forward == backward


class _HashedLock(SerializableLock):
    """SerializableLock with a fixed hash, to control set iteration order."""

    def __init__(self, hash_value: int):
        super().__init__()
        self.hash_value = hash_value

    def __hash__(self) -> int:
        return self.hash_value


def test_combined_lock_reader_and_writer_share_order() -> None:
    """Mimic the netCDF4 backend: a reader lock and a writer lock built from it.

    Deduplicating with a set made the order depend on hashes (memory addresses),
    so with these hash values a reader acquired the two shared locks in the
    opposite order of the writer, and the two deadlocked each other (GH11622).
    """
    netcdfc, hdf5, write_lock = _HashedLock(3), _HashedLock(11), _HashedLock(0)
    reader = combine_locks([netcdfc, hdf5])
    writer = combine_locks([reader, write_lock])
    assert isinstance(reader, CombinedLock)
    assert isinstance(writer, CombinedLock)

    shared = [lock for lock in writer.locks if lock in (netcdfc, hdf5)]
    assert shared == list(reader.locks)


def test_combined_lock_deduplicates_unpickled_serializable_lock() -> None:
    """An unpickled SerializableLock wraps the same lock and must not be held twice."""
    lock = SerializableLock()
    copy = pickle.loads(pickle.dumps(lock))
    assert copy is not lock
    assert copy.lock is lock.lock

    combined = CombinedLock([lock, copy])
    assert len(combined.locks) == 1
    with combined:
        assert lock.locked()
    assert not lock.locked()


def test_combined_lock_is_sorted_again_when_unpickled() -> None:
    # The lock order depends on ids, which differ between processes, so a
    # CombinedLock pickled in another process must be sorted again.
    combined = CombinedLock([SerializableLock(), SerializableLock()])
    local_order = [locks._lock_identity(lock) for lock in combined.locks]
    combined.locks = combined.locks[::-1]  # as if sorted in another process

    unpickled = pickle.loads(pickle.dumps(combined))
    assert [locks._lock_identity(lock) for lock in unpickled.locks] == local_order
