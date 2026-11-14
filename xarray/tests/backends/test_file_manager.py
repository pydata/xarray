from __future__ import annotations

import gc
import os
import pickle
import shutil
import threading
from pathlib import Path
from unittest import mock

import pytest

from xarray import open_dataset
from xarray.backends import file_manager
from xarray.backends.file_manager import (
    CachingFileManager,
    PickleableFileManager,
    _close_unless_pinned,
)
from xarray.backends.locks import CombinedLock, SerializableLock
from xarray.backends.lru_cache import LRUCache
from xarray.core.options import set_options
from xarray.tests import assert_no_warnings, requires_h5netcdf, requires_netCDF4


@pytest.fixture(params=[1, 2, 3, None])
def file_cache(request):
    maxsize = request.param
    if maxsize is None:
        yield {}
    else:
        yield LRUCache(maxsize)


def test_file_manager_mock_write(file_cache) -> None:
    mock_file = mock.Mock()
    opener = mock.Mock(spec=open, return_value=mock_file)
    lock = mock.MagicMock(spec=threading.Lock())

    manager = CachingFileManager(opener, "filename", lock=lock, cache=file_cache)
    f = manager.acquire()
    f.write("contents")
    manager.close()

    assert not file_cache
    opener.assert_called_once_with("filename")
    mock_file.write.assert_called_once_with("contents")
    mock_file.close.assert_called_once_with()
    lock.__enter__.assert_has_calls([mock.call(), mock.call()])


@pytest.mark.parametrize("warn_for_unclosed_files", [True, False])
def test_file_manager_autoclose(warn_for_unclosed_files) -> None:
    mock_file = mock.Mock()
    opener = mock.Mock(return_value=mock_file)
    cache: dict = {}

    manager = CachingFileManager(opener, "filename", cache=cache)
    manager.acquire()
    assert cache

    # can no longer use pytest.warns(None)
    if warn_for_unclosed_files:
        ctx = pytest.warns(RuntimeWarning)
    else:
        ctx = assert_no_warnings()  # type: ignore[assignment]

    with set_options(warn_for_unclosed_files=warn_for_unclosed_files):
        with ctx:
            del manager
            gc.collect()

    assert not cache
    mock_file.close.assert_called_once_with()


def _wait_for_close_threads() -> None:
    for thread in threading.enumerate():
        if thread.name == "xarray-close-file":
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_file_manager_autoclose_while_locked() -> None:
    mock_file = mock.Mock()
    opener = mock.Mock(return_value=mock_file)
    lock = threading.Lock()
    cache: dict = {}

    manager = CachingFileManager(opener, "filename", lock=lock, cache=cache)
    manager.acquire()
    assert cache

    lock.acquire()

    with set_options(warn_for_unclosed_files=False):
        del manager
        gc.collect()

    # can't clear the cache while locked, but also don't block in __del__
    assert cache
    mock_file.close.assert_not_called()

    # the file is closed once the lock is free (GH11088)
    lock.release()
    _wait_for_close_threads()
    assert not cache
    mock_file.close.assert_called_once_with()


def test_file_manager_autoclose_while_locked_reused() -> None:
    mock_file = mock.Mock()
    opener = mock.Mock(return_value=mock_file)
    lock = threading.Lock()
    cache: dict = {}

    manager = CachingFileManager(opener, "filename", lock=lock, cache=cache)
    manager_id = manager._manager_id
    manager.acquire()

    lock.acquire()
    with set_options(warn_for_unclosed_files=False):
        del manager
        gc.collect()

    # a new manager for the same file, like an unpickled one, keeps it open
    manager2 = CachingFileManager(
        opener, "filename", lock=lock, cache=cache, manager_id=manager_id
    )
    lock.release()
    _wait_for_close_threads()
    assert cache
    mock_file.close.assert_not_called()
    assert manager2.acquire() is mock_file
    opener.assert_called_once_with("filename")
    manager2.close()


def test_file_manager_repr() -> None:
    opener = mock.Mock()
    manager = CachingFileManager(opener, "my-file")
    assert "my-file" in repr(manager)


def test_file_manager_cache_and_refcounts() -> None:
    mock_file = mock.Mock()
    opener = mock.Mock(spec=open, return_value=mock_file)
    cache: dict = {}
    ref_counts: dict = {}

    manager = CachingFileManager(opener, "filename", cache=cache, ref_counts=ref_counts)
    assert ref_counts[manager._key] == 1

    assert not cache
    manager.acquire()
    assert len(cache) == 1

    with set_options(warn_for_unclosed_files=False):
        del manager
        gc.collect()

    assert not ref_counts
    assert not cache


def test_file_manager_cache_repeated_open() -> None:
    mock_file = mock.Mock()
    opener = mock.Mock(spec=open, return_value=mock_file)
    cache: dict = {}

    manager = CachingFileManager(opener, "filename", cache=cache)
    manager.acquire()
    assert len(cache) == 1

    manager2 = CachingFileManager(opener, "filename", cache=cache)
    manager2.acquire()
    assert len(cache) == 2

    with set_options(warn_for_unclosed_files=False):
        del manager
        gc.collect()

    assert len(cache) == 1

    with set_options(warn_for_unclosed_files=False):
        del manager2
        gc.collect()

    assert not cache


def test_shared_read_file_manager_lifetime() -> None:
    file = mock.Mock()
    opener = mock.Mock(return_value=file)
    first = CachingFileManager(opener, "filename", mode="r", share=True)
    second = CachingFileManager(opener, "filename", mode="r", share=True)

    assert first.acquire() is second.acquire()
    opener.assert_called_once_with("filename", mode="r")
    first.close()
    assert file_manager.REF_COUNTS[second._key] == 1
    del first
    gc.collect()
    file.close.assert_not_called()
    assert second.acquire() is file
    second.close()
    assert second._key not in file_manager.REF_COUNTS
    file.close.assert_called_once_with()

    third = CachingFileManager(opener, "filename", mode="r", share=True)
    third.acquire()
    assert opener.call_count == 2
    third.close()


def test_shared_read_file_manager_pickle() -> None:
    manager = CachingFileManager(open, __file__, mode="r", share=True)
    restored = pickle.loads(pickle.dumps(manager))
    try:
        assert manager.acquire() is restored.acquire()
        manager.close()
        assert not restored.acquire().closed
    finally:
        restored.close()


def test_shared_read_file_manager_pickle_combined_lock() -> None:
    lock = CombinedLock(
        [SerializableLock("shared-read-a"), SerializableLock("shared-read-b")]
    )
    manager = CachingFileManager(open, __file__, mode="r", share=True, lock=lock)
    restored = pickle.loads(pickle.dumps(manager))
    try:
        assert manager.acquire() is restored.acquire()
    finally:
        manager.close()
        restored.close()


def test_shared_read_file_manager_reopen() -> None:
    opener = mock.Mock(side_effect=[mock.Mock(), mock.Mock()])
    manager = CachingFileManager(opener, "filename", mode="r", share=True)
    first = manager.acquire()
    manager.close()
    assert manager._key not in file_manager.REF_COUNTS
    second = manager.acquire()
    assert file_manager.REF_COUNTS[manager._key] == 1
    assert first is not second
    manager.close()


def test_shared_read_file_manager_distinct_options() -> None:
    opener = mock.Mock(side_effect=[mock.Mock(), mock.Mock()])
    first = CachingFileManager(
        opener, "filename", mode="r", share=True, kwargs={"option": 1}
    )
    second = CachingFileManager(
        opener, "filename", mode="r", share=True, kwargs={"option": 2}
    )
    try:
        assert first.acquire() is not second.acquire()
        assert opener.call_count == 2
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("warn_for_unclosed_files", [True, False])
def test_shared_read_file_manager_autoclose(warn_for_unclosed_files: bool) -> None:
    file = mock.Mock()
    manager = CachingFileManager(
        mock.Mock(return_value=file), "filename", mode="r", share=True
    )
    manager.acquire()

    ctx = (
        pytest.warns(RuntimeWarning, match="deallocating CachingFileManager")
        if warn_for_unclosed_files
        else assert_no_warnings()
    )
    with set_options(warn_for_unclosed_files=warn_for_unclosed_files), ctx:
        del manager
        gc.collect()

    file.close.assert_called_once_with()


def test_shared_read_file_manager_close_while_locked() -> None:
    file = mock.Mock()
    lock = threading.Lock()
    manager = CachingFileManager(
        mock.Mock(return_value=file), "filename", mode="r", share=True, lock=lock
    )
    manager.acquire()

    with lock:
        manager.close(needs_lock=False)

    file.close.assert_called_once_with()


def test_shared_read_file_manager_autoclose_while_locked() -> None:
    file = mock.Mock()
    lock = threading.Lock()
    manager = CachingFileManager(
        mock.Mock(return_value=file), "filename", mode="r", share=True, lock=lock
    )
    manager.acquire()

    lock.acquire()
    with set_options(warn_for_unclosed_files=False):
        del manager
        gc.collect()

    file.close.assert_not_called()
    lock.release()
    _wait_for_close_threads()
    file.close.assert_called_once_with()


@requires_h5netcdf
def test_h5netcdf_read_only_groups_share_file(tmp_path: Path) -> None:
    import h5netcdf

    path = tmp_path / "groups.nc"
    with h5netcdf.File(path, "w") as file:
        for name in ("a", "b"):
            group = file.create_group(name)
            group.dimensions["x"] = 1
            group.create_variable("value", ("x",), data=[1])

    first = open_dataset(path, group="a", engine="h5netcdf")
    second = open_dataset(path, group="b", engine="h5netcdf")
    try:
        assert len(file_manager.FILE_CACHE) == 1
        first.close()
        assert len(file_manager.FILE_CACHE) == 1
        assert second["value"].values.tolist() == [1]
    finally:
        first.close()
        second.close()
    assert len(file_manager.FILE_CACHE) == 0

    with h5netcdf.File(path, "a") as file:
        file["b/value"][:] = [2]
    with open_dataset(path, group="b", engine="h5netcdf") as reopened:
        assert reopened["value"].values.tolist() == [2]


@requires_h5netcdf
def test_h5netcdf_custom_lock_does_not_share_file(tmp_path: Path) -> None:
    import h5netcdf

    path = tmp_path / "data.nc"
    with h5netcdf.File(path, "w") as file:
        file.create_variable("a", (), data=1)

    first = open_dataset(path, engine="h5netcdf", lock=threading.Lock())
    second = open_dataset(path, engine="h5netcdf", lock=threading.Lock())
    try:
        assert len(file_manager.FILE_CACHE) == 2
    finally:
        first.close()
        second.close()


@requires_netCDF4
def test_netcdf4_hdf5_read_only_groups_share_file(tmp_path: Path) -> None:
    import netCDF4

    path = tmp_path / "groups.nc"
    with netCDF4.Dataset(path, "w") as file:
        for name in ("a", "b"):
            group = file.createGroup(name)
            group.createDimension("x", 1)
            group.createVariable("value", "i4", ("x",))[:] = [1]

    first = open_dataset(path, group="a", engine="netcdf4")
    second = open_dataset(path, group="b", engine="netcdf4")
    try:
        assert len(file_manager.FILE_CACHE) == 1
        first.close()
        assert second["value"].values.tolist() == [1]
    finally:
        first.close()
        second.close()
    assert len(file_manager.FILE_CACHE) == 0


@requires_netCDF4
def test_netcdf4_netcdf3_still_opens_separately(tmp_path: Path) -> None:
    import netCDF4

    path = tmp_path / "classic.nc"
    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as file:
        file.createDimension("x", 1)
        file.createVariable("value", "i4", ("x",))[:] = [1]

    first = open_dataset(path, engine="netcdf4")
    second = open_dataset(path, engine="netcdf4")
    try:
        assert len(file_manager.FILE_CACHE) == 2
    finally:
        first.close()
        second.close()


@pytest.mark.skipif(os.name == "nt", reason="Windows cannot overwrite open files")
@requires_h5netcdf
@pytest.mark.parametrize(
    "engine", ["h5netcdf", pytest.param("netcdf4", marks=requires_netCDF4)]
)
def test_shared_hdf5_refreshes_after_all_datasets_close(
    tmp_path: Path, engine: str
) -> None:
    import h5netcdf

    path = tmp_path / "data.nc"
    replacement = tmp_path / "replacement.nc"
    with h5netcdf.File(path, "w") as file:
        file.create_variable("a", (), data=1)
    with h5netcdf.File(replacement, "w") as file:
        file.create_variable("a", (), data=1)
        file.create_variable("b", (), data=2)

    first = open_dataset(path, engine=engine)
    try:
        shutil.copyfile(replacement, path)
        second = open_dataset(path, engine=engine)
        try:
            assert "b" not in second
        finally:
            second.close()
    finally:
        first.close()

    with open_dataset(path, engine=engine) as reopened:
        assert "b" in reopened


def test_file_manager_cache_with_pickle(tmpdir) -> None:
    path = str(tmpdir.join("testing.txt"))
    with open(path, "w") as f:
        f.write("data")
    cache: dict = {}

    with mock.patch("xarray.backends.file_manager.FILE_CACHE", cache):
        assert not cache

        manager = CachingFileManager(open, path, mode="r")
        manager.acquire()
        assert len(cache) == 1

        manager2 = pickle.loads(pickle.dumps(manager))
        manager2.acquire()
        assert len(cache) == 1

        with set_options(warn_for_unclosed_files=False):
            del manager
            gc.collect()
        # assert len(cache) == 1

        with set_options(warn_for_unclosed_files=False):
            del manager2
            gc.collect()
        assert not cache


def test_file_manager_write_consecutive(tmpdir, file_cache) -> None:
    path1 = str(tmpdir.join("testing1.txt"))
    path2 = str(tmpdir.join("testing2.txt"))
    manager1 = CachingFileManager(open, path1, mode="w", cache=file_cache)
    manager2 = CachingFileManager(open, path2, mode="w", cache=file_cache)
    f1a = manager1.acquire()
    f1a.write("foo")
    f1a.flush()
    f2 = manager2.acquire()
    f2.write("bar")
    f2.flush()
    f1b = manager1.acquire()
    f1b.write("baz")
    assert (getattr(file_cache, "maxsize", float("inf")) > 1) == (f1a is f1b)
    manager1.close()
    manager2.close()

    with open(path1) as f:
        assert f.read() == "foobaz"
    with open(path2) as f:
        assert f.read() == "bar"


def test_file_manager_write_concurrent(tmpdir, file_cache) -> None:
    path = str(tmpdir.join("testing.txt"))
    manager = CachingFileManager(open, path, mode="w", cache=file_cache)
    f1 = manager.acquire()
    f2 = manager.acquire()
    f3 = manager.acquire()
    assert f1 is f2
    assert f2 is f3
    f1.write("foo")
    f1.flush()
    f2.write("bar")
    f2.flush()
    f3.write("baz")
    f3.flush()
    manager.close()

    with open(path) as f:
        assert f.read() == "foobarbaz"


def test_file_manager_write_pickle(tmpdir, file_cache) -> None:
    path = str(tmpdir.join("testing.txt"))
    manager = CachingFileManager(open, path, mode="w", cache=file_cache)
    f = manager.acquire()
    f.write("foo")
    f.flush()
    manager2 = pickle.loads(pickle.dumps(manager))
    f2 = manager2.acquire()
    f2.write("bar")
    manager2.close()
    manager.close()

    with open(path) as f:
        assert f.read() == "foobar"


def test_file_manager_read(tmpdir, file_cache) -> None:
    path = str(tmpdir.join("testing.txt"))

    with open(path, "w") as f:
        f.write("foobar")

    manager = CachingFileManager(open, path, cache=file_cache)
    f = manager.acquire()
    assert f.read() == "foobar"
    manager.close()


def test_file_manager_acquire_context(tmpdir, file_cache) -> None:
    path = str(tmpdir.join("testing.txt"))

    with open(path, "w") as f:
        f.write("foobar")

    class AcquisitionError(Exception):
        pass

    manager = CachingFileManager(open, path, cache=file_cache)
    with pytest.raises(AcquisitionError):
        with manager.acquire_context() as f:
            assert f.read() == "foobar"
            raise AcquisitionError
    assert not file_cache  # file was *not* already open

    with manager.acquire_context() as f:
        assert f.read() == "foobar"

    with pytest.raises(AcquisitionError):
        with manager.acquire_context() as f:
            f.seek(0)
            assert f.read() == "foobar"
            raise AcquisitionError
    assert file_cache  # file *was* already open

    manager.close()


def test_file_manager_acquire_context_survives_eviction() -> None:
    # A file evicted by another manager while in use must stay open until the
    # context exits, and be reused rather than opened a second time.
    cache: LRUCache = LRUCache(maxsize=1, on_evict=_close_unless_pinned)
    file1, file2 = mock.Mock(), mock.Mock()
    opener1 = mock.Mock(return_value=file1)
    manager1 = CachingFileManager(opener1, "file1", cache=cache)
    manager2 = CachingFileManager(mock.Mock(return_value=file2), "file2", cache=cache)

    with manager1.acquire_context() as f:
        assert f is file1
        manager2.acquire()  # evicts file1
        file1.close.assert_not_called()
        assert manager1.acquire() is file1  # evicts file2
        file2.close.assert_called_once_with()
        manager2.acquire()  # evicts file1 again
    file1.close.assert_called_once_with()
    opener1.assert_called_once()

    manager1.close()
    manager2.close()


def test_file_manager_close_while_pinned_and_evicted() -> None:
    cache: LRUCache = LRUCache(maxsize=1, on_evict=_close_unless_pinned)
    file1 = mock.Mock()
    manager1 = CachingFileManager(mock.Mock(return_value=file1), "file1", cache=cache)
    manager2 = CachingFileManager(mock.Mock(), "file2", cache=cache)

    with manager1.acquire_context():
        manager2.acquire()  # evicts file1
        manager1.close()
        file1.close.assert_called_once_with()
    file1.close.assert_called_once_with()
    manager2.close()


def test_pickleable_file_manager_write_pickle(tmpdir) -> None:
    path = str(tmpdir.join("testing.txt"))
    manager = PickleableFileManager(open, path, mode="w")
    f = manager.acquire()
    f.write("foo")
    f.flush()
    manager2 = pickle.loads(pickle.dumps(manager))
    f2 = manager2.acquire()
    f2.write("bar")
    manager2.close()
    manager.close()

    with open(path) as f:
        assert f.read() == "foobar"


def test_pickleable_file_manager_preserves_closed(tmpdir) -> None:
    path = str(tmpdir.join("testing.txt"))
    manager = PickleableFileManager(open, path, mode="w")
    f = manager.acquire()
    f.write("foo")
    manager.close()
    manager2 = pickle.loads(pickle.dumps(manager))
    assert manager2._closed
    assert repr(manager2) == "<closed PickleableFileManager>"


def test_file_manager_del_while_pin_lock_held() -> None:
    # Garbage collection can run a manager's __del__ while the same thread is
    # updating the pin state, so closing from __del__ must not deadlock.
    manager = CachingFileManager(mock.Mock(return_value=mock.Mock()), cache={})
    manager.acquire()
    done = threading.Event()

    def delete_while_locked():
        nonlocal manager
        with file_manager._PIN_LOCK:
            del manager  # runs __del__, which closes the file
        done.set()

    # the manager is deliberately deallocated without closing it first
    with set_options(warn_for_unclosed_files=False):
        thread = threading.Thread(target=delete_while_locked, daemon=True)
        thread.start()
        assert done.wait(timeout=5)
