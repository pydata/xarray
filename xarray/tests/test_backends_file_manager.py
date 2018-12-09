import gc
import pickle
import threading
try:
    from unittest import mock
except ImportError:
    import mock  # noqa: F401

import pytest

from xarray.backends.file_manager import CachingFileManager
from xarray.backends.lru_cache import LRUCache


@pytest.fixture(params=[1, 2, 3, None])
def file_cache(request):
    maxsize = request.param
    if maxsize is None:
        yield {}
    else:
        yield LRUCache(maxsize)


def test_file_manager_mock_write(file_cache):
    mock_file = mock.Mock()
    opener = mock.Mock(spec=open, return_value=mock_file)
    lock = mock.MagicMock(spec=threading.Lock())

    manager = CachingFileManager(
        opener, 'filename', lock=lock, cache=file_cache)
    f = manager.acquire()
    f.write('contents')
    manager.close()

    assert not file_cache
    opener.assert_called_once_with('filename')
    mock_file.write.assert_called_once_with('contents')
    mock_file.close.assert_called_once_with()
    lock.__enter__.assert_has_calls([mock.call(), mock.call()])


def test_file_manager_autoclose(cache):
    mock_file = mock.Mock()
    opener = mock.Mock(return_value=mock_file)
    cache = {}

    manager = CachingFileManager(opener, 'filename', cache=cache)
    manager.acquire()
    assert cache
    del manager
    gc.collect()

    assert not cache
    mock_file.close.assert_called_once_with()


def test_file_manager_write_consecutive(tmpdir, file_cache):
    path1 = str(tmpdir.join('testing1.txt'))
    path2 = str(tmpdir.join('testing2.txt'))
    manager1 = CachingFileManager(open, path1, mode='w', cache=file_cache)
    manager2 = CachingFileManager(open, path2, mode='w', cache=file_cache)
    f1a = manager1.acquire()
    f1a.write('foo')
    f1a.flush()
    f2 = manager2.acquire()
    f2.write('bar')
    f2.flush()
    f1b = manager1.acquire()
    f1b.write('baz')
    assert (getattr(file_cache, 'maxsize', float('inf')) > 1) == (f1a is f1b)
    manager1.close()
    manager2.close()

    with open(path1, 'r') as f:
        assert f.read() == 'foobaz'
    with open(path2, 'r') as f:
        assert f.read() == 'bar'


def test_file_manager_write_concurrent(tmpdir, file_cache):
    path = str(tmpdir.join('testing.txt'))
    manager = CachingFileManager(open, path, mode='w', cache=file_cache)
    f1 = manager.acquire()
    f2 = manager.acquire()
    f3 = manager.acquire()
    assert f1 is f2
    assert f2 is f3
    f1.write('foo')
    f1.flush()
    f2.write('bar')
    f2.flush()
    f3.write('baz')
    f3.flush()
    manager.close()

    with open(path, 'r') as f:
        assert f.read() == 'foobarbaz'


def test_file_manager_write_pickle(tmpdir, file_cache):
    path = str(tmpdir.join('testing.txt'))
    manager = CachingFileManager(open, path, mode='w', cache=file_cache)
    f = manager.acquire()
    f.write('foo')
    f.flush()
    manager2 = pickle.loads(pickle.dumps(manager))
    f2 = manager2.acquire()
    f2.write('bar')
    manager2.close()
    manager.close()

    with open(path, 'r') as f:
        assert f.read() == 'foobar'


def test_file_manager_read(tmpdir, file_cache):
    path = str(tmpdir.join('testing.txt'))

    with open(path, 'w') as f:
        f.write('foobar')

    manager = CachingFileManager(open, path, cache=file_cache)
    f = manager.acquire()
    assert f.read() == 'foobar'
    manager.close()


def test_file_manager_invalid_kwargs():
    with pytest.raises(TypeError):
        CachingFileManager(open, 'dummy', mode='w', invalid=True)
