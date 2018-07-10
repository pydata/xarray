import pickle
try:
    from unittest import mock
except ImportError:
    import mock  # noqa: F401

import pytest

from xarray.backends.file_manager import FileManager, FILE_CACHE
from xarray.core.pycompat import suppress


@pytest.fixture(scope='module', params=[1, 2, 3])
def file_cache(request):
    contents = FILE_CACHE.items()
    maxsize = FILE_CACHE.maxsize
    FILE_CACHE.clear()
    FILE_CACHE.maxsize = request.param
    yield FILE_CACHE
    FILE_CACHE.maxsize = maxsize
    FILE_CACHE.clear()
    FILE_CACHE.update(contents)


def test_file_manager_mock_write(file_cache):
    mock_file = mock.Mock()
    opener = mock.Mock(return_value=mock_file)

    manager = FileManager(opener, 'filename')
    with manager.acquire() as f:
        f.write('contents')
    manager.close()

    opener.assert_called_once_with('filename')
    mock_file.write.assert_called_once_with('contents')
    mock_file.close.assert_called_once_with()


def test_file_manager_mock_error(file_cache):
    mock_file = mock.Mock()
    opener = mock.Mock(return_value=mock_file)

    manager = FileManager(opener, 'mydata')
    with suppress(ValueError):
        with manager.acquire():
            raise ValueError
    manager.close()

    opener.assert_called_once_with('mydata')
    mock_file.close.assert_called_once_with()


def test_file_manager_write_consecutive(tmpdir, file_cache):
    path = str(tmpdir.join('testing.txt'))
    manager = FileManager(open, path, mode='w')
    with manager.acquire() as f:
        f.write('foo')
    with manager.acquire() as f:
        f.write('bar')
    manager.close()

    with open(path, 'r') as f:
        assert f.read() == 'foobar'


def test_file_manager_write_concurrent(tmpdir, file_cache):
    path = str(tmpdir.join('testing.txt'))
    manager = FileManager(open, path, mode='w')
    with manager.acquire() as f1:
        with manager.acquire() as f2:
            with manager.acquire() as f3:
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
    manager = FileManager(open, path, mode='w')
    with manager.acquire() as f:
        f.write('foo')
        f.flush()
    manager2 = pickle.loads(pickle.dumps(manager))
    with manager2.acquire() as f:
        f.write('bar')
    manager2.close()
    manager.close()

    with open(path, 'r') as f:
        assert f.read() == 'foobar'


def test_file_manager_read(tmpdir, file_cache):
    path = str(tmpdir.join('testing.txt'))

    with open(path, 'w') as f:
        f.write('foobar')

    manager = FileManager(open, path)
    with manager.acquire() as f:
        assert f.read() == 'foobar'
    manager.close()
