import functools
import pickle

import pytest

from xarray.backends.file_manager import (
    ExplicitFileManager, LazyFileManager, AutoclosingFileManager
)

FILE_MANAGERS = [
    ExplicitFileManager, LazyFileManager, AutoclosingFileManager,
]


@pytest.mark.parametrize('manager_type', FILE_MANAGERS)
def test_file_manager_write_consecutive(tmpdir, manager_type):
    path = str(tmpdir.join('testing.txt'))
    manager = manager_type(functools.partial(open, path), mode='w')
    with manager.acquire() as f:
        f.write('foo')
    with manager.acquire() as f:
        f.write('bar')
    manager.close()

    with open(path, 'r') as f:
        assert f.read() == 'foobar'


@pytest.mark.parametrize('manager_type', FILE_MANAGERS)
def test_file_manager_write_concurrent(tmpdir, manager_type):
    path = str(tmpdir.join('testing.txt'))
    manager = manager_type(functools.partial(open, path), mode='w')
    with manager.acquire() as f1:
        with manager.acquire() as f2:
            f1.write('foo')
            f2.write('bar')
    manager.close()

    with open(path, 'r') as f:
        assert f.read() == 'foobar'


@pytest.mark.parametrize('manager_type', FILE_MANAGERS)
def test_file_manager_write_pickle(tmpdir, manager_type):
    path = str(tmpdir.join('testing.txt'))
    manager = manager_type(
        functools.partial(open, path), mode='w')
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


@pytest.mark.parametrize('manager_type', FILE_MANAGERS)
def test_file_manager_read(tmpdir, manager_type):
    path = str(tmpdir.join('testing.txt'))

    with open(path, 'w') as f:
        f.write('foobar')

    manager = manager_type(functools.partial(open, path))
    with manager.acquire() as f:
        assert f.read() == 'foobar'
    manager.close()
