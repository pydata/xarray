import pickle

from xarray.backends.file_manager import FileManager, FILE_CACHE


def test_file_manager_write_consecutive(tmpdir):
    path = str(tmpdir.join('testing.txt'))
    manager = FileManager(open, path, mode='w')
    with manager.acquire() as f:
        f.write('foo')
    with manager.acquire() as f:
        f.write('bar')
    manager.close()

    with open(path, 'r') as f:
        assert f.read() == 'foobar'


def test_file_manager_write_concurrent(tmpdir):
    path = str(tmpdir.join('testing.txt'))
    manager = FileManager(open, path, mode='w')
    with manager.acquire() as f1:
        with manager.acquire() as f2:
            f1.write('foo')
            f2.write('bar')
    manager.close()

    with open(path, 'r') as f:
        assert f.read() == 'foobar'


def test_file_manager_write_pickle(tmpdir):
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


def test_file_manager_read(tmpdir):
    path = str(tmpdir.join('testing.txt'))

    with open(path, 'w') as f:
        f.write('foobar')

    manager = FileManager(open, path)
    with manager.acquire() as f:
        assert f.read() == 'foobar'
    manager.close()


# TODO(shoyer): add test coverage for exceeding the max size of the file cache
