import contextlib
import threading


class FileManager(object):
    """Base class for context managers for managing file objects.

    Unlike files, FileManager objects should be safely. They must be explicitly
    closed.

    Example usage:

        import functools

        manager = FileManager(functools.partial(open, filename), mode='w')
        with manager.acquire() as f:
            f.write(...)
        manager.close()
    """

    def __init__(self, opener, mode=None):
        """Initialize a FileManager.

        Parameters
        ----------
        opener : callable
            Callable that opens a given file when called, returning a file
            object.
        mode : str, optional
            If provided, passed to opener as a keyword argument. 
        """
        raise NotImplementedError

    @contextlib.contextmanager
    def acquire(self):
        """Context manager for acquiring a file object.

        This method must be thread-safe: it should be safe to simultaneously
        acquire a file in multiple threads at the same time (assuming that
        the underlying file object is thread-safe).

        Yields
        ------
        Open file object, as returned by opener().
        """
        raise NotImplementedError

    def close(self):
        """Explicitly close any associated file object (if necessary)."""
        raise NotImplementedError


_DEFAULT_MODE = object()


def _open(opener, mode):
    return opener() if mode is _DEFAULT_MODE else opener(mode=mode)


class ExplicitFileManager(FileManager):
    """A file manager that holds a file open until explicitly closed.

    This is mostly a reference implementation: must real use cases should use
    ExplicitLazyFileContext for better performance.
    """

    def __init__(self, opener, mode=_DEFAULT_MODE):
        self._opener = opener
        # file has already been created, don't override when restoring
        self._mode = 'a' if mode == 'w' else mode
        self._file = _open(opener, mode)

    @contextlib.contextmanager
    def acquire(self):
        yield self._file

    def close(self):
        self._file.close()

    def __getstate__(self):
        return {'opener': self._opener, 'mode': self._mode}

    def __setstate__(self, state):
        self.__init__(**state)


class LazyFileManager(FileManager):
    """An explicit file manager that lazily opens files."""

    def __init__(self, opener, mode=_DEFAULT_MODE):
        self._opener = opener
        self._mode = mode
        self._lock = threading.Lock()
        self._file = None

    @contextlib.contextmanager
    def acquire(self):
        with self._lock:
            if self._file is None:
                self._file = _open(self._opener, self._mode)
                # file has already been created, don't override when restoring
                if self._mode == 'w':
                    self._mode = 'a'
        yield self._file

    def close(self):
        if self._file is not None:
            self._file.close()

    def __getstate__(self):
        return {'opener': self._opener, 'mode': self._mode}

    def __setstate__(self, state):
        self.__init__(**state)


class AutoclosingFileManager(FileManager):
    """A FileManager that automatically opens/closes files when used."""

    def __init__(self, opener, mode=_DEFAULT_MODE):
        self._opener = opener
        self._mode = mode
        self._lock = threading.Lock()
        self._file = None
        self._references = 0

    @contextlib.contextmanager
    def acquire(self):
        with self._lock:
            if self._file is None:
                self._file = _open(self._opener, self._mode)
                # file has already been created, don't override when restoring
                if self._mode == 'w':
                    self._mode = 'a'
            self._references += 1

        yield self._file

        with self._lock:
            self._references -= 1
            if not self._references:
                self._file.close()
                self._file = None

    def close(self):
        pass

    def __getstate__(self):
        return {'opener': self._opener, 'mode': self._mode}

    def __setstate__(self, state):
        self.__init__(**state)


# TODO: write a FileManager that makes use of an LRU cache.
