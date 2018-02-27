# flake8: noqa

from __future__ import absolute_import, division, print_function

import sys

import numpy as np

PY2 = sys.version_info[0] < 3
PY3 = sys.version_info[0] >= 3

if PY3:  # pragma: no cover
    basestring = str
    unicode_type = str
    bytes_type = bytes
    native_int_types = (int,)

    def iteritems(d):
        return iter(d.items())

    def itervalues(d):
        return iter(d.values())

    range = range
    zip = zip
    from functools import reduce
    import builtins
    from urllib.request import urlretrieve
    from inspect import getfullargspec as getargspec
else:  # pragma: no cover
    # Python 2
    basestring = basestring  # noqa
    unicode_type = unicode  # noqa
    bytes_type = str
    native_int_types = (int, long)  # noqa

    def iteritems(d):
        return d.iteritems()

    def itervalues(d):
        return d.itervalues()

    range = xrange
    from itertools import izip as zip, imap as map
    reduce = reduce
    import __builtin__ as builtins
    from urllib import urlretrieve
    from inspect import getargspec

integer_types = native_int_types + (np.integer,)

try:
    from cyordereddict import OrderedDict
except ImportError:  # pragma: no cover
    try:
        from collections import OrderedDict
    except ImportError:
        from ordereddict import OrderedDict

try:
    # solely for isinstance checks
    import dask.array
    dask_array_type = (dask.array.Array,)
except ImportError:  # pragma: no cover
    dask_array_type = ()

try:
    try:
        from pathlib import Path
    except ImportError as e:
        from pathlib2 import Path
    path_type = (Path, )
except ImportError as e:
    path_type = ()


try:
    from contextlib import suppress
except ImportError:
    # Backport from CPython 3.5:
    # Used under the terms of Python's license, see licenses/PYTHON_LICENSE.

    class suppress:
        """Context manager to suppress specified exceptions

        After the exception is suppressed, execution proceeds with the next
        statement following the with statement.

             with suppress(FileNotFoundError):
                 os.remove(somefile)
             # Execution still resumes here if the file was already removed
        """

        def __init__(self, *exceptions):
            self._exceptions = exceptions

        def __enter__(self):
            pass

        def __exit__(self, exctype, excinst, exctb):
            # Unlike isinstance and issubclass, CPython exception handling
            # currently only looks at the concrete type hierarchy (ignoring
            # the instance and subclass checking hooks). While Guido considers
            # that a bug rather than a feature, it's a fairly hard one to fix
            # due to various internal implementation details. suppress provides
            # the simpler issubclass based semantics, rather than trying to
            # exactly reproduce the limitations of the CPython interpreter.
            #
            # See http://bugs.python.org/issue12029 for more details
            return exctype is not None and issubclass(
                exctype, self._exceptions)
try:
    from contextlib import ExitStack
except ImportError:
    # backport from Python 3.5:
    from collections import deque

    # Inspired by discussions on http://bugs.python.org/issue13585
    class ExitStack(object):
        """Context manager for dynamic management of a stack of exit callbacks

        For example:

            with ExitStack() as stack:
                files = [stack.enter_context(open(fname)) for fname in filenames]
                # All opened files will automatically be closed at the end of
                # the with statement, even if attempts to open files later
                # in the list raise an exception

        """

        def __init__(self):
            self._exit_callbacks = deque()

        def pop_all(self):
            """Preserve the context stack by transferring it to a new instance"""
            new_stack = type(self)()
            new_stack._exit_callbacks = self._exit_callbacks
            self._exit_callbacks = deque()
            return new_stack

        def _push_cm_exit(self, cm, cm_exit):
            """Helper to correctly register callbacks to __exit__ methods"""
            def _exit_wrapper(*exc_details):
                return cm_exit(cm, *exc_details)
            _exit_wrapper.__self__ = cm
            self.push(_exit_wrapper)

        def push(self, exit):
            """Registers a callback with the standard __exit__ method signature

            Can suppress exceptions the same way __exit__ methods can.

            Also accepts any object with an __exit__ method (registering a call
            to the method instead of the object itself)
            """
            # We use an unbound method rather than a bound method to follow
            # the standard lookup behaviour for special methods
            _cb_type = type(exit)
            try:
                exit_method = _cb_type.__exit__
            except AttributeError:
                # Not a context manager, so assume its a callable
                self._exit_callbacks.append(exit)
            else:
                self._push_cm_exit(exit, exit_method)
            return exit  # Allow use as a decorator

        def callback(self, callback, *args, **kwds):
            """Registers an arbitrary callback and arguments.

            Cannot suppress exceptions.
            """
            def _exit_wrapper(exc_type, exc, tb):
                callback(*args, **kwds)
            # We changed the signature, so using @wraps is not appropriate, but
            # setting __wrapped__ may still help with introspection
            _exit_wrapper.__wrapped__ = callback
            self.push(_exit_wrapper)
            return callback  # Allow use as a decorator

        def enter_context(self, cm):
            """Enters the supplied context manager

            If successful, also pushes its __exit__ method as a callback and
            returns the result of the __enter__ method.
            """
            # We look up the special methods on the type to match the with
            # statement
            _cm_type = type(cm)
            _exit = _cm_type.__exit__
            result = _cm_type.__enter__(cm)
            self._push_cm_exit(cm, _exit)
            return result

        def close(self):
            """Immediately unwind the context stack"""
            self.__exit__(None, None, None)

        def __enter__(self):
            return self

        def __exit__(self, *exc_details):
            received_exc = exc_details[0] is not None

            # We manipulate the exception state so it behaves as though
            # we were actually nesting multiple with statements
            frame_exc = sys.exc_info()[1]

            def _fix_exception_context(new_exc, old_exc):
                # Context may not be correct, so find the end of the chain
                while True:
                    exc_context = new_exc.__context__
                    if exc_context is old_exc:
                        # Context is already set correctly (see issue 20317)
                        return
                    if exc_context is None or exc_context is frame_exc:
                        break
                    new_exc = exc_context
                # Change the end of the chain to point to the exception
                # we expect it to reference
                new_exc.__context__ = old_exc

            # Callbacks are invoked in LIFO order to match the behaviour of
            # nested context managers
            suppressed_exc = False
            pending_raise = False
            while self._exit_callbacks:
                cb = self._exit_callbacks.pop()
                try:
                    if cb(*exc_details):
                        suppressed_exc = True
                        pending_raise = False
                        exc_details = (None, None, None)
                except BaseException:
                    new_exc_details = sys.exc_info()
                    # simulate the stack of exceptions by setting the context
                    _fix_exception_context(new_exc_details[1], exc_details[1])
                    pending_raise = True
                    exc_details = new_exc_details
            if pending_raise:
                try:
                    # bare "raise exc_details[1]" replaces our carefully
                    # set-up context
                    fixed_ctx = exc_details[1].__context__
                    raise exc_details[1]
                except BaseException:
                    exc_details[1].__context__ = fixed_ctx
                    raise
            return received_exc and suppressed_exc
