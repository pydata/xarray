import warnings
from contextlib import contextmanager


@contextmanager
def suppress_warning(category, message=""):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=category, message=message)

        yield
