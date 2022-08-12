import inspect
import warnings
from functools import wraps

POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY


def _deprecate_positional_args(version):
    """Decorator for methods that issues warnings for positional arguments

    Using the keyword-only argument syntax in pep 3102, arguments after the
    ``*`` will issue a warning when passed as a positional argument.

    Parameters
    ----------
    version : str
        version of the library when the positional arguments were deprecated

    Examples
    --------
    Deprecate passing `b` as positional argument:

    def func(a, b=1):
        pass

    @_deprecate_positional_args("v0.1.0")
    def func(a, *, b=2):
        pass

    func(1, 2)

    Notes
    -----
    This function is adapted from scikit-learn under the terms of its license. See
    licences/SCIKIT_LEARN_LICENSE
    """

    def _decorator(f):

        signature = inspect.signature(f)

        pos_or_kw_args = []
        kwonly_args = []
        for name, param in signature.parameters.items():
            if param.kind == POSITIONAL_OR_KEYWORD:
                pos_or_kw_args.append(name)
            elif param.kind == KEYWORD_ONLY:
                kwonly_args.append(name)
            elif param.kind == POSITIONAL_ONLY:
                raise TypeError("Cannot handle positional-only params")
                # because all args are coverted to kwargs below

        @wraps(f)
        def inner(*args, **kwargs):
            n_extra_args = len(args) - len(pos_or_kw_args)
            if n_extra_args > 0:

                extra_args = ", ".join(kwonly_args[:n_extra_args])

                warnings.warn(
                    f"Passing '{extra_args}' as positional argument(s) "
                    f"was deprecated in version {version} and will raise an error two "
                    "releases later. Please pass them as keyword arguments."
                    "",
                    FutureWarning,
                )

            kwargs.update({name: arg for name, arg in zip(pos_or_kw_args, args)})
            return f(**kwargs)

        return inner

    return _decorator
