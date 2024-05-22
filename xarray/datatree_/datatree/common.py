"""
This file and class only exists because it was easier to copy the code for AttrAccessMixin from xarray.core.common
with some slight modifications than it was to change the behaviour of an inherited xarray internal here.

The modifications are marked with # TODO comments.
"""

import warnings
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from typing import Any


class TreeAttrAccessMixin:
    """Mixin class that allows getting keys with attribute access"""

    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        """Verify that all subclasses explicitly define ``__slots__``. If they don't,
        raise error in the core xarray module and a FutureWarning in third-party
        extensions.
        """
        if not hasattr(object.__new__(cls), "__dict__"):
            pass
        # TODO reinstate this once integrated upstream
        # elif cls.__module__.startswith("datatree."):
        #    raise AttributeError(f"{cls.__name__} must explicitly define __slots__")
        # else:
        #    cls.__setattr__ = cls._setattr_dict
        #    warnings.warn(
        #        f"xarray subclass {cls.__name__} should explicitly define __slots__",
        #        FutureWarning,
        #        stacklevel=2,
        #    )
        super().__init_subclass__(**kwargs)

    @property
    def _attr_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for attribute-style access"""
        yield from ()

    @property
    def _item_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for key-autocompletion"""
        yield from ()

    def __getattr__(self, name: str) -> Any:
        if name not in {"__dict__", "__setstate__"}:
            # this avoids an infinite loop when pickle looks for the
            # __setstate__ attribute before the xarray object is initialized
            for source in self._attr_sources:
                with suppress(KeyError):
                    return source[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    # This complicated two-method design boosts overall performance of simple operations
    # - particularly DataArray methods that perform a _to_temp_dataset() round-trip - by
    # a whopping 8% compared to a single method that checks hasattr(self, "__dict__") at
    # runtime before every single assignment. All of this is just temporary until the
    # FutureWarning can be changed into a hard crash.
    def _setattr_dict(self, name: str, value: Any) -> None:
        """Deprecated third party subclass (see ``__init_subclass__`` above)"""
        object.__setattr__(self, name, value)
        if name in self.__dict__:
            # Custom, non-slotted attr, or improperly assigned variable?
            warnings.warn(
                f"Setting attribute {name!r} on a {type(self).__name__!r} object. Explicitly define __slots__ "
                "to suppress this warning for legitimate custom attributes and "
                "raise an error when attempting variables assignments.",
                FutureWarning,
                stacklevel=2,
            )

    def __setattr__(self, name: str, value: Any) -> None:
        """Objects with ``__slots__`` raise AttributeError if you try setting an
        undeclared attribute. This is desirable, but the error message could use some
        improvement.
        """
        try:
            object.__setattr__(self, name, value)
        except AttributeError as e:
            # Don't accidentally shadow custom AttributeErrors, e.g.
            # DataArray.dims.setter
            if str(e) != f"{type(self).__name__!r} object has no attribute {name!r}":
                raise
            raise AttributeError(
                f"cannot set attribute {name!r} on a {type(self).__name__!r} object. Use __setitem__ style"
                "assignment (e.g., `ds['name'] = ...`) instead of assigning variables."
            ) from e

    def __dir__(self) -> list[str]:
        """Provide method name lookup and completion. Only provide 'public'
        methods.
        """
        extra_attrs = {
            item
            for source in self._attr_sources
            for item in source
            if isinstance(item, str)
        }
        return sorted(set(dir(type(self))) | extra_attrs)
