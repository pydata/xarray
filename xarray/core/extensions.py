from __future__ import annotations

import warnings
from typing import Callable, Generic, TypeVar, overload


class AccessorRegistrationWarning(Warning):
    """Warning for conflicts in accessor registration."""


_Accessor = TypeVar("_Accessor")


class _CachedAccessor(Generic[_Accessor]):
    """Custom property-like object (descriptor) for caching accessors."""

    _name: str
    _accessor: type[_Accessor]

    def __init__(self, name: str, accessor: type[_Accessor]):
        self._name = name
        self._accessor = accessor

    @overload
    def __get__(self, obj: None, cls) -> type[_Accessor]:
        ...

    @overload
    def __get__(self, obj: object, cls) -> _Accessor:
        ...

    def __get__(self, obj: None | object, cls) -> type[_Accessor] | _Accessor:
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor

        # Use the same dict as @pandas.util.cache_readonly.
        # It must be explicitly declared in obj.__slots__.
        try:
            cache = obj._cache  # type: ignore[attr-defined]
        except AttributeError:
            cache = obj._cache = {}  # type: ignore[attr-defined]

        try:
            return cache[self._name]
        except KeyError:
            pass

        try:
            accessor_obj = self._accessor(obj)  # type: ignore[call-arg]
        except AttributeError:
            # __getattr__ on data object will swallow any AttributeErrors
            # raised when initializing the accessor, so we need to raise as
            # something else (GH933):
            raise RuntimeError(f"error initializing {self._name!r} accessor.")

        cache[self._name] = accessor_obj
        return accessor_obj


def _register_accessor(
    name: str, cls: type[object]
) -> Callable[[type[_Accessor]], type[_Accessor]]:
    def decorator(accessor: type[_Accessor]) -> type[_Accessor]:
        if hasattr(cls, name):
            warnings.warn(
                f"registration of accessor {accessor!r} under name {name!r} for type {cls!r} is "
                "overriding a preexisting attribute with the same name.",
                AccessorRegistrationWarning,
                stacklevel=2,
            )
        setattr(cls, name, _CachedAccessor(name, accessor))
        return accessor

    return decorator


def register_dataarray_accessor(
    name: str,
) -> Callable[[type[_Accessor]], type[_Accessor]]:
    """Register a custom accessor on xarray.DataArray objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    See Also
    --------
    register_dataset_accessor
    """
    from .dataarray import DataArray

    return _register_accessor(name, DataArray)


def register_dataset_accessor(
    name: str,
) -> Callable[[type[_Accessor]], type[_Accessor]]:
    """Register a custom property on xarray.Dataset objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Examples
    --------
    In your library code:

    >>> @xr.register_dataset_accessor("geo")
    ... class GeoAccessor:
    ...     def __init__(self, xarray_obj):
    ...         self._obj = xarray_obj
    ...
    ...     @property
    ...     def center(self):
    ...         # return the geographic center point of this dataset
    ...         lon = self._obj.latitude
    ...         lat = self._obj.longitude
    ...         return (float(lon.mean()), float(lat.mean()))
    ...
    ...     def plot(self):
    ...         # plot this array's data on a map, e.g., using Cartopy
    ...         pass

    Back in an interactive IPython session:

    >>> ds = xr.Dataset(
    ...     {"longitude": np.linspace(0, 10), "latitude": np.linspace(0, 20)}
    ... )
    >>> ds.geo.center
    (10.0, 5.0)
    >>> ds.geo.plot()  # plots data on a map

    See Also
    --------
    register_dataarray_accessor
    """
    from .dataset import Dataset

    return _register_accessor(name, Dataset)
