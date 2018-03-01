from __future__ import absolute_import, division, print_function

import traceback
import warnings

from .dataarray import DataArray
from .dataset import Dataset
from .pycompat import PY2


class AccessorRegistrationWarning(Warning):
    """Warning for conflicts in accessor registration."""


class _CachedAccessor(object):
    """Custom property-like object (descriptor) for caching accessors."""

    def __init__(self, name, accessor):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor
        try:
            accessor_obj = self._accessor(obj)
        except AttributeError:
            # __getattr__ on data object will swallow any AttributeErrors
            # raised when initializing the accessor, so we need to raise as
            # something else (GH933):
            msg = 'error initializing %r accessor.' % self._name
            if PY2:
                msg += ' Full traceback:\n' + traceback.format_exc()
            raise RuntimeError(msg)
        # Replace the property with the accessor object. Inspired by:
        # http://www.pydanny.com/cached-property.html
        # We need to use object.__setattr__ because we overwrite __setattr__ on
        # AttrAccessMixin.
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


def _register_accessor(name, cls):
    def decorator(accessor):
        if hasattr(cls, name):
            warnings.warn(
                'registration of accessor %r under name %r for type %r is '
                'overriding a preexisting attribute with the same name.'
                % (accessor, name, cls),
                AccessorRegistrationWarning,
                stacklevel=2)
        setattr(cls, name, _CachedAccessor(name, accessor))
        return accessor
    return decorator


def register_dataarray_accessor(name):
    """Register a custom accessor on xarray.DataArray objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    See also
    --------
    register_dataset_accessor
    """
    return _register_accessor(name, DataArray)


def register_dataset_accessor(name):
    """Register a custom property on xarray.Dataset objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Examples
    --------

    In your library code::

        import xarray as xr

        @xr.register_dataset_accessor('geo')
        class GeoAccessor(object):
            def __init__(self, xarray_obj):
                self._obj = xarray_obj

            @property
            def center(self):
                # return the geographic center point of this dataset
                lon = self._obj.latitude
                lat = self._obj.longitude
                return (float(lon.mean()), float(lat.mean()))

            def plot(self):
                # plot this array's data on a map, e.g., using Cartopy
                pass

    Back in an interactive IPython session:

        >>> ds = xarray.Dataset({'longitude': np.linspace(0, 10),
        ...                      'latitude': np.linspace(0, 20)})
        >>> ds.geo.center
        (5.0, 10.0)
        >>> ds.geo.plot()
        # plots data on a map

    See also
    --------
    register_dataarray_accessor
    """
    return _register_accessor(name, Dataset)
