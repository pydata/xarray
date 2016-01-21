from collections import Mapping
from contextlib import contextmanager
import pandas as pd

from . import formatting
from .merge import merge_dataarray_coords
from .pycompat import iteritems, basestring, OrderedDict


def _coord_merge_finalize(target, other, target_conflicts, other_conflicts,
                          promote_dims=None):
    if promote_dims is None:
        promote_dims = {}
    for k in target_conflicts:
        del target[k]
    for k, v in iteritems(other):
        if k not in other_conflicts:
            var = v.variable
            if k in promote_dims:
                var = var.expand_dims(promote_dims[k])
            target[k] = var


def _common_shape(*args):
    dims = OrderedDict()
    for arg in args:
        for dim in arg.dims:
            size = arg.shape[arg.get_axis_num(dim)]
            if dim in dims and size != dims[dim]:
                # sometimes we may not have checked the index first
                raise ValueError('index %r not aligned' % dim)
            dims[dim] = size
    return dims


def _dim_shape(var):
    return [(dim, size) for dim, size in zip(var.dims, var.shape)]


class AbstractCoordinates(Mapping):
    def __getitem__(self, key):
        if (key in self._names or
            (isinstance(key, basestring) and
             key.split('.')[0] in self._names)):
            # allow indexing current coordinates or components
            return self._data[key]
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        self.update({key: value})

    def __iter__(self):
        # needs to be in the same order as the dataset variables
        for k in self._variables:
            if k in self._names:
                yield k

    def __len__(self):
        return len(self._names)

    def __contains__(self, key):
        return key in self._names

    def __repr__(self):
        return formatting.coords_repr(self)

    @property
    def dims(self):
        return self._data.dims

    def to_index(self, ordered_dims=None):
        """Convert all index coordinates into a :py:class:`pandas.MultiIndex`
        """
        if ordered_dims is None:
            ordered_dims = self.dims
        indexes = [self._variables[k].to_index() for k in ordered_dims]
        return pd.MultiIndex.from_product(indexes, names=list(ordered_dims))

    def _merge_validate(self, other):
        """Determine conflicting variables to be dropped from either self or
        other (or unresolvable conflicts that should just raise)
        """
        self_conflicts = set()
        other_conflicts = set()
        promote_dims = {}
        for k in self:
            if k in other:
                self_var = self._variables[k]
                other_var = other[k].variable
                if not self_var.broadcast_equals(other_var):
                    if k in self.dims and k in other.dims:
                        raise ValueError('index %r not aligned' % k)
                    if k not in self.dims:
                        self_conflicts.add(k)
                    if k not in other.dims:
                        other_conflicts.add(k)
                elif _dim_shape(self_var) != _dim_shape(other_var):
                    promote_dims[k] = _common_shape(self_var, other_var)
                    self_conflicts.add(k)
        return self_conflicts, other_conflicts, promote_dims

    @contextmanager
    def _merge_inplace(self, other):
        if other is None:
            yield
        else:
            # ignore conflicts in self because we don't want to remove
            # existing coords in an in-place update
            _, other_conflicts, promote_dims = self._merge_validate(other)
            # treat promoted dimensions as a conflict, also because we don't
            # want to modify existing coords
            other_conflicts.update(promote_dims)
            yield
            _coord_merge_finalize(self, other, {}, other_conflicts)

    def merge(self, other):
        """Merge two sets of coordinates to create a new Dataset

        The method implments the logic used for joining coordinates in the
        result of a binary operation performed on xarray objects:

        - If two index coordinates conflict (are not equal), an exception is
          raised.
        - If an index coordinate and a non-index coordinate conflict, the non-
          index coordinate is dropped.
        - If two non-index coordinates conflict, both are dropped.

        Parameters
        ----------
        other : DatasetCoordinates or DataArrayCoordinates
            The coordinates from another dataset or data array.

        Returns
        -------
        merged : Dataset
            A new Dataset with merged coordinates.
        """
        ds = self.to_dataset()
        if other is not None:
            conflicts = self._merge_validate(other)
            _coord_merge_finalize(ds.coords, other, *conflicts)
        return ds


class DatasetCoordinates(AbstractCoordinates):
    """Dictionary like container for Dataset coordinates.

    Essentially an immutable OrderedDict with keys given by the array's
    dimensions and the values given by the corresponding xarray.Coordinate
    objects.
    """
    def __init__(self, dataset):
        self._data = dataset

    @property
    def _names(self):
        return self._data._coord_names

    @property
    def _variables(self):
        return self._data._variables

    def to_dataset(self):
        """Convert these coordinates into a new Dataset
        """
        return self._data._copy_listed(self._names)

    def update(self, other):
        self._data.update(other)
        self._names.update(other.keys())

    def __delitem__(self, key):
        if key in self:
            del self._data[key]
        else:
            raise KeyError(key)


class DataArrayCoordinates(AbstractCoordinates):
    """Dictionary like container for DataArray coordinates.

    Essentially an OrderedDict with keys given by the array's
    dimensions and the values given by the corresponding xarray.Coordinate
    objects.
    """
    def __init__(self, dataarray):
        self._data = dataarray

    @property
    def _names(self):
        return set(self._data._coords)

    @property
    def _variables(self):
        return self._data._coords

    def _to_dataset(self, shallow_copy=True):
        from .dataset import Dataset
        coords = OrderedDict((k, v.copy(deep=False) if shallow_copy else v)
                             for k, v in self._data._coords.items())
        dims = dict(zip(self.dims, self._data.shape))
        return Dataset._construct_direct(coords, coord_names=set(self._names),
                                         dims=dims, attrs=None)

    def to_dataset(self):
        return self._to_dataset()

    def update(self, other):
        new_vars = merge_dataarray_coords(
            self._data.indexes, self._data._coords, other)

        self._data._coords = new_vars

    def __delitem__(self, key):
        if key in self.dims:
            raise ValueError('cannot delete a coordinate corresponding to a '
                             'DataArray dimension')
        del self._data._coords[key]


class Indexes(Mapping):
    def __init__(self, source):
        self._source = source

    def __iter__(self):
        return iter(self._source.dims)

    def __len__(self):
        return len(self._source.dims)

    def __contains__(self, key):
        return key in self._source.dims

    def __getitem__(self, key):
        if key in self:
            return self._source[key].to_index()
        else:
            raise KeyError(key)

    def __repr__(self):
        return formatting.indexes_repr(self)
