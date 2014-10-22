from collections import Mapping
from contextlib import contextmanager

from .pycompat import iteritems, basestring
from . import formatting
from . import utils


def _coord_merge_finalize(target, other, target_conflicts, other_conflicts):
    for k in target_conflicts:
        del target[k]
    for k, v in iteritems(other):
        if k not in other_conflicts:
            target[k] = v.variable


class AbstractCoordinates(Mapping):
    @property
    def _names(self):
        return self._dataset._coord_names

    def __getitem__(self, key):
        if (key in self._names
                or (isinstance(key, basestring)
                    and key.split('.')[0] in self._names)):
            # allow indexing current coordinates or components
            return self._dataset[key]
        else:
            raise KeyError(key)

    def __iter__(self):
        # needs to be in the same order as the dataset variables
        for k in self._dataset._arrays:
            if k in self._names:
                yield k

    def __len__(self):
        return len(self._names)

    def __contains__(self, key):
        return key in self._names

    def __delitem__(self, key):
        if key in self:
            del self._dataset[key]
        else:
            raise KeyError(key)

    def __repr__(self):
        return formatting.coords_repr(self)

    @property
    def dims(self):
        return self._dataset.dims

    def to_dataset(self):
        """Convert these coordinates into a new Dataset
        """
        return self._dataset._copy_listed(self._names)

    def to_index(self, ordered_dims=None):
        """Convert all index coordinates into a :py:class:`pandas.MultiIndex`
        """
        if ordered_dims is None:
            ordered_dims = self.dims
        indexes = [self._dataset._arrays[k].to_index() for k in ordered_dims]
        return utils.multi_index_from_product(indexes,
                                              names=list(ordered_dims))

    def _merge_validate(self, other):
        """Determine conflicting variables to be dropped from either self or
        other (or unresolvable conflicts that should just raise)
        """
        self_conflicts = set()
        other_conflicts = set()
        for k in self:
            if k in other:
                var = self._dataset._arrays[k]
                if not var.equals(other[k]):
                    in_self_dims = k in self.dims
                    in_other_dims = k in other.dims
                    if in_self_dims and in_other_dims:
                        raise ValueError('index %r not aligned' % k)
                    if not in_self_dims:
                        self_conflicts.add(k)
                    if not in_other_dims:
                        other_conflicts.add(k)
        return self_conflicts, other_conflicts

    @contextmanager
    def _merge_inplace(self, other):
        if other is None:
            yield
        else:
            # ignore conflicts in self because we don't want to remove
            # existing coords in an in-place update
            _, other_conflicts = self._merge_validate(other)
            yield
            _coord_merge_finalize(self, other, {}, other_conflicts)

    def merge(self, other):
        """Merge two sets of coordinates to create a new Dataset

        The method implments the logic used for joining coordinates in the
        result of a binary operation performed on xray objects:

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
    dimensions and the values given by the corresponding xray.Coordinate
    objects.
    """
    def __init__(self, dataset):
        self._dataset = dataset

    def __setitem__(self, key, value):
        self._dataset[key] = value
        self._names.add(key)


class DataArrayCoordinates(AbstractCoordinates):
    """Dictionary like container for DataArray coordinates.

    Essentially an OrderedDict with keys given by the array's
    dimensions and the values given by the corresponding xray.Coordinate
    objects.
    """
    def __init__(self, dataarray):
        self._dataarray = dataarray
        self._dataset = dataarray._dataset

    def __setitem__(self, key, value):
        with self._dataarray._set_new_dataset() as ds:
            ds.coords[key] = value
            bad_dims = [d for d in ds._arrays[key].dims
                        if d not in self.dims]
            if bad_dims:
                raise ValueError('DataArray does not include all coordinate '
                                 'dimensions: %s' % bad_dims)

    @property
    def dims(self):
        return self._dataarray.dims


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
