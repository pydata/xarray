from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import warnings
from itertools import product
from collections import MutableMapping

from .. import Variable
from ..core import indexing
from ..core.utils import FrozenOrderedDict, close_on_error, HiddenKeyDict
from ..core.pycompat import iteritems, bytes_type, unicode_type, OrderedDict

from .common import (WritableCFDataStore, AbstractWritableDataStore,
                     DataStorePickleMixin)




# most of the other stores have some kind of wrapper class like
# class BaseNetCDF4Array(NdimSizeLenMixin, DunderArrayMixin):
# class H5NetCDFArrayWrapper(BaseNetCDF4Array):
# class NioArrayWrapper(NdimSizeLenMixin, DunderArrayMixin):
# we problaby need something like this

# the first question is whether it should be based on BaseNetCDF4Array or
# NdimSizeLenMixing?

# or maybe we don't need wrappers at all? probably not true


# also most have a custom opener

# keyword args for zarr.group
# store=None, overwrite=False, chunk_store=None, synchronizer=None, path=None
# the group name is called "path" in the zarr lexicon

def _open_zarr_group(store, overwrite, chunk_store, synchronizer, path):
    import zarr
    zarr_group = zarr.group(store=store, overwrite=overwrite,
            chunk_store=chunk_store, synchronizer=synchronizer, path=path)
    return zarr_group


def _dask_chunks_to_zarr_chunks(chunks):
    # zarr chunks needs to be uniform for each array
    # http://zarr.readthedocs.io/en/latest/spec/v1.html#chunks
    # dask chunks can be variable sized
    # http://dask.pydata.org/en/latest/array-design.html#chunks
    # this function dask chunks syntax to zarr chunks
    if chunks is None:
        return chunks

    all_chunks = product(*chunks)
    first_chunk = all_chunks.next()
    for this_chunk in all_chunks:
        if not (this_chunk == first_chunk):
            raise ValueError("zarr requires uniform chunk sizes, found %s" %
                             repr(chunks))
    return first_chunk


def _get_zarr_dims_and_attrs(zarr_obj, dimension_key):
    # Zarr arrays do not have dimenions. To get around this problem, we add
    # an attribute that specifies the dimension. We have to hide this attribute
    # when we send the attributes to the user.
    # zarr_obj can be either a zarr group or zarr array
    dimensions = zarr_obj.attrs.get(dimension_key)
    attributes = HiddenKeyDict(zarr_obj.attrs, dimension_key)
    return dimensions, attributes


class ZarrStore(AbstractWritableDataStore, DataStorePickleMixin):
    """Store for reading and writing data via zarr
    """

    # need some special secret attributes to tell us the dimensions
    _dimension_key = '_XARRAY_DIMENSIONS'

    def __init__(self, store=None, overwrite=False, chunk_store=None,
                 synchronizer=None, path=None, writer=None, autoclose=False):
        opener = functools.partial(_open_zarr_group, store, overwrite,
                                   chunk_store, synchronizer, path)
        self.ds = opener()
        if autoclose:
            raise NotImplementedError('autoclose=True is not implemented '
                                      'for the zarr backend')
        self._autoclose = False
        self._isopen = True
        self._opener = opener

        # initialize hidden dimension attribute
        self.ds.attrs[self._dimension_key] = {}

        # do we need to define attributes for all of the opener keyword args?
        super(ZarrStore, self).__init__(writer)

    def open_store_variable(self, name, zarr_array):
        # I don't see why it is necessary to wrap self.ds[name]
        # zarr seems to implement the required ndarray interface
        # TODO: possibly wrap zarr array in dask with aligned chunks
        data = indexing.LazilyIndexedArray(zarr_array)
        dimensions, attributes = _get_zarr_dims_and_attrs(
                                    zarr_array, self._dimension_key)
        return Variable(dimensions, data, attributes)

    def get_variables(self):
        with self.ensure_open(autoclose=False):
            return FrozenOrderedDict((k, self.open_store_variable(k, v))
                                     for k, v in self.ds.arrays())

    def get_attrs(self):
        with self.ensure_open(autoclose=True):
            _, attributes = _get_zarr_dims_and_attrs(self.ds,
                                                     self._dimension_key)
            attrs = FrozenOrderedDict(attributes)
            return attrs

    def get_dimensions(self):
        with self.ensure_open(autoclose=True):
            dimensions, _ = _get_zarr_dims_and_attrs(self.ds,
                                                     self._dimension_key)
            return dimensions

    def set_dimension(self, name, length):
        with self.ensure_open(autoclose=False):
            self.ds.attrs[self._dimension_key][name] = length

    def set_attribute(self, key, value):
        with self.ensure_open(autoclose=False):
            _, attributes = _get_zarr_dims_and_attrs(self.ds,
                                self._dimension_key)
            attributes[key] = value

    def prepare_variable(self, name, variable, check_encoding=False,
                         unlimited_dims=None):

        attrs = variable.attrs.copy()
        dims = variable.dims
        dtype = variable.dtype
        shape = variable.shape
        chunks = _dask_chunks_to_zarr_chunks(variable.chunks)

        # TODO: figure ouw how zarr should deal with unlimited dimensions
        self.set_necessary_dimensions(variable, unlimited_dims=unlimited_dims)

        # let's try keeping this fill value stuff
        fill_value = attrs.pop('_FillValue', None)
        if fill_value in ['\x00']:
            fill_value = None

        # TODO: figure out what encoding is needed for zarr

        ### arguments for zarr.create
        # zarr.creation.create(shape, chunks=None, dtype=None, compressor='default',
        # fill_value=0, order='C', store=None, synchronizer=None, overwrite=False,
        # path=None, chunk_store=None, filters=None, cache_metadata=True, **kwargs)

        # TODO: figure out how to pass along all those other arguments

        zarr_array = self.ds.create(name, shape=shape, dtype=dtype,
                                    chunks=chunks, fill_value=fill_value)
        zarr_array.attrs[self._dimension_key] = dims
        _, attributes = _get_zarr_dims_and_attrs(zarr_array,
                                                 self._dimension_key)

        for k, v in iteritems(attrs):
            attributes[k] = v

        return zarr_array, variable.data

    # sync() and close() methods should not be needed with zarr
