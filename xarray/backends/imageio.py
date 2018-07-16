from __future__ import absolute_import, division, print_function

import functools

import numpy as np

from xarray import Variable
from xarray.core.indexing import NumpyIndexingAdapter
from xarray.core.pycompat import iteritems
from xarray.core.utils import Frozen, FrozenOrderedDict
from xarray.backends.common import (
    BackendArray, DataStorePickleMixin, AbstractWritableDataStore)


class ImageioArrayWrapper(BackendArray):

    def __init__(self, index, datastore):
        self.datastore = datastore
        self.index = index
        array = self.get_array()
        self.shape = array.shape
        self.dtype = np.dtype(array.dtype.kind +
                              str(array.dtype.itemsize))

    def get_array(self):
        self.datastore.assert_open()
        return self.datastore.ds.get_data(self.index)

    def __getitem__(self, key):
        with self.datastore.ensure_open(autoclose=True):
            data = NumpyIndexingAdapter(self.get_array())[key]
            # Since data is always read to memory for imageio no need for copy
            # to ensure their presence in the memory.
            return np.array(data, dtype=self.dtype, copy=False)


class ImageioDataStore(AbstractWritableDataStore, DataStorePickleMixin):
    """Store for reading and writing data via imageio.
    """

    def __init__(self, filename_or_obj, mode='r', format=None, order='xyzct',
                 writer=None, autoclose=False, lock=None, **kwargs):
        import imageio

        if format is None or format == 'volume':
            fmt_str = 'v'
        elif format == 'image':
            fmt_str = 'i'
        else:
            raise ValueError('invalid format for imageio backend: %r' % format)

        opener = functools.partial(imageio.get_reader,
                                   uri=filename_or_obj, mode=fmt_str, **kwargs)
        self._ds = opener()
        self._autoclose = autoclose
        self._isopen = True
        self._opener = opener
        self._mode = mode
        self._order = order

        super(ImageioDataStore, self).__init__(writer, lock=lock)

    def open_store_variable(self, index):
        """Read image data from the file, using the image index."""
        with self.ensure_open(autoclose=False):
            data = ImageioArrayWrapper(index, self)
            array = data.get_array()
            dimensions = {d: None for d in self._order[:array.ndim][::-1]}
            return Variable(dimensions, data, array.meta)

    def get_variables(self):
        with self.ensure_open(autoclose=False):
            return FrozenOrderedDict((str(i), self.open_store_variable(i))
                                     for i in range(len(self.ds)))

    def get_attrs(self):
        with self.ensure_open(autoclose=True):
            # 'get_meta_data' retrieves the file's global meta data when index
            # is omitted or None. Meta data for the returned image is returned
            # as an atrribute of that image when called by 'get_data'.
            return Frozen(self.ds.get_meta_data())

    def get_dimensions(self):
        with self.ensure_open(autoclose=True):
            # Data must exist at the first index, use it as the template.
            data = ImageioArrayWrapper(0, self)
            array = data.get_array()
            dim_key = self._order[:array.ndim][::-1]
            dim_len = data.shape
            return Frozen({k: v for k, v in zip(dim_key, dim_len)})

    # TODO
    def set_dimension(self, name, length, is_unlimited=False):
        with self.ensure_open(autoclose=False):
            if name in self.ds.dimensions:
                raise ValueError('%s does not support modifying dimensions'
                                 % type(self).__name__)
            dim_length = length if not is_unlimited else None
            self.ds.createDimension(name, dim_length)

    def set_attribute(self, key, value):
        with self.ensure_open(autoclose=False):
            self.ds[key] = value

    # TODO
    def prepare_variable(self, name, variable, check_encoding=False,
                         unlimited_dims=None):
        if check_encoding and variable.encoding:
            if variable.encoding != {'_FillValue': None}:
                raise ValueError('unexpected encoding for scipy backend: %r'
                                 % list(variable.encoding))

        data = variable.data
        # nb. this still creates a numpy array in all memory, even though we
        # don't write the data yet; scipy.io.netcdf does not not support
        # incremental writes.
        if name not in self.ds.variables:
            self.ds.createVariable(name, data.dtype, variable.dims)
        scipy_var = self.ds.variables[name]
        for k, v in iteritems(variable.attrs):
            setattr(scipy_var, k, v)

        target = ImageioArrayWrapper(name, self)

        return target, data

    # TODO
    def sync(self, compute=True):
        if not compute:
            raise NotImplementedError(
                'compute=False is not supported for the scipy backend yet')
        with self.ensure_open(autoclose=True):
            super(ImageioDataStore, self).sync(compute=compute)
            # TODO retrieve 'Writer' object
            raise NotImplementedError
            #self.ds.flush()

    def close(self):
        self.ds.close()
        self._isopen = False

    def __exit__(self, type, value, tb):
        self.close()

    def __setstate__(self, state):
        filename = state['_opener'].keywords['filename']
        if hasattr(filename, 'seek'):
            # it's a file-like object
            # seek to the start of the file so scipy can read it
            filename.seek(0)
        super(ImageioDataStore, self).__setstate__(state)
        self._ds = None
        self._isopen = False
