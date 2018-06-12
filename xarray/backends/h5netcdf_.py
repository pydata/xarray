from __future__ import absolute_import, division, print_function

import functools

import numpy as np

from .. import Variable
from ..core import indexing
from ..core.pycompat import OrderedDict, bytes_type, iteritems, unicode_type
from ..core.utils import FrozenOrderedDict, close_on_error
from .common import (
    HDF5_LOCK, DataStorePickleMixin, WritableCFDataStore, find_root)
from .netCDF4_ import (
    BaseNetCDF4Array, _encode_nc4_variable, _extract_nc4_variable_encoding,
    _get_datatype, _nc4_require_group)


class H5NetCDFArrayWrapper(BaseNetCDF4Array):
    def __getitem__(self, key):
        key, np_inds = indexing.decompose_indexer(
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR)

        # h5py requires using lists for fancy indexing:
        # https://github.com/h5py/h5py/issues/992
        key = tuple(list(k) if isinstance(k, np.ndarray) else k for k in
                    key.tuple)
        with self.datastore.ensure_open(autoclose=True):
            array = self.get_array()[key]

        if len(np_inds.tuple) > 0:
            array = indexing.NumpyIndexingAdapter(array)[np_inds]

        return array


def maybe_decode_bytes(txt):
    if isinstance(txt, bytes_type):
        return txt.decode('utf-8')
    else:
        return txt


def _read_attributes(h5netcdf_var):
    # GH451
    # to ensure conventions decoding works properly on Python 3, decode all
    # bytes attributes to strings
    attrs = OrderedDict()
    for k, v in h5netcdf_var.attrs.items():
        if k not in ['_FillValue', 'missing_value']:
            v = maybe_decode_bytes(v)
        attrs[k] = v
    return attrs


_extract_h5nc_encoding = functools.partial(
    _extract_nc4_variable_encoding,
    lsd_okay=False, h5py_okay=True, backend='h5netcdf')


def _h5netcdf_create_group(dataset, name):
    return dataset.create_group(name)


def _open_h5netcdf_group(filename, mode, group):
    import h5netcdf
    ds = h5netcdf.File(filename, mode=mode)
    with close_on_error(ds):
        return _nc4_require_group(
            ds, group, mode, create_group=_h5netcdf_create_group)


class H5NetCDFStore(WritableCFDataStore, DataStorePickleMixin):
    """Store for reading and writing data via h5netcdf
    """

    def __init__(self, filename, mode='r', format=None, group=None,
                 writer=None, autoclose=False, lock=HDF5_LOCK):
        if format not in [None, 'NETCDF4']:
            raise ValueError('invalid format for h5netcdf backend')
        opener = functools.partial(_open_h5netcdf_group, filename, mode=mode,
                                   group=group)
        self._ds = opener()
        if autoclose:
            raise NotImplementedError('autoclose=True is not implemented '
                                      'for the h5netcdf backend pending '
                                      'further exploration, e.g., bug fixes '
                                      '(in h5netcdf?)')
        self._autoclose = False
        self._isopen = True
        self.format = format
        self._opener = opener
        self._filename = filename
        self._mode = mode
        super(H5NetCDFStore, self).__init__(writer, lock=lock)

    def open_store_variable(self, name, var):
        import h5py

        with self.ensure_open(autoclose=False):
            dimensions = var.dimensions
            data = indexing.LazilyOuterIndexedArray(
                H5NetCDFArrayWrapper(name, self))
            attrs = _read_attributes(var)

            # netCDF4 specific encoding
            encoding = {
                'chunksizes': var.chunks,
                'fletcher32': var.fletcher32,
                'shuffle': var.shuffle,
            }
            # Convert h5py-style compression options to NetCDF4-Python
            # style, if possible
            if var.compression == 'gzip':
                encoding['zlib'] = True
                encoding['complevel'] = var.compression_opts
            elif var.compression is not None:
                encoding['compression'] = var.compression
                encoding['compression_opts'] = var.compression_opts

            # save source so __repr__ can detect if it's local or not
            encoding['source'] = self._filename
            encoding['original_shape'] = var.shape

            vlen_dtype = h5py.check_dtype(vlen=var.dtype)
            if vlen_dtype is unicode_type:
                encoding['dtype'] = str
            elif vlen_dtype is not None:  # pragma: no cover
                # xarray doesn't support writing arbitrary vlen dtypes yet.
                pass
            else:
                encoding['dtype'] = var.dtype

        return Variable(dimensions, data, attrs, encoding)

    def get_variables(self):
        with self.ensure_open(autoclose=False):
            return FrozenOrderedDict((k, self.open_store_variable(k, v))
                                     for k, v in iteritems(self.ds.variables))

    def get_attrs(self):
        with self.ensure_open(autoclose=True):
            return FrozenOrderedDict(_read_attributes(self.ds))

    def get_dimensions(self):
        with self.ensure_open(autoclose=True):
            return self.ds.dimensions

    def get_encoding(self):
        with self.ensure_open(autoclose=True):
            encoding = {}
            encoding['unlimited_dims'] = {
                k for k, v in self.ds.dimensions.items() if v is None}
        return encoding

    def set_dimension(self, name, length, is_unlimited=False):
        with self.ensure_open(autoclose=False):
            if is_unlimited:
                self.ds.dimensions[name] = None
                self.ds.resize_dimension(name, length)
            else:
                self.ds.dimensions[name] = length

    def set_attribute(self, key, value):
        with self.ensure_open(autoclose=False):
            self.ds.attrs[key] = value

    def encode_variable(self, variable):
        return _encode_nc4_variable(variable)

    def prepare_variable(self, name, variable, check_encoding=False,
                         unlimited_dims=None):
        import h5py

        attrs = variable.attrs.copy()
        dtype = _get_datatype(
            variable, raise_on_invalid_encoding=check_encoding)

        fillvalue = attrs.pop('_FillValue', None)
        if dtype is str and fillvalue is not None:
            raise NotImplementedError(
                'h5netcdf does not yet support setting a fill value for '
                'variable-length strings '
                '(https://github.com/shoyer/h5netcdf/issues/37). '
                "Either remove '_FillValue' from encoding on variable %r "
                "or set {'dtype': 'S1'} in encoding to use the fixed width "
                'NC_CHAR type.' % name)

        if dtype is str:
            dtype = h5py.special_dtype(vlen=unicode_type)

        encoding = _extract_h5nc_encoding(variable,
                                          raise_on_invalid=check_encoding)
        kwargs = {}

        # Convert from NetCDF4-Python style compression settings to h5py style
        # If both styles are used together, h5py takes precedence
        # If set_encoding=True, raise ValueError in case of mismatch
        if encoding.pop('zlib', False):
            if (check_encoding and encoding.get('compression')
                    not in (None, 'gzip')):
                raise ValueError("'zlib' and 'compression' encodings mismatch")
            encoding.setdefault('compression', 'gzip')

        if (check_encoding and
                'complevel' in encoding and 'compression_opts' in encoding and
                encoding['complevel'] != encoding['compression_opts']):
            raise ValueError("'complevel' and 'compression_opts' encodings "
                             "mismatch")
        complevel = encoding.pop('complevel', 0)
        if complevel != 0:
            encoding.setdefault('compression_opts', complevel)

        encoding['chunks'] = encoding.pop('chunksizes', None)

        for key in ['compression', 'compression_opts', 'shuffle',
                    'chunks', 'fletcher32']:
            if key in encoding:
                kwargs[key] = encoding[key]
        if name not in self.ds:
            nc4_var = self.ds.create_variable(
                name, dtype=dtype, dimensions=variable.dims,
                fillvalue=fillvalue, **kwargs)
        else:
            nc4_var = self.ds[name]

        for k, v in iteritems(attrs):
            nc4_var.attrs[k] = v

        target = H5NetCDFArrayWrapper(name, self)

        return target, variable.data

    def sync(self, compute=True):
        if not compute:
            raise NotImplementedError(
                'compute=False is not supported for the h5netcdf backend yet')
        with self.ensure_open(autoclose=True):
            super(H5NetCDFStore, self).sync(compute=compute)
            self.ds.sync()

    def close(self):
        if self._isopen:
            # netCDF4 only allows closing the root group
            ds = find_root(self.ds)
            if not ds._closed:
                ds.close()
            self._isopen = False
