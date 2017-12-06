from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

from .. import Variable
from ..core import indexing
from ..core.utils import FrozenOrderedDict, close_on_error
from ..core.pycompat import iteritems, bytes_type, unicode_type, OrderedDict

from .common import WritableCFDataStore, DataStorePickleMixin, find_root
from .netCDF4_ import (_nc4_group, _nc4_values_and_dtype,
                       _extract_nc4_variable_encoding, BaseNetCDF4Array)


class H5NetCDFArrayWrapper(BaseNetCDF4Array):
    def __getitem__(self, key):
        key = indexing.unwrap_explicit_indexer(
            key, self, allow=(indexing.BasicIndexer, indexing.OuterIndexer))
        with self.datastore.ensure_open(autoclose=True):
            return self.get_array()[key]


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
    for k in h5netcdf_var.ncattrs():
        v = h5netcdf_var.getncattr(k)
        if k not in ['_FillValue', 'missing_value']:
            v = maybe_decode_bytes(v)
        attrs[k] = v
    return attrs


_extract_h5nc_encoding = functools.partial(_extract_nc4_variable_encoding,
                                           lsd_okay=False, backend='h5netcdf')


def _open_h5netcdf_group(filename, mode, group):
    import h5netcdf.legacyapi
    ds = h5netcdf.legacyapi.Dataset(filename, mode=mode)
    with close_on_error(ds):
        return _nc4_group(ds, group, mode)


class H5NetCDFStore(WritableCFDataStore, DataStorePickleMixin):
    """Store for reading and writing data via h5netcdf
    """
    def __init__(self, filename, mode='r', format=None, group=None,
                 writer=None, autoclose=False):
        if format not in [None, 'NETCDF4']:
            raise ValueError('invalid format for h5netcdf backend')
        opener = functools.partial(_open_h5netcdf_group, filename, mode=mode,
                                   group=group)
        self.ds = opener()
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
        super(H5NetCDFStore, self).__init__(writer)

    def open_store_variable(self, name, var):
        with self.ensure_open(autoclose=False):
            dimensions = var.dimensions
            data = indexing.LazilyIndexedArray(
                H5NetCDFArrayWrapper(name, self))
            attrs = _read_attributes(var)

            # netCDF4 specific encoding
            encoding = dict(var.filters())
            chunking = var.chunking()
            encoding['chunksizes'] = chunking \
                if chunking != 'contiguous' else None

            # save source so __repr__ can detect if it's local or not
            encoding['source'] = self._filename
            encoding['original_shape'] = var.shape

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
                self.ds.createDimension(name, size=None)
                self.ds.resize_dimension(name, length)
            else:
                self.ds.createDimension(name, size=length)

    def set_attribute(self, key, value):
        with self.ensure_open(autoclose=False):
            self.ds.setncattr(key, value)

    def prepare_variable(self, name, variable, check_encoding=False,
                         unlimited_dims=None):
        import h5py

        attrs = variable.attrs.copy()
        variable, dtype = _nc4_values_and_dtype(variable)

        self.set_necessary_dimensions(variable, unlimited_dims=unlimited_dims)

        fill_value = attrs.pop('_FillValue', None)
        if dtype is str and fill_value is not None:
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

        for key in ['zlib', 'complevel', 'shuffle',
                    'chunksizes', 'fletcher32']:
            if key in encoding:
                kwargs[key] = encoding[key]
        nc4_var = self.ds.createVariable(name, dtype, variable.dims,
                                         fill_value=fill_value, **kwargs)

        for k, v in iteritems(attrs):
            nc4_var.setncattr(k, v)
        return nc4_var, variable.data

    def sync(self):
        with self.ensure_open(autoclose=True):
            super(H5NetCDFStore, self).sync()
            self.ds.sync()

    def close(self):
        if self._isopen:
            # netCDF4 only allows closing the root group
            ds = find_root(self.ds)
            if not ds._closed:
                ds.close()
            self._isopen = False
