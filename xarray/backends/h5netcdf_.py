import functools

from .. import Variable
from ..conventions import cf_encoder
from ..core import indexing
from ..core.utils import FrozenOrderedDict, close_on_error, Frozen
from ..core.pycompat import iteritems, bytes_type, unicode_type, OrderedDict

from .common import WritableCFDataStore
from .netCDF4_ import _nc4_group, _nc4_values_and_dtype, _extract_nc4_encoding


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


_extract_h5nc_encoding = functools.partial(_extract_nc4_encoding,
                                           lsd_okay=False, backend='h5netcdf')


class H5NetCDFStore(WritableCFDataStore):
    """Store for reading and writing data via h5netcdf
    """
    def __init__(self, filename, mode='r', format=None, group=None,
                 writer=None):
        import h5netcdf.legacyapi
        if format not in [None, 'NETCDF4']:
            raise ValueError('invalid format for h5netcdf backend')
        ds = h5netcdf.legacyapi.Dataset(filename, mode=mode)
        with close_on_error(ds):
            self.ds = _nc4_group(ds, group, mode)
        self.format = format
        self._filename = filename
        super(H5NetCDFStore, self).__init__(writer)

    def open_store_variable(self, var):
        dimensions = var.dimensions
        data = indexing.LazilyIndexedArray(var)
        attrs = _read_attributes(var)

        # netCDF4 specific encoding
        encoding = dict(var.filters())
        chunking = var.chunking()
        encoding['chunksizes'] = chunking if chunking != 'contiguous' else None

        # save source so __repr__ can detect if it's local or not
        encoding['source'] = self._filename
        encoding['original_shape'] = var.shape

        return Variable(dimensions, data, attrs, encoding)

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(v))
                                 for k, v in iteritems(self.ds.variables))

    def get_attrs(self):
        return Frozen(_read_attributes(self.ds))

    def get_dimensions(self):
        return self.ds.dimensions

    def set_dimension(self, name, length):
        self.ds.createDimension(name, size=length)

    def set_attribute(self, key, value):
        self.ds.setncattr(key, value)

    def prepare_variable(self, name, variable, check_encoding=False):
        import h5py

        attrs = variable.attrs.copy()
        variable, dtype = _nc4_values_and_dtype(variable)
        if dtype is str:
            dtype = h5py.special_dtype(vlen=unicode_type)

        self.set_necessary_dimensions(variable)

        fill_value = attrs.pop('_FillValue', None)
        if fill_value in ['\x00']:
            fill_value = None

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
        super(H5NetCDFStore, self).sync()
        self.ds.sync()

    def close(self):
        ds = self.ds
        # netCDF4 only allows closing the root group
        while ds.parent is not None:
            ds = ds.parent
        ds.close()
