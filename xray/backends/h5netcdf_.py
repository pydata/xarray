# TODO: handle encoding of netCDF4 specific options
from .. import Variable
from ..conventions import cf_encoder
from ..core import indexing
from ..core.utils import FrozenOrderedDict, close_on_error
from ..core.pycompat import iteritems, basestring, unicode_type, OrderedDict

from .common import AbstractWritableDataStore
from .netCDF4_ import _nc4_group, _nc4_values_and_dtype


class H5NetCDFStore(AbstractWritableDataStore):
    """Store for reading and writing data via h5netcdf
    """
    def __init__(self, filename, mode='r', group=None):
        import h5netcdf.legacyapi
        ds = h5netcdf.legacyapi.Dataset(filename, mode=mode)
        with close_on_error(ds):
            self.ds = _nc4_group(ds, group, mode)
        self.format = format
        self._filename = filename

    def store(self, variables, attributes):
        # All NetCDF files get CF encoded by default, without this attempting
        # to write times, for example, would fail.
        cf_variables, cf_attrs = cf_encoder(variables, attributes)
        AbstractWritableDataStore.store(self, cf_variables, cf_attrs)

    def open_store_variable(self, var):
        dimensions = var.dimensions
        data = indexing.LazilyIndexedArray(var)
        attributes = OrderedDict((k, var.getncattr(k))
                                 for k in var.ncattrs())

        # netCDF4 specific encoding
        encoding = dict(var.filters())
        chunking = var.chunking()
        encoding['chunksizes'] = chunking if chunking != 'contiguous' else None

        # save source so __repr__ can detect if it's local or not
        encoding['source'] = self._filename

        return Variable(dimensions, data, attributes, encoding)

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(v))
                                 for k, v in iteritems(self.ds.variables))

    def get_attrs(self):
        return FrozenOrderedDict((k, self.ds.getncattr(k))
                                 for k in self.ds.ncattrs())

    def get_dimensions(self):
        return self.ds.dimensions

    def set_dimension(self, name, length):
        self.ds.createDimension(name, size=length)

    def set_attribute(self, key, value):
        self.ds.setncattr(key, value)

    def prepare_variable(self, name, variable):
        import h5py

        attrs = variable.attrs.copy()
        variable, dtype = _nc4_values_and_dtype(variable)
        if dtype is str:
            dtype = h5py.special_dtype(vlen=unicode_type)

        self.set_necessary_dimensions(variable)

        fill_value = attrs.pop('_FillValue', None)
        if fill_value in ['\x00']:
            fill_value = None

        encoding = variable.encoding
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
        self.ds.sync()

    def close(self):
        ds = self.ds
        # netCDF4 only allows closing the root group
        while ds.parent is not None:
            ds = ds.parent
        ds.close()
