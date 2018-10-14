#
# Copyright 2017-2018 European Centre for Medium-Range Weather Forecasts.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors:
#   Alessandro Amici - B-Open - https://bopen.eu
#

from __future__ import absolute_import, division, print_function

import numpy as np

from .. import Variable
from ..core import indexing
from ..core.utils import Frozen, FrozenOrderedDict
from .common import AbstractDataStore, BackendArray
from .locks import ensure_lock, SerializableLock

# FIXME: Add a dedicated lock, even if ecCodes is supposed to be thread-safe
# in most circumstances. See:
#   https://confluence.ecmwf.int/display/ECC/Frequently+Asked+Questions
ECCODES_LOCK = SerializableLock()


class CfGribArrayWrapper(BackendArray):
    def __init__(self, datastore, array):
        self.datastore = datastore
        self.shape = array.shape
        self.dtype = array.dtype
        self.array = array

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem)

    def _getitem(self, key):
        with self.datastore.lock:
            return self.array[key]


class CfGribDataStore(AbstractDataStore):
    """
    Implements the ``xr.AbstractDataStore`` read-only API for a GRIB file.
    """
    def __init__(self, filename, lock=None, **backend_kwargs):
        import cfgrib
        if lock is None:
            lock = ECCODES_LOCK
        self.lock = ensure_lock(lock)

        # NOTE: filter_by_keys is a dict, but CachingFileManager only accepts
        #   hashable types.
        if 'filter_by_keys' in backend_kwargs:
            filter_by_keys_items = backend_kwargs['filter_by_keys'].items()
            backend_kwargs['filter_by_keys'] = tuple(filter_by_keys_items)

        self.ds = cfgrib.open_file(filename, mode='r', **backend_kwargs)

    def open_store_variable(self, name, var):
        if isinstance(var.data, np.ndarray):
            data = var.data
        else:
            wrapped_array = CfGribArrayWrapper(self, var.data)
            data = indexing.LazilyOuterIndexedArray(wrapped_array)

        encoding = self.ds.encoding.copy()
        encoding['original_shape'] = var.data.shape

        return Variable(var.dimensions, data, var.attributes, encoding)

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(k, v))
                                 for k, v in self.ds.variables.items())

    def get_attrs(self):
        return Frozen(self.ds.attributes)

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)

    def get_encoding(self):
        dims = self.get_dimensions()
        encoding = {
            'unlimited_dims': {k for k, v in dims.items() if v is None},
        }
        return encoding
