#
# Copyright 2017-2018 European Centre for Medium-Range Weather Forecasts (ECMWF).
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

from . import common
from .. import core


class WrapGrib(common.BackendArray):
    def __init__(self, backend_array):
        self.backend_array = backend_array

    def __getattr__(self, item):
        return getattr(self.backend_array, item)

    def __getitem__(self, item):
        key, np_inds = core.indexing.decompose_indexer(
            item, self.shape, core.indexing.IndexingSupport.OUTER_1VECTOR)

        array = self.backend_array[key.tuple]

        if len(np_inds.tuple) > 0:
            array = core.indexing.NumpyIndexingAdapter(array)[np_inds]

        return array


FLAVOURS = {
    'eccodes': {
        'dataset': {
            'encode_time': False,
            'encode_vertical': False,
            'encode_geography': False,
        },
    },
    'ecmwf': {
        'variable_map': {},
    },
    'cds': {
        'variable_map': {
            'number': 'realization',
            'time': 'forecast_reference_time',
            'valid_time': 'time',
            'step': 'leadtime',
            'air_pressure': 'plev',
            'latitude': 'lat',
            'longitude': 'lon',
        },
    },
}


class CfGribDataStore(common.AbstractDataStore):
    """
    Implements the ``xr.AbstractDataStore`` read-only API for a GRIB file.
    """
    def __init__(self, ds, variable_map={}, lock=None):
        self.ds = ds
        self.variable_map = variable_map.copy()
        self.lock = lock

    @classmethod
    def from_path(cls, path, lock=None, flavour_name='ecmwf', errors='ignore', **kwargs):
        import cfgrib
        flavour = FLAVOURS[flavour_name].copy()
        config = flavour.pop('dataset', {}).copy()
        config.update(kwargs)
        ds = cfgrib.open_file(path, errors=errors, **config)
        return cls(ds=ds, lock=lock, **flavour)

    def open_store_variable(self, name, var):
        if isinstance(var.data, np.ndarray):
            data = var.data
        else:
            data = core.indexing.LazilyOuterIndexedArray(WrapGrib(var.data))

        dimensions = tuple(self.variable_map.get(dim, dim) for dim in var.dimensions)
        attrs = var.attributes

        # the coordinates attributes need a special treatment
        if 'coordinates' in attrs:
            coordinates = [self.variable_map.get(d, d) for d in attrs['coordinates'].split()]
            attrs['coordinates'] = ' '.join(coordinates)

        encoding = self.ds.encoding.copy()
        encoding['original_shape'] = var.data.shape

        return core.variable.Variable(dimensions, data, attrs, encoding)

    def get_variables(self):
        variables = []
        for k, v in self.ds.variables.items():
            variables.append((self.variable_map.get(k, k), self.open_store_variable(k, v)))
        return core.utils.FrozenOrderedDict(variables)

    def get_attrs(self):
        return core.utils.FrozenOrderedDict(self.ds.attributes)

    def get_dimensions(self):
        return core.utils.FrozenOrderedDict((self.variable_map.get(d, d), s)
                                            for d, s in self.ds.dimensions.items())

    def get_encoding(self):
        encoding = {}
        encoding['unlimited_dims'] = {k for k, v in self.ds.dimensions.items() if v is None}
        return encoding
