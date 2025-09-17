# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Small-scope CFVariableMixin benchmark tests."""

import numpy as np

from iris import coords
from iris.common.metadata import AncillaryVariableMetadata

LONG_NAME = "air temperature"
STANDARD_NAME = "air_temperature"
VAR_NAME = "air_temp"
UNITS = "degrees"
ATTRIBUTES = dict(a=1)
DICT = dict(
    standard_name=STANDARD_NAME,
    long_name=LONG_NAME,
    var_name=VAR_NAME,
    units=UNITS,
    attributes=ATTRIBUTES,
)
METADATA = AncillaryVariableMetadata(**DICT)
TUPLE = tuple(DICT.values())


class CFVariableMixin:
    def setup(self):
        data_1d = np.zeros(1000)

        # These benchmarks are from a user perspective, so using a user-level
        # subclass of CFVariableMixin to test behaviour. AncillaryVariable is
        # the simplest so using that.
        self.cfm_proxy = coords.AncillaryVariable(data_1d)
        self.cfm_proxy.long_name = "test"

    def time_get_long_name(self):
        self.cfm_proxy.long_name

    def time_set_long_name(self):
        self.cfm_proxy.long_name = LONG_NAME

    def time_get_standard_name(self):
        self.cfm_proxy.standard_name

    def time_set_standard_name(self):
        self.cfm_proxy.standard_name = STANDARD_NAME

    def time_get_var_name(self):
        self.cfm_proxy.var_name

    def time_set_var_name(self):
        self.cfm_proxy.var_name = VAR_NAME

    def time_get_units(self):
        self.cfm_proxy.units

    def time_set_units(self):
        self.cfm_proxy.units = UNITS

    def time_get_attributes(self):
        self.cfm_proxy.attributes

    def time_set_attributes(self):
        self.cfm_proxy.attributes = ATTRIBUTES

    def time_get_metadata(self):
        self.cfm_proxy.metadata

    def time_set_metadata__dict(self):
        self.cfm_proxy.metadata = DICT

    def time_set_metadata__tuple(self):
        self.cfm_proxy.metadata = TUPLE

    def time_set_metadata__metadata(self):
        self.cfm_proxy.metadata = METADATA
