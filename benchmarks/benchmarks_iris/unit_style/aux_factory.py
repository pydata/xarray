# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Small-scope AuxFactory benchmark tests."""

import numpy as np

from iris import aux_factory, coords


class FactoryCommon:
    # TODO: once https://github.com/airspeed-velocity/asv/pull/828 is released:
    #       * make class an ABC
    #       * remove NotImplementedError
    #       * combine setup_common into setup
    """Run a generalised suite of benchmarks for any factory.

    A base class running a generalised suite of benchmarks for any factory.
    Factory to be specified in a subclass.

    ASV will run the benchmarks within this class for any subclasses.

    Should only be instantiated within subclasses, but cannot enforce this
    since ASV cannot handle classes that include abstract methods.
    """

    def setup(self):
        """Prevent ASV instantiating (must therefore override setup() in any subclasses.)."""
        raise NotImplementedError

    def setup_common(self):
        """Shared setup code that can be called by subclasses."""
        self.factory = self.create()

    def time_create(self):
        """Create an instance of the benchmarked factory.

        Create method is specified in the subclass.
        """
        self.create()


class HybridHeightFactory(FactoryCommon):
    def setup(self):
        data_1d = np.zeros(1000)
        self.coord = coords.AuxCoord(points=data_1d, units="m")

        self.setup_common()

    def create(self):
        return aux_factory.HybridHeightFactory(delta=self.coord)
