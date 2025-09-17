# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Small-scope Coord benchmark tests."""

import numpy as np

from iris import coords

from .. import disable_repeat_between_setup


def setup():
    """General variables needed by multiple benchmark classes."""
    global data_1d

    data_1d = np.zeros(1000)


class CoordCommon:
    # TODO: once https://github.com/airspeed-velocity/asv/pull/828 is released:
    #       * make class an ABC
    #       * remove NotImplementedError
    #       * combine setup_common into setup
    """Run a generalised suite of benchmarks for any coord.

    A base class running a generalised suite of benchmarks for any coord.
    Coord to be specified in a subclass.

    ASV will run the benchmarks within this class for any subclasses.

    Should only be instantiated within subclasses, but cannot enforce this
    since ASV cannot handle classes that include abstract methods.
    """

    def setup(self):
        """Prevent ASV instantiating (must therefore override setup() in any subclasses.)."""
        raise NotImplementedError

    def setup_common(self):
        """Shared setup code that can be called by subclasses."""
        self.component = self.create()

    def time_create(self):
        """Create an instance of the benchmarked factory.

        Create method is specified in the subclass.
        """
        self.create()


class DimCoord(CoordCommon):
    def setup(self):
        point_values = np.arange(1000)
        bounds = np.array([point_values - 1, point_values + 1]).transpose()

        self.create_kwargs = {
            "points": point_values,
            "bounds": bounds,
            "units": "days since 1970-01-01",
            "climatological": True,
        }

        self.setup_common()

    def create(self):
        return coords.DimCoord(**self.create_kwargs)

    def time_regular(self):
        coords.DimCoord.from_regular(0, 1, 1000)


class AuxCoord(CoordCommon):
    def setup(self):
        bounds = np.array([data_1d - 1, data_1d + 1]).transpose()

        self.create_kwargs = {
            "points": data_1d,
            "bounds": bounds,
            "units": "days since 1970-01-01",
            "climatological": True,
        }

        self.setup_common()

    def create(self):
        return coords.AuxCoord(**self.create_kwargs)

    def time_points(self):
        _ = self.component.points

    def time_bounds(self):
        _ = self.component.bounds


@disable_repeat_between_setup
class AuxCoordLazy(AuxCoord):
    """Lazy equivalent of :class:`AuxCoord`."""

    def setup(self):
        super().setup()
        self.create_kwargs["points"] = self.component.lazy_points()
        self.create_kwargs["bounds"] = self.component.lazy_bounds()
        self.setup_common()


class CellMeasure(CoordCommon):
    def setup(self):
        self.setup_common()

    def create(self):
        return coords.CellMeasure(data_1d)


class CellMethod(CoordCommon):
    def setup(self):
        self.setup_common()

    def create(self):
        return coords.CellMethod("test")


class AncillaryVariable(CoordCommon):
    def setup(self):
        self.setup_common()

    def create(self):
        return coords.AncillaryVariable(data_1d)
