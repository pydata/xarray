# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Import iris benchmarking."""

from importlib import import_module, reload

################
# Prepare info for reset_colormaps:

# Import and capture colormaps.
from matplotlib import colormaps  # isort:skip

_COLORMAPS_ORIG = set(colormaps)

# Import iris.palette, which modifies colormaps.
import iris.palette

# Derive which colormaps have been added by iris.palette.
_COLORMAPS_MOD = set(colormaps)
COLORMAPS_EXTRA = _COLORMAPS_MOD - _COLORMAPS_ORIG

# Touch iris.palette to prevent linters complaining.
_ = iris.palette

################


class Iris:
    @staticmethod
    def _import(module_name, reset_colormaps=False):
        """Have experimented with adding sleep() commands into the imported modules.

        The results reveal:

        ASV avoids invoking `import x` if nothing gets called in the
        benchmark (some imports were timed, but only those where calls
        happened during import).

        Using reload() is not identical to importing, but does produce
        results that are very close to expected import times, so this is fine
        for monitoring for regressions.
        It is also ideal for accurate repetitions, without the need to mess
        with the ASV `number` attribute etc, since cached imports are not used
        and the repetitions are therefore no faster than the first run.
        """
        mod = import_module(module_name)

        if reset_colormaps:
            # Needed because reload() will attempt to register new colormaps a
            #  second time, which errors by default.
            for cm_name in COLORMAPS_EXTRA:
                colormaps.unregister(cm_name)

        reload(mod)

    def time_iris(self):
        self._import("iris")

    def time__concatenate(self):
        self._import("iris._concatenate")

    def time__constraints(self):
        self._import("iris._constraints")

    def time__data_manager(self):
        self._import("iris._data_manager")

    def time__deprecation(self):
        self._import("iris._deprecation")

    def time__lazy_data(self):
        self._import("iris._lazy_data")

    def time__merge(self):
        self._import("iris._merge")

    def time__representation(self):
        self._import("iris._representation")

    def time_analysis(self):
        self._import("iris.analysis")

    def time_analysis__area_weighted(self):
        self._import("iris.analysis._area_weighted")

    def time_analysis__grid_angles(self):
        self._import("iris.analysis._grid_angles")

    def time_analysis__interpolation(self):
        self._import("iris.analysis._interpolation")

    def time_analysis__regrid(self):
        self._import("iris.analysis._regrid")

    def time_analysis__scipy_interpolate(self):
        self._import("iris.analysis._scipy_interpolate")

    def time_analysis_calculus(self):
        self._import("iris.analysis.calculus")

    def time_analysis_cartography(self):
        self._import("iris.analysis.cartography")

    def time_analysis_geomerty(self):
        self._import("iris.analysis.geometry")

    def time_analysis_maths(self):
        self._import("iris.analysis.maths")

    def time_analysis_stats(self):
        self._import("iris.analysis.stats")

    def time_analysis_trajectory(self):
        self._import("iris.analysis.trajectory")

    def time_aux_factory(self):
        self._import("iris.aux_factory")

    def time_common(self):
        self._import("iris.common")

    def time_common_lenient(self):
        self._import("iris.common.lenient")

    def time_common_metadata(self):
        self._import("iris.common.metadata")

    def time_common_mixin(self):
        self._import("iris.common.mixin")

    def time_common_resolve(self):
        self._import("iris.common.resolve")

    def time_config(self):
        self._import("iris.config")

    def time_coord_categorisation(self):
        self._import("iris.coord_categorisation")

    def time_coord_systems(self):
        self._import("iris.coord_systems")

    def time_coords(self):
        self._import("iris.coords")

    def time_cube(self):
        self._import("iris.cube")

    def time_exceptions(self):
        self._import("iris.exceptions")

    def time_experimental(self):
        self._import("iris.experimental")

    def time_fileformats(self):
        self._import("iris.fileformats")

    def time_fileformats__ff(self):
        self._import("iris.fileformats._ff")

    def time_fileformats__ff_cross_references(self):
        self._import("iris.fileformats._ff_cross_references")

    def time_fileformats__pp_lbproc_pairs(self):
        self._import("iris.fileformats._pp_lbproc_pairs")

    def time_fileformats_structured_array_identification(self):
        self._import("iris.fileformats._structured_array_identification")

    def time_fileformats_abf(self):
        self._import("iris.fileformats.abf")

    def time_fileformats_cf(self):
        self._import("iris.fileformats.cf")

    def time_fileformats_dot(self):
        self._import("iris.fileformats.dot")

    def time_fileformats_name(self):
        self._import("iris.fileformats.name")

    def time_fileformats_name_loaders(self):
        self._import("iris.fileformats.name_loaders")

    def time_fileformats_netcdf(self):
        self._import("iris.fileformats.netcdf")

    def time_fileformats_nimrod(self):
        self._import("iris.fileformats.nimrod")

    def time_fileformats_nimrod_load_rules(self):
        self._import("iris.fileformats.nimrod_load_rules")

    def time_fileformats_pp(self):
        self._import("iris.fileformats.pp")

    def time_fileformats_pp_load_rules(self):
        self._import("iris.fileformats.pp_load_rules")

    def time_fileformats_pp_save_rules(self):
        self._import("iris.fileformats.pp_save_rules")

    def time_fileformats_rules(self):
        self._import("iris.fileformats.rules")

    def time_fileformats_um(self):
        self._import("iris.fileformats.um")

    def time_fileformats_um__fast_load(self):
        self._import("iris.fileformats.um._fast_load")

    def time_fileformats_um__fast_load_structured_fields(self):
        self._import("iris.fileformats.um._fast_load_structured_fields")

    def time_fileformats_um__ff_replacement(self):
        self._import("iris.fileformats.um._ff_replacement")

    def time_fileformats_um__optimal_array_structuring(self):
        self._import("iris.fileformats.um._optimal_array_structuring")

    def time_fileformats_um_cf_map(self):
        self._import("iris.fileformats.um_cf_map")

    def time_io(self):
        self._import("iris.io")

    def time_io_format_picker(self):
        self._import("iris.io.format_picker")

    def time_iterate(self):
        self._import("iris.iterate")

    def time_palette(self):
        self._import("iris.palette", reset_colormaps=True)

    def time_plot(self):
        self._import("iris.plot")

    def time_quickplot(self):
        self._import("iris.quickplot")

    def time_std_names(self):
        self._import("iris.std_names")

    def time_symbols(self):
        self._import("iris.symbols")

    def time_tests(self):
        self._import("iris.tests")

    def time_time(self):
        self._import("iris.time")

    def time_util(self):
        self._import("iris.util")

    # third-party imports

    def time_third_party_cartopy(self):
        self._import("cartopy")

    def time_third_party_cf_units(self):
        self._import("cf_units")

    def time_third_party_cftime(self):
        self._import("cftime")

    def time_third_party_matplotlib(self):
        self._import("matplotlib")

    def time_third_party_numpy(self):
        self._import("numpy")

    def time_third_party_scipy(self):
        self._import("scipy")
