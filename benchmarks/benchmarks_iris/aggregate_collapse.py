# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Benchmarks relating to :meth:`iris.cube.CubeList.merge` and ``concatenate``."""

import warnings

import numpy as np

from iris import analysis, coords, cube
from iris.warnings import IrisVagueMetadataWarning

from .generate_data.stock import realistic_4d_w_everything


class AggregationMixin:
    params = [[False, True]]
    param_names = ["Lazy operations"]

    def setup(self, lazy_run: bool):
        warnings.filterwarnings("ignore", message="Ignoring a datum")
        warnings.filterwarnings("ignore", category=IrisVagueMetadataWarning)
        cube = realistic_4d_w_everything(lazy=lazy_run)

        for cm in cube.cell_measures():
            cube.remove_cell_measure(cm)
        for av in cube.ancillary_variables():
            cube.remove_ancillary_variable(av)

        agg_mln_data = np.arange(0, 70, 10)
        agg_mln_repeat = np.repeat(agg_mln_data, 10)

        cube = cube[..., :10, :10]

        self.mln_aux = "aggregatable"
        self.mln = "model_level_number"
        agg_mln_coord = coords.AuxCoord(points=agg_mln_repeat, long_name=self.mln_aux)

        if lazy_run:
            agg_mln_coord.points = agg_mln_coord.lazy_points()
        cube.add_aux_coord(agg_mln_coord, 1)
        self.cube = cube


class Aggregation(AggregationMixin):
    def time_aggregated_by_MEAN(self, _):
        _ = self.cube.aggregated_by(self.mln_aux, analysis.MEAN).data

    def time_aggregated_by_COUNT(self, _):
        _ = self.cube.aggregated_by(
            self.mln_aux, analysis.COUNT, function=lambda values: values > 280
        ).data

    def time_aggregated_by_GMEAN(self, _):
        _ = self.cube.aggregated_by(self.mln_aux, analysis.GMEAN).data

    def time_aggregated_by_HMEAN(self, _):
        _ = self.cube.aggregated_by(self.mln_aux, analysis.HMEAN).data

    def time_aggregated_by_MAX_RUN(self, _):
        _ = self.cube.aggregated_by(
            self.mln_aux, analysis.MAX_RUN, function=lambda values: values > 280
        ).data

    def time_aggregated_by_MAX(self, _):
        _ = self.cube.aggregated_by(self.mln_aux, analysis.MAX).data

    def time_aggregated_by_MEDIAN(self, _):
        _ = self.cube.aggregated_by(self.mln_aux, analysis.MEDIAN).data

    def time_aggregated_by_MIN(self, _):
        _ = self.cube.aggregated_by(self.mln_aux, analysis.MIN).data

    def time_aggregated_by_PEAK(self, _):
        _ = self.cube.aggregated_by(self.mln_aux, analysis.PEAK).data

    def time_aggregated_by_PERCENTILE(self, _):
        _ = self.cube.aggregated_by(
            self.mln_aux, analysis.PERCENTILE, percent=[10, 50, 90]
        ).data

    def time_aggregated_by_FAST_PERCENTILE(self, _):
        _ = self.cube.aggregated_by(
            self.mln_aux,
            analysis.PERCENTILE,
            mdtol=0,
            percent=[10, 50, 90],
            fast_percentile_method=True,
        ).data

    def time_aggregated_by_PROPORTION(self, _):
        _ = self.cube.aggregated_by(
            self.mln_aux,
            analysis.PROPORTION,
            function=lambda values: values > 280,
        ).data

    def time_aggregated_by_STD_DEV(self, _):
        _ = self.cube.aggregated_by(self.mln_aux, analysis.STD_DEV).data

    def time_aggregated_by_VARIANCE(self, _):
        _ = self.cube.aggregated_by(self.mln_aux, analysis.VARIANCE).data

    def time_aggregated_by_RMS(self, _):
        _ = self.cube.aggregated_by(self.mln_aux, analysis.RMS).data

    def time_collapsed_by_MEAN(self, _):
        _ = self.cube.collapsed(self.mln, analysis.MEAN).data

    def time_collapsed_by_COUNT(self, _):
        _ = self.cube.collapsed(
            self.mln, analysis.COUNT, function=lambda values: values > 280
        ).data

    def time_collapsed_by_GMEAN(self, _):
        _ = self.cube.collapsed(self.mln, analysis.GMEAN).data

    def time_collapsed_by_HMEAN(self, _):
        _ = self.cube.collapsed(self.mln, analysis.HMEAN).data

    def time_collapsed_by_MAX_RUN(self, _):
        _ = self.cube.collapsed(
            self.mln, analysis.MAX_RUN, function=lambda values: values > 280
        ).data

    def time_collapsed_by_MAX(self, _):
        _ = self.cube.collapsed(self.mln, analysis.MAX).data

    def time_collapsed_by_MEDIAN(self, _):
        _ = self.cube.collapsed(self.mln, analysis.MEDIAN).data

    def time_collapsed_by_MIN(self, _):
        _ = self.cube.collapsed(self.mln, analysis.MIN).data

    def time_collapsed_by_PEAK(self, _):
        _ = self.cube.collapsed(self.mln, analysis.PEAK).data

    def time_collapsed_by_PERCENTILE(self, _):
        _ = self.cube.collapsed(
            self.mln, analysis.PERCENTILE, percent=[10, 50, 90]
        ).data

    def time_collapsed_by_FAST_PERCENTILE(self, _):
        _ = self.cube.collapsed(
            self.mln,
            analysis.PERCENTILE,
            mdtol=0,
            percent=[10, 50, 90],
            fast_percentile_method=True,
        ).data

    def time_collapsed_by_PROPORTION(self, _):
        _ = self.cube.collapsed(
            self.mln, analysis.PROPORTION, function=lambda values: values > 280
        ).data

    def time_collapsed_by_STD_DEV(self, _):
        _ = self.cube.collapsed(self.mln, analysis.STD_DEV).data

    def time_collapsed_by_VARIANCE(self, _):
        _ = self.cube.collapsed(self.mln, analysis.VARIANCE).data

    def time_collapsed_by_RMS(self, _):
        _ = self.cube.collapsed(self.mln, analysis.RMS).data


class WeightedAggregation(AggregationMixin):
    def setup(self, lazy_run):
        super().setup(lazy_run)

        weights = np.linspace(0, 1, 70)
        weights = np.broadcast_to(weights, self.cube.shape[:2])
        weights = np.broadcast_to(weights.T, self.cube.shape[::-1])
        weights = weights.T

        self.weights = weights

    ## currently has problems with indexing weights
    # def time_w_aggregated_by_WPERCENTILE(self, _):
    #     _ = self.cube.aggregated_by(
    #         self.mln_aux, analysis.WPERCENTILE, weights=self.weights, percent=[10, 50, 90]
    #     ).data

    def time_w_aggregated_by_SUM(self, _):
        _ = self.cube.aggregated_by(
            self.mln_aux, analysis.SUM, weights=self.weights
        ).data

    def time_w_aggregated_by_RMS(self, _):
        _ = self.cube.aggregated_by(
            self.mln_aux, analysis.RMS, weights=self.weights
        ).data

    def time_w_aggregated_by_MEAN(self, _):
        _ = self.cube.aggregated_by(
            self.mln_aux, analysis.MEAN, weights=self.weights
        ).data

    def time_w_collapsed_by_WPERCENTILE(self, _):
        _ = self.cube.collapsed(
            self.mln, analysis.WPERCENTILE, weights=self.weights, percent=[10, 50, 90]
        ).data

    def time_w_collapsed_by_SUM(self, _):
        _ = self.cube.collapsed(self.mln, analysis.SUM, weights=self.weights).data

    def time_w_collapsed_by_RMS(self, _):
        _ = self.cube.collapsed(self.mln, analysis.RMS, weights=self.weights).data

    def time_w_collapsed_by_MEAN(self, _):
        _ = self.cube.collapsed(self.mln, analysis.MEAN, weights=self.weights).data
