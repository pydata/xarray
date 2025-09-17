# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Common code for benchmarks."""

from os import environ

import iris

from . import generate_data
from .generate_data.um_files import create_um_files


def disable_repeat_between_setup(benchmark_object):
    """Benchmark where object persistence would be inappropriate (decorator).

    E.g:

    * Benchmarking data realisation
    * Benchmarking Cube coord addition

    Can be applied to benchmark classes/methods/functions.

    https://asv.readthedocs.io/en/stable/benchmarks.html#timing-benchmarks

    """
    # Prevent repeat runs between setup() runs - object(s) will persist after 1st.
    benchmark_object.number = 1
    # Compensate for reduced certainty by increasing number of repeats.
    #  (setup() is run between each repeat).
    #  Minimum 5 repeats, run up to 30 repeats / 20 secs whichever comes first.
    benchmark_object.repeat = (5, 30, 20.0)
    # ASV uses warmup to estimate benchmark time before planning the real run.
    #  Prevent this, since object(s) will persist after first warmup run,
    #  which would give ASV misleading info (warmups ignore ``number``).
    benchmark_object.warmup_time = 0.0

    return benchmark_object


def on_demand_benchmark(benchmark_object):
    """Disable these benchmark(s) unless ON_DEMAND_BENCHARKS env var is set.

    This is a decorator.

    For benchmarks that, for whatever reason, should not be run by default.
    E.g:

    * Require a local file
    * Used for scalability analysis instead of commit monitoring.

    Can be applied to benchmark classes/methods/functions.

    """
    if "ON_DEMAND_BENCHMARKS" in environ:
        return benchmark_object


@on_demand_benchmark
class ValidateSetup:
    """Simple benchmarks that exercise all elements of our setup."""

    params = [1, 2]

    def setup(self, param):
        generate_data.REUSE_DATA = False
        (self.file_path,) = create_um_files(
            param, param, param, param, False, ["NetCDF"]
        ).values()

    def time_validate(self, param):
        _ = iris.load(self.file_path)

    def tracemalloc_validate(self, param):
        _ = iris.load(self.file_path)
