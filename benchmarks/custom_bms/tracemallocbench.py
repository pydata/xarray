# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Benchmark for growth in process resident memory, repeating for accuracy.

Uses a modified version of the repeat logic in
:class:`asv_runner.benchmarks.time.TimeBenchmark`.
"""

import re
from timeit import Timer
import tracemalloc
from typing import Callable

from asv_runner.benchmarks.time import TimeBenchmark, wall_timer


class TracemallocBenchmark(TimeBenchmark):
    """Benchmark for growth in process resident memory, repeating for accuracy.

    Obviously limited as to what it actually measures : Relies on the current
    process not having significant unused (de-allocated) memory when the
    tested codeblock runs, and only reliable when the code allocates a
    significant amount of new memory.

    Benchmark operations prefixed with ``tracemalloc_`` or ``Tracemalloc`` will
    use this benchmark class.

    Inherits behaviour from :class:`asv_runner.benchmarks.time.TimeBenchmark`,
    with modifications for memory measurement. See the below Attributes section
    and https://asv.readthedocs.io/en/stable/writing_benchmarks.html#timing-benchmarks.

    Attributes
    ----------
    Mostly identical to :class:`asv_runner.benchmarks.time.TimeBenchmark`. See
    https://asv.readthedocs.io/en/stable/benchmarks.html#timing-benchmarks
    Make sure to use the inherited ``repeat`` attribute if greater accuracy
    is needed. Below are the attributes where inherited behaviour is
    overridden.

    number : int
        The number of times the benchmarked operation will be called per
        ``repeat``. Memory growth is measured after ALL calls -
        i.e. `number` should make no difference to the result if the operation
        has perfect garbage collection. The parent class's intelligent
        modification of `number` is NOT inherited. A minimum value of ``1`` is
        enforced.
    warmup_time, sample_time, min_run_count, timer
        Not used.
    type : str = "tracemalloc"
        The name of this benchmark type.
    unit : str = "bytes"
        The units of the measured metric (i.e. the growth in memory).

    """

    name_regex = re.compile("^(Tracemalloc[A-Z_].+)|(tracemalloc_.+)$")

    param: tuple

    def __init__(self, name: str, func: Callable, attr_sources: list) -> None:
        """Initialize a new instance of `TracemallocBenchmark`.

        Parameters
        ----------
        name : str
            The name of the benchmark.
        func : callable
            The function to benchmark.
        attr_sources : list
            A list of objects from which to draw attributes.
        """
        super().__init__(name, func, attr_sources)
        self.type = "tracemalloc"
        self.unit = "bytes"

    def _load_vars(self):
        """Load benchmark variables from attribute sources.

        Downstream handling of ``number`` is not the same as in the parent, so
        need to make sure it is at least 1.
        """
        super()._load_vars()
        self.number = max(1, self.number)

    def run(self, *param: tuple) -> dict:
        """Run the benchmark with the given parameters.

        Downstream handling of ``param`` is not the same as in the parent, so
        need to store it now.

        Parameters
        ----------
        *param : tuple
            The parameters to pass to the benchmark function.

        Returns
        -------
        dict
            A dictionary with the benchmark results. It contains the samples
            taken, and "the number of times the function was called in each
            sample" - for this benchmark that is always ``1`` to avoid the
            parent class incorrectly modifying the results.
        """
        self.param = param
        return super().run(*param)

    def benchmark_timing(
        self,
        timer: Timer,
        min_repeat: int,
        max_repeat: int,
        max_time: float,
        warmup_time: float,
        number: int,
        min_run_count: int,
    ) -> tuple[list[int], int]:
        """Benchmark the timing of the function execution.

        Heavily modified from the parent method
        - Directly performs setup and measurement (parent used timeit).
        - `number` used differently (see Parameters).
        - No warmup phase.

        Parameters
        ----------
        timer : timeit.Timer
            Not used.
        min_repeat : int
            The minimum number of times to repeat the function execution.
        max_repeat : int
            The maximum number of times to repeat the function execution.
        max_time : float
            The maximum total time to spend on the benchmarking.
        warmup_time : float
            Not used.
        number : int
            The number of times the benchmarked operation will be called per
            repeat. Memory growth is measured after ALL calls - i.e. `number`
            should make no difference to the result if the operation
            has perfect garbage collection. The parent class's intelligent
            modification of `number` is NOT inherited.
        min_run_count : int
            Not used.

        Returns
        -------
        list
            A list of the measured memory growths, in bytes.
        int = 1
            Part of the inherited return signature. Must be 1 to avoid
            the parent incorrectly modifying the results.
        """
        start_time = wall_timer()
        samples: list[int] = []

        def too_slow(num_samples) -> bool:
            """Stop taking samples if limits exceeded.

            Parameters
            ----------
            num_samples : int
                The number of samples taken so far.

            Returns
            -------
            bool
                True if the benchmark should stop, False otherwise.
            """
            if num_samples < min_repeat:
                return False
            return wall_timer() > start_time + max_time

        # Collect samples
        while len(samples) < max_repeat:
            self.redo_setup()
            tracemalloc.start()
            for _ in range(number):
                __ = self.func(*self.param)
            _, peak_mem_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            samples.append(peak_mem_bytes)

            if too_slow(len(samples)):
                break

        # ``number`` is not used in the same way as in the parent class. Must
        #  be returned as 1 to avoid parent incorrectly modifying the results.
        return samples, 1


# https://asv.readthedocs.io/projects/asv-runner/en/latest/development/benchmark_plugins.html
export_as_benchmark = [TracemallocBenchmark]
