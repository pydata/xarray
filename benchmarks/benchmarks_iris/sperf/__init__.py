# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Benchmarks for the SPerf scheme of the UK Met Office's NG-VAT project.

SPerf = assessing performance against a series of increasingly large LFRic
datasets.
"""

from iris import load_cube

from ..generate_data.ugrid import make_cubesphere_testfile


class FileMixin:
    """For use in any benchmark classes that work on a file."""

    # Allows time for large file generation.
    timeout = 3600.0
    # Largest file with these params: ~90GB.
    #  Total disk space: ~410GB.
    params = [
        [12, 384, 640, 960, 1280, 1668],
        [1, 36, 72],
        [1, 3, 10],
    ]
    param_names = ["cubesphere_C<N>", "N levels", "N time steps"]
    # cubesphere_C<N>: notation refers to faces per panel.
    #  e.g. C1 is 6 faces, 8 nodes

    def setup(self, c_size, n_levels, n_times):
        self.file_path = make_cubesphere_testfile(
            c_size=c_size, n_levels=n_levels, n_times=n_times
        )

    def load_cube(self):
        return load_cube(str(self.file_path))
