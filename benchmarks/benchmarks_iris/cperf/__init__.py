# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Benchmarks for the CPerf scheme of the UK Met Office's NG-VAT project.

CPerf = comparing performance working with data in UM versus LFRic formats.

Files available from the UK Met Office:
  moo ls moose:/adhoc/projects/avd/asv/data_for_nightly_tests/
"""

import numpy as np

from iris import load_cube

from ..generate_data import BENCHMARK_DATA
from ..generate_data.ugrid import make_cubesphere_testfile

# The data of the core test UM files has dtype=np.float32 shape=(1920, 2560)
_UM_DIMS_YX = (1920, 2560)
# The closest cubesphere size in terms of datapoints is sqrt(1920*2560 / 6)
#  This gives ~= 905, i.e. "C905"
_N_CUBESPHERE_UM_EQUIVALENT = int(np.sqrt(np.prod(_UM_DIMS_YX) / 6))


class SingleDiagnosticMixin:
    """For use in any benchmark classes that work on a single diagnostic file."""

    params = [
        ["LFRic", "UM", "UM_lbpack0", "UM_netcdf"],
        [False, True],
        [False, True],
    ]
    param_names = ["file type", "height dim (len 71)", "time dim (len 3)"]

    def setup(self, file_type, three_d, three_times):
        if file_type == "LFRic":
            # Generate an appropriate synthetic LFRic file.
            if three_times:
                n_times = 3
            else:
                n_times = 1

            # Use a cubesphere size ~equivalent to our UM test data.
            cells_per_panel_edge = _N_CUBESPHERE_UM_EQUIVALENT
            create_kwargs = dict(c_size=cells_per_panel_edge, n_times=n_times)

            if three_d:
                create_kwargs["n_levels"] = 71

            # Will reuse a file if already present.
            file_path = make_cubesphere_testfile(**create_kwargs)

        else:
            # Locate the appropriate UM file.
            if three_times:
                # pa/pb003 files
                numeric = "003"
            else:
                # pa/pb000 files
                numeric = "000"

            if three_d:
                # theta diagnostic, N1280 file w/ 71 levels (1920, 2560, 71)
                file_name = f"umglaa_pb{numeric}-theta"
            else:
                # surface_temp diagnostic, N1280 file (1920, 2560)
                file_name = f"umglaa_pa{numeric}-surfacetemp"

            file_suffices = {
                "UM": "",  # packed FF (WGDOS lbpack = 1)
                "UM_lbpack0": ".uncompressed",  # unpacked FF (lbpack = 0)
                "UM_netcdf": ".nc",  # UM file -> Iris -> NetCDF file
            }
            suffix = file_suffices[file_type]

            file_path = (BENCHMARK_DATA / file_name).with_suffix(suffix)
            if not file_path.exists():
                message = "\n".join(
                    [
                        f"Expected local file not found: {file_path}",
                        "Available from the UK Met Office.",
                    ]
                )
                raise FileNotFoundError(message)

        self.file_path = file_path
        self.file_type = file_type

    def load(self):
        return load_cube(str(self.file_path))
