# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Mesh data loading benchmark tests."""

from iris import load_cube as iris_load_cube
from iris.mesh import load_mesh as iris_load_mesh

from ..generate_data.stock import create_file__xios_2d_face_half_levels


def synthetic_data(**kwargs):
    # Ensure all uses of the synthetic data function use the common directory.
    # File location is controlled by :mod:`generate_data`, hence temp_file_dir=None.
    return create_file__xios_2d_face_half_levels(temp_file_dir=None, **kwargs)


def load_cube(*args, **kwargs):
    return iris_load_cube(*args, **kwargs)


def load_mesh(*args, **kwargs):
    return iris_load_mesh(*args, **kwargs)


class BasicLoading:
    params = [1, int(2e5)]
    param_names = ["number of faces"]

    def setup_common(self, **kwargs):
        self.data_path = synthetic_data(**kwargs)

    def setup(self, *args):
        self.setup_common(dataset_name="Loading", n_faces=args[0])

    def time_load_file(self, *args):
        _ = load_cube(str(self.data_path))

    def time_load_mesh(self, *args):
        _ = load_mesh(str(self.data_path))


class BasicLoadingTime(BasicLoading):
    """Same as BasicLoading, but scaling over a time series - an unlimited dimension."""

    # NOTE iris#4834 - careful how big the time dimension is (time dimension
    #  is UNLIMITED).

    param_names = ["number of time steps"]

    def setup(self, *args):
        self.setup_common(dataset_name="Loading", n_faces=1, n_times=args[0])


class DataRealisation:
    # Prevent repeat runs between setup() runs - data won't be lazy after 1st.
    number = 1
    # Compensate for reduced certainty by increasing number of repeats.
    repeat = (10, 10, 10.0)
    # Prevent ASV running its warmup, which ignores `number` and would
    # therefore get a false idea of typical run time since the data would stop
    # being lazy.
    warmup_time = 0.0
    timeout = 300.0

    params = [int(1e4), int(2e5)]
    param_names = ["number of faces"]

    def setup_common(self, **kwargs):
        data_path = synthetic_data(**kwargs)
        self.cube = load_cube(str(data_path))

    def setup(self, *args):
        self.setup_common(dataset_name="Realisation", n_faces=args[0])

    def time_realise_data(self, *args):
        assert self.cube.has_lazy_data()
        _ = self.cube.data[0]


class DataRealisationTime(DataRealisation):
    """Same as DataRealisation, but scaling over a time series - an unlimited dimension."""

    param_names = ["number of time steps"]

    def setup(self, *args):
        self.setup_common(dataset_name="Realisation", n_faces=1, n_times=args[0])


class Callback:
    params = [1, int(2e5)]
    param_names = ["number of faces"]

    def setup_common(self, **kwargs):
        def callback(cube, field, filename):
            return cube[::2]

        self.data_path = synthetic_data(**kwargs)
        self.callback = callback

    def setup(self, *args):
        self.setup_common(dataset_name="Loading", n_faces=args[0])

    def time_load_file_callback(self, *args):
        _ = load_cube(str(self.data_path), callback=self.callback)


class CallbackTime(Callback):
    """Same as Callback, but scaling over a time series - an unlimited dimension."""

    param_names = ["number of time steps"]

    def setup(self, *args):
        self.setup_common(dataset_name="Loading", n_faces=1, n_times=args[0])
