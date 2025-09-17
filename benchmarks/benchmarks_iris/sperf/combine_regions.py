# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Region combine benchmarks for the SPerf scheme of the UK Met Office's NG-VAT project."""

import os.path

from dask import array as da
import numpy as np

from iris import load, load_cube, save
from iris.mesh.utils import recombine_submeshes

from .. import on_demand_benchmark
from ..generate_data.ugrid import BENCHMARK_DATA, make_cube_like_2d_cubesphere


class Mixin:
    # Characterise time taken + memory-allocated, for various stages of combine
    # operations on cubesphere-like test data.
    timeout = 300.0
    params = [100, 200, 300, 500, 1000, 1668]
    param_names = ["cubesphere_C<N>"]
    # Fix result units for the tracking benchmarks.
    unit = "Mb"
    temp_save_path = BENCHMARK_DATA / "tmp.nc"

    def _parametrised_cache_filename(self, n_cubesphere, content_name):
        return BENCHMARK_DATA / f"cube_C{n_cubesphere}_{content_name}.nc"

    def _make_region_cubes(self, full_mesh_cube):
        """Make a fixed number of region cubes from a full meshcube."""
        # Divide the cube into regions.
        n_faces = full_mesh_cube.shape[-1]
        # Start with a simple list of face indices
        # first extend to multiple of 5
        n_faces_5s = 5 * ((n_faces + 1) // 5)
        i_faces = np.arange(n_faces_5s, dtype=int)
        # reshape (5N,) to (N, 5)
        i_faces = i_faces.reshape((n_faces_5s // 5, 5))
        # reorder [2, 3, 4, 0, 1] within each block of 5
        i_faces = np.concatenate([i_faces[:, 2:], i_faces[:, :2]], axis=1)
        # flatten to get [2 3 4 0 1 (-) 8 9 10 6 7 (-) 13 14 15 11 12 ...]
        i_faces = i_faces.flatten()
        # reduce back to original length, wrap any overflows into valid range
        i_faces = i_faces[:n_faces] % n_faces

        # Divide into regions -- always slightly uneven, since 7 doesn't divide
        n_regions = 7
        n_facesperregion = n_faces // n_regions
        i_face_regions = (i_faces // n_facesperregion) % n_regions
        region_inds = [
            np.where(i_face_regions == i_region)[0] for i_region in range(n_regions)
        ]
        # NOTE: this produces 7 regions, with near-adjacent value ranges but
        # with some points "moved" to an adjacent region.
        # Also, region-0 is bigger (because of not dividing by 7).

        # Finally, make region cubes with these indices.
        region_cubes = [full_mesh_cube[..., inds] for inds in region_inds]
        return region_cubes

    def setup_cache(self):
        """Cache all the necessary source data on disk."""
        # Control dask, to minimise memory usage + allow largest data.
        self.fix_dask_settings()

        for n_cubesphere in self.params:
            # Do for each parameter, since "setup_cache" is NOT parametrised
            mesh_cube = make_cube_like_2d_cubesphere(
                n_cube=n_cubesphere, with_mesh=True
            )
            # Save to files which include the parameter in the names.
            save(
                mesh_cube,
                self._parametrised_cache_filename(n_cubesphere, "meshcube"),
            )
            region_cubes = self._make_region_cubes(mesh_cube)
            save(
                region_cubes,
                self._parametrised_cache_filename(n_cubesphere, "regioncubes"),
            )

    def setup(self, n_cubesphere, imaginary_data=True, create_result_cube=True):
        """Combine tests "standard" setup operation.

        Load the source cubes (full-mesh + region) from disk.
        These are specific to the cubesize parameter.
        The data is cached on disk rather than calculated, to avoid any
        pre-loading of the process memory allocation.

        If 'imaginary_data' is set (default), the region cubes data is replaced
        with lazy data in the form of a da.zeros().  Otherwise, the region data
        is lazy data from the files.

        If 'create_result_cube' is set, create "self.combined_cube" containing
        the (still lazy) result.

        NOTE: various test classes override + extend this.

        """
        # Load source cubes (full-mesh and regions)
        self.full_mesh_cube = load_cube(
            self._parametrised_cache_filename(n_cubesphere, "meshcube")
        )
        self.region_cubes = load(
            self._parametrised_cache_filename(n_cubesphere, "regioncubes")
        )

        # Remove all var-names from loaded cubes, which can otherwise cause
        # problems.  Also implement 'imaginary' data.
        for cube in self.region_cubes + [self.full_mesh_cube]:
            cube.var_name = None
            for coord in cube.coords():
                coord.var_name = None
            if imaginary_data:
                # Replace cube data (lazy file data) with 'imaginary' data.
                # This has the same lazy-array attributes, but is allocated by
                # creating chunks on demand instead of loading from file.
                data = cube.lazy_data()
                data = da.zeros(data.shape, dtype=data.dtype, chunks=data.chunksize)
                cube.data = data

        if create_result_cube:
            self.recombined_cube = self.recombine()

        # Fix dask usage mode for all the subsequent performance tests.
        self.fix_dask_settings()

    def teardown(self, _):
        self.temp_save_path.unlink(missing_ok=True)

    def fix_dask_settings(self):
        """Fix "standard" dask behaviour for time+space testing.

        Currently this is single-threaded mode, with known chunksize,
        which is optimised for space saving so we can test largest data.

        """
        import dask.config as dcfg

        # Use single-threaded, to avoid process-switching costs and minimise memory usage.
        # N.B. generally may be slower, but use less memory ?
        dcfg.set(scheduler="single-threaded")
        # Configure iris._lazy_data.as_lazy_data to aim for 100Mb chunks
        dcfg.set({"array.chunk-size": "128Mib"})

    def recombine(self):
        # A handy general shorthand for the main "combine" operation.
        result = recombine_submeshes(
            self.full_mesh_cube,
            self.region_cubes,
            index_coord_name="i_mesh_face",
        )
        return result

    def save_recombined_cube(self):
        save(self.recombined_cube, self.temp_save_path)


@on_demand_benchmark
class CreateCube(Mixin):
    """Time+memory costs of creating a combined-regions cube.

    The result is lazy, and we don't do the actual calculation.

    """

    def setup(self, n_cubesphere, imaginary_data=True, create_result_cube=False):
        # In this case only, do *not* create the result cube.
        # That is the operation we want to test.
        super().setup(n_cubesphere, imaginary_data, create_result_cube)

    def time_create_combined_cube(self, n_cubesphere):
        self.recombine()

    def tracemalloc_create_combined_cube(self, n_cubesphere):
        self.recombine()


@on_demand_benchmark
class ComputeRealData(Mixin):
    """Time+memory costs of computing combined-regions data."""

    def time_compute_data(self, n_cubesphere):
        _ = self.recombined_cube.data

    def tracemalloc_compute_data(self, n_cubesphere):
        _ = self.recombined_cube.data


@on_demand_benchmark
class SaveData(Mixin):
    """Test saving *only*.

    Test saving *only*, having replaced the input cube data with 'imaginary'
    array data, so that input data is not loaded from disk during the save
    operation.

    """

    def time_save(self, n_cubesphere):
        # Save to disk, which must compute data + stream it to file.
        self.save_recombined_cube()

    def tracemalloc_save(self, n_cubesphere):
        self.save_recombined_cube()

    def track_filesize_saved(self, n_cubesphere):
        self.save_recombined_cube()
        return self.temp_save_path.stat().st_size * 1.0e-6


@on_demand_benchmark
class FileStreamedCalc(Mixin):
    """Test the whole cost of file-to-file streaming.

    Uses the combined cube which is based on lazy data loading from the region
    cubes on disk.

    """

    def setup(self, n_cubesphere, imaginary_data=False, create_result_cube=True):
        # In this case only, do *not* replace the loaded regions data with
        # 'imaginary' data, as we want to test file-to-file calculation+save.
        super().setup(n_cubesphere, imaginary_data, create_result_cube)

    def time_stream_file2file(self, n_cubesphere):
        # Save to disk, which must compute data + stream it to file.
        self.save_recombined_cube()

    def tracemalloc_stream_file2file(self, n_cubesphere):
        self.save_recombined_cube()
