# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Scripts for generating supporting data for UGRID-related benchmarking."""

from iris import load_cube as iris_loadcube

from . import BENCHMARK_DATA, REUSE_DATA, load_realised, run_function_elsewhere
from .stock import (
    create_file__xios_2d_face_half_levels,
    create_file__xios_3d_face_half_levels,
)


def generate_cube_like_2d_cubesphere(n_cube: int, with_mesh: bool, output_path: str):
    """Construct and save to file an LFRIc cubesphere-like cube.

    Construct and save to file an LFRIc cubesphere-like cube for a given
    cubesphere size, *or* a simpler structured (UM-like) cube of equivalent
    size.

    NOTE: this function is *NEVER* called from within this actual package.
    Instead, it is to be called via benchmarks.remote_data_generation,
    so that it can use up-to-date facilities, independent of the ASV controlled
    environment which contains the "Iris commit under test".

    This means:

    * it must be completely self-contained : i.e. it includes all its
      own imports, and saves results to an output file.

    """
    from iris import save
    from iris.tests.stock.mesh import sample_mesh, sample_mesh_cube

    n_face_nodes = n_cube * n_cube
    n_faces = 6 * n_face_nodes

    # Set n_nodes=n_faces and n_edges=2*n_faces
    # : Not exact, but similar to a 'real' cubesphere.
    n_nodes = n_faces
    n_edges = 2 * n_faces
    if with_mesh:
        mesh = sample_mesh(
            n_nodes=n_nodes, n_faces=n_faces, n_edges=n_edges, lazy_values=True
        )
        cube = sample_mesh_cube(mesh=mesh, n_z=1)
    else:
        cube = sample_mesh_cube(nomesh_faces=n_faces, n_z=1)

    # Strip off the 'extra' aux-coord mapping the mesh, which sample-cube adds
    # but which we don't want.
    cube.remove_coord("mesh_face_aux")

    # Save the result to a named file.
    save(cube, output_path)


def make_cube_like_2d_cubesphere(n_cube: int, with_mesh: bool):
    """Generate an LFRIc cubesphere-like cube.

    Generate an LFRIc cubesphere-like cube for a given cubesphere size,
    *or* a simpler structured (UM-like) cube of equivalent size.

    All the cube data, coords and mesh content are LAZY, and produced without
    allocating large real arrays (to allow peak-memory testing).

    NOTE: the actual cube generation is done in a stable Iris environment via
    benchmarks.remote_data_generation, so it is all channeled via cached netcdf
    files in our common testdata directory.

    """
    identifying_filename = f"cube_like_2d_cubesphere_C{n_cube}_Mesh={with_mesh}.nc"
    filepath = BENCHMARK_DATA / identifying_filename
    if not filepath.exists():
        # Create the required testfile, by running the generation code remotely
        #  in a 'fixed' python environment.
        run_function_elsewhere(
            generate_cube_like_2d_cubesphere,
            n_cube,
            with_mesh=with_mesh,
            output_path=str(filepath),
        )

    # File now *should* definitely exist: content is simply the desired cube.
    cube = iris_loadcube(str(filepath))

    # Ensure correct laziness.
    _ = cube.data
    for coord in cube.coords(mesh_coords=False):
        assert not coord.has_lazy_points()
        assert not coord.has_lazy_bounds()
    if cube.mesh:
        for coord in cube.mesh.coords():
            assert coord.has_lazy_points()
        for conn in cube.mesh.connectivities():
            assert conn.has_lazy_indices()

    return cube


def make_cube_like_umfield(xy_dims):
    """Create a "UM-like" cube with lazy content, for save performance testing.

    Roughly equivalent to a single current UM cube, to be compared with
    a "make_cube_like_2d_cubesphere(n_cube=_N_CUBESPHERE_UM_EQUIVALENT)"
    (see below).

    Note: probably a bit over-simplified, as there is no time coord, but that
    is probably equally true of our LFRic-style synthetic data.

    Parameters
    ----------
    xy_dims : 2-tuple
        Set the horizontal dimensions = n-lats, n-lons.

    """

    def _external(xy_dims_, save_path_):
        from dask import array as da
        import numpy as np

        from iris import save
        from iris.coords import DimCoord
        from iris.cube import Cube

        nz, ny, nx = (1,) + xy_dims_

        # Base data : Note this is float32 not float64 like LFRic/XIOS outputs.
        lazy_data = da.zeros((nz, ny, nx), dtype=np.float32)
        cube = Cube(lazy_data, long_name="structured_phenom")

        # Add simple dim coords also.
        z_dimco = DimCoord(np.arange(nz), long_name="level", units=1)
        y_dimco = DimCoord(
            np.linspace(-90.0, 90.0, ny),
            standard_name="latitude",
            units="degrees",
        )
        x_dimco = DimCoord(
            np.linspace(-180.0, 180.0, nx),
            standard_name="longitude",
            units="degrees",
        )
        for idim, co in enumerate([z_dimco, y_dimco, x_dimco]):
            cube.add_dim_coord(co, idim)

        save(cube, save_path_)

    save_path = (BENCHMARK_DATA / f"make_cube_like_umfield_{xy_dims}").with_suffix(
        ".nc"
    )
    if not REUSE_DATA or not save_path.is_file():
        _ = run_function_elsewhere(_external, xy_dims, str(save_path))
    with load_realised():
        cube = iris_loadcube(str(save_path))

    return cube


def make_cubesphere_testfile(c_size, n_levels=0, n_times=1):
    """Build a C<c_size> cubesphere testfile in a given directory.

    Build a C<c_size> cubesphere testfile in a given directory, with a standard naming.
    If n_levels > 0 specified: 3d file with the specified number of levels.
    Return the file path.

    TODO: is create_file__xios... still appropriate now we can properly save Mesh Cubes?

    """
    n_faces = 6 * c_size * c_size
    stem_name = f"mesh_cubesphere_C{c_size}_t{n_times}"
    kwargs = dict(
        temp_file_dir=None,
        dataset_name=stem_name,  # N.B. function adds the ".nc" extension
        n_times=n_times,
        n_faces=n_faces,
    )

    three_d = n_levels > 0
    if three_d:
        kwargs["n_levels"] = n_levels
        kwargs["dataset_name"] += f"_{n_levels}levels"
        func = create_file__xios_3d_face_half_levels
    else:
        func = create_file__xios_2d_face_half_levels

    file_path = func(**kwargs)
    return file_path
