# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Wrappers for using :mod:`iris.tests.stock` methods for benchmarking.

See :mod:`benchmarks.generate_data` for an explanation of this structure.
"""

from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path

import iris
from iris import cube
from iris.mesh import load_mesh

from . import BENCHMARK_DATA, REUSE_DATA, load_realised, run_function_elsewhere


def hash_args(*args, **kwargs):
    """Convert arguments into a short hash - for preserving args in filenames."""
    arg_string = str(args)
    kwarg_string = json.dumps(kwargs)
    full_string = arg_string + kwarg_string
    return sha256(full_string.encode()).hexdigest()[:10]


def _create_file__xios_common(func_name, **kwargs):
    def _external(func_name_, temp_file_dir, **kwargs_):
        from iris.tests.stock import netcdf

        func = getattr(netcdf, func_name_)
        print(func(temp_file_dir, **kwargs_), end="")

    args_hash = hash_args(**kwargs)
    save_path = (BENCHMARK_DATA / f"{func_name}_{args_hash}").with_suffix(".nc")
    if not REUSE_DATA or not save_path.is_file():
        # The xios functions take control of save location so need to move to
        #  a more specific name that allows reuse.
        actual_path = run_function_elsewhere(
            _external,
            func_name_=func_name,
            temp_file_dir=str(BENCHMARK_DATA),
            **kwargs,
        )
        Path(actual_path).replace(save_path)
    return save_path


def create_file__xios_2d_face_half_levels(
    temp_file_dir, dataset_name, n_faces=866, n_times=1
):
    """Create file wrapper for :meth:`iris.tests.stock.netcdf.create_file__xios_2d_face_half_levels`.

    Have taken control of temp_file_dir

    todo: is create_file__xios_2d_face_half_levels still appropriate now we can
     properly save Mesh Cubes?
    """
    return _create_file__xios_common(
        func_name="create_file__xios_2d_face_half_levels",
        dataset_name=dataset_name,
        n_faces=n_faces,
        n_times=n_times,
    )


def create_file__xios_3d_face_half_levels(
    temp_file_dir, dataset_name, n_faces=866, n_times=1, n_levels=38
):
    """Create file wrapper for :meth:`iris.tests.stock.netcdf.create_file__xios_3d_face_half_levels`.

    Have taken control of temp_file_dir

    todo: is create_file__xios_3d_face_half_levels still appropriate now we can
     properly save Mesh Cubes?
    """
    return _create_file__xios_common(
        func_name="create_file__xios_3d_face_half_levels",
        dataset_name=dataset_name,
        n_faces=n_faces,
        n_times=n_times,
        n_levels=n_levels,
    )


def sample_mesh(n_nodes=None, n_faces=None, n_edges=None, lazy_values=False):
    """Sample mesh wrapper for :meth:iris.tests.stock.mesh.sample_mesh`."""

    def _external(*args, **kwargs):
        from iris.mesh import save_mesh
        from iris.tests.stock.mesh import sample_mesh

        save_path_ = kwargs.pop("save_path")
        # Always saving, so laziness is irrelevant. Use lazy to save time.
        kwargs["lazy_values"] = True
        new_mesh = sample_mesh(*args, **kwargs)
        save_mesh(new_mesh, save_path_)

    arg_list = [n_nodes, n_faces, n_edges]
    args_hash = hash_args(*arg_list)
    save_path = (BENCHMARK_DATA / f"sample_mesh_{args_hash}").with_suffix(".nc")
    if not REUSE_DATA or not save_path.is_file():
        _ = run_function_elsewhere(_external, *arg_list, save_path=str(save_path))
    if not lazy_values:
        # Realise everything.
        with load_realised():
            mesh = load_mesh(str(save_path))
    else:
        mesh = load_mesh(str(save_path))
    return mesh


def sample_meshcoord(sample_mesh_kwargs=None, location="face", axis="x"):
    """Sample meshcoord wrapper for :meth:`iris.tests.stock.mesh.sample_meshcoord`.

    Parameters deviate from the original as cannot pass a
    :class:`iris.mesh.Mesh to the separate Python instance - must
    instead generate the Mesh as well.

    MeshCoords cannot be saved to file, so the _external method saves the
    MeshCoord's Mesh, then the original Python instance loads in that Mesh and
    regenerates the MeshCoord from there.
    """

    def _external(sample_mesh_kwargs_, save_path_):
        from iris.mesh import save_mesh
        from iris.tests.stock.mesh import sample_mesh, sample_meshcoord

        if sample_mesh_kwargs_:
            input_mesh = sample_mesh(**sample_mesh_kwargs_)
        else:
            input_mesh = None
        # Don't parse the location or axis arguments - only saving the Mesh at
        #  this stage.
        new_meshcoord = sample_meshcoord(mesh=input_mesh)
        save_mesh(new_meshcoord.mesh, save_path_)

    args_hash = hash_args(**sample_mesh_kwargs)
    save_path = (BENCHMARK_DATA / f"sample_mesh_coord_{args_hash}").with_suffix(".nc")
    if not REUSE_DATA or not save_path.is_file():
        _ = run_function_elsewhere(
            _external,
            sample_mesh_kwargs_=sample_mesh_kwargs,
            save_path_=str(save_path),
        )
    with load_realised():
        source_mesh = load_mesh(str(save_path))
    # Regenerate MeshCoord from its Mesh, which we saved.
    return source_mesh.to_MeshCoord(location=location, axis=axis)


def realistic_4d_w_everything(w_mesh=False, lazy=False) -> iris.cube.Cube:
    """Run :func:`iris.tests.stock.realistic_4d_w_everything` in ``DATA_GEN_PYTHON``.

    Parameters
    ----------
    w_mesh : bool
        See :func:`iris.tests.stock.realistic_4d_w_everything` for details.
    lazy : bool
        If True, the Cube will be returned with all arrays as they would
        normally be loaded from file (i.e. most will still be lazy Dask
        arrays). If False, all arrays (except derived coordinates) will be
        realised NumPy arrays.

    """

    def _external(w_mesh_: str, save_path_: str):
        import iris
        from iris.tests.stock import realistic_4d_w_everything

        cube = realistic_4d_w_everything(w_mesh=bool(w_mesh_))
        iris.save(cube, save_path_)

    save_path = (BENCHMARK_DATA / f"realistic_4d_w_everything_{w_mesh}").with_suffix(
        ".nc"
    )
    if not REUSE_DATA or not save_path.is_file():
        _ = run_function_elsewhere(_external, w_mesh_=w_mesh, save_path_=str(save_path))
    context = nullcontext() if lazy else load_realised()
    with context:
        return iris.load_cube(save_path, "air_potential_temperature")
