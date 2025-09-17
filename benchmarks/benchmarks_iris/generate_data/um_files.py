# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Generate FF, PP and NetCDF files based on a minimal synthetic FF file.

NOTE: uses the Mule package, so depends on an environment with Mule installed.
"""


def _create_um_files(
    len_x: int, len_y: int, len_z: int, len_t: int, compress, save_paths: dict
) -> None:
    """Generate an FF object of given shape and compression, save to FF/PP/NetCDF.

    This is run externally
    (:func:`benchmarks.generate_data.run_function_elsewhere`), so all imports
    are self-contained and input parameters are simple types.
    """
    from copy import deepcopy
    from datetime import datetime
    from tempfile import NamedTemporaryFile

    from mule import ArrayDataProvider, Field3, FieldsFile
    import mule.ff
    from mule.pp import fields_to_pp_file
    import numpy as np

    from iris import load_cube
    from iris import save as save_cube

    def to_bytes_patch(self, field):
        data = field.get_data()
        dtype = mule.ff._DATA_DTYPES[self.WORD_SIZE][field.lbuser1]
        data = data.astype(dtype)
        return data.tobytes(), data.size

    # TODO: remove this patch when fixed in mule, see https://github.com/MetOffice/simulation-systems/discussions/389
    mule.ff._WriteFFOperatorUnpacked.to_bytes = to_bytes_patch

    template = {
        "fixed_length_header": {"dataset_type": 3, "grid_staggering": 3},
        "integer_constants": {
            "num_p_levels": len_z,
            "num_cols": len_x,
            "num_rows": len_y,
        },
        "real_constants": {},
        "level_dependent_constants": {"dims": (len_z + 1, None)},
    }
    new_ff = FieldsFile.from_template(deepcopy(template))

    data_array = np.arange(len_x * len_y).reshape(len_x, len_y)
    array_provider = ArrayDataProvider(data_array)

    def add_field(level_: int, time_step_: int) -> None:
        """Add a minimal field to the new :class:`~mule.FieldsFile`.

        Includes the minimum information to allow Mule saving and Iris
        loading, as well as incrementation for vertical levels and time
        steps to allow generation of z and t dimensions.
        """
        new_field = Field3.empty()
        # To correspond to the header-release 3 class used.
        new_field.lbrel = 3
        # Mule uses the first element of the lookup to test for
        #  unpopulated fields (and skips them), so the first element should
        #  be set to something. The year will do.
        new_field.raw[1] = datetime.now().year

        # Horizontal.
        new_field.lbcode = 1
        new_field.lbnpt = len_x
        new_field.lbrow = len_y
        new_field.bdx = new_ff.real_constants.col_spacing
        new_field.bdy = new_ff.real_constants.row_spacing
        new_field.bzx = new_ff.real_constants.start_lon - 0.5 * new_field.bdx
        new_field.bzy = new_ff.real_constants.start_lat - 0.5 * new_field.bdy

        # Hemisphere.
        new_field.lbhem = 32
        # Processing.
        new_field.lbproc = 0

        # Vertical.
        # Hybrid height values by simulating sequences similar to those in a
        #  theta file.
        new_field.lbvc = 65
        if level_ == 0:
            new_field.lblev = 9999
        else:
            new_field.lblev = level_

        level_1 = level_ + 1
        six_rec = 20 / 3
        three_rec = six_rec / 2

        new_field.blev = level_1**2 * six_rec - six_rec
        new_field.brsvd1 = level_1**2 * six_rec + (six_rec * level_1) - three_rec

        brsvd2_simulated = np.linspace(0.995, 0, len_z)
        shift = min(len_z, 2)
        bhrlev_simulated = np.concatenate([np.ones(shift), brsvd2_simulated[:-shift]])
        new_field.brsvd2 = brsvd2_simulated[level_]
        new_field.bhrlev = bhrlev_simulated[level_]

        # Time.
        new_field.lbtim = 11

        new_field.lbyr = time_step_
        for attr_name in ["lbmon", "lbdat", "lbhr", "lbmin", "lbsec"]:
            setattr(new_field, attr_name, 0)

        new_field.lbyrd = time_step_ + 1
        for attr_name in ["lbmond", "lbdatd", "lbhrd", "lbmind", "lbsecd"]:
            setattr(new_field, attr_name, 0)

        # Data and packing.
        new_field.lbuser1 = 1
        new_field.lbpack = int(compress)
        new_field.bacc = 0
        new_field.bmdi = -1
        new_field.lbext = 0
        new_field.set_data_provider(array_provider)

        new_ff.fields.append(new_field)

    for time_step in range(len_t):
        for level in range(len_z):
            add_field(level, time_step + 1)

    ff_path = save_paths.get("FF", None)
    pp_path = save_paths.get("PP", None)
    nc_path = save_paths.get("NetCDF", None)

    if ff_path:
        new_ff.to_file(ff_path)
    if pp_path:
        fields_to_pp_file(str(pp_path), new_ff.fields)
    if nc_path:
        temp_ff_path = None
        # Need an Iris Cube from the FF content.
        if ff_path:
            # Use the existing file.
            ff_cube = load_cube(ff_path)
        else:
            # Make a temporary file.
            temp_ff_path = NamedTemporaryFile()
            new_ff.to_file(temp_ff_path.name)
            ff_cube = load_cube(temp_ff_path.name)

        save_cube(ff_cube, nc_path, zlib=compress)
        if temp_ff_path:
            temp_ff_path.close()


FILE_EXTENSIONS = {"FF": "", "PP": ".pp", "NetCDF": ".nc"}


def create_um_files(
    len_x: int,
    len_y: int,
    len_z: int,
    len_t: int,
    compress: bool,
    file_types: list,
) -> dict:
    """Generate FF-based FF / PP / NetCDF files with specified shape and compression.

    All files representing a given shape are saved in a dedicated directory. A
    dictionary of the saved paths is returned.

    If the required files exist, they are re-used, unless
    :const:`benchmarks.REUSE_DATA` is ``False``.
    """
    # Self contained imports to avoid linting confusion with _create_um_files().
    from . import BENCHMARK_DATA, REUSE_DATA, run_function_elsewhere

    save_name_sections = ["UM", len_x, len_y, len_z, len_t]
    save_name = "_".join(str(section) for section in save_name_sections)
    save_dir = BENCHMARK_DATA / save_name
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)

    save_paths = {}
    files_exist = True
    for file_type in file_types:
        file_ext = FILE_EXTENSIONS[file_type]
        save_path = (save_dir / f"{compress}").with_suffix(file_ext)
        files_exist = files_exist and save_path.is_file()
        save_paths[file_type] = str(save_path)

    if not REUSE_DATA or not files_exist:
        _ = run_function_elsewhere(
            _create_um_files, len_x, len_y, len_z, len_t, compress, save_paths
        )

    return save_paths
