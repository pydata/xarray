# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Install the SciTools custom benchmarks for detection by ASV.

See the requirements for being detected as an ASV plugin:
https://asv.readthedocs.io/projects/asv-runner/en/latest/development/benchmark_plugins.html
"""

from pathlib import Path
import shutil
from subprocess import run
from tempfile import TemporaryDirectory

this_dir = Path(__file__).parent


def package_files(new_dir: Path) -> None:
    """Package SciTools' custom benchmarks for detection by ASV.

    Parameters
    ----------
    new_dir : Path
        The directory to package the custom benchmarks in.
    """
    asv_bench_scitools = new_dir / "asv_bench_scitools"
    benchmarks = asv_bench_scitools / "benchmarks"
    benchmarks.mkdir(parents=True)
    (asv_bench_scitools / "__init__.py").touch()

    for py_file in this_dir.glob("*.py"):
        if py_file != Path(__file__):
            shutil.copy2(py_file, benchmarks)

    # Create this on the fly, as having multiple pyproject.toml files in 1
    #  project causes problems.
    py_project = new_dir / "pyproject.toml"
    py_project.write_text(
        """
        [project]
        name = "asv_bench_scitools"
        version = "0.1"
        """
    )


def main():
    with TemporaryDirectory() as temp_dir:
        package_files(Path(temp_dir))
        run(["python", "-m", "pip", "install", temp_dir])


if __name__ == "__main__":
    main()
