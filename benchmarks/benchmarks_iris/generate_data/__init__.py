# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Scripts for generating supporting data for benchmarking.

Data generated using this repo should use :func:`run_function_elsewhere`, which
means that data is generated using a fixed version of this repo and a fixed
environment, rather than those that get changed when the benchmarking run
checks out a new commit.

Downstream use of data generated 'elsewhere' requires saving; usually in a
NetCDF file. Could also use pickling but there is a potential risk if the
benchmark sequence runs over two different Python versions.

"""

from contextlib import contextmanager
from inspect import getsource
from os import environ
from pathlib import Path
from subprocess import CalledProcessError, check_output, run
from textwrap import dedent
from warnings import warn

from iris._lazy_data import as_concrete_data
from iris.fileformats import netcdf

#: Python executable used by :func:`run_function_elsewhere`, set via env
#:  variable of same name. Must be path of Python within an environment that
#:  includes this repo (including dependencies and test modules) and Mule.
try:
    DATA_GEN_PYTHON = environ["DATA_GEN_PYTHON"]
    _ = check_output([DATA_GEN_PYTHON, "-c", "a = True"])
except KeyError:
    error = "Env variable DATA_GEN_PYTHON not defined."
    raise KeyError(error)
except (CalledProcessError, FileNotFoundError, PermissionError):
    error = "Env variable DATA_GEN_PYTHON not a runnable python executable path."
    raise ValueError(error)

# The default location of data files used in benchmarks. Used by CI.
default_data_dir = (Path(__file__).parents[2] / ".data").resolve()
# Optionally override the default data location with environment variable.
BENCHMARK_DATA = Path(environ.get("BENCHMARK_DATA", default_data_dir))
if BENCHMARK_DATA == default_data_dir:
    BENCHMARK_DATA.mkdir(exist_ok=True)
    message = (
        f"No BENCHMARK_DATA env var, defaulting to {BENCHMARK_DATA}. "
        "Note that some benchmark files are GB in size."
    )
    warn(message)
elif not BENCHMARK_DATA.is_dir():
    message = f"Not a directory: {BENCHMARK_DATA} ."
    raise ValueError(message)

# Manual flag to allow the rebuilding of synthetic data.
#  False forces a benchmark run to re-make all the data files.
REUSE_DATA = True


class DataGenerationError(Exception):
    """Exception raised for errors during data generation."""

    pass


def run_function_elsewhere(func_to_run, *args, **kwargs):
    """Run a given function using the :const:`DATA_GEN_PYTHON` executable.

    This structure allows the function to be written natively.

    Parameters
    ----------
    func_to_run : FunctionType
        The function object to be run.
        NOTE: the function must be completely self-contained, i.e. perform all
        its own imports (within the target :const:`DATA_GEN_PYTHON`
        environment).
    *args : tuple, optional
        Function call arguments. Must all be expressible as simple literals,
        i.e. the ``repr`` must be a valid literal expression.
    **kwargs: dict, optional
        Function call keyword arguments. All values must be expressible as
        simple literals (see ``*args``).

    Returns
    -------
    str
        The ``stdout`` from the run.

    """
    func_string = dedent(getsource(func_to_run))
    func_string = func_string.replace("@staticmethod\n", "")
    func_call_term_strings = [repr(arg) for arg in args]
    func_call_term_strings += [f"{name}={repr(val)}" for name, val in kwargs.items()]
    func_call_string = (
        f"{func_to_run.__name__}(" + ",".join(func_call_term_strings) + ")"
    )
    python_string = "\n".join([func_string, func_call_string])

    try:
        result = run(
            [DATA_GEN_PYTHON, "-c", python_string],
            capture_output=True,
            check=True,
            text=True,
        )
    except CalledProcessError as error_:
        # From None 'breaks' the error chain - we don't want the original
        #  traceback since it is long and confusing.
        raise DataGenerationError(error_.stderr) from None

    return result.stdout


@contextmanager
def load_realised():
    """Force NetCDF loading with realised arrays.

    Since passing between data generation and benchmarking environments is via
    file loading, but some benchmarks are only meaningful if starting with real
    arrays.
    """
    from iris.fileformats._nc_load_rules import helpers
    from iris.fileformats.netcdf.loader import _get_cf_var_data as pre_patched

    def patched(*args, **kwargs):
        return as_concrete_data(pre_patched(*args, **kwargs))

    netcdf.loader._get_cf_var_data = patched
    helpers._get_cf_var_data = patched
    yield
    netcdf.loader._get_cf_var_data = pre_patched
    helpers._get_cf_var_data = pre_patched
