import inspect
import typing as T
import entrypoints
import sys

path = sys.path.copy()
path = set(path)

ENGINES: T.Dict[str, T.Dict[str, T.Any]] = {
    "h5netcdf": {
        "open_dataset": entrypoints.get_single("xarray.backends", "h5netcdf", path=path).load(),
    },
    "zarr": {
        "open_dataset": entrypoints.get_single("xarray.backends", "zarr", path=path).load(),
    },
    "cfgrib": {
        "open_dataset": entrypoints.get_single("xarray.backends", "cfgrib", path=path).load(),
    },
}


for engine in ENGINES.values():
    if "signature" not in engine:
        parameters = inspect.signature(engine["open_dataset"]).parameters
        for name, param in parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                raise TypeError(
                    f'All the parameters in {engine["open_dataset"]!r} signature should be explicit. '
                    "*args and **kwargs is not supported"
                )
        engine["signature"] = set(parameters)
