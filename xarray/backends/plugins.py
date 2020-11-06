import inspect
import typing as T

from . import cfgrib_, h5netcdf_, zarr

ENGINES: T.Dict[str, T.Dict[str, T.Any]] = {
    "h5netcdf": {
        "open_dataset": h5netcdf_.open_backend_dataset_h5necdf,
    },
    "zarr": {
        "open_dataset": zarr.open_backend_dataset_zarr,
    },
    "cfgrib": {
        "open_dataset": cfgrib_.open_backend_dataset_cfgrib,
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
