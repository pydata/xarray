import pytest

from xarray.duckarray import duckarray_module

pint = pytest.importorskip("pint")
ureg = pint.UnitRegistry(force_ndarray_like=True)


def create(data, method):
    if method in ("prod",):
        units = "dimensionless"
    else:
        units = "m"
    return ureg.Quantity(data, units)


TestPint = duckarray_module(
    "pint",
    create,
    marks={
        "TestDataset.test_reduce": [pytest.mark.skip(reason="not implemented yet")],
    },
    global_marks=[
        pytest.mark.filterwarnings("error::pint.UnitStrippedWarning"),
    ],
)
