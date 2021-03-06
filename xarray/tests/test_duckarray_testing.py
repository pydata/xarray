import pytest

from xarray.duckarray import duckarray_module

pint = pytest.importorskip("pint")
sparse = pytest.importorskip("sparse")
ureg = pint.UnitRegistry(force_ndarray_like=True)


def create_pint(data, method):
    if method in ("prod",):
        units = "dimensionless"
    else:
        units = "m"
    return ureg.Quantity(data, units)


def create_sparse(data, method):
    return sparse.COO.from_numpy(data)


TestPint = duckarray_module(
    "pint",
    create_pint,
    marks={
        "TestDataset.test_reduce[prod]": [
            pytest.mark.skip(reason="not implemented yet")
        ],
    },
    global_marks=[
        pytest.mark.filterwarnings("error::pint.UnitStrippedWarning"),
    ],
)

TestSparse = duckarray_module(
    "sparse",
    create_sparse,
    marks={
        "TestDataset.test_reduce": {
            "[median]": [pytest.mark.skip(reason="not implemented, yet")],
            "[std]": [pytest.mark.skip(reason="not implemented, yet")],
            "[var]": [pytest.mark.skip(reason="not implemented, yet")],
        },
    },
)
