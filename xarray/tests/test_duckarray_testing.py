import pytest

from .duckarray_testing import duckarray_module
from .test_units import assert_units_equal

da = pytest.importorskip("dask.array")
pint = pytest.importorskip("pint")
sparse = pytest.importorskip("sparse")
ureg = pint.UnitRegistry(force_ndarray_like=True)


def create_pint(data, method):
    if method in ("cumprod",):
        units = "dimensionless"
    else:
        units = "m"
    return ureg.Quantity(data, units)


def create_pint_dask(data, method):
    data = da.from_array(data, chunks=(2,))
    return create_pint(data, method)


def create_sparse(data, method):
    return sparse.COO.from_numpy(data)


TestPint = duckarray_module(
    "pint",
    create_pint,
    extra_asserts=assert_units_equal,
    marks={
        "TestVariable.test_reduce": {
            "[argsort]": [
                pytest.mark.skip(reason="xarray.Variable.argsort does not support dim")
            ],
            "[prod]": [pytest.mark.skip(reason="nanprod drops units")],
        },
    },
    global_marks=[
        pytest.mark.filterwarnings("error::pint.UnitStrippedWarning"),
    ],
)

TestPintDask = duckarray_module(
    "pint_dask",
    create_pint_dask,
    extra_asserts=assert_units_equal,
    marks={
        "TestVariable.test_reduce": {
            "[argsort]": [
                pytest.mark.skip(reason="xarray.Variable.argsort does not support dim")
            ],
            "[cumsum]": [pytest.mark.skip(reason="nancumsum drops the units")],
            "[median]": [pytest.mark.skip(reason="median does not support dim")],
            "[prod]": [pytest.mark.skip(reason="prod drops the units")],
            "[cumprod]": [pytest.mark.skip(reason="cumprod drops the units")],
            "[std]": [pytest.mark.skip(reason="nanstd drops the units")],
            "[sum]": [pytest.mark.skip(reason="sum drops the units")],
            "[var]": [pytest.mark.skip(reason="var drops the units")],
        },
    },
    global_marks=[
        pytest.mark.filterwarnings("error::pint.UnitStrippedWarning"),
    ],
)

TestSparse = duckarray_module(
    "sparse",
    create_sparse,
    marks={
        "TestVariable.test_reduce": {
            "[argmax]": [pytest.mark.skip(reason="not implemented by sparse")],
            "[argmin]": [pytest.mark.skip(reason="not implemented by sparse")],
            "[argsort]": [pytest.mark.skip(reason="not implemented by sparse")],
            "[cumprod]": [pytest.mark.skip(reason="not implemented by sparse")],
            "[cumsum]": [pytest.mark.skip(reason="not implemented by sparse")],
            "[median]": [pytest.mark.skip(reason="not implemented by sparse")],
            "[std]": [pytest.mark.skip(reason="nanstd not implemented, yet")],
            "[var]": [pytest.mark.skip(reason="nanvar not implemented, yet")],
        },
    },
)
