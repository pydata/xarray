import pytest

pint = pytest.importorskip("pint")
from xarray.tests.test_units.pint import unit_registry


def parametrize_unit_compatibility(error, allow_compatible_unit=True):
    return pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, error, id="no_unit"),
            pytest.param(unit_registry.dimensionless, error, id="dimensionless"),
            pytest.param(unit_registry.s, error, id="incompatible_unit"),
            pytest.param(
                unit_registry.mm,
                None if allow_compatible_unit else error,
                id="compatible_unit",
            ),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
        ids=repr,
    )
