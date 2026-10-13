"""Preflight checks for network tests that depend on the external OPeNDAP server."""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pytest


def _require_opendap_endpoint(url: str) -> None:
    """Skip network integration tests only when the remote server is unavailable.

    Unexpected HTTP responses such as 404 are not treated as transient outages,
    so missing test fixtures still fail the suite.
    """
    try:
        with urlopen(url, timeout=8):
            pass
    except HTTPError as exc:
        if exc.code in {502, 503, 504}:
            pytest.skip(f"OPeNDAP test server unavailable (HTTP {exc.code}): {url}")
        raise
    except (URLError, TimeoutError) as exc:
        pytest.skip(f"OPeNDAP test server unreachable: {url}: {exc}")


@pytest.fixture(scope="session")
def opendap_dap4_available() -> None:
    _require_opendap_endpoint(
        "https://test.opendap.org/opendap/dap4/SimpleGroup.nc4.h5.dmr"
    )


@pytest.fixture(scope="session")
def opendap_hyrax_available() -> None:
    _require_opendap_endpoint(
        "https://test.opendap.org/opendap/hyrax/data/nc/coads_climatology.nc.dmr"
    )
