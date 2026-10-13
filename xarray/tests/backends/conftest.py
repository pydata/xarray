"""Preflight checks for network tests that depend on the external OPeNDAP server."""

from __future__ import annotations

from socket import gaierror
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
            # HTTPError owns a response stream; skipping must close it.
            exc.close()
            pytest.skip(f"OPeNDAP test server unavailable (HTTP {exc.code}): {url}")
        raise
    except URLError as exc:
        # An invalid certificate or another unexpected URL error must still fail.
        if isinstance(exc.reason, (ConnectionError, TimeoutError, gaierror)):
            pytest.skip(f"OPeNDAP test server unreachable: {url}: {exc}")
        raise
    except TimeoutError as exc:
        pytest.skip(f"OPeNDAP test server timed out: {url}: {exc}")


def _hyrax_metadata_url(protocol: str) -> str:
    """Return the actual metadata URL requested by the DAP2/DAP4 test."""
    suffix = {"dap2": ".dds", "dap4": ".dmr"}[protocol]
    return "http://test.opendap.org/opendap/hyrax/data/nc/coads_climatology.nc" + suffix


def _datatree_metadata_urls() -> tuple[str, ...]:
    """Probe all three external data fixtures exercised by DataTree tests."""
    base = "http://test.opendap.org/opendap/dap4/"
    return tuple(
        base + name + ".dmr"
        for name in (
            "unaligned_simple_datatree.nc.h5",
            "all_aligned_child_nodes.nc.h5",
            "SimpleGroup.nc4.h5",
        )
    )


@pytest.fixture(scope="class")
def opendap_dap4_available() -> None:
    for url in _datatree_metadata_urls():
        _require_opendap_endpoint(url)


@pytest.fixture
def opendap_hyrax_available(protocol: str) -> None:
    _require_opendap_endpoint(_hyrax_metadata_url(protocol))
