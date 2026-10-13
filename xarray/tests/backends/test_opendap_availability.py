"""Unit tests for OPeNDAP network-test outage detection (no network needed)."""

from __future__ import annotations

from contextlib import nullcontext
from email.message import Message
from socket import gaierror
from ssl import SSLCertVerificationError
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from xarray.tests.backends.conftest import (
    _datatree_metadata_urls,
    _hyrax_metadata_url,
    _require_opendap_endpoint,
)


def test_available_opendap_endpoint_is_not_skipped() -> None:
    with patch("xarray.tests.backends.conftest.urlopen", return_value=nullcontext()):
        _require_opendap_endpoint("https://test.opendap.org/test")


@pytest.mark.parametrize("status_code", [502, 503, 504])
def test_opendap_server_outages_are_skipped(status_code: int) -> None:
    error = HTTPError(
        "https://test.opendap.org/test", status_code, "outage", Message(), None
    )
    with patch("xarray.tests.backends.conftest.urlopen", side_effect=error):
        with pytest.raises(pytest.skip.Exception, match=f"HTTP {status_code}"):
            _require_opendap_endpoint("https://test.opendap.org/test")


@pytest.mark.parametrize("status_code", [401, 404, 500])
def test_other_http_errors_remain_failures(status_code: int) -> None:
    error = HTTPError(
        "https://test.opendap.org/test", status_code, "error", Message(), None
    )
    with patch("xarray.tests.backends.conftest.urlopen", side_effect=error):
        with pytest.raises(HTTPError) as raised:
            _require_opendap_endpoint("https://test.opendap.org/test")
    assert raised.value.code == status_code


@pytest.mark.parametrize(
    "error",
    [
        URLError(ConnectionRefusedError("connection refused")),
        URLError(gaierror(-2, "DNS unavailable")),
    ],
)
def test_connection_failures_are_skipped(error: Exception) -> None:
    with patch("xarray.tests.backends.conftest.urlopen", side_effect=error):
        with pytest.raises(pytest.skip.Exception, match="unreachable"):
            _require_opendap_endpoint("https://test.opendap.org/test")


def test_direct_timeout_is_skipped() -> None:
    with patch(
        "xarray.tests.backends.conftest.urlopen", side_effect=TimeoutError("timeout")
    ):
        with pytest.raises(pytest.skip.Exception, match="timed out"):
            _require_opendap_endpoint("https://test.opendap.org/test")


@pytest.mark.parametrize(
    "error",
    [
        URLError(SSLCertVerificationError("invalid certificate")),
        URLError("unexpected URL failure"),
    ],
)
def test_unexpected_url_errors_remain_failures(error: URLError) -> None:
    with patch("xarray.tests.backends.conftest.urlopen", side_effect=error):
        with pytest.raises(URLError):
            _require_opendap_endpoint("https://test.opendap.org/test")


@pytest.mark.parametrize("protocol, suffix", [("dap2", ".dds"), ("dap4", ".dmr")])
def test_hyrax_probe_matches_pydap_protocol(protocol: str, suffix: str) -> None:
    assert _hyrax_metadata_url(protocol) == (
        "http://test.opendap.org/opendap/hyrax/data/nc/coads_climatology.nc" + suffix
    )


def test_datatree_probe_covers_all_external_test_files() -> None:
    assert set(_datatree_metadata_urls()) == {
        "http://test.opendap.org/opendap/dap4/unaligned_simple_datatree.nc.h5.dmr",
        "http://test.opendap.org/opendap/dap4/all_aligned_child_nodes.nc.h5.dmr",
        "http://test.opendap.org/opendap/dap4/SimpleGroup.nc4.h5.dmr",
    }
