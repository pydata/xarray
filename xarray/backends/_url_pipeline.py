"""Recognize Zarr URL pipeline strings (ZEP 8).

A URL pipeline such as ``"s3://bucket/repo|icechunk://tag.v1/"`` is resolved
by zarr-python. :func:`as_zarr_url_pipeline` is the only place in xarray that
decides whether a value is one: it turns a pipeline string into a
``zarr.storage.URLPipeline`` and returns everything else unchanged. With zarr
releases that predate URL pipelines it never converts anything, so xarray's
behavior is unchanged.
"""

from __future__ import annotations

import functools
import sys
from typing import TYPE_CHECKING

from xarray.namedarray.utils import module_available

if TYPE_CHECKING:
    from zarr.storage import URLPipeline  # type: ignore[attr-defined,unused-ignore]


@functools.cache
def _has_url_pipeline() -> bool:
    """Whether the installed zarr supports URL pipelines (feature detection)."""
    if not module_available("zarr"):
        return False
    import zarr.storage  # optional dependency, only imported once a "|" is seen

    return hasattr(zarr.storage, "URLPipeline")


def as_zarr_url_pipeline[T](store: T) -> T | URLPipeline:
    """Return a zarr URLPipeline for a Zarr URL pipeline string, else `store` unchanged.

    zarr's parser decides: a string it rejects (e.g. a root without a URL
    scheme, such as a local path containing ``|``) keeps its usual meaning. A
    string that parses but names an adapter that isn't installed is still a
    pipeline, and fails when it is opened.
    """
    # cheap pre-check first, so that ordinary paths never import zarr
    if not isinstance(store, str) or "|" not in store or not _has_url_pipeline():
        return store
    # optional dependency, and only present in zarr versions with URL pipelines
    from zarr.errors import URLPipelineError  # type: ignore[attr-defined,unused-ignore]
    from zarr.storage import URLPipeline  # type: ignore[attr-defined,unused-ignore]

    try:
        return URLPipeline.from_url(store)
    except URLPipelineError:
        return store  # not a pipeline: keep the string's existing meaning


def _is_zarr_url_pipeline(store: object) -> bool:
    """Whether `store` is a URLPipeline, or a string that as_zarr_url_pipeline converts."""
    store = as_zarr_url_pipeline(store)
    # a URLPipeline object can only exist once zarr.storage has been imported
    url_pipeline = getattr(sys.modules.get("zarr.storage"), "URLPipeline", None)
    return url_pipeline is not None and isinstance(store, url_pipeline)
