"""
This module exposes Xarray's custom exceptions & warnings.
"""


class AlignmentError(ValueError):
    """Error class for alignment failures due to incompatible arguments."""


class MergeError(ValueError):
    """Error class for merge failures due to incompatible arguments."""


class InvalidTreeError(Exception):
    """Raised when user attempts to create an invalid tree in some way."""


class NotFoundInTreeError(ValueError):
    """Raised when operation can't be completed because one node is not part of the expected tree."""


class TreeIsomorphismError(ValueError):
    """Error raised if two tree objects do not share the same node structure."""


class SerializationWarning(RuntimeWarning):
    """Warnings about encoding/decoding issues in serialization."""


__all__ = [
    "AlignmentError",
    "InvalidTreeError",
    "MergeError",
    "SerializationWarning",
    "TreeIsomorphismError",
]
