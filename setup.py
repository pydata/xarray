#!/usr/bin/env python
from setuptools import setup

try:
    setup(use_scm_version=True)
except LookupError as e:
    # .git has been removed, and this is not a package created by sdist
    # This is the case e.g. of a remote deployment with PyCharm Professional
    if not str(e).startswith("setuptools-scm was unable to detect version"):
        raise
    setup(version="999")
