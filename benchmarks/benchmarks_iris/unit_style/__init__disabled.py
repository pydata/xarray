# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Small-scope benchmarks that can help with performance investigations.

By renaming ``__init__.py`` these are all disabled by default:

- They bloat benchmark run-time.
- They are too vulnerable to 'noise' due to their small scope - small objects,
  short operations - they report a lot of false positive regressions.
- We rely on the wider-scope integration-style benchmarks to flag performance
  changes, upon which we expect to do some manual investigation - these
  smaller benchmarks can be run then.

"""
