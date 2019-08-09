# The remove_unused_levels defined here was copied based on the source code
# defined in pandas.core.indexes.muli.py

# For reference, here is a copy of the pandas copyright notice:

# (c) 2011-2012, Lambda Foundry, Inc. and PyData Development Team
# All rights reserved.

# Copyright (c) 2008-2011 AQR Capital Management, LLC
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the copyright holder nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from distutils.version import LooseVersion

import numpy as np
import pandas as pd

# allow ourselves to type checks for Panel even after it's removed
if LooseVersion(pd.__version__) < "0.25.0":
    Panel = pd.Panel
else:

    class Panel:  # type: ignore
        pass


# for pandas 0.19
def remove_unused_levels(self):
    """
    create a new MultiIndex from the current that removing
    unused levels, meaning that they are not expressed in the labels
    The resulting MultiIndex will have the same outward
    appearance, meaning the same .values and ordering. It will also
    be .equals() to the original.
    .. versionadded:: 0.20.0
    Returns
    -------
    MultiIndex
    Examples
    --------
    >>> i = pd.MultiIndex.from_product([range(2), list('ab')])
    MultiIndex(levels=[[0, 1], ['a', 'b']],
               codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> i[2:]
    MultiIndex(levels=[[0, 1], ['a', 'b']],
               codes=[[1, 1], [0, 1]])
    The 0 from the first level is not represented
    and can be removed
    >>> i[2:].remove_unused_levels()
    MultiIndex(levels=[[1], ['a', 'b']],
               codes=[[0, 0], [0, 1]])
    """
    import pandas.core.algorithms as algos

    new_levels = []
    new_labels = []

    changed = False
    for lev, lab in zip(self.levels, self.labels):

        # Since few levels are typically unused, bincount() is more
        # efficient than unique() - however it only accepts positive values
        # (and drops order):
        uniques = np.where(np.bincount(lab + 1) > 0)[0] - 1
        has_na = int(len(uniques) and (uniques[0] == -1))

        if len(uniques) != len(lev) + has_na:
            # We have unused levels
            changed = True

            # Recalculate uniques, now preserving order.
            # Can easily be cythonized by exploiting the already existing
            # "uniques" and stop parsing "lab" when all items are found:
            uniques = algos.unique(lab)
            if has_na:
                na_idx = np.where(uniques == -1)[0]
                # Just ensure that -1 is in first position:
                uniques[[0, na_idx[0]]] = uniques[[na_idx[0], 0]]

            # labels get mapped from uniques to 0:len(uniques)
            # -1 (if present) is mapped to last position
            label_mapping = np.zeros(len(lev) + has_na)
            # ... and reassigned value -1:
            label_mapping[uniques] = np.arange(len(uniques)) - has_na

            lab = label_mapping[lab]

            # new levels are simple
            lev = lev.take(uniques[has_na:])

        new_levels.append(lev)
        new_labels.append(lab)

    result = self._shallow_copy()

    if changed:
        result._reset_identity()
        result._set_levels(new_levels, validate=False)
        result._set_labels(new_labels, validate=False)

    return result
