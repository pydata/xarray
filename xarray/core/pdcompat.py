import numpy as np
import pandas as pd
import pandas.core.algorithms as algos


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
               labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> i[2:]
    MultiIndex(levels=[[0, 1], ['a', 'b']],
               labels=[[1, 1], [0, 1]])
    The 0 from the first level is not represented
    and can be removed
    >>> i[2:].remove_unused_levels()
    MultiIndex(levels=[[1], ['a', 'b']],
               labels=[[0, 0], [0, 1]])
    """

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
