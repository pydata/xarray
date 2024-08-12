import collections


def frequencies(seq):
    """ Find number of occurrences of each value in seq

    >>> frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])  #doctest: +SKIP
    {'cat': 3, 'ox': 1, 'pig': 2}

    See Also:
        countby
        groupby
    """
    d = collections.defaultdict(int)
    for item in seq:
        d[item] += 1
    return dict(d)
