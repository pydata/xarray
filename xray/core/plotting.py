"""
Plotting functions are implemented here and also monkeypatched in to
DataArray and DataSet classes
"""

# TODO - Is there a better way to import matplotlib in the function?
# Decorators don't preserve the argument names


class FacetGrid():
    pass


def plot(darray, ax=None, *args, **kwargs):
    """
    Default plot of DataArray using matplotlib / pylab.

    Parameters
    ----------
    darray : DataArray
        Must be 1 dimensional
    ax : matplotlib axes object
        If not passed, uses plt.gca()
    args, kwargs
        Additional arguments to matplotlib
    """
    return plot_line(darray, ax)


def plot_line(darray, ax=None, *args, **kwargs):
    """
    Line plot of DataArray using matplotlib / pylab.

    Parameters
    ----------
    darray : DataArray
        Must be 1 dimensional
    ax : matplotlib axes object
        If not passed, uses plt.gca()
    """
    import matplotlib.pyplot as plt

    ndims = len(darray.dims)
    if ndims != 1:
        raise ValueError('Line plots are for 1 dimensional DataArrays. '
        'Passed DataArray has {} dimensions'.format(ndims))

    if not ax:
        ax = plt.gca()

    xlabel, x = list(darray.indexes.items())[0]

    ax.plot(x, darray.values, *args, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(darray.name)

    return ax


def plot_contourf(darray, ax=None, *args, **kwargs):
    """
    Contour plot
    """
    import matplotlib.pyplot as plt

    if not ax:
        ax = plt.gca()

    # x axis is by default the one corresponding to the 0th axis
    xlabel, x = list(darray[0].indexes.items())[0]
    ylabel, y = list(darray[:, 0].indexes.items())[0]

    # TODO - revisit needing the transpose here
    ax.contourf(x, y, darray.values, *args, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def plot_hist(darray, ax=None, *args, **kwargs):
    """
    Histogram of DataArray using matplotlib / pylab.

    Parameters
    ----------
    darray : DataArray
        Can be 
    ax : matplotlib axes object
        If not passed, uses plt.gca()
    """
    import matplotlib.pyplot as plt

    if not ax:
        ax = plt.gca()

    ax.hist(np.ravel(darray))

    return ax
