"""
Plotting functions are implemented here and also monkeypatched in to
DataArray and DataSet classes
"""

# TODO - Is there a better way to import matplotlib in the function?
# Decorators don't preserve the argument names


class FacetGrid():
    pass


def _plot_line(darray, *args, **kwargs):
    """
    Line plot
    """
    import matplotlib.pyplot as plt

    xlabel, x = list(darray.indexes.items())[0]

    # Probably should be using the lower level matplotlib API
    plt.plot(x, darray.values, *args, **kwargs)
    ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(darray.name)

    return ax


def _plot_contourf(darray, *args, **kwargs):
    """
    Contour plot
    """
    import matplotlib.pyplot as plt

    # x axis is by default the one corresponding to the 0th axis
    xlabel, x = list(darray[0].indexes.items())[0]
    ylabel, y = list(darray[:, 0].indexes.items())[0]

    # TODO - revisit needing the transpose here
    plt.contourf(x, y, darray.values, *args, **kwargs)
    ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax
