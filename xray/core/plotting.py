"""
Plotting functions are implemented here and then monkeypatched in to
DataArray and DataSet classes
"""

def _plot_dataarray(darray, *args, **kwargs):
    """
    Plot a DataArray
    """
    import matplotlib.pyplot as plt

    xlabel = darray.indexes.keys()[0]
    x = darray.indexes[xlabel].values

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

    xlabel, ylabel = darray.indexes.keys()[0:2]
    x = darray.indexes[xlabel].values
    y = darray.indexes[ylabel].values

    # Assume 2d matrix with x on dim_0, y on dim_1
    z = darray.values.T

    plt.contourf(x, y, z, *args, **kwargs)
    ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax
