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
    y = darray.values

    # Probably should be using the lower level matplotlib API
    plt.plot(x, y, *args, **kwargs)
    ax = plt.gca()
    ax.set_xlabel(xlabel)

    return ax


def _plot_contourf(dset, *args, **kwargs):
    """
    Plot a Dataset
    """
    import matplotlib.pyplot as plt

    xlabel = darray.indexes.keys()[0]
    x = darray.indexes[xlabel].values
    y = darray.values

    # Probably should be using the lower level matplotlib API
    plt.plot(x, y, *args, **kwargs)
    ax = plt.gca()
    ax.set_xlabel(xlabel)

    return ax
