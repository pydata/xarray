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
