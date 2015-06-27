def _plot(darray, *args, **kwargs):
    """Plot a DataArray
    """
    import matplotlib.pyplot as plt
    return plt.plot(darray, *args, **kwargs)
