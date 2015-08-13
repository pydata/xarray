import numpy as np


class FacetGrid_seaborn(object):
    '''
    Copied from Seaborn
    '''

    def __init__(self, data, row=None, col=None, col_wrap=None,
                 sharex=True, sharey=True, size=3, aspect=1,
                 dropna=True, legend_out=True, despine=True,
                 margin_titles=False, xlim=None, ylim=None, subplot_kws=None,
                 gridspec_kws=None):

        import matplotlib as mpl
        import matplotlib.pyplot as plt

        MPL_GRIDSPEC_VERSION = LooseVersion('1.4')
        OLD_MPL = LooseVersion(mpl.__version__) < MPL_GRIDSPEC_VERSION

        if row:
            row_names = data[row].values
            nrow = len(row_names)
        else:
            nrow = 1

        if col:
            col_names = data[col].values
            ncol = len(col_names)
        else:
            ncol = 1

        # Compute the grid shape
        self._n_facets = ncol * nrow

        self._col_wrap = col_wrap
        if col_wrap is not None:
            if row is not None:
                err = "Cannot use `row` and `col_wrap` together."
                raise ValueError(err)
            ncol = col_wrap
            nrow = int(np.ceil(len(data[col].unique()) / col_wrap))
        self._ncol = ncol
        self._nrow = nrow

        # Calculate the base figure size
        # This can get stretched later by a legend
        figsize = (ncol * size * aspect, nrow * size)

        # Validate some inputs
        if col_wrap is not None:
            margin_titles = False

        # Build the subplot keyword dictionary
        subplot_kws = {} if subplot_kws is None else subplot_kws.copy()
        gridspec_kws = {} if gridspec_kws is None else gridspec_kws.copy()
        if xlim is not None:
            subplot_kws["xlim"] = xlim
        if ylim is not None:
            subplot_kws["ylim"] = ylim

        # Initialize the subplot grid
        if col_wrap is None:
            kwargs = dict(figsize=figsize, squeeze=False,
                          sharex=sharex, sharey=sharey,
                          subplot_kw=subplot_kws,
                          gridspec_kw=gridspec_kws)

            if OLD_MPL:
                _ = kwargs.pop('gridspec_kw', None)
                if gridspec_kws:
                    msg = "gridspec module only available in mpl >= {}"
                    warnings.warn(msg.format(MPL_GRIDSPEC_VERSION))

            fig, axes = plt.subplots(nrow, ncol, **kwargs)
            self.axes = axes

        else:
            # If wrapping the col variable we need to make the grid ourselves
            if gridspec_kws:
                warnings.warn("`gridspec_kws` ignored when using `col_wrap`")

            n_axes = len(col_names)
            fig = plt.figure(figsize=figsize)
            axes = np.empty(n_axes, object)
            axes[0] = fig.add_subplot(nrow, ncol, 1, **subplot_kws)
            if sharex:
                subplot_kws["sharex"] = axes[0]
            if sharey:
                subplot_kws["sharey"] = axes[0]
            for i in range(1, n_axes):
                axes[i] = fig.add_subplot(nrow, ncol, i + 1, **subplot_kws)
            self.axes = axes

            # Now we turn off labels on the inner axes
            if sharex:
                for ax in self._not_bottom_axes:
                    for label in ax.get_xticklabels():
                        label.set_visible(False)
                    ax.xaxis.offsetText.set_visible(False)
            if sharey:
                for ax in self._not_left_axes:
                    for label in ax.get_yticklabels():
                        label.set_visible(False)
                    ax.yaxis.offsetText.set_visible(False)

        # Set up the class attributes
        # ---------------------------

        # First the public API
        self.data = data
        self.fig = fig
        self.axes = axes

        #self.row_names = row_names
        #self.col_names = col_names

        # Next the private variables
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col

        self._margin_titles = margin_titles
        self._col_wrap = col_wrap
        self._legend_out = legend_out
        self._legend = None
        self._legend_data = {}
        self._x_var = None
        self._y_var = None


class FacetGrid(object):
    '''
    Mostly copied from Seaborn
    '''

    def __init__(self, darray, col=None, col_wrap=None):
        import matplotlib.pyplot as plt
        self.darray = darray
        #self.row = row
        self.col = col
        self.col_wrap = col_wrap

        self.nfacet = len(darray[col])

        # Compute grid shape
        if col_wrap is not None:
            self.ncol = col_wrap
        else:
            # TODO- add heuristic for inference here to get a nice shape
            # like 3 x 4
            self.ncol = self.nfacet

        self.nrow = int(np.ceil(self.nfacet / self.ncol))

        self.fig, self.axes = plt.subplots(self.nrow, self.ncol,
                sharex=True, sharey=True)

    def __iter__(self):
        return self.axes.flat

    def map_dataarray(self, func, *args, **kwargs):
        """Apply a plotting function to each facet's subset of the data.

        Differs from Seaborn style - requires the func to know how to plot a
        dataarray.
        
        For now I'm going to write this assuming func is an xray 2d
        plotting function

        Parameters
        ----------
        func : callable
            A plotting function with the first argument an xray dataarray 
        args :
            positional arguments to func
        kwargs :
            keyword arguments to func

        Returns
        -------
        self : object
            Returns self.

        """
        import matplotlib.pyplot as plt

        defaults = dict(add_colorbar=False,
                add_labels=False,
                vmin=float(self.darray.min()),
                vmax=float(self.darray.max()),
                )

        defaults.update(kwargs)

        for ax, (name, data) in zip(self, self.darray.groupby(self.col)):

            plt.sca(ax)

            mappable = func(data, *args, **defaults)

            plt.title('{coord} = {val}'.format(coord=self.col,
                val=str(name)[:10]))

        plt.colorbar(mappable, ax=self.axes.ravel().tolist())
        return self

    def map(self, func, *args, **kwargs):
        """Apply a plotting function to each facet's subset of the data.

        True to Seaborn style

        Parameters
        ----------
        func : callable
            A plotting function that takes data and keyword arguments. It
            must plot to the currently active matplotlib Axes and take a
            `color` keyword argument. If faceting on the `hue` dimension,
            it must also take a `label` keyword argument.
        args : strings
            Column names in self.data that identify variables with data to
            plot. The data for each variable is passed to `func` in the
            order the variables are specified in the call.
        kwargs : keyword arguments
            All keyword arguments are passed to the plotting function.

        Returns
        -------
        self : object
            Returns self.

        """
        import matplotlib.pyplot as plt

        for ax, (name, data) in zip(self, self.darray.groupby(self.col)):

            kwargs['add_colorbar'] = False
            plt.sca(ax)

            innerargs = [data[a] for a in args]
            func(*innerargs, **kwargs)

        return self
