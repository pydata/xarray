import warnings

import numpy as np


class FacetGrid_seaborn(object):
    '''
    Copied from Seaborn
    '''

    def __init__(self, data, row=None, col=None, col_wrap=None,
                 margin_titles=False, xlim=None, ylim=None, subplot_kws=None,
                 gridspec_kws=None):

        import matplotlib as mpl
        import matplotlib.pyplot as plt

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

        # Validate some inputs
        if col_wrap is not None:
            margin_titles = False

        # Initialize the subplot grid
        if col_wrap is None:
            kwargs = dict(figsize=figsize, squeeze=False,
                          sharex=True, sharey=True,
                          )

            fig, axes = plt.subplots(nrow, ncol, **kwargs)
            self.axes = axes

        else:
            # If wrapping the col variable we need to make the grid ourselves
            n_axes = len(col_names)
            fig = plt.figure(figsize=figsize)
            axes = np.empty(n_axes, object)
            axes[0] = fig.add_subplot(nrow, ncol, 1, **subplot_kws)

            for i in range(1, n_axes):
                axes[i] = fig.add_subplot(nrow, ncol, i + 1, **subplot_kws)
            self.axes = axes

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

    def __init__(self, darray, col=None, row=None, col_wrap=None):
        import matplotlib.pyplot as plt
        self.darray = darray
        self.row = row
        self.col = col
        self.col_wrap = col_wrap

        # _group is the grouping variable, if there is only one
        if col and row:
            self._group = False
            self._nrow = len(darray[row])
            self._ncol = len(darray[col])
            self._margin_titles = True
            if col_wrap is not None:
                warnings.warn("Can't use col_wrap when both col and row are passed")
        elif row and not col:
            self._group = row
            self._margin_titles = False
        elif not row and col:
            self._group = col
            self._margin_titles = False
        else:
            raise ValueError('Pass a coordinate name as an argument for row or col')

        # Compute grid shape
        if self._group:
            self.nfacet = len(darray[self._group])
            if col:
                # TODO - could add heuristic for nice shape like 3x4
                self._ncol = self.nfacet
            if row:
                self._ncol = 1
            if col_wrap is not None:
                # Overrides previous settings
                self._ncol = col_wrap
            self._nrow = int(np.ceil(self.nfacet / self._ncol))

        self.fig, self.axes = plt.subplots(self._nrow, self._ncol,
                sharex=True, sharey=True)

        # Set up the lists of names for the row and column facet variables
        if row is None:
            row_names = []
        else:
            row_names = list(darray[row].values)

        if col is None:
            col_names = []
        else:
            col_names = list(darray[col].values)

        self.row_names = row_names
        self.col_names = col_names

        # Next the private variables
        self._row_var = row
        self._col_var = col
        self._col_wrap = col_wrap

        self.set_titles()

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

        if self._group:
            # TODO - bug should groupby _group
            for ax, (name, data) in zip(self, self.darray.groupby(self.col)):
                plt.sca(ax)
                mappable = func(data, *args, **defaults)
                #plt.title('{coord} = {val}'.format(coord=self.col,
                #    val=str(name)[:10]))
        else:
            # Looping over the indices helps keep sanity
            for col in range(self._ncol):
                for row in range(self._nrow):
                    plt.sca(self.axes[row, col])
                    # Similar to groupby
                    group = self.darray[{self.row: row, self.col: col}]
                    mappable = func(group, *args, **defaults)

        cbar = plt.colorbar(mappable, ax=self.axes.ravel().tolist())
        cbar.set_label(self.darray.name, rotation=270)

        return self

    def set_titles(self, template=None, row_template="{row_var} = {row_name}",
                   col_template="{col_var} = {col_name}", maxchar=10,
                   **kwargs):
        '''
        Draw titles either above each facet or on the grid margins.

        Parameters
        ----------
        template : string
            Template for all titles with the formatting keys {col_var} and
            {col_name} (if using a `col` faceting variable) and/or {row_var}
            and {row_name} (if using a `row` faceting variable).
        row_template:
            Template for the row variable when titles are drawn on the grid
            margins. Must have {row_var} and {row_name} formatting keys.
        col_template:
            Template for the row variable when titles are drawn on the grid
            margins. Must have {col_var} and {col_name} formatting keys.
        maxchar : int
            Truncate strings at maxchar

        Returns
        -------
        self: object
            Returns self.

        '''
        import matplotlib as mpl

        args = dict(row_var=self._row_var, col_var=self._col_var)
        kwargs["size"] = kwargs.pop("size", mpl.rcParams["axes.labelsize"])

        # Establish default templates
        if template is None:
            if self._row_var is None:
                template = col_template
            elif self._col_var is None:
                template = row_template
            else:
                template = " | ".join([row_template, col_template])

        def shorten(name, maxchar=maxchar):
            return str(name)[:maxchar]

        if self._margin_titles:
            if self.row_names:
                # Draw the row titles on the right edge of the grid
                for i, row_name in enumerate(self.row_names):
                    ax = self.axes[i, -1]
                    args['row_name'] = shorten(row_name)
                    title = row_template.format(**args)
                    ax.annotate(title, xy=(1.02, .5), xycoords="axes fraction",
                                rotation=270, ha="left", va="center", **kwargs)
            if self.col_names:
                # Draw the column titles as normal titles
                for j, col_name in enumerate(self.col_names):
                    args['col_name'] = shorten(col_name)
                    title = col_template.format(**args)
                    self.axes[0, j].set_title(title, **kwargs)

            return self

        # Otherwise title each facet with all the necessary information
        if (self._row_var is not None) and (self._col_var is not None):
            for i, row_name in enumerate(self.row_names):
                for j, col_name in enumerate(self.col_names):
                    args['row_name'] = shorten(row_name)
                    args['col_name'] = shorten(col_name)
                    title = template.format(**args)
                    self.axes[i, j].set_title(title, **kwargs)
        elif self.row_names is not None and len(self.row_names):
            for i, row_name in enumerate(self.row_names):
                args['row_name'] = shorten(row_name)
                title = template.format(**args)
                self.axes[i, 0].set_title(title, **kwargs)
        elif self.col_names is not None and len(self.col_names):
            for i, col_name in enumerate(self.col_names):
                args['col_name'] = shorten(col_name)
                title = template.format(**args)
                # Index the flat array so col_wrap works
                self.axes.flat[i].set_title(title, **kwargs)
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



def map_dataarray2(self, func, *args, **kwargs):
    """Experimenting with row and col
    """
    import matplotlib.pyplot as plt

    defaults = dict(add_colorbar=False,
            add_labels=False,
            vmin=float(self.darray.min()),
            vmax=float(self.darray.max()),
            )

    defaults.update(kwargs)

    # Looping over the indices helps keep sanity
    for col in range(ncol):
        for row in range(nrow):
            plt.sca(axes[row, col])
            # Similar to groupby
            group = darray[{self.row: row, self.col: col}]
            mappable = func(group, *args, **defaults)

    plt.colorbar(mappable, ax=self.axes.ravel().tolist())
    return self
