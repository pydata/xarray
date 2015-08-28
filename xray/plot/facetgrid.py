from __future__ import division

import warnings
import itertools

import numpy as np
import pandas as pd

from ..core.formatting import format_item
from .plot import _determine_cmap_params


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

        # _single_group is the grouping variable, if there is only one
        if col and row:
            self._single_group = False
            self._nrow = len(darray[row])
            self._ncol = len(darray[col])
            self.nfacet = self._nrow * self._ncol
            self._margin_titles = True
            if col_wrap is not None:
                warnings.warn("Can't use col_wrap when both col and row are passed")
        elif row and not col:
            self._single_group = row
            self._margin_titles = False
        elif not row and col:
            self._single_group = col
            self._margin_titles = False
        else:
            raise ValueError('Pass a coordinate name as an argument for row or col')

        # Relying on short circuit behavior
        # Not sure when nonunique coordinates are a problem
        rep_col = col is not None and not self.darray[col].to_index().is_unique
        rep_row = row is not None and not self.darray[row].to_index().is_unique
        if rep_col or rep_row:
            raise ValueError('Coordinates used for faceting cannot '
                             'contain repeated (nonunique) values.')

        # Compute grid shape
        if self._single_group:
            self.nfacet = len(darray[self._single_group])
            if col:
                # TODO - could add heuristic for nice shapes like 3x4
                self._ncol = self.nfacet
            if row:
                self._ncol = 1
            if col_wrap is not None:
                # Overrides previous settings
                self._ncol = col_wrap
            self._nrow = int(np.ceil(self.nfacet / self._ncol))

        self.fig, self.axes = plt.subplots(self._nrow, self._ncol,
                sharex=True, sharey=True)

        # subplots flattens this array if one dimension
        self.axes.shape = self._nrow, self._ncol

        # Set up the lists of names for the row and column facet variables
        if row is None:
            row_names = []
        else:
            row_names = list(darray[row].values)

        if col is None:
            col_names = []
        else:
            col_names = list(darray[col].values)

        # TODO-
        # Refactor to have two data structures 
        # name_dicts - a list of dicts corresponding to the flattened axes arrays
        # allows iterating through without hitting the sentinel value
        # name_array - an array of dicts corresponding to axes

        if self._single_group:
            full = [{self._single_group: x} for x in
                      self.darray[self._single_group].values]
            empty = [None for x in range(self._nrow * self._ncol - len(full))]
            name_dicts = full + empty
        else:
            rowcols = itertools.product(row_names, col_names)
            name_dicts = [{row: r, col: c} for r, c in rowcols]

        self.name_dicts = np.array(name_dicts).reshape(self._nrow, self._ncol)

        self.row_names = row_names
        self.col_names = col_names

        # Next the private variables
        self._row_var = row
        self._col_var = col
        self._col_wrap = col_wrap

        self.set_titles()

    def __iter__(self):
        return self.axes.flat

    def map_dataarray(self, plotfunc, *args, **kwargs):
        """Apply a plotting function to a 2d facet's subset of the data.

        This is more convenient and less general than the map method.

        Parameters
        ----------
        plotfunc : callable
            A plotting function with the same signature as a 2d xray
            plotting method such as `xray.plot.imshow`
        args :
            positional arguments to plotfunc
        kwargs :
            keyword arguments to plotfunc

        Returns
        -------
        self : FacetGrid object
            the same FacetGrid on which the method was called

        """
        import matplotlib.pyplot as plt

        # These should be consistent with xray.plot._plot2d
        cmap_kwargs = {'plot_data': self.darray.values,
                'vmin': None,
                'vmax': None,
                'cmap': None,
                'center': None,
                'robust': False,
                'extend': None,
                'levels': 7 if 'contour' in plotfunc.__name__ else None, # MPL default
                'filled': plotfunc.__name__ != 'contour',
                }

        # Allow kwargs to override these defaults
        for param in kwargs:
            if param in cmap_kwargs:
                cmap_kwargs[param] = kwargs[param]

        # colormap inference has to happen here since all the data in self.darray
        # is required to make the right choice
        cmap_params = _determine_cmap_params(**cmap_kwargs)

        if 'contour' in plotfunc.__name__:
            # extend is a keyword argument only for contour and contourf, but
            # passing it to the colorbar is sufficient for imshow and
            # pcolormesh
            kwargs['extend'] = cmap_params['extend']
            kwargs['levels'] = cmap_params['levels']

        defaults = {
                'add_colorbar': False,
                'add_labels': False,
                'norm': cmap_params.pop('cnorm'),
                }

        # Order is important
        defaults.update(cmap_params)
        defaults.update(kwargs)

        ## Color limit calculations
        #robust = defaults['robust']
        #calc_data = self.darray.values
        #calc_data = calc_data[~pd.isnull(calc_data)]

        ## TODO - use percentile as global variable from other module
        #vmin = np.percentile(calc_data, 2) if robust else calc_data.min()
        #vmax = np.percentile(calc_data, 98) if robust else calc_data.max()
        #defaults.setdefault('vmin', vmin)
        #defaults.setdefault('vmax', vmax)

        for d, ax in zip(self.name_dicts.flat, self.axes.flat):
            # Handle the sentinel value
            if d is not None:
                subset = self.darray.loc[d]
                plotfunc(subset, ax=ax, *args, **defaults)

        # Add the labels to the bottom left plot
        # => plotting this one twice
        defaults['add_labels'] = True
        bottomleft = self.axes[-1, 0]
        oldtitle = bottomleft.get_title()
        mappable = plotfunc(self.darray.loc[self.name_dicts[-1, 0]],
                ax=bottomleft, *args, **defaults)
        bottomleft.set_title(oldtitle)

        # colorbar
        if kwargs.get('add_colorbar', True):
            cbar = plt.colorbar(mappable, ax=self.axes.ravel().tolist(),
                    extend=cmap_params['extend'])
            cbar.set_label(self.darray.name, rotation=270,
                    verticalalignment='bottom')

        return self


    def set_titles(self, template="{coord} = {value}", maxchar=30, **kwargs):
        '''
        Draw titles either above each facet or on the grid margins.

        Parameters
        ----------
        template : string
            Template for plot titles
        maxchar : int
            Truncate titles at maxchar
            # TODO - may want to append '...' to indicate

        Returns
        -------
        self: object
            Returns self.

        '''
        import matplotlib as mpl

        kwargs["size"] = kwargs.pop("size", mpl.rcParams["axes.labelsize"])

        if self._single_group:
            for d, ax in zip(self.name_dicts.flat, self.axes.flat):
                # TODO Remove check for sentinel value
                if d is not None:
                    coord, value = list(d.items()).pop()
                    prettyvalue = format_item(value)
                    title = template.format(coord=coord, value=prettyvalue)
                    title = title[:maxchar]
                    ax.set_title(title)
        else:
            # The row titles on the left edge of the grid
            for ax, row_name in zip(self.axes[:, -1], self.row_names):
                title = template.format(coord=self.row, value=format_item(row_name))
                title = title[:maxchar]
                ax.annotate(title, xy=(1.02, .5), xycoords="axes fraction",
                            rotation=270, ha="left", va="center", **kwargs)

            # The column titles on the top row
            for ax, col_name in zip(self.axes[0, :], self.col_names):
                title = template.format(coord=self.col, value=format_item(col_name))
                title = title[:maxchar]
                ax.set_title(title)

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
