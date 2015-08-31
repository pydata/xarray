from __future__ import division

import warnings
import itertools
import functools

import numpy as np
import pandas as pd

from ..core.formatting import format_item
from .plot import _determine_cmap_params


# Using this over mpl.rcParams["axes.labelsize"] since there are many of
# these strings, and they can get long
_TITLESIZE = 'small'


def _nicetitle(coord, value, maxchar, template):
    '''
    Put coord, value in template and truncate
    '''
    prettyvalue = format_item(value)
    title = template.format(coord=coord, value=prettyvalue)

    if len(title) > maxchar:
        title = title[:(maxchar - 3)] + '...'

    return title


class FacetGrid(object):
    '''
    Mostly copied from Seaborn
    '''

    def __init__(self, darray, col=None, row=None, col_wrap=None):

        import matplotlib.pyplot as plt

        # Handle corner case of nonunique coordinates
        rep_col = col is not None and not darray[col].to_index().is_unique
        rep_row = row is not None and not darray[row].to_index().is_unique
        if rep_col or rep_row:
            raise ValueError('Coordinates used for faceting cannot '
                             'contain repeated (nonunique) values.')

        # self._single_group is the grouping variable, if there is exactly one
        if col and row:
            self._single_group = False
            self._nrow = len(darray[row])
            self._ncol = len(darray[col])
            self.nfacet = self._nrow * self._ncol
            if col_wrap is not None:
                warnings.warn('Ignoring col_wrap since both col and row '
                              'were passed')
        elif row and not col:
            self._single_group = row
        elif not row and col:
            self._single_group = col
        else:
            raise ValueError('Pass a coordinate name as an argument for row or col')

        # Compute grid shape
        if self._single_group:
            self.nfacet = len(darray[self._single_group])
            if col:
                # idea - could add heuristic for nice shapes like 3x4
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
        col_names = list(darray[col].values) if col else []
        row_names = list(darray[row].values) if row else []

        if self._single_group:
            full = [{self._single_group: x} for x in
                      darray[self._single_group].values]
            empty = [None for x in range(self._nrow * self._ncol - len(full))]
            name_dicts = full + empty
        else:
            rowcols = itertools.product(row_names, col_names)
            name_dicts = [{row: r, col: c} for r, c in rowcols]

        self.name_dicts = np.array(name_dicts).reshape(self._nrow, self._ncol)

        self.row_names = row_names
        self.col_names = col_names
        self.darray = darray
        self.row = row
        self.col = col
        self.col_wrap = col_wrap


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

        # Plot the last subset again to determine x and y values
        try:
            dummyfig = plt.figure()
            dummyax = dummyfig.add_axes((0, 0, 0, 0))
            defaults['add_labels'] = True

            mappable = plotfunc(subset,
                    ax=dummyax, *args, **defaults)

            xlab, ylab = dummyax.get_xlabel(), dummyax.get_ylabel()
            bottomleft = self.axes[-1, 0]
            bottomleft.set_xlabel(xlab)
            bottomleft.set_ylabel(ylab)
            # Something to discuss- these labels could be centered on the
            # whole figure instead of the bottom left axes
            #self.fig.text(0.5, 0, xlab)
            #self.fig.text(0, 0.5, ylab)
        finally:
            plt.close(dummyfig)

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
        kwargs : keyword args
            additional arguments to matplotlib.text

        Returns
        -------
        self: object
            Returns self.

        '''
        import matplotlib as mpl

        kwargs.setdefault('size', _TITLESIZE)

        nicetitle = functools.partial(_nicetitle, maxchar=maxchar,
                template=template)

        if self._single_group:
            for d, ax in zip(self.name_dicts.flat, self.axes.flat):
                if d is not None:
                    coord, value = list(d.items()).pop()
                    title = nicetitle(coord, value, maxchar=maxchar)
                    ax.set_title(title, **kwargs)
        else:
            # The row titles on the left edge of the grid
            for ax, row_name in zip(self.axes[:, -1], self.row_names):
                title = nicetitle(coord=self.row, value=row_name, maxchar=maxchar)
                ax.annotate(title, xy=(1.02, .5), xycoords="axes fraction",
                            rotation=270, ha="left", va="center", **kwargs)

            # The column titles on the top row
            for ax, col_name in zip(self.axes[0, :], self.col_names):
                title = nicetitle(coord=self.col, value=col_name, maxchar=maxchar)
                ax.set_title(title, **kwargs)

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
